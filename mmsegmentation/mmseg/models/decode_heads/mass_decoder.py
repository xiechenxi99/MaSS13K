import math


import logging
from typing import Callable, Dict, Optional, Union
import torch
from torch import nn
import torch.nn.functional as F

from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmdet.models.layers import  SinePositionalEncoding

from mmcv.ops import ModulatedDeformConv2d
from mmcv.cnn import build_norm_layer

def window_partition(x, window_size):
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0,5, 1, 3, 2, 4).contiguous().view(B,-1, H, W)
    return x

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, norm='BN', groups=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              bias=False,
                              groups=groups)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU()
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)




class DeformLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 deconv_kernel=4,
                 deconv_stride=2,
                 deconv_padding=1,
                 deconv_out_padding=0,
                 num_groups=1,
                 deform_groups=1,
                 dilation=1,
                 norm_cfg=dict(type='BN'),
                 with_upsample=True):
        super(DeformLayer, self).__init__()
        self.with_upsample = with_upsample
        self.deform_groups = deform_groups
        self.kernel_size = kernel_size

        # Offset and mask generator
        offset_channels = 3 * kernel_size * kernel_size  # offset_x/y + mask
        self.dcn_offset = nn.Conv2d(
            in_channels,
            offset_channels * deform_groups,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            dilation=dilation)

        # Deformable Convolution
        self.dcn = ModulatedDeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            deform_groups=deform_groups)

        # Normalization after DCN
        self.dcn_bn = build_norm_layer(norm_cfg, out_channels)[1]

        # Optional upsampling
        if self.with_upsample:
            self.up_sample = nn.ConvTranspose2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=deconv_kernel,
                stride=deconv_stride,
                padding=deconv_padding,
                output_padding=deconv_out_padding,
                bias=False)
            self._deconv_init()
            self.up_bn = build_norm_layer(norm_cfg, out_channels)[1]

        self.relu = nn.ReLU(inplace=True)

        self.init_weights()

    def init_weights(self):
        for m in [self.dcn]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.dcn_offset.weight, 0)
        nn.init.constant_(self.dcn_offset.bias, 0)

    def _deconv_init(self):
        """Initialize the up-sample conv transpose weights like bilinear."""
        w = self.up_sample.weight.data
        f = math.ceil(w.size(2) / 2)
        c = (2 * f - 1 - f % 2) / (2. * f)
        for i in range(w.size(2)):
            for j in range(w.size(3)):
                w[0, 0, i, j] = \
                    (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
        for c in range(1, w.size(0)):
            w[c, 0, :, :] = w[0, 0, :, :]

    def forward(self, x):
        offset_mask = self.dcn_offset(x)
        offset_x, offset_y, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((offset_x, offset_y), dim=1)
        mask = mask.sigmoid()
        x = self.dcn(x, offset, mask)
        x = self.dcn_bn(x)
        x = self.relu(x)

        if self.with_upsample:
            x_up = self.up_sample(x)
            x_up = self.up_bn(x_up)
            x_up = self.relu(x_up)
            return x, x_up
        else:
            return x

class CSAttn(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.pos= SinePositionalEncoding(num_feats=dim//2, normalize=True)
        self.cross_attn = MultiheadAttention(embed_dims=dim,num_heads=8,batch_first=True)
        self.self_attn =  MultiheadAttention(embed_dims=dim,num_heads=8,batch_first=True)
        self.ffn1        = FFN(embed_dims=dim)
        self.ffn2        = FFN(embed_dims=dim)
    def forward(self,x_32,x_16):
        B,C,H,W=x_32.shape
        ws = 16
        x_32=window_partition(x_32,ws)
        x_16=window_partition(x_16,ws)
        # import ipdb
        # ipdb.set_trace()
        pos = self.pos(x_16.new_zeros((x_16.shape[0],)+x_16.shape[-2:])).flatten(2).permute(0, 2, 1) # B N D
        x_32 = x_32.flatten(2).permute(0,2,1)
        x_16 = x_16.flatten(2).permute(0,2,1)
        out = self.cross_attn(query=x_32,key=x_16,value=x_16,query_pos=pos,key_pos=pos)
        out = self.ffn1(out)
        out = self.self_attn(query=out,key=out,value=out,query_pos=pos,key_pos=pos)
        out = self.ffn2(out)
        out = window_reverse(out,ws,H,W)
        # out = out.permute(0,2,1).contiguous().view(B,C,H,W)
        return out

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) module for feature transformation.

    This module applies two convolutional layers with batch normalization and ReLU activation
    to transform input feature maps.

    Args:
        dim (int): The input feature dimensionality.
        factor (int): The reduction factor for the intermediate feature dimensionality.
    """
    def __init__(self, dim, factor, norm,outdim=128):
        super().__init__()

        self.mlp = nn.Sequential(nn.Conv2d(dim, dim // factor, 1, bias=False),
                                 nn.BatchNorm2d( dim // factor),
                                 nn.ReLU(),
                                 nn.Conv2d(dim // factor, dim, 1, bias=False),
                                 )

    def forward(self, x):
        return self.mlp(x)

class SEB(nn.Module):
    def __init__(self, hidden_dim, factor, norm):
        super().__init__()
        self.mlp = nn.Sequential(MLP(hidden_dim, factor, norm),
                                 nn.Sigmoid())

    def forward(self, x):
        x_avg = x.mean(dim=[2, 3], keepdim=True)
        x_w = self.mlp(x_avg).sigmoid()
        return x + x * x_w



class RFB(nn.Module):
    def __init__(self,inter_channels,sizelist=[1,3,5,7]):
        super().__init__()
        self.deformaspp = nn.ModuleList([DeformLayer(inter_channels, inter_channels//len(sizelist), kernel_size,with_upsample=False) for kernel_size in sizelist])
        self.conv1=nn.Conv2d(inter_channels,inter_channels,1)
        self.bn=nn.BatchNorm2d(inter_channels)
        self.relu=nn.ReLU()
        self.upconv=nn.Conv2d(inter_channels,inter_channels,3,padding=1)
        self.bn2=nn.BatchNorm2d(inter_channels)
    def forward(self,x):
        out=[]
        for conv in self.deformaspp:
            out.append(conv(x))
        y = torch.cat(out,dim=1)
        y = self.relu(self.bn(self.conv1(y)))

        y_up=F.interpolate(y,size=(y.shape[2]*2,y.shape[3]*2),mode='bilinear',align_corners=False)
        y_up = self.relu(self.bn2(self.upconv(y_up)))
        return y,y_up



class LSE(nn.Module):
    def __init__(self,in_dim,inter_channels):
        super().__init__()
        self.cbn1=ConvBNReLU(in_dim//2+3,inter_channels//2,1,padding=0,norm=nn.BatchNorm2d)
        self.dcn3 = DeformLayer(in_dim//2+3, inter_channels//2, 3,with_upsample=False)
        self.conv1 = nn.Conv2d(2,1,kernel_size=7,padding=3)
        self.conv2 = nn.Conv2d(2,1,kernel_size=7,padding=3)
        self.sep = nn.Conv2d(inter_channels,inter_channels,3,padding=1,groups=inter_channels)
        self.cbn2=ConvBNReLU(inter_channels,inter_channels,1,padding=0,norm=nn.BatchNorm2d)
    def forward(self,img,feat4):
        feat4 = F.interpolate(feat4,size=(img.shape[2]//2,img.shape[3]//2),mode='bilinear',align_corners=False)
        img = F.interpolate(img,size=(img.shape[2]//2,img.shape[3]//2),mode='bilinear',align_corners=False)
        x1,x2 = torch.chunk(feat4,2,dim=1)
        # import ipdb
        # ipdb.set_trace()
        x1 = self.cbn1(torch.cat([img,x1],dim=1))
        x2 = self.dcn3(torch.cat([img,x2],dim=1))

        x1_mean = torch.mean(x1,dim=1,keepdim=True)
        x1_max,_  = torch.max(x1,dim=1,keepdim=True)
        x1_attn = self.conv1(torch.sigmoid(torch.cat([x1_mean,x1_max],dim=1)))

        x2_mean = torch.mean(x2,dim=1,keepdim=True)
        x2_max,_  = torch.max(x2,dim=1,keepdim=True)
        x2_attn = self.conv2(torch.sigmoid(torch.cat([x2_mean,x2_max],dim=1)))
        x1 = x1*x1_attn+x1
        x2 = x2*x2_attn+x2

        y  = torch.cat([x1,x2],dim=1)
        # x2 = self.sep(x1)
        z = self.cbn2(y)
        return z


class ERM(nn.Module):
    def __init__(self,in_channels,inter_channels):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels,inter_channels,1,1,0)
        # self.convsep3 = nn.Conv2d(64,64,3,1,1,groups=64)
        self.conv2 = nn.Conv2d(inter_channels,in_channels,1,1,0)
        self.dcn = DeformLayer(inter_channels, inter_channels, 3,with_upsample=False)
        self.act = nn.GELU()
        self.norm = nn.BatchNorm2d(in_channels)
    def forward(self,edge,body):
        edge = torch.sigmoid(edge)
        edge = edge*body
        edge = self.norm(self.conv2(self.dcn(self.conv1(edge))))
        # edge = self.norm(self.act(edge))
        return edge+body

class CST(nn.Module):
    def __init__(self,inter_channels):
        super().__init__()
        self.dcn1=DeformLayer(in_channels=inter_channels,out_channels=inter_channels,kernel_size=3,norm_cfg=dict(type='BN'),with_upsample=True)
        self.dcn2=DeformLayer(in_channels=inter_channels,out_channels=inter_channels,kernel_size=3,norm_cfg=dict(type='BN'),with_upsample=True)
        self.cross_self_attn = CSAttn(inter_channels)
    def forward(self,f4,s4,s3):
        x_32       = s4 + f4
        x_32, x_up = self.dcn1(x_32)
        x_16, x_up  = self.dcn2(self.cross_self_attn(x_up,s3))
        return x_32,x_16,x_up


        

class HR_Pixel_Decoder(nn.Module):

    def __init__(
        self,
        conv_dim: int = 128,
        mask_dim: int = 256,
        bkc = [256,512,1024,2048],
        inter_channels=128,
        norm: Optional[Union[str, Callable]] = nn.BatchNorm2d,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__()

        self.norm = norm
        self.in_channels = [256,512,1024,2048]
        # import ipdb
        # ipdb.set_trace()
        inter_channels=inter_channels
        # self.in_channels=[64,128,256,512]

        self.in_projections = nn.ModuleList([ConvBNReLU(in_dim, inter_channels, 1, norm=norm, padding=0)
                                             for in_dim in self.in_channels])
        self.out = nn.ModuleList([ConvBNReLU(inter_channels, inter_channels*2, 1, padding=0, norm=norm) for _ in range(5)])

        self.conv_avg = ConvBNReLU(self.in_channels[-1], inter_channels, 1, padding=0, norm=norm)

        self.SEB = nn.ModuleList([SEB(inter_channels, factor=2, norm=norm) for _ in range(4)])
        self.LSE =LSE(inter_channels,inter_channels)
        self.CST = CST(inter_channels)
        self.edge_conv = nn.Sequential(ConvBNReLU(inter_channels,16,1,padding=0,norm=norm),nn.Conv2d(16,1,1,1,padding=0))
        self.edge_rec = ERM(inter_channels,inter_channels//2)
        self.RFB=RFB(inter_channels,[1,3,5,7])
    def forward(self, features):
        in_features = []

        for feature, projection in zip(features[1:], self.in_projections):
            in_features.append(projection(feature))

        conv_avg = self.conv_avg(features[-1].mean(dim=[2, 3], keepdim=True))
        x_32 = self.SEB[0](in_features[3])
        x_16 = self.SEB[1](in_features[2])
        x_8  = self.SEB[2](in_features[1])
        x_2  = self.SEB[3](in_features[0])

        x_32,x_16,x_up = self.CST(conv_avg,x_32,x_16)
        x_8,x_up=self.RFB(x_up+x_8)

        img = features[0]
        x_2 = self.LSE(img,x_2)

        x_up = F.interpolate(x_8,size=(features[0].shape[2]//2,features[0].shape[3]//2),mode='bilinear',align_corners=False)
        x_edge = self.edge_conv(x_up+x_2)
        x_out = self.edge_rec(x_2,x_up)

        return self.out[0](x_out),x_edge,[self.out[1](x_32),self.out[2](x_16),self.out[3](x_8)]

if __name__=='__main__':
    HR_Pixel_Decoder()
