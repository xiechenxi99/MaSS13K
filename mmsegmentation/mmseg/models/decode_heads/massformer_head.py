# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple,Union,Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import (BaseModule, ModuleList, caffe2_xavier_init,
                            normal_init, xavier_init)

try:
    from mmdet.models.dense_heads import \
        Mask2FormerHead as MMDET_Mask2FormerHead
except ModuleNotFoundError:
    MMDET_Mask2FormerHead = BaseModule

from mmengine.structures import InstanceData
from torch import Tensor
from mmcv.cnn import Conv2d, ConvModule
from mmseg.registry import MODELS
from mmseg.structures.seg_data_sample import SegDataSample
from mmseg.utils import ConfigType, SampleList
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmdet.models.layers import  SinePositionalEncoding
from mmdet.utils import reduce_mean
from mmdet.models.utils import get_uncertain_point_coords_with_randomness,multi_apply

from mmcv.ops import point_sample,ModulatedDeformConv2d
from .mass_decoder import HR_Pixel_Decoder
def get_uncertainty(mask_preds: Tensor, labels: Tensor) -> Tensor:
    """Estimate uncertainty based on pred logits.

    We estimate uncertainty as L1 distance between 0.0 and the logits
    prediction in 'mask_preds' for the foreground class in `classes`.

    Args:
        mask_preds (Tensor): mask predication logits, shape (num_rois,
            num_classes, mask_height, mask_width).

        labels (Tensor): Either predicted or ground truth label for
            each predicted mask, of length num_rois.

    Returns:
        scores (Tensor): Uncertainty scores with the most uncertain
            locations having the highest uncertainty score,
            shape (num_rois, 1, mask_height, mask_width)
    """
    if mask_preds.shape[1] == 1:
        gt_class_logits = mask_preds.clone()
    else:
        inds = torch.arange(mask_preds.shape[0], device=mask_preds.device)
        gt_class_logits = mask_preds[inds, labels].unsqueeze(1)
    return -torch.abs(gt_class_logits)
def get_uncertain_point_coords_with_randomness_reweight(
        mask_preds: Tensor, labels: Tensor, num_points: int,
        oversample_ratio: float, importance_sample_ratio: float,weight: Tensor) -> Tensor:
    """Get ``num_points`` most uncertain points with random points during
    train.

    Sample points in [0, 1] x [0, 1] coordinate space based on their
    uncertainty. The uncertainties are calculated for each point using
    'get_uncertainty()' function that takes point's logit prediction as
    input.

    Args:
        mask_preds (Tensor): A tensor of shape (num_rois, num_classes,
            mask_height, mask_width) for class-specific or class-agnostic
            prediction.
        labels (Tensor): The ground truth class for each instance.
        num_points (int): The number of points to sample.
        oversample_ratio (float): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled
            via importnace sampling.

    Returns:
        point_coords (Tensor): A tensor of shape (num_rois, num_points, 2)
            that contains the coordinates sampled points.
    """
    assert oversample_ratio >= 1
    assert 0 <= importance_sample_ratio <= 1
    batch_size = mask_preds.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(
        batch_size, num_sampled, 2, device=mask_preds.device)
    point_logits = point_sample(mask_preds, point_coords)
    weight       = point_sample(weight, point_coords)
    # It is crucial to calculate uncertainty based on the sampled
    # prediction value for the points. Calculating uncertainties of the
    # coarse predictions first and sampling them for points leads to
    # incorrect results.  To illustrate this: assume uncertainty func(
    # logits)=-abs(logits), a sampled point between two coarse
    # predictions with -1 and 1 logits has 0 logits, and therefore 0
    # uncertainty value. However, if we calculate uncertainties for the
    # coarse predictions first, both will have -1 uncertainty,
    # and sampled point will get -1 uncertainty.
    point_uncertainties = get_uncertainty(point_logits, labels)*weight
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(
        point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(
        batch_size, dtype=torch.long, device=mask_preds.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        batch_size, num_uncertain_points, 2)
    if num_random_points > 0:
        rand_roi_coords = torch.rand(
            batch_size, num_random_points, 2, device=mask_preds.device)
        point_coords = torch.cat((point_coords, rand_roi_coords), dim=1)
    return point_coords

class FPNbaseline(nn.Module):
    def __init__(self,in_channels: Union[List[int],
                                    Tuple[int]] = [256, 512, 1024, 2048],
                        feat_channels: int=256,
                        norm_cfg: ConfigType = dict(type='LN'),
                        act_cfg: ConfigType = dict(type='GELU'),
                        pixel_decoder=None):
        super().__init__()

        self.pixel_decoder = HR_Pixel_Decoder()


    def forward(self,x):
        out = x
        mask_features,edge_feature, multi_scale_memorys = self.pixel_decoder(out)

        return mask_features,edge_feature, multi_scale_memorys
    def init_weights(self):
        pass


@MODELS.register_module()
class MaSSFormerHead(MMDET_Mask2FormerHead):
    """Implements the Mask2Former head.

    See `Mask2Former: Masked-attention Mask Transformer for Universal Image
    Segmentation <https://arxiv.org/abs/2112.01527>`_ for details.

    Args:
        num_classes (int): Number of classes. Default: 150.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        ignore_index (int): The label index to be ignored. Default: 255.
    """

    def __init__(self,
                 num_classes,
                 align_corners=False,
                 ignore_index=255,
                 **kwargs):
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.align_corners = align_corners
        self.out_channels = num_classes
        self.ignore_index = ignore_index
        self.pixel_decoder=FPNbaseline(pixel_decoder=kwargs['pixel_decoder'])
        self.loss_edge = nn.BCELoss()

        feat_channels = kwargs['feat_channels']
        
        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)

    def _forward_aux(self,mask_embed,decoder_out,mask_feature):
        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        # shape (num_queries, batch_size, c)
        mask_embed = mask_embed(decoder_out)
        # shape (num_queries, batch_size, h, w)
        mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)
        return mask_pred

    def _seg_data_to_instance_data(self, batch_data_samples: SampleList):
        """Perform forward propagation to convert paradigm from MMSegmentation
        to MMDetection to ensure ``MMDET_Mask2FormerHead`` could be called
        normally. Specifically, ``batch_gt_instances`` would be added.

        Args:
            batch_data_samples (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.

        Returns:
            tuple[Tensor]: A tuple contains two lists.

                - batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                    gt_instance. It usually includes ``labels``, each is
                    unique ground truth label id of images, with
                    shape (num_gt, ) and ``masks``, each is ground truth
                    masks of each instances of a image, shape (num_gt, h, w).
                - batch_img_metas (list[dict]): List of image meta information.
        """
        batch_img_metas = []
        batch_gt_instances = []

        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            gt_sem_seg = data_sample.gt_sem_seg.data
            classes = torch.unique(
                gt_sem_seg,
                sorted=False,
                return_inverse=False,
                return_counts=False)

            # remove ignored region
            gt_labels = classes[classes != self.ignore_index]

            masks = []
            for class_id in gt_labels:
                masks.append(gt_sem_seg == class_id)

            if len(masks) == 0:
                gt_masks = torch.zeros(
                    (0, gt_sem_seg.shape[-2],
                     gt_sem_seg.shape[-1])).to(gt_sem_seg).long()
            else:
                gt_masks = torch.stack(masks).squeeze(1).long()

            instance_data = InstanceData(labels=gt_labels, masks=gt_masks)
            batch_gt_instances.append(instance_data)
        return batch_gt_instances, batch_img_metas

    def _forward_head(self, decoder_out: Tensor, mask_feature: Tensor,
                      attn_mask_target_size: Tuple[int, int],coord=None) -> Tuple[Tensor]:
        """Forward for head part which is called after every decoder layer.

        Args:
            decoder_out (Tensor): in shape (batch_size, num_queries, c).
            mask_feature (Tensor): in shape (batch_size, c, h, w).
            attn_mask_target_size (tuple[int, int]): target attention
                mask size.

        Returns:
            tuple: A tuple contain three elements.

                - cls_pred (Tensor): Classification scores in shape \
                    (batch_size, num_queries, cls_out_channels). \
                    Note `cls_out_channels` should includes background.
                - mask_pred (Tensor): Mask scores in shape \
                    (batch_size, num_queries,h, w).
                - attn_mask (Tensor): Attention mask in shape \
                    (batch_size * num_heads, num_queries, h, w).
        """
        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        # shape (num_queries, batch_size, c)
        cls_pred = self.cls_embed(decoder_out)
        # shape (num_queries, batch_size, c)
        mask_embed = self.mask_embed(decoder_out)
        # shape (num_queries, batch_size, h, w)
        mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)
        
        attn_mask = F.interpolate(
            mask_pred,
            attn_mask_target_size,
            mode='bilinear',
            align_corners=False)
        # shape (num_queries, batch_size, h, w) ->
        #   (batch_size * num_head, num_queries, h, w)
        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat((1, self.num_heads, 1, 1)).flatten(0, 1)
        attn_mask = attn_mask.sigmoid() < 0.5
        attn_mask = attn_mask.detach()
        if coord is not None:
            coord = coord[0,:,0].unsqueeze(0).unsqueeze(0).repeat(attn_mask.shape[0],attn_mask.shape[1],1)
            attn_mask = attn_mask.gather(2,coord)
        # mask_pred =  F.interpolate(mask_pred,mask_feature_size,mode='bilinear',align_corners=False)
        return cls_pred, mask_pred, attn_mask


    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        """Perform forward propagation and loss calculation of the decoder head
        on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.
            train_cfg (ConfigType): Training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components.
        """
        # batch SegDataSample to InstanceDataSample
        batch_gt_instances, batch_img_metas = self._seg_data_to_instance_data(
            batch_data_samples)
        gt_edge_segs = [
            data_sample.gt_edge_map.data for data_sample in batch_data_samples
        ]
        gt_edge_segs = torch.stack(gt_edge_segs, dim=0)
        # ipdb.set_trace()

        # forward
        all_cls_scores, all_mask_preds,edge_out = self(x, batch_data_samples)
        # ipdb.set_trace()
        # with torch.no_grad():
        #     _, all_mask_preds_t = self(x_t, batch_data_samples)
        # ipdb.set_trace()

        # loss
        losses = self.loss_by_feat(all_cls_scores, all_mask_preds,
                                   batch_gt_instances, batch_img_metas,gt_edge_segs,edge_out)

        return losses

    def predict(self, x: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tuple[Tensor]:
        """Test without augmentaton.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_img_metas (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.
            test_cfg (ConfigType): Test config.

        Returns:
            Tensor: A tensor of segmentation mask.
        """
        batch_data_samples = [
            SegDataSample(metainfo=metainfo) for metainfo in batch_img_metas
        ]
        # gt_edge_segs = [
        #     data_sample.gt_edge_map.data for data_sample in batch_data_samples
        # ]
        # gt_edge_segs = torch.stack(gt_edge_segs, dim=0)
        all_cls_scores, all_mask_preds,edge_out = self(x, batch_data_samples)
        # edge_out = torch.sigmoid(edge_out)
        # vis_edge(gt_edge_segs)
        # vis_edge(edge_out)
        # ipdb.set_trace()
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]
        if 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape']
        else:
            size = batch_img_metas[0]['img_shape']
        # upsample mask
        size = batch_img_metas[0]['img_shape']
        mask_pred_results = F.interpolate(
            mask_pred_results, size=size, mode='bilinear', align_corners=False)
        cls_score = F.softmax(mask_cls_results, dim=-1)[..., :-1]
        mask_pred = mask_pred_results.sigmoid()
        seg_logits = torch.einsum('bqc, bqhw->bchw', cls_score, mask_pred)
        return seg_logits


    def forward(self, x: List[Tensor],
                batch_data_samples: SampleList) -> Tuple[List[Tensor]]:
        """Forward function.

        Args:
            x (list[Tensor]): Multi scale Features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            tuple[list[Tensor]]: A tuple contains two elements.

                - cls_pred_list (list[Tensor)]: Classification logits \
                    for each decoder layer. Each is a 3D-tensor with shape \
                    (batch_size, num_queries, cls_out_channels). \
                    Note `cls_out_channels` should includes background.
                - mask_pred_list (list[Tensor]): Mask logits for each \
                    decoder layer. Each with shape (batch_size, num_queries, \
                    h, w).
        """
        batch_size = x[0].shape[0]
        mask_features,edge_out, multi_scale_memorys = self.pixel_decoder(x)
        
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        decoder_coords = []
        k = multi_scale_memorys[0].shape[2]*multi_scale_memorys[0].shape[3]//4
        # ipdb.set_trace()
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            # k = decoder_input.shape[1]//16
            # uncertainty = decoder_input.softmax(dim=-1)
            # uncertainty = torch.sum(uncertainty * torch.log(uncertainty), dim=-1)# B N
            # oversample_rate = 4
            # oversample_coord = torch.topk(uncertainty,k=oversample_rate*k,dim=1)[1] # B 2k
            # # ipdb.set_trace()
            # coord = random_select_from_tensor(oversample_coord, k,oversample_rate*k).unsqueeze(2).expand(batch_size,k,decoder_input.shape[-1])


            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # decoder_input_sample = decoder_input.gather(1,coord)
            decoder_input_sample = decoder_input
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            mask = decoder_input.new_zeros(
                (batch_size, ) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(
                mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2).permute(0, 2, 1)
            # decoder_positional_encoding_sample = decoder_positional_encoding.gather(1,coord)
            decoder_positional_encoding_sample = decoder_positional_encoding
            decoder_inputs.append(decoder_input_sample)
            decoder_positional_encodings.append(decoder_positional_encoding_sample)
            decoder_coords.append(None)
            # decoder_coords.append(coord)
        # shape (num_queries, c) -> (batch_size, num_queries, c)
        query_feat = self.query_feat.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))

        cls_pred_list = []
        mask_pred_list = []
        # ipdb.set_trace()
        mask_features_d =F.interpolate(mask_features,size=(mask_features.shape[2]//2,mask_features.shape[3]//2),mode='bilinear',align_corners=False)
        # mask_features_d=mask_features
        cls_pred, mask_pred, attn_mask = self._forward_head(
            query_feat, mask_features_d, multi_scale_memorys[0].shape[-2:],decoder_coords[0])
        
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level # modify
            # if a mask is all True(all background), then set it all False.
            mask_sum = (attn_mask.sum(-1) != attn_mask.shape[-1]).unsqueeze(-1)
            attn_mask = attn_mask & mask_sum #bh 
            # cross_attn + self_attn
            # layer = self.transformer_decoder.layers[i]
            # query_feat = layer(query_feat,decoder_inputs,level_idx,attn_mask,decoder_positional_encodings,query_embed)


            layer = self.transformer_decoder.layers[i]
            # ipdb.set_trace()
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)
        
            next_size = multi_scale_memorys[(i+1) % self.num_transformer_feat_level].shape[-2:]
            if i == self.num_transformer_decoder_layers-1:
                
                cls_pred, mask_pred, attn_mask = self._forward_head(
                    query_feat,mask_features,next_size,decoder_coords[(i+1) % self.num_transformer_feat_level] ) # modify
            else:
                cls_pred, mask_pred, attn_mask = self._forward_head(
                    query_feat,mask_features_d,next_size,decoder_coords[(i+1) % self.num_transformer_feat_level]  ) # modify
            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)

        return cls_pred_list, mask_pred_list,edge_out

    def _loss_by_feat_single(self, cls_scores: Tensor, mask_preds: Tensor,
                             batch_gt_instances: List[InstanceData],
                             batch_img_metas: List[dict]) -> Tuple[Tensor]:
        """Loss function for outputs from a single decoder layer.

        Args:
            cls_scores (Tensor): Mask score logits from a single decoder layer
                for all images. Shape (batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            mask_preds (Tensor): Mask logits for a pixel decoder for all
                images. Shape (batch_size, num_queries, h, w).
            batch_gt_instances (list[obj:`InstanceData`]): each contains
                ``labels`` and ``masks``.
            batch_img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[Tensor]: Loss components for outputs from a single \
                decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         avg_factor) = self.get_targets(cls_scores_list, mask_preds_list,
                                        batch_gt_instances, batch_img_metas)
        # shape (batch_size, num_queries)
        labels = torch.stack(labels_list, dim=0)
        # shape (batch_size, num_queries)
        label_weights = torch.stack(label_weights_list, dim=0)
        # shape (num_total_gts, h, w)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        # shape (batch_size, num_queries)
        mask_weights = torch.stack(mask_weights_list, dim=0)

        # classfication loss
        # shape (batch_size * num_queries, )
        cls_scores = cls_scores.flatten(0, 1)
        label_pos  = labels[mask_weights > 0] # extract positive_labels
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)

        class_weight = cls_scores.new_tensor(self.class_weight)
        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            avg_factor=class_weight[labels].sum())

        num_total_masks = reduce_mean(cls_scores.new_tensor([avg_factor]))
        num_total_masks = max(num_total_masks, 1)

        mask_preds = mask_preds[mask_weights > 0]

        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            return loss_cls, loss_mask, loss_dice

        with torch.no_grad():
            mask_weits=torch.abs(F.avg_pool2d(mask_targets.unsqueeze(1).float(), kernel_size=5, stride=1, padding=2)-mask_targets.unsqueeze(1).float())
            points_coords = get_uncertain_point_coords_with_randomness_reweight(
                mask_preds.unsqueeze(1), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio,1-mask_weits)
            
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            
            mask_point_targets = point_sample(
                mask_targets.unsqueeze(1).float(), points_coords).squeeze(1)

            mask_point_weits   = point_sample(
                mask_weits,points_coords).squeeze(1)

        mask_point_preds = point_sample(
            mask_preds.unsqueeze(1), points_coords).squeeze(1)

        # dice loss

        loss_dice = self.loss_dice(
            mask_point_preds, mask_point_targets,avg_factor=num_total_masks)
        
        # mask loss
        # shape (num_queries, num_points) -> (num_queries * num_points, )
        mask_point_preds = mask_point_preds.reshape(-1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        mask_point_targets = mask_point_targets.reshape(-1)
        mask_point_weits   = mask_point_weits.reshape(-1)

        loss_mask = self.loss_mask(
            mask_point_preds,
            mask_point_targets,
            weight = 1+5*mask_point_weits,
            avg_factor=num_total_masks * self.num_points)

        return loss_cls, loss_mask, loss_dice#,loss_edge

    def loss_by_feat(self, all_cls_scores: Tensor, all_mask_preds: Tensor,
                     batch_gt_instances: List[InstanceData],
                     batch_img_metas: List[dict],gt_edge,edge_out) -> Dict[str, Tensor]:
        """Loss function.

        Args:
            all_cls_scores (Tensor): Classification scores for all decoder
                layers with shape (num_decoder, batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            all_mask_preds (Tensor): Mask scores for all decoder layers with
                shape (num_decoder, batch_size, num_queries, h, w).
            batch_gt_instances (list[obj:`InstanceData`]): each contains
                ``labels`` and ``masks``.
            batch_img_metas (list[dict]): List of image meta information.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_dec_layers = len(all_cls_scores)
        batch_gt_instances_list = [
            batch_gt_instances for _ in range(num_dec_layers)
        ]
        img_metas_list = [batch_img_metas for _ in range(num_dec_layers)]
        losses_cls, losses_mask, losses_dice = multi_apply(
            self._loss_by_feat_single, all_cls_scores, all_mask_preds,
            batch_gt_instances_list, img_metas_list)

        loss_dict = dict()
        edge_out = F.interpolate(edge_out,size=gt_edge.shape[-2:],mode='bilinear',align_corners=False)

        gt_edge = (gt_edge!=255)
        edge_loss = self.loss_edge(torch.sigmoid(edge_out),gt_edge.float())*10
        mask = gt_edge!=255
        edge_loss = (edge_loss*mask).sum()/mask.sum()

        loss_dict['edge_loss'] = edge_loss
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_mask'] = losses_mask[-1]
        loss_dict['loss_dice'] = losses_dice[-1]

        num_dec_layer = 0
        for loss_cls_i, loss_mask_i, loss_dice_i  in zip(
                losses_cls[:-1], losses_mask[:-1], losses_dice[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
            loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
            # loss_dict[f'd{num_dec_layer}.loss_edge'] = loss_edge_i
            num_dec_layer += 1
        return loss_dict
