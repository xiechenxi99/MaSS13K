# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log
from torch import Tensor
import time
from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor
import ipdb
import math
import torchvision
import torch
import torch.nn.functional as F

def rotate_and_crop_tensor(tensor, angle):
    # 获取输入张量的形状
    B, C, H, W = tensor.shape

    # 计算旋转矩阵
    theta = math.radians(angle)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    rotation_matrix = torch.tensor([[cos_theta, -sin_theta],
                                    [sin_theta, cos_theta]], dtype=tensor.dtype, device=tensor.device)

    # 将旋转矩阵的形状调整为 (B, 2, 3)
    rotation_matrix = torch.cat([rotation_matrix.unsqueeze(0).repeat(B, 1, 1), torch.zeros(B, 2, 1, dtype=tensor.dtype, device=tensor.device)], dim=2)

    # 创建采样网格
    grid = F.affine_grid(rotation_matrix, tensor.size(), align_corners=True)

    # 使用双线性插值进行旋转
    rotated_tensor = F.grid_sample(tensor, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    # 计算旋转后图像的有效区域大小和位置
    angle_rad = math.radians(angle)
    width_rotated = abs(W * math.cos(angle_rad)) + abs(H * math.sin(angle_rad))
    height_rotated = abs(W * math.sin(angle_rad)) + abs(H * math.cos(angle_rad))
    x_offset = int((W - width_rotated) / 2)
    y_offset = int((H - height_rotated) / 2)

    # 从旋转后的图像中裁剪出最大的矩形区域
    cropped_tensor = rotated_tensor[:, :, y_offset:y_offset+int(height_rotated), x_offset:x_offset+int(width_rotated)]

    return cropped_tensor

@MODELS.register_module()
class EncoderDecoder_multi(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.

    1. The ``loss`` method is used to calculate the loss of model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2) Call the decode head loss function to forward decode head model and
    calculate losses.

    .. code:: text

     loss(): extract_feat() -> _decode_head_forward_train() -> _auxiliary_head_forward_train (optional)
     _decode_head_forward_train(): decode_head.loss()
     _auxiliary_head_forward_train(): auxiliary_head.loss (optional)

    2. The ``predict`` method is used to predict segmentation results,
    which includes two steps: (1) Run inference function to obtain the list of
    seg_logits (2) Call post-processing function to obtain list of
    ``SegDataSample`` including ``pred_sem_seg`` and ``seg_logits``.

    .. code:: text

     predict(): inference() -> postprocess_result()
     inference(): whole_inference()/slide_inference()
     whole_inference()/slide_inference(): encoder_decoder()
     encoder_decoder(): extract_feat() -> decode_head.predict()

    3. The ``_forward`` method is used to output the tensor by running the model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2)Call the decode head forward function to forward decode head model.

    .. code:: text

     _forward(): extract_feat() -> _decode_head.forward()

    Args:

        backbone (ConfigType): The config for the backnone of segmentor.
        decode_head (ConfigType): The config for the decode head of segmentor.
        neck (OptConfigType): The config for the neck of segmentor.
            Defaults to None.
        auxiliary_head (OptConfigType): The config for the auxiliary head of
            segmentor. Defaults to None.
        train_cfg (OptConfigType): The config for training. Defaults to None.
        test_cfg (OptConfigType): The config for testing. Defaults to None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
        pretrained (str, optional): The path for pretrained model.
            Defaults to None.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.
    """  # noqa: E501

    def __init__(self,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None,
                 ratio: List[int]=[1,0]):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.ratio = ratio

        assert self.with_decode_head

    def _init_decode_head(self, decode_head: ConfigType) -> None:
        """Initialize ``decode_head``"""
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def _init_auxiliary_head(self, auxiliary_head: ConfigType) -> None:
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)
    
    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        # print((inputs))
        # inputs = self.image_to_batch(inputs,self.ratio)
        # x1 = self.backbone(inputs,droplast=True)
        x1 = self.backbone(inputs)
        # ipdb.set_trace()
        # x2 = self.slide_extract(F.interpolate(inputs,scale_factor=0.5,mode='bilinear',align_corners=False))
        # with torch.no_grad():
        #     x2 = self.backbone(F.interpolate(inputs,scale_factor=0.5,mode='bilinear',align_corners=False))
        
        # x = zip(x1, x2)
        # x = list(x)

        # x.append((None,x2[-1]))
        # ipdb.set_trace()
        return x1
    def image_to_batch(self, inputs: Tensor, ratio: List[int]) -> Tensor:
        B,C,H,W = inputs.shape
        patch=[]
        split_height = H//(2**max(ratio))
        split_width = W//(2**max(ratio))
        for j in ratio: # 1 0
            scale = 2**(j-max(ratio))
            if scale!=1.0:
                inputs = F.interpolate(inputs,scale_factor=scale,mode='bilinear',align_corners=False)
            x = inputs.view(B,C,split_height,2**j,split_width,2**j).permute(0,3,5,1,2,4).contiguous() # B nh nw C h w
            patch.append(x.view(-1,C,split_height,split_width))
        inputs = torch.cat(patch,dim=0)
        return inputs
        
    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(inputs)
        seg_logits = self.decode_head.predict(x, batch_img_metas,
                                              self.test_cfg)

        return seg_logits

    def _decode_head_forward_train(self, inputs: List[Tensor],
                                   data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.loss(inputs, data_samples,
                                            self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _auxiliary_head_forward_train(self, inputs: List[Tensor],
                                      data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(inputs, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(inputs, data_samples,
                                                self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(inputs)
        # visualize_features_single(x[0][0],'/home/notebook/data/personal/S9055029/mmsegmentation/vis','input.png')
        # inputs_t = rotate_and_crop_tensor(inputs,5)
        # inputs_t = torchvision.transforms.functional.rotate(img=inputs, angle=5, interpolation= InterpolationMode.BILINEAR)
        # x_t = self.extract_feat(inputs_t)
        # visualize_features_single(x_t[0][0],'/home/notebook/data/personal/S9055029/mmsegmentation/vis','input_T.png')
        # ipdb.set_trace()

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)

        return losses

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`], optional): The seg data
                samples. It usually includes information such as `metainfo`
                and `gt_sem_seg`.

        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]
        # start_time = time.perf_counter()
        seg_logits = self.inference(inputs, batch_img_metas)
        # print(seg_logits.shape)
        # end_time = time.perf_counter()
        # inference_time = end_time - start_time
        # print(inference_time)
        return self.postprocess_result(seg_logits, data_samples)

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        x = self.extract_feat(inputs)
        return self.decode_head.forward(x,data_samples)
    def slide_extract(self, inputs: Tensor,droplast=False) -> Tensor:
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """
        batch_size, _, h_img, w_img = inputs.size()
        h_stride, w_stride = h_img//8*3,w_img//8*3
        h_crop, w_crop = h_img//8*5,w_img//8*5

        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = []
        count_mat=[]
        for i in range(4):
            preds.append(inputs.new_zeros((batch_size, 256*(2**i), h_img//(4*(2**i)), w_img//(4*(2**i)))))
            count_mat.append(inputs.new_zeros((batch_size, 1, h_img//(4*(2**i)), w_img//(4*(2**i)))))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.backbone(crop_img,droplast)
                # ipdb.set_trace()
                for i in range(len(crop_seg_logit)):
                    y1 = h_idx * h_stride // (4*(2**i))
                    x1 = w_idx * w_stride // (4*(2**i))
                    y2 = min(y1 + h_crop // (4*(2**i)) , h_img // (4*(2**i)) )
                    x2 = min(x1 + w_crop // (4*(2**i)) , w_img // (4*(2**i)))
                    y1 = max(y2 - h_crop // (4*(2**i)), 0)
                    x1 = max(x2 - w_crop // (4*(2**i)), 0)
                    preds[i] += F.pad(crop_seg_logit[i],
                                (int(x1), int(preds[i].shape[3] - x2), int(y1),
                                    int(preds[i].shape[2] - y2)))

                    count_mat[i][:, :, y1:y2, x1:x2] += 1
                    # ipdb.set_trace()
        for i in range(len(crop_seg_logit)):  
            assert (count_mat[i] == 0).sum() == 0
        seg_logits=[]
        for i in range(len(crop_seg_logit)):
            seg_logits.append(preds[i] / count_mat[i])

        return tuple(seg_logits)
    def slide_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                # change the image shape to patch shape
                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                # the output of encode_decode is seg logits tensor map
                # with shape [N, C, H, W]
                crop_seg_logit = self.encode_decode(crop_img, batch_img_metas)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        return seg_logits

    def whole_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference with full image.

        Args:
            inputs (Tensor): The tensor should have a shape NxCxHxW, which
                contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        seg_logits = self.encode_decode(inputs, batch_img_metas)

        return seg_logits

    def inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """Inference with slide/whole style.

        Args:
            inputs (Tensor): The input image of shape (N, 3, H, W).
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', 'pad_shape', and 'padding_size'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """
        assert self.test_cfg.get('mode', 'whole') in ['slide', 'whole'], \
            f'Only "slide" or "whole" test mode are supported, but got ' \
            f'{self.test_cfg["mode"]}.'
        ori_shape = batch_img_metas[0]['ori_shape']
        if not all(_['ori_shape'] == ori_shape for _ in batch_img_metas):
            print_log(
                'Image shapes are different in the batch.',
                logger='current',
                level=logging.WARN)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(inputs, batch_img_metas)
        else:
            seg_logit = self.whole_inference(inputs, batch_img_metas)

        return seg_logit

    def aug_test(self, inputs, batch_img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(inputs[0], batch_img_metas[0], rescale)
        for i in range(1, len(inputs)):
            cur_seg_logit = self.inference(inputs[i], batch_img_metas[i],
                                           rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(inputs)
        seg_pred = seg_logit.argmax(dim=1)
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
