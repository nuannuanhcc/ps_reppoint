from __future__ import division

import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
from mmdet import ops
from mmdet.core import force_fp32
from ..registry import ROI_EXTRACTORS


@ROI_EXTRACTORS.register_module
class SingleRoIExtractor(nn.Module):
    """Extract RoI features from a single level feature map.

    If there are mulitple input feature levels, each RoI is mapped to a level
    according to its scale.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (int): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0.
    """

    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 finest_scale=56):
        super(SingleRoIExtractor, self).__init__()
        self.roi_layers = self.build_roi_layers(roi_layer, featmap_strides)
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.finest_scale = finest_scale
        self.fp16_enabled = False

    @property
    def num_inputs(self):
        """int: Input feature map levels."""
        return len(self.featmap_strides)

    def init_weights(self):
        pass

    def build_roi_layers(self, layer_cfg, featmap_strides):
        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')
        assert hasattr(ops, layer_type)
        layer_cls = getattr(ops, layer_type)
        roi_layers = nn.ModuleList(
            [layer_cls(spatial_scale=1 / s, **cfg) for s in featmap_strides])
        return roi_layers

    def map_roi_levels(self, rois, num_levels):
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        scale = torch.sqrt(
            (rois[:, 3] - rois[:, 1] + 1) * (rois[:, 4] - rois[:, 2] + 1))
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls

    def roi_rescale(self, rois, scale_factor):
        cx = (rois[:, 1] + rois[:, 3]) * 0.5
        cy = (rois[:, 2] + rois[:, 4]) * 0.5
        w = rois[:, 3] - rois[:, 1] + 1
        h = rois[:, 4] - rois[:, 2] + 1
        new_w = w * scale_factor
        new_h = h * scale_factor
        x1 = cx - new_w * 0.5 + 0.5
        x2 = cx + new_w * 0.5 - 0.5
        y1 = cy - new_h * 0.5 + 0.5
        y2 = cy + new_h * 0.5 - 0.5
        new_rois = torch.stack((rois[:, 0], x1, y1, x2, y2), dim=-1)
        return new_rois

    @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None):
        if len(feats) == 1:
            return self.roi_layers[0](feats[0], rois)

        out_size = self.roi_layers[0].out_size
        num_levels = len(feats)
        target_lvls = self.map_roi_levels(rois, num_levels)
        roi_feats = feats[0].new_zeros(
            rois.size(0), self.out_channels, *_pair(out_size))
        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)
        for i in range(num_levels):
            inds = target_lvls == i
            if inds.any():
                rois_ = rois[inds, :]
                # roi_feats_t = self.roi_layers[i](feats[i], rois_)
                roi_feats_t = self.roi_point(feats[i], rois_, self.featmap_strides[i], out_size)
                roi_feats[inds] = roi_feats_t
        return roi_feats

    def roi_point(self, feats, rois, s, out_size):

        rois_feat = []
        for i in range(feats.shape[0]):
            inds = rois[:, 0].int() == i
            if inds.any():
                box = rois[inds][:, 1:] / s
                feat = self.stn(feats[i], box, out_size)
                rois_feat.append(feat)
        return torch.cat(rois_feat, dim=0)

    def stn(self, x, box, output_size):
        x = x.squeeze()
        if x.dim() == 3:
            x = x.expand(box.shape[0], -1, -1, -1)
        box_norm = box.clone()
        # normalization the weight and height 0~w/h => -1~1
        # min~max => a~b  x_norm = (b-a)/(max-min)*(x-min)+a
        box_norm[:, (0, 2)] = 2 * (box[:, (0, 2)]) / x.shape[-1] - 1
        box_norm[:, (1, 3)] = 2 * (box[:, (1, 3)]) / x.shape[-2] - 1
        # calculate the affine parameter
        theta = torch.zeros((x.shape[0], 6)).cuda()
        theta[:, 0] = (box_norm[:, 2] - box_norm[:, 0]) / 2
        theta[:, 2] = (box_norm[:, 2] + box_norm[:, 0]) / 2
        theta[:, 4] = (box_norm[:, 3] - box_norm[:, 1]) / 2
        theta[:, 5] = (box_norm[:, 3] + box_norm[:, 1]) / 2
        theta = theta.view(-1, 2, 3)

        new_size = torch.Size([*x.shape[:2], *output_size])
        grid = torch.nn.functional.affine_grid(theta, new_size)

        x = torch.nn.functional.grid_sample(x, grid)
        return x

    # def stn(self, x, box, output_size):
    #     x = x.squeeze()
    #     if x.dim() == 3:
    #         x = x.expand(box.shape[0], -1, -1, -1)
    #     box_norm = box.clone()
    #     # normalization the weight and height 0~w/h => -1~1
    #     # min~max => a~b  x_norm = (b-a)/(max-min)*(x-min)+a
    #     box_norm[:, (0, 2)] = 2 * (box[:, (0, 2)]) / x.shape[-1] - 1
    #     box_norm[:, (1, 3)] = 2 * (box[:, (1, 3)]) / x.shape[-2] - 1
    #
    #     len_w = box_norm[:, 2] - box_norm[:, 0]
    #     len_h = box_norm[:, 3] - box_norm[:, 1]
    #     grid = []
    #     for i in range(len(box_norm)):
    #         if output_size[1] == 1:  # h*w
    #             shift_x = (box_norm[i, 0] + box_norm[i, 1]) / 2
    #         else:
    #             shift_x = box_norm[i, 0] + len_w[i] / (output_size[1] - 1) * torch.arange(0., output_size[1])
    #
    #         if output_size[0] == 1:
    #             shift_y = (box_norm[i, 1] + box_norm[i, 3]) / 2
    #         else:
    #             shift_y = box_norm[i, 1] + len_h[i] / (output_size[0] - 1) * torch.arange(0., output_size[0])
    #
    #         shift_yy, shift_xx = torch.meshgrid(shift_y, shift_x)
    #         grid.append(torch.stack([shift_xx, shift_yy], dim=-1).unsqueeze(0).cuda())
    #     grid = torch.cat(grid, dim=0)
    #     x = torch.nn.functional.grid_sample(x, grid)
    #     return x
