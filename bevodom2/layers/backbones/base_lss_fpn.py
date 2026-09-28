"""Monocular LSS backbone: image features -> depth distribution and context -> voxel pooling to BEV.

Adapted from BEVDepth (https://github.com/Megvii-BaseDetection/BEVDepth),
Copyright (c) Megvii Inc., released under the MIT License.

Example:
    backbone = BaseLSSFPN(**backbone_conf)
    bev_feat, depth, img_feat, geom_xyz = backbone(imgs, mats_dict)
"""
import torch
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer
from mmdet3d.models import build_neck
from mmdet.models import build_backbone
from mmdet.models.backbones.resnet import BasicBlock
from torch import nn

from bevodom2.ops.voxel_pooling_train import voxel_pooling_train

__all__ = ['BaseLSSFPN']


class _ASPPModule(nn.Module):
    """Single atrous convolution branch of ASPP."""

    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super().__init__()
        self.atrous_conv = nn.Conv2d(
            inplanes, planes,
            kernel_size=kernel_size, stride=1,
            padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()
        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    """Atrous spatial pyramid pooling."""

    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super().__init__()
        dilations = [1, 6, 12, 18]
        self.aspp1 = _ASPPModule(
            inplanes, mid_channels, 1,
            padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(
            inplanes, mid_channels, 3,
            padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(
            inplanes, mid_channels, 3,
            padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(
            inplanes, mid_channels, 3,
            padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(mid_channels * 5, mid_channels, 1, bias=False)
        self.bn1 = BatchNorm(mid_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(
            x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Mlp(nn.Module):
    """Two-layer MLP."""

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.ReLU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SELayer(nn.Module):
    """Squeeze-and-excitation conditioned on camera parameters."""

    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class DepthNet(nn.Module):
    """Camera-aware prediction of the depth distribution and context features."""

    def __init__(self, in_channels, mid_channels, context_channels,
                 depth_channels):
        super().__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.context_conv = nn.Conv2d(
            mid_channels, context_channels, 1, stride=1, padding=0)
        self.bn = nn.BatchNorm1d(27)
        self.depth_mlp = Mlp(27, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)
        self.context_mlp = Mlp(27, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)
        self.depth_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            ASPP(mid_channels, mid_channels),
            build_conv_layer(cfg=dict(
                type='DCN',
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3, padding=1,
                groups=4, im2col_step=128,
            )),
            # Output layer: one channel per depth bin defined by d_bound
            nn.Conv2d(mid_channels, depth_channels, 1, stride=1, padding=0),
        )

    def forward(self, x, mats_dict):
        """x: (B, C, h, w) image features; returns concatenated [depth logits, context features]."""
        intrins = mats_dict['intrin_mats'][:, 0:1, ..., :3, :3]
        batch_size = intrins.shape[0]
        num_cams = intrins.shape[2]
        ida = mats_dict['ida_mats'][:, 0:1, ...]
        sensor2ego = mats_dict['sensor2ego_mats'][:, 0:1, ..., :3, :]
        bda = mats_dict['bda_mat'].view(
            batch_size, 1, 1, 4, 4).repeat(1, 1, num_cams, 1, 1)

        # 27-D camera input: [fx, fy, cx, cy, ida(6), bda(5), sensor2ego(12)]
        mlp_input = torch.cat([
            torch.stack([
                intrins[:, 0:1, ..., 0, 0],
                intrins[:, 0:1, ..., 1, 1],
                intrins[:, 0:1, ..., 0, 2],
                intrins[:, 0:1, ..., 1, 2],
                ida[:, 0:1, ..., 0, 0],
                ida[:, 0:1, ..., 0, 1],
                ida[:, 0:1, ..., 0, 3],
                ida[:, 0:1, ..., 1, 0],
                ida[:, 0:1, ..., 1, 1],
                ida[:, 0:1, ..., 1, 3],
                bda[:, 0:1, ..., 0, 0],
                bda[:, 0:1, ..., 0, 1],
                bda[:, 0:1, ..., 1, 0],
                bda[:, 0:1, ..., 1, 1],
                bda[:, 0:1, ..., 2, 2],
            ], dim=-1),
            sensor2ego.view(batch_size, 1, num_cams, -1),
        ], dim=-1)

        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))

        x = self.reduce_conv(x)

        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)
        context = self.context_conv(context)

        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)
        depth = self.depth_conv(depth)

        return torch.cat([depth, context], dim=1)


class BaseLSSFPN(nn.Module):
    """Monocular LSS backbone (ResNet + SECONDFPN + DepthNet + voxel pooling).

    Args:
        x_bound / y_bound / z_bound: BEV voxel range [min, max, step] in meters.
        d_bound: depth bins [min, max, step] in meters.
        final_dim: input image size (H, W).
        downsample_factor: stride of the image features w.r.t. the input image.
        output_channels: number of context (BEV) feature channels.
        img_backbone_conf / img_neck_conf / depth_net_conf: sub-network configs.
    """

    def __init__(self, x_bound, y_bound, z_bound, d_bound, final_dim,
                 downsample_factor, output_channels, img_backbone_conf,
                 img_neck_conf, depth_net_conf):
        super().__init__()
        self.downsample_factor = downsample_factor
        self.d_bound = d_bound
        self.final_dim = final_dim
        self.output_channels = output_channels

        self.register_buffer(
            'voxel_size',
            torch.Tensor([row[2] for row in [x_bound, y_bound, z_bound]]))
        self.register_buffer(
            'voxel_coord',
            torch.Tensor([row[0] + row[2] / 2.0
                          for row in [x_bound, y_bound, z_bound]]))
        self.register_buffer(
            'voxel_num',
            torch.LongTensor([(row[1] - row[0]) / row[2]
                              for row in [x_bound, y_bound, z_bound]]))

        self.register_buffer('frustum', self.create_frustum())
        self.depth_channels, _, _, _ = self.frustum.shape

        self.img_backbone = build_backbone(img_backbone_conf)
        self.img_neck = build_neck(img_neck_conf)
        self.depth_net = DepthNet(
            depth_net_conf['in_channels'],
            depth_net_conf['mid_channels'],
            self.output_channels,
            self.depth_channels,
        )

        self.img_neck.init_weights()
        self.img_backbone.init_weights()

    def create_frustum(self):
        """Frustum grid in image coordinates, (D, fH, fW, 4) with points (x, y, d, 1)."""
        ogfH, ogfW = self.final_dim
        fH = ogfH // self.downsample_factor
        fW = ogfW // self.downsample_factor

        d_coords = torch.arange(
            *self.d_bound, dtype=torch.float
        ).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = d_coords.shape

        x_coords = torch.linspace(
            0, ogfW - 1, fW, dtype=torch.float
        ).view(1, 1, fW).expand(D, fH, fW)
        y_coords = torch.linspace(
            0, ogfH - 1, fH, dtype=torch.float
        ).view(1, fH, 1).expand(D, fH, fW)
        paddings = torch.ones_like(d_coords)

        return torch.stack((x_coords, y_coords, d_coords, paddings), -1)

    def get_geometry(self, sensor2ego_mat, intrin_mat, ida_mat, bda_mat):
        """Transform frustum points to the ego frame; returns (B, N, D, fH, fW, 3)."""
        batch_size, num_cams, _, _ = sensor2ego_mat.shape

        # Undo image-space augmentation
        points = self.frustum
        ida_mat = ida_mat.view(batch_size, num_cams, 1, 1, 1, 4, 4)
        points = ida_mat.inverse().matmul(points.unsqueeze(-1))

        # Scale x/y by depth, then map camera coordinates to the ego frame
        points = torch.cat(
            (points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
             points[:, :, :, :, :, 2:]), dim=5)

        combine = sensor2ego_mat.matmul(torch.inverse(intrin_mat))
        points = combine.view(
            batch_size, num_cams, 1, 1, 1, 4, 4).matmul(points)

        # BEV-space augmentation
        bda_mat = bda_mat.unsqueeze(1).repeat(
            1, num_cams, 1, 1
        ).view(batch_size, num_cams, 1, 1, 1, 4, 4)
        points = (bda_mat @ points).squeeze(-1)

        return points[..., :3]

    def splat(self, feat, depth, geom_xyz):
        """Outer product of depth distribution and features, voxel-pooled to BEV. feat: (B, C, h, w), depth: (B, D, h, w)."""
        feat_with_depth = depth.unsqueeze(1) * feat.unsqueeze(2)       # (B, C, D, h, w)
        feat_with_depth = feat_with_depth.unsqueeze(1).permute(0, 1, 3, 4, 5, 2)  # (B, 1, D, h, w, C)
        bev = voxel_pooling_train(
            geom_xyz, feat_with_depth.contiguous(), self.voxel_num.cuda())
        return bev.contiguous()

    def forward(self, imgs, mats_dict):
        """imgs: (B, 3, H, W) monocular images.

        Returns:
            bev_feat: (B, C, X, Y) BEV features.
            depth: (B, D, h, w) depth distribution (softmax).
            img_feat: (B, C_img, h, w) FPN image features (used by the PV branch).
            geom_xyz: (B, 1, D, h, w, 3) voxel indices of the frustum points.
        """
        img_feat = self.img_neck(self.img_backbone(imgs))[0]
        depth_feature = self.depth_net(img_feat, mats_dict)
        depth = depth_feature[:, :self.depth_channels].softmax(
            dim=1, dtype=depth_feature.dtype)

        geom_xyz = self.get_geometry(
            mats_dict['sensor2ego_mats'][:, 0, ...],
            mats_dict['intrin_mats'][:, 0, ...],
            mats_dict['ida_mats'][:, 0, ...],
            mats_dict['bda_mat'],
        )
        geom_xyz = ((geom_xyz - (self.voxel_coord - self.voxel_size / 2.0))
                    / self.voxel_size).int()

        context = depth_feature[
            :, self.depth_channels:self.depth_channels + self.output_channels]
        bev_feat = self.splat(context, depth, geom_xyz)
        return bev_feat, depth, img_feat, geom_xyz
