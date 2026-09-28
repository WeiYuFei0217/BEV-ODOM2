"""BEV-ODOM2 network: monocular LSS backbone, BEV branch (dense rigid BEV flow + 3-DoF pose)
and PV branch (auxiliary 5-DoF pose).

Example:
    net = BEVODOM2(cfg['backbone_conf'], cfg['dataset_type'], **cfg['model_conf']).cuda()
    out = net(imgs, mats_dict)    # imgs: (2B, 3, H, W), rows 2i / 2i+1 are frames t / t+1
    out.R, out.t                   # 3-DoF relative pose, (B, 3, 3) / (B, 3, 1)
"""
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from spatial_correlation_sampler import SpatialCorrelationSampler

from bevodom2.layers.backbones.base_lss_fpn import BaseLSSFPN
from bevodom2.models.pose_losses import pose_loss_3dof, pose_loss_5dof

__all__ = ['BEVODOM2', 'ModelOutput']

# R/t: 3-DoF pose of the BEV branch; flow: predicted rigid BEV flow; pv_R/pv_t: 5-DoF pose of the PV branch
ModelOutput = namedtuple('ModelOutput', ['R', 't', 'flow', 'pv_R', 'pv_t'])

HEAD_CONV_STRIDE = 2  # stride of the first convolution of each PoseHead (BEV and PV branches)


def conv_stack(channels, use_bn):
    """Stack of 3x3 convolutions (stride 2 for the first, 1 otherwise), each followed by BN+LeakyReLU or ReLU."""
    layers = []
    for i, (c_in, c_out) in enumerate(zip(channels[:-1], channels[1:])):
        stride = HEAD_CONV_STRIDE if i == 0 else 1
        layers.append(nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, padding_mode='replicate'))
        if use_bn:
            layers += [nn.BatchNorm2d(c_out), nn.LeakyReLU(0.01, inplace=True)]
        else:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class PoseHead(nn.Module):
    """Pose regression head: conv stack -> FC(512) -> rotation / translation FC(64) branches -> tanh."""

    def __init__(self, conv_channels, in_features, rot_dim, trans_dim, use_bn):
        super().__init__()
        self.conv = conv_stack(conv_channels, use_bn)
        self.fc = nn.Linear(in_features, 512)
        self.fc_rot = nn.Linear(512, 64)
        self.fc_trans = nn.Linear(512, 64)
        self.out_rot = nn.Linear(64, rot_dim)
        self.out_trans = nn.Linear(64, trans_dim)

    def forward(self, x):
        x = torch.flatten(self.conv(x), 1)
        x = F.leaky_relu(self.fc(x), 0.01)
        rot = torch.tanh(self.out_rot(F.leaky_relu(self.fc_rot(x), 0.01)))
        trans = torch.tanh(self.out_trans(F.leaky_relu(self.fc_trans(x), 0.01)))
        return rot, trans


def quaternion_to_matrix(q):
    """(B, 4) unit quaternion (qx, qy, qz, qw) -> (B, 3, 3) rotation matrix."""
    qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.zeros((q.shape[0], 3, 3), device=q.device)
    R[:, 0, 0] = 1 - 2 * (qy**2 + qz**2)
    R[:, 0, 1] = 2 * (qx * qy - qz * qw)
    R[:, 0, 2] = 2 * (qx * qz + qy * qw)
    R[:, 1, 0] = 2 * (qx * qy + qz * qw)
    R[:, 1, 1] = 1 - 2 * (qx**2 + qz**2)
    R[:, 1, 2] = 2 * (qy * qz - qx * qw)
    R[:, 2, 0] = 2 * (qx * qz - qy * qw)
    R[:, 2, 1] = 2 * (qy * qz + qx * qw)
    R[:, 2, 2] = 1 - 2 * (qx**2 + qy**2)
    return R


class BEVODOM2(nn.Module):
    """BEV-ODOM2 dual-branch pose network.

    Args:
        backbone_conf: BaseLSSFPN config (x/y/z/d_bound, final_dim, ...).
        dataset_type: 'NCLT' or 'oxford'; selects the BEV half-plane the camera faces.
        corr_patch_size: side length of the local correlation window.
        use_leakyrelu_bn: BN+LeakyReLU (True) or ReLU (False) in the pose heads and FlowUNet.
        max_dis: translation output bound in meters (translation = tanh(.) * max_dis).
        lambda_flow / alpha / beta: loss weights lambda_2, alpha (3-DoF rotation), beta (5-DoF rotation).
    """

    def __init__(self, backbone_conf, dataset_type, corr_patch_size, use_leakyrelu_bn,
                 max_dis, lambda_flow=0.1, alpha=10, beta=10):
        super().__init__()
        if dataset_type not in ('NCLT', 'oxford'):
            raise ValueError(f"dataset_type must be 'NCLT' or 'oxford', got {dataset_type!r}")
        self.dataset_type = dataset_type
        self.max_dis = max_dis
        self.lambda_flow = lambda_flow
        self.alpha = alpha
        self.beta = beta

        # BEV grid: bev_size x bev_size cells of bev_res meters. The half facing the camera is cropped to
        # bev_size/2 squared; after correlation, the bev_size/4 square next to the vehicle is kept
        # (region of flow prediction and supervision).
        x_min, x_max, self.bev_res = backbone_conf['x_bound']
        self.bev_size = int(round((x_max - x_min) / self.bev_res))
        near_size = self.bev_size // 4
        corr_channels = corr_patch_size ** 2

        self.backbone = BaseLSSFPN(**backbone_conf)
        self.correlation_sampler = SpatialCorrelationSampler(
            kernel_size=1, patch_size=corr_patch_size,
            stride=1, padding=0, dilation=1, dilation_patch=1)

        # BEV branch: [BEV correlation, PV correlation splatted to BEV] -> FlowUNet -> decoder features -> 3-DoF pose
        self.flow_net = FlowUNet(corr_channels * 2, use_leakyrelu_bn=use_leakyrelu_bn)
        bev_feat_size = near_size // HEAD_CONV_STRIDE
        self.bev_head = PoseHead([32, 32, 8], 8 * bev_feat_size ** 2,
                                 rot_dim=2, trans_dim=2, use_bn=use_leakyrelu_bn)

        # PV branch: image feature correlation -> 5-DoF pose (quaternion + 3D translation)
        img_h, img_w = backbone_conf['final_dim']
        feat_h = img_h // backbone_conf['downsample_factor']
        feat_w = img_w // backbone_conf['downsample_factor']
        pv_feat_numel = 8 * -(-feat_h // HEAD_CONV_STRIDE) * -(-feat_w // HEAD_CONV_STRIDE)
        self.pv_head = PoseHead([corr_channels, 64, 32, 8], pv_feat_numel,
                                rot_dim=4, trans_dim=3, use_bn=use_leakyrelu_bn)

    def crop_view(self, x):
        """Keep the laterally centred half of the BEV grid that faces the camera (bev_size -> bev_size/2)."""
        W = x.shape[-1]
        x = x[..., W // 4:W * 3 // 4]
        H = x.shape[-2]
        return x[..., :H // 2, :] if self.dataset_type == 'NCLT' else x[..., H // 2:, :]

    def crop_near(self, x):
        """Keep the laterally centred region of a crop_view output that is closest to the vehicle (side halved)."""
        H, W = x.shape[-2:]
        x = x[..., W // 4:W * 3 // 4]
        return x[..., H // 2:, :] if self.dataset_type == 'NCLT' else x[..., :H // 2, :]

    def forward(self, imgs, mats_dict):
        """imgs: (2B, 3, H, W); rows 2i / 2i+1 are frames t / t+1 of pair i."""
        bev_feat, depth, img_feat, geom_xyz = self.backbone(imgs, mats_dict)
        B = imgs.shape[0] // 2

        # PV branch: correlation between the image features of t and t+1
        img_feat = img_feat.reshape(B, 2, *img_feat.shape[1:])
        pv_corr = self.correlation_sampler(img_feat[:, 0].contiguous(), img_feat[:, 1].contiguous())
        pv_corr = pv_corr.reshape(B, -1, *pv_corr.shape[-2:])
        depth_t = depth.reshape(B, 2, *depth.shape[1:])[:, 0]
        pv_bev = self.backbone.splat(pv_corr, depth_t, geom_xyz[::2].contiguous())

        pv_rot, pv_trans = self.pv_head(pv_corr)
        pv_R = quaternion_to_matrix(F.normalize(pv_rot, p=2, dim=1))
        pv_t = (pv_trans * self.max_dis).reshape(B, 3, 1)

        # BEV branch: correlation between the BEV features of t and t+1, concatenated with the splatted PV correlation
        bev_feat = self.crop_view(torch.rot90(bev_feat, k=1, dims=[2, 3]))
        pv_bev = self.crop_view(pv_bev)
        bev_feat = bev_feat.reshape(B, 2, *bev_feat.shape[1:])
        bev_corr = self.correlation_sampler(bev_feat[:, 0].contiguous(), bev_feat[:, 1].contiguous())
        bev_corr = self.crop_near(bev_corr)
        bev_corr = bev_corr.reshape(B, -1, *bev_corr.shape[-2:])
        fused = torch.cat((bev_corr, self.crop_near(pv_bev)), dim=1)

        flow, dec_feat = self.flow_net(fused)
        rot, trans = self.bev_head(dec_feat)
        rot = F.normalize(rot, p=2, dim=1)
        trans = trans * self.max_dis
        cos_z, sin_z = rot[:, 0], rot[:, 1]
        zeros, ones = torch.zeros_like(cos_z), torch.ones_like(cos_z)
        R = torch.stack([cos_z, -sin_z, zeros,
                         sin_z, cos_z, zeros,
                         zeros, zeros, ones], dim=-1).reshape(-1, 3, 3)
        t = torch.cat([trans, torch.zeros(B, 1, device=trans.device)], dim=1).reshape(B, 3, 1)

        return ModelOutput(R, t, flow, pv_R, pv_t)

    def compute_flow_gt(self, T_planar):
        """Dense rigid BEV flow ground truth from the planar relative pose (B, 4, 4).

        Each cell (u, v) is mapped to vehicle coordinates p = [(o_y - v) r, (u - o_x) r, 0, 1],
        transformed by T_rel and projected back; the flow is the pixel displacement, cropped to
        the region of the flow prediction. Returns (B, 2, h, w).
        """
        device = T_planar.device
        B = T_planar.shape[0]
        H = W = self.bev_size
        res = self.bev_res
        off = (self.bev_size / 2, self.bev_size / 2)

        u, v = torch.meshgrid(
            torch.arange(W, device=device, dtype=torch.float32),
            torch.arange(H, device=device, dtype=torch.float32),
            indexing='xy')
        X = (off[1] - v) * res
        Y = (u - off[0]) * res
        pts = torch.stack([
            X.flatten(), Y.flatten(),
            torch.zeros(H * W, device=device),
            torch.ones(H * W, device=device),
        ], dim=0)
        pts = pts.unsqueeze(0).repeat(B, 1, 1).float()
        pts_tgt = torch.bmm(T_planar.float(), pts)

        u_prime = off[0] + pts_tgt[:, 1, :] / res
        v_prime = off[1] - pts_tgt[:, 0, :] / res
        u_flat = u.flatten().unsqueeze(0).repeat(B, 1)
        v_flat = v.flatten().unsqueeze(0).repeat(B, 1)

        flow = torch.stack([
            (u_prime - u_flat).reshape(B, H, W),
            (v_prime - v_flat).reshape(B, H, W),
        ], dim=1)
        return self.crop_near(self.crop_view(flow))

    @staticmethod
    def flow_loss(flow_pred, flow_gt):
        """Flow loss: end-point error averaged over all cells and samples."""
        return torch.norm(flow_pred - flow_gt, p=2, dim=1).mean()

    def bev_loss(self, R, t, T_planar, flow_pred, flow_gt):
        """3-DoF pose loss (planar target) + lambda_2 * flow loss."""
        loss, logs = pose_loss_3dof(R, t, T_planar, alpha=self.alpha)
        flow_loss = self.flow_loss(flow_pred, flow_gt)
        logs['flow_loss'] = flow_loss
        return loss + self.lambda_flow * flow_loss, logs

    def pv_loss(self, R, t, T_rel):
        """5-DoF PV pose loss (full 3D target)."""
        return pose_loss_5dof(R, t, T_rel, beta=self.beta)


class DoubleConv(nn.Module):
    """Two 3x3 convolutions with optional BN."""

    def __init__(self, in_ch, out_ch, use_bn=True, use_leakyrelu=False):
        super().__init__()
        act = nn.LeakyReLU(0.01, True) if use_leakyrelu else nn.ReLU(True)
        layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(act)
        layers.append(nn.Conv2d(out_ch, out_ch, 3, padding=1))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(act)
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class DoubleConvFirst(nn.Module):
    """Input block: three 3x3 convolutions (expand, then reduce channels)."""

    def __init__(self, in_ch, out_ch, use_leakyrelu=False):
        super().__init__()
        act = nn.LeakyReLU(0.01, True) if use_leakyrelu else nn.ReLU(True)
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch * 2, 3, padding=1), nn.BatchNorm2d(in_ch * 2), act,
            nn.Conv2d(in_ch * 2, in_ch, 3, padding=1), nn.BatchNorm2d(in_ch), act,
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), act,
        )

    def forward(self, x):
        return self.double_conv(x)


class FlowUNet(nn.Module):
    """UNet for rigid BEV flow; returns (flow, last decoder features used by the pose head).

    use_leakyrelu_bn=True: LeakyReLU and no BN in the decoder; False: ReLU and BN in the decoder.
    """

    def __init__(self, in_channels, use_leakyrelu_bn=False):
        super().__init__()
        lk = use_leakyrelu_bn

        self.inc = DoubleConvFirst(in_channels, 32, use_leakyrelu=lk)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64, True, lk))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128, True, lk))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256, True, lk))

        dec_bn = not use_leakyrelu_bn
        self.up1 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.conv_up1 = DoubleConv(384, 128, use_bn=dec_bn, use_leakyrelu=lk)
        self.up2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.conv_up2 = DoubleConv(192, 64, use_bn=dec_bn, use_leakyrelu=lk)
        self.up3 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.conv_up3 = DoubleConv(96, 32, use_bn=dec_bn, use_leakyrelu=lk)

        self.outc = nn.Conv2d(32, 2, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = torch.cat([self.up1(x4), x3], dim=1)
        x = self.conv_up1(x)
        x = torch.cat([self.up2(x), x2], dim=1)
        x = self.conv_up2(x)
        x = torch.cat([self.up3(x), x1], dim=1)
        x = self.conv_up3(x)

        return self.outc(x), x
