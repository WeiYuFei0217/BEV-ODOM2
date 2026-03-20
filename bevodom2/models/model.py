"""
model.py - BEV-ODOM2 Model Definition

Contains BaseBEVODOM2 (PV-BEV fusion odometry model), FlowUNet (optical flow
estimation network), and DepthLoss.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp.autocast_mode import autocast
from spatial_correlation_sampler import SpatialCorrelationSampler

from bevodom2.layers.backbones.base_lss_fpn import BaseLSSFPN
from bevodom2.models.pose_losses import supervised_loss, supervised_loss_6dof

__all__ = ['BaseBEVODOM2']

# PVFlow FC1 input dimensions per dataset
_PVFLOW_FC1_IN = {
    "NCLT": 672,
    "oxford": 1600,
}


class BaseBEVODOM2(nn.Module):
    """PV-BEV fusion odometry network with multi-level supervision."""

    def __init__(self, backbone_conf, head_conf, matching_conf, model_conf,
                 is_train_depth=False, IS_MONO_CUT=False, ALL_DATA_INPUT=False,
                 FREEZE_BEV=False, NO_BN=False, NO_WEIGHT_SCORES=False,
                 dataset_type="NCLT", featuresize_before_corr="32*32_corr11",
                 wrap_level_num=1, max_dis=4, freeze_bev=False,
                 corr_patch_size=11, use_leakyrelu_bn=False):
        super(BaseBEVODOM2, self).__init__()

        # Remove dataset_type from backbone_conf if present (not a backbone param)
        backbone_conf.pop("model_type", None)

        # Basic config
        self.IS_MONO_CUT = IS_MONO_CUT
        self.dataset_type = dataset_type
        self.max_dis = max_dis
        self.freeze_bev = freeze_bev
        self.feature_size = matching_conf["cart_pixel_width"]
        self.feature_division = matching_conf["cart_resolution"]
        self.use_leakyrelu_bn = use_leakyrelu_bn
        self.corr_patch_size = corr_patch_size
        corr_channels = corr_patch_size ** 2

        # ---- Backbone ----
        self.backbone = BaseLSSFPN(**backbone_conf, use_pvflow=True)
        use_pretrained = model_conf.get("use_pretrained_model", False)
        if use_pretrained and (FREEZE_BEV or freeze_bev):
            self._load_and_freeze_pretrained(model_conf, dataset_type)

        # ---- Correlation sampler ----
        self.correlation_sampler = SpatialCorrelationSampler(
            kernel_size=1, patch_size=corr_patch_size,
            stride=1, padding=0, dilation=1, dilation_patch=1)

        # ---- BEV branch (FlowUNet + pose regression) ----
        self._build_flowunet_head(corr_channels)

        # ---- PV branch (6-DoF auxiliary supervision) ----
        if IS_MONO_CUT:
            self._build_pvflow_head(corr_channels, dataset_type)

    # ----------------------------------------------------------------
    # Sub-network construction
    # ----------------------------------------------------------------

    def _build_flowunet_head(self, corr_channels):
        """Build BEV branch: FlowUNet + pose regression head."""
        self.preconv = nn.Conv2d(80, 80, 3, stride=2, padding=1)

        if self.use_leakyrelu_bn:
            self.conv121_1 = nn.Sequential(
                nn.Conv2d(32, 32, 3, stride=2, padding=1, padding_mode='replicate'),
                nn.BatchNorm2d(32), nn.LeakyReLU(0.01, inplace=True))
            self.conv121_2 = nn.Sequential(
                nn.Conv2d(32, 8, 3, stride=1, padding=1, padding_mode='replicate'),
                nn.BatchNorm2d(8), nn.LeakyReLU(0.01, inplace=True))
        else:
            self.conv121_1 = nn.Sequential(
                nn.Conv2d(32, 32, 3, stride=2, padding=1, padding_mode='replicate'),
                nn.ReLU(inplace=True))
            self.conv121_2 = nn.Sequential(
                nn.Conv2d(32, 8, 3, stride=1, padding=1, padding_mode='replicate'),
                nn.ReLU(inplace=True))

        self.fc1 = nn.Linear(2048, 512)
        self.fc2_r = nn.Linear(512, 64)
        self.fc2_t = nn.Linear(512, 64)
        self.fc_output_r = nn.Linear(64, 2)
        self.fc_output_t = nn.Linear(64, 2)

        # BEV corr + PV-to-BEV projected features
        flowunet_in = corr_channels * 2
        self.flow_net = FlowUNet(
            in_channels=flowunet_in,
            use_leakyrelu_bn=self.use_leakyrelu_bn,
            use_first_conv=True)
        self.flow_loss_weight = 0.1

    def _build_pvflow_head(self, corr_channels, dataset_type):
        """Build PVFlow 6DoF auxiliary branch."""
        if self.use_leakyrelu_bn:
            self.pvflow_conv = nn.Sequential(
                nn.Conv2d(corr_channels, 64, 3, stride=2, padding=1, padding_mode='replicate'),
                nn.BatchNorm2d(64), nn.LeakyReLU(0.01, inplace=True),
                nn.Conv2d(64, 32, 3, stride=1, padding=1, padding_mode='replicate'),
                nn.BatchNorm2d(32), nn.LeakyReLU(0.01, inplace=True),
                nn.Conv2d(32, 8, 3, stride=1, padding=1, padding_mode='replicate'),
                nn.BatchNorm2d(8), nn.LeakyReLU(0.01, inplace=True))
        else:
            self.pvflow_conv = nn.Sequential(
                nn.Conv2d(corr_channels, 64, 3, stride=2, padding=1, padding_mode='replicate'),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 32, 3, stride=1, padding=1, padding_mode='replicate'),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 8, 3, stride=1, padding=1, padding_mode='replicate'),
                nn.ReLU(inplace=True))

        pvflow_fc1_in = _PVFLOW_FC1_IN.get(dataset_type, 1600)
        self.pvflow_fc1 = nn.Linear(pvflow_fc1_in, 512)
        self.pvflow_fc2_r = nn.Linear(512, 64)
        self.pvflow_fc2_t = nn.Linear(512, 64)
        self.pvflow_fc_output_r = nn.Linear(64, 4)
        self.pvflow_fc_output_t = nn.Linear(64, 3)

    # ----------------------------------------------------------------
    # Pretrained weight loading
    # ----------------------------------------------------------------
    def _load_and_freeze_pretrained(self, model_conf, dataset_type):
        """Load and freeze pretrained backbone weights."""
        print("Use pretrained model!")
        state_dict = None
        if dataset_type == "NCLT":
            path = model_conf.get("pretrained_model_path_nclt", "")
            if path:
                state_dict = torch.load(os.path.expanduser(path), map_location="cuda")
        elif dataset_type == "oxford":
            path = model_conf.get("pretrained_model_path_oxford", "")
            if path:
                state_dict = torch.load(os.path.expanduser(path), map_location="cuda")
        if state_dict is None:
            print("Pretrained model path not provided, skip.")
            return

        compatible_keys = ['img_backbone', 'img_neck', 'depth_net']
        backbone_sd = self.backbone.state_dict()
        weights = {}
        skipped_shape = 0
        skipped_missing = 0
        for key, val in state_dict.get("model", {}).items():
            if key.startswith('module.encoder.'):
                ck = key.replace('module.encoder.', '')
                if any(ck.startswith(p) for p in compatible_keys):
                    if ck not in backbone_sd:
                        print(f"[WARNING] Pretrained key '{ck}' not found in backbone, skipped.")
                        skipped_missing += 1
                    elif backbone_sd[ck].shape != val.shape:
                        print(f"[WARNING] Shape mismatch for '{ck}': "
                              f"pretrained {val.shape} vs model {backbone_sd[ck].shape}, skipped.")
                        skipped_shape += 1
                    else:
                        weights[ck] = val

        self.backbone.load_state_dict(weights, strict=False)
        loaded = set(weights.keys())
        print(f"Loaded {len(loaded)} pretrained tensors "
              f"(skipped: {skipped_shape} shape mismatch, {skipped_missing} missing).")

        unfrozen = 0
        for name, param in self.backbone.named_parameters():
            if name in loaded:
                param.requires_grad = False
            else:
                param.requires_grad = True
                unfrozen += 1
        if unfrozen > 0:
            print(f"[INFO] {unfrozen} backbone parameters remain trainable (not loaded from pretrained).")
        else:
            print("[INFO] All backbone parameters frozen.")

    # ----------------------------------------------------------------
    # BEV cropping helpers
    # ----------------------------------------------------------------
    def _crop_bev(self, x):
        """Crop BEV feature map for monocular front-view input."""
        if not self.IS_MONO_CUT:
            return x
        if self.dataset_type == "NCLT":
            x = x[:, :, :, x.shape[3] // 4:x.shape[3] * 3 // 4]
            x = x[:, :, :x.shape[2] // 2, :]
        elif self.dataset_type == "oxford":
            x = x[:, :, :, x.shape[3] // 4:x.shape[3] * 3 // 4]
            x = x[:, :, x.shape[2] // 2:, :]
        return x

    def _crop_corr(self, vol):
        """Crop correlation volume (supports 5D and 4D tensors)."""
        if vol.dim() == 5:
            if self.dataset_type == "NCLT":
                return vol[:, :, :, vol.shape[-2] // 2:, vol.shape[-1] // 4:vol.shape[-1] * 3 // 4]
            elif self.dataset_type == "oxford":
                return vol[:, :, :, :vol.shape[-2] // 2, vol.shape[-1] // 4:vol.shape[-1] * 3 // 4]
        else:
            if self.dataset_type == "NCLT":
                return vol[:, :, vol.shape[-2] // 2:, vol.shape[-1] // 4:vol.shape[-1] * 3 // 4]
            elif self.dataset_type == "oxford":
                return vol[:, :, :vol.shape[-2] // 2, vol.shape[-1] // 4:vol.shape[-1] * 3 // 4]
        return vol

    # ----------------------------------------------------------------
    # PV Flow branch
    # ----------------------------------------------------------------
    def _compute_pvflow(self, pvflow_feat, depth_pred):
        """PV Flow branch: compute 6DoF auxiliary pose and PV-to-BEV projected features."""
        pvflow_feat = pvflow_feat.reshape(
            pvflow_feat.shape[0] // 2, 2, *pvflow_feat.shape[1:])
        pv_front = pvflow_feat[:, 0]
        pv_back = pvflow_feat[:, 1]
        depth_split = depth_pred.reshape(
            depth_pred.shape[0] // 2, 2, *depth_pred.shape[1:])
        depth_front = depth_split[:, 0]

        pvflow_R = pvflow_t = pv_bev_feat = None
        for cam in range(pv_front.shape[1]):
            pv_f = pv_front[:, cam].contiguous()
            pv_b = pv_back[:, cam].contiguous()
            pv_corr = self.correlation_sampler(pv_f, pv_b)
            B = pv_corr.shape[0]
            pv_corr = pv_corr.reshape(B, -1, pv_corr.shape[3], pv_corr.shape[4])
            pv_bev_feat = self.backbone.convert_pv_feat_to_bev_feat(pv_corr, depth_front)

            x = self.pvflow_conv(pv_corr)
            x = torch.flatten(x, 1)
            x = F.leaky_relu(self.pvflow_fc1(x), 0.01)
            fc_r = F.leaky_relu(self.pvflow_fc2_r(x), 0.01)
            rot = torch.tanh(self.pvflow_fc_output_r(fc_r))
            fc_t = F.leaky_relu(self.pvflow_fc2_t(x), 0.01)
            trans = torch.tanh(self.pvflow_fc_output_t(fc_t)) * self.max_dis

            # Quaternion -> rotation matrix
            rot_n = F.normalize(rot, p=2, dim=1)
            qx, qy, qz, qw = rot_n[:, 0], rot_n[:, 1], rot_n[:, 2], rot_n[:, 3]
            pvflow_R = torch.zeros((B, 3, 3), device=rot_n.device)
            pvflow_R[:, 0, 0] = 1 - 2 * (qy**2 + qz**2)
            pvflow_R[:, 0, 1] = 2 * (qx * qy - qz * qw)
            pvflow_R[:, 0, 2] = 2 * (qx * qz + qy * qw)
            pvflow_R[:, 1, 0] = 2 * (qx * qy + qz * qw)
            pvflow_R[:, 1, 1] = 1 - 2 * (qx**2 + qz**2)
            pvflow_R[:, 1, 2] = 2 * (qy * qz - qx * qw)
            pvflow_R[:, 2, 0] = 2 * (qx * qz - qy * qw)
            pvflow_R[:, 2, 1] = 2 * (qy * qz + qx * qw)
            pvflow_R[:, 2, 2] = 1 - 2 * (qx**2 + qy**2)
            pvflow_t = trans.reshape(B, 3, 1)

        return pvflow_R, pvflow_t, pv_bev_feat

    def _make_vis(self, corr_vol):
        """Generate correlation volume visualization tensor."""
        v = corr_vol[0, :, :, :, :].permute(0, 2, 1, 3)
        v = v.reshape(v.shape[0] * v.shape[1], v.shape[2] * v.shape[3])
        v = v.cpu().detach().numpy()
        v = ((v - v.min()) / (v.max() - v.min() + 1e-8) * 255).astype('uint8')
        return torch.tensor(v).cuda()

    # ----------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------
    def forward(self, x, mats_dict, timestamps=None, gt_Rt=None, count_num=0):
        """Forward pass: PV-BEV dual-branch fusion with multi-level supervision."""
        x, depth_pred, pvflow_feat = self.backbone(x, mats_dict, timestamps, is_return_depth=True)
        pvflow_R, pvflow_t, pv_bev_feat = self._compute_pvflow(pvflow_feat, depth_pred)

        x = torch.rot90(x, k=1, dims=[2, 3])
        x = self._crop_bev(x)
        pv_bev_feat = self._crop_bev(pv_bev_feat)

        if self.feature_size == 128 and self.IS_MONO_CUT:
            x = self.preconv(x)

        x_pairs = x.reshape(x.shape[0] // 2, 2, *x.shape[1:])
        x_front = x_pairs[:, 0].contiguous()
        x_back = x_pairs[:, 1].contiguous()
        corr_vol = self.correlation_sampler(x_front, x_back)
        B = corr_vol.shape[0]

        cut_x_save = self._make_vis(corr_vol)
        corr_vol = self._crop_corr(corr_vol)
        pv_bev_feat = self._crop_corr(pv_bev_feat)

        corr_vol = corr_vol.reshape(B, -1, corr_vol.shape[-2], corr_vol.shape[-1])
        combined = torch.cat((corr_vol, pv_bev_feat), dim=1)

        flow_pred, feat = self.flow_net(combined)
        feat = self.conv121_1(feat)
        feat = self.conv121_2(feat)
        feat = torch.flatten(feat, 1)
        feat = F.leaky_relu(self.fc1(feat), 0.01)

        fc_r = F.leaky_relu(self.fc2_r(feat), 0.01)
        rot = torch.tanh(self.fc_output_r(fc_r))
        fc_t = F.leaky_relu(self.fc2_t(feat), 0.01)
        trans = torch.tanh(self.fc_output_t(fc_t))

        rot_n = F.normalize(rot, p=2, dim=1)
        trans = trans * self.max_dis
        cos_z, sin_z = rot_n[:, 0], rot_n[:, 1]
        R = torch.stack([
            cos_z, -sin_z, torch.zeros_like(cos_z),
            sin_z,  cos_z, torch.zeros_like(cos_z),
            torch.zeros_like(cos_z), torch.zeros_like(cos_z), torch.ones_like(cos_z),
        ], dim=-1).reshape(-1, 3, 3)
        t = torch.cat([trans, torch.zeros(B, 1, device=trans.device)], dim=1).reshape(B, 3, 1)

        return R, t, {'R': R, 't': t}, depth_pred, cut_x_save, flow_pred, pvflow_R, pvflow_t

    # ----------------------------------------------------------------
    # Loss
    # ----------------------------------------------------------------
    def compute_flow_gt(self, T_tgt_src):
        """Compute ground truth BEV optical flow from pose transform."""
        device = T_tgt_src.device
        B = T_tgt_src.shape[0]
        fs = int(self.feature_size * 2) if self.IS_MONO_CUT else self.feature_size
        H = W = fs
        res = self.feature_division
        off = (fs / 2, fs / 2)

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
        pts_tgt = torch.bmm(T_tgt_src.float(), pts)

        u_prime = off[0] + pts_tgt[:, 1, :] / res
        v_prime = off[1] - pts_tgt[:, 0, :] / res
        u_flat = u.flatten().unsqueeze(0).repeat(B, 1)
        v_flat = v.flatten().unsqueeze(0).repeat(B, 1)

        flow_full = torch.stack([
            (u_prime - u_flat).reshape(B, H, W),
            (v_prime - v_flat).reshape(B, H, W),
        ], dim=1)

        if self.IS_MONO_CUT:
            if self.dataset_type == "NCLT":
                flow_gt = flow_full[:, :, :H // 2, W // 4:W * 3 // 4]
            elif self.dataset_type == "oxford":
                flow_gt = flow_full[:, :, H // 2:, W // 4:W * 3 // 4]
            else:
                flow_gt = flow_full
        else:
            flow_gt = flow_full

        flow_gt = self._crop_corr(flow_gt)
        return flow_gt

    def flow_loss_func(self, flow_pred, flow_gt, valid_mask=None):
        """Compute optical flow EPE loss."""
        if valid_mask is None:
            valid_mask = torch.ones_like(flow_gt[:, 0:1, :, :])
        epe = torch.sqrt(torch.sum((flow_pred - flow_gt) ** 2, dim=1, keepdim=True))
        return (epe * valid_mask).sum() / (valid_mask.sum() + 1e-6)

    def RTloss(self, R_pred, t_pred, T_gt, flow_pred=None, flow_gt=None, testmode=False):
        """Compute rotation and translation loss with optional flow loss."""
        if (flow_pred is not None and flow_gt is not None) or testmode:
            svd_loss, dict_loss = supervised_loss(R_pred, t_pred, T_gt)
        else:
            svd_loss, dict_loss = supervised_loss_6dof(R_pred, t_pred, T_gt)

        if flow_pred is not None and flow_gt is not None:
            fl = self.flow_loss_func(flow_pred, flow_gt)
            svd_loss = svd_loss + fl * self.flow_loss_weight
            dict_loss["flow_loss"] = fl

        return svd_loss, dict_loss

    # ----------------------------------------------------------------
    # Visualization
    # ----------------------------------------------------------------
    def visualize_flow(self, flow, title="", save_path=None):
        """Visualize optical flow field as quiver plot."""
        if flow.dim() == 4:
            flow = flow[0]
        f = flow.cpu().numpy()
        u, v = f[0], f[1]
        H, W = u.shape

        fig, ax = plt.subplots(figsize=(10, (H / W) * 10 if W > 0 else 10))
        ax.set_facecolor('black')
        fig.set_facecolor('black')
        step = max(2, min(H, W) // 32)
        yg, xg = np.mgrid[step // 2:H:step, step // 2:W:step].astype(int)
        ax.quiver(xg, yg, u[yg, xg], v[yg, xg], color='white', angles='xy',
                  scale_units='xy', scale=1, headwidth=4, headlength=4, width=0.005)
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')
        plt.tight_layout(pad=0)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0, facecolor='black')
        plt.show()

    def visualize_error(self, flow_gt, flow_pred, mode='epe', save_path=None):
        """Visualize flow error (EPE heatmap or error flow field)."""
        if flow_gt.dim() == 4:
            flow_gt = flow_gt[0]
        if flow_pred.dim() == 4:
            flow_pred = flow_pred[0]
        err = flow_pred - flow_gt

        if mode == 'epe':
            epe = torch.norm(err, p=2, dim=0).cpu().numpy()
            H, W = epe.shape
            fig, ax = plt.subplots(figsize=(10, (H / W) * 10 if W > 0 else 10))
            ax.imshow(epe, cmap='viridis', vmin=0, vmax=1)
            ax.axis('off')
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.show()
        elif mode == 'flow':
            self.visualize_flow(err)


# ================================================================
# FlowUNet and building blocks
# ================================================================
class DoubleConv(nn.Module):
    """Double convolution block."""

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
    """First double convolution block with 3-layer conv (channel expand then reduce)."""

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
    """UNet for optical flow estimation.

    Args:
        in_channels: Number of input channels.
        use_leakyrelu_bn: If True, use LeakyReLU without BN in decoder;
            if False, use ReLU with BN.
        use_first_conv: If True, use DoubleConvFirst for inc;
            if False, use DoubleConv.
    """

    def __init__(self, in_channels, use_leakyrelu_bn=False, use_first_conv=True):
        super().__init__()
        lk = use_leakyrelu_bn

        # Encoder
        if use_first_conv:
            self.inc = DoubleConvFirst(in_channels, 32, use_leakyrelu=lk)
        else:
            self.inc = DoubleConv(in_channels, 32, use_bn=True, use_leakyrelu=lk)

        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64, True, lk))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128, True, lk))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256, True, lk))

        # Decoder
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


# ================================================================
# DepthLoss
# ================================================================
class DepthLoss:
    """Binary cross-entropy loss for depth prediction."""

    def __init__(self, backbone_conf):
        self.downsample_factor = backbone_conf['downsample_factor']
        self.dbound = backbone_conf['d_bound']
        self.depth_channels = int((self.dbound[1] - self.dbound[0]) / self.dbound[2])

    def __call__(self, depth_labels, depth_preds):
        depth_labels = self._get_downsampled_gt(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.depth_channels)
        fg = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels.to(depth_preds.device)

        with autocast(enabled=False):
            loss = (F.binary_cross_entropy(
                depth_preds[fg], depth_labels[fg], reduction='none',
            ).sum() / max(1.0, fg.sum()))
        loss = loss / 10.0
        return loss, {'depth_loss': loss.item()}

    def _get_downsampled_gt(self, gt):
        """Downsample depth ground truth."""
        B, N, H, W = gt.shape
        ds = self.downsample_factor
        gt = gt.view(B * N, H // ds, ds, W // ds, ds, 1)
        gt = gt.permute(0, 1, 3, 5, 2, 4).contiguous().view(-1, ds * ds)
        gt = torch.where(gt == 0.0, 1e5 * torch.ones_like(gt), gt)
        gt = torch.min(gt, dim=-1).values.view(B * N, H // ds, W // ds)
        gt = (gt - (self.dbound[0] - self.dbound[2])) / self.dbound[2]
        gt = torch.where(
            (gt < self.depth_channels + 1) & (gt >= 0.0), gt, torch.zeros_like(gt))
        gt = F.one_hot(gt.long(), self.depth_channels + 1).view(-1, self.depth_channels + 1)[:, 1:]
        return gt.float()
