"""
Training monitor module for logging, visualization, and evaluation.

- convert_to_radar_frame: pixel coords to BEV metric coords
- convert_to_radar_frame_mono_cut: mono-cut mode coord conversion
- project_to_image: project BEV coords to camera images
- MonitorBase: training monitor base class (TensorBoard logging, validation)
"""

import os
from time import time

import cv2
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor

from bevodom2.modules.utils.utils import (
    computeMedianError, save_tum_trajectory,
    eval_kitti, eval_kitti_zjh, tum_to_kitti,
)
from bevodom2.modules.utils.losses import supervised_loss
from bevodom2.modules.utils.vis import (
    draw_batch, plot_sequences,
    draw_batch_all_for_casestudy, draw_batch_all_for_casestudy2,
    plot_2d_trajectory,
)


def convert_to_radar_frame(pixel_coords, config):
    """Convert pixel coords (B, N, 2) to BEV metric coords."""
    cart_pixel_width = config['cart_pixel_width']
    cart_resolution = config['cart_resolution']
    gpuid = config['gpuid']
    if (cart_pixel_width % 2) == 0:
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
    else:
        cart_min_range = cart_pixel_width // 2 * cart_resolution
    B, N, _ = pixel_coords.size()
    R = torch.tensor([[0, -cart_resolution], [cart_resolution, 0]]).expand(B, 2, 2).to(gpuid)
    t = torch.tensor([[cart_min_range], [-cart_min_range]]).expand(B, 2, N).to(gpuid)
    return (torch.bmm(R, pixel_coords.transpose(2, 1)) + t).transpose(2, 1)


def convert_to_radar_frame_mono_cut(pixel_coords, config, dataset_type="NCLT"):
    """Convert pixel coords to metric coords for mono-cut mode."""
    cart_pixel_width = config['cart_pixel_width'] * 2
    cart_resolution = config['cart_resolution']
    gpuid = config['gpuid']
    width = cart_pixel_width
    height = cart_pixel_width

    if dataset_type in ('NCLT', 'kitti', 'ZJH'):
        pixel_coords[:, :, 0] += width // 4
    elif dataset_type == "oxford":
        pixel_coords[:, :, 0] += width // 4
        pixel_coords[:, :, 1] += height // 2

    if (cart_pixel_width % 2) == 0:
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
    else:
        cart_min_range = cart_pixel_width // 2 * cart_resolution

    B, N, _ = pixel_coords.size()
    R = torch.tensor([[0, -cart_resolution], [cart_resolution, 0]]).expand(B, 2, 2).to(gpuid)
    t = torch.tensor([[cart_min_range], [-cart_min_range]]).expand(B, 2, N).to(gpuid)
    return (torch.bmm(R, pixel_coords.transpose(2, 1)) + t).transpose(2, 1)


def project_to_image(coords, sensor2ego_mats, intrin_mats, match_weights, nms, max_w,
                     bda_mats=None, image_size=(224, 384)):
    """Project BEV coords to camera image coordinates."""
    coords_images = []
    coords_with_height = torch.cat(
        (coords, torch.zeros(coords.shape[0], coords.shape[1], 1),
         torch.ones(coords.shape[0], coords.shape[1], 1)), dim=2
    )
    coords_with_height = coords_with_height.squeeze(0)

    for i in range(len(sensor2ego_mats)):
        sensor2ego_mat = sensor2ego_mats[i]
        intrin_mat = intrin_mats[i]
        ego_to_sensor_mat = torch.inverse(sensor2ego_mat)
        coords_cam = torch.matmul(ego_to_sensor_mat, coords_with_height.transpose(0, 1)).transpose(0, 1)
        valid_indices = (coords_cam[:, 2] > 0) & (match_weights > nms * max_w)
        coords_cam = coords_cam[valid_indices]
        coords_img_homogeneous = torch.matmul(intrin_mat[:3, :3], coords_cam[:, :3].transpose(0, 1)).transpose(0, 1)
        coords_img = coords_img_homogeneous[:, :2] / coords_img_homogeneous[:, 2].unsqueeze(1)
        coords_images.append(coords_img)

    return coords_images


def denormalize(img_np):
    """Denormalize an ImageNet-normalized image."""
    normalize_mean = [0.485, 0.456, 0.406]
    normalize_std = [0.229, 0.224, 0.225]
    img_np = img_np * np.array(normalize_std) + np.array(normalize_mean)
    img_np = np.clip(img_np, 0, 1)
    return img_np


class MonitorBase(object):
    """Training monitor base class for logging, validation, and visualization."""

    def __init__(self, model, config):
        self.model = model
        self.log_dir = config['log_dir']
        self.config = config
        self.gpuid = config['gpuid']
        self.counter = 0
        self.dt = 0
        self.current_time = 0
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)
        print('monitor running and saving to {}'.format(self.log_dir))

    def step(self, loss, R_loss, t_loss, depth_loss, flow_loss=0, costmap_loss=0,
             R_loss_pv=0, t_loss_pv=0, costmap_pred=None, costmap_gt=None):
        """Record training step: write losses to TensorBoard."""
        self.counter += 1
        self.dt = time() - self.current_time
        self.current_time = time()

        self.writer.add_scalar('train/loss', loss.detach().cpu().item(), self.counter)
        self.writer.add_scalar('train/Rloss', R_loss, self.counter)
        self.writer.add_scalar('train/tloss', t_loss, self.counter)
        self.writer.add_scalar('train/depth_loss', depth_loss, self.counter)
        self.writer.add_scalar('train/flow_loss', flow_loss, self.counter)
        self.writer.add_scalar('train/costmap_loss', costmap_loss, self.counter)
        self.writer.add_scalar('train/R_loss_pv', R_loss_pv, self.counter)
        self.writer.add_scalar('train/t_loss_pv', t_loss_pv, self.counter)

        if costmap_pred is not None and costmap_gt is not None and self.counter % 50 == 0:
            costmap_vis = self.visualize_costmap(costmap_pred, costmap_gt)
            self.writer.add_image('train/costmap_comparison', costmap_vis, self.counter)

        return self.counter

    def visualize_costmap(self, costmap_pred, costmap_gt):
        """Visualize costmap: concatenate prediction, difference heatmap, and ground truth."""
        if costmap_pred.dim() == 3:
            costmap_pred = costmap_pred.unsqueeze(0)
        if costmap_gt.dim() == 3:
            costmap_gt = costmap_gt.unsqueeze(0)

        pred = costmap_pred[0, 0].detach().cpu().numpy()
        gt = costmap_gt[0, 0].detach().cpu().numpy()
        diff = np.abs(pred - gt)

        pred_img = (pred * 255).astype(np.uint8)
        diff_img = (diff * 255).astype(np.uint8)
        gt_img = (gt * 255).astype(np.uint8)

        pred_rgb = cv2.cvtColor(pred_img, cv2.COLOR_GRAY2RGB)
        gt_rgb = cv2.cvtColor(gt_img, cv2.COLOR_GRAY2RGB)
        diff_colored = cv2.applyColorMap(diff_img, cv2.COLORMAP_JET)
        diff_colored = cv2.cvtColor(diff_colored, cv2.COLOR_BGR2RGB)

        combined = np.concatenate([pred_rgb, diff_colored, gt_rgb], axis=1)
        combined = combined.transpose(2, 0, 1)
        combined = torch.from_numpy(combined).float() / 255.0
        return combined

    def clean_poses_2d(self, poses):
        """Clean poses to ensure valid SE(3) by reconstructing from x, y, yaw."""
        cleaned = []
        for T in poses:
            dx, dy = T[0, 3], T[1, 3]
            yaw = np.arctan2(T[1, 0], T[0, 0])
            T_new = np.eye(4)
            T_new[0, 0] = np.cos(yaw)
            T_new[0, 1] = -np.sin(yaw)
            T_new[1, 0] = np.sin(yaw)
            T_new[1, 1] = np.cos(yaw)
            T_new[0, 3] = dx
            T_new[1, 3] = dy
            T_new[2, 3] = T[2, 3]
            cleaned.append(T_new)
        return cleaned

    def step_val_iros(self, T_gt, T_pred, timestamps, file_path_gt, file_path_pred,
                      cut_x_save_vis=None, test_num_seq=1,
                      costmap_pred=None, costmap_gt=None):
        """Validation step: compute error metrics, save trajectories, generate plots."""
        results = computeMedianError(T_gt, T_pred)

        if cut_x_save_vis is not None:
            cut_x_save_vis_tensor = ToTensor()(cut_x_save_vis)
            self.writer.add_image('val/cut_x_save_vis_' + str(test_num_seq),
                                  cut_x_save_vis_tensor, self.counter)

        if costmap_pred is not None and costmap_gt is not None:
            costmap_vis = self.visualize_costmap(costmap_pred, costmap_gt)
            self.writer.add_image('val/costmap_comparison_' + str(test_num_seq),
                                  costmap_vis, self.counter)

        # Save TUM trajectories and convert to KITTI format
        save_tum_trajectory(file_path_gt, timestamps, T_gt)
        save_tum_trajectory(file_path_pred, timestamps, T_pred)
        file_path_gt_kitti = file_path_gt.replace(".txt", "_kitti.txt")
        file_path_pred_kitti = file_path_pred.replace(".txt", "_kitti.txt")
        tum_to_kitti(file_path_gt, file_path_gt_kitti)
        tum_to_kitti(file_path_pred, file_path_pred_kitti)

        # Compute KITTI-style odometry metrics
        kitti_t_err, kitti_r_err = eval_kitti(file_path_gt_kitti, file_path_pred_kitti)
        kitti_t_err *= 100
        kitti_r_err *= 18000 / np.pi

        # Plot trajectory
        flip = True
        if self.config.get('dataset_type', '') == 'JZ':
            flip = False
            T_gt = self.clean_poses_2d(T_gt)
            T_pred = self.clean_poses_2d(T_pred)

        imgs = plot_sequences(T_gt, T_pred, [len(T_pred)], flip=flip)
        for i, img in enumerate(imgs):
            self.writer.add_image('val/trajectory_' + str(test_num_seq), img, self.counter)

        self.current_time = time()
        return results[4], results[5], kitti_t_err, kitti_r_err

    def step_val_iros_record(self, t_err_avg_record, R_err_avg_record,
                             kitti_t_err_record, kitti_R_err_record,
                             costmap_loss_avg_record=None,
                             costmap_metrics_dict=None):
        """Record validation summary metrics to TensorBoard."""
        self.writer.add_scalar('val/t_err_avg', t_err_avg_record, self.counter)
        self.writer.add_scalar('val/R_err_avg', R_err_avg_record, self.counter)
        self.writer.add_scalar('val/kitti_t_err_avg', kitti_t_err_record, self.counter)
        self.writer.add_scalar('val/kitti_r_err_avg', kitti_R_err_record, self.counter)

        if costmap_loss_avg_record is not None:
            self.writer.add_scalar('val/costmap_loss_avg', costmap_loss_avg_record, self.counter)

        if costmap_metrics_dict is not None:
            self.writer.add_scalar('val/costmap_iou', costmap_metrics_dict['iou'], self.counter)
            self.writer.add_scalar('val/costmap_dice', costmap_metrics_dict['dice'], self.counter)
            self.writer.add_scalar('val/costmap_precision', costmap_metrics_dict['precision'], self.counter)
            self.writer.add_scalar('val/costmap_recall', costmap_metrics_dict['recall'], self.counter)
            self.writer.add_scalar('val/costmap_accuracy', costmap_metrics_dict['accuracy'], self.counter)

        self.current_time = time()
