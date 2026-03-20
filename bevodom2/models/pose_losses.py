"""
pose_losses.py - Pose Supervision Loss Functions

Provides 3DoF (yaw + xy) and 6DoF (full 3D) supervised losses.
"""

import torch


def supervised_loss(R_tgt_src_pred, t_tgt_src_pred, T_tgt_src, R_LOSS_enhenace=10):
    """3DoF supervised loss (yaw rotation + xy translation).

    Computes L1 loss between predicted and ground truth 2D rotation and translation.

    Args:
        R_tgt_src_pred: (B, 3, 3) predicted rotation matrix.
        t_tgt_src_pred: (B, 3, 1) predicted translation vector.
        T_tgt_src:      (B, 4, 4) ground truth transformation matrix.
        R_LOSS_enhenace: Weight factor for rotation loss.

    Returns:
        svd_loss: Combined loss scalar.
        dict_loss: Dictionary with 'R_loss' and 't_loss'.
    """
    batch_size = R_tgt_src_pred.shape[0]

    R_tgt_src = T_tgt_src[:, :2, :2].float().cuda()
    R_tgt_src_pred = R_tgt_src_pred[:, :2, :2].float().cuda()
    t_tgt_src = T_tgt_src[:, :2, 3].unsqueeze(-1).float().cuda()
    t_tgt_src_pred = t_tgt_src_pred[:, :2, :]

    identity = torch.eye(2).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
    loss_fn = torch.nn.L1Loss()
    R_loss = loss_fn(torch.matmul(R_tgt_src_pred.transpose(2, 1), R_tgt_src), identity)
    t_loss = loss_fn(t_tgt_src_pred, t_tgt_src)

    svd_loss = t_loss + R_LOSS_enhenace * R_loss
    dict_loss = {'R_loss': R_loss, 't_loss': t_loss}
    return svd_loss, dict_loss


def supervised_loss_6dof(R_tgt_src_pred, t_tgt_src_pred, T_tgt_src, R_LOSS_enhenace=10):
    """6DoF supervised loss (full 3D rotation + 3D translation).

    Computes L1 loss between predicted and ground truth 3D rotation and translation.

    Args:
        R_tgt_src_pred: (B, 3, 3) predicted rotation matrix.
        t_tgt_src_pred: (B, 3, 1) predicted translation vector.
        T_tgt_src:      (B, 4, 4) ground truth transformation matrix.
        R_LOSS_enhenace: Weight factor for rotation loss.

    Returns:
        svd_loss: Combined loss scalar.
        dict_loss: Dictionary with 'R_loss' and 't_loss'.
    """
    batch_size = R_tgt_src_pred.shape[0]

    R_tgt_src = T_tgt_src[:, :3, :3].float().cuda()
    t_tgt_src = T_tgt_src[:, :3, 3].unsqueeze(-1).float().cuda()
    R_tgt_src_pred = R_tgt_src_pred.float().cuda()
    t_tgt_src_pred = t_tgt_src_pred.float().cuda()

    identity = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
    loss_fn = torch.nn.L1Loss()
    R_loss = loss_fn(torch.matmul(R_tgt_src_pred.transpose(2, 1), R_tgt_src), identity)
    t_loss = loss_fn(t_tgt_src_pred, t_tgt_src)

    svd_loss = t_loss + R_LOSS_enhenace * R_loss
    dict_loss = {'R_loss': R_loss, 't_loss': t_loss}
    return svd_loss, dict_loss
