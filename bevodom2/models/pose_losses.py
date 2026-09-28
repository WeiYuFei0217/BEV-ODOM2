"""Pose losses: 3-DoF BEV pose loss, 5-DoF PV pose loss and the planar part of a relative pose.

Example:
    from bevodom2.models.pose_losses import planar_part, pose_loss_3dof, pose_loss_5dof
    loss, logs = pose_loss_3dof(R_pred, t_pred, planar_part(T_rel), alpha=10)
"""

import torch


def planar_part(T):
    """Planar part of a relative pose: yaw rotation and planar translation.

    Args:
        T: (B, 4, 4) relative pose T_rel.

    Returns:
        (B, 4, 4) transform with R = Rz(yaw), yaw = atan2(R10, R00), t = [tx, ty, 0].
    """
    yaw = torch.atan2(T[:, 1, 0], T[:, 0, 0])
    cos, sin = torch.cos(yaw), torch.sin(yaw)
    P = torch.zeros_like(T)
    P[:, 0, 0], P[:, 0, 1] = cos, -sin
    P[:, 1, 0], P[:, 1, 1] = sin, cos
    P[:, 2, 2] = 1
    P[:, 3, 3] = 1
    P[:, :2, 3] = T[:, :2, 3]
    return P


def pose_loss_3dof(R_pred, t_pred, T_gt, alpha=10):
    """3-DoF loss: |tx - tx_gt| + |ty - ty_gt| + alpha * |theta - theta_gt|.

    Args:
        R_pred: (B, 3, 3) predicted rotation (yaw only).
        t_pred: (B, 3, 1) predicted translation.
        T_gt:   (B, 4, 4) planar ground truth transform.
        alpha: Weight of the rotation term.

    Returns:
        loss: Batch mean of the loss.
        logs: Dictionary with 'R_loss' (|dtheta|, rad) and 't_loss'.
    """
    T_gt = T_gt.to(R_pred.device, torch.float32)
    cos_p, sin_p = R_pred[:, 0, 0], R_pred[:, 1, 0]
    cos_g, sin_g = T_gt[:, 0, 0], T_gt[:, 1, 0]
    # Angle difference wrapped to (-pi, pi]
    d_theta = torch.atan2(sin_p * cos_g - cos_p * sin_g, cos_p * cos_g + sin_p * sin_g)

    t_loss = (t_pred[:, :2, 0] - T_gt[:, :2, 3]).abs().sum(dim=1).mean()
    R_loss = d_theta.abs().mean()

    loss = t_loss + alpha * R_loss
    logs = {'R_loss': R_loss, 't_loss': t_loss}
    return loss, logs


def pose_loss_5dof(R_pred, t_pred, T_gt, beta=10, eps=1e-6):
    """5-DoF loss: ||t_hat - t_hat_gt||_1 + beta * ||R - R_gt||_F, t_hat = t / (||t||_2 + eps).

    Args:
        R_pred: (B, 3, 3) predicted rotation.
        t_pred: (B, 3, 1) predicted translation.
        T_gt:   (B, 4, 4) full 3D ground truth transform.
        beta: Weight of the rotation term.
        eps: Stabilizer of the translation normalization.

    Returns:
        loss: Batch mean of the loss.
        logs: Dictionary with 'R_loss' and 't_loss'.
    """
    T_gt = T_gt.to(R_pred.device, torch.float32)
    t_pred = t_pred.reshape(-1, 3)
    t_gt = T_gt[:, :3, 3]
    t_hat_pred = t_pred / (t_pred.norm(dim=1, keepdim=True) + eps)
    t_hat_gt = t_gt / (t_gt.norm(dim=1, keepdim=True) + eps)

    t_loss = (t_hat_pred - t_hat_gt).abs().sum(dim=1).mean()
    R_loss = torch.linalg.norm(R_pred - T_gt[:, :3, :3], dim=(1, 2)).mean()

    loss = t_loss + beta * R_loss
    logs = {'R_loss': R_loss, 't_loss': t_loss}
    return loss, logs
