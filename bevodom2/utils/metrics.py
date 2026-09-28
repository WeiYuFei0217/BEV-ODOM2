"""Odometry metrics and trajectory export. All inputs are lists of 4x4 relative poses.

- segment_errors: RTE / RRE averaged over sub-trajectories of 100, 200, ..., 800 m.
- frame_errors: mean per-frame relative translation / rotation error.
- compute_ate: mean position error after one global SE(3) or Sim(3) alignment.
- save_tum_trajectory: chained trajectory in TUM format.

Example:
    rte, rre = segment_errors(T_gt, T_pred)
    ate_sim3 = compute_ate(T_gt, T_pred, with_scale=True)
"""
import numpy as np
import torch
from scipy.spatial.transform import Rotation

SEGMENT_LENGTHS = [100, 200, 300, 400, 500, 600, 700, 800]
SEGMENT_STEP = 10  # frame step between sub-trajectory start points


def accumulate(rel_poses):
    """Chain relative poses; returns (translation (3,), quaternion xyzw (4,)) per frame."""
    T_cumulative = np.identity(4)
    states = []
    for T in rel_poses:
        T_cumulative = np.dot(T_cumulative, T)
        states.append((T_cumulative[:3, 3], Rotation.from_matrix(T_cumulative[:3, :3]).as_quat()))
    return states


def save_tum_trajectory(file_path, timestamps, rel_poses):
    """TUM format: one "timestamp tx ty tz qx qy qz qw" line per frame."""
    with open(file_path, 'w') as f:
        for timestamp, (t, q) in zip(timestamps, accumulate(rel_poses)):
            f.write(f"{timestamp.item()} {t[0]} {t[1]} {t[2]} {q[0]} {q[1]} {q[2]} {q[3]}\n")


def trajectory_matrices(rel_poses):
    """(N, 4, 4) float64 chained trajectory, rotations rebuilt from quaternions (identical to the TUM file)."""
    poses = np.zeros((len(rel_poses), 4, 4), dtype=np.float64)
    for i, (t, q) in enumerate(accumulate(rel_poses)):
        qx, qy, qz, qw = (float(v) for v in q)
        poses[i, :3, :3] = [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
        ]
        poses[i, :3, 3] = [float(v) for v in t]
        poses[i, 3, 3] = 1.0
    return torch.from_numpy(poses)


def _trajectory_distances(poses):
    diffs = poses[1:, :3, 3] - poses[:-1, :3, 3]
    distances = torch.zeros(len(poses), dtype=torch.float64)
    distances[1:] = torch.cumsum(torch.norm(diffs, dim=1), dim=0)
    return distances


def _last_frame_from_segment_length(dist, first_frame, segment_length):
    matching = torch.nonzero(dist[first_frame:] - dist[first_frame] > segment_length)
    return -1 if len(matching) == 0 else matching[0].item() + first_frame


def segment_errors(T_gt, T_pred):
    """Returns (RTE, RRE): mean length-normalized sub-trajectory error, as a ratio and in rad/m."""
    poses_gt, poses_pred = trajectory_matrices(T_gt), trajectory_matrices(T_pred)
    if len(poses_gt) != len(poses_pred):
        raise ValueError("Ground truth and predicted trajectories must have the same length")
    dist = _trajectory_distances(poses_gt)
    inv_gt, inv_pred = torch.inverse(poses_gt), torch.inverse(poses_pred)

    t_errs, r_errs = [], []
    for first_frame in range(0, len(poses_gt), SEGMENT_STEP):
        for length in SEGMENT_LENGTHS:
            last_frame = _last_frame_from_segment_length(dist, first_frame, length)
            if last_frame == -1:
                continue
            delta_gt = torch.matmul(inv_gt[first_frame], poses_gt[last_frame])
            delta_pred = torch.matmul(inv_pred[first_frame], poses_pred[last_frame])
            error = torch.matmul(torch.inverse(delta_pred), delta_gt)
            cos_angle = torch.clamp((torch.trace(error[:3, :3]) - 1) / 2.0, min=-1.0, max=1.0)
            r_errs.append(torch.acos(cos_angle).item() / length)
            t_errs.append(torch.norm(error[:3, 3]).item() / length)
    if not t_errs:
        raise ValueError("No valid 100-800 m sub-trajectory for evaluation")
    return sum(t_errs) / len(t_errs), sum(r_errs) / len(r_errs)


def _inverse_se3_f32(T):
    """Inverse of a rigid transform (float32 result)."""
    T2 = np.identity(4, dtype=np.float32)
    R = T[0:3, 0:3]
    T2[0:3, 0:3] = R.transpose()
    T2[0:3, 3:] = np.matmul(-1 * R.transpose(), T[0:3, 3].reshape(3, 1))
    return T2


def frame_errors(T_gt, T_pred):
    """Mean per-frame relative pose error: (planar translation error in m, rotation error in deg)."""
    t_error, r_error = [], []
    for T, T_p in zip(T_gt, T_pred):
        E = np.matmul(T, _inverse_se3_f32(T_p))
        t_error.append(np.sqrt(E[0, 3] ** 2 + E[1, 3] ** 2))
        cos_angle = 0.5 * (np.trace(E[0:3, 0:3]) - 1)
        r_error.append(180 * np.arccos(max(min(cos_angle, 1.0), -1.0)) / np.pi)
    return np.mean(np.array(t_error)), np.mean(np.array(r_error))


def umeyama_alignment(src, dst, with_scale):
    """Least-squares SE(3) / Sim(3) alignment of src to dst (N x 3); returns (R, t, s)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    u, d, vt = np.linalg.svd(xd.T @ xs / len(src))
    w = np.eye(src.shape[1])
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        w[-1, -1] = -1
    rot = u @ w @ vt
    s = float((d * np.diag(w)).sum() / ((xs ** 2).sum() / len(src))) if with_scale else 1.0
    return rot, mu_d - s * rot @ mu_s, s


def compute_ate(T_gt, T_pred, with_scale):
    """ATE in meters: mean (not RMSE) position error after one global alignment of the full trajectory.

    with_scale=False: SE(3) alignment; with_scale=True: Sim(3) alignment.
    """
    def positions(rel_poses):
        T_cumulative, xyz = np.identity(4), []
        for T in rel_poses:
            T_cumulative = np.dot(T_cumulative, T)
            xyz.append(T_cumulative[:3, 3])
        return np.array(xyz)

    gt_xyz, pred_xyz = positions(T_gt), positions(T_pred)
    rot, t, s = umeyama_alignment(pred_xyz, gt_xyz, with_scale)
    return float(np.linalg.norm(s * pred_xyz @ rot.T + t - gt_xyz, axis=1).mean())


def evaluate_trajectory(T_gt, T_pred):
    """All metrics of one sequence: rte (%), rre (deg/100 m), t_err_avg, R_err_avg, ate_se3, ate_sim3."""
    rte, rre = segment_errors(T_gt, T_pred)
    t_err_avg, R_err_avg = frame_errors(T_gt, T_pred)
    return {
        't_err_avg': t_err_avg,
        'R_err_avg': R_err_avg,
        'rte': rte * 100,
        'rre': rre * 18000 / np.pi,
        'ate_se3': compute_ate(T_gt, T_pred, with_scale=False),
        'ate_sim3': compute_ate(T_gt, T_pred, with_scale=True),
    }
