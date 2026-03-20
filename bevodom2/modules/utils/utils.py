"""
Utility functions for pose estimation, trajectory evaluation, and data processing.

Includes:
- SE(3) transform operations (inverse, compose, enforce orthogonality)
- Lie algebra conversions (se3 <-> SE3)
- Error metrics (rotation, translation, KITTI evaluation)
- Trajectory I/O (TUM format, KITTI format)
- Coordinate transforms (pixel to radar frame, normalization)
- Optical flow utilities (bilinear sampling, warping, cycle consistency)
"""

import csv
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from scipy import interpolate
from scipy.spatial.transform import Rotation as R


# ===========================================================================
# SE(3) Transform Operations
# ===========================================================================

def get_inverse_tf(T):
    """Compute the inverse of a 4x4 homogeneous transform."""
    T2 = np.identity(4, dtype=np.float32)
    R = T[0:3, 0:3]
    t = T[0:3, 3].reshape(3, 1)
    T2[0:3, 0:3] = R.transpose()
    T2[0:3, 3:] = np.matmul(-1 * R.transpose(), t)
    return T2


def get_transform(x, y, theta):
    """Build a 4x4 SE(2) transform from x, y, theta."""
    T = np.identity(4, dtype=np.float32)
    T[0:2, 0:2] = np.array([[np.cos(theta), np.sin(theta)],
                             [-np.sin(theta), np.cos(theta)]])
    T[0, 3] = x
    T[1, 3] = y
    return T


def get_transform2(R, t):
    """Build a 4x4 SE(3) transform from rotation matrix R and translation t."""
    T = np.identity(4, dtype=np.float32)
    T[0:3, 0:3] = R
    T[0:3, 3] = t.squeeze()
    return T


def enforce_orthog(T, dim=3):
    """Enforce orthogonality of the rotation part of a transform."""
    if dim == 2:
        if abs(np.linalg.det(T[0:2, 0:2]) - 1) < 1e-10:
            return T
        R = T[0:2, 0:2]
        epsilon = 0.001
        if abs(R[0, 0] - R[1, 1]) > epsilon or abs(R[1, 0] + R[0, 1]) > epsilon:
            print("WARNING: this is not a proper rigid transformation:", R)
            return T
        a = (R[0, 0] + R[1, 1]) / 2
        b = (-R[1, 0] + R[0, 1]) / 2
        s = np.sqrt(a**2 + b**2)
        a /= s
        b /= s
        R[0, 0] = a
        R[0, 1] = b
        R[1, 0] = -b
        R[1, 1] = a
        T[0:2, 0:2] = R
    if dim == 3:
        if abs(np.linalg.det(T[0:3, 0:3]) - 1) < 1e-10:
            return T
        c1 = T[0:3, 1]
        c2 = T[0:3, 2]
        c1 /= np.linalg.norm(c1)
        c2 /= np.linalg.norm(c2)
        newcol0 = np.cross(c1, c2)
        newcol1 = np.cross(c2, newcol0)
        T[0:3, 0] = newcol0
        T[0:3, 1] = newcol1
        T[0:3, 2] = c2
    return T


# ===========================================================================
# Lie Algebra Operations
# ===========================================================================

def carrot(xbar):
    """Skew-symmetric (hat) operator for 3-vectors or 6-vectors."""
    x = xbar.squeeze()
    if x.shape[0] == 3:
        return np.array([[0, -x[2], x[1]],
                         [x[2], 0, -x[0]],
                         [-x[1], x[0], 0]])
    elif x.shape[0] == 6:
        return np.array([[0, -x[5], x[4], x[0]],
                         [x[5], 0, -x[3], x[1]],
                         [-x[4], x[3], 0, x[2]],
                         [0, 0, 0, 1]])
    print('WARNING: attempted carrot operator on invalid vector shape')
    return xbar


def se3ToSE3(xi):
    """Convert se(3) Lie algebra element to SE(3) matrix via exponential map."""
    T = np.identity(4, dtype=np.float32)
    rho = xi[0:3].reshape(3, 1)
    phibar = xi[3:6].reshape(3, 1)
    phi = np.linalg.norm(phibar)
    R = np.identity(3)
    if phi != 0:
        phibar /= phi
        I = np.identity(3)
        R = np.cos(phi) * I + (1 - np.cos(phi)) * phibar @ phibar.T + np.sin(phi) * carrot(phibar)
        J = I * np.sin(phi) / phi + (1 - np.sin(phi) / phi) * phibar @ phibar.T + \
            carrot(phibar) * (1 - np.cos(phi)) / phi
        rho = J @ rho
    T[0:3, 0:3] = R
    T[0:3, 3:] = rho
    return T


def SE3tose3(T):
    """Convert SE(3) matrix to se(3) Lie algebra element via logarithmic map."""
    R = T[0:3, 0:3]
    evals, evecs = np.linalg.eig(R)
    idx = -1
    for i in range(3):
        if evals[i].real != 0 and evals[i].imag == 0:
            idx = i
            break
    assert(idx != -1)
    abar = evecs[idx].real.reshape(3, 1)
    phi = np.arccos((np.trace(R) - 1) / 2)
    rho = T[0:3, 3:]
    if phi != 0:
        I = np.identity(3)
        J = I * np.sin(phi) / phi + (1 - np.sin(phi) / phi) * abar @ abar.T + \
            carrot(abar) * (1 - np.cos(phi)) / phi
        rho = np.linalg.inv(J) @ rho
    xi = np.zeros((6, 1))
    xi[0:3, 0:] = rho
    xi[3:, 0:] = phi * abar
    return xi


# ===========================================================================
# Error Metrics
# ===========================================================================

def rotationError(T):
    """Compute rotation error in radians from a 4x4 error transform."""
    d = 0.5 * (np.trace(T[0:3, 0:3]) - 1)
    return np.arccos(max(min(d, 1.0), -1.0))


def translationError(T, dim=2):
    """Compute translation error from a 4x4 error transform."""
    if dim == 2:
        return np.sqrt(T[0, 3]**2 + T[1, 3]**2)
    return np.sqrt(T[0, 3]**2 + T[1, 3]**2 + T[2, 3]**2)


def computeRelativePoseError(T_gt, T_pred, delta):
    """Compute relative pose error (RPE) at interval delta."""
    rpe_t_error = []
    rpe_r_error = []
    for i in range(0, len(T_gt) - delta, delta):
        T_gt_rel = np.matmul(get_inverse_tf(T_gt[i]), T_gt[i + delta])
        T_pred_rel = np.matmul(get_inverse_tf(T_pred[i]), T_pred[i + delta])
        T_error_rel = np.matmul(get_inverse_tf(T_gt_rel), T_pred_rel)
        rpe_t_error.append(translationError(T_error_rel))
        rpe_r_error.append(180 * rotationError(T_error_rel) / np.pi)
    return rpe_t_error, rpe_r_error


def computeMedianError(T_gt, T_pred, delta=1):
    """Compute median/mean ATE and RPE errors."""
    t_error = []
    r_error = []
    for i, T in enumerate(T_gt):
        T_error = np.matmul(T, get_inverse_tf(T_pred[i]))
        t_error.append(translationError(T_error))
        r_error.append(180 * rotationError(T_error) / np.pi)
    t_error = np.array(t_error)
    r_error = np.array(r_error)

    rpe_t_error, rpe_r_error = computeRelativePoseError(T_gt, T_pred, delta)
    rpe_t_error = np.array(rpe_t_error)
    rpe_r_error = np.array(rpe_r_error)

    return [np.median(t_error), np.std(t_error), np.median(r_error), np.std(r_error),
            np.mean(t_error), np.mean(r_error), np.mean(rpe_t_error), np.mean(rpe_r_error)]


# ===========================================================================
# KITTI Evaluation (numpy-based)
# ===========================================================================

def trajectoryDistances(poses):
    """Compute cumulative trajectory distances from a list of poses."""
    dist = [0]
    for i in range(1, len(poses)):
        P1 = get_inverse_tf(poses[i - 1])
        P2 = get_inverse_tf(poses[i])
        dx = P1[0, 3] - P2[0, 3]
        dy = P1[1, 3] - P2[1, 3]
        dist.append(dist[i-1] + np.sqrt(dx**2 + dy**2))
    return dist


def lastFrameFromSegmentLength(dist, first_frame, length):
    """Find last frame index for a given segment length."""
    for i in range(first_frame, len(dist)):
        if dist[i] > dist[first_frame] + length:
            return i
    return -1


def calcSequenceErrors(poses_gt, poses_pred):
    """Calculate sequence errors at standard KITTI segment lengths."""
    lengths = [100, 200, 300, 400, 500, 600, 700, 800]
    err = []
    step_size = 4
    dist = trajectoryDistances(poses_gt)

    for first_frame in range(0, len(poses_gt), step_size):
        for length in lengths:
            last_frame = lastFrameFromSegmentLength(dist, first_frame, length)
            if last_frame == -1:
                continue
            pose_delta_gt = np.matmul(poses_gt[last_frame], get_inverse_tf(poses_gt[first_frame]))
            pose_delta_res = np.matmul(poses_pred[last_frame], get_inverse_tf(poses_pred[first_frame]))
            pose_error = np.matmul(pose_delta_gt, get_inverse_tf(pose_delta_res))
            r_err = rotationError(pose_error)
            t_err = translationError(pose_error)
            num_frames = float(last_frame - first_frame + 1)
            speed = float(length) / (0.25 * num_frames)
            err.append([first_frame, r_err/float(length), t_err/float(length), length, speed])
    return err


def getStats(err):
    """Compute average translation and rotation errors."""
    t_err = 0
    r_err = 0
    for e in err:
        t_err += e[2]
        r_err += e[1]
    t_err /= float(len(err))
    r_err /= float(len(err))
    return t_err, r_err


def computeKittiMetrics(T_gt, T_pred, seq_lens):
    """Compute KITTI rotation and translation metrics per sequence, then average.

    Args:
        T_gt: list of 4x4 transforms (frame t to t+1)
        T_pred: list of 4x4 transforms (frame t to t+1)
        seq_lens: list of sequence lengths

    Returns:
        t_err (float): average translation error (%)
        r_err (float): average rotation error (deg/m)
    """
    seq_indices = []
    idx = 0
    for s in seq_lens:
        seq_indices.append(list(range(idx, idx + s - 1)))
        idx += (s - 1)
    err_list = []
    for indices in seq_indices:
        T_gt_ = np.identity(4)
        T_pred_ = np.identity(4)
        poses_gt = []
        poses_pred = []
        for i in indices:
            T_gt_ = np.matmul(T_gt[i], T_gt_)
            T_pred_ = np.matmul(T_pred[i], T_pred_)
            enforce_orthog(T_gt_)
            enforce_orthog(T_pred_)
            poses_gt.append(T_gt_)
            poses_pred.append(T_pred_)
        err = calcSequenceErrors(poses_gt, poses_pred)
        t_err, r_err = getStats(err)
        err_list.append([t_err, r_err])
    err_list = np.asarray(err_list)
    avg = np.mean(err_list, axis=0)
    return avg[0] * 100, avg[1] * 180 / np.pi


def saveKittiErrors(err, fname):
    pickle.dump(err, open(fname, 'wb'))


def loadKittiErrors(fname):
    return pickle.load(open(fname, 'rb'))


# ===========================================================================
# KITTI Evaluation (torch-based, optimized)
# ===========================================================================

class Errors_kitti_eval:
    def __init__(self, first_frame, r_err, t_err, len, speed):
        self.first_frame = first_frame
        self.r_err = r_err
        self.t_err = t_err
        self.len = len
        self.speed = speed


def load_poses(file_name):
    """Load poses from KITTI format file with double precision."""
    poses = []
    with open(file_name, 'r') as f:
        for line in f:
            data = line.strip().split()
            if len(data) == 12:
                pose_data = np.array(data, dtype=np.float64).reshape(3, 4)
                poses.append(pose_data)

    n = len(poses)
    pose_array = np.zeros((n, 4, 4), dtype=np.float64)
    for i in range(n):
        pose_array[i, :3, :4] = poses[i]
        pose_array[i, 3, 3] = 1.0
    return torch.from_numpy(pose_array).double()


def trajectory_distances(poses):
    """Compute trajectory distances efficiently using torch."""
    diffs = poses[1:, :3, 3] - poses[:-1, :3, 3]
    point_distances = torch.norm(diffs, dim=1)
    distances = torch.zeros(len(poses), dtype=torch.float64)
    distances[1:] = torch.cumsum(point_distances, dim=0)
    return distances


def last_frame_from_segment_length(dist, first_frame, segment_length):
    """Find last frame using tensor operations."""
    relevant_dists = dist[first_frame:] - dist[first_frame]
    mask = relevant_dists > segment_length
    matching_frames = torch.nonzero(mask)
    return -1 if len(matching_frames) == 0 else matching_frames[0].item() + first_frame


def rotation_error(pose_error):
    """Compute rotation error with double precision."""
    r = pose_error[:3, :3]
    d = (torch.trace(r) - 1) / 2.0
    d = torch.clamp(d, min=-1.0, max=1.0)
    return torch.acos(d)


def translation_error(pose_error):
    """Compute translation error with double precision."""
    return torch.norm(pose_error[:3, 3])


def calc_sequence_errors(poses_gt, poses_result):
    """Calculate sequence errors with optimized tensor operations."""
    err = []
    step_size = 10
    dist = trajectory_distances(poses_gt)
    inv_poses_gt = torch.inverse(poses_gt)
    inv_poses_result = torch.inverse(poses_result)
    lengths = [100, 200, 300, 400, 500, 600, 700, 800]

    for first_frame in range(0, len(poses_gt), step_size):
        for lens in lengths:
            last_frame = last_frame_from_segment_length(dist, first_frame, lens)
            if last_frame == -1:
                continue
            pose_delta_gt = torch.matmul(inv_poses_gt[first_frame], poses_gt[last_frame])
            pose_delta_result = torch.matmul(inv_poses_result[first_frame], poses_result[last_frame])
            pose_error = torch.matmul(torch.inverse(pose_delta_result), pose_delta_gt)
            r_err = rotation_error(pose_error)
            t_err = translation_error(pose_error)
            num_frames = last_frame - first_frame + 1
            speed = lens / (0.1 * num_frames)
            err.append(Errors_kitti_eval(first_frame, r_err.item() / lens,
                                         t_err.item() / lens, lens, speed))
    return err


def calc_sequence_errors_zjh(poses_gt, poses_result):
    """Calculate sequence errors for ZJH dataset."""
    err = []
    step_size = 10
    dist = trajectory_distances(poses_gt)
    inv_poses_gt = torch.inverse(poses_gt)
    inv_poses_result = torch.inverse(poses_result)
    lengths = [100, 200, 300, 400, 500, 600, 700, 800]

    for first_frame in range(0, len(poses_gt), step_size):
        for lens in lengths:
            last_frame = last_frame_from_segment_length(dist, first_frame, lens)
            if last_frame == -1:
                continue
            pose_delta_gt = torch.matmul(inv_poses_gt[first_frame], poses_gt[last_frame])
            pose_delta_result = torch.matmul(inv_poses_result[first_frame], poses_result[last_frame])
            pose_error = torch.matmul(torch.inverse(pose_delta_result), pose_delta_gt)
            r_err = rotation_error(pose_error)
            t_err = translation_error(pose_error)
            num_frames = last_frame - first_frame + 1
            speed = lens / (0.1 * num_frames)
            err.append(Errors_kitti_eval(first_frame, r_err.item() / lens,
                                         t_err.item() / lens, lens, speed))
    return err


def eval_kitti(gt_file, pred_file):
    """Evaluate KITTI odometry metrics."""
    poses_gt = load_poses(gt_file)
    poses_result = load_poses(pred_file)
    if len(poses_gt) != len(poses_result):
        raise ValueError("Ground truth and result poses must have the same length")
    print(len(poses_gt))

    errors = calc_sequence_errors(poses_gt, poses_result)
    if not errors:
        print("No errors found!!!")
        return 10000, 10000

    t_err_sum = sum(e.t_err for e in errors)
    r_err_sum = sum(e.r_err for e in errors)
    count = len(errors)
    return t_err_sum / count, r_err_sum / count


def eval_kitti_zjh(gt_file, pred_file):
    """Evaluate KITTI odometry metrics for ZJH dataset."""
    poses_gt = load_poses(gt_file)
    poses_result = load_poses(pred_file)
    if len(poses_gt) != len(poses_result):
        raise ValueError("Ground truth and result poses must have the same length")
    print(len(poses_gt))

    errors = calc_sequence_errors_zjh(poses_gt, poses_result)
    if not errors:
        print("No errors found!!!")
        return 10000, 10000

    t_err_sum = sum(e.t_err for e in errors)
    r_err_sum = sum(e.r_err for e in errors)
    count = len(errors)
    return t_err_sum / count, r_err_sum / count


# ===========================================================================
# Trajectory I/O
# ===========================================================================

def save_in_yeti_format(T_gt, T_pred, timestamps, seq_lens, seq_names, root='./'):
    """Save results in YETI CSV format."""
    seq_indices = []
    idx = 0
    for s in seq_lens:
        seq_indices.append(list(range(idx, idx + s - 1)))
        idx += (s - 1)

    for s, indices in enumerate(seq_indices):
        fname = root + 'accuracy' + seq_names[s] + '.csv'
        with open(fname, 'w') as f:
            f.write('x,y,yaw,gtx,gty,gtyaw,time1,time2\n')
            for i in indices:
                R_pred = T_pred[i][:3, :3]
                t_pred = T_pred[i][:3, 3:]
                yaw = -1 * np.arcsin(R_pred[0, 1])
                gtyaw = -1 * np.arcsin(T_gt[i][0, 1])
                t = np.matmul(-1 * R_pred.transpose(), np.reshape(t_pred, (3, 1)))
                T = get_inverse_tf(T_gt[i])
                f.write('{},{},{},{},{},{},{},{}\n'.format(
                    t[0, 0], t[1, 0], yaw, T[0, 3], T[1, 3], gtyaw,
                    timestamps[i][0], timestamps[i][1]))


def load_icra21_results(results_loc, seq_names, seq_lens):
    """Load ICRA21 results from CSV files."""
    T_icra = []
    for i, seq_name in enumerate(seq_names):
        fname = results_loc + 'accuracy' + seq_name + '.csv'
        with open(fname, 'r') as f:
            f.readline()
            lines = f.readlines()
            count = 0
            for line in lines:
                line = line.split(',')
                T_icra.append(get_inverse_tf(get_transform(float(line[11]), float(line[12]), float(line[13]))))
                count += 1
            if count < seq_lens[i]:
                print('WARNING: ICRA results shorter than seq_len by {}'.format(seq_lens[i] - count))
            while count < seq_lens[i]:
                T_icra.append(T_icra[-1])
                count += 1
    return T_icra


def save_tum_trajectory(file_path, timestamps, poses):
    """Save trajectory in TUM format (timestamp tx ty tz qx qy qz qw)."""
    with open(file_path, 'w') as f:
        T_cumulative = np.identity(4)
        for i in range(len(timestamps)):
            timestamp = timestamps[i]
            T = poses[i]
            T_cumulative = np.dot(T_cumulative, T)
            translation = T_cumulative[:3, 3]
            rotation_matrix = T_cumulative[:3, :3]
            rotation = R.from_matrix(rotation_matrix).as_quat()
            pose_str = (f"{translation[0]} {translation[1]} {translation[2]} "
                        f"{rotation[0]} {rotation[1]} {rotation[2]} {rotation[3]}")
            f.write(f"{timestamp.item()} {pose_str}\n")


def write_lists_to_file(filename, *lists):
    """Write multiple lists to a file with space-separated values."""
    try:
        with open(filename, 'w') as f:
            n = len(lists[0])
            for i in range(n):
                line = ' '.join(str(lst[i]) for lst in lists)
                f.write(line + '\n')
    except Exception as e:
        print(f"Error writing to file: {e}")


def tum_to_kitti(TUMfilename, KITTIfilename):
    """Convert TUM format trajectory to KITTI format."""
    t1, t2, t3 = [], [], []
    q1, q2, q3, qw = [], [], [], []

    try:
        with open(TUMfilename, newline='') as csvfile:
            datareader = csv.reader(csvfile, delimiter=' ')
            for row in datareader:
                t1.append(float(row[1]))
                t2.append(float(row[2]))
                t3.append(float(row[3]))
                q1.append(float(row[4]))
                q2.append(float(row[5]))
                q3.append(float(row[6]))
                qw.append(float(row[7]))
    except Exception as e:
        print(f"Error reading TUM file: {e}")
        return

    R1, R2, R3 = [], [], []
    R4, R5, R6 = [], [], []
    R7, R8, R9 = [], [], []

    for i in range(len(q1)):
        r1 = 1 - 2 * (q2[i] * q2[i] + q3[i] * q3[i])
        r2 = 2 * (q1[i] * q2[i] - qw[i] * q3[i])
        r3 = 2 * (q1[i] * q3[i] + qw[i] * q2[i])
        r4 = 2 * (q1[i] * q2[i] + qw[i] * q3[i])
        r5 = 1 - 2 * (q1[i] * q1[i] + q3[i] * q3[i])
        r6 = 2 * (q2[i] * q3[i] - qw[i] * q1[i])
        r7 = 2 * (q1[i] * q3[i] - qw[i] * q2[i])
        r8 = 2 * (q2[i] * q3[i] + qw[i] * q1[i])
        r9 = 1 - 2 * (q1[i] * q1[i] + q2[i] * q2[i])
        R1.append(r1); R2.append(r2); R3.append(r3)
        R4.append(r4); R5.append(r5); R6.append(r6)
        R7.append(r7); R8.append(r8); R9.append(r9)

    write_lists_to_file(KITTIfilename,
                        R1, R2, R3, t1,
                        R4, R5, R6, t2,
                        R7, R8, R9, t3)


# ===========================================================================
# Coordinate Transforms & Keypoint Utilities
# ===========================================================================

def normalize_coords(coords_2D, width, height):
    """Normalize 2D coords to [-1, 1] range."""
    batch_size = coords_2D.size(0)
    u_norm = (2 * coords_2D[:, :, 0].reshape(batch_size, -1) / (width - 1)) - 1
    v_norm = (2 * coords_2D[:, :, 1].reshape(batch_size, -1) / (height - 1)) - 1
    return torch.stack([u_norm, v_norm], dim=2)


def convert_to_radar_frame(pixel_coords, config):
    """Convert pixel coords to metric BEV coords."""
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
    """Convert pixel coords to metric BEV coords for mono-cut mode."""
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


def convert_to_radar_frame_mono_cut_3232(pixel_coords, config, dataset_type="NCLT"):
    """Convert pixel coords to metric BEV coords for 32x32 mono-cut mode."""
    cart_pixel_width = config['cart_pixel_width'] * 2
    cart_resolution = config['cart_resolution']
    gpuid = config['gpuid']
    width = cart_pixel_width
    height = cart_pixel_width

    if dataset_type in ('NCLT', 'kitti', 'ZJH'):
        pixel_coords[:, :, 0] += (width * 3) // 8
        pixel_coords[:, :, 1] += height // 4
    elif dataset_type == "oxford":
        pixel_coords[:, :, 0] += (width * 3) // 8
        pixel_coords[:, :, 1] += height // 2

    if (cart_pixel_width % 2) == 0:
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
    else:
        cart_min_range = cart_pixel_width // 2 * cart_resolution

    B, N, _ = pixel_coords.size()
    R = torch.tensor([[0, -cart_resolution], [cart_resolution, 0]]).expand(B, 2, 2).to(gpuid)
    t = torch.tensor([[cart_min_range], [-cart_min_range]]).expand(B, 2, N).to(gpuid)
    return (torch.bmm(R, pixel_coords.transpose(2, 1)) + t).transpose(2, 1)


def get_indices(batch_size, window_size):
    """Get src/tgt index pairs for consecutive frames in each window."""
    src_ids = []
    tgt_ids = []
    for i in range(batch_size):
        for j in range(window_size - 1):
            idx = i * window_size + j
            src_ids.append(idx)
            tgt_ids.append(idx + 1)
    return src_ids, tgt_ids


def get_indices2(batch_size, window_size, asTensor=False):
    """Get src/tgt index pairs where src is always the first frame in each window."""
    src_ids = []
    tgt_ids = []
    for i in range(batch_size):
        idx = i * window_size
        for j in range(idx + 1, idx + window_size):
            tgt_ids.append(j)
            src_ids.append(idx)
    if asTensor:
        src_ids = np.asarray(src_ids, dtype=np.int64)
        tgt_ids = np.asarray(tgt_ids, dtype=np.int64)
        return torch.from_numpy(src_ids), torch.from_numpy(tgt_ids)
    return src_ids, tgt_ids


def get_lr(optimizer):
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_T_ba(out, a, b):
    """Get relative transform T_ba from network output."""
    T_b0 = np.eye(4)
    T_b0[:3, :3] = out['R'][0, b].detach().cpu().numpy()
    T_b0[:3, 3:4] = out['t'][0, b].detach().cpu().numpy()
    T_a0 = np.eye(4)
    T_a0[:3, :3] = out['R'][0, a].detach().cpu().numpy()
    T_a0[:3, 3:4] = out['t'][0, a].detach().cpu().numpy()
    return np.matmul(T_b0, get_inverse_tf(T_a0))


def convert_to_weight_matrix(w, window_id, T_aug=[]):
    """Convert weight scores to full weight matrices."""
    z_weight = 9.2103  # log(1e4), 1e4 is inverse variance of 1cm std dev
    if w.size(1) == 1:
        # Scalar weight
        A = torch.zeros(w.size(0), 9, device=w.device)
        A[:, (0, 4)] = torch.exp(w)
        A[:, 8] = torch.exp(torch.tensor(z_weight))
        A = A.reshape((-1, 3, 3))
        d = torch.zeros(w.size(0), 3, device=w.device)
        d[:, 0:2] += w
        d[:, 2] += z_weight
    elif w.size(1) == 3:
        # 2x2 matrix weight
        L = torch.zeros(w.size(0), 4, device=w.device)
        L[:, (0, 3)] = 1
        L[:, 2] = w[:, 0]
        L = L.reshape((-1, 2, 2))
        D = torch.zeros(w.size(0), 4, device=w.device)
        D[:, (0, 3)] = torch.exp(w[:, 1:])
        D = D.reshape((-1, 2, 2))
        A2x2 = L @ D @ L.transpose(1, 2)

        if T_aug:
            Rot = T_aug[window_id].to(w.device)[:2, :2].unsqueeze(0)
            A2x2 = Rot.transpose(1, 2) @ A2x2 @ Rot

        A = torch.zeros(w.size(0), 3, 3, device=w.device)
        A[:, 0:2, 0:2] = A2x2
        A[:, 2, 2] = torch.exp(torch.tensor(z_weight))
        d = torch.ones(w.size(0), 3, device=w.device) * z_weight
        d[:, 0:2] = w[:, 1:]
    else:
        assert False, "Weight scores should be dim 1 or 3"

    return A, d


def mask_intensity_filter(data, patch_size, patch_mean_thres=0.05):
    """Filter keypoints by patch mean intensity threshold."""
    int_patches = F.unfold(data, kernel_size=patch_size, stride=patch_size)
    keypoint_int = torch.mean(int_patches, dim=1, keepdim=True)
    return keypoint_int >= patch_mean_thres


# ===========================================================================
# Time & Undistortion Utilities
# ===========================================================================

def wrapto2pi(phi):
    """Wrap angle to [0, 2*pi)."""
    if phi < 0:
        return phi + 2 * np.pi * np.ceil(phi / (-2 * np.pi))
    elif phi >= 2 * np.pi:
        return (phi / (2 * np.pi) % 1) * 2 * np.pi
    return phi


def getApproxTimeStamps(points, times, flip_y=False):
    """Approximate per-point timestamps from azimuth angles."""
    azimuth_step = (2 * np.pi) / 400
    timestamps = []
    for i, p in enumerate(points):
        p = points[i]
        ptimes = times[i]
        delta_t = ptimes[-1] - ptimes[-2]
        ptimes = np.append(ptimes, int(ptimes[-1] + delta_t))
        point_times = []
        for k in range(p.shape[0]):
            x = p[k, 0]
            y = p[k, 1]
            if flip_y:
                y *= -1
            phi = np.arctan2(y, x)
            phi = wrapto2pi(phi)
            time_idx = phi / azimuth_step
            t1 = ptimes[int(np.floor(time_idx))]
            t2 = ptimes[int(np.ceil(time_idx))]
            ratio = time_idx % 1
            t = int(t1 + ratio * (t2 - t1))
            point_times.append(t)
        timestamps.append(np.array(point_times))
    return timestamps


def undistort_pointcloud(points, point_times, t_refs, solver):
    """Undistort pointcloud using interpolated poses."""
    for i, p in enumerate(points):
        p = points[i]
        ptimes = point_times[i]
        t_ref = t_refs[i]
        for j, ptime in enumerate(ptimes):
            T_0a = np.identity(4, dtype=np.float32)
            solver.getPoseBetweenTimes(T_0a, ptime, t_ref)
            pbar = T_0a @ p[j].reshape(4, 1)
            p[j, :] = pbar[:]
        points[i] = p
    return points


# ===========================================================================
# Optical Flow Utilities
# ===========================================================================

def load_ckpt(model, path):
    """Load model checkpoint."""
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False)


def resize_data(img1, img2, flow, factor=1.0):
    """Resize image pair and flow by a scale factor."""
    _, _, h, w = img1.shape
    h = int(h * factor)
    w = int(w * factor)
    img1 = F.interpolate(img1, (h, w), mode='area')
    img2 = F.interpolate(img2, (h, w), mode='area')
    flow = F.interpolate(flow, (h, w), mode='area') * factor
    return img1, img2, flow


class InputPadder:
    """Pads images such that dimensions are divisible by 8."""

    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def forward_interpolate(flow):
    """Forward-interpolate optical flow to next frame."""
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]
    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy
    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata((x1, y1), dx, (x0, y0), method='nearest', fill_value=0)
    flow_y = interpolate.griddata((x1, y1), dy, (x0, y0), method='nearest', fill_value=0)
    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """Wrapper for grid_sample using pixel coordinates."""
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1
    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)
    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()
    return img


def coords_grid(batch, ht, wd, device):
    """Create a coordinate grid of shape (batch, 2, ht, wd)."""
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    """Upsample flow by 8x."""
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


# ===========================================================================
# Depth & Reprojection Utilities
# ===========================================================================

def transform(T, p):
    """Apply 4x4 transform to point array."""
    assert T.shape == (4, 4)
    return np.einsum('H W j, i j -> H W i', p, T[:3, :3]) + T[:3, 3]


def from_homog(x):
    """Convert from homogeneous coordinates."""
    return x[..., :-1] / x[..., [-1]]


def reproject(depth1, pose1, pose2, K1, K2):
    """Reproject depth map from camera 1 to camera 2."""
    H, W = depth1.shape
    x, y = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    img_1_coords = np.stack((x, y, np.ones_like(x)), axis=-1).astype(np.float64)
    cam1_coords = np.einsum('H W, H W j, i j -> H W i', depth1, img_1_coords, np.linalg.inv(K1))
    rel_pose = np.linalg.inv(pose2) @ pose1
    cam2_coords = transform(rel_pose, cam1_coords)
    return from_homog(np.einsum('H W j, i j -> H W i', cam2_coords, K2))


def induced_flow(depth0, depth1, data):
    """Compute induced optical flow from depth maps and poses."""
    H, W = depth0.shape
    coords1 = reproject(depth0, data['T0'], data['T1'], data['K0'], data['K1'])
    x, y = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    coords0 = np.stack([x, y], axis=-1)
    flow_01 = coords1 - coords0

    H, W = depth1.shape
    coords1 = reproject(depth1, data['T1'], data['T0'], data['K1'], data['K0'])
    x, y = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    coords0 = np.stack([x, y], axis=-1)
    flow_10 = coords1 - coords0

    return flow_01, flow_10


def check_cycle_consistency(flow_01, flow_10):
    """Check cycle consistency of bidirectional flows."""
    flow_01 = torch.from_numpy(flow_01).permute(2, 0, 1)[None]
    flow_10 = torch.from_numpy(flow_10).permute(2, 0, 1)[None]
    H, W = flow_01.shape[-2:]
    coords = coords_grid(1, H, W, flow_01.device)
    coords1 = coords + flow_01
    flow_reprojected = bilinear_sampler(flow_10, coords1.permute(0, 2, 3, 1))
    cycle = flow_reprojected + flow_01
    cycle = torch.norm(cycle, dim=1)
    mask = (cycle < 0.1 * min(H, W)).float()
    return mask[0].numpy()