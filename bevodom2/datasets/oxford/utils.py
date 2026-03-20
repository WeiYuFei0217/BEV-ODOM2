"""
Oxford RobotCar dataset utility functions (simplified 2D version).

Provides pose reading, coordinate transformation, and point cloud operations.
RPY2Rot uses a simplified 2D rotation (yaw + xy translation only), suitable
for planar motion assumptions.
"""

import os

import numpy as np
import torch
from scipy.spatial import distance_matrix

from bevodom2.datasets.oxford.python.transform import *

# Faulty point clouds (scans with 0 points)
FAULTY_POINTCLOUDS = []

# Test region centre coordinates (Oxford sequences)
TEST_REGION_CENTRES = np.array([[5735400, 620000]])

# Test region radius
TEST_REGION_RADIUS = 220

# Buffer distance between train/test regions to prevent overlap
TEST_TRAIN_BOUNDARY = 50


def in_train_split(pos):
    """Check whether positions fall within the training region."""
    assert pos.ndim == 2
    assert pos.shape[1] == 2
    dist = distance_matrix(pos, TEST_REGION_CENTRES)
    mask = (dist > TEST_REGION_RADIUS + TEST_TRAIN_BOUNDARY).all(axis=1)
    return mask


def in_test_split(pos):
    """Check whether positions fall within the test region."""
    assert pos.ndim == 2
    assert pos.shape[1] == 2
    dist = distance_matrix(pos, TEST_REGION_CENTRES)
    mask = (dist < TEST_REGION_RADIUS).any(axis=1)
    return mask


def find_nearest_ndx(ts, timestamps):
    """Find the index of the closest timestamp in a sorted array."""
    ndx = np.searchsorted(timestamps, ts)
    if ndx == 0:
        return ndx
    elif ndx == len(timestamps):
        return ndx - 1
    else:
        assert timestamps[ndx - 1] <= ts <= timestamps[ndx]
        if ts - timestamps[ndx - 1] < timestamps[ndx] - ts:
            return ndx - 1
        else:
            return ndx


def read_ts_file(ts_filepath: str):
    """Read a timestamp file and return an int64 timestamp array."""
    with open(ts_filepath, "r") as h:
        txt_ts = h.readlines()

    n = len(txt_ts)
    ts = np.zeros((n,), dtype=np.int64)

    for ndx, timestamp in enumerate(txt_ts):
        temp = [e.strip() for e in timestamp.split(' ')]
        assert len(temp) == 2, f'Invalid line in timestamp file: {temp}'
        ts[ndx] = int(temp[0])

    return ts


def read_lidar_poses(poses_filepath: str, left_lidar_filepath: str,
                     pose_time_tolerance: float = 1.):
    """Read global poses from CSV and match them to LiDAR scan timestamps.

    Args:
        poses_filepath: Path to the global poses CSV file.
        left_lidar_filepath: Directory containing LiDAR scan .bin files.
        pose_time_tolerance: Time tolerance in seconds.

    Returns:
        lidar_timestamps: Array of matched LiDAR timestamps.
        lidar_poses: Array of corresponding 4x4 pose matrices.
    """
    with open(poses_filepath, "r") as h:
        txt_poses = h.readlines()

    n = len(txt_poses)
    print("pose num:", n)
    system_timestamps = np.zeros((n,), dtype=np.int64)
    poses = np.zeros((n, 4, 4), dtype=np.float64)

    for ndx, pose in enumerate(txt_poses):
        temp = [e.strip() for e in pose.split(',')]
        if ndx == 0:
            continue
        ndx -= 1
        assert len(temp) == 15, f'Invalid line in global poses file: {temp}'
        system_timestamps[ndx] = int(temp[0])
        poses[ndx] = RPY2Rot(
            float(temp[5]), float(temp[6]), float(temp[7]),
            float(temp[12]), float(temp[13]), float(temp[14])
        )

    # Sort by timestamp in ascending order
    sorted_ndx = np.argsort(system_timestamps, axis=0)
    system_timestamps = system_timestamps[sorted_ndx]
    poses = poses[sorted_ndx]

    # Collect LiDAR scan timestamps
    left_lidar_timestamps = [
        int(os.path.splitext(f)[0])
        for f in os.listdir(left_lidar_filepath)
        if os.path.splitext(f)[1] == '.bin'
    ]
    left_lidar_timestamps.sort()

    lidar_timestamps = []
    lidar_poses = []
    count_rejected = 0

    for ndx, lidar_ts in enumerate(left_lidar_timestamps):
        if lidar_ts in FAULTY_POINTCLOUDS:
            continue

        closest_ts_ndx = find_nearest_ndx(lidar_ts, system_timestamps)
        delta = abs(system_timestamps[closest_ts_ndx] - lidar_ts)
        # Timestamps are in nanoseconds (1e-9 second)
        if delta > pose_time_tolerance * 10000000:
            count_rejected += 1
            continue

        lidar_timestamps.append(lidar_ts)
        lidar_poses.append(poses[closest_ts_ndx])

    lidar_timestamps = np.array(lidar_timestamps, dtype=np.int64)
    lidar_poses = np.array(lidar_poses, dtype=np.float64)

    print(f'{len(lidar_timestamps)} scans with valid pose, '
          f'{count_rejected} rejected due to unknown pose')
    return lidar_timestamps, lidar_poses


def read_lidar_poses_RPY(poses_filepath: str, left_lidar_filepath: str,
                         pose_time_tolerance: float = 1.):
    """Read global poses and match to LiDAR timestamps, also returning RPY angles.

    Args:
        poses_filepath: Path to the global poses CSV file.
        left_lidar_filepath: Directory containing LiDAR scan .bin files.
        pose_time_tolerance: Time tolerance in seconds.

    Returns:
        lidar_timestamps: Array of matched LiDAR timestamps.
        lidar_poses: Array of corresponding 4x4 pose matrices.
        lidar_poses_RPY: Array of 6DoF Euler angles [x, y, z, roll, pitch, yaw].
    """
    with open(poses_filepath, "r") as h:
        txt_poses = h.readlines()

    n = len(txt_poses)
    print("pose num:", n)
    system_timestamps = np.zeros((n,), dtype=np.int64)
    poses = np.zeros((n, 4, 4), dtype=np.float64)
    poses_RPY = np.zeros((n, 6), dtype=np.float64)

    for ndx, pose in enumerate(txt_poses):
        temp = [e.strip() for e in pose.split(',')]
        if ndx == 0:
            continue
        ndx -= 1
        assert len(temp) == 15, f'Invalid line in global poses file: {temp}'
        system_timestamps[ndx] = int(temp[0])
        poses[ndx] = RPY2Rot(
            float(temp[5]), float(temp[6]), float(temp[7]),
            float(temp[12]), float(temp[13]), float(temp[14])
        )
        poses_RPY[ndx] = [
            float(temp[5]), float(temp[6]), float(temp[7]),
            float(temp[12]), float(temp[13]), float(temp[14])
        ]

    # Sort by timestamp in ascending order
    sorted_ndx = np.argsort(system_timestamps, axis=0)
    system_timestamps = system_timestamps[sorted_ndx]
    poses = poses[sorted_ndx]
    poses_RPY = poses_RPY[sorted_ndx]

    # Collect LiDAR scan timestamps
    left_lidar_timestamps = [
        int(os.path.splitext(f)[0])
        for f in os.listdir(left_lidar_filepath)
        if os.path.splitext(f)[1] == '.bin'
    ]
    left_lidar_timestamps.sort()

    lidar_timestamps = []
    lidar_poses = []
    lidar_poses_RPY = []
    count_rejected = 0

    for ndx, lidar_ts in enumerate(left_lidar_timestamps):
        if lidar_ts in FAULTY_POINTCLOUDS:
            continue

        closest_ts_ndx = find_nearest_ndx(lidar_ts, system_timestamps)
        delta = abs(system_timestamps[closest_ts_ndx] - lidar_ts)
        if delta > pose_time_tolerance * 10000000:
            count_rejected += 1
            continue

        lidar_timestamps.append(lidar_ts)
        lidar_poses.append(poses[closest_ts_ndx])
        lidar_poses_RPY.append(poses_RPY[closest_ts_ndx])

    lidar_timestamps = np.array(lidar_timestamps, dtype=np.int64)
    lidar_poses = np.array(lidar_poses, dtype=np.float64)
    lidar_poses_RPY = np.array(lidar_poses_RPY, dtype=np.float64)

    print(f'{len(lidar_timestamps)} scans with valid pose, '
          f'{count_rejected} rejected due to unknown pose')
    return lidar_timestamps, lidar_poses, lidar_poses_RPY


def relative_pose(m1, m2):
    """Compute relative transformation between two poses.

    SE(3) pose is a 4x4 matrix:
        Pw = [R | T] @ [P]
             [0 | 1]   [1]

    Args:
        m1: Transformation from frame 1 to world.
        m2: Transformation from frame 2 to world.

    Returns:
        Relative transformation from frame 1 to frame 2.
    """
    return np.linalg.inv(m2) @ m1


def RPY2Rot(x, y, z, roll, pitch, yaw):
    """Convert translation and RPY Euler angles to a 4x4 homogeneous matrix (simplified 2D).

    Uses yaw rotation and xy translation only, suitable for planar motion.
    """
    T = np.identity(4, dtype=np.float32)
    T[0:2, 0:2] = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw),  np.cos(yaw)]
    ])
    T[0, 3] = x
    T[1, 3] = y
    return T


def random_rotation(xyz, angle_range=(-np.pi, np.pi)):
    """Apply a random rotation around the Z axis."""
    angle = np.random.uniform(*angle_range)
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0,              0,             1]
    ]).transpose()
    return np.dot(xyz, rotation_matrix)


def euler2se3(x, y, z, roll, pitch, yaw):
    """Convert Euler angles and translation to a 4x4 SE(3) transformation matrix."""
    se3 = np.eye(4, dtype=np.float64)
    R_x = np.array([
        [1, 0,             0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])
    R_y = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [ 0,             1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,            0,           1]
    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    se3[:3, :3] = R
    se3[:3, 3] = np.array([x, y, z])
    return se3


def apply_transform(pc: torch.Tensor, m: torch.Tensor):
    """Apply an SE(3) transformation to a point cloud.

    Supports 4x4 matrix on (N, 3) clouds or 3x3 matrix on (N, 2) clouds.
    """
    assert pc.ndim == 2
    n_dim = pc.shape[1]
    assert n_dim == 2 or n_dim == 3
    assert m.shape == (n_dim + 1, n_dim + 1)
    pc = pc @ m[:n_dim, :n_dim].transpose(1, 0) + m[:n_dim, -1]
    return pc
