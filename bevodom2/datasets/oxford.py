"""Oxford Radar RobotCar monocular (mono_rear) sequence dataset.

Frames are defined by the velodyne_left/*.bin timestamps; each frame uses the nearest mono_rear_rect
image and gps/ins.csv pose.
Layout of data_root/<seq>/: velodyne_left/, mono_rear_rect/, mono_rear.timestamps, gps/ins.csv.

Example:
    ds = OxfordSequence('/path/to/oxford', '2019-01-11-12-26-55', phase='eval', frame_stride=5)
    images, poses, poses_3dof, timestamp = ds[0]
"""
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from bevodom2.datasets import IMAGE_TRANSFORM, load_pairs, pose_3dof, sample_pair

POSE_TIME_TOLERANCE = 1.0


def find_nearest_ndx(ts, timestamps):
    """Index of the timestamp closest to ts in a sorted array."""
    ndx = np.searchsorted(timestamps, ts)
    if ndx == 0:
        return ndx
    if ndx == len(timestamps):
        return ndx - 1
    assert timestamps[ndx - 1] <= ts <= timestamps[ndx]
    return ndx - 1 if ts - timestamps[ndx - 1] < timestamps[ndx] - ts else ndx


def read_ts_file(ts_filepath):
    """Read <sensor>.timestamps into an int64 array."""
    with open(ts_filepath, "r") as h:
        lines = h.readlines()
    ts = np.zeros((len(lines),), dtype=np.int64)
    for ndx, line in enumerate(lines):
        fields = [e.strip() for e in line.split(' ')]
        assert len(fields) == 2, f'Invalid line in timestamp file: {fields}'
        ts[ndx] = int(fields[0])
    return ts


def pose_6dof(x, y, z, roll, pitch, yaw):
    """R = Rz(yaw) Ry(pitch) Rx(roll) (as in the RobotCar SDK build_se3_transform); returns a 4x4 float64 matrix."""
    se3 = np.eye(4, dtype=np.float64)
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    se3[:3, :3] = np.dot(R_z, np.dot(R_y, R_x))
    se3[:3, 3] = np.array([x, y, z])
    return se3


def read_lidar_poses(poses_filepath, left_lidar_dir, pose_time_tolerance=POSE_TIME_TOLERANCE):
    """Match INS poses to the left LiDAR scan timestamps. Returns (timestamps, 6-DoF poses, planar poses).

    Timestamps are in microseconds; scans whose nearest pose is farther than
    pose_time_tolerance * 1e7 are dropped.
    """
    with open(poses_filepath, "r") as h:
        lines = h.readlines()[1:]  # skip the header row
    n = len(lines)
    system_timestamps = np.zeros((n,), dtype=np.int64)
    poses = np.zeros((n, 4, 4), dtype=np.float64)
    poses_3dof = np.zeros((n, 4, 4), dtype=np.float64)
    for ndx, line in enumerate(lines):
        fields = [e.strip() for e in line.split(',')]
        assert len(fields) == 15, f'Invalid line in global poses file: {fields}'
        x, y, z, roll, pitch, yaw = (float(fields[i]) for i in (5, 6, 7, 12, 13, 14))
        system_timestamps[ndx] = int(fields[0])
        poses[ndx] = pose_6dof(x, y, z, roll, pitch, yaw)
        poses_3dof[ndx] = pose_3dof(x, y, yaw)

    sorted_ndx = np.argsort(system_timestamps, axis=0)
    system_timestamps = system_timestamps[sorted_ndx]
    poses, poses_3dof = poses[sorted_ndx], poses_3dof[sorted_ndx]

    lidar_timestamps = sorted(int(os.path.splitext(f)[0]) for f in os.listdir(left_lidar_dir)
                              if os.path.splitext(f)[1] == '.bin')
    keep_ts, keep = [], []
    for lidar_ts in lidar_timestamps:
        closest = find_nearest_ndx(lidar_ts, system_timestamps)
        if abs(system_timestamps[closest] - lidar_ts) > pose_time_tolerance * 10000000:
            continue
        keep_ts.append(lidar_ts)
        keep.append(closest)
    return np.array(keep_ts, dtype=np.int64), poses[keep], poses_3dof[keep]


class OxfordSequence(Dataset):
    def __init__(self, data_root, seq, phase, frame_stride=5, pair_file=None):
        if phase not in ('train', 'eval', 'pairs'):
            raise ValueError(f"Unknown phase: {phase}")
        self.phase = phase
        self.seq_path = os.path.join(os.path.expanduser(data_root), seq)
        pose_file = os.path.join(self.seq_path, 'gps', 'ins.csv')
        left_lidar_dir = os.path.join(self.seq_path, 'velodyne_left')
        for path in (pose_file, left_lidar_dir):
            if not os.path.exists(path):
                raise FileNotFoundError(f'Cannot access {path}')

        self.mono_rear_ts = read_ts_file(os.path.join(self.seq_path, 'mono_rear.timestamps'))
        self.timestamps, self.poses, self.poses_3dof = read_lidar_poses(pose_file, left_lidar_dir)
        if phase == 'eval':
            self.timestamps = self.timestamps[::frame_stride]
            self.poses = self.poses[::frame_stride]
            self.poses_3dof = self.poses_3dof[::frame_stride]
        if phase == 'train':
            self.s_std, self.s_high = load_pairs(pair_file, len(self.timestamps))

    def __len__(self):
        return len(self.timestamps) - 1 if self.phase == 'eval' else len(self.timestamps)

    def __getitem__(self, idx):
        if self.phase == 'train':
            frames = sample_pair(idx, self.s_std, self.s_high, len(self))
        elif self.phase == 'eval':
            frames = (idx, idx + 1)
        else:
            raise RuntimeError("phase 'pairs' provides frame poses only")

        images = [self.load_image(i) for i in frames]
        poses = np.stack([self.poses[i] for i in frames])
        poses_3dof = np.stack([self.poses_3dof[i] for i in frames])
        return torch.stack(images), poses, poses_3dof, float(self.timestamps[frames[0]])

    def load_image(self, idx):
        image_ts = self.mono_rear_ts[find_nearest_ndx(self.timestamps[idx], self.mono_rear_ts)]
        path = os.path.join(self.seq_path, 'mono_rear_rect', f'{image_ts}.png')
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f'Cannot read image: {path}')
        return IMAGE_TRANSFORM(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
