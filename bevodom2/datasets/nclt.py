"""NCLT monocular (Cam5) sequence dataset.

Layout of data_root/<date>/:
    lb3_u_s_384/Cam1, Cam5/<timestamp>.jpg   images (the Cam1 file names define the frame order)
    ground_truth/groundtruth_<date>.csv      or groundtruth_<date>.csv (t, x, y, z, roll, pitch, yaw)

Example:
    ds = NCLTSequence('/path/to/NCLT', '2012-02-02', phase='eval', frame_stride=5)
    images, poses, poses_3dof, timestamp = ds[0]
"""
import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from bevodom2.datasets import IMAGE_TRANSFORM, load_pairs, pose_3dof, sample_pair


def gt_csv_path(data_root, date):
    """Ground-truth file: in ground_truth/ for most sessions, in the session root for 2012-05-26."""
    candidates = [os.path.join(data_root, date, "ground_truth", f"groundtruth_{date}.csv"),
                  os.path.join(data_root, date, f"groundtruth_{date}.csv")]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"NCLT ground truth not found: {candidates}")


def pose_6dof(x, y, z, roll, pitch, yaw):
    """R = Rz(yaw) Ry(pitch) Rx(roll) (NCLT convention, radians); returns a 4x4 float64 matrix."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr, x],
                     [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr, y],
                     [-sp,   cp*sr,            cp*cr,            z],
                     [0, 0, 0, 1]])


class NCLTSequence(Dataset):
    def __init__(self, data_root, date, phase, frame_stride=5, pair_file=None):
        if phase not in ('train', 'eval', 'pairs'):
            raise ValueError(f"Unknown phase: {phase}")
        self.phase = phase
        self.image_dir = os.path.join(data_root, date, "lb3_u_s_384")
        self.gt = pd.read_csv(gt_csv_path(data_root, date), header=None)
        self.image_names = sorted(os.listdir(os.path.join(self.image_dir, "Cam1")))

        if phase == 'train':
            self.s_std, self.s_high = load_pairs(pair_file, len(self.image_names))
            self.frames = self.image_names[:-1]
        elif phase == 'eval':
            self.frames = self.image_names[::frame_stride]
        else:
            self.frames = self.image_names

    def __len__(self):
        return len(self.frames) - 1 if self.phase == 'eval' else len(self.frames)

    def nearest_gt_row(self, timestamp):
        """Ground-truth row closest to the image timestamp (ties resolve to the earlier row)."""
        return self.gt.iloc[(self.gt[0] - timestamp).abs().idxmin()]

    def __getitem__(self, idx):
        if self.phase == 'train':
            idx1, idx2 = sample_pair(idx, self.s_std, self.s_high, len(self))
            names = [self.image_names[idx1], self.image_names[idx2]]
        elif self.phase == 'eval':
            names = [self.frames[idx], self.frames[idx + 1]]
        else:
            raise RuntimeError("phase 'pairs' provides frame poses only")

        images = []
        poses = np.zeros((2, 4, 4), dtype=np.float64)
        poses_3dof = np.zeros((2, 4, 4), dtype=np.float64)
        timestamp = None
        for k, name in enumerate(names):
            stem = name.split('.')[0]
            row = self.nearest_gt_row(float(stem)).iloc[:].tolist()
            if timestamp is None:
                timestamp = float(row[0])
            x, y, z, roll, pitch, yaw = (float(v) for v in row[1:7])
            poses[k] = pose_6dof(x, y, z, roll, pitch, yaw)
            poses_3dof[k] = pose_3dof(x, y, yaw)
            images.append(self.load_image(stem))
        return torch.stack(images), poses, poses_3dof, timestamp

    def load_image(self, stem):
        path = os.path.join(self.image_dir, "Cam5", f"{stem}.jpg")
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        return IMAGE_TRANSFORM(image)
