"""NCLT / Oxford monocular sequence datasets and enhanced rotation sampling.

Each sample is a frame pair (t, t+1): images (2, 3, H, W), 6-DoF global poses (2, 4, 4),
planar global poses (2, 4, 4) and the timestamp of the first frame.
Phases:
    'train': frame i paired with a frame from its pair lists (S_high with probability 0.7, see sample_pair);
    'eval':  every frame_stride-th frame, consecutive pairs in temporal order (val/test);
    'pairs': frame timestamps and poses only (used by tools/gen_pairs.py).

Example:
    from bevodom2.datasets import build_sequences
    dataset = build_sequences('NCLT', data_root, ['2012-02-02'], phase='eval', frame_stride=5)
"""
import pickle
import random

import numpy as np
from torch.utils.data import ConcatDataset
from torchvision import transforms

HIGH_ROT_RATIO = 7  # out of 10: probability of drawing from S_high when both lists are non-empty

# RGB image -> normalized tensor (ImageNet statistics)
IMAGE_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def pose_3dof(x, y, yaw):
    """Planar pose from (x, y, yaw); returns a 4x4 float32 matrix."""
    T = np.identity(4, dtype=np.float32)
    T[0:2, 0:2] = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    T[0, 3] = x
    T[1, 3] = y
    return T


def load_pairs(pair_file, num_frames):
    """Load the pair lists [S_standard, S_high]; both must have one entry per frame."""
    with open(pair_file, 'rb') as f:
        s_std, s_high = pickle.load(f)
    if not len(s_std) == len(s_high) == num_frames:
        raise ValueError(f"Pair lists in {pair_file} do not match the {num_frames} frames of the sequence")
    return s_std, s_high


def sample_pair(idx, s_std, s_high, num_samples):
    """Draw a partner frame for idx (a new random idx is drawn if both lists are empty). Returns (idx1, idx2)."""
    std, high = s_std[idx], s_high[idx]
    while not std and not high:
        idx = random.randint(0, num_samples - 1)
        std, high = s_std[idx], s_high[idx]
    if high and std:
        pair = random.choice(high) if random.randint(1, 10) <= HIGH_ROT_RATIO else random.choice(std)
    elif high:
        pair = random.choice(high)
    else:
        pair = random.choice(std)
    return idx, pair


def build_sequences(dataset_type, data_root, sequences, phase, frame_stride=5, pair_files=None):
    """Concatenate several sequences into one dataset."""
    from bevodom2.datasets.nclt import NCLTSequence
    from bevodom2.datasets.oxford import OxfordSequence
    cls = NCLTSequence if dataset_type == 'NCLT' else OxfordSequence
    pair_files = pair_files or [None] * len(sequences)
    return ConcatDataset([cls(data_root, seq, phase, frame_stride, pair_file)
                          for seq, pair_file in zip(sequences, pair_files)])
