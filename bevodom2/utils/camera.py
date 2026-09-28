"""Monocular camera intrinsics/extrinsics and the camera matrix dict (mats_dict) of the backbone.

image_meta.pkl in data_root holds 'K' (intrinsics) and 'T' (cam_T_body extrinsics) per camera.
NCLT uses Cam5 (last entry); Oxford uses mono_rear (third of the first three entries).

Example:
    intrin, cam_T_body = load_mono_camera(data_root, 'NCLT')
    mats_dict = build_mats_dict(intrin, cam_T_body, n=32)
"""
import os
import pickle

import numpy as np
import torch


def load_mono_camera(data_root, dataset_type):
    """Returns (intrinsics (1, 4, 4), cam_T_body (1, 1, 4, 4)) as float32 CPU tensors."""
    with open(os.path.join(data_root, "image_meta.pkl"), 'rb') as handle:
        image_meta = pickle.load(handle)
    K, T = image_meta['K'], image_meta['T']
    if dataset_type == 'oxford':
        K, T = K[:3], T[:3]
    intrins = torch.from_numpy(np.array(K)).float()[-1:, ...]
    cam_T_body = torch.from_numpy(np.array(T)).unsqueeze(0).float()[:1, -1:, ...]
    return intrinsics_4x4(intrins).unsqueeze(0), cam_T_body


def intrinsics_4x4(K):
    """(N, 3|4, 3|4) intrinsics -> (N, 4, 4) keeping fx, fy, cx, cy."""
    out = torch.zeros(K.shape[0], 4, 4, dtype=torch.float32, device=K.device)
    out[:, 0, 0] = K[:, 0, 0]
    out[:, 1, 1] = K[:, 1, 1]
    out[:, 0, 2] = K[:, 0, 2]
    out[:, 1, 2] = K[:, 1, 2]
    out[:, 2, 2] = 1.0
    out[:, 3, 3] = 1.0
    return out


def rigid_inverse(a):
    """Batched inverse of rigid transforms (N, 4, 4)."""
    inv = a.clone()
    r_transpose = a[:, :3, :3].transpose(1, 2)
    inv[:, :3, :3] = r_transpose
    inv[:, :3, 3:4] = -torch.matmul(r_transpose, a[:, :3, 3:4])
    return inv


def build_mats_dict(intrin, cam_T_body, n):
    """Camera matrices shared by n images (GPU): (n, 1, 1, 4, 4), bda_mat (n, 4, 4); no augmentation."""
    pix_T_cams = intrin.repeat(n, 1, 1, 1).cuda()
    cams_T_body = cam_T_body.repeat(n, 1, 1, 1).cuda()
    body_T_cams = rigid_inverse(cams_T_body.reshape(-1, 4, 4)).reshape(n, -1, 4, 4)
    ida_mats = torch.from_numpy(np.eye(4)).repeat(n, 1, 1).cuda().view(n, 1, 1, 4, 4)
    bda_mat = torch.from_numpy(np.eye(4)).repeat(n, 1, 1).cuda()
    return {
        'sensor2ego_mats': body_T_cams.view(n, 1, 1, 4, 4).float(),
        'intrin_mats': pix_T_cams.view(n, 1, 1, 4, 4).float(),
        'ida_mats': ida_mats.float(),
        'bda_mat': bda_mat.float(),
    }
