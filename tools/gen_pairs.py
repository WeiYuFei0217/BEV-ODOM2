"""Generate the training pair lists for enhanced rotation sampling.

For every frame i of a training sequence, each other frame j with |t_j - t_i| <= pair_window_s and
||t_rel||_2 <= pair_max_dist (T_rel = G_j^{-1} G_i) goes to S_high if its absolute relative yaw lies in
pair_high_rot_deg, or to S_standard if it is below that range. All parameters are read from the yaml.
Output: pickle of [S_standard, S_high], each a per-frame list of ascending frame indices.

Example:
    python tools/gen_pairs.py -c bevodom2/config_files/NCLT.yaml
    python tools/gen_pairs.py -c bevodom2/config_files/Oxford.yaml --out_root /tmp/pairs --sequences 2019-01-11-13-24-51
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def nclt_frames(data_root, date):
    """NCLT: timestamps (s) and 6-DoF poses of the frames, in NCLTSequence order."""
    from bevodom2.datasets.nclt import NCLTSequence, pose_6dof
    ds = NCLTSequence(data_root, date, phase="pairs")
    t_img = np.array([float(name.split('.')[0]) for name in ds.image_names])
    # Nearest ground-truth row, as in NCLTSequence.nearest_gt_row (ties resolve to the earlier row)
    t_gt = ds.gt[0].to_numpy(dtype=np.float64)
    if np.any(np.diff(t_gt) < 0):
        raise ValueError(f"Ground truth timestamps are not sorted: {date}")
    right = np.clip(np.searchsorted(t_gt, t_img), 1, len(t_gt) - 1)
    rows = np.where(t_img - t_gt[right - 1] <= t_gt[right] - t_img, right - 1, right)
    gt = ds.gt.to_numpy(dtype=np.float64)[rows]
    poses = np.stack([pose_6dof(*row[1:7]) for row in gt])
    return t_img * 1e-6, poses


def oxford_frames(data_root, seq):
    """Oxford: timestamps (s) and 6-DoF poses of the frames, in OxfordSequence order."""
    from bevodom2.datasets.oxford import OxfordSequence
    ds = OxfordSequence(data_root, seq, phase="pairs")
    return ds.timestamps.astype(np.float64) * 1e-6, ds.poses


def gen_pairs(timestamps, poses, window_s, max_dist, high_rot_deg):
    """Returns (S_standard, S_high)."""
    if np.any(np.diff(timestamps) < 0):
        raise ValueError("Frame timestamps are not sorted")
    R, p = poses[:, :3, :3], poses[:, :3, 3]
    lo_deg, hi_deg = high_rot_deg
    lo = np.searchsorted(timestamps, timestamps - window_s, side='left')
    hi = np.searchsorted(timestamps, timestamps + window_s, side='right')
    s_std, s_high = [], []
    for i in range(len(timestamps)):
        j = np.arange(lo[i], hi[i])
        j = j[j != i]
        # T_rel = G_j^{-1} G_i: R_rel = R_j^T R_i, ||t_rel|| = ||p_i - p_j||
        dist = np.linalg.norm(p[i] - p[j], axis=1)
        R_rel = np.einsum('nki,kl->nil', R[j], R[i])
        yaw = np.degrees(np.abs(np.arctan2(R_rel[:, 1, 0], R_rel[:, 0, 0])))
        near = dist <= max_dist
        s_std.append(j[near & (yaw < lo_deg)].tolist())
        s_high.append(j[near & (yaw >= lo_deg) & (yaw <= hi_deg)].tolist())
    return s_std, s_high


def main():
    parser = argparse.ArgumentParser(description='Generate enhanced rotation sampling pair lists.')
    parser.add_argument('-c', '--config', required=True, help='dataset yaml (NCLT.yaml / Oxford.yaml)')
    parser.add_argument('--sequences', nargs='+', default=None, help='default: train_sequences in the yaml')
    parser.add_argument('--out_root', default=None, help='default: pair_root in the yaml')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    dataset_type, params = cfg['dataset_type'], cfg['train_conf']
    if dataset_type not in ('NCLT', 'oxford'):
        raise ValueError(f"dataset_type must be 'NCLT' or 'oxford', got {dataset_type!r}")
    out_root = args.out_root or cfg['pair_root']
    sequences = args.sequences or params['train_sequences']
    load_frames = nclt_frames if dataset_type == 'NCLT' else oxford_frames

    for seq in sequences:
        out_path = os.path.join(out_root, seq, params['pair_file'])
        if os.path.exists(out_path) and not args.overwrite:
            raise FileExistsError(f"{out_path} exists (use --overwrite)")
        timestamps, poses = load_frames(cfg['data_root'], seq)
        s_std, s_high = gen_pairs(timestamps, poses, params['pair_window_s'],
                                  params['pair_max_dist'], params['pair_high_rot_deg'])
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'wb') as f:
            pickle.dump([s_std, s_high], f)
        n_high = sum(len(v) > 0 for v in s_high)
        print(f"{seq}: {len(s_std)} frames, S_standard {sum(map(len, s_std))} pairs, "
              f"S_high {sum(map(len, s_high))} pairs ({n_high} frames with S_high) -> {out_path}")


if __name__ == '__main__':
    main()
