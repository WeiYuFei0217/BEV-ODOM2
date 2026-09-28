"""BEV-ODOM2 training and evaluation entry point (single GPU).

Train:  python bevodom2/train_model/train.py -c bevodom2/config_files/NCLT.yaml -g 0
Resume: python bevodom2/train_model/train.py -c bevodom2/config_files/NCLT.yaml -g 0 --resume <run_name>
Test:   python bevodom2/train_model/train.py -c bevodom2/config_files/NCLT.yaml -g 0 --test --weights weights/bevodom2_nclt.pth

Training evaluates on the validation sequences after every epoch, keeps the epoch with the lowest mean
validation RTE as best.pth and evaluates it once on the test sequences at the end.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import csv
import datetime
import logging
import os
import random
import warnings

import numpy as np
import torch
import yaml
from torch.multiprocessing import set_start_method
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from bevodom2.datasets import build_sequences
from bevodom2.models.pose_losses import planar_part
from bevodom2.utils.camera import build_mats_dict, load_mono_camera
from bevodom2.utils.checkpoints import load_pretrained, load_weights
from bevodom2.utils.metrics import evaluate_trajectory, save_tum_trajectory
from bevodom2.utils.monitor import Monitor

warnings.filterwarnings("ignore", category=UserWarning)

METRIC_KEYS = ('rte', 'rre', 't_err_avg', 'R_err_avg', 'ate_se3', 'ate_sim3')
TRAIN_LOG_KEYS = ('loss', 'rot_loss', 'trans_loss', 'flow_loss', 'pv_rot_loss', 'pv_trans_loss')


def parse_args():
    parser = argparse.ArgumentParser(description='BEV-ODOM2 training / evaluation')
    parser.add_argument('-c', '--config', required=True, help='dataset yaml')
    parser.add_argument('-g', '--gpu', required=True, help='ID of the single GPU to use, e.g. "0"')
    parser.add_argument('--test', action='store_true',
                        help='test only: evaluate --weights once on the sequences of --split')
    parser.add_argument('--weights', default=None, help='network weights (.pth) for --test')
    parser.add_argument('--split', choices=['test', 'val'], default='test', help='sequences evaluated in --test mode')
    parser.add_argument('--resume', default=None, help='run name to resume from <output_root>/model_save/model_<run>/last.pth')
    args = parser.parse_args()
    if ',' in args.gpu:
        parser.error('only single-GPU training/testing is supported (batch size 16 on one GPU)')
    if args.test and args.weights is None:
        parser.error('--test requires --weights')
    if not args.test and args.weights is not None:
        parser.error('--weights is only used with --test')
    if args.test and args.resume is not None:
        parser.error('--resume is only used for training')
    return args


def check_splits(splits):
    """Train/val/test sequence lists must be non-empty, free of duplicates and pairwise disjoint."""
    for name, seqs in splits.items():
        if not seqs or len(set(seqs)) != len(seqs):
            raise ValueError(f"{name}_sequences must be non-empty and free of duplicates: {seqs}")
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = sorted(set(splits[a]) & set(splits[b]))
            if overlap:
                raise ValueError(f"{a}_sequences and {b}_sequences overlap: {overlap}")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def relative_poses(poses):
    """(B, 2, 4, 4) global poses -> T_rel = G_{t+1}^{-1} G_t (CPU, float64)."""
    return torch.matmul(torch.from_numpy(np.linalg.inv(poses[:, 1].numpy())), poses[:, 0])


def print_results(split, results):
    """Print per-sequence and mean metrics."""
    print(f"================ FINAL {split.upper()} RESULTS ================")
    # RTE / RRE: 100-800 m sub-trajectories; ATE: mean position error after one global SE(3) / Sim(3) alignment
    for k, m in enumerate(results, start=1):
        print(f"Sequence {k} ({m['sequence']}) - RTE(%): {m['rte']:.6f}, RRE(deg/100m): {m['rre']:.6f}, "
              f"t_err_avg: {m['t_err_avg']:.6f}, R_err_avg: {m['R_err_avg']:.6f}, "
              f"ATE_SE3(m): {m['ate_se3']:.6f}, ATE_Sim3(m): {m['ate_sim3']:.6f}")
    if len(results) > 1:
        mean = {key: np.mean([m[key] for m in results]) for key in METRIC_KEYS}
        print("---------------- AVERAGE RESULTS ----------------")
        print(f"Mean RTE(%): {mean['rte']:.6f}, Mean RRE(deg/100m): {mean['rre']:.6f}, "
              f"Mean t_err_avg: {mean['t_err_avg']:.6f}, Mean R_err_avg: {mean['R_err_avg']:.6f}, "
              f"Mean ATE_SE3(m): {mean['ate_se3']:.6f}, Mean ATE_Sim3(m): {mean['ate_sim3']:.6f}")
    print("====================================================")


class Runner:
    def __init__(self, cfg, args):
        self.args = args
        self.dataset_type = cfg['dataset_type']
        self.data_root = cfg['data_root']
        self.train_conf = cfg['train_conf']
        self.eval_conf = cfg['eval_conf']
        self.splits = {'train': list(self.train_conf['train_sequences']),
                       'val': list(self.eval_conf['val_sequences']),
                       'test': list(self.eval_conf['test_sequences'])}
        check_splits(self.splits)
        set_seed(self.train_conf['seed'])

        output_root = cfg['output_root']
        self.run_name = args.resume or datetime.datetime.now().strftime("%y%m%d_%H%M%S") + "_" + self.dataset_type
        self.model_dir = os.path.join(output_root, "model_save", "model_" + self.run_name)
        self.traj_dir = os.path.join(output_root, "evo")
        log_dir = os.path.join(output_root, f"log_{self.run_name}")
        for path in (self.traj_dir, log_dir):
            os.makedirs(path, exist_ok=True)
        self.log_path = os.path.join(log_dir, "loss_record.txt")
        self.monitor = Monitor(log_dir)

        # Imported here so that spawned DataLoader workers do not import mmcv/mmdet/mmdet3d
        from mmcv.utils import get_logger
        # Register the 'mmcv' logger at WARNING level before mmcv does so at INFO (init_weights messages)
        get_logger('mmcv', log_level=logging.WARNING)
        from bevodom2.models.model import BEVODOM2
        model_conf, tc = cfg['model_conf'], self.train_conf
        self.net = BEVODOM2(cfg['backbone_conf'], self.dataset_type,
                            corr_patch_size=model_conf['corr_patch_size'],
                            use_leakyrelu_bn=model_conf['use_leakyrelu_bn'],
                            max_dis=model_conf['max_dis'],
                            lambda_flow=tc['lambda_flow'], alpha=tc['alpha'], beta=tc['beta']).cuda()
        self.camera = load_mono_camera(self.data_root, self.dataset_type)
        self.num_workers = tc['num_workers']

    def build_eval_loaders(self, split):
        """One loader per val/test sequence: batch size 1, consecutive pairs after frame_stride subsampling, in order."""
        return [(seq, DataLoader(build_sequences(self.dataset_type, self.data_root, [seq], 'eval',
                                                 self.eval_conf['frame_stride']),
                                 batch_size=1, shuffle=False, num_workers=self.num_workers, pin_memory=True))
                for seq in self.splits[split]]

    def evaluate(self, split, loaders, epoch):
        """Chain the predicted relative poses of each sequence and compare with the 3-DoF ground truth."""
        self.net.eval()
        mats_dict = build_mats_dict(*self.camera, n=2)
        results = []
        for k, (seq, loader) in enumerate(loaders, start=1):
            T_gt, T_pred, timestamps = [], [], []
            for images, _, poses_3dof, timestamp in tqdm(loader, desc=f'{split} {seq}'):
                with torch.no_grad():
                    out = self.net(images.reshape(-1, *images.shape[2:]).cuda(), mats_dict)
                T_gt.append(relative_poses(poses_3dof)[0].numpy())
                T = np.eye(4)
                T[:3, :3] = out.R[0].cpu().numpy()
                T[:3, 3] = out.t[0].cpu().numpy().reshape(3)
                T_pred.append(T)
                timestamps.extend(timestamp)
            if not T_pred:
                raise RuntimeError(f"Empty evaluation loader for {split} sequence {seq}")

            name = f'{k}_{split}_{self.run_name}_{epoch}.txt'
            save_tum_trajectory(os.path.join(self.traj_dir, 'tum_trajectory_gt_' + name), timestamps, T_gt)
            save_tum_trajectory(os.path.join(self.traj_dir, 'tum_trajectory_pred_' + name), timestamps, T_pred)
            metrics = evaluate_trajectory(T_gt, T_pred)
            metrics['sequence'] = seq
            results.append(metrics)
            self.monitor.log_trajectory(split, k, T_gt, T_pred)
            with open(self.log_path, "a") as f:
                f.write(f"epoch {epoch} {split} {seq}: " +
                        ", ".join(f"{key} {metrics[key]:.6f}" for key in METRIC_KEYS) + "\n")

        self.monitor.log_split_mean(split, results, METRIC_KEYS)
        torch.cuda.empty_cache()
        return results

    def test(self):
        load_weights(self.net, self.args.weights)
        print(f"Loaded weights: {self.args.weights}")
        split = self.args.split
        print_results(split, self.evaluate(split, self.build_eval_loaders(split), epoch=0))

    def build_train_loader(self):
        tc = self.train_conf
        pair_files = [os.path.join(self.pair_root, seq, tc['pair_file']) for seq in self.splits['train']]
        missing = [path for path in pair_files if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError(f"Sampling pair lists not found: {missing}. "
                                    f"Generate them with tools/gen_pairs.py -c <config>")
        dataset = build_sequences(self.dataset_type, self.data_root, self.splits['train'], 'train',
                                  pair_files=pair_files)
        return DataLoader(dataset, batch_size=tc['batch_size'], shuffle=True, drop_last=True,
                          num_workers=self.num_workers, pin_memory=True)

    def train_step(self, batch, mats_dict, optimizer):
        images, poses, _, _ = batch
        T_rel = relative_poses(poses).cuda()
        # The 3-DoF loss and the flow target use the planar part of T_rel; the 5-DoF loss uses the full T_rel
        T_planar = planar_part(T_rel)
        out = self.net(images.reshape(-1, *images.shape[2:]).cuda(), mats_dict)
        loss_bev, logs = self.net.bev_loss(out.R, out.t, T_planar, out.flow, self.net.compute_flow_gt(T_planar))
        loss_pv, logs_pv = self.net.pv_loss(out.pv_R, out.pv_t, T_rel)
        loss = loss_bev + self.train_conf['lambda_5dof'] * loss_pv

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {'loss': loss.item(), 'rot_loss': logs['R_loss'].item(), 'trans_loss': logs['t_loss'].item(),
                'flow_loss': logs['flow_loss'].item(), 'pv_rot_loss': logs_pv['R_loss'].item(),
                'pv_trans_loss': logs_pv['t_loss'].item()}

    def train(self, cfg):
        tc = self.train_conf
        self.pair_root = cfg['pair_root']
        optimizer = Adam(self.net.parameters(), lr=tc['lr'], weight_decay=tc['weight_decay'])
        scheduler = ExponentialLR(optimizer, gamma=tc['lr_decay'])
        best_path = os.path.join(self.model_dir, "best.pth")   # network weights of the selected epoch
        last_path = os.path.join(self.model_dir, "last.pth")   # full training state for --resume
        val_csv_path = os.path.join(self.model_dir, "val_metrics.csv")
        os.makedirs(self.model_dir, exist_ok=True)

        start_epoch, best_val_rte, best_epoch = 0, None, None
        if self.args.resume:
            # last.pth is written by this script and holds optimizer state, so it is read without weights_only
            checkpoint = load_weights(self.net, last_path, trusted_checkpoint=True)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch, self.monitor.counter = checkpoint['epoch'], checkpoint['counter']
            best_val_rte, best_epoch = checkpoint['best_val_rte'], checkpoint['best_epoch']
            print(f"Resumed {self.run_name} after epoch {start_epoch}")
        elif tc.get('pretrained'):
            frozen = load_pretrained(self.net, tc['pretrained'])
            for name, param in self.net.named_parameters():
                if name in frozen:
                    param.requires_grad = False
            print(f"Loaded {len(frozen)} pretrained tensors from {tc['pretrained']}; "
                  f"frozen for the first {tc['freeze_epochs']} epoch(s)")

        train_loader = self.build_train_loader()
        val_loaders = self.build_eval_loaders('val')
        mats_dict = build_mats_dict(*self.camera, n=2 * tc['batch_size'])
        with open(self.log_path, "a") as f:
            f.write(f"Start: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")

        for epoch in range(start_epoch + 1, tc['max_epochs'] + 1):
            print(f"{self.run_name} epoch {epoch}")
            if epoch > tc['freeze_epochs']:
                for param in self.net.backbone.parameters():
                    param.requires_grad = True
            self.net.train()
            sums = dict.fromkeys(TRAIN_LOG_KEYS, 0.0)
            for batch in tqdm(train_loader):
                values = self.train_step(batch, mats_dict, optimizer)
                self.monitor.log_train_step(values)
                for key in TRAIN_LOG_KEYS:
                    sums[key] += values[key]
            scheduler.step()
            with open(self.log_path, "a") as f:
                f.write(f"epoch {epoch} train: " +
                        ", ".join(f"{key} {sums[key] / len(train_loader):.6f}" for key in TRAIN_LOG_KEYS) + "\n")

            # Checkpoint selection: RTE averaged over the validation sequences
            val_results = self.evaluate('val', val_loaders, epoch)
            val_mean = {key: float(np.mean([m[key] for m in val_results])) for key in METRIC_KEYS}
            new_csv = not os.path.exists(val_csv_path)
            with open(val_csv_path, "a", newline='') as f:
                writer = csv.writer(f)
                if new_csv:
                    writer.writerow(['epoch', 'sequence', 'rte_pct', 'rre_deg_per_100m', 'ate_se3_m', 'ate_sim3_m'])
                for seq, m in [(m['sequence'], m) for m in val_results] + [('__mean__', val_mean)]:
                    writer.writerow([epoch, seq] + [f"{m[key]:.6f}" for key in ('rte', 'rre', 'ate_se3', 'ate_sim3')])

            # Update only on a strict improvement (ties keep the earlier epoch)
            if best_val_rte is None or val_mean['rte'] < best_val_rte:
                best_val_rte, best_epoch = val_mean['rte'], epoch
                torch.save(self.net.state_dict(), best_path)
            torch.save({
                'model_state_dict': self.net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'counter': self.monitor.counter,
                'epoch': epoch,
                'best_val_rte': best_val_rte,
                'best_epoch': best_epoch,
            }, last_path)
            print(f"epoch {epoch}: mean val RTE {val_mean['rte']:.6f}; best {best_val_rte:.6f} at epoch {best_epoch}")

        # Evaluate the selected checkpoint once on the test sequences
        if best_epoch is None:
            raise RuntimeError(f"No validation-selected checkpoint in {self.model_dir}")
        print(f"Loading best checkpoint (epoch {best_epoch}, mean val RTE {best_val_rte:.6f}): {best_path}")
        load_weights(self.net, best_path)
        print_results('test', self.evaluate('test', self.build_eval_loaders('test'), epoch=best_epoch))


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    runner = Runner(cfg, args)
    if args.test:
        runner.test()
    else:
        runner.train(cfg)


if __name__ == "__main__":
    main()
