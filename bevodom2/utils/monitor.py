"""TensorBoard logging: training losses, val/test metrics and top-view trajectory plots.

Example:
    monitor = Monitor(log_dir)
    monitor.log_train_step({'loss': 1.2, 'rot_loss': 0.1})
    monitor.log_trajectory('val', 1, T_gt, T_pred)
"""
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor

IMG_SIZE = 1000
MARGIN = 50


class Monitor:
    """SummaryWriter wrapper; counter is the training step (restored from last.pth on resume)."""

    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        self.counter = 0

    def log_train_step(self, values):
        self.counter += 1
        for key, value in values.items():
            self.writer.add_scalar(f'train/{key}', value, self.counter)

    def log_split_mean(self, split, results, keys):
        """Mean of each metric over the sequences of a split."""
        for key in keys:
            self.writer.add_scalar(f'{split}/{key}', float(np.mean([m[key] for m in results])), self.counter)

    def log_trajectory(self, split, seq_index, T_gt, T_pred):
        self.writer.add_image(f'{split}/trajectory_{seq_index}',
                              ToTensor()(draw_trajectories(T_gt, T_pred)), self.counter)


def planar_positions(rel_poses):
    """Top-view (x, y) positions from chained relative poses (y/z axes flipped for display)."""
    T_flip = np.diag([1.0, -1.0, -1.0, 1.0])
    T_acc = T_flip.copy()
    xs, ys = [], []
    for T in rel_poses[:-1]:
        T_acc = np.matmul(T, T_acc)
        R, t = T_acc[:3, :3], T_acc[:3, 3]
        position = -R.T @ t
        xs.append(position[0])
        ys.append(position[1])
    return xs, ys


def draw_trajectories(T_gt, T_pred):
    """Top-view trajectory plot: ground truth in black, prediction in blue."""
    x_gt, y_gt = planar_positions(T_gt)
    x_pred, y_pred = planar_positions(T_pred)
    img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color='white')
    xs, ys = x_gt + x_pred, y_gt + y_pred
    if not xs or not np.all(np.isfinite(xs + ys)):
        return img
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)
    scale = (IMG_SIZE - 2 * MARGIN) / span
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    draw = ImageDraw.Draw(img)

    def to_pixels(x, y):
        return [((a - cx) * scale + IMG_SIZE / 2, (b - cy) * scale + IMG_SIZE / 2) for a, b in zip(x, y)]

    draw.line(to_pixels(x_gt, y_gt), fill='black', width=3)
    draw.line(to_pixels(x_pred, y_pred), fill='blue', width=3)
    draw.text((100, 50), 'black: ground truth', fill='black')
    draw.text((100, 65), 'blue: prediction', fill='blue')
    return img
