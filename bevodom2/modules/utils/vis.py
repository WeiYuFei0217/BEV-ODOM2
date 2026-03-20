"""
Visualization utilities for BEV feature maps, keypoint matches, and trajectories.

Includes:
- convert_plt_to_img/tensor: matplotlib figure to PIL/tensor conversion
- draw_batch: visualize radar scans, scores, and keypoint matches
- draw_batch_all_for_casestudy/2: batch-level case study visualizations
- draw_matches: overlay keypoint matches on radar images
- draw_batch_steam: steam-based keypoint visualization
- histogram_equalization, min_max_scaling: image enhancement
- plot_sequences: top-down trajectory plotting
- plot_2d_trajectory: 2D trajectory comparison
- draw_plot: PIL-based trajectory drawing
"""

import io
import os

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import PIL.Image
import torch
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from PIL import Image, ImageDraw
from matplotlib import cm

from bevodom2.modules.utils.utils import (
    enforce_orthog, get_inverse_tf, get_T_ba,
    getApproxTimeStamps, wrapto2pi,
)


# ===========================================================================
# Plot Conversion Helpers
# ===========================================================================

def convert_plt_to_img():
    """Convert current matplotlib figure to PIL Image."""
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return PIL.Image.open(buf)


def convert_plt_to_tensor():
    """Convert current matplotlib figure to torch tensor."""
    return ToTensor()(convert_plt_to_img())


# ===========================================================================
# Image Enhancement
# ===========================================================================

def histogram_equalization(img):
    """Apply histogram equalization to a single-channel image."""
    flat_img = img.flatten()
    histogram, bin_edges = np.histogram(flat_img, bins=256, range=(flat_img.min(), flat_img.max()))
    cdf = histogram.cumsum()
    cdf_normalized = cdf / cdf.max()
    equalized_img = np.interp(flat_img, bin_edges[:-1], cdf_normalized)
    equalized_img = equalized_img.reshape(img.shape)
    return equalized_img


def min_max_scaling(img):
    """Scale image to [0, 1] range."""
    img_min = img.min()
    img_max = img.max()
    return (img - img_min) / (img_max - img_min)


# ===========================================================================
# BEV Feature & Keypoint Visualization
# ===========================================================================

def _prepare_bev_images(BEV_feature):
    """Extract and process BEV feature maps into RGB images."""
    radar_img = []
    x_save = torch.mean(BEV_feature, dim=1)
    x_save = x_save.reshape(BEV_feature.shape[0], BEV_feature.shape[2], BEV_feature.shape[3])
    for shape_0 in range(x_save.shape[0]):
        img = x_save[shape_0].detach().cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min()) * 255.0
        img = img.astype(np.uint8)
        img = cv2.equalizeHist(img)
        img_rgb = np.stack((img, img, img), axis=0)
        radar_img.append(torch.from_numpy(img_rgb.astype('uint8')))
    return radar_img, x_save


def _paste_mono_cut_image(radar_np_tensor, config, dataset_type, shapeshape="None",
                          scale=1, canvas_size_multiplier=2):
    """Create a canvas and paste an image for mono-cut mode."""
    w = config['cart_pixel_width'] * canvas_size_multiplier * scale
    h = config['cart_pixel_width'] * canvas_size_multiplier * scale
    canvas = Image.new('RGB', (w, h), color='black')
    radar_np = radar_np_tensor.numpy()
    radar_np = np.transpose(radar_np, (1, 2, 0))
    img_part = Image.fromarray(radar_np.astype('uint8'))
    if scale != 1:
        img_part = img_part.resize((config['cart_pixel_width'] * scale, config['cart_pixel_width'] * scale))

    if dataset_type in ('NCLT', 'kitti', 'ZJH'):
        if shapeshape in ("None", "3232"):
            canvas.paste(img_part, (config['cart_pixel_width'] * scale // 2, 0))
    elif dataset_type == 'oxford':
        if shapeshape in ("None", "3232"):
            canvas.paste(img_part, (config['cart_pixel_width'] * scale // 2,
                                    config['cart_pixel_width'] * scale))
    return canvas


def draw_batch(feature_size, dict_matching_draw, config, IS_MONO_CUT=False,
               dataset_type='NCLT', shapeshape="None"):
    """Create an image grid of radar scans, scores, and keypoint matches for a single batch."""
    radar_img, _ = _prepare_bev_images(dict_matching_draw['x'])

    src = dict_matching_draw['src'][0].squeeze().detach().cpu().numpy()
    tgt = dict_matching_draw['tgt'][0].squeeze().detach().cpu().numpy()
    match_weights = dict_matching_draw['match_weights'][0].squeeze().detach().cpu().numpy()
    nms = config['vis_keypoint_nms']
    max_w = np.max(match_weights)

    # Build match image
    if IS_MONO_CUT:
        match_img = _paste_mono_cut_image(radar_img[0], config, dataset_type, shapeshape)
    else:
        match_img = radar_img[0].numpy()
        match_img = np.transpose(match_img, (1, 2, 0))
        match_img = Image.fromarray(match_img.astype('uint8'))

    # Draw keypoint matches
    draw = ImageDraw.Draw(match_img)
    for i in range(src.shape[0]):
        if match_weights[i] < nms * max_w:
            continue
        draw.line([(int(src[i, 0]), int(src[i, 1])),
                   (int(tgt[i, 0]), int(tgt[i, 1]))], fill=(0, 0, 255), width=1)
        draw.point((int(src[i, 0]), int(src[i, 1])), fill=(0, 255, 0))
        draw.point((int(tgt[i, 0]), int(tgt[i, 1])), fill=(255, 0, 0))
    match_img = transforms.ToTensor()(match_img)

    # Score image
    scores = dict_matching_draw['scores'][0].squeeze().detach().cpu().numpy()
    print("scores.min(), scores.max(), scores.mean():", scores.min(), scores.max(), scores.mean())
    if scores.max() != scores.min():
        scaled_scores = ((scores - scores.min()) / (scores.max() - scores.min()) * 255).astype('uint8')
    else:
        scaled_scores = ((scores - (scores.min()-1)) / (scores.max() - (scores.min()-1)) * 255).astype('uint8')
    print("scaled_scores.min(), scaled_scores.max(), scaled_scores.mean():",
          scaled_scores.min(), scaled_scores.max(), scaled_scores.mean())
    score_img = Image.fromarray(scaled_scores)
    score_img = score_img.convert('RGB')
    score_img = transforms.ToTensor()(score_img)

    if IS_MONO_CUT:
        radar_img_0 = _paste_mono_cut_image(radar_img[0], config, dataset_type, shapeshape)
        radar_img_1 = _paste_mono_cut_image(radar_img[1], config, dataset_type, shapeshape)

        # Score image on canvas
        score_img_new = Image.new('RGB', (config['cart_pixel_width'] * 2, config['cart_pixel_width'] * 2), color='black')
        score_vis = score_img * 255
        score_vis = score_vis.numpy()
        score_vis = np.transpose(score_vis, (1, 2, 0))
        score_vis = Image.fromarray(score_vis.astype('uint8'))
        if dataset_type in ('NCLT', 'kitti', 'ZJH'):
            if shapeshape == "None":
                score_img_new.paste(score_vis, (config['cart_pixel_width'] // 2, 0))
            elif shapeshape == "3232":
                radar_img_0_pil = radar_img_0  # already PIL
                # For 3232 mode, paste score onto radar_img_0
                pass
        elif dataset_type == 'oxford':
            if shapeshape in ("None", "3232"):
                score_img_new.paste(score_vis, (config['cart_pixel_width'] // 2, config['cart_pixel_width']))

        radar_img_0 = transforms.ToTensor()(radar_img_0)
        radar_img_1 = transforms.ToTensor()(radar_img_1)
        score_img_new = transforms.ToTensor()(score_img_new)
        return vutils.make_grid([radar_img_0, radar_img_1, score_img_new, match_img])

    return vutils.make_grid([radar_img[0], radar_img[1], score_img, match_img])


def draw_batch_all_for_casestudy(feature_size, dict_matching_draws, config,
                                  IS_MONO_CUT=False, dataset_type='NCLT',
                                  output_folder='./outputs/evo_casestudy/case_study/'):
    """Create and save images of radar scans, scores, and keypoint matches for each batch."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not dict_matching_draws:
        raise ValueError("dict_matching_draws is empty.")

    for idx, dict_matching_draw in enumerate(dict_matching_draws):
        BEV_feature = dict_matching_draw['x']
        x_save = torch.mean(BEV_feature, dim=1)
        x_save = x_save.reshape(BEV_feature.shape[0], BEV_feature.shape[2], BEV_feature.shape[3])

        # Colormap BEV images
        radar_img = []
        for shape_0 in range(x_save.shape[0]):
            img = x_save[shape_0].detach().cpu().numpy()
            img_ori = x_save[shape_0].detach().cpu().numpy()
            equalized_img = histogram_equalization(img)
            img = min_max_scaling(equalized_img)
            img_colormap = cm.viridis((img - img.min()) / (img.max() - img.min()))
            img_colormap[img_ori == 0] = [0, 0, 0, 1]
            img_rgb = (img_colormap[:, :, :3] * 255).astype(np.uint8)
            img_rgb = np.transpose(img_rgb, (2, 0, 1))
            radar_img.append(torch.from_numpy(img_rgb))

        # Grayscale BEV images for drawing
        radar_img_draw = []
        for shape_0 in range(x_save.shape[0]):
            img = x_save[shape_0].detach().cpu().numpy()
            img = (img - img.min()) / (img.max() - img.min()) * 255.0
            img_rgb = np.stack((img, img, img), axis=0)
            radar_img_draw.append(torch.from_numpy(img_rgb.astype('uint8')))

        src = dict_matching_draw['src'][0].squeeze().detach().cpu().numpy()
        tgt = dict_matching_draw['tgt'][0].squeeze().detach().cpu().numpy()
        match_weights = dict_matching_draw['match_weights'][0].squeeze().detach().cpu().numpy()
        nms = config['vis_keypoint_nms']
        max_w = np.max(match_weights)

        # Build match image
        if IS_MONO_CUT:
            match_img = _paste_mono_cut_image(radar_img_draw[0], config, dataset_type)
        else:
            match_img = radar_img[0].numpy()
            match_img = np.transpose(match_img, (1, 2, 0))
            match_img = Image.fromarray(match_img.astype('uint8'))

        draw = ImageDraw.Draw(match_img)
        for i in range(src.shape[0]):
            if match_weights[i] < nms * max_w:
                continue
            draw.line([(int(src[i, 0]), int(src[i, 1])),
                       (int(tgt[i, 0]), int(tgt[i, 1]))], fill=(0, 0, 255), width=1)
            draw.point((int(src[i, 0]), int(src[i, 1])), fill=(0, 255, 0))
            draw.point((int(tgt[i, 0]), int(tgt[i, 1])), fill=(255, 0, 0))
        match_img = transforms.ToTensor()(match_img)

        scores = dict_matching_draw['scores'][0].squeeze().detach().cpu().numpy()
        scaled_scores = ((scores - scores.min()) / (scores.max() - scores.min()) * 255).astype('uint8')
        score_img = Image.fromarray(scaled_scores)
        score_img = score_img.convert('RGB')
        score_img = transforms.ToTensor()(score_img)

        if IS_MONO_CUT:
            radar_img_0 = _paste_mono_cut_image(radar_img[0], config, dataset_type)
            radar_img_1 = _paste_mono_cut_image(radar_img[1], config, dataset_type)
            score_img_new = Image.new('RGB', (config['cart_pixel_width'] * 2, config['cart_pixel_width'] * 2), color='black')
            score_vis = score_img * 255
            score_vis = score_vis.numpy()
            score_vis = np.transpose(score_vis, (1, 2, 0))
            score_vis = Image.fromarray(score_vis.astype('uint8'))
            if dataset_type in ('NCLT', 'kitti', 'ZJH'):
                score_img_new.paste(score_vis, (config['cart_pixel_width'] // 2, 0))
            elif dataset_type == 'oxford':
                score_img_new.paste(score_vis, (config['cart_pixel_width'] // 2, config['cart_pixel_width']))
            radar_img_0 = transforms.ToTensor()(radar_img_0)
            radar_img_1 = transforms.ToTensor()(radar_img_1)
            score_img_new = transforms.ToTensor()(score_img_new)
            grid_img = vutils.make_grid([radar_img_0, radar_img_1, score_img_new, match_img])
        else:
            grid_img = vutils.make_grid([radar_img[0], radar_img[1], score_img, match_img])

        output_path = os.path.join(output_folder, f"batch_{idx:04d}.png")
        vutils.save_image(grid_img, output_path)


def draw_batch_all_for_casestudy2(feature_size, dict_matching_draws, config,
                                   IS_MONO_CUT=False, dataset_type='NCLT',
                                   output_folder='./outputs/evo_casestudy2/'):
    """Create and save high-resolution case study images with enlarged keypoints."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not dict_matching_draws:
        raise ValueError("dict_matching_draws is empty.")

    SCALE = 16
    POINT_RADIUS = 8

    for idx, dict_matching_draw in enumerate(dict_matching_draws):
        print(idx)
        BEV_feature = dict_matching_draw['x']
        x_save = torch.mean(BEV_feature, dim=1)
        x_save = x_save.reshape(BEV_feature.shape[0], BEV_feature.shape[2], BEV_feature.shape[3])

        # Colormap BEV images
        radar_img = []
        for shape_0 in range(x_save.shape[0]):
            img = x_save[shape_0].detach().cpu().numpy()
            img_ori = x_save[shape_0].detach().cpu().numpy()
            equalized_img = histogram_equalization(img)
            img = min_max_scaling(equalized_img)
            img_colormap = cm.viridis((img - img.min()) / (img.max() - img.min()))
            img_colormap[img_ori == 0] = [0, 0, 0, 1]
            img_rgb = (img_colormap[:, :, :3] * 255).astype(np.uint8)
            img_rgb = np.transpose(img_rgb, (2, 0, 1))
            radar_img.append(torch.from_numpy(img_rgb))

        # Grayscale + equalized BEV images for drawing
        radar_img_draw = []
        for shape_0 in range(x_save.shape[0]):
            img = x_save[shape_0].detach().cpu().numpy()
            equalized_img = histogram_equalization(img)
            img = min_max_scaling(equalized_img) * 128
            img_rgb = np.stack((img, img, img), axis=0)
            radar_img_draw.append(torch.from_numpy(img_rgb.astype('uint8')))

        src = dict_matching_draw['src'][0].squeeze().detach().cpu().numpy()
        tgt = dict_matching_draw['tgt'][0].squeeze().detach().cpu().numpy()
        match_weights = dict_matching_draw['match_weights'][0].squeeze().detach().cpu().numpy()
        nms = config['vis_keypoint_nms']
        max_w = np.max(match_weights)

        # Build match image at high resolution
        if IS_MONO_CUT:
            match_img = _paste_mono_cut_image(radar_img_draw[1], config, dataset_type, scale=SCALE)
            radar_img_save0 = _paste_mono_cut_image(radar_img[0], config, dataset_type, scale=SCALE)
            radar_img_save1 = _paste_mono_cut_image(radar_img[1], config, dataset_type, scale=SCALE)
        else:
            match_img = radar_img[0].numpy()
            match_img = np.transpose(match_img, (1, 2, 0))
            match_img = Image.fromarray(match_img.astype('uint8'))

        # Draw scaled keypoint matches
        draw = ImageDraw.Draw(match_img)
        for i in range(src.shape[0]):
            if match_weights[i] < nms * max_w:
                continue
            sx, sy = int(src[i, 0] * SCALE), int(src[i, 1] * SCALE)
            tx, ty = int(tgt[i, 0] * SCALE), int(tgt[i, 1] * SCALE)
            draw.line([(sx, sy), (tx, ty)], fill=(0, 0, 255), width=SCALE // 2)
            draw.ellipse([(sx - POINT_RADIUS, sy - POINT_RADIUS),
                          (sx + POINT_RADIUS, sy + POINT_RADIUS)], fill=(0, 255, 0))
            draw.ellipse([(tx - POINT_RADIUS, ty - POINT_RADIUS),
                          (tx + POINT_RADIUS, ty + POINT_RADIUS)], fill=(255, 0, 0))
        match_img = transforms.ToTensor()(match_img)
        if IS_MONO_CUT:
            radar_img_save0 = transforms.ToTensor()(radar_img_save0)
            radar_img_save1 = transforms.ToTensor()(radar_img_save1)

        scores = dict_matching_draw['scores'][0].squeeze().detach().cpu().numpy()
        scaled_scores = ((scores - scores.min()) / (scores.max() - scores.min()) * 255).astype('uint8')
        score_img = Image.fromarray(scaled_scores)
        score_img = score_img.convert('RGB')

        if IS_MONO_CUT:
            # Build small grid images for thumbnails
            radar_img_0 = _paste_mono_cut_image(radar_img[0], config, dataset_type)
            radar_img_1 = _paste_mono_cut_image(radar_img[1], config, dataset_type)
            score_img_new = Image.new('RGB', (config['cart_pixel_width'] * 2, config['cart_pixel_width'] * 2), color='black')
            score_img_t = transforms.ToTensor()(score_img)
            score_vis = score_img_t * 255
            score_vis = score_vis.numpy()
            score_vis = np.transpose(score_vis, (1, 2, 0))
            score_vis = Image.fromarray(score_vis.astype('uint8'))
            if dataset_type in ('NCLT', 'kitti', 'ZJH'):
                score_img_new.paste(score_vis, (config['cart_pixel_width'] // 2, 0))
            elif dataset_type == 'oxford':
                score_img_new.paste(score_vis, (config['cart_pixel_width'] // 2, config['cart_pixel_width']))
            radar_img_0 = transforms.ToTensor()(radar_img_0)
            radar_img_1 = transforms.ToTensor()(radar_img_1)
            score_img_new = transforms.ToTensor()(score_img_new)

        # Resize score image for high-res output
        score_img = score_img.resize((config['cart_pixel_width'] * SCALE, config['cart_pixel_width'] * SCALE))
        score_img = transforms.ToTensor()(score_img)

        # Save individual images
        output_path = os.path.join(output_folder, f"match_{idx:04d}.png")
        output_path_save0 = os.path.join(output_folder, f"save0_{idx:04d}.png")
        output_path_save1 = os.path.join(output_folder, f"save1_{idx:04d}.png")
        output_path_score = os.path.join(output_folder, f"score_{idx:04d}.png")
        vutils.save_image(match_img, output_path)
        if IS_MONO_CUT:
            vutils.save_image(radar_img_save0, output_path_save0)
            vutils.save_image(radar_img_save1, output_path_save1)
        vutils.save_image(score_img, output_path_score)


# ===========================================================================
# Match Visualization with Transforms
# ===========================================================================

def draw_matches(batch, out, config, solver):
    """Overlay raw and transformed keypoint matches on radar images."""
    azimuth_step = (2 * np.pi) / 400
    cart_pixel_width = config['cart_pixel_width']
    cart_resolution = config['cart_resolution']
    if (cart_pixel_width % 2) == 0:
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
    else:
        cart_min_range = cart_pixel_width // 2 * cart_resolution
    T_met_pix = np.array([[0, -cart_resolution, 0, cart_min_range],
                          [cart_resolution, 0, 0, -cart_min_range],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
    T_pix_met = np.linalg.inv(T_met_pix)

    keypoint_ints = out['keypoint_ints']
    ids = torch.nonzero(keypoint_ints[0, 0] > 0, as_tuple=False).squeeze(1)
    src = out['src_rc'][0, ids].squeeze().detach().cpu().numpy()
    tgt = out['tgt_rc'][0, ids].squeeze().detach().cpu().numpy()
    radar = batch['data'][0].squeeze().numpy()
    _, axs = plt.subplots(1, 3, tight_layout=True)

    # Raw locations
    axs[0].imshow(radar, cmap='gray', extent=(0, 640, 640, 0), interpolation='none')
    axs[0].set_axis_off()
    axs[0].set_title('raw')
    for i in range(src.shape[0]):
        axs[0].plot([src[i, 0], tgt[i, 0]], [src[i, 1], tgt[i, 1]], c='w', linewidth=1, zorder=2)
        axs[0].scatter(src[i, 0], src[i, 1], c='limegreen', s=2, zorder=3)
        axs[0].scatter(tgt[i, 0], tgt[i, 1], c='r', s=2, zorder=4)

    # Rigid transform aligned
    src = out['src'][0, ids].squeeze().detach().cpu().numpy()
    tgt = out['tgt'][0, ids].squeeze().detach().cpu().numpy()
    axs[1].imshow(radar, cmap='gray', extent=(0, 640, 640, 0), interpolation='none')
    axs[1].set_axis_off()
    axs[1].set_title('rigid')
    T_tgt_src = get_T_ba(out, a=0, b=1)
    error = np.zeros((src.shape[0], 2))
    for i in range(src.shape[0]):
        x1 = np.array([src[i, 0], src[i, 1], 0, 1]).reshape(4, 1)
        x2 = np.array([tgt[i, 0], tgt[i, 1], 0, 1]).reshape(4, 1)
        x1 = T_tgt_src @ x1
        e = x1 - x2
        error[i, 1] = np.sqrt(e.T @ e)
        error[i, 0] = int(wrapto2pi(np.arctan2(x2[1, 0], x2[0, 0])) // azimuth_step)
        x1 = T_pix_met @ x1
        x2 = T_pix_met @ x2
        axs[1].plot([x1[0, 0], x2[0, 0]], [x1[1, 0], x2[1, 0]], c='w', linewidth=1, zorder=2)
        axs[1].scatter(x1[0, 0], x1[1, 0], c='limegreen', s=2, zorder=3)
        axs[1].scatter(x2[0, 0], x2[1, 0], c='r', s=2, zorder=4)

    # Interpolated poses
    t1 = batch['timestamps'][0].numpy().squeeze()
    t2 = batch['timestamps'][1].numpy().squeeze()
    times1 = getApproxTimeStamps([src], [t1])[0]
    times2 = getApproxTimeStamps([tgt], [t2])[0]
    t_refs = batch['t_ref'].numpy()

    T_1a = np.identity(4, dtype=np.float32)
    T_1b = np.identity(4, dtype=np.float32)
    axs[2].imshow(radar, cmap='gray', extent=(0, 640, 640, 0), interpolation='none')
    axs[2].set_axis_off()
    axs[2].set_title('interp')
    error2 = np.zeros((src.shape[0], 2))
    for i in range(src.shape[0]):
        solver.getPoseBetweenTimes(T_1a, times1[i], t_refs[1, 0, 0])
        solver.getPoseBetweenTimes(T_1b, times2[i], t_refs[1, 0, 0])
        x1 = np.array([src[i, 0], src[i, 1], 0, 1]).reshape(4, 1)
        x2 = np.array([tgt[i, 0], tgt[i, 1], 0, 1]).reshape(4, 1)
        x1 = T_1a @ x1
        x2 = T_1b @ x2
        e = x1 - x2
        error2[i, 1] = np.sqrt(e.T @ e)
        error2[i, 0] = int(wrapto2pi(np.arctan2(x2[1, 0], x2[0, 0])) // azimuth_step)
        x1 = T_pix_met @ x1
        x2 = T_pix_met @ x2
        axs[2].plot([x1[0, 0], x2[0, 0]], [x1[1, 0], x2[1, 0]], c='w', linewidth=1, zorder=2)
        axs[2].scatter(x1[0, 0], x1[1, 0], c='limegreen', s=2, zorder=3)
        axs[2].scatter(x2[0, 0], x2[1, 0], c='r', s=2, zorder=4)

    plt.savefig('matches.pdf', bbox_inches='tight', pad_inches=0.0)
    plt.figure()

    idx = np.argsort(error[:, 0])
    error = error[idx, :]
    idx = np.argsort(error2[:, 0])
    error2 = error2[idx, :]
    plt.plot(error[:, 0], error[:, 1], color='b', label='raw error', linewidth=1)
    plt.plot(error2[:, 0], error2[:, 1], color='r', label='interp error', linewidth=1)
    plt.title('raw error')
    plt.legend()
    plt.savefig('matches2.pdf', bbox_inches='tight', pad_inches=0.0)


def draw_batch_steam(batch, out, config):
    """Create visualization of radar scans with steam-based keypoint matches."""
    radar = batch['data'][0].squeeze().numpy()
    radar_tgt = batch['data'][-1].squeeze().numpy()
    plt.imshow(np.concatenate((radar, radar_tgt), axis=1), cmap='gray')
    plt.title('radar src-tgt pair')
    radar_img = convert_plt_to_tensor()

    # Keypoint matches (horizontal layout)
    src = out['src_rc'][-1].squeeze().detach().cpu().numpy()
    tgt = out['tgt_rc'][-1].squeeze().detach().cpu().numpy()
    keypoint_ints = out['keypoint_ints']
    ids = torch.nonzero(keypoint_ints[-1, 0] > 0, as_tuple=False).squeeze(1)
    ids_cpu = ids.cpu()

    plt.imshow(np.concatenate((radar, radar_tgt), axis=1), cmap='gray')
    delta = radar.shape[1]
    for i in range(src.shape[0]):
        if i in ids_cpu:
            plt.plot([src[i, 0], tgt[i, 0] + delta], [src[i, 1], tgt[i, 1]], c='y', linewidth=0.5, zorder=2)
            plt.scatter(src[i, 0], src[i, 1], c='g', s=5, zorder=3)
            plt.scatter(tgt[i, 0] + delta, tgt[i, 1], c='g', s=5, zorder=4)
    plt.title('matches')
    match_img = convert_plt_to_tensor()

    # Keypoint matches (vertical layout)
    plt.imshow(np.concatenate((radar, radar_tgt), axis=0), cmap='gray')
    delta = radar.shape[1]
    for i in range(src.shape[0]):
        if i in ids_cpu:
            plt.plot([src[i, 0], tgt[i, 0]], [src[i, 1], tgt[i, 1] + delta], c='y', linewidth=0.5, zorder=2)
            plt.scatter(src[i, 0], src[i, 1], c='g', s=5, zorder=3)
            plt.scatter(tgt[i, 0], tgt[i, 1] + delta, c='g', s=5, zorder=4)
    plt.title('matches')
    match_img2 = convert_plt_to_tensor()

    # Scores
    scores = out['scores'][-1]
    if scores.size(0) == 3:
        scores = scores[1] + scores[2]
    scores = scores.squeeze().detach().cpu().numpy()
    plt.imshow(scores, cmap='inferno')
    plt.colorbar()
    plt.title('log det weight (weight score vis)')
    score_img = convert_plt_to_tensor()

    # Detector scores
    detector_scores = out['detector_scores'][-1].squeeze().detach().cpu().numpy()
    plt.imshow(detector_scores, cmap='inferno')
    plt.colorbar()
    plt.title('detector score')
    dscore_img = convert_plt_to_tensor()

    # Point-to-point error
    src_p = out['src'][-1].squeeze().T
    tgt_p = out['tgt'][-1].squeeze().T
    R_tgt_src = out['R'][0, -1, :2, :2]
    t_st_in_t = out['t'][0, -1, :2, :]
    error = tgt_p - (R_tgt_src @ src_p + t_st_in_t)
    error2_sqrt = torch.sqrt(torch.sum(error * error, dim=0).squeeze())
    error2_sqrt = error2_sqrt[ids_cpu].detach().cpu().numpy()

    plt.imshow(radar, cmap='gray')
    plt.scatter(src[ids_cpu, 0], src[ids_cpu, 1], c=error2_sqrt, s=5, zorder=2, cmap='rainbow')
    plt.clim(0.0, 1)
    plt.colorbar()
    plt.title('P2P error')
    p2p_img = convert_plt_to_tensor()

    return (vutils.make_grid([dscore_img, score_img, radar_img]),
            vutils.make_grid([match_img, match_img2]),
            vutils.make_grid([p2p_img]))


# ===========================================================================
# Trajectory Plotting
# ===========================================================================

def plot_sequences(T_gt, T_pred, seq_lens, returnTensor=True, T_icra=None,
                   savePDF=False, fnames=None, flip=True):
    """Create top-down plots of predicted odometry vs. ground truth."""
    seq_indices = []
    idx = 0
    for s in seq_lens:
        seq_indices.append(list(range(idx, idx + s - 1)))
        idx += (s - 1)

    T_flip = np.identity(4)
    T_flip[1, 1] = -1
    T_flip[2, 2] = -1
    imgs = []
    for seq_i, indices in enumerate(seq_indices):
        T_gt_ = np.identity(4)
        T_pred_ = np.identity(4)
        if flip:
            T_gt_ = np.matmul(T_flip, T_gt_)
            T_pred_ = np.matmul(T_flip, T_pred_)
        x_gt, y_gt = [], []
        x_pred, y_pred = [], []
        for i in indices:
            T_gt_ = np.matmul(T_gt[i], T_gt_)
            T_pred_ = np.matmul(T_pred[i], T_pred_)
            enforce_orthog(T_gt_)
            enforce_orthog(T_pred_)
            T_gt_temp = get_inverse_tf(T_gt_)
            T_pred_temp = get_inverse_tf(T_pred_)
            x_gt.append(T_gt_temp[0, 3])
            y_gt.append(T_gt_temp[1, 3])
            x_pred.append(T_pred_temp[0, 3])
            y_pred.append(T_pred_temp[1, 3])
            print(T_gt_temp, T_pred_temp)

        img = draw_plot(x_gt, y_gt, x_pred, y_pred)
        if returnTensor:
            imgs.append(transforms.ToTensor()(img))
        else:
            imgs.append(img)

    return imgs


def plot_2d_trajectory(T_gt, T_pred, plane='xy'):
    """Plot 2D trajectory comparison on a specified plane (xy/yz/xz)."""
    fig, ax = plt.subplots()
    x_gt, y_gt = [], []
    x_pred, y_pred = [], []
    T_gt_accum = np.identity(4)
    T_pred_accum = np.identity(4)

    plane_indices = {'xy': (0, 1), 'yz': (1, 2), 'xz': (0, 2)}
    ix, iy = plane_indices.get(plane, (0, 1))

    for T_g, T_p in zip(T_gt, T_pred):
        T_gt_accum = np.matmul(T_gt_accum, T_g)
        T_pred_accum = np.matmul(T_pred_accum, T_p)
        x_gt.append(T_gt_accum[ix, 3])
        y_gt.append(T_gt_accum[iy, 3])
        x_pred.append(T_pred_accum[ix, 3])
        y_pred.append(T_pred_accum[iy, 3])

    ax.plot(x_gt, y_gt, label='Ground Truth')
    ax.plot(x_pred, y_pred, label='Prediction')
    ax.set_xlabel(plane[0].upper())
    ax.set_ylabel(plane[1].upper())
    ax.legend()
    return fig


def draw_plot(x_gt, y_gt, x_pred, y_pred):
    """Draw a top-down trajectory plot using PIL (no matplotlib dependency)."""
    if (max(x_gt + x_pred) == float('inf') or min(x_gt + x_pred) == float('-inf') or
            max(y_gt + y_pred) == float('inf') or min(y_gt + y_pred) == float('-inf')):
        return Image.new('RGB', (1000, 1000), color='white')

    center_x = (max(x_gt + x_pred) + min(x_gt + x_pred)) / 2
    center_y = (max(y_gt + y_pred) + min(y_gt + y_pred)) / 2
    img_width = 1000
    img_height = 1000

    img = Image.new('RGB', (img_width, img_height), color='white')
    draw = ImageDraw.Draw(img)

    scale_factor = min((img_width - 100) / (max(x_gt + x_pred) - min(x_gt + x_pred)),
                       (img_height - 100) / (max(y_gt + y_pred) - min(y_gt + y_pred)))

    # Draw axes
    draw.line([(0, 950), (img_width, 950)], fill='black', width=2)
    draw.line([(50, 0), (50, img_height)], fill='black', width=2)

    axis_bounds_width = img_width / scale_factor
    axis_bounds_height = img_height / scale_factor

    print(f"Plot bounds: x[{min(x_gt + x_pred):.2f}, {max(x_gt + x_pred):.2f}], "
          f"y[{min(y_gt + y_pred):.2f}, {max(y_gt + y_pred):.2f}], scale:{scale_factor:.2f}")

    # Draw tick marks
    step_x = int(axis_bounds_width / 22) if int(axis_bounds_width / 22) != 0 else max(1, int(axis_bounds_width / 5))
    step_y = int(axis_bounds_height / 22) if int(axis_bounds_height / 22) != 0 else max(1, int(axis_bounds_height / 5))

    for i in range(-int(axis_bounds_width / 2), int(axis_bounds_width / 2), step_x):
        tick_x = int((i * scale_factor) + img_width / 2)
        if 0 <= tick_x <= img_width:
            draw.line([(tick_x, 945), (tick_x, 955)], fill='black', width=2)
            draw.text((tick_x - 5, 960), str(i), fill='black', font=None)

    for i in range(-int(axis_bounds_height / 2), int(axis_bounds_height / 2), step_y):
        tick_y = int((-i * scale_factor) + img_height / 2)
        if 0 <= tick_y <= img_height:
            draw.line([(45, tick_y), (55, tick_y)], fill='black', width=2)
            draw.text((25, tick_y - 5), str(i), fill='black', font=None)

    # Draw ground truth trajectory
    for i in range(len(x_gt) - 1):
        draw.line([(x_gt[i] - center_x) * scale_factor + img_width / 2,
                   (y_gt[i] - center_y) * scale_factor + img_height / 2,
                   (x_gt[i + 1] - center_x) * scale_factor + img_width / 2,
                   (y_gt[i + 1] - center_y) * scale_factor + img_height / 2],
                  fill='black', width=3)
    draw.text((100, 50), 'black-gt', fill='black', font=None)

    # Draw predicted trajectory
    for i in range(len(x_pred) - 1):
        draw.line([(x_pred[i] - center_x) * scale_factor + img_width / 2,
                   (y_pred[i] - center_y) * scale_factor + img_height / 2,
                   (x_pred[i + 1] - center_x) * scale_factor + img_width / 2,
                   (y_pred[i + 1] - center_y) * scale_factor + img_height / 2],
                  fill='blue', width=3)
    draw.text((100, 65), 'blue-pred', fill='blue', font=None)

    return img