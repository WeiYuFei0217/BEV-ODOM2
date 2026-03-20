"""
Oxford RobotCar Velodyne HDL-32E point cloud loading and parsing utilities.

Supports loading Velodyne point clouds from binary (.bin) and raw (.png) files,
and converting raw data to XYZI point clouds.
"""

import os
from typing import AnyStr

import cv2
import numpy as np

# HDL-32E sensor hard-coded parameters
hdl32e_range_resolution = 0.002  # m / pixel
hdl32e_minimum_range = 1.0
hdl32e_elevations = np.array([
    -0.1862, -0.1628, -0.1396, -0.1164, -0.0930,
    -0.0698, -0.0466, -0.0232, 0., 0.0232, 0.0466, 0.0698,
    0.0930, 0.1164, 0.1396, 0.1628, 0.1862, 0.2094, 0.2327,
    0.2560, 0.2793, 0.3025, 0.3259, 0.3491, 0.3723, 0.3957,
    0.4189, 0.4421, 0.4655, 0.4887, 0.5119, 0.5353
])[:, np.newaxis]
hdl32e_base_to_fire_height = 0.090805
hdl32e_cos_elevations = np.cos(hdl32e_elevations)
hdl32e_sin_elevations = np.sin(hdl32e_elevations)


def load_velodyne_binary(velodyne_bin_path: AnyStr):
    """Load a Velodyne point cloud from a binary file.

    Args:
        velodyne_bin_path: Path to the binary point cloud file (.bin).

    Returns:
        ptcld: XYZI point cloud data with shape (4, N).

    Note:
        The pre-computed points are NOT motion compensated.
    """
    ext = os.path.splitext(velodyne_bin_path)[1]
    if ext != ".bin":
        raise RuntimeError(
            f"Velodyne binary pointcloud file should have `.bin` extension "
            f"but had: {ext}"
        )
    if not os.path.isfile(velodyne_bin_path):
        raise FileNotFoundError(
            f"Could not find velodyne bin example: {velodyne_bin_path}"
        )
    data = np.fromfile(velodyne_bin_path, dtype=np.float32)
    ptcld = data.reshape((-1, 4)).transpose()
    return ptcld


def load_velodyne_raw(velodyne_raw_path: AnyStr):
    """Load raw Velodyne scan data from a PNG file.

    Args:
        velodyne_raw_path: Path to the raw scan file (.png).

    Returns:
        ranges: Range measurements in metres (0 = invalid), shape (32, N).
        intensities: Intensity values (0 = invalid), shape (32, N).
        angles: Azimuth angles in radians, shape (1, N).
        approximate_timestamps: Linearly interpolated timestamps, shape (1, N).

    Note:
        Reference: https://velodynelidar.com/lidar/products/manual/
        63-9113%20HDL-32E%20manual_Rev%20E_NOV2012.pdf
    """
    ext = os.path.splitext(velodyne_raw_path)[1]
    if ext != ".png":
        raise RuntimeError(
            f"Velodyne raw file should have `.png` extension but had: {ext}"
        )
    if not os.path.isfile(velodyne_raw_path):
        raise FileNotFoundError(
            f"Could not find velodyne raw example: {velodyne_raw_path}"
        )

    example = cv2.imread(velodyne_raw_path, cv2.IMREAD_GRAYSCALE)
    intensities, ranges_raw, angles_raw, timestamps_raw = np.array_split(
        example, [32, 96, 98], 0
    )

    ranges = np.ascontiguousarray(ranges_raw.transpose()).view(np.uint16).transpose()
    ranges = ranges * hdl32e_range_resolution

    angles = np.ascontiguousarray(angles_raw.transpose()).view(np.uint16).transpose()
    angles = angles * (2. * np.pi) / 36000

    approximate_timestamps = (
        np.ascontiguousarray(timestamps_raw.transpose()).view(np.int64).transpose()
    )

    return ranges, intensities, angles, approximate_timestamps


def velodyne_raw_to_pointcloud(ranges: np.ndarray, intensities: np.ndarray,
                               angles: np.ndarray):
    """Convert raw Velodyne data to an XYZI point cloud.

    Args:
        ranges: Raw range measurements.
        intensities: Raw intensity values.
        angles: Raw azimuth angles.

    Returns:
        pointcloud: XYZI point cloud with shape (4, N).

    Note:
        Does NOT perform motion compensation. Using load_velodyne_binary is
        approximately 2x faster at the cost of 8x storage space.
    """
    valid = ranges > hdl32e_minimum_range
    z = hdl32e_sin_elevations * ranges - hdl32e_base_to_fire_height
    xy = hdl32e_cos_elevations * ranges
    x = np.sin(angles) * xy
    y = -np.cos(angles) * xy

    xf = x[valid].reshape(-1)
    yf = y[valid].reshape(-1)
    zf = z[valid].reshape(-1)
    intensityf = intensities[valid].reshape(-1).astype(np.float32)

    ptcld = np.stack((xf, yf, zf, intensityf), 0)
    return ptcld
