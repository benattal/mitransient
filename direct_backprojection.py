#!/usr/bin/env python3
"""
Direct Backprojection for NLOS Reconstruction

This script implements simple direct backprojection for reconstructing hidden
scenes from confocal transient NLOS data.

Algorithm:
1. Load transient data from render_transient.py output
2. Detect direct peaks to recover relay wall geometry
3. For each voxel, sum indirect transient intensity / r² from all relay points

Unlike phasor field backprojection, this method:
- Does not use virtual wavelength or frequency domain processing
- Directly deposits transient intensity into the volume
- Uses 1/r² distance falloff (radiometric falloff)

Usage Examples:
--------------

Basic usage (simple scenes with auto volume bounds):
    python direct_backprojection.py results/wall_box_transient.npy

Complex scenes (e.g., ourbox) with explicit volume bounds:
    python direct_backprojection.py results/ourbox_confocal_transient_both.npy \
        --scene-file scenes/ourbox_confocal.xml \
        --voxel-resolution 128 \
        --volume-from-camera \
        --volume-min -1 -1 -1 \
        --volume-max 1 1 1

Using separate direct/indirect captures (for better geometry recovery):
    python direct_backprojection.py results/scene_transient.npy \
        --direct-file results/scene_direct_only.npy \
        --indirect-file results/scene_indirect_only.npy \
        --scene-file scenes/my_scene.xml

Key Parameters:
- --voxel-resolution: Higher values (128, 256) give finer detail but slower
- --volume-min/max: Manually specify reconstruction bounds (x, y, z) in meters
- --volume-from-camera: Auto-compute bounds from camera frustum (good starting point)
- --no-falloff: Disable 1/r² distance falloff

Output:
- *_orthographic.png: Combined front/side/top views (grayscale intensity)
- *_orthographic_color.png: Combined views with RGB color
- *_orthographic_overlay.png: Reconstruction overlaid with relay wall point cloud
- *_volume.npy: 3D reconstruction volume (Vx, Vy, Vz)
- *_volume_rgb.npy: 3D RGB reconstruction volume (Vx, Vy, Vz, 3)
"""

import os
import argparse
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

# Set Mitsuba variant before importing
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import mitransient as mitr
import drjit as dr


def parse_args():
    parser = argparse.ArgumentParser(
        description='Direct backprojection for NLOS reconstruction'
    )
    parser.add_argument('transient_file', type=str,
                        help='Path to *_transient.npy file (used for indirect if --direct-file not specified)')
    parser.add_argument('--direct-file', type=str, default=None,
                        help='Separate file for direct peak detection (geometry). If not specified, uses transient_file.')
    parser.add_argument('--indirect-file', type=str, default=None,
                        help='Separate file for indirect transient (reconstruction). If not specified, uses transient_file.')
    parser.add_argument('--scene-file', type=str, default='scenes/wall_box_behind.xml',
                        help='Path to XML scene file to extract parameters (default: scenes/wall_box_behind.xml)')
    parser.add_argument('--start-opl', type=float, default=None,
                        help='Start OPL from scene (meters). Overrides scene file value.')
    parser.add_argument('--bin-width', type=float, default=None,
                        help='Bin width OPL from scene (meters). Overrides scene file value.')
    parser.add_argument('--voxel-resolution', type=int, default=64,
                        help='Voxels per dimension (default: 64)')
    parser.add_argument('--volume-min', type=float, nargs=3, default=None,
                        help='Volume min bounds (x, y, z). Default: auto-computed from relay wall.')
    parser.add_argument('--volume-max', type=float, nargs=3, default=None,
                        help='Volume max bounds (x, y, z). Default: auto-computed from relay wall.')
    parser.add_argument('--volume-from-camera', action='store_true',
                        help='Compute volume bounds by backprojecting camera frustum corners.')
    parser.add_argument('--volume-depth-min', type=float, default=None,
                        help='Near depth for camera frustum backprojection (meters).')
    parser.add_argument('--volume-depth-max', type=float, default=None,
                        help='Far depth for camera frustum backprojection (meters).')
    parser.add_argument('--volume-margin', type=float, default=0.1,
                        help='Margin to add around camera-derived volume bounds (fraction, default: 0.1).')
    parser.add_argument('--no-falloff', action='store_true',
                        help='Disable 1/r² distance falloff in backprojection')
    parser.add_argument('--bin-threshold', type=float, default=0.0,
                        help='Only splat to voxels whose corresponding time bin has indirect signal above this fraction of the relay point\'s max (0.0-1.0, default: 0.0 = no threshold)')
    parser.add_argument('--min-relay-distance', type=float, default=0.0,
                        help='Minimum distance from relay point to voxel for contribution (meters, default: 0.0)')
    parser.add_argument('--no-hemisphere-filter', action='store_true',
                        help='Disable hemisphere filtering based on relay wall normals. By default, only voxels in the positive normal hemisphere (away from camera) contribute.')
    parser.add_argument('--output-dir', type=str, default='vis/',
                        help='Output directory (default: vis/)')
    parser.add_argument('--output-name', type=str, default=None,
                        help='Output name prefix (default: derived from input)')
    parser.add_argument('--debug-pixel', type=int, nargs=2, default=None, metavar=('H', 'W'),
                        help='Pixel (row, col) to use for debug visualizations (default: center)')
    parser.add_argument('--normalize-slices', type=str, default=None, choices=['x', 'y', 'z', 'none'],
                        help='Normalize intensity per slice along axis (x, y, z, or none). Default: none')
    parser.add_argument('--vis-transform', type=str, default='none',
                        choices=['none', 'log', 'sqrt', 'cbrt', 'exp', 'sigmoid'],
                        help='Non-linear transform for volume visualization: none (linear), log (log10), sqrt, cbrt (cube root), exp (exponential), sigmoid. Default: none')
    parser.add_argument('--vis-scale', type=float, default=1.0,
                        help='Scale factor applied before non-linearity: output = transform(scale * input + bias). Default: 1.0')
    parser.add_argument('--vis-bias', type=float, default=0.0,
                        help='Bias (offset) applied before non-linearity: output = transform(scale * input + bias). Default: 0.0')
    parser.add_argument('--transient-log', action='store_true',
                        help='Use log scale for transient visualization plots. Default: disabled')
    return parser.parse_args()


def load_scene_params(scene_path):
    """
    Load scene from XML and extract relevant parameters.

    Args:
        scene_path: Path to the XML scene file

    Returns:
        dict with keys:
            - start_opl: Start optical path length (meters)
            - bin_width: OPL per bin (meters)
            - width, height: Film dimensions
            - sensor: Mitsuba sensor object
            - camera_origin: Camera position in world space (3,)
            - camera_to_world: 4x4 camera-to-world transform matrix
    """
    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"Scene file not found: {scene_path}")

    print(f"  Loading scene from: {scene_path}")
    scene = mi.load_file(scene_path)

    sensor = scene.sensors()[0]
    film = sensor.film()

    # Extract film parameters
    start_opl = float(film.start_opl)
    bin_width = float(film.bin_width_opl)
    width, height = film.size()

    # Extract camera transform
    camera_to_world = sensor.world_transform()
    # Get camera origin (translation component) - flatten to 1D
    camera_origin = np.array(camera_to_world.translation(), dtype=np.float32).flatten()

    print(f"  Film size: {width}x{height}")
    print(f"  Start OPL: {start_opl} m")
    print(f"  Bin width: {bin_width} m")
    print(f"  Camera origin: {camera_origin}")

    return {
        'start_opl': start_opl,
        'bin_width': bin_width,
        'width': width,
        'height': height,
        'sensor': sensor,
        'camera_origin': camera_origin,
        'camera_to_world': camera_to_world,
        'scene': scene,
    }


def compute_relay_positions_from_camera(relay_depths, sensor, camera_origin):
    """
    Compute 3D positions of relay wall points by unprojecting camera rays.

    Uses the camera sensor to generate rays for each pixel, then computes
    the intersection point using the depth from direct peak detection.

    For confocal NLOS, the direct peak bin indicates round-trip distance:
        total_opl = start_opl + bin_index * bin_width
        depth = total_opl / 2  (assuming camera and laser are co-located)

    Args:
        relay_depths: (H, W) depth values from direct peak detection
        sensor: Mitsuba sensor object
        camera_origin: (3,) camera position in world space

    Returns:
        relay_pos: (H, W, 3) array of 3D world-space positions
    """
    H, W = relay_depths.shape

    # Get film size for normalized coordinates
    film = sensor.film()
    film_w, film_h = film.size()

    # Create grid of normalized pixel coordinates (vectorized)
    j_coords = np.arange(W, dtype=np.float32)
    i_coords = np.arange(H, dtype=np.float32)
    jj, ii = np.meshgrid(j_coords, i_coords)

    u_coords = (jj + 0.5) / film_w
    v_coords = (ii + 0.5) / film_h

    # Flatten for batch processing
    u_flat = u_coords.flatten()
    v_flat = v_coords.flatten()

    # Sample rays in batch using DrJit
    sample_pos = mi.Point2f(mi.Float(u_flat), mi.Float(v_flat))
    rays, _ = sensor.sample_ray(
        time=mi.Float(0.0),
        sample1=mi.Float(0.0),
        sample2=sample_pos,
        sample3=mi.Point2f(mi.Float(0.5), mi.Float(0.5))
    )

    # Extract ray origins and directions
    ray_o_x = np.array(rays.o.x, dtype=np.float32)
    ray_o_y = np.array(rays.o.y, dtype=np.float32)
    ray_o_z = np.array(rays.o.z, dtype=np.float32)
    ray_d_x = np.array(rays.d.x, dtype=np.float32)
    ray_d_y = np.array(rays.d.y, dtype=np.float32)
    ray_d_z = np.array(rays.d.z, dtype=np.float32)

    # Normalize ray directions to ensure unit length
    ray_d_norm = np.sqrt(ray_d_x**2 + ray_d_y**2 + ray_d_z**2)
    ray_d_x = ray_d_x / ray_d_norm
    ray_d_y = ray_d_y / ray_d_norm
    ray_d_z = ray_d_z / ray_d_norm

    # Flatten depths for vectorized computation
    depths_flat = relay_depths.flatten()

    # Compute world positions: origin + depth * direction
    world_x = ray_o_x + depths_flat * ray_d_x
    world_y = ray_o_y + depths_flat * ray_d_y
    world_z = ray_o_z + depths_flat * ray_d_z

    # Reshape back to (H, W, 3)
    relay_pos = np.stack([
        world_x.reshape(H, W),
        world_y.reshape(H, W),
        world_z.reshape(H, W)
    ], axis=-1)

    return relay_pos


def compute_relay_normals(relay_pos, camera_origin):
    """
    Compute surface normals at each relay wall point.

    Uses finite differences on neighboring positions to compute tangent vectors,
    then cross product to get normals. Normals are oriented to point away from camera.

    Args:
        relay_pos: (H, W, 3) relay wall 3D positions
        camera_origin: (3,) camera position

    Returns:
        normals: (H, W, 3) unit normal vectors at each relay point
    """
    H, W, _ = relay_pos.shape

    # Compute tangent vectors using finite differences
    tangent_u = np.zeros_like(relay_pos)
    tangent_u[:, 1:-1, :] = relay_pos[:, 2:, :] - relay_pos[:, :-2, :]
    tangent_u[:, 0, :] = relay_pos[:, 1, :] - relay_pos[:, 0, :]
    tangent_u[:, -1, :] = relay_pos[:, -1, :] - relay_pos[:, -2, :]

    tangent_v = np.zeros_like(relay_pos)
    tangent_v[1:-1, :, :] = relay_pos[2:, :, :] - relay_pos[:-2, :, :]
    tangent_v[0, :, :] = relay_pos[1, :, :] - relay_pos[0, :, :]
    tangent_v[-1, :, :] = relay_pos[-1, :, :] - relay_pos[-2, :, :]

    # Cross product to get normal
    normals = np.cross(tangent_u, tangent_v)

    # Normalize
    norm_length = np.linalg.norm(normals, axis=-1, keepdims=True)
    norm_length = np.maximum(norm_length, 1e-8)
    normals = normals / norm_length

    # Orient normals to point away from camera (into the hidden scene)
    relay_to_camera = camera_origin - relay_pos
    dot = np.sum(normals * relay_to_camera, axis=-1)
    flip_mask = dot > 0
    normals[flip_mask] = -normals[flip_mask]

    return -normals.astype(np.float32)


def load_transient(path, return_rgb=False):
    """
    Load transient data and convert to luminance (and optionally return RGB).

    Args:
        path: Path to .npy file with shape (H, W, T, C)
        return_rgb: If True, also return RGB channels

    Returns:
        luminance: (H, W, T) array
        rgb: (H, W, T, 3) array (only if return_rgb=True)
    """
    data = np.load(path)
    print(f"  Loaded shape: {data.shape}")

    # Convert to luminance for reconstruction
    if data.shape[-1] >= 3:
        luminance = (0.2126 * data[..., 0] +
                     0.7152 * data[..., 1] +
                     0.0722 * data[..., 2])
        rgb = data[..., :3].astype(np.float32)
    else:
        luminance = data[..., 0]
        rgb = np.stack([data[..., 0]] * 3, axis=-1).astype(np.float32)

    if return_rgb:
        return luminance.astype(np.float32), rgb
    return luminance.astype(np.float32)


def detect_direct_peaks(transient, start_opl, bin_width, threshold_percentile=95, peak_width=2,
                        min_signal_threshold=None):
    """
    Detect direct reflection peaks to recover relay wall depth (vectorized).

    For each pixel, find the first significant peak - this is the direct
    return from the relay wall. Uses argmax on thresholded data for speed.

    Args:
        transient: (H, W, T) transient data
        start_opl: Start optical path length (meters)
        bin_width: OPL per bin (meters)
        threshold_percentile: Percentile for peak detection threshold
        peak_width: Half-width of direct peak mask (bins)
        min_signal_threshold: Minimum signal to consider a pixel valid.

    Returns:
        relay_depths: (H, W) array of wall depths (meters)
        direct_mask: (H, W, T) boolean mask of direct light
        indirect: (H, W, T) transient with direct peaks removed
        valid_mask: (H, W) boolean mask of pixels with valid signal
    """
    H, W, T = transient.shape

    # Compute per-pixel max values for local thresholding
    pixel_max = np.max(transient, axis=-1)

    # Valid pixels have sufficient signal strength
    valid_mask = pixel_max > 0.0

    # Per-pixel threshold: fraction of local max
    local_threshold = pixel_max * 0.5

    # Find first bin above threshold for each pixel (vectorized)
    above_threshold = transient > local_threshold[:, :, np.newaxis]

    # For pixels with signal, find first bin above threshold
    first_above = np.argmax(above_threshold, axis=2)
    direct_bins = first_above

    # Compute relay depths from bin indices
    opl = start_opl + direct_bins.astype(np.float32) * bin_width
    relay_depths = opl / 2.0

    # Create direct mask (vectorized)
    bin_indices = np.arange(T)[np.newaxis, np.newaxis, :]
    direct_bins_expanded = direct_bins[:, :, np.newaxis]

    direct_mask = (bin_indices >= direct_bins_expanded - peak_width) & \
                  (bin_indices <= direct_bins_expanded + peak_width)

    # Create indirect transient (remove direct peaks)
    indirect = transient.copy()
    indirect[direct_mask] = 0

    # Report statistics only for valid pixels
    valid_depths = relay_depths[valid_mask]

    if len(valid_depths) > 0:
        print(f"  Relay depth range (valid pixels): [{valid_depths.min():.3f}, {valid_depths.max():.3f}] m")
    else:
        print(f"  WARNING: No valid pixels detected!")

    print(f"  Valid pixels: {valid_mask.sum()}/{valid_mask.size} ({100*valid_mask.sum()/valid_mask.size:.1f}%)")
    print(f"  Direct light fraction: {direct_mask.sum() / direct_mask.size:.2%}")

    return relay_depths, direct_mask, indirect, valid_mask


def direct_backproject_drjit(indirect_transient, relay_pos,
                              volume_min, volume_max, voxel_res,
                              start_opl, bin_width,
                              camera_pos=None, no_falloff=False,
                              bin_threshold=0.0, min_relay_distance=0.0,
                              relay_normals=None):
    """
    Direct backprojection using DrJit for GPU acceleration.

    For each voxel, sums contributions from all relay points:
    U(voxel) = sum_relay [ transient(relay, t) / r² ]

    where t is the time bin corresponding to the round-trip OPL and
    r is the distance from relay to voxel.

    Args:
        indirect_transient: (H, W, T) indirect transient data
        relay_pos: (H, W, 3) relay wall 3D positions
        volume_min/max: (3,) reconstruction volume bounds
        voxel_res: Number of voxels per dimension
        start_opl, bin_width: Temporal parameters
        camera_pos: (3,) camera position, defaults to origin
        no_falloff: If True, disable 1/r² distance falloff
        bin_threshold: Only splat to voxels whose corresponding time bin has
                       indirect signal above this fraction of the relay point's
                       max signal (0.0-1.0).
        min_relay_distance: Minimum distance from relay point to voxel (meters).
        relay_normals: (H, W, 3) unit normal vectors at each relay point.

    Returns:
        volume: (Vx, Vy, Vz) reconstructed intensity
    """
    H, W, T = indirect_transient.shape
    N_relay = H * W

    if camera_pos is None:
        camera_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    else:
        camera_pos = np.array(camera_pos, dtype=np.float32)

    # Flatten data for DrJit
    relay_flat = relay_pos.reshape(N_relay, 3).astype(np.float32)
    transient_flat = indirect_transient.reshape(N_relay, T).astype(np.float32)

    # Precompute camera-to-relay distances
    d_cam_relay = np.linalg.norm(relay_flat - camera_pos, axis=1).astype(np.float32)

    # Create voxel grid
    vx = np.linspace(volume_min[0], volume_max[0], voxel_res, dtype=np.float32)
    vy = np.linspace(volume_min[1], volume_max[1], voxel_res, dtype=np.float32)
    vz = np.linspace(volume_min[2], volume_max[2], voxel_res, dtype=np.float32)
    Vx, Vy, Vz = len(vx), len(vy), len(vz)
    N_voxels = Vx * Vy * Vz

    voxel_grid = np.stack(np.meshgrid(vx, vy, vz, indexing='ij'), axis=-1)
    voxels_flat = voxel_grid.reshape(N_voxels, 3).astype(np.float32)

    print(f"  N_relay: {N_relay}, N_voxels: {N_voxels}")
    print(f"  Uploading to GPU...")

    # Convert to DrJit arrays (on GPU)
    relay_x = mi.Float(relay_flat[:, 0])
    relay_y = mi.Float(relay_flat[:, 1])
    relay_z = mi.Float(relay_flat[:, 2])
    d_cam_relay_dr = mi.Float(d_cam_relay)

    # Store transient as DrJit TensorXf for efficient indexing
    transient_tensor = mi.TensorXf(transient_flat)

    # Upload normals if provided
    use_hemisphere_filter = relay_normals is not None
    if use_hemisphere_filter:
        normals_flat = relay_normals.reshape(N_relay, 3).astype(np.float32)
        normal_x = mi.Float(normals_flat[:, 0])
        normal_y = mi.Float(normals_flat[:, 1])
        normal_z = mi.Float(normals_flat[:, 2])
        print(f"  Hemisphere filtering: enabled (only positive normal hemisphere)")

    # Compute per-relay-point bin thresholds if requested
    use_bin_threshold = bin_threshold > 0.0
    if use_bin_threshold:
        relay_max = transient_flat.max(axis=1, keepdims=True)
        relay_thresholds = (bin_threshold * relay_max).flatten()
        relay_thresholds_dr = mi.Float(relay_thresholds)
        print(f"  Bin threshold: {bin_threshold:.2%} of each relay point's max")
        valid_bins = (transient_flat > relay_thresholds[:, np.newaxis]).sum()
        total_bins = N_relay * T
        print(f"    Bins above threshold: {valid_bins}/{total_bins} ({100*valid_bins/total_bins:.1f}%)")

    if min_relay_distance > 0.0:
        print(f"  Min relay distance: {min_relay_distance:.4f} m")

    # Process all voxels in parallel
    voxel_x = mi.Float(voxels_flat[:, 0])
    voxel_y = mi.Float(voxels_flat[:, 1])
    voxel_z = mi.Float(voxels_flat[:, 2])

    # Initialize accumulator
    vol_intensity = dr.zeros(mi.Float, N_voxels)

    print(f"  Backprojecting {N_relay} relay points...")

    # Loop over relay points (each contributes to all voxels)
    for r_idx in range(N_relay):
        # Get relay point position (scalar, broadcast to all voxels)
        rx = dr.gather(mi.Float, relay_x, mi.UInt32(r_idx))
        ry = dr.gather(mi.Float, relay_y, mi.UInt32(r_idx))
        rz = dr.gather(mi.Float, relay_z, mi.UInt32(r_idx))
        d_cam_r = dr.gather(mi.Float, d_cam_relay_dr, mi.UInt32(r_idx))

        # Distance from relay to each voxel (vectorized over voxels)
        dx = voxel_x - rx
        dy = voxel_y - ry
        dz = voxel_z - rz
        d_relay_voxel = dr.sqrt(dx * dx + dy * dy + dz * dz)

        # Total OPL: camera -> relay -> voxel -> relay -> camera
        opl_total = 2.0 * d_cam_r + 2.0 * d_relay_voxel

        # Convert to bin index
        bin_idx_float = (opl_total - start_opl) / bin_width
        bin_idx = mi.Int32(dr.floor(bin_idx_float))

        # Fractional part for linear interpolation
        frac = bin_idx_float - dr.floor(bin_idx_float)

        # Valid mask
        valid = (bin_idx >= 0) & (bin_idx < T - 1)

        # Apply minimum distance threshold: skip voxels too close to relay
        if min_relay_distance > 0.0:
            valid = valid & (d_relay_voxel > min_relay_distance)

        # Apply hemisphere filter: only voxels in positive normal hemisphere
        if use_hemisphere_filter:
            rnx = dr.gather(mi.Float, normal_x, mi.UInt32(r_idx))
            rny = dr.gather(mi.Float, normal_y, mi.UInt32(r_idx))
            rnz = dr.gather(mi.Float, normal_z, mi.UInt32(r_idx))
            dot_normal = dx * rnx + dy * rny + dz * rnz
            valid = valid & (dot_normal > 0.0)

        # Clamp indices for safe gathering
        idx0 = dr.clip(bin_idx, 0, T - 2)
        idx1 = idx0 + 1

        # Linear index into flattened transient array: r_idx * T + bin
        lin_idx0 = mi.UInt32(r_idx * T) + mi.UInt32(idx0)
        lin_idx1 = mi.UInt32(r_idx * T) + mi.UInt32(idx1)

        # Gather transient values
        val0 = dr.gather(mi.Float, transient_tensor.array, lin_idx0)
        val1 = dr.gather(mi.Float, transient_tensor.array, lin_idx1)

        # Linear interpolation
        transient_val = val0 * (1.0 - frac) + val1 * frac

        # Apply bin threshold: skip contributions where indirect signal is too weak
        if use_bin_threshold:
            relay_thresh = dr.gather(mi.Float, relay_thresholds_dr, mi.UInt32(r_idx))
            valid = valid & (transient_val > relay_thresh)

        # 1/r² distance weighting (radiometric falloff) or no falloff
        if no_falloff:
            weight = 1.0
        else:
            inv_r_sq = dr.rcp(dr.maximum(d_relay_voxel * d_relay_voxel, 0.000001))
            weight = inv_r_sq

        # Contribution: transient intensity weighted by 1/r²
        contrib = transient_val * weight

        # Accumulate with valid mask
        vol_intensity += dr.select(valid, contrib, 0.0)

        # Periodically evaluate to avoid graph explosion
        if (r_idx + 1) % 100 == 0:
            dr.eval(vol_intensity)
            print(f"    Progress: {r_idx + 1}/{N_relay} relay points", end='\r')

    print(f"    Progress: {N_relay}/{N_relay} relay points - Done!")

    dr.eval(vol_intensity)

    # Transfer back to numpy
    volume = np.array(vol_intensity).reshape(Vx, Vy, Vz)

    return volume


def apply_vis_transform(data, transform='none', scale=1.0, bias=0.0):
    """
    Apply a non-linear transform to data for visualization.

    The transform is: output = transform(scale * input + bias)

    Args:
        data: Input array
        transform: 'none' (linear), 'log' (log10), 'sqrt', 'cbrt' (cube root),
                   'exp' (exponential), 'sigmoid'
        scale: Scale factor applied before non-linearity
        bias: Bias (offset) applied before non-linearity

    Returns:
        Transformed data, label for colorbar
    """
    # Apply scale and bias first
    scaled_data = scale * data + bias

    eps = 1e-10
    if transform == 'log':
        return np.log10(scaled_data + eps), 'log₁₀(intensity)'
    elif transform == 'sqrt':
        return np.sqrt(np.maximum(scaled_data, 0)), '√intensity'
    elif transform == 'cbrt':
        return np.cbrt(scaled_data), '∛intensity'
    elif transform == 'exp':
        return np.exp(scaled_data), 'exp(intensity)'
    elif transform == 'sigmoid':
        return 1.0 / (1.0 + np.exp(-scaled_data)), 'σ(intensity)'
    else:  # 'none'
        return scaled_data, 'intensity'


def visualize_orthographic(volume, volume_min, volume_max, output_dir, output_name,
                           vis_transform='none', vis_scale=1.0, vis_bias=0.0):
    """
    Create front, side, top orthographic projections using max intensity projection.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Compute all projections first
    front = np.max(volume, axis=2)
    side = np.max(volume, axis=0)
    top = np.max(volume, axis=1)

    # Apply visualization transform
    front, label = apply_vis_transform(front, vis_transform, vis_scale, vis_bias)
    side, _ = apply_vis_transform(side, vis_transform, vis_scale, vis_bias)
    top, _ = apply_vis_transform(top, vis_transform, vis_scale, vis_bias)

    # Find global min/max across all projections for consistent normalization
    all_projs = [front, side, top]
    global_min = min(p.min() for p in all_projs)
    global_max = max(p.max() for p in all_projs)

    # Combined 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Build title suffix based on transform
    title_suffix = f' ({vis_transform})' if vis_transform != 'none' else ''

    im0 = axes[0].imshow(
        front.T, origin='lower', cmap='hot',
        extent=[volume_min[0], volume_max[0], volume_min[1], volume_max[1]],
        aspect='equal', vmin=global_min, vmax=global_max
    )
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')
    axes[0].set_title(f'Front View (XY){title_suffix}')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label=label)

    im1 = axes[1].imshow(
        side, origin='lower', cmap='hot',
        extent=[volume_min[2], volume_max[2], volume_min[1], volume_max[1]],
        aspect='equal', vmin=global_min, vmax=global_max
    )
    axes[1].set_xlabel('Z (m)')
    axes[1].set_ylabel('Y (m)')
    axes[1].set_title(f'Side View (ZY){title_suffix}')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label=label)

    im2 = axes[2].imshow(
        top.T, origin='lower', cmap='hot',
        extent=[volume_min[0], volume_max[0], volume_min[2], volume_max[2]],
        aspect='equal', vmin=global_min, vmax=global_max
    )
    axes[2].set_xlabel('X (m)')
    axes[2].set_ylabel('Z (m)')
    axes[2].set_title(f'Top View (XZ){title_suffix}')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label=label)

    plt.tight_layout()
    combined_path = os.path.join(output_dir, f'{output_name}_orthographic.png')
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {combined_path}")


def visualize_orthographic_color(volume_rgb, volume_min, volume_max, output_dir, output_name,
                                  vis_transform='none', vis_scale=1.0, vis_bias=0.0):
    """
    Create color orthographic projections (R, G, B channels) with global normalization.
    """
    os.makedirs(output_dir, exist_ok=True)

    front_rgb = np.max(volume_rgb, axis=2)
    side_rgb = np.max(volume_rgb, axis=0)
    top_rgb = np.max(volume_rgb, axis=1)

    # Apply scale and bias first
    front_rgb = vis_scale * front_rgb + vis_bias
    side_rgb = vis_scale * side_rgb + vis_bias
    top_rgb = vis_scale * top_rgb + vis_bias

    # Apply visualization transform to each channel
    eps = 1e-10
    if vis_transform == 'log':
        front_rgb = np.log10(front_rgb + eps)
        side_rgb = np.log10(side_rgb + eps)
        top_rgb = np.log10(top_rgb + eps)
        # Normalize log values to [0, 1] range
        all_projs = [front_rgb, side_rgb, top_rgb]
        global_min = min(p.min() for p in all_projs)
        global_max = max(p.max() for p in all_projs)
        if global_max > global_min:
            front_rgb = (front_rgb - global_min) / (global_max - global_min)
            side_rgb = (side_rgb - global_min) / (global_max - global_min)
            top_rgb = (top_rgb - global_min) / (global_max - global_min)
    elif vis_transform == 'sqrt':
        front_rgb = np.sqrt(np.maximum(front_rgb, 0))
        side_rgb = np.sqrt(np.maximum(side_rgb, 0))
        top_rgb = np.sqrt(np.maximum(top_rgb, 0))
        global_max = max(front_rgb.max(), side_rgb.max(), top_rgb.max())
        if global_max > 0:
            front_rgb /= global_max
            side_rgb /= global_max
            top_rgb /= global_max
    elif vis_transform == 'cbrt':
        front_rgb = np.cbrt(front_rgb)
        side_rgb = np.cbrt(side_rgb)
        top_rgb = np.cbrt(top_rgb)
        global_max = max(front_rgb.max(), side_rgb.max(), top_rgb.max())
        if global_max > 0:
            front_rgb /= global_max
            side_rgb /= global_max
            top_rgb /= global_max
    elif vis_transform == 'exp':
        front_rgb = np.exp(front_rgb)
        side_rgb = np.exp(side_rgb)
        top_rgb = np.exp(top_rgb)
        global_max = max(front_rgb.max(), side_rgb.max(), top_rgb.max())
        if global_max > 0:
            front_rgb /= global_max
            side_rgb /= global_max
            top_rgb /= global_max
    elif vis_transform == 'sigmoid':
        front_rgb = 1.0 / (1.0 + np.exp(-front_rgb))
        side_rgb = 1.0 / (1.0 + np.exp(-side_rgb))
        top_rgb = 1.0 / (1.0 + np.exp(-top_rgb))
        # Sigmoid output is already in [0, 1] range, just normalize
        global_max = max(front_rgb.max(), side_rgb.max(), top_rgb.max())
        if global_max > 0:
            front_rgb /= global_max
            side_rgb /= global_max
            top_rgb /= global_max
    else:  # 'none'
        # Linear normalization
        all_projs = [front_rgb, side_rgb, top_rgb]
        global_max = max(p.max() for p in all_projs)
        if global_max > 0:
            front_rgb = front_rgb / global_max
            side_rgb = side_rgb / global_max
            top_rgb = top_rgb / global_max

    front_rgb = np.clip(front_rgb, 0, 1)
    side_rgb = np.clip(side_rgb, 0, 1)
    top_rgb = np.clip(top_rgb, 0, 1)

    # Build title suffix based on transform
    title_suffix = f' ({vis_transform})' if vis_transform != 'none' else ''

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(
        np.transpose(front_rgb, (1, 0, 2)), origin='lower',
        extent=[volume_min[0], volume_max[0], volume_min[1], volume_max[1]],
        aspect='equal'
    )
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')
    axes[0].set_title(f'Front View (XY) - Color{title_suffix}')

    axes[1].imshow(
        side_rgb, origin='lower',
        extent=[volume_min[2], volume_max[2], volume_min[1], volume_max[1]],
        aspect='equal'
    )
    axes[1].set_xlabel('Z (m)')
    axes[1].set_ylabel('Y (m)')
    axes[1].set_title(f'Side View (ZY) - Color{title_suffix}')

    axes[2].imshow(
        np.transpose(top_rgb, (1, 0, 2)), origin='lower',
        extent=[volume_min[0], volume_max[0], volume_min[2], volume_max[2]],
        aspect='equal'
    )
    axes[2].set_xlabel('X (m)')
    axes[2].set_ylabel('Z (m)')
    axes[2].set_title(f'Top View (XZ) - Color{title_suffix}')

    plt.tight_layout()
    combined_path = os.path.join(output_dir, f'{output_name}_orthographic_color.png')
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {combined_path}")


def visualize_orthographic_with_pointcloud(volume_rgb, relay_pos, relay_colors,
                                            volume_min, volume_max, output_dir, output_name,
                                            point_alpha=0.7, point_size=2,
                                            vis_transform='none', vis_scale=1.0, vis_bias=0.0):
    """
    Create color orthographic projections overlaid with the unprojected color point cloud.
    """
    os.makedirs(output_dir, exist_ok=True)

    points_3d = relay_pos.reshape(-1, 3)
    colors_flat = relay_colors.reshape(-1, 3)

    color_max = colors_flat.max()
    if color_max > 0:
        colors_normalized = colors_flat / color_max
    else:
        colors_normalized = colors_flat
    colors_normalized = np.clip(colors_normalized, 0, 1)

    front_rgb = np.max(volume_rgb, axis=2)
    side_rgb = np.max(volume_rgb, axis=0)
    top_rgb = np.max(volume_rgb, axis=1)

    # Apply scale and bias first
    front_rgb = vis_scale * front_rgb + vis_bias
    side_rgb = vis_scale * side_rgb + vis_bias
    top_rgb = vis_scale * top_rgb + vis_bias

    # Apply visualization transform
    eps = 1e-10
    if vis_transform == 'log':
        front_rgb = np.log10(front_rgb + eps)
        side_rgb = np.log10(side_rgb + eps)
        top_rgb = np.log10(top_rgb + eps)
        # Normalize log values to [0, 1] range
        all_projs = [front_rgb, side_rgb, top_rgb]
        global_min = min(p.min() for p in all_projs)
        global_max = max(p.max() for p in all_projs)
        if global_max > global_min:
            front_rgb = (front_rgb - global_min) / (global_max - global_min)
            side_rgb = (side_rgb - global_min) / (global_max - global_min)
            top_rgb = (top_rgb - global_min) / (global_max - global_min)
    elif vis_transform == 'sqrt':
        front_rgb = np.sqrt(np.maximum(front_rgb, 0))
        side_rgb = np.sqrt(np.maximum(side_rgb, 0))
        top_rgb = np.sqrt(np.maximum(top_rgb, 0))
        global_max = max(front_rgb.max(), side_rgb.max(), top_rgb.max())
        if global_max > 0:
            front_rgb /= global_max
            side_rgb /= global_max
            top_rgb /= global_max
    elif vis_transform == 'cbrt':
        front_rgb = np.cbrt(front_rgb)
        side_rgb = np.cbrt(side_rgb)
        top_rgb = np.cbrt(top_rgb)
        global_max = max(front_rgb.max(), side_rgb.max(), top_rgb.max())
        if global_max > 0:
            front_rgb /= global_max
            side_rgb /= global_max
            top_rgb /= global_max
    elif vis_transform == 'exp':
        front_rgb = np.exp(front_rgb)
        side_rgb = np.exp(side_rgb)
        top_rgb = np.exp(top_rgb)
        global_max = max(front_rgb.max(), side_rgb.max(), top_rgb.max())
        if global_max > 0:
            front_rgb /= global_max
            side_rgb /= global_max
            top_rgb /= global_max
    elif vis_transform == 'sigmoid':
        front_rgb = 1.0 / (1.0 + np.exp(-front_rgb))
        side_rgb = 1.0 / (1.0 + np.exp(-side_rgb))
        top_rgb = 1.0 / (1.0 + np.exp(-top_rgb))
        global_max = max(front_rgb.max(), side_rgb.max(), top_rgb.max())
        if global_max > 0:
            front_rgb /= global_max
            side_rgb /= global_max
            top_rgb /= global_max
    else:  # 'none'
        all_projs = [front_rgb, side_rgb, top_rgb]
        global_max = max(p.max() for p in all_projs)
        if global_max > 0:
            front_rgb = front_rgb / global_max
            side_rgb = side_rgb / global_max
            top_rgb = top_rgb / global_max

    front_rgb = np.clip(front_rgb, 0, 1)
    side_rgb = np.clip(side_rgb, 0, 1)
    top_rgb = np.clip(top_rgb, 0, 1)

    px, py, pz = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(
        np.transpose(front_rgb, (1, 0, 2)), origin='lower',
        extent=[volume_min[0], volume_max[0], volume_min[1], volume_max[1]],
        aspect='equal'
    )
    axes[0].scatter(px, py, c=colors_normalized, s=point_size, alpha=point_alpha, marker='.')
    axes[0].set_xlim([volume_min[0], volume_max[0]])
    axes[0].set_ylim([volume_min[1], volume_max[1]])
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')
    axes[0].set_title('Front View (XY) - Reconstruction + Point Cloud')

    axes[1].imshow(
        side_rgb, origin='lower',
        extent=[volume_min[2], volume_max[2], volume_min[1], volume_max[1]],
        aspect='equal'
    )
    axes[1].scatter(pz, py, c=colors_normalized, s=point_size, alpha=point_alpha, marker='.')
    axes[1].set_xlim([volume_min[2], volume_max[2]])
    axes[1].set_ylim([volume_min[1], volume_max[1]])
    axes[1].set_xlabel('Z (m)')
    axes[1].set_ylabel('Y (m)')
    axes[1].set_title('Side View (ZY) - Reconstruction + Point Cloud')

    axes[2].imshow(
        np.transpose(top_rgb, (1, 0, 2)), origin='lower',
        extent=[volume_min[0], volume_max[0], volume_min[2], volume_max[2]],
        aspect='equal'
    )
    axes[2].scatter(px, pz, c=colors_normalized, s=point_size, alpha=point_alpha, marker='.')
    axes[2].set_xlim([volume_min[0], volume_max[0]])
    axes[2].set_ylim([volume_min[2], volume_max[2]])
    axes[2].set_xlabel('X (m)')
    axes[2].set_ylabel('Z (m)')
    axes[2].set_title('Top View (XZ) - Reconstruction + Point Cloud')

    plt.tight_layout()
    combined_path = os.path.join(output_dir, f'{output_name}_orthographic_overlay.png')
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {combined_path}")


def visualize_direct_vs_indirect(transient_original, indirect, direct_mask,
                                  start_opl, bin_width, output_dir, output_name,
                                  pixel_h=None, pixel_w=None, use_log_scale=True):
    """
    Visualize the original transient vs the filtered indirect transient at a single pixel.

    Args:
        use_log_scale: If True, use log scale for transient traces
    """
    os.makedirs(output_dir, exist_ok=True)
    H, W, T = transient_original.shape
    time_axis = start_opl + np.arange(T) * bin_width

    if pixel_h is None:
        pixel_h = H // 2
    if pixel_w is None:
        pixel_w = W // 2

    pixel_h = max(0, min(pixel_h, H - 1))
    pixel_w = max(0, min(pixel_w, W - 1))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    eps = 1e-10
    orig_max = np.max(transient_original, axis=2)
    indirect_max = np.max(indirect, axis=2)
    direct_energy = np.sum(transient_original * direct_mask, axis=2)

    # Apply log scale for spatial images if enabled
    if use_log_scale:
        im00 = axes[0, 0].imshow(np.log10(orig_max + eps), cmap='hot', origin='lower')
        plt.colorbar(im00, ax=axes[0, 0], label='log₁₀')
        axes[0, 0].set_title('Original: Max over time (log)')
    else:
        im00 = axes[0, 0].imshow(orig_max, cmap='hot', origin='lower')
        plt.colorbar(im00, ax=axes[0, 0])
        axes[0, 0].set_title('Original: Max over time')
    axes[0, 0].scatter([pixel_w], [pixel_h], c='cyan', s=100, marker='x', linewidths=2)
    axes[0, 0].set_xlabel('Pixel X')
    axes[0, 0].set_ylabel('Pixel Y')

    if use_log_scale:
        im01 = axes[0, 1].imshow(np.log10(indirect_max + eps), cmap='hot', origin='lower')
        plt.colorbar(im01, ax=axes[0, 1], label='log₁₀')
        axes[0, 1].set_title('Indirect: Max over time (log)')
    else:
        im01 = axes[0, 1].imshow(indirect_max, cmap='hot', origin='lower')
        plt.colorbar(im01, ax=axes[0, 1])
        axes[0, 1].set_title('Indirect: Max over time')
    axes[0, 1].scatter([pixel_w], [pixel_h], c='cyan', s=100, marker='x', linewidths=2)
    axes[0, 1].set_xlabel('Pixel X')
    axes[0, 1].set_ylabel('Pixel Y')

    if use_log_scale:
        im02 = axes[0, 2].imshow(np.log10(direct_energy + eps), cmap='hot', origin='lower')
        plt.colorbar(im02, ax=axes[0, 2], label='log₁₀')
        axes[0, 2].set_title('Direct Peak Energy (log)')
    else:
        im02 = axes[0, 2].imshow(direct_energy, cmap='hot', origin='lower')
        plt.colorbar(im02, ax=axes[0, 2])
        axes[0, 2].set_title('Direct Peak Energy')
    axes[0, 2].scatter([pixel_w], [pixel_h], c='cyan', s=100, marker='x', linewidths=2)
    axes[0, 2].set_xlabel('Pixel X')
    axes[0, 2].set_ylabel('Pixel Y')

    orig_trace = transient_original[pixel_h, pixel_w, :]
    indirect_trace = indirect[pixel_h, pixel_w, :]
    mask_trace = direct_mask[pixel_h, pixel_w, :]

    # Transient trace plots
    axes[1, 0].plot(time_axis, orig_trace + (eps if use_log_scale else 0), 'b-', linewidth=1)
    axes[1, 0].set_xlabel('OPL (m)')
    axes[1, 0].set_ylabel('Intensity')
    axes[1, 0].set_title(f'Original Transient at ({pixel_h}, {pixel_w})')
    axes[1, 0].set_xlim([time_axis[0], time_axis[-1]])
    if use_log_scale:
        axes[1, 0].set_yscale('log')

    axes[1, 1].plot(time_axis, indirect_trace + (eps if use_log_scale else 0), 'g-', linewidth=1)
    axes[1, 1].set_xlabel('OPL (m)')
    axes[1, 1].set_ylabel('Intensity')
    axes[1, 1].set_title(f'Indirect Transient at ({pixel_h}, {pixel_w})')
    axes[1, 1].set_xlim([time_axis[0], time_axis[-1]])
    if use_log_scale:
        axes[1, 1].set_yscale('log')

    axes[1, 2].plot(time_axis, orig_trace + (eps if use_log_scale else 0), 'b-', linewidth=1, alpha=0.7, label='Original')
    axes[1, 2].plot(time_axis, indirect_trace + (eps if use_log_scale else 0), 'g-', linewidth=1, alpha=0.7, label='Indirect')
    mask_indices = np.where(mask_trace)[0]
    if len(mask_indices) > 0:
        mask_start = time_axis[mask_indices[0]]
        mask_end = time_axis[mask_indices[-1]]
        axes[1, 2].axvspan(mask_start, mask_end, alpha=0.3, color='red', label='Direct mask')
    axes[1, 2].set_xlabel('OPL (m)')
    axes[1, 2].set_ylabel('Intensity')
    axes[1, 2].set_title(f'Comparison at ({pixel_h}, {pixel_w})')
    axes[1, 2].legend()
    axes[1, 2].set_xlim([time_axis[0], time_axis[-1]])
    if use_log_scale:
        axes[1, 2].set_yscale('log')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{output_name}_direct_vs_indirect.png')
    plt.savefig(save_path, dpi=250, bbox_inches='tight')
    plt.close()
    print(f"  Saved direct vs indirect visualization: {save_path}")


def visualize_depths(relay_depths, transient, direct_mask, start_opl, bin_width,
                     output_dir, output_name, use_log_scale=True):
    """
    Visualize detected relay wall depths and direct peak detection.

    Args:
        use_log_scale: If True, use log scale for transient traces
    """
    os.makedirs(output_dir, exist_ok=True)
    H, W, T = transient.shape
    eps = 1e-10

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    im0 = axes[0, 0].imshow(relay_depths, cmap='viridis', origin='lower')
    axes[0, 0].set_title('Detected Relay Wall Depths')
    axes[0, 0].set_xlabel('Pixel X')
    axes[0, 0].set_ylabel('Pixel Y')
    plt.colorbar(im0, ax=axes[0, 0], label='Depth (m)')

    axes[0, 1].hist(relay_depths.flatten(), bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Depth (m)')
    axes[0, 1].set_ylabel('Pixel Count')
    axes[0, 1].set_title('Depth Distribution')
    axes[0, 1].axvline(relay_depths.mean(), color='r', linestyle='--',
                       label=f'Mean: {relay_depths.mean():.3f}m')
    axes[0, 1].legend()

    direct_bins = (relay_depths * 2 - start_opl) / bin_width
    im2 = axes[0, 2].imshow(direct_bins, cmap='plasma', origin='lower')
    axes[0, 2].set_title('Direct Peak Bin Index')
    axes[0, 2].set_xlabel('Pixel X')
    axes[0, 2].set_ylabel('Pixel Y')
    plt.colorbar(im2, ax=axes[0, 2], label='Bin Index')

    center_h, center_w = H // 2, W // 2
    time_axis = start_opl + np.arange(T) * bin_width

    offsets = [(0, 0), (-H//4, 0), (H//4, 0), (0, -W//4), (0, W//4)]
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    for (dh, dw), color in zip(offsets, colors):
        h_idx = center_h + dh
        w_idx = center_w + dw
        if 0 <= h_idx < H and 0 <= w_idx < W:
            trace = transient[h_idx, w_idx, :]
            depth = relay_depths[h_idx, w_idx]
            axes[1, 0].plot(time_axis, trace + (eps if use_log_scale else 0), color=color, alpha=0.7,
                           label=f'({h_idx},{w_idx}), d={depth:.2f}m')
            peak_opl = depth * 2
            axes[1, 0].axvline(peak_opl, color=color, linestyle='--', alpha=0.3)

    axes[1, 0].set_xlabel('OPL (m)')
    axes[1, 0].set_ylabel('Intensity')
    axes[1, 0].set_title('Example Transient Traces' + (' (log scale)' if use_log_scale else ''))
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].set_xlim([time_axis[0], time_axis[-1]])
    if use_log_scale:
        axes[1, 0].set_yscale('log')

    temporal_sum = transient.sum(axis=(0, 1))
    axes[1, 1].plot(time_axis, temporal_sum + (eps if use_log_scale else 0), 'b-')
    axes[1, 1].set_xlabel('OPL (m)')
    axes[1, 1].set_ylabel('Total Intensity')
    axes[1, 1].set_title('Spatially-Summed Transient' + (' (log scale)' if use_log_scale else ''))
    axes[1, 1].axvline(relay_depths.mean() * 2, color='r', linestyle='--',
                       label=f'Mean direct OPL: {relay_depths.mean()*2:.2f}m')
    axes[1, 1].legend()
    if use_log_scale:
        axes[1, 1].set_yscale('log')

    direct_energy = transient[direct_mask].sum()
    total_energy = transient.sum()
    indirect_energy = total_energy - direct_energy

    axes[1, 2].bar(['Direct', 'Indirect', 'Total'],
                   [direct_energy, indirect_energy, total_energy],
                   color=['orange', 'blue', 'green'])
    axes[1, 2].set_ylabel('Total Energy')
    axes[1, 2].set_title('Energy Distribution')
    if use_log_scale:
        axes[1, 2].set_yscale('log')
    for i, v in enumerate([direct_energy, indirect_energy, total_energy]):
        axes[1, 2].text(i, v * 1.1, f'{v:.2e}', ha='center', fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{output_name}_depths_debug.png')
    plt.savefig(save_path, dpi=250, bbox_inches='tight')
    plt.close()
    print(f"  Saved depth visualization: {save_path}")


def normalize_volume_by_slices(volume, axis='y'):
    """
    Normalize volume intensity per slice along the specified axis.
    """
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if axis not in axis_map:
        print(f"  WARNING: Invalid axis '{axis}', skipping normalization")
        return volume

    ax = axis_map[axis]
    volume_norm = volume.copy()

    if volume.ndim == 4:
        n_slices = volume.shape[ax]
        for i in range(n_slices):
            if ax == 0:
                slice_data = volume_norm[i, :, :, :]
            elif ax == 1:
                slice_data = volume_norm[:, i, :, :]
            else:
                slice_data = volume_norm[:, :, i, :]

            slice_max = slice_data.max()
            if slice_max > 0:
                slice_data /= slice_max
    else:
        n_slices = volume.shape[ax]
        for i in range(n_slices):
            if ax == 0:
                slice_data = volume_norm[i, :, :]
            elif ax == 1:
                slice_data = volume_norm[:, i, :]
            else:
                slice_data = volume_norm[:, :, i]

            slice_max = slice_data.max()
            if slice_max > 0:
                if ax == 0:
                    volume_norm[i, :, :] = slice_data / slice_max
                elif ax == 1:
                    volume_norm[:, i, :] = slice_data / slice_max
                else:
                    volume_norm[:, :, i] = slice_data / slice_max

    print(f"  Normalized volume by {axis}-slices ({n_slices} slices)")
    return volume_norm


def extract_direct_colors(transient_rgb, direct_mask):
    """
    Extract colors at the direct peak for each pixel.
    """
    H, W, T, C = transient_rgb.shape

    colors = np.zeros((H, W, 3), dtype=np.float32)

    for c in range(3):
        channel_data = transient_rgb[..., c]
        colors[..., c] = np.sum(channel_data * direct_mask, axis=2)

    return colors


def save_results(volume, relay_depths, output_dir, output_name):
    """Save reconstruction results to numpy files."""
    os.makedirs(output_dir, exist_ok=True)

    volume_path = os.path.join(output_dir, f'{output_name}_volume.npy')
    np.save(volume_path, volume)
    print(f"  Saved volume: {volume_path}")

    depths_path = os.path.join(output_dir, f'{output_name}_relay_depths.npy')
    np.save(depths_path, relay_depths)
    print(f"  Saved relay depths: {depths_path}")


def compute_volume_bounds_from_camera(sensor, depth_min, depth_max, margin=0.1):
    """
    Compute volume bounds by backprojecting camera frustum corners.
    """
    sample_points = [
        (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0),
        (0.5, 0.0), (0.5, 1.0), (0.0, 0.5), (1.0, 0.5), (0.5, 0.5),
    ]

    all_points = []

    for u, v in sample_points:
        sample_pos = mi.Point2f(mi.Float(u), mi.Float(v))
        rays, _ = sensor.sample_ray(
            time=mi.Float(0.0),
            sample1=mi.Float(0.0),
            sample2=sample_pos,
            sample3=mi.Point2f(mi.Float(0.5), mi.Float(0.5))
        )

        ray_o = np.array([np.array(rays.o.x)[0], np.array(rays.o.y)[0], np.array(rays.o.z)[0]])
        ray_d = np.array([np.array(rays.d.x)[0], np.array(rays.d.y)[0], np.array(rays.d.z)[0]])
        ray_d = ray_d / np.linalg.norm(ray_d)

        point_near = ray_o + depth_min * ray_d
        point_far = ray_o + depth_max * ray_d

        all_points.append(point_near)
        all_points.append(point_far)

    all_points = np.array(all_points)

    volume_min = all_points.min(axis=0)
    volume_max = all_points.max(axis=0)

    extent = volume_max - volume_min
    volume_min = volume_min - margin * extent
    volume_max = volume_max + margin * extent

    return volume_min.astype(np.float32), volume_max.astype(np.float32)


def compute_default_volume_bounds(relay_pos, camera_origin, valid_mask=None):
    """
    Compute default volume bounds based on relay wall positions.
    """
    if valid_mask is not None:
        valid_positions = relay_pos[valid_mask]
        if len(valid_positions) == 0:
            print("  WARNING: No valid positions, using all pixels for bounds")
            valid_positions = relay_pos.reshape(-1, 3)
    else:
        valid_positions = relay_pos.reshape(-1, 3)

    relay_min = valid_positions.min(axis=0)
    relay_max = valid_positions.max(axis=0)
    relay_center = (relay_min + relay_max) / 2

    wall_extent = relay_max - relay_min
    wall_size = max(wall_extent[0], wall_extent[1])

    cam_to_wall = relay_center - camera_origin
    wall_normal = cam_to_wall / (np.linalg.norm(cam_to_wall) + 1e-8)

    volume_depth = wall_size * 2.0

    margin = wall_size * 0.2
    volume_min = np.array([
        relay_min[0] - margin,
        relay_min[1] - margin,
        relay_center[2] + volume_depth * 0.1
    ], dtype=np.float32)

    volume_max = np.array([
        relay_max[0] + margin,
        relay_max[1] + margin,
        relay_center[2] + volume_depth
    ], dtype=np.float32)

    return volume_min, volume_max


def main():
    args = parse_args()

    print("=" * 60)
    print("Direct Backprojection for NLOS Reconstruction")
    print("=" * 60)

    # 1. Load scene parameters from XML
    print("\n[1/6] Loading scene parameters...")
    scene_params = load_scene_params(args.scene_file)

    start_opl = args.start_opl if args.start_opl is not None else scene_params['start_opl']
    bin_width = args.bin_width if args.bin_width is not None else scene_params['bin_width']

    print(f"  Using start_opl: {start_opl} m")
    print(f"  Using bin_width: {bin_width} m")

    # 2. Load transient data
    direct_file = args.direct_file if args.direct_file else args.transient_file
    indirect_file = args.indirect_file if args.indirect_file else args.transient_file

    using_separate_files = (direct_file != indirect_file)

    if using_separate_files:
        print("\n[2/6] Loading transient data (separate files)...")
        print(f"  Direct file (geometry): {direct_file}")
        print(f"  Indirect file (reconstruction): {indirect_file}")

        print("  Loading direct file...")
        transient_direct, transient_direct_rgb = load_transient(direct_file, return_rgb=True)
        print(f"    Luminance shape: {transient_direct.shape}")

        print("  Loading indirect file...")
        transient_indirect, transient_indirect_rgb = load_transient(indirect_file, return_rgb=True)
        print(f"    Luminance shape: {transient_indirect.shape}")
        print(f"    RGB shape: {transient_indirect_rgb.shape}")
    else:
        print("\n[2/6] Loading transient data...")
        transient_direct, transient_direct_rgb = load_transient(args.transient_file, return_rgb=True)
        transient_indirect = transient_direct
        transient_indirect_rgb = transient_direct_rgb
        print(f"  Luminance shape: {transient_direct.shape}")
        print(f"  RGB shape: {transient_direct_rgb.shape}")

    # 3. Detect direct peaks and extract relay geometry
    print("\n[3/6] Detecting direct peaks...")
    relay_depths, direct_mask, indirect, valid_mask = detect_direct_peaks(
        transient_direct, start_opl, bin_width
    )

    # Create indirect data (remove direct peaks from indirect file)
    indirect_rgb = transient_indirect_rgb.copy()
    for c in range(3):
        indirect_rgb[..., c][direct_mask] = 0

    transient_rgb = transient_direct_rgb

    # Determine output name
    output_name = args.output_name or os.path.splitext(os.path.basename(args.transient_file))[0].replace('_transient', '_direct_bp')

    # Visualize depths
    print("\n  Generating depth visualizations...")
    visualize_depths(relay_depths, transient_direct, direct_mask, start_opl, bin_width,
                     args.output_dir, output_name, use_log_scale=args.transient_log)

    # Visualize direct vs indirect transient
    print("\n  Generating direct vs indirect visualization...")
    debug_pixel_h = args.debug_pixel[0] if args.debug_pixel else None
    debug_pixel_w = args.debug_pixel[1] if args.debug_pixel else None
    visualize_direct_vs_indirect(transient_direct, indirect, direct_mask,
                                  start_opl, bin_width, args.output_dir, output_name,
                                  pixel_h=debug_pixel_h, pixel_w=debug_pixel_w,
                                  use_log_scale=args.transient_log)

    # 4. Compute relay wall 3D positions
    print("\n[4/6] Computing relay wall positions (unprojecting camera rays)...")
    relay_pos = compute_relay_positions_from_camera(
        relay_depths,
        scene_params['sensor'],
        scene_params['camera_origin']
    )

    valid_pos = relay_pos[valid_mask]
    if len(valid_pos) > 0:
        print(f"  Relay position range (valid): X=[{valid_pos[:,0].min():.2f}, {valid_pos[:,0].max():.2f}]")
        print(f"                                Y=[{valid_pos[:,1].min():.2f}, {valid_pos[:,1].max():.2f}]")
        print(f"                                Z=[{valid_pos[:,2].min():.2f}, {valid_pos[:,2].max():.2f}]")
    else:
        print(f"  WARNING: No valid relay positions!")

    # Compute relay wall normals (for hemisphere filtering)
    if args.no_hemisphere_filter:
        print("\n  Hemisphere filtering: DISABLED")
        relay_normals = None
    else:
        print("\n  Computing relay wall normals (for hemisphere filtering)...")
        camera_origin = scene_params['camera_origin']
        relay_normals = compute_relay_normals(relay_pos, camera_origin)

    # 5. Determine volume bounds
    print("\n[5/6] Setting up reconstruction volume...")
    camera_origin = scene_params['camera_origin']

    if args.volume_min is not None and args.volume_max is not None:
        volume_min = np.array(args.volume_min, dtype=np.float32)
        volume_max = np.array(args.volume_max, dtype=np.float32)
        print(f"  Using user-specified volume bounds")
    elif args.volume_from_camera:
        valid_depths = relay_depths[valid_mask]
        if len(valid_depths) > 0:
            depth_min = args.volume_depth_min if args.volume_depth_min is not None else valid_depths.min()
            depth_max = args.volume_depth_max if args.volume_depth_max is not None else valid_depths.max() * 2.0
        else:
            depth_min = args.volume_depth_min if args.volume_depth_min is not None else relay_depths.min()
            depth_max = args.volume_depth_max if args.volume_depth_max is not None else relay_depths.max() * 2.0
        print(f"  Computing volume from camera frustum:")
        print(f"    Depth range: [{depth_min:.3f}, {depth_max:.3f}] m")
        print(f"    Margin: {args.volume_margin * 100:.1f}%")
        volume_min, volume_max = compute_volume_bounds_from_camera(
            scene_params['sensor'], depth_min, depth_max, margin=args.volume_margin
        )
        print(f"  Volume bounds from camera frustum backprojection")
    else:
        volume_min, volume_max = compute_default_volume_bounds(relay_pos, camera_origin, valid_mask)
        print(f"  Auto-computed volume bounds from relay wall (valid pixels only)")

    print(f"  Volume min: [{volume_min[0]:.2f}, {volume_min[1]:.2f}, {volume_min[2]:.2f}]")
    print(f"  Volume max: [{volume_max[0]:.2f}, {volume_max[1]:.2f}, {volume_max[2]:.2f}]")

    # 6. Direct backprojection with DrJit (GPU)
    falloff_str = "no falloff" if args.no_falloff else "1/r² falloff"
    print(f"\n[6/7] Backprojecting luminance to {args.voxel_resolution}^3 volume (DrJit/GPU, {falloff_str})...")

    volume = direct_backproject_drjit(
        indirect, relay_pos,
        volume_min, volume_max, args.voxel_resolution,
        start_opl, bin_width,
        camera_origin, no_falloff=args.no_falloff,
        bin_threshold=args.bin_threshold,
        min_relay_distance=args.min_relay_distance,
        relay_normals=relay_normals
    )
    print(f"  Volume range: [{volume.min():.6f}, {volume.max():.6f}]")

    # 7. Direct backprojection for RGB channels
    print(f"\n[7/8] Backprojecting RGB channels...")
    channel_names = ['Red', 'Green', 'Blue']
    volume_rgb = np.zeros((args.voxel_resolution, args.voxel_resolution, args.voxel_resolution, 3),
                          dtype=np.float32)

    for c in range(3):
        print(f"  Processing {channel_names[c]} channel...")
        volume_rgb[..., c] = direct_backproject_drjit(
            indirect_rgb[..., c], relay_pos,
            volume_min, volume_max, args.voxel_resolution,
            start_opl, bin_width,
            camera_origin, no_falloff=args.no_falloff,
            bin_threshold=args.bin_threshold,
            min_relay_distance=args.min_relay_distance,
            relay_normals=relay_normals
        )
        print(f"    {channel_names[c]} range: [{volume_rgb[..., c].min():.6f}, {volume_rgb[..., c].max():.6f}]")

    # 8. Optional slice normalization
    if args.normalize_slices and args.normalize_slices != 'none':
        print(f"\n[8/9] Normalizing volumes by {args.normalize_slices}-slices...")
        volume = normalize_volume_by_slices(volume, axis=args.normalize_slices)
        volume_rgb = normalize_volume_by_slices(volume_rgb, axis=args.normalize_slices)

    # 9. Extract direct colors for point cloud overlay
    print(f"\n[9/10] Extracting direct colors for point cloud...")
    relay_colors = extract_direct_colors(transient_rgb, direct_mask)

    # 10. Visualize and save
    print(f"\n[10/10] Generating visualizations and saving...")
    visualize_orthographic(volume, volume_min, volume_max, args.output_dir, output_name,
                           vis_transform=args.vis_transform, vis_scale=args.vis_scale, vis_bias=args.vis_bias)
    visualize_orthographic_color(volume_rgb, volume_min, volume_max, args.output_dir, output_name,
                                  vis_transform=args.vis_transform, vis_scale=args.vis_scale, vis_bias=args.vis_bias)
    visualize_orthographic_with_pointcloud(volume_rgb, relay_pos, relay_colors,
                                           volume_min, volume_max, args.output_dir, output_name,
                                           vis_transform=args.vis_transform, vis_scale=args.vis_scale, vis_bias=args.vis_bias)
    save_results(volume, relay_depths, args.output_dir, output_name)

    # Also save RGB volume
    volume_rgb_path = os.path.join(args.output_dir, f'{output_name}_volume_rgb.npy')
    np.save(volume_rgb_path, volume_rgb)
    print(f"  Saved RGB volume: {volume_rgb_path}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
