#!/usr/bin/env python3
"""
Phasor Field Backprojection for NLOS Reconstruction

This script implements Rayleigh-Sommerfeld diffraction via direct summation
for reconstructing hidden scenes from confocal transient NLOS data.

Algorithm:
1. Load transient data from render_transient.py output
2. Detect direct peaks to recover relay wall geometry
3. Apply phasor field convolution (virtual wavelength)
4. Backproject using DrJit GPU acceleration
5. Visualize orthographic projections (front, side, top)
"""

import os
import argparse
import numpy as np
import scipy.signal
import scipy.ndimage
import matplotlib.pyplot as plt

# Set Mitsuba variant before importing
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import mitransient as mitr
import drjit as dr


def parse_args():
    parser = argparse.ArgumentParser(
        description='Phasor field backprojection for NLOS reconstruction'
    )
    parser.add_argument('transient_file', type=str,
                        help='Path to *_transient.npy file')
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
    parser.add_argument('--wavelength', type=float, default=0.05,
                        help='Virtual wavelength for phasor field (meters, default: 0.05)')
    parser.add_argument('--output-dir', type=str, default='vis/',
                        help='Output directory (default: vis/)')
    parser.add_argument('--output-name', type=str, default=None,
                        help='Output name prefix (default: derived from input)')
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
    # j corresponds to columns (x), i corresponds to rows (y)
    j_coords = np.arange(W, dtype=np.float32)
    i_coords = np.arange(H, dtype=np.float32)
    jj, ii = np.meshgrid(j_coords, i_coords)  # shape (H, W)

    u_coords = (jj + 0.5) / film_w  # shape (H, W)
    v_coords = (ii + 0.5) / film_h  # shape (H, W)

    # Flatten for batch processing
    u_flat = u_coords.flatten()
    v_flat = v_coords.flatten()
    N = len(u_flat)

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


def load_transient(path):
    """
    Load transient data and convert to luminance.

    Args:
        path: Path to .npy file with shape (H, W, T, C)

    Returns:
        luminance: (H, W, T) array
    """
    data = np.load(path)
    print(f"  Loaded shape: {data.shape}")

    # Convert to luminance for reconstruction
    if data.shape[-1] >= 3:
        luminance = (0.2126 * data[..., 0] +
                     0.7152 * data[..., 1] +
                     0.0722 * data[..., 2])
    else:
        luminance = data[..., 0]

    return luminance.astype(np.float32)


def detect_direct_peaks(transient, start_opl, bin_width, threshold_percentile=95, peak_width=5):
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

    Returns:
        relay_depths: (H, W) array of wall depths (meters)
        direct_mask: (H, W, T) boolean mask of direct light
        indirect: (H, W, T) transient with direct peaks removed
    """
    H, W, T = transient.shape

    # Compute per-pixel max values for local thresholding
    pixel_max = np.max(transient, axis=2)  # (H, W)

    # Global threshold based on non-zero max values
    nonzero_max = pixel_max[pixel_max > 0]
    if len(nonzero_max) > 0:
        global_threshold = np.percentile(nonzero_max, threshold_percentile) * 0.1
    else:
        global_threshold = 1e-10

    # Per-pixel threshold: fraction of local max, but at least global_threshold
    local_threshold = pixel_max * 0.1  # (H, W)
    threshold = np.maximum(local_threshold, global_threshold)  # (H, W)

    # Find first bin above threshold for each pixel (vectorized)
    # Create mask of bins above threshold
    above_threshold = transient > threshold[:, :, np.newaxis]  # (H, W, T)

    # Find first bin index above threshold using argmax on cumsum
    # argmax returns first True when applied to boolean array
    # But we need to handle case where no bin is above threshold
    any_above = np.any(above_threshold, axis=2)  # (H, W)

    # For pixels with signal, find first bin above threshold
    # argmax on boolean gives index of first True
    first_above = np.argmax(above_threshold, axis=2)  # (H, W)

    # For pixels with signal, refine to find local maximum near first_above
    # Look in a window around first_above to find the actual peak
    window_size = 10
    direct_bins = np.zeros((H, W), dtype=np.int32)

    # Vectorized peak refinement: for each pixel, find max in window around first_above
    for di in range(-window_size, window_size + 1):
        candidate_bins = np.clip(first_above + di, 0, T - 1)
        # Gather values at candidate bins
        ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        candidate_vals = transient[ii, jj, candidate_bins]
        current_vals = transient[ii, jj, direct_bins]
        # Update where candidate is larger
        better = candidate_vals > current_vals
        direct_bins = np.where(better, candidate_bins, direct_bins)

    # Default for pixels with no signal
    default_bin = T // 4
    direct_bins = np.where(any_above, direct_bins, default_bin)

    # Compute relay depths from bin indices
    # OPL = start_opl + bin_index * bin_width (round trip)
    # depth = OPL / 2
    opl = start_opl + direct_bins.astype(np.float32) * bin_width
    relay_depths = opl / 2.0

    # Create direct mask (vectorized)
    # For each pixel, mask bins in range [direct_bin - peak_width, direct_bin + peak_width]
    bin_indices = np.arange(T)[np.newaxis, np.newaxis, :]  # (1, 1, T)
    direct_bins_expanded = direct_bins[:, :, np.newaxis]  # (H, W, 1)

    direct_mask = (bin_indices >= direct_bins_expanded - peak_width) & \
                  (bin_indices <= direct_bins_expanded + peak_width)  # (H, W, T)

    # Create indirect transient (remove direct peaks)
    indirect = transient.copy()
    indirect[direct_mask] = 0

    print(f"  Relay depth range: [{relay_depths.min():.3f}, {relay_depths.max():.3f}] m")
    print(f"  Direct light fraction: {direct_mask.sum() / direct_mask.size:.2%}")

    return relay_depths, direct_mask, indirect


def phasor_convolution(transient, bin_width, wavelength, cycles=5):
    """
    Convolve transient with virtual wavelength to create phasor representation.

    This creates a complex wavefield where the phase encodes time-of-flight.
    The phasor field method uses: P = H * [cos(ωt) + i·sin(ωt)]

    Args:
        transient: (H, W, T) indirect transient data
        bin_width: OPL per bin (meters)
        wavelength: Virtual wavelength (meters)
        cycles: Number of wave cycles in kernel

    Returns:
        phasor_cos: (H, W, T) cosine component (real part)
        phasor_sin: (H, W, T) sine component (imaginary part)
    """
    # Create wavelet kernel
    kernel_duration = cycles * wavelength  # Total OPL span
    kernel_bins = max(int(kernel_duration / bin_width), 3)
    t_kernel = np.arange(kernel_bins) * bin_width

    # Angular frequency: omega = 2*pi / wavelength (in OPL space)
    omega = 2.0 * np.pi / wavelength

    kernel_cos = np.cos(omega * t_kernel).astype(np.float32)
    kernel_sin = np.sin(omega * t_kernel).astype(np.float32)

    # Normalize kernels
    norm = np.sqrt(np.sum(kernel_cos ** 2))
    if norm > 0:
        kernel_cos /= norm
        kernel_sin /= norm

    print(f"  Kernel size: {kernel_bins} bins")

    # Convolve along temporal axis (vectorized over spatial dims)
    phasor_cos = scipy.ndimage.convolve1d(
        transient, kernel_cos, axis=2, mode='constant'
    ).astype(np.float32)
    phasor_sin = scipy.ndimage.convolve1d(
        transient, kernel_sin, axis=2, mode='constant'
    ).astype(np.float32)

    return phasor_cos, phasor_sin


def phasor_backproject_drjit(phasor_cos, phasor_sin, relay_pos,
                              volume_min, volume_max, voxel_res,
                              start_opl, bin_width, wavelength,
                              camera_pos=None):
    """
    Phasor field backprojection using DrJit for GPU acceleration.

    Implements Rayleigh-Sommerfeld diffraction via direct summation:
    U(voxel) = sum_relay [ phasor(relay, t) * exp(i*k*r) / r ]

    where k = 2*pi/wavelength and r = distance(relay, voxel)

    Args:
        phasor_cos, phasor_sin: (H, W, T) phasor components
        relay_pos: (H, W, 3) relay wall 3D positions
        volume_min/max: (3,) reconstruction volume bounds
        voxel_res: Number of voxels per dimension
        start_opl, bin_width: Temporal parameters
        wavelength: Virtual wavelength (meters)
        camera_pos: (3,) camera position, defaults to origin

    Returns:
        volume: (Vx, Vy, Vz) reconstructed intensity
    """
    H, W, T = phasor_cos.shape
    N_relay = H * W

    if camera_pos is None:
        camera_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    else:
        camera_pos = np.array(camera_pos, dtype=np.float32)

    # Flatten data for DrJit
    relay_flat = relay_pos.reshape(N_relay, 3).astype(np.float32)
    phasor_cos_flat = phasor_cos.reshape(N_relay, T).astype(np.float32)
    phasor_sin_flat = phasor_sin.reshape(N_relay, T).astype(np.float32)

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

    # Wave number (as Python float for DrJit compatibility)
    k = float(2.0 * np.pi / wavelength)

    print(f"  N_relay: {N_relay}, N_voxels: {N_voxels}")
    print(f"  Uploading to GPU...")

    # Convert to DrJit arrays (on GPU)
    relay_x = mi.Float(relay_flat[:, 0])
    relay_y = mi.Float(relay_flat[:, 1])
    relay_z = mi.Float(relay_flat[:, 2])
    d_cam_relay_dr = mi.Float(d_cam_relay)

    # Store phasors as DrJit TensorXf for efficient indexing
    phasor_cos_tensor = mi.TensorXf(phasor_cos_flat)
    phasor_sin_tensor = mi.TensorXf(phasor_sin_flat)

    # Process all voxels in parallel
    voxel_x = mi.Float(voxels_flat[:, 0])
    voxel_y = mi.Float(voxels_flat[:, 1])
    voxel_z = mi.Float(voxels_flat[:, 2])

    # Initialize accumulators
    vol_real = dr.zeros(mi.Float, N_voxels)
    vol_imag = dr.zeros(mi.Float, N_voxels)

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

        # Clamp indices for safe gathering
        idx0 = dr.clip(bin_idx, 0, T - 2)
        idx1 = idx0 + 1

        # Linear index into flattened phasor array: r_idx * T + bin
        lin_idx0 = mi.UInt32(r_idx * T) + mi.UInt32(idx0)
        lin_idx1 = mi.UInt32(r_idx * T) + mi.UInt32(idx1)

        # Gather phasor values
        cos0 = dr.gather(mi.Float, phasor_cos_tensor.array, lin_idx0)
        cos1 = dr.gather(mi.Float, phasor_cos_tensor.array, lin_idx1)
        sin0 = dr.gather(mi.Float, phasor_sin_tensor.array, lin_idx0)
        sin1 = dr.gather(mi.Float, phasor_sin_tensor.array, lin_idx1)

        # Linear interpolation
        phasor_r = cos0 * (1.0 - frac) + cos1 * frac
        phasor_i = sin0 * (1.0 - frac) + sin1 * frac

        # Rayleigh-Sommerfeld: multiply by exp(i*k*r) / r
        # exp(i*k*r) = cos(k*r) + i*sin(k*r)
        phase = k * d_relay_voxel
        rs_cos = dr.cos(phase)
        rs_sin = dr.sin(phase)

        # Inverse distance weighting (1/r falloff)
        inv_r = dr.rcp(dr.maximum(d_relay_voxel, 0.001))

        # Complex multiplication: (phasor_r + i*phasor_i) * (rs_cos + i*rs_sin) * inv_r
        contrib_real = (phasor_r * rs_cos - phasor_i * rs_sin) * inv_r
        contrib_imag = (phasor_r * rs_sin + phasor_i * rs_cos) * inv_r

        # Accumulate with valid mask
        vol_real += dr.select(valid, contrib_real, 0.0)
        vol_imag += dr.select(valid, contrib_imag, 0.0)

        # Periodically evaluate to avoid graph explosion
        if (r_idx + 1) % 100 == 0:
            dr.eval(vol_real, vol_imag)
            print(f"    Progress: {r_idx + 1}/{N_relay} relay points", end='\r')

    print(f"    Progress: {N_relay}/{N_relay} relay points - Done!")

    # Compute magnitude
    volume_flat = dr.sqrt(vol_real * vol_real + vol_imag * vol_imag)
    dr.eval(volume_flat)

    # Transfer back to numpy
    volume = np.array(volume_flat).reshape(Vx, Vy, Vz)

    return volume


def visualize_orthographic(volume, volume_min, volume_max, output_dir, output_name):
    """
    Create front, side, top orthographic projections using max intensity projection.

    Args:
        volume: (Vx, Vy, Vz) 3D reconstruction
        volume_min/max: Physical bounds
        output_dir: Output directory
        output_name: Output filename prefix
    """
    os.makedirs(output_dir, exist_ok=True)

    # Compute all projections first
    # Front view (XY plane, max over Z - axis 2)
    front = np.max(volume, axis=2)
    # Side view (ZY plane, max over X - axis 0)
    side = np.max(volume, axis=0)
    # Top view (XZ plane, max over Y - axis 1)
    top = np.max(volume, axis=1)

    # Find global min/max across all projections for consistent normalization
    all_projs = [front, side, top]
    global_min = min(p.min() for p in all_projs)
    global_max = max(p.max() for p in all_projs)

    # Combined 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(
        front.T, origin='lower', cmap='hot',
        extent=[volume_min[0], volume_max[0], volume_min[1], volume_max[1]],
        aspect='equal', vmin=global_min, vmax=global_max
    )
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')
    axes[0].set_title('Front View (XY)')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(
        side, origin='lower', cmap='hot',
        extent=[volume_min[2], volume_max[2], volume_min[1], volume_max[1]],
        aspect='equal', vmin=global_min, vmax=global_max
    )
    axes[1].set_xlabel('Z (m)')
    axes[1].set_ylabel('Y (m)')
    axes[1].set_title('Side View (ZY)')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(
        top.T, origin='lower', cmap='hot',
        extent=[volume_min[0], volume_max[0], volume_min[2], volume_max[2]],
        aspect='equal', vmin=global_min, vmax=global_max
    )
    axes[2].set_xlabel('X (m)')
    axes[2].set_ylabel('Z (m)')
    axes[2].set_title('Top View (XZ)')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    combined_path = os.path.join(output_dir, f'{output_name}_orthographic.png')
    plt.savefig(combined_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {combined_path}")

    # Individual high-res projections (also with consistent normalization)
    # Note: front and top need .T, but side does not (different axis ordering)
    projections = [
        ('front', front.T, [volume_min[0], volume_max[0], volume_min[1], volume_max[1]], 'X (m)', 'Y (m)'),
        ('side', side, [volume_min[2], volume_max[2], volume_min[1], volume_max[1]], 'Z (m)', 'Y (m)'),
        ('top', top.T, [volume_min[0], volume_max[0], volume_min[2], volume_max[2]], 'X (m)', 'Z (m)')
    ]

    for name, proj, extent, xlabel, ylabel in projections:
        plt.figure(figsize=(8, 8))
        plt.imshow(proj, origin='lower', cmap='hot', extent=extent, aspect='equal',
                   vmin=global_min, vmax=global_max)
        plt.colorbar(label='Intensity')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f'{name.capitalize()} View (Max Intensity Projection)')

        save_path = os.path.join(output_dir, f'{output_name}_{name}.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")


def save_results(volume, relay_depths, output_dir, output_name):
    """Save reconstruction results to numpy files."""
    os.makedirs(output_dir, exist_ok=True)

    volume_path = os.path.join(output_dir, f'{output_name}_volume.npy')
    np.save(volume_path, volume)
    print(f"  Saved volume: {volume_path}")

    depths_path = os.path.join(output_dir, f'{output_name}_relay_depths.npy')
    np.save(depths_path, relay_depths)
    print(f"  Saved relay depths: {depths_path}")


def visualize_depths(relay_depths, transient, direct_mask, start_opl, bin_width,
                     output_dir, output_name):
    """
    Visualize detected relay wall depths and direct peak detection.

    Args:
        relay_depths: (H, W) detected depths per pixel
        transient: (H, W, T) original transient data
        direct_mask: (H, W, T) mask of direct light bins
        start_opl: Start optical path length
        bin_width: Bin width in OPL
        output_dir: Output directory
        output_name: Output filename prefix
    """
    os.makedirs(output_dir, exist_ok=True)
    H, W, T = transient.shape

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Relay depths image
    im0 = axes[0, 0].imshow(relay_depths, cmap='viridis', origin='lower')
    axes[0, 0].set_title('Detected Relay Wall Depths')
    axes[0, 0].set_xlabel('Pixel X')
    axes[0, 0].set_ylabel('Pixel Y')
    plt.colorbar(im0, ax=axes[0, 0], label='Depth (m)')

    # 2. Depth histogram
    axes[0, 1].hist(relay_depths.flatten(), bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Depth (m)')
    axes[0, 1].set_ylabel('Pixel Count')
    axes[0, 1].set_title('Depth Distribution')
    axes[0, 1].axvline(relay_depths.mean(), color='r', linestyle='--',
                       label=f'Mean: {relay_depths.mean():.3f}m')
    axes[0, 1].legend()

    # 3. Direct peak bin indices
    direct_bins = (relay_depths * 2 - start_opl) / bin_width
    im2 = axes[0, 2].imshow(direct_bins, cmap='plasma', origin='lower')
    axes[0, 2].set_title('Direct Peak Bin Index')
    axes[0, 2].set_xlabel('Pixel X')
    axes[0, 2].set_ylabel('Pixel Y')
    plt.colorbar(im2, ax=axes[0, 2], label='Bin Index')

    # 4. Example transient traces at center pixels
    center_h, center_w = H // 2, W // 2
    time_axis = start_opl + np.arange(T) * bin_width

    # Plot a few example traces
    offsets = [(0, 0), (-H//4, 0), (H//4, 0), (0, -W//4), (0, W//4)]
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    for (dh, dw), color in zip(offsets, colors):
        h_idx = center_h + dh
        w_idx = center_w + dw
        if 0 <= h_idx < H and 0 <= w_idx < W:
            trace = transient[h_idx, w_idx, :]
            depth = relay_depths[h_idx, w_idx]
            axes[1, 0].plot(time_axis, trace, color=color, alpha=0.7,
                           label=f'({h_idx},{w_idx}), d={depth:.2f}m')
            # Mark direct peak location
            peak_opl = depth * 2
            axes[1, 0].axvline(peak_opl, color=color, linestyle='--', alpha=0.3)

    axes[1, 0].set_xlabel('OPL (m)')
    axes[1, 0].set_ylabel('Intensity')
    axes[1, 0].set_title('Example Transient Traces (dashed = detected peak)')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].set_xlim([time_axis[0], time_axis[-1]])

    # 5. Sum of transient over spatial dims (temporal profile)
    temporal_sum = transient.sum(axis=(0, 1))
    axes[1, 1].plot(time_axis, temporal_sum, 'b-')
    axes[1, 1].set_xlabel('OPL (m)')
    axes[1, 1].set_ylabel('Total Intensity')
    axes[1, 1].set_title('Spatially-Summed Transient')
    axes[1, 1].axvline(relay_depths.mean() * 2, color='r', linestyle='--',
                       label=f'Mean direct OPL: {relay_depths.mean()*2:.2f}m')
    axes[1, 1].legend()

    # 6. Direct vs indirect energy
    direct_energy = transient[direct_mask].sum()
    total_energy = transient.sum()
    indirect_energy = total_energy - direct_energy

    axes[1, 2].bar(['Direct', 'Indirect', 'Total'],
                   [direct_energy, indirect_energy, total_energy],
                   color=['orange', 'blue', 'green'])
    axes[1, 2].set_ylabel('Total Energy')
    axes[1, 2].set_title('Energy Distribution')
    for i, v in enumerate([direct_energy, indirect_energy, total_energy]):
        axes[1, 2].text(i, v * 1.02, f'{v:.2e}', ha='center', fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{output_name}_depths_debug.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved depth visualization: {save_path}")


def visualize_phasors(phasor_cos, phasor_sin, start_opl, bin_width, wavelength,
                      output_dir, output_name):
    """
    Visualize phasor field components (magnitude and phase).

    Args:
        phasor_cos: (H, W, T) cosine (real) component
        phasor_sin: (H, W, T) sine (imaginary) component
        start_opl: Start optical path length
        bin_width: Bin width in OPL
        wavelength: Virtual wavelength used
        output_dir: Output directory
        output_name: Output filename prefix
    """
    os.makedirs(output_dir, exist_ok=True)
    H, W, T = phasor_cos.shape

    # Compute magnitude and phase
    phasor_mag = np.sqrt(phasor_cos**2 + phasor_sin**2)
    phasor_phase = np.arctan2(phasor_sin, phasor_cos)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    time_axis = start_opl + np.arange(T) * bin_width

    # 1. Max magnitude over time (spatial distribution)
    mag_max = phasor_mag.max(axis=2)
    im0 = axes[0, 0].imshow(mag_max, cmap='hot', origin='lower')
    axes[0, 0].set_title('Phasor Magnitude (max over time)')
    axes[0, 0].set_xlabel('Pixel X')
    axes[0, 0].set_ylabel('Pixel Y')
    plt.colorbar(im0, ax=axes[0, 0], label='Magnitude')

    # 2. Argmax of magnitude (when is phasor strongest)
    mag_argmax = phasor_mag.argmax(axis=2)
    mag_argmax_opl = start_opl + mag_argmax * bin_width
    im1 = axes[0, 1].imshow(mag_argmax_opl, cmap='viridis', origin='lower')
    axes[0, 1].set_title('OPL of Peak Phasor Magnitude')
    axes[0, 1].set_xlabel('Pixel X')
    axes[0, 1].set_ylabel('Pixel Y')
    plt.colorbar(im1, ax=axes[0, 1], label='OPL (m)')

    # 3. Phase at peak magnitude
    phase_at_peak = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            phase_at_peak[i, j] = phasor_phase[i, j, mag_argmax[i, j]]
    im2 = axes[0, 2].imshow(phase_at_peak, cmap='twilight', origin='lower',
                            vmin=-np.pi, vmax=np.pi)
    axes[0, 2].set_title('Phase at Peak Magnitude')
    axes[0, 2].set_xlabel('Pixel X')
    axes[0, 2].set_ylabel('Pixel Y')
    plt.colorbar(im2, ax=axes[0, 2], label='Phase (rad)')

    # 4. Example phasor traces at center
    center_h, center_w = H // 2, W // 2
    offsets = [(0, 0), (-H//4, 0), (H//4, 0)]
    colors = ['blue', 'green', 'red']

    for (dh, dw), color in zip(offsets, colors):
        h_idx = center_h + dh
        w_idx = center_w + dw
        if 0 <= h_idx < H and 0 <= w_idx < W:
            mag_trace = phasor_mag[h_idx, w_idx, :]
            axes[1, 0].plot(time_axis, mag_trace, color=color, alpha=0.7,
                           label=f'({h_idx},{w_idx})')

    axes[1, 0].set_xlabel('OPL (m)')
    axes[1, 0].set_ylabel('Magnitude')
    axes[1, 0].set_title('Phasor Magnitude Traces')
    axes[1, 0].legend(fontsize=8)

    # 5. Cos and Sin components at center pixel
    cos_trace = phasor_cos[center_h, center_w, :]
    sin_trace = phasor_sin[center_h, center_w, :]
    axes[1, 1].plot(time_axis, cos_trace, 'b-', label='Cos (real)', alpha=0.7)
    axes[1, 1].plot(time_axis, sin_trace, 'r-', label='Sin (imag)', alpha=0.7)
    axes[1, 1].set_xlabel('OPL (m)')
    axes[1, 1].set_ylabel('Amplitude')
    axes[1, 1].set_title(f'Phasor Components at Center ({center_h},{center_w})')
    axes[1, 1].legend()

    # 6. Spatially summed phasor magnitude
    mag_sum = phasor_mag.sum(axis=(0, 1))
    axes[1, 2].plot(time_axis, mag_sum, 'purple')
    axes[1, 2].set_xlabel('OPL (m)')
    axes[1, 2].set_ylabel('Total Magnitude')
    axes[1, 2].set_title('Spatially-Summed Phasor Magnitude')

    # Add wavelength info
    fig.suptitle(f'Phasor Field Analysis (wavelength = {wavelength} m)', fontsize=12)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{output_name}_phasors_debug.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved phasor visualization: {save_path}")


def compute_default_volume_bounds(relay_pos, camera_origin):
    """
    Compute default volume bounds based on relay wall positions.

    The hidden scene is typically behind the relay wall (positive Z in wall-space).
    We extend the volume from the relay wall position outward.

    Args:
        relay_pos: (H, W, 3) relay wall positions
        camera_origin: (3,) camera position

    Returns:
        volume_min: (3,) minimum bounds
        volume_max: (3,) maximum bounds
    """
    # Get relay wall bounding box
    relay_min = relay_pos.min(axis=(0, 1))
    relay_max = relay_pos.max(axis=(0, 1))
    relay_center = (relay_min + relay_max) / 2

    # Estimate wall size
    wall_extent = relay_max - relay_min
    wall_size = max(wall_extent[0], wall_extent[1])

    # For NLOS, hidden scene is usually behind the wall (opposite camera direction)
    # Determine which direction is "behind" based on camera position
    cam_to_wall = relay_center - camera_origin
    wall_normal = cam_to_wall / (np.linalg.norm(cam_to_wall) + 1e-8)

    # Volume extends in the direction opposite to camera
    # For typical setup: camera at origin, wall in front, hidden scene behind wall
    # We create a volume that spans the hidden region

    # Default: extend 2x wall size behind the wall
    volume_depth = wall_size * 2.0

    # X-Y bounds match relay wall with some margin
    margin = wall_size * 0.2
    volume_min = np.array([
        relay_min[0] - margin,
        relay_min[1] - margin,
        relay_center[2] + volume_depth * 0.1  # Start just behind wall
    ], dtype=np.float32)

    volume_max = np.array([
        relay_max[0] + margin,
        relay_max[1] + margin,
        relay_center[2] + volume_depth  # Extend behind wall
    ], dtype=np.float32)

    return volume_min, volume_max


def main():
    args = parse_args()

    print("=" * 60)
    print("Phasor Field Backprojection for NLOS Reconstruction")
    print("=" * 60)

    # 1. Load scene parameters from XML
    print("\n[1/7] Loading scene parameters...")
    scene_params = load_scene_params(args.scene_file)

    # Use command-line overrides if provided, otherwise use scene values
    start_opl = args.start_opl if args.start_opl is not None else scene_params['start_opl']
    bin_width = args.bin_width if args.bin_width is not None else scene_params['bin_width']

    print(f"  Using start_opl: {start_opl} m")
    print(f"  Using bin_width: {bin_width} m")

    # 2. Load transient data
    print("\n[2/7] Loading transient data...")
    transient = load_transient(args.transient_file)
    print(f"  Final shape: {transient.shape}")

    # 3. Detect direct peaks and extract relay geometry
    print("\n[3/7] Detecting direct peaks...")
    relay_depths, direct_mask, indirect = detect_direct_peaks(
        transient, start_opl, bin_width
    )

    # Determine output name early for visualizations
    output_name = args.output_name or os.path.splitext(os.path.basename(args.transient_file))[0].replace('_transient', '_reconstruction')

    # Visualize depths for debugging
    print("\n  Generating depth visualizations...")
    visualize_depths(relay_depths, transient, direct_mask, start_opl, bin_width,
                     args.output_dir, output_name)

    # 4. Compute relay wall 3D positions by unprojecting camera rays
    print("\n[4/7] Computing relay wall positions (unprojecting camera rays)...")
    relay_pos = compute_relay_positions_from_camera(
        relay_depths,
        scene_params['sensor'],
        scene_params['camera_origin']
    )
    print(f"  Relay position range: X=[{relay_pos[...,0].min():.2f}, {relay_pos[...,0].max():.2f}]")
    print(f"                        Y=[{relay_pos[...,1].min():.2f}, {relay_pos[...,1].max():.2f}]")
    print(f"                        Z=[{relay_pos[...,2].min():.2f}, {relay_pos[...,2].max():.2f}]")

    # 5. Determine volume bounds
    print("\n[5/7] Setting up reconstruction volume...")
    camera_origin = scene_params['camera_origin']

    if args.volume_min is not None and args.volume_max is not None:
        volume_min = np.array(args.volume_min, dtype=np.float32)
        volume_max = np.array(args.volume_max, dtype=np.float32)
        print(f"  Using user-specified volume bounds")
    else:
        volume_min, volume_max = compute_default_volume_bounds(relay_pos, camera_origin)
        print(f"  Auto-computed volume bounds from relay wall")

    print(f"  Volume min: [{volume_min[0]:.2f}, {volume_min[1]:.2f}, {volume_min[2]:.2f}]")
    print(f"  Volume max: [{volume_max[0]:.2f}, {volume_max[1]:.2f}, {volume_max[2]:.2f}]")

    # 6. Phasor field convolution
    print(f"\n[6/7] Computing phasor field (wavelength={args.wavelength}m)...")
    phasor_cos, phasor_sin = phasor_convolution(
        indirect, bin_width, args.wavelength
    )

    # Visualize phasors for debugging
    print("\n  Generating phasor visualizations...")
    visualize_phasors(phasor_cos, phasor_sin, start_opl, bin_width, args.wavelength,
                      args.output_dir, output_name)

    # 7. Phasor backprojection with DrJit (GPU)
    print(f"\n[7/7] Backprojecting to {args.voxel_resolution}^3 volume (DrJit/GPU)...")

    volume = phasor_backproject_drjit(
        phasor_cos, phasor_sin, relay_pos,
        volume_min, volume_max, args.voxel_resolution,
        start_opl, bin_width, args.wavelength,
        camera_origin
    )
    print(f"  Volume range: [{volume.min():.6f}, {volume.max():.6f}]")

    # 8. Visualize and save
    print(f"\n[8/8] Generating visualizations and saving...")
    visualize_orthographic(volume, volume_min, volume_max, args.output_dir, output_name)
    save_results(volume, relay_depths, args.output_dir, output_name)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
