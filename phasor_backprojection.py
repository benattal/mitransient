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

Usage Examples:
--------------

Basic usage (simple scenes with auto volume bounds):
    python phasor_backprojection.py results/wall_box_transient.npy

Complex scenes (e.g., ourbox) with explicit volume bounds:
    python phasor_backprojection.py results/ourbox_confocal_transient_both.npy \\
        --scene-file scenes/ourbox_confocal.xml \\
        --voxel-resolution 128 \\
        --volume-from-camera \\
        --wavelength 0.05 \\
        --volume-min -1 -1 -1 \\
        --volume-max 1 1 1

Using separate direct/indirect captures (for better geometry recovery):
    python phasor_backprojection.py results/scene_transient.npy \\
        --direct-file results/scene_direct_only.npy \\
        --indirect-file results/scene_indirect_only.npy \\
        --scene-file scenes/my_scene.xml

Key Parameters:
- --voxel-resolution: Higher values (128, 256) give finer detail but slower
- --wavelength: Virtual wavelength in meters; smaller = sharper but noisier
- --volume-min/max: Manually specify reconstruction bounds (x, y, z) in meters
- --volume-from-camera: Auto-compute bounds from camera frustum (good starting point)
- --no-falloff: Disable 1/r distance falloff (can help with some scenes)

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
    parser.add_argument('--wavelength', type=float, default=0.05,
                        help='Virtual wavelength for phasor field (meters, default: 0.05)')
    parser.add_argument('--no-falloff', action='store_true',
                        help='Disable 1/r distance falloff in backprojection (use inv_r=1.0)')
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
    parser.add_argument('--ortho-views', type=str, nargs='+', default=['front', 'top'],
                        choices=['front', 'side', 'top'],
                        help='Which orthographic views to show (front, side, top). Default: front top')
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
    # Tangent in u direction (along columns)
    tangent_u = np.zeros_like(relay_pos)
    tangent_u[:, 1:-1, :] = relay_pos[:, 2:, :] - relay_pos[:, :-2, :]
    tangent_u[:, 0, :] = relay_pos[:, 1, :] - relay_pos[:, 0, :]
    tangent_u[:, -1, :] = relay_pos[:, -1, :] - relay_pos[:, -2, :]

    # Tangent in v direction (along rows)
    tangent_v = np.zeros_like(relay_pos)
    tangent_v[1:-1, :, :] = relay_pos[2:, :, :] - relay_pos[:-2, :, :]
    tangent_v[0, :, :] = relay_pos[1, :, :] - relay_pos[0, :, :]
    tangent_v[-1, :, :] = relay_pos[-1, :, :] - relay_pos[-2, :, :]

    # Cross product to get normal
    normals = np.cross(tangent_u, tangent_v)

    # Normalize
    norm_length = np.linalg.norm(normals, axis=-1, keepdims=True)
    norm_length = np.maximum(norm_length, 1e-8)  # Avoid division by zero
    normals = normals / norm_length

    # Orient normals to point away from camera (into the hidden scene)
    # Compute vector from relay to camera
    relay_to_camera = camera_origin - relay_pos  # (H, W, 3)

    # Dot product with normal
    dot = np.sum(normals * relay_to_camera, axis=-1)  # (H, W)

    # Flip normals that point toward camera
    flip_mask = dot > 0
    normals[flip_mask] = -normals[flip_mask]

    return -normals.astype(np.float32)


def visualize_relay_normals(relay_pos, normals, camera_origin, output_dir, output_name):
    """
    Visualize computed relay wall normals as an image.

    Args:
        relay_pos: (H, W, 3) relay wall positions
        normals: (H, W, 3) normal vectors
        camera_origin: (3,) camera position
        output_dir: Output directory
        output_name: Output filename prefix
    """
    os.makedirs(output_dir, exist_ok=True)
    H, W, _ = relay_pos.shape

    # Convert normals to RGB visualization (map [-1, 1] to [0, 1])
    normals_vis = (normals + 1.0) / 2.0
    normals_vis = np.clip(normals_vis, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Normal visualization (RGB = XYZ)
    axes[0].imshow(normals_vis, origin='lower')
    axes[0].set_title('Computed Normals (RGB = XYZ)')
    axes[0].set_xlabel('Pixel X')
    axes[0].set_ylabel('Pixel Y')
    axes[0].axis('off')

    # Individual normal components
    axes[1].imshow(normals[:, :, 2], cmap='RdBu', vmin=-1, vmax=1, origin='lower')
    axes[1].set_title('Normal Z Component')
    axes[1].set_xlabel('Pixel X')
    axes[1].set_ylabel('Pixel Y')
    plt.colorbar(axes[1].images[0], ax=axes[1], fraction=0.046)

    # Depth visualization for reference
    depths = np.linalg.norm(relay_pos - camera_origin, axis=-1)
    axes[2].imshow(depths, cmap='viridis', origin='lower')
    axes[2].set_title('Relay Wall Depth')
    axes[2].set_xlabel('Pixel X')
    axes[2].set_ylabel('Pixel Y')
    plt.colorbar(axes[2].images[0], ax=axes[2], fraction=0.046)

    save_path = os.path.join(output_dir, f'{output_name}_relay_normals.png')
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  Saved relay normals visualization: {save_path}")


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
        # Replicate single channel to RGB
        rgb = np.stack([data[..., 0]] * 3, axis=-1).astype(np.float32)

    if return_rgb:
        return luminance.astype(np.float32), rgb
    return luminance.astype(np.float32)


def detect_direct_peaks(transient, start_opl, bin_width, threshold_percentile=95, peak_width=5,
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
                              If None, uses 1% of global max.

    Returns:
        relay_depths: (H, W) array of wall depths (meters)
        direct_mask: (H, W, T) boolean mask of direct light
        indirect: (H, W, T) transient with direct peaks removed
        valid_mask: (H, W) boolean mask of pixels with valid signal
    """
    H, W, T = transient.shape

    # Compute per-pixel max values for local thresholding
    pixel_max = np.max(transient, axis=-1)  # (H, W)

    # Valid pixels have sufficient signal strength
    valid_mask = pixel_max > 0.0  # (H, W)

    # Per-pixel threshold: fraction of local max
    local_threshold = pixel_max * 0.5  # (H, W)

    # Find first bin above threshold for each pixel (vectorized)
    above_threshold = transient > local_threshold[:, :, np.newaxis]

    # For pixels with signal, find first bin above threshold
    first_above = np.argmax(above_threshold, axis=2)  # (H, W)
    direct_bins = first_above

    # Compute relay depths from bin indices
    opl = start_opl + direct_bins.astype(np.float32) * bin_width
    relay_depths = opl / 2.0

    # Create direct mask (vectorized)
    bin_indices = np.arange(T)[np.newaxis, np.newaxis, :]  # (1, 1, T)
    direct_bins_expanded = direct_bins[:, :, np.newaxis]  # (H, W, 1)

    direct_mask = (bin_indices >= direct_bins_expanded - peak_width) & \
                  (bin_indices <= direct_bins_expanded + peak_width)  # (H, W, T)
    # direct_mask = (bin_indices <= direct_bins_expanded)

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
                              camera_pos=None, no_falloff=False,
                              bin_threshold=0.0, min_relay_distance=0.0,
                              indirect_transient=None, relay_normals=None):
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
        no_falloff: If True, disable 1/r distance falloff (inv_r = 1.0)
        bin_threshold: Only splat to voxels whose corresponding time bin has
                       indirect signal above this fraction of the relay point's
                       max signal (0.0-1.0). Requires indirect_transient.
        min_relay_distance: Minimum distance from relay point to voxel (meters).
                            Voxels closer than this distance are skipped.
        indirect_transient: (H, W, T) raw indirect transient for bin thresholding.
                            Required if bin_threshold > 0.
        relay_normals: (H, W, 3) unit normal vectors at each relay point.
                       If provided, only voxels in the positive hemisphere
                       (dot(voxel - relay, normal) > 0) receive contributions.

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

    # Upload normals if provided
    use_hemisphere_filter = relay_normals is not None
    if use_hemisphere_filter:
        normals_flat = relay_normals.reshape(N_relay, 3).astype(np.float32)
        normal_x = mi.Float(normals_flat[:, 0])
        normal_y = mi.Float(normals_flat[:, 1])
        normal_z = mi.Float(normals_flat[:, 2])
        print(f"  Hemisphere filtering: enabled (only positive normal hemisphere)")

    # Compute per-relay-point bin thresholds if requested
    use_bin_threshold = bin_threshold > 0.0 and indirect_transient is not None
    if use_bin_threshold:
        # Compute per-relay-point max values
        indirect_flat = indirect_transient.reshape(N_relay, T).astype(np.float32)
        relay_max = indirect_flat.max(axis=1, keepdims=True)  # (N_relay, 1)
        # Compute absolute threshold per relay point
        relay_thresholds = (bin_threshold * relay_max).flatten()  # (N_relay,)
        # Upload indirect transient and thresholds to GPU
        indirect_tensor = mi.TensorXf(indirect_flat)
        relay_thresholds_dr = mi.Float(relay_thresholds)
        print(f"  Bin threshold: {bin_threshold:.2%} of each relay point's max")
        # Report how many bins pass threshold overall
        valid_bins = (indirect_flat > relay_thresholds[:, np.newaxis]).sum()
        total_bins = N_relay * T
        print(f"    Bins above threshold: {valid_bins}/{total_bins} ({100*valid_bins/total_bins:.1f}%)")
    elif bin_threshold > 0.0 and indirect_transient is None:
        print(f"  WARNING: bin_threshold specified but indirect_transient not provided, ignoring")
        use_bin_threshold = False

    if min_relay_distance > 0.0:
        print(f"  Min relay distance: {min_relay_distance:.4f} m")

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

        # Apply minimum distance threshold: skip voxels too close to relay
        if min_relay_distance > 0.0:
            valid = valid & (d_relay_voxel > min_relay_distance)

        # Apply hemisphere filter: only voxels in positive normal hemisphere
        if use_hemisphere_filter:
            # Get normal for this relay point
            rnx = dr.gather(mi.Float, normal_x, mi.UInt32(r_idx))
            rny = dr.gather(mi.Float, normal_y, mi.UInt32(r_idx))
            rnz = dr.gather(mi.Float, normal_z, mi.UInt32(r_idx))
            # Dot product of (voxel - relay) with normal
            # dx, dy, dz already computed above as voxel - relay
            dot_normal = dx * rnx + dy * rny + dz * rnz
            valid = valid & (dot_normal > 0.0)

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

        # Apply bin threshold: skip contributions where indirect signal is too weak
        if use_bin_threshold:
            # Get the indirect signal value at this bin for this relay point
            indirect_val0 = dr.gather(mi.Float, indirect_tensor.array, lin_idx0)
            indirect_val1 = dr.gather(mi.Float, indirect_tensor.array, lin_idx1)
            indirect_val = indirect_val0 * (1.0 - frac) + indirect_val1 * frac
            # Get the threshold for this relay point
            relay_thresh = dr.gather(mi.Float, relay_thresholds_dr, mi.UInt32(r_idx))
            valid = valid & (indirect_val > relay_thresh)

        # Rayleigh-Sommerfeld: multiply by exp(i*k*r) / r
        # exp(i*k*r) = cos(k*r) + i*sin(k*r)
        phase = k * d_relay_voxel
        rs_cos = dr.cos(phase)
        rs_sin = dr.sin(phase)

        # Inverse distance weighting (1/r falloff) or no falloff
        if no_falloff:
            inv_r = 1.0
        else:
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
                           vis_transform='none', vis_scale=1.0, vis_bias=0.0,
                           views=None):
    """
    Create front, side, top orthographic projections using max intensity projection.

    Args:
        volume: (Vx, Vy, Vz) 3D reconstruction
        volume_min/max: Physical bounds
        output_dir: Output directory
        output_name: Output filename prefix
        vis_transform: Non-linear transform for visualization
        vis_scale: Scale factor before transform
        vis_bias: Bias before transform
        views: List of views to show ('front', 'side', 'top'). Default: ['front', 'top']
    """
    if views is None:
        views = ['front', 'top']

    os.makedirs(output_dir, exist_ok=True)

    # Compute all projections first
    # Front view (XY plane, max over Z - axis 2)
    front = np.max(volume, axis=2)
    # Side view (ZY plane, max over X - axis 0)
    side = np.max(volume, axis=0)
    # Top view (XZ plane, max over Y - axis 1)
    top = np.max(volume, axis=1)

    # Apply visualization transform
    front, label = apply_vis_transform(front, vis_transform, vis_scale, vis_bias)
    side, _ = apply_vis_transform(side, vis_transform, vis_scale, vis_bias)
    top, _ = apply_vis_transform(top, vis_transform, vis_scale, vis_bias)

    # Find global min/max across all projections for consistent normalization
    all_projs = [front, side, top]
    global_min = min(p.min() for p in all_projs)
    global_max = max(p.max() for p in all_projs)

    # Build title suffix based on transform
    title_suffix = f' ({vis_transform})' if vis_transform != 'none' else ''

    # Build list of views to display
    n_views = len(views)
    fig, axes = plt.subplots(1, n_views, figsize=(5 * n_views, 5))
    if n_views == 1:
        axes = [axes]

    ax_idx = 0
    if 'front' in views:
        im = axes[ax_idx].imshow(
            front.T, origin='lower', cmap='hot',
            extent=[volume_min[0], volume_max[0], volume_min[1], volume_max[1]],
            aspect='equal', vmin=global_min, vmax=global_max
        )
        axes[ax_idx].set_xlabel('X (m)')
        axes[ax_idx].set_ylabel('Y (m)')
        axes[ax_idx].set_title(f'Front View (XY){title_suffix}')
        plt.colorbar(im, ax=axes[ax_idx], fraction=0.046, pad=0.04, label=label)
        ax_idx += 1

    if 'side' in views:
        im = axes[ax_idx].imshow(
            side, origin='lower', cmap='hot',
            extent=[volume_min[2], volume_max[2], volume_min[1], volume_max[1]],
            aspect='equal', vmin=global_min, vmax=global_max
        )
        axes[ax_idx].set_xlabel('Z (m)')
        axes[ax_idx].set_ylabel('Y (m)')
        axes[ax_idx].set_title(f'Side View (ZY){title_suffix}')
        plt.colorbar(im, ax=axes[ax_idx], fraction=0.046, pad=0.04, label=label)
        ax_idx += 1

    if 'top' in views:
        im = axes[ax_idx].imshow(
            top.T, origin='lower', cmap='hot',
            extent=[volume_min[0], volume_max[0], volume_min[2], volume_max[2]],
            aspect='equal', vmin=global_min, vmax=global_max
        )
        axes[ax_idx].set_xlabel('X (m)')
        axes[ax_idx].set_ylabel('Z (m)')
        axes[ax_idx].set_title(f'Top View (XZ){title_suffix}')
        plt.colorbar(im, ax=axes[ax_idx], fraction=0.046, pad=0.04, label=label)

    plt.tight_layout()
    combined_path = os.path.join(output_dir, f'{output_name}_orthographic.png')
    plt.savefig(combined_path, dpi=300)
    plt.close()
    print(f"  Saved: {combined_path}")


def visualize_orthographic_color(volume_rgb, volume_min, volume_max, output_dir, output_name,
                                  views=None):
    """
    Create color orthographic projections (R, G, B channels) with global normalization.

    Args:
        volume_rgb: (Vx, Vy, Vz, 3) 3D RGB reconstruction
        volume_min/max: Physical bounds
        output_dir: Output directory
        output_name: Output filename prefix
        views: List of views to show ('front', 'side', 'top'). Default: ['front', 'top']
    """
    if views is None:
        views = ['front', 'top']

    os.makedirs(output_dir, exist_ok=True)

    # Compute all projections for each channel
    # Front view (XY plane, max over Z - axis 2)
    front_rgb = np.max(volume_rgb, axis=2)  # (Vx, Vy, 3)
    # Side view (ZY plane, max over X - axis 0)
    side_rgb = np.max(volume_rgb, axis=0)   # (Vy, Vz, 3)
    # Top view (XZ plane, max over Y - axis 1)
    top_rgb = np.max(volume_rgb, axis=1)    # (Vx, Vz, 3)

    # Global normalization across all projections and channels
    all_projs = [front_rgb, side_rgb, top_rgb]
    global_max = max(p.max() for p in all_projs)

    if global_max > 0:
        front_rgb = front_rgb / global_max
        side_rgb = side_rgb / global_max
        top_rgb = top_rgb / global_max

    # Clip to [0, 1] range
    front_rgb = np.clip(front_rgb, 0, 1)
    side_rgb = np.clip(side_rgb, 0, 1)
    top_rgb = np.clip(top_rgb, 0, 1)

    # Combined n-panel figure
    n_views = len(views)
    fig, axes = plt.subplots(1, n_views, figsize=(5 * n_views, 5))
    if n_views == 1:
        axes = [axes]

    ax_idx = 0
    if 'front' in views:
        axes[ax_idx].imshow(
            np.transpose(front_rgb, (1, 0, 2)), origin='lower',
            extent=[volume_min[0], volume_max[0], volume_min[1], volume_max[1]],
            aspect='equal'
        )
        axes[ax_idx].set_xlabel('X (m)')
        axes[ax_idx].set_ylabel('Y (m)')
        axes[ax_idx].set_title('Front View (XY) - Color')
        ax_idx += 1

    if 'side' in views:
        axes[ax_idx].imshow(
            side_rgb, origin='lower',
            extent=[volume_min[2], volume_max[2], volume_min[1], volume_max[1]],
            aspect='equal'
        )
        axes[ax_idx].set_xlabel('Z (m)')
        axes[ax_idx].set_ylabel('Y (m)')
        axes[ax_idx].set_title('Side View (ZY) - Color')
        ax_idx += 1

    if 'top' in views:
        axes[ax_idx].imshow(
            np.transpose(top_rgb, (1, 0, 2)), origin='lower',
            extent=[volume_min[0], volume_max[0], volume_min[2], volume_max[2]],
            aspect='equal'
        )
        axes[ax_idx].set_xlabel('X (m)')
        axes[ax_idx].set_ylabel('Z (m)')
        axes[ax_idx].set_title('Top View (XZ) - Color')

    plt.tight_layout()
    combined_path = os.path.join(output_dir, f'{output_name}_orthographic_color.png')
    plt.savefig(combined_path, dpi=300)
    plt.close()
    print(f"  Saved: {combined_path}")

    # Individual high-res color projections (only for selected views)
    projections = []
    if 'front' in views:
        projections.append(('front', np.transpose(front_rgb, (1, 0, 2)),
         [volume_min[0], volume_max[0], volume_min[1], volume_max[1]], 'X (m)', 'Y (m)'))
    if 'side' in views:
        projections.append(('side', side_rgb,
         [volume_min[2], volume_max[2], volume_min[1], volume_max[1]], 'Z (m)', 'Y (m)'))
    if 'top' in views:
        projections.append(('top', np.transpose(top_rgb, (1, 0, 2)),
         [volume_min[0], volume_max[0], volume_min[2], volume_max[2]], 'X (m)', 'Z (m)'))

    for name, proj, extent, xlabel, ylabel in projections:
        plt.figure(figsize=(10, 10))
        plt.imshow(proj, origin='lower', extent=extent, aspect='equal')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f'{name.capitalize()} View - Color (Max Intensity Projection)')

        save_path = os.path.join(output_dir, f'{output_name}_{name}_color.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"  Saved: {save_path}")


def visualize_orthographic_with_pointcloud(volume_rgb, relay_pos, relay_colors,
                                            volume_min, volume_max, output_dir, output_name,
                                            point_alpha=0.7, point_size=2, views=None):
    """
    Create color orthographic projections overlaid with the unprojected color point cloud.

    Args:
        volume_rgb: (Vx, Vy, Vz, 3) 3D RGB reconstruction
        relay_pos: (H, W, 3) 3D positions of relay wall points
        relay_colors: (H, W, 3) colors at each relay wall point
        volume_min/max: Physical bounds
        output_dir: Output directory
        output_name: Output filename prefix
        point_alpha: Alpha for point cloud overlay
        point_size: Size of points in scatter plot
        views: List of views to show ('front', 'side', 'top'). Default: ['front', 'top']
    """
    if views is None:
        views = ['front', 'top']

    os.makedirs(output_dir, exist_ok=True)

    H, W = relay_pos.shape[:2]

    # Flatten relay positions and colors
    points_3d = relay_pos.reshape(-1, 3)  # (N, 3)
    colors_flat = relay_colors.reshape(-1, 3)  # (N, 3)

    # Normalize colors for display
    color_max = colors_flat.max()
    if color_max > 0:
        colors_normalized = colors_flat / color_max
    else:
        colors_normalized = colors_flat
    colors_normalized = np.clip(colors_normalized, 0, 1)

    # Compute volume projections (same as visualize_orthographic_color)
    front_rgb = np.max(volume_rgb, axis=2)  # (Vx, Vy, 3)
    side_rgb = np.max(volume_rgb, axis=0)   # (Vy, Vz, 3)
    top_rgb = np.max(volume_rgb, axis=1)    # (Vx, Vz, 3)

    # Global normalization for volume
    all_projs = [front_rgb, side_rgb, top_rgb]
    global_max = max(p.max() for p in all_projs)
    if global_max > 0:
        front_rgb = front_rgb / global_max
        side_rgb = side_rgb / global_max
        top_rgb = top_rgb / global_max

    front_rgb = np.clip(front_rgb, 0, 1)
    side_rgb = np.clip(side_rgb, 0, 1)
    top_rgb = np.clip(top_rgb, 0, 1)

    # Extract point cloud coordinates for each view
    px, py, pz = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]

    # Combined n-panel figure with overlay
    n_views = len(views)
    fig, axes = plt.subplots(1, n_views, figsize=(5 * n_views, 5))
    if n_views == 1:
        axes = [axes]

    ax_idx = 0
    if 'front' in views:
        axes[ax_idx].imshow(
            np.transpose(front_rgb, (1, 0, 2)), origin='lower',
            extent=[volume_min[0], volume_max[0], volume_min[1], volume_max[1]],
            aspect='equal'
        )
        axes[ax_idx].scatter(px, py, c=colors_normalized, s=point_size, alpha=point_alpha, marker='.')
        axes[ax_idx].set_xlim([volume_min[0], volume_max[0]])
        axes[ax_idx].set_ylim([volume_min[1], volume_max[1]])
        axes[ax_idx].set_xlabel('X (m)')
        axes[ax_idx].set_ylabel('Y (m)')
        axes[ax_idx].set_title('Front View (XY) - Reconstruction + Point Cloud')
        ax_idx += 1

    if 'side' in views:
        axes[ax_idx].imshow(
            side_rgb, origin='lower',
            extent=[volume_min[2], volume_max[2], volume_min[1], volume_max[1]],
            aspect='equal'
        )
        axes[ax_idx].scatter(pz, py, c=colors_normalized, s=point_size, alpha=point_alpha, marker='.')
        axes[ax_idx].set_xlim([volume_min[2], volume_max[2]])
        axes[ax_idx].set_ylim([volume_min[1], volume_max[1]])
        axes[ax_idx].set_xlabel('Z (m)')
        axes[ax_idx].set_ylabel('Y (m)')
        axes[ax_idx].set_title('Side View (ZY) - Reconstruction + Point Cloud')
        ax_idx += 1

    if 'top' in views:
        axes[ax_idx].imshow(
            np.transpose(top_rgb, (1, 0, 2)), origin='lower',
            extent=[volume_min[0], volume_max[0], volume_min[2], volume_max[2]],
            aspect='equal'
        )
        axes[ax_idx].scatter(px, pz, c=colors_normalized, s=point_size, alpha=point_alpha, marker='.')
        axes[ax_idx].set_xlim([volume_min[0], volume_max[0]])
        axes[ax_idx].set_ylim([volume_min[2], volume_max[2]])
        axes[ax_idx].set_xlabel('X (m)')
        axes[ax_idx].set_ylabel('Z (m)')
        axes[ax_idx].set_title('Top View (XZ) - Reconstruction + Point Cloud')

    plt.tight_layout()
    combined_path = os.path.join(output_dir, f'{output_name}_orthographic_overlay.png')
    plt.savefig(combined_path, dpi=300)
    plt.close()
    print(f"  Saved: {combined_path}")

    # Individual high-res overlaid projections (only for selected views)
    projections = []
    if 'front' in views:
        projections.append(('front', np.transpose(front_rgb, (1, 0, 2)),
         [volume_min[0], volume_max[0], volume_min[1], volume_max[1]],
         'X (m)', 'Y (m)', px, py))
    if 'side' in views:
        projections.append(('side', side_rgb,
         [volume_min[2], volume_max[2], volume_min[1], volume_max[1]],
         'Z (m)', 'Y (m)', pz, py))
    if 'top' in views:
        projections.append(('top', np.transpose(top_rgb, (1, 0, 2)),
         [volume_min[0], volume_max[0], volume_min[2], volume_max[2]],
         'X (m)', 'Z (m)', px, pz))

    for name, proj, extent, xlabel, ylabel, scatter_x, scatter_y in projections:
        plt.figure(figsize=(10, 10))
        plt.imshow(proj, origin='lower', extent=extent, aspect='equal')
        plt.scatter(scatter_x, scatter_y, c=colors_normalized, s=point_size, alpha=point_alpha, marker='.')
        plt.xlim([extent[0], extent[1]])
        plt.ylim([extent[2], extent[3]])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f'{name.capitalize()} View - Reconstruction + Point Cloud')

        save_path = os.path.join(output_dir, f'{output_name}_{name}_overlay.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"  Saved: {save_path}")


def visualize_direct_vs_indirect(transient_original, indirect, direct_mask,
                                  start_opl, bin_width, output_dir, output_name,
                                  pixel_h=None, pixel_w=None, use_log_scale=False):
    """
    Visualize the original transient vs the filtered indirect transient at a single pixel.

    Args:
        transient_original: (H, W, T) original transient data (with direct peak)
        indirect: (H, W, T) indirect transient data (direct peak removed)
        direct_mask: (H, W, T) boolean mask of direct light bins
        start_opl: Start optical path length
        bin_width: Bin width in OPL
        output_dir: Output directory
        output_name: Output filename prefix
        pixel_h: Row index of pixel to visualize (default: center)
        pixel_w: Column index of pixel to visualize (default: center)
        use_log_scale: If True, use log scale for transient traces
    """
    os.makedirs(output_dir, exist_ok=True)
    H, W, T = transient_original.shape
    time_axis = start_opl + np.arange(T) * bin_width

    # Default to center pixel
    if pixel_h is None:
        pixel_h = H // 2
    if pixel_w is None:
        pixel_w = W // 2

    # Clamp to valid range
    pixel_h = max(0, min(pixel_h, H - 1))
    pixel_w = max(0, min(pixel_w, W - 1))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    eps = 1e-10
    orig_max = np.max(transient_original, axis=2)
    indirect_max = np.max(indirect, axis=2)
    direct_energy = np.sum(transient_original * direct_mask, axis=2)

    # Row 1: Spatial views with selected pixel marked
    # Apply log scale for spatial images if enabled
    im00 = axes[0, 0].imshow(orig_max, cmap='hot', origin='upper')
    plt.colorbar(im00, ax=axes[0, 0])
    axes[0, 0].set_title('Original: Max over time')
    axes[0, 0].scatter([pixel_w], [pixel_h], c='cyan', s=100, marker='x', linewidths=2)
    axes[0, 0].set_xlabel('Pixel X')
    axes[0, 0].set_ylabel('Pixel Y')

    im01 = axes[0, 1].imshow(indirect_max, cmap='hot', origin='upper')
    plt.colorbar(im01, ax=axes[0, 1])
    axes[0, 1].set_title('Indirect: Max over time')
    axes[0, 1].scatter([pixel_w], [pixel_h], c='cyan', s=100, marker='x', linewidths=2)
    axes[0, 1].set_xlabel('Pixel X')
    axes[0, 1].set_ylabel('Pixel Y')

    im02 = axes[0, 2].imshow(direct_energy, cmap='hot', origin='upper')
    plt.colorbar(im02, ax=axes[0, 2])
    axes[0, 2].set_title('Direct Peak Energy')
    axes[0, 2].scatter([pixel_w], [pixel_h], c='cyan', s=100, marker='x', linewidths=2)
    axes[0, 2].set_xlabel('Pixel X')
    axes[0, 2].set_ylabel('Pixel Y')

    # Row 2: Temporal traces at selected pixel
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

    save_path = os.path.join(output_dir, f'{output_name}_direct_vs_indirect.png')
    plt.savefig(save_path, dpi=250)
    plt.close()
    print(f"  Saved direct vs indirect visualization: {save_path}")


def normalize_volume_by_slices(volume, axis='y'):
    """
    Normalize volume intensity per slice along the specified axis.

    Each slice is normalized independently so that its max value is 1.
    This helps visualize details when there's intensity falloff along an axis.

    Args:
        volume: (Vx, Vy, Vz) or (Vx, Vy, Vz, C) 3D volume
        axis: 'x' (axis 0), 'y' (axis 1), or 'z' (axis 2)

    Returns:
        Normalized volume with same shape
    """
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if axis not in axis_map:
        print(f"  WARNING: Invalid axis '{axis}', skipping normalization")
        return volume

    ax = axis_map[axis]
    volume_norm = volume.copy()

    # Handle RGB volumes (4D)
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
        # Grayscale volume (3D)
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

    Args:
        transient_rgb: (H, W, T, 3) RGB transient data
        direct_mask: (H, W, T) boolean mask of direct light bins

    Returns:
        colors: (H, W, 3) color at direct peak for each pixel
    """
    H, W, T, C = transient_rgb.shape

    # Find the bin index of the direct peak for each pixel
    # Use the center of the direct mask region
    direct_bins = np.argmax(direct_mask, axis=2)  # First True in mask

    # For each pixel, sum the color within the direct mask region
    # This gives a more robust color estimate than a single bin
    colors = np.zeros((H, W, 3), dtype=np.float32)

    for c in range(3):
        channel_data = transient_rgb[..., c]  # (H, W, T)
        # Sum within direct mask
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


def visualize_depths(relay_depths, transient, direct_mask, start_opl, bin_width,
                     output_dir, output_name, use_log_scale=False):
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
        use_log_scale: If True, use log scale for transient traces
    """
    os.makedirs(output_dir, exist_ok=True)
    H, W, T = transient.shape
    eps = 1e-10

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Relay depths image
    im0 = axes[0, 0].imshow(relay_depths, cmap='viridis', origin='upper')
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
    im2 = axes[0, 2].imshow(direct_bins, cmap='plasma', origin='upper')
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
            axes[1, 0].plot(time_axis, trace + (eps if use_log_scale else 0), color=color, alpha=0.7,
                           label=f'({h_idx},{w_idx}), d={depth:.2f}m')
            # Mark direct peak location
            peak_opl = depth * 2
            axes[1, 0].axvline(peak_opl, color=color, linestyle='--', alpha=0.3)

    axes[1, 0].set_xlabel('OPL (m)')
    axes[1, 0].set_ylabel('Intensity')
    axes[1, 0].set_title('Example Transient Traces' + (' (log scale)' if use_log_scale else ''))
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].set_xlim([time_axis[0], time_axis[-1]])
    if use_log_scale:
        axes[1, 0].set_yscale('log')

    # 5. Sum of transient over spatial dims (temporal profile)
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

    # 6. Direct vs indirect energy
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

    save_path = os.path.join(output_dir, f'{output_name}_depths_debug.png')
    plt.savefig(save_path, dpi=250)
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

    save_path = os.path.join(output_dir, f'{output_name}_phasors_debug.png')
    plt.savefig(save_path, dpi=250)
    plt.close()
    print(f"  Saved phasor visualization: {save_path}")


def compute_volume_bounds_from_camera(sensor, depth_min, depth_max, margin=0.1):
    """
    Compute volume bounds by backprojecting camera frustum corners.

    Samples rays at the corners and edges of the camera sensor, then
    backprojects them to the specified depth range to determine the
    3D bounding box.

    Args:
        sensor: Mitsuba sensor object
        depth_min: Near depth for frustum (meters)
        depth_max: Far depth for frustum (meters)
        margin: Fractional margin to add around bounds (default: 0.1 = 10%)

    Returns:
        volume_min: (3,) minimum bounds
        volume_max: (3,) maximum bounds
    """
    film = sensor.film()
    film_w, film_h = film.size()

    # Sample points: corners + edge midpoints + center
    # Normalized coordinates [0, 1]
    sample_points = [
        (0.0, 0.0),    # top-left
        (1.0, 0.0),    # top-right
        (0.0, 1.0),    # bottom-left
        (1.0, 1.0),    # bottom-right
        (0.5, 0.0),    # top-center
        (0.5, 1.0),    # bottom-center
        (0.0, 0.5),    # left-center
        (1.0, 0.5),    # right-center
        (0.5, 0.5),    # center
    ]

    all_points = []

    for u, v in sample_points:
        # Sample ray for this pixel
        sample_pos = mi.Point2f(mi.Float(u), mi.Float(v))
        rays, _ = sensor.sample_ray(
            time=mi.Float(0.0),
            sample1=mi.Float(0.0),
            sample2=sample_pos,
            sample3=mi.Point2f(mi.Float(0.5), mi.Float(0.5))
        )

        # Extract ray origin and direction (convert DrJit arrays to numpy, then to scalars)
        ray_o = np.array([np.array(rays.o.x)[0], np.array(rays.o.y)[0], np.array(rays.o.z)[0]])
        ray_d = np.array([np.array(rays.d.x)[0], np.array(rays.d.y)[0], np.array(rays.d.z)[0]])

        # Normalize ray direction to ensure unit length
        ray_d = ray_d / np.linalg.norm(ray_d)

        # Backproject to near and far depths
        point_near = ray_o + depth_min * ray_d
        point_far = ray_o + depth_max * ray_d

        all_points.append(point_near)
        all_points.append(point_far)

    all_points = np.array(all_points)

    # Compute bounding box
    volume_min = all_points.min(axis=0)
    volume_max = all_points.max(axis=0)

    # Add margin
    extent = volume_max - volume_min
    volume_min = volume_min - margin * extent
    volume_max = volume_max + margin * extent

    return volume_min.astype(np.float32), volume_max.astype(np.float32)


def compute_default_volume_bounds(relay_pos, camera_origin, valid_mask=None):
    """
    Compute default volume bounds based on relay wall positions.

    The hidden scene is typically behind the relay wall (positive Z in wall-space).
    We extend the volume from the relay wall position outward.

    Args:
        relay_pos: (H, W, 3) relay wall positions
        camera_origin: (3,) camera position
        valid_mask: (H, W) boolean mask of valid pixels. If None, uses all pixels.

    Returns:
        volume_min: (3,) minimum bounds
        volume_max: (3,) maximum bounds
    """
    # Filter to only valid relay positions
    if valid_mask is not None:
        valid_positions = relay_pos[valid_mask]  # (N_valid, 3)
        if len(valid_positions) == 0:
            print("  WARNING: No valid positions, using all pixels for bounds")
            valid_positions = relay_pos.reshape(-1, 3)
    else:
        valid_positions = relay_pos.reshape(-1, 3)

    # Get relay wall bounding box from valid positions only
    relay_min = valid_positions.min(axis=0)
    relay_max = valid_positions.max(axis=0)
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
    # Determine which files to use for direct peaks and indirect transient
    direct_file = args.direct_file if args.direct_file else args.transient_file
    indirect_file = args.indirect_file if args.indirect_file else args.transient_file

    using_separate_files = (direct_file != indirect_file)

    if using_separate_files:
        print("\n[2/7] Loading transient data (separate files)...")
        print(f"  Direct file (geometry): {direct_file}")
        print(f"  Indirect file (reconstruction): {indirect_file}")

        # Load direct file for peak detection
        print("  Loading direct file...")
        transient_direct, transient_direct_rgb = load_transient(direct_file, return_rgb=True)
        print(f"    Luminance shape: {transient_direct.shape}")

        # Load indirect file for reconstruction
        print("  Loading indirect file...")
        transient_indirect, transient_indirect_rgb = load_transient(indirect_file, return_rgb=True)
        print(f"    Luminance shape: {transient_indirect.shape}")
        print(f"    RGB shape: {transient_indirect_rgb.shape}")
    else:
        print("\n[2/7] Loading transient data...")
        transient_direct, transient_direct_rgb = load_transient(args.transient_file, return_rgb=True)
        transient_indirect = transient_direct
        transient_indirect_rgb = transient_direct_rgb
        print(f"  Luminance shape: {transient_direct.shape}")
        print(f"  RGB shape: {transient_direct_rgb.shape}")

    # 3. Detect direct peaks and extract relay geometry (using direct file)
    print("\n[3/7] Detecting direct peaks...")
    relay_depths, direct_mask, indirect, valid_mask = detect_direct_peaks(
        transient_direct, start_opl, bin_width
    )

    # Create indirect data (remove direct peaks from indirect file)
    # If using same file, this removes direct peaks; if separate files, we still
    # apply the mask to ensure consistency
    indirect_rgb = transient_indirect_rgb.copy()

    # Apply direct mask to indirect data
    # Note: if files have different content, direct_mask from direct_file is applied to indirect_file
    direct_energy_before = transient_indirect[direct_mask].sum()
    total_energy_before = transient_indirect.sum()

    for c in range(3):
        indirect_rgb[..., c][direct_mask] = 0

    # Report direct peak filtering
    print(f"  Direct peak filtering:")
    print(f"    Masked bins: {direct_mask.sum()} ({100*direct_mask.sum()/direct_mask.size:.1f}% of all bins)")
    print(f"    Energy removed: {direct_energy_before:.4e} ({100*direct_energy_before/total_energy_before:.1f}% of total)")
    print(f"    Remaining indirect energy: {indirect.sum():.4e}")

    # Keep reference to original RGB for color extraction (from direct file)
    transient_rgb = transient_direct_rgb

    # Determine output name early for visualizations
    output_name = args.output_name or os.path.splitext(os.path.basename(args.transient_file))[0].replace('_transient', '_reconstruction')

    # Create output directory: vis/{output_name}/
    output_dir = os.path.join(args.output_dir, output_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"  Output directory: {output_dir}")

    # Visualize depths for debugging (using direct file transient)
    print("\n  Generating depth visualizations...")
    visualize_depths(relay_depths, transient_direct, direct_mask, start_opl, bin_width,
                     output_dir, output_name, use_log_scale=args.transient_log)

    # Visualize direct vs indirect transient
    print("\n  Generating direct vs indirect visualization...")
    debug_pixel_h = args.debug_pixel[0] if args.debug_pixel else None
    debug_pixel_w = args.debug_pixel[1] if args.debug_pixel else None
    visualize_direct_vs_indirect(transient_direct, indirect, direct_mask,
                                  start_opl, bin_width, output_dir, output_name,
                                  pixel_h=debug_pixel_h, pixel_w=debug_pixel_w,
                                  use_log_scale=args.transient_log)

    # 4. Compute relay wall 3D positions by unprojecting camera rays
    print("\n[4/7] Computing relay wall positions (unprojecting camera rays)...")
    relay_pos = compute_relay_positions_from_camera(
        relay_depths,
        scene_params['sensor'],
        scene_params['camera_origin']
    )
    # Report position range for valid pixels only
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
        print(f"  Normal range: X=[{relay_normals[:,:,0].min():.3f}, {relay_normals[:,:,0].max():.3f}]")
        print(f"                Y=[{relay_normals[:,:,1].min():.3f}, {relay_normals[:,:,1].max():.3f}]")
        print(f"                Z=[{relay_normals[:,:,2].min():.3f}, {relay_normals[:,:,2].max():.3f}]")

        # Visualize normals (computed from finite differences)
        print("\n  Generating relay normals visualization...")
        visualize_relay_normals(relay_pos, relay_normals, camera_origin, output_dir, output_name)

    # 5. Determine volume bounds
    print("\n[5/7] Setting up reconstruction volume...")
    camera_origin = scene_params['camera_origin']

    if args.volume_min is not None and args.volume_max is not None:
        volume_min = np.array(args.volume_min, dtype=np.float32)
        volume_max = np.array(args.volume_max, dtype=np.float32)
        print(f"  Using user-specified volume bounds")
    elif args.volume_from_camera:
        # Compute bounds from camera frustum backprojection
        # Use relay depth stats for default depth range if not specified (only valid pixels)
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

    # 6. Phasor field convolution (luminance)
    print(f"\n[6/8] Computing phasor field (wavelength={args.wavelength}m)...")
    phasor_cos, phasor_sin = phasor_convolution(
        indirect, bin_width, args.wavelength
    )

    # Visualize phasors for debugging
    print("\n  Generating phasor visualizations...")
    visualize_phasors(phasor_cos, phasor_sin, start_opl, bin_width, args.wavelength,
                      output_dir, output_name)

    # 7. Phasor backprojection with DrJit (GPU) - Luminance
    falloff_str = "no falloff" if args.no_falloff else "1/r falloff"
    print(f"\n[7/8] Backprojecting luminance to {args.voxel_resolution}^3 volume (DrJit/GPU, {falloff_str})...")

    volume = phasor_backproject_drjit(
        phasor_cos, phasor_sin, relay_pos,
        volume_min, volume_max, args.voxel_resolution,
        start_opl, bin_width, args.wavelength,
        camera_origin, no_falloff=args.no_falloff,
        bin_threshold=args.bin_threshold,
        min_relay_distance=args.min_relay_distance,
        indirect_transient=indirect,
        relay_normals=relay_normals
    )
    print(f"  Volume range: [{volume.min():.6f}, {volume.max():.6f}]")

    # 8. Phasor backprojection for RGB channels
    print(f"\n[8/8] Backprojecting RGB channels...")
    channel_names = ['Red', 'Green', 'Blue']
    volume_rgb = np.zeros((args.voxel_resolution, args.voxel_resolution, args.voxel_resolution, 3),
                          dtype=np.float32)

    for c in range(3):
        print(f"  Processing {channel_names[c]} channel...")
        # Phasor convolution for this channel
        phasor_cos_c, phasor_sin_c = phasor_convolution(
            indirect_rgb[..., c], bin_width, args.wavelength
        )
        # Backproject
        volume_rgb[..., c] = phasor_backproject_drjit(
            phasor_cos_c, phasor_sin_c, relay_pos,
            volume_min, volume_max, args.voxel_resolution,
            start_opl, bin_width, args.wavelength,
            camera_origin, no_falloff=args.no_falloff,
            bin_threshold=args.bin_threshold,
            min_relay_distance=args.min_relay_distance,
            indirect_transient=indirect_rgb[..., c],
            relay_normals=relay_normals
        )
        print(f"    {channel_names[c]} range: [{volume_rgb[..., c].min():.6f}, {volume_rgb[..., c].max():.6f}]")

    # 9. Optional slice normalization
    if args.normalize_slices and args.normalize_slices != 'none':
        print(f"\n[9/11] Normalizing volumes by {args.normalize_slices}-slices...")
        volume = normalize_volume_by_slices(volume, axis=args.normalize_slices)
        volume_rgb = normalize_volume_by_slices(volume_rgb, axis=args.normalize_slices)

    # 10. Extract direct colors for point cloud overlay
    print(f"\n[10/11] Extracting direct colors for point cloud...")
    relay_colors = extract_direct_colors(transient_rgb, direct_mask)
    print(f"  Relay colors range: R=[{relay_colors[...,0].min():.2e}, {relay_colors[...,0].max():.2e}]")
    print(f"                      G=[{relay_colors[...,1].min():.2e}, {relay_colors[...,1].max():.2e}]")
    print(f"                      B=[{relay_colors[...,2].min():.2e}, {relay_colors[...,2].max():.2e}]")

    # 11. Visualize and save
    print(f"\n[11/11] Generating visualizations and saving...")
    visualize_orthographic(volume, volume_min, volume_max, output_dir, output_name,
                           vis_transform=args.vis_transform, vis_scale=args.vis_scale, vis_bias=args.vis_bias,
                           views=args.ortho_views)
    visualize_orthographic_color(volume_rgb, volume_min, volume_max, output_dir, output_name,
                                  views=args.ortho_views)
    visualize_orthographic_with_pointcloud(volume_rgb, relay_pos, relay_colors,
                                           volume_min, volume_max, output_dir, output_name,
                                           views=args.ortho_views)
    save_results(volume, relay_depths, output_dir, output_name)

    # Also save RGB volume
    volume_rgb_path = os.path.join(output_dir, f'{output_name}_volume_rgb.npy')
    np.save(volume_rgb_path, volume_rgb)
    print(f"  Saved RGB volume: {volume_rgb_path}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
