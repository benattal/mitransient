# NLOS Transient Rendering and Reconstruction Guide

This document describes the complete workflow for NLOS (Non-Line-of-Sight) imaging:
1. Rendering transient data with `render_transient.py`
2. Reconstructing hidden scenes with `phasor_backprojection.py` or `direct_backprojection.py`

---

## Part 1: Rendering Transient Data

### The `render_transient.py` Script

This script renders a Mitsuba scene with transient (time-resolved) output.

```bash
python render_transient.py <scene_file.xml> [options]
```

### Basic Usage

```bash
# Basic render with default settings
python render_transient.py scenes/ourbox_confocal.xml

# High-quality render with more samples
python render_transient.py scenes/ourbox_confocal.xml --spp 100000

# Custom output location and name
python render_transient.py scenes/ourbox_confocal.xml \
    --output-dir results/experiment1 \
    --output-name my_render
```

### Render Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `scene_file` | (required) | Path to the XML scene file |
| `--spp` | 100000 | Samples per pixel (higher = less noise) |
| `--clip-max` | 10.0 | Maximum clipping for video tonemapping |
| `--output-dir` | `results/` | Output directory |
| `--output-name` | (from scene) | Output filename prefix |
| `--plot-pixel-x` | center | X coordinate for transient plot |
| `--plot-pixel-y` | center | Y coordinate for transient plot |
| `--no-plot` | off | Disable transient plot generation |

### Render Outputs

Files are saved to `{output-dir}/{output-name}/`:

| File | Description |
|------|-------------|
| `steady.exr` | Steady-state image (HDR) |
| `steady.png` | Steady-state image (tonemapped) |
| `transient.mp4` | Transient video (time evolution) |
| `transient.npy` | Raw transient data `(H, W, T, C)` for reconstruction |
| `transient_pixel_X_Y.png` | Transient plot at specified pixel |

---

## Part 2: Separating Direct and NLOS Light

For better NLOS reconstruction, it's recommended to render the direct component and NLOS component separately. This is controlled by the `use_nlos_only` flag in the scene XML.

### The `use_nlos_only` Flag

In your scene XML file, the integrator has a `use_nlos_only` boolean parameter:

```xml
<integrator type='transient_path'>
    <boolean name="use_nlos_only" value="false"/>  <!-- false = full render -->
    <!-- ... other parameters ... -->
</integrator>
```

| Value | Behavior |
|-------|----------|
| `false` | Renders **all light**. The transient includes both direct reflections from visible surfaces (relay wall) and light that bounced off hidden (NLOS) surfaces. |
| `true` | Renders **NLOS light only**. Only captures light paths that interacted with non-line-of-sight (hidden) surfaces. This excludes direct reflections from visible geometry like the relay wall. |

### Creating Separate Scene Files

To render direct and NLOS components separately, create two versions of your scene:

**1. `scene.xml` (or `scene_full.xml`)** - Full render with `use_nlos_only="false"`:
```xml
<integrator type='transient_path'>
    <boolean name="use_nlos_only" value="false"/>
    <!-- ... -->
</integrator>
```

**2. `scene_nlos.xml`** - NLOS surfaces only with `use_nlos_only="true"`:
```xml
<integrator type='transient_path'>
    <boolean name="use_nlos_only" value="true"/>
    <!-- ... -->
</integrator>
```

### Rendering Workflow for Best Results

```bash
# Step 1: Render full transient (all light - for relay wall depth detection)
python render_transient.py scenes/ourbox_confocal.xml \
    --output-dir results \
    --output-name ourbox_full \
    --spp 100000

# Step 2: Create NLOS-only scene (copy and set use_nlos_only to true)
# Then render NLOS-only transient (for reconstruction)
python render_transient.py scenes/ourbox_confocal_nlos.xml \
    --output-dir results \
    --output-name ourbox_nlos \
    --spp 100000

# Step 3: Backproject using both files
python direct_backprojection.py results/ourbox_full/transient.npy \
    --indirect-file results/ourbox_nlos/transient.npy \
    --scene-file scenes/ourbox_confocal.xml \
    --voxel-resolution 128
```

### Why Separate Full/NLOS Renders?

- **Full render** (`use_nlos_only=false`): Contains all light paths - both reflections from visible surfaces (relay wall, boxes) and light from hidden NLOS surfaces. The direct peak in this render is used to detect relay wall geometry (depth at each pixel).
- **NLOS-only render** (`use_nlos_only=true`): Contains only light that interacted with hidden (NLOS) surfaces. This is what we actually want to backproject for reconstruction.

When using only the full render, the backprojection script must detect and remove the direct peak algorithmically, which can be imperfect. Using separate files gives cleaner results because the NLOS-only render has no direct peak to remove.

---

## Part 3: Scene Configuration

### Key Scene Parameters

The scene XML must include a `transient_hdr_film` with temporal parameters:

```xml
<film type="transient_hdr_film">
    <integer name="width" value="128"/>
    <integer name="height" value="128"/>
    <integer name="temporal_bins" value="700"/>
    <float name="start_opl" value="0"/>
    <float name="bin_width_opl" value="0.01"/>
    <rfilter type="box"/>
</film>
```

| Parameter | Description |
|-----------|-------------|
| `width`, `height` | Spatial resolution (relay wall pixels) |
| `temporal_bins` | Number of time frames |
| `start_opl` | When recording starts (meters of optical path length) |
| `bin_width_opl` | Duration of each bin (meters OPL). 0.01m ≈ 33 picoseconds |

### Integrator Parameters

```xml
<integrator type='transient_path'>
    <boolean name="camera_unwarp" value="false"/>
    <boolean name="use_nlos_only" value="false"/>
    <integer name="max_depth" value="8"/>
    <string name="temporal_filter" value="box"/>
</integrator>
```

| Parameter | Description |
|-----------|-------------|
| `camera_unwarp` | Account for camera-to-intersection distance in timing |
| `use_nlos_only` | Only render indirect light (see above) |
| `max_depth` | Maximum path bounces |
| `temporal_filter` | Temporal filtering: `box` or `gaussian` |

---

## Part 4: Backprojection

## Overview

Both scripts reconstruct a 3D volume from transient (time-resolved) captures:

| Script | Method | Best For |
|--------|--------|----------|
| `phasor_backprojection.py` | Phasor field (frequency domain) | Sharp reconstructions, wave-based focusing |
| `direct_backprojection.py` | Direct intensity (time domain) | Simple, interpretable, fast |

### Key Difference

- **Phasor**: Convolves transient with virtual wavelength, uses complex wave propagation with `exp(i*k*r) / r`. Produces sharper results but requires tuning wavelength.
- **Direct**: Deposits transient intensity directly into voxels weighted by `1/r²`. Simpler and more intuitive.

## Input Requirements

1. **Transient data** (`.npy` file): Shape `(H, W, T, C)` where:
   - `H, W`: Spatial resolution (relay wall pixels)
   - `T`: Temporal bins
   - `C`: Color channels (typically 3 for RGB)

2. **Scene file** (`.xml`): Mitsuba scene with `transient_hdr_film` containing:
   - `start_opl`: When recording starts (meters)
   - `bin_width_opl`: Duration per bin (meters)

## Basic Usage

### Single Transient File (combined direct+indirect)

```bash
# Direct backprojection
python direct_backprojection.py results/scene/transient.npy \
    --scene-file scenes/my_scene.xml \
    --voxel-resolution 64

# Phasor backprojection
python phasor_backprojection.py results/scene/transient.npy \
    --scene-file scenes/my_scene.xml \
    --voxel-resolution 64 \
    --wavelength 0.05
```

### Separate Direct/Indirect Files (recommended for better results)

When you have separate renders for direct (relay wall only) and indirect (hidden scene only) light:

```bash
python direct_backprojection.py results/scene_direct/transient.npy \
    --indirect-file results/scene_indirect/transient.npy \
    --scene-file scenes/my_scene.xml \
    --voxel-resolution 128
```

The first positional argument (or `--direct-file`) is used for:
- Detecting relay wall geometry (direct peak detection)

The `--indirect-file` is used for:
- Actual reconstruction (backprojection)

## Volume Bounds

### Auto-compute from Camera Frustum (recommended)

```bash
--volume-from-camera
```

Optionally specify depth range:
```bash
--volume-from-camera --volume-depth-min 0.5 --volume-depth-max 3.0
```

### Manual Bounds

```bash
--volume-min -1 -0.1 -1 --volume-max 1 1 1
```

Coordinates are in meters: `(x_min, y_min, z_min)` and `(x_max, y_max, z_max)`.

## Key Parameters

### Common to Both Scripts

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--voxel-resolution` | 64 | Voxels per dimension (64³, 128³, etc.) |
| `--volume-min/max` | auto | Manual volume bounds (x, y, z) in meters |
| `--volume-from-camera` | off | Auto-compute bounds from camera frustum |
| `--bin-threshold` | 0.0 | Only use bins above this fraction of max (0.0-1.0) |
| `--min-relay-distance` | 0.0 | Skip voxels closer than this to relay (meters) |
| `--no-falloff` | off | Disable distance falloff weighting |
| `--no-hemisphere-filter` | off | Disable hemisphere filtering (by default, only voxels in the positive normal hemisphere contribute) |
| `--output-dir` | `vis/` | Output directory |
| `--output-name` | auto | Output filename prefix |

### Phasor-Specific

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--wavelength` | 0.05 | Virtual wavelength in meters (smaller = sharper but noisier) |

### Visualization Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--vis-transform` | none | Volume visualization: `none`, `log`, `sqrt`, `cbrt`, `exp`, `sigmoid` |
| `--vis-scale` | 1.0 | Scale factor: `transform(scale * input + bias)` |
| `--vis-bias` | 0.0 | Bias offset: `transform(scale * input + bias)` |
| `--transient-log` | off | Use log scale for transient debug plots |
| `--normalize-slices` | none | Normalize per slice: `x`, `y`, `z`, or `none` |

## Example Commands

### Basic Reconstruction

```bash
python direct_backprojection.py results/ourbox_confocal/transient.npy \
    --scene-file scenes/ourbox_confocal.xml \
    --voxel-resolution 64
```

### High-Quality with Single File

```bash
python direct_backprojection.py results/ourbox_confocal/transient.npy \
    --scene-file scenes/ourbox_confocal_hidden.xml \
    --voxel-resolution 128 \
    --volume-from-camera \
    --volume-min -1 -0.1 -1 \
    --volume-max 1 1 1 \
    --bin-threshold 0.001 \
    --min-relay-distance 0.1
```

### High-Quality with Separate Direct/Indirect Files

```bash
python direct_backprojection.py results/ourbox_confocal_direct/transient.npy \
    --indirect-file results/ourbox_confocal_indirect/transient.npy \
    --scene-file scenes/ourbox_confocal_hidden.xml \
    --voxel-resolution 128 \
    --volume-from-camera \
    --volume-min -1 -0.1 -1 \
    --volume-max 1 1 1 \
    --bin-threshold 0.01 \
    --min-relay-distance 0.1
```

### Phasor with Wavelength Tuning

```bash
python phasor_backprojection.py results/scene/transient.npy \
    --scene-file scenes/my_scene.xml \
    --voxel-resolution 128 \
    --wavelength 0.03 \
    --volume-from-camera
```

### With Visualization Transform

```bash
# Log scale for high dynamic range
python direct_backprojection.py results/scene/transient.npy \
    --scene-file scenes/scene.xml \
    --vis-transform log

# Square root (gamma-like compression)
python direct_backprojection.py results/scene/transient.npy \
    --scene-file scenes/scene.xml \
    --vis-transform sqrt

# Sigmoid with custom scale/bias
python direct_backprojection.py results/scene/transient.npy \
    --scene-file scenes/scene.xml \
    --vis-transform sigmoid \
    --vis-scale 10 \
    --vis-bias -5
```

### Wall Box HI NLOS (Retroreflective Letters)

```bash
python direct_backprojection.py renders/wall_box_hi_nlos/transient.npy \
    --scene-file scenes/wall_box_hi_nlos.xml \
    --voxel-resolution 128 \
    --volume-min -1 -1 -0.9 \
    --volume-max 1 0 0 \
    --bin-threshold 0.0 \
    --min-relay-distance 0.0 \
    --vis-transform none \
    --no-hemisphere-filter \
    --transient-log \
    --debug-pixel 50 64 \
    --no-falloff
```

## Output Files

Files are saved to `{output-dir}/{output-name}/`:

Both scripts generate:

| File | Description |
|------|-------------|
| `orthographic.png` | Combined front/side/top max-intensity projections (grayscale) |
| `orthographic_color.png` | Combined RGB projections |
| `orthographic_overlay.png` | Reconstruction overlaid with relay wall point cloud |
| `volume.npy` | 3D reconstruction volume `(Vx, Vy, Vz)` |
| `volume_rgb.npy` | 3D RGB volume `(Vx, Vy, Vz, 3)` |
| `relay_depths.npy` | Detected relay wall depths `(H, W)` |
| `depths_debug.png` | Depth detection visualization |
| `direct_vs_indirect.png` | Transient decomposition visualization |

Phasor additionally generates:

| File | Description |
|------|-------------|
| `phasors_debug.png` | Phasor field visualization |
| `relay_normals.png` | Computed relay wall normals |

## Tips

1. **Start with low resolution** (`--voxel-resolution 64`) to quickly iterate on volume bounds.

2. **Use separate files** when possible - rendering direct and indirect separately gives cleaner geometry detection.

3. **Adjust `--bin-threshold`** to filter noise. Start with 0.01 (1% of max signal).

4. **Set `--min-relay-distance`** to avoid artifacts near the relay wall (try 0.05-0.2 meters).

5. **For phasor**, smaller wavelength gives sharper results but more noise. Try 0.02-0.1 meters.

6. **Check debug visualizations** (`*_depths_debug.png`, `*_direct_vs_indirect.png`) to verify correct direct peak detection.

## Algorithm Details

### Direct Backprojection

For each voxel at position `v`:
```
intensity(v) = Σ_relay [ transient(relay, t) / |v - relay|² ]
```
where `t` is the time bin corresponding to the round-trip OPL:
```
t = (2 * |camera - relay| + 2 * |relay - v| - start_opl) / bin_width
```

### Phasor Backprojection

1. Convolve transient with virtual wavelength kernel:
   ```
   phasor = transient * [cos(ωt) + i·sin(ωt)]
   ```

2. Backproject with Rayleigh-Sommerfeld diffraction:
   ```
   U(v) = Σ_relay [ phasor(relay, t) * exp(i·k·r) / r ]
   ```
   where `k = 2π/wavelength` and `r = |v - relay|`.

3. Take magnitude: `intensity(v) = |U(v)|`

## Troubleshooting

### No reconstruction visible
- Check `*_depths_debug.png` to verify direct peak detection is correct
- Try increasing `--voxel-resolution`
- Verify volume bounds contain the hidden scene

### Noisy reconstruction
- Increase `--bin-threshold` (e.g., 0.01 to 0.1)
- Increase `--min-relay-distance`
- For phasor, increase `--wavelength`

### Artifacts near relay wall
- Increase `--min-relay-distance` (try 0.1-0.3 meters)
- Check if direct peak removal is working (`*_direct_vs_indirect.png`)

### Wrong depth scale
- Verify `start_opl` and `bin_width_opl` in scene file match render parameters
- Check that scene file matches the one used for rendering

---

## Quick Reference: Complete Workflow

Here's the complete workflow from scene to reconstruction:

```bash
# 1. Prepare scene files
#    - scene.xml: with use_nlos_only="false" (full render)
#    - scene_nlos.xml: with use_nlos_only="true" (NLOS surfaces only)

# 2. Render full transient (all light - for relay wall depth detection)
python render_transient.py scenes/scene.xml \
    --output-dir results --output-name scene_full --spp 100000 --no-plot

# 3. Render NLOS-only transient (for reconstruction)
python render_transient.py scenes/scene_nlos.xml \
    --output-dir results --output-name scene_nlos --spp 100000 --no-plot

# 4. Backproject with both files
python direct_backprojection.py results/scene_full/transient.npy \
    --indirect-file results/scene_nlos/transient.npy \
    --scene-file scenes/scene.xml \
    --voxel-resolution 128 \
    --volume-min -1 -0.1 -1 --volume-max 1 1 1 \
    --bin-threshold 0.01 \
    --min-relay-distance 0.1

# Or with single combined file (simpler but less accurate):
python direct_backprojection.py results/scene_full/transient.npy \
    --scene-file scenes/scene.xml \
    --voxel-resolution 128 \
    --volume-min -1 -0.1 -1 --volume-max 1 1 1
```

### Minimal Example (single file)

```bash
# Render
python render_transient.py scenes/ourbox_confocal.xml --spp 50000

# Reconstruct
python direct_backprojection.py results/ourbox_confocal/transient.npy \
    --scene-file scenes/ourbox_confocal.xml \
    --voxel-resolution 64
```
