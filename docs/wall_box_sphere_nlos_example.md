# Wall-Box-Sphere NLOS Example

This guide demonstrates rendering and backprojection reconstruction on the `wall_box_sphere_nlos` scene - a simple NLOS scene with a relay wall, a box, and a green sphere.

## Scene Overview

The scene (`scenes/wall_box_sphere_nlos.xml`) contains:

- **Camera** at origin, looking down -Z axis (45° FOV)
- **Relay wall** at z=-2.0 (4m × 4m gray rectangle)
- **Hidden objects**:
  - Red box (floor-like surface at y=-1.0)
  - Green sphere (center at y=-0.75, z=-0.75, radius 0.25)
- **Confocal projector** emitter for NLOS capture

Key film parameters:
- Resolution: 128×128 pixels
- Temporal bins: 1000
- Bin width: 0.01m (10mm per bin)
- Start OPL: 0.0m

## Step 1: Render Transient Data

```bash
python render_transient.py scenes/wall_box_sphere_nlos.xml --clip-max 0.001
```

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--clip-max` | 0.001 | Low clipping value for tonemapping (scene has low radiance) |

The low `--clip-max` is needed because NLOS scenes have very low radiance values (light bounces multiple times).

### Outputs

Files are saved to `results/`:
- `wall_box_sphere_nlos_steady.exr` - Steady-state HDR image
- `wall_box_sphere_nlos_steady.png` - Steady-state tonemapped image
- `wall_box_sphere_nlos_transient.mp4` - Transient video
- `wall_box_sphere_nlos_transient.npy` - Raw transient data for reconstruction

## Step 2: Run Backprojection

```bash
python direct_backprojection.py results/wall_box_sphere_nlos_transient.npy \
    --scene-file scenes/wall_box_sphere_nlos.xml \
    --voxel-resolution 128 \
    --volume-min -1 -1 -0.9 \
    --volume-max 1 0 0 \
    --bin-threshold 0.01 \
    --min-relay-distance 0.01 \
    --vis-transform none \
    --no-hemisphere-filter
```

### Parameters Explained

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--voxel-resolution` | 128 | 128³ voxel grid for reconstruction |
| `--volume-min` | -1 -1 -0.9 | Volume bounds (x_min, y_min, z_min) in meters |
| `--volume-max` | 1 0 0 | Volume bounds (x_max, y_max, z_max) in meters |
| `--bin-threshold` | 0.01 | Only use bins above 1% of max signal (noise filter) |
| `--min-relay-distance` | 0.01 | Skip voxels within 1cm of relay wall |
| `--vis-transform` | none | No intensity transform (linear scale) |
| `--no-hemisphere-filter` | - | Disable hemisphere filtering |

### Volume Bounds

The volume bounds are chosen to capture the hidden scene region:
- **X**: -1 to 1 (full width behind relay wall)
- **Y**: -1 to 0 (from floor level to just above the sphere)
- **Z**: -0.9 to 0 (between relay wall at z=-2 and camera)

### Outputs

Files are saved to `vis/`:
- `*_orthographic.png` - Max-intensity projections (front/side/top views)
- `*_orthographic_color.png` - RGB projections
- `*_volume.npy` - 3D reconstruction volume
- `*_depths_debug.png` - Relay wall depth detection visualization

## Complete Workflow

```bash
# Step 1: Render
python render_transient.py scenes/wall_box_sphere_nlos.xml --clip-max 0.001

# Step 2: Reconstruct
python direct_backprojection.py results/wall_box_sphere_nlos_transient.npy \
    --scene-file scenes/wall_box_sphere_nlos.xml \
    --voxel-resolution 128 \
    --volume-min -1 -1 -0.9 \
    --volume-max 1 0 0 \
    --bin-threshold 0.01 \
    --min-relay-distance 0.01 \
    --vis-transform none \
    --no-hemisphere-filter
```

## Tips

1. **Adjust `--bin-threshold`** if reconstruction is noisy (try 0.02-0.05).

2. **Increase `--min-relay-distance`** (e.g., 0.05) if there are artifacts near the relay wall.

3. **Try different volume bounds** to focus on specific regions of interest.

4. **Use `--vis-transform sqrt` or `log`** if reconstruction has high dynamic range.

5. **For higher quality**, increase render samples:
   ```bash
   python render_transient.py scenes/wall_box_sphere_nlos.xml --spp 200000 --clip-max 0.001
   ```
