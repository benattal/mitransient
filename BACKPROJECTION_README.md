# Box Position Sweep Experiment

This experiment systematically varies the position of a hidden box in a transient rendering NLOS scene and reconstructs the scene using phasor field backprojection.

## Experiment Overview

**Date:** 2025-11-28

**Scene Setup:**
- Camera at origin (0, 0, 0), looking down -Z axis
- Relay wall: rectangle at z=-2.0, size 4m × 4m (scaled 2x2)
- Hidden box: flat cube (0.5m × 0.5m × 0.001m) behind camera at z > 0
- Confocal transient imaging with 128×128 pixels, 1000 temporal bins

**Parameter Sweep:**
- X translation: -0.25, 0.0, +0.25 meters
- Z translation: 0.25, 0.5, 0.75 meters
- Total configurations: 9 (3 × 3 grid)

## Directory Structure

```
experiments/box_position_sweep/
├── README.md              # This file
├── run_experiment.py      # Automation script
├── scenes/                # Generated scene XML files
│   ├── box_x-0.25_z0.25.xml
│   ├── box_x-0.25_z0.50.xml
│   ├── box_x-0.25_z0.75.xml
│   ├── box_x+0.00_z0.25.xml
│   ├── box_x+0.00_z0.50.xml
│   ├── box_x+0.00_z0.75.xml
│   ├── box_x+0.25_z0.25.xml
│   ├── box_x+0.25_z0.50.xml
│   └── box_x+0.25_z0.75.xml
├── renders/               # Transient rendering outputs
│   ├── box_x*_steady.exr     # Steady-state images
│   ├── box_x*_steady.png     # PNG versions
│   ├── box_x*_transient.mp4  # Transient videos
│   └── box_x*_transient.npy  # Raw transient data (128×128×1000×3)
└── backprojections/       # Reconstruction outputs
    ├── box_x*_orthographic.png  # Combined 3-panel view (front/side/top)
    ├── box_x*_front.png         # Front view (XY plane)
    ├── box_x*_side.png          # Side view (ZY plane)
    ├── box_x*_top.png           # Top view (XZ plane)
    ├── box_x*_volume.npy        # 3D volume (128³)
    ├── box_x*_relay_depths.npy  # Detected relay wall depths
    ├── box_x*_depths_debug.png  # Depth detection visualization
    └── box_x*_phasors_debug.png # Phasor field visualization
```

## Commands Used

### Rendering (per configuration)

```bash
python render_transient.py scenes/box_x{X}_z{Z}.xml \
    --clip-max 0.001 \
    --output-dir experiments/box_position_sweep/renders \
    --output-name box_x{X}_z{Z} \
    --no-plot
```

**Rendering parameters:**
- 10,000 samples per pixel (default)
- Temporal bins: 1000
- Bin width: 0.01 m (OPL)
- Start OPL: 1.0 m

### Backprojection (per configuration)

```bash
python phasor_backprojection.py \
    experiments/box_position_sweep/renders/box_x{X}_z{Z}_transient.npy \
    --scene-file experiments/box_position_sweep/scenes/box_x{X}_z{Z}.xml \
    --voxel-resolution 128 \
    --volume-min -1 -1 0 \
    --volume-max 1 1 1 \
    --wavelength 0.05 \
    --output-dir experiments/box_position_sweep/backprojections \
    --output-name box_x{X}_z{Z}
```

**Backprojection parameters:**
- Voxel resolution: 128³
- Volume bounds: [-1, 1] × [-1, 1] × [0, 1] meters
- Virtual wavelength: 0.05 m (5 cm)

### Running All Experiments

```bash
python experiments/box_position_sweep/run_experiment.py
```

## Results Summary

All 9 configurations completed successfully:

| X (m)  | Z (m) | Name              | Status |
|--------|-------|-------------------|--------|
| -0.25  | 0.25  | box_x-0.25_z0.25  | OK     |
| -0.25  | 0.50  | box_x-0.25_z0.50  | OK     |
| -0.25  | 0.75  | box_x-0.25_z0.75  | OK     |
|  0.00  | 0.25  | box_x+0.00_z0.25  | OK     |
|  0.00  | 0.50  | box_x+0.00_z0.50  | OK     |
|  0.00  | 0.75  | box_x+0.00_z0.75  | OK     |
| +0.25  | 0.25  | box_x+0.25_z0.25  | OK     |
| +0.25  | 0.50  | box_x+0.25_z0.50  | OK     |
| +0.25  | 0.75  | box_x+0.25_z0.75  | OK     |

## Viewing Results

### Orthographic Projections (recommended)
View the `*_orthographic.png` files for a combined front/side/top view:
```bash
eog backprojections/box_x+0.00_z0.50_orthographic.png
```

### Individual Projections
- `*_front.png` - XY plane (front view)
- `*_side.png` - ZY plane (side view)
- `*_top.png` - XZ plane (top view)

### Transient Videos
```bash
vlc renders/box_x+0.00_z0.50_transient.mp4
```

### Loading Volume Data
```python
import numpy as np
volume = np.load('backprojections/box_x+0.00_z0.50_volume.npy')
print(f"Volume shape: {volume.shape}")  # (128, 128, 128)
```

## Expected Observations

1. **X translation**: Box should shift left/right in front view (XY) and top view (XZ)
2. **Z translation**: Box should shift up/down in side view (ZY) and top view (XZ)
3. **Depth effect**: Boxes at larger Z should appear deeper in the reconstructed volume
4. **Reconstruction quality**: May vary with distance from relay wall center

## Scene Geometry Reference

```
                     +Y
                      |
                      |
        Hidden Box    |
        (behind cam)  |
           z > 0      |
                      |
    -------[Camera]---+-------> +Z (behind camera)
                      |
                      |
                      |
          Relay Wall  |
          z = -2.0    |
                      |
                     -Z (in front of camera)
```

The camera looks toward -Z. The relay wall is at z=-2 (in front). The hidden box is placed at positive z values (behind the camera), and light bounces off the relay wall to illuminate it.
