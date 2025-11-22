# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`mitransient` is a Python library that extends Mitsuba 3 to support transient light transport simulations. It enables time-resolved rendering by tracking when photons arrive at the camera based on their travel distance and the speed of light. The library is particularly powerful for non-line-of-sight (NLOS) imaging simulations and supports polarization tracking and differentiable transient rendering.

**Key concept:** Time is measured in optical path length (OPL), not seconds. OPL = distance × refractive_index. Light travels 1 meter in ~3.33×10⁻⁹ seconds.

## Architecture

### Plugin System

The library extends Mitsuba 3's plugin architecture with transient-aware components:

- **Integrators** (`mitransient/integrators/`): Core rendering algorithms
  - `TransientPath`: Standard transient path tracing for line-of-sight scenes
  - `TransientNLOSPath`: NLOS-optimized path tracing with laser/hidden geometry sampling
  - `TransientPRBVolPath`: Volumetric path tracing with time resolution
  - All inherit from `TransientADIntegrator` in `common.py`

- **Films** (`mitransient/films/`): Image storage with temporal dimension
  - `TransientHDRFilm`: Stores a video (list of images) instead of single image
  - `PhasorHDRFilm`: Frequency-domain alternative to time-domain storage
  - Films maintain both steady-state and transient data blocks

- **Sensors** (`mitransient/sensors/`): Data capture configurations
  - `NLOSCaptureMeter`: Uniformly samples points on relay wall geometry
  - `NLOSSensor`: Helper for NLOS scene configurations

- **Emitters** (`mitransient/emitters/`):
  - `AngularArea`: Area emitter with restricted angular range

### Core Components

- **`render/`**: Custom image block implementations
  - `TransientImageBlock`: Histogram-based temporal binning
  - `PhasorImageBlock`: Frequency-domain accumulation

- **`nlos.py`**: NLOS-specific utilities for focusing laser emitters on relay walls

- **`utils.py`**: Shared utilities including `speed_of_light` constant and example scenes

- **`vis` module**: Visualization tools (auto-selects polarized vs unpolarized based on variant)

### Variant System

`mitransient` requires setting a Mitsuba variant before import. The variant determines:
- Color representation (RGB, monochromatic, spectral)
- Execution backend (CPU: `llvm_*`, GPU: `cuda_*`)
- Differentiability (`ad` variants for autodiff)

**Always use this pattern:**
```python
import mitsuba as mi
mi.set_variant('llvm_ad_rgb')  # or cuda_ad_rgb for GPU
import mitransient as mitr
```

Scalar variants (`scalar_*`) disable most plugins. Prefer `llvm_*` or `cuda_*` variants.

## Development Commands

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/integration/test_nlos.py

# Tests are in tests/integration/
```

Test configuration: `pytest.ini` sets `testpaths="tests"` and `norecursedirs="ext"`

### Installation

**Standard installation (includes mitsuba dependency):**
```bash
pip install mitransient
```

**Development installation (editable, if using custom Mitsuba 3):**
```bash
pip install mitransient --no-dependencies
# Then set PYTHONPATH to your compiled Mitsuba 3 build
```

**Environment setup (conda):**
```bash
conda env create -f environment.yml
conda activate mitransient
```

The environment.yml specifies Python 3.11, mitsuba 3.7.0, and recommended packages (numpy, matplotlib, opencv-python).

### Building Documentation

Documentation uses Sphinx with ReadTheDocs hosting. The `docs/generate_plugin_doc.py` script auto-generates plugin documentation from docstrings.

## Common Patterns

### Scene Configuration

Transient scenes use the same XML/dict format as Mitsuba 3, but with transient-specific parameters:

```python
scene = mi.load_dict({
    'type': 'scene',
    'integrator': {
        'type': 'transient_path',
        'max_depth': 8,
        'temporal_filter': 'box',  # or 'gaussian'
        'camera_unwarp': False     # whether to account for camera-to-intersection distance
    },
    'sensor': {
        'type': 'perspective',
        'film': {
            'type': 'transient_hdr_film',
            'width': 256,
            'height': 256,
            'temporal_bins': 400,        # number of time frames
            'start_opl': 1.0,            # when recording starts (in meters)
            'bin_width_opl': 0.01,       # duration of each frame (in meters)
            'rfilter': {'type': 'box'}
        }
    }
})
```

### Rendering

```python
# Returns tuple: (steady_state_image, transient_video)
img_steady, img_transient = mi.render(scene, spp=1024)

# Transient shape: (width, height, temporal_bins, channels)
```

### NLOS Workflows

For NLOS scenes, use helper functions to focus the laser on relay wall points:

```python
import mitransient as mitr

# Focus laser at specific pixel of the capture meter
mitr.nlos.focus_emitter_at_relay_wall_pixel(
    mi.Point2f(x, y),
    relay_wall_shape,
    laser_emitter
)

# Or at 3D point in space
mitr.nlos.focus_emitter_at_relay_wall_3dpoint(
    mi.Point3f(x, y, z),
    relay_wall_shape,
    laser_emitter
)
```

### Integrator Preparation

Custom integrators must call `prepare_transient()` before rendering:

```python
integrator = scene.integrator()
integrator.prepare_transient(scene, sensor=0)
data_steady, data_transient = integrator.render(scene)
```

## Important Constraints

- **Variant must be set before import**: The library checks this in `__init__.py` and raises helpful errors
- **Version compatibility**: Enforced in `version.py` with `check_compatibility()`
- **No scalar variants**: Scalar variants disable plugins; use vectorized/JIT variants
- **OPL units**: All temporal parameters use optical path length, not time
- **Steady + Transient**: Films always maintain both steady-state and transient data

## Examples Location

The `examples/` directory contains Jupyter notebooks organized by feature:
- `examples/transient/`: Basic transient rendering
- `examples/transient-nlos/`: NLOS capture simulations
- `examples/polarization/`: Polarization tracking
- `examples/diff-transient/`: Differentiable rendering
- `examples/angulararea-emitter/`: Angular emitter demos

Start with `examples/transient/0-render_cbox_diffuse.ipynb` for quickstart.

## Related Tools

- **TAL toolkit** (https://github.com/diegoroyo/tal): Shell interface for creating/simulating NLOS scenes, easier than direct Python scripting
- **Mitsuba 3 docs**: https://mitsuba.readthedocs.io - foundational concepts apply
