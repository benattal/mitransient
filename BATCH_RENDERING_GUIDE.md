# Batch Rendering with integrator.sample() in mitransient

This guide explains the updated `cornell_confocal.ipynb` notebook that demonstrates batch rendering using the `integrator.sample()` function.

## Overview

The notebook has been updated to show two rendering approaches:

1. **Batch Rendering** (NEW): Uses `integrator.sample()` to render all pixels at once
2. **Standard Rendering**: Uses `mi.render()` for comparison

## Key Components

### 1. RayBatch Helper Class

```python
class RayBatch(NamedTuple):
    """Batch of rays for rendering"""
    origins: mi.Point3f
    directions: mi.Vector3f
    wavelengths: mi.Color0f
```

This simple data structure holds a batch of rays for rendering.

### 2. sample_image_as_ray_batch() Function

This function extracts all rays from a sensor as a batch:

```python
def sample_image_as_ray_batch(
    sensor: mi.Sensor,
    ref_image: Optional[mi.TensorXf] = None,
    ref_mask: Optional[mi.TensorXf] = None
) -> RayBatch
```

**What it does:**
- Gets the film dimensions (width × height)
- Generates pixel indices for all pixels
- Converts to normalized coordinates [0, 1]
- Samples rays for all pixels in one call
- Returns a RayBatch containing all rays

**Key implementation details:**
- Uses `dr.arange()` to create indices for all pixels efficiently
- Computes pixel coordinates: `x = idx % width`, `y = idx // width`
- Normalizes to [0, 1]: `pos = (pixel + 0.5) / dimension`
- Calls `sensor.sample_ray()` with all positions at once

### 3. Batch Rendering Pipeline

The batch rendering process consists of several steps:

#### Step 1: Setup
```python
integrator = scene.integrator()
film = sensor.film()
sampler = sensor.sampler().clone()

# Configure for batch rendering
spp = 256
sampler.set_sample_count(spp)
sampler.set_samples_per_wavefront(spp)
wavefront_size = W * H * spp
sampler.seed(0, wavefront_size)

# Prepare integrator
integrator.prepare_transient(scene, sensor=0)
```

#### Step 2: Sample All Rays
```python
ray_batch = sample_image_as_ray_batch(sensor)
rays = mi.Ray3f(ray_batch.origins, ray_batch.directions)
rays.wavelengths = ray_batch.wavelengths
```

#### Step 3: Define Transient Callback
```python
transient_contributions = []
distances_list = []

def add_transient_callback(spec, distance, wavelengths, active_mask):
    """Callback to accumulate transient contributions"""
    transient_contributions.append(dr.detach(spec))
    distances_list.append(dr.detach(distance))
```

This callback is called by the integrator whenever it accumulates a contribution to the transient image.

#### Step 4: Run Integrator
```python
L, valid, aovs, state = integrator.sample(
    mode=dr.ADMode.Primal,
    scene=scene,
    sampler=sampler,
    ray=rays,
    β=mi.Spectrum(1.0),
    δL=None,
    state_in=None,
    active=mi.Bool(True),
    add_transient=add_transient_callback
)

dr.eval(L)
```

#### Step 5: Reconstruct Images

**Steady-state image:**
```python
L_np = np.array(L)
data_steady_batch = L_np.reshape(H, W, channels)
```

**Transient video:**
```python
data_transient_batch = np.zeros((H, W, temporal_bins, channels))

for contrib, distance in zip(transient_contributions, distances_list):
    # Compute temporal bin index
    bin_indices = ((distance - start_opl) / bin_width_opl).astype(int)

    # Accumulate into appropriate bins
    for each valid contribution:
        data_transient_batch[y, x, t, :] += contribution
```

## Advantages of Batch Rendering

### 1. **Full Control**
- Access to intermediate path tracing results
- Can inspect contributions at each bounce
- Customize sampling strategies

### 2. **Debugging**
- See exactly what the integrator is computing
- Track distances and contributions separately
- Verify temporal binning logic

### 3. **Flexibility**
- Can modify how contributions are accumulated
- Easy to implement custom filters or processing
- Can render subsets of pixels efficiently

### 4. **Performance**
- All rays processed in parallel (SIMD/GPU)
- Single kernel launch for all pixels
- Efficient memory access patterns

## Understanding the Callback

The `add_transient_callback` function is crucial for transient rendering:

```python
def add_transient_callback(spec, distance, wavelengths, active_mask):
    """
    Called by integrator whenever a contribution is made

    Args:
        spec: Spectral contribution (mi.Spectrum)
        distance: Optical path length traveled (mi.Float)
        wavelengths: Wavelength samples
        active_mask: Which rays are active (mi.Bool)
    """
```

**When is it called?**
- After hitting an emitter (direct lighting)
- After next-event estimation (emitter sampling)
- At each bounce with accumulated radiance

**What to do with the data?**
1. Store the contribution and distance
2. Later, bin contributions by distance into temporal bins
3. Accumulate in the appropriate time slot

## Temporal Binning

The transient film divides time into discrete bins:

```
start_opl = 3.5 meters
bin_width_opl = 0.02 meters
temporal_bins = 1000

Total range: [3.5, 23.5] meters of optical path length
```

For each contribution at distance `d`:
```python
bin_index = (d - start_opl) / bin_width_opl
if 0 <= bin_index < temporal_bins:
    data_transient[pixel_y, pixel_x, bin_index] += contribution
```

## Comparison with Standard Rendering

The notebook includes a side-by-side comparison:

```python
# Standard
data_steady_std, data_transient_std = mi.render(scene, spp=256)

# Batch
L, valid, aovs, state = integrator.sample(...)
```

**Expected results:**
- Both should produce similar images (within Monte Carlo variance)
- Batch rendering gives you access to intermediate data
- Standard rendering is simpler but less flexible

## Use Cases

### When to use batch rendering:
1. **Research**: Need to inspect path tracing internals
2. **Custom algorithms**: Implementing modified integrators
3. **Debugging**: Understanding how rendering works
4. **Special effects**: Custom temporal filtering or processing

### When to use standard rendering:
1. **Production**: Just need final images
2. **Simplicity**: Don't need intermediate results
3. **Standard workflow**: Using built-in features

## Complete Workflow Summary

```
1. Setup scene and sensor
   ↓
2. Sample all rays as batch (sample_image_as_ray_batch)
   ↓
3. Initialize sampler for wavefront size
   ↓
4. Prepare integrator for transient rendering
   ↓
5. Define callback to collect contributions
   ↓
6. Call integrator.sample() with batch
   ↓
7. Reconstruct steady image from L
   ↓
8. Bin transient contributions by distance
   ↓
9. Visualize results
```

## Tips and Tricks

### Memory Management
- For large images, process in tiles
- Use `dr.eval()` to force computation and free memory
- `dr.detach()` contributions to avoid holding references

### Performance
- Use `cuda_ad_rgb` variant for GPU acceleration
- Adjust wavefront size for your hardware
- Profile with smaller images first

### Debugging
- Print shapes of intermediate results
- Check contribution counts
- Verify temporal bin assignments
- Compare with standard rendering

### Common Issues

**Issue:** Transient image is all zeros
- **Solution:** Check `add_transient_callback` is being called
- Verify temporal bin range covers the scene distances

**Issue:** Different results from standard rendering
- **Solution:** Use same spp for fair comparison
- Check Monte Carlo variance (render multiple times)

**Issue:** Out of memory
- **Solution:** Reduce image size or render in tiles
- Lower samples per pixel (spp)

## Further Reading

- mitransient documentation: https://mitransient.readthedocs.io
- Mitsuba 3 integrator guide: https://mitsuba.readthedocs.io/en/latest/src/api_reference.html
- DrJIT documentation: https://drjit.readthedocs.io
