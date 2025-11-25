import os
import argparse
import numpy as np
import mitsuba as mi

# Set Mitsuba variant
mi.set_variant('cuda_ad_rgb')

import drjit as dr
import mitransient as mitr
# Import the confocal projector emitter to register the plugin
from mitransient.emitters.confocal_projector import ConfocalProjector

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Render Cornell box with confocal projector')
parser.add_argument('--non-confocal', action='store_true',
                    help='Use non-confocal mode (projector frame is fixed, not dynamic)')
parser.add_argument('--clip-max', type=float, default=1.0,
                    help='Maximum clipping value for tonemapping (default: 1.0)')
args = parser.parse_args()

is_confocal = not args.non_confocal

print(f"Using Mitsuba version: {mi.__version__}")
print(f"Using Mitransient version: {mitr.__version__}")
print(f"Confocal mode: {is_confocal}")

# Rendering configuration
spp = 100000

# Get the cornell box scene dictionary
d = mitr.cornell_box()

# Remove existing light and integrator
d.pop('light', None)
d.pop('integrator', None)

# Update film parameters
d['sensor']['film']['temporal_bins'] = 1000
d['sensor']['film']['width'] = 256
d['sensor']['film']['height'] = 256

# Create the ConfocalProjector emitter (single spot, same spread as in cornell_confocal.ipynb)
# sigma=0.001, intensity=100000.0 (single spot), fov=0.55, is_confocal=True
projector = mi.load_dict({
    "type": "confocal_projector",
    "grid_rows": 1,
    "grid_cols": 1,
    "grid_sigma": 0.001,
    "grid_intensity": 100000.0,
    "grid_spacing": "uniform",
    "fov": 0.55,
    "is_confocal": is_confocal,  # Confocal mode - frame computed dynamically from camera ray
    "max_rejection_samples": 4,
})

print(f"Confocal projector created:")
print(projector.to_string())

# Setup integrator with reference to the projector
integrator = mi.load_dict({
    "type": "transient_path",
    "temporal_filter": "box",
    "use_nlos_only": False,
    "camera_unwarp": False,
    "max_depth": 8,
    "confocal_projector": projector,
})
d['integrator'] = integrator

print(f"Integrator confocal_projector: {integrator.confocal_projector}")

# Load the scene
scene = mi.load_dict(d)
print("Scene loaded successfully!")

# Render the scene
print(f"Rendering with {spp} samples per pixel...")
data_steady, data_transient = mi.render(scene, spp=spp)

print(f"Rendering complete!")
print(f"Steady shape: {data_steady.shape}")
print(f"Transient shape: {data_transient.shape}")

# Tonemap and save transient video
data_transient_clipped = dr.clip(data_transient, 0.0, args.clip_max)
data_transient_tonemapped = mitr.vis.tonemap_transient(data_transient_clipped)

mode_suffix = 'confocal' if is_confocal else 'non_confocal'
output_video_path = os.path.join(os.path.dirname(__file__), 'results', f'{mode_suffix}_output.mp4')
mitr.vis.save_video(
    output_video_path,
    data_transient_tonemapped,
    axis_video=2,  # Time axis is the 3rd dimension
)
print(f"Transient video saved to: {output_video_path}")

# Save raw transient data as .npy file
output_npy_path = os.path.join(os.path.dirname(__file__), 'results', f'{mode_suffix}_transient.npy')
np.save(output_npy_path, np.array(data_transient))
print(f"Transient data saved to: {output_npy_path}")
