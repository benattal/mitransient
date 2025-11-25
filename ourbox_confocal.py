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
parser = argparse.ArgumentParser(description='Render ourbox scene with confocal projector')
parser.add_argument('--non-confocal', action='store_true',
                    help='Use non-confocal mode (projector frame is fixed, not dynamic)')
parser.add_argument('--clip-max', type=float, default=10.0,
                    help='Maximum clipping value for tonemapping (default: 10.0)')
args = parser.parse_args()

is_confocal = not args.non_confocal

print(f"Using Mitsuba version: {mi.__version__}")
print(f"Using Mitransient version: {mitr.__version__}")
print(f"Confocal mode: {is_confocal}")

# Rendering configuration
spp = 100000

# Load the scene from XML file
scene_path = os.path.join(os.path.dirname(__file__), 'scenes', 'ourbox_confocal.xml')
print(f"Loading scene from: {scene_path}")
scene = mi.load_file(scene_path)
print("Scene loaded successfully!")

# Extract scene params and update is_confocal setting
params = mi.traverse(scene)
print(f"Available params: {list(params.keys())}")

# Print scene info
sensor = scene.sensors()[0]
film = sensor.film()
print(f"Film size: {film.size()}")
print(f"Integrator: {scene.integrator()}")

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
output_video_path = os.path.join(os.path.dirname(__file__), 'results', f'ourbox_{mode_suffix}_output.mp4')
mitr.vis.save_video(
    output_video_path,
    data_transient_tonemapped,
    axis_video=2,  # Time axis is the 3rd dimension
)
print(f"Transient video saved to: {output_video_path}")

# Save raw transient data as .npy file
output_npy_path = os.path.join(os.path.dirname(__file__), 'results', f'ourbox_{mode_suffix}_transient.npy')
np.save(output_npy_path, np.array(data_transient))
print(f"Transient data saved to: {output_npy_path}")
