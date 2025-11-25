import os
import numpy as np
import mitsuba as mi

# Set Mitsuba variant
mi.set_variant('cuda_ad_rgb')

import drjit as dr
import mitransient as mitr
# Import the confocal projector emitter to register the plugin
from mitransient.emitters.confocal_projector import ConfocalProjector

print(f"Using Mitsuba version: {mi.__version__}")
print(f"Using Mitransient version: {mitr.__version__}")

# Rendering configuration
spp = 100000

# Load the scene from XML file
scene_path = os.path.join(os.path.dirname(__file__), 'scenes', 'ourbox_confocal.xml')
print(f"Loading scene from: {scene_path}")
scene = mi.load_file(scene_path)
print("Scene loaded successfully!")

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
data_transient_clipped = dr.clip(data_transient, 0.0, 10.0)
data_transient_tonemapped = mitr.vis.tonemap_transient(data_transient_clipped)

output_video_path = os.path.join(os.path.dirname(__file__), 'results', 'ourbox_confocal_output.mp4')
mitr.vis.save_video(
    output_video_path,
    data_transient_tonemapped,
    axis_video=2,  # Time axis is the 3rd dimension
)
print(f"Transient video saved to: {output_video_path}")

# Save raw transient data as .npy file
output_npy_path = os.path.join(os.path.dirname(__file__), 'results', 'ourbox_confocal_transient.npy')
np.save(output_npy_path, np.array(data_transient))
print(f"Transient data saved to: {output_npy_path}")
