import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import mitsuba as mi

# Set Mitsuba variant
mi.set_variant('cuda_ad_rgb')

import drjit as dr
import mitransient as mitr
# Import the confocal projector emitter to register the plugin
from mitransient.emitters.confocal_projector import ConfocalProjector

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Render transient scene from XML file')
parser.add_argument('scene_file', type=str,
                    help='Path to the XML scene file')
parser.add_argument('--spp', type=int, default=100000,
                    help='Samples per pixel (default: 100000)')
parser.add_argument('--clip-max', type=float, default=10.0,
                    help='Maximum clipping value for tonemapping (default: 10.0)')
parser.add_argument('--output-dir', type=str, default=None,
                    help='Output directory (default: results/)')
parser.add_argument('--output-name', type=str, default=None,
                    help='Output filename prefix (default: derived from scene file)')
parser.add_argument('--plot-pixel-x', type=int, default=None,
                    help='X coordinate of pixel to plot transient (default: center)')
parser.add_argument('--plot-pixel-y', type=int, default=None,
                    help='Y coordinate of pixel to plot transient (default: center)')
parser.add_argument('--no-plot', action='store_true',
                    help='Disable transient plot')
args = parser.parse_args()

print(f"Using Mitsuba version: {mi.__version__}")
print(f"Using Mitransient version: {mitr.__version__}")

# Load the scene from XML file
scene_path = os.path.abspath(args.scene_file)
if not os.path.exists(scene_path):
    raise FileNotFoundError(f"Scene file not found: {scene_path}")

print(f"Loading scene from: {scene_path}")
scene = mi.load_file(scene_path)
print("Scene loaded successfully!")

# Extract scene params
params = mi.traverse(scene)
print(f"Available params: {list(params.keys())}")

# Print scene info
sensor = scene.sensors()[0]
film = sensor.film()
print(f"Film size: {film.size()}")
print(f"Integrator: {scene.integrator()}")

# Render the scene
print(f"Rendering with {args.spp} samples per pixel...")
data_steady, data_transient = mi.render(scene, spp=args.spp)

print(f"Rendering complete!")
print(f"Steady shape: {data_steady.shape}")

# Determine output name first
if args.output_name:
    output_name = args.output_name
else:
    # Derive from scene filename (without extension)
    output_name = os.path.splitext(os.path.basename(scene_path))[0]

# Create output directory: results/{output_name}/ by default
base_output_dir = args.output_dir if args.output_dir else os.path.join(os.path.dirname(__file__), 'renders')
output_dir = os.path.join(base_output_dir, output_name)
os.makedirs(output_dir, exist_ok=True)

# Save steady-state image
output_steady_exr_path = os.path.join(output_dir, 'steady.exr')
mi.util.write_bitmap(output_steady_exr_path, data_steady)
print(f"Steady-state image saved to: {output_steady_exr_path}")

output_steady_png_path = os.path.join(output_dir, 'steady.png')
mi.util.write_bitmap(output_steady_png_path, data_steady)
print(f"Steady-state image saved to: {output_steady_png_path}")

# Tonemap and save transient video
data_transient_clipped = dr.clip(data_transient, 0.0, args.clip_max)
data_transient_tonemapped = mitr.vis.tonemap_transient(data_transient_clipped)

output_video_path = os.path.join(output_dir, 'transient.mp4')
mitr.vis.save_video(
    output_video_path,
    data_transient_tonemapped,
    axis_video=2,  # Time axis is the 3rd dimension
)
print(f"Transient video saved to: {output_video_path}")

# Save raw transient data as .npy file
output_npy_path = os.path.join(output_dir, 'transient.npy')
np.save(output_npy_path, np.array(data_transient))
print(f"Transient data saved to: {output_npy_path}")

# Plot transient at a specific pixel
if not args.no_plot:
    # Convert to numpy for plotting
    transient_np = np.array(data_transient_clipped)
    height, width = transient_np.shape[:2]

    # Default to center pixel if not specified
    pixel_x = args.plot_pixel_x if args.plot_pixel_x is not None else width // 2
    pixel_y = args.plot_pixel_y if args.plot_pixel_y is not None else height // 2

    # Clamp to valid range
    pixel_x = max(0, min(pixel_x, width - 1))
    pixel_y = max(0, min(pixel_y, height - 1))

    print(f"Plotting transient at pixel ({pixel_x}, {pixel_y})")

    # Extract transient at the pixel (shape: [temporal_bins, channels])
    pixel_transient = transient_np[pixel_y, pixel_x, :, :]

    # Get temporal parameters from film if available
    try:
        start_opl = film.start_opl()
        bin_width = film.bin_width_opl()
        num_bins = pixel_transient.shape[0]
        time_axis = np.arange(num_bins) * bin_width + start_opl
        xlabel = 'Optical Path Length (m)'
    except:
        time_axis = np.arange(pixel_transient.shape[0])
        xlabel = 'Time bin'

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # If RGB, plot each channel and total intensity
    if pixel_transient.shape[1] >= 3:
        ax.plot(time_axis, pixel_transient[:, 0], 'r-', alpha=0.7, label='R')
        ax.plot(time_axis, pixel_transient[:, 1], 'g-', alpha=0.7, label='G')
        ax.plot(time_axis, pixel_transient[:, 2], 'b-', alpha=0.7, label='B')
        # Plot luminance
        luminance = 0.2126 * pixel_transient[:, 0] + 0.7152 * pixel_transient[:, 1] + 0.0722 * pixel_transient[:, 2]
        ax.plot(time_axis, luminance, 'k-', linewidth=2, label='Luminance')
    else:
        ax.plot(time_axis, pixel_transient[:, 0], 'k-', linewidth=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Radiance')
    ax.set_title(f'Transient at pixel ({pixel_x}, {pixel_y})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save plot
    output_plot_path = os.path.join(output_dir, f'transient_pixel_{pixel_x}_{pixel_y}.png')
    plt.savefig(output_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Transient plot saved to: {output_plot_path}")
