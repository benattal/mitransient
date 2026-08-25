import argparse
import math
import os
import sys
from pathlib import Path

import mitsuba as mi
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render transient scene from XML file")
    parser.add_argument("scene_file", type=str, help="Path to the XML scene file")
    parser.add_argument(
        "--variant",
        type=str,
        default="cuda_ad_rgb",
        help="Mitsuba variant to use, e.g. llvm_ad_rgb or cuda_ad_rgb",
    )
    parser.add_argument(
        "--spp",
        type=int,
        default=100000,
        help="Samples per pixel (default: 100000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic Mitsuba render seed (default: 0)",
    )
    parser.add_argument(
        "--clip-max",
        type=float,
        default=10.0,
        help="Maximum clipping value for tonemapping (default: 10.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for rendered data and images (default: derived from base-output-dir/output-name/)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Output filename prefix (default: derived from scene file)",
    )
    parser.add_argument(
        "--plot-pixel-x",
        type=int,
        default=None,
        help="X coordinate of pixel to plot transient (default: center)",
    )
    parser.add_argument(
        "--plot-pixel-y",
        type=int,
        default=None,
        help="Y coordinate of pixel to plot transient (default: center)",
    )
    parser.add_argument("--no-plot", action="store_true", help="Disable transient plot")
    parser.add_argument("--no-video", action="store_true", help="Disable transient video output")
    parser.add_argument(
        "--pulse-samples",
        type=int,
        default=None,
        help="Number of pulse samples per path vertex (default: use scene value)",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Use log scale for y-axis in transient plot",
    )
    parser.add_argument(
        "--pixel-filter",
        type=str,
        default="box",
        choices=["box", "delta"],
        help="Pixel filter mode: box keeps subpixel jitter, delta point-samples at pixel centers (default: box)",
    )
    parser.add_argument(
        "--spp-per-pass",
        type=int,
        default=None,
        help=(
            "Maximum SPP per render pass. Splits --spp into deterministic "
            "passes and forms an SPP-weighted average to limit GPU memory."
        ),
    )
    parser.add_argument(
        "--preserve-shape-ids",
        action="store_true",
        help=(
            "Disable Mitsuba's compatible-mesh merge so integrator shape-id "
            "filters remain exact."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    mi.set_variant(args.variant)
    print("Mitsuba current variant:", mi.variant())

    import drjit as dr
    import mitransient as mitr
    from mitransient.bsdfs.retroreflector import Retroreflector
    from mitransient.emitters.confocal_projector import ConfocalProjector

    print(f"Using Mitsuba version: {mi.__version__}")
    print(f"Using Mitransient version: {mitr.__version__}")

    scene_path = os.path.abspath(args.scene_file)
    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"Scene file not found: {scene_path}")

    print(f"Loading scene from: {scene_path}")
    if args.preserve_shape_ids:
        config = mi.parser.ParserConfig(args.variant)
        config.merge_meshes = False
        previous_cwd = os.getcwd()
        try:
            os.chdir(os.path.dirname(scene_path))
            state = mi.parser.parse_file(config, scene_path)
            mi.parser.transform_all(config, state)
            scene = mi.parser.instantiate(config, state)
        finally:
            os.chdir(previous_cwd)
    else:
        scene = mi.load_file(scene_path)
    print("Scene loaded successfully!")

    params = mi.traverse(scene)
    print(f"Available params: {list(params.keys())}")

    sensor = scene.sensors()[0]
    film = sensor.film()
    print(f"Film size: {film.size()}")
    print(f"Integrator: {scene.integrator()}")

    if args.pulse_samples is not None:
        integrator = scene.integrator()
        if hasattr(integrator, "pulse_samples"):
            integrator.pulse_samples = args.pulse_samples
            print(f"Using {args.pulse_samples} pulse samples per vertex")
        else:
            print("Warning: Integrator does not support pulse_samples parameter")

    integrator = scene.integrator()
    if hasattr(integrator, "pixel_filter"):
        integrator.pixel_filter = args.pixel_filter
        print(f"Using pixel filter mode: {args.pixel_filter}")
    else:
        print("Warning: Integrator does not support pixel_filter parameter")

    if args.spp_per_pass is None or args.spp_per_pass >= args.spp:
        print(f"Rendering with {args.spp} samples per pixel...")
        data_steady, data_transient = mi.render(
            scene, spp=args.spp, seed=args.seed
        )
    else:
        if args.spp_per_pass < 1:
            raise ValueError("--spp-per-pass must be positive")
        num_passes = math.ceil(args.spp / args.spp_per_pass)
        print(
            f"Rendering {num_passes} passes with at most "
            f"{args.spp_per_pass} spp each (total {args.spp} spp)..."
        )
        steady_acc = None
        transient_acc = None
        completed_spp = 0
        for pass_index in range(num_passes):
            pass_spp = min(args.spp_per_pass, args.spp - completed_spp)
            print(
                f"  Pass {pass_index + 1}/{num_passes}: {pass_spp} spp...",
                flush=True,
            )
            steady, transient = mi.render(
                scene, spp=pass_spp, seed=args.seed + pass_index
            )
            steady_np = np.asarray(steady, dtype=np.float32)
            transient_np = np.asarray(transient, dtype=np.float32)
            if steady_acc is None:
                steady_acc = steady_np * pass_spp
                transient_acc = transient_np * pass_spp
            else:
                steady_acc += steady_np * pass_spp
                transient_acc += transient_np * pass_spp
            completed_spp += pass_spp
            del steady, transient, steady_np, transient_np
        data_steady = mi.TensorXf(steady_acc / completed_spp)
        data_transient = mi.TensorXf(transient_acc / completed_spp)

    print("Rendering complete!")
    print(f"Steady shape: {data_steady.shape}")

    if args.output_name:
        output_name = args.output_name
    else:
        output_name = os.path.splitext(os.path.basename(scene_path))[0]

    if args.output_dir is not None:
        base_output_dir = args.output_dir
    else:
        base_output_dir = os.path.join(os.path.dirname(__file__), "../renders_fwp")

    output_dir = os.path.join(base_output_dir, output_name)
    os.makedirs(output_dir, exist_ok=True)

    output_steady_exr_path = os.path.join(output_dir, "steady.exr")
    mi.util.write_bitmap(output_steady_exr_path, data_steady)
    print(f"Steady-state image saved to: {output_steady_exr_path}")

    output_steady_png_path = os.path.join(output_dir, "steady.png")
    mi.util.write_bitmap(output_steady_png_path, data_steady)
    print(f"Steady-state image saved to: {output_steady_png_path}")

    if not args.no_video:
        data_transient_tonemapped = mitr.vis.tonemap_transient(data_transient)
        data_transient_tonemapped = np.clip(np.array(data_transient_tonemapped), 0.0, 1.0)
        output_video_path = os.path.join(output_dir, "transient.mp4")
        mitr.vis.save_video(
            output_video_path,
            data_transient_tonemapped,
            axis_video=2,
        )
        print(f"Transient video saved to: {output_video_path}")
    else:
        print("Skipping transient video (--no-video)")

    output_npy_path = os.path.join(output_dir, "transient.npy")
    np.save(output_npy_path, np.array(data_transient))
    print(f"Transient data saved to: {output_npy_path}")

    if not args.no_plot:
        import matplotlib.pyplot as plt

        data_transient_clipped = dr.clip(data_transient, 0.0, args.clip_max)
        transient_np = np.array(data_transient_clipped)
        height, width = transient_np.shape[:2]

        pixel_x = args.plot_pixel_x if args.plot_pixel_x is not None else width // 2
        pixel_y = args.plot_pixel_y if args.plot_pixel_y is not None else height // 2

        pixel_x = max(0, min(pixel_x, width - 1))
        pixel_y = max(0, min(pixel_y, height - 1))

        print(f"Plotting transient at pixel ({pixel_x}, {pixel_y})")

        pixel_transient = transient_np[pixel_y, pixel_x, :, :]

        try:
            start_opl = film.start_opl()
            bin_width = film.bin_width_opl()
            num_bins = pixel_transient.shape[0]
            time_axis = np.arange(num_bins) * bin_width + start_opl
            xlabel = "Optical Path Length (m)"
        except Exception:
            time_axis = np.arange(pixel_transient.shape[0])
            xlabel = "Time bin"

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time_axis, pixel_transient[:, 0])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Radiance")
        ax.set_title(f"Transient at pixel ({pixel_x}, {pixel_y})")
        ax.grid(True, alpha=0.3)

        if args.log_scale:
            ax.set_yscale("log")
            ax.set_ylabel("Radiance (log scale)")

        output_plot_path = os.path.join(output_dir, f"transient_pixel_{pixel_x}_{pixel_y}.png")
        plt.savefig(output_plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Transient plot saved to: {output_plot_path}")
    else:
        print("Skipping transient plot (--no-plot)")


if __name__ == "__main__":
    main()
