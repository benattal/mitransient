import argparse
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process rendered transient output")
    parser.add_argument("scene_file", type=str, help="Path to the transient .npy file")
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="Device used for RGB reduction (production runs pass CUDA explicitly)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("processing")

    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA preprocessing requested but CUDA is unavailable")
    print(f"RGB reduction device: {device}")

    path = Path(args.scene_file)
    transient = torch.from_numpy(np.load(path)).to(device)
    reduced = transient.sum(dim=-1)
    transformed = reduced.reshape(-1, reduced.shape[-1]).cpu().numpy()

    output_path = path.parent / f"{path.stem}_processed.npy"
    np.save(output_path, transformed)
    print(f"Processed transient saved to: {output_path}")


if __name__ == "__main__":
    main()
