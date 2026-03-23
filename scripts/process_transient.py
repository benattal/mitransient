import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process rendered transient output")
    parser.add_argument("scene_file", type=str, help="Path to the transient .npy file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("processing")

    path = Path(args.scene_file)
    transient = np.load(path).sum(axis=-1)
    transformed = transient.reshape(-1, transient.shape[-1])

    output_path = path.parent / f"{path.stem}_processed.npy"
    np.save(output_path, transformed)
    print(f"Processed transient saved to: {output_path}")


if __name__ == "__main__":
    main()
