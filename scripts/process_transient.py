import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

print('processing')

# Parse command-line arguments
parser = argparse.ArgumentParser(description='rendered transient')
parser.add_argument('scene_file', type=str,
                    help='Path to the transient')
args = parser.parse_args()

path = Path(args.scene_file)

transient = np.load(path).sum(axis=-1)

transformed = transient.reshape(-1,transient.shape[-1])

output_path = f'{path.parent}/{path.stem}_processed.npy'
np.save(output_path, transformed)
print(f"Processed transient saved to: {output_path}")