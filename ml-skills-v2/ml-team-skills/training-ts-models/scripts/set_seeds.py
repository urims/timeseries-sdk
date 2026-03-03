#!/usr/bin/env python3
"""Set all random seeds for reproducible training.

Usage:
    python scripts/set_seeds.py [--seed 42]

Sets seeds for: Python random, NumPy, PyTorch (CPU + CUDA), and PYTHONHASHSEED.
Enables deterministic mode (may slow training slightly).
"""

import argparse
import os
import random
import sys

def set_all_seeds(seed: int = 42) -> dict:
    """Set all random seeds and return metadata about what was configured."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    metadata = {"seed": seed, "python_random": True, "pythonhashseed": True}

    try:
        import numpy as np
        np.random.seed(seed)
        metadata["numpy"] = True
    except ImportError:
        metadata["numpy"] = False

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        metadata["pytorch"] = True
        metadata["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        metadata["pytorch"] = False

    return metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set reproducibility seeds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    result = set_all_seeds(args.seed)
    print(f"Seeds set to {args.seed}")
    for key, value in result.items():
        if key != "seed":
            print(f"  {key}: {value}")
