#!/usr/bin/env python3
"""Capture reproducibility metadata for the current environment.

Usage:
    python scripts/pin_environment.py --output reproducibility_metadata.json

Records:
    - Python version
    - Installed package versions (pip freeze)
    - PyTorch and CUDA versions (if available)
    - Git commit hash and dirty status
    - Timestamp
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone

def capture_metadata():
    """Capture full environment metadata."""
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
    }

    # Git info
    try:
        metadata["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        metadata["git_dirty"] = bool(dirty)
    except (subprocess.CalledProcessError, FileNotFoundError):
        metadata["git_commit"] = "not_a_git_repo"
        metadata["git_dirty"] = None

    # PyTorch and CUDA
    try:
        import torch
        metadata["pytorch_version"] = torch.__version__
        metadata["cuda_available"] = torch.cuda.is_available()
        metadata["cuda_version"] = torch.version.cuda if torch.cuda.is_available() else None
    except ImportError:
        metadata["pytorch_version"] = None

    # NumPy
    try:
        import numpy as np
        metadata["numpy_version"] = np.__version__
    except ImportError:
        metadata["numpy_version"] = None

    # Pip freeze
    try:
        freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], stderr=subprocess.DEVNULL
        ).decode().strip()
        metadata["pip_freeze"] = freeze.split("\n") if freeze else []
    except subprocess.CalledProcessError:
        metadata["pip_freeze"] = []

    return metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture environment metadata")
    parser.add_argument("--output", default="reproducibility_metadata.json",
                        help="Output file path")
    args = parser.parse_args()

    meta = capture_metadata()

    with open(args.output, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Environment metadata saved to {args.output}")
    print(f"  Python: {meta['python_version'].split()[0]}")
    print(f"  Git: {meta['git_commit'][:8]}{'*' if meta.get('git_dirty') else ''}")
    print(f"  PyTorch: {meta.get('pytorch_version', 'not installed')}")
    print(f"  Packages: {len(meta.get('pip_freeze', []))} recorded")
