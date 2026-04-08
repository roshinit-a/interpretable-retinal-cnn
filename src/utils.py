"""
utils.py — Shared utilities for the interpretable-retinal-cnn project.

Currently provides:
    seed_everything(seed)  — fix all RNG sources for reproducibility.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """
    Fix all random number generator seeds for fully reproducible training.

    Covers:
        - Python's built-in ``random`` module
        - NumPy
        - PyTorch CPU ops
        - PyTorch CUDA ops (all devices)
        - cuDNN algorithm selection

    Args:
        seed: Integer seed value.  Default 42.

    Note on performance:
        ``torch.backends.cudnn.deterministic = True`` disables cuDNN's
        non-deterministic algorithms (e.g. atomic reductions in some
        convolution back-props).  This may slow training by ~5-15% on
        some architectures.  Disable it if speed is more important than
        bit-exact reproducibility.

    Example::

        from src.utils import seed_everything
        seed_everything(42)
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)          # for multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"[reproducibility] All RNG seeds fixed to {seed}.")
