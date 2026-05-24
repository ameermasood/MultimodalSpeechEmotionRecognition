"""General utilities used by training entry points."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def to_abs(path: str) -> str:
    """Expand user markers and return an absolute path."""
    return os.path.abspath(os.path.expanduser(path))


def safe_makedirs(path: str) -> None:
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
