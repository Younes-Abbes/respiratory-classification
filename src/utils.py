"""Small reusable utilities: seeding, logging, config loading."""

import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and torch (CPU + CUDA) for reproducibility.

    Note: for full determinism on CUDA, also set
        torch.use_deterministic_algorithms(True)
    but this is incompatible with several AST kernels — we don't enforce it.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(path: str | Path) -> dict:
    """Load a YAML configuration file and return the dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class AverageMeter:
    """Running average of a scalar value, useful for logging the loss."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.value = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.value = value
        self.sum += value * n
        self.count += n

    @property
    def average(self) -> float:
        return self.sum / max(self.count, 1)


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return (total_parameters, trainable_parameters)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
