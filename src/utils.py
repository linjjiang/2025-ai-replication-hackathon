"""
Utility functions for the retro-cue RNN replication.
Derived from the methods described in Piwek et al. (2023).
"""

from __future__ import annotations

import datetime
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d


ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "results/logs/replication_log.txt"


def log_message(message: str, level: str = "INFO") -> None:
    """Append a message to the replication log with simple tagging."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().isoformat()
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{stamp} [{level.upper()}] {message}\n")


def log_metric(name: str, value: float, level: str = "INFO") -> None:
    """Structured logging for scalar metrics."""
    log_message(f"{name}={value:.6f}", level=level)


def angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest signed angular difference (radians) between a and b."""
    diff = (a - b + np.pi) % (2 * np.pi) - np.pi
    return diff


def circular_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Alias for clarity."""
    return np.abs(angle_diff(a, b))


def von_mises_tuning(centers: np.ndarray, stimulus: float, kappa: float = 5.0) -> np.ndarray:
    """Compute von Mises activations for a given stimulus (radians)."""
    return np.exp(kappa * np.cos(stimulus - centers)) / (2 * np.pi * np.i0(kappa))


def make_colour_basis(num_units: int = 17) -> np.ndarray:
    """Equally spaced preferred colours for input/output channels."""
    return np.linspace(0, 2 * np.pi, num_units, endpoint=False)


def bin_colour(angle: float, num_bins: int = 4) -> int:
    """Bin a colour angle into one of num_bins equal segments."""
    wrapped = angle % (2 * np.pi)
    edges = np.linspace(0, 2 * np.pi, num_bins + 1)
    # last bin inclusive of 2pi
    return int(np.digitize(wrapped, edges, right=False) - 1)


def smooth_and_derivative(loss_history: np.ndarray, sigma: float = 4.0) -> Tuple[np.ndarray, np.ndarray]:
    """Gaussian smooth the loss and compute its derivative."""
    smoothed = gaussian_filter1d(loss_history, sigma=sigma, mode="nearest")
    derivative = np.gradient(smoothed)
    return smoothed, derivative


def detect_plateau_epoch(loss_history: np.ndarray, initial_loss: float) -> int:
    """
    Detect the first plateau epoch as described in Methods:
    first local maximum of the derivative after leaving 5% band of initial loss.
    """
    smoothed, derivative = smooth_and_derivative(loss_history)
    threshold_loss = initial_loss * 0.95
    valid_epochs = np.where(loss_history < threshold_loss)[0]
    if valid_epochs.size == 0:
        return int(len(loss_history) - 1)
    start_idx = valid_epochs[0]
    for i in range(start_idx + 1, len(derivative) - 1):
        if derivative[i - 1] < derivative[i] > derivative[i + 1]:
            return i
    return int(len(loss_history) - 1)


def save_json(data: dict, path: os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def ensure_dir(path: os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def eq21_loss(logits: torch.Tensor, target_idx: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """
    Implement Eq. 2.1 from the paper: weighted MSE scaled by circular distance.

    Args:
        logits: (batch, output_dim)
        target_idx: (batch,) integer indices of correct output channel
        basis: (output_dim,) tensor of preferred angles (radians)
    """
    probs = F.softmax(logits, dim=-1)
    target_onehot = F.one_hot(target_idx, num_classes=logits.shape[-1]).float().to(logits.device)
    target_angles = basis.to(logits.device)[target_idx]
    # broadcast angular distance to all output units per sample
    angle_term = (basis.to(logits.device)[None, :] - target_angles[:, None])
    loss = torch.mean(((target_onehot - probs) * angle_term) ** 2)
    return loss

