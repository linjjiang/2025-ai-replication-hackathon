"""
Task construction utilities for the retro-cue RNN replication.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

from .utils import make_colour_basis, von_mises_tuning


@dataclass
class TaskConfig:
    pre_delay: int = 7
    post_delay: int = 7
    stim_duration: int = 1
    cue_duration: int = 1
    num_colours: int = 16  # unique colours in dataset
    tuning_units: int = 17  # von Mises channels per location and output
    kappa: float = 5.0
    device: str = "cpu"

    @property
    def input_dim(self) -> int:
        return 2 + 2 * self.tuning_units

    @property
    def output_dim(self) -> int:
        return self.tuning_units

    @property
    def total_time(self) -> int:
        return self.stim_duration + self.pre_delay + self.cue_duration + self.post_delay

    @property
    def pre_cue_time_index(self) -> int:
        """0-indexed timestep at end of pre-cue delay."""
        return self.stim_duration + self.pre_delay - 1

    @property
    def post_cue_time_index(self) -> int:
        """0-indexed timestep at end of post-cue delay."""
        return self.total_time - 1


class RetroCueDataset:
    """
    Deterministic dataset enumerating all colour pairs and cue locations.
    Each sample is a single trial (batch size 1 during training for fidelity).
    """

    def __init__(self, config: TaskConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        self.colours = np.linspace(0, 2 * np.pi, config.num_colours, endpoint=False)
        self.basis = make_colour_basis(config.tuning_units)
        self.trials = self._enumerate_trials()
        self.basis_tensor = torch.from_numpy(self.basis).float().to(self.device)

    def _encode_colour(self, angle: float) -> np.ndarray:
        activations = von_mises_tuning(self.basis, angle, kappa=self.config.kappa)
        return activations / activations.max()  # rescale to [0,1] (Methods)

    def _enumerate_trials(self) -> List[Dict]:
        trials = []
        for c1 in self.colours:
            for c2 in self.colours:
                for cue_loc in [0, 1]:  # 0 -> location1, 1 -> location2
                    trials.append({"c1": c1, "c2": c2, "cue": cue_loc})
        return trials

    def __len__(self) -> int:
        return len(self.trials)

    def __getitem__(self, idx: int) -> Dict:
        return self.build_trial(self.trials[idx])

    def build_trial(self, spec: Dict) -> Dict:
        cfg = self.config
        time = cfg.total_time
        inp = np.zeros((time, cfg.input_dim), dtype=np.float32)
        # stimulus period (both items)
        stim_vec1 = self._encode_colour(spec["c1"])
        stim_vec2 = self._encode_colour(spec["c2"])
        inp[0, 0] = 0.0  # cue channels inactive
        inp[0, 1] = 0.0
        inp[0, 2 : 2 + cfg.tuning_units] = stim_vec1
        inp[0, 2 + cfg.tuning_units :] = stim_vec2

        # cue
        cue_start = cfg.stim_duration + cfg.pre_delay
        cue_end = cue_start + cfg.cue_duration
        inp[cue_start:cue_end, spec["cue"]] = 1.0

        # torch tensors
        inputs = torch.from_numpy(inp).to(self.device)
        # target: index of output channel closest to cued colour
        target_angle = spec["c1"] if spec["cue"] == 0 else spec["c2"]
        target_idx = int(np.argmin(np.abs(self.basis - target_angle)))
        metadata = {
            "c1": spec["c1"],
            "c2": spec["c2"],
            "cue": spec["cue"],
            "target_angle": target_angle,
            "target_idx": target_idx,
        }
        return {"inputs": inputs, "target_idx": target_idx, "metadata": metadata}


def generate_repetitions(dataset: RetroCueDataset, repetitions: int = 1) -> List[Dict]:
    """Expand deterministic dataset by repeating trials (noise will add variability)."""
    return [dataset[i % len(dataset)] for i in range(len(dataset) * repetitions)]

