"""
Stress test across post-cue delay lengths (Fig 5B).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.analysis import compute_geometry
from src.model import ModelConfig, RetroCueRNN
from src.task import RetroCueDataset, TaskConfig
from src.utils import eq21_loss, log_message, log_metric, set_seed


BASE_DIR = Path(__file__).resolve().parents[1]
FIG_PATH = BASE_DIR / "results" / "figures" / "fig5B_alignment_vs_delay.png"


def train_for_delay(task_cfg: TaskConfig, model_cfg: ModelConfig, epochs: int, lr: float, noise_std: float) -> RetroCueRNN:
    dataset = RetroCueDataset(task_cfg, device=model_cfg.device)
    model = RetroCueRNN(model_cfg)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    basis = dataset.basis_tensor
    losses: List[float] = []

    for epoch in range(epochs):
        order = np.random.permutation(len(dataset))
        epoch_loss = 0.0
        for idx in order:
            sample = dataset[idx]
            inputs = sample["inputs"].unsqueeze(0)
            target_idx = torch.tensor([sample["target_idx"]], device=model_cfg.device)
            logits, _ = model(inputs, noise_std=noise_std)
            loss = eq21_loss(logits[:, -1, :], target_idx, basis)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(dataset))
        if len(losses) >= 15:
            recent = np.array(losses[-15:])
            slope = np.polyfit(np.arange(len(recent)), recent, 1)[0]
            if slope > -2e-5 and recent.mean() < 0.0036:
                break
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--post-delays", type=int, nargs="+", default=[0, 2, 4, 6, 7])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=200)
    parser.add_argument("--noise-std", type=float, default=0.0)  # per Methods for stress test
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-runs", type=int, default=3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ai_pre_all = []
    ai_post_all = []
    for post_delay in args.post_delays:
        ai_pre_runs = []
        ai_post_runs = []
        for r in range(args.n_runs):
            set_seed(args.seed + r)
            task_cfg = TaskConfig(post_delay=post_delay, device=device)
            model_cfg = ModelConfig(
                input_dim=task_cfg.input_dim,
                output_dim=task_cfg.output_dim,
                hidden_dim=args.hidden_dim,
                noise_std=args.noise_std,
                device=device,
            )
            model = train_for_delay(task_cfg, model_cfg, epochs=args.epochs, lr=args.lr, noise_std=args.noise_std)
            dataset = RetroCueDataset(task_cfg, device=device)
            hidden, metadata = [], []
            for idx in range(len(dataset)):
                sample = dataset[idx]
                with torch.no_grad():
                    _, h = model(sample["inputs"].unsqueeze(0), noise_std=0.0)
                hidden.append(h.squeeze(0).cpu().numpy())
                metadata.append(sample["metadata"])
            hidden_arr = np.stack(hidden)
            geom_pre = compute_geometry(hidden_arr, metadata, time_idx=task_cfg.pre_cue_time_index)
            geom_post = compute_geometry(hidden_arr, metadata, time_idx=task_cfg.post_cue_time_index)
            ai_pre_runs.append(geom_pre["ai"])
            ai_post_runs.append(geom_post["ai"])
            log_metric(f"delay{post_delay}_run{r}_AI_pre", geom_pre["ai"])
            log_metric(f"delay{post_delay}_run{r}_AI_post", geom_post["ai"])
        ai_pre_all.append(ai_pre_runs)
        ai_post_all.append(ai_post_runs)

    ai_pre_all = np.array(ai_pre_all)
    ai_post_all = np.array(ai_post_all)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(args.post_delays, ai_pre_all.mean(axis=1), marker="o", color="red", label="Pre-cue")
    ax.plot(args.post_delays, ai_post_all.mean(axis=1), marker="^", color="blue", label="Post-cue")
    for i, d in enumerate(args.post_delays):
        ax.scatter(np.full(args.n_runs, d) - 0.05, ai_pre_all[i], color="red", alpha=0.4, s=15)
        ax.scatter(np.full(args.n_runs, d) + 0.05, ai_post_all[i], color="blue", alpha=0.4, s=15)
    ax.set_xlabel("Post-cue delay length (cycles)")
    ax.set_ylabel("Alignment Index")
    ax.set_title("Fig 5B: AI vs post-cue delay")
    ax.set_ylim(0, 1.05)
    ax.legend()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=300)
    plt.close(fig)

    log_message("Stress test complete")


if __name__ == "__main__":
    main()

