"""
Training script for replicating the retro-cue RNN (Fig 4A loss curve).
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import List

import numpy as np
import torch

from src.analysis import plot_loss_curve
from src.model import ModelConfig, RetroCueRNN
from src.task import RetroCueDataset, TaskConfig
from src.utils import detect_plateau_epoch, eq21_loss, log_message, log_metric, set_seed


BASE_DIR = Path(__file__).resolve().parents[1]
FIG_PATH = BASE_DIR / "results" / "figures" / "fig4A_loss.png"
CKPT_DIR = BASE_DIR / "results" / "checkpoints"
LOG_JSON = BASE_DIR / "results" / "logs" / "training_summary.json"


def train_model(args) -> dict:
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    all_summaries = []
    loss_plot_saved = False

    for run in range(args.n_runs):
        set_seed(args.seed + run)
        task_cfg = TaskConfig(
            pre_delay=args.pre_delay,
            post_delay=args.post_delay,
            stim_duration=1,
            cue_duration=1,
            device=device,
        )
        model_cfg = ModelConfig(
            input_dim=task_cfg.input_dim,
            output_dim=task_cfg.output_dim,
            hidden_dim=args.hidden_dim,
            noise_std=args.noise_std,
            device=device,
        )
        dataset = RetroCueDataset(task_cfg, device=device)
        model = RetroCueRNN(model_cfg)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)

        losses: List[float] = []
        state_history: List[dict] = [copy.deepcopy(model.state_dict())]
        basis = dataset.basis_tensor

        log_message(f"[run {run}] Starting training on device={device}")
        for epoch in range(args.epochs):
            order = np.random.permutation(len(dataset))
            epoch_loss = 0.0
            for idx in order:
                sample = dataset[idx]
                inputs = sample["inputs"].unsqueeze(0)
                target_idx = torch.tensor([sample["target_idx"]], device=device)
                logits, _ = model(inputs, noise_std=args.noise_std)
                loss = eq21_loss(logits[:, -1, :], target_idx, basis)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(dataset)
            losses.append(epoch_loss)
            log_metric(f"run{run}_epoch_loss", epoch_loss)
            state_history.append(copy.deepcopy(model.state_dict()))

            if len(losses) >= 15:
                recent = np.array(losses[-15:])
                slope = np.polyfit(np.arange(len(recent)), recent, 1)[0]
                if slope > -2e-5 and recent.mean() < 0.0036:
                    log_message(f"[run {run}] Stopping early at epoch {epoch} (slope={slope:.6e}, loss={recent.mean():.6f})")
                    break

        losses = np.array(losses)
        plateau_idx = detect_plateau_epoch(losses, initial_loss=losses[0])
        log_message(f"[run {run}] Plateau detected at epoch {plateau_idx}")

        run_ckpt = CKPT_DIR / f"run{run}"
        run_ckpt.mkdir(parents=True, exist_ok=True)
        torch.save(state_history[0], run_ckpt / "untrained.pt")
        torch.save(state_history[min(plateau_idx + 1, len(state_history) - 1)], run_ckpt / "plateau.pt")
        torch.save(state_history[-1], run_ckpt / "trained.pt")

        if not loss_plot_saved:
            plot_loss_curve(losses, plateau_idx=plateau_idx, save_path=str(FIG_PATH))
            loss_plot_saved = True

        summary = {
            "run": run,
            "device": device,
            "epochs_ran": len(losses),
            "plateau_epoch": int(plateau_idx),
            "final_loss": float(losses[-1]),
            "loss_history": losses.tolist(),
            "task_config": task_cfg.__dict__,
            "model_config": model_cfg.__dict__,
        }
        all_summaries.append(summary)
        log_message(f"[run {run}] Training complete: final loss {losses[-1]:.6f}")

    LOG_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_JSON, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    return {"runs": all_summaries}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=200)
    parser.add_argument("--noise-std", type=float, default=0.07)
    parser.add_argument("--pre-delay", type=int, default=7)
    parser.add_argument("--post-delay", type=int, default=7)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-runs", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(args)

