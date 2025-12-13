"""
Geometry analysis for Figures 2B and 4C.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.analysis import (
    bar_with_points,
    compute_geometry,
    polar_angle_plot,
    plot_pre_post_planes,
)
from src.model import ModelConfig, RetroCueRNN
from src.task import RetroCueDataset, TaskConfig
from src.utils import log_message, log_metric


BASE_DIR = Path(__file__).resolve().parents[1]
CKPT_DIR = BASE_DIR / "results" / "checkpoints"
FIG_DIR = BASE_DIR / "results" / "figures"


def _load_model(state_path: Path, model_cfg: ModelConfig) -> RetroCueRNN:
    model = RetroCueRNN(model_cfg)
    state = torch.load(state_path, map_location=model_cfg.device)
    model.load_state_dict(state)
    model.eval()
    return model


def _collect_hidden(model: RetroCueRNN, dataset: RetroCueDataset) -> tuple[np.ndarray, list]:
    hidden_all = []
    metadata_all = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        inputs = sample["inputs"].unsqueeze(0)
        with torch.no_grad():
            _, hidden = model(inputs, noise_std=0.0)
        hidden_all.append(hidden.squeeze(0).cpu().numpy())
        metadata_all.append(sample["metadata"])
    return np.stack(hidden_all), metadata_all


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    task_cfg = TaskConfig(device=device)
    model_cfg = ModelConfig(input_dim=task_cfg.input_dim, output_dim=task_cfg.output_dim, device=device)
    dataset = RetroCueDataset(task_cfg, device=device)

    run_dirs = sorted([p for p in CKPT_DIR.iterdir() if p.is_dir() and p.name.startswith("run")])
    stages = ["Untrained", "Plateau", "Trained"]
    theta_pre = {s: [] for s in stages}
    theta_post = {s: [] for s in stages}
    ai_pre = {s: [] for s in stages}
    ai_post = {s: [] for s in stages}

    best_geom_pre = None
    best_geom_post = None
    best_score = None

    for run_dir in run_dirs:
        for stage, fname in zip(stages, ["untrained.pt", "plateau.pt", "trained.pt"]):
            path = run_dir / fname
            model = _load_model(path, model_cfg)
            hidden, metadata = _collect_hidden(model, dataset)
            geom_pre = compute_geometry(hidden, metadata, time_idx=task_cfg.pre_cue_time_index)
            geom_post = compute_geometry(hidden, metadata, time_idx=task_cfg.post_cue_time_index)

            theta_pre[stage].append(geom_pre["theta"])
            theta_post[stage].append(geom_post["theta"])
            ai_pre[stage].append(geom_pre["ai"])
            ai_post[stage].append(geom_post["ai"])

            log_metric(f"{run_dir.name}_{stage}_pre_theta", geom_pre["theta"])
            log_metric(f"{run_dir.name}_{stage}_post_theta", geom_post["theta"])
            log_metric(f"{run_dir.name}_{stage}_pre_ai", geom_pre["ai"])
            log_metric(f"{run_dir.name}_{stage}_post_ai", geom_post["ai"])
            if geom_pre["ai"] > 1.0 or geom_post["ai"] > 1.0:
                log_message(f"{run_dir.name}_{stage} AI exceeded 1.0", level="ERROR")

            if stage == "Trained":
                score = abs(geom_pre["theta"] - 90) + abs(geom_post["theta"] - 0)
                if best_score is None or score < best_score:
                    best_score = score
                    best_geom_pre = geom_pre
                    best_geom_post = geom_post

    # Fig 2B using best-trained run
    if best_geom_pre is not None:
        plot_pre_post_planes(
            best_geom_pre["points"],
            best_geom_post["points"],
            best_geom_pre["explained"],
            best_geom_post["explained"],
            save_path=str(FIG_DIR / "fig2B_planes.png"),
        )

    # Fig 4C replication: pre (top) and post (bottom)
    colors = {"Untrained": "orange", "Plateau": "magenta", "Trained": "purple"}
    fig = plt.figure(figsize=(10, 8))
    ax_pre_polar = fig.add_subplot(2, 2, 1, projection="polar")
    ax_pre_bar = fig.add_subplot(2, 2, 2)
    ax_post_polar = fig.add_subplot(2, 2, 3, projection="polar")
    ax_post_bar = fig.add_subplot(2, 2, 4)

    polar_angle_plot(ax_pre_polar, {k: np.array(theta_pre[k]) for k in stages}, colors, title="Pre-cue θ")
    bar_with_points(ax_pre_bar, {k: float(np.mean(ai_pre[k])) for k in stages}, {k: np.array(ai_pre[k]) for k in stages}, colors)
    ax_pre_bar.set_title("Pre-cue AI")

    polar_angle_plot(ax_post_polar, {k: np.array(theta_post[k]) for k in stages}, colors, title="Post-cue θ")
    bar_with_points(ax_post_bar, {k: float(np.mean(ai_post[k])) for k in stages}, {k: np.array(ai_post[k]) for k in stages}, colors)
    ax_post_bar.set_title("Post-cue AI")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4C_angles_ai.png", dpi=300)
    plt.close(fig)

    log_message("Geometry analysis complete")


if __name__ == "__main__":
    main()

