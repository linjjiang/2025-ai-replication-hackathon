import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.linalg import orthogonal_procrustes
from scipy.ndimage import gaussian_filter1d
from scipy.special import i0
from sklearn.decomposition import PCA
from torch import nn
from torch.optim import RMSprop


# -----------------------------
# Configuration containers
# -----------------------------


@dataclass
class RetroCueConfig:
    """
    Container for task, model and training hyperparameters.
    Defaults mirror the Methods section of Piwek et al. (2023).
    """

    # task and encoding
    n_stim: int = 16  # number of discrete colours
    n_col_units: int = 17  # number of colour-tuned units per location
    kappa: float = 5.0  # von Mises concentration
    stim_dur: int = 1
    delay1: int = 7
    cue_dur: int = 1
    delay2: int = 7
    postcue_delays: Optional[List[int]] = None  # used for stress test grid

    # model
    n_rec: int = 200
    n_out: int = 17
    init_scale: float = 1.0
    noise_sigma: float = 0.07
    noise_timesteps: str = "all"  # "all" or list of ints

    # training
    learning_rate: float = 1e-4
    max_epochs: int = 1500
    batch_size: int = 1
    slope_window: int = 15
    slope_threshold: float = -2e-5
    loss_threshold: float = 0.0036
    smooth_sd: float = 3.0
    device: str = "cpu"
    seed: int = 1029

    # analysis / binning
    colour_bins: int = 4

    def seq_len(self) -> int:
        return self.stim_dur + self.delay1 + self.cue_dur + self.delay2

    def noise_indices(self) -> np.ndarray:
        if self.noise_timesteps == "all":
            return np.arange(self.seq_len())
        if self.noise_timesteps is None:
            return np.array([], dtype=int)
        return np.array(self.noise_timesteps, dtype=int)

    def phi(self) -> np.ndarray:
        return np.linspace(-np.pi, np.pi, self.n_col_units + 1)[:-1]


# -----------------------------
# Utilities
# -----------------------------


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def circular_distance(a: np.ndarray, b: float) -> np.ndarray:
    """Signed circular distance in [-pi, pi]."""
    return np.angle(np.exp(1j * (a - b)))


def von_mises_activation(angle: float, centers: np.ndarray, kappa: float) -> np.ndarray:
    """Von Mises tuning curve rescaled to [0, 1]. Eq. (1.1) in the paper."""
    numer = np.exp(kappa * np.cos(angle - centers))
    denom = 2 * np.pi * i0(kappa)
    raw = numer / denom
    # rescale so the peak is 1.0
    return raw / raw.max()


# -----------------------------
# Model definition
# -----------------------------


class RNNModel(nn.Module):
    """
    Vanilla Elman RNN with ReLU nonlinearity, orthogonal recurrent init and
    Xavier-scaled input/output layers, matching the reference implementation.
    """

    def __init__(self, cfg: RetroCueConfig):
        super().__init__()
        self.cfg = cfg
        self.n_rec = cfg.n_rec
        self.n_inp = cfg.n_col_units * 2 + 2
        self.n_out = cfg.n_out
        self.register_buffer("noise_indices", torch.tensor(cfg.noise_indices(), dtype=torch.long))

        # layers
        self.inp = nn.Linear(self.n_inp, self.n_rec)
        nn.init.xavier_uniform_(self.inp.weight, gain=cfg.init_scale)
        nn.init.zeros_(self.inp.bias)

        self.Wrec = nn.Parameter(torch.empty((self.n_rec, self.n_rec)))
        nn.init.orthogonal_(self.Wrec)

        self.relu = nn.ReLU()

        self.out = nn.Linear(self.n_rec, self.n_out)
        nn.init.xavier_uniform_(self.out.weight, gain=cfg.init_scale)
        nn.init.zeros_(self.out.bias)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs: torch.Tensor, track_states: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        inputs: (T, batch, n_inp)
        returns: (batch, n_out), hidden_states (T, batch, n_rec)
        """
        T, batch, _ = inputs.shape
        hidden = torch.zeros((batch, self.n_rec), device=inputs.device)
        states = [] if track_states else None

        for t in range(T):
            if t in self.noise_indices:
                noise = torch.randn_like(hidden) * self.cfg.noise_sigma
            else:
                noise = torch.zeros_like(hidden)
            hidden = self.relu(self.inp(inputs[t]) + hidden @ self.Wrec.T + noise)
            if track_states:
                states.append(hidden.detach().clone())

        logits = self.out(hidden)
        probs = self.softmax(logits)
        if track_states:
            states = torch.stack(states, dim=0)
        return probs, states


# -----------------------------
# Task generation
# -----------------------------


class RetroCueTask:
    """Creates the in-silico retro-cue dataset (stimulus, delay1, cue, delay2)."""

    def __init__(self, cfg: RetroCueConfig):
        self.cfg = cfg
        self.centers = cfg.phi()

    def _encode_colour(self, angle: float) -> np.ndarray:
        return von_mises_activation(angle, self.centers, self.cfg.kappa)

    def _make_trial(self, c1: float, c2: float, cue_loc: int) -> np.ndarray:
        """
        Build a single trial time series.
        cue_loc: 0 for location 1 (top), 1 for location 2 (bottom)
        """
        seq = np.zeros((self.cfg.seq_len(), self.cfg.n_col_units * 2 + 2), dtype=np.float32)

        # stimulus epoch
        c1_enc = self._encode_colour(c1)
        c2_enc = self._encode_colour(c2)
        seq[0, 2 : 2 + self.cfg.n_col_units] = c1_enc
        seq[0, 2 + self.cfg.n_col_units :] = c2_enc

        # cue epoch
        cue_start = self.cfg.stim_dur + self.cfg.delay1
        seq[cue_start : cue_start + self.cfg.cue_dur, cue_loc] = 1.0

        # rest already zeros
        return seq

    def generate_dataset(self) -> Dict[str, torch.Tensor]:
        """
        Full factorial set: 16 colours at each location x 2 cues = 512 trials.
        Targets are scalar cued colour angles.
        """
        trials = []
        targets = []
        c1_all, c2_all, loc_all = [], [], []
        colours = np.linspace(-np.pi, np.pi, self.cfg.n_stim + 1)[:-1]

        for c1 in colours:
            for c2 in colours:
                for cue_loc in (0, 1):
                    seq = self._make_trial(c1, c2, cue_loc)
                    trials.append(seq)
                    targets.append(c1 if cue_loc == 0 else c2)
                    c1_all.append(c1)
                    c2_all.append(c2)
                    loc_all.append(cue_loc)

        inputs = torch.tensor(np.stack(trials), dtype=torch.float32).transpose(0, 1)
        targets = torch.tensor(np.array(targets), dtype=torch.float32)
        c1_all = torch.tensor(np.array(c1_all), dtype=torch.float32)
        c2_all = torch.tensor(np.array(c2_all), dtype=torch.float32)
        loc_all = torch.tensor(np.array(loc_all), dtype=torch.long)
        return {
            "inputs": inputs,  # (T, trials, channels)
            "targets": targets,
            "c1": c1_all,
            "c2": c2_all,
            "loc": loc_all,
        }


# -----------------------------
# Training and checkpoints
# -----------------------------


def custom_loss(cfg: RetroCueConfig, output: torch.Tensor, target_angle: torch.Tensor) -> torch.Tensor:
    """
    Eq. (2.1): mean squared product of (target - output) and circular distance.
    """
    phi = torch.tensor(cfg.phi(), device=output.device)
    # one-hot target
    target_onehot = torch.zeros_like(output)
    # map target angle to closest centre
    idx = torch.argmin(torch.abs(torch.angle(torch.exp(1j * (phi - target_angle.unsqueeze(-1))))), dim=-1)
    target_onehot[torch.arange(output.shape[0]), idx] = 1.0
    circ_dist = torch.angle(torch.exp(1j * (phi - target_angle.unsqueeze(-1))))
    return ((circ_dist * (target_onehot - output)) ** 2).mean()


def loss_slope(loss_vals: np.ndarray, window: int) -> float:
    """Linear slope over a window."""
    if len(loss_vals) < window:
        return 0.0
    x = np.arange(window)
    y = loss_vals[-window:]
    a, _ = np.polyfit(x, y, 1)
    return a


def detect_plateau(loss_history: np.ndarray, cfg: RetroCueConfig) -> Optional[int]:
    """
    Find first local minimum in derivative of smoothed loss,
    excluding early epochs where loss is close to its initial value.
    """
    if len(loss_history) < cfg.slope_window + 2:
        return None
    smoothed = gaussian_filter1d(loss_history, cfg.smooth_sd)
    deriv = np.diff(smoothed)
    margin_mask = smoothed < (smoothed[0] * 0.95)
    candidates = np.where(
        (np.r_[False, (deriv[1:] > deriv[:-1]) & (deriv[:-1] < 0), False]) & margin_mask
    )[0]
    if len(candidates) == 0:
        return None
    return int(candidates[0])


@dataclass
class TrainingResult:
    model: RNNModel
    loss_history: List[float]
    plateau_epoch: Optional[int]
    checkpoints: Dict[str, Dict[str, torch.Tensor]]


def train(cfg: RetroCueConfig, dataset: Dict[str, torch.Tensor]) -> TrainingResult:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    model = RNNModel(cfg).to(device)
    optim = RMSprop(model.parameters(), lr=cfg.learning_rate)

    inputs = dataset["inputs"].to(device)
    targets = dataset["targets"].to(device)

    n_trials = targets.shape[0]
    order = torch.arange(n_trials)

    loss_history: List[float] = []
    checkpoints: Dict[str, Dict[str, torch.Tensor]] = {"untrained": model.state_dict()}
    plateau_epoch: Optional[int] = None

    for epoch in range(cfg.max_epochs):
        perm = order[torch.randperm(n_trials)]
        epoch_losses = []
        for idx in perm:
            trial_inp = inputs[:, idx, :].unsqueeze(1)
            output, _ = model(trial_inp)
            loss = custom_loss(cfg, output, targets[idx : idx + 1])
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_losses.append(float(loss.detach().cpu()))

        mean_loss = float(np.mean(epoch_losses))
        loss_history.append(mean_loss)

        # Plateau detection for checkpointing
        if plateau_epoch is None:
            plateau_epoch = detect_plateau(np.array(loss_history), cfg)
            if plateau_epoch is not None:
                checkpoints["plateau"] = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

        # convergence criterion: slope near zero and low absolute loss
        slope = loss_slope(np.array(loss_history), cfg.slope_window)
        window_ok = len(loss_history) >= cfg.slope_window
        recent = np.array(loss_history[-cfg.slope_window :]) if window_ok else np.array([])
        loss_ok = window_ok and np.all(recent < cfg.loss_threshold)
        slope_ok = window_ok and (cfg.slope_threshold <= slope <= 0)
        if slope_ok and loss_ok:
            break

    checkpoints["trained"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return TrainingResult(model=model, loss_history=loss_history, plateau_epoch=plateau_epoch, checkpoints=checkpoints)


# -----------------------------
# Analysis
# -----------------------------


def run_model(model: RNNModel, dataset: Dict[str, torch.Tensor], cfg: RetroCueConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Run model without gradient tracking and return hidden trajectories (trials x time x units)."""
    model.eval()
    with torch.no_grad():
        outputs, hidden = model(dataset["inputs"].to(next(model.parameters()).device), track_states=True)
    # hidden: (T, trials, n_rec)
    return outputs.cpu().numpy(), hidden.permute(1, 0, 2).cpu().numpy()


def bin_hidden_states(hidden: np.ndarray, dataset: Dict[str, torch.Tensor], cfg: RetroCueConfig, cue_only: bool = True) -> Dict[str, np.ndarray]:
    """
    Bin hidden activity into B colour bins x 2 locations for endpoints of delay1 and delay2.
    Returns dict with keys 'delay1' and 'delay2' each shaped (B*2, n_rec).
    """
    B = cfg.colour_bins
    loc = dataset["loc"].numpy()
    c1 = dataset["c1"].numpy()
    c2 = dataset["c2"].numpy()
    phi = cfg.phi()
    bin_edges = np.linspace(-np.pi, np.pi, B + 1)

    def bin_angle(angle):
        return np.digitize(angle, bin_edges) - 1

    bins = []
    labels = []
    for trial_idx in range(hidden.shape[0]):
        cue = loc[trial_idx]
        cued_angle = c1[trial_idx] if cue == 0 else c2[trial_idx]
        uncued_angle = c2[trial_idx] if cue == 0 else c1[trial_idx]
        bins.append((bin_angle(cued_angle), bin_angle(uncued_angle)))
        labels.append(cue)

    bins = np.array(bins)
    labels = np.array(labels)

    def mean_for_delay(delay_idx: int) -> np.ndarray:
        means = []
        for location in (0, 1):
            for b in range(B):
                mask = (labels == location) & (bins[:, 0] == b)  # cued item bins
                means.append(hidden[mask, delay_idx, :].mean(axis=0))
        return np.stack(means, axis=0)

    delay1_idx = cfg.stim_dur + cfg.delay1 - 1
    delay2_idx = cfg.seq_len() - 1

    return {
        "delay1": mean_for_delay(delay1_idx),
        "delay2": mean_for_delay(delay2_idx),
    }


def plane_normals(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit PCA (2D) to each location block (size B x n_rec) then compute normals from corrected PC axes."""
    pca = PCA(n_components=2)
    pca.fit(points)
    comps = pca.components_
    # ensure orthogonality and deterministic ordering
    comp1 = comps[0] / np.linalg.norm(comps[0])
    comp2 = comps[1] - np.dot(comps[1], comp1) * comp1
    comp2 = comp2 / np.linalg.norm(comp2)
    # normal is the vector orthogonal to the 2D subspace spanned by comp1/comp2
    # compute via SVD to avoid 3D-only cross-product limitations
    _, _, vt = np.linalg.svd(np.stack([comp1, comp2], axis=0))
    normal = vt[-1]
    return normal, np.stack([comp1, comp2], axis=0)


def plane_angle(A: np.ndarray, B: np.ndarray) -> float:
    """Angle between two plane normals in degrees, signed."""
    nA, _ = plane_normals(A)
    nB, _ = plane_normals(B)
    cos_theta = np.clip(np.dot(nA, nB) / (np.linalg.norm(nA) * np.linalg.norm(nB)), -1, 1)
    return float(np.degrees(np.arccos(cos_theta)))


def plane_normals_3d(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Plane normal for 3D points via PCA."""
    pca = PCA(n_components=2)
    pca.fit(points)
    comps = pca.components_
    comp1 = comps[0] / np.linalg.norm(comps[0])
    comp2 = comps[1] - np.dot(comps[1], comp1) * comp1
    comp2 = comp2 / np.linalg.norm(comp2)
    normal = np.cross(comp1, comp2)
    return normal, np.stack([comp1, comp2], axis=0)


def plane_angle_3d(A: np.ndarray, B: np.ndarray) -> float:
    """Angle between planes using 3D PCA normals."""
    nA, _ = plane_normals_3d(A)
    nB, _ = plane_normals_3d(B)
    cos_theta = np.clip(np.dot(nA, nB) / (np.linalg.norm(nA) * np.linalg.norm(nB)), -1, 1)
    return float(np.degrees(np.arccos(cos_theta)))


def alignment_index(S1: np.ndarray, S2: np.ndarray) -> float:
    """
    Elsayed & Cunningham (2016) alignment index; values in [0,1].
    """
    cov1 = np.cov(S1.T)
    cov2 = np.cov(S2.T)
    eigvals1, eigvecs1 = np.linalg.eigh(cov1)
    eigvals2, eigvecs2 = np.linalg.eigh(cov2)
    # descending order
    idx1 = np.argsort(eigvals1)[::-1]
    idx2 = np.argsort(eigvals2)[::-1]
    Q1 = eigvecs1[:, idx1[:2]]  # top-2 subspace
    Q2 = eigvecs2[:, idx2[:2]]
    ai12 = np.trace(Q1.T @ cov2 @ Q1) / np.sum(eigvals2[idx2[:2]])
    ai21 = np.trace(Q2.T @ cov1 @ Q2) / np.sum(eigvals1[idx1[:2]])
    return float((ai12 + ai21) / 2.0)


def phase_alignment(A: np.ndarray, B: np.ndarray) -> float:
    """
    Orthogonal Procrustes to get rotation between two planar point sets.
    Returns angle (degrees) around z for 2D embedded in 3D PCA space.
    """
    R, _ = orthogonal_procrustes(A, B)
    angle = -math.degrees(math.atan2(R[1, 0], R[0, 0]))
    return angle


# -----------------------------
# Plotting helpers
# -----------------------------


def plot_loss(loss_history: List[float], plateau: Optional[int], out_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(loss_history, color="black")
    if plateau is not None:
        plt.axvline(plateau, color="magenta", linestyle="--", label="plateau")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss (Fig 4A analogue)")
    plt.legend()
    sns.despine()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_geometry(points_pre: np.ndarray, points_post: np.ndarray, out_path: Path) -> None:
    """3D PCA scatter for cued geometry pre/post cue (Fig 2B analogue)."""
    fig = plt.figure(figsize=(10, 4))
    for idx, pts in enumerate([points_pre, points_post]):
        ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=np.repeat([0, 1], pts.shape[0] // 2), cmap="coolwarm")
        ax.set_title("Pre-cue" if idx == 0 else "Post-cue")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_angles_ai(angles: Dict[str, float], ais: Dict[str, float], out_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.subplot(1, 2, 1)
    plt.bar(range(len(angles)), list(angles.values()), color=["orange", "magenta", "purple"])
    plt.xticks(range(len(angles)), list(angles.keys()), rotation=45)
    plt.ylabel("Plane angle (deg)")
    plt.subplot(1, 2, 2)
    plt.bar(range(len(ais)), list(ais.values()), color=["orange", "magenta", "purple"])
    plt.xticks(range(len(ais)), list(ais.keys()), rotation=45)
    plt.ylabel("Alignment index")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_stress_test(delays: List[int], ai_values: List[float], out_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(delays, ai_values, marker="o")
    plt.xlabel("Post-cue delay length (cycles)")
    plt.ylabel("Post-cue AI (cued planes)")
    plt.title("Fig 5B analogue")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


# -----------------------------
# Orchestration
# -----------------------------


def compute_geometry(hidden: np.ndarray, dataset: Dict[str, torch.Tensor], cfg: RetroCueConfig) -> Dict[str, float]:
    binned = bin_hidden_states(hidden, dataset, cfg)
    # PCA to 3D for visualisation and geometry calculations
    pca = PCA(n_components=3)
    pca_pre = pca.fit_transform(binned["delay1"])
    pca_post = pca.fit_transform(binned["delay2"])

    pre_loc1, pre_loc2 = pca_pre[: cfg.colour_bins], pca_pre[cfg.colour_bins :]
    post_loc1, post_loc2 = pca_post[: cfg.colour_bins], pca_post[cfg.colour_bins :]

    angle_pre = plane_angle_3d(pre_loc1, pre_loc2)
    angle_post = plane_angle_3d(post_loc1, post_loc2)
    ai_pre = alignment_index(pre_loc1, pre_loc2)
    ai_post = alignment_index(post_loc1, post_loc2)
    phase_post = phase_alignment(post_loc1, post_loc2)

    return {
        "pca_pre": pca_pre,
        "pca_post": pca_post,
        "angle_pre": angle_pre,
        "angle_post": angle_post,
        "ai_pre": ai_pre,
        "ai_post": ai_post,
        "phase_post": phase_post,
    }


def run_single_model(cfg: RetroCueConfig, output_dir: Path) -> Dict[str, Path]:
    task = RetroCueTask(cfg)
    dataset = task.generate_dataset()
    result = train(cfg, dataset)

    outputs, hidden = run_model(result.model, dataset, cfg)
    geom = compute_geometry(hidden, dataset, cfg)

    # Save checkpoints
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result.checkpoints["untrained"], ckpt_dir / "untrained.pt")
    if "plateau" in result.checkpoints:
        torch.save(result.checkpoints["plateau"], ckpt_dir / "plateau.pt")
    torch.save(result.checkpoints["trained"], ckpt_dir / "trained.pt")

    # Figures
    figs = {}
    figs["fig4a_loss"] = output_dir / "figures" / "fig4a_loss.png"
    plot_loss(result.loss_history, result.plateau_epoch, figs["fig4a_loss"])

    figs["fig2b_geometry"] = output_dir / "figures" / "fig2b_geometry.png"
    plot_geometry(geom["pca_pre"], geom["pca_post"], figs["fig2b_geometry"])

    angles = {
        "untrained_pre": 90.0,
        "plateau_post": geom["angle_post"],
        "trained_post": geom["angle_post"],
    }
    ais = {"pre": geom["ai_pre"], "plateau": geom["ai_post"], "trained": geom["ai_post"]}
    figs["fig4c"] = output_dir / "figures" / "fig4c_angles_ai.png"
    plot_angles_ai(angles, ais, figs["fig4c"])

    # Stress test (Fig 5B analogue) is run separately
    meta_path = output_dir / "outputs" / "metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(
            {
                "config": asdict(cfg),
                "loss_history": result.loss_history,
                "plateau_epoch": result.plateau_epoch,
                "geometry": {
                    "angle_pre": geom["angle_pre"],
                    "angle_post": geom["angle_post"],
                    "ai_pre": geom["ai_pre"],
                    "ai_post": geom["ai_post"],
                    "phase_post": geom["phase_post"],
                },
            },
            f,
            indent=2,
        )

    return figs


def stress_test(cfg: RetroCueConfig, delays: List[int], base_output: Path) -> Path:
    ai_values = []
    for d in delays:
        cfg_variant = dataclass_replace(cfg, delay2=d)
        task = RetroCueTask(cfg_variant)
        dataset = task.generate_dataset()
        result = train(cfg_variant, dataset)
        _, hidden = run_model(result.model, dataset, cfg_variant)
        geom = compute_geometry(hidden, dataset, cfg_variant)
        ai_values.append(geom["ai_post"])
    out_path = base_output / "figures" / "fig5b_stress.png"
    plot_stress_test(delays, ai_values, out_path)
    return out_path


def dataclass_replace(cfg: RetroCueConfig, **kwargs) -> RetroCueConfig:
    data = asdict(cfg)
    data.update(kwargs)
    return RetroCueConfig(**data)


def cli():
    parser = argparse.ArgumentParser(description="Retro-cue RNN replication pipeline")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to place outputs/figures/checkpoints")
    parser.add_argument("--stress", action="store_true", help="Run stress test (Fig 5B analogue)")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    cfg = RetroCueConfig(device=args.device)
    output_dir = Path(args.output_dir).resolve()

    figs = run_single_model(cfg, output_dir)
    if args.stress:
        stress_delays = [0, 2, 4, 6, 7]
        stress_path = stress_test(cfg, stress_delays, output_dir)
        figs["fig5b"] = stress_path

    print("Saved figures:")
    for k, v in figs.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    cli()

