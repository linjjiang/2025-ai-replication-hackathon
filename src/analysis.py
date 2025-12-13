"""
Analysis utilities for geometric readouts and plotting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.linalg import svd

from .utils import bin_colour, ensure_dir


@dataclass
class PCAResult:
    projected: np.ndarray  # shape (conditions, 3)
    components: np.ndarray  # shape (3, features)
    explained: np.ndarray  # variance ratio


def compute_pca(data: np.ndarray, n_components: int = 3) -> PCAResult:
    """Simple PCA via SVD; data shape (samples, features)."""
    data_centered = data - data.mean(0, keepdims=True)
    u, s, vh = svd(data_centered, full_matrices=False)
    comps = vh[:n_components]
    projected = np.dot(data_centered, comps.T)
    explained = (s[:n_components] ** 2) / (len(data) - 1)
    explained = explained / explained.sum()
    return PCAResult(projected=projected, components=comps, explained=explained)


def build_condition_matrix(hidden: np.ndarray, metadata: List[Dict], time_idx: int, num_bins: int = 4) -> np.ndarray:
    """
    Average hidden states within colour bins for cued geometry.
    Returns matrix of shape (8, hidden_dim) ordered as L1 bins then L2 bins.
    """
    hidden_dim = hidden.shape[-1]
    accum = np.zeros((2, num_bins, hidden_dim), dtype=np.float64)
    counts = np.zeros((2, num_bins), dtype=np.int32)
    for i, meta in enumerate(metadata):
        loc = meta["cue"]
        angle = meta["target_angle"]
        bin_idx = bin_colour(angle, num_bins=num_bins)
        accum[loc, bin_idx] += hidden[i, time_idx]
        counts[loc, bin_idx] += 1
    # avoid division by zero
    counts[counts == 0] = 1
    means = accum / counts[..., None]
    return means.reshape(-1, hidden_dim)


def fit_plane(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return centroid, two basis vectors (3D), and the normal.
    Orientation is fixed by ordering polygon vertices around the centroid to avoid 0/180 flips.
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    u, s, vh = svd(centered, full_matrices=False)
    pca_basis = vh[:2]  # two dominant directions
    normal = np.cross(pca_basis[0], pca_basis[1])
    if normal[2] < 0:
        normal = -normal
        pca_basis = -pca_basis
    normal /= np.linalg.norm(normal) + 1e-9
    # project to 2D to order vertices
    coords2d = centered @ pca_basis.T
    angles = np.arctan2(coords2d[:, 1], coords2d[:, 0])
    order = np.argsort(angles)
    ref_vec = coords2d[order[0]]
    ref_vec = ref_vec / (np.linalg.norm(ref_vec) + 1e-9)
    vec1 = (ref_vec[0] * pca_basis[0]) + (ref_vec[1] * pca_basis[1])
    vec1 /= np.linalg.norm(vec1) + 1e-9
    vec2 = np.cross(normal, vec1)
    vec2 /= np.linalg.norm(vec2) + 1e-9
    basis = np.stack([vec1, vec2], axis=0)
    return centroid, basis, normal


def plane_angle(normal1: np.ndarray, normal2: np.ndarray, signed: bool = False) -> float:
    """Angle between plane normals in degrees."""
    dot_val = np.clip(np.dot(normal1, normal2), -1.0, 1.0)
    angle = np.degrees(np.arccos(dot_val))
    if signed:
        sign = np.sign(np.cross(normal1, normal2)[2])
        angle = angle * (1 if sign == 0 else sign)
    return angle


def alignment_index(S1: np.ndarray, S2: np.ndarray, subspace_dim: int = 3) -> float:
    """Symmetric alignment index following Elsayed et al., 2016."""
    def _proj(cov_a, cov_b):
        eigvals, eigvecs = np.linalg.eigh(cov_a)
        eigvals = np.maximum(eigvals, 1e-9)
        idx = np.argsort(eigvals)[::-1][:subspace_dim]
        Q = eigvecs[:, idx]
        return np.trace(Q.T @ cov_b @ Q) / np.sum(eigvals[idx])

    cov1 = np.cov(S1, rowvar=False)
    cov2 = np.cov(S2, rowvar=False)
    ai = 0.5 * (_proj(cov1, cov2) + _proj(cov2, cov1))
    return float(np.clip(ai, 0.0, 1.0))


def project_to_plane(points: np.ndarray, centroid: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project 3D points to 2D plane coordinates."""
    centered = points - centroid
    return centered @ basis.T


def phase_alignment_angle(points1: np.ndarray, points2: np.ndarray, plane1, plane2) -> float:
    """Orthogonal Procrustes alignment angle between two planar point sets."""
    c1, b1, _ = plane1
    c2, b2, _ = plane2
    p1 = project_to_plane(points1, c1, b1)
    p2 = project_to_plane(points2, c2, b2)
    # rotate plane2 onto plane1
    m = p2.T @ p1
    u, _, vt = svd(m)
    R = u @ vt
    angle = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return float(angle)


def plot_plane(ax, points: np.ndarray, plane, color: str, marker: str, label: str) -> None:
    centroid, basis, normal = plane
    coords = project_to_plane(points, centroid, basis)
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=color, marker=marker, label=label)
    # mesh for the plane
    grid = np.linspace(-1.2, 1.2, 12)
    uu, vv = np.meshgrid(grid, grid)
    span1 = coords[:, 0].max() - coords[:, 0].min()
    span2 = coords[:, 1].max() - coords[:, 1].min()
    scale1 = max(span1, 1e-3)
    scale2 = max(span2, 1e-3)
    surface = centroid + (uu * scale1)[:, :, None] * basis[0] + (vv * scale2)[:, :, None] * basis[1]
    ax.plot_surface(surface[:, :, 0], surface[:, :, 1], surface[:, :, 2], alpha=0.2, color=color)
    return scatter


def plot_pre_post_planes(pre_points: np.ndarray, post_points: np.ndarray, explained_pre: np.ndarray, explained_post: np.ndarray, save_path: str) -> None:
    """Create the 3D geometry figure for pre- and post-cue."""
    fig = plt.figure(figsize=(12, 5))
    for idx, (points, explained, title) in enumerate(
        [(pre_points, explained_pre, "Pre-cue"), (post_points, explained_post, "Post-cue")]
    ):
        ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
        loc1 = points[:4]
        loc2 = points[4:]
        plane1 = fit_plane(loc1)
        plane2 = fit_plane(loc2)
        plot_plane(ax, loc1, plane1, color="#1f77b4", marker="^", label="L1")
        plot_plane(ax, loc2, plane2, color="#ff7f0e", marker="s", label="L2")
        ax.set_title(title)
        ax.set_xlabel(f"PC1 [{explained[0]*100:.1f}%]")
        ax.set_ylabel(f"PC2 [{explained[1]*100:.1f}%]")
        ax.set_zlabel(f"PC3 [{explained[2]*100:.1f}%]")
        ax.legend()
    path = ensure_dir(Path(save_path).parent)
    fig.tight_layout()
    fig.savefig(path / Path(save_path).name, dpi=300)
    plt.close(fig)


def polar_angle_plot(ax, angles_by_stage: Dict[str, np.ndarray], colors: Dict[str, str], title: str) -> None:
    """Plot plane angles on polar axis."""
    for stage, values in angles_by_stage.items():
        ax.scatter(np.radians(values), np.ones_like(values), alpha=0.6, label=stage, color=colors[stage])
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_rlim(0, 1.2)
    ax.set_title(title)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.15))


def bar_with_points(ax, bars: Dict[str, float], samples: Dict[str, np.ndarray], colors: Dict[str, str]) -> None:
    """Bar plot with jittered sample points."""
    labels = list(bars.keys())
    bar_vals = [bars[k] for k in labels]
    ax.bar(labels, bar_vals, color=[colors[k] for k in labels], alpha=0.6)
    for i, key in enumerate(labels):
        xs = np.random.uniform(i - 0.15, i + 0.15, size=len(samples[key]))
        ax.scatter(xs, samples[key], color=colors[key], edgecolor="k", linewidth=0.5, alpha=0.8, s=20)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Alignment Index")
    ax.set_title("Subspace AI")


def compute_geometry(hidden: np.ndarray, metadata: List[Dict], time_idx: int, num_bins: int = 4, subspace_dim: int = 3):
    """Compute PCA projection, plane fits, angles, and AI for a given timestep."""
    condition_matrix = build_condition_matrix(hidden, metadata, time_idx=time_idx, num_bins=num_bins)
    pca = compute_pca(condition_matrix, n_components=3)
    points = pca.projected
    loc1 = points[:num_bins]
    loc2 = points[num_bins:]
    plane1 = fit_plane(loc1)
    plane2 = fit_plane(loc2)
    theta = plane_angle(plane1[2], plane2[2])
    ai = alignment_index(condition_matrix[:num_bins], condition_matrix[num_bins:], subspace_dim=subspace_dim)
    return {
        "condition_matrix": condition_matrix,
        "points": points,
        "explained": pca.explained,
        "planes": (plane1, plane2),
        "theta": theta,
        "ai": ai,
    }


def plot_loss_curve(losses: np.ndarray, plateau_idx: int, save_path: str) -> None:
    """Plot training loss with plateau annotation for Fig 4A."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(losses, label="Loss")
    ax.axvline(plateau_idx, color="magenta", linestyle="--", label="Plateau")
    ax.scatter([0, plateau_idx, len(losses) - 1], [losses[0], losses[plateau_idx], losses[-1]], color=["orange", "magenta", "purple"], zorder=5)
    ax.annotate("Plateau", xy=(plateau_idx, losses[plateau_idx]), xytext=(plateau_idx, losses[plateau_idx] + 0.05), arrowprops=dict(arrowstyle="->", color="magenta"))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.set_title("Training loss with plateau (Fig 4A)")
    ensure_dir(Path(save_path).parent)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

