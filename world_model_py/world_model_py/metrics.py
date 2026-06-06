"""Prediction-quality metrics for World Model outputs (pure numpy, no ROS).

The benchmark measures *speed*; these measure *whether the imagined future is any
good*. They work on plain arrays so they're reusable for an Autoware/Nav2
evaluator (feed a reference occupancy) or for self-consistency checks with no
ground truth.

- ``occupancy_iou`` / ``future_occupancy_iou`` — agreement between a predicted
  occupancy grid (sequence) and a reference one (e.g. ground truth, or another
  stack's prediction).
- ``temporal_consistency`` — how stable a predicted occupancy horizon is
  (mean IoU between consecutive steps); a jittery rollout scores low.
- ``latent_drift`` — mean cosine distance between consecutive latents, for
  latent-only backends (V-JEPA2 / I-JEPA).
"""
from __future__ import annotations

import numpy as np


def _occupied(grid: np.ndarray, threshold: int) -> np.ndarray:
    return np.asarray(grid) >= threshold


def occupancy_iou(pred: np.ndarray, ref: np.ndarray, threshold: int = 50) -> float:
    """Intersection-over-union of occupied cells. Two empty grids agree -> 1.0."""
    a = _occupied(pred, threshold)
    b = _occupied(ref, threshold)
    if a.shape != b.shape:
        raise ValueError(f"grid shape mismatch: {a.shape} vs {b.shape}")
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 1.0
    inter = int(np.logical_and(a, b).sum())
    return inter / union


def future_occupancy_iou(pred_grids, ref_grids, threshold: int = 50) -> dict:
    """Per-step IoU between two occupancy horizons (compared up to the shorter
    length). Returns ``{"per_step": [...], "mean": float, "steps": n}``."""
    n = min(len(pred_grids), len(ref_grids))
    per = [occupancy_iou(pred_grids[i], ref_grids[i], threshold) for i in range(n)]
    return {"per_step": per, "mean": float(np.mean(per)) if per else 0.0, "steps": n}


def temporal_consistency(grids, threshold: int = 50) -> float:
    """Mean IoU between consecutive predicted grids (1.0 = perfectly stable).
    Fewer than two grids -> 1.0."""
    if len(grids) < 2:
        return 1.0
    ious = [occupancy_iou(grids[i], grids[i + 1], threshold) for i in range(len(grids) - 1)]
    return float(np.mean(ious))


def latent_drift(latents) -> dict:
    """Cosine distance between consecutive latents. Returns
    ``{"per_step": [...], "mean": float}`` (empty -> mean 0.0)."""
    per = []
    for i in range(len(latents) - 1):
        a = np.asarray(latents[i], dtype=np.float64).ravel()
        b = np.asarray(latents[i + 1], dtype=np.float64).ravel()
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        per.append(float(np.clip(1.0 - np.dot(a, b) / denom, 0.0, 2.0)))
    return {"per_step": per, "mean": float(np.mean(per)) if per else 0.0}
