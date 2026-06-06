"""Example external World Model adapter for world_model_ros2.

A GPU-free, numpy-only backend that demonstrates how an *outside* package adds a
World Model: it encodes each camera frame as a normalized RGB colour histogram
(the "latent") and reports surprise as the L1 distance between successive
histograms — a cheap appearance-change / novelty signal.

It is wired in via a package entry point (see pyproject.toml), so installing
this package makes ``load_model("example")`` and ``world-model list`` work with
no changes to world_model_ros2 itself.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from world_model_py.adapters.base import (
    ActionCondition,
    FuturePrediction,
    Observation,
    WorldModelAdapter,
)

_BINS = 4  # per channel -> 64-d histogram


def color_histogram(image_hwc_uint8: np.ndarray) -> np.ndarray:
    px = np.asarray(image_hwc_uint8).reshape(-1, 3).astype(np.int64)
    q = np.clip(px // (256 // _BINS), 0, _BINS - 1)
    idx = q[:, 0] * _BINS * _BINS + q[:, 1] * _BINS + q[:, 2]
    h = np.bincount(idx, minlength=_BINS ** 3).astype(np.float32)
    return h / (h.sum() + 1e-8)


class HistogramAdapter(WorldModelAdapter):
    name = "example"

    def __init__(self, dt: float = 0.1):
        self.dt = float(dt)
        self._prev: Optional[np.ndarray] = None

    def predict_future(
        self,
        obs: Observation,
        action: Optional[ActionCondition] = None,
        horizon: int = 8,
    ) -> FuturePrediction:
        if obs.image is None or getattr(obs.image, "size", 0) == 0:
            raise ValueError("example adapter needs obs.image")
        if action is not None and action.horizon > 0:
            horizon = action.horizon
        horizon = max(1, int(horizon))

        hist = color_histogram(obs.image)
        if self._prev is None:
            risk, conf = 0.0, 0.0
        else:
            risk = float(np.clip(0.5 * np.abs(hist - self._prev).sum(), 0.0, 1.0))
            conf = 0.8
        self._prev = hist
        return FuturePrediction(
            dt=self.dt,
            latents=[hist.copy() for _ in range(horizon)],
            risk=risk,
            risk_confidence=conf,
            risk_label="example-hist",
        )

    def reset(self) -> None:
        self._prev = None


def make_example_adapter(**kwargs) -> HistogramAdapter:
    return HistogramAdapter(**kwargs)
