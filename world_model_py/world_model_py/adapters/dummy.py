"""A deterministic, GPU-free World Model used for CI, demos and smoke tests.

It produces structurally-valid outputs (latents, an occupancy grid that
"drifts" over the horizon, and a risk score that grows with action
magnitude) so the whole ROS 2 pipeline can be exercised with no real model,
no dataset and no GPU. Outputs are deterministic given the inputs.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import (
    ActionCondition,
    FuturePrediction,
    Observation,
    WorldModelAdapter,
)


class DummyAdapter(WorldModelAdapter):
    name = "dummy"

    def __init__(self, latent_dim: int = 256, grid_size: int = 32, dt: float = 0.1):
        self.latent_dim = int(latent_dim)
        self.grid_size = int(grid_size)
        self.dt = float(dt)

    def _seed_from(self, obs: Observation) -> int:
        h = 0
        if obs.ego_state is not None and obs.ego_state.size:
            h ^= int(np.abs(obs.ego_state).sum() * 1000) & 0xFFFFFFFF
        if obs.image is not None and obs.image.size:
            h ^= int(obs.image.astype(np.int64).sum()) & 0xFFFFFFFF
        h ^= (len(obs.instruction) * 2654435761) & 0xFFFFFFFF
        return h & 0xFFFFFFFF

    def predict_future(
        self,
        obs: Observation,
        action: Optional[ActionCondition] = None,
        horizon: int = 8,
    ) -> FuturePrediction:
        if action is not None and action.horizon > 0:
            horizon = action.horizon
        horizon = max(1, int(horizon))

        rng = np.random.default_rng(self._seed_from(obs))
        base = rng.standard_normal(self.latent_dim).astype(np.float32)

        latents: list[np.ndarray] = []
        occupancy: list[np.ndarray] = []
        gs = self.grid_size
        yy, xx = np.mgrid[0:gs, 0:gs]
        for k in range(horizon):
            # latent slowly evolves step to step.
            drift = 0.05 * (k + 1) * rng.standard_normal(self.latent_dim).astype(np.float32)
            latents.append((base + drift).astype(np.float32))

            # a synthetic "obstacle blob" that drifts across the grid.
            cx = (0.5 + 0.3 * np.sin(0.3 * (k + 1))) * gs
            cy = (0.5 + 0.3 * np.cos(0.3 * (k + 1))) * gs
            dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
            grid = (100.0 * np.exp(-dist2 / (2 * (gs * 0.12) ** 2))).astype(np.int8)
            occupancy.append(grid)

        # risk grows with the magnitude of the conditioning action.
        if action is not None and action.action is not None and action.action.size:
            mag = float(np.linalg.norm(action.action) / np.sqrt(action.action.size))
            risk = float(np.clip(0.1 + 0.5 * np.tanh(mag), 0.0, 1.0))
        else:
            risk = 0.1

        return FuturePrediction(
            dt=self.dt,
            latents=latents,
            occupancy=occupancy,
            risk=risk,
            risk_confidence=0.5,
            risk_label="dummy",
        )

    def info(self) -> dict:
        return {
            "name": self.name,
            "remote": False,
            "device": "cpu",
            "latent_dim": self.latent_dim,
            "grid_size": self.grid_size,
        }
