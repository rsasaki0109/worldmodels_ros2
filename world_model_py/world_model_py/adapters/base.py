"""Backend-agnostic World Model adapter interface.

Adapters operate on plain numpy arrays and dataclasses -- deliberately *not*
on ROS 2 messages -- so they can be imported, unit-tested and benchmarked
with no ROS install and no GPU. The ROS layer (``runtime_node``) is the only
place that converts between these dataclasses and ``world_model_msgs``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Observation:
    """Current observation fed into a World Model. All fields optional."""

    image: Optional[np.ndarray] = None          # (H, W, 3) uint8, or None
    ego_state: Optional[np.ndarray] = None      # (D,) float32
    action_history: Optional[np.ndarray] = None  # (T, action_dim) float32
    instruction: str = ""


@dataclass
class ActionCondition:
    """A candidate action sequence to condition a rollout on."""

    action: Optional[np.ndarray] = None         # (horizon, action_dim) float32
    dt: float = 0.1                             # seconds between steps

    @property
    def horizon(self) -> int:
        return 0 if self.action is None else int(self.action.shape[0])


@dataclass
class Latent:
    """A model-agnostic latent embedding."""

    data: np.ndarray                            # arbitrary shape
    encoding: str = ""


@dataclass
class FuturePrediction:
    """Everything a backend can say about the imagined future.

    ``latents`` is always present (index 0 == t+dt). ``frames`` and
    ``occupancy`` are optional and only filled by backends that can produce
    them. Risk fields summarise the future into a single gate-able score.
    """

    dt: float
    latents: list[np.ndarray] = field(default_factory=list)
    frames: list[np.ndarray] = field(default_factory=list)      # list of (H,W,3) uint8
    occupancy: list[np.ndarray] = field(default_factory=list)   # list of (H,W) int8 [-1,100]
    risk: float = 0.0
    risk_confidence: float = 1.0
    risk_label: str = ""

    @property
    def horizon(self) -> int:
        return len(self.latents)


class WorldModelAdapter(ABC):
    """Base class every World Model backend implements."""

    #: short stable identifier used by the registry / CLI / bench.
    name: str = "base"

    @abstractmethod
    def predict_future(
        self,
        obs: Observation,
        action: Optional[ActionCondition] = None,
        horizon: int = 8,
    ) -> FuturePrediction:
        """Imagine the future from ``obs``, optionally conditioned on ``action``."""
        raise NotImplementedError

    def encode(self, obs: Observation) -> Latent:
        """Encode an observation into a latent. Default: first rollout latent."""
        pred = self.predict_future(obs, horizon=1)
        data = pred.latents[0] if pred.latents else np.zeros((0,), dtype=np.float32)
        return Latent(data=data, encoding=self.name)

    def score_trajectory(
        self, obs: Observation, action: ActionCondition
    ) -> float:
        """Return a scalar risk in [0, 1] for executing ``action`` from ``obs``."""
        return float(self.predict_future(obs, action, horizon=action.horizon or 8).risk)

    def info(self) -> dict:
        """Backend metadata (device, dims, whether it is remote, ...)."""
        return {"name": self.name, "remote": False, "device": "cpu"}
