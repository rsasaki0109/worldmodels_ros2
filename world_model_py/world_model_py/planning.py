"""Latent-space planning: imagine a path to a goal, with no model training.

This is the "imagination" half of a world model. The JEPA adapters give us a
frozen *encoder* (frame -> latent); this module adds the two missing pieces to
turn that into goal-reaching behaviour, **without training anything**:

  1. an *episodic* (retrieval / non-parametric) latent dynamics model. Given a
     memory of real transitions ``(latent_t, action_t, latent_{t+1})`` recorded
     from a robot, it predicts the next latent for a *new* (latent, action) by
     k-nearest-neighbour regression in the joint (latent, action) space. No
     gradient steps, no weights -- it just remembers what actually happened.
     This is the retrieval analogue of DINO-WM / PLDM ("frozen features + a
     simple dynamics model"), and it runs on a CPU.

  2. a sampling planner (CEM / random-shooting) that rolls candidate action
     sequences through that dynamics and keeps the one whose imagined final
     latent is closest (cosine) to a *goal* latent -- i.e. visual foresight to
     an image goal.

Everything here is plain numpy, ROS-free and torch-free, so it is unit-tested
on CI with a toy linear system and reused by the ROS layer for the real demo.
The planner is dynamics-agnostic: any object with ``step(latent, action)`` works
(a learned V-JEPA2-AC head can replace ``RetrievalDynamics`` unchanged).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

import numpy as np


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine similarity, clipped to [0, 2]. 0 == identical direction."""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.clip(1.0 - float(np.dot(a, b) / denom), 0.0, 2.0))


class LatentDynamics(Protocol):
    """Anything that predicts the next latent from (latent, action)."""

    def step(self, latent: np.ndarray, action: np.ndarray) -> np.ndarray: ...


@dataclass
class RetrievalDynamics:
    """Non-parametric latent dynamics from a memory of real transitions.

    Built from arrays recorded off a robot / dataset:
      * ``latents``       (N, D)  latent at step t
      * ``actions``       (N, A)  action taken at step t
      * ``next_latents``  (N, D)  latent at step t+1

    ``step`` finds the ``k`` memory entries whose (latent, action) is closest to
    the query and returns a distance-weighted prediction of the *next* latent.
    It predicts the **delta** (next - cur) so it extrapolates sensibly to latents
    that sit between remembered states. No training.
    """

    latents: np.ndarray
    actions: np.ndarray
    next_latents: np.ndarray
    k: int = 8
    action_weight: float = 1.0

    def __post_init__(self) -> None:
        self.latents = np.asarray(self.latents, dtype=np.float32)
        self.actions = np.asarray(self.actions, dtype=np.float32)
        self.next_latents = np.asarray(self.next_latents, dtype=np.float32)
        n = self.latents.shape[0]
        if not (self.actions.shape[0] == self.next_latents.shape[0] == n) or n == 0:
            raise ValueError("latents/actions/next_latents must share N>0 rows")
        # L2-normalise latents so cosine geometry == euclidean on the sphere.
        self._lat_n = self._unit(self.latents)
        self._deltas = self.next_latents - self.latents
        # typical action scale, to make the action distance commensurate.
        self._act_scale = float(np.std(self.actions)) + 1e-6
        self.k = int(max(1, min(self.k, n)))

    @staticmethod
    def _unit(x: np.ndarray) -> np.ndarray:
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

    def step_with_index(self, latent: np.ndarray, action: np.ndarray) -> tuple:
        """Predict next latent; also return the index of the nearest memory entry
        (used to decode the imagined latent back to a real frame)."""
        latent = np.asarray(latent, dtype=np.float32).ravel()
        action = np.asarray(action, dtype=np.float32).ravel()
        q = latent / (np.linalg.norm(latent) + 1e-8)
        lat_d = 1.0 - self._lat_n @ q                       # (N,) cosine distance
        act_d = np.linalg.norm(self.actions - action[None, :], axis=1) / self._act_scale
        dist = lat_d + self.action_weight * act_d
        idx = np.argpartition(dist, self.k - 1)[: self.k]
        w = 1.0 / (dist[idx] + 1e-6)
        w = w / w.sum()
        delta = (w[:, None] * self._deltas[idx]).sum(axis=0)
        nearest = int(idx[int(np.argmin(dist[idx]))])
        return (latent + delta).astype(np.float32), nearest

    def step(self, latent: np.ndarray, action: np.ndarray) -> np.ndarray:
        return self.step_with_index(latent, action)[0]


@dataclass
class PlanResult:
    """The outcome of planning toward a goal latent."""

    actions: np.ndarray                                # (horizon, action_dim)
    latents: list = field(default_factory=list)        # imagined latent per step
    indices: list = field(default_factory=list)        # nearest-memory idx per step
    cost: float = 0.0                                  # final goal cosine distance
    costs: list = field(default_factory=list)          # cost per CEM iteration

    @property
    def horizon(self) -> int:
        return int(self.actions.shape[0]) if self.actions.size else 0


def _rollout(dynamics: LatentDynamics, start: np.ndarray, seq: np.ndarray):
    """Imagine a latent trajectory for one action sequence. Returns (latents,
    indices) where indices is the nearest memory entry per step (or -1)."""
    lat = np.asarray(start, dtype=np.float32).ravel()
    latents, indices = [], []
    for a in seq:
        if hasattr(dynamics, "step_with_index"):
            lat, idx = dynamics.step_with_index(lat, a)
        else:
            lat, idx = dynamics.step(lat, a), -1
        latents.append(np.asarray(lat, dtype=np.float32))
        indices.append(int(idx))
    return latents, indices


def plan_to_goal(
    dynamics: LatentDynamics,
    start_latent: np.ndarray,
    goal_latent: np.ndarray,
    *,
    action_dim: int,
    horizon: int = 12,
    samples: int = 256,
    iterations: int = 4,
    elite_frac: float = 0.1,
    action_low: float = -1.0,
    action_high: float = 1.0,
    terminal_weight: float = 1.0,
    path_weight: float = 0.1,
    seed: int = 0,
) -> PlanResult:
    """Plan an action sequence whose imagined future reaches ``goal_latent``.

    Cross-Entropy Method over action sequences: sample, roll each through
    ``dynamics``, score by closeness to the goal (terminal + a small path term so
    it makes steady progress), keep the elite, refit a Gaussian, repeat. With
    ``iterations=1`` this degrades to random shooting.

    Returns the best sequence found and its imagined latent trajectory.
    """
    rng = np.random.default_rng(seed)
    start = np.asarray(start_latent, dtype=np.float32).ravel()
    goal = np.asarray(goal_latent, dtype=np.float32).ravel()
    horizon = int(max(1, horizon))
    n_elite = max(1, int(samples * elite_frac))

    mid = 0.5 * (action_low + action_high)
    mean = np.full((horizon, action_dim), mid, dtype=np.float32)
    std = np.full((horizon, action_dim), 0.5 * (action_high - action_low), dtype=np.float32)

    best_seq, best_cost, best_traj, best_idx = None, np.inf, [], []
    iter_costs: list[float] = []

    def score(latents: list) -> float:
        terminal = cosine_distance(latents[-1], goal)
        path = float(np.mean([cosine_distance(l, goal) for l in latents]))
        return terminal_weight * terminal + path_weight * path

    for _ in range(int(max(1, iterations))):
        cand = rng.normal(mean, std, size=(samples, horizon, action_dim)).astype(np.float32)
        cand = np.clip(cand, action_low, action_high)
        costs = np.empty(samples, dtype=np.float32)
        rolls = []
        for i in range(samples):
            lat, idx = _rollout(dynamics, start, cand[i])
            costs[i] = score(lat)
            rolls.append((lat, idx))
        order = np.argsort(costs)
        elite = cand[order[:n_elite]]
        mean = elite.mean(axis=0)
        std = elite.std(axis=0) + 1e-3            # keep a floor so it can explore
        top = int(order[0])
        iter_costs.append(float(costs[top]))
        if costs[top] < best_cost:
            best_cost = float(costs[top])
            best_seq = cand[top].copy()
            best_traj, best_idx = rolls[top]

    return PlanResult(
        actions=best_seq if best_seq is not None else mean,
        latents=best_traj,
        indices=best_idx,
        cost=best_cost,
        costs=iter_costs,
    )


def decode_trajectory(
    latents: list,
    memory_latents: np.ndarray,
    memory_frames,
) -> list:
    """Turn imagined latents into pictures by nearest-neighbour retrieval.

    JEPA encoders have no pixel decoder, so we visualise the imagined future by
    fetching, for each imagined latent, the real frame whose latent is closest
    (cosine). This is how the "imagination" is made watchable.
    """
    mem = np.asarray(memory_latents, dtype=np.float32)
    mem_n = mem / (np.linalg.norm(mem, axis=-1, keepdims=True) + 1e-8)
    out = []
    for lat in latents:
        q = np.asarray(lat, dtype=np.float32).ravel()
        q = q / (np.linalg.norm(q) + 1e-8)
        out.append(memory_frames[int(np.argmax(mem_n @ q))])
    return out
