"""Convert a path into an action sequence (pure numpy, no ROS).

A candidate trajectory is a list of poses ``[x, y, yaw]``. The action between
consecutive poses is the body-frame displacement ``[ds_forward, ds_lateral,
dyaw]`` -- i.e. what the robot must *do* to follow the path. Feeding that to a
World Model's ``score_trajectory`` answers "how risky is following this path?".

Kept ROS-free so the geometry is unit-tested on its own.
"""
from __future__ import annotations

import numpy as np


def wrap_to_pi(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def path_to_action(poses_xyyaw: np.ndarray) -> np.ndarray:
    """``(N, 3)`` poses [x, y, yaw] -> ``(N-1, 3)`` body-frame actions
    [forward, lateral, dyaw]. Returns an empty ``(0, 3)`` array for < 2 poses.
    """
    p = np.asarray(poses_xyyaw, dtype=np.float32)
    if p.ndim != 2 or p.shape[0] < 2:
        return np.zeros((0, 3), dtype=np.float32)

    dxy = p[1:, :2] - p[:-1, :2]
    yaw = p[:-1, 2]
    cos, sin = np.cos(yaw), np.sin(yaw)
    forward = cos * dxy[:, 0] + sin * dxy[:, 1]
    lateral = -sin * dxy[:, 0] + cos * dxy[:, 1]
    dyaw = wrap_to_pi(p[1:, 2] - p[:-1, 2])
    return np.stack([forward, lateral, dyaw], axis=1).astype(np.float32)


def path_length(poses_xyyaw: np.ndarray) -> float:
    p = np.asarray(poses_xyyaw, dtype=np.float32)
    if p.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(p[1:, :2] - p[:-1, :2], axis=1)))
