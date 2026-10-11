"""Pinhole camera geometry in the robot base frame (pure numpy).

The occupancy head lifts image features onto the ground grid by projecting BEV
points into the camera (inverse perspective mapping). All it needs is a 3x4
projection ``P`` from base-frame points ``(x fwd, y left, z up)`` to pixels --
which a real robot gets from ``sensor_msgs/CameraInfo`` (``K``) and the
``base_link -> camera_optical`` TF, and the simulator gets from its config.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# base (x fwd, y left, z up) -> optical (x right, y down, z fwd)
_BASE_TO_OPTICAL = np.array([[0.0, -1.0, 0.0],
                             [0.0, 0.0, -1.0],
                             [1.0, 0.0, 0.0]])


@dataclass
class CameraModel:
    P: np.ndarray          # (3, 4) base-frame homogeneous point -> pixel (u, v, 1) * w
    width: int
    height: int

    def __post_init__(self) -> None:
        self.P = np.asarray(self.P, dtype=np.float64).reshape(3, 4)
        self.width, self.height = int(self.width), int(self.height)

    @classmethod
    def pinhole(cls, width: int, height: int, fx: float, fy: float, cx: float, cy: float,
                position=(0.0, 0.0, 0.0), pitch: float = 0.0) -> "CameraModel":
        """Camera at ``position`` (base frame), looking along +x, pitched down by
        ``pitch`` radians. ``cx, cy`` are in pixels (an off-centre ``cy`` is a
        shifted principal point)."""
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        c, s = np.cos(pitch), np.sin(pitch)
        # pitch down = rotate the base->camera axes about base y
        R_pitch = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        R = _BASE_TO_OPTICAL @ R_pitch.T          # base -> optical
        t = -R @ np.asarray(position, dtype=np.float64)
        return cls(P=K @ np.hstack([R, t[:, None]]), width=width, height=height)

    @classmethod
    def from_camera_info(cls, K, width: int, height: int, T_optical_base) -> "CameraModel":
        """From a CameraInfo ``K`` (3x3, row-major list ok) and the 4x4 transform
        taking base-frame points into the optical frame."""
        K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        T = np.asarray(T_optical_base, dtype=np.float64).reshape(4, 4)
        return cls(P=K @ T[:3, :], width=width, height=height)

    def project(self, points: np.ndarray):
        """(..., 3) base-frame points -> (u, v, valid). ``valid`` = in front of
        the camera and inside the image."""
        pts = np.asarray(points, dtype=np.float64)
        h = pts @ self.P[:, :3].T + self.P[:, 3]
        w = h[..., 2]
        front = w > 1e-6
        ws = np.where(front, w, 1.0)
        u, v = h[..., 0] / ws, h[..., 1] / ws
        valid = front & (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        return u, v, valid

    def to_dict(self) -> dict:
        return {"P": self.P, "width": self.width, "height": self.height}
