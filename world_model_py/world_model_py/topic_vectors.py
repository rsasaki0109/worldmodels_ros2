"""Extract numeric vectors from common ROS 2 message types (no cv_bridge)."""
from __future__ import annotations

from typing import Callable

import numpy as np


def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


_STATE_EXTRACTORS: dict[str, Callable] = {
    "nav_msgs/msg/Odometry": lambda m: [
        m.pose.pose.position.x,
        m.pose.pose.position.y,
        quat_to_yaw(
            m.pose.pose.orientation.x,
            m.pose.pose.orientation.y,
            m.pose.pose.orientation.z,
            m.pose.pose.orientation.w,
        ),
        m.twist.twist.linear.x,
        m.twist.twist.linear.y,
        m.twist.twist.angular.z,
    ],
}

_ACTION_EXTRACTORS: dict[str, Callable] = {
    "geometry_msgs/msg/Twist": lambda m: [
        m.linear.x,
        m.linear.y,
        m.linear.z,
        m.angular.x,
        m.angular.y,
        m.angular.z,
    ],
    "geometry_msgs/msg/TwistStamped": lambda m: _ACTION_EXTRACTORS["geometry_msgs/msg/Twist"](m.twist),
}


def extract_action_vector(msg, type_str: str, mode: str = "scalar") -> np.ndarray:
    """Return action vector from a message. ``scalar`` keeps angular.z (steering)."""
    if type_str not in _ACTION_EXTRACTORS:
        raise ValueError(f"unsupported action type '{type_str}'")
    full = np.asarray(_ACTION_EXTRACTORS[type_str](msg), dtype=np.float32)
    if mode == "full":
        return full
    if mode == "scalar":
        if full.size < 6:
            return full[:1]
        return np.array([full[5]], dtype=np.float32)
    raise ValueError(f"unknown action mode '{mode}'")


def supported_state_types() -> list[str]:
    return sorted(_STATE_EXTRACTORS)


def supported_action_types() -> list[str]:
    return sorted(_ACTION_EXTRACTORS)
