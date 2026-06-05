"""Turn ROS 2 messages into the numeric arrays a learning dataset needs.

Pure functions over already-deserialized messages -- no ROS runtime, no bag,
no cv_bridge (cv_bridge is built against numpy 1.x and crashes under numpy 2),
so images are decoded by hand. Each extractor is keyed by message type string
(e.g. "sensor_msgs/msg/Image") so the converter can dispatch on the bag's
declared topic types.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


# ------------------------------------------------------------------ images

def image_to_rgb(msg) -> np.ndarray:
    """sensor_msgs/Image -> (H, W, 3) uint8 RGB. Handles rgb8/bgr8/mono8."""
    h, w = int(msg.height), int(msg.width)
    if h == 0 or w == 0:
        raise ValueError("empty image")
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    enc = msg.encoding.lower()
    if enc in ("rgb8", "bgr8"):
        arr = buf.reshape(h, w, 3)
        if enc == "bgr8":
            arr = arr[:, :, ::-1]
        return np.ascontiguousarray(arr)
    if enc == "mono8":
        return np.repeat(buf.reshape(h, w, 1), 3, axis=2)
    if enc in ("rgba8", "bgra8"):
        arr = buf.reshape(h, w, 4)[:, :, :3]
        if enc == "bgra8":
            arr = arr[:, :, ::-1]
        return np.ascontiguousarray(arr)
    raise ValueError(f"unsupported image encoding '{msg.encoding}'")


# ------------------------------------------------------------------ state/action

def _quat_to_yaw(x, y, z, w) -> float:
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def odometry_to_vec(msg) -> np.ndarray:
    """nav_msgs/Odometry -> [x, y, yaw, vx, vy, wz]."""
    p = msg.pose.pose.position
    o = msg.pose.pose.orientation
    t = msg.twist.twist
    return np.array(
        [p.x, p.y, _quat_to_yaw(o.x, o.y, o.z, o.w), t.linear.x, t.linear.y, t.angular.z],
        dtype=np.float32,
    )


def jointstate_to_vec(msg) -> np.ndarray:
    """sensor_msgs/JointState -> position vector (falls back to velocity)."""
    if len(msg.position):
        return np.asarray(msg.position, dtype=np.float32)
    if len(msg.velocity):
        return np.asarray(msg.velocity, dtype=np.float32)
    return np.asarray(msg.effort, dtype=np.float32)


def twist_to_vec(msg) -> np.ndarray:
    """geometry_msgs/Twist -> [lx, ly, lz, ax, ay, az]."""
    return np.array(
        [msg.linear.x, msg.linear.y, msg.linear.z,
         msg.angular.x, msg.angular.y, msg.angular.z],
        dtype=np.float32,
    )


def twiststamped_to_vec(msg) -> np.ndarray:
    return twist_to_vec(msg.twist)


def floatarray_to_vec(msg) -> np.ndarray:
    """std_msgs/Float32MultiArray or Float64MultiArray -> data."""
    return np.asarray(msg.data, dtype=np.float32)


# message type string -> vector extractor
_VECTOR_EXTRACTORS = {
    "nav_msgs/msg/Odometry": odometry_to_vec,
    "sensor_msgs/msg/JointState": jointstate_to_vec,
    "geometry_msgs/msg/Twist": twist_to_vec,
    "geometry_msgs/msg/TwistStamped": twiststamped_to_vec,
    "std_msgs/msg/Float32MultiArray": floatarray_to_vec,
    "std_msgs/msg/Float64MultiArray": floatarray_to_vec,
}


def vector_extractor(type_str: str):
    try:
        return _VECTOR_EXTRACTORS[type_str]
    except KeyError:
        raise ValueError(
            f"no vector extractor for '{type_str}'. supported: "
            f"{sorted(_VECTOR_EXTRACTORS)}"
        ) from None


def supported_vector_types() -> list[str]:
    return sorted(_VECTOR_EXTRACTORS)


def header_stamp_ns(msg) -> Optional[int]:
    """Best-effort sensor timestamp (ns) from a message header, else None."""
    hdr = getattr(msg, "header", None)
    if hdr is None:
        return None
    return int(hdr.stamp.sec) * 1_000_000_000 + int(hdr.stamp.nanosec)
