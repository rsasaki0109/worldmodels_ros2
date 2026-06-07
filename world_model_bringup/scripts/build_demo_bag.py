#!/usr/bin/env python3
"""Build the bundled counterfactual replay demo rosbag2 (mcap).

Synthetic driving clip: 48 frames @ 10 Hz, 64x64 RGB, varying scene + steering.
GPU-free; needs ROS 2 (rosbag2_py) only.

    python3 world_model_bringup/scripts/build_demo_bag.py
    # -> world_model_bringup/demo/drive_demo.mcap
"""
from __future__ import annotations

import os

import numpy as np
import rosbag2_py
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.serialization import serialize_message
from sensor_msgs.msg import Image

IMG = "/camera/image_raw"
ODOM = "/odom"
CMD = "/cmd_vel"
N = 48
DT_NS = 100_000_000  # 10 Hz
OUT = os.path.join(os.path.dirname(__file__), "..", "demo", "drive_demo.mcap")


def _stamp(msg, t_ns: int) -> None:
    msg.header.stamp.sec = t_ns // 1_000_000_000
    msg.header.stamp.nanosec = t_ns % 1_000_000_000
    msg.header.frame_id = "camera"


def _image(t_ns: int, k: int) -> Image:
    m = Image()
    _stamp(m, t_ns)
    h = w = 64
    m.height, m.width, m.encoding = h, w, "rgb8"
    m.step = w * 3
    yy, xx = np.mgrid[0:h, 0:w]
    band = np.clip((xx.astype(np.int32) - (k * 3) % w), 0, 255)
    img = np.stack(
        [
            ((k * 11) % 256) * np.ones((h, w), np.uint8),
            band.astype(np.uint8),
            ((yy * 4) % 256).astype(np.uint8),
        ],
        axis=2,
    )
    m.data = img.tobytes()
    return m


def _odom(t_ns: int, k: int) -> Odometry:
    m = Odometry()
    _stamp(m, t_ns)
    m.pose.pose.position.x = 0.1 * k
    m.pose.pose.position.y = 0.02 * np.sin(k * 0.2)
    m.pose.pose.orientation.w = 1.0
    m.twist.twist.linear.x = 0.5
    m.twist.twist.angular.z = 0.15 * np.sin(k * 0.25)
    return m


def _cmd(k: int) -> Twist:
    m = Twist()
    m.linear.x = 0.5
    # left / straight / right segments so counterfactual branches diverge.
    if k < 16:
        m.angular.z = -0.6
    elif k < 32:
        m.angular.z = 0.0
    else:
        m.angular.z = 0.6
    return m


def build(path: str = OUT) -> str:
    import shutil

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if os.path.exists(path):
        shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=path, storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    for tid, (name, typ) in enumerate(
        (
            (IMG, "sensor_msgs/msg/Image"),
            (ODOM, "nav_msgs/msg/Odometry"),
            (CMD, "geometry_msgs/msg/Twist"),
        )
    ):
        writer.create_topic(
            rosbag2_py.TopicMetadata(id=tid, name=name, type=typ, serialization_format="cdr")
        )
    t0 = 1_000_000_000
    for k in range(N):
        t = t0 + k * DT_NS
        writer.write(IMG, serialize_message(_image(t, k)), t)
        writer.write(ODOM, serialize_message(_odom(t + 2_000_000, k)), t + 2_000_000)
        writer.write(CMD, serialize_message(_cmd(k)), t + 1_000_000)
    del writer
    return path


def main() -> None:
    path = build()
    size_kb = os.path.getsize(path) / 1024
    print(f"wrote {path} ({N} frames, {size_kb:.1f} KiB)")


if __name__ == "__main__":
    main()
