"""Synthetic camera for the anomaly-monitor demo (GPU-free).

Publishes a sensor_msgs/Image stream that is normally "nominal" (a smoothly
moving block on a gradient) and periodically injects an *anomaly* event: the
lens is briefly occluded by a dark blob. Point the monitor at it and, with a
real adapter (``ijepa``), watch ``~/anomaly`` fire during the occlusions.

    ros2 run world_model_py camera_sim
"""
from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from . import conversions as conv


class CameraSim(Node):
    def __init__(self):
        super().__init__("camera_sim")
        self.declare_parameter("rate_hz", 4.0)
        self.declare_parameter("size", 256)
        self.declare_parameter("period", 40)      # frames between anomaly events
        self.declare_parameter("event_len", 8)    # occlusion length in frames

        rate = self.get_parameter("rate_hz").get_parameter_value().double_value or 4.0
        self._n = int(self.get_parameter("size").get_parameter_value().integer_value) or 256
        self._period = int(self.get_parameter("period").get_parameter_value().integer_value)
        self._event = int(self.get_parameter("event_len").get_parameter_value().integer_value)
        self._k = 0

        s = self._n
        yy, xx = np.mgrid[0:s, 0:s]
        self._bg = np.stack([xx / s * 170 + 40, yy / s * 110 + 30, np.full((s, s), 90)], 2).astype(np.uint8)
        self._yy, self._xx = yy, xx

        self._pub = self.create_publisher(Image, "image", 10)
        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            f"camera_sim: {rate} Hz, anomaly every {self._period} frames for {self._event} frames")

    def _frame(self) -> np.ndarray:
        s = self._n
        img = self._bg.copy()
        # nominal: a block gliding back and forth
        cx = int((0.5 + 0.4 * np.sin(0.15 * self._k)) * s)
        img[s // 2 - 24:s // 2 + 24, max(0, cx - 24):cx + 24] = [210, 60, 60]
        # periodic anomaly: a dark occluder sweeps across the lens
        phase = self._k % self._period
        if phase < self._event:
            prog = phase / max(1, self._event - 1)
            ox = int(40 + prog * (s - 80))
            a = np.clip(1.25 - (((self._xx - ox) / 120.0) ** 2 + ((self._yy - s / 2) / 165.0) ** 2), 0, 1) * 0.93
            img = (img * (1 - a[..., None]) + np.array([16, 16, 20]) * a[..., None]).astype(np.uint8)
        return img

    def _tick(self) -> None:
        self._k += 1
        msg_header_img = Image()
        msg_header_img.header.stamp = self.get_clock().now().to_msg()
        msg_header_img.header.frame_id = "camera"
        out = conv.np_to_image_msg(self._frame(), msg_header_img.header)
        self._pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CameraSim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
