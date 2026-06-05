"""Publishes synthetic ``Observation`` messages so the demo works with no
robot, no bag and no dataset.

    ros2 run world_model_py sample_publisher
"""
from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node

from world_model_msgs.msg import Observation as ObservationMsg
from . import conversions as conv


class SampleObservationPublisher(Node):
    def __init__(self):
        super().__init__("sample_observation_publisher")
        self.declare_parameter("rate_hz", 5.0)
        self.declare_parameter("image_size", 64)
        self._pub = self.create_publisher(ObservationMsg, "observation", 10)
        rate = self.get_parameter("rate_hz").get_parameter_value().double_value or 5.0
        self._size = int(self.get_parameter("image_size").get_parameter_value().integer_value) or 64
        self._rng = np.random.default_rng(0)
        self._k = 0
        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(f"publishing synthetic observations on 'observation' at {rate} Hz")

    def _tick(self) -> None:
        self._k += 1
        img = (self._rng.random((self._size, self._size, 3)) * 255).astype(np.uint8)
        msg = ObservationMsg()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        msg.image = conv.np_to_image_msg(img, msg.header)
        msg.ego_state = [1.0, 0.0, 0.05 * self._k, 0.5]
        msg.action_history = [0.0, 0.0, 0.1, 0.0]
        msg.action_dim = 2
        msg.instruction = "move forward"
        self._pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SampleObservationPublisher()
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
