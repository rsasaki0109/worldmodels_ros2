"""Relay rosbag2 topics into ``world_model_msgs/Observation`` for the runtime.

Sits between ``ros2 bag play`` and ``runtime_node``: subscribes to a camera
topic (required) and optional state/action topics, then publishes an
``Observation`` on each camera frame with the latest state/action joined.

    ros2 run world_model_py bag_relay --ros-args \\
        -p image_topic:=/camera/image_raw \\
        -p action_topic:=/cmd_vel
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import rclpy
from rclpy.node import Node
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import Image

from world_model_msgs.msg import Observation as ObservationMsg

from . import conversions as conv
from .topic_vectors import _ACTION_EXTRACTORS, _STATE_EXTRACTORS, extract_action_vector


def build_observation(
    image_msg: Image,
    ego_state: Optional[list[float]],
    action_history: list[list[float]],
    action_dim: int,
    instruction: str,
) -> ObservationMsg:
    """Assemble an Observation message from already-extracted fields."""
    msg = ObservationMsg()
    msg.header = image_msg.header
    if not msg.header.frame_id:
        msg.header.frame_id = "camera"
    msg.image = image_msg
    msg.ego_state = list(ego_state) if ego_state is not None else []
    flat = [v for row in action_history for v in row]
    msg.action_history = flat
    msg.action_dim = int(action_dim) if action_dim else 0
    msg.instruction = instruction
    return msg


class BagObservationRelay(Node):
    def __init__(self):
        super().__init__("bag_observation_relay")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("state_topic", "")
        self.declare_parameter("state_type", "nav_msgs/msg/Odometry")
        self.declare_parameter("action_topic", "")
        self.declare_parameter("action_type", "geometry_msgs/msg/Twist")
        self.declare_parameter("action_mode", "full")
        self.declare_parameter("observation_topic", "/world_model_runtime/observation")
        self.declare_parameter("action_history_len", 8)
        self.declare_parameter("action_dim", 0)
        self.declare_parameter("instruction", "")

        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        state_topic = self.get_parameter("state_topic").get_parameter_value().string_value.strip()
        action_topic = self.get_parameter("action_topic").get_parameter_value().string_value.strip()
        obs_topic = self.get_parameter("observation_topic").get_parameter_value().string_value
        history_len = int(self.get_parameter("action_history_len").get_parameter_value().integer_value) or 8
        self._action_dim = int(self.get_parameter("action_dim").get_parameter_value().integer_value)
        self._instruction = self.get_parameter("instruction").get_parameter_value().string_value
        self._action_mode = self.get_parameter("action_mode").get_parameter_value().string_value

        self._latest_state: Optional[list[float]] = None
        self._action_history: deque[list[float]] = deque(maxlen=max(1, history_len))
        self._pub = self.create_publisher(ObservationMsg, obs_topic, 10)
        self._published = 0

        self.create_subscription(Image, image_topic, self._on_image, 10)
        if state_topic:
            self._subscribe_vector(
                state_topic,
                self.get_parameter("state_type").get_parameter_value().string_value,
                _STATE_EXTRACTORS,
                self._on_state,
                "state",
            )
        if action_topic:
            action_type = self.get_parameter("action_type").get_parameter_value().string_value
            self._subscribe_vector(action_topic, action_type, _ACTION_EXTRACTORS, self._on_action, "action")
            if self._action_dim <= 0:
                sample = extract_action_vector(
                    self._zero_action(action_type), action_type, self._action_mode
                )
                self._action_dim = int(sample.size)

        self.get_logger().info(
            f"bag relay: image='{image_topic}'"
            + (f", state='{state_topic}'" if state_topic else "")
            + (f", action='{action_topic}'" if action_topic else "")
            + f" -> '{obs_topic}'"
        )

    @staticmethod
    def _zero_action(type_str: str):
        from geometry_msgs.msg import Twist, TwistStamped
        from nav_msgs.msg import Odometry

        if type_str == "geometry_msgs/msg/TwistStamped":
            return TwistStamped().twist
        if type_str == "geometry_msgs/msg/Twist":
            return Twist()
        if type_str == "nav_msgs/msg/Odometry":
            return Odometry()
        return get_message(type_str)()

    def _subscribe_vector(self, topic, type_str, table, callback, label):
        if type_str not in table:
            raise ValueError(
                f"unsupported {label} type '{type_str}'. supported: {sorted(table)}"
            )
        msg_type = get_message(type_str)
        self.create_subscription(msg_type, topic, callback, 10)

    def _on_state(self, msg) -> None:
        type_str = self.get_parameter("state_type").get_parameter_value().string_value
        self._latest_state = [float(v) for v in _STATE_EXTRACTORS[type_str](msg)]

    def _on_action(self, msg) -> None:
        type_str = self.get_parameter("action_type").get_parameter_value().string_value
        vec = extract_action_vector(msg, type_str, self._action_mode)
        self._action_history.append([float(v) for v in vec])
        if self._action_dim <= 0:
            self._action_dim = int(vec.size)

    def _on_image(self, msg: Image) -> None:
        if conv.image_msg_to_np(msg) is None:
            return
        out = build_observation(
            msg,
            self._latest_state,
            list(self._action_history),
            self._action_dim,
            self._instruction,
        )
        self._pub.publish(out)
        self._published += 1
        if self._published == 1:
            self.get_logger().info("published first Observation from bag stream")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BagObservationRelay()
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
