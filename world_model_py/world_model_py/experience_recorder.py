"""Record (latent, action, next_latent) transitions while a bag is replayed.

Subscribes to the same camera/action topics as ``bag_relay``, encodes each
frame with a World Model adapter, and writes ``experience.npz`` on shutdown or
after the stream goes idle (bag finished).

    ros2 run world_model_py experience_recorder --ros-args \\
        -p output_path:=/tmp/experience.npz -p adapter:=dummy
"""
from __future__ import annotations

import time

import numpy as np
import rclpy
from rclpy.node import Node
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import Image

from world_model_py.adapters import Observation
from world_model_py.experience import save_experience
from world_model_py.registry import load_model
from world_model_py.topic_vectors import extract_action_vector

from . import conversions as conv


class ExperienceRecorder(Node):
    def __init__(self, *, parameter_overrides=None, **adapter_kwargs):
        super().__init__("experience_recorder", parameter_overrides=parameter_overrides or [])
        self.declare_parameter("adapter", "dummy")
        self.declare_parameter("model_id", "facebook/ijepa_vith14_1k")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("action_topic", "/cmd_vel")
        self.declare_parameter("action_type", "geometry_msgs/msg/Twist")
        self.declare_parameter("action_mode", "scalar")
        self.declare_parameter("output_path", "/tmp/world_model_experience.npz")
        self.declare_parameter("frame_size", 96)
        self.declare_parameter("store_frames", True)
        self.declare_parameter("idle_timeout_sec", 3.0)
        self.declare_parameter("min_transitions", 4)

        name = self.get_parameter("adapter").get_parameter_value().string_value
        if name == "ijepa":
            adapter_kwargs.setdefault(
                "model_id",
                self.get_parameter("model_id").get_parameter_value().string_value,
            )
        self._adapter = load_model(name, **adapter_kwargs)

        self._frame_size = int(self.get_parameter("frame_size").get_parameter_value().integer_value) or 96
        self._store_frames = self.get_parameter("store_frames").get_parameter_value().bool_value

        self._latents: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._frames: list[np.ndarray] = []
        self._latest_action = np.zeros(1, dtype=np.float32)
        self._stable_since = 0.0
        self._last_count = 0
        self._saved = False

        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        action_topic = self.get_parameter("action_topic").get_parameter_value().string_value.strip()
        if action_topic:
            action_type = self.get_parameter("action_type").get_parameter_value().string_value
            msg_type = get_message(action_type)
            self._action_mode = self.get_parameter("action_mode").get_parameter_value().string_value
            self._action_type = action_type
            self.create_subscription(msg_type, action_topic, self._on_action, 10)

        self.create_subscription(Image, image_topic, self._on_image, 10)
        self.create_timer(0.5, self._check_idle)
        self.get_logger().info(
            f"experience recorder: adapter='{name}', image='{image_topic}'"
            + (f", action='{action_topic}'" if action_topic else "")
            + f" -> '{self.get_parameter('output_path').value}'"
        )

    def _on_action(self, msg) -> None:
        self._latest_action = extract_action_vector(msg, self._action_type, self._action_mode)

    def _resize_frame(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        if h == self._frame_size and w == self._frame_size:
            return rgb
        # nearest-neighbour resize without PIL dependency
        ys = (np.linspace(0, h - 1, self._frame_size)).astype(np.int32)
        xs = (np.linspace(0, w - 1, self._frame_size)).astype(np.int32)
        return rgb[np.ix_(ys, xs)]

    def _on_image(self, msg: Image) -> None:
        arr = conv.image_msg_to_np(msg)
        if arr is None:
            return
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        lat = np.asarray(self._adapter.encode(Observation(image=arr)).data, np.float32).ravel()
        self._latents.append(lat)
        self._actions.append(self._latest_action.copy())
        if self._store_frames:
            self._frames.append(self._resize_frame(arr.astype(np.uint8)))

    def _check_idle(self) -> None:
        if self._saved:
            return
        n = len(self._latents)
        if n == 0:
            return
        if n != self._last_count:
            self._last_count = n
            self._stable_since = time.monotonic()
            return
        idle = float(self.get_parameter("idle_timeout_sec").get_parameter_value().double_value)
        if time.monotonic() - self._stable_since < idle:
            return
        self._flush("stream idle (bag likely finished)")

    def _flush(self, reason: str) -> None:
        if self._saved:
            return
        n = len(self._latents)
        min_tr = int(self.get_parameter("min_transitions").get_parameter_value().integer_value)
        if n < min_tr + 1:
            self.get_logger().warn(
                f"only {max(0, n - 1)} transitions; need >= {min_tr} — not saving"
            )
            return
        lat = np.stack(self._latents, axis=0)
        act = np.stack(self._actions, axis=0)
        frames = np.stack(self._frames, axis=0) if self._store_frames and self._frames else None
        out = self.get_parameter("output_path").get_parameter_value().string_value
        summary = save_experience(out, lat, act, frames=frames)
        self._saved = True
        self.get_logger().info(
            f"saved experience ({reason}): {summary['transitions']} transitions, "
            f"action_dim={summary['action_dim']} -> {summary['path']}"
        )

    def destroy_node(self):
        if not self._saved:
            self._flush("shutdown")
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ExperienceRecorder()
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
