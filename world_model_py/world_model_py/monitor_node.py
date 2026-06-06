"""Runtime anomaly / OOD monitor node.

Watches a camera through a World Model and raises a flag when the latent
surprise leaves the nominal band — a drop-in safety/OOD monitor for any ROS 2
robot. Calibrates online on nominal operation; needs no failure data.

    ros2 run world_model_py monitor_node --ros-args -p adapter:=ijepa

Subscribes:  ~/image            (sensor_msgs/Image)
Publishes:   ~/surprise         (std_msgs/Float32)   latent novelty
             ~/anomaly_threshold(std_msgs/Float32)   adaptive threshold
             ~/anomaly          (std_msgs/Bool)      this frame is anomalous
"""
from __future__ import annotations

import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool, Float32
from sensor_msgs.msg import Image

from .adapters import Observation
from .anomaly import AnomalyDetector
from . import conversions as conv
from .registry import load_model


class WorldModelMonitor(Node):
    def __init__(self, **kwargs):
        super().__init__("world_model_monitor", **kwargs)
        self.declare_parameter("adapter", "ijepa")
        self.declare_parameter("model_id", "facebook/ijepa_vith14_1k")
        self.declare_parameter("image_topic", "image")
        self.declare_parameter("window", 12)
        self.declare_parameter("k", 4.0)

        name = self.get_parameter("adapter").get_parameter_value().string_value
        kwargs = {}
        if name in ("ijepa",):
            kwargs["model_id"] = self.get_parameter("model_id").get_parameter_value().string_value
        self._adapter = load_model(name, **kwargs)
        self._det = AnomalyDetector(
            window=self.get_parameter("window").get_parameter_value().integer_value,
            k=self.get_parameter("k").get_parameter_value().double_value,
        )

        self._pub_s = self.create_publisher(Float32, "~/surprise", 10)
        self._pub_t = self.create_publisher(Float32, "~/anomaly_threshold", 10)
        self._pub_a = self.create_publisher(Bool, "~/anomaly", 10)
        topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.create_subscription(Image, topic, self._on_image, 10)
        self.get_logger().info(f"world model monitor: adapter '{name}', watching '{topic}'")

    def _on_image(self, msg: Image) -> None:
        img = conv.image_msg_to_np(msg)
        if img is None:
            return
        try:
            surprise = float(self._adapter.predict_future(Observation(image=img), horizon=1).risk)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"adapter failed: {exc}")
            return
        res = self._det.update(surprise)
        self._pub_s.publish(Float32(data=res["surprise"]))
        self._pub_t.publish(Float32(data=res["threshold"]))
        self._pub_a.publish(Bool(data=res["anomaly"]))
        if res["anomaly"]:
            self.get_logger().warn(
                f"ANOMALY: surprise {surprise:.3f} > threshold {res['threshold']:.3f}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WorldModelMonitor()
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
