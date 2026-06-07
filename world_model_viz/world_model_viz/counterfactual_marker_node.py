"""Visualise ImagineFutures counterfactual branches in RViz.

Watches for a recorded experience memory, then (when the camera stream goes
idle after a bag replay) calls ``/world_model_planning/imagine_futures`` and
publishes:

- ``~/counterfactual/mosaic``  — side-by-side endpoint frames (sensor_msgs/Image)
- ``~/counterfactual/markers`` — branch labels + coloured panels (MarkerArray)

    ros2 run world_model_viz counterfactual_marker_node

Set ``auto_imagine:=false`` and publish ``True`` on ``~/trigger`` to fire manually.
"""
from __future__ import annotations

import os
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Point, Vector3
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, ColorRGBA, Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray

from world_model_msgs.srv import ImagineFutures

from .counterfactual import (
    BRANCH_COLORS,
    branch_label,
    build_mosaic,
    image_msg_to_rgb,
    rgb_to_image_msg,
)


class CounterfactualMarkerNode(Node):
    def __init__(self):
        super().__init__("counterfactual_viz")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("memory_path", "/tmp/world_model_experience.npz")
        self.declare_parameter("service_name", "/world_model_planning/imagine_futures")
        self.declare_parameter("steering_options", [-0.7, 0.0, 0.7])
        self.declare_parameter("horizon", 12)
        self.declare_parameter("auto_imagine", True)
        self.declare_parameter("idle_timeout_sec", 3.0)
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("panel_spacing", 1.6)

        self._frame = self.get_parameter("frame_id").get_parameter_value().string_value
        self._memory_path = self.get_parameter("memory_path").get_parameter_value().string_value
        self._idle = float(self.get_parameter("idle_timeout_sec").get_parameter_value().double_value)
        self._auto = self.get_parameter("auto_imagine").get_parameter_value().bool_value
        self._spacing = float(self.get_parameter("panel_spacing").get_parameter_value().double_value)

        self._latest_image: Image | None = None
        self._last_image_t = 0.0
        self._imagined = False
        self._pending = False

        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.create_subscription(Image, image_topic, self._on_image, qos_profile_sensor_data)
        self.create_subscription(Bool, "~/trigger", self._on_trigger, 10)

        self._pub_mosaic = self.create_publisher(Image, "~/counterfactual/mosaic", 10)
        self._pub_markers = self.create_publisher(MarkerArray, "~/counterfactual/markers", 10)
        self._pub_div = self.create_publisher(Float32MultiArray, "~/counterfactual/divergence", 10)

        svc = self.get_parameter("service_name").get_parameter_value().string_value
        self._client = self.create_client(ImagineFutures, svc)
        self.create_timer(0.5, self._tick)
        self.get_logger().info(
            f"counterfactual viewer: image='{image_topic}', memory='{self._memory_path}', "
            f"service='{svc}', auto_imagine={self._auto}"
        )

    def _steering_options(self) -> list[float]:
        return list(self.get_parameter("steering_options").value)

    def _on_image(self, msg: Image) -> None:
        self._latest_image = msg
        self._last_image_t = time.monotonic()

    def _on_trigger(self, msg: Bool) -> None:
        if msg.data:
            self._imagined = False
            self._try_imagine(force=True)

    def _tick(self) -> None:
        if not self._auto or self._imagined or self._pending:
            return
        if self._latest_image is None:
            return
        if time.monotonic() - self._last_image_t < self._idle:
            return
        if not os.path.isfile(self._memory_path):
            return
        self._try_imagine()

    def _try_imagine(self, force: bool = False) -> None:
        if self._pending or self._latest_image is None:
            return
        if self._imagined and not force:
            return
        if not self._client.wait_for_service(timeout_sec=0.0):
            return

        req = ImagineFutures.Request()
        req.current_image = self._latest_image
        req.steering_options = [float(x) for x in self._steering_options()]
        req.horizon = int(self.get_parameter("horizon").value)
        self._pending = True
        fut = self._client.call_async(req)
        fut.add_done_callback(self._on_response)

    def _on_response(self, fut) -> None:
        self._pending = False
        try:
            resp = fut.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"imagine_futures failed: {exc}")
            return
        if not resp.success:
            self.get_logger().warn("imagine_futures returned success=false")
            return
        self._imagined = True
        self._publish(resp)
        self.get_logger().info(
            f"counterfactual mosaic published ({len(resp.branches)} branches, "
            f"max divergence {max(resp.divergence) if resp.divergence else 0:.3f})"
        )

    def _publish(self, resp: ImagineFutures.Response) -> None:
        opts = self._steering_options()
        labels = [branch_label(s) for s in opts]
        frames: list[np.ndarray] = []
        for branch in resp.branches:
            if branch.frames:
                frames.append(image_msg_to_rgb(branch.frames[-1]))
            else:
                frames.append(np.full((32, 32, 3), 48, dtype=np.uint8))

        header = self._latest_image.header if self._latest_image else resp.branches[0].header
        mosaic = build_mosaic(frames, labels)
        self._pub_mosaic.publish(rgb_to_image_msg(mosaic, header))

        div = Float32MultiArray()
        div.data = [float(x) for x in resp.divergence]
        self._pub_div.publish(div)

        arr = MarkerArray()
        clear = Marker()
        clear.header.frame_id = self._frame
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        n = len(resp.branches)
        for i, branch in enumerate(resp.branches):
            x = (i - (n - 1) / 2.0) * self._spacing
            color = BRANCH_COLORS[i % len(BRANCH_COLORS)]
            panel = Marker()
            panel.header = header
            panel.header.frame_id = self._frame
            panel.ns = "counterfactual_panel"
            panel.id = i
            panel.type = Marker.CUBE
            panel.action = Marker.ADD
            panel.pose.position.x = x
            panel.pose.position.z = 0.5
            panel.pose.orientation.w = 1.0
            panel.scale = Vector3(x=1.2, y=0.08, z=0.9)
            panel.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=0.85)
            arr.markers.append(panel)

            text = Marker()
            text.header = header
            text.header.frame_id = self._frame
            text.ns = "counterfactual_label"
            text.id = i
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position = Point(x=x, y=0.0, z=1.2)
            text.pose.orientation.w = 1.0
            text.scale.z = 0.25
            div_s = f"{resp.divergence[i]:.2f}" if i < len(resp.divergence) else "--"
            text.text = f"{labels[i]}  d={div_s}"
            text.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=1.0)
            arr.markers.append(text)

        self._pub_markers.publish(arr)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CounterfactualMarkerNode()
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
