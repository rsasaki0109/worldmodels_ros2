"""Turn a World Model's imagined future into RViz markers.

Subscribes to ``FutureOccupancy`` and ``RiskScore`` and republishes a
``visualization_msgs/MarkerArray`` that stacks the predicted occupancy along
+z (the higher the layer, the further into the future), colours it from green
(near) to red (far), and fades each cell by its occupancy probability. A text
marker shows the current risk score.

    ros2 run world_model_viz occupancy_marker_node

This is the "imagination viewer" -- it works with the GPU-free dummy adapter,
so you get a moving picture with no robot, no dataset and no GPU.
"""
from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node

from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point, Vector3
from visualization_msgs.msg import Marker, MarkerArray

from world_model_msgs.msg import FutureOccupancy, RiskScore


def _ramp(frac: float) -> tuple:
    """Green (near future) -> yellow -> red (far future)."""
    frac = float(np.clip(frac, 0.0, 1.0))
    return (frac, 1.0 - frac, 0.0)


class OccupancyMarkerNode(Node):
    def __init__(self):
        super().__init__("world_model_viz")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("threshold", 20)      # min occupancy [0..100] to draw
        self.declare_parameter("z_step", 0.2)        # metres between future layers
        self.declare_parameter("occupancy_topic", "/world_model_runtime/future_occupancy")
        self.declare_parameter("risk_topic", "/world_model_runtime/risk_score")

        self._frame = self.get_parameter("frame_id").get_parameter_value().string_value
        self._thresh = int(self.get_parameter("threshold").get_parameter_value().integer_value)
        self._z_step = float(self.get_parameter("z_step").get_parameter_value().double_value)
        occ_topic = self.get_parameter("occupancy_topic").get_parameter_value().string_value
        risk_topic = self.get_parameter("risk_topic").get_parameter_value().string_value

        self._risk = None
        self._pub = self.create_publisher(MarkerArray, "~/imagination", 10)
        self.create_subscription(FutureOccupancy, occ_topic, self._on_occ, 10)
        self.create_subscription(RiskScore, risk_topic, self._on_risk, 10)
        self.get_logger().info(
            f"imagination viewer: {occ_topic} -> ~/imagination (frame '{self._frame}')"
        )

    def _on_risk(self, msg: RiskScore) -> None:
        self._risk = msg

    def _on_occ(self, msg: FutureOccupancy) -> None:
        array = MarkerArray()
        # clear previous frame so cell counts can shrink between predictions.
        clear = Marker()
        clear.header.frame_id = self._frame
        clear.action = Marker.DELETEALL
        array.markers.append(clear)

        horizon = max(1, len(msg.grids))
        for k, grid in enumerate(msg.grids):
            marker = self._grid_to_marker(grid, k, horizon, msg.header.stamp)
            if marker is not None:
                array.markers.append(marker)

        array.markers.append(self._risk_marker(msg.header.stamp, horizon))
        self._pub.publish(array)

    def _grid_to_marker(self, grid, k: int, horizon: int, stamp) -> Marker | None:
        res = grid.info.resolution or 0.1
        w, h = grid.info.width, grid.info.height
        if w == 0 or h == 0:
            return None
        data = np.asarray(grid.data, dtype=np.int16).reshape(h, w)
        ys, xs = np.where(data >= self._thresh)
        if xs.size == 0:
            return None

        ox = grid.info.origin.position.x
        oy = grid.info.origin.position.y
        z = k * self._z_step + self._z_step * 0.5
        r, g, b = _ramp(k / max(1, horizon - 1))

        marker = Marker()
        marker.header.frame_id = self._frame
        marker.header.stamp = stamp
        marker.ns = "future_occupancy"
        marker.id = k
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.scale = Vector3(x=res, y=res, z=res)
        marker.pose.orientation.w = 1.0
        marker.frame_locked = True

        for j, i in zip(xs.tolist(), ys.tolist()):
            marker.points.append(
                Point(x=ox + (j + 0.5) * res, y=oy + (i + 0.5) * res, z=z)
            )
            occ = float(data[i, j]) / 100.0
            marker.colors.append(
                ColorRGBA(r=r, g=g, b=b, a=float(np.clip(0.25 + 0.65 * occ, 0.0, 1.0)))
            )
        return marker

    def _risk_marker(self, stamp, horizon: int) -> Marker:
        marker = Marker()
        marker.header.frame_id = self._frame
        marker.header.stamp = stamp
        marker.ns = "risk"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.z = horizon * self._z_step + 0.6
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.4
        if self._risk is not None:
            risk = float(self._risk.score)
            marker.text = f"risk {risk:.2f} [{self._risk.label}]"
            marker.color = ColorRGBA(r=float(risk), g=float(1.0 - risk), b=0.0, a=1.0)
        else:
            marker.text = "risk --"
            marker.color = ColorRGBA(r=0.7, g=0.7, b=0.7, a=1.0)
        return marker


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OccupancyMarkerNode()
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
