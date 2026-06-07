"""Pick a path that avoids predicted occupancy — costmap-layer logic in Python.

Subscribes to ``FutureOccupancy``, scores a few candidate paths by how much they
intersect the predicted lethal union (what ``WorldModelLayer`` would stamp),
and publishes the naive vs safest paths for RViz.

    ros2 run world_model_nav2 avoidance_demo_node
"""
from __future__ import annotations

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from world_model_msgs.msg import FutureOccupancy

from .occupancy_path import (
    default_candidates,
    merge_lethal_cells,
    path_collision_risk,
)


class AvoidanceDemoNode(Node):
    def __init__(self):
        super().__init__("avoidance_demo")
        self.declare_parameter("topic", "/world_model_runtime/future_occupancy")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("rate_hz", 2.0)

        self._frame = self.get_parameter("frame_id").get_parameter_value().string_value
        self._latest = None
        topic = self.get_parameter("topic").get_parameter_value().string_value
        self._pub = self.create_publisher(MarkerArray, "~/avoidance_paths", 10)
        self.create_subscription(FutureOccupancy, topic, self._on_occ, 10)
        rate = float(self.get_parameter("rate_hz").get_parameter_value().double_value) or 2.0
        self.create_timer(1.0 / rate, self._publish)
        self.get_logger().info(f"avoidance demo: scoring paths against '{topic}'")

    def _on_occ(self, msg: FutureOccupancy) -> None:
        self._latest = msg

    def _publish(self) -> None:
        if self._latest is None or not self._latest.grids:
            return
        lethal = merge_lethal_cells(self._latest.grids)
        scored = []
        for name, xy in default_candidates():
            risk = path_collision_risk(xy, lethal)
            scored.append((name, xy, risk))
        scored.sort(key=lambda t: t[2])
        best_name, best_xy, best_risk = scored[0]
        naive_name, naive_xy, naive_risk = scored[0]
        for name, xy, risk in scored:
            if name == "straight":
                naive_name, naive_xy, naive_risk = name, xy, risk
                break

        arr = MarkerArray()
        clear = Marker()
        clear.header.frame_id = self._frame
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        stamp = self._latest.header.stamp
        arr.markers.append(self._line(naive_xy, naive_risk, 0, "naive", highlight=False, stamp=stamp))
        arr.markers.append(self._line(best_xy, best_risk, 1, best_name, highlight=True, stamp=stamp))

        label = Marker()
        label.header.frame_id = self._frame
        label.header.stamp = stamp
        label.ns = "avoidance_label"
        label.id = 0
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position = Point(x=1.6, y=-0.35, z=0.4)
        label.pose.orientation.w = 1.0
        label.scale.z = 0.22
        label.text = (
            f"predicted lethal union  |  naive {naive_risk:.2f}  ->  "
            f"avoid {best_risk:.2f} ({best_name})"
        )
        label.color = ColorRGBA(r=0.9, g=0.9, b=0.95, a=1.0)
        arr.markers.append(label)
        self._pub.publish(arr)

    def _line(self, xy, risk, mid, name, *, highlight: bool, stamp):
        m = Marker()
        m.header.frame_id = self._frame
        m.header.stamp = stamp
        m.ns = "avoidance_paths"
        m.id = mid
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.14 if highlight else 0.08
        m.pose.orientation.w = 1.0
        if highlight:
            m.color = ColorRGBA(r=0.15, g=0.95, b=0.35, a=1.0)
        else:
            m.color = ColorRGBA(r=float(risk), g=float(1.0 - risk), b=0.05, a=0.95)
        for x, y in xy:
            m.points.append(Point(x=float(x), y=float(y), z=0.05))
        return m


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AvoidanceDemoNode()
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
