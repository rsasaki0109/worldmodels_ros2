"""Preview what the World Model costmap layer would stamp (RViz markers).

Subscribes to ``FutureOccupancy``, unions lethal cells over the horizon (same
rule as ``world_model_costmap::WorldModelLayer``), and draws them on the ground.

    ros2 run world_model_viz costmap_preview_node
"""
from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from world_model_msgs.msg import FutureOccupancy


class CostmapPreviewNode(Node):
    def __init__(self):
        super().__init__("costmap_preview")
        self.declare_parameter("topic", "/world_model_runtime/future_occupancy")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("threshold", 50)
        self.declare_parameter("cell_scale", 0.1)

        self._frame = self.get_parameter("frame_id").get_parameter_value().string_value
        self._thresh = int(self.get_parameter("threshold").get_parameter_value().integer_value)
        self._scale = float(self.get_parameter("cell_scale").get_parameter_value().double_value)
        topic = self.get_parameter("topic").get_parameter_value().string_value
        self._pub = self.create_publisher(MarkerArray, "~/predicted_lethal", 10)
        self.create_subscription(FutureOccupancy, topic, self._on_occ, 10)
        self.get_logger().info(f"costmap preview: {topic} -> ~/predicted_lethal")

    def _on_occ(self, msg: FutureOccupancy) -> None:
        arr = MarkerArray()
        clear = Marker()
        clear.header.frame_id = self._frame
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        marker = Marker()
        marker.header = msg.header
        marker.header.frame_id = self._frame
        marker.ns = "predicted_lethal"
        marker.id = 0
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale = Vector3(x=self._scale, y=self._scale, z=0.05)

        for grid in msg.grids:
            res = float(grid.info.resolution) or self._scale
            ox = float(grid.info.origin.position.x)
            oy = float(grid.info.origin.position.y)
            w, h = int(grid.info.width), int(grid.info.height)
            if w == 0 or h == 0:
                continue
            data = np.asarray(grid.data, dtype=np.int16).reshape(h, w)
            ys, xs = np.where(data >= self._thresh)
            for j, i in zip(xs.tolist(), ys.tolist()):
                occ = float(data[i, j]) / 100.0
                marker.points.append(
                    Point(x=ox + (j + 0.5) * res, y=oy + (i + 0.5) * res, z=0.02)
                )
                marker.colors.append(
                    ColorRGBA(r=1.0, g=0.2, b=0.1, a=float(np.clip(0.35 + 0.55 * occ, 0.0, 1.0)))
                )

        if marker.points:
            arr.markers.append(marker)
        self._pub.publish(arr)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CostmapPreviewNode()
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
