"""In-process check: runtime (autostart) + viewer + probe.
Verifies the imagination MarkerArray is produced from a published Observation.
"""
import sys
import time
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from world_model_msgs.msg import Observation as ObservationMsg
from visualization_msgs.msg import Marker, MarkerArray
from world_model_py.runtime_node import WorldModelRuntime
from world_model_py import conversions as conv
from world_model_viz.occupancy_marker_node import OccupancyMarkerNode


class Probe(Node):
    def __init__(self):
        super().__init__("probe")
        self.markers = None
        self.pub = self.create_publisher(ObservationMsg, "/world_model_runtime/observation", 10)
        self.create_subscription(MarkerArray, "/world_model_viz/imagination", self._m, 10)
        self.create_timer(0.2, self._tick)

    def _tick(self):
        msg = ObservationMsg()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        msg.image = conv.np_to_image_msg(np.zeros((32, 32, 3), np.uint8), msg.header)
        msg.ego_state = [1.0, 0.0, 0.0, 0.5]
        msg.action_dim = 2
        self.pub.publish(msg)

    def _m(self, m):
        self.markers = m


def main():
    rclpy.init()
    runtime = WorldModelRuntime()
    viewer = OccupancyMarkerNode()
    probe = Probe()
    ex = SingleThreadedExecutor()
    for n in (runtime, viewer, probe):
        ex.add_node(n)

    rc = 1
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 20.0:
        ex.spin_once(timeout_sec=0.1)
        if probe.markers is not None:
            ms = probe.markers.markers
            cube = [x for x in ms if x.type == Marker.CUBE_LIST and len(x.points) > 0]
            text = [x for x in ms if x.type == Marker.TEXT_VIEW_FACING]
            deleteall = [x for x in ms if x.action == Marker.DELETEALL]
            total_pts = sum(len(x.points) for x in cube)
            colors_ok = all(len(x.colors) == len(x.points) for x in cube)
            print(f"markers={len(ms)} deleteall={len(deleteall)} "
                  f"cube_lists={len(cube)} text={len(text)} total_points={total_pts} "
                  f"colors_match_points={colors_ok}")
            if cube and text and deleteall and total_pts > 0 and colors_ok:
                print(f"sample text: '{text[0].text}'")
                print("VIZ_OK")
                rc = 0
            else:
                print("VIZ_BAD")
            break
    else:
        print("VIZ_TIMEOUT")

    for n in (runtime, viewer, probe):
        n.destroy_node()
    rclpy.shutdown()
    return rc


if __name__ == "__main__":
    sys.exit(main())
