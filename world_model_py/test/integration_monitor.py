"""In-process wiring check for the anomaly monitor node (dummy adapter, no GPU).

Verifies the ROS surface end to end: an Image on the monitor's input topic
produces messages on ~/surprise, ~/anomaly_threshold and ~/anomaly. Run with
ROS sourced:

    python3 world_model_py/test/integration_monitor.py
"""
import sys
import time

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter

from std_msgs.msg import Bool, Float32
from sensor_msgs.msg import Image

from world_model_py.monitor_node import WorldModelMonitor
from world_model_py import conversions as conv


class Probe(Node):
    def __init__(self):
        super().__init__("probe")
        self.got = {"surprise": None, "threshold": None, "anomaly": None}
        self.pub = self.create_publisher(Image, "/image", 10)
        self.create_subscription(Float32, "/world_model_monitor/surprise",
                                 lambda m: self.got.__setitem__("surprise", m.data), 10)
        self.create_subscription(Float32, "/world_model_monitor/anomaly_threshold",
                                 lambda m: self.got.__setitem__("threshold", m.data), 10)
        self.create_subscription(Bool, "/world_model_monitor/anomaly",
                                 lambda m: self.got.__setitem__("anomaly", m.data), 10)
        self._rng = np.random.default_rng(0)
        self.create_timer(0.2, self._tick)

    def _tick(self):
        img = (self._rng.random((32, 32, 3)) * 255).astype(np.uint8)
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(conv.np_to_image_msg(img, msg.header))


def main():
    rclpy.init()
    monitor = WorldModelMonitor(parameter_overrides=[Parameter("adapter", value="dummy")])
    probe = Probe()
    ex = SingleThreadedExecutor()
    ex.add_node(monitor)
    ex.add_node(probe)
    rc = 1
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 15.0:
        ex.spin_once(timeout_sec=0.1)
        if all(v is not None for v in probe.got.values()):
            print("monitor outputs:", {k: (round(v, 3) if isinstance(v, float) else v)
                                        for k, v in probe.got.items()})
            print("MONITOR_OK")
            rc = 0
            break
    else:
        print("MONITOR_TIMEOUT", probe.got)
    monitor.destroy_node()
    probe.destroy_node()
    rclpy.shutdown()
    return rc


if __name__ == "__main__":
    sys.exit(main())
