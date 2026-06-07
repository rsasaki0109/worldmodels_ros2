"""Unit + in-process integration tests for the bag -> Observation relay."""
import numpy as np
import pytest

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from world_model_py.bag_relay import build_observation, _ACTION_EXTRACTORS
from world_model_py import conversions as conv


def _image() -> Image:
    hdr = Header()
    hdr.frame_id = "camera"
    arr = np.zeros((16, 16, 3), dtype=np.uint8)
    return conv.np_to_image_msg(arr, hdr)


def test_build_observation_fields():
    msg = build_observation(
        _image(),
        ego_state=[1.0, 2.0, 0.1],
        action_history=[[0.5, 0.0, 0.1], [0.4, 0.0, 0.0]],
        action_dim=3,
        instruction="drive",
    )
    assert list(msg.ego_state) == pytest.approx([1.0, 2.0, 0.1])
    assert list(msg.action_history) == pytest.approx([0.5, 0.0, 0.1, 0.4, 0.0, 0.0])
    assert msg.action_dim == 3
    assert msg.instruction == "drive"
    assert msg.image.width == 16


def test_twist_extractor_dim():
    tw = Twist()
    tw.linear.x = 1.0
    tw.angular.z = 0.2
    vec = _ACTION_EXTRACTORS["geometry_msgs/msg/Twist"](tw)
    assert len(vec) == 6
    assert vec[0] == pytest.approx(1.0)
    assert vec[5] == pytest.approx(0.2)


rclpy = pytest.importorskip("rclpy")


def test_relay_to_runtime_integration():
    """Image + Twist -> bag_relay -> runtime produces future outputs."""
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node

    from world_model_msgs.msg import FutureState as FutureStateMsg, RiskScore as RiskScoreMsg
    from world_model_py.bag_relay import BagObservationRelay
    from world_model_py.runtime_node import WorldModelRuntime

    class Probe(Node):
        def __init__(self):
            super().__init__("probe")
            self.got_future = None
            self.got_risk = None
            self.pub_img = self.create_publisher(Image, "/camera/image_raw", 10)
            self.pub_act = self.create_publisher(Twist, "/cmd_vel", 10)
            self.create_subscription(
                FutureStateMsg, "/world_model_runtime/future_state", self._f, 10
            )
            self.create_subscription(
                RiskScoreMsg, "/world_model_runtime/risk_score", self._r, 10
            )
            self.create_timer(0.2, self._tick)

        def _tick(self):
            self.pub_img.publish(_image())
            tw = Twist()
            tw.linear.x = 0.3
            self.pub_act.publish(tw)

        def _f(self, m):
            self.got_future = m

        def _r(self, m):
            self.got_risk = m

        def done(self):
            return self.got_future is not None and self.got_risk is not None

    rclpy.init()
    try:
        relay = BagObservationRelay()
        runtime = WorldModelRuntime()
        probe = Probe()
        ex = SingleThreadedExecutor()
        ex.add_node(relay)
        ex.add_node(runtime)
        ex.add_node(probe)

        import time

        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 20.0:
            ex.spin_once(timeout_sec=0.1)
            if probe.done():
                assert len(probe.got_future.latents) > 0
                assert probe.got_risk.score >= 0.0
                return
        pytest.fail("timeout waiting for relay -> runtime pipeline")
    finally:
        rclpy.shutdown()
