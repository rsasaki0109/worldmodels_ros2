"""In-process end-to-end check: runtime node (autostart) + a probe node that
publishes one Observation and waits for FutureState / RiskScore / FutureOccupancy.
Avoids cross-process discovery flakiness on this machine.
"""
import sys
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from world_model_msgs.msg import (
    Observation as ObservationMsg,
    FutureState as FutureStateMsg,
    RiskScore as RiskScoreMsg,
    FutureOccupancy as FutureOccupancyMsg,
)
from world_model_py.runtime_node import WorldModelRuntime
from world_model_py import conversions as conv


class Probe(Node):
    def __init__(self):
        super().__init__("probe")
        self.got_future = None
        self.got_risk = None
        self.got_occ = None
        self.pub = self.create_publisher(ObservationMsg, "/world_model_runtime/observation", 10)
        self.create_subscription(FutureStateMsg, "/world_model_runtime/future_state", self._f, 10)
        self.create_subscription(RiskScoreMsg, "/world_model_runtime/risk_score", self._r, 10)
        self.create_subscription(FutureOccupancyMsg, "/world_model_runtime/future_occupancy", self._o, 10)
        self.create_timer(0.2, self._tick)

    def _tick(self):
        img = (np.zeros((32, 32, 3))).astype(np.uint8)
        msg = ObservationMsg()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        msg.image = conv.np_to_image_msg(img, msg.header)
        msg.ego_state = [1.0, 0.0, 0.0, 0.5]
        msg.action_dim = 2
        msg.instruction = "go"
        self.pub.publish(msg)

    def _f(self, m): self.got_future = m
    def _r(self, m): self.got_risk = m
    def _o(self, m): self.got_occ = m

    def done(self):
        return self.got_future and self.got_risk and self.got_occ


def main():
    rclpy.init()
    runtime = WorldModelRuntime()
    probe = Probe()
    ex = SingleThreadedExecutor()
    ex.add_node(runtime)
    ex.add_node(probe)

    import time
    t0 = time.perf_counter()
    rc = 1
    while time.perf_counter() - t0 < 20.0:
        ex.spin_once(timeout_sec=0.1)
        if probe.done():
            f, r, o = probe.got_future, probe.got_risk, probe.got_occ
            print(f"future_state: dt={f.dt:.3f}, #latents={len(f.latents)}, "
                  f"latent0_len={len(f.latents[0].data) if f.latents else 0}, "
                  f"shape={list(f.latents[0].shape) if f.latents else []}")
            print(f"risk_score:   score={r.score:.3f}, confidence={r.confidence:.3f}, label='{r.label}'")
            print(f"future_occupancy: #grids={len(o.grids)}, "
                  f"grid0={o.grids[0].info.width}x{o.grids[0].info.height}" if o.grids else "no grids")
            print("INTEGRATION_OK")
            rc = 0
            break
    else:
        print("INTEGRATION_TIMEOUT: future=%s risk=%s occ=%s" % (
            bool(probe.got_future), bool(probe.got_risk), bool(probe.got_occ)))

    runtime.destroy_node()
    probe.destroy_node()
    rclpy.shutdown()
    return rc


if __name__ == "__main__":
    sys.exit(main())
