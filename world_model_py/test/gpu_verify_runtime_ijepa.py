"""Run the REAL I-JEPA model inside the ROS 2 lifecycle runtime (NOT in CI).

Proves the whole pipeline end to end through the ROS 2 contract:
    Observation msg -> conversions -> ijepa adapter (GPU) -> RiskScore msg
A scene-change image sequence is published on the runtime's observation topic;
the surprise (cosine latent distance) is read back off /risk_score and should
spike on scene cuts.

    python3 -m pytest ... NO. Run directly with ROS sourced:
    PYTHONPATH=world_model_py python3 world_model_py/test/gpu_verify_runtime_ijepa.py
"""
import sys
import time

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter

from world_model_msgs.msg import Observation as ObservationMsg, RiskScore
from world_model_py.runtime_node import WorldModelRuntime
from world_model_py import conversions as conv


def _gradient():
    yy, xx = np.mgrid[0:256, 0:256]
    return np.stack([xx / 256 * 180 + 40, yy / 256 * 120 + 30, np.full((256, 256), 90)], 2).astype(np.uint8)


def _box(base, cx, s, color):
    img = base.copy(); img[128 - s:128 + s, max(0, cx - s):cx + s] = color; return img


def _circle(base, cx, color):
    img = base.copy(); yy, xx = np.mgrid[0:256, 0:256]
    img[(xx - cx) ** 2 + (yy - 130) ** 2 <= 900] = color; return img


def make_sequence():
    rng = np.random.default_rng(3)
    frames = []
    g = _gradient()
    for k in range(8):                       # smooth: red box gliding
        frames.append(_box(g, 40 + k * 26, 34, [220, 40, 40]))
    noise = (rng.random((256, 256, 3)) * 255).astype(np.uint8)
    for k in range(5):                       # CUT -> noise scene
        frames.append(_circle(noise, 80 + k * 22, [40, 220, 90]))
    for k in range(5):                       # CUT back -> gradient + blue box
        frames.append(_box(_gradient(), 200 - k * 26, 34, [40, 90, 230]))
    return frames


class Probe(Node):
    def __init__(self, frames):
        super().__init__("probe")
        self.frames = frames
        self.sent = 0
        self.scores = []
        self.pub = self.create_publisher(ObservationMsg, "/world_model_runtime/observation", 10)
        self.create_subscription(RiskScore, "/world_model_runtime/risk_score", self._on_risk, 10)
        self.create_timer(1.0, self._kick)   # publish frame 0 until the model is up

    def _publish(self, i):
        f = self.frames[i]
        m = ObservationMsg()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "camera"
        m.image = conv.np_to_image_msg(f, m.header)
        self.pub.publish(m)

    def _kick(self):
        if self.sent == 0:
            self._publish(0)                  # keep poking until runtime is active

    def _on_risk(self, msg: RiskScore):
        idx = len(self.scores)
        if idx >= len(self.frames):
            return
        self.scores.append(round(float(msg.score), 4))
        self.sent = idx + 1
        if self.sent < len(self.frames):
            self._publish(self.sent)


def main():
    frames = make_sequence()
    rclpy.init()
    runtime = WorldModelRuntime(parameter_overrides=[
        Parameter("adapter", value="ijepa"),
        Parameter("model_id", value="facebook/ijepa_vith14_1k"),
        Parameter("horizon", value=1),
    ])
    probe = Probe(frames)
    ex = SingleThreadedExecutor()
    ex.add_node(runtime)
    ex.add_node(probe)

    print("loading I-JEPA in the lifecycle node (this blocks ~1-2 min on first activate)...")
    deadline = time.perf_counter() + 240.0
    while time.perf_counter() < deadline and len(probe.scores) < len(frames):
        ex.spin_once(timeout_sec=0.2)

    print(f"surprise over /world_model_runtime/risk_score ({len(probe.scores)}/{len(frames)}):")
    print("  ", probe.scores)
    spikes = [i for i, s in enumerate(probe.scores) if s > 0.15]
    print("  anomaly frames (>0.15):", spikes)
    ok = len(probe.scores) == len(frames) and 8 in spikes
    print("RUNTIME_IJEPA_OK" if ok else "RUNTIME_IJEPA_INCOMPLETE")

    runtime.destroy_node(); probe.destroy_node(); rclpy.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
