#!/usr/bin/env python3
"""Walk the wm_crossing "pedestrians" back and forth across the corridor.

Each ped moves along world y at its own speed and turns around near the walls
(|y| > turn_y), using the ground-truth poses bridged on /ground_truth/poses.
The bridge drops model names, so poses are matched to peds by their fixed x
and height; the named ped poses are republished on /ground_truth/peds
(world frame) for the closed-loop evaluator.

``-p seed:=N`` (N > 0) holds each ped still for a random 0-``max_delay`` s of
sim time first, so episodes see different crossing timings (the same seed
gives the same timings, for a fair with/without comparison).

``-p start_topic:=/ground_truth/ped_start``: the peds stand still until a
``std_msgs/Empty`` arrives there, and the delays count from it. The evaluator
sends it with the navigation goal, so every condition meets exactly the same
crossings; otherwise they would depend on how fast the processes started.
"""
import json
import os
import random
import sys

import rclpy
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
from rclpy.node import Node
from tf2_msgs.msg import TFMessage

PED_Z = 0.85   # ped link height in wm_crossing.sdf


class PedMover(Node):
    def __init__(self, config: str, turn_y: float = 3.0):
        super().__init__("gz_ped_mover")
        self.peds = json.load(open(config))["peds"]
        self.turn_y = float(turn_y)
        self.declare_parameter("seed", 0)
        self.declare_parameter("max_delay", 8.0)
        seed = self.get_parameter("seed").get_parameter_value().integer_value
        max_delay = self.get_parameter("max_delay").get_parameter_value().double_value
        rng = random.Random(seed)
        self.delay = {p["name"]: (rng.uniform(0.0, max_delay) if seed > 0 else 0.0) for p in self.peds}
        self.t0 = None
        self.declare_parameter("start_topic", "")
        start_topic = self.get_parameter("start_topic").get_parameter_value().string_value
        self.started = not start_topic
        if start_topic:
            self.create_subscription(Empty, start_topic, self._on_start, 10)
        self.dir = {p["name"]: (1.0 if p["y0"] < 0 else -1.0) for p in self.peds}
        self.pubs = {p["name"]: self.create_publisher(Twist, f"/model/{p['name']}/cmd_vel", 10)
                     for p in self.peds}
        self.gt_pub = self.create_publisher(TFMessage, "/ground_truth/peds", 10)
        self.create_subscription(TFMessage, "/ground_truth/poses", self._on_poses, 10)
        self.create_timer(0.1, self._tick)

    def _match(self, t) -> str:
        """Ped whose (fixed) x and height match this pose, or ""."""
        tr = t.transform.translation
        if abs(tr.z - PED_Z) > 0.05:
            return ""
        best = min(self.peds, key=lambda p: abs(p["x"] - tr.x))
        return best["name"] if abs(best["x"] - tr.x) < 0.3 else ""

    def _on_poses(self, msg: TFMessage) -> None:
        out = TFMessage()
        for t in msg.transforms:
            name = t.child_frame_id.split("::")[0] or self._match(t)
            if name not in self.dir:
                continue
            y = t.transform.translation.y
            if y > self.turn_y:
                self.dir[name] = -1.0
            elif y < -self.turn_y:
                self.dir[name] = 1.0
            t.header.frame_id = "world"
            t.header.stamp = self.get_clock().now().to_msg()
            t.child_frame_id = name
            out.transforms.append(t)
        if out.transforms:
            self.gt_pub.publish(out)

    def _on_start(self, _msg) -> None:
        if not self.started:
            self.started = True
            self.t0 = None              # delays count from now
            self.get_logger().info("start received: pedestrians walking")

    def _tick(self) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        if now <= 0.0 or not self.started:
            for pub in self.pubs.values():
                pub.publish(Twist())
            return
        if self.t0 is None:
            self.t0 = now
        for p in self.peds:
            tw = Twist()
            if now - self.t0 >= self.delay[p["name"]]:
                tw.linear.y = self.dir[p["name"]] * float(p["speed"])
            self.pubs[p["name"]].publish(tw)


def main() -> None:
    rclpy.init(args=sys.argv)
    here = os.path.dirname(os.path.abspath(__file__))
    default = os.path.join(here, "..", "gazebo", "wm_crossing_peds.json")
    node = PedMover(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else default)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
            rclpy.try_shutdown()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
