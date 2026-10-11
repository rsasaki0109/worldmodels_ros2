#!/usr/bin/env python3
"""Walk the wm_crossing "pedestrians" back and forth across the corridor.

Each ped moves along world y at its own speed and turns around near the walls
(|y| > turn_y), using the ground-truth poses bridged on /ground_truth/poses.
The bridge drops model names, so poses are matched to peds by their fixed x
and height; the named ped poses are republished on /ground_truth/peds
(world frame) for the closed-loop evaluator.
"""
import json
import os
import sys

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from tf2_msgs.msg import TFMessage

PED_Z = 0.85   # ped link height in wm_crossing.sdf


class PedMover(Node):
    def __init__(self, config: str, turn_y: float = 3.0):
        super().__init__("gz_ped_mover")
        self.peds = json.load(open(config))["peds"]
        self.turn_y = float(turn_y)
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

    def _tick(self) -> None:
        for p in self.peds:
            tw = Twist()
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
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
