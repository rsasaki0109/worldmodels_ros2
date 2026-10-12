#!/usr/bin/env python3
"""Drive the wm_crossing robot around for data collection (no Nav2 needed).

Visits random waypoints inside the corridor at 0.4-0.8 m/s with a simple
heading controller on /odom, for --duration seconds of sim time. Obstacles are
ignored on purpose (the robot has no collision geometry): this only collects
diverse camera + LiDAR views with pedestrians crossing in front of the robot.

    python3 gz_data_drive.py --duration 300 --seed 0 --ros-args -p use_sim_time:=true
"""
import argparse
import math
import random
import sys

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node


class Drive(Node):
    def __init__(self, duration: float, seed: int):
        super().__init__("gz_data_drive")
        self.rng = random.Random(seed)
        self.duration = duration
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(Odometry, "/odom", self._on_odom, 10)
        self.pose = None
        self.goal = self._new_goal()
        self.speed = 0.6
        self.t0 = None
        self.done = False
        self.create_timer(0.05, self._tick)

    def _new_goal(self):
        return (self.rng.uniform(0.5, 21.0), self.rng.uniform(-2.8, 2.8))

    def _on_odom(self, msg: Odometry) -> None:
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.pose = (p.x, p.y, yaw)

    def _tick(self) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.pose is None or now == 0.0:
            return
        if self.t0 is None:
            self.t0 = now
        if now - self.t0 > self.duration:
            self.pub.publish(Twist())
            self.done = True
            return
        x, y, yaw = self.pose
        dx, dy = self.goal[0] - x, self.goal[1] - y
        if math.hypot(dx, dy) < 0.4:
            self.goal = self._new_goal()
            self.speed = self.rng.uniform(0.4, 0.8)
            return
        err = math.atan2(dy, dx) - yaw
        err = math.atan2(math.sin(err), math.cos(err))
        tw = Twist()
        tw.angular.z = max(-1.0, min(1.0, 1.5 * err))
        tw.linear.x = self.speed * max(0.0, math.cos(err)) ** 2
        self.pub.publish(tw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=0)
    args, ros_args = ap.parse_known_args()
    rclpy.init(args=ros_args)
    node = Drive(args.duration, args.seed)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
