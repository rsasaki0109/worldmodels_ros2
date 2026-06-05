"""Build a few candidate paths, ask the scorer to rank them, print the result.

    ros2 run world_model_nav2 score_demo

Run the scorer first:
    ros2 run world_model_nav2 trajectory_scorer_node
"""
from __future__ import annotations

import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

from world_model_msgs.srv import ScoreTrajectories


def _path(points, frame="map") -> Path:
    path = Path()
    path.header.frame_id = frame
    for x, y in points:
        ps = PoseStamped()
        ps.header.frame_id = frame
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.orientation.w = 1.0
        path.poses.append(ps)
    return path


def candidate_paths() -> list:
    straight = [(0.1 * i, 0.0) for i in range(11)]                       # gentle, short steps
    fast = [(0.4 * i, 0.0) for i in range(11)]                           # big steps -> riskier
    swerve = [(0.1 * i, 0.5 * math.sin(0.6 * i)) for i in range(11)]     # wiggly
    return [_path(straight), _path(fast), _path(swerve)]


class DemoClient(Node):
    def __init__(self):
        super().__init__("score_demo")
        self.cli = self.create_client(ScoreTrajectories, "/world_model_trajectory_scorer/score_trajectories")

    def run(self) -> int:
        if not self.cli.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("scorer service not available")
            return 1
        req = ScoreTrajectories.Request()
        req.trajectories = candidate_paths()
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        res = future.result()
        if res is None:
            self.get_logger().error("service call failed")
            return 1
        labels = ["straight", "fast", "swerve"]
        for label, score in zip(labels, res.scores):
            print(f"  {label:9s} risk={score:.3f}")
        print(f"safest: {labels[res.best_index]} (index {res.best_index})")
        return 0


def main(args=None) -> int:
    rclpy.init(args=args)
    node = DemoClient()
    try:
        return node.run()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
