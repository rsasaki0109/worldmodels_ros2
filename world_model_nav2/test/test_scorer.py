"""In-process service test: scorer ranks candidate paths by risk (ROS env)."""
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from world_model_msgs.srv import ScoreTrajectories
from world_model_nav2.trajectory_scorer_node import TrajectoryScorer
from world_model_nav2.score_demo import candidate_paths


def _spin_until(ex, pred, timeout=8.0):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        ex.spin_once(timeout_sec=0.05)
        if pred():
            return True
    return False


def test_scorer_ranks_paths(tmp_path=None):
    rclpy.init()
    scorer = TrajectoryScorer()                       # dummy adapter by default
    client = Node("test_client")
    cli = client.create_client(
        ScoreTrajectories, "/world_model_trajectory_scorer/score_trajectories"
    )
    ex = SingleThreadedExecutor()
    ex.add_node(scorer)
    ex.add_node(client)
    try:
        assert _spin_until(ex, cli.service_is_ready), "service never came up"

        req = ScoreTrajectories.Request()
        req.trajectories = candidate_paths()          # [straight, fast, swerve]
        future = cli.call_async(req)
        assert _spin_until(ex, future.done), "service call timed out"

        res = future.result()
        assert len(res.scores) == 3
        assert all(0.0 <= s <= 1.0 for s in res.scores)
        # dummy risk grows with action magnitude: "fast" (0.4 steps) > "straight" (0.1)
        assert res.scores[1] > res.scores[0]
        # safest is the gentle straight path
        assert res.best_index == 0
    finally:
        scorer.destroy_node()
        client.destroy_node()
        rclpy.shutdown()
