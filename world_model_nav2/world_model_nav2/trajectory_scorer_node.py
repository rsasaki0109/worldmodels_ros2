"""Score candidate trajectories with a World Model and pick the safest.

A *mock* of the Nav2 controller/critic step: instead of compiling a C++
nav2_core plugin, this exposes a ROS 2 service that any planner can call to
rank candidate paths by model-based risk. The contract (ScoreTrajectories.srv)
is what a real Nav2 critic would eventually wrap.

    ros2 run world_model_nav2 trajectory_scorer_node --ros-args -p adapter:=dummy

Service:  ~/score_trajectories  (world_model_msgs/srv/ScoreTrajectories)
Markers:  ~/scored_paths        (visualization_msgs/MarkerArray, green=safe..red=risky)
"""
from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node

from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

from world_model_msgs.srv import ScoreTrajectories

from world_model_py.adapters import ActionCondition, Observation
from world_model_py.registry import load_model

from .path_to_action import path_to_action


def _yaw_from_quat(q) -> float:
    return float(np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                            1.0 - 2.0 * (q.y * q.y + q.z * q.z)))


def _path_to_xyyaw(path) -> np.ndarray:
    rows = []
    for ps in path.poses:
        p = ps.pose.position
        rows.append([p.x, p.y, _yaw_from_quat(ps.pose.orientation)])
    return np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 3), np.float32)


class TrajectoryScorer(Node):
    def __init__(self):
        super().__init__("world_model_trajectory_scorer")
        self.declare_parameter("adapter", "dummy")
        self.declare_parameter("frame_id", "map")
        name = self.get_parameter("adapter").get_parameter_value().string_value
        self._frame = self.get_parameter("frame_id").get_parameter_value().string_value
        self._adapter = load_model(name)

        self._markers = self.create_publisher(MarkerArray, "~/scored_paths", 1)
        self._srv = self.create_service(
            ScoreTrajectories, "~/score_trajectories", self._on_score
        )
        self.get_logger().info(f"trajectory scorer ready (adapter '{name}')")

    def _on_score(self, request, response):
        obs = self._observation(request.observation)
        scores = []
        xyyaws = []
        for path in request.trajectories:
            xyyaw = _path_to_xyyaw(path)
            xyyaws.append(xyyaw)
            action = path_to_action(xyyaw)
            if action.shape[0] == 0:
                scores.append(1.0)   # empty/degenerate path -> max risk
                continue
            risk = self._adapter.score_trajectory(obs, ActionCondition(action=action, dt=0.1))
            scores.append(float(np.clip(risk, 0.0, 1.0)))

        response.scores = scores
        response.best_index = int(np.argmin(scores)) if scores else -1
        self._publish_markers(xyyaws, scores, response.best_index)
        self.get_logger().info(
            f"scored {len(scores)} trajectories; best={response.best_index} "
            f"scores={[round(s, 3) for s in scores]}"
        )
        return response

    def _observation(self, obs_msg) -> Observation:
        ego = np.asarray(obs_msg.ego_state, dtype=np.float32) if len(obs_msg.ego_state) else None
        return Observation(ego_state=ego, instruction=obs_msg.instruction)

    def _publish_markers(self, xyyaws, scores, best) -> None:
        array = MarkerArray()
        clear = Marker()
        clear.header.frame_id = self._frame
        clear.action = Marker.DELETEALL
        array.markers.append(clear)

        for i, (xyyaw, score) in enumerate(zip(xyyaws, scores)):
            if xyyaw.shape[0] < 2:
                continue
            m = Marker()
            m.header.frame_id = self._frame
            m.ns = "scored_paths"
            m.id = i
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.08 if i != best else 0.14   # highlight the winner
            m.pose.orientation.w = 1.0
            m.color = ColorRGBA(r=float(score), g=float(1.0 - score), b=0.0, a=1.0)
            for x, y, _ in xyyaw:
                m.points.append(Point(x=float(x), y=float(y), z=0.0))
            array.markers.append(m)
        self._markers.publish(array)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrajectoryScorer()
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
