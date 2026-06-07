"""Plan to an image goal with a learning-free episodic World Model (ROS 2).

Wraps ``world_model_py.planning`` behind a ROS 2 service: given a goal image and
the current image, it imagines candidate action rollouts in the latent space of
a World Model adapter -- using a retrieval dynamics built from a memory of real
transitions -- and returns the action sequence that best reaches the goal. This
is the "World Model版 Nav2" step: plan in a learned world, no dynamics training.

    ros2 run world_model_py planning_node --ros-args \\
        -p adapter:=ijepa -p memory_path:=/path/to/experience.npz

Service: ``~/plan_to_goal``  (world_model_msgs/srv/PlanToGoal)

The experience memory (an ``.npz`` with arrays ``latents``, ``actions``,
``next_latents``) is the planner's map: real ``(latent, action, next_latent)``
transitions to roll through. Build one from any rosbag / dataset (the
``validate_real_*`` scripts show how to encode frames + actions into one).
"""
from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node

from world_model_msgs.srv import PlanToGoal
from world_model_msgs.msg import ActionCondition as ActionConditionMsg

from world_model_py.adapters import Observation
from world_model_py.conversions import image_msg_to_np
from world_model_py.planning import RetrievalDynamics, cosine_distance, plan_to_goal
from world_model_py.registry import load_model


class PlanningNode(Node):
    """ROS 2 service node that plans to an image goal in a World Model's latent
    space using a learning-free retrieval dynamics."""

    def __init__(self, **adapter_kwargs):
        super().__init__("world_model_planning")
        self.declare_parameter("adapter", "dummy")
        self.declare_parameter("memory_path", "")
        self.declare_parameter("horizon", 12)
        self.declare_parameter("samples", 256)
        self.declare_parameter("iterations", 4)
        self.declare_parameter("action_low", -1.0)
        self.declare_parameter("action_high", 1.0)
        self.declare_parameter("action_weight", 1.0)
        self.declare_parameter("k", 8)

        name = self.get_parameter("adapter").value
        self._adapter = load_model(name, **adapter_kwargs)
        self._dyn = self._load_memory(self.get_parameter("memory_path").value)
        self._srv = self.create_service(PlanToGoal, "~/plan_to_goal", self._on_plan)
        self.get_logger().info(
            f"planning node up (adapter={name}, "
            f"memory={'loaded' if self._dyn is not None else 'NONE — set memory_path'})")

    def _load_memory(self, path: str):
        if not path:
            return None
        d = np.load(path)
        return RetrievalDynamics(
            d["latents"], d["actions"], d["next_latents"],
            k=int(self.get_parameter("k").value),
            action_weight=float(self.get_parameter("action_weight").value))

    def attach_memory(self, dynamics: RetrievalDynamics) -> None:
        """Inject a dynamics directly (used by tests / in-process callers)."""
        self._dyn = dynamics

    def _encode(self, img_msg) -> np.ndarray:
        arr = image_msg_to_np(img_msg)
        lat = self._adapter.encode(Observation(image=arr)).data
        return np.asarray(lat, np.float32).ravel()

    def _on_plan(self, req, resp):
        if self._dyn is None:
            self.get_logger().warn("no experience memory; set memory_path")
            resp.success = False
            return resp
        start = self._encode(req.current_image)
        goal = self._encode(req.goal_image)
        horizon = int(req.horizon) or int(self.get_parameter("horizon").value)
        adim = int(self._dyn.actions.shape[1])
        res = plan_to_goal(
            self._dyn, start, goal, action_dim=adim, horizon=horizon,
            samples=int(self.get_parameter("samples").value),
            iterations=int(self.get_parameter("iterations").value),
            action_low=float(self.get_parameter("action_low").value),
            action_high=float(self.get_parameter("action_high").value),
        )
        ac = ActionConditionMsg()
        ac.header = req.goal_image.header
        ac.action = [float(x) for x in np.asarray(res.actions, np.float32).ravel()]
        ac.action_dim = adim
        ac.horizon = int(res.horizon)
        ac.dt = 0.0
        resp.planned_action = ac
        resp.final_cost = float(res.cost)
        resp.start_cost = float(cosine_distance(start, goal))
        resp.success = True
        self.get_logger().info(
            f"planned: start_cost {resp.start_cost:.3f} -> final_cost {resp.final_cost:.3f} "
            f"({resp.start_cost / (resp.final_cost + 1e-9):.1f}x closer)")
        return resp


def main(args=None):
    rclpy.init(args=args)
    node = PlanningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
