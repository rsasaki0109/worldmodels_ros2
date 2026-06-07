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

import os

import numpy as np
import rclpy
from rclpy.node import Node

from std_msgs.msg import Header

from world_model_msgs.srv import ImagineFutures, PlanToGoal
from world_model_msgs.msg import (
    ActionCondition as ActionConditionMsg,
    FutureState as FutureStateMsg,
    LatentState as LatentStateMsg,
)

from world_model_py.adapters import Observation
from world_model_py.conversions import image_msg_to_np, np_to_image_msg
from world_model_py.planning import (
    RetrievalDynamics,
    cosine_distance,
    decode_trajectory,
    imagine_counterfactuals,
    plan_to_goal,
)
from world_model_py.registry import load_model


class PlanningNode(Node):
    """ROS 2 service node that plans to an image goal in a World Model's latent
    space using a learning-free retrieval dynamics."""

    def __init__(self, *, parameter_overrides=None, **adapter_kwargs):
        super().__init__("world_model_planning", parameter_overrides=parameter_overrides or [])
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
        self._frames = None                       # optional decoded-frame memory
        self._memory_mtime = 0.0
        self._dyn = self._load_memory(self.get_parameter("memory_path").value)
        self._srv = self.create_service(PlanToGoal, "~/plan_to_goal", self._on_plan)
        self._imagine_srv = self.create_service(
            ImagineFutures, "~/imagine_futures", self._on_imagine)
        self.get_logger().info(
            f"planning node up (adapter={name}, "
            f"memory={'loaded' if self._dyn is not None else 'NONE — set memory_path'})")

    def _load_memory(self, path: str):
        if not path or not os.path.isfile(path):
            return None
        d = np.load(path)
        if "frames" in getattr(d, "files", []):
            self._frames = d["frames"]            # (N, H, W, 3), aligned to next_latents
        else:
            self._frames = None
        self._memory_mtime = os.path.getmtime(path)
        return RetrievalDynamics(
            d["latents"], d["actions"], d["next_latents"],
            k=int(self.get_parameter("k").value),
            action_weight=float(self.get_parameter("action_weight").value))

    def _maybe_reload_memory(self) -> None:
        path = self.get_parameter("memory_path").value
        if not path or not os.path.isfile(path):
            return
        mtime = os.path.getmtime(path)
        if mtime <= self._memory_mtime and self._dyn is not None:
            return
        self.get_logger().info(f"loading experience memory from {path}")
        self._dyn = self._load_memory(path)

    def attach_memory(self, dynamics: RetrievalDynamics, frames=None) -> None:
        """Inject a dynamics directly (used by tests / in-process callers)."""
        self._dyn = dynamics
        self._frames = frames

    def _encode(self, img_msg) -> np.ndarray:
        arr = image_msg_to_np(img_msg)
        lat = self._adapter.encode(Observation(image=arr)).data
        return np.asarray(lat, np.float32).ravel()

    def _on_plan(self, req, resp):
        self._maybe_reload_memory()
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

    def _on_imagine(self, req, resp):
        self._maybe_reload_memory()
        if self._dyn is None:
            self.get_logger().warn("no experience memory; set memory_path")
            resp.success = False
            return resp
        start = self._encode(req.current_image)
        horizon = int(req.horizon) or int(self.get_parameter("horizon").value)
        adim = int(self._dyn.actions.shape[1])
        opts = [np.full(adim, float(s), np.float32) for s in req.steering_options]
        branches = imagine_counterfactuals(self._dyn, start, opts, horizon)

        header = req.current_image.header
        endpoints = []
        for latents, _idx in branches:
            fs = FutureStateMsg()
            fs.header = header
            fs.dt = 0.0
            for lat in latents:
                ls = LatentStateMsg()
                ls.data = [float(x) for x in np.asarray(lat, np.float32).ravel()]
                ls.shape = [len(ls.data)]
                ls.encoding = self._adapter.name
                fs.latents.append(ls)
            if self._frames is not None:
                for fr in decode_trajectory(latents, self._dyn.next_latents, self._frames):
                    fs.frames.append(np_to_image_msg(np.asarray(fr, np.uint8), header))
            resp.branches.append(fs)
            endpoints.append(np.asarray(latents[-1], np.float32))
        resp.divergence = [cosine_distance(e, endpoints[0]) for e in endpoints]
        resp.success = True
        self.get_logger().info(
            f"imagined {len(endpoints)} branches, "
            f"max divergence {max(resp.divergence):.3f}")
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
