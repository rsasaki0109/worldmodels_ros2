"""In-process check of the PlanToGoal service node (ROS 2, dummy adapter).

Cross-process DDS discovery is slow on this machine, so this drives the service
inside one process with a SingleThreadedExecutor (same pattern as the other
integration_* tests). It verifies the service wiring + message contract; the
planning quality itself is covered GPU-free by test_planning.py.

    colcon test --packages-select world_model_py
"""
import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")

from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import Header

from world_model_msgs.srv import ImagineFutures, PlanToGoal

from world_model_py.adapters import Observation
from world_model_py.conversions import np_to_image_msg
from world_model_py.planning import RetrievalDynamics
from world_model_py.planning_node import PlanningNode
from world_model_py.registry import load_model


def _synthetic_memory(action_dim=3, n=48, seed=0):
    """A dummy-adapter experience memory: real (latent, action, next_latent)
    rows so the node has something to plan through."""
    rng = np.random.default_rng(seed)
    ad = load_model("dummy")
    imgs = [rng.integers(0, 255, (16, 16, 3), dtype=np.uint8) for _ in range(n)]
    lats = np.array([ad.encode(Observation(image=im)).data for im in imgs], np.float32)
    acts = rng.uniform(-1, 1, (n, action_dim)).astype(np.float32)
    nxt = np.roll(lats, -1, axis=0)
    return RetrievalDynamics(lats[:-1], acts[:-1], nxt[:-1], k=6), imgs


def test_plan_to_goal_service():
    rclpy.init()
    try:
        dyn, imgs = _synthetic_memory(action_dim=3)
        node = PlanningNode()
        node.attach_memory(dyn)
        client = rclpy.create_node("plan_test_client")
        cli = client.create_client(PlanToGoal, "/world_model_planning/plan_to_goal")

        ex = SingleThreadedExecutor()
        ex.add_node(node); ex.add_node(client)
        assert cli.wait_for_service(timeout_sec=5.0)

        req = PlanToGoal.Request()
        req.current_image = np_to_image_msg(imgs[0], Header())
        req.goal_image = np_to_image_msg(imgs[-1], Header())
        req.horizon = 8
        fut = cli.call_async(req)
        ex.spin_until_future_complete(fut, timeout_sec=15.0)
        resp = fut.result()

        assert resp is not None
        assert resp.success
        assert resp.planned_action.horizon == 8
        assert resp.planned_action.action_dim == 3
        assert len(resp.planned_action.action) == 8 * 3
        assert np.isfinite(resp.final_cost)
        node.destroy_node(); client.destroy_node()
    finally:
        rclpy.try_shutdown()


def test_imagine_futures_service():
    rclpy.init()
    try:
        dyn, imgs = _synthetic_memory(action_dim=1)      # steering-only memory
        node = PlanningNode()
        node.attach_memory(dyn)
        client = rclpy.create_node("imagine_test_client")
        cli = client.create_client(ImagineFutures, "/world_model_planning/imagine_futures")
        ex = SingleThreadedExecutor(); ex.add_node(node); ex.add_node(client)
        assert cli.wait_for_service(timeout_sec=5.0)

        req = ImagineFutures.Request()
        req.current_image = np_to_image_msg(imgs[0], Header())
        req.steering_options = [-0.7, 0.0, 0.7]
        req.horizon = 10
        fut = cli.call_async(req)
        ex.spin_until_future_complete(fut, timeout_sec=15.0)
        resp = fut.result()

        assert resp is not None and resp.success
        assert len(resp.branches) == 3                   # one future per option
        assert all(len(b.latents) == 10 for b in resp.branches)
        assert len(resp.divergence) == 3
        assert resp.divergence[0] == 0.0                 # branch 0 vs itself
        node.destroy_node(); client.destroy_node()
    finally:
        rclpy.try_shutdown()


def test_plan_without_memory_fails_gracefully():
    rclpy.init()
    try:
        node = PlanningNode()                 # no memory attached
        client = rclpy.create_node("plan_test_client2")
        cli = client.create_client(PlanToGoal, "/world_model_planning/plan_to_goal")
        ex = SingleThreadedExecutor(); ex.add_node(node); ex.add_node(client)
        assert cli.wait_for_service(timeout_sec=5.0)
        req = PlanToGoal.Request()
        req.current_image = np_to_image_msg(np.zeros((16, 16, 3), np.uint8), Header())
        req.goal_image = np_to_image_msg(np.ones((16, 16, 3), np.uint8), Header())
        fut = cli.call_async(req)
        ex.spin_until_future_complete(fut, timeout_sec=10.0)
        resp = fut.result()
        assert resp is not None and not resp.success
        node.destroy_node(); client.destroy_node()
    finally:
        rclpy.try_shutdown()
