"""Tests for experience memory build/save and recorder -> planning pipeline."""
import os
import tempfile

import numpy as np
import pytest

from world_model_py.experience import align_frames_to_next, build_transitions, save_experience


def test_build_transitions_shapes():
    lat = np.arange(12, dtype=np.float32).reshape(4, 3)
    act = np.ones((4, 1), np.float32) * np.arange(4, dtype=np.float32)[:, None]
    L, S, Nx = build_transitions(lat, act)
    assert L.shape == (3, 3)
    assert S.shape == (3, 1)
    assert Nx.shape == (3, 3)
    np.testing.assert_array_equal(L[0], lat[0])
    np.testing.assert_array_equal(Nx[0], lat[1])


def test_save_and_align_frames():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "exp.npz")
        lat = np.random.randn(6, 4).astype(np.float32)
        act = np.random.randn(6, 1).astype(np.float32)
        frames = np.random.randint(0, 255, (6, 8, 8, 3), dtype=np.uint8)
        summary = save_experience(path, lat, act, frames=frames)
        assert summary["transitions"] == 5
        d = np.load(path)
        assert d["frames"].shape == (5, 8, 8, 3)
        np.testing.assert_array_equal(d["frames"], align_frames_to_next(frames))


rclpy = pytest.importorskip("rclpy")


def test_recorder_flush_and_planning_imagines():
    """Record transitions in-process, save experience.npz, then call ImagineFutures."""
    from geometry_msgs.msg import Twist
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.parameter import Parameter
    from std_msgs.msg import Header

    from world_model_msgs.srv import ImagineFutures
    from world_model_py import conversions as conv
    from world_model_py.experience_recorder import ExperienceRecorder
    from world_model_py.planning_node import PlanningNode

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "experience.npz")
        rclpy.init()
        try:
            recorder = ExperienceRecorder(parameter_overrides=[
                Parameter("output_path", Parameter.Type.STRING, out),
                Parameter("min_transitions", Parameter.Type.INTEGER, 3),
            ])
            planner = PlanningNode(parameter_overrides=[
                Parameter("memory_path", Parameter.Type.STRING, out),
            ])
            for i in range(10):
                hdr = Header()
                arr = np.full((16, 16, 3), i * 20, dtype=np.uint8)
                tw = Twist()
                tw.angular.z = -0.5 + 0.1 * i
                recorder._on_action(tw)
                recorder._on_image(conv.np_to_image_msg(arr, hdr))
            recorder._flush("test")

            assert os.path.isfile(out)
            client = rclpy.create_node("imagine_client")
            cli = client.create_client(ImagineFutures, "/world_model_planning/imagine_futures")
            ex = SingleThreadedExecutor()
            ex.add_node(planner)
            ex.add_node(client)
            assert cli.wait_for_service(timeout_sec=5.0)

            req = ImagineFutures.Request()
            req.current_image = conv.np_to_image_msg(
                np.full((16, 16, 3), 100, dtype=np.uint8), Header()
            )
            req.steering_options = [-0.7, 0.0, 0.7]
            req.horizon = 6
            fut = cli.call_async(req)
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < 5.0 and not fut.done():
                ex.spin_once(timeout_sec=0.05)
            resp = fut.result()
            assert resp.success
            assert len(resp.branches) == 3
        finally:
            rclpy.shutdown()
