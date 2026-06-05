"""Path -> action geometry (pure numpy, no ROS)."""
import numpy as np

from world_model_nav2.path_to_action import path_to_action, path_length, wrap_to_pi


def test_straight_path_forward_only():
    poses = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    act = path_to_action(poses)
    assert act.shape == (2, 3)
    assert np.allclose(act[:, 0], [1.0, 1.0])    # forward
    assert np.allclose(act[:, 1], 0.0)           # no lateral
    assert np.allclose(act[:, 2], 0.0)           # no rotation


def test_body_frame_rotation():
    # heading 90deg, then move +y (world) -> that is "forward" in body frame
    poses = np.array([[0, 0, np.pi / 2], [0, 1, np.pi / 2]], dtype=np.float32)
    act = path_to_action(poses)
    assert abs(act[0, 0] - 1.0) < 1e-5    # forward ~ 1
    assert abs(act[0, 1]) < 1e-5          # lateral ~ 0


def test_dyaw_wrapped():
    poses = np.array([[0, 0, 3.0], [0, 0, -3.0]], dtype=np.float32)
    act = path_to_action(poses)
    # 3.0 -> -3.0 is +0.283 rad the short way, not -6.0
    assert abs(act[0, 2] - wrap_to_pi(np.array([-6.0]))[0]) < 1e-5
    assert abs(act[0, 2]) < 0.4


def test_degenerate_paths_empty():
    assert path_to_action(np.zeros((0, 3))).shape == (0, 3)
    assert path_to_action(np.array([[1, 2, 0]], dtype=np.float32)).shape == (0, 3)


def test_path_length():
    poses = np.array([[0, 0, 0], [3, 4, 0]], dtype=np.float32)
    assert abs(path_length(poses) - 5.0) < 1e-5
    assert path_length(np.zeros((1, 3))) == 0.0
