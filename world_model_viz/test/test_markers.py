"""Unit tests for the colour ramp (runs under colcon test in a ROS env)."""
from world_model_viz.occupancy_marker_node import _ramp


def test_ramp_near_is_green():
    assert _ramp(0.0) == (0.0, 1.0, 0.0)


def test_ramp_far_is_red():
    assert _ramp(1.0) == (1.0, 0.0, 0.0)


def test_ramp_mid_blends():
    r, g, b = _ramp(0.5)
    assert abs(r - 0.5) < 1e-6
    assert abs(g - 0.5) < 1e-6
    assert b == 0.0


def test_ramp_clamps():
    assert _ramp(-1.0) == (0.0, 1.0, 0.0)
    assert _ramp(2.0) == (1.0, 0.0, 0.0)
