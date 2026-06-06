"""Prediction-quality metrics. No ROS, no GPU."""
import numpy as np

from world_model_py.metrics import (
    occupancy_iou,
    future_occupancy_iou,
    temporal_consistency,
    latent_drift,
)


def _grid(cells, size=4):
    g = np.zeros((size, size), np.int8)
    for (r, c) in cells:
        g[r, c] = 100
    return g


def test_iou_identical():
    g = _grid([(0, 0), (1, 1)])
    assert occupancy_iou(g, g) == 1.0


def test_iou_disjoint():
    assert occupancy_iou(_grid([(0, 0)]), _grid([(3, 3)])) == 0.0


def test_iou_partial():
    a = _grid([(0, 0), (1, 1)])
    b = _grid([(1, 1), (2, 2)])
    assert abs(occupancy_iou(a, b) - 1 / 3) < 1e-9   # inter 1, union 3


def test_iou_both_empty_is_one():
    z = np.zeros((4, 4), np.int8)
    assert occupancy_iou(z, z) == 1.0


def test_iou_shape_mismatch_raises():
    try:
        occupancy_iou(np.zeros((4, 4), np.int8), np.zeros((3, 3), np.int8))
        assert False
    except ValueError:
        pass


def test_future_iou_aligns_shorter():
    g = _grid([(0, 0)])
    res = future_occupancy_iou([g, g, g], [g, g], threshold=50)
    assert res["steps"] == 2
    assert res["mean"] == 1.0


def test_temporal_consistency():
    g = _grid([(0, 0)])
    assert temporal_consistency([g, g, g]) == 1.0
    assert temporal_consistency([g]) == 1.0
    a, b = _grid([(0, 0)]), _grid([(3, 3)])
    assert temporal_consistency([a, b]) == 0.0


def test_latent_drift():
    same = [np.array([1.0, 0.0]), np.array([1.0, 0.0])]
    assert latent_drift(same)["mean"] < 1e-9
    orth = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    assert abs(latent_drift(orth)["mean"] - 1.0) < 1e-9
    assert latent_drift([np.array([1.0])])["mean"] == 0.0
