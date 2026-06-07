"""Unit tests for counterfactual mosaic helpers."""
import numpy as np

from world_model_viz.counterfactual import branch_label, build_mosaic


def test_branch_labels():
    assert branch_label(-0.7) == "LEFT"
    assert branch_label(0.0) == "STRAIGHT"
    assert branch_label(0.7) == "RIGHT"


def test_build_mosaic_shape():
    frames = [
        np.full((32, 32, 3), 10, np.uint8),
        np.full((32, 32, 3), 20, np.uint8),
        np.full((32, 32, 3), 30, np.uint8),
    ]
    out = build_mosaic(frames, ["LEFT", "STRAIGHT", "RIGHT"], tile=64, gap=4)
    assert out.ndim == 3
    assert out.shape[2] == 3
    assert out.shape[1] == 3 * 64 + 2 * 4
    assert out.shape[0] == 64 + 22
