"""Nearest-neighbour time sync (pure numpy, no ROS)."""
import numpy as np

from world_model_datasets.sync import nearest_indices


def test_exact_matches():
    master = np.array([0, 10, 20], dtype=np.int64)
    other = np.array([20, 0, 10], dtype=np.int64)  # unsorted on purpose
    idx = nearest_indices(master, other)
    assert other[idx].tolist() == [0, 10, 20]


def test_picks_nearest():
    master = np.array([5, 14], dtype=np.int64)
    other = np.array([0, 10, 20], dtype=np.int64)
    idx = nearest_indices(master, other)
    assert other[idx].tolist() == [0, 10]  # 5->0 (tie low), 14->10


def test_tolerance_rejects_far():
    master = np.array([0, 1000], dtype=np.int64)
    other = np.array([5, 7], dtype=np.int64)
    idx = nearest_indices(master, other, tol_ns=10)
    assert idx[0] >= 0          # 0 within 10 of 5
    assert idx[1] == -1         # 1000 far from any


def test_empty_other():
    idx = nearest_indices(np.array([1, 2, 3]), np.array([], dtype=np.int64))
    assert idx.tolist() == [-1, -1, -1]


def test_single_other():
    idx = nearest_indices(np.array([0, 100]), np.array([50]))
    assert idx.tolist() == [0, 0]
