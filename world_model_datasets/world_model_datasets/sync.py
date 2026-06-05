"""Time-synchronisation helpers (pure numpy; no ROS).

The image stream is the master clock. For every image timestamp we pick the
nearest sample from each other stream, optionally rejecting matches outside a
tolerance. This is the bit a localization/mapping engineer cares about: it is
explicit and testable, not hidden inside a learning library.
"""
from __future__ import annotations

import numpy as np


def nearest_indices(master_ns: np.ndarray, other_ns: np.ndarray, tol_ns: int | None = None) -> np.ndarray:
    """For each time in ``master_ns`` return the index of the nearest time in
    ``other_ns``. Entries with no match within ``tol_ns`` get -1.

    ``other_ns`` need not be sorted; it is sorted internally.
    """
    master = np.asarray(master_ns, dtype=np.int64)
    other = np.asarray(other_ns, dtype=np.int64)
    if other.size == 0:
        return np.full(master.shape, -1, dtype=np.int64)

    order = np.argsort(other)
    sorted_other = other[order]
    pos = np.searchsorted(sorted_other, master)
    pos = np.clip(pos, 1, len(sorted_other) - 1)
    left = sorted_other[pos - 1]
    right = sorted_other[pos]
    choose_left = (master - left) <= (right - master)
    idx_sorted = np.where(choose_left, pos - 1, pos)
    # handle the single-element / boundary cases cleanly
    idx_sorted = np.clip(idx_sorted, 0, len(sorted_other) - 1)
    result = order[idx_sorted]

    if tol_ns is not None:
        dist = np.abs(other[result] - master)
        result = np.where(dist <= tol_ns, result, -1)
    return result.astype(np.int64)
