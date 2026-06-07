"""Score paths against a predicted occupancy horizon (pure numpy).

Mirrors what ``world_model_costmap::WorldModelLayer`` does: union lethal cells
over the whole horizon, then measure how much a candidate path intersects them.
"""
from __future__ import annotations

import math

import numpy as np


def merge_lethal_cells(grids, threshold: int = 50) -> list[tuple[float, float]]:
    """Union occupied cells across a FutureOccupancy horizon -> world (x, y) centres."""
    cells: set[tuple[float, float]] = set()
    for grid in grids:
        res = float(grid.info.resolution) or 0.1
        ox = float(grid.info.origin.position.x)
        oy = float(grid.info.origin.position.y)
        w, h = int(grid.info.width), int(grid.info.height)
        if w == 0 or h == 0:
            continue
        data = np.asarray(grid.data, dtype=np.int16).reshape(h, w)
        ys, xs = np.where(data >= threshold)
        for j, i in zip(xs.tolist(), ys.tolist()):
            cells.add((ox + (j + 0.5) * res, oy + (i + 0.5) * res))
    return sorted(cells)


def sample_line(x0: float, y0: float, x1: float, y1: float, n: int = 24) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    x = x0 + (x1 - x0) * t
    y = y0 + (y1 - y0) * t
    return np.stack([x, y], axis=1)


def sample_arc(cx: float, cy: float, r: float, a0: float, a1: float, n: int = 24) -> np.ndarray:
    t = np.linspace(a0, a1, n, dtype=np.float32)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], axis=1)


def path_collision_risk(
    path_xy: np.ndarray,
    lethal: list[tuple[float, float]],
    radius: float = 0.18,
) -> float:
    """Fraction of path samples within ``radius`` of a predicted lethal cell."""
    if path_xy.shape[0] == 0 or not lethal:
        return 0.0
    hits = 0
    r2 = radius * radius
    for x, y in path_xy:
        for lx, ly in lethal:
            if (x - lx) ** 2 + (y - ly) ** 2 <= r2:
                hits += 1
                break
    return hits / path_xy.shape[0]


def default_candidates() -> list[tuple[str, np.ndarray]]:
    """Straight vs two detours through a 3.2 m map with a mid-field obstacle band."""
    straight = sample_line(0.2, 1.6, 3.0, 1.6)
    upper = sample_arc(1.6, 0.2, 1.45, math.pi * 0.15, math.pi * 0.85)
    lower = sample_arc(1.6, 3.0, 1.45, -math.pi * 0.85, -math.pi * 0.15)
    return [("straight", straight), ("upper", upper), ("lower", lower)]
