"""Tests for predicted-occupancy path scoring."""
import numpy as np

from nav_msgs.msg import OccupancyGrid

from world_model_msgs.msg import FutureOccupancy
from world_model_nav2.occupancy_path import (
    default_candidates,
    merge_lethal_cells,
    path_collision_risk,
    sample_line,
)


def _grid(cx: float, cy: float, val: int = 90) -> OccupancyGrid:
    g = OccupancyGrid()
    g.info.resolution = 0.1
    g.info.width = g.info.height = 32
    g.info.origin.position.x = 0.0
    g.info.origin.position.y = 0.0
    g.info.origin.orientation.w = 1.0
    data = np.zeros((32, 32), np.int8)
    gi = int(cx / 0.1)
    gj = int(cy / 0.1)
    data[gj, gi] = val
    g.data = data.ravel().tolist()
    return g


def test_merge_and_score():
    msg = FutureOccupancy()
    msg.grids = [_grid(1.6, 1.6)]
    lethal = merge_lethal_cells(msg.grids)
    assert len(lethal) == 1
    through = sample_line(0.2, 1.6, 3.0, 1.6)
    around = sample_line(0.2, 2.4, 3.0, 2.4)
    assert path_collision_risk(through, lethal) > path_collision_risk(around, lethal)


def test_default_candidates_pick_detour():
    msg = FutureOccupancy()
    # wall across the straight lane
    msg.grids = [_grid(x, 1.6) for x in np.linspace(1.0, 2.2, 5)]
    lethal = merge_lethal_cells(msg.grids)
    scored = [(n, path_collision_risk(xy, lethal)) for n, xy in default_candidates()]
    risks = {n: r for n, r in scored}
    assert risks["straight"] > 0.0
    assert min(risks.values()) < risks["straight"]
