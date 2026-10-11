"""Tests for LiDAR + box -> BEV occupancy label building (numpy only, CI-safe).

    pytest world_model_py/test/test_occupancy_labels.py
"""
import numpy as np
import pytest

from world_model_py.camera_model import CameraModel
from world_model_py.occupancy_labels import (
    BevGrid, StaticMap, camera_fov, height_above_ground, make_tf, points_in_boxes,
    quat_to_rot, rasterize_boxes, rasterize_points, ray_visibility, transform, yaw_of,
)

GRID = BevGrid(20, 20, 0.5, (0.0, -5.0))     # 10 m ahead, +-5 m


def _cell(x, y):
    i, j, ok = GRID.index(np.array([[x, y]]))
    assert ok[0]
    return int(i[0]), int(j[0])


def test_quaternion_and_transform_roundtrip():
    q = (np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4))      # +90 deg about z
    R = quat_to_rot(q)
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0])
    assert yaw_of(R) == pytest.approx(np.pi / 2)
    T = make_tf([1.0, 2.0, 0.0], q)
    p = np.array([[3.0, 0.0, 1.0]])
    assert np.allclose(transform(np.linalg.inv(T), transform(T, p)), p)
    assert np.allclose(transform(T, p), [[1.0, 5.0, 1.0]])


def test_rasterize_points_and_index():
    occ = rasterize_points(np.array([[2.1, 0.1], [2.2, 0.2], [9.9, 4.9], [50.0, 0.0]]), GRID)
    assert occ[_cell(2.1, 0.1)] and occ[_cell(9.9, 4.9)] and occ.sum() == 2
    assert rasterize_points(np.array([[2.1, 0.1]]), GRID, min_points=2).sum() == 0


def test_rasterize_boxes_respects_yaw():
    along_x = rasterize_boxes(np.array([[5.0, 0.0, 0.0, 4.0, 1.0]]), GRID)
    along_y = rasterize_boxes(np.array([[5.0, 0.0, np.pi / 2, 4.0, 1.0]]), GRID)
    assert along_x[_cell(6.6, 0.1)] and not along_x[_cell(5.1, 1.9)]
    assert along_y[_cell(5.1, 1.9)] and not along_y[_cell(6.6, 0.1)]
    assert rasterize_boxes(np.zeros((0, 5)), GRID).sum() == 0


def test_points_in_boxes_checks_height():
    box = np.array([[5.0, 0.0, 1.0, 0.0, 4.0, 2.0, 2.0]])    # centre z=1, height 2
    pts = np.array([[5.0, 0.0, 0.5], [5.0, 0.0, 3.5], [8.0, 0.0, 1.0]])
    assert points_in_boxes(pts, box).tolist() == [True, False, False]
    assert points_in_boxes(np.array([[7.3, 0.0, 1.0]]), box, inflate=0.5).tolist() == [True]


def test_ray_visibility_stops_at_hits():
    wall = np.stack([np.full(200, 6.0), np.linspace(-5, 5, 200)], 1)
    vis = ray_visibility(wall, (0.0, 0.0), GRID)
    assert vis[_cell(3.0, 0.0)] and vis[_cell(6.1, 0.0)]       # free space + the wall
    assert not vis[_cell(8.0, 0.0)]                             # behind it


def test_height_above_ground_handles_slope_and_stray_returns():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 40, (6000, 2))
    ground = np.column_stack([xy, 0.05 * xy[:, 0]])             # 5% slope: 2 m at 40 m
    obj = np.column_stack([rng.uniform(30, 31, (200, 2)), 1.0 + 0.05 * 30 + rng.uniform(0, 0.5, 200)])
    stray = np.array([[20.0, 20.0, -2.0]])                     # a reflection below the road
    pts = np.concatenate([ground, obj, stray])
    h = height_above_ground(pts)
    g = h[: len(ground)]
    assert np.percentile(np.abs(g), 95) < 0.15                  # road stays ~0 despite the slope
    assert np.median(h[len(ground):-1]) > 0.8                   # the object stands out
    near = np.linalg.norm(ground[:, :2] - [20, 20], axis=1) < 2
    assert np.abs(g[near]).max() < 0.3                          # the stray point didn't sink it


def test_camera_fov_and_static_map():
    cam = CameraModel.pinhole(100, 60, 50, 50, 50, 10, position=(0, 0, 1.5))
    fov = camera_fov(GRID, cam)
    assert fov[_cell(8.0, 0.0)] and not fov[_cell(0.6, 4.5)]
    m = StaticMap(voxel=0.5, min_hits=2)
    m.add(np.array([[1.1, 1.1, 1.1], [3.0, 3.0, 1.0]]))
    m.add(np.array([[1.2, 1.2, 1.2]]))
    pts = m.points()
    assert len(pts) == 1 and np.allclose(pts[0], [1.25, 1.25, 1.25])
