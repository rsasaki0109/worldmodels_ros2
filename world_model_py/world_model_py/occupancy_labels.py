"""Build BEV occupancy labels from LiDAR + 3D boxes (pure numpy).

Training an occupancy head on real data needs labels: which cells are occupied
now and over the next few seconds, and which cells the sensors actually
observed. This module turns raw robot data into those labels, independent of
any dataset format (nuScenes today, a camera + LiDAR rosbag tomorrow):

  * **static structure** -- LiDAR points in an obstacle height band, accumulated
    in the world frame over a whole sequence with points inside annotated boxes
    removed (so moving things don't smear);
  * **agents** -- annotated 3D boxes rasterised as oriented footprints at the
    query time (future occupancy = boxes at ``t + k*dt``, static map unchanged);
  * **visibility** -- cells crossed by (or hit by) a LiDAR ray at the
    observation time, intersected with the camera's field of view. This mirrors
    how Occ3D builds its ``mask_camera``.

Frames follow ROS / nuScenes ego conventions: x forward, y left, z up. Grids use
the ``nav_msgs/OccupancyGrid`` layout ``grid[i, j]`` = (y row, x column) with the
lower-left corner at ``origin``.
"""
from __future__ import annotations

from dataclasses import dataclass

import warnings

import numpy as np


# --------------------------------------------------------------------- transforms
def quat_to_rot(q) -> np.ndarray:
    """Quaternion ``(w, x, y, z)`` -> 3x3 rotation matrix."""
    w, x, y, z = (float(v) for v in q)
    n = np.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def make_tf(translation, rotation_wxyz) -> np.ndarray:
    """4x4 transform taking points from the child frame into the parent frame."""
    T = np.eye(4)
    T[:3, :3] = quat_to_rot(rotation_wxyz)
    T[:3, 3] = np.asarray(translation, dtype=np.float64)
    return T


def transform(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    return pts @ T[:3, :3].T + T[:3, 3]


def yaw_of(R: np.ndarray) -> float:
    return float(np.arctan2(R[1, 0], R[0, 0]))


# -------------------------------------------------------------------------- grid
@dataclass(frozen=True)
class BevGrid:
    rows: int
    cols: int
    resolution: float
    origin: tuple            # (x, y) of the lower-left corner

    @property
    def shape(self) -> tuple:
        return (self.rows, self.cols)

    def centers(self) -> np.ndarray:
        """(rows, cols, 2) cell-centre (x, y)."""
        xs = self.origin[0] + (np.arange(self.cols) + 0.5) * self.resolution
        ys = self.origin[1] + (np.arange(self.rows) + 0.5) * self.resolution
        X, Y = np.meshgrid(xs, ys)
        return np.stack([X, Y], -1)

    def index(self, xy: np.ndarray):
        """(N, 2) points -> (row, col, inside)."""
        xy = np.asarray(xy, dtype=np.float64)
        j = np.floor((xy[:, 0] - self.origin[0]) / self.resolution).astype(np.int64)
        i = np.floor((xy[:, 1] - self.origin[1]) / self.resolution).astype(np.int64)
        inside = (i >= 0) & (i < self.rows) & (j >= 0) & (j < self.cols)
        return i, j, inside


def rasterize_points(xy: np.ndarray, grid: BevGrid, min_points: int = 1) -> np.ndarray:
    """Cells containing at least ``min_points`` of the (N, 2) points."""
    i, j, ok = grid.index(xy)
    counts = np.zeros(grid.shape, np.int32)
    np.add.at(counts, (i[ok], j[ok]), 1)
    return counts >= min_points


def rasterize_boxes(boxes: np.ndarray, grid: BevGrid, inflate: float = 0.0) -> np.ndarray:
    """Oriented footprints ``(N, 5) [x, y, yaw, length, width]`` -> (rows, cols)
    bool. A cell is occupied when its centre is inside the footprint grown by
    ``inflate`` + half a cell (so thin objects still mark a cell)."""
    occ = np.zeros(grid.shape, bool)
    if len(boxes) == 0:
        return occ
    c = grid.centers().reshape(-1, 2)
    pad = inflate + 0.5 * grid.resolution
    for x, y, yaw, length, width in np.asarray(boxes, dtype=np.float64):
        d = c - (x, y)
        cs, sn = np.cos(yaw), np.sin(yaw)
        lon = d[:, 0] * cs + d[:, 1] * sn
        lat = -d[:, 0] * sn + d[:, 1] * cs
        occ |= ((np.abs(lon) <= length / 2 + pad) & (np.abs(lat) <= width / 2 + pad)).reshape(grid.shape)
    return occ


def points_in_boxes(pts: np.ndarray, boxes: np.ndarray, inflate: float = 0.0) -> np.ndarray:
    """(N, 3) points vs (M, 7) boxes ``[x, y, z, yaw, length, width, height]``
    (z = box centre) -> (N,) bool, inside any box grown by ``inflate``."""
    inside = np.zeros(len(pts), bool)
    for x, y, z, yaw, length, width, height in np.asarray(boxes, dtype=np.float64):
        d = pts[:, :2] - (x, y)
        cs, sn = np.cos(yaw), np.sin(yaw)
        lon = d[:, 0] * cs + d[:, 1] * sn
        lat = -d[:, 0] * sn + d[:, 1] * cs
        inside |= ((np.abs(lon) <= length / 2 + inflate) & (np.abs(lat) <= width / 2 + inflate)
                   & (np.abs(pts[:, 2] - z) <= height / 2 + inflate))
    return inside


def height_above_ground(pts: np.ndarray, cell: float = 1.0, window: int = 2,
                        percentile: float = 10.0, outlier: float = 0.5) -> np.ndarray:
    """Per-point height above a local ground estimate (pure numpy).

    Ground = a low percentile of the returns in each ``cell`` x ``cell`` column
    (a percentile, not the minimum: LiDAR has stray returns below the road),
    min-filtered over a ``(2*window+1)^2`` neighbourhood so a column holding
    only a car roof borrows the road next to it. Columns more than ``outlier``
    metres below their neighbourhood median are dropped first. Robust to slopes and to
    ego-pose pitch error, which a fixed z threshold is not at 30+ m range."""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) == 0:
        return np.zeros(0)
    ij = np.floor(pts[:, :2] / cell).astype(np.int64)
    ij -= ij.min(0)
    shape = tuple(ij.max(0) + 1)
    flat = ij[:, 0] * shape[1] + ij[:, 1]
    order = np.lexsort((pts[:, 2], flat))                 # by cell, then z
    fs, zs = flat[order], pts[order, 2]
    starts = np.flatnonzero(np.r_[True, fs[1:] != fs[:-1]])
    counts = np.diff(np.r_[starts, len(fs)])
    pick = starts + np.floor((counts - 1) * percentile / 100.0).astype(np.int64)
    ground = np.full(shape[0] * shape[1], np.inf)
    ground[fs[starts]] = zs[pick]
    ground = ground.reshape(shape)
    w = int(window)

    def shifts(a, fill):
        p = np.pad(a, w, constant_values=fill)
        return np.stack([p[dy:dy + shape[0], dx:dx + shape[1]]
                         for dy in range(2 * w + 1) for dx in range(2 * w + 1)])

    # a column far below its neighbourhood's median is a stray return, not road
    g = np.where(np.isfinite(ground), ground, np.nan)
    with warnings.catch_warnings():               # all-nan windows -> nan median
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(shifts(g, np.nan), axis=0)
    ground = np.where(ground < med - outlier, np.inf, ground)
    local = shifts(ground, np.inf).min(axis=0)
    return pts[:, 2] - local[ij[:, 0], ij[:, 1]]


def ray_visibility(hits_xy: np.ndarray, origin_xy, grid: BevGrid,
                   step: float | None = None, azimuth_bin_deg: float = 0.2) -> np.ndarray:
    """Cells crossed by a 2D ray from ``origin_xy`` to any hit, hit cells included.
    Free space a ray passed through and surfaces it hit are both *observed*;
    cells behind every hit stay unobserved."""
    hits = np.asarray(hits_xy, dtype=np.float64)
    vis = np.zeros(grid.shape, bool)
    if len(hits) == 0:
        return vis
    o = np.asarray(origin_xy, dtype=np.float64)
    step = step or grid.resolution * 0.5
    # A ray only matters up to the farthest grid corner, and along one azimuth
    # the longest ray covers every shorter one: keep one clipped ray per bin.
    d = hits - o
    rng = np.linalg.norm(d, axis=1)
    corners = np.array([[grid.origin[0], grid.origin[1]],
                        [grid.origin[0] + grid.cols * grid.resolution, grid.origin[1]],
                        [grid.origin[0], grid.origin[1] + grid.rows * grid.resolution],
                        [grid.origin[0] + grid.cols * grid.resolution,
                         grid.origin[1] + grid.rows * grid.resolution]])
    reach = np.linalg.norm(corners - o, axis=1).max()
    az = np.floor((np.arctan2(d[:, 1], d[:, 0]) + np.pi) / np.radians(azimuth_bin_deg)).astype(np.int64)
    order = np.lexsort((-rng, az))
    first = order[np.r_[True, az[order][1:] != az[order][:-1]]]
    scale = np.minimum(rng[first], reach) / np.maximum(rng[first], 1e-9)
    hits = o + d[first] * scale[:, None]
    length = np.linalg.norm(hits - o, axis=1)
    n = int(np.ceil(length.max() / step)) + 1
    s = np.linspace(0.0, 1.0, n)
    # chunk to bound memory on dense scans
    for a in range(0, len(hits), 4096):
        h = hits[a:a + 4096]
        pts = o + s[None, :, None] * (h - o)[:, None, :]          # (m, n, 2)
        i, j, ok = grid.index(pts.reshape(-1, 2))
        vis[i[ok], j[ok]] = True
    return vis


def camera_fov(grid: BevGrid, camera, z: float = 0.0, max_range: float | None = None) -> np.ndarray:
    """Cells whose point at height ``z`` projects into the camera image."""
    c = grid.centers().reshape(-1, 2)
    pts = np.concatenate([c, np.full((len(c), 1), z)], 1)
    _, _, ok = camera.project(pts)
    if max_range is not None:
        ok &= np.linalg.norm(c, axis=1) <= max_range
    return ok.reshape(grid.shape)


class StaticMap:
    """World-frame static obstacle points accumulated over a sequence, voxel-
    downsampled; a voxel counts once it was hit in ``min_hits`` scans."""

    def __init__(self, voxel: float = 0.25, min_hits: int = 2):
        self.voxel = float(voxel)
        self.min_hits = int(min_hits)
        self._hits: dict = {}

    def add(self, world_pts: np.ndarray) -> None:
        if len(world_pts) == 0:
            return
        keys = np.unique(np.floor(np.asarray(world_pts) / self.voxel).astype(np.int64), axis=0)
        for k in map(tuple, keys):
            self._hits[k] = self._hits.get(k, 0) + 1

    def points(self) -> np.ndarray:
        keep = [k for k, v in self._hits.items() if v >= self.min_hits]
        if not keep:
            return np.zeros((0, 3))
        return (np.asarray(keep, dtype=np.float64) + 0.5) * self.voxel
