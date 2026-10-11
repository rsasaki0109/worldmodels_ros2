"""nuScenes -> (camera frame, ego command, future BEV occupancy, visibility).

A minimal reader over the nuScenes JSON tables (no devkit dependency) that
produces training samples for :class:`~world_model_py.occupancy_head.OccupancyHead`
with labels from :mod:`world_model_py.occupancy_labels`:

  * every ``CAM_FRONT`` frame (keyframes + sweeps, ~12 Hz) is an observation;
  * future occupancy at ``t + k*dt`` = the scene's static LiDAR map plus all
    annotated boxes interpolated to that time, in the ego frame at ``t``;
  * visibility = cells a LiDAR ray reached at ``t`` within the camera's view.

Download ``v1.0-mini`` (public, ~4 GB; nuScenes terms of use apply -- register at
nuscenes.org) and point ``root`` at the extracted folder. Only ``CAM_FRONT``,
``LIDAR_TOP`` and the JSON tables are read.
"""
from __future__ import annotations

import bisect
import json
import os
from collections import defaultdict

import numpy as np

from .camera_model import CameraModel
from .occupancy_labels import (
    BevGrid, StaticMap, camera_fov, height_above_ground, make_tf, points_in_boxes, quat_to_rot,
    rasterize_boxes, rasterize_points, ray_visibility, transform, yaw_of,
)

#: obstacle height band (metres above the local ground) for static points
Z_BAND = (0.3, 2.5)


class NuScenesTables:
    """Just the JSON tables we need, indexed once.

    ``v1.0-trainval`` metadata is large (millions of ``sample_data`` / ``ego_pose``
    rows), so only the requested channels are kept, and only scenes whose files
    are present on disk are reported (e.g. one of the ten trainval blobs)."""

    def __init__(self, root: str, version: str = "v1.0-mini",
                 channels=("CAM_FRONT", "LIDAR_TOP")):
        self.root = root
        self.version = version

        def load(name):
            with open(os.path.join(root, version, f"{name}.json")) as f:
                return json.load(f)

        self.scene = load("scene")
        self.sample = {r["token"]: r for r in load("sample")}
        self.calib = {r["token"]: r for r in load("calibrated_sensor")}
        sensors = {r["token"]: r["channel"] for r in load("sensor")}
        self.channel_of = {t: sensors[c["sensor_token"]] for t, c in self.calib.items()}
        keep = {t for t, ch in self.channel_of.items() if ch in channels}

        # scene token -> channel -> [sample_data], built in one pass
        self.by_scene = defaultdict(lambda: defaultdict(list))
        poses = set()
        for d in load("sample_data"):
            if d["calibrated_sensor_token"] in keep:
                scene_tok = self.sample[d["sample_token"]]["scene_token"]
                self.by_scene[scene_tok][self.channel_of[d["calibrated_sensor_token"]]].append(d)
                poses.add(d["ego_pose_token"])
        self.ego_pose = {r["token"]: r for r in load("ego_pose") if r["token"] in poses}

        self.ann_by_scene = defaultdict(list)
        for a in load("sample_annotation"):
            self.ann_by_scene[self.sample[a["sample_token"]]["scene_token"]].append(a)

    def scene_names(self, available_only: bool = True, channel: str = "CAM_FRONT") -> list:
        """Scene names, sorted; by default only scenes whose first ``channel``
        file exists under ``root``."""
        out = []
        for s in sorted(self.scene, key=lambda s: s["name"]):
            sds = self.by_scene[s["token"]][channel]
            if not sds:
                continue
            if available_only and not os.path.exists(os.path.join(self.root, sds[0]["filename"])):
                continue
            out.append(s["name"])
        return out


class SceneOccupancy:
    """Samples of one nuScenes scene. Indices refer to ``CAM_FRONT`` frames.

    Works with the full sensor set or with LiDAR *keyframes* only (``sweeps/
    LIDAR_TOP`` not downloaded): scans whose files are missing are skipped, and
    the static map then needs fewer repeated hits (2 Hz instead of 20 Hz)."""

    def __init__(self, tables: NuScenesTables, scene_name: str, grid: BevGrid,
                 horizon: int = 4, dt: float = 0.5, image_width: int = 448,
                 camera: str = "CAM_FRONT", lidar: str = "LIDAR_TOP"):
        self.t = tables
        self.grid = grid
        self.horizon = int(horizon)
        self.dt = float(dt)
        self.image_width = int(image_width)
        scene = next(s for s in tables.scene if s["name"] == scene_name)
        self.name = scene_name
        root = tables.root
        chans = tables.by_scene[scene["token"]]
        exists = lambda d: os.path.exists(os.path.join(root, d["filename"]))
        self.cams = sorted(filter(exists, chans[camera]), key=lambda d: d["timestamp"])
        self.lidars = sorted(filter(exists, chans[lidar]), key=lambda d: d["timestamp"])
        self.lidar_sweeps = any(not d["is_key_frame"] for d in self.lidars)
        self._lidar_ts = [d["timestamp"] for d in self.lidars]
        self.times = np.array([d["timestamp"] for d in self.cams], np.int64)

        self._tracks = self._build_tracks(tables.ann_by_scene[scene["token"]])
        self._static_pts = None

    @property
    def _static(self) -> np.ndarray:
        """World static points, built on first use (needs the LiDAR files)."""
        if self._static_pts is None:
            # 20 Hz sweeps: every 2nd scan, 6 hits; 2 Hz keyframes: every scan, 3 hits
            stride, hits = (2, 6) if self.lidar_sweeps else (1, 3)
            self._static_pts = self._build_static_map(stride, hits)
        return self._static_pts

    # --- poses -----------------------------------------------------------------
    def _T_world_ego(self, sd) -> np.ndarray:
        p = self.t.ego_pose[sd["ego_pose_token"]]
        return make_tf(p["translation"], p["rotation"])

    def _T_ego_sensor(self, sd) -> np.ndarray:
        c = self.t.calib[sd["calibrated_sensor_token"]]
        return make_tf(c["translation"], c["rotation"])

    def ego_pose(self, i: int) -> np.ndarray:
        return self._T_world_ego(self.cams[i])

    def ego_command(self, i: int) -> np.ndarray:
        """[forward speed m/s, yaw rate rad/s] from the previous camera frame."""
        if i == 0:
            return np.zeros(2, np.float32)
        Ta, Tb = self.ego_pose(i - 1), self.ego_pose(i)
        dt = max((self.times[i] - self.times[i - 1]) * 1e-6, 1e-3)
        rel = np.linalg.inv(Ta) @ Tb
        return np.array([rel[0, 3] / dt, yaw_of(rel[:3, :3]) / dt], np.float32)

    # --- camera ----------------------------------------------------------------
    def camera_model(self, i: int = 0) -> CameraModel:
        sd = self.cams[i]
        c = self.t.calib[sd["calibrated_sensor_token"]]
        s = self.image_width / sd["width"]
        K = np.asarray(c["camera_intrinsic"], np.float64).copy()
        K[:2] *= s
        h = int(round(sd["height"] * s))
        return CameraModel.from_camera_info(K, self.image_width, h, np.linalg.inv(self._T_ego_sensor(sd)))

    def image(self, i: int) -> np.ndarray:
        from PIL import Image
        sd = self.cams[i]
        img = Image.open(os.path.join(self.t.root, sd["filename"])).convert("RGB")
        h = int(round(sd["height"] * self.image_width / sd["width"]))
        return np.asarray(img.resize((self.image_width, h), Image.BILINEAR))

    # --- boxes -----------------------------------------------------------------
    def _build_tracks(self, annotations) -> dict:
        tracks = defaultdict(list)
        for a in annotations:
            ts = self.t.sample[a["sample_token"]]["timestamp"]
            w, l, h = a["size"]
            yaw = yaw_of(quat_to_rot(a["rotation"]))
            tracks[a["instance_token"]].append((ts, *a["translation"], yaw, l, w, h))
        return {k: np.array(sorted(v)) for k, v in tracks.items()}

    def boxes_world(self, ts: int) -> np.ndarray:
        """All annotated boxes interpolated to ``ts`` -> (N, 7) [x,y,z,yaw,l,w,h].
        Objects are only present inside their annotated time span."""
        out = []
        for tr in self._tracks.values():
            t = tr[:, 0]
            if ts < t[0] or ts > t[-1]:
                continue
            k = int(np.searchsorted(t, ts))
            if t[k] == ts or k == 0:
                out.append(tr[k, 1:])
                continue
            a, b = tr[k - 1], tr[k]
            u = (ts - a[0]) / (b[0] - a[0])
            dyaw = (b[4] - a[4] + np.pi) % (2 * np.pi) - np.pi
            r = a[1:] + u * (b[1:] - a[1:])
            r[3] = a[4] + u * dyaw
            out.append(r)
        return np.array(out).reshape(-1, 7)

    @staticmethod
    def _boxes_to_ego(boxes_w: np.ndarray, T_ego_world: np.ndarray) -> np.ndarray:
        """World boxes (N, 7) -> ego-frame footprints (N, 5) [x, y, yaw, l, w]."""
        if len(boxes_w) == 0:
            return np.zeros((0, 5))
        xy = transform(T_ego_world, boxes_w[:, :3])[:, :2]
        yaw = boxes_w[:, 3] + yaw_of(T_ego_world[:3, :3])
        return np.column_stack([xy, yaw, boxes_w[:, 4], boxes_w[:, 5]])

    # --- lidar -----------------------------------------------------------------
    def _lidar_points(self, sd) -> np.ndarray:
        pts = np.fromfile(os.path.join(self.t.root, sd["filename"]), dtype=np.float32)
        return pts.reshape(-1, 5)[:, :3].astype(np.float64)

    def _lidar_world(self, sd):
        """(points in world, points in that scan's ego frame, sensor origin world)."""
        T_ego_s = self._T_ego_sensor(sd)
        T_w_ego = self._T_world_ego(sd)
        ego = transform(T_ego_s, self._lidar_points(sd))
        return transform(T_w_ego, ego), ego, transform(T_w_ego @ T_ego_s, np.zeros((1, 3)))[0]

    def _build_static_map(self, stride: int, min_hits: int) -> np.ndarray:
        smap = StaticMap(voxel=0.25, min_hits=min_hits)
        for sd in self.lidars[::max(1, stride)]:
            world, ego, _ = self._lidar_world(sd)
            r = np.linalg.norm(ego[:, :2], axis=1)
            near = r < 60.0
            world, ego, r = world[near], ego[near], r[near]
            hag = height_above_ground(ego)
            keep = (hag > Z_BAND[0]) & (hag < Z_BAND[1]) & (r > 3.0)
            world = world[keep]
            boxes = self.boxes_world(sd["timestamp"])
            if len(boxes):
                world = world[~points_in_boxes(world, boxes, inflate=0.8)]
            smap.add(world)
        return smap.points()

    def _nearest_lidar(self, ts: int):
        k = bisect.bisect_left(self._lidar_ts, ts)
        cands = [c for c in (k - 1, k) if 0 <= c < len(self.lidars)]
        return self.lidars[min(cands, key=lambda c: abs(self._lidar_ts[c] - ts))]

    # --- labels ----------------------------------------------------------------
    def valid_indices(self) -> np.ndarray:
        """Frames with a previous frame and the full future horizon inside the scene."""
        end = self.times[-1]
        need = int(self.horizon * self.dt * 1e6)
        return np.array([i for i in range(1, len(self.cams)) if self.times[i] + need <= end])

    def _static_ego(self, T_ego_world) -> np.ndarray:
        if len(self._static) == 0:
            return np.zeros(self.grid.shape, bool)
        ego = transform(T_ego_world, self._static)
        return rasterize_points(ego[:, :2], self.grid, min_points=1)

    def static_occupancy(self, i: int) -> np.ndarray:
        """Static structure only (no boxes) in the ego frame at ``i``."""
        return self._static_ego(np.linalg.inv(self.ego_pose(i)))

    def occupancy_at(self, i: int, offset_s: float) -> np.ndarray:
        """Occupancy at camera time ``i`` + ``offset_s`` in the ego frame at ``i``."""
        T_ego_world = np.linalg.inv(self.ego_pose(i))
        ts = int(self.times[i] + offset_s * 1e6)
        boxes = self._boxes_to_ego(self.boxes_world(ts), T_ego_world)
        return self._static_ego(T_ego_world) | rasterize_boxes(boxes, self.grid)

    def future_occupancy(self, i: int) -> np.ndarray:
        """(horizon, rows, cols) bool: occupancy at ``t + k*dt``, k = 1..horizon."""
        T_ego_world = np.linalg.inv(self.ego_pose(i))
        static = self._static_ego(T_ego_world)
        out = []
        for k in range(1, self.horizon + 1):
            ts = int(self.times[i] + k * self.dt * 1e6)
            boxes = self._boxes_to_ego(self.boxes_world(ts), T_ego_world)
            out.append(static | rasterize_boxes(boxes, self.grid))
        return np.stack(out)

    def visibility(self, i: int) -> np.ndarray:
        """Cells observed by the LiDAR scan nearest to frame ``i`` and inside the
        camera's view (ego frame at ``i``)."""
        T_ego_world = np.linalg.inv(self.ego_pose(i))
        world, _, origin = self._lidar_world(self._nearest_lidar(int(self.times[i])))
        ego = transform(T_ego_world, world)
        o = transform(T_ego_world, origin[None])[0]
        ego = ego[ego[:, 2] < Z_BAND[1]]
        vis = ray_visibility(ego[:, :2], o[:2], self.grid)
        return vis & camera_fov(self.grid, self.camera_model(i))
