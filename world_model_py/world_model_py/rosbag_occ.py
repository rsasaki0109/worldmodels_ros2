"""Camera + LiDAR rosbag (ROS 1 or ROS 2) -> occupancy-head training samples.

The bring-your-own-data path for :class:`~world_model_py.occupancy_head.OccupancyHead`:
record a bag on your robot, then train a head on it. Reading uses the pure-python
`rosbags <https://gitlab.com/ternaris/rosbags>`_ library, so no ROS install is
needed and both ``.bag`` (ROS 1) and rosbag2 directories work. Messages are
decoded with the standard definitions, not the ones stored in the bag, which
are sometimes empty in real recordings.

What a bag must contain (topics are auto-detected by type, or given explicitly):

  * a camera: ``sensor_msgs/Image`` or ``CompressedImage`` + its ``CameraInfo``;
  * a 3D LiDAR: ``sensor_msgs/PointCloud2``;
  * the robot pose: ``nav_msgs/Odometry`` (its ``child_frame_id`` is the base frame);
  * the sensor mounts: ``/tf_static`` (or static edges on ``/tf``).

Labels need no annotation. Occupancy at ``t + k*dt`` is the LiDAR returns of the
scans around that time, taken 0.3–2.0 m above a local ground estimate and moved
into the robot frame at ``t``. Moving people and robots therefore appear where
they really went. Visibility is the cells a LiDAR ray reached at ``t``, within
the camera's view. Static structure and movers are not told apart (no boxes), so
the agents-only metric is not reported for bags.
"""
from __future__ import annotations

import io
from collections import defaultdict
from pathlib import Path

import numpy as np

from .camera_model import CameraModel
from .occupancy_labels import (
    BevGrid, camera_fov, height_above_ground, make_tf, rasterize_points, ray_visibility,
    transform, yaw_of,
)

#: obstacle height band above the local ground (metres)
Z_BAND = (0.3, 2.0)

_IMAGE_TYPES = ("sensor_msgs/msg/CompressedImage", "sensor_msgs/msg/Image")
_FLOAT32 = 7   # sensor_msgs/PointField.FLOAT32


# ---------------------------------------------------------------- message parsing
def stamp_ns(header) -> int:
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def pointcloud2_xyz(msg) -> np.ndarray:
    """``sensor_msgs/PointCloud2`` -> (N, 3) float32 xyz, NaN/inf points dropped."""
    fields = {f.name: f for f in msg.fields}
    for k in ("x", "y", "z"):
        if k not in fields or fields[k].datatype != _FLOAT32:
            raise ValueError(f"PointCloud2 needs float32 x/y/z fields, got {sorted(fields)}")
    dtype = np.dtype({"names": ["x", "y", "z"],
                      "formats": ["<f4" if not msg.is_bigendian else ">f4"] * 3,
                      "offsets": [fields[k].offset for k in ("x", "y", "z")],
                      "itemsize": msg.point_step})
    n = msg.width * msg.height
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if msg.row_step * msg.height != len(raw):
        raw = raw[: msg.row_step * msg.height]
    pts = np.frombuffer(raw.tobytes()[: n * msg.point_step], dtype=dtype, count=n)
    xyz = np.stack([pts["x"], pts["y"], pts["z"]], -1).astype(np.float32)
    return xyz[np.isfinite(xyz).all(1)]


def decode_image(msg) -> np.ndarray:
    """``Image`` (rgb8/bgr8/rgba8/bgra8/mono8) or ``CompressedImage`` -> HxWx3 uint8 RGB."""
    if hasattr(msg, "format"):                       # CompressedImage
        from PIL import Image
        return np.asarray(Image.open(io.BytesIO(bytes(msg.data))).convert("RGB"))
    enc = msg.encoding.lower()
    ch = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}.get(enc)
    if ch is None:
        raise ValueError(f"unsupported image encoding '{msg.encoding}'")
    arr = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.step)
    arr = arr[:, : msg.width * ch].reshape(msg.height, msg.width, ch)
    if ch == 1:
        return np.repeat(arr, 3, 2)
    if enc.startswith("bgr"):
        arr = arr[..., [2, 1, 0]]
    return np.ascontiguousarray(arr[..., :3])


#: fallbacks for types a typestore may lack (e.g. tf2_msgs in the ROS 1 store)
_FALLBACK_MSGDEFS = {
    "tf2_msgs/msg/TFMessage": "geometry_msgs/TransformStamped[] transforms\n",
}


def _ensure_types(store, conns) -> None:
    """Register any message type the typestore lacks: from a built-in definition
    when we have one (bags can store wrong or empty ones), else from the bag."""
    from rosbags.typesys import get_types_from_msg
    for c in conns:
        if c.msgtype in store.types:
            continue
        text = _FALLBACK_MSGDEFS.get(c.msgtype, "") or getattr(getattr(c, "msgdef", None), "data", "")
        if not text:
            raise KeyError(f"no message definition for {c.msgtype} ({c.topic})")
        store.register(get_types_from_msg(text, c.msgtype))


def _tf_of(transform_msg) -> np.ndarray:
    t, q = transform_msg.translation, transform_msg.rotation
    return make_tf([t.x, t.y, t.z], [q.w, q.x, q.y, q.z])


class TfTree:
    """Static transforms, looked up between any two connected frames."""

    def __init__(self):
        self.edges = defaultdict(dict)        # frame -> {neighbour: T_frame_neighbour}

    def add(self, parent: str, child: str, T_parent_child: np.ndarray) -> None:
        parent, child = parent.lstrip("/"), child.lstrip("/")
        self.edges[parent][child] = T_parent_child
        self.edges[child][parent] = np.linalg.inv(T_parent_child)

    def lookup(self, target: str, source: str) -> np.ndarray:
        """T_target_source: maps points in ``source`` into ``target``."""
        target, source = target.lstrip("/"), source.lstrip("/")
        if target == source:
            return np.eye(4)
        seen, frontier = {target: np.eye(4)}, [target]
        while frontier:
            nxt = []
            for f in frontier:
                for g, T in self.edges[f].items():
                    if g not in seen:
                        seen[g] = seen[f] @ T
                        if g == source:
                            return seen[g]
                        nxt.append(g)
            frontier = nxt
        raise KeyError(f"no static transform between '{target}' and '{source}' "
                       f"(known frames: {sorted(self.edges)})")


# ------------------------------------------------------------------------ the bag
class BagOccupancy:
    """All samples of one bag. Indices refer to (stride-subsampled) camera frames."""

    def __init__(self, path, grid: BevGrid, horizon: int = 4, dt: float = 0.5,
                 image_width: int = 448, image_topic: str = "", camera_info_topic: str = "",
                 lidar_topic: str = "", odom_topic: str = "", camera_stride: int = 1,
                 max_range: float = 30.0, scan_window: float = 0.1, robot_radius: float = 0.5,
                 intrinsics=None, camera_frame: str = "", log=None):
        """``intrinsics=(fx, fy, cx, cy)`` (pixels at the recorded resolution) and
        ``camera_frame`` replace a missing ``CameraInfo`` topic."""
        from rosbags.typesys import Stores, get_typestore

        self.path = Path(path)
        self.name = self.path.stem
        self.grid = grid
        self.horizon, self.dt = int(horizon), float(dt)
        self.image_width = int(image_width)
        self.scan_window_ns = int(scan_window * 1e9)
        log = log or (lambda *_: None)

        # Decode with the standard message definitions only: recorders sometimes
        # store empty or stale ones (common in ROS 1 bags), and every type read
        # here is a stock sensor_msgs / nav_msgs / tf2_msgs message.
        if self.path.suffix == ".bag":
            from rosbags.rosbag1 import Reader
            store = get_typestore(Stores.ROS1_NOETIC)
            deserialize = store.deserialize_ros1
        else:
            from rosbags.rosbag2 import Reader
            store = get_typestore(Stores.LATEST)
            deserialize = store.deserialize_cdr
        with Reader(self.path) as reader:
            conns = {c.topic: c for c in reader.connections}

            def pick(given, types, prefer=()):
                if given:
                    if given not in conns:
                        raise KeyError(f"topic '{given}' not in bag {self.name}: {sorted(conns)}")
                    return conns[given]
                cands = [c for c in conns.values() if c.msgtype in types]
                if not cands:
                    raise KeyError(f"no {types} topic in bag {self.name}: "
                                   f"{sorted((c.topic, c.msgtype) for c in conns.values())}")
                for p in prefer:
                    for c in cands:
                        if p in c.topic.lower():
                            return c
                return max(cands, key=lambda c: c.msgcount)

            cam_c = pick(image_topic, _IMAGE_TYPES, ("rgb", "color", "image"))
            info_c = None
            if intrinsics is None:
                try:
                    info_c = pick(camera_info_topic, ("sensor_msgs/msg/CameraInfo",),
                                  (cam_c.topic.rsplit("/", 2)[0],))
                except KeyError as exc:
                    raise KeyError(f"{exc} -- pass intrinsics=(fx, fy, cx, cy) "
                                   "(--camera-intrinsics) for bags without CameraInfo") from None
            lidar_c = pick(lidar_topic, ("sensor_msgs/msg/PointCloud2",), ("velodyne", "lidar", "points"))
            odom_c = pick(odom_topic, ("nav_msgs/msg/Odometry",), ("odom",))
            tf_cs = [c for c in conns.values() if c.msgtype == "tf2_msgs/msg/TFMessage"]
            log(f"[bag] {self.name}: camera={cam_c.topic} info={info_c.topic if info_c else intrinsics} "
                f"lidar={lidar_c.topic} odom={odom_c.topic}")
            self.topics = {"image": cam_c.topic, "camera_info": info_c.topic if info_c else "",
                           "lidar": lidar_c.topic, "odom": odom_c.topic}

            info = None
            cams, scans, odom = [], [], []
            tf = TfTree()
            dynamic = set()
            base_frame = None
            cam_count = 0
            native = None
            used = [c for c in (cam_c, info_c, lidar_c, odom_c, *tf_cs) if c is not None]
            _ensure_types(store, used)
            for conn, _, raw in reader.messages(connections=used):
                msg = deserialize(raw, conn.msgtype)
                if conn is info_c:
                    info = info or msg
                elif conn is cam_c:
                    if cam_count % max(1, int(camera_stride)) == 0:
                        img = decode_image(msg)
                        native = native if cam_count else (img.shape[1], img.shape[0])
                        cams.append((stamp_ns(msg.header), msg.header.frame_id, self._resize(img)))
                    cam_count += 1
                elif conn is lidar_c:
                    xyz = pointcloud2_xyz(msg)
                    xyz = xyz[np.linalg.norm(xyz[:, :2], axis=1) < max_range]
                    scans.append((stamp_ns(msg.header), msg.header.frame_id, xyz))
                elif conn is odom_c:
                    p = msg.pose.pose
                    odom.append((stamp_ns(msg.header), make_tf(
                        [p.position.x, p.position.y, p.position.z],
                        [p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z])))
                    base_frame = base_frame or msg.child_frame_id
                    self.odom_frame = msg.header.frame_id
                else:   # TF: keep static edges (anything not involving the odom pose)
                    for t in msg.transforms:
                        key = (t.header.frame_id.lstrip("/"), t.child_frame_id.lstrip("/"))
                        if key in dynamic:
                            continue
                        if conn.topic.endswith("tf") and not conn.topic.endswith("tf_static"):
                            if key in tf.edges.get(key[0], {}) and key not in dynamic:
                                prev = tf.edges[key[0]][key[1]]
                                if not np.allclose(prev, _tf_of(t.transform), atol=1e-4):
                                    dynamic.add(key)       # it moves: not a mount
                                    tf.edges[key[0]].pop(key[1], None)
                                    tf.edges[key[1]].pop(key[0], None)
                                    continue
                        tf.add(key[0], key[1], _tf_of(t.transform))

        if (info is None and intrinsics is None) or not cams or not scans or not odom:
            raise ValueError(f"bag {self.name}: missing camera_info/camera/lidar/odom messages")
        self.base_frame = base_frame.lstrip("/")
        cams.sort(key=lambda c: c[0])
        scans.sort(key=lambda s: s[0])
        odom.sort(key=lambda o: o[0])
        self._odom_t = np.array([o[0] for o in odom], np.int64)
        self._odom_T = np.stack([o[1] for o in odom])

        # sensor mounts in the base frame
        cam_frame = camera_frame or (info.header.frame_id if info is not None else "") or cams[0][1]
        self.T_base_cam = tf.lookup(self.base_frame, cam_frame)
        self.T_base_lidar = tf.lookup(self.base_frame, scans[0][1])
        if info is not None:
            K = np.asarray(info.k if hasattr(info, "k") else info.K, np.float64).reshape(3, 3).copy()
            width, height = float(info.width), float(info.height)
        else:
            fx, fy, cx, cy = (float(v) for v in intrinsics)
            K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
            width, height = float(native[0]), float(native[1])
        s = self.image_width / width
        K[:2] *= s
        h = int(round(height * s))
        self.camera = CameraModel.from_camera_info(K, self.image_width, h,
                                                   np.linalg.inv(self.T_base_cam))

        # keep camera frames inside the odometry span
        t0, t1 = self._odom_t[0], self._odom_t[-1]
        cams = [c for c in cams if t0 <= c[0] <= t1]
        self.times = np.array([c[0] for c in cams], np.int64)
        self.frames = [c[2] for c in cams]

        # per scan: world obstacle points, world points for visibility, sensor origin
        self._scan_t, self._scan_obs, self._scan_all, self._scan_origin = [], [], [], []
        for ts, _, xyz in scans:
            if not t0 <= ts <= t1:
                continue
            base = transform(self.T_base_lidar, xyz)
            r = np.linalg.norm(base[:, :2], axis=1)
            hag = height_above_ground(base)
            obs = (hag > Z_BAND[0]) & (hag < Z_BAND[1]) & (r > robot_radius)
            T_w = self._pose(ts) @ self.T_base_lidar
            world = transform(T_w, xyz).astype(np.float32)
            self._scan_t.append(ts)
            self._scan_obs.append(world[obs])
            self._scan_all.append(world[hag < Z_BAND[1]])
            self._scan_origin.append(T_w[:3, 3].copy())
        self._scan_t = np.array(self._scan_t, np.int64)
        log(f"  {len(self.frames)} camera frames, {len(self._scan_t)} scans, "
            f"{(t1 - t0) / 1e9:.0f} s, base frame '{self.base_frame}'")

    # --- helpers ----------------------------------------------------------------
    def _resize(self, img: np.ndarray) -> np.ndarray:
        from PIL import Image
        h = int(round(img.shape[0] * self.image_width / img.shape[1]))
        return np.asarray(Image.fromarray(img).resize((self.image_width, h), Image.BILINEAR))

    def _pose(self, ts: int) -> np.ndarray:
        """T_odom_base at ``ts``: translation interpolated, rotation from the
        nearer sample (wheeled/legged robots turn slowly between odom messages)."""
        k = int(np.clip(np.searchsorted(self._odom_t, ts), 1, len(self._odom_t) - 1))
        ta, tb = self._odom_t[k - 1], self._odom_t[k]
        u = float(np.clip((ts - ta) / max(tb - ta, 1), 0.0, 1.0))
        T = (self._odom_T[k - 1] if u < 0.5 else self._odom_T[k]).copy()
        T[:3, 3] = (1 - u) * self._odom_T[k - 1][:3, 3] + u * self._odom_T[k][:3, 3]
        return T

    def ego_pose(self, i: int) -> np.ndarray:
        return self._pose(int(self.times[i]))

    def image(self, i: int) -> np.ndarray:
        return self.frames[i]

    def camera_model(self, i: int = 0) -> CameraModel:
        return self.camera

    def ego_command(self, i: int) -> np.ndarray:
        """[forward speed m/s, yaw rate rad/s] since the previous camera frame."""
        if i == 0:
            return np.zeros(2, np.float32)
        rel = np.linalg.inv(self.ego_pose(i - 1)) @ self.ego_pose(i)
        dt = max((self.times[i] - self.times[i - 1]) * 1e-9, 1e-3)
        return np.array([rel[0, 3] / dt, yaw_of(rel[:3, :3]) / dt], np.float32)

    # --- labels -----------------------------------------------------------------
    def valid_indices(self) -> np.ndarray:
        """Frames with a previous frame and scans covering the whole horizon."""
        end = self._scan_t[-1] - self.scan_window_ns
        need = int(self.horizon * self.dt * 1e9)
        return np.array([i for i in range(1, len(self.times))
                         if self.times[i] + need <= end and self.times[i] >= self._scan_t[0]],
                        dtype=np.int64)

    def _scans_near(self, ts: int) -> np.ndarray:
        a = np.searchsorted(self._scan_t, ts - self.scan_window_ns)
        b = np.searchsorted(self._scan_t, ts + self.scan_window_ns, side="right")
        return np.arange(a, b)

    def occupancy_at(self, i: int, offset_s: float) -> np.ndarray:
        """LiDAR obstacle returns around ``t_i + offset_s``, in the robot frame at ``t_i``."""
        T_ego_world = np.linalg.inv(self.ego_pose(i))
        idx = self._scans_near(int(self.times[i] + offset_s * 1e9))
        if len(idx) == 0:
            return np.zeros(self.grid.shape, bool)
        pts = np.concatenate([self._scan_obs[j] for j in idx])
        return rasterize_points(transform(T_ego_world, pts)[:, :2], self.grid)

    def future_occupancy(self, i: int) -> np.ndarray:
        return np.stack([self.occupancy_at(i, k * self.dt) for k in range(1, self.horizon + 1)])

    def static_occupancy(self, i: int) -> np.ndarray:
        """Not separable without boxes: nothing is marked static."""
        return np.zeros(self.grid.shape, bool)

    def visibility(self, i: int) -> np.ndarray:
        """Cells a LiDAR ray reached in the scan nearest to frame ``i``, within the
        camera's view (robot frame at ``i``)."""
        ts = int(self.times[i])
        j = int(np.argmin(np.abs(self._scan_t - ts)))
        T_ego_world = np.linalg.inv(self.ego_pose(i))
        ego = transform(T_ego_world, self._scan_all[j])
        o = transform(T_ego_world, self._scan_origin[j][None])[0]
        return ray_visibility(ego[:, :2], o[:2], self.grid) & camera_fov(self.grid, self.camera)
