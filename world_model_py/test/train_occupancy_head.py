"""Train + evaluate an encoder -> future occupancy head (NOT in CI; GPU optional).

Two data sources share one pipeline:

  * ``--data sim`` -- seeded episodes from ``world_model_py.occupancy_sim``
    (camera frames, exact future occupancy, camera-visibility mask);
  * ``--data nuscenes`` -- real ``CAM_FRONT`` frames with labels built from
    LiDAR + 3D boxes by ``world_model_py.nuscenes_occ`` (static LiDAR map +
    interpolated boxes; LiDAR-ray x camera-FOV visibility). Scenes are split
    train / val / test; labels are cached per scene in ``--cache``.

Every frame is encoded into a spatial feature map, an
:class:`~world_model_py.occupancy_head.OccupancyHead` is trained (loss on visible
cells only) and pooled IoU on held-out data -- over the cells visible at the
observation time, like Occ3D -- is reported against baselines:

  * ``prior``       -- per-cell occupancy frequency from training (no image),
  * ``persist-gt``  -- the *ground-truth* current occupancy held constant
                       (privileged: it reads the labels, not the camera),
  * ``pixels``      -- the same head on downsampled raw pixels, which tells
                       whether the learned encoder actually helps.

    # GPU-free smoke run (pixels encoder, CPU torch):
    python3 world_model_py/test/train_occupancy_head.py --encoder pixels --out /tmp/occ
    # I-JEPA on a GPU (e.g. notebooks/train_occupancy_head.ipynb on Colab):
    WM_DEVICE=cuda python3 world_model_py/test/train_occupancy_head.py --encoder ijepa --out occ_out
    # real data: nuScenes v1.0-mini extracted at $NUSC
    python3 world_model_py/test/train_occupancy_head.py --data nuscenes \
        --nuscenes-root $NUSC --encoder pixels --out occ_nus

Writes ``<out>/occupancy_head_<encoder>.npz`` (load with
``runtime_node --ros-args -p adapter:=ijepa -p occupancy_head:=...``),
``<out>/metrics.json`` and ``<out>/preview.png``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from world_model_py.occupancy_head import OccupancyHead, pooled_iou
from world_model_py.occupancy_sim import (
    SimConfig, current_occupancy, future_occupancy, rollout_episode, visibility,
)

TRAIN_SEED, VAL_SEED, TEST_SEED = 0, 100_000, 200_000
#: nuScenes-mini scene split (test: one day + one night scene)
NUSC_SPLIT = {"val": ["scene-0553"], "test": ["scene-0796", "scene-1094"]}


def log(msg: str) -> None:
    print(msg, flush=True)


def make_encoder(name: str, device: str, model_id: str):
    if name == "pixels":
        from world_model_py.adapters.jepa import PixelEncoder
        return PixelEncoder(size=64)
    if name == "ijepa":
        from world_model_py.adapters.jepa import _HFEncoder
        return _HFEncoder(model_id=model_id, device=device)
    raise SystemExit(f"unknown encoder '{name}' (pixels | ijepa)")


def encode(enc, frames, batch):
    return np.concatenate([enc.embed_map_batch(list(frames[s:s + batch]))
                           for s in range(0, len(frames), batch)])


class SimSource:
    """Seeded synthetic episodes; every frame with a full horizon is a sample."""

    def __init__(self, args):
        self.cfg = cfg = SimConfig()
        self.args = args
        self.camera = cfg.camera()
        self.resolution, self.origin, self.dt = cfg.resolution, cfg.origin, cfg.dt
        self.horizon = args.horizon
        self.heights, self.dilations = (0.05, 0.5, 1.0), (1, 1)
        self.splits = {"train": [TRAIN_SEED + e for e in range(args.episodes)],
                       "val": [VAL_SEED + e for e in range(args.val_episodes)],
                       "test": [TEST_SEED + e for e in range(args.test_episodes)]}

    def describe(self) -> str:
        c = self.cfg
        return (f"sim image={c.image_size}px grid={c.grid_size}x{c.grid_size}@{c.resolution}m "
                f"dt={c.dt}s horizon={self.horizon}")

    def frames(self, key):
        return rollout_episode(self.args.steps, seed=key, cfg=self.cfg).frames

    def episode(self, key) -> dict:
        ep = rollout_episode(self.args.steps, seed=key, cfg=self.cfg)
        ts = np.arange(1, len(ep) - self.horizon)
        return {"frames": ep.frames, "ego": ep.actions[ts - 1], "ts": ts,
                "target": np.stack([future_occupancy(ep, t, self.horizon, self.cfg) for t in ts]),
                "mask": np.stack([visibility(ep, t, self.cfg) for t in ts]),
                "current": np.stack([current_occupancy(ep, t, self.cfg) for t in ts]),
                "static": np.zeros((len(ts), self.cfg.grid_size, self.cfg.grid_size), bool)}


class NuScenesSource:
    """nuScenes scenes; every CAM_FRONT frame with a full horizon is a sample.

    Labels (no images) are cached per scene in ``--cache``, so they can be built
    once on a CPU box (``--labels-only``) and reused where the GPU is: images are
    always read from ``--nuscenes-root``."""

    def __init__(self, args):
        from world_model_py.nuscenes_occ import NuScenesTables
        from world_model_py.occupancy_labels import BevGrid

        self.args = args
        log(f"[nuscenes] loading {args.nuscenes_version} tables")
        self.tables = NuScenesTables(args.nuscenes_root, args.nuscenes_version)
        self.grid = BevGrid(48, 48, 0.5, (0.0, -12.0))          # 24 m ahead, +-12 m
        self.resolution, self.origin, self.dt = self.grid.resolution, self.grid.origin, 0.5
        self.horizon = min(args.horizon, 4)                      # 2 s at 0.5 s
        self.heights, self.dilations = (0.0, 0.75, 1.5), (1, 2, 4)
        names = self.tables.scene_names()
        if args.max_scenes:
            names = names[: args.max_scenes]
        if args.nuscenes_version == "v1.0-mini":
            held = set(NUSC_SPLIT["val"]) | set(NUSC_SPLIT["test"])
            self.splits = {"train": [n for n in names if n not in held],
                           "val": [n for n in NUSC_SPLIT["val"] if n in names],
                           "test": [n for n in NUSC_SPLIT["test"] if n in names]}
        else:   # deterministic by sorted scene name: 1/8 test, 1/8 val
            self.splits = {"train": [n for i, n in enumerate(names) if i % 8 not in (0, 4)],
                           "val": [n for i, n in enumerate(names) if i % 8 == 4],
                           "test": [n for i, n in enumerate(names) if i % 8 == 0]}
        self.cache = args.cache or os.path.join(args.out, "nuscenes_cache")
        os.makedirs(self.cache, exist_ok=True)
        self.camera = self._scene(self.splits["train"][0]).camera_model(0)

    def describe(self) -> str:
        g = self.grid
        return (f"nuscenes {self.args.nuscenes_version} image={self.args.image_width}px "
                f"grid={g.rows}x{g.cols}@{g.resolution}m dt={self.dt}s horizon={self.horizon} "
                f"scenes train/val/test={[len(v) for v in self.splits.values()]}")

    def _scene(self, name):
        from world_model_py.nuscenes_occ import SceneOccupancy
        return SceneOccupancy(self.tables, name, self.grid, horizon=self.horizon, dt=self.dt,
                              image_width=self.args.image_width)

    def labels(self, name, sc=None) -> dict:
        path = os.path.join(self.cache, f"{name}_labels_h{self.horizon}.npz")
        if os.path.exists(path):
            with np.load(path) as z:
                return {k: z[k] for k in z.files}
        sc = sc or self._scene(name)
        ts = sc.valid_indices()
        lab = {"n_cams": np.int64(len(sc.cams)), "ts": ts,
               "ego": np.stack([sc.ego_command(i) for i in ts]),
               "target": np.stack([sc.future_occupancy(i) for i in ts]),
               "mask": np.stack([sc.visibility(i) for i in ts]),
               "current": np.stack([sc.occupancy_at(i, 0.0) for i in ts]),
               "static": np.stack([sc.static_occupancy(i) for i in ts])}
        np.savez_compressed(path, **lab)
        return lab

    def frames(self, name, sc=None):
        from concurrent.futures import ThreadPoolExecutor
        sc = sc or self._scene(name)
        with ThreadPoolExecutor(8) as pool:          # JPEG decode releases the GIL
            return np.stack(list(pool.map(sc.image, range(len(sc.cams)))))

    def episode(self, name) -> dict:
        sc = self._scene(name)
        lab = self.labels(name, sc)
        if int(lab["n_cams"]) != len(sc.cams):
            raise RuntimeError(f"{name}: cached labels were built for {int(lab['n_cams'])} "
                               f"frames but {len(sc.cams)} are on disk")
        return {**lab, "frames": self.frames(name, sc)}


class RosbagSource:
    """Camera + LiDAR rosbags (ROS 1 ``.bag`` or rosbag2): labels from future LiDAR
    scans (``world_model_py.rosbag_occ``). Each bag is cut into ``--segment-s``
    episodes; with >= 4 bags, whole bags are held out for val / test so the same
    place never appears on both sides of the split."""

    def __init__(self, args):
        import glob
        from world_model_py.occupancy_labels import BevGrid
        from world_model_py.rosbag_occ import BagOccupancy

        self.args = args
        paths = []
        for b in args.bags:
            paths += sorted(glob.glob(os.path.join(b, "*.bag"))) if os.path.isdir(b) and not \
                os.path.exists(os.path.join(b, "metadata.yaml")) else [b]
        if not paths:
            raise SystemExit("--data rosbag needs --bags <file.bag | rosbag2 dir | dir of .bag> ...")
        n, res = args.grid_cells, args.grid_res
        self.grid = BevGrid(n, n, res, (0.0, -n * res / 2.0))
        self.resolution, self.origin, self.dt = res, self.grid.origin, 0.5
        self.horizon = min(args.horizon, 4)
        self.heights, self.dilations = (0.0, 0.5, 1.0), (1, 2, 4)
        self.bags = [BagOccupancy(p, self.grid, horizon=self.horizon, dt=self.dt,
                                  image_width=args.image_width, image_topic=args.image_topic,
                                  camera_info_topic=args.camera_info_topic,
                                  lidar_topic=args.lidar_topic, odom_topic=args.odom_topic,
                                  camera_stride=args.camera_stride,
                                  intrinsics=([float(v) for v in args.camera_intrinsics.split(",")]
                                              if args.camera_intrinsics else None),
                                  camera_frame=args.camera_frame, log=log) for p in paths]
        self.camera = self.bags[0].camera
        for b in self.bags[1:]:
            if not np.allclose(b.camera.P, self.camera.P, rtol=0.05, atol=2.0):
                log(f"  WARNING: {b.name} camera differs from {self.bags[0].name}; "
                    "the head stores one camera model")
        # episodes: (bag index, segment) over the valid frames
        self.segments = {}
        for bi, b in enumerate(self.bags):
            vi = b.valid_indices()
            if len(vi) == 0:
                continue
            seg = ((b.times[vi] - b.times[vi[0]]) // int(args.segment_s * 1e9)).astype(int)
            for k in np.unique(seg):
                self.segments[f"{b.name}#{k:03d}"] = (bi, vi[seg == k])
        keys = sorted(self.segments)
        if len(self.bags) >= 4:
            names = sorted({b.name for b in self.bags})
            role = {n: ("test" if i % 4 == 0 else "val" if i % 4 == 2 else "train")
                    for i, n in enumerate(names)}
            pick = lambda r: [k for k in keys if role[k.split("#")[0]] == r]
        else:
            pick = lambda r: [k for i, k in enumerate(keys)
                              if (r == "test") == (i % 8 == 0) and (r == "val") == (i % 8 == 4)]
        self.splits = {r: pick(r) for r in ("train", "val", "test")}
        self.cache = args.cache or os.path.join(args.out, "rosbag_cache")
        os.makedirs(self.cache, exist_ok=True)

    def describe(self) -> str:
        g = self.grid
        return (f"rosbag x{len(self.bags)} image={self.args.image_width}px "
                f"grid={g.rows}x{g.cols}@{g.resolution}m dt={self.dt}s horizon={self.horizon} "
                f"episodes train/val/test={[len(v) for v in self.splits.values()]}")

    def labels(self, key) -> dict:
        g = self.grid
        path = os.path.join(self.cache, f"{key.replace('#', '_')}_g{g.rows}r{g.resolution}"
                                        f"_h{self.horizon}.npz")
        if os.path.exists(path):
            with np.load(path) as z:
                return {k: z[k] for k in z.files}
        bi, idx = self.segments[key]
        b = self.bags[bi]
        lab = {"ts": idx,
               "ego": np.stack([b.ego_command(i) for i in idx]),
               "target": np.stack([b.future_occupancy(i) for i in idx]),
               "mask": np.stack([b.visibility(i) for i in idx]),
               "current": np.stack([b.occupancy_at(i, 0.0) for i in idx]),
               "static": np.stack([b.static_occupancy(i) for i in idx])}
        np.savez_compressed(path, **lab)
        return lab

    def episode(self, key) -> dict:
        bi, _ = self.segments[key]
        lab = self.labels(key)
        # earlier frames reach back max(prev_offsets) (clipped at the bag start;
        # build_split drops samples without enough history)
        lo = max(0, int(lab["ts"].min()) - max(self.args.prev_offsets))
        frames = np.stack(self.bags[bi].frames[lo: int(lab["ts"].max()) + 1])
        return {**lab, "ts": lab["ts"] - lo, "frames": frames}

    def frames(self, key):
        return self.episode(key)["frames"]


def frame_dt(source) -> float:
    """Seconds between consecutive frames the head sees (its prev_offsets unit)."""
    if hasattr(source, "bags"):
        return float(np.median(np.diff(source.bags[0].times)) * 1e-9)
    cfg = getattr(source, "cfg", None)
    return float(getattr(cfg, "dt", 0.0) or 0.0)


def build_split(source, keys, encoders, pcas, batch, keep_frames=0, prev_offsets=(1,), sample_stride=1):
    """Encode a split. Per encoder: reduced maps for t and for t - o, o in
    ``prev_offsets`` (float16, (N, P, ...)); shared: ego command, future
    targets, visibility and current occupancy. Samples without enough history
    are dropped; ``sample_stride`` keeps every Nth sample."""
    z = {k: [] for k in encoders}
    zp = {k: [] for k in encoders}
    egos, targets, masks, current, static, frames = [], [], [], [], [], []
    for e, key in enumerate(keys):
        ep = source.episode(key)
        keep = np.nonzero(ep["ts"] >= max(prev_offsets))[0][::max(1, int(sample_stride))]
        ts = ep["ts"][keep]
        for k, enc in encoders.items():
            mu, comp, std = pcas[k]
            m = ((encode(enc, ep["frames"], batch) - mu) @ comp / std).astype(np.float16)
            z[k].append(m[ts])
            zp[k].append(np.stack([m[ts - o] for o in prev_offsets], 1))
        egos.append(ep["ego"][keep])
        targets.append(ep["target"][keep])
        masks.append(ep["mask"][keep])
        current.append(ep["current"][keep])
        static.append(ep["static"][keep])
        if e < keep_frames:
            frames.append(ep["frames"][ts])
        if (e + 1) % 10 == 0 or e + 1 == len(keys):
            log(f"    {e + 1}/{len(keys)} episodes")
    out = {"z": {k: np.concatenate(v) for k, v in z.items()},
           "zp": {k: np.concatenate(v) for k, v in zp.items()},
           "ego": np.concatenate(egos), "target": np.concatenate(targets),
           "mask": np.concatenate(masks), "current": np.concatenate(current),
           "static": np.concatenate(static)}
    if frames:
        out["frames"] = np.concatenate(frames)
    return out


def per_step_iou(pred, ref, mask=None) -> list:
    return [pooled_iou(pred[:, k], ref[:, k], mask) for k in range(ref.shape[1])]


def bev(grid: np.ndarray) -> np.ndarray:
    """(y, x) grid -> display (forward up, left on the left)."""
    return grid.T[::-1, ::-1]


def save_preview(path, frames, target, mask, proba, threshold, n=4, steps=None):
    """camera | (GT vs pred) at a few horizon steps: white = hit, green = missed,
    red = false alarm; dark blue = not visible at t (not scored), with predictions
    there in dim purple."""
    try:
        from PIL import Image
    except ImportError:
        log("  (PIL missing; skipping preview)")
        return
    h = frames.shape[1]
    hz = target.shape[1]
    steps = steps or sorted({0, hz // 2 - 1 if hz > 2 else 0, hz - 1})
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    rows = []
    for i in idx:
        tiles = [frames[i]]
        for k in steps:
            if k >= target.shape[1]:
                continue
            g, p, m = bev(target[i, k]), bev(proba[i, k] >= threshold), bev(mask[i])
            tile = np.zeros(g.shape + (3,), np.uint8) + 25
            tile[~m] = (20, 30, 70)
            tile[~m & p] = (90, 50, 110)
            tile[m & g & p] = (255, 255, 255)
            tile[m & g & ~p] = (40, 200, 70)
            tile[m & ~g & p] = (230, 60, 60)
            rep = max(1, h // g.shape[0])
            t = np.kron(tile, np.ones((rep, rep, 1), np.uint8))
            tiles.append(np.pad(t, ((0, h - t.shape[0]), (0, 0), (0, 0))))
        rows.append(np.concatenate([np.pad(t, ((2, 2), (2, 2), (0, 0))) for t in tiles], 1))
    Image.fromarray(np.concatenate(rows, 0)).save(path)
    log(f"  wrote {path}")


def predict(head, d, name, batch=16):
    # numpy im2col on a 48x48 grid is ~15 MB per sample per conv: keep batches small
    return np.concatenate([
        head.predict_proba_reduced(d["z"][name][s:s + batch], d["zp"][name][s:s + batch],
                                   d["ego"][s:s + batch])
        for s in range(0, len(d["ego"]), batch)])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="sim", choices=["sim", "nuscenes", "rosbag"])
    ap.add_argument("--bags", nargs="*", default=[],
                    help="rosbag: .bag files, rosbag2 dirs, or dirs of .bag files")
    ap.add_argument("--image-topic", default="")
    ap.add_argument("--camera-info-topic", default="")
    ap.add_argument("--lidar-topic", default="")
    ap.add_argument("--odom-topic", default="")
    ap.add_argument("--camera-intrinsics", default="",
                    help="rosbag without CameraInfo: fx,fy,cx,cy in recorded-image pixels")
    ap.add_argument("--camera-frame", default="", help="rosbag: override the camera TF frame")
    ap.add_argument("--camera-stride", type=int, default=2, help="rosbag: keep every Nth frame")
    ap.add_argument("--prev-offsets", type=lambda v: tuple(int(x) for x in v.split(",")), default=(1,),
                    help="earlier frames the head sees, in frames back (e.g. 1,2,4)")
    ap.add_argument("--sample-stride", type=int, default=1,
                    help="train/score every Nth frame (history still uses every frame)")
    ap.add_argument("--segment-s", type=float, default=20.0, help="rosbag: episode length")
    ap.add_argument("--grid-cells", type=int, default=48, help="rosbag: BEV cells per side")
    ap.add_argument("--grid-res", type=float, default=0.25, help="rosbag: metres per cell")
    ap.add_argument("--nuscenes-root", default=os.environ.get("NUSCENES_ROOT", ""))
    ap.add_argument("--nuscenes-version", default="v1.0-mini")
    ap.add_argument("--image-width", type=int, default=448, help="nuScenes frames are resized")
    ap.add_argument("--cache", default="", help="nuScenes label cache dir")
    ap.add_argument("--max-scenes", type=int, default=0, help="nuScenes: cap scenes (0 = all)")
    ap.add_argument("--labels-only", action="store_true",
                    help="nuScenes: build the label cache and exit (no encoder / GPU)")
    ap.add_argument("--encoder", default="pixels", choices=["pixels", "ijepa"])
    ap.add_argument("--model-id", default="facebook/ijepa_vith14_1k")
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--val-episodes", type=int, default=10)
    ap.add_argument("--test-episodes", type=int, default=20)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--channels", type=int, default=32, help="channel PCA size")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--batch", type=int, default=32, help="encoder batch size")
    ap.add_argument("--no-pixel-baseline", action="store_true")
    ap.add_argument("--device", default=os.environ.get("WM_DEVICE", ""))
    ap.add_argument("--out", default="occ_out")
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    if args.data == "nuscenes" and not args.nuscenes_root:
        raise SystemExit("--data nuscenes needs --nuscenes-root (or NUSCENES_ROOT)")
    source = {"nuscenes": NuScenesSource, "rosbag": RosbagSource}.get(args.data, SimSource)(args)
    args.horizon = source.horizon
    if args.labels_only:
        if args.data == "sim":
            raise SystemExit("--labels-only is for --data nuscenes | rosbag")
        keys = [k for v in source.splits.values() for k in v]
        log(f"[labels] {source.describe()}")
        for e, key in enumerate(keys):
            t0 = time.time()
            lab = source.labels(key)
            log(f"  {e + 1}/{len(keys)} {key}: {len(lab['ts'])} samples in {time.time() - t0:.1f}s")
        return 0

    names = [args.encoder]
    if args.encoder != "pixels" and not args.no_pixel_baseline:
        names.append("pixels")
    log(f"device={device} encoders={names} {source.describe()}")
    t0 = time.time()
    encoders = {k: make_encoder(k, device, args.model_id) for k in names}
    log(f"loaded encoders in {time.time() - t0:.1f}s")

    # channel PCA from a few training episodes (keeps ~1000-d token maps small)
    pcas = {}
    pca_keys = source.splits["train"][:5 if args.data == "sim" else 2]
    if not all(source.splits.values()):
        raise SystemExit(f"empty split: {source.describe()} -- need more data")
    for k, enc in encoders.items():
        sample = np.concatenate([encode(enc, source.frames(key), args.batch) for key in pca_keys])
        pcas[k] = OccupancyHead.fit_channel_pca(sample, channels=args.channels)
        log(f"[pca] {k}: map {sample.shape[1:]} -> {pcas[k][1].shape[1]} channels")

    data = {}
    for split in ("train", "val", "test"):
        keys = source.splits[split]
        log(f"[{split}] labelling + encoding {len(keys)} episodes")
        t0 = time.time()
        data[split] = build_split(source, keys, encoders, pcas, args.batch,
                                  keep_frames=3 if split == "test" else 0,
                                  prev_offsets=args.prev_offsets, sample_stride=args.sample_stride)
        log(f"  {len(data[split]['ego'])} samples in {time.time() - t0:.1f}s")

    tr, va, te = data["train"], data["val"], data["test"]
    results = {"config": {**vars(args), "device": device, "source": source.describe(),
                          "resolution": source.resolution, "dt": source.dt,
                          "occupancy_rate": float(tr["target"].mean()),
                          "visible_rate": float(tr["mask"].mean())}}
    methods, methods_all, methods_agents = {}, {}, {}
    has_static = bool(te["static"].any())

    def score(name, pred):
        """visible-cell IoU, all-cell IoU, and agents-only IoU (static structure
        excluded from both prediction and target, so only movers count)."""
        methods[name] = per_step_iou(pred, te["target"], te["mask"])
        methods_all[name] = per_step_iou(pred, te["target"])
        dyn = ~te["static"][:, None]
        methods_agents[name] = per_step_iou(pred & dyn, te["target"] & dyn, te["mask"])

    # baseline: per-cell prior (threshold picked on val, like the heads)
    prior = tr["target"].mean(0)
    best = max(np.linspace(0.05, 0.95, 19),
               key=lambda t: pooled_iou(np.broadcast_to(prior >= t, va["target"].shape),
                                        va["target"], va["mask"]))
    score("prior", np.broadcast_to(prior >= best, te["target"].shape))

    # privileged reference: ground-truth current occupancy held constant
    score("persist-gt", np.repeat(te["current"][:, None], args.horizon, axis=1))

    previews = {}
    for name in names:
        log(f"[fit] {name} head (map {tr['z'][name].shape[1:]})")
        t0 = time.time()
        head = OccupancyHead.fit(
            tr["z"][name], tr["zp"][name], tr["ego"], tr["target"], tr["mask"],
            camera=source.camera, channel_pca=pcas[name], resolution=source.resolution,
            origin=source.origin, heights=source.heights, dt=source.dt, hidden=args.hidden,
            dilations=source.dilations, epochs=args.epochs, device=device, log=log,
            prev_offsets=args.prev_offsets, frame_dt=frame_dt(source))
        thr = head.calibrate(va["z"][name], va["zp"][name], va["ego"], va["target"], va["mask"])
        log(f"  trained in {time.time() - t0:.1f}s, threshold {thr:.2f}")
        proba = predict(head, te, name)
        score(name, proba >= thr)
        path = os.path.join(args.out, f"occupancy_head_{name}.npz")
        head.save(path)
        log(f"  wrote {path} ({os.path.getsize(path) / 1e6:.2f} MB)")
        previews[name] = (proba, thr)

    results["iou_per_step"] = methods
    results["iou_mean"] = {k: float(np.mean(v)) for k, v in methods.items()}
    results["iou_mean_all_cells"] = {k: float(np.mean(v)) for k, v in methods_all.items()}
    if has_static:
        results["agents_iou_per_step"] = methods_agents
        results["agents_iou_mean"] = {k: float(np.mean(v)) for k, v in methods_agents.items()}
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    log(f"\npooled IoU on held-out {args.data} data, cells visible at t (higher is better)")
    log("method".ljust(12) + "".join(f"  t+{k + 1:<3d}" for k in range(args.horizon))
        + "   mean   (all cells)")
    for k, v in methods.items():
        log(k.ljust(12) + "".join(f"  {x:.3f}" for x in v)
            + f"  {np.mean(v):.3f}   ({np.mean(methods_all[k]):.3f})")
    if has_static:
        log("\nagents-only IoU (static structure excluded; visible cells)")
        for k, v in methods_agents.items():
            log(k.ljust(12) + "".join(f"  {x:.3f}" for x in v) + f"  {np.mean(v):.3f}")

    if "frames" in te:
        n = len(te["frames"])
        proba, thr = previews[args.encoder]
        save_preview(os.path.join(args.out, "preview.png"), te["frames"],
                     te["target"][:n], te["mask"][:n], proba[:n], thr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
