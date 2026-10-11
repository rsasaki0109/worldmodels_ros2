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
                "current": np.stack([current_occupancy(ep, t, self.cfg) for t in ts])}


class NuScenesSource:
    """nuScenes scenes; every CAM_FRONT frame with a full horizon is a sample."""

    def __init__(self, args):
        from world_model_py.nuscenes_occ import NuScenesTables
        from world_model_py.occupancy_labels import BevGrid

        self.args = args
        self.tables = NuScenesTables(args.nuscenes_root, args.nuscenes_version)
        self.grid = BevGrid(48, 48, 0.5, (0.0, -12.0))          # 24 m ahead, +-12 m
        self.resolution, self.origin, self.dt = self.grid.resolution, self.grid.origin, 0.5
        self.horizon = min(args.horizon, 4)                      # 2 s at 0.5 s
        self.heights, self.dilations = (0.0, 0.75, 1.5), (1, 2, 4)
        names = self.tables.scene_names()
        held = set(NUSC_SPLIT["val"]) | set(NUSC_SPLIT["test"])
        self.splits = {"train": [n for n in names if n not in held],
                       "val": [n for n in NUSC_SPLIT["val"] if n in names],
                       "test": [n for n in NUSC_SPLIT["test"] if n in names]}
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

    def episode(self, name) -> dict:
        path = os.path.join(self.cache, f"{name}_w{self.args.image_width}_h{self.horizon}.npz")
        if os.path.exists(path):
            with np.load(path) as z:
                return {k: z[k] for k in z.files}
        sc = self._scene(name)
        ts = sc.valid_indices()
        ep = {"frames": np.stack([sc.image(i) for i in range(len(sc.cams))]),
              "ego": np.stack([sc.ego_command(i) for i in ts]), "ts": ts,
              "target": np.stack([sc.future_occupancy(i) for i in ts]),
              "mask": np.stack([sc.visibility(i) for i in ts]),
              "current": np.stack([sc.occupancy_at(i, 0.0) for i in ts])}
        np.savez(path, **ep)
        return ep

    def frames(self, key):
        return self.episode(key)["frames"]


def build_split(source, keys, encoders, pcas, batch, keep_frames=0):
    """Encode a split. Per encoder: reduced maps for t and t-1 (float16);
    shared: ego command, future targets, visibility and current occupancy."""
    z = {k: [] for k in encoders}
    zp = {k: [] for k in encoders}
    egos, targets, masks, current, frames = [], [], [], [], []
    for e, key in enumerate(keys):
        ep = source.episode(key)
        ts = ep["ts"]
        for k, enc in encoders.items():
            mu, comp, std = pcas[k]
            m = ((encode(enc, ep["frames"], batch) - mu) @ comp / std).astype(np.float16)
            z[k].append(m[ts])
            zp[k].append(m[ts - 1])
        egos.append(ep["ego"])
        targets.append(ep["target"])
        masks.append(ep["mask"])
        current.append(ep["current"])
        if e < keep_frames:
            frames.append(ep["frames"][ts])
        if (e + 1) % 10 == 0 or e + 1 == len(keys):
            log(f"    {e + 1}/{len(keys)} episodes")
    out = {"z": {k: np.concatenate(v) for k, v in z.items()},
           "zp": {k: np.concatenate(v) for k, v in zp.items()},
           "ego": np.concatenate(egos), "target": np.concatenate(targets),
           "mask": np.concatenate(masks), "current": np.concatenate(current)}
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


def predict(head, d, name, batch=256):
    return np.concatenate([
        head.predict_proba_reduced(d["z"][name][s:s + batch], d["zp"][name][s:s + batch],
                                   d["ego"][s:s + batch])
        for s in range(0, len(d["ego"]), batch)])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="sim", choices=["sim", "nuscenes"])
    ap.add_argument("--nuscenes-root", default=os.environ.get("NUSCENES_ROOT", ""))
    ap.add_argument("--nuscenes-version", default="v1.0-mini")
    ap.add_argument("--image-width", type=int, default=448, help="nuScenes frames are resized")
    ap.add_argument("--cache", default="", help="nuScenes label cache dir")
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
    source = NuScenesSource(args) if args.data == "nuscenes" else SimSource(args)
    args.horizon = source.horizon

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
                                  keep_frames=3 if split == "test" else 0)
        log(f"  {len(data[split]['ego'])} samples in {time.time() - t0:.1f}s")

    tr, va, te = data["train"], data["val"], data["test"]
    results = {"config": {**vars(args), "device": device, "source": source.describe(),
                          "resolution": source.resolution, "dt": source.dt,
                          "occupancy_rate": float(tr["target"].mean()),
                          "visible_rate": float(tr["mask"].mean())}}
    methods, methods_all = {}, {}

    # baseline: per-cell prior (threshold picked on val, like the heads)
    prior = tr["target"].mean(0)
    best = max(np.linspace(0.05, 0.95, 19),
               key=lambda t: pooled_iou(np.broadcast_to(prior >= t, va["target"].shape),
                                        va["target"], va["mask"]))
    pred = np.broadcast_to(prior >= best, te["target"].shape)
    methods["prior"] = per_step_iou(pred, te["target"], te["mask"])
    methods_all["prior"] = per_step_iou(pred, te["target"])

    # privileged reference: ground-truth current occupancy held constant
    pred = np.repeat(te["current"][:, None], args.horizon, axis=1)
    methods["persist-gt"] = per_step_iou(pred, te["target"], te["mask"])
    methods_all["persist-gt"] = per_step_iou(pred, te["target"])

    previews = {}
    for name in names:
        log(f"[fit] {name} head (map {tr['z'][name].shape[1:]})")
        t0 = time.time()
        head = OccupancyHead.fit(
            tr["z"][name], tr["zp"][name], tr["ego"], tr["target"], tr["mask"],
            camera=source.camera, channel_pca=pcas[name], resolution=source.resolution,
            origin=source.origin, heights=source.heights, dt=source.dt, hidden=args.hidden,
            dilations=source.dilations, epochs=args.epochs, device=device, log=log)
        thr = head.calibrate(va["z"][name], va["zp"][name], va["ego"], va["target"], va["mask"])
        log(f"  trained in {time.time() - t0:.1f}s, threshold {thr:.2f}")
        proba = predict(head, te, name)
        methods[name] = per_step_iou(proba >= thr, te["target"], te["mask"])
        methods_all[name] = per_step_iou(proba >= thr, te["target"])
        path = os.path.join(args.out, f"occupancy_head_{name}.npz")
        head.save(path)
        log(f"  wrote {path} ({os.path.getsize(path) / 1e6:.2f} MB)")
        previews[name] = (proba, thr)

    results["iou_per_step"] = methods
    results["iou_mean"] = {k: float(np.mean(v)) for k, v in methods.items()}
    results["iou_mean_all_cells"] = {k: float(np.mean(v)) for k, v in methods_all.items()}
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    log(f"\npooled IoU on held-out {args.data} data, cells visible at t (higher is better)")
    log("method".ljust(12) + "".join(f"  t+{k + 1:<3d}" for k in range(args.horizon))
        + "   mean   (all cells)")
    for k, v in methods.items():
        log(k.ljust(12) + "".join(f"  {x:.3f}" for x in v)
            + f"  {np.mean(v):.3f}   ({np.mean(methods_all[k]):.3f})")

    if "frames" in te:
        n = len(te["frames"])
        proba, thr = previews[args.encoder]
        save_preview(os.path.join(args.out, "preview.png"), te["frames"],
                     te["target"][:n], te["mask"][:n], proba[:n], thr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
