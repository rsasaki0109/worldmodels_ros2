"""Train + evaluate a latent -> future occupancy head (NOT in CI; GPU optional).

Generates seeded episodes with ``world_model_py.occupancy_sim`` (camera frames +
exact future occupancy), encodes every frame with the chosen encoder, trains an
:class:`~world_model_py.occupancy_head.OccupancyHead` and reports pooled IoU on
held-out episodes against baselines:

  * ``prior``         -- per-cell occupancy frequency from training (no image),
  * ``persist-gt``    -- the *ground-truth* current occupancy held constant
                         (privileged: it sees the labels, not the camera),
  * ``pixels``        -- the same head on downsampled raw pixels, which tells
                         whether the learned encoder actually helps.

    # GPU-free smoke run (pixels encoder, CPU torch):
    python3 world_model_py/test/train_occupancy_head.py --encoder pixels --out /tmp/occ
    # I-JEPA on a GPU (e.g. notebooks/train_occupancy_head.ipynb on Colab):
    WM_DEVICE=cuda python3 world_model_py/test/train_occupancy_head.py --encoder ijepa --out occ_out

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
    SimConfig, current_occupancy, future_occupancy, rollout_episode,
)

TRAIN_SEED, VAL_SEED, TEST_SEED = 0, 100_000, 200_000


def log(msg: str) -> None:
    print(msg, flush=True)


def make_encoder(name: str, device: str, model_id: str):
    if name == "pixels":
        from world_model_py.adapters.jepa import PixelEncoder
        return PixelEncoder()
    if name == "ijepa":
        from world_model_py.adapters.jepa import _HFEncoder
        return _HFEncoder(model_id=model_id, device=device)
    raise SystemExit(f"unknown encoder '{name}' (pixels | ijepa)")


def build_split(n_episodes, seed0, steps, horizon, cfg, encoders, batch, keep_frames=0):
    """Simulate + encode a split. Returns per-encoder latents and shared labels."""
    lat = {k: [] for k in encoders}
    prev = {k: [] for k in encoders}
    egos, targets, current, frames = [], [], [], []
    for e in range(n_episodes):
        ep = rollout_episode(steps, seed=seed0 + e, cfg=cfg)
        ts = list(range(1, len(ep) - horizon))
        for k, enc in encoders.items():
            z = np.concatenate([enc.embed_batch(list(ep.frames[s:s + batch]))
                                for s in range(0, len(ep), batch)])
            lat[k].append(z[ts])
            prev[k].append(z[[t - 1 for t in ts]])
        egos.append(ep.actions[[t - 1 for t in ts]])
        targets.append(np.stack([future_occupancy(ep, t, horizon, cfg) for t in ts]))
        current.append(np.stack([current_occupancy(ep, t, cfg) for t in ts]))
        if e < keep_frames:
            frames.append(ep.frames[ts])
        if (e + 1) % 10 == 0 or e + 1 == n_episodes:
            log(f"    {e + 1}/{n_episodes} episodes")
    out = {
        "lat": {k: np.concatenate(v) for k, v in lat.items()},
        "prev": {k: np.concatenate(v) for k, v in prev.items()},
        "ego": np.concatenate(egos),
        "target": np.concatenate(targets),
        "current": np.concatenate(current),
    }
    if frames:
        out["frames"] = np.concatenate(frames)
    return out


def per_step_iou(pred, ref) -> list:
    return [pooled_iou(pred[:, k], ref[:, k]) for k in range(ref.shape[1])]


def bev(grid: np.ndarray) -> np.ndarray:
    """(y, x) grid -> display (forward up, left on the left)."""
    return grid.T[::-1, ::-1]


def save_preview(path, frames, target, proba, threshold, n=4, steps=(0, 3, 7)):
    """camera | (GT vs pred) at a few horizon steps: white = hit, green = missed,
    red = false alarm."""
    try:
        from PIL import Image
    except ImportError:
        log("  (PIL missing; skipping preview)")
        return
    s = frames.shape[1]
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    rows = []
    for i in idx:
        tiles = [frames[i]]
        for k in steps:
            if k >= target.shape[1]:
                continue
            g, p = bev(target[i, k]), bev(proba[i, k] >= threshold)
            tile = np.zeros(g.shape + (3,), np.uint8) + 25
            tile[g & p] = (255, 255, 255)
            tile[g & ~p] = (40, 200, 70)
            tile[~g & p] = (230, 60, 60)
            rep = s // g.shape[0]
            tiles.append(np.kron(tile, np.ones((rep, rep, 1), np.uint8)))
        rows.append(np.concatenate([np.pad(t, ((2, 2), (2, 2), (0, 0))) for t in tiles], 1))
    Image.fromarray(np.concatenate(rows, 0)).save(path)
    log(f"  wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--encoder", default="pixels", choices=["pixels", "ijepa"])
    ap.add_argument("--model-id", default="facebook/ijepa_vith14_1k")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--val-episodes", type=int, default=10)
    ap.add_argument("--test-episodes", type=int, default=15)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--components", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32, help="encoder batch size")
    ap.add_argument("--no-pixel-baseline", action="store_true")
    ap.add_argument("--device", default=os.environ.get("WM_DEVICE", ""))
    ap.add_argument("--out", default="occ_out")
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    cfg = SimConfig()

    names = [args.encoder]
    if args.encoder != "pixels" and not args.no_pixel_baseline:
        names.append("pixels")
    log(f"device={device} encoders={names} horizon={args.horizon} "
        f"grid={cfg.grid_size}x{cfg.grid_size}@{cfg.resolution}m dt={cfg.dt}s")
    t0 = time.time()
    encoders = {k: make_encoder(k, device, args.model_id) for k in names}
    log(f"loaded encoders in {time.time() - t0:.1f}s")

    data = {}
    for split, n, seed in (("train", args.episodes, TRAIN_SEED),
                           ("val", args.val_episodes, VAL_SEED),
                           ("test", args.test_episodes, TEST_SEED)):
        log(f"[{split}] simulating + encoding {n} episodes x {args.steps} steps")
        t0 = time.time()
        data[split] = build_split(n, seed, args.steps, args.horizon, cfg, encoders,
                                  args.batch, keep_frames=3 if split == "test" else 0)
        log(f"  {len(data[split]['ego'])} samples in {time.time() - t0:.1f}s")

    tr, va, te = data["train"], data["val"], data["test"]
    results = {"config": {**vars(args), "device": device, "grid": cfg.grid_size,
                          "resolution": cfg.resolution, "dt": cfg.dt,
                          "occupancy_rate": float(tr["target"].mean())}}
    methods = {}

    # baseline: per-cell prior (threshold picked on val, like the heads)
    prior = tr["target"].mean(0)
    best = max(np.linspace(0.05, 0.95, 19),
               key=lambda t: pooled_iou(np.broadcast_to(prior >= t, va["target"].shape), va["target"]))
    pred = np.broadcast_to(prior >= best, te["target"].shape)
    methods["prior"] = per_step_iou(pred, te["target"])

    # privileged reference: ground-truth current occupancy held constant
    pred = np.repeat(te["current"][:, None], args.horizon, axis=1)
    methods["persist-gt"] = per_step_iou(pred, te["target"])

    previews = {}
    for name in names:
        log(f"[fit] {name} head (latent dim {tr['lat'][name].shape[1]})")
        t0 = time.time()
        head = OccupancyHead.fit(
            tr["lat"][name], tr["prev"][name], tr["ego"], tr["target"],
            resolution=cfg.resolution, origin=cfg.origin, dt=cfg.dt,
            n_components=args.components, hidden=args.hidden, epochs=args.epochs,
            device=device, log=log)
        thr = head.calibrate(va["lat"][name], va["prev"][name], va["ego"], va["target"])
        log(f"  trained in {time.time() - t0:.1f}s, threshold {thr:.2f}")
        proba = head.predict_proba_batch(te["lat"][name], te["prev"][name], te["ego"])
        methods[name] = per_step_iou(proba >= thr, te["target"])
        path = os.path.join(args.out, f"occupancy_head_{name}.npz")
        head.save(path)
        log(f"  wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
        previews[name] = (proba, thr)

    results["iou_per_step"] = methods
    results["iou_mean"] = {k: float(np.mean(v)) for k, v in methods.items()}
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    log("\npooled IoU on held-out episodes (higher is better)")
    hdr = "method".ljust(12) + "".join(f"  t+{k + 1:<3d}" for k in range(args.horizon)) + "   mean"
    log(hdr)
    for k, v in methods.items():
        log(k.ljust(12) + "".join(f"  {x:.3f}" for x in v) + f"  {np.mean(v):.3f}")

    if "frames" in te:
        n = len(te["frames"])
        proba, thr = previews[args.encoder]
        save_preview(os.path.join(args.out, "preview.png"), te["frames"],
                     te["target"][:n], proba[:n], thr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
