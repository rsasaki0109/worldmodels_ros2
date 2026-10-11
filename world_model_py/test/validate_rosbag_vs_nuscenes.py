"""Cross-check the rosbag labeller against the nuScenes labeller on real data (NOT in CI).

Converts a nuScenes scene into a plain rosbag2 (``nuscenes_to_rosbag.py``), then
labels it twice: with ``rosbag_occ.BagOccupancy`` (future LiDAR scans, no
annotations) and with ``nuscenes_occ.SceneOccupancy`` (static LiDAR map +
annotated boxes). The two label different things by design, so it checks:

  * exact: the camera model and the ego commands recovered from TF / CameraInfo /
    odometry equal the ones read from nuScenes' own calibration tables;
  * agreement: visibility IoU, and for occupancy the *precision* of the rosbag
    labels against nuScenes' (within 1 cell, on visible cells). A raw scan
    should rarely mark a cell the annotated labels call free. Recall is lower
    by construction, because one sparse scan sees less than an accumulated map.

    python3 world_model_py/test/validate_rosbag_vs_nuscenes.py --nuscenes-root /data/nuscenes
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

import numpy as np

from world_model_py.nuscenes_occ import NuScenesTables, SceneOccupancy
from world_model_py.occupancy_labels import BevGrid
from world_model_py.rosbag_occ import BagOccupancy

sys.path.insert(0, os.path.dirname(__file__))
from nuscenes_to_rosbag import convert  # noqa: E402


def dilate(a: np.ndarray, r: int = 1) -> np.ndarray:
    out = a.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out |= np.roll(np.roll(a, dy, 0), dx, 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nuscenes-root", required=True)
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--scenes", nargs="*", default=["scene-0796", "scene-1094"])
    ap.add_argument("--every", type=int, default=5, help="check every Nth frame")
    args = ap.parse_args()

    tables = NuScenesTables(args.nuscenes_root, args.version)
    grid = BevGrid(48, 48, 0.5, (0.0, -12.0))
    ok = True
    for name in args.scenes:
        sc = SceneOccupancy(tables, name, grid, horizon=4, dt=0.5, image_width=448)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, name)
            convert(tables, name, path)
            bag = BagOccupancy(path, grid, horizon=4, dt=0.5, image_width=448, camera_stride=1)
        # pair frames by timestamp (bag: ns, nuScenes: us)
        pos = {int(t): i for i, t in enumerate(bag.times)}
        pairs = [(i, pos[int(sc.times[i]) * 1000]) for i in sc.valid_indices()[:: args.every]
                 if int(sc.times[i]) * 1000 in pos and pos[int(sc.times[i]) * 1000] in set(bag.valid_indices())]
        cam_same = np.allclose(bag.camera.P, sc.camera_model(0).P, rtol=1e-6, atol=1e-6)
        ego_err = max(float(np.abs(bag.ego_command(b) - sc.ego_command(n)).max()) for n, b in pairs)
        vis_iou, prec, rec = [], [], []
        for n, b in pairs:
            vb, vn = bag.visibility(b), sc.visibility(n)
            vis_iou.append((vb & vn).sum() / max((vb | vn).sum(), 1))
            m = vb & vn
            for k, (ob, on) in enumerate(zip(np.concatenate([bag.occupancy_at(b, 0.0)[None], bag.future_occupancy(b)]),
                                             np.concatenate([sc.occupancy_at(n, 0.0)[None], sc.future_occupancy(n)]))):
                if k not in (0, 4):
                    continue
                pb, pn = ob & m, on & m
                if pb.any():
                    prec.append((k, (pb & dilate(pn)).sum() / pb.sum()))
                if pn.any():
                    rec.append((k, (pn & dilate(pb)).sum() / pn.sum()))
        p0 = np.mean([v for k, v in prec if k == 0]); p4 = np.mean([v for k, v in prec if k == 4])
        r0 = np.mean([v for k, v in rec if k == 0]); r4 = np.mean([v for k, v in rec if k == 4])
        print(f"{name}: {len(pairs)} frames  camera model identical={cam_same}  "
              f"max ego-command diff={ego_err:.2e}  visibility IoU={np.mean(vis_iou):.3f}")
        print(f"   rosbag-label precision vs nuScenes labels (±1 cell): now {p0:.3f}  t+2s {p4:.3f}   "
              f"recall: now {r0:.3f}  t+2s {r4:.3f}")
        ok &= bool(cam_same) and ego_err < 1e-3 and np.mean(vis_iou) > 0.8 and p0 > 0.8
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
