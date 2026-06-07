#!/usr/bin/env python3
"""Render docs/nav2_avoidance.gif from the dummy World Model (GPU-free).

    python3 world_model_nav2/test/render_nav2_avoidance_gif.py
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw

from world_model_py.registry import load_model
from world_model_py.adapters import Observation
from world_model_nav2.occupancy_path import default_candidates, merge_lethal_cells, path_collision_risk

OUT = os.path.join("docs", "nav2_avoidance.gif")
W, H = 520, 300
N = 36


def _world_to_px(x, y):
    px = int(40 + x / 3.2 * (W - 80))
    py = int(H - 40 - y / 3.2 * (H - 80))
    return px, py


def main():
    wm = load_model("dummy")
    obs = Observation(ego_state=np.array([1.0, 0, 0, 0.5], np.float32))
    frames = []
    for f in range(N):
        pred = wm.predict_future(obs, horizon=8)
        # build pseudo-grids like runtime
        class G:
            pass

        grids = []
        for g in pred.occupancy:
            og = G()
            og.info = G()
            og.info.resolution = 0.1
            og.info.width = og.info.height = g.shape[0]
            og.info.origin = G()
            og.info.origin.position = G()
            og.info.origin.position.x = 0.0
            og.info.origin.position.y = 0.0
            og.data = np.asarray(g, np.int16).ravel().tolist()
            grids.append(og)
        lethal = merge_lethal_cells(grids)
        cands = default_candidates()
        scored = [(n, xy, path_collision_risk(xy, lethal)) for n, xy in cands]
        best = min(scored, key=lambda t: t[2])
        naive = next(t for t in scored if t[0] == "straight")

        im = Image.new("RGB", (W, H), (28, 28, 32))
        dr = ImageDraw.Draw(im)
        dr.text((12, 10), "World Model predicted occupancy -> avoid lethal union", fill=(230, 230, 235))
        dr.text((12, 28), "GPU-free dummy adapter · same rule as world_model_costmap layer", fill=(130, 130, 140))

        for lx, ly in lethal:
            px, py = _world_to_px(lx, ly)
            dr.rectangle([px - 3, py - 3, px + 3, py + 3], fill=(200, 60, 40))

        def draw_path(xy, color, width):
            pts = [_world_to_px(x, y) for x, y in xy]
            for i in range(len(pts) - 1):
                dr.line([pts[i], pts[i + 1]], fill=color, width=width)

        draw_path(naive[1], (220, 80, 60), 3)
        draw_path(best[1], (60, 220, 100), 5)
        dr.text((12, H - 24), f"naive {naive[2]:.2f}  ->  avoid {best[2]:.2f} ({best[0]})", fill=(200, 200, 210))
        frames.append(im)
        obs = Observation(ego_state=np.array([1.0, 0, 0.05 * f, 0.5], np.float32))

    os.makedirs("docs", exist_ok=True)
    frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=120, loop=0)
    print(f"wrote {OUT} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
