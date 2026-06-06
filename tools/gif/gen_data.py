"""Generate the data.json that drives each README GIF, from the real pipeline.

    python3 gen_data.py imagination --out imagination/data.json
    python3 gen_data.py nav2        --out nav2/data.json
    python3 gen_data.py ijepa       --out ijepa/data.json     # needs GPU + weights

Run with ROS 2 sourced (so world_model_py / world_model_nav2 import). The
imagination and nav2 datasets are GPU-free (dummy adapter); ijepa runs the real
I-JEPA model on a GPU. Outputs are consumed by the matching render_*.html.
"""
import argparse
import base64
import io
import json

import numpy as np

from world_model_py.registry import load_model
from world_model_py.adapters import Observation, ActionCondition


def gen_imagination():
    wm = load_model("dummy", latent_dim=64, grid_size=32)
    obs = Observation(ego_state=np.array([1.0, 0, 0, 0.5], np.float32), instruction="move forward")
    pred = wm.predict_future(obs, horizon=8)
    layers = [np.asarray(g, np.int16).tolist() for g in pred.occupancy]
    N = 32
    frames = []
    for f in range(N):
        phase = f / N
        mag = 1.0 - np.cos(2 * np.pi * phase)
        act = ActionCondition(action=np.full((6, 2), mag * 0.7, np.float32), dt=0.1)
        frames.append({"step": phase * 8.0, "risk": float(wm.score_trajectory(obs, act))})
    return {"grid_size": 32, "horizon": 8, "layers": layers, "frames": frames}


def gen_nav2():
    from world_model_nav2.path_to_action import path_to_action

    wm = load_model("dummy")
    obs = Observation(ego_state=np.zeros(4, np.float32))
    L, n = 3.0, 12
    cands = []
    for k in np.linspace(-0.75, 0.75, 7):
        s = np.linspace(0, L, n)
        if abs(k) < 1e-6:
            x, y, yaw = s, np.zeros_like(s), np.zeros_like(s)
        else:
            th = k * s
            x, y, yaw = np.sin(th) / k, (1 - np.cos(th)) / k, th
        poses = np.stack([x, y, yaw], 1).astype(np.float32)
        risk = float(wm.score_trajectory(obs, ActionCondition(action=path_to_action(poses), dt=0.1)))
        cands.append({"pts": np.stack([x, y], 1).round(4).tolist(), "risk": round(risk, 4)})
    best = int(np.argmin([c["risk"] for c in cands]))
    return {"candidates": cands, "best": best, "world": {"L": L}}


def gen_ijepa():
    from PIL import Image

    def gradient():
        yy, xx = np.mgrid[0:256, 0:256]
        return np.stack([xx / 256 * 180 + 40, yy / 256 * 120 + 30, np.full((256, 256), 90)], 2).astype(np.uint8)

    def box(base, cx, s, color):
        img = base.copy(); img[128 - s:128 + s, max(0, cx - s):cx + s] = color; return img

    def circle(base, cx, color):
        img = base.copy(); yy, xx = np.mgrid[0:256, 0:256]
        img[(xx - cx) ** 2 + (yy - 130) ** 2 <= 900] = color; return img

    rng = np.random.default_rng(3)
    frames_img = []
    g = gradient()
    for k in range(8):
        frames_img.append(box(g, 40 + k * 26, 34, [220, 40, 40]))
    noise = (rng.random((256, 256, 3)) * 255).astype(np.uint8)
    for k in range(5):
        frames_img.append(circle(noise, 80 + k * 22, [40, 220, 90]))
    for k in range(5):
        frames_img.append(box(gradient(), 200 - k * 26, 34, [40, 90, 230]))

    wm = load_model("ijepa", model_id="facebook/ijepa_vith14_1k", device="cuda", dtype="float16")
    out = []
    for f in frames_img:
        pred = wm.predict_future(Observation(image=f), horizon=1)
        buf = io.BytesIO()
        Image.fromarray(f).resize((128, 128), Image.BILINEAR).save(buf, "PNG")
        out.append({
            "img": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
            "surprise": round(float(pred.risk), 4),
            "conf": round(float(pred.risk_confidence), 3),
        })
    return {"frames": out}


def _scene_change_frames():
    """Smooth motion -> scene cut -> different scene -> cut back (256x256 RGB)."""
    def gradient():
        yy, xx = np.mgrid[0:256, 0:256]
        return np.stack([xx / 256 * 180 + 40, yy / 256 * 120 + 30, np.full((256, 256), 90)], 2).astype(np.uint8)

    def box(base, cx, s, color):
        img = base.copy(); img[128 - s:128 + s, max(0, cx - s):cx + s] = color; return img

    def circle(base, cx, color):
        img = base.copy(); yy, xx = np.mgrid[0:256, 0:256]
        img[(xx - cx) ** 2 + (yy - 130) ** 2 <= 900] = color; return img

    rng = np.random.default_rng(3)
    frames = []
    g = gradient()
    for k in range(8):
        frames.append(box(g, 40 + k * 26, 34, [220, 40, 40]))
    noise = (rng.random((256, 256, 3)) * 255).astype(np.uint8)
    for k in range(5):
        frames.append(circle(noise, 80 + k * 22, [40, 220, 90]))
    for k in range(5):
        frames.append(box(gradient(), 200 - k * 26, 34, [40, 90, 230]))
    return frames


def gen_compare():
    """Run BOTH real JEPA encoders on the same sequence: image (I-JEPA) vs
    video (V-JEPA 2) surprise. Needs a GPU + both sets of weights."""
    import base64
    import io
    from PIL import Image

    frames_img = _scene_change_frames()
    ijepa = load_model("ijepa", model_id="facebook/ijepa_vith14_1k", device="cuda", dtype="float16")
    vjepa2 = load_model("vjepa2", entry="vjepa2_vit_large", device="cuda", dtype="float16", clip_len=16)

    out = []
    for f in frames_img:
        obs = Observation(image=f)
        ij = float(ijepa.predict_future(obs, horizon=1).risk)
        vj = float(vjepa2.predict_future(obs, horizon=1).risk)
        buf = io.BytesIO()
        Image.fromarray(f).resize((128, 128), Image.BILINEAR).save(buf, "PNG")
        out.append({
            "img": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
            "ijepa": round(ij, 4),
            "vjepa2": round(vj, 4),
        })
    return {"frames": out}


GENERATORS = {
    "imagination": gen_imagination,
    "nav2": gen_nav2,
    "ijepa": gen_ijepa,
    "compare": gen_compare,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=sorted(GENERATORS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = GENERATORS[args.kind]()
    with open(args.out, "w") as fh:
        json.dump(data, fh)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
