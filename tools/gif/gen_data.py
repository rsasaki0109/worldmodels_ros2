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


def gen_hero():
    """Hero GIF: a *runtime anomaly monitor*. Real LeRobot SO-101 robot-camera
    footage runs through the real V-JEPA 2 encoder; the AnomalyDetector
    self-calibrates on nominal operation and flags an unexpected event (here the
    camera is briefly occluded — an object passes in front of the lens). Follows
    World-Model failure/OOD-monitoring research (no failure data needed).

    Needs network (Hugging Face LeRobot dataset), ffmpeg, and a GPU.
    """
    import base64
    import glob
    import io
    import os
    import subprocess
    import tempfile
    import urllib.request
    from PIL import Image

    from world_model_py.anomaly import AnomalyDetector

    base = "https://huggingface.co/datasets/lerobot/svla_so101_pickplace/resolve/main/videos"
    SIDE = base + "/observation.images.side/chunk-000/file-000.mp4"
    N, EV0, EV1 = 24, 14, 20                  # frames; occlusion event window

    tmp = tempfile.mkdtemp()
    side = os.path.join(tmp, "side.mp4")
    urllib.request.urlretrieve(SIDE, side)
    d = os.path.join(tmp, "fr")
    os.makedirs(d, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-ss", "6", "-t", "6.4", "-i", side,
         "-vf", "fps=4,crop=480:480:80:0,scale=256:256", os.path.join(d, "f%03d.png")],
        check=True, capture_output=True,
    )
    files = sorted(glob.glob(os.path.join(d, "*.png")))[:N]
    frames = [np.asarray(Image.open(f).convert("RGB")).astype(np.uint8) for f in files]

    # occlude the event window: a dark soft blob sweeps across (object near lens)
    yy, xx = np.mgrid[0:256, 0:256]
    for i in range(EV0, EV1):
        prog = (i - EV0) / max(1, (EV1 - 1 - EV0))
        cx = int(40 + prog * 200)
        a = np.clip(1.25 - (((xx - cx) / 120.0) ** 2 + ((yy - 128) / 165.0) ** 2), 0, 1) * 0.93
        frames[i] = (frames[i] * (1 - a[..., None]) + np.array([16, 16, 20]) * a[..., None]).astype(np.uint8)

    dev = os.environ.get("WM_HERO_DEVICE", "cuda")   # set WM_HERO_DEVICE=cpu if the GPU is busy
    clip_len = int(os.environ.get("WM_HERO_CLIPLEN", "16"))   # smaller = much faster on CPU
    prime = int(os.environ.get("WM_HERO_PRIME", "16"))
    dtype = "float16" if dev == "cuda" else "float32"
    wm = load_model("vjepa2", entry="vjepa2_vit_large", device=dev, dtype=dtype, clip_len=clip_len)
    for _ in range(prime):
        wm.predict_future(Observation(image=frames[0]), horizon=1)
    wm.reset()
    det = AnomalyDetector(window=10, k=4.0, warmup=4, floor=0.02)

    out = []
    for f in frames:
        s = float(wm.predict_future(Observation(image=f), horizon=1).risk)
        r = det.update(s)
        buf = io.BytesIO()
        Image.fromarray(f).resize((200, 200), Image.BILINEAR).save(buf, "JPEG", quality=82)
        out.append({"img": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(),
                    "surprise": round(s, 4), "threshold": round(r["threshold"], 4),
                    "anomaly": bool(r["anomaly"]), "latched": bool(det.latched)})
    return {"frames": out, "event": [EV0, EV1]}


GENERATORS = {
    "imagination": gen_imagination,
    "nav2": gen_nav2,
    "ijepa": gen_ijepa,
    "compare": gen_compare,
    "hero": gen_hero,
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
