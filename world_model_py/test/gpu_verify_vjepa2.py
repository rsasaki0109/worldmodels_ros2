"""Manual GPU verification of the real V-JEPA 2 video adapter (NOT in CI).

Loads V-JEPA 2 (ViT-L) via torch.hub, streams a scene-change frame sequence
through the rolling-clip encoder, and prints per-step surprise (cosine distance
between successive clip latents). Surprise should jump on scene cuts.

    python3 world_model_py/test/gpu_verify_vjepa2.py
"""
import sys

import numpy as np

from world_model_py.registry import load_model
from world_model_py.adapters.base import Observation


def _gradient():
    yy, xx = np.mgrid[0:256, 0:256]
    return np.stack([xx / 256 * 180 + 40, yy / 256 * 120 + 30, np.full((256, 256), 90)], 2).astype(np.uint8)


def _box(base, cx, color):
    img = base.copy(); img[96:160, max(0, cx - 32):cx + 32] = color; return img


def main():
    print("loading V-JEPA 2 (downloads ViT-L weights on first run)...")
    wm = load_model("vjepa2", entry="vjepa2_vit_large", device="cuda", dtype="float16", clip_len=16)
    print("info:", wm.info())

    rng = np.random.default_rng(3)
    frames = []
    g = _gradient()
    for k in range(6):
        frames.append(_box(g, 40 + k * 30, [220, 40, 40]))          # smooth motion
    noise = (rng.random((256, 256, 3)) * 255).astype(np.uint8)
    for _ in range(6):
        frames.append(noise.copy())                                 # scene cut -> spike
    for k in range(6):
        frames.append(_box(_gradient(), 200 - k * 30, [40, 90, 230]))  # cut back

    surprise = []
    for f in frames:
        surprise.append(round(float(wm.predict_future(Observation(image=f), horizon=1).risk), 4))
    print("surprise:", surprise)
    spikes = [i for i, s in enumerate(surprise) if s > 0.1]
    print("spikes (>0.1):", spikes)
    ok = len(surprise) == len(frames) and any(s > 0.1 for s in surprise[6:9])
    print("VJEPA2_OK" if ok else "VJEPA2_WEAK_SIGNAL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
