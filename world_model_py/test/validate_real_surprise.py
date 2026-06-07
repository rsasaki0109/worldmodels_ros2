"""Quantify the surprise signal on PUBLIC real robot data.

Claim under test: the World Model's surprise (cosine latent distance) is small
within a scene and large across a scene/view change — i.e. the anomaly/OOD
signal is semantically grounded, not noise.

Uses real LeRobot SO-101 footage (two camera views) through a real JEPA encoder.
A scene change registers in 1 frame for an *image* encoder (`ijepa`) but over
the whole clip for a *video* encoder (`vjepa2`), so the comparison uses a
``window`` = peak surprise in the frames right after each cut vs the
within-scene baseline. ``ijepa`` is the default (per-frame; runs in minutes on
CPU when the GPU is busy).

    python3 world_model_py/test/validate_real_surprise.py
    WM_ADAPTER=vjepa2 WM_DEVICE=cuda python3 world_model_py/test/validate_real_surprise.py
"""
import glob
import os
import subprocess
import sys
import tempfile
import urllib.request

import numpy as np
from PIL import Image

from world_model_py.registry import load_model
from world_model_py.adapters import Observation

REPO = "https://huggingface.co/datasets/lerobot/svla_so101_pickplace/resolve/main/videos"
SIDE = REPO + "/observation.images.side/chunk-000/file-000.mp4"
UP = REPO + "/observation.images.up/chunk-000/file-000.mp4"
SEG = 10


def _seg(mp4, ss, tmp, tag):
    d = os.path.join(tmp, tag)
    os.makedirs(d, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-ss", str(ss), "-t", "2.6", "-i", mp4,
                    "-vf", "fps=4,scale=256:256", os.path.join(d, "f%03d.png")],
                   check=True, capture_output=True)
    return [np.asarray(Image.open(f).convert("RGB")).astype(np.uint8)
            for f in sorted(glob.glob(d + "/*.png"))][:SEG]


def main():
    tmp = tempfile.mkdtemp()
    side = os.path.join(tmp, "side.mp4"); urllib.request.urlretrieve(SIDE, side)
    up = os.path.join(tmp, "up.mp4"); urllib.request.urlretrieve(UP, up)
    # scene A: side episode 1 | scene B: up view | scene C: side episode 2 (far apart)
    segs = [_seg(side, 6, tmp, "a"), _seg(up, 10, tmp, "b"), _seg(side, 240, tmp, "c")]
    frames = [f for s in segs for f in s]
    boundaries = [SEG, 2 * SEG]

    dev = os.environ.get("WM_DEVICE", "cpu")
    adapter = os.environ.get("WM_ADAPTER", "ijepa")     # ijepa (image) or vjepa2 (video)
    dtype = "float16" if dev == "cuda" else "float32"
    if adapter == "ijepa":
        kw, window = {"model_id": "facebook/ijepa_vith14_1k"}, 1
    else:
        clip = 8
        kw, window = {"entry": "vjepa2_vit_large", "clip_len": clip}, clip
    print(f"loading {adapter} on {dev} (window={window}) ...")
    wm = load_model(adapter, device=dev, dtype=dtype, **kw)
    for _ in range(window):                              # prime the (clip) buffer
        wm.predict_future(Observation(image=frames[0]), horizon=1)
    wm.reset()

    s = [float(wm.predict_future(Observation(image=f), horizon=1).risk) for f in frames]

    trans = set()
    for b in boundaries:
        trans.update(range(b, min(len(s), b + window)))
    peaks = [max(s[b:b + window]) for b in boundaries]
    baseline = [s[i] for i in range(3, len(s)) if i not in trans]
    bmean, bmax = float(np.mean(baseline)), float(np.max(baseline))

    print(f"=== surprise on REAL LeRobot SO-101 footage ({adapter}) ===")
    print(f"within-scene baseline  : mean {bmean:.3f}  max {bmax:.3f}")
    print(f"scene-change peaks      : {[round(p, 3) for p in peaks]}")
    print(f"separation ratio       : {min(peaks) / (bmean + 1e-9):.1f}x  "
          f"(every cut > every within-scene frame: {min(peaks) > bmax})")
    ok = min(peaks) > bmax
    print("VALIDATION_OK" if ok else "WEAK_SEPARATION")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
