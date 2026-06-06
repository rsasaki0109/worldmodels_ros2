"""Quantify the surprise signal on PUBLIC real robot data.

Claim under test: the World Model's surprise (cosine latent distance) is small
between consecutive frames of the *same* scene and large across a scene/view
change — i.e. the anomaly/OOD signal is semantically grounded, not noise.

Uses real LeRobot SO-101 footage (two camera views) through the real I-JEPA
encoder. I-JEPA (per-frame) is used so it runs in minutes on CPU when the GPU is
busy. Needs network + a ROS-free Python env (torch/transformers).

    python3 world_model_py/test/validate_real_surprise.py            # GPU if free
    WM_DEVICE=cpu python3 world_model_py/test/validate_real_surprise.py
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
PAN = 6


def _seg(mp4, ss, tmp, tag):
    d = os.path.join(tmp, tag)
    os.makedirs(d, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-ss", str(ss), "-t", "1.6", "-i", mp4,
                    "-vf", "fps=4,scale=256:256", os.path.join(d, "f%03d.png")],
                   check=True, capture_output=True)
    return [np.asarray(Image.open(f).convert("RGB")).astype(np.uint8)
            for f in sorted(glob.glob(d + "/*.png"))][:PAN]


def main():
    tmp = tempfile.mkdtemp()
    side = os.path.join(tmp, "side.mp4"); urllib.request.urlretrieve(SIDE, side)
    up = os.path.join(tmp, "up.mp4"); urllib.request.urlretrieve(UP, up)
    # scene A: side episode 1 | scene B: up view | scene C: side episode 2 (far apart)
    segs = [_seg(side, 6, tmp, "a"), _seg(up, 10, tmp, "b"), _seg(side, 240, tmp, "c")]
    frames = [f for s in segs for f in s]
    boundaries = {PAN, 2 * PAN}            # indices whose step crosses a scene cut

    dev = os.environ.get("WM_DEVICE", "cpu")
    print(f"loading I-JEPA on {dev} ...")
    wm = load_model("ijepa", model_id="facebook/ijepa_vith14_1k", device=dev,
                    dtype="float16" if dev == "cuda" else "float32")

    within, across = [], []
    prev = None
    for i, f in enumerate(frames):
        s = float(wm.predict_future(Observation(image=f), horizon=1).risk)
        if i > 0:
            (across if i in boundaries else within).append(s)
        prev = s
    wm.reset()

    wmean, wmax = float(np.mean(within)), float(np.max(within))
    amean = float(np.mean(across))
    print("=== surprise on REAL LeRobot SO-101 footage (I-JEPA) ===")
    print(f"within-scene surprise : mean {wmean:.3f}  max {wmax:.3f}  (consecutive same-scene frames)")
    print(f"scene-change surprise : {[round(a,3) for a in across]}  mean {amean:.3f}")
    print(f"separation ratio      : {amean / (wmean + 1e-9):.1f}x  "
          f"(every scene change > every within-scene frame: {min(across) > wmax})")
    ok = min(across) > wmax              # perfectly separable
    print("VALIDATION_OK" if ok else "WEAK_SEPARATION")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
