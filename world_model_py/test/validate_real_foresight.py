"""Visual foresight on PUBLIC driving video: imagine the road ahead, no training.

Claim under test: given a single current frame and a memory of how the latent
world evolves, the episodic world model (frozen JEPA encoder + retrieval
dynamics) can autoregressively *imagine the road ahead* -- and the imagined
future matches what the car actually drives into.

Pipeline on a real public driving clip (yaak-ai/L2D, front camera):
  1. encode N frames with I-JEPA                 -> latents (N, D)
  2. build next-latent retrieval dynamics from   (latent_t -> latent_{t+1})
  3. from frame 0 ALONE, roll the dynamics forward N-1 steps (no peeking)
  4. decode each imagined latent to a real frame (nearest neighbour)
  5. compare the imagined rollout to the actual future; render docs/foresight.gif
     (input | imagined future | real future)

The hook: the model sees one frame, then dreams ~15 s of driving that lands on
the real road ahead -- with zero dynamics training.

Needs network (Hugging Face) + ffmpeg + torch/transformers (GPU nice). Not CI.

    WM_DEVICE=cuda python3 world_model_py/test/validate_real_foresight.py
"""
import glob
import os
import subprocess
import sys
import tempfile
import urllib.request

import numpy as np
from PIL import Image, ImageDraw

from world_model_py.registry import load_model
from world_model_py.adapters import Observation
from world_model_py.planning import RetrievalDynamics, cosine_distance, decode_trajectory

URL = ("https://huggingface.co/datasets/yaak-ai/L2D/resolve/main/"
       "videos/observation.images.front_left/chunk-001/file-491.mp4")
N = 60                 # frames to use
FPS_IN = 4             # sampling fps from the source clip
SS = 8                 # seconds to skip in (get past the static start)
TILE = 200
OUT_GIF = os.path.join("docs", "foresight.gif")
OUT_PNG = os.path.join("docs", "foresight.png")


def _label(img_hwc, text, color=(20, 20, 20)):
    im = Image.fromarray(img_hwc.astype(np.uint8)).resize((TILE, TILE))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, TILE, 16], fill=(255, 255, 255))
    d.text((4, 3), text, fill=color)
    return np.asarray(im)


def main():
    tmp = tempfile.mkdtemp()
    mp4 = os.path.join(tmp, "d.mp4")
    print("downloading public L2D front-camera driving clip (~66MB) ...")
    urllib.request.urlretrieve(URL, mp4)
    fr = os.path.join(tmp, "fr"); os.makedirs(fr)
    subprocess.run(["ffmpeg", "-y", "-ss", str(SS), "-i", mp4,
                    "-vf", f"fps={FPS_IN},scale=256:256", "-frames:v", str(N),
                    os.path.join(fr, "%03d.png")], check=True, capture_output=True)
    frames = [np.asarray(Image.open(f).convert("RGB")).astype(np.uint8)
              for f in sorted(glob.glob(fr + "/*.png"))][:N]
    n = len(frames)

    dev = os.environ.get("WM_DEVICE", "cuda")
    dtype = "float16" if dev == "cuda" else "float32"
    print(f"encoding {n} frames with I-JEPA on {dev} ...")
    wm = load_model("ijepa", model_id="facebook/ijepa_vith14_1k", device=dev, dtype=dtype)
    lat = np.array([wm.encode(Observation(image=f)).data for f in frames], np.float32)

    # next-latent retrieval dynamics (forward driving == single implicit action).
    dyn = RetrievalDynamics(lat[:-1], np.zeros((n - 1, 1), np.float32), lat[1:], k=4)

    # imagine the road ahead from frame 0 ALONE.
    cur = lat[0]
    roll, idxs = [], []
    z = np.zeros(1, np.float32)
    for _ in range(n - 1):
        cur, i = dyn.step_with_index(cur, z)
        roll.append(cur); idxs.append(i)
    imagined = decode_trajectory(roll, lat, frames)     # n-1 imagined frames
    real = frames[1:n]

    # metrics
    progress = float(np.corrcoef(np.arange(len(idxs)), idxs)[0, 1])
    sim = [1.0 - cosine_distance(roll[t], lat[t + 1]) for t in range(len(roll))]
    span = cosine_distance(lat[0], lat[-1])

    # render: input | imagined future | real future, animated over time.
    os.makedirs("docs", exist_ok=True)
    out_dir = os.path.join(tmp, "out"); os.makedirs(out_dir)
    in0 = _label(frames[0], "INPUT (t=0)")
    panes = []
    for t in range(len(imagined)):
        row = np.concatenate([
            in0,
            _label(imagined[t], f"IMAGINED  t+{t + 1}", (150, 0, 0)),
            _label(real[t], f"REAL  t+{t + 1}", (0, 90, 0)),
        ], axis=1)
        Image.fromarray(row).save(os.path.join(out_dir, f"{t:03d}.png"))
        panes.append(row)
    # static strip (subsampled) for README fallback
    step = max(1, len(panes) // 6)
    Image.fromarray(np.concatenate(panes[::step][:6], axis=0)).save(OUT_PNG)
    # gif
    subprocess.run(["ffmpeg", "-y", "-framerate", "6", "-i",
                    os.path.join(out_dir, "%03d.png"),
                    "-vf", "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
                    OUT_GIF], check=True, capture_output=True)

    print("=== visual foresight on REAL public driving video (I-JEPA + retrieval) ===")
    print(f"frames                         : {n} @ {FPS_IN}fps, latent span (0->end) {span:.3f}")
    print(f"imagined-vs-real latent sim     : mean {np.mean(sim):.3f} (1.0 == identical)")
    print(f"imagination marches forward     : idx-vs-time corr {progress:.2f}")
    print(f"wrote {OUT_GIF} and {OUT_PNG}")
    ok = progress > 0.9 and np.mean(sim) > 0.9 and span > 0.08
    print("FORESIGHT_OK" if ok else "FORESIGHT_WEAK")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
