"""Counterfactual imagination on PUBLIC driving data: "what if I steer L/S/R?"

Claim under test: from a single current frame, an episodic world model (frozen
JEPA encoder + action-conditioned retrieval dynamics) can imagine *different
futures for different actions* -- turn left vs go straight vs turn right -- with
ZERO dynamics training. This is the "what-if rollout" a planner needs.

How it stays honest (unlike a pixel-replaying reconstruction): the dynamics is
conditioned on the steering action, and the imagined branches *diverge* by
action. We do not claim pixel-accurate prediction; we show that conditioning on
a counterfactual action produces a correspondingly different imagined future,
retrieved from real driving experience.

Pipeline on real public driving video (yaak-ai/L2D, front camera, 10 fps):
  1. collect several short *turning* episodes -> an experience memory
  2. encode every frame with I-JEPA                       -> latents
  3. build action-conditioned retrieval dynamics
       (latent_t, steering_t) -> latent_{t+1}
  4. from one current frame, roll the dynamics forward under a fixed
     counterfactual steering of LEFT / STRAIGHT / RIGHT
  5. decode each imagined latent to a real frame (nearest neighbour)
  6. report the branch divergence and render docs/counterfactual.gif

Needs network (Hugging Face) + ffmpeg + torch/transformers (GPU nice). Not CI.

    WM_DEVICE=cuda python3 world_model_py/test/validate_real_counterfactual.py
"""
import glob
import os
import subprocess
import sys
import tempfile
import urllib.request

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

from world_model_py.registry import load_model
from world_model_py.adapters import Observation
from world_model_py.planning import RetrievalDynamics, cosine_distance, decode_trajectory

B = "https://huggingface.co/datasets/yaak-ai/L2D/resolve/main/"
VURL = B + "videos/observation.images.front_left/chunk-000/file-120.mp4"
DATA = B + "data/chunk-000/file-004.parquet"
# turning episodes that share video file-120 / data file-004 (start seconds).
EPISODES = [(21701, 414.0), (25117, 784.3), (25167, 814.3),
            (25304, 844.3), (25339, 859.3)]
FPS = 5
DUR = 30
W = 0.4              # action weight: steering matters, latent still anchors
H = 14              # imagination horizon (steps)
TILE = 220
OUT_GIF = os.path.join("docs", "counterfactual.gif")
OUT_PNG = os.path.join("docs", "counterfactual.png")


def _tile(img, text, color):
    im = Image.fromarray(img.astype(np.uint8)).resize((TILE, TILE))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, TILE, 18], fill=(255, 255, 255))
    d.text((5, 4), text, fill=color)
    return np.asarray(im)


def main():
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, "d.parquet")
    print("downloading L2D action data ...")
    urllib.request.urlretrieve(DATA, p)
    t = pq.read_table(p, columns=["episode_index", "frame_index", "action.continuous"])
    EI = np.array(t.column("episode_index").to_pylist())
    FI = np.array(t.column("frame_index").to_pylist())
    AC = np.array(t.column("action.continuous").to_pylist())

    def steer_of(ep):
        m = EI == ep
        ac = AC[m][np.argsort(FI[m])]
        return ac[::2, 2].astype(np.float32)        # ::2 -> ~5 fps to match FPS

    dev = os.environ.get("WM_DEVICE", "cuda")
    dtype = "float16" if dev == "cuda" else "float32"
    print(f"encoding turning episodes with I-JEPA on {dev} ...")
    wm = load_model("ijepa", model_id="facebook/ijepa_vith14_1k", device=dev, dtype=dtype)

    L, S, Nx, mem_frames, cur0 = [], [], [], [], None
    for ep, ss in EPISODES:
        d = os.path.join(tmp, str(ep)); os.makedirs(d, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-ss", str(ss), "-i", VURL, "-t", str(DUR),
                        "-vf", f"fps={FPS},scale=256:256", os.path.join(d, "%03d.png")],
                       check=True, capture_output=True)
        fr = [np.asarray(Image.open(f).convert("RGB")).astype(np.uint8)
              for f in sorted(glob.glob(d + "/*.png"))]
        st = steer_of(ep)
        n = min(len(fr), len(st)); fr, st = fr[:n], st[:n]
        lat = np.array([wm.encode(Observation(image=f)).data for f in fr], np.float32)
        # within-episode transitions; frames aligned to the *next* latent so a
        # decoded latent shows where you ARRIVE.
        L.append(lat[:-1]); S.append(st[:-1]); Nx.append(lat[1:]); mem_frames.extend(fr[1:])
        if ep == EPISODES[0][0]:
            cur0, cur0_frame = lat[6], fr[6]        # the "current" observation
        print(f"  ep{ep}: {n} frames, steer[{st.min():.2f},{st.max():.2f}]")

    L = np.concatenate(L); S = np.concatenate(S); Nx = np.concatenate(Nx)
    print(f"experience memory: {len(L)} transitions from {len(EPISODES)} episodes")

    dyn = RetrievalDynamics(L, S[:, None], Nx, k=4, action_weight=W)
    branches = {}
    for name, s in [("LEFT", -0.7), ("STRAIGHT", 0.0), ("RIGHT", 0.7)]:
        c = cur0; traj = []
        for _ in range(H):
            c = dyn.step(c, np.array([s], np.float32)); traj.append(c)
        branches[name] = decode_trajectory(traj, L, mem_frames)

    # divergence on imagined latent endpoints
    def endlat(s):
        c = cur0
        for _ in range(H):
            c = dyn.step(c, np.array([s], np.float32))
        return c
    eL, eS, eR = endlat(-0.7), endlat(0.0), endlat(0.7)

    # render: INPUT | LEFT | STRAIGHT | RIGHT, animated.
    os.makedirs("docs", exist_ok=True)
    out_dir = os.path.join(tmp, "out"); os.makedirs(out_dir)
    in_tile = _tile(cur0_frame, "INPUT (now)", (20, 20, 20))
    BL, BK, BR = (0, 70, 200), (40, 40, 40), (200, 40, 0)
    panes = []
    for ti in range(H):
        row = np.concatenate([
            in_tile,
            _tile(branches["LEFT"][ti], f"what-if LEFT  t+{ti + 1}", BL),
            _tile(branches["STRAIGHT"][ti], f"what-if STRAIGHT  t+{ti + 1}", BK),
            _tile(branches["RIGHT"][ti], f"what-if RIGHT  t+{ti + 1}", BR),
        ], axis=1)
        Image.fromarray(row).save(os.path.join(out_dir, f"{ti:03d}.png"))
        panes.append(row)
    step = max(1, len(panes) // 5)
    Image.fromarray(np.concatenate(panes[::step][:5], axis=0)).save(OUT_PNG)
    subprocess.run(["ffmpeg", "-y", "-framerate", "4", "-i",
                    os.path.join(out_dir, "%03d.png"),
                    "-vf", "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
                    OUT_GIF], check=True, capture_output=True)

    print("=== counterfactual imagination on REAL public driving (I-JEPA + retrieval) ===")
    print(f"endpoint divergence  L-S {cosine_distance(eL, eS):.3f}  "
          f"R-S {cosine_distance(eR, eS):.3f}  L-R {cosine_distance(eL, eR):.3f}")
    print(f"wrote {OUT_GIF} and {OUT_PNG}")
    ok = cosine_distance(eL, eR) > 0.2 and cosine_distance(eL, eS) > 0.1
    print("COUNTERFACTUAL_OK" if ok else "COUNTERFACTUAL_WEAK")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
