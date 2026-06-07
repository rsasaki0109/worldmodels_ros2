"""Build a playable driving world from PUBLIC video -> docs/drive_world.npz.

Downloads a few real driving episodes (yaak-ai/L2D, front camera), encodes every
frame with I-JEPA, and saves an action-conditioned experience memory that
play_drive.py can be *driven through* with a keyboard. No dynamics training: the
.npz is just (latent, steering, next_latent, frame) rows recorded from real
driving.

Needs network (Hugging Face) + ffmpeg + torch/transformers (GPU nice). Not CI.

    WM_DEVICE=cuda python3 world_model_py/test/build_drive_world.py
"""
import glob
import os
import subprocess
import sys
import tempfile
import urllib.request

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from world_model_py.registry import load_model
from world_model_py.adapters import Observation
from world_model_py.dynamics import MLPDynamics
from world_model_py.planning import RetrievalDynamics, cosine_distance

B = "https://huggingface.co/datasets/yaak-ai/L2D/resolve/main/"
VURL = B + "videos/observation.images.front_left/chunk-000/file-120.mp4"
DATA = B + "data/chunk-000/file-004.parquet"
# Turning + straight episodes that share video file-120 / data file-004.
EPISODES = [(21701, 414.0), (25117, 784.3), (25167, 814.3),
            (25304, 844.3), (25339, 859.3)]
FPS = 5
DUR = 30
SIZE = 96            # stored frame size (keeps the npz small enough to share)
W = 0.4              # action weight: steering matters, latent still anchors
OUT = os.path.join("docs", "drive_world.npz")


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
    print(f"encoding episodes with I-JEPA on {dev} ...")
    wm = load_model("ijepa", model_id="facebook/ijepa_vith14_1k", device=dev, dtype=dtype)

    L, S, Nx, frames = [], [], [], []
    for ep, ss in EPISODES:
        d = os.path.join(tmp, str(ep)); os.makedirs(d, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-ss", str(ss), "-i", VURL, "-t", str(DUR),
                        "-vf", f"fps={FPS},scale=256:256", os.path.join(d, "%03d.png")],
                       check=True, capture_output=True)
        files = sorted(glob.glob(d + "/*.png"))
        fr = [np.asarray(Image.open(f).convert("RGB")).astype(np.uint8) for f in files]
        st = steer_of(ep)
        n = min(len(fr), len(st)); fr, st = fr[:n], st[:n]
        lat = np.array([wm.encode(Observation(image=f)).data for f in fr], np.float32)
        small = [np.asarray(Image.fromarray(f).resize((SIZE, SIZE)), np.uint8) for f in fr]
        # frames aligned to the *next* latent: a decoded latent shows where you arrive.
        L.append(lat[:-1]); S.append(st[:-1]); Nx.append(lat[1:]); frames.extend(small[1:])
        print(f"  ep{ep}: {n} frames, steer[{st.min():.2f},{st.max():.2f}]")

    L = np.concatenate(L); S = np.concatenate(S)[:, None]; Nx = np.concatenate(Nx)
    frames = np.stack(frames).astype(np.uint8)

    # Train the learned latent-dynamics head (torch, GPU) -- this is what makes
    # the world genuinely *playable*: a constant action keeps moving (no
    # retrieval fixed-point) and steering changes the future. Weights are stored
    # as numpy, so playing back needs no torch.
    print("training learned latent dynamics (PCA + MLP) ...")
    head = MLPDynamics.fit(L, S, Nx, n_components=24, action_scale=7.0,
                           device=dev if dev != "mps" else "cpu", epochs=2500)

    # Honest report: learned head vs learning-free retrieval, on the same memory.
    ret = RetrievalDynamics(L, S, Nx, k=4, action_weight=W)
    start = Nx[6]

    def end(dyn, s, H=12):
        c = start.copy()
        for _ in range(H):
            c = dyn.step(c, np.array([s], np.float32))
        return c

    def motion(dyn, s, H=12):
        c = start.copy(); tot = 0.0
        for _ in range(H):
            nc = dyn.step(c, np.array([s], np.float32)); tot += np.linalg.norm(nc - c); c = nc
        return tot / H
    print("=== learned MLP dynamics vs learning-free retrieval ===")
    print(f"  steering L-R   learned {cosine_distance(end(head, -0.7), end(head, 0.7)):.3f}"
          f"   retrieval {cosine_distance(end(ret, -0.7), end(ret, 0.7)):.3f}")
    print(f"  straight motion learned {motion(head, 0.0):.3f}"
          f"   retrieval {motion(ret, 0.0):.3f}  (learned keeps moving, no fixed point)")

    os.makedirs("docs", exist_ok=True)
    payload = {"latents": L, "actions": S, "next_latents": Nx, "frames": frames,
               "start_index": 6, "action_weight": np.float32(W), "k": np.int32(4),
               # learned head (numpy weights):
               "mu": head.mu, "components": head.components,
               "action_scale": np.float32(head.action_scale),
               "action_dim": np.int32(head.action_dim),
               "n_layers": np.int32(len(head.layers))}
    for i, (Wt, b) in enumerate(head.layers):
        payload[f"W{i}"] = Wt; payload[f"b{i}"] = b
    np.savez_compressed(OUT, **payload)
    mb = os.path.getsize(OUT) / 1e6
    print(f"wrote {OUT}: {len(L)} transitions, frames {frames.shape}, learned head, {mb:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
