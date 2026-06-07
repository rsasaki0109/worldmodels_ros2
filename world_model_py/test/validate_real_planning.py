"""Visual foresight on PUBLIC real robot data: plan to an image goal, no training.

Claim under test: with only a frozen JEPA *encoder* and a memory of real robot
transitions, the latent planner (episodic retrieval dynamics + CEM) can imagine
an action sequence that reaches a *goal image* -- the "World Model版 Nav2"
behaviour -- without training any dynamics model.

Pipeline on a real LeRobot SO-101 pick-and-place episode:
  1. encode the first N frames with I-JEPA  -> latents (N, D)
  2. build a transition memory (latent_t, real_action_t, latent_{t+1})
  3. start = latents[0], goal = latents[-1] (the task-completed frame)
  4. plan_to_goal: CEM over action sequences through RetrievalDynamics
  5. decode the imagined latents to real frames (nearest-neighbour)
  6. report: did the imagined endpoint reach the goal? beat random/no-op?
     does the imagined rollout march forward through the episode?

Writes an imagination-vs-reality strip to docs/foresight.png.

Needs network (Hugging Face) + ffmpeg + torch/transformers + a GPU is nice.
Not in CI.

    WM_DEVICE=cuda python3 world_model_py/test/validate_real_planning.py
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
from world_model_py.planning import (
    RetrievalDynamics,
    cosine_distance,
    decode_trajectory,
    plan_to_goal,
)

REPO = "https://huggingface.co/datasets/lerobot/svla_so101_pickplace/resolve/main"
VIDEO = REPO + "/videos/observation.images.side/chunk-000/file-000.mp4"
PARQUET = REPO + "/data/chunk-000/file-000.parquet"
N = 60                          # frames of the episode to use as the memory
OUT = os.path.join("docs", "foresight.png")


def _strip(rows, path, scale=160):
    """Save a grid PNG: one row per sequence, each row a horizontal frame strip."""
    h = w = scale
    ncol = max(len(r) for r in rows)
    canvas = np.full((h * len(rows), w * ncol, 3), 255, np.uint8)
    for i, row in enumerate(rows):
        for j, fr in enumerate(row):
            im = np.asarray(Image.fromarray(fr.astype(np.uint8)).resize((w, h)))
            canvas[i * h:(i + 1) * h, j * w:(j + 1) * w] = im[:, :, :3]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Image.fromarray(canvas).save(path)


def main():
    tmp = tempfile.mkdtemp()
    mp4 = os.path.join(tmp, "v.mp4")
    parq = os.path.join(tmp, "d.parquet")
    print("downloading public LeRobot SO-101 episode ...")
    urllib.request.urlretrieve(VIDEO, mp4)
    urllib.request.urlretrieve(PARQUET, parq)

    fr = os.path.join(tmp, "fr"); os.makedirs(fr)
    subprocess.run(["ffmpeg", "-y", "-i", mp4, "-frames:v", str(N),
                    "-vf", "scale=256:256", os.path.join(fr, "%03d.png")],
                   check=True, capture_output=True)
    frames = [np.asarray(Image.open(f).convert("RGB")).astype(np.uint8)
              for f in sorted(glob.glob(fr + "/*.png"))][:N]
    actions = np.array(pq.read_table(parq).column("action").to_pylist()[:N], np.float32)
    n = min(len(frames), len(actions))
    frames, actions = frames[:n], actions[:n]

    dev = os.environ.get("WM_DEVICE", "cuda")
    dtype = "float16" if dev == "cuda" else "float32"
    print(f"encoding {n} frames with I-JEPA on {dev} ...")
    wm = load_model("ijepa", model_id="facebook/ijepa_vith14_1k", device=dev, dtype=dtype)
    latents = np.array([wm.encode(Observation(image=f)).data for f in frames], np.float32)

    # transition memory: (latent_t, action_t, latent_{t+1})
    dyn = RetrievalDynamics(latents[:-1], actions[:-1], latents[1:], k=6)
    start, goal = latents[0], latents[-1]
    alo, ahi = float(actions.min()), float(actions.max())
    horizon = n - 1

    print(f"planning to goal image over horizon {horizon} (action box [{alo:.2f},{ahi:.2f}]) ...")
    res = plan_to_goal(dyn, start, goal, action_dim=actions.shape[1], horizon=horizon,
                       samples=512, iterations=6, action_low=alo, action_high=ahi,
                       terminal_weight=1.0, path_weight=0.2, seed=0)

    # baselines: do nothing, and random actions.
    rng = np.random.default_rng(1)
    noop = [start]
    for _ in range(horizon):
        noop.append(dyn.step(noop[-1], np.zeros(actions.shape[1], np.float32)))
    rand_seq = rng.uniform(alo, ahi, size=(horizon, actions.shape[1])).astype(np.float32)
    rlat = start
    for a in rand_seq:
        rlat = dyn.step(rlat, a)

    d0 = cosine_distance(start, goal)
    d_plan = cosine_distance(res.latents[-1], goal)
    d_noop = cosine_distance(noop[-1], goal)
    d_rand = cosine_distance(rlat, goal)

    # decode the imagined rollout to real frames; do the indices march forward?
    imagined = decode_trajectory(res.latents, latents, frames)
    idx = [i for i in res.indices if i >= 0]
    progress = (np.corrcoef(np.arange(len(idx)), idx)[0, 1] if len(idx) > 1 else 0.0)

    # picture: top = imagined future (decoded), bottom = real episode (subsampled)
    step = max(1, len(imagined) // 8)
    img_row = [start_frame := frames[0]] + imagined[::step][:8] + [frames[-1]]
    real_row = frames[::max(1, n // len(img_row))][:len(img_row)]
    _strip([img_row, real_row], OUT)

    print("=== visual foresight on REAL LeRobot SO-101 (I-JEPA + retrieval planner) ===")
    print(f"start->goal distance      : {d0:.3f}")
    print(f"planned endpoint distance : {d_plan:.3f}   ({d0 / (d_plan + 1e-9):.1f}x closer)")
    print(f"  vs no-op                : {d_noop:.3f}")
    print(f"  vs random actions       : {d_rand:.3f}")
    print(f"imagination marches forward (idx vs time corr): {progress:.2f}")
    print(f"strip written             : {OUT}")
    ok = d_plan < d0 and d_plan < d_noop and d_plan < d_rand
    print("FORESIGHT_OK" if ok else "FORESIGHT_WEAK")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
