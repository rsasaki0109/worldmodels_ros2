"""Drive *inside* the world model -- interactively, or record a session to GIF.

Loads a playable driving world (docs/drive_world.npz from build_drive_world.py)
and lets you steer through it. Each frame is retrieved from real driving
experience, conditioned on your steering -- no simulator, no training.

Interactive (needs matplotlib + a display):
    python3 world_model_py/test/play_drive.py
    # left / right arrows steer, up = straight, q = quit

Record a scripted session to docs/drive.gif (needs ffmpeg, no display):
    python3 world_model_py/test/play_drive.py --record
    python3 world_model_py/test/play_drive.py --record "L,L,L,S,S,R,R,R,R,S,S,L,L"
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw

from world_model_py.play import load_world

WORLD = os.path.join("docs", "drive_world.npz")
DECODER = os.path.join("docs", "drive_decoder.pt")
OUT_GIF = os.path.join("docs", "drive.gif")
TILE = 256
KEYMAP = {"L": -0.7, "S": 0.0, "R": 0.7}


def _make_sim(generative):
    """Load the world; attach the learned decoder when --generative is set."""
    dpath = DECODER if generative and os.path.exists(DECODER) else None
    if generative and dpath is None:
        print("no decoder found; run validate_generative_decode.py first "
              "(falling back to retrieval)", file=sys.stderr)
    return load_world(WORLD, decoder_path=dpath)


def _hud(frame, steering, t):
    """Upscale a retrieved frame and draw a steering HUD on it."""
    im = Image.fromarray(np.asarray(frame, np.uint8)).resize((TILE, TILE))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, TILE, 22], fill=(255, 255, 255))
    d.text((6, 5), f"DRIVING INSIDE THE WORLD MODEL   t+{t}", fill=(20, 20, 20))
    # steering bar at the bottom
    cx = TILE // 2
    d.rectangle([0, TILE - 24, TILE, TILE], fill=(0, 0, 0))
    d.line([cx, TILE - 22, cx, TILE - 2], fill=(90, 90, 90))
    x = int(cx + steering / 0.7 * (TILE // 2 - 8))
    col = (0, 120, 255) if steering < -0.05 else (255, 80, 0) if steering > 0.05 else (180, 180, 180)
    d.rectangle([min(cx, x), TILE - 20, max(cx, x), TILE - 6], fill=col)
    label = "LEFT" if steering < -0.05 else "RIGHT" if steering > 0.05 else "STRAIGHT"
    d.text((6, TILE - 20), label, fill=(255, 255, 255))
    return np.asarray(im)


def record(seq, generative=False):
    sim = _make_sim(generative)
    tmp = tempfile.mkdtemp()
    frame = sim.reset()
    panes = [_hud(frame, 0.0, 0)]
    for i, key in enumerate(seq, 1):
        s = KEYMAP[key]
        frame, _ = sim.step(s)
        panes.append(_hud(frame, s, i))
    for i, im in enumerate(panes):
        Image.fromarray(im).save(os.path.join(tmp, f"{i:03d}.png"))
    os.makedirs("docs", exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-framerate", "4", "-i",
                    os.path.join(tmp, "%03d.png"),
                    "-vf", "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
                    OUT_GIF], check=True, capture_output=True)
    print(f"wrote {OUT_GIF} ({len(panes)} frames, steering {''.join(seq)})")


def interactive(generative=False):
    import matplotlib.pyplot as plt

    sim = _make_sim(generative)
    state = {"steer": 0.0}
    fig, ax = plt.subplots(figsize=(5, 5))
    img = ax.imshow(_hud(sim.reset(), 0.0, 0))
    ax.axis("off")

    def on_key(event):
        if event.key == "q":
            plt.close(fig); return
        state["steer"] = {"left": -0.7, "right": 0.7, "up": 0.0}.get(event.key, state["steer"])
        frame, _ = sim.step(state["steer"])
        img.set_data(_hud(frame, state["steer"], sim.t))
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)
    print("drive: left/right arrows steer, up = straight, q = quit")
    plt.show()


def main():
    if not os.path.exists(WORLD):
        print(f"missing {WORLD}; run build_drive_world.py first", file=sys.stderr)
        return 1
    args = sys.argv[1:]
    generative = "--generative" in args
    args = [a for a in args if a != "--generative"]
    if args and args[0] == "--record":
        seq = args[1].split(",") if len(args) > 1 else list("SSLLLLSSRRRRRSSLLLSS")
        record([k.strip().upper() for k in seq], generative=generative)
    else:
        interactive(generative=generative)
    return 0


if __name__ == "__main__":
    sys.exit(main())
