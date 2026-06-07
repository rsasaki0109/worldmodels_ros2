"""Train a latent->pixel decoder and *generate* the drive instead of retrieving it.

Loads a playable world (docs/drive_world.npz from build_drive_world.py), trains a
small conv decoder ``latent -> image`` on the experience frames, and renders the
difference between *retrieving* the imagined future (nearest real frame) and
*generating* it (decoded pixels) along the same learned-dynamics rollout.

Outputs:
  * docs/drive_decoder.pt   -- the trained decoder (gitignored)
  * docs/decoder.png        -- real | reconstruction pairs + a latent interpolation
  * docs/generated_drive.gif-- RETRIEVED vs GENERATED, side by side, as you steer

Needs torch + the world npz; GPU recommended. Not CI.

    WM_DEVICE=cuda python3 world_model_py/test/validate_generative_decode.py
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw

from world_model_py.decoder import LatentDecoder
from world_model_py.play import load_world

WORLD = os.path.join("docs", "drive_world.npz")
DEC = os.path.join("docs", "drive_decoder.pt")
PNG = os.path.join("docs", "decoder.png")
GIF = os.path.join("docs", "generated_drive.gif")
TILE = 192
KEYMAP = {"L": -0.7, "S": 0.0, "R": 0.7}


def _label(img, text, color):
    im = Image.fromarray(np.asarray(img, np.uint8)).resize((TILE, TILE))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, TILE, 18], fill=(255, 255, 255))
    d.text((5, 4), text, fill=color)
    return np.asarray(im)


def main():
    if not os.path.exists(WORLD):
        print(f"missing {WORLD}; run build_drive_world.py first", file=sys.stderr)
        return 1
    dev = os.environ.get("WM_DEVICE", "cuda")
    z = np.load(WORLD)
    lat, frames = z["next_latents"].astype(np.float32), z["frames"]

    print(f"training latent->pixel decoder on {len(lat)} frames ({dev}) ...")
    dec = LatentDecoder.fit(lat, frames, epochs=1500, device=dev)
    os.makedirs("docs", exist_ok=True)
    dec.save(DEC)
    print(f"reconstruction PSNR {dec.psnr(lat, frames):.1f} dB  ->  wrote {DEC}")

    # decoder.png: real | recon pairs, then a latent interpolation (generation of
    # frames that were never observed).
    sel = [6, len(lat) // 4, len(lat) // 2, int(len(lat) * 0.7)]
    rec = dec.decode(lat[sel])
    top = np.concatenate([np.concatenate([_label(frames[s], "real", (20, 20, 20)),
                                          _label(rec[i], "decoded", (200, 40, 0))], 1)
                          for i, s in enumerate(sel)], 1)
    a, b = lat[sel[0]], lat[sel[2]]
    interp = np.stack([a * (1 - t) + b * t for t in np.linspace(0, 1, 8)]).astype(np.float32)
    gen = dec.decode(interp)
    bot = np.concatenate([_label(g, "generated" if 0 < i < 7 else "endpoint",
                                 (0, 70, 200) if 0 < i < 7 else (20, 20, 20))
                          for i, g in enumerate(gen)], 1)
    w = max(top.shape[1], bot.shape[1])
    pad = lambda r: np.concatenate([r, np.zeros((r.shape[0], w - r.shape[1], 3), np.uint8)], 1)
    Image.fromarray(np.concatenate([pad(top), pad(bot)], 0)).save(PNG)
    print(f"wrote {PNG}")

    # generated_drive.gif: same learned-dynamics ride, RETRIEVED vs GENERATED.
    seq = list("SSLLLLLSSRRRRRSS")
    ret_sim = load_world(WORLD)                       # nearest-neighbour decode
    gen_sim = load_world(WORLD); gen_sim.decoder = dec   # generated decode
    ret_sim.reset(); gen_sim.reset()
    tmp = tempfile.mkdtemp()
    panes = []
    for i, key in enumerate([None] + seq):
        if key is not None:
            ret_sim.step(KEYMAP[key]); gen_sim.step(KEYMAP[key])
        row = np.concatenate([
            _label(ret_sim.frame(), f"RETRIEVED  t+{i}", (40, 40, 40)),
            _label(gen_sim.frame(), f"GENERATED  t+{i}", (0, 120, 255)),
        ], axis=1)
        Image.fromarray(row).save(os.path.join(tmp, f"{i:03d}.png"))
        panes.append(row)
    subprocess.run(["ffmpeg", "-y", "-framerate", "4", "-i", os.path.join(tmp, "%03d.png"),
                    "-vf", "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse", GIF],
                   check=True, capture_output=True)
    print(f"wrote {GIF} ({len(panes)} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
