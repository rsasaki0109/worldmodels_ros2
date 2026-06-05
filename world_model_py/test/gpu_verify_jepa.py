"""Manual GPU verification of the real JEPA encoder (NOT run in CI).

Downloads an I-JEPA checkpoint, encodes a few synthetic frames on the GPU, and
prints latent dim + surprise so the heavy backend path is proven end to end.

    python3 -m world_model_py.test.gpu_verify_jepa            # default model
    WM_JEPA_MODEL=facebook/ijepa_vith14_1k python3 ...
"""
import os
import time

import numpy as np

from world_model_py.registry import load_model
from world_model_py.adapters.base import Observation


def _frame(seed, kind="noise"):
    rng = np.random.default_rng(seed)
    if kind == "noise":
        return (rng.random((256, 256, 3)) * 255).astype(np.uint8)
    # a structured frame: gradient + block
    h, w = 256, 256
    yy, xx = np.mgrid[0:h, 0:w]
    img = ((xx / w) * 255).astype(np.uint8)[:, :, None].repeat(3, 2)
    img[60:180, 60:180] = [220, 30, 30]
    return img


def main():
    model_id = os.environ.get("WM_JEPA_MODEL", "facebook/ijepa_vith14_1k")
    print(f"loading ijepa adapter: model_id={model_id}")
    t0 = time.perf_counter()
    wm = load_model("ijepa", model_id=model_id, device="cuda", dtype="float16")
    print(f"loaded in {time.perf_counter() - t0:.1f}s; info={wm.info()}")

    # frame 1 (structured)
    p1 = wm.predict_future(Observation(image=_frame(0, "struct")), horizon=4)
    print(f"frame1: latent_dim={len(p1.latents[0])} horizon={p1.horizon} "
          f"risk={p1.risk:.3f} conf={p1.risk_confidence:.2f} (baseline)")

    # frame 2 (same structured frame -> low surprise)
    p2 = wm.predict_future(Observation(image=_frame(0, "struct")), horizon=4)
    print(f"frame2 (same):    risk={p2.risk:.4f} conf={p2.risk_confidence:.2f}")

    # frame 3 (different, noise -> higher surprise)
    p3 = wm.predict_future(Observation(image=_frame(7, "noise")), horizon=4)
    print(f"frame3 (changed): risk={p3.risk:.4f} conf={p3.risk_confidence:.2f}")

    ok = (
        len(p1.latents[0]) > 0
        and p1.risk == 0.0
        and p2.risk < p3.risk
    )
    print("GPU_VERIFY_OK" if ok else "GPU_VERIFY_BAD")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
