"""Build and save experience memories for retrieval planning (pure numpy)."""
from __future__ import annotations

import os
from typing import Optional

import numpy as np


def build_transitions(
    latents: np.ndarray,
    actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn per-frame latents and actions into (latent, action, next_latent) rows."""
    lat = np.asarray(latents, dtype=np.float32)
    act = np.asarray(actions, dtype=np.float32)
    if lat.ndim != 2 or act.ndim != 2:
        raise ValueError("latents and actions must be 2-D arrays")
    if len(lat) < 2:
        raise ValueError("need at least 2 frames for transitions")
    if len(lat) != len(act):
        raise ValueError(f"latents/actions length mismatch: {len(lat)} vs {len(act)}")
    return lat[:-1].copy(), act[:-1].copy(), lat[1:].copy()


def align_frames_to_next(frames: np.ndarray) -> np.ndarray:
    """Frames aligned to arrival latents (index i shows state at latent[i+1])."""
    fr = np.asarray(frames)
    if len(fr) < 2:
        raise ValueError("need at least 2 frames")
    return fr[1:].copy()


def save_experience(
    path: str,
    latents: np.ndarray,
    actions: np.ndarray,
    frames: Optional[np.ndarray] = None,
    **meta,
) -> dict:
    """Write an experience ``.npz`` and return a short summary dict."""
    L, S, Nx = build_transitions(latents, actions)
    payload = {
        "latents": L,
        "actions": S,
        "next_latents": Nx,
    }
    if frames is not None:
        payload["frames"] = align_frames_to_next(np.asarray(frames, dtype=np.uint8))
    for key, val in meta.items():
        payload[key] = val
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez_compressed(path, **payload)
    return {
        "path": path,
        "transitions": int(len(L)),
        "action_dim": int(S.shape[1]) if S.ndim == 2 else 0,
        "latent_dim": int(L.shape[1]) if L.ndim == 2 else 0,
        "has_frames": frames is not None,
    }


def experience_summary(path: str) -> dict:
    d = np.load(path)
    n = int(len(d["latents"]))
    return {
        "path": path,
        "transitions": n,
        "action_dim": int(d["actions"].shape[1]),
        "latent_dim": int(d["latents"].shape[1]),
        "has_frames": "frames" in d.files,
    }
