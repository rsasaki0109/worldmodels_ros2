"""JSON wire format shared by the remote adapter and the reference server.

Both halves of the local<->remote split import these functions, so the request
the client sends and the response the server returns can never drift apart, and
the round-trip is unit-testable with no network. Plain Python lists/dicts only
(JSON-serialisable); numpy is used only to normalise shapes.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .adapters.base import ActionCondition, FuturePrediction, Observation


def _list(a) -> Optional[list]:
    return None if a is None else np.asarray(a).tolist()


# ----------------------------------------------------------- request (client -> server)

def request_payload(obs: Observation, action: Optional[ActionCondition], horizon: int) -> dict:
    return {
        "observation": {
            "image": _list(obs.image),
            "ego_state": _list(obs.ego_state),
            "action_history": _list(obs.action_history),
            "instruction": obs.instruction,
        },
        "action": None if action is None or action.action is None
        else {"action": _list(action.action), "dt": action.dt},
        "horizon": int(horizon),
    }


def observation_from_payload(payload: dict) -> Observation:
    o = payload.get("observation") or {}

    def f32(x):
        return None if x is None else np.asarray(x, dtype=np.float32)

    img = o.get("image")
    return Observation(
        image=None if img is None else np.asarray(img, dtype=np.uint8),
        ego_state=f32(o.get("ego_state")),
        action_history=f32(o.get("action_history")),
        instruction=o.get("instruction", "") or "",
    )


def action_from_payload(payload: dict) -> Optional[ActionCondition]:
    a = payload.get("action")
    if not a or a.get("action") is None:
        return None
    return ActionCondition(action=np.asarray(a["action"], dtype=np.float32), dt=float(a.get("dt", 0.1)))


# ----------------------------------------------------------- response (server -> client)

def prediction_to_response(pred: FuturePrediction) -> dict:
    return {
        "dt": float(pred.dt),
        "latents": [np.asarray(x, dtype=np.float32).tolist() for x in pred.latents],
        "occupancy": [np.asarray(x, dtype=np.int8).tolist() for x in pred.occupancy],
        "frames": [np.asarray(x, dtype=np.uint8).tolist() for x in pred.frames],
        "risk": float(pred.risk),
        "risk_confidence": float(pred.risk_confidence),
        "risk_label": pred.risk_label,
    }


def prediction_from_response(body: dict) -> FuturePrediction:
    return FuturePrediction(
        dt=float(body.get("dt", 0.1)),
        latents=[np.asarray(x, dtype=np.float32) for x in body.get("latents", [])],
        occupancy=[np.asarray(x, dtype=np.int8) for x in body.get("occupancy", [])],
        frames=[np.asarray(x, dtype=np.uint8) for x in body.get("frames", [])],
        risk=float(body.get("risk", 0.0)),
        risk_confidence=float(body.get("risk_confidence", 1.0)),
        risk_label=str(body.get("risk_label", "remote")),
    )
