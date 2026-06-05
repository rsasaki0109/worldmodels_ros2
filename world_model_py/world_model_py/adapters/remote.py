"""HTTP adapter that forwards observations to a remote World Model server.

Heavy backends (Cosmos 16B/64B, DreamZero, ...) do not fit on a 16 GB GPU,
so they are expected to run on a separate machine and be reached over a
simple JSON/HTTP boundary. This adapter is the local stub for that: it POSTs
the observation and parses a ``FuturePrediction``-shaped JSON reply.

Only the Python stdlib is used (urllib) so the package has no extra runtime
dependency just to talk to a server.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Optional

import numpy as np

from .base import (
    ActionCondition,
    FuturePrediction,
    Observation,
    WorldModelAdapter,
)


class RemoteAdapterError(RuntimeError):
    pass


class RemoteAdapter(WorldModelAdapter):
    name = "remote"

    def __init__(self, url: str = "http://localhost:8080/predict_future", timeout: float = 30.0):
        self.url = url
        self.timeout = float(timeout)

    def _payload(self, obs, action, horizon) -> dict:
        def arr(a):
            return None if a is None else np.asarray(a).tolist()

        return {
            "observation": {
                "image": arr(obs.image),
                "ego_state": arr(obs.ego_state),
                "action_history": arr(obs.action_history),
                "instruction": obs.instruction,
            },
            "action": None
            if action is None
            else {"action": arr(action.action), "dt": action.dt},
            "horizon": int(horizon),
        }

    def predict_future(
        self,
        obs: Observation,
        action: Optional[ActionCondition] = None,
        horizon: int = 8,
    ) -> FuturePrediction:
        if action is not None and action.horizon > 0:
            horizon = action.horizon
        data = json.dumps(self._payload(obs, action, horizon)).encode("utf-8")
        req = urllib.request.Request(
            self.url, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            raise RemoteAdapterError(f"remote world model at {self.url} failed: {exc}") from exc

        return self._parse(body)

    @staticmethod
    def _parse(body: dict) -> FuturePrediction:
        latents = [np.asarray(x, dtype=np.float32) for x in body.get("latents", [])]
        occupancy = [np.asarray(x, dtype=np.int8) for x in body.get("occupancy", [])]
        frames = [np.asarray(x, dtype=np.uint8) for x in body.get("frames", [])]
        return FuturePrediction(
            dt=float(body.get("dt", 0.1)),
            latents=latents,
            occupancy=occupancy,
            frames=frames,
            risk=float(body.get("risk", 0.0)),
            risk_confidence=float(body.get("risk_confidence", 1.0)),
            risk_label=str(body.get("risk_label", "remote")),
        )

    def info(self) -> dict:
        return {"name": self.name, "remote": True, "url": self.url, "device": "remote"}
