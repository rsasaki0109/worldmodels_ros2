"""HTTP adapter that forwards observations to a remote World Model server.

Heavy backends (Cosmos 16B/64B, DreamZero, ...) do not fit on a 16 GB GPU, so
they run on a separate machine reached over a simple JSON/HTTP boundary. This
is the local stub: it POSTs the observation and parses a FuturePrediction-shaped
reply. The request/response format lives in ``world_model_py.wire`` and is
shared with the reference server (``world_model_py.server``) so the two cannot
drift. Only the Python stdlib is used (urllib) -- no extra runtime dependency.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Optional

from .base import ActionCondition, FuturePrediction, Observation, WorldModelAdapter
from .. import wire


class RemoteAdapterError(RuntimeError):
    pass


class RemoteAdapter(WorldModelAdapter):
    name = "remote"

    def __init__(self, url: str = "http://localhost:8080/predict_future", timeout: float = 30.0):
        self.url = url
        self.timeout = float(timeout)

    def _post(self, path_url: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            path_url, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            raise RemoteAdapterError(f"remote world model at {path_url} failed: {exc}") from exc

    def predict_future(
        self,
        obs: Observation,
        action: Optional[ActionCondition] = None,
        horizon: int = 8,
    ) -> FuturePrediction:
        if action is not None and action.horizon > 0:
            horizon = action.horizon
        body = self._post(self.url, wire.request_payload(obs, action, horizon))
        return wire.prediction_from_response(body)

    def health(self) -> dict:
        """GET the server's /health sibling of the predict URL."""
        base = self.url.rsplit("/", 1)[0]
        try:
            with urllib.request.urlopen(base + "/health", timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            raise RemoteAdapterError(f"remote health check failed: {exc}") from exc

    def info(self) -> dict:
        return {"name": self.name, "remote": True, "url": self.url, "device": "remote"}
