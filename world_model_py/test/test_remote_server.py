"""End-to-end local<->remote over real HTTP, stdlib only (no ROS, no GPU).

Starts the reference server (dummy backend) on an ephemeral port and drives it
through the RemoteAdapter, proving the two halves of the split agree.
"""
import threading

import numpy as np

from world_model_py.adapters import ActionCondition, Observation, RemoteAdapter
from world_model_py.registry import load_model
from world_model_py.server import build_server


class _Server:
    def __init__(self, adapter_name="dummy"):
        self.httpd, _ = build_server(adapter_name, host="127.0.0.1", port=0)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.thread.join(timeout=5)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}/predict_future"


def _obs():
    return Observation(ego_state=np.array([1.0, 0.0, 0.0, 0.5], np.float32), instruction="go")


def test_health():
    with _Server() as s:
        info = RemoteAdapter(url=s.url).health()
        assert info["status"] == "ok"
        assert info["adapter"]["name"] == "dummy"


def test_predict_matches_local_dummy():
    with _Server() as s:
        remote = RemoteAdapter(url=s.url)
        local = load_model("dummy")
        action = ActionCondition(action=np.zeros((6, 2), np.float32), dt=0.1)

        r = remote.predict_future(_obs(), action)
        l = local.predict_future(_obs(), action)

        assert r.horizon == l.horizon == 6
        assert len(r.occupancy) == len(l.occupancy)
        assert np.allclose(r.latents[0], l.latents[0], atol=1e-5)
        assert abs(r.risk - l.risk) < 1e-5
        assert r.risk_label == l.risk_label


def test_score_trajectory_over_http():
    with _Server() as s:
        remote = RemoteAdapter(url=s.url)
        small = ActionCondition(action=np.full((4, 2), 0.01, np.float32))
        large = ActionCondition(action=np.full((4, 2), 2.0, np.float32))
        assert remote.score_trajectory(_obs(), large) > remote.score_trajectory(_obs(), small)


def test_unreachable_server_raises():
    from world_model_py.adapters import RemoteAdapterError

    remote = RemoteAdapter(url="http://127.0.0.1:9/predict_future", timeout=1.0)
    try:
        remote.predict_future(_obs())
        assert False, "expected RemoteAdapterError"
    except RemoteAdapterError:
        pass
