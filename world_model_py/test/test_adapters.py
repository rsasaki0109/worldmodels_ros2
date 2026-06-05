"""Adapter / registry / bench tests. No ROS, no GPU required."""
import numpy as np

from world_model_py import load_model, available_models
from world_model_py.adapters import ActionCondition, Observation
from world_model_py.bench import run_bench, render_html


def _obs():
    return Observation(
        image=np.zeros((16, 16, 3), dtype=np.uint8),
        ego_state=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        instruction="go",
    )


def test_registry_lists_dummy_and_remote():
    models = available_models()
    assert "dummy" in models
    assert "remote" in models


def test_dummy_predict_shapes():
    wm = load_model("dummy", latent_dim=64, grid_size=24)
    pred = wm.predict_future(_obs(), horizon=5)
    assert pred.horizon == 5
    assert len(pred.occupancy) == 5
    assert pred.latents[0].shape == (64,)
    assert pred.occupancy[0].shape == (24, 24)
    assert 0.0 <= pred.risk <= 1.0


def test_dummy_is_deterministic():
    wm = load_model("dummy")
    a = wm.predict_future(_obs(), horizon=3)
    b = wm.predict_future(_obs(), horizon=3)
    assert np.allclose(a.latents[0], b.latents[0])


def test_risk_grows_with_action_magnitude():
    wm = load_model("dummy")
    small = ActionCondition(action=np.full((4, 2), 0.01, dtype=np.float32))
    large = ActionCondition(action=np.full((4, 2), 2.0, dtype=np.float32))
    assert wm.score_trajectory(_obs(), large) > wm.score_trajectory(_obs(), small)


def test_horizon_follows_action():
    wm = load_model("dummy")
    act = ActionCondition(action=np.zeros((7, 2), dtype=np.float32))
    pred = wm.predict_future(_obs(), act)
    assert pred.horizon == 7


def test_encode_returns_latent():
    wm = load_model("dummy", latent_dim=32)
    latent = wm.encode(_obs())
    assert latent.data.shape == (32,)
    assert latent.encoding == "dummy"


def test_bench_runs_and_renders_html():
    result = run_bench("dummy", runs=5, warmup=1)
    assert result.runs == 5
    assert result.latency_ms_p50 >= 0.0
    assert result.throughput_hz > 0.0
    html = render_html(result)
    assert "<html" in html
    assert "dummy" in html
