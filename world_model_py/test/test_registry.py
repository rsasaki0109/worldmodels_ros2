"""Registry: built-ins, register(), and entry-point discovery. No ROS/GPU."""
import numpy as np

from world_model_py import registry
from world_model_py.adapters import DummyAdapter
from world_model_py.adapters.base import FuturePrediction, Observation, WorldModelAdapter


class _Toy(WorldModelAdapter):
    name = "toy"

    def predict_future(self, obs, action=None, horizon=8):
        return FuturePrediction(dt=0.1, latents=[np.zeros(2, np.float32)] * horizon, risk=0.5)


def _reset():
    registry._EXTRA.clear()
    registry._DISCOVERED = None


def test_builtins_present():
    _reset()
    for name in ("dummy", "remote", "ijepa", "vjepa2"):
        assert name in registry.available_models()


def test_register_adds_and_loads():
    _reset()
    registry.register("toy", _Toy)
    assert "toy" in registry.available_models()
    assert isinstance(registry.load_model("toy"), _Toy)


def test_unknown_raises():
    _reset()
    try:
        registry.load_model("nope")
        assert False
    except KeyError:
        pass


def test_entry_point_discovery(monkeypatch):
    _reset()
    monkeypatch.setattr(registry, "_discover_entry_points", lambda: {"toy_ep": _Toy})
    assert "toy_ep" in registry.available_models()
    assert isinstance(registry.load_model("toy_ep"), _Toy)


def test_builtin_beats_entry_point(monkeypatch):
    _reset()
    # an entry point cannot hijack a built-in name
    monkeypatch.setattr(registry, "_discover_entry_points", lambda: {"dummy": _Toy})
    assert isinstance(registry.load_model("dummy"), DummyAdapter)


def test_bad_entry_point_is_skipped(monkeypatch):
    _reset()
    from importlib import metadata

    class _BadEP:
        name = "broken"

        def load(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(metadata, "entry_points", lambda **k: [_BadEP()])
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        models = registry.available_models()  # must not raise
    assert "dummy" in models
