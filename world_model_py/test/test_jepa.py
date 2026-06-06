"""Surprise / rollout logic for the JEPA adapter, tested with a fake encoder.

No torch, no transformers, no GPU, no model download -- proves the adapter
logic independently of the heavy backend (which has its own GPU script).
"""
import numpy as np

from world_model_py.adapters.base import Observation, ActionCondition
from world_model_py.adapters.jepa import IJepaAdapter


class FakeEncoder:
    """Returns whatever latent it is told to, in sequence."""

    def __init__(self, vectors):
        self._vectors = [np.asarray(v, dtype=np.float32) for v in vectors]
        self._i = 0
        self.calls = 0

    def embed(self, image_hwc_uint8):
        self.calls += 1
        v = self._vectors[min(self._i, len(self._vectors) - 1)]
        self._i += 1
        return v

    def info(self):
        return {"model_id": "fake", "device": "cpu", "latent_dim": None}


def _obs():
    return Observation(image=np.zeros((8, 8, 3), dtype=np.uint8))


def test_encode_returns_latent():
    enc = FakeEncoder([[1.0, 0.0, 0.0]])
    wm = IJepaAdapter(enc)
    latent = wm.encode(_obs())
    assert latent.encoding == "ijepa"
    assert latent.data.shape == (3,)


def test_first_frame_has_zero_surprise_and_confidence():
    wm = IJepaAdapter(FakeEncoder([[1.0, 0.0, 0.0]]))
    pred = wm.predict_future(_obs(), horizon=4)
    assert pred.risk == 0.0
    assert pred.risk_confidence == 0.0
    assert pred.horizon == 4
    assert pred.risk_label == "ijepa-surprise"


def test_identical_frames_low_surprise():
    wm = IJepaAdapter(FakeEncoder([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    wm.predict_future(_obs())
    pred = wm.predict_future(_obs())
    assert pred.risk < 1e-6
    assert pred.risk_confidence == 0.9


def test_orthogonal_change_is_high_surprise():
    wm = IJepaAdapter(FakeEncoder([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    wm.predict_future(_obs())
    pred = wm.predict_future(_obs())
    assert pred.risk > 0.9  # cosine distance of orthogonal vectors == 1


def test_opposite_change_clamped_to_one():
    wm = IJepaAdapter(FakeEncoder([[1.0, 0.0], [-1.0, 0.0]]))
    wm.predict_future(_obs())
    pred = wm.predict_future(_obs())
    assert pred.risk <= 1.0
    assert pred.risk > 0.99


def test_horizon_follows_action():
    wm = IJepaAdapter(FakeEncoder([[1.0, 2.0, 3.0]]))
    act = ActionCondition(action=np.zeros((6, 2), dtype=np.float32))
    pred = wm.predict_future(_obs(), act)
    assert pred.horizon == 6
    # persistence rollout: every step is the same latent
    assert all(np.allclose(pred.latents[0], l) for l in pred.latents)


def test_missing_image_raises():
    wm = IJepaAdapter(FakeEncoder([[1.0]]))
    try:
        wm.predict_future(Observation(image=None))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_reset_clears_baseline():
    wm = IJepaAdapter(FakeEncoder([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]))
    wm.predict_future(_obs())
    wm.reset()
    pred = wm.predict_future(_obs())  # first frame after reset
    assert pred.risk == 0.0
    assert pred.risk_confidence == 0.0


def test_registry_lists_jepa_backends():
    from world_model_py.registry import available_models

    models = available_models()
    assert "ijepa" in models
    assert "vjepa2" in models


def test_name_param_sets_label_and_encoding():
    wm = IJepaAdapter(FakeEncoder([[1.0, 0.0], [0.0, 1.0]]), name="vjepa2")
    assert wm.encode(_obs()).encoding == "vjepa2"
    pred = wm.predict_future(_obs())
    assert pred.risk_label == "vjepa2-surprise"
