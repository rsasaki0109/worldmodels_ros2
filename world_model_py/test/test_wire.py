"""Wire round-trip: client payload <-> server observation, and prediction
response <-> FuturePrediction. No network, no ROS."""
import numpy as np

from world_model_py import wire
from world_model_py.adapters.base import (
    ActionCondition,
    FuturePrediction,
    Observation,
)


def test_observation_round_trip():
    obs = Observation(
        image=np.zeros((4, 4, 3), np.uint8),
        ego_state=np.array([1.0, 2.0], np.float32),
        action_history=np.array([[0.1, 0.2]], np.float32),
        instruction="go",
    )
    payload = wire.request_payload(obs, None, horizon=5)
    assert payload["horizon"] == 5
    assert payload["action"] is None

    back = wire.observation_from_payload(payload)
    assert back.instruction == "go"
    assert np.allclose(back.ego_state, obs.ego_state)
    assert back.image.shape == (4, 4, 3)


def test_action_round_trip():
    act = ActionCondition(action=np.ones((3, 2), np.float32), dt=0.2)
    payload = wire.request_payload(Observation(), act, horizon=3)
    back = wire.action_from_payload(payload)
    assert back is not None
    assert back.action.shape == (3, 2)
    assert abs(back.dt - 0.2) < 1e-6


def test_no_action_payload():
    payload = wire.request_payload(Observation(), None, 8)
    assert wire.action_from_payload(payload) is None


def test_prediction_round_trip():
    pred = FuturePrediction(
        dt=0.1,
        latents=[np.array([1.0, 2.0, 3.0], np.float32), np.array([4.0, 5.0, 6.0], np.float32)],
        occupancy=[np.full((2, 2), 50, np.int8)],
        risk=0.42, risk_confidence=0.8, risk_label="dummy",
    )
    body = wire.prediction_to_response(pred)
    back = wire.prediction_from_response(body)
    assert back.dt == 0.1
    assert len(back.latents) == 2
    assert np.allclose(back.latents[1], [4.0, 5.0, 6.0])
    assert len(back.occupancy) == 1
    assert back.occupancy[0].shape == (2, 2)
    assert abs(back.risk - 0.42) < 1e-6
    assert back.risk_label == "dummy"
