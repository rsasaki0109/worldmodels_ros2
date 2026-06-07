"""Tests for the learned latent-dynamics heads.

The numpy inference path (forward math, save/load) and the closed-form
``LinearDynamics`` are tested GPU-free on CI. The torch ``MLPDynamics.fit`` is
exercised end-to-end on a toy rotation world but skipped where torch is absent.

    pytest world_model_py/test/test_dynamics.py
"""
import numpy as np
import pytest

from world_model_py.dynamics import MLPDynamics, _gelu_tanh
from world_model_py.planning import LinearDynamics, cosine_distance


# --- LinearDynamics (closed form, numpy) --------------------------------------
def test_linear_recovers_linear_system():
    """next = latent + 0.5*action (per-dim): the fit should recover it."""
    rng = np.random.default_rng(0)
    lat = rng.normal(size=(400, 3)).astype(np.float32)
    act = rng.normal(size=(400, 3)).astype(np.float32)
    nxt = lat + 0.5 * act
    dyn = LinearDynamics.fit(lat, act, nxt, ridge=1e-3)
    pred = dyn.step(np.zeros(3, np.float32), np.array([1.0, 0.0, -1.0], np.float32))
    assert np.allclose(pred, [0.5, 0.0, -0.5], atol=1e-2)


def test_linear_action_changes_prediction():
    rng = np.random.default_rng(1)
    lat = rng.normal(size=(300, 4)).astype(np.float32)
    act = rng.normal(size=(300, 2)).astype(np.float32)
    nxt = lat.copy()
    nxt[:, :2] += act                                  # action drives first 2 dims
    dyn = LinearDynamics.fit(lat, act, nxt, ridge=1e-3)
    a = dyn.step(np.zeros(4, np.float32), np.array([1.0, 0.0], np.float32))
    b = dyn.step(np.zeros(4, np.float32), np.array([-1.0, 0.0], np.float32))
    assert cosine_distance(a, b) > 0.5


# --- MLPDynamics numpy inference (no torch) -----------------------------------
def _identity_head(d=3, k=2, action_dim=1):
    """A hand-built MLPDynamics whose MLP outputs zeros -> step is identity."""
    mu = np.zeros(d, np.float32)
    comp = np.eye(d, k, dtype=np.float32)              # first k dims
    # 2-layer MLP with zero output weights => delta == 0 for any input.
    layers = [(np.zeros((4, k + action_dim), np.float32), np.zeros(4, np.float32)),
              (np.zeros((k, 4), np.float32), np.zeros(k, np.float32))]
    return MLPDynamics(mu=mu, components=comp, layers=layers, action_dim=action_dim)


def test_mlp_forward_zero_delta_is_identity():
    dyn = _identity_head()
    x = np.array([0.3, -0.7, 0.1], np.float32)
    out = dyn.step(x, np.array([0.5], np.float32))
    # PCA keeps only the first k=2 dims, so the projected part is preserved and
    # the dropped dim collapses; the zero-delta MLP adds nothing.
    assert out.shape == (3,)
    assert np.allclose(out[:2], x[:2], atol=1e-5)


def test_mlp_gelu_matches_formula():
    x = np.array([-2.0, 0.0, 1.5], np.float32)
    g = _gelu_tanh(x)
    assert g[1] == 0.0 and g[2] > 0 and -0.1 < g[0] < 0.0


def test_mlp_save_load_roundtrip(tmp_path):
    dyn = _identity_head()
    # give it non-trivial weights so the round-trip is meaningful
    dyn.layers = [(np.full((4, 3), 0.1, np.float32), np.arange(4, dtype=np.float32)),
                  (np.full((2, 4), 0.2, np.float32), np.zeros(2, np.float32))]
    dyn._comp_t = dyn.components.T.copy()
    p = str(tmp_path / "head.npz")
    dyn.save(p)
    back = MLPDynamics.load(p)
    x = np.array([0.2, -0.4, 0.9], np.float32)
    assert np.allclose(dyn.step(x, np.array([0.3], np.float32)),
                       back.step(x, np.array([0.3], np.float32)), atol=1e-6)


# --- MLPDynamics end-to-end training (torch) ----------------------------------
def test_mlp_fit_learns_steering():
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    n = 800
    ang = rng.uniform(-np.pi, np.pi, n).astype(np.float32)
    steer = rng.uniform(-1, 1, n).astype(np.float32)
    cur = np.stack([np.cos(ang), np.sin(ang)], 1).astype(np.float32)
    na = ang + steer * 0.4
    nxt = np.stack([np.cos(na), np.sin(na)], 1).astype(np.float32)
    dyn = MLPDynamics.fit(cur, steer[:, None], nxt, n_components=2, hidden=64,
                          epochs=1500, action_scale=1.0, seed=0)
    start = np.array([1.0, 0.0], np.float32)
    # one-step prediction is close to the true rotated heading
    pred = dyn.step(start, np.array([0.5], np.float32))
    true = np.array([np.cos(0.5 * 0.4), np.sin(0.5 * 0.4)], np.float32)
    assert cosine_distance(pred, true) < 0.05
    # opposite steering diverges over a rollout (it is genuinely playable)
    cl = start.copy(); crr = start.copy()
    for _ in range(8):
        cl = dyn.step(cl, np.array([0.7], np.float32))
        crr = dyn.step(crr, np.array([-0.7], np.float32))
    assert cosine_distance(cl, crr) > 0.3
