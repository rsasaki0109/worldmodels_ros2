"""GPU-free unit tests for the playable world model (DriveSim).

Uses a toy *rotation* world: latents are unit 2-D headings and the recorded
dynamics is "rotate the heading by steering * dtheta". A real action-conditioned
RetrievalDynamics should recover that, so steering left vs right drives the
imagined heading apart -- the same divergence the real driving demo shows, but
checkable with no torch, no network and no GPU.

    pytest world_model_py/test/test_play.py
"""
import numpy as np

from world_model_py.planning import cosine_distance
from world_model_py.play import DriveSim


def _rotation_world(n=600, dtheta=0.3, seed=0):
    """Memory of (heading, steering) -> heading rotated by steering*dtheta."""
    rng = np.random.default_rng(seed)
    ang = rng.uniform(-np.pi, np.pi, n).astype(np.float32)
    steer = rng.uniform(-1.0, 1.0, n).astype(np.float32)
    cur = np.stack([np.cos(ang), np.sin(ang)], 1).astype(np.float32)
    nang = ang + steer * dtheta
    nxt = np.stack([np.cos(nang), np.sin(nang)], 1).astype(np.float32)
    # frame i encodes its arrival heading as a flat grey tile (checkable decode).
    frames = [np.full((8, 8, 3), int(((a % (2 * np.pi)) / (2 * np.pi)) * 255), np.uint8)
              for a in nang]
    from world_model_py.planning import RetrievalDynamics
    dyn = RetrievalDynamics(cur, steer[:, None], nxt, k=8, action_weight=0.3)
    return DriveSim(dyn, nxt, frames, start_latent=np.array([1.0, 0.0], np.float32))


def test_reset_returns_frame():
    sim = _rotation_world()
    f = sim.reset()
    assert f.shape == (8, 8, 3) and f.dtype == np.uint8
    assert sim.t == 0


def test_step_advances_and_returns_frame():
    sim = _rotation_world()
    sim.reset()
    f, lat = sim.step(0.5)
    assert f.shape == (8, 8, 3)
    assert lat.shape == (2,)
    assert sim.t == 1


def test_steering_diverges_left_vs_right():
    """The whole point: opposite steering -> different imagined futures."""
    sim = _rotation_world()
    H = 8
    sim.reset()
    for _ in range(H):
        sim.step(+0.7)
    left = sim.latent.copy()
    sim.reset()
    for _ in range(H):
        sim.step(-0.7)
    right = sim.latent.copy()
    assert cosine_distance(left, right) > 0.2


def test_deterministic():
    seq = [0.3, -0.2, 0.8, 0.0, -0.5]
    a = _rotation_world().drive(seq)
    b = _rotation_world().drive(seq)
    assert len(a) == len(seq) + 1
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_straight_keeps_heading():
    """Zero steering should barely rotate the heading."""
    sim = _rotation_world()
    sim.reset()
    start = sim.latent.copy()
    for _ in range(6):
        sim.step(0.0)
    assert cosine_distance(start, sim.latent) < 0.2


def test_frame_length_mismatch_raises():
    from world_model_py.planning import RetrievalDynamics
    lat = np.eye(2, dtype=np.float32)
    dyn = RetrievalDynamics(lat, np.zeros((2, 1), np.float32), lat, k=1)
    try:
        DriveSim(dyn, lat, [np.zeros((4, 4, 3), np.uint8)], start_latent=lat[0])
        assert False, "expected ValueError"
    except ValueError:
        pass
