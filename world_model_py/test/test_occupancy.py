"""Tests for the occupancy pipeline: camera geometry, simulator, head, adapter.

Everything except ``OccupancyHead.fit`` (and the conv parity check) is numpy
only and runs on CI with no torch, ROS or GPU.

    pytest world_model_py/test/test_occupancy.py
"""
import numpy as np
import pytest

from world_model_py import wire
from world_model_py.adapters.base import FuturePrediction, Observation
from world_model_py.adapters.jepa import IJepaAdapter, PixelEncoder
from world_model_py.camera_model import CameraModel
from world_model_py.occupancy_head import OccupancyHead, _conv_np, lift_table, pooled_iou
from world_model_py.occupancy_sim import (
    Episode, SimConfig, current_occupancy, future_occupancy, rollout_episode, visibility,
)


# --- camera ------------------------------------------------------------------------
def test_pinhole_projects_ground_point_ahead():
    cam = CameraModel.pinhole(256, 256, 128.0, 128.0, 128.0, 25.6, position=(0, 0, 1.5))
    u, v, ok = cam.project(np.array([[4.0, 0.0, 0.0], [4.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]))
    assert ok.tolist() == [True, True, False]           # behind the camera is invalid
    assert u[0] == pytest.approx(128.0)
    assert v[0] == pytest.approx(25.6 + 128.0 * 1.5 / 4.0)
    assert u[1] < u[0]                                  # +y (left) lands left of centre


def test_pitch_down_raises_the_horizon_in_the_image():
    flat = CameraModel.pinhole(100, 100, 50, 50, 50, 50, position=(0, 0, 1.0))
    down = CameraModel.pinhole(100, 100, 50, 50, 50, 50, position=(0, 0, 1.0), pitch=0.2)
    far = np.array([[1000.0, 0.0, 1.0]])
    assert flat.project(far)[1][0] == pytest.approx(50.0, abs=1e-3)
    assert down.project(far)[1][0] < 50.0


def test_from_camera_info_matches_pinhole():
    cam = CameraModel.pinhole(64, 48, 40, 41, 32, 20, position=(0.2, 0.0, 1.0))
    T = np.eye(4)
    T[:3, :] = np.linalg.inv(np.array([[40, 0, 32], [0, 41, 20], [0, 0, 1.0]])) @ cam.P
    other = CameraModel.from_camera_info([40, 0, 32, 0, 41, 20, 0, 0, 1], 64, 48, T)
    pts = np.array([[3.0, 0.5, 0.0], [5.0, -1.0, 0.3]])
    assert np.allclose(cam.project(pts)[0], other.project(pts)[0])
    assert np.allclose(cam.project(pts)[1], other.project(pts)[1])


# --- simulator ---------------------------------------------------------------------
def test_rollout_is_deterministic_and_shaped():
    cfg = SimConfig(image_size=64)
    a = rollout_episode(12, seed=3, cfg=cfg)
    b = rollout_episode(12, seed=3, cfg=cfg)
    assert a.frames.shape == (12, 64, 64, 3) and a.frames.dtype == np.uint8
    assert np.array_equal(a.frames, b.frames) and np.array_equal(a.obstacles, b.obstacles)
    fut = future_occupancy(a, 2, 4, cfg)
    assert fut.shape == (4, cfg.grid_size, cfg.grid_size) and fut.dtype == bool
    with pytest.raises(IndexError):
        future_occupancy(a, 10, 4, cfg)


def _one_obstacle_episode(x, y, radius=0.5, height=2.0):
    pts = np.array([[[x, y]]], np.float32)
    return Episode(frames=np.zeros((1, 8, 8, 3), np.uint8), actions=np.zeros((1, 2), np.float32),
                   ego=np.zeros((1, 3), np.float32), obstacles=pts,
                   radii=np.array([radius], np.float32), heights=np.array([height], np.float32))


def test_occupancy_lands_in_the_right_cell():
    cfg = SimConfig()
    occ = current_occupancy(_one_obstacle_episode(4.25, 0.25, radius=0.1), 0, cfg)
    i = int((0.25 - cfg.origin[1]) // cfg.resolution)   # row = y
    j = int((4.25 - cfg.origin[0]) // cfg.resolution)   # col = x
    assert occ[i, j] and occ.sum() == 1


def test_visibility_fov_and_occlusion():
    cfg = SimConfig()
    ep = _one_obstacle_episode(3.0, 0.0, radius=0.6, height=2.0)   # taller than the camera
    vis = visibility(ep, 0, cfg)
    centers = cfg.cell_centers()
    row = np.argmin(np.abs(centers[:, 0, 1] - 0.25))
    col = lambda x: np.argmin(np.abs(centers[0, :, 0] - x))
    assert vis[row, col(2.25)]                    # in front of the obstacle
    assert vis[row, col(3.25)]                    # the obstacle's own cells
    assert not vis[row, col(6.25)]                # behind it: occluded
    assert not vis[0, col(0.75)]                  # far left, close: outside the FOV
    open_vis = visibility(_one_obstacle_episode(3.0, 0.0, height=0.1), 0, cfg)
    assert open_vis[row, col(6.25)]               # a low obstacle doesn't hide the ground


# --- head ---------------------------------------------------------------------------
def _random_head(fmap_c=5, channels=3, horizon=4, rows=8, cols=8, hidden=6, seed=0):
    rng = np.random.default_rng(seed)
    cam = CameraModel.pinhole(64, 64, 32, 32, 32, 6.4, position=(0, 0, 1.5))
    mu = rng.normal(size=fmap_c).astype(np.float32)
    comp = np.linalg.qr(rng.normal(size=(fmap_c, channels)))[0].astype(np.float32)
    head = OccupancyHead(camera=cam, ch_mu=mu, ch_comp=comp, ch_std=np.ones(channels),
                         layers=[], horizon=horizon, grid_shape=(rows, cols), resolution=1.0,
                         origin=(0.0, -4.0))
    f = head.in_channels
    shapes = [(hidden, f, 1), (hidden, hidden, 3), (hidden, hidden, 3), (horizon, hidden, 1)]
    head.layers = [(0.3 * rng.normal(size=(o, i, k, k)).astype(np.float32),
                    0.1 * rng.normal(size=o).astype(np.float32)) for o, i, k in shapes]
    return head


def test_lift_table_weights_are_bilinear():
    cam = CameraModel.pinhole(64, 64, 32, 32, 32, 6.4, position=(0, 0, 1.5))
    idx, w = lift_table(cam, (8, 8), 1.0, (0.0, -4.0), (0.0, 1.0), (16, 16))
    assert idx.shape == (2, 64, 4) and w.shape == (2, 64, 4)
    s = w.sum(-1)
    assert np.all((np.isclose(s, 1.0)) | (s == 0.0))     # valid -> 1, invalid -> 0
    assert (s == 0.0).any() and (s > 0).any()
    assert idx.min() >= 0 and idx.max() < 16 * 16


def test_head_predicts_probabilities_and_roundtrips(tmp_path):
    head = _random_head()
    rng = np.random.default_rng(1)
    fmap, prev = rng.normal(size=(2, 10, 12, 5)).astype(np.float32)
    p = head.predict_proba(fmap, prev, ego=np.array([0.5, 0.1]))
    assert p.shape == (4, 8, 8) and np.all((p >= 0) & (p <= 1))
    assert not np.allclose(p, head.predict_proba(fmap, None, None))   # motion/ego matter
    path = str(tmp_path / "head.npz")
    head.threshold = 0.3
    head.save(path)
    back = OccupancyHead.load(path)
    assert back.threshold == pytest.approx(0.3) and np.allclose(back.heights, head.heights)
    assert np.allclose(back.predict_proba(fmap, prev, [0.5, 0.1]), p, atol=1e-5)


def test_occupancy_values_map_threshold_to_fifty():
    head = _random_head()
    head.threshold = 0.2
    v = head.to_occupancy_values(np.array([0.0, 0.1, 0.2, 0.6, 1.0]))
    assert v.dtype == np.int8 and v.tolist() == [0, 25, 50, 75, 100]


def test_pooled_iou_with_visibility_mask():
    pred = np.zeros((1, 2, 2, 2), bool)
    ref = np.zeros_like(pred)
    pred[0, :, 0, 0] = ref[0, :, 0, 0] = True        # hit in a visible cell
    pred[0, :, 1, 1] = True                           # false alarm in a hidden cell
    mask = np.array([[[True, True], [True, False]]])
    assert pooled_iou(pred, ref) == pytest.approx(0.5)
    assert pooled_iou(pred, ref, mask) == pytest.approx(1.0)
    assert pooled_iou(np.zeros_like(pred), np.zeros_like(ref)) == 1.0


def test_conv_matches_torch():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 7, 9, 3)).astype(np.float32)
    W = rng.normal(size=(4, 3, 5, 5)).astype(np.float32)
    b = rng.normal(size=4).astype(np.float32)
    ref = torch.nn.functional.conv2d(torch.from_numpy(x).permute(0, 3, 1, 2),
                                     torch.from_numpy(W), torch.from_numpy(b), padding=2)
    assert np.allclose(_conv_np(x, W, b), ref.permute(0, 2, 3, 1).numpy(), atol=1e-4)
    ref2 = torch.nn.functional.conv2d(torch.from_numpy(x).permute(0, 3, 1, 2),
                                      torch.from_numpy(W), torch.from_numpy(b), padding=4,
                                      dilation=2)
    assert np.allclose(_conv_np(x, W, b, 2), ref2.permute(0, 2, 3, 1).numpy(), atol=1e-4)


def test_fit_trains_a_numpy_head():
    pytest.importorskip("torch")
    head0 = _random_head()
    rng = np.random.default_rng(2)
    z = rng.normal(size=(24, 10, 12, 3)).astype(np.float32)
    tgt = rng.random((24, 4, 8, 8)) < 0.1
    head = OccupancyHead.fit(z, z, rng.normal(size=(24, 2)), tgt, np.ones((24, 8, 8), bool),
                             camera=head0.camera,
                             channel_pca=(np.zeros(3), np.eye(3, dtype=np.float32), np.ones(3)),
                             resolution=1.0, origin=(0.0, -4.0), hidden=8, epochs=2,
                             dilations=(1, 2))
    assert [W.shape[0] for W, _ in head.layers] == [8, 8, 8, 4]
    assert head.dilations == (1, 1, 2, 1)
    thr = head.calibrate(z, z, np.zeros((24, 2)), tgt)
    assert 0.0 < thr < 1.0
    assert head.predict_proba_reduced(z[:2], z[:2]).shape == (2, 4, 8, 8)


# --- adapter / wire -----------------------------------------------------------------
def test_adapter_publishes_head_occupancy():
    head = _random_head(fmap_c=3, channels=3, horizon=4)
    wm = IJepaAdapter(PixelEncoder(size=16), occupancy_head=head)
    img = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    obs = Observation(image=img, action_history=np.array([[0.4, 0.0], [0.5, 0.1]], np.float32))
    pred = wm.predict_future(obs, horizon=8)
    assert len(pred.latents) == 8
    assert len(pred.occupancy) == 4                   # never past the head's horizon
    assert pred.occupancy[0].shape == (8, 8) and pred.occupancy[0].dtype == np.int8
    fov = head.fov_mask()
    assert fov.any() and not fov.all()
    assert np.all(pred.occupancy[0][~fov] == -1)          # unseen -> unknown
    assert np.all(pred.occupancy[0][fov] >= 0)
    assert pred.occupancy_resolution == 1.0 and pred.occupancy_origin == (0.0, -4.0)
    assert wm.dt == head.dt
    assert wm.predict_future(obs, horizon=2).occupancy.__len__() == 2
    assert "occupancy_head" in wm.info()


def test_adapter_requires_map_encoder_for_head():
    class LatentOnly:
        def embed(self, image):
            return np.ones(4, np.float32)

        def info(self):
            return {}

    wm = IJepaAdapter(LatentOnly(), occupancy_head=_random_head())
    with pytest.raises(TypeError):
        wm.predict_future(Observation(image=np.zeros((8, 8, 3), np.uint8)))


def test_wire_carries_occupancy_geometry():
    pred = FuturePrediction(dt=0.2, latents=[np.zeros(3, np.float32)],
                            occupancy=[np.zeros((2, 2), np.int8)],
                            occupancy_resolution=0.5, occupancy_origin=(0.0, -4.0))
    back = wire.prediction_from_response(wire.prediction_to_response(pred))
    assert back.occupancy_resolution == 0.5 and back.occupancy_origin == (0.0, -4.0)
    legacy = wire.prediction_from_response({"latents": [], "occupancy": []})
    assert legacy.occupancy_resolution == 0.1 and legacy.occupancy_origin == (0.0, 0.0)
