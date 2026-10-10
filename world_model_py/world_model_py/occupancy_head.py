"""A learned *encoder feature map -> future occupancy* head (IPM lift + BEV CNN).

JEPA encoders give latents, but Nav2 consumers (``world_model_costmap``,
``world_model_dwb_critics``) need ``FutureOccupancy``. This head closes that gap:
from the encoder's spatial feature map (patch tokens) of the current and the
previous frame plus the last ego command, it predicts an occupancy probability
per BEV cell for each of the next ``horizon`` steps.

Why this shape (each choice was measured, see ``test/train_occupancy_head.py``):

  * **Geometry, not a dense image->BEV map.** Every BEV cell is projected into
    the camera (inverse perspective mapping, a short pillar of heights above the
    ground) and samples the feature map there (bilinear). A dense learned
    image->grid mapping memorises positions and does not generalise; the lift
    is fixed by the camera, so the learned part is shared across cells.
  * **Spatial tokens, not the pooled latent.** A mean-pooled latent has lost
    *where* things are. The adapter's latent is still the token mean, so the
    existing contract is unchanged.
  * **A small BEV CNN** (1x1 -> kxk -> kxk -> 1x1) turns the lifted features
    (current, change since the previous frame, ego command, cell position) into
    per-step logits; its receptive field covers how far obstacles move over the
    horizon.

Training uses torch (GPU, e.g. Colab); **inference is pure numpy** (bilinear
gathers + im2col convolutions), so the ROS runtime and CI need no torch and a
trained head is one ``.npz``. Channels are first reduced by a PCA so a ~1000-d
token map trains from a few thousand frames.

The output is calibrated for the costmap rule: the probability threshold that
maximised IoU on validation data is stored, and :meth:`predict_grids` maps it to
occupancy value 50 -- so a consumer using ``occupied_threshold: 50`` applies the
validated decision boundary.

Grids are ego-centric at the observation time: ``grid[i, j]`` = (y row, x
column), lower-left corner at ``origin``, ``resolution`` metres per cell (the
``nav_msgs/OccupancyGrid`` layout in the robot base frame).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .camera_model import CameraModel
from .dynamics import _gelu_tanh


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


# --------------------------------------------------------------------------- lift
def lift_table(camera: CameraModel, grid_shape, resolution, origin, heights, fmap_hw):
    """Bilinear sample table for projecting BEV pillars into a feature map.

    Returns ``(idx, w)``: ``idx`` (K, cells, 4) flat indices into the
    ``Hf * Wf`` feature map and ``w`` (K, cells, 4) bilinear weights, zero where
    the point is behind the camera or outside the image (so invalid samples read
    as zero features)."""
    rows, cols = grid_shape
    hf, wf = fmap_hw
    xs = origin[0] + (np.arange(cols) + 0.5) * resolution
    ys = origin[1] + (np.arange(rows) + 0.5) * resolution
    X, Y = np.meshgrid(xs, ys)
    idx = np.zeros((len(heights), rows * cols, 4), np.int64)
    wts = np.zeros((len(heights), rows * cols, 4), np.float32)
    for k, z in enumerate(heights):
        pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], -1)
        u, v, ok = camera.project(pts)
        # pixel -> feature-map continuous coords (cell centres at +0.5)
        fu = u / camera.width * wf - 0.5
        fv = v / camera.height * hf - 0.5
        u0, v0 = np.floor(fu), np.floor(fv)
        du, dv = fu - u0, fv - v0
        corners = [(0, 0, (1 - du) * (1 - dv)), (0, 1, du * (1 - dv)),
                   (1, 0, (1 - du) * dv), (1, 1, du * dv)]
        for c, (oy, ox, w) in enumerate(corners):
            uu = np.clip(u0 + ox, 0, wf - 1).astype(np.int64)
            vv = np.clip(v0 + oy, 0, hf - 1).astype(np.int64)
            idx[k, :, c] = vv * wf + uu
            wts[k, :, c] = np.where(ok, w, 0.0)
    return idx, wts


def _conv_np(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """'same' conv, x (N, R, C, Cin), W torch layout (Cout, Cin, k, k)."""
    cout, cin, k, _ = W.shape
    if k == 1:
        return x @ W[:, :, 0, 0].T + b
    p = k // 2
    n, r, c, _ = x.shape
    xp = np.pad(x, ((0, 0), (p, p), (p, p), (0, 0)))
    cols = np.concatenate([xp[:, ky:ky + r, kx:kx + c] for ky in range(k) for kx in range(k)],
                          axis=-1)                                   # (N, R, C, k*k*Cin)
    Wm = W.transpose(2, 3, 1, 0).reshape(k * k * cin, cout)
    return cols @ Wm + b


@dataclass
class OccupancyHead:
    """Feature map (+ previous map + ego command) -> (horizon, rows, cols) probability."""

    camera: CameraModel
    ch_mu: np.ndarray              # (C,)    channel PCA mean
    ch_comp: np.ndarray            # (C, c)  channel PCA basis
    ch_std: np.ndarray             # (c,)    std of reduced channels
    layers: list                   # [(W (Cout, Cin, k, k), b), ...]
    horizon: int
    grid_shape: tuple              # (rows, cols)
    resolution: float = 0.5
    origin: tuple = (0.0, -4.0)
    heights: tuple = (0.05, 0.5, 1.0)
    dt: float = 0.2
    ego_dim: int = 2
    ego_scale: float = 1.0
    threshold: float = 0.5

    _tables: dict = field(init=False, default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.ch_mu = np.asarray(self.ch_mu, np.float32)
        self.ch_comp = np.asarray(self.ch_comp, np.float32)
        self.ch_std = np.asarray(self.ch_std, np.float32)
        self.layers = [(np.asarray(W, np.float32), np.asarray(b, np.float32))
                       for (W, b) in self.layers]
        self.horizon = int(self.horizon)
        self.grid_shape = tuple(int(v) for v in self.grid_shape)
        self.origin = tuple(float(v) for v in self.origin)
        self.heights = tuple(float(v) for v in self.heights)

    @property
    def channels(self) -> int:
        return int(self.ch_comp.shape[1])

    @property
    def in_channels(self) -> int:
        k = len(self.heights)
        return 2 * k * self.channels + k + self.ego_dim + 2

    # --- features ---------------------------------------------------------------
    def reduce(self, fmap: np.ndarray) -> np.ndarray:
        """(..., Hf, Wf, C) raw feature map -> (..., Hf, Wf, c) reduced channels."""
        return ((np.asarray(fmap, np.float32) - self.ch_mu) @ self.ch_comp) / self.ch_std

    def _table(self, fmap_hw):
        key = tuple(int(v) for v in fmap_hw)
        if key not in self._tables:
            self._tables[key] = lift_table(self.camera, self.grid_shape, self.resolution,
                                           self.origin, self.heights, key)
        return self._tables[key]

    def _cell_pos(self) -> np.ndarray:
        rows, cols = self.grid_shape
        span_x, span_y = cols * self.resolution, rows * self.resolution
        xs = self.origin[0] + (np.arange(cols) + 0.5) * self.resolution
        ys = self.origin[1] + (np.arange(rows) + 0.5) * self.resolution
        X, Y = np.meshgrid(xs, ys)
        return np.stack([X / span_x, Y / (span_y / 2)], -1).astype(np.float32)

    def bev_input(self, z, z_prev=None, egos=None) -> np.ndarray:
        """Reduced maps (N, Hf, Wf, c) -> BEV network input (N, rows, cols, F)."""
        z = np.asarray(z, np.float32)
        n, hf, wf, c = z.shape
        idx, w = self._table((hf, wf))
        rows, cols = self.grid_shape

        def lift(m):
            flat = m.reshape(n, hf * wf, c)
            g = flat[:, idx]                                    # (N, K, cells, 4, c)
            return (g * w[None, ..., None]).sum(3)              # (N, K, cells, c)

        cur = lift(z)
        diff = cur - lift(np.asarray(z_prev, np.float32)) if z_prev is not None \
            else np.zeros_like(cur)
        valid = np.broadcast_to((w.sum(-1) > 0).astype(np.float32)[None], (n,) + w.shape[:2])
        k = len(self.heights)
        feats = [cur.transpose(0, 2, 1, 3).reshape(n, rows * cols, k * c),
                 diff.transpose(0, 2, 1, 3).reshape(n, rows * cols, k * c),
                 valid.transpose(0, 2, 1)]
        if self.ego_dim:
            e = np.zeros((n, self.ego_dim), np.float32) if egos is None else \
                np.asarray(egos, np.float32).reshape(n, -1)[:, : self.ego_dim]
            feats.append(np.broadcast_to((e * self.ego_scale)[:, None], (n, rows * cols, self.ego_dim)))
        feats.append(np.broadcast_to(self._cell_pos().reshape(1, rows * cols, 2), (n, rows * cols, 2)))
        return np.concatenate(feats, -1).reshape(n, rows, cols, -1).astype(np.float32)

    # --- inference (pure numpy) -------------------------------------------------
    def _net(self, x: np.ndarray) -> np.ndarray:
        for i, (W, b) in enumerate(self.layers):
            x = _conv_np(x, W, b)
            if i < len(self.layers) - 1:
                x = _gelu_tanh(x)
        return x                                                # (N, rows, cols, horizon)

    def predict_proba_reduced(self, z, z_prev=None, egos=None) -> np.ndarray:
        """Reduced maps (N, Hf, Wf, c) -> (N, horizon, rows, cols) probabilities."""
        logits = self._net(self.bev_input(z, z_prev, egos))
        return _sigmoid(logits).transpose(0, 3, 1, 2).astype(np.float32)

    def predict_proba(self, fmap, prev_fmap=None, ego=None) -> np.ndarray:
        """One raw feature map (Hf, Wf, C) -> (horizon, rows, cols) probabilities."""
        z = self.reduce(fmap)[None]
        zp = None if prev_fmap is None else self.reduce(prev_fmap)[None]
        e = None if ego is None else np.asarray(ego, np.float32)[None]
        return self.predict_proba_reduced(z, zp, e)[0]

    def to_occupancy_values(self, proba: np.ndarray) -> np.ndarray:
        """Probabilities -> OccupancyGrid values (int8 0..100), with the stored
        IoU-optimal ``threshold`` mapped to 50."""
        p = np.asarray(proba, np.float32)
        t = float(np.clip(self.threshold, 1e-3, 1 - 1e-3))
        v = np.where(p <= t, 50.0 * p / t, 50.0 + 50.0 * (p - t) / (1.0 - t))
        return np.clip(np.rint(v), 0, 100).astype(np.int8)

    def predict_grids(self, fmap, prev_fmap=None, ego=None) -> list:
        """One observation -> list of ``horizon`` (rows, cols) int8 grids."""
        return list(self.to_occupancy_values(self.predict_proba(fmap, prev_fmap, ego)))

    # --- persistence ------------------------------------------------------------
    def save(self, path: str) -> None:
        flat = {
            "camera_P": self.camera.P, "camera_size": np.int32([self.camera.width, self.camera.height]),
            "ch_mu": self.ch_mu, "ch_comp": self.ch_comp, "ch_std": self.ch_std,
            "horizon": np.int32(self.horizon), "grid_shape": np.int32(self.grid_shape),
            "resolution": np.float32(self.resolution), "origin": np.float32(self.origin),
            "heights": np.float32(self.heights), "dt": np.float32(self.dt),
            "ego_dim": np.int32(self.ego_dim), "ego_scale": np.float32(self.ego_scale),
            "threshold": np.float32(self.threshold), "n_layers": np.int32(len(self.layers)),
        }
        for i, (W, b) in enumerate(self.layers):
            flat[f"W{i}"] = W
            flat[f"b{i}"] = b
        np.savez_compressed(path, **flat)

    @classmethod
    def load(cls, path_or_npz) -> "OccupancyHead":
        z = np.load(path_or_npz) if isinstance(path_or_npz, str) else path_or_npz
        w, h = (int(v) for v in z["camera_size"])
        return cls(
            camera=CameraModel(P=z["camera_P"], width=w, height=h),
            ch_mu=z["ch_mu"], ch_comp=z["ch_comp"], ch_std=z["ch_std"],
            layers=[(z[f"W{i}"], z[f"b{i}"]) for i in range(int(z["n_layers"]))],
            horizon=int(z["horizon"]), grid_shape=tuple(z["grid_shape"].tolist()),
            resolution=float(z["resolution"]), origin=tuple(z["origin"].tolist()),
            heights=tuple(z["heights"].tolist()), dt=float(z["dt"]),
            ego_dim=int(z["ego_dim"]), ego_scale=float(z["ego_scale"]),
            threshold=float(z["threshold"]),
        )

    # --- training (torch, GPU; weights stored as numpy) --------------------------
    @staticmethod
    def fit_channel_pca(fmaps: np.ndarray, channels: int = 32, max_vectors: int = 200_000,
                        seed: int = 0):
        """Channel PCA over feature vectors of sample maps (..., C). Returns
        ``(mu, comp, std)``; ``comp`` is (C, c). When ``channels >= C`` the basis
        is the identity (raw channels, standardised)."""
        x = np.asarray(fmaps, np.float32).reshape(-1, np.shape(fmaps)[-1])
        if len(x) > max_vectors:
            x = x[np.random.default_rng(seed).choice(len(x), max_vectors, replace=False)]
        mu = x.mean(0)
        d = x.shape[1]
        if channels >= d:
            comp = np.eye(d, dtype=np.float32)
        else:
            _, _, vt = np.linalg.svd(x - mu, full_matrices=False)
            comp = vt[:channels].T.astype(np.float32)
        std = ((x - mu) @ comp).std(0) + 1e-6
        return mu, comp, std.astype(np.float32)

    @classmethod
    def fit(
        cls,
        z: np.ndarray,
        z_prev: np.ndarray,
        egos: np.ndarray,
        targets: np.ndarray,
        masks: Optional[np.ndarray] = None,
        *,
        camera: CameraModel,
        channel_pca: tuple,
        resolution: float = 0.5,
        origin: tuple = (0.0, -4.0),
        heights: tuple = (0.05, 0.5, 1.0),
        dt: float = 0.2,
        hidden: int = 64,
        kernel: int = 5,
        epochs: int = 30,
        batch_size: int = 64,
        lr: float = 2e-3,
        weight_decay: float = 1e-4,
        ego_scale: float = 1.0,
        device: str = "cpu",
        seed: int = 0,
        log=None,
    ) -> "OccupancyHead":
        """Train on *reduced* maps ``z``/``z_prev`` (N, Hf, Wf, c) (see
        :meth:`fit_channel_pca` / :meth:`reduce`), (N, horizon, rows, cols) bool
        ``targets`` and optional (N, rows, cols) visibility ``masks`` (loss only
        on visible cells). Returns a numpy-only head; torch is imported here only.
        Call :meth:`calibrate` afterwards to pick the decision threshold."""
        import torch
        import torch.nn as nn

        n = len(z)
        horizon, rows, cols = np.shape(targets)[1:]
        ego_dim = int(np.asarray(egos).reshape(n, -1).shape[1])
        mu, comp, std = channel_pca
        head = cls(camera=camera, ch_mu=mu, ch_comp=comp, ch_std=std, layers=[],
                   horizon=horizon, grid_shape=(rows, cols), resolution=resolution,
                   origin=origin, heights=heights, dt=dt, ego_dim=ego_dim, ego_scale=ego_scale)

        torch.manual_seed(seed)
        f_in = head.in_channels
        p = kernel // 2
        model = nn.Sequential(
            nn.Conv2d(f_in, hidden, 1), nn.GELU(approximate="tanh"),
            nn.Conv2d(hidden, hidden, kernel, padding=p), nn.GELU(approximate="tanh"),
            nn.Conv2d(hidden, hidden, kernel, padding=p), nn.GELU(approximate="tanh"),
            nn.Conv2d(hidden, horizon, 1),
        ).to(device)
        tgt = np.asarray(targets, bool)
        msk = np.ones((n, rows, cols), bool) if masks is None else np.asarray(masks, bool)
        prior = float(np.clip(tgt[np.broadcast_to(msk[:, None], tgt.shape)].mean(), 1e-4, 1 - 1e-4))
        with torch.no_grad():
            model[-1].bias.fill_(float(np.log(prior / (1 - prior))))
        pos_weight = torch.tensor(min(10.0, (1 - prior) / prior), device=device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        steps = max(1, int(epochs) * ((n + batch_size - 1) // batch_size))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)

        rng = np.random.default_rng(seed)
        model.train()
        for ep in range(int(epochs)):
            perm = rng.permutation(n)
            total, count = 0.0, 0
            for s in range(0, n, batch_size):
                ix = np.sort(perm[s:s + batch_size])
                x = head.bev_input(z[ix], z_prev[ix], np.asarray(egos)[ix])
                xt = torch.from_numpy(x).permute(0, 3, 1, 2).to(device)
                yt = torch.from_numpy(tgt[ix].astype(np.float32)).to(device)
                mt = torch.from_numpy(msk[ix].astype(np.float32)).to(device)[:, None]
                logits = model(xt)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, yt, pos_weight=pos_weight, reduction="none")
                loss = (loss * mt).sum() / (mt.sum() * horizon).clamp_min(1.0)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                total += loss.item() * len(ix)
                count += len(ix)
            if log is not None and (ep % 5 == 0 or ep == int(epochs) - 1):
                log(f"  epoch {ep:4d}  masked bce {total / max(count, 1):.4f}")
        model.eval()

        head.layers = [(m.weight.detach().cpu().numpy().astype(np.float32),
                        m.bias.detach().cpu().numpy().astype(np.float32))
                       for m in model if isinstance(m, nn.Conv2d)]
        return head

    def calibrate(self, z, z_prev, egos, targets, masks=None, candidates=None,
                  batch: int = 256) -> float:
        """Pick the probability threshold maximising pooled (visible-cell) IoU on
        held-out data; stores and returns it."""
        proba = np.concatenate([self.predict_proba_reduced(z[s:s + batch], z_prev[s:s + batch],
                                                           np.asarray(egos)[s:s + batch])
                                for s in range(0, len(z), batch)])
        best_t, best_iou = 0.5, -1.0
        for t in candidates if candidates is not None else np.linspace(0.05, 0.95, 19):
            iou = pooled_iou(proba >= t, targets, masks)
            if iou > best_iou:
                best_t, best_iou = float(t), iou
        self.threshold = best_t
        return best_t


def pooled_iou(pred: np.ndarray, ref: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """IoU pooled over every cell of every sample (empty-vs-empty samples don't
    inflate the score the way a per-sample mean would). ``mask`` (broadcastable,
    e.g. (N, rows, cols) visibility for (N, H, rows, cols) grids) restricts the
    count to those cells. Both empty -> 1.0."""
    a, b = np.asarray(pred, bool), np.asarray(ref, bool)
    if mask is not None:
        m = np.asarray(mask, bool)
        if m.ndim == a.ndim - 1:
            m = m[:, None]
        m = np.broadcast_to(m, a.shape)
        a, b = a & m, b & m
    union = int(np.logical_or(a, b).sum())
    return 1.0 if union == 0 else int(np.logical_and(a, b).sum()) / union


def load_head(head: Optional[object]) -> Optional[OccupancyHead]:
    """Accept a head, a path to a saved ``.npz``, or None/empty string."""
    if head is None or head == "":
        return None
    return head if isinstance(head, OccupancyHead) else OccupancyHead.load(str(head))
