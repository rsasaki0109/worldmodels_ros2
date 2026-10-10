"""A learned *latent -> future occupancy* head (PCA + small MLP).

JEPA encoders give latents, but Nav2 consumers (``world_model_costmap``,
``world_model_dwb_critics``) need ``FutureOccupancy``. This head closes that gap:
from the current latent, the change since the previous latent (motion) and the
last ego command, it predicts an occupancy probability per BEV cell for each of
the next ``horizon`` steps.

Same recipe as :class:`world_model_py.dynamics.MLPDynamics`:

  * **PCA first** -- JEPA latents are ~1000-d; a small standardized subspace
    keeps the head tiny and trainable from a few thousand frames.
  * **training uses torch (GPU, e.g. Colab) but inference is pure numpy**, so
    the ROS runtime and CI need no torch, and a trained head is one ``.npz``.

The output is calibrated for the costmap rule: the probability threshold that
maximised IoU on validation data is stored, and :meth:`predict_grids` maps it to
occupancy value 50 -- so a consumer using ``occupied_threshold: 50`` applies the
validated decision boundary.

Grids follow the ``world_model_py.occupancy_sim`` convention: ego-centric at the
observation time, ``grid[i, j]`` = (y row, x column), lower-left corner at
``origin``, ``resolution`` metres per cell.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .dynamics import _gelu_tanh


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


@dataclass
class OccupancyHead:
    """Latent (+ motion + ego command) -> (horizon, G, G) occupancy probability."""

    mu: np.ndarray                 # (D,)   PCA mean
    components: np.ndarray         # (D, K) PCA basis (columns)
    z_std: np.ndarray              # (K,)   std of PCA coordinates
    dz_std: np.ndarray             # (K,)   std of PCA deltas (motion feature)
    layers: list                   # [(W, b), ...] numpy MLP weights
    horizon: int
    grid_shape: tuple              # (rows, cols)
    resolution: float = 0.25
    origin: tuple = (0.0, -4.0)
    dt: float = 0.2
    ego_dim: int = 2
    ego_scale: float = 1.0
    threshold: float = 0.5

    def __post_init__(self) -> None:
        self.mu = np.asarray(self.mu, np.float32)
        self.components = np.asarray(self.components, np.float32)
        self.z_std = np.asarray(self.z_std, np.float32)
        self.dz_std = np.asarray(self.dz_std, np.float32)
        self.layers = [(np.asarray(W, np.float32), np.asarray(b, np.float32))
                       for (W, b) in self.layers]
        self.horizon = int(self.horizon)
        self.grid_shape = tuple(int(v) for v in self.grid_shape)
        self.origin = tuple(float(v) for v in self.origin)

    # --- features ---------------------------------------------------------------
    def _pca(self, latent: np.ndarray) -> np.ndarray:
        return (np.asarray(latent, np.float32) - self.mu) @ self.components

    def features(self, latents, prev_latents=None, egos=None) -> np.ndarray:
        """Batch features (N, 2K + ego_dim). ``prev_latents`` / ``egos`` may be
        None (no history yet / no command) and are then zero-filled."""
        lat = np.atleast_2d(np.asarray(latents, np.float32))
        z = self._pca(lat)
        if prev_latents is None:
            dz = np.zeros_like(z)
        else:
            dz = z - self._pca(np.atleast_2d(np.asarray(prev_latents, np.float32)))
        if egos is None or self.ego_dim == 0:
            ego = np.zeros((lat.shape[0], self.ego_dim), np.float32)
        else:
            ego = np.atleast_2d(np.asarray(egos, np.float32))[:, : self.ego_dim]
        return np.concatenate([z / self.z_std, dz / self.dz_std, ego * self.ego_scale],
                              axis=1).astype(np.float32)

    # --- inference (pure numpy) -------------------------------------------------
    def _mlp(self, h: np.ndarray) -> np.ndarray:
        for i, (W, b) in enumerate(self.layers):
            h = h @ W.T + b
            if i < len(self.layers) - 1:
                h = _gelu_tanh(h)
        return h

    def predict_proba(self, latent, prev_latent=None, ego=None) -> np.ndarray:
        """One observation -> (horizon, rows, cols) float32 probabilities."""
        return self.predict_proba_batch(
            latent, None if prev_latent is None else prev_latent,
            None if ego is None else ego)[0]

    def predict_proba_batch(self, latents, prev_latents=None, egos=None) -> np.ndarray:
        """Batch -> (N, horizon, rows, cols) float32 probabilities."""
        logits = self._mlp(self.features(latents, prev_latents, egos))
        return _sigmoid(logits).reshape(-1, self.horizon, *self.grid_shape).astype(np.float32)

    def to_occupancy_values(self, proba: np.ndarray) -> np.ndarray:
        """Probabilities -> OccupancyGrid values (int8 0..100), with the stored
        IoU-optimal ``threshold`` mapped to 50."""
        p = np.asarray(proba, np.float32)
        t = float(np.clip(self.threshold, 1e-3, 1 - 1e-3))
        v = np.where(p <= t, 50.0 * p / t, 50.0 + 50.0 * (p - t) / (1.0 - t))
        return np.clip(np.rint(v), 0, 100).astype(np.int8)

    def predict_grids(self, latent, prev_latent=None, ego=None) -> list:
        """One observation -> list of ``horizon`` (rows, cols) int8 grids."""
        return list(self.to_occupancy_values(self.predict_proba(latent, prev_latent, ego)))

    # --- persistence ------------------------------------------------------------
    def save(self, path: str) -> None:
        flat = {
            "mu": self.mu, "components": self.components,
            "z_std": self.z_std, "dz_std": self.dz_std,
            "horizon": np.int32(self.horizon),
            "grid_shape": np.asarray(self.grid_shape, np.int32),
            "resolution": np.float32(self.resolution),
            "origin": np.asarray(self.origin, np.float32),
            "dt": np.float32(self.dt),
            "ego_dim": np.int32(self.ego_dim),
            "ego_scale": np.float32(self.ego_scale),
            "threshold": np.float32(self.threshold),
            "n_layers": np.int32(len(self.layers)),
        }
        for i, (W, b) in enumerate(self.layers):
            # float16 halves the file; inference upcasts to float32
            flat[f"W{i}"] = W.astype(np.float16)
            flat[f"b{i}"] = b
        np.savez_compressed(path, **flat)

    @classmethod
    def load(cls, path_or_npz) -> "OccupancyHead":
        z = np.load(path_or_npz) if isinstance(path_or_npz, str) else path_or_npz
        nl = int(z["n_layers"])
        return cls(
            mu=z["mu"], components=z["components"], z_std=z["z_std"], dz_std=z["dz_std"],
            layers=[(z[f"W{i}"], z[f"b{i}"]) for i in range(nl)],
            horizon=int(z["horizon"]), grid_shape=tuple(z["grid_shape"].tolist()),
            resolution=float(z["resolution"]), origin=tuple(z["origin"].tolist()),
            dt=float(z["dt"]), ego_dim=int(z["ego_dim"]), ego_scale=float(z["ego_scale"]),
            threshold=float(z["threshold"]),
        )

    # --- training (torch, GPU; weights stored as numpy) --------------------------
    @classmethod
    def fit(
        cls,
        latents: np.ndarray,
        prev_latents: np.ndarray,
        egos: np.ndarray,
        targets: np.ndarray,
        *,
        resolution: float = 0.25,
        origin: tuple = (0.0, -4.0),
        dt: float = 0.2,
        n_components: int = 64,
        hidden: int = 256,
        epochs: int = 60,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        ego_scale: float = 1.0,
        device: str = "cpu",
        seed: int = 0,
        log=None,
    ) -> "OccupancyHead":
        """Train with torch on (N, horizon, rows, cols) bool ``targets``; return a
        numpy-only head. torch is imported here only. Call :meth:`calibrate`
        afterwards to pick the decision threshold on held-out data."""
        import torch
        import torch.nn as nn

        lat = np.asarray(latents, np.float32)
        prev = np.asarray(prev_latents, np.float32)
        ego = np.asarray(egos, np.float32).reshape(lat.shape[0], -1)
        tgt = np.asarray(targets, np.float32)
        n, d = lat.shape
        horizon, rows, cols = tgt.shape[1:]
        k = int(min(n_components, d, n))

        mu = lat.mean(0)
        _, _, vt = np.linalg.svd(lat - mu, full_matrices=False)
        comp = vt[:k].T.astype(np.float32)
        z = (lat - mu) @ comp
        dz = z - (prev - mu) @ comp
        head = cls(
            mu=mu, components=comp, z_std=z.std(0) + 1e-6, dz_std=dz.std(0) + 1e-6,
            layers=[], horizon=horizon, grid_shape=(rows, cols), resolution=resolution,
            origin=origin, dt=dt, ego_dim=ego.shape[1], ego_scale=ego_scale,
        )
        x = head.features(lat, prev, ego)
        y = tgt.reshape(n, -1)

        torch.manual_seed(seed)
        model = nn.Sequential(
            nn.Linear(x.shape[1], hidden), nn.GELU(approximate="tanh"),
            nn.Linear(hidden, hidden), nn.GELU(approximate="tanh"),
            nn.Linear(hidden, y.shape[1]),
        ).to(device)
        # start from the occupancy prior so early training isn't spent on the bias
        prior = np.clip(y.mean(0), 1e-4, 1 - 1e-4)
        with torch.no_grad():
            model[-1].bias.copy_(torch.tensor(np.log(prior / (1 - prior)), dtype=torch.float32))
        pos = float(y.mean())
        pos_weight = torch.tensor(min(10.0, (1 - pos) / max(pos, 1e-6)), device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, int(epochs)))

        xt = torch.tensor(x, device=device)
        yt = torch.tensor(y, device=device)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        model.train()
        for ep in range(int(epochs)):
            perm = torch.randperm(n, generator=gen).to(device)
            total = 0.0
            for s in range(0, n, batch_size):
                idx = perm[s:s + batch_size]
                opt.zero_grad()
                loss = loss_fn(model(xt[idx]), yt[idx])
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            sched.step()
            if log is not None and (ep % 10 == 0 or ep == int(epochs) - 1):
                log(f"  epoch {ep:4d}  bce {total / n:.4f}")
        model.eval()

        head.layers = [
            (m.weight.detach().cpu().numpy().astype(np.float32),
             m.bias.detach().cpu().numpy().astype(np.float32))
            for m in model if isinstance(m, nn.Linear)
        ]
        return head

    def calibrate(self, latents, prev_latents, egos, targets,
                  candidates=None) -> float:
        """Pick the probability threshold maximising pooled IoU on held-out
        ``targets``; stores and returns it."""
        proba = self.predict_proba_batch(latents, prev_latents, egos)
        tgt = np.asarray(targets, bool)
        best_t, best_iou = 0.5, -1.0
        for t in candidates if candidates is not None else np.linspace(0.05, 0.95, 19):
            iou = pooled_iou(proba >= t, tgt)
            if iou > best_iou:
                best_t, best_iou = float(t), iou
        self.threshold = best_t
        return best_t


def pooled_iou(pred: np.ndarray, ref: np.ndarray) -> float:
    """IoU pooled over every cell of every sample (empty-vs-empty samples don't
    inflate the score the way a per-sample mean would). Both empty -> 1.0."""
    a, b = np.asarray(pred, bool), np.asarray(ref, bool)
    union = int(np.logical_or(a, b).sum())
    return 1.0 if union == 0 else int(np.logical_and(a, b).sum()) / union


def load_head(head: Optional[object]) -> Optional[OccupancyHead]:
    """Accept a head, a path to a saved ``.npz``, or None/empty string."""
    if head is None or head == "":
        return None
    return head if isinstance(head, OccupancyHead) else OccupancyHead.load(str(head))
