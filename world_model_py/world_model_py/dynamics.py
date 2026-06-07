"""A *learned* action-conditioned latent dynamics head (PCA + small MLP).

This is the trainable counterpart of the learning-free ``RetrievalDynamics``.
The retrieval model is great for one-shot *counterfactual* comparison but is not
a smooth simulator: a constant action collapses onto a remembered fixed point,
so you cannot keep *driving*. A tiny learned head fixes that -- it predicts a
smooth latent delta for **any** (latent, action), so constant actions keep
moving and the action genuinely steers the future.

Two design choices make it work on a 16 GB GPU with only a few hundred real
transitions:

  * **PCA first.** JEPA latents are ~1000-d; with so few samples the action
    signal drowns in the latent dimensions and the model ignores steering. We
    project to a small subspace (``n_components``, default 40) where the action
    has relative weight, learn the dynamics there, and map back.
  * **Tiny MLP.** Two hidden layers predicting the *delta* in PCA space.

Crucially, **training uses torch (GPU) but inference is pure numpy**: ``fit``
extracts the weights as numpy arrays, so ``step`` -- and therefore the planner,
DriveSim and the ROS node -- need no torch at all, and a trained world is a
single self-contained ``.npz``. It is a drop-in for the planner / DriveSim (same
``step`` / ``step_with_index`` interface); decode-to-frame still uses the
experience frames via :func:`world_model_py.planning.decode_trajectory`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _gelu_tanh(x: np.ndarray) -> np.ndarray:
    """tanh approximation of GELU (matches torch nn.GELU(approximate='tanh'))."""
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


@dataclass
class MLPDynamics:
    """Learned latent dynamics: PCA encode -> MLP delta -> PCA decode.

    Built by :meth:`fit` from a memory of real ``(latent, action, next_latent)``
    transitions. Inference (:meth:`step`) is pure numpy. Persist / restore a
    trained head with :meth:`save` / :meth:`load`.
    """

    mu: np.ndarray                       # (D,)   PCA mean
    components: np.ndarray               # (D, K) PCA basis (columns)
    layers: list                         # [(W, b), ...] numpy MLP weights
    action_scale: float = 1.0
    action_dim: int = 1

    _comp_t: np.ndarray = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.mu = np.asarray(self.mu, dtype=np.float32)
        self.components = np.asarray(self.components, dtype=np.float32)
        self.layers = [(np.asarray(W, np.float32), np.asarray(b, np.float32))
                       for (W, b) in self.layers]
        self._comp_t = self.components.T.copy()   # (K, D)

    # --- inference (pure numpy) ------------------------------------------------
    def _encode(self, latent: np.ndarray) -> np.ndarray:
        return (latent - self.mu) @ self.components          # (K,)

    def _decode(self, z: np.ndarray) -> np.ndarray:
        return z @ self._comp_t + self.mu                    # (D,)

    def _mlp(self, h: np.ndarray) -> np.ndarray:
        for i, (W, b) in enumerate(self.layers):
            h = h @ W.T + b
            if i < len(self.layers) - 1:
                h = _gelu_tanh(h)
        return h

    def step_with_index(self, latent, action, after: int = -1) -> tuple:
        latent = np.asarray(latent, dtype=np.float32).ravel()
        action = np.asarray(action, dtype=np.float32).ravel()
        z = self._encode(latent)
        inp = np.concatenate([z, action * self.action_scale]).astype(np.float32)
        z_next = z + self._mlp(inp)
        return self._decode(z_next).astype(np.float32), -1

    def step(self, latent, action, after: int = -1) -> np.ndarray:
        return self.step_with_index(latent, action, after)[0]

    # --- persistence -----------------------------------------------------------
    def save(self, path: str) -> None:
        flat = {"mu": self.mu, "components": self.components,
                "action_scale": np.float32(self.action_scale),
                "action_dim": np.int32(self.action_dim),
                "n_layers": np.int32(len(self.layers))}
        for i, (W, b) in enumerate(self.layers):
            flat[f"W{i}"] = W
            flat[f"b{i}"] = b
        np.savez(path, **flat)

    @classmethod
    def load(cls, path_or_npz) -> "MLPDynamics":
        z = np.load(path_or_npz) if isinstance(path_or_npz, str) else path_or_npz
        nl = int(z["n_layers"])
        layers = [(z[f"W{i}"], z[f"b{i}"]) for i in range(nl)]
        return cls(mu=z["mu"], components=z["components"], layers=layers,
                   action_scale=float(z["action_scale"]),
                   action_dim=int(z["action_dim"]))

    # --- training (torch, GPU; weights stored as numpy) ------------------------
    @classmethod
    def fit(
        cls,
        latents: np.ndarray,
        actions: np.ndarray,
        next_latents: np.ndarray,
        *,
        n_components: int = 40,
        hidden: int = 256,
        epochs: int = 2500,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        action_scale: float = 1.0,
        device: str = "cpu",
        seed: int = 0,
    ) -> "MLPDynamics":
        """Train the head with torch; return a numpy-only :class:`MLPDynamics`.

        torch is imported here only -- inference never needs it.
        """
        import torch
        import torch.nn as nn

        lat = np.asarray(latents, dtype=np.float32)
        act = np.asarray(actions, dtype=np.float32)
        nxt = np.asarray(next_latents, dtype=np.float32)
        if act.ndim == 1:
            act = act[:, None]
        n, d = lat.shape
        k = int(min(n_components, d, n))

        mu = lat.mean(0)
        _, _, vt = np.linalg.svd(lat - mu, full_matrices=False)
        comp = vt[:k].T.astype(np.float32)               # (D, K)
        z = (lat - mu) @ comp                            # (N, K)
        zn = (nxt - mu) @ comp
        delta = (zn - z).astype(np.float32)

        torch.manual_seed(seed)
        model = nn.Sequential(
            nn.Linear(k + act.shape[1], hidden), nn.GELU(approximate="tanh"),
            nn.Linear(hidden, hidden), nn.GELU(approximate="tanh"),
            nn.Linear(hidden, k),
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        zt = torch.tensor(z, device=device)
        at = torch.tensor(act * action_scale, device=device)
        dt = torch.tensor(delta, device=device)
        inp = torch.cat([zt, at], dim=-1)
        model.train()
        for _ in range(int(epochs)):
            opt.zero_grad()
            loss = ((model(inp) - dt) ** 2).mean()
            loss.backward()
            opt.step()
        model.eval()

        layers = []
        for m in model:
            if isinstance(m, nn.Linear):
                layers.append((m.weight.detach().cpu().numpy().astype(np.float32),
                               m.bias.detach().cpu().numpy().astype(np.float32)))
        return cls(mu=mu, components=comp, layers=layers,
                   action_scale=float(action_scale), action_dim=int(act.shape[1]))
