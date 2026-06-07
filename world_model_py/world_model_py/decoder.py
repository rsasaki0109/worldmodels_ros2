"""A learned latent -> pixel decoder, so imagined latents become *generated*
frames instead of retrieved ones.

Both the counterfactual demo and DriveSim visualise the imagination by *nearest
neighbour*: for each imagined latent they fetch the closest real frame. That is
honest but limited -- you can only ever see frames that actually happened. This
module trains a small convolutional decoder ``latent -> image`` on the experience
frames, so an imagined latent (including ones that fall *between* remembered
states) is rendered as a synthesised picture. That is the difference between
*retrieving* the future and *generating* it -- a step toward
DIAMOND/Oasis-style world models, at reconstruction quality bounded by the small
public dataset.

Unlike the dynamics head (numpy inference), the decoder is a conv net, so it
needs torch for both training and inference -- it is therefore an *optional*
component, lazy-imported exactly like the JEPA adapters. The retrieval decode
(:func:`world_model_py.planning.decode_trajectory`) stays the torch-free default;
the decoder is opt-in.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _build_module(latent_dim: int, out_size: int, base: int = 6, base_ch: int = 256):
    """latent -> (3, out_size, out_size) conv decoder. out_size must be base*2^k."""
    import torch.nn as nn

    n_up = int(round(np.log2(out_size / base)))
    if base * (2 ** n_up) != out_size:
        raise ValueError(f"out_size {out_size} must be {base}*2^k")

    def up(i, o):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(i, o, 3, 1, 1), nn.GroupNorm(8, o), nn.GELU())

    chans = [base_ch]
    for _ in range(n_up):
        chans.append(max(16, chans[-1] // 2))
    blocks = [up(chans[i], chans[i + 1]) for i in range(n_up)]
    return nn.Sequential(
        nn.Linear(latent_dim, base_ch * base * base),
        nn.Unflatten(1, (base_ch, base, base)),
        *blocks,
        nn.Conv2d(chans[-1], 3, 3, 1, 1), nn.Sigmoid(),
    )


@dataclass
class LatentDecoder:
    """Render a latent to an RGB frame with a learned conv decoder (torch)."""

    mu: np.ndarray                 # (D,) latent mean (normalisation)
    sd: np.ndarray                 # (D,) latent std
    out_size: int
    latent_dim: int
    base_ch: int = 256
    _module: object = field(default=None, repr=False)

    def _norm(self, lat: np.ndarray) -> np.ndarray:
        return ((np.asarray(lat, np.float32) - self.mu) / self.sd).astype(np.float32)

    def decode(self, latents) -> np.ndarray:
        """latent(s) -> uint8 frame(s). Accepts (D,) or (N, D); returns
        (out_size, out_size, 3) or (N, out_size, out_size, 3)."""
        import torch

        arr = np.asarray(latents, np.float32)
        single = arr.ndim == 1
        if single:
            arr = arr[None]
        x = torch.tensor(self._norm(arr))
        dev = next(self._module.parameters()).device
        with torch.no_grad():
            y = self._module(x.to(dev)).cpu().numpy()        # (N,3,H,W)
        img = (np.clip(y.transpose(0, 2, 3, 1), 0, 1) * 255).astype(np.uint8)
        return img[0] if single else img

    # --- training --------------------------------------------------------------
    @classmethod
    def fit(
        cls,
        latents: np.ndarray,
        frames: np.ndarray,
        *,
        epochs: int = 1500,
        lr: float = 2e-3,
        batch: int = 128,
        l1: float = 0.3,
        base_ch: int = 256,
        device: str = "cpu",
        seed: int = 0,
    ) -> "LatentDecoder":
        import torch

        lat = np.asarray(latents, np.float32)
        fr = np.asarray(frames, np.float32) / 255.0
        n, d = lat.shape
        out_size = int(fr.shape[1])
        mu = lat.mean(0)
        sd = lat.std(0) + 1e-6
        x_all = ((lat - mu) / sd).astype(np.float32)

        torch.manual_seed(seed)
        module = _build_module(d, out_size, base_ch=base_ch).to(device)
        opt = torch.optim.Adam(module.parameters(), lr=lr)
        xt = torch.tensor(x_all, device=device)
        yt = torch.tensor(fr.transpose(0, 3, 1, 2), device=device)
        rng = np.random.default_rng(seed)
        idx = np.arange(n)
        module.train()
        for _ in range(int(epochs)):
            rng.shuffle(idx)
            for b in range(0, n, batch):
                bi = idx[b:b + batch]
                opt.zero_grad()
                pred = module(xt[bi])
                diff = pred - yt[bi]
                loss = (diff ** 2).mean() + l1 * diff.abs().mean()
                loss.backward()
                opt.step()
        module.eval()
        self = cls(mu=mu, sd=sd, out_size=out_size, latent_dim=d, base_ch=base_ch)
        self._module = module
        return self

    def psnr(self, latents: np.ndarray, frames: np.ndarray) -> float:
        """Reconstruction PSNR (dB) on (latent, frame) pairs -- the honest
        quality number."""
        rec = self.decode(latents).astype(np.float32) / 255.0
        mse = float(np.mean((rec - np.asarray(frames, np.float32) / 255.0) ** 2))
        return float(-10.0 * np.log10(mse + 1e-12))

    # --- persistence -----------------------------------------------------------
    def save(self, path: str) -> None:
        import torch

        torch.save({"mu": self.mu, "sd": self.sd, "out_size": self.out_size,
                    "latent_dim": self.latent_dim, "base_ch": self.base_ch,
                    "state": self._module.state_dict()}, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "LatentDecoder":
        import torch

        ck = torch.load(path, map_location=device, weights_only=False)
        self = cls(mu=np.asarray(ck["mu"], np.float32), sd=np.asarray(ck["sd"], np.float32),
                   out_size=int(ck["out_size"]), latent_dim=int(ck["latent_dim"]),
                   base_ch=int(ck["base_ch"]))
        module = _build_module(self.latent_dim, self.out_size, base_ch=self.base_ch).to(device)
        module.load_state_dict(ck["state"])
        module.eval()
        self._module = module
        return self
