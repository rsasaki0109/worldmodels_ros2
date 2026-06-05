"""JEPA latent adapter: camera frame -> latent embedding -> surprise score.

The encoder is a self-supervised JEPA image model (I-JEPA today; V-JEPA2 video
weights slot into the same place once available). The adapter turns a stream of
observations into:
  * a latent embedding per frame (FutureState latents),
  * a temporal *surprise* score = how far the current latent moved from the
    previous one (cosine distance), surfaced as RiskScore.

The encoder is deliberately split out (``_HFEncoder``) and lazily imports
torch/transformers, so:
  * ``import world_model_py`` never pulls in torch,
  * the surprise / rollout logic is unit-tested with a fake encoder (no GPU),
  * the real model is exercised by a separate GPU verification script.

This is an *encoder*, not a dynamics model: ``predict_future`` returns a
persistence rollout (latent held constant over the horizon). Action-conditioned
prediction arrives with V-JEPA2-AC; the contract does not change.
"""
from __future__ import annotations

from typing import Optional, Protocol

import numpy as np

from .base import (
    ActionCondition,
    FuturePrediction,
    Latent,
    Observation,
    WorldModelAdapter,
)


class FrameEncoder(Protocol):
    """Minimal interface the adapter needs from any image encoder."""

    def embed(self, image_hwc_uint8: np.ndarray) -> np.ndarray: ...

    def info(self) -> dict: ...


class IJepaAdapter(WorldModelAdapter):
    """Wrap a frame encoder into the World Model contract (latent + surprise)."""

    name = "ijepa"

    def __init__(self, encoder: FrameEncoder, dt: float = 0.1):
        self._enc = encoder
        self.dt = float(dt)
        self._prev: Optional[np.ndarray] = None

    def _latent(self, obs: Observation) -> np.ndarray:
        if obs.image is None or getattr(obs.image, "size", 0) == 0:
            raise ValueError("ijepa adapter requires obs.image (a camera frame)")
        return np.asarray(self._enc.embed(obs.image), dtype=np.float32).ravel()

    def encode(self, obs: Observation) -> Latent:
        return Latent(data=self._latent(obs), encoding=self.name)

    def _surprise(self, latent: np.ndarray) -> tuple:
        """Cosine distance from the previous latent. Returns (score, confidence)."""
        if self._prev is None:
            score, confidence = 0.0, 0.0          # no baseline yet
        else:
            a, b = latent, self._prev
            denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
            cos = float(np.dot(a, b) / denom)
            score, confidence = float(np.clip(1.0 - cos, 0.0, 1.0)), 0.9
        self._prev = latent
        return score, confidence

    def predict_future(
        self,
        obs: Observation,
        action: Optional[ActionCondition] = None,
        horizon: int = 8,
    ) -> FuturePrediction:
        if action is not None and action.horizon > 0:
            horizon = action.horizon
        horizon = max(1, int(horizon))

        latent = self._latent(obs)
        score, confidence = self._surprise(latent)
        return FuturePrediction(
            dt=self.dt,
            latents=[latent.copy() for _ in range(horizon)],   # persistence rollout
            risk=score,
            risk_confidence=confidence,
            risk_label="ijepa-surprise",
        )

    def reset(self) -> None:
        self._prev = None

    def info(self) -> dict:
        d = {"name": self.name, "remote": False, "kind": "jepa-encoder"}
        try:
            d.update(self._enc.info())
        except Exception:  # noqa: BLE001
            pass
        return d


class _HFEncoder:
    """I-JEPA (or any HF image model) wrapped as a FrameEncoder. Loads torch +
    transformers lazily so the rest of the package stays import-light."""

    # ImageNet stats, used by the manual-preprocess fallback.
    _MEAN = (0.485, 0.456, 0.406)
    _STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        model_id: str = "facebook/ijepa_vith14_1k",
        device: str = "auto",
        dtype: str = "float16",
        image_size: int = 224,
    ):
        import torch
        from transformers import AutoModel

        self.model_id = model_id
        self.image_size = int(image_size)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._torch = torch
        torch_dtype = getattr(torch, dtype) if device == "cuda" else torch.float32
        self._dtype = torch_dtype

        self.model = AutoModel.from_pretrained(model_id, torch_dtype=torch_dtype)
        self.model.to(device).eval()

        # Prefer the model's own image processor; fall back to manual transform.
        self._processor = None
        try:
            from transformers import AutoImageProcessor

            self._processor = AutoImageProcessor.from_pretrained(model_id)
        except Exception:  # noqa: BLE001
            self._processor = None

        self._latent_dim = None

    def _to_pixel_values(self, image_hwc_uint8: np.ndarray):
        torch = self._torch
        img = np.asarray(image_hwc_uint8)
        if img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, axis=2)
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)

        if self._processor is not None:
            from PIL import Image

            pil = Image.fromarray(img.astype(np.uint8)[:, :, :3])
            out = self._processor(images=pil, return_tensors="pt")
            return out["pixel_values"].to(self.device, self._dtype)

        # manual: resize + normalise (no torchvision dependency required path).
        from PIL import Image

        pil = Image.fromarray(img.astype(np.uint8)[:, :, :3]).resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        arr = (arr - np.array(self._MEAN, np.float32)) / np.array(self._STD, np.float32)
        chw = np.transpose(arr, (2, 0, 1))[None]  # (1,3,H,W)
        return torch.from_numpy(chw).to(self.device, self._dtype)

    def embed(self, image_hwc_uint8: np.ndarray) -> np.ndarray:
        torch = self._torch
        pixel_values = self._to_pixel_values(image_hwc_uint8)
        with torch.no_grad():
            out = self.model(pixel_values=pixel_values)
        hidden = out.last_hidden_state  # (1, tokens, dim)
        latent = hidden.mean(dim=1).squeeze(0).float().cpu().numpy()
        self._latent_dim = int(latent.shape[-1])
        return latent.astype(np.float32)

    def info(self) -> dict:
        return {
            "model_id": self.model_id,
            "device": self.device,
            "dtype": str(self._dtype).replace("torch.", ""),
            "latent_dim": self._latent_dim,
            "processor": "hf" if self._processor is not None else "manual",
        }


def make_ijepa_adapter(**kwargs) -> IJepaAdapter:
    """Registry factory: build an I-JEPA adapter backed by a real HF encoder.

    Accepts encoder kwargs (model_id, device, dtype, image_size). Importing the
    encoder (and thus torch/transformers) happens only when this is called.
    """
    adapter_dt = kwargs.pop("dt", 0.1)
    encoder = _HFEncoder(**kwargs)
    return IJepaAdapter(encoder, dt=adapter_dt)
