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

Attach a trained :class:`world_model_py.occupancy_head.OccupancyHead`
(``occupancy_head=path.npz``) and the adapter also predicts ``FutureOccupancy``
from the latent, so a real encoder can drive the Nav2 costmap layer / critic.
"""
from __future__ import annotations

from typing import Optional, Protocol

import numpy as np

from ..occupancy_head import load_head
from .base import (
    ActionCondition,
    FuturePrediction,
    Latent,
    Observation,
    WorldModelAdapter,
)


class FrameEncoder(Protocol):
    """Minimal interface the adapter needs from any image encoder.

    Encoders that can drive an occupancy head also implement
    ``embed_with_map(image) -> (latent (D,), feature_map (Hf, Wf, C))`` -- the
    spatial patch tokens the latent is pooled from -- in one forward pass."""

    def embed(self, image_hwc_uint8: np.ndarray) -> np.ndarray: ...

    def info(self) -> dict: ...


def _token_grid(tokens: np.ndarray) -> np.ndarray:
    """(T, C) patch tokens -> (s, s, C); drops a leading CLS token if present."""
    t = tokens.shape[0]
    s = int(round(np.sqrt(t)))
    if s * s != t:
        s = int(round(np.sqrt(t - 1)))
        if s * s != t - 1:
            raise ValueError(f"cannot arrange {t} tokens into a square grid")
        tokens = tokens[1:]
    return tokens.reshape(s, s, -1)


class IJepaAdapter(WorldModelAdapter):
    """Wrap a frame encoder into the World Model contract (latent + surprise)."""

    name = "ijepa"

    def __init__(self, encoder: FrameEncoder, dt: float = 0.1, name: str = "ijepa",
                 occupancy_head=None):
        self._enc = encoder
        self.name = name
        self._prev: Optional[np.ndarray] = None
        self._prev_map: Optional[np.ndarray] = None
        self._head = load_head(occupancy_head)
        # the head was trained at a fixed step; its dt is the truth for the rollout
        self.dt = float(self._head.dt) if self._head is not None else float(dt)

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

        fmap = None
        if self._head is not None:
            if obs.image is None or getattr(obs.image, "size", 0) == 0:
                raise ValueError("ijepa adapter requires obs.image (a camera frame)")
            if not hasattr(self._enc, "embed_with_map"):
                raise TypeError("occupancy_head needs an encoder with embed_with_map()")
            latent, fmap = self._enc.embed_with_map(obs.image)
            latent = np.asarray(latent, dtype=np.float32).ravel()
        else:
            latent = self._latent(obs)
        score, confidence = self._surprise(latent)
        pred = FuturePrediction(
            dt=self.dt,
            latents=[latent.copy() for _ in range(horizon)],   # persistence rollout
            risk=score,
            risk_confidence=confidence,
            risk_label=f"{self.name}-surprise",
        )
        if self._head is not None:
            ego = None
            if obs.action_history is not None and np.size(obs.action_history):
                ego = np.atleast_2d(np.asarray(obs.action_history, np.float32))[-1]
            # the head predicts a fixed horizon; never extrapolate past it
            pred.occupancy = self._head.predict_grids(fmap, self._prev_map, ego)[:horizon]
            self._prev_map = fmap
            pred.occupancy_resolution = self._head.resolution
            pred.occupancy_origin = self._head.origin
        return pred

    def reset(self) -> None:
        self._prev = None
        self._prev_map = None

    def info(self) -> dict:
        d = {"name": self.name, "remote": False, "kind": "jepa-encoder"}
        if self._head is not None:
            d["occupancy_head"] = {"horizon": self._head.horizon,
                                   "grid": list(self._head.grid_shape),
                                   "resolution": self._head.resolution,
                                   "dt": self._head.dt}
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

    def _tokens(self, pixel_values):
        with self._torch.no_grad():
            out = self.model(pixel_values=pixel_values)
        return out.last_hidden_state.float().cpu().numpy()      # (N, tokens, dim)

    def embed(self, image_hwc_uint8: np.ndarray) -> np.ndarray:
        tokens = self._tokens(self._to_pixel_values(image_hwc_uint8))[0]
        latent = tokens.mean(axis=0)
        self._latent_dim = int(latent.shape[-1])
        return latent.astype(np.float32)

    def embed_with_map(self, image_hwc_uint8: np.ndarray):
        """(latent (D,), patch-token map (s, s, D)) from one forward pass. The
        latent is the token mean, identical to :meth:`embed`."""
        tokens = self._tokens(self._to_pixel_values(image_hwc_uint8))[0]
        self._latent_dim = int(tokens.shape[-1])
        return tokens.mean(axis=0).astype(np.float32), _token_grid(tokens).astype(np.float32)

    def embed_map_batch(self, images) -> np.ndarray:
        """Patch-token maps for a list of frames in one forward pass -> (N, s, s, D)."""
        pv = self._torch.cat([self._to_pixel_values(im) for im in images], dim=0)
        tokens = self._tokens(pv)
        return np.stack([_token_grid(t) for t in tokens]).astype(np.float32)

    def info(self) -> dict:
        return {
            "model_id": self.model_id,
            "device": self.device,
            "dtype": str(self._dtype).replace("torch.", ""),
            "latent_dim": self._latent_dim,
            "processor": "hf" if self._processor is not None else "manual",
        }


class PixelEncoder:
    """GPU-free baseline FrameEncoder: the downsampled image itself is the
    feature map (and its flattening the latent). Used to check whether a learned
    encoder actually helps a head, and to exercise the training pipeline with no
    model download."""

    def __init__(self, size: int = 32):
        self.size = int(size)

    def embed_map(self, image_hwc_uint8: np.ndarray) -> np.ndarray:
        """Area-average pool to (size, size, 3), scaled to [0, 1]."""
        img = np.asarray(image_hwc_uint8, np.float32)
        if img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, axis=2)
        h, w = img.shape[:2]
        ys = (np.arange(self.size + 1) * h) // self.size
        xs = (np.arange(self.size + 1) * w) // self.size
        rows = np.add.reduceat(img[:, :, :3], ys[:-1], axis=0) / np.diff(ys)[:, None, None]
        pooled = np.add.reduceat(rows, xs[:-1], axis=1) / np.diff(xs)[None, :, None]
        return (pooled / 255.0).astype(np.float32)

    def embed(self, image_hwc_uint8: np.ndarray) -> np.ndarray:
        return self.embed_map(image_hwc_uint8).ravel()

    def embed_with_map(self, image_hwc_uint8: np.ndarray):
        m = self.embed_map(image_hwc_uint8)
        return m.ravel(), m

    def embed_map_batch(self, images) -> np.ndarray:
        return np.stack([self.embed_map(im) for im in images])

    def info(self) -> dict:
        return {"encoder": "pixels", "size": self.size, "latent_dim": self.size * self.size * 3}


def make_ijepa_adapter(**kwargs) -> IJepaAdapter:
    """Registry factory: build an I-JEPA adapter backed by a real HF encoder.

    Accepts encoder kwargs (model_id, device, dtype, image_size) plus
    ``occupancy_head`` (a trained head ``.npz`` path). Importing the encoder
    (and thus torch/transformers) happens only when this is called.
    """
    adapter_dt = kwargs.pop("dt", 0.1)
    head = kwargs.pop("occupancy_head", None)
    encoder = _HFEncoder(**kwargs)
    return IJepaAdapter(encoder, dt=adapter_dt, occupancy_head=head)


class _VJepa2Encoder:
    """V-JEPA 2 video encoder (loaded via torch.hub) wrapped as a FrameEncoder.

    V-JEPA 2 is a *video* model, so this keeps a rolling buffer of recent frames
    and encodes the whole clip on each call (the encoder interpolates its
    positional encoding, so the clip length need not be the training default).
    torch / torch.hub are imported lazily.
    """

    _MEAN = (0.485, 0.456, 0.406)
    _STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        entry: str = "vjepa2_vit_large",
        repo: str = "facebookresearch/vjepa2",
        device: str = "auto",
        dtype: str = "float16",
        clip_len: int = 16,
        img_size: int = 256,
    ):
        import torch
        from collections import deque

        if clip_len % 2 != 0:
            clip_len += 1  # tubelet_size is 2
        self.entry = entry
        self.clip_len = int(clip_len)
        self.img_size = int(img_size)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._torch = torch
        self._dtype = getattr(torch, dtype) if device == "cuda" else torch.float32

        loaded = torch.hub.load(repo, entry, trust_repo=True)
        self.encoder = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
        self.encoder.to(device=device, dtype=self._dtype).eval()
        self._buffer = deque(maxlen=self.clip_len)
        self._latent_dim = None

    def _prep(self, image_hwc_uint8: np.ndarray) -> np.ndarray:
        from PIL import Image

        img = np.asarray(image_hwc_uint8)
        if img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, axis=2)
        pil = Image.fromarray(img.astype(np.uint8)[:, :, :3]).resize(
            (self.img_size, self.img_size), Image.BILINEAR
        )
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        arr = (arr - np.array(self._MEAN, np.float32)) / np.array(self._STD, np.float32)
        return np.transpose(arr, (2, 0, 1))  # (3, H, W)

    def _clip_tokens(self, image_hwc_uint8: np.ndarray) -> np.ndarray:
        torch = self._torch
        self._buffer.append(self._prep(image_hwc_uint8))
        frames = list(self._buffer)
        while len(frames) < self.clip_len:          # pad front by repeating oldest
            frames.insert(0, frames[0])
        clip = np.stack(frames, axis=1)             # (3, T, H, W)
        x = torch.from_numpy(clip[None]).to(self.device, self._dtype)  # (1, 3, T, H, W)
        with torch.no_grad():
            out = self.encoder(x)                    # (1, num_tokens, dim)
        tokens = out.squeeze(0).float().cpu().numpy()
        self._latent_dim = int(tokens.shape[-1])
        return tokens

    def embed(self, image_hwc_uint8: np.ndarray) -> np.ndarray:
        return self._clip_tokens(image_hwc_uint8).mean(axis=0).astype(np.float32)

    def embed_with_map(self, image_hwc_uint8: np.ndarray):
        """(clip latent, spatial map of the *latest* tubelet (s, s, D))."""
        tokens = self._clip_tokens(image_hwc_uint8)
        s = self.img_size // 16                      # patch size 16
        grid = tokens.reshape(-1, s, s, tokens.shape[-1])[-1]
        return tokens.mean(axis=0).astype(np.float32), grid.astype(np.float32)

    def info(self) -> dict:
        return {
            "entry": self.entry,
            "device": self.device,
            "dtype": str(self._dtype).replace("torch.", ""),
            "clip_len": self.clip_len,
            "latent_dim": self._latent_dim,
        }


def make_vjepa2_adapter(**kwargs) -> IJepaAdapter:
    """Registry factory: a V-JEPA 2 video adapter (camera clip -> latent ->
    surprise). Imports torch / torch.hub only when called."""
    adapter_dt = kwargs.pop("dt", 0.1)
    head = kwargs.pop("occupancy_head", None)
    encoder = _VJepa2Encoder(**kwargs)
    return IJepaAdapter(encoder, dt=adapter_dt, name="vjepa2", occupancy_head=head)
