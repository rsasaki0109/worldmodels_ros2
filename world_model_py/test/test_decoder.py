"""Tests for the learned latent->pixel decoder.

The decoder is a conv net, so it needs torch for both training and inference;
these tests are skipped where torch is absent (like the JEPA / MLP-fit tests).
They use tiny 24x24 images so the end-to-end fit runs fast on a CPU.

    pytest world_model_py/test/test_decoder.py
"""
import numpy as np
import pytest

pytest.importorskip("torch")

from world_model_py.decoder import LatentDecoder, _build_module


def _toy(n=64, d=16, size=24, seed=0):
    """Latents that linearly control a coloured gradient image, so a decoder
    can actually learn latent -> pixels."""
    rng = np.random.default_rng(seed)
    lat = rng.normal(size=(n, d)).astype(np.float32)
    ramp = np.linspace(0, 1, size, dtype=np.float32)
    frames = np.empty((n, size, size, 3), np.float32)
    for i in range(n):
        frames[i, ..., 0] = ramp[None, :] * (0.5 + 0.5 * np.tanh(lat[i, 0]))
        frames[i, ..., 1] = ramp[:, None] * (0.5 + 0.5 * np.tanh(lat[i, 1]))
        frames[i, ..., 2] = 0.5 + 0.5 * np.tanh(lat[i, 2])
    return lat, (frames * 255).astype(np.uint8)


def test_build_module_rejects_bad_size():
    with pytest.raises(ValueError):
        _build_module(8, out_size=20)        # 20 != 6*2^k


def test_decode_shapes():
    lat, frames = _toy()
    dec = LatentDecoder.fit(lat, frames, epochs=30, device="cpu")
    one = dec.decode(lat[0])
    assert one.shape == (24, 24, 3) and one.dtype == np.uint8
    many = dec.decode(lat[:5])
    assert many.shape == (5, 24, 24, 3)


def test_decoder_learns_and_psnr_improves():
    lat, frames = _toy()
    dec = LatentDecoder.fit(lat, frames, epochs=600, lr=3e-3, device="cpu")
    assert dec.psnr(lat, frames) > 18.0          # recognisable reconstruction


def test_save_load_roundtrip(tmp_path):
    lat, frames = _toy()
    dec = LatentDecoder.fit(lat, frames, epochs=20, device="cpu")
    p = str(tmp_path / "dec.pt")
    dec.save(p)
    back = LatentDecoder.load(p)
    a = dec.decode(lat[:3]).astype(np.int16)
    b = back.decode(lat[:3]).astype(np.int16)
    assert np.abs(a - b).max() <= 1              # identical up to rounding
