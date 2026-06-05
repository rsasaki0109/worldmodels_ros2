"""Adapter registry: ``load_model("dummy")`` -> a WorldModelAdapter.

This is the ``transformers``-style entry point for the project. New backends
register a factory here (or, later, via package entry points) and become
reachable by name from the CLI, the runtime node and the benchmark.
"""
from __future__ import annotations

from typing import Callable, Dict

from .adapters import DummyAdapter, RemoteAdapter, WorldModelAdapter


def _make_ijepa(**kwargs) -> WorldModelAdapter:
    # Lazy: importing jepa pulls torch/transformers only when actually used.
    from .adapters.jepa import make_ijepa_adapter

    return make_ijepa_adapter(**kwargs)


# name -> factory(**kwargs) -> adapter
_REGISTRY: Dict[str, Callable[..., WorldModelAdapter]] = {
    "dummy": DummyAdapter,
    "remote": RemoteAdapter,
    "ijepa": _make_ijepa,
}


def register(name: str, factory: Callable[..., WorldModelAdapter]) -> None:
    """Register a new adapter factory under ``name`` (overwrites if present)."""
    _REGISTRY[name] = factory


def available_models() -> list[str]:
    return sorted(_REGISTRY)


def load_model(name: str, **kwargs) -> WorldModelAdapter:
    """Instantiate the adapter registered under ``name``.

    Extra keyword arguments are passed to the adapter constructor, e.g.
    ``load_model("remote", url="http://gpu-box:8080/predict_future")``.
    """
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown world model '{name}'. available: {available_models()}"
        ) from None
    return factory(**kwargs)
