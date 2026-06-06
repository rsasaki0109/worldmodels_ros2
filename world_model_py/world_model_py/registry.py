"""Adapter registry: ``load_model("dummy")`` -> a WorldModelAdapter.

This is the ``transformers``-style entry point for the project. Backends become
reachable by name from the CLI, the runtime node and the benchmark in three ways:

1. built-in adapters (dummy / remote / ijepa / vjepa2),
2. in-process ``register("name", factory)``,
3. **package entry points** — an external pip package exposes adapters under the
   ``world_model_ros2.adapters`` group, so installing it makes new World Models
   available with no edits here. This is what makes it an adapter *hub*.

   In your package's pyproject.toml:

       [project.entry-points."world_model_ros2.adapters"]
       mymodel = "my_pkg.adapter:make_my_adapter"
"""
from __future__ import annotations

import warnings
from typing import Callable, Dict

from .adapters import DummyAdapter, RemoteAdapter, WorldModelAdapter

ENTRY_POINT_GROUP = "world_model_ros2.adapters"


def _make_ijepa(**kwargs) -> WorldModelAdapter:
    # Lazy: importing jepa pulls torch/transformers only when actually used.
    from .adapters.jepa import make_ijepa_adapter

    return make_ijepa_adapter(**kwargs)


def _make_vjepa2(**kwargs) -> WorldModelAdapter:
    # Lazy: pulls torch / torch.hub V-JEPA 2 only when actually used.
    from .adapters.jepa import make_vjepa2_adapter

    return make_vjepa2_adapter(**kwargs)


# built-in name -> factory(**kwargs) -> adapter
_BUILTIN: Dict[str, Callable[..., WorldModelAdapter]] = {
    "dummy": DummyAdapter,
    "remote": RemoteAdapter,
    "ijepa": _make_ijepa,
    "vjepa2": _make_vjepa2,
}

# adapters added at runtime via register()
_EXTRA: Dict[str, Callable[..., WorldModelAdapter]] = {}

# cache of entry-point-discovered factories (None until first discovery)
_DISCOVERED: Dict[str, Callable[..., WorldModelAdapter]] | None = None


def _discover_entry_points() -> Dict[str, Callable[..., WorldModelAdapter]]:
    """Load adapter factories advertised by installed packages. Cached; a single
    bad entry point warns rather than breaking the whole registry."""
    from importlib.metadata import entry_points

    found: Dict[str, Callable[..., WorldModelAdapter]] = {}
    try:
        eps = entry_points(group=ENTRY_POINT_GROUP)
    except TypeError:  # very old importlib.metadata
        eps = entry_points().get(ENTRY_POINT_GROUP, [])  # type: ignore[attr-defined]
    for ep in eps:
        try:
            found[ep.name] = ep.load()
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"failed to load adapter entry point '{ep.name}': {exc}")
    return found


def _registry() -> Dict[str, Callable[..., WorldModelAdapter]]:
    global _DISCOVERED
    if _DISCOVERED is None:
        _DISCOVERED = _discover_entry_points()
    # precedence: discovered < built-in < explicit register()
    return {**_DISCOVERED, **_BUILTIN, **_EXTRA}


def register(name: str, factory: Callable[..., WorldModelAdapter]) -> None:
    """Register a new adapter factory under ``name`` (overwrites if present)."""
    _EXTRA[name] = factory


def available_models() -> list[str]:
    return sorted(_registry())


def load_model(name: str, **kwargs) -> WorldModelAdapter:
    """Instantiate the adapter registered under ``name``.

    Extra keyword arguments are passed to the adapter factory, e.g.
    ``load_model("remote", url="http://gpu-box:8080/predict_future")``.
    """
    try:
        factory = _registry()[name]
    except KeyError:
        raise KeyError(
            f"unknown world model '{name}'. available: {available_models()}"
        ) from None
    return factory(**kwargs)
