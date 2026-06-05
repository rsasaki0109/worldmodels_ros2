"""Benchmark a World Model adapter and emit an evidence HTML report.

The strategic point of this package is *evidence, not vibes*: latency
percentiles, throughput, and (when torch is present) peak VRAM -- written to
a self-contained HTML file. It runs with no ROS install and no GPU; missing
measurements are reported as "n/a" rather than silently skipped.
"""
from __future__ import annotations

import html
import platform
import time
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from .adapters import ActionCondition, Observation
from .registry import load_model


@dataclass
class BenchResult:
    adapter: str
    runs: int
    horizon: int
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_mean: float
    throughput_hz: float
    vram_peak_mb: Optional[float]
    device: str
    remote: bool


def _sample_observation(image_hw=(64, 64)) -> Observation:
    h, w = image_hw
    rng = np.random.default_rng(0)
    return Observation(
        image=(rng.random((h, w, 3)) * 255).astype(np.uint8),
        ego_state=np.array([1.0, 0.0, 0.0, 0.5], dtype=np.float32),
        action_history=rng.standard_normal((4, 2)).astype(np.float32),
        instruction="move forward",
    )


def _read_vram_peak_mb() -> Optional[float]:
    try:
        import torch  # noqa: WPS433 (optional dependency)

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:  # noqa: BLE001 -- torch absent or no CUDA is fine
        return None
    return None


def run_bench(
    adapter_name: str,
    runs: int = 50,
    horizon: int = 8,
    warmup: int = 3,
    **adapter_kwargs,
) -> BenchResult:
    adapter = load_model(adapter_name, **adapter_kwargs)
    obs = _sample_observation()
    action = ActionCondition(
        action=np.zeros((horizon, 2), dtype=np.float32), dt=0.1
    )

    for _ in range(max(0, warmup)):
        adapter.predict_future(obs, action, horizon=horizon)

    samples_ms: list[float] = []
    t_start = time.perf_counter()
    for _ in range(runs):
        t0 = time.perf_counter()
        adapter.predict_future(obs, action, horizon=horizon)
        samples_ms.append((time.perf_counter() - t0) * 1000.0)
    wall = time.perf_counter() - t_start

    arr = np.asarray(samples_ms)
    info = adapter.info()
    return BenchResult(
        adapter=adapter_name,
        runs=runs,
        horizon=horizon,
        latency_ms_p50=float(np.percentile(arr, 50)),
        latency_ms_p95=float(np.percentile(arr, 95)),
        latency_ms_mean=float(arr.mean()),
        throughput_hz=float(runs / wall) if wall > 0 else float("inf"),
        vram_peak_mb=_read_vram_peak_mb(),
        device=str(info.get("device", "cpu")),
        remote=bool(info.get("remote", False)),
    )


def _row(k: str, v) -> str:
    return f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>"


def render_html(result: BenchResult) -> str:
    d = asdict(result)
    d["vram_peak_mb"] = "n/a" if result.vram_peak_mb is None else round(result.vram_peak_mb, 1)
    for key in ("latency_ms_p50", "latency_ms_p95", "latency_ms_mean", "throughput_hz"):
        d[key] = round(d[key], 3)
    rows = "\n".join(_row(k, v) for k, v in d.items())
    rows += _row("python", platform.python_version())
    rows += _row("platform", platform.platform())
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>world_model_ros2 bench: {html.escape(result.adapter)}</title>
<style>
 body{{font-family:system-ui,sans-serif;margin:2rem;color:#1a1a1a}}
 h1{{font-size:1.3rem}} .tag{{color:#666}}
 table{{border-collapse:collapse;margin-top:1rem}}
 th,td{{border:1px solid #ddd;padding:.4rem .8rem;text-align:left}}
 th{{background:#f5f5f5}}
</style></head><body>
<h1>world_model_ros2 &middot; adapter benchmark</h1>
<p class="tag">Evidence, not vibes. Latency / throughput / VRAM for the
<code>{html.escape(result.adapter)}</code> adapter.</p>
<table>{rows}</table>
</body></html>"""


def bench_to_file(adapter_name: str, out_path: str, **kwargs) -> BenchResult:
    result = run_bench(adapter_name, **kwargs)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(render_html(result))
    return result
