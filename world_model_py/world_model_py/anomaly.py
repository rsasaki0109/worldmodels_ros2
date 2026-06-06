"""Runtime anomaly / OOD detector from a World Model's surprise signal.

Recent work uses a World Model's latent prediction error as a *runtime monitor*:
calibrate a threshold on nominal operation, then flag inputs whose error exceeds
it as failures / out-of-distribution — crucially, **without needing failure
data** (e.g. "World Models for Anomaly Detection during Model-Based RL",
arXiv:2503.02552; "Detecting failures without failure data", arXiv:2503.08558;
"Foundational World Models Detect Bimanual Manipulator Failures",
arXiv:2603.06987).

This is the small, ROS-free core: feed it the per-step surprise (cosine latent
distance from the `ijepa`/`vjepa2` adapters) and it maintains a rolling baseline
of *nominal* samples and an adaptive threshold ``mean + k·std``. Samples flagged
as anomalous are not folded back into the baseline, so a sustained anomaly does
not silently recalibrate the monitor.
"""
from __future__ import annotations

from collections import deque

import numpy as np


class AnomalyDetector:
    def __init__(self, window: int = 12, k: float = 4.0, warmup: int = 5, floor: float = 0.02):
        """``window`` nominal samples for the baseline; flag when surprise exceeds
        ``mean + k·std`` (and at least ``floor``); ignore the first ``warmup``
        samples while the baseline fills."""
        self.window = int(window)
        self.k = float(k)
        self.warmup = int(warmup)
        self.floor = float(floor)
        self._baseline = deque(maxlen=self.window)
        self._n = 0
        self.latched = False

    def threshold(self) -> float:
        if len(self._baseline) < 3:
            return self.floor
        arr = np.asarray(self._baseline, dtype=np.float64)
        return max(self.floor, float(arr.mean() + self.k * arr.std()))

    def update(self, surprise: float) -> dict:
        """Push one surprise value; return
        ``{anomaly, threshold, surprise, baseline_mean}``."""
        self._n += 1
        thr = self.threshold()
        ready = self._n > self.warmup and len(self._baseline) >= 3
        anomaly = bool(ready and surprise > thr)
        if anomaly:
            self.latched = True
        else:
            self._baseline.append(float(surprise))   # learn only from nominal
        mean = float(np.mean(self._baseline)) if self._baseline else 0.0
        return {"anomaly": anomaly, "threshold": thr, "surprise": float(surprise),
                "baseline_mean": mean}

    def reset(self) -> None:
        self._baseline.clear()
        self._n = 0
        self.latched = False
