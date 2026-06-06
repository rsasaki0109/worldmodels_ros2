"""Runtime anomaly detector (adaptive threshold). No ROS, no GPU."""
from world_model_py.anomaly import AnomalyDetector


def test_nominal_stays_quiet():
    det = AnomalyDetector(window=8, k=4.0, warmup=4, floor=0.02)
    flags = [det.update(0.01)["anomaly"] for _ in range(20)]
    assert not any(flags)


def test_spike_after_calibration_flags():
    det = AnomalyDetector(window=8, k=4.0, warmup=4, floor=0.02)
    for _ in range(10):
        det.update(0.01)               # calibrate on nominal
    assert det.update(0.5)["anomaly"]  # clear outlier -> flagged
    assert det.latched


def test_warmup_suppresses_early_flags():
    det = AnomalyDetector(window=8, k=4.0, warmup=5, floor=0.02)
    # a big value during warmup must not flag (baseline not trusted yet)
    early = [det.update(0.4)["anomaly"] for _ in range(4)]
    assert not any(early)


def test_anomaly_not_folded_into_baseline():
    det = AnomalyDetector(window=6, k=3.0, warmup=3, floor=0.01)
    for _ in range(8):
        det.update(0.02)
    base_thr = det.threshold()
    for _ in range(5):                 # sustained anomaly
        det.update(0.9)
    # threshold barely moves because anomalies are excluded from the baseline
    assert det.threshold() < base_thr * 3


def test_threshold_floor():
    det = AnomalyDetector(window=8, k=4.0, warmup=2, floor=0.05)
    for _ in range(8):
        det.update(0.0)                # zero-variance nominal
    assert det.threshold() == 0.05     # floor prevents hypersensitivity
    assert not det.update(0.04)["anomaly"]
    assert det.update(0.2)["anomaly"]
