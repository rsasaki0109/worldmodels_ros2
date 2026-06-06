"""Logic test for the example adapter (no ROS, no GPU)."""
import numpy as np

from wm_example_adapter import HistogramAdapter, color_histogram
from world_model_py.adapters.base import Observation


def _img(color):
    return np.tile(np.array(color, np.uint8), (16, 16, 1))


def test_histogram_normalized():
    h = color_histogram(_img([255, 0, 0]))
    assert abs(h.sum() - 1.0) < 1e-5
    assert h.shape == (64,)


def test_first_frame_zero_surprise():
    wm = HistogramAdapter()
    pred = wm.predict_future(Observation(image=_img([10, 10, 10])), horizon=3)
    assert pred.risk == 0.0
    assert pred.horizon == 3
    assert pred.risk_label == "example-hist"


def test_same_image_low_surprise():
    wm = HistogramAdapter()
    wm.predict_future(Observation(image=_img([10, 200, 10])))
    pred = wm.predict_future(Observation(image=_img([10, 200, 10])))
    assert pred.risk < 1e-6


def test_color_change_high_surprise():
    wm = HistogramAdapter()
    wm.predict_future(Observation(image=_img([255, 0, 0])))
    pred = wm.predict_future(Observation(image=_img([0, 0, 255])))
    assert pred.risk > 0.9  # entirely different colour bin
