"""Message -> vector/image extractors (pure, no ROS; fake message objects)."""
from types import SimpleNamespace as NS

import numpy as np

from world_model_datasets import extractors as ex


def test_image_rgb8():
    data = np.arange(2 * 2 * 3, dtype=np.uint8).tobytes()
    msg = NS(height=2, width=2, encoding="rgb8", data=data)
    rgb = ex.image_to_rgb(msg)
    assert rgb.shape == (2, 2, 3)
    assert rgb[0, 0].tolist() == [0, 1, 2]


def test_image_bgr8_swaps():
    data = bytes([10, 20, 30])  # one bgr pixel
    msg = NS(height=1, width=1, encoding="bgr8", data=data)
    rgb = ex.image_to_rgb(msg)
    assert rgb[0, 0].tolist() == [30, 20, 10]


def test_image_mono8_expands():
    msg = NS(height=1, width=2, encoding="mono8", data=bytes([5, 9]))
    rgb = ex.image_to_rgb(msg)
    assert rgb.shape == (1, 2, 3)
    assert rgb[0, 0].tolist() == [5, 5, 5]


def test_image_unsupported_raises():
    msg = NS(height=1, width=1, encoding="yuv422", data=bytes([0, 0]))
    try:
        ex.image_to_rgb(msg)
        assert False
    except ValueError:
        pass


def test_twist_vector():
    msg = NS(linear=NS(x=1.0, y=2.0, z=3.0), angular=NS(x=4.0, y=5.0, z=6.0))
    assert ex.twist_to_vec(msg).tolist() == [1, 2, 3, 4, 5, 6]


def test_odometry_yaw_and_twist():
    # yaw 90deg -> quaternion (0,0,sqrt2/2,sqrt2/2)
    s = np.sqrt(0.5)
    msg = NS(
        pose=NS(pose=NS(position=NS(x=1.0, y=2.0, z=0.0),
                        orientation=NS(x=0.0, y=0.0, z=s, w=s))),
        twist=NS(twist=NS(linear=NS(x=0.5, y=0.0, z=0.0),
                          angular=NS(x=0.0, y=0.0, z=0.1))),
    )
    v = ex.odometry_to_vec(msg)
    assert v[0] == 1.0 and v[1] == 2.0
    assert abs(v[2] - np.pi / 2) < 1e-5
    assert abs(v[3] - 0.5) < 1e-6 and abs(v[5] - 0.1) < 1e-6


def test_jointstate_prefers_position():
    msg = NS(position=[0.1, 0.2], velocity=[9.0], effort=[])
    assert ex.jointstate_to_vec(msg).tolist() == [
        np.float32(0.1).item(), np.float32(0.2).item()
    ]


def test_vector_extractor_dispatch_and_error():
    assert ex.vector_extractor("geometry_msgs/msg/Twist") is ex.twist_to_vec
    try:
        ex.vector_extractor("foo/msg/Bar")
        assert False
    except ValueError:
        pass


def test_header_stamp_ns():
    msg = NS(header=NS(stamp=NS(sec=2, nanosec=500)))
    assert ex.header_stamp_ns(msg) == 2_000_000_500
    assert ex.header_stamp_ns(NS(x=1)) is None
