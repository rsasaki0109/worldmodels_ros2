"""End-to-end: synthesise a rosbag2, convert it, validate the dataset.

Needs a ROS 2 environment (rosbag2_py). Runs under colcon test.
"""
import json
import os

import numpy as np
import pyarrow.parquet as pq

import rosbag2_py
from rclpy.serialization import serialize_message

from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

from world_model_datasets.converter import convert

IMG_TOPIC = "/camera/image_raw"
ODOM_TOPIC = "/odom"
CMD_TOPIC = "/cmd_vel"


def _stamp(msg, t_ns):
    msg.header.stamp.sec = t_ns // 1_000_000_000
    msg.header.stamp.nanosec = t_ns % 1_000_000_000


def _image(t_ns, k):
    m = Image()
    _stamp(m, t_ns)
    m.height, m.width, m.encoding = 32, 48, "rgb8"
    m.step = m.width * 3
    m.data = ((np.ones((32, 48, 3), np.uint8) * (k * 7 % 255))).tobytes()
    return m


def _odom(t_ns, k):
    m = Odometry()
    _stamp(m, t_ns)
    m.pose.pose.position.x = float(k)
    m.pose.pose.orientation.w = 1.0
    m.twist.twist.linear.x = 0.5
    return m


def _cmd(k):
    m = Twist()
    m.linear.x = 0.5
    m.angular.z = 0.1 * k
    return m


def _make_bag(path, n=8):
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=path, storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    for tid, (name, typ) in enumerate((
        (IMG_TOPIC, "sensor_msgs/msg/Image"),
        (ODOM_TOPIC, "nav_msgs/msg/Odometry"),
        (CMD_TOPIC, "geometry_msgs/msg/Twist"),
    )):
        writer.create_topic(
            rosbag2_py.TopicMetadata(
                id=tid, name=name, type=typ, serialization_format="cdr"
            )
        )
    dt = 100_000_000  # 0.1 s
    for k in range(n):
        t = 1_000_000_000 + k * dt
        writer.write(IMG_TOPIC, serialize_message(_image(t, k)), t)
        writer.write(ODOM_TOPIC, serialize_message(_odom(t + 5_000_000, k)), t + 5_000_000)
        writer.write(CMD_TOPIC, serialize_message(_cmd(k)), t + 3_000_000)
    del writer


def test_convert_end_to_end(tmp_path):
    bag = str(tmp_path / "bag")
    out = str(tmp_path / "ds")
    _make_bag(bag, n=8)

    summary = convert(
        bag=bag, image_topic=IMG_TOPIC, out=out,
        state_topic=ODOM_TOPIC, action_topic=CMD_TOPIC,
        fps=10.0, tol_ms=50.0, task="drive", robot_type="turtlebot",
    )
    assert summary["frames_kept"] == 8
    assert summary["frames_skipped"] == 0

    info = json.load(open(os.path.join(out, "meta", "info.json")))
    assert info["total_frames"] == 8
    assert info["features"]["observation.state"]["shape"] == [6]   # odom vec
    assert info["features"]["action"]["shape"] == [6]              # twist vec

    table = pq.read_table(os.path.join(out, "data", "chunk-000", "episode_000000.parquet"))
    assert table.num_rows == 8
    # state x increments with k (odometry position.x = k)
    xs = [row[0] for row in table.column("observation.state").to_pylist()]
    assert xs == sorted(xs)
    mp4 = os.path.join(out, "videos", "chunk-000",
                       "observation.images.camera_image_raw", "episode_000000.mp4")
    assert os.path.exists(mp4)


def test_convert_rejects_out_of_tolerance(tmp_path):
    bag = str(tmp_path / "bag2")
    out = str(tmp_path / "ds2")
    _make_bag(bag, n=5)
    # tiny tolerance -> odom is 5ms off the image stamp, so all frames dropped
    summary = convert(
        bag=bag, image_topic=IMG_TOPIC, out=out,
        state_topic=ODOM_TOPIC, action_topic=None,
        fps=10.0, tol_ms=1.0, task="", robot_type="x",
    )
    assert summary["frames_kept"] == 0
    assert summary["frames_skipped"] == 5
