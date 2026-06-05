"""Read a rosbag2 (sqlite3 or mcap) into per-topic message streams.

Uses rosbag2_py + rosidl deserialization, so it needs a ROS 2 environment
(this is a ROS package). Storage backend is auto-detected from the bag's
metadata, so the same code reads .db3 and .mcap bags.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

import rosbag2_py


@dataclass
class Stamped:
    t_ns: int          # bag receive time (ns)
    msg: object        # deserialized message


def topic_types(bag_path: str) -> dict:
    """topic name -> message type string, e.g. '/odom' -> 'nav_msgs/msg/Odometry'."""
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id=""),
        rosbag2_py.ConverterOptions("", ""),
    )
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    del reader
    return types


def read_topics(bag_path: str, topics: list[str]) -> Iterator[tuple]:
    """Yield (topic, Stamped) for the requested topics, in bag order."""
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id=""),
        rosbag2_py.ConverterOptions("", ""),
    )
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    wanted = set(topics)
    missing = wanted - set(type_map)
    if missing:
        raise ValueError(
            f"topics not in bag: {sorted(missing)}. available: {sorted(type_map)}"
        )
    msg_classes = {name: get_message(type_map[name]) for name in wanted}

    if hasattr(reader, "set_filter"):
        try:
            reader.set_filter(rosbag2_py.StorageFilter(topics=list(wanted)))
        except Exception:  # noqa: BLE001 -- filter is an optimisation only
            pass

    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic not in wanted:
            continue
        yield topic, Stamped(t_ns=int(t_ns), msg=deserialize_message(data, msg_classes[topic]))
