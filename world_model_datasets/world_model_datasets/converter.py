"""rosbag2 -> LeRobot-compatible dataset.

    ros2 run world_model_datasets export_lerobot \
        --bag ./my_bag \
        --image-topic /camera/image_raw \
        --state-topic /odom \
        --action-topic /cmd_vel \
        --fps 10 \
        --out ./hf_dataset

The image topic is the master clock; state/action are nearest-neighbour joined
to each image timestamp (header stamp when present, else bag receive time),
rejecting matches outside --tol-ms. The whole bag becomes one episode.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from . import extractors
from .lerobot_writer import LeRobotWriter
from .rosbag_reader import read_topics, topic_types
from .sync import nearest_indices


def _camera_key(topic: str) -> str:
    base = topic.strip("/").replace("/", "_") or "cam"
    return f"observation.images.{base}"


def _collect(bag, topics):
    streams = {t: [] for t in topics}
    for topic, st in read_topics(bag, topics):
        streams[topic].append(st)
    return streams


def _times(stamped_list, use_header=True):
    out = []
    for st in stamped_list:
        ts = extractors.header_stamp_ns(st.msg) if use_header else None
        out.append(ts if ts else st.t_ns)
    return np.asarray(out, dtype=np.int64)


def convert(
    bag: str,
    image_topic: str,
    out: str,
    state_topic: str | None = None,
    action_topic: str | None = None,
    fps: float = 10.0,
    tol_ms: float = 100.0,
    task: str = "",
    robot_type: str = "unknown",
) -> dict:
    types = topic_types(bag)
    if image_topic not in types:
        raise ValueError(f"image topic '{image_topic}' not in bag ({sorted(types)})")

    wanted = [image_topic] + [t for t in (state_topic, action_topic) if t]
    streams = _collect(bag, wanted)

    imgs = streams[image_topic]
    if not imgs:
        raise ValueError(f"no messages on image topic '{image_topic}'")
    master_t = _times(imgs)
    t0 = int(master_t[0])
    tol_ns = int(tol_ms * 1e6)

    def joined(topic, extractor):
        if not topic:
            return None, None
        lst = streams[topic]
        if not lst:
            raise ValueError(f"no messages on topic '{topic}'")
        idx = nearest_indices(master_t, _times(lst), tol_ns)
        vecs = [extractor(s.msg) for s in lst]
        return idx, vecs

    s_idx, s_vecs = joined(state_topic, extractors.vector_extractor(types[state_topic])) \
        if state_topic else (None, None)
    a_idx, a_vecs = joined(action_topic, extractors.vector_extractor(types[action_topic])) \
        if action_topic else (None, None)

    cam_key = _camera_key(image_topic)
    writer = LeRobotWriter(out, fps=fps, camera_keys=[cam_key], robot_type=robot_type)
    writer.begin_episode(task=task)

    kept = skipped = 0
    for i, st in enumerate(imgs):
        if s_idx is not None and s_idx[i] < 0:
            skipped += 1
            continue
        if a_idx is not None and a_idx[i] < 0:
            skipped += 1
            continue
        state = s_vecs[s_idx[i]] if s_idx is not None else np.zeros(0, np.float32)
        action = a_vecs[a_idx[i]] if a_idx is not None else np.zeros(0, np.float32)
        rgb = extractors.image_to_rgb(st.msg)
        writer.add_frame(
            state=state, action=action,
            images={cam_key: rgb},
            timestamp=(int(master_t[i]) - t0) / 1e9,
        )
        kept += 1

    writer.end_episode()
    writer.close()

    summary = {
        "bag": bag, "out": out, "frames_kept": kept, "frames_skipped": skipped,
        "camera_key": cam_key,
        "state_topic": state_topic, "action_topic": action_topic, "fps": fps,
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="export_lerobot", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bag", required=True, help="rosbag2 directory (sqlite3 or mcap)")
    p.add_argument("--image-topic", required=True)
    p.add_argument("--state-topic", default=None)
    p.add_argument("--action-topic", default=None)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--tol-ms", type=float, default=100.0, help="max sync gap (ms)")
    p.add_argument("--task", default="", help="natural-language task label")
    p.add_argument("--robot-type", default="unknown")
    p.add_argument("--out", required=True, help="output dataset directory")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not os.path.exists(args.bag):
        print(f"bag not found: {args.bag}", file=sys.stderr)
        return 2
    summary = convert(
        bag=args.bag, image_topic=args.image_topic, out=args.out,
        state_topic=args.state_topic, action_topic=args.action_topic,
        fps=args.fps, tol_ms=args.tol_ms, task=args.task, robot_type=args.robot_type,
    )
    print(f"exported {summary['frames_kept']} frames "
          f"(skipped {summary['frames_skipped']}) -> {args.out}")
    print(f"  camera: {summary['camera_key']}  state: {summary['state_topic']}  "
          f"action: {summary['action_topic']}  fps: {summary['fps']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
