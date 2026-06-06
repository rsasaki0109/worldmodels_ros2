"""Validate the rosbag2 -> LeRobot converter on PUBLIC real robot data.

Downloads a real LeRobot SO-101 pick-and-place episode (real camera video + real
6-DoF joint state + action from Hugging Face), builds a rosbag2 from it, runs
``export_lerobot``, and checks that the real values survive the round-trip.

Needs network (Hugging Face) + ffmpeg + a ROS 2 environment. Not in CI.

    python3 world_model_datasets/test/validate_real_lerobot.py
"""
import glob
import json
import os
import subprocess
import sys
import tempfile
import urllib.request

import numpy as np
import pyarrow.parquet as pq
from PIL import Image as PILImage

import rosbag2_py
from rclpy.serialization import serialize_message
from sensor_msgs.msg import Image, JointState

from world_model_datasets.converter import convert

REPO = "https://huggingface.co/datasets/lerobot/svla_so101_pickplace/resolve/main"
VIDEO = REPO + "/videos/observation.images.side/chunk-000/file-000.mp4"
PARQUET = REPO + "/data/chunk-000/file-000.parquet"
N = 24
IMG_T, STATE_T, ACT_T = "/camera/image_raw", "/joint_states", "/target_joints"


def _stamp(msg, tns):
    msg.header.stamp.sec = tns // 10**9
    msg.header.stamp.nanosec = tns % 10**9


def main():
    tmp = tempfile.mkdtemp()
    mp4 = os.path.join(tmp, "v.mp4")
    parq = os.path.join(tmp, "d.parquet")
    print("downloading public LeRobot SO-101 episode ...")
    urllib.request.urlretrieve(VIDEO, mp4)
    urllib.request.urlretrieve(PARQUET, parq)

    fr = os.path.join(tmp, "fr")
    os.makedirs(fr)
    subprocess.run(["ffmpeg", "-y", "-i", mp4, "-frames:v", str(N),
                    os.path.join(fr, "%03d.png")], check=True, capture_output=True)
    frames = [np.asarray(PILImage.open(f).convert("RGB")) for f in sorted(glob.glob(fr + "/*.png"))][:N]
    h, w = frames[0].shape[:2]

    t = pq.read_table(parq)
    state = t.column("observation.state").to_pylist()[:N]
    action = t.column("action").to_pylist()[:N]
    stamps = t.column("timestamp").to_pylist()[:N]

    bag = os.path.join(tmp, "bag")
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=bag, storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    for tid, (name, ty) in enumerate([(IMG_T, "sensor_msgs/msg/Image"),
                                      (STATE_T, "sensor_msgs/msg/JointState"),
                                      (ACT_T, "sensor_msgs/msg/JointState")]):
        writer.create_topic(rosbag2_py.TopicMetadata(
            id=tid, name=name, type=ty, serialization_format="cdr"))
    base = 1_000_000_000
    for k in range(N):
        tns = base + int(stamps[k] * 1e9)
        im = Image(); _stamp(im, tns)
        im.height, im.width, im.encoding, im.step = h, w, "rgb8", w * 3
        im.data = np.ascontiguousarray(frames[k].astype(np.uint8)).tobytes()
        writer.write(IMG_T, serialize_message(im), tns)
        for topic, vec in ((STATE_T, state[k]), (ACT_T, action[k])):
            js = JointState(); _stamp(js, tns); js.position = [float(x) for x in vec]
            writer.write(topic, serialize_message(js), tns)
    del writer

    out = os.path.join(tmp, "ds")
    summary = convert(bag=bag, image_topic=IMG_T, out=out, state_topic=STATE_T,
                      action_topic=ACT_T, fps=30.0, tol_ms=50.0,
                      task="pick and place", robot_type="so101")

    info = json.load(open(os.path.join(out, "meta", "info.json")))
    ot = pq.read_table(os.path.join(out, "data", "chunk-000", "episode_000000.parquet"))
    out_state = np.array(ot.column("observation.state").to_pylist())
    out_action = np.array(ot.column("action").to_pylist())
    mp4_out = os.path.join(out, "videos", "chunk-000",
                           "observation.images.camera_image_raw", "episode_000000.mp4")
    state_ok = np.allclose(out_state, np.array(state), atol=1e-3)
    action_ok = np.allclose(out_action, np.array(action), atol=1e-3)

    print("=== rosbag2 -> LeRobot on REAL public data (LeRobot SO-101) ===")
    print(f"input            : real {w}x{h} frames + real 6-DoF joint state/action, {N} frames")
    print(f"frames_kept      : {summary['frames_kept']} (skipped {summary['frames_skipped']})")
    print(f"info features    : state{info['features']['observation.state']['shape']} "
          f"action{info['features']['action']['shape']} fps={info['fps']}")
    print(f"state round-trip : {'MATCH' if state_ok else 'MISMATCH'}")
    print(f"action round-trip: {'MATCH' if action_ok else 'MISMATCH'}")
    print(f"video written    : {os.path.exists(mp4_out)}")
    ok = (summary["frames_kept"] == N and state_ok and action_ok
          and os.path.exists(mp4_out) and ot.num_rows == N)
    print("VALIDATION_OK" if ok else "VALIDATION_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
