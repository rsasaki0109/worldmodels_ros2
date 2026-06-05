"""LeRobotWriter output structure (no ROS; uses pyarrow + ffmpeg)."""
import json
import os
import shutil
import subprocess

import numpy as np
import pyarrow.parquet as pq

from world_model_datasets.lerobot_writer import LeRobotWriter

CAM = "observation.images.cam"


def _write_dataset(tmp, episodes=1, frames=5):
    w = LeRobotWriter(tmp, fps=10.0, camera_keys=[CAM], robot_type="testbot")
    rng = np.random.default_rng(0)
    for _ in range(episodes):
        w.begin_episode(task="drive")
        for _ in range(frames):
            w.add_frame(
                state=rng.standard_normal(6).astype(np.float32),
                action=rng.standard_normal(2).astype(np.float32),
                images={CAM: (rng.random((48, 64, 3)) * 255).astype(np.uint8)},
            )
        w.end_episode()
    w.close()
    return w


def test_layout_and_counts(tmp_path):
    tmp = str(tmp_path / "ds")
    _write_dataset(tmp, episodes=2, frames=4)

    assert os.path.exists(os.path.join(tmp, "meta", "info.json"))
    assert os.path.exists(os.path.join(tmp, "meta", "tasks.jsonl"))
    assert os.path.exists(os.path.join(tmp, "meta", "episodes.jsonl"))
    assert os.path.exists(os.path.join(tmp, "meta", "stats.json"))

    info = json.load(open(os.path.join(tmp, "meta", "info.json")))
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 8
    assert info["robot_type"] == "testbot"
    assert info["features"]["observation.state"]["shape"] == [6]
    assert info["features"]["action"]["shape"] == [2]
    assert info["features"][CAM]["dtype"] == "video"


def test_parquet_roundtrip(tmp_path):
    tmp = str(tmp_path / "ds")
    _write_dataset(tmp, episodes=1, frames=5)
    p = os.path.join(tmp, "data", "chunk-000", "episode_000000.parquet")
    table = pq.read_table(p)
    assert table.num_rows == 5
    cols = set(table.column_names)
    assert {"observation.state", "action", "timestamp", "frame_index",
            "episode_index", "index", "next.done", "task_index"} <= cols
    df = table.to_pandas()
    assert df["next.done"].tolist() == [False, False, False, False, True]
    assert len(df["observation.state"].iloc[0]) == 6


def test_video_is_playable(tmp_path):
    if shutil.which("ffprobe") is None:
        return
    tmp = str(tmp_path / "ds")
    _write_dataset(tmp, episodes=1, frames=6)
    mp4 = os.path.join(tmp, "videos", "chunk-000", CAM, "episode_000000.mp4")
    assert os.path.exists(mp4)
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,width,height", "-of", "csv=p=0", mp4],
        capture_output=True, text=True,
    )
    assert out.returncode == 0
    assert "h264" in out.stdout


def test_stats_present(tmp_path):
    tmp = str(tmp_path / "ds")
    _write_dataset(tmp, episodes=1, frames=5)
    stats = json.load(open(os.path.join(tmp, "meta", "stats.json")))
    assert "observation.state" in stats and "action" in stats
    assert len(stats["observation.state"]["mean"]) == 6
    assert stats["observation.state"]["count"] == [5]
