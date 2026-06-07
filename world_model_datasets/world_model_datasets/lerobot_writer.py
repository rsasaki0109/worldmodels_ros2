"""Write frames into a LeRobot-compatible dataset (v2.1 layout).

Pure Python: pyarrow for parquet, ffmpeg for video, json for metadata. No ROS,
no torch, no lerobot package -- so it is unit-testable on its own.

Honesty note: this produces the v2.1 *layout* (meta/info.json, meta/tasks.jsonl,
meta/episodes.jsonl, meta/episodes_stats.jsonl, meta/stats.json,
data/chunk-*/episode_*.parquet, videos/chunk-*/<key>/episode_*.mp4). Validated
structurally and, when ``lerobot`` is installed, via ``test/validate_lerobot_loader.py``
(v2.1 -> v3.0 convert + ``LeRobotDataset`` load).
"""
from __future__ import annotations

import json
import os
import subprocess
from typing import Dict, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CODEBASE_VERSION = "v2.1"
CHUNK = "chunk-000"


class _VideoEncoder:
    """One ffmpeg process per camera per episode, fed raw RGB frames."""

    def __init__(self, path: str, fps: float):
        self.path = path
        self.fps = float(fps)
        self._proc = None
        self._shape = None
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def add(self, rgb: np.ndarray) -> None:
        rgb = np.ascontiguousarray(rgb.astype(np.uint8))
        if self._proc is None:
            h, w = rgb.shape[:2]
            self._shape = (h, w)
            self._proc = subprocess.Popen(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{w}x{h}", "-r", f"{self.fps}", "-i", "-",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    self.path,
                ],
                stdin=subprocess.PIPE,
            )
        elif rgb.shape[:2] != self._shape:
            raise ValueError(
                f"frame size {rgb.shape[:2]} != episode size {self._shape} for {self.path}"
            )
        self._proc.stdin.write(rgb.tobytes())

    def close(self) -> None:
        if self._proc is not None:
            self._proc.stdin.close()
            rc = self._proc.wait()
            self._proc = None
            if rc != 0:
                raise RuntimeError(f"ffmpeg failed ({rc}) writing {self.path}")


class _RunningStats:
    def __init__(self):
        self.n = 0
        self.lo = None
        self.hi = None
        self.sum = None
        self.sumsq = None

    def update(self, v: np.ndarray) -> None:
        v = np.asarray(v, dtype=np.float64)
        if self.n == 0:
            self.lo, self.hi = v.copy(), v.copy()
            self.sum, self.sumsq = v.copy(), v * v
        else:
            self.lo = np.minimum(self.lo, v)
            self.hi = np.maximum(self.hi, v)
            self.sum += v
            self.sumsq += v * v
        self.n += 1

    def as_dict(self) -> dict:
        if self.n == 0:
            return {}
        mean = self.sum / self.n
        var = np.maximum(self.sumsq / self.n - mean * mean, 0.0)
        return {
            "min": self.lo.astype(np.float32).tolist(),
            "max": self.hi.astype(np.float32).tolist(),
            "mean": mean.astype(np.float32).tolist(),
            "std": np.sqrt(var).astype(np.float32).tolist(),
            "count": [self.n],
        }


class LeRobotWriter:
    def __init__(
        self,
        out_dir: str,
        fps: float,
        camera_keys: list[str],
        robot_type: str = "unknown",
    ):
        self.out = out_dir
        self.fps = float(fps)
        self.camera_keys = list(camera_keys)
        self.robot_type = robot_type

        os.makedirs(os.path.join(out_dir, "data", CHUNK), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "meta"), exist_ok=True)

        self._episodes = []          # [{episode_index, tasks, length}]
        self._tasks = {}             # task string -> task_index
        self._global_index = 0
        self._state_dim = None
        self._action_dim = None
        self._img_shapes: Dict[str, tuple] = {}
        self._stats = {"observation.state": _RunningStats(), "action": _RunningStats()}

        # per-episode stats for meta/episodes_stats.jsonl (LeRobot v2.1 / v3 convert)
        self._episode_stats: list[dict] = []
        self._ep_open = False
        self._rows = None
        self._encoders: Dict[str, _VideoEncoder] = {}
        self._ep_task = None

    # -- episode lifecycle -------------------------------------------------
    def begin_episode(self, task: str = "") -> None:
        assert not self._ep_open, "episode already open"
        self._ep_open = True
        self._ep_task = task
        if task not in self._tasks:
            self._tasks[task] = len(self._tasks)
        ep_idx = len(self._episodes)
        self._rows = {k: [] for k in
                      ("index", "episode_index", "frame_index", "timestamp",
                       "task_index", "observation.state", "action", "next.done")}
        self._ep_stats = {"observation.state": _RunningStats(), "action": _RunningStats()}
        self._encoders = {
            key: _VideoEncoder(
                os.path.join(self.out, "videos", CHUNK, key, f"episode_{ep_idx:06d}.mp4"),
                self.fps,
            )
            for key in self.camera_keys
        }

    def add_frame(
        self,
        state: np.ndarray,
        action: np.ndarray,
        images: Dict[str, np.ndarray],
        timestamp: Optional[float] = None,
    ) -> None:
        assert self._ep_open, "call begin_episode() first"
        ep_idx = len(self._episodes)
        fidx = len(self._rows["index"])
        state = np.asarray(state, dtype=np.float32).ravel()
        action = np.asarray(action, dtype=np.float32).ravel()

        if self._state_dim is None:
            self._state_dim, self._action_dim = state.size, action.size
        ts = fidx / self.fps if timestamp is None else float(timestamp)

        self._rows["index"].append(self._global_index)
        self._rows["episode_index"].append(ep_idx)
        self._rows["frame_index"].append(fidx)
        self._rows["timestamp"].append(np.float32(ts))
        self._rows["task_index"].append(self._tasks[self._ep_task])
        self._rows["observation.state"].append(state.tolist())
        self._rows["action"].append(action.tolist())
        self._rows["next.done"].append(False)
        self._global_index += 1

        self._stats["observation.state"].update(state)
        self._stats["action"].update(action)
        self._ep_stats["observation.state"].update(state)
        self._ep_stats["action"].update(action)

        for key, enc in self._encoders.items():
            img = images[key]
            self._img_shapes.setdefault(key, img.shape[:2] + (3,))
            enc.add(img)

    def _vector_column(self, key: str, dim: int | None) -> pa.Array:
        """Parquet column compatible with lerobot v3 ``LeRobotDataset`` loading."""
        vals = self._rows[key]
        if not vals:
            return pa.array([], type=pa.list_(pa.float32()))
        if dim is None:
            dim = len(vals[0])
        if dim == 1:
            # info.json shape [1] -> HF Value (scalar), not Sequence.
            return pa.array([v[0] for v in vals], type=pa.float32())
        return pa.array(vals, type=pa.list_(pa.float32(), dim))

    def end_episode(self) -> None:
        assert self._ep_open, "no episode open"
        length = len(self._rows["index"])
        if length:
            self._rows["next.done"][-1] = True
        ep_idx = len(self._episodes)

        table = pa.table(
            {
                "observation.state": self._vector_column("observation.state", self._state_dim),
                "action": self._vector_column("action", self._action_dim),
                "timestamp": pa.array(self._rows["timestamp"], pa.float32()),
                "frame_index": pa.array(self._rows["frame_index"], pa.int64()),
                "episode_index": pa.array(self._rows["episode_index"], pa.int64()),
                "index": pa.array(self._rows["index"], pa.int64()),
                "task_index": pa.array(self._rows["task_index"], pa.int64()),
                "next.done": pa.array(self._rows["next.done"], pa.bool_()),
            }
        )
        pq.write_table(
            table,
            os.path.join(self.out, "data", CHUNK, f"episode_{ep_idx:06d}.parquet"),
        )
        for enc in self._encoders.values():
            enc.close()

        self._episodes.append(
            {"episode_index": ep_idx, "tasks": [self._ep_task], "length": length}
        )
        self._episode_stats.append(
            {"episode_index": ep_idx, "stats": {k: s.as_dict() for k, s in self._ep_stats.items() if s.n}}
        )
        self._ep_open = False
        self._rows = None
        self._encoders = {}

    # -- finalise ----------------------------------------------------------
    def close(self) -> None:
        assert not self._ep_open, "end the open episode before close()"
        self._write_meta()

    def _features(self) -> dict:
        feats = {
            "observation.state": {
                "dtype": "float32", "shape": [int(self._state_dim or 0)], "names": None,
            },
            "action": {
                "dtype": "float32", "shape": [int(self._action_dim or 0)], "names": None,
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "next.done": {"dtype": "bool", "shape": [1], "names": None},
        }
        for key in self.camera_keys:
            h, w, c = self._img_shapes.get(key, (0, 0, 3))
            feats[key] = {
                "dtype": "video",
                "shape": [int(h), int(w), int(c)],
                "names": ["height", "width", "channels"],
                "info": {"video.fps": self.fps, "video.codec": "h264"},
            }
        return feats

    def _write_meta(self) -> None:
        meta = os.path.join(self.out, "meta")
        total_frames = sum(e["length"] for e in self._episodes)
        info = {
            "codebase_version": CODEBASE_VERSION,
            "robot_type": self.robot_type,
            "total_episodes": len(self._episodes),
            "total_frames": total_frames,
            "total_tasks": len(self._tasks),
            "total_videos": len(self._episodes) * len(self.camera_keys),
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": self.fps,
            "splits": {"train": f"0:{len(self._episodes)}"},
            "data_path": "data/" + CHUNK + "/episode_{episode_index:06d}.parquet",
            "video_path": "videos/" + CHUNK + "/{video_key}/episode_{episode_index:06d}.mp4",
            "features": self._features(),
        }
        with open(os.path.join(meta, "info.json"), "w") as fh:
            json.dump(info, fh, indent=2)

        with open(os.path.join(meta, "tasks.jsonl"), "w") as fh:
            for task, idx in sorted(self._tasks.items(), key=lambda kv: kv[1]):
                fh.write(json.dumps({"task_index": idx, "task": task}) + "\n")

        with open(os.path.join(meta, "episodes.jsonl"), "w") as fh:
            for e in self._episodes:
                fh.write(json.dumps(e) + "\n")

        with open(os.path.join(meta, "episodes_stats.jsonl"), "w") as fh:
            for row in self._episode_stats:
                fh.write(json.dumps(row) + "\n")

        stats = {k: s.as_dict() for k, s in self._stats.items() if s.n}
        with open(os.path.join(meta, "stats.json"), "w") as fh:
            json.dump(stats, fh, indent=2)
