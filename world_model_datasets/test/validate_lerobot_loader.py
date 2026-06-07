"""Validate export_lerobot output loads with the real ``lerobot`` package.

Converts our v2.1 layout to v3.0 (lerobot >= 3.0 requirement) then opens
``LeRobotDataset``. Skipped in CI when ``lerobot`` is not installed.

    pip install lerobot
    cd world_model_datasets && python3 -m pytest test/validate_lerobot_loader.py -q
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("lerobot")

from world_model_datasets.lerobot_writer import LeRobotWriter


def _write_toy_dataset(out: str, n: int = 6) -> None:
    w = LeRobotWriter(out, fps=10.0, camera_keys=["observation.images.cam"], robot_type="test")
    w.begin_episode(task="pick")
    for i in range(n):
        rgb = np.full((16, 16, 3), i * 30, dtype=np.uint8)
        w.add_frame(
            state=np.array([0.1 * i], np.float32),
            action=np.array([0.2 * i], np.float32),
            images={"observation.images.cam": rgb},
        )
    w.end_episode()
    w.close()
    assert os.path.isfile(os.path.join(out, "meta", "episodes_stats.jsonl"))


def test_lerobot_dataset_loads_after_v30_convert(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = str(tmp_path / "ds")
    _write_toy_dataset(root, n=6)

    subprocess.run(
        [
            sys.executable, "-m", "lerobot.scripts.convert_dataset_v21_to_v30",
            f"--repo-id={os.path.basename(root)}",
            f"--root={root}",
            "--push-to-hub=false",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    info = json.load(open(os.path.join(root, "meta", "info.json")))
    assert info.get("codebase_version", "").startswith("v3")

    ds = LeRobotDataset(repo_id=os.path.basename(root), root=root)
    assert len(ds) == 6
    sample = ds[0]
    state = sample["observation.state"]
    action = sample["action"]
    state_dim = 1 if state.ndim == 0 else int(state.shape[-1])
    action_dim = 1 if action.ndim == 0 else int(action.shape[-1])
    assert state_dim == 1
    assert action_dim == 1
