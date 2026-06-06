# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`vjepa2` adapter** — real V-JEPA 2 video encoder via `torch.hub`
  (`facebookresearch/vjepa2`): rolling-clip latent + temporal surprise. Verified
  on a 16 GB GPU (ViT-L, fp16). torch stays optional/lazy-imported.
- **`world_model_costmap`** — a compiled `nav2_costmap_2d::Layer` (C++/pluginlib)
  that stamps predicted `FutureOccupancy` into the Nav2 costmap.
- README hero GIFs (imagined occupancy, Nav2 scoring, real I-JEPA surprise) and
  a reproducible generator under `tools/gif/`.
- `CONTRIBUTING.md`, issue forms, and a PR template.
- `world-model bench-compare` — one evidence dashboard across adapters; spins a
  local reference server to measure the remote adapter's HTTP overhead.

## [0.1.0] - 2026-06-06

First public release: a ROS 2 runtime / adapter / benchmark / visualization
layer for using existing World Models. ROS 2 Jazzy.

### Added
- **world_model_msgs** — the typed contract: `Observation`, `ActionCondition`,
  `FutureState`, `FutureOccupancy`, `LatentState`, `RiskScore`, `Rollout`,
  `PredictFuture.action`, and `srv/ScoreTrajectories`.
- **world_model_py** — adapter SDK:
  - `load_model()` registry; adapters operate on numpy/dataclasses (no ROS, no
    GPU to test).
  - `dummy` adapter — deterministic and GPU-free (CI, demos, smoke tests).
  - `ijepa` adapter — real self-supervised JEPA encoder: image → latent →
    surprise score. torch/transformers are optional (lazy-imported). Verified on
    a 16 GB GPU (I-JEPA ViT-H/14, fp16, 1280-d latents).
  - `remote` adapter + `world-model-server` — local↔remote split over a shared
    JSON wire format (`wire.py`); swap the server's backing adapter to a heavy
    model (Cosmos/DreamZero) on a GPU box without changing ROS clients.
  - lifecycle `runtime_node` (self-activating via `autostart`), sample observation
    publisher, `world-model` CLI, and an HTML benchmark.
- **world_model_viz** — RViz "imagination" viewer: imagined `FutureOccupancy`
  and `RiskScore` rendered as a `MarkerArray` (future stacked on +z, green→red).
- **world_model_datasets** — `export_lerobot`: convert rosbag2 (sqlite3/mcap)
  into a LeRobot-compatible v2.1 dataset (parquet + per-camera mp4 + meta).
  GPU-free; nearest-neighbour time sync with the image stream as master clock.
- **world_model_nav2** — score Nav2 candidate trajectories by model-based risk
  (a service mock of a controller critic) with risk-coloured path markers.
- **world_model_bringup** — launch files for the dummy and remote runtimes.
- GitHub Actions CI: a ROS-free unit job and a full ROS 2 Jazzy
  `colcon build` + `colcon test` job.

### Notes
- The dataset writer is validated structurally (parquet round-trips, mp4 is
  ffprobe-readable, counts are consistent), not against the `lerobot` loader.
- The Nav2 integration is a ROS service mock, not yet a compiled `nav2_core`
  C++ plugin.

[0.1.0]: https://github.com/rsasaki0109/worldmodels_ros2/releases/tag/v0.1.0
