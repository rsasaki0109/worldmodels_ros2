# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-06-08

The imagination layer reaches ROS 2: plan to an image goal and run counterfactual
"what-if" rollouts as services, on top of the learning-free planner from 0.3.0.

### Added
- **`PlanToGoal` ROS 2 service + `planning_node`** — plan to an image goal in a
  World Model's latent space via the learning-free retrieval planner; returns a
  planned `ActionCondition`, final/start goal cost. New
  `world_model_msgs/srv/PlanToGoal.srv`.
- **`ImagineFutures` ROS 2 service** — counterfactual rollout: one imagined
  `FutureState` (latents, and decoded frames if the memory has them) per steering
  option, plus per-branch divergence. The README hero demo, as a service. New
  `world_model_msgs/srv/ImagineFutures.srv`.
- **`planning.rollout_action` / `imagine_counterfactuals`** — reusable
  counterfactual primitives shared by the demo and the node.
- In-process service tests (3) for both services.

### Changed
- Promoted the counterfactual imagination GIF to the README hero; moved the
  anomaly-monitor GIF into its own section.

## [0.3.0] - 2026-06-08

Learning-free imagination & planning: turn the frozen JEPA encoder into
goal-reaching and counterfactual behaviour with no dynamics training, validated
on real public driving and robot data.

### Added
- **Latent planning** (`world_model_py.planning`) — `RetrievalDynamics`
  (non-parametric latent dynamics from a memory of real `(latent, action,
  next_latent)` transitions; the retrieval analogue of DINO-WM / PLDM),
  `plan_to_goal` (CEM / random-shooting planner to a goal latent), and
  `decode_trajectory` (nearest-neighbour decode of imagined latents to frames).
  Pure numpy, ROS-free, torch-free; 6 unit tests on a toy linear system.
- **Counterfactual imagination demo** (`test/validate_real_counterfactual.py`) —
  action-conditioned retrieval on real public **L2D** driving video: imagine
  "what if I steer left / straight / right"; the branches diverge by action
  (L–R latent distance 0.65). Renders `docs/counterfactual.gif`.
- **Visual-foresight demos** on real public data —
  `validate_real_planning.py` (plan to an image goal on LeRobot SO-101;
  reconstructs the real trajectory, corr 0.97, beats no-op/random) and
  `validate_real_foresight.py` (roll the road ahead from one frame, corr 1.00).

### Changed
- README: new "Counterfactual imagination & planning" section + Highlights entry.

## [0.2.0] - 2026-06-07

A second real model, a compiled Nav2 plugin, a plugin hub, and the first
practical runtime application (anomaly/OOD monitoring) — plus benchmarks,
metrics, demos and a live dashboard.

### Added
- **Runtime anomaly / OOD monitor** — `AnomalyDetector` (self-calibrating
  adaptive threshold on the World Model's latent surprise, no failure data
  needed) + `monitor_node` (publishes `~/surprise`, `~/anomaly_threshold`,
  `~/anomaly`). Follows recent world-model failure/OOD-monitoring research.
- **Prediction-quality metrics** (`world_model_py.metrics`) — occupancy IoU,
  future-occupancy IoU vs a reference, temporal consistency, and latent drift;
  pure numpy, the reusable core for an Autoware/Nav2 evaluator.
- **Runnable external adapter example** (`examples/`) registered purely via an
  entry point — install it and `world-model list` shows it.
- **Entry-point adapter discovery** — external pip packages can add World Model
  backends under the `world_model_ros2.adapters` group; `load_model()` /
  `world-model list` pick them up with no edits to this repo.
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
- A live demo landing + benchmark dashboard published to GitHub Pages, and the
  `camera_sim` + `monitor_demo` one-command anomaly-monitor demo.

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

[0.2.0]: https://github.com/rsasaki0109/worldmodels_ros2/releases/tag/v0.2.0
[0.1.0]: https://github.com/rsasaki0109/worldmodels_ros2/releases/tag/v0.1.0
