# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.1] - 2026-06-08

LeRobot loader validation and a compiled Nav2 DWB critic plugin.

### Added
- **LeRobot loader validation** — `test/validate_lerobot_loader.py` exports a toy
  dataset, converts v2.1 → v3.0, and loads it with `LeRobotDataset` when
  `lerobot` is installed. Parquet writer fix: 1-D state/action columns are stored
  as scalars so v3 loading succeeds.
- **`world_model_dwb_critics`** — compiled `dwb_core::TrajectoryCritic` plugin
  (`WorldModelOccupancyCritic`) that scores DWB rollouts against predicted
  `FutureOccupancy` using the same lethal-union rule as `world_model_costmap`.
- **Nav2 loopback integration** — `nav2_loopback_world_model.launch.py` patches
  Nav2 bringup params with the costmap layer + DWB critic; `smoke_nav2_loopback.py`
  verifies both plugins initialise in a live stack (requires
  `ros-jazzy-nav2-loopback-sim`).

### Notes
- Loopback smoke confirms plugin load + subscription, not full navigation to a goal.
  LeRobot validation is optional (`pip install lerobot`).

## [0.7.0] - 2026-06-08

From rosbag replay to counterfactual imagination and Nav2-style avoidance — the
full “use your own data” loop, plus one-command demos.

### Added
- **Rosbag replay** — `bag_relay` node + `replay_imagination.launch.py`: play a
  bag through the World Model runtime and RViz imagination viewer.
- **Experience recording** — `experience_recorder` writes `experience.npz` when a
  replay finishes; `planning_node` hot-reloads the memory for `ImagineFutures`.
- **Counterfactual demo** — bundled `drive_demo.mcap`,
  `replay_counterfactual_demo.launch.py`, `counterfactual_marker_node` (mosaic +
  MarkerArray), and Foxglove layout `foxglove/counterfactual.json`.
- **Nav2 avoidance preview** — `costmap_preview_node` (predicted lethal union),
  `avoidance_demo_node` (pick a detour path using the same rule as
  `world_model_costmap::WorldModelLayer`), `nav2_avoidance_demo.launch.py`, and
  `docs/nav2_avoidance.gif`.

### Notes
- Nav2 avoidance demo visualises the costmap-layer logic in Python + RViz; closed-
  loop Nav2 in Gazebo is still future work (see `world_model_costmap/README.md`).

## [0.6.0] - 2026-06-08

Generate the future instead of retrieving it: a learned `latent → pixel` decoder
turns imagined latents into synthesised frames, including states that were never
observed.

### Added
- **`LatentDecoder`** (`world_model_py.decoder`) — a small convolutional decoder
  trained on `(latent, frame)` pairs so an imagined latent is *rendered* as pixels
  instead of snapped to the nearest real frame. Plugs into `DriveSim`
  (`frame()` generates when a decoder is attached) and any imagined trajectory.
  Optional / lazy-imported torch, like the JEPA adapters; retrieval decode stays
  the torch-free default.
- **`test/validate_generative_decode.py`** — trains the decoder on a built world,
  reports reconstruction PSNR (~31 dB on public L2D), and renders
  `docs/generated_drive.gif` (retrieved vs generated, side by side) and
  `docs/decoder.png` (reconstructions + a latent interpolation that generates
  unobserved frames).
- **`play_drive.py --generative`** — drive with generated frames.
- 4 torch-gated unit tests for the decoder (shapes, learning, save/load).

### Notes
- Honest scope: reconstruction-grade rendering bounded by the small public
  dataset (~750 frames) — softer than real, latent interpolations blur. A step
  toward but not yet DIAMOND/Oasis-quality generation; the same interface takes a
  stronger decoder or a V-JEPA 2-AC head with more data.

## [0.5.0] - 2026-06-08

A *learned* world-model head: train a tiny action-conditioned latent dynamics on
real driving and **drive inside it** — smoothly, with steering control — where
the learning-free retrieval model could only do one-shot what-ifs.

### Added
- **`MLPDynamics`** (`world_model_py.dynamics`) — a learned action-conditioned
  latent dynamics head: PCA to a small subspace + a tiny MLP predicting the
  latent delta. Trained with torch (GPU), but **inference is pure numpy** (weights
  stored as numpy), so a trained world is a self-contained `.npz` and the planner
  / DriveSim / ROS node need no torch. Drop-in for the existing `step()` interface.
- **`LinearDynamics`** (`world_model_py.planning`) — a closed-form ridge-regression
  latent dynamics head; the honest baseline (predicts well, keeps moving, but
  can't capture steering). Pure numpy, unit-tested on CI.
- **`DriveSim`** (`world_model_py.play`) + **playable demo** — steer through the
  world model in real time (`test/play_drive.py`, keyboard or `--record` to GIF);
  `test/build_drive_world.py` builds a playable world from real public **L2D**
  driving video (encode → train head → one self-contained `.npz`). New README
  hero-style GIF `docs/drive.gif`.
- Unit tests (12) for the numpy inference path, `LinearDynamics`, and `DriveSim`;
  the torch trainer is exercised on a toy rotation world (skipped without torch).

### Notes
- Honest result, measured on the same memory: only the learned MLP head *both*
  keeps moving under a constant action *and* responds to steering (L–R 0.43).
  Retrieval collapses to a fixed point under a held action; a linear head moves
  but ignores steering. Still nearest-neighbour decode (no pixel synthesis).

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
