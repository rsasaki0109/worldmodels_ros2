# world_model_ros2

[![ci](https://github.com/rsasaki0109/worldmodels_ros2/actions/workflows/ci.yml/badge.svg)](https://github.com/rsasaki0109/worldmodels_ros2/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![ROS 2](https://img.shields.io/badge/ROS%202-Jazzy-22314E.svg)](https://docs.ros.org/en/jazzy/)

**A ROS 2-native runtime, adapter hub, benchmark and visualization layer for
using *existing* World Models in robotics and autonomous driving.**

> Run V-JEPA2 / Cosmos / dummy world models from ROS 2. Predict future states,
> score trajectories, visualize imagination in RViz, and export rosbag2 data to
> robot-learning datasets.

`world_model_ros2` is **not** another foundation model. It is the
[Nav2](https://github.com/ros-navigation/navigation2)/`transformers`-style
**boundary** that lets a ROS 2 developer call a World Model the same way every
time — locally for light models, remotely for heavy ones — and get back typed
futures, occupancy, latents and risk.

```
ROS 2 observation  (camera / lidar / map / ego_state / instruction / action)
        │
        ▼
world_model_ros2 runtime
   local adapter:  dummy, V-JEPA2, small predictor
   remote adapter: Cosmos, DreamZero, custom server
        │
        ▼
outputs:  future frames · future occupancy · latent state
          action-conditioned rollout · risk / surprise score
        │
        ▼
consumers: RViz/Foxglove · Nav2 plugin · Autoware evaluator · VLA Zoo · Walking Zoo
```

## What this is / isn't

**It is:**
- Run local **and** remote World Model adapters from ROS 2
- Predict future states from sensor streams and candidate actions
- Score trajectories and commands with model-based risk
- Visualize imagined futures in RViz/Foxglove
- Benchmark latency, VRAM and temporal consistency
- Convert rosbag2 data into robot-learning / world-model datasets

**It is not:** a foundation model · a Dreamer reimplementation · a simulator
replacement · a safety-certified controller · a replacement for Nav2 or Autoware.

## Try it in 5 minutes — no robot, no dataset, no training

The `dummy` adapter is deterministic and **GPU-free**, so the whole pipeline
runs anywhere.

### 1. Standalone CLI (no ROS needed)

```bash
cd world_model_py
python -m world_model_py.cli list
python -m world_model_py.cli bench --adapter dummy --out report.html
# -> report.html with latency p50/p95, throughput, VRAM
```

### 2. ROS 2 runtime

```bash
# build
colcon build
source install/setup.bash

# launch the runtime + a synthetic observation publisher (dummy adapter)
ros2 launch world_model_bringup dummy_runtime.launch.py

# in another terminal
ros2 topic echo /world_model_runtime/future_state
ros2 topic echo /world_model_runtime/risk_score
```

### 3. See the imagination (RViz)

One command brings up the dummy runtime, synthetic observations, the
imagination viewer and RViz. The predicted occupancy is stacked along +z
(higher = further into the future) and coloured green→red (near→far), with a
risk readout on top — all from the GPU-free dummy model.

```bash
ros2 launch world_model_viz imagination_demo.launch.py     # rviz:=false on headless
```

The viewer republishes `MarkerArray` on `/world_model_viz/imagination`.

## Packages

| package | type | what |
|---|---|---|
| `world_model_msgs` | ament_cmake | the message/action **contract**: `Observation`, `ActionCondition`, `FutureState`, `FutureOccupancy`, `LatentState`, `RiskScore`, `Rollout`, `PredictFuture.action` |
| `world_model_py` | ament_python | adapter SDK (`load_model`), `dummy` + `remote` adapters, lifecycle runtime node, sample publisher, benchmark, `world-model` CLI |
| `world_model_viz` | ament_python | imagination viewer: imagined `FutureOccupancy` + `RiskScore` → RViz `MarkerArray` |
| `world_model_datasets` | ament_python | `export_lerobot`: rosbag2 → LeRobot-compatible dataset (parquet + mp4 + meta), GPU-free |
| `world_model_nav2` | ament_python | score Nav2 candidate trajectories by model-based risk (`ScoreTrajectories` service + risk-coloured path markers) |
| `world_model_bringup` | ament_cmake | launch files + demo config |

## Adapter SDK

Adapters operate on plain numpy/dataclasses — **no ROS, no GPU required to
test** — so backends are unit-testable and benchmarkable in isolation. The ROS
layer is the only place that touches `world_model_msgs`.

```python
from world_model_py import load_model
from world_model_py.adapters import Observation, ActionCondition
import numpy as np

wm = load_model("dummy")                       # or "remote", url=...
obs = Observation(ego_state=np.zeros(4, np.float32))
future = wm.predict_future(obs, horizon=8)     # -> FuturePrediction
risk = wm.score_trajectory(obs, ActionCondition(action=np.zeros((8, 2), np.float32)))
```

Adding a backend = subclass `WorldModelAdapter`, implement `predict_future`,
and `register("vjepa2", VJepa2Adapter)`.

### Real model: JEPA latent server (`ijepa`)

The first real backend turns a camera stream into latents and a **surprise**
score (cosine distance between successive latents — a model-based novelty /
anomaly signal). It loads a self-supervised JEPA image encoder (I-JEPA today;
V-JEPA2 video weights drop into the same `jepa.py` once `transformers` ships
them).

```bash
# optional extras, not required for dummy/remote or CI:
pip install "torch" "transformers>=4.40"

ros2 launch world_model_bringup dummy_runtime.launch.py \
    adapter:=ijepa model_id:=facebook/ijepa_vith14_1k
ros2 topic echo /world_model_runtime/risk_score    # surprise per frame
```

Verified on a 16 GB GPU (GPU): I-JEPA ViT-H/14, fp16, 1280-d
latents — identical frames ≈ 0 surprise, a changed frame ≈ 0.65. torch and
transformers are **optional**: imported lazily, so `dummy`/`remote` and CI need
neither.

## rosbag2 → robot-learning dataset

Turn a recording into a LeRobot-compatible dataset — **no GPU**. The image
topic is the master clock; state/action are nearest-neighbour joined to each
frame (header stamp when present) and matches outside `--tol-ms` are dropped.

```bash
ros2 run world_model_datasets export_lerobot \
    --bag ./my_bag \
    --image-topic /camera/image_raw \
    --state-topic /odom \
    --action-topic /cmd_vel \
    --fps 10 --task "drive forward" --out ./hf_dataset
```

Output (v2.1 layout): `meta/{info,episodes,tasks,stats}.json[l]`,
`data/chunk-000/episode_*.parquet`, `videos/chunk-000/<key>/episode_*.mp4`.
Supported state/action types: `nav_msgs/Odometry`, `sensor_msgs/JointState`,
`geometry_msgs/Twist[Stamped]`, `std_msgs/Float{32,64}MultiArray`.

> Honesty: validated structurally (parquet round-trips, mp4 is ffprobe-readable,
> counts consistent), **not** against the `lerobot` loader (not a dependency
> here). Verify against your `lerobot` version before training.

## Nav2 trajectory scoring (mock)

Rank candidate paths by model-based risk before the robot commits to one — a
mock of a Nav2 controller critic (a service today, a compiled `nav2_core`
plugin later). Each path becomes a body-frame action sequence and is scored by
the World Model; the safest wins.

```bash
ros2 launch world_model_nav2 scorer_demo.launch.py
# straight  risk=0.129
# fast      risk=0.213
# swerve    risk=0.165
# -> safest: straight
```

Service `~/score_trajectories` (`world_model_msgs/srv/ScoreTrajectories`) takes
`nav_msgs/Path[]` and returns a risk per path plus the safest index; scored
paths are published as risk-coloured (green→red) markers for RViz.

## Roadmap (90-day MVP)

- **0–30d (this scaffold):** msgs, adapter SDK, dummy + remote adapters,
  lifecycle node, CLI, HTML smoke/bench report — **all GPU-free**. ✅
- **31–60d:** JEPA latent adapter (image→latent→surprise) ✅, RViz imagination
  markers ✅, rosbag2 replay demo (next).
- **61–90d:** rosbag2 → LeRobotDataset converter ✅, Nav2 trajectory-scoring
  mock ✅, Cosmos remote adapter, benchmark dashboard, VLA Zoo / Walking Zoo
  examples.

## License

Apache-2.0. See [LICENSE](LICENSE).
