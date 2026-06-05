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

## Roadmap (90-day MVP)

- **0–30d (this scaffold):** msgs, adapter SDK, dummy + remote adapters,
  lifecycle node, CLI, HTML smoke/bench report — **all GPU-free**. ✅
- **31–60d:** V-JEPA2 local adapter, image→latent→surprise, RViz markers,
  rosbag2 replay demo.
- **61–90d:** Nav2 trajectory-scoring mock, Cosmos remote adapter, rosbag2 →
  LeRobotDataset converter, benchmark dashboard, VLA Zoo / Walking Zoo examples.

## License

Apache-2.0. See [LICENSE](LICENSE).
