# GIF generation

Reproduces the README GIFs from the **real pipeline** — no faked data. Each GIF
is: `gen_data.py` (run the actual adapters) → `render_*.html` (canvas) →
Playwright screenshots (system Chrome) → ffmpeg palette GIF.

| GIF | source | GPU |
|---|---|---|
| `docs/imagination.gif` | `dummy` adapter `FutureOccupancy` + `RiskScore` | no |
| `docs/nav2_scoring.gif` | `path_to_action` + `dummy.score_trajectory` over a path fan | no |
| `docs/ijepa_surprise.gif` | **real I-JEPA ViT-H/14** per-frame surprise | yes |

## Setup

```bash
source /opt/ros/jazzy/setup.bash
source ../../install/setup.bash        # for world_model_py / world_model_nav2
npm install playwright-core            # uses system google-chrome, no download
# for the ijepa GIF only:
pip install torch "transformers>=4.40"
```

## Run

```bash
./build.sh imagination     # GPU-free
./build.sh nav2            # GPU-free
./build.sh ijepa           # needs GPU + weights
./build.sh all
```

Output GIFs are written to `../../docs/`. Frame counts / fps live in `build.sh`.
