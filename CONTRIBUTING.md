# Contributing to world_model_ros2

Thanks for your interest! This project is the **ROS 2 layer for World Models** —
a runtime, adapter hub, benchmark, and visualization for *using* existing models,
not a place to train new ones. Contributions that widen that boundary (new
adapters, consumers, datasets, viz) are especially welcome.

## Build & test

ROS 2 Jazzy + Python 3.12.

```bash
# from the workspace root
colcon build
source install/setup.bash
colcon test && colcon test-result --verbose
```

The fast, ROS-free unit tests (adapters, wire, server) need only numpy:

```bash
cd world_model_py && python3 -m pytest test/test_adapters.py test/test_jepa.py \
    test/test_wire.py test/test_remote_server.py -q
```

CI mirrors this: a ROS-free `unit` job and a full ROS 2 Jazzy `colcon` job. Both
must stay green. GPU-only checks (`test/gpu_verify_*.py`) are **not** in CI — run
them locally if you touch the `ijepa` path.

## Design rules (please keep these intact)

1. **Adapters take numpy/dataclasses, never ROS messages.** `runtime_node` +
   `conversions.py` are the only place that touch `world_model_msgs`. This keeps
   adapters unit-testable with no ROS and no GPU.
2. **Heavy/optional deps are lazy-imported** (e.g. torch/transformers inside the
   encoder, not at module top). `import world_model_py`, `dummy`, `remote`, and
   CI must work without them.
3. **The wire format is shared.** Remote adapter and `server.py` both use
   `wire.py`; change them together.
4. **Be honest about verification.** If something is a mock or validated only
   structurally (not against the real loader/model), say so in code and README.

## Adding a World Model adapter

Most contributions are new adapters. Subclass `WorldModelAdapter`:

```python
from world_model_py.adapters.base import WorldModelAdapter, FuturePrediction

class MyAdapter(WorldModelAdapter):
    name = "mymodel"
    def predict_future(self, obs, action=None, horizon=8) -> FuturePrediction:
        ...  # numpy in, FuturePrediction out

# register it (lazy factory if it imports heavy deps)
from world_model_py.registry import register
register("mymodel", MyAdapter)
```

Add unit tests with a fake backend so they run in CI without the real weights
(see `test/test_jepa.py`), and a `gpu_verify_*.py` script for the real model.

**Ship an adapter from your own package** — no edits here needed. Expose it
under the `world_model_ros2.adapters` entry-point group:

```toml
# pyproject.toml in your package
[project.entry-points."world_model_ros2.adapters"]
mymodel = "my_pkg.adapter:make_my_adapter"
```

Once installed, `load_model("mymodel")` and `world-model list` pick it up
automatically. Built-in names take precedence, so an entry point can't hijack
`dummy`/`remote`/`ijepa`/`vjepa2`.

## Pull requests

- Branch off `main`; keep PRs focused.
- Run `colcon test` and the ROS-free unit tests before pushing.
- Update the README / `CHANGELOG.md` when behavior or interfaces change.
- Describe what you verified and how (this project values evidence over vibes).

## Reporting issues

Use the issue templates. For bugs, include your ROS distro, the adapter, and the
exact command + output.

## License

By contributing you agree your contributions are licensed under Apache-2.0.
