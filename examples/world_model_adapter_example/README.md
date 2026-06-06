# Example external World Model adapter

A tiny, **separate** pip package that adds a World Model backend to
`world_model_ros2` **without editing the main repo** — the entry-point hub in
action. The adapter (`example`) encodes each frame as an RGB colour histogram
and reports surprise as the histogram change (a GPU-free appearance-novelty
signal). Copy this layout to ship your own model.

## Try it

```bash
source /opt/ros/jazzy/setup.bash
source ../../install/setup.bash          # provides world_model_py
pip install -e .                         # registers the entry point

world-model list                         # -> ... example ...
world-model info --adapter example
```

In ROS 2 it then works like any backend:

```bash
ros2 run world_model_py runtime_node --ros-args -p adapter:=example
```

## The wiring

```toml
# pyproject.toml
[project.entry-points."world_model_ros2.adapters"]
example = "wm_example_adapter:make_example_adapter"
```

`make_example_adapter(**kwargs)` returns a `WorldModelAdapter`. That's the whole
contract — see `wm_example_adapter/__init__.py`.
