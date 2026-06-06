# world_model_costmap

A **real, compiled** `nav2_costmap_2d::Layer` plugin (C++/pluginlib) that stamps
a World Model's predicted `FutureOccupancy` into the Nav2 costmap. Any planner or
controller then avoids where the model expects obstacles over the horizon — no
controller-specific integration needed.

This is the production-grade counterpart to the `world_model_nav2` service mock:
the mock *ranks* candidate paths offline; this layer feeds the live costmap.

## How it works

- Subscribes to `world_model_msgs/FutureOccupancy` (default
  `/world_model_runtime/future_occupancy`, published by the runtime node).
- Each update it clears its buffer and marks the **union of predicted occupancy
  over the whole horizon**: any cell `>= occupied_threshold` becomes
  `LETHAL_OBSTACLE`, then `updateWithMax` merges it into the master costmap.
- Assumes the grids are in the costmap's global frame (TF reprojection of a
  differing frame is future work).

## Use it

Add the layer to your Nav2 costmap params:

```yaml
local_costmap:
  local_costmap:
    ros__parameters:
      plugins: ["obstacle_layer", "world_model_layer", "inflation_layer"]
      world_model_layer:
        plugin: "world_model_costmap::WorldModelLayer"
        enabled: true
        topic: /world_model_runtime/future_occupancy
        occupied_threshold: 50      # OccupancyGrid value [0..100]
```

| param | default | meaning |
|---|---|---|
| `enabled` | `true` | toggle the layer |
| `topic` | `/world_model_runtime/future_occupancy` | FutureOccupancy source |
| `occupied_threshold` | `50` | min occupancy to mark a cell lethal |

## Verification status

Compiles against Nav2 Jazzy, exports the pluginlib class, and registers in the
ament index (so Nav2 discovers it). **Closed-loop behavior in a running Nav2
stack is not yet validated here** — try it in a Gazebo/Nav2 sim before relying
on it.
