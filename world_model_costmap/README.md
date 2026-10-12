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
  The previous cycle's marked extent is re-reported in `updateBounds`, so cells
  a prediction no longer covers are cleared from the master. Cells outside the
  camera's view are published as `-1` and ignored.
- Each grid is moved from its header frame (e.g. `base_link`, where a camera
  head predicts) into the costmap's global frame through TF, at the grid's
  stamp when TF has it, else the latest transform if it is within
  `transform_tolerance`. An empty `frame_id` means the grid is already in the
  global frame. On a costmap finer than the grid, every costmap cell under a
  predicted cell is marked, not just the centre.

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
| `transform_tolerance` | `0.5` | s; skip a grid whose TF is older than this |
| `step_costs` | `[]` (all `LETHAL`) | cost per prediction step; the last entry repeats for later steps |
| `robot_clear_radius` | `0.0` | m; never mark predictions this close to the robot |

`step_costs` lets a planner trade a predicted obstacle off against other costs
instead of treating every predicted cell as a wall. Graded, sub-inscribed costs
such as `[220, 180, 140, 100]` are one option. The Gazebo evaluation measures
both encodings. Neither prevents contacts there, and the all-`LETHAL` one
leaves the most clearance (see the top-level README).

## Verification status

Compiles against Nav2 Jazzy, exports the pluginlib class, and registers in the
ament index. **Nav2 loopback smoke** (`smoke_nav2_loopback.py`) confirms the
layer subscribes to `future_occupancy` inside a live stack.
`world_model_py/test/integration_occupancy_nav2.py` feeds real camera
predictions (I-JEPA + a trained occupancy head) through a standalone costmap.
It checks that the lethal cells equal the predicted union every frame, with the
costmap in `odom` and the robot off the origin and rotated (TF path).
Navigation to a goal with the layer in both Nav2 costmaps runs in Gazebo:
`world_model_bringup/scripts/gz_nav_eval.py` (see the top-level README).
