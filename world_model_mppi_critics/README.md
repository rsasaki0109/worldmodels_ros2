# world_model_mppi_critics

An MPPI critic plugin for Nav2 Jazzy, `WorldModelTimeCritic`. It scores the
controller's sampled trajectories against a World Model's predicted
`world_model_msgs/FutureOccupancy` **in time**.

## How it works

- **Time matching.** The trajectory point reached at time t is checked against
  the prediction step for t. Step k of the message covers
  `stamp + (k + 1) · dt`. The prediction's age, now minus stamp, is added, and
  each point is matched to the nearest step. Points past the horizon are not
  scored.
- **Why it differs from the costmap layer.** `world_model_costmap` marks the
  union of the whole horizon, which loses *when* a cell will be occupied. With
  this critic, waiting for a pedestrian to pass costs nothing. Only being where
  the pedestrian is predicted to be, at that moment, is penalised.
- **Geometry.** Grids are moved from their header frame (e.g. `base_link` at
  the prediction stamp) into the costmap's global frame through TF. When a
  message arrives, each step's occupied cells are dilated once by `radius`
  (robot plus obstacle clearance).
- **Cost.** A trajectory's cost is `cost_weight · (seconds spent in predicted
  occupancy) ^ cost_power`. Predictions older than `max_age` are ignored.

## Use it

Add it to MPPI's critics list:

```yaml
controller_server:
  ros__parameters:
    FollowPath:
      plugin: "nav2_mppi_controller::MPPIController"
      critics: [..., "WorldModelTimeCritic"]
      WorldModelTimeCritic:
        enabled: true
        topic: /world_model_runtime/future_occupancy
        occupied_threshold: 50   # OccupancyGrid value [0..100]
        radius: 0.55             # m: robot radius + clearance
        cost_weight: 60.0        # per second in predicted occupancy
        cost_power: 1
        max_age: 1.0             # s
        transform_tolerance: 0.5 # s
```

MPPI looks critics up as `mppi::critics::<name>`. The plugin is therefore
registered under the name `mppi::critics::WorldModelTimeCritic`.

## Verification status

- **gtests** (`colcon test`) cover the time-to-step matching and the dilation.
- **Gazebo closed loop.** `world_model_bringup` runs the critic with
  `gz_nav.launch.py wm_mode:=critic`. Over 15 paired seeds against the
  LiDAR-only baseline:
  - clearance is +0.14 m (larger on 11/15 seeds, p = 0.12);
  - time to goal is unchanged;
  - contacts are 20 against 23, which is not significant.

  The predictions come from a raw-pixel head. The top-level README has the
  full comparison.
