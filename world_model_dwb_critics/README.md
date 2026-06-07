# world_model_dwb_critics

A **compiled** `dwb_core::TrajectoryCritic` plugin that scores candidate DWB
trajectories against a World Model's predicted `FutureOccupancy`. It uses the
same lethal-union rule as `world_model_costmap::WorldModelLayer`.

## Use with Nav2 DWB

Add the critic to your DWB controller params (alongside the usual critics):

```yaml
controller_server:
  ros__parameters:
    FollowPath:
      plugin: "dwb_core::DWBLocalPlanner"
      critics: [
        "RotateToGoal", "Oscillation", "BaseObstacle", "GoalAlign", "PathAlign", "PathDist", "GoalDist",
        "world_model_dwb_critics::WorldModelOccupancyCritic"
      ]
      world_model_dwb_critics::WorldModelOccupancyCritic:
        plugin: "world_model_dwb_critics::WorldModelOccupancyCritic"
        enabled: true
        topic: /world_model_runtime/future_occupancy
        occupied_threshold: 50
        collision_radius: 0.18
        collision_penalty: 10.0
        scale: 1.0
```

Run a World Model runtime publishing `future_occupancy` on the same machine.

## Verification status

Builds against Nav2 Jazzy and exports the pluginlib class. **Nav2 loopback
smoke** (`world_model_bringup/scripts/smoke_nav2_loopback.py`) confirms the
critic loads inside a live DWB controller. Full navigation-to-goal in Gazebo is
not yet validated here.
