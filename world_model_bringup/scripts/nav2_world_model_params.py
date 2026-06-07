"""Merge Nav2 bringup params with World Model costmap layer + DWB critic."""
from __future__ import annotations

import copy
import os
import tempfile
from typing import Any

import yaml


def _dwb_follow_path() -> dict[str, Any]:
    return {
        "plugin": "dwb_core::DWBLocalPlanner",
        "debug_trajectory_details": False,
        "min_vel_x": 0.0,
        "min_vel_y": 0.0,
        "max_vel_x": 0.26,
        "max_vel_y": 0.0,
        "max_vel_theta": 1.0,
        "min_speed_xy": 0.0,
        "max_speed_xy": 0.26,
        "min_speed_theta": 0.0,
        "acc_lim_x": 2.5,
        "acc_lim_y": 0.0,
        "acc_lim_theta": 3.2,
        "decel_lim_x": -2.5,
        "decel_lim_y": 0.0,
        "decel_lim_theta": -3.2,
        "vx_samples": 12,
        "vy_samples": 1,
        "vtheta_samples": 12,
        "sim_time": 1.5,
        "linear_granularity": 0.05,
        "angular_granularity": 0.025,
        "transform_tolerance": 0.2,
        "trans_stopped_velocity": 0.25,
        "short_circuit_trajectory_evaluation": True,
        "stateful": True,
        "critics": [
            "RotateToGoal",
            "Oscillation",
            "BaseObstacle",
            "GoalAlign",
            "PathAlign",
            "PathDist",
            "GoalDist",
            "world_model_dwb_critics::WorldModelOccupancyCritic",
        ],
        "BaseObstacle.scale": 0.02,
        "PathAlign.scale": 0.0,
        "GoalAlign.scale": 0.0,
        "PathDist.scale": 32.0,
        "GoalDist.scale": 24.0,
        "RotateToGoal.scale": 32.0,
        "RotateToGoal.slowing_factor": 5.0,
        "RotateToGoal.lookahead_time": -1.0,
        "world_model_dwb_critics::WorldModelOccupancyCritic": {
            "plugin": "world_model_dwb_critics::WorldModelOccupancyCritic",
            "enabled": True,
            "topic": "/world_model_runtime/future_occupancy",
            "occupied_threshold": 50,
            "collision_radius": 0.18,
            "collision_penalty": 10.0,
            "scale": 1.0,
        },
    }


def merge_nav2_world_model_params(base_path: str) -> str:
    """Return a temp yaml path: base Nav2 params + World Model plugins."""
    with open(base_path, encoding="utf-8") as fh:
        params = yaml.safe_load(fh)

    params = copy.deepcopy(params)
    params["controller_server"]["ros__parameters"]["FollowPath"] = _dwb_follow_path()

    local = params["local_costmap"]["local_costmap"]["ros__parameters"]
    plugins = list(local.get("plugins", []))
    if "world_model_layer" not in plugins:
        insert_at = plugins.index("inflation_layer") if "inflation_layer" in plugins else len(plugins)
        plugins.insert(insert_at, "world_model_layer")
    local["plugins"] = plugins
    local["world_model_layer"] = {
        "plugin": "world_model_costmap::WorldModelLayer",
        "enabled": True,
        "topic": "/world_model_runtime/future_occupancy",
        "occupied_threshold": 50,
    }

    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="nav2_world_model_")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(params, fh, sort_keys=False)
    return path
