"""Nav2 TB3 loopback sim + World Model runtime (costmap layer + DWB critic).

Requires ``ros-jazzy-nav2-loopback-sim`` and a built workspace with
``world_model_costmap`` + ``world_model_dwb_critics``.

    ros2 launch world_model_bringup nav2_loopback_world_model.launch.py use_rviz:=false

Headless smoke:

    python3 $(ros2 pkg prefix world_model_bringup)/share/world_model_bringup/scripts/smoke_nav2_loopback.py
"""
from __future__ import annotations

import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node


def _setup(context, *args, **kwargs):
    bringup_dir = get_package_share_directory("nav2_bringup")
    bringup_share = get_package_share_directory("world_model_bringup")
    scripts_dir = os.path.join(bringup_share, "scripts")
    sys.path.insert(0, scripts_dir)
    from nav2_world_model_params import merge_nav2_world_model_params

    merged = merge_nav2_world_model_params(
        os.path.join(bringup_dir, "params", "nav2_params.yaml")
    )
    adapter = LaunchConfiguration("adapter").perform(context)
    use_rviz = LaunchConfiguration("use_rviz").perform(context)

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_dir, "launch", "tb3_loopback_simulation.launch.py")
        ),
        launch_arguments={
            "params_file": merged,
            "use_rviz": use_rviz,
            "autostart": "true",
        }.items(),
    )
    runtime = LifecycleNode(
        package="world_model_py",
        executable="runtime_node",
        name="world_model_runtime",
        namespace="",
        parameters=[{"adapter": adapter, "horizon": 8, "autostart": True}],
        output="screen",
    )
    sample = Node(
        package="world_model_py",
        executable="sample_publisher",
        name="sample_observation_publisher",
        remappings=[("observation", "/world_model_runtime/observation")],
        parameters=[{"rate_hz": 4.0}],
        output="screen",
    )
    return [nav2, runtime, sample]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("adapter", default_value="dummy"),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            OpaqueFunction(function=_setup),
        ]
    )
