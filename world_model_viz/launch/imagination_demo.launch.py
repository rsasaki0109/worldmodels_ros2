"""End-to-end imagination demo: dummy World Model runtime + synthetic
observations + the imagination viewer + RViz.

    ros2 launch world_model_viz imagination_demo.launch.py

Set rviz:=false on headless machines.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    bringup = get_package_share_directory("world_model_bringup")
    viz_share = get_package_share_directory("world_model_viz")
    rviz_cfg = os.path.join(viz_share, "rviz", "imagination.rviz")

    runtime = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup, "launch", "dummy_runtime.launch.py")
        )
    )

    viewer = Node(
        package="world_model_viz",
        executable="occupancy_marker_node",
        name="world_model_viz",
        output="screen",
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_cfg],
        condition=IfCondition(LaunchConfiguration("rviz")),
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("rviz", default_value="true"),
            runtime,
            viewer,
            rviz,
        ]
    )
