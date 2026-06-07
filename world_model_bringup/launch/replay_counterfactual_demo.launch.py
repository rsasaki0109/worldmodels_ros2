"""One-command counterfactual replay demo (bundled bag, no robot, GPU-free).

Replays a small synthetic driving bag, records an experience memory, auto-calls
``ImagineFutures`` when the bag ends, and shows the L / straight / R mosaic in
RViz.

    ros2 launch world_model_bringup replay_counterfactual_demo.launch.py
    # headless: rviz:=false
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
    viz = get_package_share_directory("world_model_viz")
    bag = os.path.join(bringup, "demo", "drive_demo.mcap")
    rviz_cfg = os.path.join(viz, "rviz", "counterfactual.rviz")

    replay = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(bringup, "launch", "replay_imagination.launch.py")),
        launch_arguments={
            "bag": bag,
            "adapter": LaunchConfiguration("adapter"),
            "record_experience": "true",
            "experience_out": "/tmp/world_model_experience.npz",
            "rviz": "false",
            "start_delay": "2.0",
            "rate": LaunchConfiguration("rate"),
        }.items(),
    )

    counterfactual = Node(
        package="world_model_viz",
        executable="counterfactual_marker_node",
        name="counterfactual_viz",
        parameters=[
            {"memory_path": "/tmp/world_model_experience.npz"},
            {"auto_imagine": True},
            {"idle_timeout_sec": 3.0},
        ],
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
            DeclareLaunchArgument("adapter", default_value="dummy"),
            DeclareLaunchArgument("rate", default_value="2.0"),
            DeclareLaunchArgument("rviz", default_value="true"),
            replay,
            counterfactual,
            rviz,
        ]
    )
