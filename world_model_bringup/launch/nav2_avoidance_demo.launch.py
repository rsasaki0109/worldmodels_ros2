"""World Model predicted occupancy -> avoid the lethal union (GPU-free).

Shows what ``world_model_costmap::WorldModelLayer`` would stamp into Nav2, then
picks a detour path that misses the predicted lethal cells. No Gazebo / full
Nav2 stack required.

    ros2 launch world_model_bringup nav2_avoidance_demo.launch.py
    # rviz:=false on headless
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node


def generate_launch_description() -> LaunchDescription:
    viz_share = get_package_share_directory("world_model_viz")
    rviz_cfg = os.path.join(viz_share, "rviz", "nav2_avoidance.rviz")

    runtime = LifecycleNode(
        package="world_model_py",
        executable="runtime_node",
        name="world_model_runtime",
        namespace="",
        parameters=[{"adapter": LaunchConfiguration("adapter"), "horizon": 8, "autostart": True}],
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
    preview = Node(
        package="world_model_viz",
        executable="costmap_preview_node",
        name="costmap_preview",
        output="screen",
    )
    avoidance = Node(
        package="world_model_nav2",
        executable="avoidance_demo_node",
        name="avoidance_demo",
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
            DeclareLaunchArgument("rviz", default_value="true"),
            runtime,
            sample,
            preview,
            avoidance,
            rviz,
        ]
    )
