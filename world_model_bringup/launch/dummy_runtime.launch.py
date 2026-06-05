"""Bring up the World Model runtime with the GPU-free dummy adapter, plus a
synthetic observation publisher. The runtime is a lifecycle node but
``autostart`` (default true) makes it self-transition to active on boot, so
no separate lifecycle manager is needed.

    ros2 launch world_model_bringup dummy_runtime.launch.py

Then, in another terminal:
    ros2 topic echo /world_model_runtime/future_state
    ros2 topic echo /world_model_runtime/risk_score
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node


def generate_launch_description() -> LaunchDescription:
    adapter = LaunchConfiguration("adapter")
    horizon = LaunchConfiguration("horizon")

    runtime = LifecycleNode(
        package="world_model_py",
        executable="runtime_node",
        name="world_model_runtime",
        namespace="",
        parameters=[{"adapter": adapter, "horizon": horizon, "autostart": True}],
        output="screen",
    )

    sample = Node(
        package="world_model_py",
        executable="sample_publisher",
        name="sample_observation_publisher",
        remappings=[("observation", "/world_model_runtime/observation")],
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("adapter", default_value="dummy"),
            DeclareLaunchArgument("horizon", default_value="8"),
            runtime,
            sample,
        ]
    )
