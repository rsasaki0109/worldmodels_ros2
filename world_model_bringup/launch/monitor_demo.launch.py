"""Runtime anomaly-monitor demo: a synthetic camera (with periodic occlusion
events) feeding the World Model monitor.

    ros2 launch world_model_bringup monitor_demo.launch.py            # dummy (GPU-free)
    ros2 launch world_model_bringup monitor_demo.launch.py adapter:=ijepa   # real detection

Watch it flag the occlusion events:
    ros2 topic echo /world_model_monitor/anomaly
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    adapter = LaunchConfiguration("adapter")

    camera = Node(
        package="world_model_py",
        executable="camera_sim",
        name="camera_sim",
        output="screen",
    )
    monitor = Node(
        package="world_model_py",
        executable="monitor_node",
        name="world_model_monitor",
        parameters=[{"adapter": adapter}],
        remappings=[("image", "/camera_sim/image")],
        output="screen",
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("adapter", default_value="dummy"),
            camera,
            monitor,
        ]
    )
