"""Run the trajectory scorer and fire the demo client once.

    ros2 launch world_model_nav2 scorer_demo.launch.py

Scores three candidate paths (straight / fast / swerve) with the dummy World
Model and prints the safest. Risk-coloured path markers are published on
/world_model_trajectory_scorer/scored_paths for RViz.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    adapter = LaunchConfiguration("adapter")

    scorer = Node(
        package="world_model_nav2",
        executable="trajectory_scorer_node",
        name="world_model_trajectory_scorer",
        parameters=[{"adapter": adapter}],
        output="screen",
    )
    # give the service a moment to come up, then run the one-shot demo client.
    demo = TimerAction(
        period=2.0,
        actions=[Node(package="world_model_nav2", executable="score_demo", output="screen")],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("adapter", default_value="dummy"),
            scorer,
            demo,
        ]
    )
