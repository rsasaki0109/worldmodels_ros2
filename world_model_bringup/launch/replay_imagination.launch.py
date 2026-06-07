"""Replay a rosbag2 through the World Model runtime and imagination viewer.

    ros2 launch world_model_bringup replay_imagination.launch.py \\
        bag:=/path/to/my_drive

With experience recording + counterfactual planning (learning-free):

    ros2 launch world_model_bringup replay_imagination.launch.py \\
        bag:=/path/to/my_drive record_experience:=true

After the bag finishes, call ``ImagineFutures`` (memory reloads automatically):

    ros2 service call /world_model_planning/imagine_futures \\
        world_model_msgs/srv/ImagineFutures \\
        "{steering_options: [-0.7, 0.0, 0.7], horizon: 12}"

Defaults assume a driving bag with ``/camera/image_raw``, ``/cmd_vel`` and
``/odom``. Set ``rviz:=false`` on headless machines.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node


def generate_launch_description() -> LaunchDescription:
    viz_share = get_package_share_directory("world_model_viz")
    rviz_cfg = os.path.join(viz_share, "rviz", "imagination.rviz")

    bag = LaunchConfiguration("bag")
    adapter = LaunchConfiguration("adapter")
    horizon = LaunchConfiguration("horizon")
    rate = LaunchConfiguration("rate")
    image_topic = LaunchConfiguration("image_topic")
    action_topic = LaunchConfiguration("action_topic")
    state_topic = LaunchConfiguration("state_topic")
    loop = LaunchConfiguration("loop")
    start_delay = LaunchConfiguration("start_delay")
    record_experience = LaunchConfiguration("record_experience")
    experience_out = LaunchConfiguration("experience_out")

    runtime = LifecycleNode(
        package="world_model_py",
        executable="runtime_node",
        name="world_model_runtime",
        namespace="",
        parameters=[{"adapter": adapter, "horizon": horizon, "autostart": True}],
        output="screen",
    )

    relay = Node(
        package="world_model_py",
        executable="bag_relay",
        name="bag_observation_relay",
        parameters=[
            {"image_topic": image_topic},
            {"action_topic": action_topic},
            {"state_topic": state_topic},
            {"action_mode": "full"},
            {"observation_topic": "/world_model_runtime/observation"},
        ],
        output="screen",
    )

    recorder = Node(
        package="world_model_py",
        executable="experience_recorder",
        name="experience_recorder",
        condition=IfCondition(record_experience),
        parameters=[
            {"adapter": adapter},
            {"image_topic": image_topic},
            {"action_topic": action_topic},
            {"action_mode": "scalar"},
            {"output_path": experience_out},
            {"idle_timeout_sec": 2.0},
        ],
        output="screen",
    )

    planning = Node(
        package="world_model_py",
        executable="planning_node",
        name="world_model_planning",
        condition=IfCondition(record_experience),
        parameters=[
            {"adapter": adapter},
            {"memory_path": experience_out},
            {"horizon": horizon},
        ],
        output="screen",
    )

    bag_play_cmd = ["ros2", "bag", "play", bag, "--rate", rate]
    bag_play_loop = ExecuteProcess(
        cmd=bag_play_cmd + ["--loop"],
        condition=IfCondition(loop),
        output="screen",
    )
    bag_play_once = ExecuteProcess(
        cmd=bag_play_cmd,
        condition=UnlessCondition(loop),
        output="screen",
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

    delayed_bag = TimerAction(
        period=start_delay,
        actions=[bag_play_loop, bag_play_once],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("bag", description="Path to a rosbag2 directory or .mcap file"),
            DeclareLaunchArgument("adapter", default_value="dummy"),
            DeclareLaunchArgument("horizon", default_value="8"),
            DeclareLaunchArgument("rate", default_value="1.0"),
            DeclareLaunchArgument("image_topic", default_value="/camera/image_raw"),
            DeclareLaunchArgument("action_topic", default_value="/cmd_vel"),
            DeclareLaunchArgument("state_topic", default_value="/odom"),
            DeclareLaunchArgument("loop", default_value="false"),
            DeclareLaunchArgument("start_delay", default_value="3.0"),
            DeclareLaunchArgument("rviz", default_value="true"),
            DeclareLaunchArgument(
                "record_experience",
                default_value="false",
                description="Record experience.npz and start planning_node for ImagineFutures",
            ),
            DeclareLaunchArgument(
                "experience_out",
                default_value="/tmp/world_model_experience.npz",
                description="Where to write the recorded experience memory",
            ),
            runtime,
            relay,
            recorder,
            planning,
            viewer,
            rviz,
            delayed_bag,
        ]
    )
