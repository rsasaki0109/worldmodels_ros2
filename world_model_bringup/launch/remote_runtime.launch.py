"""Local<->remote demo: start the reference World Model server (dummy backend)
and a runtime node whose adapter is ``remote``, talking to it over HTTP.

    ros2 launch world_model_bringup remote_runtime.launch.py

On a real deployment the server runs on a GPU box with a heavy adapter
(cosmos/dreamzero); here it is the stdlib dummy server, so the whole
local<->remote path runs with no GPU.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node


def generate_launch_description() -> LaunchDescription:
    port = LaunchConfiguration("port")
    server_adapter = LaunchConfiguration("server_adapter")
    url = ["http://127.0.0.1:", port, "/predict_future"]

    server = ExecuteProcess(
        cmd=["python3", "-m", "world_model_py.server",
             "--adapter", server_adapter, "--host", "127.0.0.1", "--port", port],
        output="screen",
    )

    runtime = LifecycleNode(
        package="world_model_py",
        executable="runtime_node",
        name="world_model_runtime",
        namespace="",
        parameters=[{"adapter": "remote", "remote_url": url, "autostart": True}],
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
            DeclareLaunchArgument("port", default_value="8080"),
            DeclareLaunchArgument("server_adapter", default_value="dummy"),
            server,
            runtime,
            sample,
        ]
    )
