"""Headless Gazebo crossing world + ros_gz bridge + sensor TF + walking peds.

    ros2 launch world_model_bringup gz_crossing.launch.py

Topics: /camera/image_raw, /camera/camera_info, /lidar/points, /odom, /tf,
/tf_static, /ground_truth/peds; drive with /cmd_vel. Runs on a CPU-only box
(software rendering): --headless-rendering.

Collect a training rosbag (JPEG images, for ``train_occupancy_head.py --data rosbag``):

    ros2 launch world_model_bringup gz_crossing.launch.py record:=/data/gz/run0 drive:=300 seed:=0

``drive:=N`` wanders the robot around for N s of sim time (gz_data_drive.py);
``record:=DIR`` records camera (compressed), camera_info, LiDAR, odom, TF and
the ped ground truth to DIR. The launch shuts down when the drive finishes.
"""
import os
import shutil
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, ExecuteProcess, OpaqueFunction,
                            RegisterEventHandler, SetEnvironmentVariable, TimerAction)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

RECORD_TOPICS = ["/camera/image_raw/compressed", "/camera/camera_info", "/lidar/points", "/odom",
                 "/tf", "/tf_static", "/ground_truth/peds"]


def _optional(context, share):
    out = []
    record = LaunchConfiguration("record").perform(context)
    drive = LaunchConfiguration("drive").perform(context)
    rtf = float(LaunchConfiguration("rtf").perform(context))
    if rtf != 1.0:   # slower than real time: CPU headroom for Nav2 + a world model
        out.append(TimerAction(period=2.0, actions=[ExecuteProcess(cmd=[
            "gz", "service", "-s", "/world/wm_crossing/set_physics", "--reqtype", "gz.msgs.Physics",
            "--reptype", "gz.msgs.Boolean", "--timeout", "5000",
            "--req", f"max_step_size: 0.004, real_time_factor: {rtf}"], output="screen")]))
    if record:
        out.append(Node(package="image_transport", executable="republish", name="camera_compress",
                        parameters=[{"use_sim_time": True, "in_transport": "raw", "out_transport": "compressed"}],
                        remappings=[("in", "/camera/image_raw"), ("out/compressed", "/camera/image_raw/compressed")]))
        ros2 = [sys.executable, shutil.which("ros2") or "ros2"]
        out.append(ExecuteProcess(cmd=ros2 + ["bag", "record", "-s", "sqlite3", "-o", record] + RECORD_TOPICS,
                                  output="screen"))
    if drive:
        drv = ExecuteProcess(cmd=[sys.executable, os.path.join(share, "scripts", "gz_data_drive.py"),
                                  "--duration", drive, "--seed", LaunchConfiguration("seed").perform(context),
                                  "--ros-args", "-p", "use_sim_time:=true"], output="screen")
        # the drive ends the run: shut everything down (closes the bag cleanly)
        out += [drv, RegisterEventHandler(OnProcessExit(target_action=drv,
                                                        on_exit=[EmitEvent(event=Shutdown(reason="drive done"))]))]
    return out


def generate_launch_description():
    share = get_package_share_directory("world_model_bringup")
    gz_dir = os.path.join(share, "gazebo")
    world = os.path.join(gz_dir, "wm_crossing.sdf")
    sim = {"use_sim_time": True}
    return LaunchDescription([
        DeclareLaunchArgument("record", default_value="", description="rosbag output dir (empty: no recording)"),
        DeclareLaunchArgument("drive", default_value="", description="data-collection drive, seconds of sim time"),
        DeclareLaunchArgument("seed", default_value="0", description="drive waypoint seed"),
        DeclareLaunchArgument("rtf", default_value="1.0", description="Gazebo real-time factor"),
        DeclareLaunchArgument("ped_seed", default_value="0",
                              description="> 0: random per-ped start delays (closed-loop episodes)"),
        SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", gz_dir),
        ExecuteProcess(cmd=["gz", "sim", "-s", "-r", "--headless-rendering", world], output="screen"),
        Node(package="ros_gz_bridge", executable="parameter_bridge", output="screen",
             parameters=[{"config_file": os.path.join(gz_dir, "bridge.yaml")}, sim]),
        # camera optical frame: z forward, x right, y down
        Node(package="tf2_ros", executable="static_transform_publisher", parameters=[sim],
             arguments=["--x", "0.2", "--z", "0.55", "--qw", "0.5", "--qx", "-0.5", "--qy", "0.5",
                        "--qz", "-0.5", "--frame-id", "base_link", "--child-frame-id", "camera_optical"]),
        Node(package="tf2_ros", executable="static_transform_publisher", parameters=[sim],
             arguments=["--z", "0.65", "--frame-id", "base_link", "--child-frame-id", "lidar"]),
        ExecuteProcess(cmd=[sys.executable, os.path.join(share, "scripts", "gz_ped_mover.py"),
                            os.path.join(gz_dir, "wm_crossing_peds.json"),
                            "--ros-args", "-p", "use_sim_time:=true",
                            "-p", ["seed:=", LaunchConfiguration("ped_seed")]], output="screen"),
        OpaqueFunction(function=_optional, kwargs={"share": share}),
    ])
