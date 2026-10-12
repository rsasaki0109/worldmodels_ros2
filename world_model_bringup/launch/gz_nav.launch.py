"""Nav2 (+ optionally the World Model runtime) for the wm_crossing Gazebo world.

Start the world first (gz_crossing.launch.py), then:

    # baseline: LiDAR obstacle layer only
    ros2 launch world_model_bringup gz_nav.launch.py
    # + WorldModelLayer fed by a trained occupancy head on the robot camera
    ros2 launch world_model_bringup gz_nav.launch.py world_model:=true \\
        head:=/path/occupancy_head_pixels.npz model_id:=pixels

and send a NavigateToPose goal in the odom frame (gz_nav_eval.py does this for
a with/without comparison). All conditions share config/gz_nav2.yaml;
``wm_mode`` picks how the prediction is used: ``graded`` / ``lethal`` add
``world_model_layer`` to both costmaps, ``critic`` adds the time-indexed
``WorldModelTimeCritic`` to MPPI instead.
"""
import os
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

NAV_NODES = ["controller_server", "planner_server", "behavior_server", "bt_navigator"]


class _NoAliases(yaml.SafeDumper):
    """rcl's params parser rejects YAML aliases: write shared blocks out in full."""

    def ignore_aliases(self, data):
        return True


WM_MODES = ("graded", "lethal", "critic")


def _params(src: str, world_model: bool, mode: str = "graded") -> str:
    """Expand the YAML (anchors) and wire the world model in. ``mode``:
    ``graded`` / ``lethal`` add the ``WorldModelLayer`` before inflation in both
    costmaps (graded step costs, or every predicted cell LETHAL, the layer's
    default); ``critic`` leaves the costmaps LiDAR-only and adds the
    time-indexed ``WorldModelTimeCritic`` to MPPI. Returns a temp params file
    path."""
    if mode not in WM_MODES:
        raise ValueError(f"wm_mode must be one of {WM_MODES}, got '{mode}'")
    with open(src) as f:
        params = yaml.safe_load(f)
    if world_model and mode == "critic":
        params["controller_server"]["ros__parameters"]["FollowPath"]["critics"].append("WorldModelTimeCritic")
    elif world_model:
        for cm in ("local_costmap", "global_costmap"):
            p = params[cm][cm]["ros__parameters"]
            p["plugins"].insert(p["plugins"].index("inflation_layer"), "world_model_layer")
            if mode == "lethal":
                for k in ("step_costs", "robot_clear_radius"):
                    p["world_model_layer"].pop(k, None)
    out = tempfile.NamedTemporaryFile("w", suffix="_gz_nav2.yaml", delete=False)
    yaml.dump(params, out, Dumper=_NoAliases)
    out.close()
    return out.name


def _setup(context):
    arg = lambda n: LaunchConfiguration(n).perform(context)
    world_model = arg("world_model").lower() in ("1", "true", "yes")
    params = _params(arg("params_file"), world_model, arg("wm_mode"))
    sim = {"use_sim_time": True}
    nav = arg("nav").lower() in ("1", "true", "yes")
    wm_nodes = arg("wm_nodes").lower() in ("1", "true", "yes")
    nodes = [] if not nav else [
        Node(package="nav2_controller", executable="controller_server", output="screen",
             parameters=[params]),
        Node(package="nav2_planner", executable="planner_server", output="screen", parameters=[params]),
        Node(package="nav2_behaviors", executable="behavior_server", output="screen", parameters=[params]),
        Node(package="nav2_bt_navigator", executable="bt_navigator", output="screen", parameters=[params]),
        # bring the nodes up once DDS discovery between them has settled: a
        # lifecycle transition requested too early can lose its reply, and the
        # manager then waits forever
        TimerAction(period=float(arg("lifecycle_delay")), actions=[
            Node(package="nav2_lifecycle_manager", executable="lifecycle_manager",
                 name="lifecycle_manager_navigation", output="screen",
                 parameters=[sim, {"autostart": True, "node_names": NAV_NODES}])]),
    ]
    if world_model and wm_nodes:
        nodes += [
            # one BLAS thread: spinning BLAS workers on every core starve Nav2
            Node(package="world_model_py", executable="runtime_node", output="screen",
                 additional_env={"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                 parameters=[sim, {"adapter": "ijepa", "model_id": arg("model_id"),
                                   "occupancy_head": arg("head"), "occupancy_frame": "base_link",
                                   "horizon": 4, "queue_depth": 1}]),
            Node(package="world_model_py", executable="bag_relay", output="screen",
                 parameters=[sim, {"image_topic": "/camera/image_raw", "action_topic": "/cmd_vel",
                                   "action_mode": "planar",
                                   "min_period": float(arg("min_period")),
                                   "observation_topic": "/world_model_runtime/observation"}]),
        ]
    return nodes


def generate_launch_description():
    share = get_package_share_directory("world_model_bringup")
    return LaunchDescription([
        DeclareLaunchArgument("world_model", default_value="false"),
        DeclareLaunchArgument("head", default_value="", description="trained OccupancyHead .npz"),
        DeclareLaunchArgument("model_id", default_value="pixels",
                              description="encoder the head was trained on: pixels | an HF I-JEPA id"),
        DeclareLaunchArgument("min_period", default_value="0.2",
                              description="camera frame spacing the head was trained at (s)"),
        # split bringup (gz_nav_eval.py): Nav2 first, the world model nodes
        # once Nav2 is active (nav:=false wm_nodes:=true)
        DeclareLaunchArgument("nav", default_value="true", description="start Nav2"),
        DeclareLaunchArgument("wm_nodes", default_value="true",
                              description="with world_model: start runtime_node + bag_relay"),
        DeclareLaunchArgument("wm_mode", default_value="graded",
                              description="how the prediction reaches Nav2: graded | lethal (WorldModelLayer "
                                          "in the costmaps) | critic (time-indexed MPPI critic)"),
        DeclareLaunchArgument("lifecycle_delay", default_value="5.0",
                              description="s between starting the Nav2 nodes and activating them"),
        DeclareLaunchArgument("params_file", default_value=os.path.join(share, "config", "gz_nav2.yaml")),
        OpaqueFunction(function=_setup),
    ])
