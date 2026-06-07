#!/usr/bin/env python3
"""Headless smoke: Nav2 loopback starts with World Model plugins loaded."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def _plugin_xml_ok(share: str, class_name: str) -> bool:
    for name in ("world_model_costmap_plugins.xml", "world_model_dwb_critics_plugins.xml"):
        path = os.path.join(share, name)
        if not os.path.isfile(path):
            continue
        root = ET.parse(path).getroot()
        for cls in root.iter("class"):
            if cls.get("type") == class_name:
                return True
    return False


def _library_exists(prefix: str, lib: str) -> bool:
    return os.path.isfile(os.path.join(prefix, "lib", lib))


def main() -> int:
    from ament_index_python.packages import get_package_prefix, get_package_share_directory

    costmap_share = get_package_share_directory("world_model_costmap")
    dwb_share = get_package_share_directory("world_model_dwb_critics")
    costmap_prefix = get_package_prefix("world_model_costmap")
    dwb_prefix = get_package_prefix("world_model_dwb_critics")

    assert _plugin_xml_ok(costmap_share, "world_model_costmap::WorldModelLayer")
    assert _plugin_xml_ok(dwb_share, "world_model_dwb_critics::WorldModelOccupancyCritic")
    assert _library_exists(costmap_prefix, "libworld_model_layer.so")
    assert _library_exists(dwb_prefix, "libworld_model_occupancy_critic.so")

    launch = subprocess.Popen(
        [
            "ros2", "launch", "world_model_bringup", "nav2_loopback_world_model.launch.py",
            "use_rviz:=false",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
    )
    try:
        deadline = time.time() + 120.0
        log = ""
        saw_costmap = saw_dwb = False
        while time.time() < deadline:
            line = launch.stdout.readline() if launch.stdout else ""
            if line:
                log += line
                if "WorldModelLayer" in line:
                    saw_costmap = True
                if "WorldModelOccupancyCritic" in line and "subscribed" in line:
                    saw_dwb = True
                if saw_costmap and saw_dwb:
                    print("OK: Nav2 loopback loaded costmap layer + DWB critic")
                    return 0
            if launch.poll() is not None:
                break
            if not line:
                time.sleep(0.05)
        print(log[-6000:], file=sys.stderr)
        raise RuntimeError(
            f"timed out (costmap={saw_costmap}, dwb={saw_dwb}) waiting for plugin init logs"
        )
    finally:
        if launch.poll() is None:
            os.killpg(os.getpgid(launch.pid), signal.SIGTERM)
            launch.wait(timeout=20)


if __name__ == "__main__":
    sys.exit(main())
