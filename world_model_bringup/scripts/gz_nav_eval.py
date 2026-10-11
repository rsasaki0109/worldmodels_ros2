#!/usr/bin/env python3
"""Closed-loop Nav2 evaluation in the wm_crossing world: with vs without a World Model.

Each episode restarts the world (pedestrian start delays from ``ped_seed``,
the same for every condition), starts Nav2 (``gz_nav.launch.py``), sends the
robot from x=0 to the end of the corridor and records, from Gazebo ground
truth:

  * success     -- goal reached within --timeout s of sim time
  * time        -- sim seconds to the goal
  * collision   -- robot footprint (radius --robot-radius) overlapped a ped
  * contact_s   -- sim seconds spent overlapping
  * min_clear   -- closest robot-to-ped surface distance (m)

Conditions: ``baseline`` (LiDAR obstacle layer only) and ``wm`` (+ the
WorldModelLayer fed by a trained occupancy head, --head). Peds do not react to
the robot and have no collision geometry, so contacts are counted, not
simulated.

    python3 gz_nav_eval.py --episodes 10 --head occupancy_head_pixels.npz \\
        --out /tmp/gz_eval.json --ros-args -p use_sim_time:=true
"""
import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.node import Node
from tf2_msgs.msg import TFMessage

HERE = os.path.dirname(os.path.abspath(__file__))
NAV_NODES = ["controller_server", "planner_server", "behavior_server", "bt_navigator"]


def _ped_radii():
    cfg = os.path.join(HERE, "..", "gazebo", "wm_crossing_peds.json")
    return {p["name"]: float(p["radius"]) for p in json.load(open(cfg))["peds"]}


class Episode(Node):
    def __init__(self, robot_radius: float):
        super().__init__("gz_nav_eval")
        self.robot_radius = robot_radius
        self.radii = _ped_radii()
        self.peds = {}
        self.pose = None
        self.reset()
        self.create_subscription(Odometry, "/odom", self._on_odom, 50)
        self.create_subscription(TFMessage, "/ground_truth/peds", self._on_peds, 50)
        self.nav = ActionClient(self, NavigateToPose, "navigate_to_pose")

    def reset(self):
        self.min_clear = math.inf
        self.contact_s = 0.0
        self.contacts = 0
        self._in_contact = False
        self._last_t = None
        self.path_len = 0.0
        self.recording = False

    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_peds(self, msg):
        for t in msg.transforms:
            self.peds[t.child_frame_id] = (t.transform.translation.x, t.transform.translation.y)

    def _on_odom(self, msg):
        p = msg.pose.pose.position
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.recording and self.pose is not None:
            self.path_len += math.hypot(p.x - self.pose[0], p.y - self.pose[1])
        self.pose = (p.x, p.y)
        if not self.recording or not self.peds:
            self._last_t = t
            return
        clear = min(math.hypot(p.x - x, p.y - y) - self.radii.get(n, 0.3) - self.robot_radius
                    for n, (x, y) in self.peds.items())
        self.min_clear = min(self.min_clear, clear)
        contact = clear < 0.0
        if contact and self._last_t is not None:
            self.contact_s += max(0.0, t - self._last_t)
        if contact and not self._in_contact:
            self.contacts += 1
        self._in_contact = contact
        self._last_t = t

    def spin_until(self, cond, timeout_wall: float) -> bool:
        end = time.time() + timeout_wall
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
            if cond():
                return True
        return False

    def nav_active(self) -> bool:
        for n in NAV_NODES:
            cli = self.create_client(GetState, f"/{n}/get_state")
            try:
                if not cli.wait_for_service(timeout_sec=0.5):
                    return False
                fut = cli.call_async(GetState.Request())
                if not self.spin_until(fut.done, 3.0) or fut.result().current_state.label != "active":
                    return False
            finally:
                self.destroy_client(cli)
        return True


def _launch(args, log):
    ros2 = shutil.which("ros2") or "ros2"
    return subprocess.Popen([sys.executable, ros2, "launch", "world_model_bringup"] + args,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                            preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))


def _stop(procs):
    for sig, wait in ((signal.SIGINT, 15), (signal.SIGTERM, 5), (signal.SIGKILL, 2)):
        for p in procs:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, sig)
                except ProcessLookupError:
                    pass
        end = time.time() + wait
        while time.time() < end and any(p.poll() is None for p in procs):
            time.sleep(0.2)


def run_episode(node, cond, seed, a, logdir):
    log = open(os.path.join(logdir, f"ep{seed:03d}_{cond}.log"), "w")
    procs = [_launch(["gz_crossing.launch.py", f"ped_seed:={seed}", f"rtf:={a.rtf}"], log)]
    res = {"condition": cond, "seed": seed}
    try:
        node.pose, node.peds = None, {}
        if not node.spin_until(lambda: node.pose is not None and len(node.peds) == len(node.radii), 90):
            return {**res, "error": "world did not come up"}
        nav = ["gz_nav.launch.py", f"world_model:={'true' if cond == 'wm' else 'false'}"]
        if cond == "wm":
            nav += [f"head:={a.head}", f"model_id:={a.model_id}", f"min_period:={a.min_period}"]
        procs.append(_launch(nav, log))
        if not node.spin_until(node.nav_active, 120) or not node.nav.wait_for_server(timeout_sec=30):
            return {**res, "error": "nav2 did not come up"}
        node.spin_until(lambda: False, 3.0)                     # let the costmaps fill
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = "odom"
        goal.pose.pose.position.x, goal.pose.pose.position.y = a.goal
        goal.pose.pose.orientation.w = 1.0
        node.reset()
        node.recording = True
        t0 = node.now()
        send = node.nav.send_goal_async(goal)
        if not node.spin_until(send.done, 30) or not send.result().accepted:
            return {**res, "error": "goal rejected"}
        result = send.result().get_result_async()
        while not result.done() and node.now() - t0 < a.timeout:
            rclpy.spin_once(node, timeout_sec=0.1)
        node.recording = False
        elapsed = node.now() - t0
        status = result.result().status if result.done() else None
        if not result.done():
            cancel = send.result().cancel_goal_async()
            node.spin_until(cancel.done, 10)
        x, y = node.pose
        dist = math.hypot(x - a.goal[0], y - a.goal[1])
        res.update(success=bool(status == GoalStatus.STATUS_SUCCEEDED and dist < 0.8),
                   status=int(status) if status is not None else -1, time=round(elapsed, 2),
                   dist_to_goal=round(dist, 2), path_len=round(node.path_len, 2),
                   collision=node.contacts > 0, contacts=node.contacts,
                   contact_s=round(node.contact_s, 2),
                   min_clear=round(node.min_clear, 3) if math.isfinite(node.min_clear) else None)
        return res
    finally:
        _stop(procs[::-1])
        log.close()


def summarize(results):
    rows = {}
    for r in results:
        if "error" in r:
            continue
        rows.setdefault(r["condition"], []).append(r)
    out = {}
    for c, rs in rows.items():
        n = len(rs)
        ok = [r for r in rs if r["success"]]
        out[c] = {"episodes": n,
                  "success_rate": round(len(ok) / n, 3),
                  "collision_rate": round(sum(r["collision"] for r in rs) / n, 3),
                  "contacts": sum(r["contacts"] for r in rs),
                  "contact_s": round(sum(r["contact_s"] for r in rs), 2),
                  "mean_time_success": round(sum(r["time"] for r in ok) / len(ok), 2) if ok else None,
                  "mean_min_clear": round(sum(r["min_clear"] for r in rs if r["min_clear"] is not None) / n, 3)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--first-seed", type=int, default=1)
    ap.add_argument("--conditions", default="baseline,wm")
    ap.add_argument("--head", default="")
    ap.add_argument("--model-id", default="pixels")
    ap.add_argument("--min-period", type=float, default=0.2)
    ap.add_argument("--goal", type=lambda s: tuple(float(v) for v in s.split(",")), default=(21.0, 0.0))
    ap.add_argument("--timeout", type=float, default=120.0, help="sim seconds per episode")
    ap.add_argument("--rtf", type=float, default=0.5, help="Gazebo real-time factor (CPU headroom)")
    ap.add_argument("--robot-radius", type=float, default=0.3)
    ap.add_argument("--out", required=True)
    a, ros_args = ap.parse_known_args()
    conds = a.conditions.split(",")
    if "wm" in conds and not a.head:
        raise SystemExit("--head is required for the 'wm' condition")
    a.head = os.path.abspath(a.head) if a.head else ""
    logdir = os.path.splitext(a.out)[0] + "_logs"
    os.makedirs(logdir, exist_ok=True)

    rclpy.init(args=ros_args)
    node = Episode(a.robot_radius)
    results = []
    if os.path.exists(a.out):                       # resume
        results = json.load(open(a.out))["episodes"]
    done = {(r["condition"], r["seed"]) for r in results if "error" not in r}
    try:
        for seed in range(a.first_seed, a.first_seed + a.episodes):
            for cond in conds:
                if (cond, seed) in done:
                    continue
                r = run_episode(node, cond, seed, a, logdir)
                print(json.dumps(r), flush=True)
                results = [x for x in results if (x["condition"], x["seed"]) != (cond, seed)] + [r]
                with open(a.out, "w") as f:
                    json.dump({"summary": summarize(results), "episodes": results}, f, indent=1)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    print(json.dumps(summarize(results), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
