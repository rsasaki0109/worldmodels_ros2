"""Real camera -> I-JEPA + occupancy head -> FutureOccupancy -> Nav2 costmap (NOT in CI).

End-to-end check of the ROS path with a trained head on real frames:

  1. starts ``runtime_node`` (``adapter:=ijepa``, ``occupancy_head:=<npz>``) and a
     standalone ``nav2_costmap_2d`` whose only plugin is
     ``world_model_costmap::WorldModelLayer``;
  2. publishes ``world_model_msgs/Observation`` built from nuScenes ``CAM_FRONT``
     frames (image + last ego command), one at a time;
  3. checks every ``FutureOccupancy`` (horizon, resolution, origin, unknown
     outside the camera's view), that the costmap's lethal cells equal the union
     of the predicted grids, and that the runtime's grids equal the head run
     offline on the same frames (numpy, no ROS).

Needs ROS 2 Jazzy + ``ros-jazzy-nav2-costmap-2d``, a built workspace, a head
from ``train_occupancy_head.py --data nuscenes --encoder ijepa`` and nuScenes:

    source install/setup.bash
    python3 world_model_py/test/integration_occupancy_nav2.py \\
        --head occupancy_head_ijepa.npz --nuscenes-root /data/nuscenes --frames 6
"""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Header

from world_model_msgs.msg import FutureOccupancy, Observation
from world_model_py import conversions as conv

# the standalone nav2_costmap_2d executable is a lifecycle node named /costmap
# that publishes its grid on /costmap
COSTMAP_YAML = """
costmap:
  ros__parameters:
    global_frame: base_link
    robot_base_frame: base_link
    rolling_window: true
    width: 50
    height: 50
    resolution: 0.5
    update_frequency: 5.0
    publish_frequency: 5.0
    always_send_full_costmap: true
    plugins: ["world_model_layer"]
    world_model_layer:
      plugin: "world_model_costmap::WorldModelLayer"
      topic: /world_model_runtime/future_occupancy
      occupied_threshold: 50
"""


class Harness(Node):
    def __init__(self):
        super().__init__("occupancy_nav2_harness")
        self.pub = self.create_publisher(Observation, "/world_model_runtime/observation", 10)
        self.occ = []
        self.costmap = None
        self.create_subscription(FutureOccupancy, "/world_model_runtime/future_occupancy",
                                 self.occ.append, 10)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, "/costmap", self._on_costmap, qos)

    def _on_costmap(self, msg):
        self.costmap = msg

    def spin_until(self, cond, timeout):
        end = time.time() + timeout
        while time.time() < end and not cond():
            rclpy.spin_once(self, timeout_sec=0.1)
        return cond()


def grids_of(msg: FutureOccupancy) -> list:
    return [np.asarray(g.data, np.int8).reshape(g.info.height, g.info.width) for g in msg.grids]


def costmap_lethal(cm: OccupancyGrid) -> np.ndarray:
    return np.asarray(cm.data, np.int8).reshape(cm.info.height, cm.info.width) >= 99


def union_in_costmap(msg: FutureOccupancy, cm: OccupancyGrid, threshold=50) -> np.ndarray:
    """Expected lethal cells: the union of predicted grids, mapped like the layer does."""
    out = np.zeros((cm.info.height, cm.info.width), bool)
    for g in msg.grids:
        arr = np.asarray(g.data, np.int8).reshape(g.info.height, g.info.width)
        jj, ii = np.nonzero(arr >= threshold)
        wx = g.info.origin.position.x + (ii + 0.5) * g.info.resolution
        wy = g.info.origin.position.y + (jj + 0.5) * g.info.resolution
        mx = np.floor((wx - cm.info.origin.position.x) / cm.info.resolution).astype(int)
        my = np.floor((wy - cm.info.origin.position.y) / cm.info.resolution).astype(int)
        ok = (mx >= 0) & (mx < cm.info.width) & (my >= 0) & (my < cm.info.height)
        out[my[ok], mx[ok]] = True
    return out


def _exe(package: str, name: str) -> str:
    from ament_index_python.packages import get_package_prefix
    return os.path.join(get_package_prefix(package), "lib", package, name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--head", required=True)
    ap.add_argument("--nuscenes-root", required=True)
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--scene", default="scene-0796")
    ap.add_argument("--start", type=int, default=60)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--image-width", type=int, default=448)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from world_model_py.nuscenes_occ import NuScenesTables, SceneOccupancy
    from world_model_py.occupancy_head import OccupancyHead
    from world_model_py.occupancy_labels import BevGrid

    head = OccupancyHead.load(args.head)
    tables = NuScenesTables(args.nuscenes_root, args.version)
    grid = BevGrid(*head.grid_shape, head.resolution, head.origin)
    scene = SceneOccupancy(tables, args.scene, grid, horizon=head.horizon, dt=head.dt,
                           image_width=args.image_width)
    idx = list(range(args.start, args.start + args.frames))

    ros2 = [sys.executable, shutil.which("ros2")]
    nodes = subprocess.run(ros2 + ["node", "list"], capture_output=True, text=True).stdout.split()
    if "/costmap" in nodes or "/world_model_runtime" in nodes:
        raise SystemExit(f"stale ROS nodes are running ({nodes}); stop them first")

    yaml = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.write(COSTMAP_YAML)
    yaml.close()
    # python tools run with *this* interpreter: their shebangs may point at a
    # python that is not the one ROS 2 was built for
    procs = [
        subprocess.Popen([sys.executable, "-m", "world_model_py.runtime_node", "--ros-args",
                          "-p", "adapter:=ijepa", "-p", f"occupancy_head:={os.path.abspath(args.head)}",
                          "-p", f"horizon:={head.horizon}"]),
        # C++ executables are started directly (not via `ros2 run`) so the
        # shutdown signal reaches them and no stale /costmap survives a run
        subprocess.Popen([_exe("nav2_costmap_2d", "nav2_costmap_2d"), "--ros-args",
                          "--params-file", yaml.name]),
        subprocess.Popen([_exe("tf2_ros", "static_transform_publisher"),
                          "--frame-id", "odom", "--child-frame-id", "base_link"]),
    ]
    failures = []
    try:
        rclpy.init()
        node = Harness()
        # bring the costmap up (it is a lifecycle node)
        time.sleep(3.0)
        for t in ("configure", "activate"):
            r = subprocess.run(ros2 + ["lifecycle", "set", "/costmap", t], check=False,
                               capture_output=True, text=True, timeout=60)
            print(f"costmap {t}: {(r.stdout + r.stderr).strip()}", flush=True)
        if not node.spin_until(lambda: node.count_subscribers("/world_model_runtime/observation") > 0, 180):
            raise SystemExit("runtime_node never subscribed to observations")
        time.sleep(2.0)

        # offline reference: same frames through the numpy head (fresh adapter)
        from world_model_py.adapters.jepa import IJepaAdapter, _HFEncoder
        from world_model_py.adapters.base import Observation as Obs
        offline = IJepaAdapter(_HFEncoder(), occupancy_head=head)
        frames = [scene.image(i) for i in idx]
        egos = [scene.ego_command(i) for i in idx]

        results = []
        for k, (img, ego) in enumerate(zip(frames, egos)):
            msg = Observation()
            msg.header = Header(frame_id="base_link")
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.image = conv.np_to_image_msg(img, msg.header)
            msg.action_history = [float(v) for v in ego]
            msg.action_dim = 2
            n_before = len(node.occ)
            node.pub.publish(msg)
            if not node.spin_until(lambda: len(node.occ) > n_before, 120):
                failures.append(f"frame {k}: no FutureOccupancy within 120 s")
                continue
            occ = node.occ[-1]
            ref = offline.predict_future(Obs(image=img, action_history=np.asarray([ego])),
                                         horizon=head.horizon)
            got = grids_of(occ)
            g0 = occ.grids[0].info
            checks = {
                "horizon": len(got) == head.horizon,
                "resolution": abs(g0.resolution - head.resolution) < 1e-6,
                "origin": abs(g0.origin.position.x - head.origin[0]) < 1e-6
                          and abs(g0.origin.position.y - head.origin[1]) < 1e-6,
                "unknown outside FOV": all((g[~head.fov_mask()] == -1).all() for g in got),
                "matches offline head": all(np.array_equal(a, b) for a, b in zip(got, ref.occupancy)),
            }
            # the costmap reflects this prediction within a refresh cycle or two;
            # stale cells from the previous prediction must be gone too
            end = time.time() + 3.0
            cm, exp, lethal = None, None, None
            while time.time() < end:
                node.costmap = None
                node.spin_until(lambda: node.costmap is not None, 2.0)
                if node.costmap is None:
                    continue
                cm = node.costmap
                exp, lethal = union_in_costmap(occ, cm), costmap_lethal(cm)
                if np.array_equal(exp, lethal):
                    break
            if cm is not None:
                same = bool(np.array_equal(exp, lethal))
                checks["costmap == union of predicted grids"] = same
                checks["costmap has predicted obstacles"] = bool(lethal.any())
                if not same:
                    print(f"  costmap diff: {int((lethal & ~exp).sum())} extra (stale?) / "
                          f"{int((exp & ~lethal).sum())} missing of {int(exp.sum())}", flush=True)
            else:
                checks["costmap published"] = False
            results.append((k, checks, got, cm))
            status = " ".join(f"{n}={'ok' if v else 'FAIL'}" for n, v in checks.items())
            print(f"frame {k}: occupied cells/step {[int((g >= 50).sum()) for g in got]}  {status}",
                  flush=True)
            failures += [f"frame {k}: {n}" for n, v in checks.items() if not v]

        if args.out and results:
            _render(args.out, frames, results, head)
    finally:
        for p in procs:
            p.send_signal(signal.SIGINT)
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        if rclpy.ok():
            rclpy.shutdown()
        os.unlink(yaml.name)

    print("\nPASS" if not failures else "\nFAIL:\n  " + "\n  ".join(failures))
    return 0 if not failures else 1


def _render(path, frames, results, head):
    """camera | predicted union (0..100, -1 grey) | costmap lethal cells (forward up)."""
    from PIL import Image
    rows = []
    for (k, _, got, cm), img in zip(results, frames):
        h = img.shape[0]
        g = np.max(np.stack(got), 0).astype(np.int16)
        tile = np.full(g.shape + (3,), 30, np.uint8)
        tile[g < 0] = (70, 70, 70)
        hot = g >= 50
        tile[hot] = (230, 60, 60)
        warm = (g > 10) & ~hot
        tile[warm] = (120, 60, 40)
        tile = tile.transpose(1, 0, 2)[::-1, ::-1]
        rep = h // tile.shape[0]
        t1 = np.kron(tile, np.ones((rep, rep, 1), np.uint8))
        cols = [img, np.pad(t1, ((0, h - t1.shape[0]), (4, 4), (0, 0)))]
        if cm is not None:
            lethal = costmap_lethal(cm)
            c = np.where(lethal[..., None], np.array([255, 255, 255], np.uint8),
                         np.array([20, 20, 40], np.uint8)).astype(np.uint8)
            c = c[::-1]                                  # +y up -> image rows
            c = np.rot90(c, k=1)                         # forward (+x) up
            rep = max(1, h // c.shape[0])
            t2 = np.kron(c, np.ones((rep, rep, 1), np.uint8))
            cols.append(np.pad(t2, ((0, h - t2.shape[0]), (0, 0), (0, 0))))
        rows.append(np.concatenate(cols, 1))
    Image.fromarray(np.concatenate(rows, 0)).save(path)
    print(f"wrote {path}")


if __name__ == "__main__":
    sys.exit(main())
