"""Convert one nuScenes scene into a rosbag2 a robot could have recorded (NOT in CI).

Writes what ``world_model_py.rosbag_occ`` expects from a real recording, and
nothing nuScenes-specific:

  * ``/CAM_FRONT/image_raw/compressed`` (the original JPEGs) + ``/CAM_FRONT/camera_info``
  * ``/LIDAR_TOP/points`` (``PointCloud2`` x/y/z in the sensor frame)
  * ``/odom`` (``nav_msgs/Odometry``, ``odom -> base_link`` from the ego poses)
  * ``/tf_static`` (``base_link -> CAM_FRONT``, ``base_link -> LIDAR_TOP``)

Used by ``validate_rosbag_vs_nuscenes.py`` to check the rosbag path against the
nuScenes labeller on real data.

    python3 world_model_py/test/nuscenes_to_rosbag.py --nuscenes-root /data/nuscenes \\
        --scene scene-0796 --out /tmp/scene-0796
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from world_model_py.nuscenes_occ import NuScenesTables


def convert(tables: NuScenesTables, scene_name: str, out: str) -> None:
    from rosbags.rosbag2 import Writer
    from rosbags.typesys import Stores, get_typestore

    ts = get_typestore(Stores.ROS2_JAZZY)
    T = ts.types
    scene = next(s for s in tables.scene if s["name"] == scene_name)
    chans = tables.by_scene[scene["token"]]
    root = tables.root
    exists = lambda d: os.path.exists(os.path.join(root, d["filename"]))
    cams = sorted(filter(exists, chans["CAM_FRONT"]), key=lambda d: d["timestamp"])
    lidars = sorted(filter(exists, chans["LIDAR_TOP"]), key=lambda d: d["timestamp"])

    def stamp(us):
        return T["builtin_interfaces/msg/Time"](sec=int(us // 1_000_000), nanosec=int(us % 1_000_000) * 1000)

    hdr = lambda us, f: T["std_msgs/msg/Header"](stamp=stamp(us), frame_id=f)
    vec = lambda v: T["geometry_msgs/msg/Vector3"](x=float(v[0]), y=float(v[1]), z=float(v[2]))
    quat = lambda q: T["geometry_msgs/msg/Quaternion"](w=float(q[0]), x=float(q[1]), y=float(q[2]), z=float(q[3]))

    def mount(sd, child):
        c = tables.calib[sd["calibrated_sensor_token"]]
        return T["geometry_msgs/msg/TransformStamped"](
            header=hdr(sd["timestamp"], "base_link"), child_frame_id=child,
            transform=T["geometry_msgs/msg/Transform"](translation=vec(c["translation"]),
                                                       rotation=quat(c["rotation"])))

    tf_static = T["tf2_msgs/msg/TFMessage"](transforms=[mount(cams[0], "CAM_FRONT"),
                                                        mount(lidars[0], "LIDAR_TOP")])
    fields = [T["sensor_msgs/msg/PointField"](name=n, offset=4 * i, datatype=7, count=1)
              for i, n in enumerate("xyz")]
    K = np.asarray(tables.calib[cams[0]["calibrated_sensor_token"]]["camera_intrinsic"], float)

    events = [("cam", d) for d in cams] + [("lidar", d) for d in lidars]
    events.sort(key=lambda e: e[1]["timestamp"])
    with Writer(out, version=9) as w:
        c_tf = w.add_connection("/tf_static", "tf2_msgs/msg/TFMessage", typestore=ts)
        c_img = w.add_connection("/CAM_FRONT/image_raw/compressed", "sensor_msgs/msg/CompressedImage", typestore=ts)
        c_info = w.add_connection("/CAM_FRONT/camera_info", "sensor_msgs/msg/CameraInfo", typestore=ts)
        c_pts = w.add_connection("/LIDAR_TOP/points", "sensor_msgs/msg/PointCloud2", typestore=ts)
        c_odom = w.add_connection("/odom", "nav_msgs/msg/Odometry", typestore=ts)
        t0 = events[0][1]["timestamp"]
        w.write(c_tf, t0 * 1000, ts.serialize_cdr(tf_static, "tf2_msgs/msg/TFMessage"))
        for kind, sd in events:
            us = sd["timestamp"]
            p = tables.ego_pose[sd["ego_pose_token"]]
            od = T["nav_msgs/msg/Odometry"](
                header=hdr(us, "odom"), child_frame_id="base_link",
                pose=T["geometry_msgs/msg/PoseWithCovariance"](
                    pose=T["geometry_msgs/msg/Pose"](
                        position=T["geometry_msgs/msg/Point"](x=float(p["translation"][0]),
                                                              y=float(p["translation"][1]),
                                                              z=float(p["translation"][2])),
                        orientation=quat(p["rotation"])), covariance=np.zeros(36)),
                twist=T["geometry_msgs/msg/TwistWithCovariance"](
                    twist=T["geometry_msgs/msg/Twist"](linear=vec((0, 0, 0)), angular=vec((0, 0, 0))),
                    covariance=np.zeros(36)))
            w.write(c_odom, us * 1000, ts.serialize_cdr(od, "nav_msgs/msg/Odometry"))
            if kind == "cam":
                data = np.fromfile(os.path.join(root, sd["filename"]), np.uint8)
                im = T["sensor_msgs/msg/CompressedImage"](header=hdr(us, "CAM_FRONT"), format="jpeg", data=data)
                w.write(c_img, us * 1000, ts.serialize_cdr(im, "sensor_msgs/msg/CompressedImage"))
                info = T["sensor_msgs/msg/CameraInfo"](
                    header=hdr(us, "CAM_FRONT"), height=sd["height"], width=sd["width"],
                    distortion_model="plumb_bob", d=np.zeros(5), k=K.ravel(), r=np.eye(3).ravel(),
                    p=np.zeros(12), binning_x=0, binning_y=0,
                    roi=T["sensor_msgs/msg/RegionOfInterest"](x_offset=0, y_offset=0, height=0, width=0,
                                                              do_rectify=False))
                w.write(c_info, us * 1000, ts.serialize_cdr(info, "sensor_msgs/msg/CameraInfo"))
            else:
                pts = np.fromfile(os.path.join(root, sd["filename"]), np.float32).reshape(-1, 5)[:, :3]
                pc = T["sensor_msgs/msg/PointCloud2"](
                    header=hdr(us, "LIDAR_TOP"), height=1, width=len(pts), fields=fields,
                    is_bigendian=False, point_step=12, row_step=12 * len(pts),
                    data=np.frombuffer(np.ascontiguousarray(pts, "<f4").tobytes(), np.uint8),
                    is_dense=True)
                w.write(c_pts, us * 1000, ts.serialize_cdr(pc, "sensor_msgs/msg/PointCloud2"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nuscenes-root", required=True)
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--scene", default="scene-0796")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    convert(NuScenesTables(args.nuscenes_root, args.version), args.scene, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
