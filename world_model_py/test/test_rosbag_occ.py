"""Tests for rosbag -> occupancy labels on a synthetic rosbag2 (no ROS needed).

The bag holds a robot driving +x at 0.5 m/s past a static wall and a person
walking across, with a camera, a 3D LiDAR, odometry and static TF -- exactly
what a real recording carries. Needs ``rosbags`` and ``Pillow`` (skipped otherwise).

    pytest world_model_py/test/test_rosbag_occ.py
"""
import io

import numpy as np
import pytest

rosbags = pytest.importorskip("rosbags")
PIL = pytest.importorskip("PIL")

from world_model_py.occupancy_labels import BevGrid  # noqa: E402

SPEED = 0.5          # robot, m/s along +x
WALL_X = 6.0         # static wall at world x
PERSON = (3.0, 2.0, -0.5)   # x, y0, vy (m, m, m/s)
LIDAR_Z, CAM = 0.5, (0.2, 0.0, 0.4)
GRID = BevGrid(32, 32, 0.25, (0.0, -4.0))


def _world_points(t):
    """Ground (cut off behind the wall, as the wall occludes it), wall, person."""
    gx, gy = np.meshgrid(np.arange(0.0, WALL_X, 0.2), np.arange(-4.0, 4.0, 0.2))
    ground = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], 1)
    wy, wz = np.meshgrid(np.arange(-2.0, 2.0, 0.1), np.arange(0.0, 1.5, 0.1))
    wall = np.stack([np.full(wy.size, WALL_X), wy.ravel(), wz.ravel()], 1)
    ang, pz = np.meshgrid(np.linspace(0, 2 * np.pi, 12, endpoint=False), np.arange(0.0, 1.7, 0.1))
    py = PERSON[1] + PERSON[2] * t
    person = np.stack([PERSON[0] + 0.2 * np.cos(ang.ravel()), py + 0.2 * np.sin(ang.ravel()),
                       pz.ravel()], 1)
    return np.concatenate([ground, wall, person])


def _make_bag(path, duration=6.0):
    from PIL import Image
    from rosbags.rosbag2 import Writer
    from rosbags.typesys import Stores, get_typestore

    ts = get_typestore(Stores.ROS2_JAZZY)
    T = ts.types
    Time = T["builtin_interfaces/msg/Time"]
    hdr = lambda t, f: T["std_msgs/msg/Header"](stamp=Time(sec=int(t), nanosec=int(round((t % 1) * 1e9))), frame_id=f)
    vec = lambda x, y, z: T["geometry_msgs/msg/Vector3"](x=x, y=y, z=z)
    quat = lambda w, x, y, z: T["geometry_msgs/msg/Quaternion"](x=x, y=y, z=z, w=w)

    def tfs(parent, child, xyz, q):
        return T["geometry_msgs/msg/TransformStamped"](
            header=hdr(0.0, parent), child_frame_id=child,
            transform=T["geometry_msgs/msg/Transform"](translation=vec(*xyz), rotation=quat(*q)))

    # base (x fwd, y left, z up) -> optical (x right, y down, z fwd): q = (0.5,-0.5,0.5,-0.5)
    static = T["tf2_msgs/msg/TFMessage"](transforms=[
        tfs("base_link", "velodyne", (0, 0, LIDAR_Z), (1, 0, 0, 0)),
        tfs("base_link", "camera_optical", CAM, (0.5, -0.5, 0.5, -0.5))])
    fields = [T["sensor_msgs/msg/PointField"](name=n, offset=4 * i, datatype=7, count=1)
              for i, n in enumerate("xyz")]
    with Writer(path, version=9) as w:
        c_tf = w.add_connection("/tf_static", static.__msgtype__, typestore=ts)
        c_img = w.add_connection("/camera/color/image_raw/compressed",
                                 "sensor_msgs/msg/CompressedImage", typestore=ts)
        c_info = w.add_connection("/camera/color/camera_info", "sensor_msgs/msg/CameraInfo", typestore=ts)
        c_pts = w.add_connection("/velodyne_points", "sensor_msgs/msg/PointCloud2", typestore=ts)
        c_odom = w.add_connection("/odom", "nav_msgs/msg/Odometry", typestore=ts)
        w.write(c_tf, 0, ts.serialize_cdr(static, static.__msgtype__))
        for k in range(int(duration * 20) + 1):          # 20 Hz odom
            t = k / 20
            ns = int(round(t * 1e9)) + 1
            pose = T["geometry_msgs/msg/PoseWithCovariance"](
                pose=T["geometry_msgs/msg/Pose"](position=T["geometry_msgs/msg/Point"](x=SPEED * t, y=0.0, z=0.0),
                                                 orientation=quat(1, 0, 0, 0)), covariance=np.zeros(36))
            tw = T["geometry_msgs/msg/TwistWithCovariance"](
                twist=T["geometry_msgs/msg/Twist"](linear=vec(SPEED, 0, 0), angular=vec(0, 0, 0)),
                covariance=np.zeros(36))
            od = T["nav_msgs/msg/Odometry"](header=hdr(t, "odom"), child_frame_id="base_link", pose=pose, twist=tw)
            w.write(c_odom, ns, ts.serialize_cdr(od, od.__msgtype__))
            if k % 2:
                continue
            # 10 Hz lidar (in the lidar frame) + camera
            pts = _world_points(t) - (SPEED * t, 0.0, LIDAR_Z)
            pc = T["sensor_msgs/msg/PointCloud2"](
                header=hdr(t, "velodyne"), height=1, width=len(pts), fields=fields, is_bigendian=False,
                point_step=12, row_step=12 * len(pts), data=np.frombuffer(pts.astype("<f4").tobytes(), np.uint8),
                is_dense=True)
            w.write(c_pts, ns, ts.serialize_cdr(pc, pc.__msgtype__))
            buf = io.BytesIO()
            Image.fromarray(np.full((48, 64, 3), 90, np.uint8)).save(buf, "JPEG")
            im = T["sensor_msgs/msg/CompressedImage"](header=hdr(t, "camera_optical"), format="jpeg",
                                                      data=np.frombuffer(buf.getvalue(), np.uint8))
            w.write(c_img, ns, ts.serialize_cdr(im, im.__msgtype__))
            info = T["sensor_msgs/msg/CameraInfo"](
                header=hdr(t, "camera_optical"), height=48, width=64, distortion_model="plumb_bob",
                d=np.zeros(5), k=np.array([32.0, 0, 32, 0, 32, 24, 0, 0, 1]), r=np.eye(3).ravel(),
                p=np.zeros(12), binning_x=0, binning_y=0,
                roi=T["sensor_msgs/msg/RegionOfInterest"](x_offset=0, y_offset=0, height=0, width=0,
                                                          do_rectify=False))
            w.write(c_info, ns, ts.serialize_cdr(info, info.__msgtype__))


@pytest.fixture(scope="module")
def bag(tmp_path_factory):
    from world_model_py.rosbag_occ import BagOccupancy
    path = tmp_path_factory.mktemp("bags") / "synthetic"
    _make_bag(path)
    return BagOccupancy(path, GRID, horizon=4, dt=0.5, image_width=64)


def _cell(x, y):
    i, j, ok = GRID.index(np.array([[x, y]]))
    assert ok[0]
    return int(i[0]), int(j[0])


def test_topics_frames_and_mounts(bag):
    assert bag.topics["lidar"] == "/velodyne_points" and bag.topics["odom"] == "/odom"
    assert bag.base_frame == "base_link"
    assert np.allclose(bag.T_base_lidar[:3, 3], (0, 0, LIDAR_Z))
    assert np.allclose(bag.T_base_cam[:3, 3], CAM)
    u, v, ok = bag.camera.project(np.array([[5.0, 0.0, CAM[2]]]))   # straight ahead, cam height
    assert ok[0] and abs(u[0] - 32) < 1e-6 and abs(v[0] - 24) < 1e-6
    assert len(bag.frames) == 61 and bag.frames[0].shape == (48, 64, 3)


def test_ego_command_and_valid_range(bag):
    vi = bag.valid_indices()
    assert vi[0] == 1 and bag.times[vi[-1]] + 2.0e9 <= bag._scan_t[-1]
    v, w = bag.ego_command(10)
    assert v == pytest.approx(SPEED, abs=1e-3) and w == pytest.approx(0.0, abs=1e-6)


def test_occupancy_labels_static_and_moving(bag):
    i = 10                                   # t = 1.0 s, robot at x = 0.5
    t = bag.times[i] * 1e-9
    x0 = SPEED * t
    now = bag.occupancy_at(i, 0.0)
    fut = bag.future_occupancy(i)
    assert fut.shape == (4, 32, 32)
    assert now[_cell(WALL_X - x0 + 0.05, 0.1)]                    # the wall
    assert not now[_cell(2.0, -3.0)]                               # ground is not an obstacle
    y_now = PERSON[1] + PERSON[2] * t
    y_fut = PERSON[1] + PERSON[2] * (t + 1.0)                      # k = 2 -> t + 1 s
    assert now[_cell(PERSON[0] - x0, y_now)]
    assert fut[1][_cell(PERSON[0] - x0, y_fut)]                    # the person moved, in frame t
    assert not fut[1][_cell(PERSON[0] - x0, y_now)]
    assert fut[3][_cell(WALL_X - x0 + 0.05, 0.1)]                  # the wall stays put


def test_visibility(bag):
    vis = bag.visibility(10)
    x0 = SPEED * bag.times[10] * 1e-9
    assert vis[_cell(3.0, 0.0)]                                    # open floor ahead
    assert not vis[_cell(WALL_X - x0 + 1.0, 0.0)]                  # behind the wall
    assert not vis[_cell(0.3, 3.5)]                                # outside the camera's view
