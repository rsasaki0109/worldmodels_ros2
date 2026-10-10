"""Synthetic camera + ground-truth future occupancy, for training occupancy heads.

Real driving / robot datasets rarely ship *future* occupancy labels next to a
camera stream, so this is the bootstrap: a tiny top-down world (a ground plane
plus moving cylindrical obstacles) rendered through a forward pinhole camera,
with the **exact** future occupancy of every obstacle available as labels. It
exists to exercise the full ``camera -> encoder latent -> future occupancy``
pipeline end to end; the same head trains on real data once LiDAR-derived
labels replace the simulator (the head never sees anything but latents).

Everything is pure numpy and seeded, so a dataset is reproducible from its seed
and needs no GPU, no ROS and no download.

Conventions (match ``nav_msgs/OccupancyGrid`` with the robot base frame):
  * the BEV grid is ego-centric at the *observation* time ``t``: ``x`` forward
    in ``[0, grid_size * resolution)``, ``y`` left in
    ``[-grid_size * resolution / 2, +...)``;
  * ``grid[i, j]`` is row ``i`` (``y``) and column ``j`` (``x``), so
    ``grid.ravel()`` is OccupancyGrid ``data`` with origin
    ``(0, -grid_size * resolution / 2)``;
  * future grids stay in that same frame (the frame a costmap would transform
    from), so obstacle motion -- not ego motion -- is what has to be predicted.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera_model import CameraModel


@dataclass
class SimConfig:
    image_size: int = 256          # square camera image (pixels)
    hfov_deg: float = 90.0         # horizontal field of view
    cam_height: float = 1.5        # metres above the ground
    # image row of the horizon, as a fraction: < 0.5 is an off-centre principal
    # point (a sensor shifted down), which looks at the near ground the way a
    # pitched-down camera would while keeping the projection exactly pinhole.
    horizon_row: float = 0.1
    # 0.5 m cells: at 128 px focal length a cell still spans >= 1.5 px out to
    # 8 m; finer cells are below the camera's depth resolution at range.
    grid_size: int = 16            # BEV cells per side
    resolution: float = 0.5        # metres per BEV cell -> 8 m x 8 m
    dt: float = 0.2                # seconds per step
    n_obstacles: tuple = (4, 8)    # inclusive range per episode
    obstacle_speed: float = 1.0    # max obstacle speed (m/s)
    ego_speed: tuple = (0.2, 1.0)  # forward speed range (m/s)
    ego_yaw_rate: float = 0.5      # max |yaw rate| (rad/s)
    action_hold: int = 10          # steps an ego command is held
    noise: float = 3.0             # camera pixel noise (std, uint8 units)

    @property
    def origin(self) -> tuple:
        """(x, y) of the grid's lower-left corner in the ego frame."""
        return (0.0, -self.grid_size * self.resolution / 2.0)

    @property
    def focal(self) -> float:
        return (self.image_size / 2.0) / np.tan(np.radians(self.hfov_deg) / 2.0)

    def camera(self) -> CameraModel:
        """The simulator's camera as a base-frame projection (what a real robot
        builds from CameraInfo + TF)."""
        s = self.image_size
        return CameraModel.pinhole(s, s, self.focal, self.focal, s / 2.0, s * self.horizon_row,
                                   position=(0.0, 0.0, self.cam_height))

    def cell_centers(self) -> np.ndarray:
        """(G, G, 2) ego-frame (x, y) of every cell centre, ``[i, j]`` = (y, x)."""
        g, res = self.grid_size, self.resolution
        x0, y0 = self.origin
        xs = x0 + (np.arange(g) + 0.5) * res
        ys = y0 + (np.arange(g) + 0.5) * res
        X, Y = np.meshgrid(xs, ys)
        return np.stack([X, Y], -1)


@dataclass
class Episode:
    """One simulated drive: camera frames plus the full world state per step."""

    frames: np.ndarray       # (T, S, S, 3) uint8
    actions: np.ndarray      # (T, 2) float32 [v, yaw_rate] applied after frame t
    ego: np.ndarray          # (T, 3) float32 world pose [x, y, yaw] at frame t
    obstacles: np.ndarray    # (T, N, 2) float32 world positions at frame t
    radii: np.ndarray        # (N,) float32
    heights: np.ndarray = None  # (N,) float32 obstacle heights (for visibility)

    def __len__(self) -> int:
        return int(self.frames.shape[0])


def _to_ego(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """World points (..., 2) -> ego frame of ``pose`` = [x, y, yaw]."""
    c, s = np.cos(pose[2]), np.sin(pose[2])
    d = points - pose[:2]
    return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], -1)


def rasterize(ego_points: np.ndarray, radii: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """Discs at ego-frame ``ego_points`` (N, 2) -> (G, G) bool occupancy."""
    g, res = cfg.grid_size, cfg.resolution
    x0, y0 = cfg.origin
    xs = x0 + (np.arange(g) + 0.5) * res
    ys = y0 + (np.arange(g) + 0.5) * res
    occ = np.zeros((g, g), dtype=bool)
    for (px, py), r in zip(ego_points, radii):
        dx = xs[None, :] - px
        dy = ys[:, None] - py
        occ |= dx * dx + dy * dy <= (r + 0.5 * res) ** 2
    return occ


def future_occupancy(ep: Episode, t: int, horizon: int, cfg: SimConfig) -> np.ndarray:
    """Ground truth for frame ``t``: occupancy at ``t+1 .. t+horizon`` in the ego
    frame at ``t``. Returns (horizon, G, G) bool."""
    if t + horizon >= len(ep):
        raise IndexError(f"t={t} + horizon={horizon} exceeds episode length {len(ep)}")
    return np.stack([
        rasterize(_to_ego(ep.obstacles[t + k], ep.ego[t]), ep.radii, cfg)
        for k in range(1, horizon + 1)
    ])


def visibility(ep: Episode, t: int, cfg: SimConfig) -> np.ndarray:
    """Cells the camera can see at frame ``t`` (ego frame at ``t``), (G, G) bool.

    A cell is visible when its ground point projects into the image and the
    camera ray to it is not blocked by an obstacle taller than the ray where it
    enters that obstacle. A cell covered by an obstacle is not occluded by that
    same obstacle (its front face is what the camera sees). Mirrors the camera
    visibility masks real occupancy benchmarks (Occ3D) ship with their labels."""
    centers = cfg.cell_centers().reshape(-1, 2)
    _, _, vis = cfg.camera().project(np.concatenate([centers, np.zeros((len(centers), 1))], 1))
    rel = _to_ego(ep.obstacles[t], ep.ego[t])                     # (N, 2)
    heights = ep.heights if ep.heights is not None else np.full(len(rel), np.inf)
    plen2 = (centers ** 2).sum(1)                                 # (C,)
    for o, r, ho in zip(rel, ep.radii, heights):
        # closest approach of the ray origin->cell to the obstacle centre
        s = np.clip(centers @ o / plen2, 0.0, 1.0)
        d2 = ((s[:, None] * centers - o) ** 2).sum(1)
        hit = d2 < r * r
        s_in = s - np.sqrt(np.clip(r * r - d2, 0.0, None) / plen2)
        own = ((centers - o) ** 2).sum(1) <= (r + 0.5 * cfg.resolution) ** 2
        ray_h = cfg.cam_height * (1.0 - s_in)
        vis &= ~(hit & ~own & (s_in > 0) & (s_in < 1) & (ray_h <= ho))
    return vis.reshape(cfg.grid_size, cfg.grid_size)


def current_occupancy(ep: Episode, t: int, cfg: SimConfig) -> np.ndarray:
    """Occupancy at frame ``t`` itself (ego frame at ``t``), (G, G) bool."""
    return rasterize(_to_ego(ep.obstacles[t], ep.ego[t]), ep.radii, cfg)


class _Renderer:
    """Forward pinhole camera over a checkered ground plane with cylinders."""

    def __init__(self, cfg: SimConfig):
        s = cfg.image_size
        self.cfg = cfg
        self.f = cfg.focal
        self.cx = s / 2.0
        self.cy = s * cfg.horizon_row
        vv, uu = np.mgrid[0:s, 0:s].astype(np.float32) + 0.5
        below = vv > self.cy + 0.5
        dv = np.where(below, vv - self.cy, 1.0)
        self.ground_x = np.where(below, self.f * cfg.cam_height / dv, np.inf)
        self.ground_y = -(uu - self.cx) * self.ground_x / self.f
        self.below = below
        sky_t = np.clip(vv / max(self.cy, 1.0), 0.0, 1.0)[..., None]
        self.sky = ((1 - sky_t) * np.array([110, 150, 210]) + sky_t * np.array([190, 210, 230]))
        self.rows = vv[:, 0]

    def render(self, pose, obstacles, radii, heights, colors, rng) -> np.ndarray:
        cfg, s = self.cfg, self.cfg.image_size
        img = self.sky.copy()

        # ground: world-anchored 1 m checker, so ego motion shows up as texture flow
        c, si = np.cos(pose[2]), np.sin(pose[2])
        gx = np.where(self.below, self.ground_x, 0.0)
        gy = np.where(self.below, self.ground_y, 0.0)
        wx = pose[0] + c * gx - si * gy
        wy = pose[1] + si * gx + c * gy
        checker = ((np.floor(wx) + np.floor(wy)) % 2.0)[..., None]
        ground = 95.0 + 35.0 * checker + np.array([10.0, 20.0, 0.0])
        fog = np.clip(gx / 25.0, 0.0, 1.0)[..., None]
        ground = (1 - fog) * ground + fog * np.array([170.0, 180.0, 185.0])
        img = np.where(self.below[..., None], ground, img)

        # obstacles, far to near (painter's algorithm)
        rel = _to_ego(obstacles, pose)
        for i in np.argsort(-rel[:, 0]):
            x, y = rel[i]
            if x < 0.15:
                continue
            u0 = int(np.floor(self.cx - self.f * (y + radii[i]) / x))
            u1 = int(np.ceil(self.cx - self.f * (y - radii[i]) / x))
            v0 = int(np.floor(self.cy - self.f * (heights[i] - cfg.cam_height) / x))
            v1 = int(np.ceil(self.cy + self.f * cfg.cam_height / x))
            u0, u1, v0, v1 = max(u0, 0), min(u1, s), max(v0, 0), min(v1, s)
            if u0 >= u1 or v0 >= v1:
                continue
            # cylinder shading across the width + distance fog
            span = np.linspace(-1.0, 1.0, u1 - u0)
            shade = 0.65 + 0.35 * np.sqrt(np.clip(1 - span ** 2, 0, 1))
            col = colors[i][None, :] * shade[:, None]
            k = np.clip(x / 25.0, 0.0, 1.0)
            col = (1 - k) * col + k * np.array([170.0, 180.0, 185.0])
            img[v0:v1, u0:u1] = col[None, :, :]

        if cfg.noise > 0:
            img = img + rng.normal(0.0, cfg.noise, img.shape)
        return np.clip(img, 0, 255).astype(np.uint8)


def rollout_episode(steps: int, seed: int, cfg: SimConfig | None = None) -> Episode:
    """Simulate ``steps`` frames of a drive among moving obstacles (seeded)."""
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(seed)
    ren = _Renderer(cfg)
    view = cfg.grid_size * cfg.resolution

    n = int(rng.integers(cfg.n_obstacles[0], cfg.n_obstacles[1] + 1))
    radii = rng.uniform(0.25, 0.6, n).astype(np.float32)
    heights = rng.uniform(0.8, 1.8, n)
    colors = rng.uniform(40, 230, (n, 3))

    pose = np.zeros(3, dtype=np.float64)

    def spawn(far: bool) -> tuple:
        # far: beyond the BEV window, so it enters later (visible to the camera
        # first). near: anywhere ahead, for a populated first frame.
        fwd = rng.uniform(view + 3.0, view + 8.0) if far else rng.uniform(1.5, view + 2.0)
        lat = rng.uniform(-view * 0.6, view * 0.6)
        c, s = np.cos(pose[2]), np.sin(pose[2])
        p = pose[:2] + np.array([c * fwd - s * lat, s * fwd + c * lat])
        heading = rng.uniform(-np.pi, np.pi)
        speed = rng.uniform(0.0, cfg.obstacle_speed)
        return p, speed * np.array([np.cos(heading), np.sin(heading)])

    pos = np.zeros((n, 2))
    vel = np.zeros((n, 2))
    for i in range(n):
        pos[i], vel[i] = spawn(far=False)

    frames, actions, egos, obs = [], [], [], []
    action = np.zeros(2)
    for t in range(steps):
        if t % cfg.action_hold == 0:
            action = np.array([rng.uniform(*cfg.ego_speed),
                               rng.uniform(-cfg.ego_yaw_rate, cfg.ego_yaw_rate)])
        frames.append(ren.render(pose, pos, radii, heights, colors, rng))
        actions.append(action.copy())
        egos.append(pose.copy())
        obs.append(pos.copy())

        # integrate the world: unicycle ego, constant-velocity obstacles
        pose[2] += action[1] * cfg.dt
        pose[0] += action[0] * np.cos(pose[2]) * cfg.dt
        pose[1] += action[0] * np.sin(pose[2]) * cfg.dt
        pos = pos + vel * cfg.dt
        # recycle obstacles that fell far behind / aside
        rel = _to_ego(pos, pose)
        for i in np.where((rel[:, 0] < -3.0) | (np.abs(rel[:, 1]) > view * 1.5)
                          | (rel[:, 0] > view + 10.0))[0]:
            pos[i], vel[i] = spawn(far=True)

    return Episode(
        frames=np.stack(frames),
        actions=np.asarray(actions, np.float32),
        ego=np.asarray(egos, np.float32),
        obstacles=np.asarray(obs, np.float32),
        radii=radii,
        heights=np.asarray(heights, np.float32),
    )
