"""Convert between ``world_model_msgs`` and the adapter dataclasses.

Kept in one place so adapters never import ROS 2. Image conversion is done
by hand (rgb8/bgr8/mono8) to avoid a hard dependency on cv_bridge.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from std_msgs.msg import Header

from sensor_msgs.msg import Image
from nav_msgs.msg import OccupancyGrid

from world_model_msgs.msg import (
    ActionCondition as ActionConditionMsg,
    FutureState as FutureStateMsg,
    LatentState as LatentStateMsg,
    Observation as ObservationMsg,
    RiskScore as RiskScoreMsg,
)

from .adapters import ActionCondition, FuturePrediction, Observation


# ---------------------------------------------------------------- images

def image_msg_to_np(msg: Image) -> Optional[np.ndarray]:
    if msg.height == 0 or msg.width == 0 or len(msg.data) == 0:
        return None
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    enc = msg.encoding.lower()
    if enc in ("rgb8", "bgr8"):
        arr = buf.reshape(msg.height, msg.width, 3)
        if enc == "bgr8":
            arr = arr[:, :, ::-1]
        return np.ascontiguousarray(arr)
    if enc == "mono8":
        return buf.reshape(msg.height, msg.width)
    # unknown encoding: best-effort flat view, leave to the adapter.
    return buf.copy()


def np_to_image_msg(arr: np.ndarray, header: Header) -> Image:
    msg = Image()
    msg.header = header
    arr = np.ascontiguousarray(arr.astype(np.uint8))
    if arr.ndim == 2:
        msg.height, msg.width = arr.shape
        msg.encoding = "mono8"
        msg.step = msg.width
    else:
        msg.height, msg.width = arr.shape[:2]
        msg.encoding = "rgb8"
        msg.step = msg.width * 3
    msg.is_bigendian = 0
    msg.data = arr.tobytes()
    return msg


# ------------------------------------------------------------ inbound (msg -> dc)

def observation_from_msg(msg: ObservationMsg) -> Observation:
    return Observation(
        image=image_msg_to_np(msg.image),
        ego_state=np.asarray(msg.ego_state, dtype=np.float32) if len(msg.ego_state) else None,
        action_history=(
            np.asarray(msg.action_history, dtype=np.float32).reshape(-1, msg.action_dim)
            if len(msg.action_history) and msg.action_dim
            else None
        ),
        instruction=msg.instruction,
    )


def action_from_msg(msg: ActionConditionMsg) -> ActionCondition:
    if len(msg.action) and msg.action_dim:
        action = np.asarray(msg.action, dtype=np.float32).reshape(msg.horizon or -1, msg.action_dim)
    else:
        action = None
    return ActionCondition(action=action, dt=msg.dt or 0.1)


# ----------------------------------------------------------- outbound (dc -> msg)

def future_to_msg(pred: FuturePrediction, header: Header) -> FutureStateMsg:
    msg = FutureStateMsg()
    msg.header = header
    msg.dt = float(pred.dt)
    for vec in pred.latents:
        latent = LatentStateMsg()
        latent.header = header
        flat = np.asarray(vec, dtype=np.float32).ravel()
        latent.data = flat.tolist()
        latent.shape = list(np.asarray(vec).shape)
        latent.encoding = pred.risk_label
        msg.latents.append(latent)
    for frame in pred.frames:
        msg.frames.append(np_to_image_msg(np.asarray(frame), header))
    return msg


def risk_to_msg(pred: FuturePrediction, header: Header) -> RiskScoreMsg:
    msg = RiskScoreMsg()
    msg.header = header
    msg.score = float(pred.risk)
    msg.confidence = float(pred.risk_confidence)
    msg.label = pred.risk_label
    return msg


def occupancy_to_grids(pred: FuturePrediction, header: Header, resolution: float = 0.1):
    grids = []
    for g in pred.occupancy:
        grid = OccupancyGrid()
        grid.header = header
        arr = np.asarray(g, dtype=np.int8)
        grid.info.resolution = float(resolution)
        grid.info.height, grid.info.width = arr.shape
        grid.info.origin.orientation.w = 1.0
        grid.data = arr.ravel().tolist()
        grids.append(grid)
    return grids
