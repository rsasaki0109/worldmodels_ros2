"""Counterfactual imagination visuals (pure numpy, no ROS)."""
from __future__ import annotations

import numpy as np


def image_msg_to_rgb(msg) -> np.ndarray:
    """sensor_msgs/Image -> (H, W, 3) uint8 RGB."""
    h, w = int(msg.height), int(msg.width)
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    enc = msg.encoding.lower()
    if enc in ("rgb8", "bgr8"):
        arr = buf.reshape(h, w, 3)
        if enc == "bgr8":
            arr = arr[:, :, ::-1]
        return np.ascontiguousarray(arr)
    if enc == "mono8":
        g = buf.reshape(h, w)
        return np.repeat(g[:, :, None], 3, axis=2)
    raise ValueError(f"unsupported image encoding '{msg.encoding}'")


def rgb_to_image_msg(arr: np.ndarray, header) -> "object":
    from sensor_msgs.msg import Image

    msg = Image()
    msg.header = header
    arr = np.ascontiguousarray(arr.astype(np.uint8))
    msg.height, msg.width = arr.shape[:2]
    msg.encoding = "rgb8"
    msg.step = msg.width * 3
    msg.is_bigendian = 0
    msg.data = arr.tobytes()
    return msg


def _resize_rgb(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == size and w == size:
        return img
    ys = np.linspace(0, h - 1, size).astype(np.int32)
    xs = np.linspace(0, w - 1, size).astype(np.int32)
    return img[np.ix_(ys, xs)]


def branch_label(steer: float) -> str:
    if steer < -0.2:
        return "LEFT"
    if steer > 0.2:
        return "RIGHT"
    return "STRAIGHT"


def build_mosaic(
    frames: list[np.ndarray],
    labels: list[str],
    tile: int = 160,
    gap: int = 8,
) -> np.ndarray:
    """Horizontal mosaic of branch endpoint frames with a simple label bar."""
    if not frames:
        raise ValueError("need at least one frame")
    tiles = [_resize_rgb(np.asarray(f, dtype=np.uint8), tile) for f in frames]
    bar_h = 22
    out_w = len(tiles) * tile + (len(tiles) - 1) * gap
    out = np.zeros((tile + bar_h, out_w, 3), dtype=np.uint8)
    x = 0
    for i, t in enumerate(tiles):
        out[bar_h:, x : x + tile] = t
        label = labels[i] if i < len(labels) else f"branch {i}"
        # simple inverted bar as label background
        out[:bar_h, x : x + tile] = (40, 40, 40)
        _draw_label(out, x + 6, 4, label)
        x += tile + gap
    return out


def _draw_label(canvas: np.ndarray, x0: int, y0: int, text: str) -> None:
    """Tiny 5x7 bitmap font for ASCII labels (no PIL)."""
    font = {
        "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
        "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
        "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
        "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
        "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
        "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
        "G": ["01111", "10000", "10000", "10111", "10001", "10001", "01111"],
        "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
        "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
        "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
        "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
        "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
        "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
        "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
        "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
        " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
        "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
        ".": ["00000", "00000", "00000", "00000", "00000", "00100", "00100"],
        "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
        "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
        "2": ["01110", "10001", "00001", "00110", "01000", "10000", "11111"],
        "3": ["01110", "10001", "00001", "00110", "00001", "10001", "01110"],
        "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
        "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
        "6": ["01110", "10000", "11110", "10001", "10001", "10001", "01110"],
        "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
        "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
        "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    }
    cx = x0
    for ch in text.upper():
        glyph = font.get(ch, font[" "])
        for row, bits in enumerate(glyph):
            for col, bit in enumerate(bits):
                if bit == "1":
                    yy, xx = y0 + row, cx + col
                    if 0 <= yy < canvas.shape[0] and 0 <= xx < canvas.shape[1]:
                        canvas[yy, xx] = (230, 230, 230)
        cx += 6


BRANCH_COLORS = [
    (0.2, 0.5, 1.0),   # left  — blue
    (0.2, 0.9, 0.3),   # straight — green
    (1.0, 0.55, 0.1),  # right — orange
]
