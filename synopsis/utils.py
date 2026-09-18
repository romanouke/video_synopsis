from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from .types import BBox


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resize_keep_aspect(frame: np.ndarray, max_width: int = 960) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / float(width)
    new_size = (max_width, max(1, int(height * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def clamp_bbox(bbox: BBox, width: int, height: int) -> BBox:
    x, y, w, h = bbox
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return x, y, w, h


def bbox_area(bbox: BBox) -> int:
    _, _, w, h = bbox
    return max(0, w) * max(0, h)


def bbox_centroid(bbox: BBox) -> Tuple[float, float]:
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0


def bbox_iou(a: BBox, b: BBox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0

    union = aw * ah + bw * bh - inter
    return inter / float(union) if union > 0 else 0.0


def create_local_mask(contour: np.ndarray, bbox: BBox) -> np.ndarray:
    x, y, w, h = bbox
    mask = np.zeros((h, w), dtype=np.uint8)
    shifted = contour.copy()
    shifted[:, 0, 0] -= x
    shifted[:, 0, 1] -= y
    cv2.drawContours(mask, [shifted], -1, 255, thickness=cv2.FILLED)
    return mask


def alpha_blend(canvas: np.ndarray, crop: np.ndarray, mask: np.ndarray, bbox: BBox) -> None:
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return

    mask_f = (mask.astype(np.float32) / 255.0)[..., None]
    roi = canvas[y : y + h, x : x + w]
    if roi.shape[:2] != crop.shape[:2]:
        return
    blended = (crop.astype(np.float32) * mask_f) + (roi.astype(np.float32) * (1.0 - mask_f))
    canvas[y : y + h, x : x + w] = blended.astype(np.uint8)
