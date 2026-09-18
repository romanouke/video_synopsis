from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from .types import Detection
from .utils import bbox_area, clamp_bbox, create_local_mask


@dataclass
class YoloDetectorConfig:
    model: str = "yolov8n.pt"  # model name/path for ultralytics YOLO if available
    conf_threshold: float = 0.3
    classes: list[int] | None = None


class YoloDetector:
    """Optional YOLO detector using `ultralytics` if installed.

    Falls back by raising ImportError when `ultralytics` is not available.
    """
    def __init__(self, config: YoloDetectorConfig | None = None) -> None:
        cfg = config or YoloDetectorConfig()
        try:
            from ultralytics import YOLO

            self._model = YOLO(cfg.model)
        except Exception as exc:  # intentionally broad - model may not be available
            raise ImportError("ultralytics YOLO not available") from exc
        self.config = cfg

    def detect(self, frame: np.ndarray, frame_index: int) -> List[Detection]:
        # ultralytics expects BGR or RGB depending; pass as numpy array
        res = self._model(frame, conf=self.config.conf_threshold, classes=self.config.classes)
        detections: List[Detection] = []
        height, width = frame.shape[:2]
        # results may be a list; take first
        results = res[0] if isinstance(res, (list, tuple)) else res
        boxes = getattr(results, "boxes", None)
        if boxes is None:
            return detections

        for box in boxes:
            # ultralytics Box: xyxy, conf, cls
            xyxy = box.xyxy[0].cpu().numpy() if hasattr(box, "xyxy") else box[:4]
            conf = float(box.conf[0]) if hasattr(box, "conf") else float(box[4])
            x1, y1, x2, y2 = map(int, xyxy.tolist())
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            bbox = clamp_bbox((x1, y1, w, h), width, height)
            x, y, w, h = bbox
            crop = frame[y : y + h, x : x + w].copy()
            mask = 255 * np.ones((h, w), dtype=np.uint8)
            detections.append(Detection(frame_index=frame_index, bbox=bbox, confidence=conf, crop=crop, mask=mask))

        detections.sort(key=lambda d: bbox_area(d.bbox), reverse=True)
        return detections


@dataclass
class MotionDetectorConfig:
    min_area: int = 800
    blur_size: int = 5
    morph_kernel: int = 3
    shadow_threshold: int = 200


class MotionDetector:
    def __init__(self, config: MotionDetectorConfig | None = None) -> None:
        self.config = config or MotionDetectorConfig()
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=250,
            varThreshold=28,
            detectShadows=False,
        )

    def detect(self, frame: np.ndarray, frame_index: int) -> List[Detection]:
        blurred = cv2.GaussianBlur(frame, (self.config.blur_size, self.config.blur_size), 0)
        fg_mask = self.bg_subtractor.apply(blurred)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.config.morph_kernel, self.config.morph_kernel))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: List[Detection] = []
        height, width = frame.shape[:2]

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config.min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            bbox = clamp_bbox((x, y, w, h), width, height)
            if bbox_area(bbox) < self.config.min_area:
                continue

            x, y, w, h = bbox
            crop = frame[y : y + h, x : x + w].copy()
            local_mask = create_local_mask(contour, bbox)
            bbox_ratio = bbox_area(bbox) / float(width * height)
            fill_ratio = float(area) / float(max(1, bbox_area(bbox)))
            confidence = float(np.clip(0.2 + 0.5 * min(1.0, fill_ratio) + 0.3 * min(1.0, bbox_ratio / 0.08), 0.0, 1.0))
            detections.append(
                Detection(
                    frame_index=frame_index,
                    bbox=bbox,
                    confidence=confidence,
                    crop=crop,
                    mask=local_mask,
                )
            )

        detections.sort(key=lambda d: bbox_area(d.bbox), reverse=True)
        return detections
