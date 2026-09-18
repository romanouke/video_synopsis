from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .types import Detection, Track, TrackObservation
from .utils import bbox_centroid, bbox_iou


@dataclass
class BoTSORTConfig:
    max_lost: int = 18
    min_hits: int = 2
    high_confidence_threshold: float = 0.45
    low_confidence_threshold: float = 0.15
    iou_threshold: float = 0.15
    appearance_threshold: float = 0.25
    cost_threshold: float = 0.95
    iou_weight: float = 0.55
    appearance_weight: float = 0.30
    motion_weight: float = 0.15
    histogram_bins: Tuple[int, int, int] = (8, 8, 8)


class _KalmanBoxFilter:
    def __init__(self, bbox: tuple[int, int, int, int]) -> None:
        self.x = np.zeros((8, 1), dtype=np.float32)
        self.x[:4, 0] = np.array(bbox, dtype=np.float32)

        self.p = np.eye(8, dtype=np.float32) * 10.0
        self.f = np.eye(8, dtype=np.float32)
        for idx in range(4):
            self.f[idx, idx + 4] = 1.0

        self.h = np.zeros((4, 8), dtype=np.float32)
        self.h[:4, :4] = np.eye(4, dtype=np.float32)
        self.q = np.eye(8, dtype=np.float32) * 0.01
        self.r = np.eye(4, dtype=np.float32) * 5.0

    def predict(self) -> tuple[int, int, int, int]:
        self.x = self.f @ self.x
        self.p = self.f @ self.p @ self.f.T + self.q
        return self.bbox()

    def update(self, bbox: tuple[int, int, int, int]) -> None:
        z = np.array(bbox, dtype=np.float32).reshape(4, 1)
        y = z - (self.h @ self.x)
        s = self.h @ self.p @ self.h.T + self.r
        k = self.p @ self.h.T @ np.linalg.inv(s)
        self.x = self.x + (k @ y)
        identity = np.eye(8, dtype=np.float32)
        self.p = (identity - (k @ self.h)) @ self.p

    def bbox(self) -> tuple[int, int, int, int]:
        x, y, w, h = self.x[:4, 0]
        return (
            int(round(float(x))),
            int(round(float(y))),
            max(1, int(round(float(w)))),
            max(1, int(round(float(h)))),
        )


@dataclass
class _ActiveTrack:
    track: Track
    filter: _KalmanBoxFilter
    appearance: np.ndarray
    hits: int = 1
    lost: int = 0
    confirmed: bool = False


class BoTSORTTracker:
    def __init__(self, config: BoTSORTConfig | None = None) -> None:
        self.config = config or BoTSORTConfig()
        self._next_id = 1
        self._active: Dict[int, _ActiveTrack] = {}
        self._finished: List[Track] = []

    def update(self, detections: List[Detection]) -> None:
        features = [self._extract_feature(detection.crop) for detection in detections]

        for active_track in self._active.values():
            active_track.filter.predict()

        if not detections:
            self._age_unmatched_tracks(set(self._active.keys()))
            return

        high_indices = [
            idx for idx, detection in enumerate(detections) if detection.confidence >= self.config.high_confidence_threshold
        ]
        low_indices = [
            idx
            for idx, detection in enumerate(detections)
            if self.config.low_confidence_threshold <= detection.confidence < self.config.high_confidence_threshold
        ]

        track_ids = list(self._active.keys())
        matches, unmatched_tracks, unmatched_high = self._associate(track_ids, detections, features, high_indices)
        self._apply_matches(matches, detections, features)

        if unmatched_tracks and low_indices:
            low_matches, unmatched_tracks, _ = self._associate(list(unmatched_tracks), detections, features, low_indices)
            self._apply_matches(low_matches, detections, features)

        self._age_unmatched_tracks(unmatched_tracks)

        for det_idx in unmatched_high:
            detection = detections[det_idx]
            if detection.confidence >= self.config.high_confidence_threshold:
                self._start_new_track(detection, features[det_idx])

    def finalize(self) -> List[Track]:
        for active_track in list(self._active.values()):
            if active_track.track.observations:
                self._finished.append(active_track.track)
        self._active.clear()
        return sorted(self._finished, key=lambda track: (track.start_frame, -track.duration))

    def _associate(
        self,
        track_ids: List[int],
        detections: List[Detection],
        features: List[np.ndarray],
        det_indices: List[int],
    ) -> tuple[List[tuple[int, int]], set[int], set[int]]:
        if not track_ids or not det_indices:
            return [], set(track_ids), set(det_indices)

        candidates: List[tuple[float, int, int]] = []
        for track_id in track_ids:
            active_track = self._active.get(track_id)
            if active_track is None:
                continue
            pred_bbox = active_track.filter.bbox()
            track_center = bbox_centroid(pred_bbox)
            track_diag = self._bbox_diagonal(pred_bbox)

            for det_idx in det_indices:
                detection = detections[det_idx]
                det_bbox = detection.bbox
                iou = bbox_iou(pred_bbox, det_bbox)
                det_center = bbox_centroid(det_bbox)
                motion = float(np.hypot(track_center[0] - det_center[0], track_center[1] - det_center[1])) / max(
                    track_diag, self._bbox_diagonal(det_bbox), 1.0
                )
                appearance = self._cosine_similarity(active_track.appearance, features[det_idx])

                if iou < self.config.iou_threshold and appearance < self.config.appearance_threshold:
                    continue

                cost = (
                    self.config.iou_weight * (1.0 - iou)
                    + self.config.appearance_weight * (1.0 - appearance)
                    + self.config.motion_weight * min(1.0, motion)
                )
                candidates.append((cost, track_id, det_idx))

        candidates.sort(key=lambda item: item[0])
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        matches: List[tuple[int, int]] = []

        for cost, track_id, det_idx in candidates:
            if cost > self.config.cost_threshold:
                continue
            if track_id in matched_tracks or det_idx in matched_detections:
                continue
            matched_tracks.add(track_id)
            matched_detections.add(det_idx)
            matches.append((track_id, det_idx))

        unmatched_tracks = set(track_ids) - matched_tracks
        unmatched_detections = set(det_indices) - matched_detections
        return matches, unmatched_tracks, unmatched_detections

    def _apply_matches(self, matches: List[tuple[int, int]], detections: List[Detection], features: List[np.ndarray]) -> None:
        for track_id, det_idx in matches:
            active_track = self._active.get(track_id)
            if active_track is None:
                continue

            detection = detections[det_idx]
            active_track.filter.update(detection.bbox)
            active_track.track.observations.append(
                TrackObservation(
                    frame_index=detection.frame_index,
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    crop=detection.crop,
                    mask=detection.mask,
                )
            )
            if np.any(features[det_idx]):
                active_track.appearance = self._normalize(0.85 * active_track.appearance + 0.15 * features[det_idx])
            active_track.hits += 1
            active_track.lost = 0
            active_track.confirmed = active_track.confirmed or active_track.hits >= self.config.min_hits

    def _age_unmatched_tracks(self, unmatched_track_ids: set[int]) -> None:
        for track_id in list(unmatched_track_ids):
            active_track = self._active.get(track_id)
            if active_track is None:
                continue
            active_track.lost += 1
            if active_track.lost > self.config.max_lost:
                if active_track.track.observations:
                    self._finished.append(active_track.track)
                del self._active[track_id]

    def _start_new_track(self, detection: Detection, appearance: np.ndarray) -> None:
        track = Track(track_id=self._next_id)
        track.observations.append(
            TrackObservation(
                frame_index=detection.frame_index,
                bbox=detection.bbox,
                confidence=detection.confidence,
                crop=detection.crop,
                mask=detection.mask,
            )
        )
        self._active[track.track_id] = _ActiveTrack(
            track=track,
            filter=_KalmanBoxFilter(detection.bbox),
            appearance=appearance,
            hits=1,
            lost=0,
            confirmed=False,
        )
        self._next_id += 1

    def _extract_feature(self, crop: np.ndarray) -> np.ndarray:
        if crop.size == 0:
            return np.zeros(8 * 8 * 8, dtype=np.float32)

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, list(self.config.histogram_bins), [0, 180, 0, 256, 0, 256])
        hist = hist.flatten().astype(np.float32)
        return self._normalize(hist)

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-8:
            return np.zeros_like(vector, dtype=np.float32)
        return (vector / norm).astype(np.float32)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        if a.size == 0 or b.size == 0:
            return 0.0
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a <= 1e-8 or norm_b <= 1e-8:
            return 0.0
        return float(np.clip(np.dot(a, b) / (norm_a * norm_b), 0.0, 1.0))

    def _bbox_diagonal(self, bbox: tuple[int, int, int, int]) -> float:
        _, _, w, h = bbox
        return float(np.hypot(max(1, w), max(1, h)))


TrackerConfig = BoTSORTConfig
SimpleTubeTracker = BoTSORTTracker
