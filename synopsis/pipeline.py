from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .detection import MotionDetector, MotionDetectorConfig, YoloDetector, YoloDetectorConfig
from .tracking import BoTSORTConfig, BoTSORTTracker
from .types import ScheduledTrack, SynopsisResult, Track
from .utils import alpha_blend, ensure_dir, resize_keep_aspect, bbox_iou


@dataclass
class SynopsisConfig:
    target_fps: float = 12.0
    max_width: int = 640
    min_area: int = 800
    use_yolo: bool = False
    yolo_model: str = "yolov8n.pt"
    yolo_conf_threshold: float = 0.3
    bot_max_lost: int = 18
    bot_min_hits: int = 2
    bot_high_confidence_threshold: float = 0.45
    bot_low_confidence_threshold: float = 0.15
    bot_iou_threshold: float = 0.15
    bot_appearance_threshold: float = 0.25
    bot_cost_threshold: float = 0.95
    min_tube_length: int = 4
    collision_iou_threshold: float = 0.08


class VideoSynopsisProcessor:
    def __init__(self, config: SynopsisConfig | None = None) -> None:
        self.config = config or SynopsisConfig()

    def process(self, input_path: str | Path, output_dir: str | Path) -> SynopsisResult:
        start_time = time.time()
        input_path = str(input_path)
        output_dir = ensure_dir(Path(output_dir))
        output_path = output_dir / "synopsis.mp4"
        background_path = output_dir / "background.jpg"

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError("Tidak dapat membuka file video input.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        original_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        source_fps = fps if fps > 0 else self.config.target_fps
        stride = max(1, int(round(source_fps / self.config.target_fps))) if source_fps > self.config.target_fps else 1
        process_fps = source_fps / stride if stride > 0 else self.config.target_fps

        # prefer YOLO detector when configured and available
        detector = None
        notes_detector = "motion"
        if self.config.use_yolo:
            try:
                ycfg = YoloDetectorConfig(model=self.config.yolo_model, conf_threshold=self.config.yolo_conf_threshold)
                detector = YoloDetector(ycfg)
                notes_detector = "yolo"
            except Exception:
                detector = MotionDetector(MotionDetectorConfig(min_area=self.config.min_area))
                notes_detector = "motion(fallback)"

        if detector is None:
            detector = MotionDetector(MotionDetectorConfig(min_area=self.config.min_area))
        tracker = BoTSORTTracker(
            BoTSORTConfig(
                max_lost=self.config.bot_max_lost,
                min_hits=self.config.bot_min_hits,
                high_confidence_threshold=self.config.bot_high_confidence_threshold,
                low_confidence_threshold=self.config.bot_low_confidence_threshold,
                iou_threshold=self.config.bot_iou_threshold,
                appearance_threshold=self.config.bot_appearance_threshold,
                cost_threshold=self.config.bot_cost_threshold,
            )
        )

        background_accum: Optional[np.ndarray] = None
        background_count = 0
        sampled_frames = 0
        sampled_frame_index = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if sampled_frame_index % stride != 0:
                sampled_frame_index += 1
                continue

            frame = resize_keep_aspect(frame, self.config.max_width)
            frame_float = frame.astype(np.float32)
            if background_accum is None:
                background_accum = frame_float
            else:
                alpha = 1.0 / float(background_count + 1)
                background_accum = (1.0 - alpha) * background_accum + alpha * frame_float
            background_count += 1

            detections = detector.detect(frame, sampled_frames)
            tracker.update(detections)
            sampled_frames += 1
            sampled_frame_index += 1

        cap.release()

        tracks = tracker.finalize()
        tracks = [track for track in tracks if track.duration >= self.config.min_tube_length]
        if not tracks:
            raise RuntimeError("Tidak ada tube yang cukup stabil terdeteksi. Coba video dengan objek bergerak yang lebih jelas.")

        scheduled_tracks = self._schedule_tracks(tracks)
        synopsis_frames = max((item.start + item.track.duration for item in scheduled_tracks), default=0)

        if background_accum is None:
            raise RuntimeError("Gagal membangun background.")

        background = np.clip(background_accum, 0, 255).astype(np.uint8)
        cv2.imwrite(str(background_path), background)

        height, width = background.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        writer = cv2.VideoWriter(str(output_path), fourcc, max(process_fps, 1.0), (width, height))
        if not writer.isOpened():
            raise RuntimeError("Tidak dapat membuat file output video.")

        for frame_idx in range(synopsis_frames):
            canvas = background.copy()
            for item in scheduled_tracks:
                local_idx = frame_idx - item.start
                if local_idx < 0 or local_idx >= item.track.duration:
                    continue
                obs = item.track.observations[local_idx]
                alpha_blend(canvas, obs.crop, obs.mask, obs.bbox)
            writer.write(canvas)

        writer.release()

        runtime_seconds = time.time() - start_time
        compression_ratio = (sampled_frames / synopsis_frames) if synopsis_frames > 0 else 0.0
        notes = [
            f"Pipeline ini memakai {notes_detector} detection + BoT-SORT-style tracking untuk membangun tube.",
            "Tube yang tidak saling collision signifikan dapat bertumpuk pada timeline synopsis.",
        ]

        return SynopsisResult(
            input_path=str(input_path),
            output_path=str(output_path),
            background_path=str(background_path),
            original_frames=original_frames,
            sampled_frames=sampled_frames,
            tubes=len(tracks),
            synopsis_frames=synopsis_frames,
            compression_ratio=compression_ratio,
            runtime_seconds=runtime_seconds,
            notes=notes,
        )

    def _schedule_tracks(self, tracks: List[Track]) -> List[ScheduledTrack]:
        ordered = sorted(tracks, key=lambda track: (track.duration, -track.start_frame), reverse=True)
        scheduled: List[ScheduledTrack] = []
        current_end = 0

        for track in ordered:
            placed = False
            for start in range(0, current_end + 1):
                if self._can_place(track, start, scheduled):
                    scheduled.append(ScheduledTrack(track=track, start=start))
                    current_end = max(current_end, start + track.duration)
                    placed = True
                    break
            if not placed:
                scheduled.append(ScheduledTrack(track=track, start=current_end))
                current_end += track.duration

        return sorted(scheduled, key=lambda item: item.start)

    def _can_place(self, track: Track, start: int, scheduled: List[ScheduledTrack]) -> bool:
        for item in scheduled:
            other = item.track
            other_start = item.start
            overlap_start = max(start, other_start)
            overlap_end = min(start + track.duration, other_start + other.duration)
            if overlap_start >= overlap_end:
                continue

            for global_frame in range(overlap_start, overlap_end):
                local_a = global_frame - start
                local_b = global_frame - other_start
                obs_a = track.observations[local_a]
                obs_b = other.observations[local_b]
                if bbox_iou(obs_a.bbox, obs_b.bbox) > self.config.collision_iou_threshold:
                    return False
        return True
