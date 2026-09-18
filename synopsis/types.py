from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

BBox = Tuple[int, int, int, int]


@dataclass
class Detection:
    frame_index: int
    bbox: BBox
    confidence: float
    crop: np.ndarray
    mask: np.ndarray


@dataclass
class TrackObservation:
    frame_index: int
    bbox: BBox
    confidence: float
    crop: np.ndarray
    mask: np.ndarray


@dataclass
class Track:
    track_id: int
    observations: List[TrackObservation] = field(default_factory=list)

    @property
    def start_frame(self) -> int:
        return self.observations[0].frame_index if self.observations else 0

    @property
    def end_frame(self) -> int:
        return self.observations[-1].frame_index if self.observations else 0

    @property
    def duration(self) -> int:
        return len(self.observations)


@dataclass
class ScheduledTrack:
    track: Track
    start: int


@dataclass
class SynopsisResult:
    input_path: str
    output_path: str
    background_path: str
    original_frames: int
    sampled_frames: int
    tubes: int
    synopsis_frames: int
    compression_ratio: float
    runtime_seconds: float
    notes: List[str]
