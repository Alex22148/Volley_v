from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class FramePacket:
    frame_id: int
    source_path: str
    timestamp_host_ns: int
    timestamp_source_ns: Optional[int]
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DetectionPacket:
    bbox_xyxy: List[float]
    score: float
    class_id: int
    label: str = ""


@dataclass(slots=True)
class TrackPacket:
    track_id: int
    bbox_xyxy: List[float]
    score: float
    class_id: int
    label: str = ""


@dataclass(slots=True)
class RuntimeStatsSnapshot:
    frames_seen: int = 0
    frames_processed: int = 0
    avg_load_ms: float = 0.0
    avg_convert_ms: float = 0.0
    avg_preprocess_ms: float = 0.0
    avg_infer_ms: float = 0.0
    avg_track_ms: float = 0.0
    avg_total_ms: float = 0.0
    counters: Dict[str, int] = field(default_factory=dict)
    gauges: Dict[str, float] = field(default_factory=dict)
