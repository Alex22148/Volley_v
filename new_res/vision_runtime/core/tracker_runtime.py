# file: vision_runtime/core/tracker_runtime.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from vision_runtime.ipc.messages import DetectionPacket, TrackPacket


@dataclass(slots=True)
class TrackerConfig:
    max_age_frames: int = 8
    iou_threshold: float = 0.3


@dataclass(slots=True)
class _TrackState:
    track_id: int
    bbox_xyxy: List[float]
    score: float
    class_id: int
    label: str
    age_frames: int = 0


def _bbox_iou(box_a: List[float], box_b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0

    return inter_area / union


class SimpleTrackerRuntime:
    def __init__(self, config: Optional[TrackerConfig] = None) -> None:
        self.config = config or TrackerConfig()
        self._next_track_id = 1
        self._tracks: Dict[int, _TrackState] = {}

    def reset(self) -> None:
        self._next_track_id = 1
        self._tracks.clear()

    def update(self, detections: List[DetectionPacket]) -> List[TrackPacket]:
        if not detections:
            self._age_unmatched_tracks(set())
            self._drop_expired_tracks()
            return self._export_tracks()

        unmatched_track_ids = set(self._tracks.keys())
        matched_detection_indices: set[int] = set()

        candidate_matches: List[Tuple[float, int, int]] = []
        for det_idx, det in enumerate(detections):
            for track_id, track in self._tracks.items():
                if track.class_id != det.class_id:
                    continue
                iou = _bbox_iou(track.bbox_xyxy, det.bbox_xyxy)
                if iou >= self.config.iou_threshold:
                    candidate_matches.append((iou, track_id, det_idx))

        candidate_matches.sort(key=lambda item: item[0], reverse=True)

        used_track_ids: set[int] = set()
        for _, track_id, det_idx in candidate_matches:
            if track_id in used_track_ids or det_idx in matched_detection_indices:
                continue

            det = detections[det_idx]
            track = self._tracks[track_id]
            track.bbox_xyxy = list(det.bbox_xyxy)
            track.score = float(det.score)
            track.class_id = int(det.class_id)
            track.label = str(det.label)
            track.age_frames = 0

            used_track_ids.add(track_id)
            matched_detection_indices.add(det_idx)
            unmatched_track_ids.discard(track_id)

        for det_idx, det in enumerate(detections):
            if det_idx in matched_detection_indices:
                continue

            track_id = self._next_track_id
            self._next_track_id += 1
            self._tracks[track_id] = _TrackState(
                track_id=track_id,
                bbox_xyxy=list(det.bbox_xyxy),
                score=float(det.score),
                class_id=int(det.class_id),
                label=str(det.label),
                age_frames=0,
            )

        self._age_unmatched_tracks(unmatched_track_ids)
        self._drop_expired_tracks()
        return self._export_tracks()

    def _age_unmatched_tracks(self, unmatched_track_ids: set[int]) -> None:
        for track_id in unmatched_track_ids:
            track = self._tracks.get(track_id)
            if track is not None:
                track.age_frames += 1

    def _drop_expired_tracks(self) -> None:
        expired = [
            track_id
            for track_id, track in self._tracks.items()
            if track.age_frames > self.config.max_age_frames
        ]
        for track_id in expired:
            self._tracks.pop(track_id, None)

    def _export_tracks(self) -> List[TrackPacket]:
        out: List[TrackPacket] = []
        for track_id in sorted(self._tracks.keys()):
            track = self._tracks[track_id]
            out.append(
                TrackPacket(
                    track_id=int(track.track_id),
                    bbox_xyxy=list(track.bbox_xyxy),
                    score=float(track.score),
                    class_id=int(track.class_id),
                    label=str(track.label),
                )
            )
        return out