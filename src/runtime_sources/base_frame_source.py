"""Neutral data contracts for the runtime benchmark / simulator.

These dataclasses define the boundary between frame sources, color
converters, and inference engines. A future real-camera grabber adapter
can produce RawFrame / FrameBatch4Cam without changing the inference or
benchmark code. Nothing here imports from capture/* or live_runtime/* on
purpose, to keep the production pipeline untouched.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, Mapping, Optional, Tuple

import numpy as np


CAMERA_ROLES: Tuple[str, str, str, str] = ("LEFT", "CENTER_L", "CENTER_R", "RIGHT")
SUPPORTED_BAYER_PATTERNS: Tuple[str, ...] = ("RG", "BG", "GR", "GB")
SUPPORTED_RAW_DTYPES: Tuple[str, ...] = ("uint8", "uint16")


@dataclass(slots=True)
class RawFrame:
    """A single RAW/Bayer frame from one camera at one timestamp."""

    camera_id: str
    role: str
    frame_id: int
    timestamp_ns: int
    image: np.ndarray
    width: int
    height: int
    bayer_pattern: str
    dtype: str

    def validate(self) -> None:
        if self.role not in CAMERA_ROLES:
            raise ValueError(f"Unknown camera role: {self.role!r}")
        if self.bayer_pattern not in SUPPORTED_BAYER_PATTERNS:
            raise ValueError(f"Unsupported bayer pattern: {self.bayer_pattern!r}")
        if self.dtype not in SUPPORTED_RAW_DTYPES:
            raise ValueError(f"Unsupported raw dtype: {self.dtype!r}")
        if self.image.ndim != 2:
            raise ValueError(
                f"RAW frame must be 2D (HxW), got shape={self.image.shape}"
            )
        if self.image.shape != (self.height, self.width):
            raise ValueError(
                f"RAW frame shape {self.image.shape} does not match "
                f"({self.height}, {self.width})"
            )


@dataclass(slots=True)
class ColorFrame:
    """A single debayered/color-converted frame ready for inference."""

    camera_id: str
    role: str
    frame_id: int
    timestamp_ns: int
    image: np.ndarray
    color_format: str = "BGR"
    source_raw_shape: Tuple[int, int] = (0, 0)

    def validate(self) -> None:
        if self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError(
                f"ColorFrame must be HxWx3, got shape={self.image.shape}"
            )


@dataclass(slots=True)
class FrameBatch4Cam:
    """A synchronized packet of 4 RAW frames, one per camera role.

    Frames are kept in CAMERA_ROLES order (LEFT, CENTER_L, CENTER_R, RIGHT).
    """

    packet_id: int
    timestamp_ns: int
    frames: Tuple[RawFrame, RawFrame, RawFrame, RawFrame]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if len(self.frames) != 4:
            raise ValueError(f"FrameBatch4Cam must contain 4 frames, got {len(self.frames)}")
        seen_roles = tuple(f.role for f in self.frames)
        if seen_roles != CAMERA_ROLES:
            raise ValueError(
                f"FrameBatch4Cam roles {seen_roles} != expected {CAMERA_ROLES}"
            )
        for f in self.frames:
            f.validate()


@dataclass(slots=True)
class InferenceResult:
    """Per-frame inference output."""

    packet_id: int
    role: str
    frame_id: int
    timestamp_ns: int
    detections: list
    timing_ms: dict
    image_size: int = 0
    backend: str = ""

    @property
    def num_detections(self) -> int:
        return len(self.detections)


class BaseFrameSource(ABC):
    """Abstract source that yields synchronized 4-camera RAW packets.

    Implementations must guarantee:
      - exactly 4 RawFrame per packet,
      - frames in CAMERA_ROLES order,
      - shared timestamp_ns per packet,
      - monotonic packet_id starting at 0.
    """

    @abstractmethod
    def iter_packets(self) -> Iterator[FrameBatch4Cam]:
        ...

    def close(self) -> None:
        return None

    @property
    def description(self) -> str:
        return self.__class__.__name__

    def expected_total_packets(self) -> Optional[int]:
        return None
