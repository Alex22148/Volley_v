"""Frame sources for the runtime benchmark / simulator.

Public API:
    BaseFrameSource, RawFrame, ColorFrame, FrameBatch4Cam,
    InferenceResult, CAMERA_ROLES, SyntheticRaw4CamSource
"""
from .base_frame_source import (
    BaseFrameSource,
    CAMERA_ROLES,
    ColorFrame,
    FrameBatch4Cam,
    InferenceResult,
    RawFrame,
)
from .synthetic_raw_4cam_source import (
    SyntheticRaw4CamSource,
    SyntheticSourceConfig,
)

__all__ = [
    "BaseFrameSource",
    "CAMERA_ROLES",
    "ColorFrame",
    "FrameBatch4Cam",
    "InferenceResult",
    "RawFrame",
    "SyntheticRaw4CamSource",
    "SyntheticSourceConfig",
]
