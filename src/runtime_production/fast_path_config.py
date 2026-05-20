"""Environment-driven configuration for the production fast path.

Activation:
    VOLLEYHUB_FAST_PATH=1                       (required)
    VOLLEYHUB_TRT_ENGINE=path/to/engine         (required when fast path on)

Optional overrides:
    VOLLEYHUB_COLOR_BACKEND=native_cuda_npp     (default; only allowed value for now)
    VOLLEYHUB_INFERENCE_BACKEND=tensorrt        (default)
    VOLLEYHUB_YOLO_IMGSZ=640                    (default — must match engine)
    VOLLEYHUB_YOLO_BATCH=4                      (default — must match engine)
    VOLLEYHUB_BAYER_PATTERN=RG                  (default)
    VOLLEYHUB_FAST_PATH_HALF=1                  (default)
    VOLLEYHUB_FAST_PATH_DEVICE=cuda             (default)
    VOLLEYHUB_FAST_PATH_CONF=0.25
    VOLLEYHUB_FAST_PATH_IOU=0.45
    VOLLEYHUB_FAST_PATH_BALL_CLASS_ID=0
    VOLLEYHUB_FAST_PATH_REQUIRED_ROLES=LEFT,CENTER_L,CENTER_R,RIGHT
    VOLLEYHUB_FAST_PATH_WARMUP=3
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

_DEFAULT_ROLES: Tuple[str, str, str, str] = ("LEFT", "CENTER_L", "CENTER_R", "RIGHT")


def _read_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "")
    if raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _read_int(name: str, default: int) -> int:
    try:
        raw = os.environ.get(name, "")
        if raw == "":
            return default
        return int(raw.strip())
    except Exception:
        return default


def _read_float(name: str, default: float) -> float:
    try:
        raw = os.environ.get(name, "")
        if raw == "":
            return default
        return float(raw.strip())
    except Exception:
        return default


@dataclass(slots=True)
class FastPathConfig:
    enabled: bool
    engine_path: str
    color_backend: str = "native_cuda_npp"
    inference_backend: str = "tensorrt"
    # int (square) or (H, W) tuple/list. Stride-32 alignment is checked in validate().
    imgsz: object = 640
    batch_size: int = 4
    bayer_pattern: str = "RG"
    half: bool = True
    device: str = "cuda"
    confidence: float = 0.25
    iou: float = 0.45
    ball_class_id: int = 0
    max_det: int = 300
    warmup_iterations: int = 3
    use_unified_image_processor: bool = False
    required_roles: Tuple[str, ...] = _DEFAULT_ROLES
    frame_read_retries: int = 1
    frame_read_delay_s: float = 0.0005
    extras: dict = field(default_factory=dict)

    def imgsz_hw(self) -> Tuple[int, int]:
        v = self.imgsz
        if isinstance(v, (tuple, list)):
            return (int(v[0]), int(v[1]))
        iv = int(v)  # type: ignore[arg-type]
        return (iv, iv)

    def imgsz_for_predict(self):
        v = self.imgsz
        if isinstance(v, (tuple, list)):
            return [int(v[0]), int(v[1])]
        return int(v)  # type: ignore[arg-type]

    def validate(self) -> Optional[str]:
        """Return None if valid, otherwise an error message.

        Validation is intentionally split into two phases:
        1. logical/runtime contract checks that do not touch the filesystem,
        2. engine file checks.

        This makes diagnostics useful even on a development machine that does
        not have the final TensorRT engine copied into the project yet.
        """
        if not self.enabled:
            return None

        if self.color_backend != "native_cuda_npp":
            return f"unsupported color_backend for fast path: {self.color_backend}"
        if self.inference_backend != "tensorrt":
            return f"unsupported inference_backend for fast path: {self.inference_backend}"

        v = self.imgsz
        try:
            if isinstance(v, (tuple, list)):
                if len(v) != 2:
                    return f"imgsz tuple must be (H, W); got {v}"
                ih, iw = int(v[0]), int(v[1])
                if ih <= 0 or iw <= 0 or ih % 32 != 0 or iw % 32 != 0:
                    return f"imgsz dims must be > 0 and divisible by 32, got {v}"
            else:
                iv = int(v)  # type: ignore[arg-type]
                if iv <= 0 or iv % 32 != 0:
                    return f"imgsz must be > 0 and divisible by 32, got {iv}"
        except Exception:
            return f"imgsz must be an int or (H, W), got {v!r}"

        if self.batch_size <= 0:
            return f"batch_size must be > 0, got {self.batch_size}"
        if self.batch_size != len(self.required_roles):
            return (
                f"batch_size ({self.batch_size}) must match number of required_roles "
                f"({len(self.required_roles)}: {self.required_roles}); "
                "static b4 engine cannot consume a different batch size."
            )
        if self.bayer_pattern not in ("RG", "BG", "GR", "GB"):
            return f"unknown bayer_pattern: {self.bayer_pattern}"

        if not self.engine_path:
            return "VOLLEYHUB_TRT_ENGINE is empty"
        p = Path(self.engine_path)
        if not p.exists():
            return f"engine file not found: {self.engine_path}"
        if p.stat().st_size < 1024:
            return f"engine file too small (<1024 B): {self.engine_path}"
        return None


def load_fast_path_config_from_env() -> FastPathConfig:
    """Build a FastPathConfig from environment variables. Always succeeds —
    `enabled` is False if VOLLEYHUB_FAST_PATH is not set; validation is a
    separate step."""
    roles_raw = os.environ.get("VOLLEYHUB_FAST_PATH_REQUIRED_ROLES", "").strip()
    if roles_raw:
        roles = tuple(r.strip().upper() for r in roles_raw.split(",") if r.strip())
    else:
        roles = _DEFAULT_ROLES

    return FastPathConfig(
        enabled=_read_bool("VOLLEYHUB_FAST_PATH", False),
        engine_path=os.environ.get("VOLLEYHUB_TRT_ENGINE", "").strip(),
        color_backend=os.environ.get("VOLLEYHUB_COLOR_BACKEND", "native_cuda_npp").strip(),
        inference_backend=os.environ.get("VOLLEYHUB_INFERENCE_BACKEND", "tensorrt").strip(),
        imgsz=_read_int("VOLLEYHUB_YOLO_IMGSZ", 640),
        batch_size=_read_int("VOLLEYHUB_YOLO_BATCH", 4),
        bayer_pattern=os.environ.get("VOLLEYHUB_BAYER_PATTERN", "RG").strip().upper() or "RG",
        half=_read_bool("VOLLEYHUB_FAST_PATH_HALF", True),
        device=os.environ.get("VOLLEYHUB_FAST_PATH_DEVICE", "cuda").strip() or "cuda",
        confidence=_read_float("VOLLEYHUB_FAST_PATH_CONF", 0.25),
        iou=_read_float("VOLLEYHUB_FAST_PATH_IOU", 0.45),
        ball_class_id=_read_int("VOLLEYHUB_FAST_PATH_BALL_CLASS_ID", 0),
        max_det=_read_int("VOLLEYHUB_FAST_PATH_MAX_DET", 300),
        warmup_iterations=_read_int("VOLLEYHUB_FAST_PATH_WARMUP", 3),
        use_unified_image_processor=_read_bool("VOLLEYHUB_USE_UNIFIED_IMAGE_PROCESSOR", False),
        required_roles=roles,
    )
