# file: vision_runtime/adapters/debayer_adapter.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Protocol

import numpy as np

try:
    import cv2
except Exception as exc:
    cv2 = None
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None


DebayerBackendName = Literal["cpu_opencv", "future_cuda_npp"]


@dataclass(slots=True)
class DebayerConfig:
    backend: DebayerBackendName = "cpu_opencv"
    bayer_pattern: Literal["RG", "BG", "GR", "GB"] = "BG"
    output_format: Literal["BGR"] = "BGR"
    library_path: Optional[str] = None


class DebayerAdapter(Protocol):
    def debayer(self, frame_bayer: np.ndarray) -> np.ndarray:
        ...

    def close(self) -> None:
        ...


def _require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError(f"OpenCV import failed: {_CV2_IMPORT_ERROR}")


def _normalize_bayer_pattern(pattern: str) -> str:
    value = str(pattern or "").strip().upper()
    if value not in {"RG", "BG", "GR", "GB"}:
        raise ValueError(f"Unsupported Bayer pattern: {pattern}")
    return value


def _get_bayer_code(pattern: str) -> int:
    _require_cv2()
    mapping = {
        "RG": cv2.COLOR_BAYER_RG2BGR,
        "BG": cv2.COLOR_BAYER_BG2BGR,
        "GR": cv2.COLOR_BAYER_GR2BGR,
        "GB": cv2.COLOR_BAYER_GB2BGR,
    }
    return mapping[_normalize_bayer_pattern(pattern)]


def _ensure_uint8_contiguous(frame: np.ndarray) -> np.ndarray:
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8, copy=False)
    if not frame.flags["C_CONTIGUOUS"]:
        frame = np.ascontiguousarray(frame)
    return frame


class CpuOpenCVDebayerAdapter:
    def __init__(self, config: DebayerConfig) -> None:
        _require_cv2()
        self.config = config
        self._code = _get_bayer_code(config.bayer_pattern)

        if str(config.output_format).upper() != "BGR":
            raise ValueError(f"Unsupported output_format: {config.output_format}")

    def debayer(self, frame_bayer: np.ndarray) -> np.ndarray:
        if frame_bayer is None:
            raise ValueError("frame_bayer is None")

        if not isinstance(frame_bayer, np.ndarray):
            raise TypeError(f"frame_bayer must be np.ndarray, got {type(frame_bayer)}")

        # Obraz już gotowy 3-kanałowy -> tylko walidacja
        if frame_bayer.ndim == 3 and frame_bayer.shape[2] == 3:
            return _ensure_uint8_contiguous(frame_bayer)

        # Bayer / mono 2D -> debayer CPU OpenCV
        if frame_bayer.ndim == 2:
            src = _ensure_uint8_contiguous(frame_bayer)
            out = cv2.cvtColor(src, self._code)
            return _ensure_uint8_contiguous(out)

        # BGRA -> BGR
        if frame_bayer.ndim == 3 and frame_bayer.shape[2] == 4:
            src = _ensure_uint8_contiguous(frame_bayer)
            out = cv2.cvtColor(src, cv2.COLOR_BGRA2BGR)
            return _ensure_uint8_contiguous(out)

        raise ValueError(
            f"Unsupported frame shape={frame_bayer.shape}, dtype={frame_bayer.dtype}"
        )

    def close(self) -> None:
        return


class FutureCudaNppDebayerAdapter:
    """
    Placeholder boundary for future C++/CUDA/NPP backend.

    Expected contract:
    - input:  2D Bayer uint8 numpy array
    - output: 3-channel BGR uint8 numpy array
    """

    def __init__(self, config: DebayerConfig) -> None:
        self.config = config

    def debayer(self, frame_bayer: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "future_cuda_npp backend is not implemented yet. "
            "Attach C++/CUDA/NPP backend here."
        )

    def close(self) -> None:
        return


def build_debayer_adapter(config: DebayerConfig) -> DebayerAdapter:
    backend = str(config.backend).strip().lower()

    if backend == "cpu_opencv":
        return CpuOpenCVDebayerAdapter(config)

    if backend == "future_cuda_npp":
        return FutureCudaNppDebayerAdapter(config)

    raise ValueError(f"Unsupported debayer backend: {config.backend}")