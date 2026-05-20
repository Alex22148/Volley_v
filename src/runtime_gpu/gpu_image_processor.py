"""Shared image preprocessing contract for benchmark and production paths.

This module intentionally keeps imports light at module import time.  Torch and
OpenCV are imported only while processing frames so the contract can be imported
on machines without CUDA or optional GPU backends.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional

import numpy as np


InputColor = Literal["RGB", "BGR", "BAYER"]
OutputLayout = Literal["BCHW"]


@dataclass(slots=True)
class ImageProcessorConfig:
    target_size: int | tuple[int, int] = 640
    input_color: InputColor = "RGB"
    bayer_pattern: str = "RG"
    normalize: bool = True
    device: str = "auto"
    prefer_cuda: bool = True
    half_precision: bool = False
    output_layout: OutputLayout = "BCHW"

    def validate(self) -> None:
        if self.input_color not in ("RGB", "BGR", "BAYER"):
            raise ValueError(f"unknown input_color: {self.input_color}")
        if self.output_layout != "BCHW":
            raise ValueError(f"unsupported output_layout: {self.output_layout}")
        h, w = self.target_hw()
        if h <= 0 or w <= 0:
            raise ValueError(f"target_size must be > 0, got {self.target_size!r}")
        if self.bayer_pattern.upper() not in ("RG", "BG", "GR", "GB"):
            raise ValueError(f"unknown bayer_pattern: {self.bayer_pattern}")

    def target_hw(self) -> tuple[int, int]:
        value = self.target_size
        if isinstance(value, (tuple, list)):
            if len(value) != 2:
                raise ValueError(f"target_size must be int or (H, W), got {value!r}")
            return int(value[0]), int(value[1])
        iv = int(value)
        return iv, iv


@dataclass(slots=True)
class ImageProcessingResult:
    tensor: object
    stage_ms: dict[str, float] = field(default_factory=dict)
    device: str = "cpu"
    dtype: str = ""
    layout: OutputLayout = "BCHW"
    color_format: str = "RGB"
    normalized: bool = True
    source_shapes: list[tuple[int, ...]] = field(default_factory=list)


class GpuImageProcessor:
    """Convert RGB/BGR/Bayer numpy frames into a BCHW torch tensor.

    The implementation is CPU-testable and CUDA-ready.  When ``device="auto"``
    it selects CUDA only if torch imports successfully and CUDA is available.
    """

    def __init__(self, config: ImageProcessorConfig) -> None:
        config.validate()
        self.config = config

    def resolve_device(self):
        torch = _import_torch()
        requested = str(self.config.device or "auto").lower()
        if requested == "auto":
            if self.config.prefer_cuda and torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        if requested.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(requested)

    def process_frame(self, frame: np.ndarray) -> ImageProcessingResult:
        return self.process_batch([frame])

    def process_batch(self, frames: Iterable[np.ndarray]) -> ImageProcessingResult:
        torch = _import_torch()
        stage_ms: dict[str, float] = {}
        source_shapes: list[tuple[int, ...]] = []
        target_h, target_w = self.config.target_hw()
        device = self.resolve_device()
        out_dtype = torch.float16 if self.config.half_precision else torch.float32
        planes = []
        t_total = time.perf_counter()

        for frame in frames:
            arr = np.asarray(frame)
            source_shapes.append(tuple(arr.shape))

            t_color = time.perf_counter()
            rgb = self._to_rgb_uint8(arr)
            stage_ms["color_convert_ms"] = stage_ms.get("color_convert_ms", 0.0) + (
                time.perf_counter() - t_color
            ) * 1000.0

            t_tensor = time.perf_counter()
            if not rgb.flags.c_contiguous:
                rgb = np.ascontiguousarray(rgb)
            tensor = torch.from_numpy(rgb).to(device=device, non_blocking=True)
            tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(out_dtype)
            stage_ms["to_tensor_ms"] = stage_ms.get("to_tensor_ms", 0.0) + (
                time.perf_counter() - t_tensor
            ) * 1000.0

            t_norm = time.perf_counter()
            if self.config.normalize:
                tensor = tensor * (1.0 / 255.0)
            stage_ms["normalize_ms"] = stage_ms.get("normalize_ms", 0.0) + (
                time.perf_counter() - t_norm
            ) * 1000.0

            t_resize = time.perf_counter()
            if tuple(tensor.shape[-2:]) != (target_h, target_w):
                import torch.nn.functional as F

                tensor = F.interpolate(
                    tensor,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )
            stage_ms["resize_ms"] = stage_ms.get("resize_ms", 0.0) + (
                time.perf_counter() - t_resize
            ) * 1000.0
            planes.append(tensor.contiguous())

        if not planes:
            raise ValueError("process_batch requires at least one frame")

        t_stack = time.perf_counter()
        batch = torch.cat(planes, dim=0).contiguous()
        stage_ms["stack_ms"] = (time.perf_counter() - t_stack) * 1000.0
        _sync_if_cuda(torch, device)
        stage_ms["total_ms"] = (time.perf_counter() - t_total) * 1000.0

        return ImageProcessingResult(
            tensor=batch,
            stage_ms=stage_ms,
            device=str(device),
            dtype=str(batch.dtype).replace("torch.", ""),
            layout="BCHW",
            color_format="RGB",
            normalized=bool(self.config.normalize),
            source_shapes=source_shapes,
        )

    def _to_rgb_uint8(self, frame: np.ndarray) -> np.ndarray:
        if self.config.input_color == "RGB":
            rgb = _ensure_hwc3(frame)
        elif self.config.input_color == "BGR":
            rgb = _ensure_hwc3(frame)[..., ::-1]
        else:
            rgb = _debayer_to_rgb(frame, self.config.bayer_pattern)

        if rgb.dtype == np.uint8:
            return rgb
        if np.issubdtype(rgb.dtype, np.integer):
            max_value = max(1, int(np.iinfo(rgb.dtype).max))
            return np.clip((rgb.astype(np.float32) / max_value) * 255.0, 0, 255).astype(
                np.uint8
            )
        return np.clip(rgb, 0, 255).astype(np.uint8)


def _import_torch():
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError("torch is required to process frames into BCHW tensors") from exc
    return torch


def _sync_if_cuda(torch, device) -> None:
    try:
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
    except Exception:
        pass


def _ensure_hwc3(frame: np.ndarray) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected HWC frame with 3 channels, got {frame.shape}")
    return frame


def _debayer_to_rgb(frame: np.ndarray, pattern: str) -> np.ndarray:
    if frame.ndim != 2:
        raise ValueError(f"expected 2D Bayer frame, got {frame.shape}")
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for Bayer conversion") from exc

    pattern = pattern.upper()
    codes = {
        "RG": cv2.COLOR_BAYER_RG2RGB,
        "BG": cv2.COLOR_BAYER_BG2RGB,
        "GR": cv2.COLOR_BAYER_GR2RGB,
        "GB": cv2.COLOR_BAYER_GB2RGB,
    }
    return cv2.cvtColor(frame, codes[pattern])
