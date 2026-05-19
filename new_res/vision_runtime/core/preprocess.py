# file: vision_runtime/core/preprocess.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import cv2
import numpy as np


@dataclass(slots=True)
class PreprocessConfig:
    image_size: int = 640
    input_color: Literal["BGR", "RGB"] = "BGR"
    normalize_01: bool = True


def _validate_frame(frame_bgr: np.ndarray) -> np.ndarray:
    if frame_bgr is None:
        raise ValueError("frame_bgr is None")
    if not isinstance(frame_bgr, np.ndarray):
        raise TypeError(f"frame_bgr must be np.ndarray, got {type(frame_bgr)}")
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"Expected 3-channel image, got shape={frame_bgr.shape}")
    if frame_bgr.dtype != np.uint8:
        frame_bgr = frame_bgr.astype(np.uint8, copy=False)
    if not frame_bgr.flags["C_CONTIGUOUS"]:
        frame_bgr = np.ascontiguousarray(frame_bgr)
    return frame_bgr


def _letterbox_resize(image: np.ndarray, target_size: int) -> tuple[np.ndarray, dict]:
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image size: shape={image.shape}")

    scale = min(target_size / float(w), target_size / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    meta = {
        "orig_h": int(h),
        "orig_w": int(w),
        "resized_h": int(new_h),
        "resized_w": int(new_w),
        "pad_x": int(pad_x),
        "pad_y": int(pad_y),
        "scale": float(scale),
        "target_size": int(target_size),
    }
    return canvas, meta


def minimal_preprocess(frame_bgr: np.ndarray, config: PreprocessConfig) -> Tuple[np.ndarray, dict]:
    frame_bgr = _validate_frame(frame_bgr)

    if int(config.image_size) <= 0:
        raise ValueError(f"image_size must be > 0, got {config.image_size}")

    image, meta = _letterbox_resize(frame_bgr, int(config.image_size))

    if config.input_color == "RGB":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif config.input_color != "BGR":
        raise ValueError(f"Unsupported input_color: {config.input_color}")

    if config.normalize_01:
        image_out = image.astype(np.float32) / 255.0
    else:
        image_out = image

    if not image_out.flags["C_CONTIGUOUS"]:
        image_out = np.ascontiguousarray(image_out)

    meta["input_color"] = str(config.input_color)
    meta["normalized_01"] = bool(config.normalize_01)
    return image_out, meta


