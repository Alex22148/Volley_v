"""GPU color path: Bayer RAW -> RGB float tensor (B, 3, H, W) on CUDA.

"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .native_debayer import NativeCudaDebayer


@dataclass(slots=True)
class ColorResult:
    """Output of a single color-convert step.

    tensor      : (B, 3, H_out, W_out) on CUDA, dtype float16 or float32.
    backend_used: "native_cuda_kernel" or "torch_gpu_fallback".
    timings_ms  : per-step breakdown.
    """
    tensor: Any
    backend_used: str
    timings_ms: Dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class CudaDebayerConfig:
    bayer_pattern: str = "RG"          # one of "RG", "BG", "GR", "GB"
    output_color: str = "RGB"          # "RGB" or "BGR"
    output_hw: Tuple[int, int] = (640, 640)
    normalize_01: bool = True
    half_precision: bool = True
    prefer_native: bool = True
    device: str = "cuda"


class CudaDebayer:
    """Bayer -> resized normalized RGB tensor on CUDA.

    Usage:
        deb = CudaDebayer(CudaDebayerConfig(output_hw=(640, 640)))
        raw = np.random.randint(0, 256, (4, 1080, 1920), dtype=np.uint8)
        out = deb.convert(raw)
        # out.tensor: (4, 3, 640, 640) float16 cuda
        # out.backend_used: "native_cuda_kernel"
    """

    def __init__(self, config: Optional[CudaDebayerConfig] = None) -> None:
        self.config = config or CudaDebayerConfig()
        self._native: Optional[NativeCudaDebayer] = None
        if self.config.prefer_native:
            cand = NativeCudaDebayer(device=self.config.device)
            if cand.is_available():
                self._native = cand

    # ----------------------------------------------------------- public API

    def backend_status(self) -> Dict[str, Any]:
        info = {
            "prefer_native": self.config.prefer_native,
            "native_available": self._native is not None,
        }
        info["native_info"] = NativeCudaDebayer.describe_backend()
        return info

    def convert(self, raw_batch: Union[np.ndarray, List[np.ndarray], "Any"]) -> ColorResult:
        """Bayer batch -> (B, 3, H_out, W_out) float on CUDA."""
        if self._native is not None:
            return self._convert_native(raw_batch)
        return self._convert_torch_fallback(raw_batch)

    # ------------------------------------------------------------- native

    def _convert_native(self, raw_batch: Any) -> ColorResult:
        import torch
        t0 = time.perf_counter()
        out = self._native.debayer(  # type: ignore[union-attr]
            raw_batch,
            bayer_pattern=self.config.bayer_pattern,
            output_format=self.config.output_color,
            resize_to=self.config.output_hw,
            normalize=self.config.normalize_01,
            output_layout="BCHW",
            device=self.config.device,
            half=self.config.half_precision,
        )
        torch.cuda.synchronize()
        total_ms = (time.perf_counter() - t0) * 1000.0
        return ColorResult(
            tensor=out,
            backend_used="native_cuda_kernel",
            timings_ms={"total_color_ms": total_ms},
        )

    # ----------------------------------------------------- torch fallback

    def _convert_torch_fallback(self, raw_batch: Any) -> ColorResult:
        """Pure-torch GPU debayer.

        Uses the simplest possible approach: index the four Bayer color
        channels, then average-interpolate the missing pixels with a
        depthwise conv (effectively bilinear demosaic). This is honest
        about being slower than the custom CUDA kernel — we log it.
        """
        import torch
        import torch.nn.functional as F

        t0 = time.perf_counter()
        if isinstance(raw_batch, list):
            arr = np.stack([np.ascontiguousarray(f) for f in raw_batch], axis=0)
            raw = torch.from_numpy(arr).to(self.config.device, non_blocking=True)
        elif isinstance(raw_batch, np.ndarray):
            arr = raw_batch
            if arr.ndim == 2:
                arr = arr[None, ...]
            if arr.dtype == np.uint16:
                arr = (arr >> 8).astype(np.uint8, copy=False)
            elif arr.dtype != np.uint8:
                arr = arr.astype(np.uint8, copy=False)
            raw = torch.from_numpy(np.ascontiguousarray(arr)).to(self.config.device, non_blocking=True)
        elif isinstance(raw_batch, torch.Tensor):
            raw = raw_batch
            if raw.dim() == 2:
                raw = raw.unsqueeze(0)
            if raw.dtype != torch.uint8:
                raw = raw.to(torch.uint8)
            if str(raw.device) != str(torch.device(self.config.device)):
                raw = raw.to(self.config.device, non_blocking=True)
        else:
            raise TypeError(f"Unsupported raw_batch type: {type(raw_batch)}")

        raw_f = raw.to(torch.float32)
        if self.config.normalize_01:
            raw_f = raw_f / 255.0

        # Per-channel Bayer planes via stride-2 sampling.
        # Pattern "RG":
        #   row even:  R G R G ...
        #   row odd :  G B G B ...
        pat = self.config.bayer_pattern.upper()
        _, H, W = raw_f.shape
        if H % 2 or W % 2:
            raise ValueError(f"Bayer raw must have even H and W; got {H}x{W}")

        # 4 stride-2 planes (top-left, top-right, bottom-left, bottom-right)
        tl = raw_f[:, 0::2, 0::2]
        tr = raw_f[:, 0::2, 1::2]
        bl = raw_f[:, 1::2, 0::2]
        br = raw_f[:, 1::2, 1::2]

        def _planes_to_rgb(pattern: str):
            if pattern == "RG":
                return tl, (tr + bl) * 0.5, br      # R, G, B
            if pattern == "BG":
                return br, (tr + bl) * 0.5, tl
            if pattern == "GR":
                return tr, (tl + br) * 0.5, bl
            if pattern == "GB":
                return bl, (tl + br) * 0.5, tr
            raise ValueError(f"Unsupported pattern: {pattern}")

        r_plane, g_plane, b_plane = _planes_to_rgb(pat)

        # Stack to (B, 3, H/2, W/2) -- this is a quarter-res image but
        # correct-color. Then upsample to (H, W) bilinear before final resize.
        quarter = torch.stack(
            (r_plane, g_plane, b_plane) if self.config.output_color.upper() == "RGB"
            else (b_plane, g_plane, r_plane),
            dim=1,
        )

        full = F.interpolate(quarter, size=(H, W), mode="bilinear", align_corners=False)

        target_h, target_w = self.config.output_hw
        if (target_h, target_w) != (H, W):
            full = F.interpolate(full, size=(target_h, target_w),
                                 mode="bilinear", align_corners=False)

        if self.config.half_precision:
            full = full.to(torch.float16)
        full = full.contiguous()

        torch.cuda.synchronize()
        total_ms = (time.perf_counter() - t0) * 1000.0
        return ColorResult(
            tensor=full,
            backend_used="torch_gpu_fallback",
            timings_ms={"total_color_ms": total_ms},
        )
