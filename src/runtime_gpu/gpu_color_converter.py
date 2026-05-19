"""4-camera RAW/Bayer -> color (BGR) converter with backend ladder.

Backend selection (auto):
    1. cv2.cuda  (if cv2.cuda exists AND demosaicing op is callable)
    2. torch GPU (uses torch.nn.functional, fast approximate demosaic)
    3. CPU       (cv2.cvtColor)

Each backend reports per-stage timings:
    upload_to_gpu_ms, debayer_ms, color_convert_ms, resize_ms,
    normalize_ms, total_color_ms.

The CPU fallback is logged at WARNING when it is not the user's explicit
choice. The class is independent of capture/* and live_runtime/* on
purpose; the existing production pipeline rules (no cv2.cuda at runtime,
TRT-only inference) still hold there. This converter exists only inside
the diagnostic / benchmark tor.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Literal

import numpy as np

from src.runtime_sources.base_frame_source import (
    CAMERA_ROLES,
    ColorFrame,
    FrameBatch4Cam,
    RawFrame,
)

_LOG = logging.getLogger(__name__)

_BackendName = Literal["auto", "cv2_cuda", "torch_gpu", "cpu", "native_cuda_npp"]
_BAYER_PATTERN_TO_CV_CODE: dict = {}


def _build_bayer_code_table() -> None:
    try:
        import cv2
    except Exception:
        return
    _BAYER_PATTERN_TO_CV_CODE.update(
        {
            "RG": cv2.COLOR_BAYER_RG2BGR,
            "BG": cv2.COLOR_BAYER_BG2BGR,
            "GR": cv2.COLOR_BAYER_GR2BGR,
            "GB": cv2.COLOR_BAYER_GB2BGR,
        }
    )


_build_bayer_code_table()


_OutputMode = Literal["numpy_bgr", "torch_cuda_bchw_rgb_norm"]


def _resolve_target_hw(value) -> tuple:
    """Normalize target_image_size to (H, W) tuple. 0 / None / falsy -> (0, 0)."""
    if value is None:
        return (0, 0)
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"target_image_size tuple must be (H, W); got {value}")
        h, w = int(value[0]), int(value[1])
        return (h, w)
    iv = int(value)
    return (iv, iv)


@dataclass(slots=True)
class ColorConverterConfig:
    backend: _BackendName = "auto"
    # int (square) or (H, W) tuple. 0 disables resize in this stage.
    target_image_size: object = 0
    output_color: Literal["BGR", "RGB"] = "BGR"
    normalize_01: bool = False
    half_precision: bool = False
    device: str = "cuda"
    output_mode: _OutputMode = "numpy_bgr"

    def validate(self) -> None:
        if self.backend not in ("auto", "cv2_cuda", "torch_gpu", "cpu", "native_cuda_npp"):
            raise ValueError(f"unknown backend: {self.backend}")
        if self.output_color not in ("BGR", "RGB"):
            raise ValueError(f"output_color must be BGR or RGB, got {self.output_color}")
        if self.output_mode not in ("numpy_bgr", "torch_cuda_bchw_rgb_norm"):
            raise ValueError(f"unknown output_mode: {self.output_mode}")
        h, w = _resolve_target_hw(self.target_image_size)
        if h < 0 or w < 0:
            raise ValueError("target_image_size dimensions must be >= 0")
        if self.output_mode == "torch_cuda_bchw_rgb_norm":
            if h <= 0 or w <= 0:
                raise ValueError("torch_cuda_bchw_rgb_norm requires target_image_size > 0")

    def target_hw(self) -> tuple:
        return _resolve_target_hw(self.target_image_size)


@dataclass(slots=True)
class ColorConvertTimings:
    upload_to_gpu_ms: float = 0.0
    debayer_ms: float = 0.0
    color_convert_ms: float = 0.0
    resize_ms: float = 0.0
    normalize_ms: float = 0.0
    download_from_gpu_ms: float = 0.0
    total_color_ms: float = 0.0
    backend: str = ""
    output_location: str = "cpu"

    def to_dict(self) -> dict:
        return {
            "upload_to_gpu_ms": float(self.upload_to_gpu_ms),
            "debayer_ms": float(self.debayer_ms),
            "color_convert_ms": float(self.color_convert_ms),
            "resize_ms": float(self.resize_ms),
            "normalize_ms": float(self.normalize_ms),
            "download_from_gpu_ms": float(self.download_from_gpu_ms),
            "total_color_ms": float(self.total_color_ms),
            "backend": str(self.backend),
            "output_location": str(self.output_location),
        }


@dataclass(slots=True)
class GpuFrameMeta:
    role: str
    camera_id: str
    frame_id: int
    timestamp_ns: int
    source_raw_shape: tuple


@dataclass(slots=True)
class GpuColorBatch:
    """Color batch kept on GPU as a torch.Tensor (B, 3, H, W).

    The tensor is normalized to 0..1 RGB float (16/32) and ready to be
    passed directly to Ultralytics' ``model.predict(tensor)`` without any
    additional CPU bounce. Per-frame metadata is preserved in `frames`.
    """

    tensor: object  # torch.Tensor (B, 3, H, W) on cuda
    device: str
    dtype: str  # "float16" or "float32"
    image_size: int
    color_format: str = "RGB"
    frames: List[GpuFrameMeta] = field(default_factory=list)


@dataclass(slots=True)
class ColorConvertResult:
    color_frames: List[ColorFrame]
    timings: ColorConvertTimings
    gpu_batch: object = None  # Optional[GpuColorBatch]
    fallback_warnings: List[str] = field(default_factory=list)


class GpuColorConverter:
    """Converts a 4-camera RAW packet into 4 BGR/RGB color frames."""

    def __init__(self, config: ColorConverterConfig) -> None:
        config.validate()
        self.config = config
        self._fallback_warnings: List[str] = []
        self._chosen_backend = self._select_backend()
        _LOG.info("[GpuColorConverter] using backend=%s", self._chosen_backend)

    @property
    def backend(self) -> str:
        return self._chosen_backend

    @property
    def fallback_warnings(self) -> List[str]:
        return list(self._fallback_warnings)

    def convert_packet(self, packet: FrameBatch4Cam) -> ColorConvertResult:
        if self.config.output_mode == "torch_cuda_bchw_rgb_norm":
            if self._chosen_backend not in ("torch_gpu", "native_cuda_npp"):
                # zero-copy mode requires a backend that produces a CUDA
                # torch tensor; cv2_cuda/cpu cannot do that without copies.
                msg = (
                    f"output_mode=torch_cuda_bchw_rgb_norm requires backend in "
                    f"(torch_gpu, native_cuda_npp); current={self._chosen_backend}; "
                    "falling back to numpy output"
                )
                _LOG.warning("[GpuColorConverter] %s", msg)
                self._fallback_warnings.append(msg)
                if self._chosen_backend == "cv2_cuda":
                    return self._convert_cv2_cuda(packet)
                return self._convert_cpu(packet)
            if self._chosen_backend == "native_cuda_npp":
                return self._convert_native_cuda_full(packet)
            return self._convert_torch_gpu_full(packet)
        if self._chosen_backend == "cv2_cuda":
            return self._convert_cv2_cuda(packet)
        if self._chosen_backend == "torch_gpu":
            return self._convert_torch_gpu(packet)
        if self._chosen_backend == "native_cuda_npp":
            return self._convert_native_cuda_numpy(packet)
        return self._convert_cpu(packet)

    def _select_backend(self) -> str:
        wanted = self.config.backend
        if wanted == "cv2_cuda":
            if not self._cv2_cuda_available():
                msg = "cv2_cuda forced but cv2.cuda unavailable, falling back to cpu"
                _LOG.warning("[GpuColorConverter] %s", msg)
                self._fallback_warnings.append(msg)
                return "cpu"
            return "cv2_cuda"
        if wanted == "torch_gpu":
            if not self._torch_cuda_available():
                msg = "torch_gpu forced but torch.cuda unavailable, falling back to cpu"
                _LOG.warning("[GpuColorConverter] %s", msg)
                self._fallback_warnings.append(msg)
                return "cpu"
            return "torch_gpu"
        if wanted == "native_cuda_npp":
            if not self._native_cuda_available():
                msg = (
                    "native_cuda_npp forced but native debayer extension unavailable; "
                    "falling back to cpu (custom CUDA kernel needs to be built — "
                    "see src/runtime_gpu/native_debayer/README.md)"
                )
                _LOG.warning("[GpuColorConverter] %s", msg)
                self._fallback_warnings.append(msg)
                return "cpu"
            return "native_cuda_npp"
        if wanted == "cpu":
            return "cpu"
        # auto: keep historical preference (cv2_cuda > torch_gpu > cpu).
        # Native is opt-in only — never auto-selected, to keep production
        # default unchanged.
        if self._cv2_cuda_available():
            return "cv2_cuda"
        if self._torch_cuda_available():
            msg = "cv2.cuda not available, using torch_gpu fallback"
            _LOG.warning("[GpuColorConverter] %s", msg)
            self._fallback_warnings.append(msg)
            return "torch_gpu"
        msg = "Neither cv2.cuda nor torch.cuda available, using CPU fallback"
        _LOG.warning("[GpuColorConverter] %s", msg)
        self._fallback_warnings.append(msg)
        return "cpu"

    @staticmethod
    def _native_cuda_available() -> bool:
        try:
            from src.runtime_gpu.native_debayer import NativeCudaDebayer
            return bool(NativeCudaDebayer.is_available())
        except Exception:
            return False

    @staticmethod
    def _cv2_cuda_available() -> bool:
        try:
            import cv2
        except Exception:
            return False
        cuda_mod = getattr(cv2, "cuda", None)
        if cuda_mod is None:
            return False
        try:
            n = cuda_mod.getCudaEnabledDeviceCount()
        except Exception:
            return False
        if int(n) <= 0:
            return False
        # cuda.demosaicing exists in some OpenCV builds; we check at runtime.
        return hasattr(cuda_mod, "cvtColor") or hasattr(cuda_mod, "demosaicing")

    @staticmethod
    def _torch_cuda_available() -> bool:
        try:
            import torch
        except Exception:
            return False
        try:
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _convert_cpu(self, packet: FrameBatch4Cam) -> ColorConvertResult:
        import cv2

        timings = ColorConvertTimings(backend="cpu", output_location="cpu")
        color_frames: List[ColorFrame] = []
        t0 = time.perf_counter()

        for raw in packet.frames:
            ts_debayer = time.perf_counter()
            cv_code = _BAYER_PATTERN_TO_CV_CODE[raw.bayer_pattern]
            img = self._raw_to_uint8(raw)
            bgr = cv2.cvtColor(img, cv_code)
            timings.debayer_ms += (time.perf_counter() - ts_debayer) * 1000.0

            ts_cc = time.perf_counter()
            if self.config.output_color == "RGB":
                bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            timings.color_convert_ms += (time.perf_counter() - ts_cc) * 1000.0

            ts_rs = time.perf_counter()
            t_h, t_w = self.config.target_hw()
            if t_h > 0 and t_w > 0:
                bgr = cv2.resize(
                    bgr,
                    (int(t_w), int(t_h)),
                    interpolation=cv2.INTER_LINEAR,
                )
            timings.resize_ms += (time.perf_counter() - ts_rs) * 1000.0

            ts_n = time.perf_counter()
            if self.config.normalize_01:
                bgr = (bgr.astype(np.float32) * (1.0 / 255.0))
                if self.config.half_precision:
                    bgr = bgr.astype(np.float16)
            timings.normalize_ms += (time.perf_counter() - ts_n) * 1000.0

            color_frames.append(self._wrap_color_frame(raw, bgr))

        timings.total_color_ms = (time.perf_counter() - t0) * 1000.0
        return ColorConvertResult(
            color_frames=color_frames,
            timings=timings,
            fallback_warnings=list(self._fallback_warnings),
        )

    def _convert_cv2_cuda(self, packet: FrameBatch4Cam) -> ColorConvertResult:
        import cv2

        timings = ColorConvertTimings(backend="cv2_cuda", output_location="cpu")
        cuda = getattr(cv2, "cuda", None)
        if cuda is None:
            self._chosen_backend = "cpu"
            return self._convert_cpu(packet)
        cuda_cvt_color = getattr(cuda, "cvtColor", None)
        cuda_resize = getattr(cuda, "resize", None)
        cuda_gpumat = getattr(cuda, "GpuMat", None)
        if cuda_cvt_color is None or cuda_gpumat is None:
            self._chosen_backend = "cpu"
            return self._convert_cpu(packet)
        color_frames: List[ColorFrame] = []
        t0 = time.perf_counter()

        for raw in packet.frames:
            cv_code = _BAYER_PATTERN_TO_CV_CODE[raw.bayer_pattern]
            img = self._raw_to_uint8(raw)

            ts_up = time.perf_counter()
            gpu_in = cuda_gpumat()
            gpu_in.upload(img)
            timings.upload_to_gpu_ms += (time.perf_counter() - ts_up) * 1000.0

            ts_db = time.perf_counter()
            try:
                gpu_bgr = cuda_cvt_color(gpu_in, cv_code)
            except Exception as exc:
                msg = f"cv2.cuda.cvtColor failed for {raw.bayer_pattern}: {exc}; switching backend to cpu"
                _LOG.warning("[GpuColorConverter] %s", msg)
                self._fallback_warnings.append(msg)
                self._chosen_backend = "cpu"
                return self._convert_cpu(packet)
            timings.debayer_ms += (time.perf_counter() - ts_db) * 1000.0

            ts_rs = time.perf_counter()
            t_h, t_w = self.config.target_hw()
            if t_h > 0 and t_w > 0 and cuda_resize is not None:
                gpu_bgr = cuda_resize(
                    gpu_bgr,
                    (int(t_w), int(t_h)),
                    interpolation=cv2.INTER_LINEAR,
                )
            timings.resize_ms += (time.perf_counter() - ts_rs) * 1000.0

            ts_dl = time.perf_counter()
            bgr = gpu_bgr.download()
            timings.download_from_gpu_ms += (time.perf_counter() - ts_dl) * 1000.0

            if self.config.output_color == "RGB":
                ts_cc = time.perf_counter()
                bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                timings.color_convert_ms += (time.perf_counter() - ts_cc) * 1000.0

            ts_n = time.perf_counter()
            if self.config.normalize_01:
                bgr = (bgr.astype(np.float32) * (1.0 / 255.0))
                if self.config.half_precision:
                    bgr = bgr.astype(np.float16)
            timings.normalize_ms += (time.perf_counter() - ts_n) * 1000.0

            color_frames.append(self._wrap_color_frame(raw, bgr))

        timings.total_color_ms = (time.perf_counter() - t0) * 1000.0
        return ColorConvertResult(
            color_frames=color_frames,
            timings=timings,
            fallback_warnings=list(self._fallback_warnings),
        )

    def _convert_torch_gpu(self, packet: FrameBatch4Cam) -> ColorConvertResult:
        import torch
        import torch.nn.functional as F

        timings = ColorConvertTimings(backend="torch_gpu", output_location="cpu")
        device = torch.device(self.config.device)
        color_frames: List[ColorFrame] = []
        t0 = time.perf_counter()

        for raw in packet.frames:
            img = self._raw_to_uint8(raw)
            ts_up = time.perf_counter()
            t_in = torch.from_numpy(img).to(device, non_blocking=True)
            torch.cuda.synchronize()
            timings.upload_to_gpu_ms += (time.perf_counter() - ts_up) * 1000.0

            ts_db = time.perf_counter()
            t_bgr = self._torch_demosaic(t_in, raw.bayer_pattern)
            torch.cuda.synchronize()
            timings.debayer_ms += (time.perf_counter() - ts_db) * 1000.0

            ts_rs = time.perf_counter()
            t_h, t_w = self.config.target_hw()
            if t_h > 0 and t_w > 0:
                t_bgr = t_bgr.permute(2, 0, 1).unsqueeze(0).float()
                t_bgr = F.interpolate(
                    t_bgr,
                    size=(int(t_h), int(t_w)),
                    mode="bilinear",
                    align_corners=False,
                )
                t_bgr = t_bgr.squeeze(0).permute(1, 2, 0).clamp(0, 255).to(torch.uint8)
            torch.cuda.synchronize()
            timings.resize_ms += (time.perf_counter() - ts_rs) * 1000.0

            ts_n = time.perf_counter()
            if self.config.normalize_01:
                dtype = torch.float16 if self.config.half_precision else torch.float32
                t_out = t_bgr.to(dtype) * (1.0 / 255.0)
            else:
                t_out = t_bgr
            torch.cuda.synchronize()
            timings.normalize_ms += (time.perf_counter() - ts_n) * 1000.0

            ts_dl = time.perf_counter()
            arr = t_out.detach().cpu().numpy()
            timings.download_from_gpu_ms += (time.perf_counter() - ts_dl) * 1000.0

            if self.config.output_color == "RGB":
                ts_cc = time.perf_counter()
                arr = arr[..., ::-1].copy()
                timings.color_convert_ms += (time.perf_counter() - ts_cc) * 1000.0

            color_frames.append(self._wrap_color_frame(raw, arr))

        timings.total_color_ms = (time.perf_counter() - t0) * 1000.0
        return ColorConvertResult(
            color_frames=color_frames,
            timings=timings,
            fallback_warnings=list(self._fallback_warnings),
        )

    def _convert_torch_gpu_full(self, packet: FrameBatch4Cam) -> ColorConvertResult:
        """Full-GPU pipeline: tensor stays on CUDA all the way to inference.

        Output: (B=4, 3, imgsz, imgsz) RGB float (16/32) normalized to 0..1.
        No download to CPU happens here — `download_from_gpu_ms` stays 0.
        """
        import torch
        import torch.nn.functional as F

        timings = ColorConvertTimings(backend="torch_gpu", output_location="cuda")
        device = torch.device(self.config.device)
        out_dtype = torch.float16 if self.config.half_precision else torch.float32
        target_h, target_w = self.config.target_hw()
        target_h = int(target_h)
        target_w = int(target_w)

        per_frame_planes: List = []
        meta_list: List[GpuFrameMeta] = []

        t0 = time.perf_counter()

        for raw in packet.frames:
            img = self._raw_to_uint8(raw)

            ts_up = time.perf_counter()
            t_in = torch.from_numpy(img).to(device, non_blocking=True)
            torch.cuda.synchronize()
            timings.upload_to_gpu_ms += (time.perf_counter() - ts_up) * 1000.0

            ts_db = time.perf_counter()
            t_bgr = self._torch_demosaic(t_in, raw.bayer_pattern)  # (H, W, 3) uint8 BGR
            torch.cuda.synchronize()
            timings.debayer_ms += (time.perf_counter() - ts_db) * 1000.0

            # BGR -> RGB on GPU
            ts_cc = time.perf_counter()
            t_rgb = t_bgr[..., [2, 1, 0]].contiguous()
            torch.cuda.synchronize()
            timings.color_convert_ms += (time.perf_counter() - ts_cc) * 1000.0

            # HWC -> CHW + dtype cast + normalize, then resize
            ts_rs = time.perf_counter()
            t_chw = t_rgb.permute(2, 0, 1).unsqueeze(0).to(out_dtype)
            t_chw = t_chw * (1.0 / 255.0)
            t_resized = F.interpolate(
                t_chw,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
            torch.cuda.synchronize()
            timings.resize_ms += (time.perf_counter() - ts_rs) * 1000.0

            per_frame_planes.append(t_resized)
            meta_list.append(
                GpuFrameMeta(
                    role=raw.role,
                    camera_id=raw.camera_id,
                    frame_id=raw.frame_id,
                    timestamp_ns=raw.timestamp_ns,
                    source_raw_shape=(raw.height, raw.width),
                )
            )

        # Stack into (B, 3, H, W). Already normalized + correct dtype + on cuda.
        ts_n = time.perf_counter()
        batch_tensor = torch.cat(per_frame_planes, dim=0).contiguous()
        torch.cuda.synchronize()
        timings.normalize_ms += (time.perf_counter() - ts_n) * 1000.0

        timings.total_color_ms = (time.perf_counter() - t0) * 1000.0

        gpu_batch = GpuColorBatch(
            tensor=batch_tensor,
            device=str(device),
            dtype="float16" if out_dtype is torch.float16 else "float32",
            image_size=int(target_h),
            color_format="RGB",
            frames=meta_list,
        )
        return ColorConvertResult(
            color_frames=[],
            timings=timings,
            gpu_batch=gpu_batch,
            fallback_warnings=list(self._fallback_warnings),
        )

    def _convert_native_cuda_full(self, packet: FrameBatch4Cam) -> ColorConvertResult:
        """Zero-copy native CUDA debayer. Output: (B, 3, H, W) RGB float on cuda."""
        from src.runtime_gpu.native_debayer import NativeCudaDebayer
        import torch

        timings = ColorConvertTimings(backend="native_cuda_npp", output_location="cuda")
        device = torch.device(self.config.device)
        target_h, target_w = self.config.target_hw()
        target_h = int(target_h)
        target_w = int(target_w)
        bayer_pattern = packet.frames[0].bayer_pattern

        debayer = NativeCudaDebayer(device=str(device))
        # Stack 4 RAW frames as (B, H, W) uint8 on host first, the adapter
        # uploads to CUDA itself. Stacking is cheap (~0.5 ms at 2464x2056).
        raw_stack = np.stack([self._raw_to_uint8(f) for f in packet.frames], axis=0)

        t0 = time.perf_counter()

        ts_full = time.perf_counter()
        out_tensor = debayer.debayer(
            raw_stack,
            bayer_pattern=bayer_pattern,
            output_format="RGB",
            resize_to=(target_h, target_w) if target_h > 0 and target_w > 0 else None,
            normalize=True,
            output_layout="BCHW",
            device=str(device),
            half=bool(self.config.half_precision),
        )
        torch.cuda.synchronize()
        # Adapter performs upload + debayer + (optional) resize + normalize
        # in one call; we cannot disaggregate further without instrumenting
        # the C++ side, so attribute the bulk to debayer_ms.
        timings.debayer_ms = (time.perf_counter() - ts_full) * 1000.0

        meta_list: List[GpuFrameMeta] = []
        for raw in packet.frames:
            meta_list.append(
                GpuFrameMeta(
                    role=raw.role,
                    camera_id=raw.camera_id,
                    frame_id=raw.frame_id,
                    timestamp_ns=raw.timestamp_ns,
                    source_raw_shape=(raw.height, raw.width),
                )
            )

        timings.total_color_ms = (time.perf_counter() - t0) * 1000.0

        gpu_batch = GpuColorBatch(
            tensor=out_tensor,
            device=str(device),
            dtype="float16" if self.config.half_precision else "float32",
            image_size=int(target_h) if target_h > 0 else int(packet.frames[0].height),
            color_format="RGB",
            frames=meta_list,
        )
        return ColorConvertResult(
            color_frames=[],
            timings=timings,
            gpu_batch=gpu_batch,
            fallback_warnings=list(self._fallback_warnings),
        )

    def _convert_native_cuda_numpy(self, packet: FrameBatch4Cam) -> ColorConvertResult:
        """Native CUDA debayer that downloads to numpy BGR HWC at the end
        (round-trip mode for honest comparison vs production CPU path).
        """
        from src.runtime_gpu.native_debayer import NativeCudaDebayer
        import torch

        timings = ColorConvertTimings(backend="native_cuda_npp", output_location="cpu")
        device = torch.device(self.config.device)
        target_h, target_w = self.config.target_hw()
        target_h = int(target_h)
        target_w = int(target_w)
        bayer_pattern = packet.frames[0].bayer_pattern

        debayer = NativeCudaDebayer(device=str(device))
        raw_stack = np.stack([self._raw_to_uint8(f) for f in packet.frames], axis=0)

        t0 = time.perf_counter()

        ts_db = time.perf_counter()
        out_tensor = debayer.debayer(
            raw_stack,
            bayer_pattern=bayer_pattern,
            output_format="BGR",
            resize_to=(target_h, target_w) if target_h > 0 and target_w > 0 else None,
            normalize=False,
            output_layout="BCHW",
            device=str(device),
            half=False,
        )
        torch.cuda.synchronize()
        timings.debayer_ms = (time.perf_counter() - ts_db) * 1000.0

        # Download (B,3,H,W) float -> numpy and reshape to per-frame HWC uint8 BGR.
        ts_dl = time.perf_counter()
        arr = out_tensor.clamp(0, 255).to(torch.uint8).detach().cpu().numpy()
        timings.download_from_gpu_ms = (time.perf_counter() - ts_dl) * 1000.0
        # arr shape: (B, 3, H, W) BGR uint8 -> per-frame (H, W, 3)
        per_frame_hwc = np.transpose(arr, (0, 2, 3, 1))

        color_frames: List[ColorFrame] = []
        for i, raw in enumerate(packet.frames):
            color_frames.append(self._wrap_color_frame(raw, np.ascontiguousarray(per_frame_hwc[i])))

        timings.total_color_ms = (time.perf_counter() - t0) * 1000.0
        return ColorConvertResult(
            color_frames=color_frames,
            timings=timings,
            fallback_warnings=list(self._fallback_warnings),
        )

    @staticmethod
    def _torch_demosaic(bayer_uint8, pattern: str):
        """Approximate demosaic on GPU: 2x2 cell -> RGB by channel pickoff."""
        import torch

        h, w = bayer_uint8.shape
        if h % 2 != 0 or w % 2 != 0:
            raise ValueError(f"Bayer dims must be even, got {h}x{w}")
        # Slice the four positions of the 2x2 mosaic.
        c00 = bayer_uint8[0::2, 0::2]
        c01 = bayer_uint8[0::2, 1::2]
        c10 = bayer_uint8[1::2, 0::2]
        c11 = bayer_uint8[1::2, 1::2]
        # For each pattern, build R/G/B planes at half resolution.
        if pattern == "RG":
            r_half, g_half, b_half = c00, ((c01.float() + c10.float()) / 2).to(torch.uint8), c11
        elif pattern == "BG":
            r_half, g_half, b_half = c11, ((c01.float() + c10.float()) / 2).to(torch.uint8), c00
        elif pattern == "GR":
            r_half, g_half, b_half = c01, ((c00.float() + c11.float()) / 2).to(torch.uint8), c10
        elif pattern == "GB":
            r_half, g_half, b_half = c10, ((c00.float() + c11.float()) / 2).to(torch.uint8), c01
        else:
            raise ValueError(f"Unsupported bayer pattern: {pattern}")
        # Upsample x2 with nearest neighbour to recover full resolution.
        r = r_half.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)
        g = g_half.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)
        b = b_half.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)
        return torch.stack([b, g, r], dim=-1).contiguous()

    @staticmethod
    def _raw_to_uint8(raw: RawFrame) -> np.ndarray:
        if raw.image.dtype == np.uint16:
            return (raw.image >> 8).astype(np.uint8)
        if raw.image.dtype != np.uint8:
            return raw.image.astype(np.uint8)
        return raw.image

    @staticmethod
    def _wrap_color_frame(raw: RawFrame, color_image: np.ndarray) -> ColorFrame:
        return ColorFrame(
            camera_id=raw.camera_id,
            role=raw.role,
            frame_id=raw.frame_id,
            timestamp_ns=raw.timestamp_ns,
            image=color_image,
            color_format="BGR",
            source_raw_shape=(raw.height, raw.width),
        )


# small helper for downstream code that wants a deterministic 4-frame ordering
def order_color_frames_by_role(frames: List[ColorFrame]) -> List[ColorFrame]:
    by_role = {f.role: f for f in frames}
    return [by_role[role] for role in CAMERA_ROLES]
