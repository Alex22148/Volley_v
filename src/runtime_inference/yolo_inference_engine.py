"""YOLO inference engine for the runtime benchmark.

Loads a model from .pt / .onnx / .engine via Ultralytics (preferred) or
in dry-run mode skips inference entirely. The engine is independent of
vision/yolo_module.py; that production class is left untouched.

Backends:
    ultralytics : YOLO(model_path) — supports .pt / .onnx / .engine
    tensorrt    : strict .engine file via Ultralytics (no silent fallback)
    auto        : tensorrt if model_path endswith .engine, else ultralytics

Per-call timings are reported as InferenceTimings.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np

from src.runtime_sources.base_frame_source import (
    CAMERA_ROLES,
    ColorFrame,
    InferenceResult,
)

_LOG = logging.getLogger(__name__)

_BackendName = Literal["auto", "ultralytics", "tensorrt", "dry_run"]
_MIN_TRT_ENGINE_BYTES = 1024


@dataclass(slots=True)
class InferenceConfig:
    model_path: Optional[str] = None
    backend: _BackendName = "auto"
    batch_size: int = 4
    # int (square) or (H, W) tuple. Ultralytics predict() accepts both.
    image_size: object = 640
    confidence: float = 0.25
    iou: float = 0.45
    device: str = "cuda"
    use_half: bool = True
    warmup_iterations: int = 3
    ball_class_id: Optional[int] = 0
    max_det: int = 300
    dry_run: bool = False

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        ish = self.image_size
        if isinstance(ish, (tuple, list)):
            if len(ish) != 2:
                raise ValueError(f"image_size tuple must be (H, W); got {ish}")
            if int(ish[0]) <= 0 or int(ish[1]) <= 0:
                raise ValueError("image_size dims must be > 0")
        else:
            if int(ish) <= 0:  # type: ignore[arg-type]
                raise ValueError("image_size must be > 0")
        if self.warmup_iterations < 0:
            raise ValueError("warmup_iterations must be >= 0")
        if self.backend not in ("auto", "ultralytics", "tensorrt", "dry_run"):
            raise ValueError(f"unknown backend: {self.backend}")
        if not self.dry_run and self.backend != "dry_run" and not self.model_path:
            raise ValueError("model_path is required unless dry_run=True")

    def image_hw(self) -> tuple:
        """Return (H, W) tuple regardless of int/tuple input."""
        ish = self.image_size
        if isinstance(ish, (tuple, list)):
            return (int(ish[0]), int(ish[1]))
        v = int(ish)  # type: ignore[arg-type]
        return (v, v)

    def imgsz_for_predict(self):
        """Format expected by Ultralytics model.predict(): int or [H, W]."""
        ish = self.image_size
        if isinstance(ish, (tuple, list)):
            return [int(ish[0]), int(ish[1])]
        return int(ish)  # type: ignore[arg-type]


@dataclass(slots=True)
class InferenceTimings:
    preprocess_ms: float = 0.0
    upload_to_gpu_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0
    total_inference_ms: float = 0.0
    backend: str = ""
    input_location: str = "cpu"
    zero_copy_to_inference: bool = False
    gpu_debayer_only_not_full_gpu_pipeline: bool = False
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "preprocess_ms": float(self.preprocess_ms),
            "upload_to_gpu_ms": float(self.upload_to_gpu_ms),
            "inference_ms": float(self.inference_ms),
            "postprocess_ms": float(self.postprocess_ms),
            "total_inference_ms": float(self.total_inference_ms),
            "backend": str(self.backend),
            "input_location": str(self.input_location),
            "zero_copy_to_inference": bool(self.zero_copy_to_inference),
            "gpu_debayer_only_not_full_gpu_pipeline": bool(
                self.gpu_debayer_only_not_full_gpu_pipeline
            ),
            "notes": str(self.notes),
        }


@dataclass(slots=True)
class _BatchOutput:
    results: List[InferenceResult]
    timings: InferenceTimings


class YoloInferenceEngine:
    """Batch inference for 4-camera color frames.

    The engine accepts already color-converted frames (from
    GpuColorConverter.convert_packet). Letterbox + tensor preprocess is
    performed inside Ultralytics for the .pt/.onnx/.engine path. For
    timing diagnostics, we measure the wall-clock around predict().
    """

    def __init__(self, config: InferenceConfig) -> None:
        config.validate()
        self.config = config
        self._model = None
        self._effective_backend: str = "dry_run" if config.dry_run else self._resolve_backend()
        self._is_dry = self._effective_backend == "dry_run"
        if not self._is_dry:
            self._load_model()
        else:
            _LOG.warning("[YoloInferenceEngine] dry-run mode — no model loaded")

    @property
    def backend(self) -> str:
        return self._effective_backend

    @property
    def is_dry_run(self) -> bool:
        return self._is_dry

    def _resolve_backend(self) -> str:
        cfg = self.config
        if cfg.backend == "dry_run":
            return "dry_run"
        if cfg.backend == "tensorrt":
            self._validate_engine_path()
            return "tensorrt"
        if cfg.backend == "ultralytics":
            return "ultralytics"
        # auto: prefer engine if path is .engine
        path = Path(str(cfg.model_path or ""))
        if path.suffix.lower() == ".engine":
            self._validate_engine_path()
            return "tensorrt"
        return "ultralytics"

    def _validate_engine_path(self) -> None:
        path = Path(str(self.config.model_path or ""))
        if not path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {path}")
        if path.stat().st_size < _MIN_TRT_ENGINE_BYTES:
            raise RuntimeError(
                f"TensorRT engine too small ({path.stat().st_size} B), refusing to load: {path}"
            )

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Ultralytics is required for inference. Install: pip install ultralytics"
            ) from exc

        path = str(self.config.model_path)
        _LOG.info("[YoloInferenceEngine] loading model: %s (backend=%s)", path, self._effective_backend)
        model = YOLO(path)
        try:
            model.to(self.config.device)
        except Exception as exc:
            _LOG.warning("[YoloInferenceEngine] model.to(%s) skipped: %s", self.config.device, exc)
        self._model = model

    def warmup(self) -> None:
        if self._is_dry or self._model is None:
            return
        if self.config.warmup_iterations <= 0:
            return
        warm_h, warm_w = self.config.image_hw()
        dummy = np.zeros((warm_h, warm_w, 3), dtype=np.uint8)
        batch = [dummy.copy() for _ in range(self.config.batch_size)]
        _LOG.info(
            "[YoloInferenceEngine] warmup: %d iterations x batch=%d @ imgsz=%dx%d",
            self.config.warmup_iterations,
            self.config.batch_size,
            warm_h, warm_w,
        )
        for _ in range(self.config.warmup_iterations):
            try:
                self._model.predict(
                    batch,
                    imgsz=self.config.imgsz_for_predict(),
                    conf=self.config.confidence,
                    iou=self.config.iou,
                    device=self.config.device,
                    half=self._effective_half(),
                    verbose=False,
                )
            except Exception as exc:
                _LOG.warning("[YoloInferenceEngine] warmup iter failed: %s", exc)
                break

    def _effective_half(self) -> bool:
        if not self.config.use_half:
            return False
        return str(self.config.device).startswith("cuda")

    def infer_color_frames(self, color_frames: List[ColorFrame]) -> _BatchOutput:
        """Run one batched inference call across the supplied color frames.

        Input is on CPU (numpy). Ultralytics will internally upload to GPU,
        letterbox, normalize, and BGR->RGB. We attribute its `preprocess`
        speed bucket to `preprocess_ms` (pre_transform) and approximate
        `upload_to_gpu_ms` from a side-channel measurement so the report is
        honest about the hidden round-trip cost.
        """
        timings = InferenceTimings(
            backend=self._effective_backend,
            input_location="cpu",
            zero_copy_to_inference=False,
        )
        t_total = time.perf_counter()

        t_pre = time.perf_counter()
        ordered = self._order_for_batch(color_frames)
        np_batch = [f.image for f in ordered]
        timings.preprocess_ms = (time.perf_counter() - t_pre) * 1000.0

        if self._is_dry or self._model is None:
            # measure a representative side-channel upload so dry-run still
            # shows what the hidden round-trip would cost
            timings.upload_to_gpu_ms = self._measure_upload_cost(np_batch)
            timings.total_inference_ms = (time.perf_counter() - t_total) * 1000.0
            results = [self._empty_result(f, timings) for f in ordered]
            return _BatchOutput(results=results, timings=timings)

        # Side-channel: time how long a torch upload of this batch would take.
        # This isolates "would-be CPU->GPU cost" from Ultralytics' opaque preprocess.
        timings.upload_to_gpu_ms = self._measure_upload_cost(np_batch)

        try:
            ult_results = self._model.predict(
                np_batch,
                imgsz=self.config.imgsz_for_predict(),
                conf=self.config.confidence,
                iou=self.config.iou,
                device=self.config.device,
                half=self._effective_half(),
                max_det=self.config.max_det,
                verbose=False,
            )
        except Exception as exc:
            _LOG.error("[YoloInferenceEngine] inference call failed: %s", exc)
            timings.total_inference_ms = (time.perf_counter() - t_total) * 1000.0
            results = [self._empty_result(f, timings) for f in ordered]
            return _BatchOutput(results=results, timings=timings)
        self._merge_ultralytics_speed(ult_results, timings)

        t_post = time.perf_counter()
        per_frame: List[List[dict]] = [self._convert_ult_result(r) for r in ult_results]
        post_local_ms = (time.perf_counter() - t_post) * 1000.0
        timings.postprocess_ms = max(timings.postprocess_ms, post_local_ms)

        timings.total_inference_ms = (time.perf_counter() - t_total) * 1000.0
        results: List[InferenceResult] = []
        for frame, dets in zip(ordered, per_frame):
            results.append(
                InferenceResult(
                    packet_id=-1,
                    role=frame.role,
                    frame_id=frame.frame_id,
                    timestamp_ns=frame.timestamp_ns,
                    detections=dets,
                    timing_ms=timings.to_dict(),
                    image_size=self.config.image_hw()[0],
                    backend=self._effective_backend,
                )
            )
        return _BatchOutput(results=results, timings=timings)

    def infer_gpu_batch(self, gpu_batch) -> _BatchOutput:
        """Inference on a CUDA tensor batch, ideally without any CPU bounce.

        Expects a GpuColorBatch (B, 3, H, W) RGB float on cuda. If
        Ultralytics still copies internally, we set
        ``gpu_debayer_only_not_full_gpu_pipeline=True`` so the report does
        not mislead about a "full GPU pipeline".
        """
        timings = InferenceTimings(
            backend=self._effective_backend,
            input_location="cuda",
            zero_copy_to_inference=True,
        )
        t_total = time.perf_counter()

        ordered_meta = list(gpu_batch.frames)
        tensor = gpu_batch.tensor

        # If model wasn't loaded (dry run), we still validate tensor shape and report.
        if self._is_dry or self._model is None:
            timings.upload_to_gpu_ms = 0.0
            timings.total_inference_ms = (time.perf_counter() - t_total) * 1000.0
            return _BatchOutput(
                results=[self._empty_meta_result(m, timings) for m in ordered_meta],
                timings=timings,
            )

        # Sanity: tensor must be on CUDA. If not, mark as fallback and recopy.
        is_cuda = self._tensor_is_cuda(tensor)
        if not is_cuda:
            timings.zero_copy_to_inference = False
            timings.input_location = "cpu"
            timings.gpu_debayer_only_not_full_gpu_pipeline = True
            timings.notes = "input tensor was not on cuda; treated as round-trip"

        # Time the .to(device) call (no-op when already on cuda; serves as proof).
        t_up = time.perf_counter()
        try:
            import torch
            tensor = tensor.to(self.config.device, non_blocking=True)
            if str(self.config.device).startswith("cuda"):
                torch.cuda.synchronize()
        except Exception as exc:
            _LOG.warning("[YoloInferenceEngine] tensor.to(device) failed: %s", exc)
        timings.upload_to_gpu_ms = (time.perf_counter() - t_up) * 1000.0

        try:
            ult_results = self._model.predict(
                tensor,
                imgsz=self.config.imgsz_for_predict(),
                conf=self.config.confidence,
                iou=self.config.iou,
                device=self.config.device,
                half=self._effective_half(),
                max_det=self.config.max_det,
                verbose=False,
            )
        except Exception as exc:
            _LOG.warning(
                "[YoloInferenceEngine] tensor predict failed (%s); "
                "Ultralytics likely cannot consume this tensor zero-copy on this version",
                exc,
            )
            timings.zero_copy_to_inference = False
            timings.gpu_debayer_only_not_full_gpu_pipeline = True
            timings.notes = (
                f"Ultralytics rejected CUDA tensor input ({type(exc).__name__}); "
                "falling back to CPU round-trip"
            )
            return self._fallback_tensor_to_numpy(gpu_batch, ordered_meta, timings, t_total)
        self._merge_ultralytics_speed(ult_results, timings)

        t_post = time.perf_counter()
        per_frame: List[List[dict]] = [self._convert_ult_result(r) for r in ult_results]
        post_local_ms = (time.perf_counter() - t_post) * 1000.0
        timings.postprocess_ms = max(timings.postprocess_ms, post_local_ms)

        timings.total_inference_ms = (time.perf_counter() - t_total) * 1000.0

        if timings.preprocess_ms > 1.0:
            timings.gpu_debayer_only_not_full_gpu_pipeline = True
            timings.notes = (
                f"Ultralytics preprocess={timings.preprocess_ms:.2f}ms even with CUDA tensor; "
                "library still performs internal copies/transforms"
            )

        results: List[InferenceResult] = []
        for meta, dets in zip(ordered_meta, per_frame):
            results.append(
                InferenceResult(
                    packet_id=-1,
                    role=meta.role,
                    frame_id=meta.frame_id,
                    timestamp_ns=meta.timestamp_ns,
                    detections=dets,
                    timing_ms=timings.to_dict(),
                    image_size=self.config.image_hw()[0],
                    backend=self._effective_backend,
                )
            )
        return _BatchOutput(results=results, timings=timings)

    def _fallback_tensor_to_numpy(self, gpu_batch, ordered_meta, timings, t_total) -> _BatchOutput:
        model = self._model
        if model is None:
            timings.total_inference_ms = (time.perf_counter() - t_total) * 1000.0
            return _BatchOutput(
                results=[self._empty_meta_result(m, timings) for m in ordered_meta],
                timings=timings,
            )
        try:
            tensor = gpu_batch.tensor
            arr = tensor.detach().to("cpu").numpy()  # (B, 3, H, W)
            # Convert to list of HWC uint8 BGR for Ultralytics numpy path.
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            arr = arr.transpose(0, 2, 3, 1)  # BCHW -> BHWC (still RGB)
            np_batch = [a[..., ::-1].copy() for a in arr]  # RGB -> BGR
            ult_results = model.predict(
                np_batch,
                imgsz=self.config.imgsz_for_predict(),
                conf=self.config.confidence,
                iou=self.config.iou,
                device=self.config.device,
                half=self._effective_half(),
                max_det=self.config.max_det,
                verbose=False,
            )
            self._merge_ultralytics_speed(ult_results, timings)
            per_frame = [self._convert_ult_result(r) for r in ult_results]
        except Exception as exc:
            _LOG.error("[YoloInferenceEngine] fallback inference failed: %s", exc)
            per_frame = [[] for _ in ordered_meta]

        timings.total_inference_ms = (time.perf_counter() - t_total) * 1000.0
        results: List[InferenceResult] = []
        for meta, dets in zip(ordered_meta, per_frame):
            results.append(
                InferenceResult(
                    packet_id=-1,
                    role=meta.role,
                    frame_id=meta.frame_id,
                    timestamp_ns=meta.timestamp_ns,
                    detections=dets,
                    timing_ms=timings.to_dict(),
                    image_size=self.config.image_hw()[0],
                    backend=self._effective_backend,
                )
            )
        return _BatchOutput(results=results, timings=timings)

    def _measure_upload_cost(self, np_batch: List[np.ndarray]) -> float:
        """Side-channel measurement of CPU->GPU upload for the supplied batch.

        Does not affect the actual inference call, just gives an honest
        number to attribute to the round-trip mode.
        """
        if not str(self.config.device).startswith("cuda"):
            return 0.0
        try:
            import torch
            if not torch.cuda.is_available():
                return 0.0
            arr = np.stack(np_batch, axis=0) if len(np_batch) > 1 else np_batch[0][None, ...]
            t = time.perf_counter()
            tensor = torch.from_numpy(arr).to(self.config.device, non_blocking=True)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t) * 1000.0
            del tensor
            return float(elapsed)
        except Exception:
            return 0.0

    @staticmethod
    def _tensor_is_cuda(tensor) -> bool:
        try:
            return bool(getattr(tensor, "is_cuda", False))
        except Exception:
            return False

    @staticmethod
    def _merge_ultralytics_speed(ult_results, timings: InferenceTimings) -> None:
        """Pull Ultralytics' per-stage `speed` dict (ms) into our timings.

        Ultralytics returns one speed dict per result (same per batch).
        """
        if not ult_results:
            return
        speed = getattr(ult_results[0], "speed", None)
        if not isinstance(speed, dict):
            return
        try:
            timings.preprocess_ms = float(speed.get("preprocess", 0.0))
            timings.inference_ms = float(speed.get("inference", 0.0))
            timings.postprocess_ms = float(speed.get("postprocess", 0.0))
        except Exception:
            pass

    def _order_for_batch(self, color_frames: List[ColorFrame]) -> List[ColorFrame]:
        # Keep CAMERA_ROLES order if frames span all 4 roles; otherwise as-is.
        roles_seen = {f.role for f in color_frames}
        if roles_seen == set(CAMERA_ROLES):
            by_role = {f.role: f for f in color_frames}
            return [by_role[r] for r in CAMERA_ROLES]
        return list(color_frames)

    def _convert_ult_result(self, r) -> List[dict]:
        out: List[dict] = []
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            return out
        try:
            xyxy = boxes.xyxy.detach().cpu().numpy() if hasattr(boxes.xyxy, "detach") else np.asarray(boxes.xyxy)
            conf = boxes.conf.detach().cpu().numpy() if hasattr(boxes.conf, "detach") else np.asarray(boxes.conf)
            cls = boxes.cls.detach().cpu().numpy() if hasattr(boxes.cls, "detach") else np.asarray(boxes.cls)
        except Exception as exc:
            _LOG.warning("[YoloInferenceEngine] result extraction failed: %s", exc)
            return out
        ball_id = self.config.ball_class_id
        for i in range(len(xyxy)):
            cls_i = int(cls[i])
            if ball_id is not None and cls_i != int(ball_id):
                continue
            out.append(
                {
                    "x1": float(xyxy[i, 0]),
                    "y1": float(xyxy[i, 1]),
                    "x2": float(xyxy[i, 2]),
                    "y2": float(xyxy[i, 3]),
                    "confidence": float(conf[i]),
                    "class_id": cls_i,
                }
            )
        return out

    @staticmethod
    def _empty_result(frame: ColorFrame, timings: InferenceTimings) -> InferenceResult:
        return InferenceResult(
            packet_id=-1,
            role=frame.role,
            frame_id=frame.frame_id,
            timestamp_ns=frame.timestamp_ns,
            detections=[],
            timing_ms=timings.to_dict(),
            image_size=0,
            backend=timings.backend,
        )

    @staticmethod
    def _empty_meta_result(meta, timings: InferenceTimings) -> InferenceResult:
        return InferenceResult(
            packet_id=-1,
            role=meta.role,
            frame_id=meta.frame_id,
            timestamp_ns=meta.timestamp_ns,
            detections=[],
            timing_ms=timings.to_dict(),
            image_size=0,
            backend=timings.backend,
        )
