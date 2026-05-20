"""Fast-path executor for the production live pipeline.

Owns the heavy resources (TRT model, NativeCudaDebayer adapter) and the
per-role tracker state for the duration of a live session. Stateless
across packets EXCEPT for tracker continuity.

Critical contract:
    process_packet(raw_frames, roles, ts_ns) -> FastPathPacketResult

    Inputs:
      raw_frames : List[np.ndarray] of (H, W) uint8 — exactly len == batch_size
      roles      : List[str]                       — one per frame, in order
      ts_ns      : int                             — packet-level timestamp

    Output:
      FastPathPacketResult with per-role detections, per-role tracks,
      stage timings, and metadata flags.

The tensor handed to model.predict is RGB float on CUDA in BCHW layout —
no .cpu(), no .numpy(), no PIL conversion in the hot path.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from src.runtime_production.fast_path_config import FastPathConfig

_LOG = logging.getLogger(__name__)


@dataclass(slots=True)
class FastPathPacketResult:
    roles: List[str]
    detections_per_role: List[List[Dict[str, Any]]]
    tracks_per_role: List[List[Dict[str, Any]]]
    stage_ms: Dict[str, float]
    ts_ns: int
    color_backend: str
    inference_backend: str
    color_output_location: str
    inference_input_location: str
    zero_copy_to_inference: bool
    preprocessing_path: str = ""
    tensor_device: str = ""
    tensor_shape: tuple = ()
    tensor_layout: str = "BCHW"
    fallback_used: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class FastPathExecutor:
    """Owns the TRT model + native debayer; processes 4-camera packets."""

    def __init__(self, config: FastPathConfig) -> None:
        self.config = config
        self._model = None
        self._debayer = None
        self._tracks_per_role: Dict[str, Dict[int, dict]] = defaultdict(dict)
        self._next_track_id = 1
        self._initialized = False
        self._init_error: Optional[str] = None

    # ------------------------------------------------------------------ init

    def initialize(self) -> bool:
        if self._initialized:
            return True
        err = self.config.validate()
        if err:
            self._init_error = err
            _LOG.warning("[FastPathExecutor] config invalid: %s", err)
            return False

        try:
            from src.runtime_gpu.native_debayer import NativeCudaDebayer
        except Exception as exc:
            self._init_error = f"native debayer import failed: {exc!r}"
            _LOG.warning("[FastPathExecutor] %s", self._init_error)
            return False

        debayer = NativeCudaDebayer(device=self.config.device)
        if not NativeCudaDebayer.is_available():
            info = NativeCudaDebayer.describe_backend()
            self._init_error = f"native debayer not available: {info.get('error')}"
            _LOG.warning("[FastPathExecutor] %s", self._init_error)
            return False
        self._debayer = debayer

        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:
            self._init_error = f"ultralytics import failed: {exc!r}"
            _LOG.warning("[FastPathExecutor] %s", self._init_error)
            return False

        try:
            _LOG.info(
                "[FastPathExecutor] loading TRT engine: %s (b%d, imgsz=%d)",
                self.config.engine_path,
                self.config.batch_size,
                self.config.imgsz,
            )
            self._model = YOLO(self.config.engine_path)
        except Exception as exc:
            self._init_error = f"engine load failed: {exc!r}"
            _LOG.error("[FastPathExecutor] %s", self._init_error)
            return False

        self._warmup()
        self._initialized = True
        return True

    def _warmup(self) -> None:
        if self.config.warmup_iterations <= 0 or self._model is None or self._debayer is None:
            return
        h, w = self.config.imgsz_hw()
        dummy = np.zeros((self.config.batch_size, h * 2, w * 2), dtype=np.uint8)
        for i in range(int(self.config.warmup_iterations)):
            try:
                tensor = self._debayer.debayer(
                    dummy,
                    bayer_pattern=self.config.bayer_pattern,
                    output_format="RGB",
                    resize_to=(h, w),
                    normalize=True,
                    output_layout="BCHW",
                    device=self.config.device,
                    half=self.config.half,
                )
                _ = self._model.predict(
                    tensor,
                    imgsz=self.config.imgsz_for_predict(),
                    conf=self.config.confidence,
                    iou=self.config.iou,
                    device=self.config.device,
                    half=self.config.half,
                    max_det=self.config.max_det,
                    verbose=False,
                )
            except Exception as exc:
                _LOG.warning("[FastPathExecutor] warmup iter %d failed: %s", i, exc)
                break

    # --------------------------------------------------------- accessors

    @property
    def is_ready(self) -> bool:
        return bool(self._initialized and self._model is not None and self._debayer is not None)

    @property
    def init_error(self) -> Optional[str]:
        return self._init_error

    def describe(self) -> dict:
        try:
            from src.runtime_gpu.native_debayer import NativeCudaDebayer
            native_info = NativeCudaDebayer.describe_backend()
        except Exception:
            native_info = {"available": False, "detail": "?"}
        return {
            "ready": self.is_ready,
            "engine_path": self.config.engine_path,
            "batch_size": self.config.batch_size,
            "imgsz": self.config.imgsz,
            "bayer_pattern": self.config.bayer_pattern,
            "half": self.config.half,
            "device": self.config.device,
            "color_backend": self.config.color_backend,
            "use_unified_image_processor": bool(self.config.use_unified_image_processor),
            "native_backend_detail": native_info.get("detail"),
            "inference_backend": self.config.inference_backend,
            "init_error": self._init_error,
        }

    # ---------------------------------------------------------- processing

    def process_packet(
        self,
        raw_frames: List[np.ndarray],
        roles: List[str],
        ts_ns: int,
    ) -> FastPathPacketResult:
        if not self.is_ready:
            return self._error_result(roles, ts_ns, f"executor not ready: {self._init_error}")

        if len(raw_frames) != self.config.batch_size:
            return self._error_result(
                roles, ts_ns,
                f"batch size mismatch: got {len(raw_frames)} frames, "
                f"engine expects {self.config.batch_size}",
            )
        if len(roles) != len(raw_frames):
            return self._error_result(roles, ts_ns, "len(roles) != len(raw_frames)")
        if self._debayer is None or self._model is None:
            return self._error_result(roles, ts_ns, "executor not initialized")

        try:
            import torch
        except Exception as exc:
            return self._error_result(roles, ts_ns, f"torch unavailable: {exc}")

        stage_ms: Dict[str, float] = {}
        t_total = time.perf_counter()

        # Stack 4 frames (host).
        t_st = time.perf_counter()
        try:
            stack = np.stack([np.ascontiguousarray(f) for f in raw_frames], axis=0)
            if stack.dtype != np.uint8:
                stack = stack.astype(np.uint8, copy=False)
        except Exception as exc:
            return self._error_result(roles, ts_ns, f"frame stacking failed: {exc}")
        stage_ms["stack_ms"] = (time.perf_counter() - t_st) * 1000.0

        # Native debayer + resize + normalize, all on CUDA.
        fallback_used = None
        try:
            tensor, preprocessing_path = self._prepare_bchw_tensor(stack, stage_ms, torch)
        except Exception as exc:
            return self._error_result(roles, ts_ns, f"native debayer failed: {exc}")
        fallback_used = stage_ms.pop("_fallback_used", None)

        zero_copy = bool(getattr(tensor, "is_cuda", False))
        color_output_location = "cuda" if zero_copy else "cpu"
        tensor_device = self._tensor_device_label(tensor)
        tensor_shape = self._tensor_shape_tuple(tensor)

        # TensorRT inference on the CUDA tensor.
        t_inf = time.perf_counter()
        try:
            ult_results = self._model.predict(
                tensor,
                imgsz=self.config.imgsz_for_predict(),
                conf=self.config.confidence,
                iou=self.config.iou,
                device=self.config.device,
                half=self.config.half,
                max_det=self.config.max_det,
                verbose=False,
            )
            if str(self.config.device).startswith("cuda"):
                torch.cuda.synchronize()
        except Exception as exc:
            return self._error_result(roles, ts_ns, f"predict failed: {exc}")
        stage_ms["inference_ms"] = (time.perf_counter() - t_inf) * 1000.0

        inference_input_location = "cuda" if zero_copy else "cpu"

        # Postprocess: per-role detections + tracker update.
        t_post = time.perf_counter()
        per_role_dets = self._extract_detections(ult_results, len(roles))
        per_role_tracks = self._update_tracks_per_role(per_role_dets, roles)
        stage_ms["postprocess_ms"] = (time.perf_counter() - t_post) * 1000.0

        stage_ms["total_packet_ms"] = (time.perf_counter() - t_total) * 1000.0

        return FastPathPacketResult(
            roles=list(roles),
            detections_per_role=per_role_dets,
            tracks_per_role=per_role_tracks,
            stage_ms=stage_ms,
            ts_ns=int(ts_ns),
            color_backend=self.config.color_backend,
            inference_backend=self.config.inference_backend,
            color_output_location=color_output_location,
            inference_input_location=inference_input_location,
            zero_copy_to_inference=zero_copy,
            preprocessing_path=preprocessing_path,
            tensor_device=tensor_device,
            tensor_shape=tensor_shape,
            tensor_layout="BCHW",
            fallback_used=fallback_used,
            error=None,
        )

    # ---------------------------------------------------------- helpers

    def _prepare_bchw_tensor(self, stack: np.ndarray, stage_ms: Dict[str, float], torch):
        if self.config.use_unified_image_processor:
            try:
                return self._prepare_bchw_tensor_unified(stack, stage_ms, torch)
            except Exception as exc:
                stage_ms["_fallback_used"] = f"unified_image_processor_failed: {type(exc).__name__}: {exc}"
                _LOG.warning(
                    "[FastPathExecutor] unified image processor failed; falling back to native debayer: %s",
                    exc,
                )
        return self._prepare_bchw_tensor_native(stack, stage_ms, torch)

    def _prepare_bchw_tensor_unified(self, stack: np.ndarray, stage_ms: Dict[str, float], torch):
        from src.runtime_gpu.gpu_image_processor import (
            GpuImageProcessor,
            ImageProcessorConfig,
        )

        h, w = self.config.imgsz_hw()
        t_color = time.perf_counter()
        processor = GpuImageProcessor(
            ImageProcessorConfig(
                target_size=(h, w),
                input_color="BAYER",
                bayer_pattern=self.config.bayer_pattern,
                normalize=True,
                device=self.config.device,
                prefer_cuda=True,
                half_precision=self.config.half,
            )
        )
        result = processor.process_batch(stack)
        tensor = result.tensor
        if str(self.config.device).startswith("cuda") and not bool(getattr(tensor, "is_cuda", False)):
            raise RuntimeError("unified image processor returned a non-CUDA tensor")
        self._validate_bchw_tensor(tensor, h, w, source="unified")
        if str(self.config.device).startswith("cuda"):
            torch.cuda.synchronize()
        for key, value in result.stage_ms.items():
            stage_ms[f"unified_{key}"] = float(value)
        stage_ms["color_ms"] = (time.perf_counter() - t_color) * 1000.0
        return tensor, "unified"

    def _prepare_bchw_tensor_native(self, stack: np.ndarray, stage_ms: Dict[str, float], torch):
        if self._debayer is None:
            raise RuntimeError("native debayer unavailable")
        t_color = time.perf_counter()
        h, w = self.config.imgsz_hw()
        tensor = self._debayer.debayer(
            stack,
            bayer_pattern=self.config.bayer_pattern,
            output_format="RGB",
            resize_to=(h, w),
            normalize=True,
            output_layout="BCHW",
            device=self.config.device,
            half=self.config.half,
        )
        if str(self.config.device).startswith("cuda"):
            torch.cuda.synchronize()
        self._validate_bchw_tensor(tensor, h, w, source="native")
        stage_ms["color_ms"] = (time.perf_counter() - t_color) * 1000.0
        return tensor, "native"

    def _validate_bchw_tensor(self, tensor, h: int, w: int, source: str) -> None:
        shape = self._tensor_shape_tuple(tensor)
        expected = (self.config.batch_size, 3, int(h), int(w))
        if shape != expected:
            raise RuntimeError(f"unexpected {source} tensor shape: {shape}, expected {expected}")

    @staticmethod
    def _tensor_shape_tuple(tensor) -> tuple:
        try:
            return tuple(int(x) for x in getattr(tensor, "shape", ()))
        except Exception:
            return ()

    @staticmethod
    def _tensor_device_label(tensor) -> str:
        try:
            device = getattr(tensor, "device", None)
            if device is not None:
                return str(device)
        except Exception:
            pass
        try:
            return "cuda" if bool(getattr(tensor, "is_cuda", False)) else "cpu"
        except Exception:
            return ""

    def _error_result(self, roles: List[str], ts_ns: int, msg: str) -> FastPathPacketResult:
        return FastPathPacketResult(
            roles=list(roles),
            detections_per_role=[[] for _ in roles],
            tracks_per_role=[[] for _ in roles],
            stage_ms={},
            ts_ns=int(ts_ns),
            color_backend=self.config.color_backend,
            inference_backend=self.config.inference_backend,
            color_output_location="",
            inference_input_location="",
            zero_copy_to_inference=False,
            fallback_used="error",
            error=msg,
        )

    def _extract_detections(self, ult_results, n_expected: int) -> List[List[Dict[str, Any]]]:
        out: List[List[Dict[str, Any]]] = [[] for _ in range(n_expected)]
        if not ult_results:
            return out
        try:
            import numpy as _np
        except Exception:
            return out
        for i, r in enumerate(ult_results[:n_expected]):
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            try:
                xyxy = boxes.xyxy.detach().cpu().numpy() if hasattr(boxes.xyxy, "detach") else _np.asarray(boxes.xyxy)
                conf = boxes.conf.detach().cpu().numpy() if hasattr(boxes.conf, "detach") else _np.asarray(boxes.conf)
                cls = boxes.cls.detach().cpu().numpy() if hasattr(boxes.cls, "detach") else _np.asarray(boxes.cls)
            except Exception:
                continue
            ball_id = self.config.ball_class_id
            dets: List[Dict[str, Any]] = []
            for j in range(len(xyxy)):
                cls_j = int(cls[j])
                if ball_id is not None and cls_j != int(ball_id):
                    continue
                x1, y1, x2, y2 = (
                    float(xyxy[j, 0]), float(xyxy[j, 1]),
                    float(xyxy[j, 2]), float(xyxy[j, 3]),
                )
                dets.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(conf[j]),
                    "class_id": cls_j,
                })
            out[i] = dets
        return out

    def _update_tracks_per_role(
        self,
        per_role_dets: List[List[Dict[str, Any]]],
        roles: List[str],
        max_dist_px: float = 120.0,
        max_missed: int = 5,
    ) -> List[List[Dict[str, Any]]]:
        per_role_tracks: List[List[Dict[str, Any]]] = []
        for role, dets in zip(roles, per_role_dets):
            tracks = self._tracks_per_role[role]
            det_centers = [
                (0.5 * (d["bbox"][0] + d["bbox"][2]), 0.5 * (d["bbox"][1] + d["bbox"][3]))
                for d in dets
            ]
            used = set()
            for tid in list(tracks.keys()):
                tr = tracks[tid]
                best_i = -1
                best_d = 1e18
                for i, (cx, cy) in enumerate(det_centers):
                    if i in used:
                        continue
                    d = (cx - tr["cx"]) ** 2 + (cy - tr["cy"]) ** 2
                    if d < best_d:
                        best_d = d
                        best_i = i
                if best_i >= 0 and best_d <= (max_dist_px * max_dist_px):
                    used.add(best_i)
                    cx, cy = det_centers[best_i]
                    det = dets[best_i]
                    tr["cx"] = cx
                    tr["cy"] = cy
                    tr["bbox"] = list(det["bbox"])
                    tr["confidence"] = float(det["confidence"])
                    tr["missed"] = 0
                else:
                    tr["missed"] += 1
                    if tr["missed"] > max_missed:
                        tracks.pop(tid, None)
            for i, det in enumerate(dets):
                if i in used:
                    continue
                cx, cy = det_centers[i]
                tid = self._next_track_id
                self._next_track_id += 1
                tracks[tid] = {
                    "id": tid,
                    "cx": cx,
                    "cy": cy,
                    "bbox": list(det["bbox"]),
                    "confidence": float(det["confidence"]),
                    "missed": 0,
                }
            per_role_tracks.append([
                {"track_id": int(tid), "bbox": list(tr["bbox"]),
                 "confidence": float(tr["confidence"])}
                for tid, tr in tracks.items()
                if tr.get("missed", 0) <= max_missed
            ])
        return per_role_tracks
