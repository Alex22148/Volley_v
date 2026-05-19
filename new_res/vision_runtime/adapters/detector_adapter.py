# file: vision_runtime/adapters/detector_adapter.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Protocol

import numpy as np

from vision_runtime.ipc.messages import DetectionPacket

try:
    import torch
except Exception:
    torch = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


DetectorBackendName = Literal["ultralytics", "tensorrt"]
MIN_TRT_ENGINE_BYTES = 1024


@dataclass(slots=True)
class DetectorConfig:
    backend: DetectorBackendName = "ultralytics"
    model_path: str = "best.pt"
    trt_engine_path: str = ""
    device: str = "cuda"
    image_size: int = 640
    confidence: float = 0.25
    ball_class_id: Optional[int] = None
    max_det: int = 300
    use_half: bool = True


class DetectorAdapter(Protocol):
    def warmup(self) -> None:
        ...

    def infer(self, frame_bgr: np.ndarray) -> List[DetectionPacket]:
        ...

    def close(self) -> None:
        ...


def _require_ultralytics() -> None:
    if YOLO is None:
        raise RuntimeError("Ultralytics is not available. Install: pip install ultralytics")


def _torch_cuda_available() -> bool:
    return bool(torch is not None and torch.cuda.is_available())


def _validate_image(frame_bgr: np.ndarray) -> np.ndarray:
    if frame_bgr is None:
        raise ValueError("frame_bgr is None")
    if not isinstance(frame_bgr, np.ndarray):
        raise TypeError(f"frame_bgr must be np.ndarray, got {type(frame_bgr)}")
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"Expected BGR image with shape [H,W,3], got {frame_bgr.shape}")
    if frame_bgr.dtype != np.uint8:
        frame_bgr = frame_bgr.astype(np.uint8, copy=False)
    if not frame_bgr.flags["C_CONTIGUOUS"]:
        frame_bgr = np.ascontiguousarray(frame_bgr)
    return frame_bgr


def _validate_ultralytics_model_path(model_path: str) -> Path:
    path = Path(str(model_path or "").strip())
    if not path:
        raise ValueError("model_path is empty")
    if not path.exists():
        raise FileNotFoundError(f"Ultralytics model file not found: {path}")
    if path.suffix.lower() not in {".pt", ".onnx", ".engine"}:
        raise ValueError(f"Unsupported Ultralytics model extension: {path.suffix}")
    return path


def _validate_tensorrt_engine_path(engine_path: str) -> Path:
    path = Path(str(engine_path or "").strip())
    if not path:
        raise ValueError("trt_engine_path is empty")
    if not path.exists():
        raise FileNotFoundError(f"TensorRT engine file not found: {path}")
    if path.suffix.lower() not in {".engine", ".rt"}:
        raise ValueError(f"TensorRT engine must have .engine or .rt extension: {path}")
    if path.stat().st_size < MIN_TRT_ENGINE_BYTES:
        raise ValueError(f"TensorRT engine file looks invalid or too small: {path}")
    return path


def _should_use_half(config: DetectorConfig) -> bool:
    return bool(
        config.use_half
        and str(config.device).startswith("cuda")
        and _torch_cuda_available()
    )


def _result_to_detections(result, *, class_filter: Optional[int]) -> List[DetectionPacket]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    xyxy = boxes.xyxy.detach().cpu().numpy() if getattr(boxes, "xyxy", None) is not None else np.zeros((0, 4))
    confs = boxes.conf.detach().cpu().numpy() if getattr(boxes, "conf", None) is not None else np.zeros((0,))
    classes = boxes.cls.detach().cpu().numpy() if getattr(boxes, "cls", None) is not None else np.zeros((0,))

    names_map = getattr(result, "names", {}) or {}
    out: List[DetectionPacket] = []

    for idx in range(int(xyxy.shape[0])):
        class_id = int(classes[idx]) if idx < len(classes) else 0
        if class_filter is not None and class_id != class_filter:
            continue

        x1, y1, x2, y2 = [float(v) for v in xyxy[idx].tolist()]
        score = float(confs[idx]) if idx < len(confs) else 0.0
        label = str(names_map.get(class_id, "")) if isinstance(names_map, dict) else ""

        out.append(
            DetectionPacket(
                bbox_xyxy=[x1, y1, x2, y2],
                score=score,
                class_id=class_id,
                label=label,
            )
        )

    return out


class _BaseUltralyticsAdapter:
    def __init__(self, config: DetectorConfig, *, load_path: Path) -> None:
        _require_ultralytics()
        self.config = config
        self.load_path = load_path
        self.model = YOLO(str(load_path))
        self._half = _should_use_half(config)
        self._sync_runtime_device()

    def _sync_runtime_device(self) -> None:
        try:
            if hasattr(self.model, "to"):
                self.model.to(self.config.device)
        except Exception:
            pass

        inner = getattr(self.model, "model", None)
        if torch is None or inner is None:
            return

        try:
            if hasattr(inner, "to"):
                inner.to(self.config.device)
            if hasattr(inner, "half") and self._half:
                inner.half()
            elif hasattr(inner, "float"):
                inner.float()
        except Exception:
            pass

    def warmup(self) -> None:
        dummy = np.zeros(
            (self.config.image_size, self.config.image_size, 3),
            dtype=np.uint8,
        )
        try:
            self.model.predict(
                source=dummy,
                imgsz=self.config.image_size,
                conf=self.config.confidence,
                device=self.config.device,
                verbose=False,
                max_det=self.config.max_det,
                half=self._half,
            )
        except TypeError:
            self.model.predict(
                source=dummy,
                imgsz=self.config.image_size,
                conf=self.config.confidence,
                device=self.config.device,
                verbose=False,
                max_det=self.config.max_det,
            )

    def infer(self, frame_bgr: np.ndarray) -> List[DetectionPacket]:
        frame_bgr = _validate_image(frame_bgr)

        predict_kwargs = {
            "source": frame_bgr,
            "imgsz": self.config.image_size,
            "conf": self.config.confidence,
            "device": self.config.device,
            "verbose": False,
            "max_det": self.config.max_det,
        }

        if self.config.ball_class_id is not None:
            predict_kwargs["classes"] = [self.config.ball_class_id]

        try:
            results = self.model.predict(
                **predict_kwargs,
                half=self._half,
            )
        except TypeError:
            results = self.model.predict(**predict_kwargs)

        if not results:
            return []

        return _result_to_detections(
            results[0],
            class_filter=self.config.ball_class_id,
        )

    def close(self) -> None:
        self.model = None


class UltralyticsDetectorAdapter(_BaseUltralyticsAdapter):
    def __init__(self, config: DetectorConfig) -> None:
        model_path = _validate_ultralytics_model_path(config.model_path)
        config = DetectorConfig(
            backend=config.backend,
            model_path=config.model_path,
            trt_engine_path=config.trt_engine_path,
            device=config.device,
            image_size=config.image_size,
            confidence=config.confidence,
            ball_class_id=config.ball_class_id,
            max_det=config.max_det,
            use_half=False,
        )
        super().__init__(config=config, load_path=model_path)


class TensorRtDetectorAdapter(_BaseUltralyticsAdapter):
    def __init__(self, config: DetectorConfig) -> None:
        if not str(config.device).startswith("cuda"):
            raise ValueError("TensorRT backend requires device='cuda'")
        if not _torch_cuda_available():
            raise RuntimeError("TensorRT backend requires CUDA-enabled torch runtime")

        engine_path = _validate_tensorrt_engine_path(config.trt_engine_path)
        super().__init__(config=config, load_path=engine_path)

    def warmup(self) -> None:
        if Path(self.load_path).suffix.lower() not in {".engine", ".rt"}:
            raise RuntimeError("TensorRT adapter must be loaded from .engine or .rt")
        super().warmup()


def build_detector_adapter(config: DetectorConfig) -> DetectorAdapter:
    backend = str(config.backend).strip().lower()

    if backend == "ultralytics":
        return UltralyticsDetectorAdapter(config)

    if backend == "tensorrt":
        return TensorRtDetectorAdapter(config)

    raise ValueError(f"Unsupported detector backend: {config.backend}")