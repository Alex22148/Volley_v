# file: vision_runtime/core/pipeline_runner.py
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from vision_runtime.adapters.debayer_adapter import DebayerConfig, build_debayer_adapter
from vision_runtime.adapters.detector_adapter import DetectorConfig, build_detector_adapter
from vision_runtime.adapters.image_source import FolderImageSource
from vision_runtime.core.preprocess import PreprocessConfig, minimal_preprocess


@dataclass(slots=True)
class RunnerConfig:
    input_dir: Path
    output_dir: Path
    detector_backend: str = "ultralytics"
    model_path: str = "best.pt"
    trt_engine_path: str = ""
    device: str = "cuda"
    image_size: int = 640
    confidence: float = 0.25
    debayer_backend: str = "cpu_opencv"
    bayer_pattern: str = "BG"
    save_overlays: bool = True
    save_json: bool = True


class OfflinePipelineRunner:
    def __init__(self, config: RunnerConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.overlay_dir = self.output_dir / "overlays"
        self.json_dir = self.output_dir / "json"

    def run(self) -> dict:
        self._ensure_dirs()

        source = FolderImageSource(Path(self.config.input_dir))
        files = source.list_files()
        if not files:
            raise FileNotFoundError(f"No images found in: {self.config.input_dir}")

        detector = build_detector_adapter(
            DetectorConfig(
                backend=str(self.config.detector_backend),
                model_path=str(self.config.model_path),
                trt_engine_path=str(self.config.trt_engine_path),
                device=str(self.config.device),
                image_size=int(self.config.image_size),
                confidence=float(self.config.confidence),
            )
        )
        debayer = build_debayer_adapter(
            DebayerConfig(
                backend=str(self.config.debayer_backend),  # type: ignore[arg-type]
                bayer_pattern=str(self.config.bayer_pattern).upper(),  # type: ignore[arg-type]
                output_format="BGR",
            )
        )
        preprocess_cfg = PreprocessConfig(
            image_size=int(self.config.image_size),
            input_color="BGR",
            normalize_01=False,
        )

        summary: Dict[str, Any] = {
            "config": {
                "input_dir": str(self.config.input_dir),
                "output_dir": str(self.config.output_dir),
                "detector_backend": str(self.config.detector_backend),
                "model_path": str(self.config.model_path),
                "trt_engine_path": str(self.config.trt_engine_path),
                "device": str(self.config.device),
                "image_size": int(self.config.image_size),
                "confidence": float(self.config.confidence),
                "debayer_backend": str(self.config.debayer_backend),
                "bayer_pattern": str(self.config.bayer_pattern),
                "save_overlays": bool(self.config.save_overlays),
                "save_json": bool(self.config.save_json),
            },
            "frames_total": 0,
            "frames_processed": 0,
            "detections_total": 0,
            "tracks_total": 0,
            "avg_load_ms": 0.0,
            "avg_convert_ms": 0.0,
            "avg_preprocess_ms": 0.0,
            "avg_infer_ms": 0.0,
            "avg_track_ms": 0.0,
            "avg_total_ms": 0.0,
            "per_frame": [],
        }

        load_samples: List[float] = []
        convert_samples: List[float] = []
        preprocess_samples: List[float] = []
        infer_samples: List[float] = []
        track_samples: List[float] = []
        total_samples: List[float] = []

        try:
            detector.warmup()

            for item in source.iter_items():
                t0 = time.perf_counter()

                load_t0 = time.perf_counter()
                image = cv2.imread(str(item.path), cv2.IMREAD_UNCHANGED)
                load_ms = (time.perf_counter() - load_t0) * 1000.0

                if image is None:
                    summary["per_frame"].append(
                        {
                            "frame_id": int(item.frame_id),
                            "source_path": str(item.path),
                            "error": "cv2.imread returned None",
                        }
                    )
                    continue

                convert_t0 = time.perf_counter()
                frame_bgr = self._convert_input_to_bgr(image, debayer)
                convert_ms = (time.perf_counter() - convert_t0) * 1000.0

                preprocess_t0 = time.perf_counter()
                preprocessed, preprocess_meta = minimal_preprocess(frame_bgr, preprocess_cfg)
                preprocess_ms = (time.perf_counter() - preprocess_t0) * 1000.0

                infer_t0 = time.perf_counter()
                detections = detector.infer(self._to_uint8_bgr(preprocessed, preprocess_cfg))
                infer_ms = (time.perf_counter() - infer_t0) * 1000.0

                track_t0 = time.perf_counter()
                tracks: List[dict] = []
                track_ms = (time.perf_counter() - track_t0) * 1000.0

                total_ms = (time.perf_counter() - t0) * 1000.0

                frame_record = {
                    "frame_id": int(item.frame_id),
                    "source_path": str(item.path),
                    "timestamp_host_ns": int(item.timestamp_host_ns),
                    "detections_count": int(len(detections)),
                    "tracks_count": int(len(tracks)),
                    "stage_ms": {
                        "load_ms": float(load_ms),
                        "convert_ms": float(convert_ms),
                        "preprocess_ms": float(preprocess_ms),
                        "infer_ms": float(infer_ms),
                        "track_ms": float(track_ms),
                        "total_ms": float(total_ms),
                    },
                    "preprocess_meta": preprocess_meta,
                    "detections": [asdict(d) for d in detections],
                    "tracks": tracks,
                }
                summary["per_frame"].append(frame_record)

                if self.config.save_overlays:
                    overlay = self._draw_overlay(frame_bgr, detections)
                    out_name = f"{item.frame_id:06d}_{item.path.stem}.jpg"
                    cv2.imwrite(str(self.overlay_dir / out_name), overlay)

                load_samples.append(load_ms)
                convert_samples.append(convert_ms)
                preprocess_samples.append(preprocess_ms)
                infer_samples.append(infer_ms)
                track_samples.append(track_ms)
                total_samples.append(total_ms)

                summary["frames_processed"] += 1
                summary["detections_total"] += len(detections)
                summary["tracks_total"] += len(tracks)

        finally:
            try:
                detector.close()
            except Exception:
                pass
            try:
                debayer.close()
            except Exception:
                pass

        summary["frames_total"] = len(files)
        summary["avg_load_ms"] = self._avg(load_samples)
        summary["avg_convert_ms"] = self._avg(convert_samples)
        summary["avg_preprocess_ms"] = self._avg(preprocess_samples)
        summary["avg_infer_ms"] = self._avg(infer_samples)
        summary["avg_track_ms"] = self._avg(track_samples)
        summary["avg_total_ms"] = self._avg(total_samples)

        if self.config.save_json:
            summary_path = self.json_dir / "summary.json"
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        return summary

    def _ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.config.save_overlays:
            self.overlay_dir.mkdir(parents=True, exist_ok=True)
        if self.config.save_json:
            self.json_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _avg(values: List[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    @staticmethod
    def _convert_input_to_bgr(image: np.ndarray, debayer) -> np.ndarray:
        if image.ndim == 2:
            return debayer.debayer(image)

        if image.ndim == 3 and image.shape[2] == 3:
            if image.dtype != np.uint8:
                image = image.astype(np.uint8, copy=False)
            if not image.flags["C_CONTIGUOUS"]:
                image = np.ascontiguousarray(image)
            return image

        if image.ndim == 3 and image.shape[2] == 4:
            bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            if not bgr.flags["C_CONTIGUOUS"]:
                bgr = np.ascontiguousarray(bgr)
            return bgr

        raise ValueError(f"Unsupported input image shape={image.shape}, dtype={image.dtype}")

    @staticmethod
    def _to_uint8_bgr(image: np.ndarray, preprocess_cfg: PreprocessConfig) -> np.ndarray:
        if image.dtype == np.uint8:
            out = image
        elif np.issubdtype(image.dtype, np.floating):
            out = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            out = image.astype(np.uint8, copy=False)

        if preprocess_cfg.input_color == "RGB":
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

        if not out.flags["C_CONTIGUOUS"]:
            out = np.ascontiguousarray(out)
        return out

    @staticmethod
    def _draw_overlay(frame_bgr: np.ndarray, detections) -> np.ndarray:
        out = frame_bgr.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(round(v)) for v in det.bbox_xyxy]
            score = float(det.score)
            label = det.label or str(det.class_id)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                out,
                f"{label} {score:.2f}",
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        return out