"""Programmatic API for runtime studies.

Two studies are exposed:

    A) run_max_resolution_study(target_ms, candidates, cfg)
       Sweep a list of resolutions and find the largest one whose median
       compute time fits within the requested budget (e.g. 20 ms / 50 fps).

    B) run_max_fps_study(width, height, cfg)
       Measure compute time at a fixed resolution and report the maximum
       FPS the system can sustain (median-based "aggressive" and p95-based
       "safe" estimates).

Both are thin wrappers around `run_micro_benchmark`, which builds the
benchmark pipeline (source + color + inference) and returns aggregated
metrics. This module is intentionally free of argparse and Tkinter so it
can be reused by both the CLI and the GUI.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from src.runtime_benchmark._runtime_metrics import summarize_samples
from src.runtime_gpu.gpu_color_converter import (
    ColorConverterConfig,
    GpuColorConverter,
)
from src.runtime_inference.yolo_inference_engine import (
    InferenceConfig,
    YoloInferenceEngine,
)
from src.runtime_sources.synthetic_raw_4cam_source import (
    SyntheticRaw4CamSource,
    SyntheticSourceConfig,
)

_LOG = logging.getLogger(__name__)

ProgressFn = Callable[[str], None]


def _noop(_msg: str) -> None:
    return None


@dataclass(slots=True)
class StudyConfig:
    """Knobs shared by every study run."""

    pipeline_mode: str = "auto"          # auto | cpu | gpu_roundtrip | gpu_zero_copy
    color_backend: str = "auto"          # auto | cv2_cuda | torch_gpu | cpu | native_cuda_npp
    inference_backend: str = "auto"      # auto | ultralytics | tensorrt | dry_run
    bayer_pattern: str = "RG"
    dtype: str = "uint8"
    # int (square) or (H, W) tuple/list for rectangular inference shapes.
    imgsz: object = 640
    batch_size: int = 4
    device: str = "cuda"
    half: bool = True
    num_frames: int = 200
    warmup_iterations: int = 10
    model_path: Optional[str] = None
    dry_run: bool = True
    confidence: float = 0.25
    iou: float = 0.45
    ball_class_id: Optional[int] = 0
    max_det: int = 300
    seed: int = 1234

    def validate(self) -> None:
        if self.pipeline_mode not in ("auto", "cpu", "gpu_roundtrip", "gpu_zero_copy"):
            raise ValueError(f"unknown pipeline_mode={self.pipeline_mode}")
        ish = self.imgsz
        if isinstance(ish, (tuple, list)):
            if len(ish) != 2 or int(ish[0]) <= 0 or int(ish[1]) <= 0:
                raise ValueError(f"imgsz tuple must be (H, W) > 0; got {ish}")
        else:
            if int(ish) <= 0:  # type: ignore[arg-type]
                raise ValueError("imgsz must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.num_frames <= 0:
            raise ValueError("num_frames must be > 0")
        if self.bayer_pattern not in ("RG", "BG", "GR", "GB"):
            raise ValueError(f"unknown bayer pattern: {self.bayer_pattern}")


@dataclass(slots=True)
class MicroBenchmarkResult:
    width: int
    height: int
    pipeline_mode: str
    color_backend: str
    inference_backend: str
    color_output_location: str
    inference_input_location: str
    zero_copy_to_inference: bool
    gpu_debayer_only_not_full_gpu_pipeline: bool
    samples: int
    median_compute_ms: float
    p95_compute_ms: float
    mean_compute_ms: float
    max_compute_ms: float
    median_color_ms: float
    median_inference_ms: float
    median_color_download_ms: float
    median_inference_upload_ms: float
    max_fps_from_median: float
    max_fps_from_p95: float
    fallback_warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class MaxResolutionStudyResult:
    target_ms: float
    target_fps: float
    candidates: List[MicroBenchmarkResult]
    chosen: Optional[MicroBenchmarkResult]
    pipeline_mode: str

    def to_dict(self) -> dict:
        return {
            "target_ms": self.target_ms,
            "target_fps": self.target_fps,
            "pipeline_mode": self.pipeline_mode,
            "candidates": [c.to_dict() for c in self.candidates],
            "chosen": self.chosen.to_dict() if self.chosen else None,
        }


@dataclass(slots=True)
class MaxFpsStudyResult:
    width: int
    height: int
    measurement: MicroBenchmarkResult
    suggested_safe_fps: int
    suggested_aggressive_fps: int

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "suggested_safe_fps": self.suggested_safe_fps,
            "suggested_aggressive_fps": self.suggested_aggressive_fps,
            "measurement": self.measurement.to_dict(),
        }


# Reasonable Bayer-friendly resolutions up to the camera's native 2464x2056.
DEFAULT_CANDIDATE_RESOLUTIONS: List[Tuple[int, int]] = [
    (640, 480),
    (800, 600),
    (1024, 768),
    (1280, 720),
    (1280, 960),
    (1600, 1200),
    (1920, 1080),
    (1920, 1440),
    (2048, 1536),
    (2304, 1728),
    (2464, 2056),
]


def _build_color_config(cfg: StudyConfig) -> ColorConverterConfig:
    if cfg.pipeline_mode == "gpu_zero_copy":
        # Explicit color_backend override wins (e.g. native_cuda_npp);
        # otherwise default to torch_gpu — the only backend that auto
        # produced CUDA tensors before native was added.
        if cfg.color_backend in ("torch_gpu", "native_cuda_npp"):
            backend = cfg.color_backend
        else:
            backend = "torch_gpu"
        output_mode = "torch_cuda_bchw_rgb_norm"
        normalize_01 = True
        half_precision = bool(cfg.half)
    elif cfg.pipeline_mode == "gpu_roundtrip":
        backend = "torch_gpu" if cfg.color_backend == "auto" else cfg.color_backend
        output_mode = "numpy_bgr"
        normalize_01 = False
        half_precision = False
    elif cfg.pipeline_mode == "cpu":
        backend = "cpu" if cfg.color_backend == "auto" else cfg.color_backend
        output_mode = "numpy_bgr"
        normalize_01 = False
        half_precision = False
    else:  # auto
        try:
            import torch
            if torch.cuda.is_available() and cfg.device.startswith("cuda"):
                backend = "torch_gpu"
                output_mode = "torch_cuda_bchw_rgb_norm"
                normalize_01 = True
                half_precision = bool(cfg.half)
            else:
                backend = "cpu"
                output_mode = "numpy_bgr"
                normalize_01 = False
                half_precision = False
        except Exception:
            backend = "cpu"
            output_mode = "numpy_bgr"
            normalize_01 = False
            half_precision = False
    return ColorConverterConfig(
        backend=backend,
        target_image_size=cfg.imgsz,  # int or (H, W) tuple
        output_color="BGR",
        normalize_01=normalize_01,
        half_precision=half_precision,
        device=cfg.device,
        output_mode=output_mode,  # type: ignore[arg-type]
    )


def _build_inference_config(cfg: StudyConfig) -> InferenceConfig:
    backend = "dry_run" if cfg.dry_run else cfg.inference_backend
    return InferenceConfig(
        model_path=cfg.model_path,
        backend=backend,  # type: ignore[arg-type]
        batch_size=int(cfg.batch_size),
        image_size=cfg.imgsz,  # int or (H, W) tuple
        confidence=float(cfg.confidence),
        iou=float(cfg.iou),
        device=cfg.device,
        use_half=bool(cfg.half),
        warmup_iterations=int(cfg.warmup_iterations),
        ball_class_id=cfg.ball_class_id,
        max_det=int(cfg.max_det),
        dry_run=bool(cfg.dry_run),
    )


def run_micro_benchmark(
    width: int,
    height: int,
    cfg: StudyConfig,
    progress: Optional[ProgressFn] = None,
) -> MicroBenchmarkResult:
    """Run one short benchmark at (width, height) and return aggregated metrics.

    Frames generated are synthetic (no camera required). The function
    handles its own warmup and only reports timings from the measurement
    phase.
    """
    cfg.validate()
    p = progress or _noop

    src_cfg = SyntheticSourceConfig(
        mode="generated",
        width=int(width),
        height=int(height),
        fps=120.0,  # source itself never paces; benchmark loop runs flat-out
        num_frames=int(cfg.num_frames + cfg.warmup_iterations),
        dtype=cfg.dtype,
        bayer_pattern=cfg.bayer_pattern,
        ball_enabled=True,
        seed=int(cfg.seed),
    )
    try:
        source = SyntheticRaw4CamSource(src_cfg)
    except Exception as exc:
        return _error_result(width, height, cfg, str(exc))

    try:
        color_cfg = _build_color_config(cfg)
        converter = GpuColorConverter(color_cfg)
    except Exception as exc:
        return _error_result(width, height, cfg, f"color converter init: {exc}")

    try:
        inf_cfg = _build_inference_config(cfg)
        engine = YoloInferenceEngine(inf_cfg)
    except Exception as exc:
        return _error_result(width, height, cfg, f"inference engine init: {exc}")

    try:
        engine.warmup()
    except Exception as exc:
        _LOG.warning("warmup failed: %s", exc)

    is_zero_copy = (color_cfg.output_mode == "torch_cuda_bchw_rgb_norm")

    compute_samples: List[float] = []
    color_samples: List[float] = []
    inference_samples: List[float] = []
    color_download_samples: List[float] = []
    inference_upload_samples: List[float] = []
    color_output_loc = "cpu"
    inference_input_loc = "cpu"
    zero_copy_observed = False
    gpu_debayer_only_flag = False

    p(f"  measuring {width}x{height} ({cfg.num_frames} packets, {cfg.warmup_iterations} warmup)...")

    n_total = cfg.num_frames + cfg.warmup_iterations
    idx = 0
    try:
        for packet in source.iter_packets():
            t0 = time.perf_counter()
            color_result = converter.convert_packet(packet)
            color_output_loc = color_result.timings.output_location

            inference_ms = 0.0
            inference_upload_ms = 0.0
            if is_zero_copy and color_result.gpu_batch is not None:
                out = engine.infer_gpu_batch(color_result.gpu_batch)
                inference_input_loc = out.timings.input_location
                if out.timings.zero_copy_to_inference:
                    zero_copy_observed = True
                if out.timings.gpu_debayer_only_not_full_gpu_pipeline:
                    gpu_debayer_only_flag = True
                inference_upload_ms = out.timings.upload_to_gpu_ms
            else:
                # numpy path: cpu or gpu_roundtrip
                out = engine.infer_color_frames(list(color_result.color_frames))
                inference_input_loc = out.timings.input_location
                inference_upload_ms = out.timings.upload_to_gpu_ms
                if is_zero_copy and color_result.gpu_batch is None:
                    gpu_debayer_only_flag = True
            inference_ms = out.timings.total_inference_ms
            compute_ms = (time.perf_counter() - t0) * 1000.0

            if idx >= cfg.warmup_iterations:
                compute_samples.append(compute_ms)
                color_samples.append(color_result.timings.total_color_ms)
                inference_samples.append(inference_ms)
                color_download_samples.append(color_result.timings.download_from_gpu_ms)
                inference_upload_samples.append(inference_upload_ms)
            idx += 1
            if idx >= n_total:
                break
    except Exception as exc:
        return _error_result(width, height, cfg, f"during measurement: {exc}")
    finally:
        try:
            source.close()
        except Exception:
            pass

    if not compute_samples:
        return _error_result(width, height, cfg, "no compute samples collected")

    cs = summarize_samples(compute_samples)
    color_s = summarize_samples(color_samples)
    inf_s = summarize_samples(inference_samples)
    cd_s = summarize_samples(color_download_samples)
    iu_s = summarize_samples(inference_upload_samples)

    median_ms = float(cs["median_ms"]) or 0.0001
    p95_ms = float(cs["p95_ms"]) or 0.0001

    return MicroBenchmarkResult(
        width=int(width),
        height=int(height),
        pipeline_mode=cfg.pipeline_mode,
        color_backend=converter.backend,
        inference_backend=engine.backend,
        color_output_location=color_output_loc,
        inference_input_location=inference_input_loc,
        zero_copy_to_inference=zero_copy_observed,
        gpu_debayer_only_not_full_gpu_pipeline=gpu_debayer_only_flag,
        samples=int(cs["count"]),
        median_compute_ms=median_ms,
        p95_compute_ms=p95_ms,
        mean_compute_ms=float(cs["mean_ms"]),
        max_compute_ms=float(cs["max_ms"]),
        median_color_ms=float(color_s["median_ms"]),
        median_inference_ms=float(inf_s["median_ms"]),
        median_color_download_ms=float(cd_s["median_ms"]),
        median_inference_upload_ms=float(iu_s["median_ms"]),
        max_fps_from_median=1000.0 / median_ms,
        max_fps_from_p95=1000.0 / p95_ms,
        fallback_warnings=list(converter.fallback_warnings),
    )


def _error_result(width: int, height: int, cfg: StudyConfig, msg: str) -> MicroBenchmarkResult:
    return MicroBenchmarkResult(
        width=int(width),
        height=int(height),
        pipeline_mode=cfg.pipeline_mode,
        color_backend="",
        inference_backend="",
        color_output_location="",
        inference_input_location="",
        zero_copy_to_inference=False,
        gpu_debayer_only_not_full_gpu_pipeline=False,
        samples=0,
        median_compute_ms=float("inf"),
        p95_compute_ms=float("inf"),
        mean_compute_ms=float("inf"),
        max_compute_ms=float("inf"),
        median_color_ms=0.0,
        median_inference_ms=0.0,
        median_color_download_ms=0.0,
        median_inference_upload_ms=0.0,
        max_fps_from_median=0.0,
        max_fps_from_p95=0.0,
        error=msg,
    )


def run_max_resolution_study(
    target_ms: float,
    candidates: Sequence[Tuple[int, int]],
    cfg: StudyConfig,
    progress: Optional[ProgressFn] = None,
) -> MaxResolutionStudyResult:
    """Sweep `candidates` and return the largest WxH whose median compute
    time fits within `target_ms`.
    """
    if target_ms <= 0:
        raise ValueError("target_ms must be > 0")
    if not candidates:
        raise ValueError("candidates must be non-empty")

    p = progress or _noop
    target_fps = 1000.0 / float(target_ms)
    p(f"Study A: max resolution at <= {target_ms:.2f} ms ({target_fps:.1f} fps)")
    p(f"  pipeline_mode={cfg.pipeline_mode} imgsz={cfg.imgsz} batch={cfg.batch_size} half={cfg.half}")

    results: List[MicroBenchmarkResult] = []
    for (w, h) in sorted(candidates, key=lambda x: x[0] * x[1]):
        result = run_micro_benchmark(w, h, cfg, progress=p)
        results.append(result)
        if result.error:
            p(f"  {w}x{h}  ERROR: {result.error}")
            continue
        verdict = "PASS" if result.median_compute_ms <= target_ms else "FAIL"
        p(
            f"  {w}x{h:>5}  median={result.median_compute_ms:7.2f} ms "
            f"p95={result.p95_compute_ms:7.2f} ms  "
            f"max_fps_med={result.max_fps_from_median:6.1f}  "
            f"col_loc={result.color_output_location:>4}  "
            f"inf_loc={result.inference_input_location:>4}  "
            f"[{verdict}]"
        )

    passing = [r for r in results if not r.error and r.median_compute_ms <= target_ms]
    chosen = max(passing, key=lambda r: r.width * r.height) if passing else None
    if chosen:
        p(f"  -> chosen: {chosen.width}x{chosen.height} "
          f"(median {chosen.median_compute_ms:.2f} ms, "
          f"max FPS @ median = {chosen.max_fps_from_median:.1f})")
    else:
        p("  -> no candidate fits the budget")
    return MaxResolutionStudyResult(
        target_ms=float(target_ms),
        target_fps=float(target_fps),
        candidates=results,
        chosen=chosen,
        pipeline_mode=cfg.pipeline_mode,
    )


def run_max_fps_study(
    width: int,
    height: int,
    cfg: StudyConfig,
    progress: Optional[ProgressFn] = None,
) -> MaxFpsStudyResult:
    """Measure compute time at fixed (width, height) and return max FPS."""
    p = progress or _noop
    p(f"Study B: max FPS at {width}x{height}")
    p(f"  pipeline_mode={cfg.pipeline_mode} imgsz={cfg.imgsz} batch={cfg.batch_size} half={cfg.half}")
    result = run_micro_benchmark(width, height, cfg, progress=p)
    if result.error:
        p(f"  ERROR: {result.error}")
        return MaxFpsStudyResult(
            width=int(width),
            height=int(height),
            measurement=result,
            suggested_safe_fps=0,
            suggested_aggressive_fps=0,
        )
    safe_fps = int(np.floor(result.max_fps_from_p95))
    aggressive_fps = int(np.floor(result.max_fps_from_median))
    p(f"  median compute = {result.median_compute_ms:.2f} ms -> aggressive max FPS = {aggressive_fps}")
    p(f"  p95    compute = {result.p95_compute_ms:.2f} ms -> safe       max FPS = {safe_fps}")
    p(f"  color_loc={result.color_output_location} inf_loc={result.inference_input_location} "
      f"zero_copy={result.zero_copy_to_inference} "
      f"warn_round_trip={result.gpu_debayer_only_not_full_gpu_pipeline}")
    return MaxFpsStudyResult(
        width=int(width),
        height=int(height),
        measurement=result,
        suggested_safe_fps=safe_fps,
        suggested_aggressive_fps=aggressive_fps,
    )
