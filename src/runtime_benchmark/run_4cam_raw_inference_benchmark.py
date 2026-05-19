"""4-camera RAW -> color -> YOLO benchmark runner.

Diagnostic / simulator tor — completely separate from the production
camera pipeline. No imports from gui.py, capture/*, or live_runtime/*.

Usage:
    python -m src.runtime_benchmark.run_4cam_raw_inference_benchmark \
        --source synthetic \
        --model-path models/best.pt \
        --width 2464 --height 2056 --fps 77 \
        --num-frames 1000 --batch-size 4 --imgsz 640 \
        --device cuda --half --bayer-pattern RG \
        --output-dir reports/runtime_benchmark/test_001

Run with --dry-run to skip the YOLO call entirely and time only the
generation + color-conversion path.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.runtime_benchmark._runtime_metrics import (
    GpuStatsSample,
    collect_runtime_env,
    sample_gpu_stats,
    summarize_samples,
    utc_now_iso,
    write_frames_csv,
    write_json,
)
from src.runtime_gpu.gpu_color_converter import (
    ColorConverterConfig,
    ColorConvertResult,
    GpuColorConverter,
)
from src.runtime_inference.yolo_inference_engine import (
    InferenceConfig,
    YoloInferenceEngine,
)
from src.runtime_sources.base_frame_source import CAMERA_ROLES
from src.runtime_sources.synthetic_raw_4cam_source import (
    SyntheticRaw4CamSource,
    SyntheticSourceConfig,
)

_LOG = logging.getLogger("runtime_benchmark")


_FRAME_CSV_COLUMNS: List[str] = [
    "packet_id",
    "frame_id",
    "timestamp_ns",
    "pipeline_mode",
    "color_total_ms",
    "color_upload_ms",
    "color_debayer_ms",
    "color_resize_ms",
    "color_normalize_ms",
    "color_download_ms",
    "color_output_location",
    "inference_total_ms",
    "inference_preprocess_ms",
    "inference_inference_ms",
    "inference_postprocess_ms",
    "inference_upload_ms",
    "inference_input_location",
    "zero_copy_to_inference",
    "gpu_roundtrip",
    "gpu_debayer_only_not_full_gpu_pipeline",
    "packet_total_ms",
    "packet_total_without_pacing_ms",
    "fps_instant",
    "drops_so_far",
    "detections_total",
    "detections_per_role",
    "gpu_alloc_mb",
    "gpu_reserved_mb",
    "gpu_util_pct",
]


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="4-camera RAW -> color -> YOLO benchmark for VolleyHub."
    )
    parser.add_argument(
        "--source",
        choices=("synthetic", "folder", "replay", "basler_emu"),
        default="synthetic",
    )
    parser.add_argument("--folder-path", type=Path, default=None,
                        help="Required for --source folder/replay.")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to .pt / .onnx / .engine. Required unless --dry-run.")
    parser.add_argument("--width", type=int, default=2464)
    parser.add_argument("--height", type=int, default=2056)
    parser.add_argument("--fps", type=float, default=77.0)
    parser.add_argument("--num-frames", type=int, default=1000,
                        help="Number of 4-cam packets to generate (i.e. timestamps).")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Inference batch size in number of images (multiple of 4 for batch_per_packet).")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--bayer-pattern", type=str, default="RG", choices=("RG", "BG", "GR", "GB"))
    parser.add_argument("--dtype", type=str, default="uint8", choices=("uint8", "uint16"))
    parser.add_argument("--noise-sigma", type=float, default=0.0)
    parser.add_argument("--ball", action="store_true", default=True,
                        help="Render a synthetic moving ball in generated mode (default on).")
    parser.add_argument("--no-ball", dest="ball", action="store_false")
    parser.add_argument("--color-backend", type=str, default="auto",
                        choices=("auto", "cv2_cuda", "torch_gpu", "cpu", "native_cuda_npp"))
    parser.add_argument("--inference-backend", type=str, default="auto",
                        choices=("auto", "ultralytics", "tensorrt", "dry_run"))
    parser.add_argument(
        "--pipeline-mode",
        type=str,
        default="auto",
        choices=("auto", "cpu", "gpu_roundtrip", "gpu_zero_copy"),
        help=(
            "High-level pipeline shape. cpu = numpy throughout (matches today's "
            "production debayer). gpu_roundtrip = GPU debayer but result is "
            "downloaded to numpy before YOLO (Ultralytics re-uploads). "
            "gpu_zero_copy = full GPU pipeline (debayer/resize/normalize/BCHW on "
            "CUDA, tensor stays on GPU into YOLO)."
        ),
    )
    parser.add_argument("--enforce-fps-pacing", action="store_true",
                        help="In generated mode, sleep between packets to honor --fps. "
                             "When off, the benchmark runs as fast as possible and "
                             "packet_total_ms == packet_total_without_pacing_ms.")
    parser.add_argument("--batch-mode", type=str, default="batch_per_packet",
                        choices=("batch_per_packet", "sequence_batch"))
    parser.add_argument("--sequence-len", type=int, default=1,
                        help="Number of packets to merge into one inference batch in sequence_batch mode.")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--ball-class-id", type=int, default=0)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--save-debug-images", type=int, default=0,
                        help="If > 0, save the first N color frames with bboxes overlaid.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip YOLO inference; measure generation + color conversion only.")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args(argv)


def _setup_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _build_source(args: argparse.Namespace):
    if args.source == "basler_emu":
        from src.runtime_sources.basler_emulated_source import (
            BaslerEmulatedConfig,
            BaslerEmulatedSource,
        )
        cfg = BaslerEmulatedConfig(
            width=args.width,
            height=args.height,
            fps=args.fps,
            num_frames=args.num_frames,
            bayer_pattern=args.bayer_pattern,
            dtype=args.dtype,
        )
        return BaslerEmulatedSource(cfg)
    cfg = SyntheticSourceConfig(
        mode=args.source if args.source in ("folder", "replay") else "generated",
        width=args.width,
        height=args.height,
        fps=args.fps,
        num_frames=args.num_frames,
        dtype=args.dtype,
        bayer_pattern=args.bayer_pattern,
        noise_sigma=float(args.noise_sigma),
        ball_enabled=bool(args.ball),
        folder_path=args.folder_path,
    )
    return SyntheticRaw4CamSource(cfg)


def _resolve_pipeline_mode(args: argparse.Namespace) -> str:
    if args.pipeline_mode != "auto":
        return args.pipeline_mode
    try:
        import torch
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            return "gpu_zero_copy"
    except Exception:
        pass
    return "cpu"


def _build_color_converter(args: argparse.Namespace, pipeline_mode: str) -> GpuColorConverter:
    if pipeline_mode == "cpu":
        backend = "cpu" if args.color_backend == "auto" else args.color_backend
        output_mode = "numpy_bgr"
        normalize_01 = False
        half_precision = False
    elif pipeline_mode == "gpu_roundtrip":
        backend = "torch_gpu" if args.color_backend == "auto" else args.color_backend
        output_mode = "numpy_bgr"
        normalize_01 = False
        half_precision = False
    elif pipeline_mode == "gpu_zero_copy":
        backend = "torch_gpu"  # only torch can produce a CUDA torch tensor
        output_mode = "torch_cuda_bchw_rgb_norm"
        normalize_01 = True
        half_precision = bool(args.half)
    else:
        raise SystemExit(f"unknown pipeline_mode={pipeline_mode}")
    cfg = ColorConverterConfig(
        backend=backend,
        target_image_size=int(args.imgsz),
        output_color="BGR",
        normalize_01=normalize_01,
        half_precision=half_precision,
        device=args.device,
        output_mode=output_mode,  # type: ignore[arg-type]
    )
    return GpuColorConverter(cfg)


def _build_inference_engine(args: argparse.Namespace) -> YoloInferenceEngine:
    backend = "dry_run" if args.dry_run else args.inference_backend
    cfg = InferenceConfig(
        model_path=args.model_path,
        backend=backend,
        batch_size=int(args.batch_size),
        image_size=int(args.imgsz),
        confidence=float(args.confidence),
        iou=float(args.iou),
        device=args.device,
        use_half=bool(args.half),
        warmup_iterations=int(args.warmup),
        ball_class_id=int(args.ball_class_id),
        max_det=int(args.max_det),
        dry_run=bool(args.dry_run),
    )
    return YoloInferenceEngine(cfg)


def _save_config(args: argparse.Namespace, run_dir: Path) -> None:
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    payload = {
        "started_utc": utc_now_iso(),
        "argv": sys.argv,
        "args": config,
        "env": collect_runtime_env(),
    }
    write_json(run_dir / "benchmark_config.json", payload)


def _save_debug_image(
    run_dir: Path,
    color_frame,
    detections: List[dict],
    index: int,
) -> None:
    try:
        import cv2
    except Exception:
        return
    out_dir = run_dir / "debug_images"
    out_dir.mkdir(parents=True, exist_ok=True)
    img = color_frame.image.copy()
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    for det in detections:
        x1, y1, x2, y2 = (
            int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"]),
        )
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img, f"{det['confidence']:.2f}", (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
        )
    fname = f"{index:04d}_{color_frame.role}_packet{color_frame.frame_id}.png"
    cv2.imwrite(str(out_dir / fname), img)


def _accumulate_sequence_batch(
    color_results: List[ColorConvertResult],
) -> List:
    flat = []
    for cr in color_results:
        flat.extend(cr.color_frames)
    return flat


def _instant_fps(packet_total_ms: float) -> float:
    if packet_total_ms <= 0:
        return 0.0
    return 1000.0 / packet_total_ms


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    _setup_logging(args.log_level)

    if args.batch_mode == "batch_per_packet" and args.batch_size % 4 != 0:
        _LOG.warning(
            "batch_per_packet expects batch_size to be multiple of 4 (got %d). "
            "Inference will still run but will not be aligned to packet boundaries.",
            args.batch_size,
        )
    if args.batch_mode == "sequence_batch" and args.sequence_len < 1:
        raise SystemExit("sequence_batch requires --sequence-len >= 1")

    pipeline_mode = _resolve_pipeline_mode(args)
    is_zero_copy_mode = pipeline_mode == "gpu_zero_copy"
    is_gpu_roundtrip_mode = pipeline_mode == "gpu_roundtrip"

    run_dir: Path = args.output_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    _save_config(args, run_dir)

    source = _build_source(args)
    converter = _build_color_converter(args, pipeline_mode)
    engine = _build_inference_engine(args)

    _LOG.info("pipeline_mode: %s", pipeline_mode)
    _LOG.info("source: %s", source.description)
    _LOG.info("color backend: %s", converter.backend)
    _LOG.info("inference backend: %s (dry_run=%s)", engine.backend, engine.is_dry_run)

    engine.warmup()

    rows: List[dict] = []
    color_total_samples: List[float] = []
    color_download_samples: List[float] = []
    inference_total_samples: List[float] = []
    inference_upload_samples: List[float] = []
    packet_total_samples: List[float] = []
    packet_compute_samples: List[float] = []
    drops = 0
    saved_debug = 0
    detections_total = 0
    period_s = 1.0 / float(args.fps) if args.fps > 0 else 0.0
    next_deadline = time.perf_counter()
    enforce_pacing = bool(args.enforce_fps_pacing) and period_s > 0

    pending_color_results: List[ColorConvertResult] = []
    flagged_gpu_debayer_only = False

    t_run_start = time.perf_counter()
    t_iter_top = time.perf_counter()

    try:
        for packet in source.iter_packets():
            t_packet_start = time.perf_counter()

            color_result = converter.convert_packet(packet)
            color_total_samples.append(color_result.timings.total_color_ms)
            color_download_samples.append(color_result.timings.download_from_gpu_ms)

            batch_color_results: List[ColorConvertResult] = []
            if args.batch_mode == "batch_per_packet":
                batch_color_results = [color_result]
            else:
                pending_color_results.append(color_result)
                if len(pending_color_results) >= int(args.sequence_len):
                    batch_color_results = pending_color_results
                    pending_color_results = []

            inference_ms = 0.0
            inference_upload_ms = 0.0
            inference_input_loc = "cpu"
            zero_copy = False
            gpu_debayer_only_flag = False
            packet_detections = 0
            per_role_counts = {r: 0 for r in CAMERA_ROLES}
            if batch_color_results:
                inf_t0 = time.perf_counter()
                if is_zero_copy_mode:
                    inference_input_loc = "cuda"
                    for cr in batch_color_results:
                        if cr.gpu_batch is None:
                            # Color converter could not produce a GPU tensor;
                            # treat the run as round-trip and flag honestly.
                            gpu_debayer_only_flag = True
                            flagged_gpu_debayer_only = True
                            break
                        out = engine.infer_gpu_batch(cr.gpu_batch)
                        if out.timings.gpu_debayer_only_not_full_gpu_pipeline:
                            gpu_debayer_only_flag = True
                            flagged_gpu_debayer_only = True
                        if out.timings.zero_copy_to_inference:
                            zero_copy = True
                        inference_upload_ms = max(inference_upload_ms, out.timings.upload_to_gpu_ms)
                        inference_input_loc = out.timings.input_location
                        for res in out.results:
                            n = res.num_detections
                            packet_detections += n
                            if res.role in per_role_counts:
                                per_role_counts[res.role] += n
                else:
                    flat_color = _accumulate_sequence_batch(batch_color_results)
                    for chunk_start in range(0, len(flat_color), int(args.batch_size)):
                        chunk = flat_color[chunk_start: chunk_start + int(args.batch_size)]
                        out = engine.infer_color_frames(chunk)
                        inference_upload_ms = max(inference_upload_ms, out.timings.upload_to_gpu_ms)
                        inference_input_loc = out.timings.input_location
                        for res in out.results:
                            n = res.num_detections
                            packet_detections += n
                            if res.role in per_role_counts:
                                per_role_counts[res.role] += n
                            if saved_debug < int(args.save_debug_images) and n > 0:
                                _save_debug_image(
                                    run_dir,
                                    next(f for f in chunk if f.frame_id == res.frame_id and f.role == res.role),
                                    res.detections,
                                    saved_debug,
                                )
                                saved_debug += 1
                inference_ms = (time.perf_counter() - inf_t0) * 1000.0
                inference_total_samples.append(inference_ms)
            inference_upload_samples.append(inference_upload_ms)

            t_compute_end = time.perf_counter()
            packet_compute_ms = (t_compute_end - t_packet_start) * 1000.0
            packet_compute_samples.append(packet_compute_ms)

            if enforce_pacing:
                next_deadline += period_s
                slack = next_deadline - t_compute_end
                if slack > 0:
                    time.sleep(slack)
                elif slack < -period_s:
                    drops += 1
                    next_deadline = time.perf_counter()
            else:
                if period_s > 0 and packet_compute_ms > period_s * 1000.0:
                    drops += 1

            packet_total_ms = (time.perf_counter() - t_iter_top) * 1000.0
            packet_total_samples.append(packet_total_ms)
            t_iter_top = time.perf_counter()

            detections_total += packet_detections

            color_output_loc = color_result.timings.output_location
            color_download_ms_local = color_result.timings.download_from_gpu_ms
            # round-trip is meaningful only when GPU touched the data but
            # YOLO had to consume CPU input.
            if pipeline_mode == "cpu":
                gpu_roundtrip_local = False
            elif pipeline_mode == "gpu_roundtrip":
                gpu_roundtrip_local = True
            else:  # gpu_zero_copy: round-trip only if zero-copy could not be honored
                gpu_roundtrip_local = (
                    color_output_loc != "cuda" or inference_input_loc != "cuda"
                )

            stats: GpuStatsSample = sample_gpu_stats() if (packet.packet_id % 32 == 0) else GpuStatsSample()

            rows.append(
                {
                    "packet_id": packet.packet_id,
                    "frame_id": packet.frames[0].frame_id,
                    "timestamp_ns": packet.timestamp_ns,
                    "pipeline_mode": pipeline_mode,
                    "color_total_ms": round(color_result.timings.total_color_ms, 4),
                    "color_upload_ms": round(color_result.timings.upload_to_gpu_ms, 4),
                    "color_debayer_ms": round(color_result.timings.debayer_ms, 4),
                    "color_resize_ms": round(color_result.timings.resize_ms, 4),
                    "color_normalize_ms": round(color_result.timings.normalize_ms, 4),
                    "color_download_ms": round(color_download_ms_local, 4),
                    "color_output_location": color_output_loc,
                    "inference_total_ms": round(inference_ms, 4),
                    "inference_preprocess_ms": "",
                    "inference_inference_ms": "",
                    "inference_postprocess_ms": "",
                    "inference_upload_ms": round(inference_upload_ms, 4),
                    "inference_input_location": inference_input_loc,
                    "zero_copy_to_inference": zero_copy,
                    "gpu_roundtrip": gpu_roundtrip_local,
                    "gpu_debayer_only_not_full_gpu_pipeline": gpu_debayer_only_flag,
                    "packet_total_ms": round(packet_total_ms, 4),
                    "packet_total_without_pacing_ms": round(packet_compute_ms, 4),
                    "fps_instant": round(_instant_fps(packet_total_ms), 2),
                    "drops_so_far": drops,
                    "detections_total": packet_detections,
                    "detections_per_role": json.dumps(per_role_counts),
                    "gpu_alloc_mb": round(stats.allocated_mb, 2),
                    "gpu_reserved_mb": round(stats.reserved_mb, 2),
                    "gpu_util_pct": "" if stats.utilization_pct is None else round(stats.utilization_pct, 1),
                }
            )

            if (packet.packet_id + 1) % 100 == 0:
                _LOG.info(
                    "packet %d/%s color=%.2fms inf=%.2fms pkt=%.2fms drops=%d",
                    packet.packet_id + 1,
                    source.expected_total_packets() or "?",
                    color_result.timings.total_color_ms,
                    inference_ms,
                    packet_total_ms,
                    drops,
                )
    finally:
        try:
            source.close()
        except Exception as exc:
            _LOG.warning("source.close failed: %s", exc)

    elapsed_s = time.perf_counter() - t_run_start
    summary = {
        "finished_utc": utc_now_iso(),
        "source": getattr(source, "description", str(source)),
        "pipeline_mode": pipeline_mode,
        "color_backend": converter.backend,
        "color_fallback_warnings": converter.fallback_warnings,
        "inference_backend": engine.backend,
        "is_dry_run": engine.is_dry_run,
        "gpu_debayer_only_not_full_gpu_pipeline": flagged_gpu_debayer_only,
        "packets_processed": len(rows),
        "elapsed_seconds": round(elapsed_s, 4),
        "fps_effective": round(len(rows) / elapsed_s, 2) if elapsed_s > 0 else 0.0,
        "drops": drops,
        "enforce_fps_pacing": enforce_pacing,
        "detections_total": detections_total,
        "color_ms_summary": summarize_samples(color_total_samples),
        "color_download_ms_summary": summarize_samples(color_download_samples),
        "inference_ms_summary": summarize_samples(inference_total_samples),
        "inference_upload_ms_summary": summarize_samples(inference_upload_samples),
        "packet_total_ms_summary": summarize_samples(packet_total_samples),
        "packet_total_without_pacing_ms_summary": summarize_samples(packet_compute_samples),
        "args": _args_for_summary(args),
        "env": collect_runtime_env(),
    }
    write_json(run_dir / "benchmark_summary.json", summary)
    write_frames_csv(run_dir / "benchmark_frames.csv", rows, _FRAME_CSV_COLUMNS)

    _LOG.info("=" * 60)
    _LOG.info("pipeline_mode      : %s", pipeline_mode)
    _LOG.info("packets processed  : %d", len(rows))
    _LOG.info("elapsed (s)        : %.2f", elapsed_s)
    _LOG.info("FPS effective      : %.2f", summary["fps_effective"])
    _LOG.info("avg color ms       : %.3f", summary["color_ms_summary"]["mean_ms"])
    _LOG.info("avg color download : %.3f  (>0 means CPU round-trip)", summary["color_download_ms_summary"]["mean_ms"])
    _LOG.info("avg infer upload   : %.3f  (CPU->GPU cost; ~0 means zero-copy)", summary["inference_upload_ms_summary"]["mean_ms"])
    _LOG.info("avg inference ms   : %.3f", summary["inference_ms_summary"]["mean_ms"])
    _LOG.info("avg packet (compute): %.3f", summary["packet_total_without_pacing_ms_summary"]["mean_ms"])
    _LOG.info("avg packet (total) : %.3f", summary["packet_total_ms_summary"]["mean_ms"])
    _LOG.info("drops              : %d", drops)
    if flagged_gpu_debayer_only:
        _LOG.warning("gpu_debayer_only_not_full_gpu_pipeline=TRUE — color was on GPU "
                     "but inference round-tripped to CPU. Decision metric is "
                     "packet_total_without_pacing_ms, not color_total_ms alone.")
    _LOG.info("output dir         : %s", run_dir)
    _LOG.info("=" * 60)
    return 0


def _args_for_summary(args: argparse.Namespace) -> dict:
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}


if __name__ == "__main__":
    raise SystemExit(main())
