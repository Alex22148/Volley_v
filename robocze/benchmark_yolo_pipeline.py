#!/usr/bin/env python3
"""
Isolated / engine-only benchmark for VolleyHub.

Mode:
- isolated (no live queues, no router, no camera pipeline)
"""

import argparse
import random
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from benchmarking.casegen import build_case_grid
from benchmarking.io_utils import (
    collect_runtime_env,
    detect_git_commit,
    prepare_run_directory,
    utc_now_iso,
    write_json,
    write_summary_csv,
)
from benchmarking.stats import downsample, slope_per_step, summarize
from vision.yolo_module import VolleyHubYOLO


ROLES = ["CENTER_L", "CENTER_R", "LEFT", "RIGHT"]


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def list_images(images_dir: Path, extensions: Iterable[str]) -> list[Path]:
    exts = {e.lower().strip() for e in extensions}
    files = []
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    files.sort()
    return files


def scale_scene_keep_canvas(frame_bgr: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-6:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    sw = max(1, int(round(w * scale)))
    sh = max(1, int(round(h * scale)))
    small = cv2.resize(frame_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros_like(frame_bgr)
    x0 = (w - sw) // 2
    y0 = (h - sh) // 2
    canvas[y0 : y0 + sh, x0 : x0 + sw] = small
    return canvas


def bgr_to_bayer_rg8(frame_bgr: np.ndarray) -> np.ndarray:
    b = frame_bgr[..., 0]
    g = frame_bgr[..., 1]
    r = frame_bgr[..., 2]
    out = np.empty(frame_bgr.shape[:2], dtype=np.uint8)
    out[0::2, 0::2] = r[0::2, 0::2]
    out[0::2, 1::2] = g[0::2, 1::2]
    out[1::2, 0::2] = g[1::2, 0::2]
    out[1::2, 1::2] = b[1::2, 1::2]
    return out


def pick_frame(pool: list[np.ndarray], idx: int) -> np.ndarray:
    return pool[idx % len(pool)]


def run_single_case(
    yolo: VolleyHubYOLO,
    frame_pool: list[np.ndarray],
    seq_len: int,
    repeats: int,
    warmup: int,
) -> tuple[dict, dict]:
    total_ms_samples: list[float] = []
    prepare_ms_samples: list[float] = []
    forward_ms_samples: list[float] = []
    post_ms_samples: list[float] = []
    conf_samples: list[float] = []
    detections_total = 0
    detected_frames = 0

    images_per_call = int(seq_len * len(ROLES))
    total_frames = int(repeats * images_per_call)

    cursor = 0
    for i in range(warmup + repeats):
        frames_by_step = []
        for _ in range(seq_len):
            step_frames = []
            for _ in ROLES:
                step_frames.append(pick_frame(frame_pool, cursor))
                cursor += 1
            frames_by_step.append(step_frames)

        t0 = time.perf_counter()
        out = yolo.detect_ball_in_batch(frames_by_step, ROLES)
        total_ms = (time.perf_counter() - t0) * 1000.0

        if i < warmup:
            continue

        meta = out.get("meta", {})
        det_steps = out.get("detections", [])

        total_ms_samples.append(float(total_ms))
        prepare_ms_samples.append(float(meta.get("prepare_ms", 0.0) or 0.0))
        forward_ms_samples.append(float(meta.get("forward_ms", 0.0) or 0.0))
        post_ms_samples.append(float(meta.get("post_ms", 0.0) or 0.0))

        for step in det_steps:
            for role in ROLES:
                dets = step.get(role, [])
                if dets:
                    detected_frames += 1
                    detections_total += len(dets)
                    conf_samples.append(max(float(d.get("confidence", 0.0) or 0.0) for d in dets))

    total_s = summarize(total_ms_samples)
    prepare_s = summarize(prepare_ms_samples)
    forward_s = summarize(forward_ms_samples)
    post_s = summarize(post_ms_samples)
    mean_total = float(total_s["mean"])

    fps_calls = (1000.0 / mean_total) if mean_total > 0 else 0.0
    fps_images = (images_per_call * 1000.0 / mean_total) if mean_total > 0 else 0.0
    detection_rate = (detected_frames / total_frames) if total_frames > 0 else 0.0
    mean_conf = float(np.mean(conf_samples)) if conf_samples else 0.0

    aggregates = {
        "total_ms_mean": float(total_s["mean"]),
        "total_ms_p50": float(total_s["p50"]),
        "total_ms_p95": float(total_s["p95"]),
        "total_ms_p99": float(total_s["p99"]),
        "total_ms_std": float(total_s["std"]),
        "total_ms_cv": float(total_s["cv"]),
        "total_ms_slope": float(slope_per_step(total_ms_samples)),
        "prepare_ms_mean": float(prepare_s["mean"]),
        "forward_ms_mean": float(forward_s["mean"]),
        "post_ms_mean": float(post_s["mean"]),
        "fps_calls": float(fps_calls),
        "fps_images": float(fps_images),
        "detections_total": int(detections_total),
        "detection_rate_frames": float(detection_rate),
        "mean_max_conf": float(mean_conf),
    }
    samples = {
        "total_ms_samples": total_ms_samples,
        "prepare_ms_samples": prepare_ms_samples,
        "forward_ms_samples": forward_ms_samples,
        "post_ms_samples": post_ms_samples,
    }
    return aggregates, samples


def main() -> int:
    parser = argparse.ArgumentParser(description="VolleyHub isolated benchmark")
    parser.add_argument("--images-dir", type=str, required=True, help="Folder ze zdjÄ™ciami testowymi")
    parser.add_argument("--model", type=str, default="best.pt", help="ĹšcieĹĽka do modelu (.pt/.onnx/.engine)")
    parser.add_argument("--backend", type=str, default="ultralytics", choices=["ultralytics", "tensorrt"])
    parser.add_argument("--trt-engine", type=str, default="", help="Path to .engine for backend=tensorrt")
    parser.add_argument("--trt-dynamic", action="store_true")
    parser.add_argument("--trt-workspace-gb", type=float, default=2.0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--input-format", type=str, default="bgr", choices=["bgr", "synthetic_bayer_rg8"])
    parser.add_argument("--batch-sizes", type=str, default="8")
    parser.add_argument("--seq-lens", type=str, default="1,2,3,4")
    parser.add_argument("--object-scales", type=str, default="1.0,0.75,0.5,0.35,0.25")
    parser.add_argument("--image-sizes", type=str, default="640")
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--max-images", type=int, default=120)
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--ball-class-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--exts", type=str, default=".jpg,.jpeg,.png,.bmp")
    parser.add_argument("--samples-max-points", type=int, default=400)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    images_dir = Path(args.images_dir).resolve()
    if not images_dir.exists():
        raise FileNotFoundError(f"Brak katalogu: {images_dir}")

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = (Path(__file__).resolve().parent / model_path).resolve()

    batch_sizes = parse_int_list(args.batch_sizes)
    seq_lens = parse_int_list(args.seq_lens)
    object_scales = parse_float_list(args.object_scales)
    image_sizes = parse_int_list(args.image_sizes)
    exts = [x.strip() for x in args.exts.split(",") if x.strip()]

    run_base = Path(args.out_dir).resolve() if args.out_dir else Path(__file__).resolve().parent
    run_paths = prepare_run_directory(run_base, mode="isolated")
    start_ts = utc_now_iso()

    files = list_images(images_dir, exts)
    if not files:
        raise RuntimeError(f"Brak obrazĂłw wejĹ›ciowych w: {images_dir}")
    if args.max_images > 0:
        files = files[: args.max_images]

    base_frames = []
    for p in files:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is not None:
            base_frames.append(img)
    if not base_frames:
        raise RuntimeError("Nie udaĹ‚o się wczytaÄ‡ ĹĽadnego obrazu.")

    yolo = VolleyHubYOLO()
    if not yolo.set_backend(args.backend):
        raise RuntimeError(f"Nie udaĹ‚o się ustawiÄ‡ backend={args.backend}")
    yolo.configure_tensorrt(dynamic=args.trt_dynamic, workspace_gb=args.trt_workspace_gb)
    if args.trt_engine:
        if not yolo.set_tensorrt_engine(args.trt_engine):
            raise RuntimeError(f"Nie udaĹ‚o się ustawiÄ‡ TensorRT engine: {args.trt_engine}")
    if args.device != "auto":
        if not yolo.set_device(args.device):
            raise RuntimeError(f"Nie udaĹ‚o się ustawiÄ‡ device={args.device}")
    yolo.set_confidence_threshold(args.conf_threshold)
    yolo.ball_class_id = int(args.ball_class_id)
    if not yolo.load_custom_model(str(model_path)):
        raise RuntimeError("Nie udaĹ‚o się zaĹ‚adowaÄ‡ modelu.")
    if not yolo.enable():
        raise RuntimeError("Nie udaĹ‚o się wĹ‚Ä…czyÄ‡ YOLO.")

    grid = {
        "models": [str(model_path)],
        "image_sizes": image_sizes,
        "chunk_batch_sizes": batch_sizes,
        "seq_lens": seq_lens,
        "object_scales": object_scales,
    }
    cases = build_case_grid(grid)

    summary_rows: list[dict] = []
    cases_index: list[dict] = []

    for i, case in enumerate(cases, start=1):
        case_id = f"case_{i:04d}"
        imgsz = int(case["imgsz"])
        seq_len = int(case["seq_len"])
        chunk_bs = int(case["chunk_batch_size"])
        object_scale = float(case["object_scale"])

        yolo.image_size = imgsz
        yolo.set_batch_size(chunk_bs)

        frame_pool = [scale_scene_keep_canvas(fr, object_scale) for fr in base_frames]
        if args.input_format == "synthetic_bayer_rg8":
            frame_pool = [bgr_to_bayer_rg8(fr) for fr in frame_pool]

        print(
            f"[BENCH][isolated] {case_id} "
            f"imgsz={imgsz} chunk_bs={chunk_bs} seq_len={seq_len} scale={object_scale:.3f}"
        )
        aggregates, samples = run_single_case(
            yolo=yolo,
            frame_pool=frame_pool,
            seq_len=seq_len,
            repeats=args.repeats,
            warmup=args.warmup,
        )

        images_per_call = int(seq_len * len(ROLES))
        row = {
            "case_id": case_id,
            "benchmark_mode": "isolated",
            "backend": str(getattr(yolo, "inference_backend", "ultralytics")),
            "model_path": str(model_path),
            "trt_engine_path": str(getattr(yolo, "trt_engine_path", None) or ""),
            "device": str(getattr(yolo, "device", "cpu")),
            "input_format": str(args.input_format),
            "imgsz": int(imgsz),
            "seq_len": int(seq_len),
            "chunk_batch_size": int(chunk_bs),
            "images_per_call": int(images_per_call),
            "object_scale": float(object_scale),
            "repeats": int(args.repeats),
            "warmup": int(args.warmup),
            "ball_class_id": int(args.ball_class_id),
            "conf_threshold": float(args.conf_threshold),
            **aggregates,
            "batch_collect_ms_mean": None,
            "queue_wait_ms_mean": None,
            "e2e_pipeline_ms_mean": None,
            "e2e_pipeline_ms_p50": None,
            "e2e_pipeline_ms_p95": None,
            "e2e_pipeline_ms_p99": None,
            "drop_gap_frames": None,
            "batch_q_full_count": None,
            "pending_buckets_max": None,
            "ready_steps_max": None,
            "yolo_raw_q_max": None,
            "yolo_batch_q_max": None,
            "cpu_percent_mean": None,
            "ram_percent_mean": None,
            "gpu_percent_mean": None,
            "gpu_mem_percent_mean": None,
            "bbox_result_age_ms_mean": None,
            "bbox_result_age_ms_p50": None,
            "bbox_result_age_ms_p95": None,
            "bbox_result_age_ms_p99": None,
        }
        summary_rows.append(row)

        samples_down = {
            "total_ms_samples": downsample(samples["total_ms_samples"], args.samples_max_points),
            "prepare_ms_samples": downsample(samples["prepare_ms_samples"], args.samples_max_points),
            "forward_ms_samples": downsample(samples["forward_ms_samples"], args.samples_max_points),
            "post_ms_samples": downsample(samples["post_ms_samples"], args.samples_max_points),
        }

        case_json_path = run_paths["cases_dir"] / f"{case_id}.json"
        samples_full_path = run_paths["cases_dir"] / f"{case_id}_samples_full.json"
        write_json(samples_full_path, samples)
        case_payload = {
            "case_id": case_id,
            "benchmark_mode": "isolated",
            "configuration": {
                "backend": row["backend"],
                "model_path": row["model_path"],
                "trt_engine_path": row["trt_engine_path"],
                "device": row["device"],
                "input_format": row["input_format"],
                "imgsz": row["imgsz"],
                "seq_len": row["seq_len"],
                "chunk_batch_size": row["chunk_batch_size"],
                "images_per_call": row["images_per_call"],
                "object_scale": row["object_scale"],
                "repeats": row["repeats"],
                "warmup": row["warmup"],
                "ball_class_id": row["ball_class_id"],
                "conf_threshold": row["conf_threshold"],
            },
            "aggregates": row,
            "samples": samples_down,
            "artifacts": {
                "samples_full_path": str(samples_full_path.relative_to(run_paths["run_dir"])),
                "debug_trace_path": None,
            },
        }
        write_json(case_json_path, case_payload)
        cases_index.append(
            {
                "case_id": case_id,
                "configuration": case_payload["configuration"],
                "case_json_path": str(case_json_path.relative_to(run_paths["run_dir"])),
                "artifacts": case_payload["artifacts"],
            }
        )

    write_summary_csv(run_paths["summary_csv"], summary_rows)
    write_json(
        run_paths["cases_index_json"],
        {
            "schema": "benchmark_cases_index_v1",
            "benchmark_mode": "isolated",
            "generated_at": utc_now_iso(),
            "cases_count": len(cases_index),
            "cases": cases_index,
        },
    )

    end_ts = utc_now_iso()
    manifest = {
        "schema": "benchmark_manifest_v1",
        "benchmark_mode": "isolated",
        "started_at": start_ts,
        "ended_at": end_ts,
        "git_commit": detect_git_commit(Path(__file__).resolve().parent),
        "runtime": collect_runtime_env(),
        "parameter_grid": {
            "models": [str(model_path)],
            "image_sizes": image_sizes,
            "chunk_batch_sizes": batch_sizes,
            "seq_lens": seq_lens,
            "object_scales": object_scales,
            "repeats": int(args.repeats),
            "warmup": int(args.warmup),
            "conf_threshold": float(args.conf_threshold),
            "ball_class_id": int(args.ball_class_id),
            "input_format": str(args.input_format),
            "backend": str(args.backend),
            "device": str(args.device),
            "trt_engine": str(args.trt_engine or ""),
            "trt_dynamic": bool(args.trt_dynamic),
            "trt_workspace_gb": float(args.trt_workspace_gb),
        },
        "cases_count": len(cases),
        "outputs": {
            "summary_csv": str(run_paths["summary_csv"].relative_to(run_paths["run_dir"])),
            "cases_index_json": str(run_paths["cases_index_json"].relative_to(run_paths["run_dir"])),
            "cases_dir": str(run_paths["cases_dir"].relative_to(run_paths["run_dir"])),
        },
    }
    write_json(run_paths["manifest_json"], manifest)

    print(f"[BENCH][isolated] run_dir: {run_paths['run_dir']}")
    print(f"[BENCH][isolated] summary: {run_paths['summary_csv']}")
    print(f"[BENCH][isolated] cases: {run_paths['cases_dir']}")
    print(f"[BENCH][isolated] manifest: {run_paths['manifest_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

