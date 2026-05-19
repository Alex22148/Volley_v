"""CLI for runtime studies (max-resolution / max-fps).

Examples:
    python -m src.runtime_benchmark.run_runtime_study max-resolution \\
        --target-ms 20 --pipeline-mode auto --imgsz 640 --batch-size 4 --half

    python -m src.runtime_benchmark.run_runtime_study max-fps \\
        --width 1280 --height 960 --pipeline-mode auto --imgsz 640 --half
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from src.runtime_benchmark.study_runner import (
    DEFAULT_CANDIDATE_RESOLUTIONS,
    StudyConfig,
    run_max_fps_study,
    run_max_resolution_study,
)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pipeline-mode", default="auto",
                        choices=("auto", "cpu", "gpu_roundtrip", "gpu_zero_copy"))
    parser.add_argument("--color-backend", default="auto",
                        choices=("auto", "cv2_cuda", "torch_gpu", "cpu", "native_cuda_npp"))
    parser.add_argument("--inference-backend", default="auto",
                        choices=("auto", "ultralytics", "tensorrt", "dry_run"))
    parser.add_argument("--bayer-pattern", default="RG", choices=("RG", "BG", "GR", "GB"))
    parser.add_argument("--dtype", default="uint8", choices=("uint8", "uint16"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--num-frames", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Skip YOLO inference (default ON for studies).")
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--ball-class-id", type=int, default=0)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--output-json", type=Path, default=None,
                        help="Optional path to dump structured study results.")
    parser.add_argument("--log-level", default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"))


def _cfg_from_args(args: argparse.Namespace) -> StudyConfig:
    return StudyConfig(
        pipeline_mode=args.pipeline_mode,
        color_backend=args.color_backend,
        inference_backend=args.inference_backend,
        bayer_pattern=args.bayer_pattern,
        dtype=args.dtype,
        imgsz=int(args.imgsz),
        batch_size=int(args.batch_size),
        device=args.device,
        half=bool(args.half),
        num_frames=int(args.num_frames),
        warmup_iterations=int(args.warmup),
        model_path=args.model_path,
        dry_run=bool(args.dry_run),
        confidence=float(args.confidence),
        iou=float(args.iou),
        ball_class_id=int(args.ball_class_id),
        max_det=int(args.max_det),
    )


def _parse_resolutions(text: Optional[str]) -> List[Tuple[int, int]]:
    if not text:
        return list(DEFAULT_CANDIDATE_RESOLUTIONS)
    out: List[Tuple[int, int]] = []
    for token in str(text).replace(";", ",").split(","):
        token = token.strip().lower().replace(" ", "")
        if not token:
            continue
        if "x" not in token:
            raise SystemExit(f"resolution must look like '1280x960', got {token!r}")
        w_str, h_str = token.split("x", 1)
        w, h = int(w_str), int(h_str)
        if w <= 0 or h <= 0 or w % 2 != 0 or h % 2 != 0:
            raise SystemExit(f"resolution {token!r} must be positive even-numbered WxH")
        out.append((w, h))
    return out


def _setup_logging(level_name: str) -> None:
    logging.basicConfig(level=getattr(logging, level_name),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="VolleyHub runtime studies")
    sub = parser.add_subparsers(dest="study", required=True)

    a = sub.add_parser("max-resolution",
                       help="Find largest resolution whose median compute time fits a budget.")
    _add_common_args(a)
    a.add_argument("--target-ms", type=float, required=True,
                   help="Compute budget per packet, e.g. 20 for 50 fps.")
    a.add_argument("--candidate-resolutions", type=str, default=None,
                   help="Comma-separated WxH list. Default: built-in ladder up to 2464x2056.")

    b = sub.add_parser("max-fps",
                       help="Measure max sustainable FPS at a fixed resolution.")
    _add_common_args(b)
    b.add_argument("--width", type=int, required=True)
    b.add_argument("--height", type=int, required=True)

    args = parser.parse_args(argv)
    _setup_logging(args.log_level)
    cfg = _cfg_from_args(args)

    def progress(line: str) -> None:
        print(line, flush=True)

    if args.study == "max-resolution":
        candidates = _parse_resolutions(args.candidate_resolutions)
        result = run_max_resolution_study(
            target_ms=float(args.target_ms),
            candidates=candidates,
            cfg=cfg,
            progress=progress,
        )
    else:
        result = run_max_fps_study(
            width=int(args.width),
            height=int(args.height),
            cfg=cfg,
            progress=progress,
        )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result.to_dict(), indent=2, default=lambda o: getattr(o, "__dict__", str(o))),
            encoding="utf-8",
        )
        print(f"saved: {args.output_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
