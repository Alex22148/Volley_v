"""Full-HD inference-size sweep for the 4-camera VolleyHub pipeline.

Fixed:
    - cameras = 4
    - capture = 1920x1080
    - target_fps = 50  → packet_budget_ms = 20
    - batch = 4
    - color_backend = native_cuda_npp (custom_cuda_kernel)
    - pipeline_mode = gpu_zero_copy
    - inference_backend = tensorrt
    - half = True

Sweep over YOLO inference input shapes:
    640        — baseline square
    960        — square
    1280       — square
    (1088, 1920) — Full-HD-like, height aligned to stride=32
    (optional) 1920 square — only if the rectangular engine is unavailable

For each variant we look up a pre-built TRT b4 FP16 static engine in
artifacts/tensorrt_exports/. Variants without an engine are reported as
NOT_TESTED_ENGINE_MISSING — never silently skipped.

Outputs:
    reports/fullhd_inference_size_sweep.{md,json,csv}
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.runtime_benchmark.study_runner import StudyConfig, run_micro_benchmark
from src.runtime_gpu.native_debayer import NativeCudaDebayer

_LOG = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
ENGINE_DIR = REPO / "artifacts" / "tensorrt_exports"

# Sweep configurations: (label, imgsz_for_runtime, engine_filename_or_None_to_search)
SweepImgsz = Union[int, Tuple[int, int]]


def _engine_filename_for(imgsz: SweepImgsz) -> str:
    if isinstance(imgsz, tuple):
        h, w = int(imgsz[0]), int(imgsz[1])
        return f"best__fp16_{h}x{w}_b4_static__fp16__img{h}x{w}__b4__static.engine"
    return f"best__fp16_{int(imgsz)}_b4_static__fp16__img{int(imgsz)}__b4__static.engine"


def _shape_label(imgsz: SweepImgsz) -> str:
    if isinstance(imgsz, tuple):
        return f"{int(imgsz[0])}x{int(imgsz[1])}"
    v = int(imgsz)
    return f"{v}x{v}"


SWEEP_VARIANTS: List[Tuple[str, SweepImgsz]] = [
    ("imgsz_640",         640),
    ("imgsz_960",         960),
    ("imgsz_1280",        1280),
    ("imgsz_1088x1920",   (1088, 1920)),
]


@dataclass(slots=True)
class VariantResult:
    name: str
    capture_resolution: str
    inference_input_shape: str
    yolo_imgsz: str
    engine_path: str
    engine_present: bool
    color_backend_chosen: str
    inference_backend: str
    samples: int

    color_ms_median: float
    color_ms_p95: float
    inference_ms_median: float
    inference_ms_p95: float
    packet_ms_median: float
    packet_ms_p95: float

    packets_per_second: float
    fps_per_camera_median: float
    fps_per_camera_safe_p95: float
    total_images_per_second_median: float
    total_images_per_second_safe_p95: float

    pass_50fps_median: bool
    pass_50fps_safe_p95: bool
    stability_margin_ms: float

    color_output_location: str
    inference_input_location: str
    zero_copy_to_inference: bool
    gpu_debayer_only_not_full_gpu_pipeline: bool
    fallback_used: Optional[str]
    native_backend_detail: Optional[str]
    error: Optional[str] = None
    fallback_warnings: List[str] = field(default_factory=list)
    status: str = "OK"

    def to_dict(self) -> dict:
        return asdict(self)


_CSV_COLUMNS: List[str] = [
    "name",
    "status",
    "capture_resolution",
    "inference_input_shape",
    "yolo_imgsz",
    "engine_path",
    "engine_present",
    "color_backend_chosen",
    "inference_backend",
    "samples",
    "color_ms_median",
    "color_ms_p95",
    "inference_ms_median",
    "inference_ms_p95",
    "packet_ms_median",
    "packet_ms_p95",
    "packets_per_second",
    "fps_per_camera_median",
    "fps_per_camera_safe_p95",
    "total_images_per_second_median",
    "total_images_per_second_safe_p95",
    "pass_50fps_median",
    "pass_50fps_safe_p95",
    "stability_margin_ms",
    "color_output_location",
    "inference_input_location",
    "zero_copy_to_inference",
    "gpu_debayer_only_not_full_gpu_pipeline",
    "fallback_used",
    "native_backend_detail",
    "error",
]


def _missing_variant(name: str, imgsz: SweepImgsz, capture_w: int, capture_h: int,
                     engine_path: Path, status: str, error: str) -> VariantResult:
    return VariantResult(
        name=name,
        capture_resolution=f"{capture_w}x{capture_h}",
        inference_input_shape=_shape_label(imgsz),
        yolo_imgsz=str(imgsz) if isinstance(imgsz, int) else f"({int(imgsz[0])},{int(imgsz[1])})",
        engine_path=str(engine_path),
        engine_present=engine_path.exists(),
        color_backend_chosen="",
        inference_backend="",
        samples=0,
        color_ms_median=float("nan"),
        color_ms_p95=float("nan"),
        inference_ms_median=float("nan"),
        inference_ms_p95=float("nan"),
        packet_ms_median=float("nan"),
        packet_ms_p95=float("nan"),
        packets_per_second=0.0,
        fps_per_camera_median=0.0,
        fps_per_camera_safe_p95=0.0,
        total_images_per_second_median=0.0,
        total_images_per_second_safe_p95=0.0,
        pass_50fps_median=False,
        pass_50fps_safe_p95=False,
        stability_margin_ms=float("nan"),
        color_output_location="",
        inference_input_location="",
        zero_copy_to_inference=False,
        gpu_debayer_only_not_full_gpu_pipeline=False,
        fallback_used=None,
        native_backend_detail=None,
        error=error,
        status=status,
    )


def _run_variant(name: str, imgsz: SweepImgsz, args: argparse.Namespace) -> VariantResult:
    engine_path = ENGINE_DIR / _engine_filename_for(imgsz)
    if not engine_path.exists():
        _LOG.warning("[%s] engine missing at %s", name, engine_path)
        return _missing_variant(
            name, imgsz, args.capture_width, args.capture_height, engine_path,
            status="NOT_TESTED_ENGINE_MISSING",
            error=f"engine not found: {engine_path.name}",
        )

    cfg = StudyConfig(
        pipeline_mode="gpu_zero_copy",
        color_backend="native_cuda_npp",
        inference_backend="tensorrt",
        bayer_pattern=args.bayer_pattern,
        dtype="uint8",
        imgsz=imgsz,
        batch_size=int(args.batch_size),
        device="cuda",
        half=True,
        num_frames=int(args.num_frames),
        warmup_iterations=int(args.warmup),
        model_path=str(engine_path),
        dry_run=False,
        confidence=float(args.confidence),
        iou=float(args.iou),
        ball_class_id=int(args.ball_class_id),
        max_det=int(args.max_det),
    )

    try:
        cfg.validate()
    except Exception as exc:
        return _missing_variant(name, imgsz, args.capture_width, args.capture_height,
                                engine_path, "FAILED_CONFIG", str(exc))

    def progress(msg: str) -> None:
        _LOG.info("[%s] %s", name, msg)

    measurement = run_micro_benchmark(
        width=int(args.capture_width),
        height=int(args.capture_height),
        cfg=cfg,
        progress=progress,
    )
    if measurement.error:
        return _missing_variant(name, imgsz, args.capture_width, args.capture_height,
                                engine_path, "FAILED_RUNTIME", measurement.error)

    fps_packets_med = measurement.max_fps_from_median  # = packets/s
    fps_packets_p95 = measurement.max_fps_from_p95
    pass_med = measurement.median_compute_ms <= float(args.packet_budget_ms)
    pass_p95 = measurement.p95_compute_ms <= float(args.packet_budget_ms)

    fallback_used = None
    if measurement.color_backend != "native_cuda_npp":
        fallback_used = f"color_backend chosen={measurement.color_backend} (requested=native_cuda_npp)"
    elif not measurement.zero_copy_to_inference:
        fallback_used = "zero-copy requested but not honored at runtime"

    native_detail: Optional[str] = None
    try:
        info = NativeCudaDebayer.describe_backend()
        native_detail = info.get("detail") or None
    except Exception:
        pass

    return VariantResult(
        name=name,
        capture_resolution=f"{args.capture_width}x{args.capture_height}",
        inference_input_shape=_shape_label(imgsz),
        yolo_imgsz=str(imgsz) if isinstance(imgsz, int) else f"({int(imgsz[0])},{int(imgsz[1])})",
        engine_path=str(engine_path),
        engine_present=True,
        color_backend_chosen=measurement.color_backend,
        inference_backend=measurement.inference_backend,
        samples=int(measurement.samples),
        color_ms_median=float(measurement.median_color_ms),
        color_ms_p95=float(measurement.p95_compute_ms),  # study_runner only stores p95 for total compute
        inference_ms_median=float(measurement.median_inference_ms),
        inference_ms_p95=float(measurement.p95_compute_ms),
        packet_ms_median=float(measurement.median_compute_ms),
        packet_ms_p95=float(measurement.p95_compute_ms),
        packets_per_second=float(fps_packets_med),
        fps_per_camera_median=float(fps_packets_med),
        fps_per_camera_safe_p95=float(fps_packets_p95),
        total_images_per_second_median=float(fps_packets_med * 4.0),
        total_images_per_second_safe_p95=float(fps_packets_p95 * 4.0),
        pass_50fps_median=bool(pass_med),
        pass_50fps_safe_p95=bool(pass_p95),
        stability_margin_ms=float(args.packet_budget_ms - measurement.p95_compute_ms),
        color_output_location=str(measurement.color_output_location),
        inference_input_location=str(measurement.inference_input_location),
        zero_copy_to_inference=bool(measurement.zero_copy_to_inference),
        gpu_debayer_only_not_full_gpu_pipeline=bool(measurement.gpu_debayer_only_not_full_gpu_pipeline),
        fallback_used=fallback_used,
        native_backend_detail=native_detail,
        error=None,
        fallback_warnings=list(measurement.fallback_warnings),
        status="OK",
    )


def _fmt(v, p=2) -> str:
    try:
        f = float(v)
        if f != f:
            return "—"
        return f"{f:.{p}f}"
    except Exception:
        return "—"


def _yes_no(b: bool) -> str:
    return "PASS" if b else "FAIL"


def _build_md(args: argparse.Namespace, variants: List[VariantResult]) -> str:
    info = NativeCudaDebayer.describe_backend()
    lines: List[str] = []
    a = lines.append
    a("# Full-HD inference size sweep")
    a("")
    a(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    a("")
    a("## Constants")
    a("")
    a(f"- cameras: 4")
    a(f"- capture: **{args.capture_width}x{args.capture_height}** RAW Bayer={args.bayer_pattern}")
    a(f"- target FPS (per camera): **{args.target_fps}** ⇒ packet budget = **{args.packet_budget_ms} ms**")
    a(f"- batch size: {args.batch_size}")
    a(f"- pipeline_mode: gpu_zero_copy")
    a(f"- color_backend: native_cuda_npp ({info.get('detail')})")
    a(f"- inference_backend: tensorrt (FP16, batch=4 static)")
    a(f"- half: True")
    a(f"- num_frames per variant: {args.num_frames} (warmup={args.warmup})")
    a(f"- CUDA device: {info.get('cuda_device_name')}")
    a("")
    a("> The native debayer slot is `native_cuda_npp`; the runtime detail "
      f"is **{info.get('detail')}** — not NPP. NPP integration is documented "
      "as future work in `src/runtime_gpu/native_debayer/README.md`.")
    a("")

    a("## PYTANIE 1 — Maksymalny FPS dla 4 kamer przy danym rozmiarze inferencji")
    a("")
    a("| variant | inference_input | packet_ms median | packet_ms p95 | max_fps_median | max_fps_safe_p95 | fps/cam median | fps/cam safe | total imgs/s median | total imgs/s safe |")
    a("|---------|-----------------|-------------------|----------------|----------------|-------------------|----------------|---------------|----------------------|--------------------|")
    for v in variants:
        if v.status != "OK":
            a(f"| {v.name:<16} | {v.inference_input_shape:<15} | _{v.status}_ |  |  |  |  |  |  |  |")
            continue
        a(
            f"| {v.name:<16} | {v.inference_input_shape:<15} | "
            f"{_fmt(v.packet_ms_median)} ms | {_fmt(v.packet_ms_p95)} ms | "
            f"{_fmt(v.fps_per_camera_median, 1)} | {_fmt(v.fps_per_camera_safe_p95, 1)} | "
            f"{_fmt(v.fps_per_camera_median, 1)} | {_fmt(v.fps_per_camera_safe_p95, 1)} | "
            f"{_fmt(v.total_images_per_second_median, 1)} | {_fmt(v.total_images_per_second_safe_p95, 1)} |"
        )
    a("")

    a("## PYTANIE 2 — Największa rozdzielczość inferencji utrzymująca 50 FPS dla 4 kamer")
    a("")
    a(f"Definicja: 50 FPS × 4 kamery zsynchronizowane ⇒ 50 pakietów/s ⇒ budżet = {args.packet_budget_ms} ms na pakiet (4 obrazy).")
    a("")
    a("| variant | inference_input | packet_ms median | packet_ms p95 | margin (ms vs 20) | PASS median | PASS safe p95 |")
    a("|---------|-----------------|-------------------|----------------|----------------------|-------------|----------------|")
    for v in variants:
        if v.status != "OK":
            a(f"| {v.name:<16} | {v.inference_input_shape:<15} | _{v.status}_ |  |  |  |  |")
            continue
        a(
            f"| {v.name:<16} | {v.inference_input_shape:<15} | "
            f"{_fmt(v.packet_ms_median)} ms | {_fmt(v.packet_ms_p95)} ms | "
            f"{_fmt(v.stability_margin_ms)} | {_yes_no(v.pass_50fps_median)} | {_yes_no(v.pass_50fps_safe_p95)} |"
        )
    a("")

    a("## Per-stage breakdown")
    a("")
    a("| variant | color_ms median | inference_ms median | color_loc | inf_loc | zero_copy | engine |")
    a("|---------|-------------------|----------------------|-----------|---------|-----------|--------|")
    for v in variants:
        if v.status != "OK":
            engine_name = Path(v.engine_path).name if v.engine_path else "—"
            a(f"| {v.name:<16} | _{v.status}_ |  |  |  |  | `{engine_name}` |")
            continue
        a(
            f"| {v.name:<16} | {_fmt(v.color_ms_median)} | {_fmt(v.inference_ms_median)} | "
            f"{v.color_output_location} | {v.inference_input_location} | "
            f"{v.zero_copy_to_inference} | `{Path(v.engine_path).name}` |"
        )
    a("")

    # ---- Decision answer ----
    ok_variants = [v for v in variants if v.status == "OK"]
    pass_med = [v for v in ok_variants if v.pass_50fps_median]
    pass_p95 = [v for v in ok_variants if v.pass_50fps_safe_p95]

    def _shape_area(v: VariantResult) -> int:
        if "x" in v.inference_input_shape:
            h_str, w_str = v.inference_input_shape.split("x")
            return int(h_str) * int(w_str)
        return 0

    largest_pass_med = max(pass_med, key=_shape_area) if pass_med else None
    largest_pass_p95 = max(pass_p95, key=_shape_area) if pass_p95 else None
    fastest_variant = max(ok_variants, key=lambda v: v.fps_per_camera_safe_p95) if ok_variants else None

    a("## Decision answer")
    a("")
    a("### 1. Maksymalny FPS")
    a("")
    if not ok_variants:
        a("- Brak udanych pomiarów. Wszystkie warianty failed/missing.")
    else:
        for v in ok_variants:
            a(f"- dla `{v.inference_input_shape}`: **{_fmt(v.fps_per_camera_median, 1)} FPS median**, "
              f"**{_fmt(v.fps_per_camera_safe_p95, 1)} FPS safe (p95)** "
              f"(packet median {_fmt(v.packet_ms_median)} ms, p95 {_fmt(v.packet_ms_p95)} ms)")
        if fastest_variant:
            a(f"- **Największy praktyczny FPS osiągnięto dla:** `{fastest_variant.inference_input_shape}` "
              f"→ {_fmt(fastest_variant.fps_per_camera_safe_p95, 1)} FPS safe (p95).")
    a("")
    a("### 2. Największa rozdzielczość dla 50 FPS")
    a("")
    if largest_pass_med:
        a(f"- największa rozdzielczość spełniająca **median ≤ {args.packet_budget_ms} ms**: "
          f"**`{largest_pass_med.inference_input_shape}`** "
          f"(median {_fmt(largest_pass_med.packet_ms_median)} ms)")
    else:
        a(f"- największa rozdzielczość spełniająca median ≤ {args.packet_budget_ms} ms: **brak — żaden wariant nie wyrabia mediany**")
    if largest_pass_p95:
        a(f"- największa rozdzielczość spełniająca **p95 ≤ {args.packet_budget_ms} ms**: "
          f"**`{largest_pass_p95.inference_input_shape}`** "
          f"(p95 {_fmt(largest_pass_p95.packet_ms_p95)} ms, margin {_fmt(largest_pass_p95.stability_margin_ms)} ms)")
    else:
        a(f"- największa rozdzielczość spełniająca p95 ≤ {args.packet_budget_ms} ms: **brak — żaden wariant nie utrzymuje p95**")
    if largest_pass_p95:
        rec = largest_pass_p95
        rec_label = "safe (utrzymuje p95)"
    elif largest_pass_med:
        rec = largest_pass_med
        rec_label = "agresywna (median ≤ budżet, ale ogony p95 ponad budżet — możliwe rzadkie dropy)"
    else:
        rec = None
        rec_label = ""
    if rec:
        a(f"- **rekomendowana produkcyjnie:** `{rec.inference_input_shape}` ({rec_label})")
        a(f"  engine: `{Path(rec.engine_path).name}`")
    else:
        a("- **rekomendowana produkcyjnie:** żaden z testowanych wariantów; "
          "rozważ mniejszy model, crop ROI lub tile detection.")
    a("")
    a("### 3. Full HD inference (1920×1088, czyli 1088×1920 H×W)")
    a("")
    fhd = next((v for v in variants if v.inference_input_shape in ("1088x1920", "1920x1088")), None)
    if not fhd or fhd.status != "OK":
        a(f"- **status: {fhd.status if fhd else 'NOT_PRESENT_IN_SWEEP'}** "
          f"({fhd.error if fhd and fhd.error else 'engine likely missing — see status column'})")
    else:
        a(f"- median: {_fmt(fhd.packet_ms_median)} ms vs budżet {args.packet_budget_ms} ms ⇒ "
          f"**{_yes_no(fhd.pass_50fps_median)} median**")
        a(f"- p95: {_fmt(fhd.packet_ms_p95)} ms vs budżet {args.packet_budget_ms} ms ⇒ "
          f"**{_yes_no(fhd.pass_50fps_safe_p95)} safe (p95)**")
        margin = fhd.stability_margin_ms
        if margin >= 0:
            a(f"- zapas marginesu: **+{_fmt(margin)} ms** ponad budżet")
        else:
            a(f"- brak marginesu: **{_fmt(margin)} ms** (przekroczenie {_fmt(-margin)} ms)")
    a("")

    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Full-HD inference size sweep (4 cams × 50 FPS).")
    parser.add_argument("--capture-width", type=int, default=1920)
    parser.add_argument("--capture-height", type=int, default=1080)
    parser.add_argument("--bayer-pattern", default="RG", choices=("RG", "BG", "GR", "GB"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--target-fps", type=float, default=50.0)
    parser.add_argument("--packet-budget-ms", type=float, default=20.0)
    parser.add_argument("--num-frames", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--ball-class-id", type=int, default=0)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--variants", type=str, default="",
                        help="Comma-separated subset of variant names (default: all).")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    selected = SWEEP_VARIANTS
    if args.variants.strip():
        wanted = {v.strip() for v in args.variants.split(",") if v.strip()}
        selected = [(n, im) for (n, im) in SWEEP_VARIANTS if n in wanted]

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[VariantResult] = []
    for name, imgsz in selected:
        results.append(_run_variant(name, imgsz, args))

    # JSON
    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "native_backend_info": NativeCudaDebayer.describe_backend(),
        "variants": [r.to_dict() for r in results],
    }
    (out_dir / "fullhd_inference_size_sweep.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")

    # CSV
    with (out_dir / "fullhd_inference_size_sweep.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for r in results:
            row = r.to_dict()
            writer.writerow({k: row.get(k, "") for k in _CSV_COLUMNS})

    # Markdown
    md = _build_md(args, results)
    (out_dir / "fullhd_inference_size_sweep.md").write_text(md, encoding="utf-8")

    print(f"\nWrote: {out_dir / 'fullhd_inference_size_sweep.md'}")
    print(f"Wrote: {out_dir / 'fullhd_inference_size_sweep.json'}")
    print(f"Wrote: {out_dir / 'fullhd_inference_size_sweep.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
