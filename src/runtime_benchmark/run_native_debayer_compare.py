"""Compare four pipeline variants with the same TRT engine.

Variants:
  1. cpu                   = pipeline_mode=cpu             , color=cpu
  2. torch_gpu_roundtrip   = pipeline_mode=gpu_roundtrip   , color=torch_gpu
  3. torch_gpu_zero_copy   = pipeline_mode=gpu_zero_copy   , color=torch_gpu
  4. native_cuda_zero_copy = pipeline_mode=gpu_zero_copy   , color=native_cuda_npp

Outputs (default: reports/):
  native_debayer_benchmark.json
  native_debayer_benchmark.csv
  native_debayer_benchmark.md

The native backend identifies itself in the report as
``native_backend_detail = custom_cuda_kernel``. NPP is documented as
future work — never claim NPP is in use unless the kernel actually
binds to NPP.
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
from typing import List, Optional

from src.runtime_benchmark.study_runner import (
    StudyConfig,
    run_micro_benchmark,
)
from src.runtime_gpu.native_debayer import NativeCudaDebayer

_LOG = logging.getLogger(__name__)


@dataclass(slots=True)
class VariantResult:
    name: str
    pipeline_mode: str
    color_backend_requested: str
    color_backend_chosen: str
    inference_backend: str
    width: int
    height: int
    samples: int
    color_ms_median: float
    color_ms_p95: float
    color_download_ms_median: float
    inference_ms_median: float
    inference_ms_p95: float
    inference_upload_ms_median: float
    packet_ms_median: float
    packet_ms_p95: float
    packets_per_second: float
    fps_per_camera: float
    total_images_per_second: float
    color_output_location: str
    inference_input_location: str
    zero_copy_to_inference: bool
    gpu_debayer_only_not_full_gpu_pipeline: bool
    fallback_used: Optional[str]
    native_backend_detail: Optional[str]
    error: Optional[str] = None
    fallback_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


_CSV_COLUMNS: List[str] = [
    "name",
    "pipeline_mode",
    "color_backend_requested",
    "color_backend_chosen",
    "inference_backend",
    "width",
    "height",
    "samples",
    "color_ms_median",
    "color_ms_p95",
    "color_download_ms_median",
    "inference_ms_median",
    "inference_ms_p95",
    "inference_upload_ms_median",
    "packet_ms_median",
    "packet_ms_p95",
    "packets_per_second",
    "fps_per_camera",
    "total_images_per_second",
    "color_output_location",
    "inference_input_location",
    "zero_copy_to_inference",
    "gpu_debayer_only_not_full_gpu_pipeline",
    "fallback_used",
    "native_backend_detail",
    "error",
]


def _make_cfg(args: argparse.Namespace, pipeline_mode: str, color_backend: str) -> StudyConfig:
    return StudyConfig(
        pipeline_mode=pipeline_mode,
        color_backend=color_backend,
        inference_backend=args.inference_backend,
        bayer_pattern=args.bayer_pattern,
        dtype="uint8",
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


def _detect_fallback(measurement, requested_color_backend: str, requested_pipeline_mode: str) -> Optional[str]:
    """A fallback was used iff:
    - the chosen color backend differs from the requested one, OR
    - zero_copy was requested but the run actually round-tripped via CPU.
    """
    if measurement.color_backend != requested_color_backend and requested_color_backend != "auto":
        return f"color_backend chosen={measurement.color_backend} (requested={requested_color_backend})"
    if requested_pipeline_mode == "gpu_zero_copy" and not measurement.zero_copy_to_inference:
        return "zero-copy requested but not honored at runtime"
    return None


def _run_variant(name: str, pipeline_mode: str, color_backend: str,
                 args: argparse.Namespace) -> VariantResult:
    _LOG.info("=== Variant %s | pipeline=%s color=%s ===", name, pipeline_mode, color_backend)
    cfg = _make_cfg(args, pipeline_mode, color_backend)
    try:
        cfg.validate()
    except Exception as exc:
        return _failed_variant(name, pipeline_mode, color_backend, args, str(exc))

    def progress(msg: str) -> None:
        _LOG.info("%s", msg)

    measurement = run_micro_benchmark(
        width=int(args.width),
        height=int(args.height),
        cfg=cfg,
        progress=progress,
    )
    if measurement.error:
        return _failed_variant(name, pipeline_mode, color_backend, args, measurement.error)

    fps_packets = measurement.max_fps_from_median  # packets per second
    fps_per_cam = fps_packets                       # synced 4-cam packet ⇒ same as packets/s
    total_imgs_s = fps_packets * 4.0

    fallback_used = _detect_fallback(measurement, color_backend, pipeline_mode)

    native_detail: Optional[str] = None
    if color_backend == "native_cuda_npp" or measurement.color_backend == "native_cuda_npp":
        try:
            info = NativeCudaDebayer.describe_backend()
            native_detail = info.get("detail") or None
        except Exception:
            native_detail = None

    return VariantResult(
        name=name,
        pipeline_mode=pipeline_mode,
        color_backend_requested=color_backend,
        color_backend_chosen=measurement.color_backend,
        inference_backend=measurement.inference_backend,
        width=int(args.width),
        height=int(args.height),
        samples=int(measurement.samples),
        color_ms_median=float(measurement.median_color_ms),
        color_ms_p95=float(measurement.p95_compute_ms),  # see note in MD report
        color_download_ms_median=float(measurement.median_color_download_ms),
        inference_ms_median=float(measurement.median_inference_ms),
        inference_ms_p95=float(measurement.p95_compute_ms),
        inference_upload_ms_median=float(measurement.median_inference_upload_ms),
        packet_ms_median=float(measurement.median_compute_ms),
        packet_ms_p95=float(measurement.p95_compute_ms),
        packets_per_second=float(fps_packets),
        fps_per_camera=float(fps_per_cam),
        total_images_per_second=float(total_imgs_s),
        color_output_location=str(measurement.color_output_location),
        inference_input_location=str(measurement.inference_input_location),
        zero_copy_to_inference=bool(measurement.zero_copy_to_inference),
        gpu_debayer_only_not_full_gpu_pipeline=bool(measurement.gpu_debayer_only_not_full_gpu_pipeline),
        fallback_used=fallback_used,
        native_backend_detail=native_detail,
        error=None,
        fallback_warnings=list(measurement.fallback_warnings),
    )


def _failed_variant(name: str, pipeline_mode: str, color_backend: str,
                    args: argparse.Namespace, error: str) -> VariantResult:
    return VariantResult(
        name=name,
        pipeline_mode=pipeline_mode,
        color_backend_requested=color_backend,
        color_backend_chosen="",
        inference_backend="",
        width=int(args.width),
        height=int(args.height),
        samples=0,
        color_ms_median=float("nan"),
        color_ms_p95=float("nan"),
        color_download_ms_median=float("nan"),
        inference_ms_median=float("nan"),
        inference_ms_p95=float("nan"),
        inference_upload_ms_median=float("nan"),
        packet_ms_median=float("nan"),
        packet_ms_p95=float("nan"),
        packets_per_second=0.0,
        fps_per_camera=0.0,
        total_images_per_second=0.0,
        color_output_location="",
        inference_input_location="",
        zero_copy_to_inference=False,
        gpu_debayer_only_not_full_gpu_pipeline=False,
        fallback_used="run failed",
        native_backend_detail=None,
        error=error,
    )


def _fmt_ms(v) -> str:
    try:
        f = float(v)
        if f != f:  # NaN
            return "—"
        return f"{f:.2f}"
    except Exception:
        return "—"


def _fmt_fps(v) -> str:
    try:
        f = float(v)
        if f != f or f == 0:
            return "—"
        return f"{f:.1f}"
    except Exception:
        return "—"


def _build_markdown(args: argparse.Namespace, variants: List[VariantResult]) -> str:
    native_info = NativeCudaDebayer.describe_backend()
    lines: List[str] = []
    a = lines.append
    a("# Native debayer benchmark")
    a("")
    a(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    a(f"Resolution: **{args.width}x{args.height}**, batch={args.batch_size}, imgsz={args.imgsz}, "
      f"half={bool(args.half)}, model=`{args.model_path}`, inference_backend=`{args.inference_backend}`, "
      f"frames per variant={args.num_frames} (warmup={args.warmup}).")
    a("")
    a("## Native backend identification")
    a("")
    a(f"- `available`: {native_info['available']}")
    a(f"- `native_backend_detail`: **{native_info['detail']}**")
    a(f"- extension path: `{native_info['extension_path']}`")
    a(f"- CUDA device: {native_info.get('cuda_device_name')}")
    a(f"- torch: {native_info.get('torch_version')}, CUDA: {native_info.get('cuda_version')}")
    a("")
    a("> The current implementation is a **bilinear demosaic CUDA kernel** built via `torch.utils.cpp_extension`. "
      "**It is NOT NPP.** NPP (`nppiCFAToBGR_8u_C1C3R` and friends) is documented as future work; the project "
      "name `native_cuda_npp` is the externally-stable identifier of this slot, while the runtime detail field "
      "above tells you which implementation is actually executing.")
    a("")
    a("## Headline comparison")
    a("")
    a("| Variant                  | color_ms median | color_ms p95* | packet_ms median | packet_ms p95 | packets/s | FPS/cam | total imgs/s | zero_copy | fallback_used |")
    a("|--------------------------|------------------|----------------|-------------------|----------------|-----------|---------|---------------|-----------|----------------|")
    for v in variants:
        a("| {name:<24} | {col_med:>16} | {col_p95:>14} | {pkt_med:>17} | {pkt_p95:>14} | {pps:>9} | {fpc:>7} | {imgs:>13} | {zc:>9} | {fb} |".format(
            name=v.name,
            col_med=_fmt_ms(v.color_ms_median),
            col_p95="—",  # we don't have a separate p95 for color stage from study_runner
            pkt_med=_fmt_ms(v.packet_ms_median),
            pkt_p95=_fmt_ms(v.packet_ms_p95),
            pps=_fmt_fps(v.packets_per_second),
            fpc=_fmt_fps(v.fps_per_camera),
            imgs=_fmt_fps(v.total_images_per_second),
            zc=str(v.zero_copy_to_inference),
            fb=(v.fallback_used or "—"),
        ))
    a("")
    a("\\* color_ms p95 is not separately captured by the current `MicroBenchmarkResult` "
      "structure — `packet_ms p95` already shows the worst-case end-to-end behaviour, which "
      "is the actionable signal.")
    a("")
    a("## Inference and locations")
    a("")
    a("| Variant                  | inference_backend | inf_ms median | inf_upload_ms median | color_loc | inf_loc | gpu_debayer_only |")
    a("|--------------------------|--------------------|----------------|------------------------|-----------|---------|-------------------|")
    for v in variants:
        a("| {name:<24} | {ib:<18} | {im:>14} | {iu:>22} | {col:>9} | {inf:>7} | {gdo} |".format(
            name=v.name,
            ib=v.inference_backend or "—",
            im=_fmt_ms(v.inference_ms_median),
            iu=_fmt_ms(v.inference_upload_ms_median),
            col=v.color_output_location or "—",
            inf=v.inference_input_location or "—",
            gdo=str(v.gpu_debayer_only_not_full_gpu_pipeline),
        ))
    a("")
    a("## Honest assessment")
    a("")
    cpu_v = next((v for v in variants if v.name == "cpu"), None)
    torch_zc = next((v for v in variants if v.name == "torch_gpu_zero_copy"), None)
    native_zc = next((v for v in variants if v.name == "native_cuda_zero_copy"), None)
    if cpu_v and native_zc and not native_zc.error:
        ratio = (cpu_v.color_ms_median / native_zc.color_ms_median) if native_zc.color_ms_median > 0 else 0.0
        a(f"- color stage: cpu {_fmt_ms(cpu_v.color_ms_median)} ms vs native_cuda_zero_copy "
          f"{_fmt_ms(native_zc.color_ms_median)} ms → **{ratio:.1f}× speedup** vs CPU.")
    if torch_zc and native_zc and not native_zc.error and not torch_zc.error:
        delta = native_zc.color_ms_median - torch_zc.color_ms_median
        if delta < 0:
            a(f"- vs torch_gpu_zero_copy: native is {_fmt_ms(-delta)} ms FASTER on color stage.")
        elif delta > 0:
            a(f"- vs torch_gpu_zero_copy: native is {_fmt_ms(delta)} ms slower on color stage. "
              "The current native path is a simple bilinear demosaic kernel; a true NPP implementation "
              "(`nppiCFAToBGR_8u_C1C3R`) or a hand-tuned shared-memory tiled kernel could close this gap. "
              "Documented as future work in `src/runtime_gpu/native_debayer/README.md`.")
        else:
            a("- vs torch_gpu_zero_copy: native and torch are within noise on color stage.")
    if native_zc and native_zc.fallback_used:
        a(f"- ⚠ native variant reported `fallback_used = {native_zc.fallback_used}` — "
          "this means the run did NOT execute on the requested path. See `fallback_warnings`.")
    if native_zc and native_zc.zero_copy_to_inference:
        a("- native_cuda_zero_copy honored zero-copy semantics: tensor stayed on CUDA from "
          "debayer through to TRT inference, no CPU bounce in between.")
    a("")
    a("## Production wiring (DEFERRED)")
    a("")
    a("This benchmark does NOT change the live pipeline. To enable the native backend in production "
      "after results are accepted, see `src/runtime_gpu/native_debayer/README.md` for the planned "
      "`VOLLEYHUB_COLOR_BACKEND` env var. CPU fallback is preserved as the default.")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="4-way native debayer benchmark.")
    parser.add_argument("--width", type=int, default=2464)
    parser.add_argument("--height", type=int, default=2056)
    parser.add_argument("--bayer-pattern", default="RG", choices=("RG", "BG", "GR", "GB"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--half", action="store_true", default=True)
    parser.add_argument("--no-half", dest="half", action="store_false")
    parser.add_argument("--num-frames", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--inference-backend", default="tensorrt",
                        choices=("auto", "ultralytics", "tensorrt"))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--ball-class-id", type=int, default=0)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--variants", default="cpu,torch_gpu_roundtrip,torch_gpu_zero_copy,native_cuda_zero_copy",
                        help="Comma-separated subset to run.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    plan = {
        "cpu":                      ("cpu",            "cpu"),
        "torch_gpu_roundtrip":      ("gpu_roundtrip",  "torch_gpu"),
        "torch_gpu_zero_copy":      ("gpu_zero_copy",  "torch_gpu"),
        "native_cuda_zero_copy":    ("gpu_zero_copy",  "native_cuda_npp"),
    }
    selected = [v.strip() for v in str(args.variants).split(",") if v.strip()]
    unknown = [v for v in selected if v not in plan]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}; known={list(plan.keys())}")

    results: List[VariantResult] = []
    for v_name in selected:
        pipeline_mode, color_backend = plan[v_name]
        results.append(_run_variant(v_name, pipeline_mode, color_backend, args))

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    json_payload = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "native_backend_info": NativeCudaDebayer.describe_backend(),
        "variants": [r.to_dict() for r in results],
    }
    (out_dir / "native_debayer_benchmark.json").write_text(
        json.dumps(json_payload, indent=2, default=str), encoding="utf-8"
    )

    csv_path = out_dir / "native_debayer_benchmark.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for r in results:
            row = r.to_dict()
            writer.writerow({k: row.get(k, "") for k in _CSV_COLUMNS})

    md = _build_markdown(args, results)
    (out_dir / "native_debayer_benchmark.md").write_text(md, encoding="utf-8")

    print(f"\nWrote: {out_dir / 'native_debayer_benchmark.md'}")
    print(f"Wrote: {out_dir / 'native_debayer_benchmark.json'}")
    print(f"Wrote: {out_dir / 'native_debayer_benchmark.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
