"""Compare legacy/native preprocessing against the unified image processor.

This diagnostic benchmark does not run inference and does not affect the
production default path. It verifies whether both preprocessing paths produce
compatible BCHW tensors and records timing/device/fallback information.
"""
from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.runtime_benchmark._runtime_metrics import (
    summarize_samples,
    utc_now_iso,
    write_json,
)
from src.runtime_gpu.gpu_color_converter import ColorConverterConfig, GpuColorConverter
from src.runtime_gpu.gpu_image_processor import GpuImageProcessor, ImageProcessorConfig
from src.runtime_sources.base_frame_source import FrameBatch4Cam
from src.runtime_sources.synthetic_raw_4cam_source import (
    SyntheticRaw4CamSource,
    SyntheticSourceConfig,
)


@dataclass(slots=True)
class PreprocessPathResult:
    path: str
    tensor_shape: tuple
    tensor_device: str
    tensor_dtype: str
    tensor_layout: str
    stage_ms: dict
    fallback_used: str = ""
    error: str = ""

    @property
    def is_bchw(self) -> bool:
        return len(self.tensor_shape) == 4 and self.tensor_shape[1] == 3


@dataclass(slots=True)
class PreprocessComparisonResult:
    packet_id: int
    native: PreprocessPathResult
    unified: PreprocessPathResult
    bchw_compatible: bool
    same_shape: bool

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["native"]["tensor_shape"] = list(self.native.tensor_shape)
        payload["unified"]["tensor_shape"] = list(self.unified.tensor_shape)
        return payload


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare native preprocessing with unified image processor."
    )
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=64)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--bayer-pattern", type=str, default="RG", choices=("RG", "BG", "GR", "GB"))
    parser.add_argument("--dtype", type=str, default="uint8", choices=("uint8", "uint16"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--half", action="store_true")
    parser.add_argument(
        "--native-backend",
        type=str,
        default="cpu",
        choices=("auto", "cv2_cuda", "torch_gpu", "cpu", "native_cuda_npp"),
        help="Backend used by the legacy/native color converter.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def build_synthetic_source(args: argparse.Namespace) -> SyntheticRaw4CamSource:
    return SyntheticRaw4CamSource(
        SyntheticSourceConfig(
            mode="generated",
            width=int(args.width),
            height=int(args.height),
            fps=float(args.fps),
            num_frames=int(args.num_frames),
            dtype=str(args.dtype),
            bayer_pattern=str(args.bayer_pattern),
            ball_enabled=True,
        )
    )


def run_native_preprocessing(
    packet: FrameBatch4Cam,
    *,
    imgsz: int,
    device: str,
    half: bool,
    native_backend: str,
) -> PreprocessPathResult:
    stage_ms: dict = {}
    fallback_used = ""
    t0 = time.perf_counter()
    try:
        converter = GpuColorConverter(
            ColorConverterConfig(
                backend=native_backend,
                target_image_size=int(imgsz),
                output_color="BGR",
                normalize_01=False,
                half_precision=bool(half),
                device=device,
                output_mode="torch_cuda_bchw_rgb_norm",
            )
        )
        color_result = converter.convert_packet(packet)
        stage_ms.update(color_result.timings.to_dict())
        if color_result.gpu_batch is not None:
            tensor = color_result.gpu_batch.tensor
        else:
            fallback_used = "native_color_frames_to_unified_bchw"
            processor = GpuImageProcessor(
                ImageProcessorConfig(
                    target_size=int(imgsz),
                    input_color="BGR",
                    normalize=True,
                    device=device,
                    prefer_cuda=True,
                    half_precision=bool(half),
                )
            )
            proc_result = processor.process_batch([f.image for f in color_result.color_frames])
            tensor = proc_result.tensor
            for key, value in proc_result.stage_ms.items():
                stage_ms[f"adapter_{key}"] = float(value)
        if color_result.fallback_warnings:
            fallback_used = "; ".join([fallback_used] + list(color_result.fallback_warnings)).strip("; ")
        stage_ms["total_path_ms"] = (time.perf_counter() - t0) * 1000.0
        return _path_result("native", tensor, stage_ms, fallback_used=fallback_used)
    except Exception as exc:
        stage_ms["total_path_ms"] = (time.perf_counter() - t0) * 1000.0
        return PreprocessPathResult(
            path="native",
            tensor_shape=(),
            tensor_device="",
            tensor_dtype="",
            tensor_layout="BCHW",
            stage_ms=stage_ms,
            error=f"{type(exc).__name__}: {exc}",
        )


def run_unified_preprocessing(
    packet: FrameBatch4Cam,
    *,
    imgsz: int,
    device: str,
    half: bool,
    bayer_pattern: str,
) -> PreprocessPathResult:
    stage_ms: dict = {}
    t0 = time.perf_counter()
    try:
        processor = GpuImageProcessor(
            ImageProcessorConfig(
                target_size=int(imgsz),
                input_color="BAYER",
                bayer_pattern=str(bayer_pattern),
                normalize=True,
                device=device,
                prefer_cuda=True,
                half_precision=bool(half),
            )
        )
        proc_result = processor.process_batch([f.image for f in packet.frames])
        stage_ms.update(proc_result.stage_ms)
        stage_ms["total_path_ms"] = (time.perf_counter() - t0) * 1000.0
        return _path_result("unified", proc_result.tensor, stage_ms)
    except Exception as exc:
        stage_ms["total_path_ms"] = (time.perf_counter() - t0) * 1000.0
        return PreprocessPathResult(
            path="unified",
            tensor_shape=(),
            tensor_device="",
            tensor_dtype="",
            tensor_layout="BCHW",
            stage_ms=stage_ms,
            error=f"{type(exc).__name__}: {exc}",
        )


def compare_preprocessing_packet(
    packet: FrameBatch4Cam,
    *,
    imgsz: int,
    device: str = "auto",
    half: bool = False,
    native_backend: str = "cpu",
    bayer_pattern: str = "RG",
) -> PreprocessComparisonResult:
    native = run_native_preprocessing(
        packet,
        imgsz=imgsz,
        device=device,
        half=half,
        native_backend=native_backend,
    )
    unified = run_unified_preprocessing(
        packet,
        imgsz=imgsz,
        device=device,
        half=half,
        bayer_pattern=bayer_pattern,
    )
    same_shape = bool(native.tensor_shape and native.tensor_shape == unified.tensor_shape)
    bchw_compatible = bool(native.is_bchw and unified.is_bchw and same_shape)
    return PreprocessComparisonResult(
        packet_id=int(packet.packet_id),
        native=native,
        unified=unified,
        bchw_compatible=bchw_compatible,
        same_shape=same_shape,
    )


def run_comparison(args: argparse.Namespace) -> list[PreprocessComparisonResult]:
    source = build_synthetic_source(args)
    results: list[PreprocessComparisonResult] = []
    try:
        for packet in source.iter_packets():
            results.append(
                compare_preprocessing_packet(
                    packet,
                    imgsz=int(args.imgsz),
                    device=str(args.device),
                    half=bool(args.half),
                    native_backend=str(args.native_backend),
                    bayer_pattern=str(args.bayer_pattern),
                )
            )
    finally:
        source.close()
    return results


def write_comparison_outputs(
    output_dir: Path,
    args: argparse.Namespace,
    results: list[PreprocessComparisonResult],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in results:
        rows.extend(_rows_for_result(result))
    _write_csv(output_dir / "preprocessing_compare.csv", rows)
    summary = {
        "finished_utc": utc_now_iso(),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "packets_processed": len(results),
        "bchw_compatible_packets": sum(1 for r in results if r.bchw_compatible),
        "native_total_ms_summary": summarize_samples(
            [r.native.stage_ms.get("total_path_ms", 0.0) for r in results]
        ),
        "unified_total_ms_summary": summarize_samples(
            [r.unified.stage_ms.get("total_path_ms", 0.0) for r in results]
        ),
        "fallback_packets": sum(1 for r in results if r.native.fallback_used or r.unified.fallback_used),
        "errors": [
            r.to_dict() for r in results if r.native.error or r.unified.error
        ],
    }
    write_json(output_dir / "preprocessing_compare_summary.json", summary)
    write_json(output_dir / "preprocessing_compare_details.json", [r.to_dict() for r in results])


def _path_result(path: str, tensor, stage_ms: dict, fallback_used: str = "") -> PreprocessPathResult:
    return PreprocessPathResult(
        path=path,
        tensor_shape=_tensor_shape(tensor),
        tensor_device=_tensor_device(tensor),
        tensor_dtype=str(getattr(tensor, "dtype", "")).replace("torch.", ""),
        tensor_layout="BCHW",
        stage_ms={k: float(v) if isinstance(v, (int, float)) else v for k, v in stage_ms.items()},
        fallback_used=fallback_used,
    )


def _tensor_shape(tensor) -> tuple:
    try:
        return tuple(int(x) for x in getattr(tensor, "shape", ()))
    except Exception:
        return ()


def _tensor_device(tensor) -> str:
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


def _rows_for_result(result: PreprocessComparisonResult) -> list[dict]:
    rows = []
    for path_result in (result.native, result.unified):
        rows.append(
            {
                "packet_id": result.packet_id,
                "path": path_result.path,
                "tensor_shape": "x".join(str(x) for x in path_result.tensor_shape),
                "tensor_device": path_result.tensor_device,
                "tensor_dtype": path_result.tensor_dtype,
                "tensor_layout": path_result.tensor_layout,
                "is_bchw": path_result.is_bchw,
                "bchw_compatible": result.bchw_compatible,
                "same_shape": result.same_shape,
                "total_path_ms": round(float(path_result.stage_ms.get("total_path_ms", 0.0)), 4),
                "fallback_used": path_result.fallback_used,
                "error": path_result.error,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "packet_id",
        "path",
        "tensor_shape",
        "tensor_device",
        "tensor_dtype",
        "tensor_layout",
        "is_bchw",
        "bchw_compatible",
        "same_shape",
        "total_path_ms",
        "fallback_used",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    results = run_comparison(args)
    write_comparison_outputs(args.output_dir, args, results)
    return 1 if any(r.native.error or r.unified.error for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
