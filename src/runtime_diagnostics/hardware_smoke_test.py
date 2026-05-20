"""Synthetic hardware smoke test for CUDA/engine readiness.

This diagnostic tool does not change production defaults. It can run a CPU
preprocessing smoke without an engine, and it attempts a short fast-path run
only when CUDA and a valid engine path are available.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from src.runtime_benchmark.run_preprocessing_compare import (
    _parse_args as _parse_preprocess_args,
    run_comparison,
    write_comparison_outputs,
)
from src.runtime_diagnostics.runtime_readiness_check import check_runtime_readiness


def run_hardware_smoke_test(
    *,
    output_dir: str | Path,
    engine_path: str | Path | None = None,
    width: int = 128,
    height: int = 96,
    num_frames: int = 5,
    imgsz: int = 64,
    batch_size: int = 4,
    preprocess_backend: str = "compare",
    device: str = "auto",
    skip_inference: bool = False,
) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []

    readiness = check_runtime_readiness(engine_path)
    warnings.extend(readiness.get("status", {}).get("warnings", []))
    errors.extend(readiness.get("status", {}).get("errors", []))

    preprocess_report = _run_preprocess_smoke(
        out_dir=out_dir,
        width=width,
        height=height,
        num_frames=num_frames,
        imgsz=imgsz,
        preprocess_backend=preprocess_backend,
        device=device,
    )
    artifacts.extend(preprocess_report.get("artifacts", []))
    warnings.extend(preprocess_report.get("warnings", []))
    errors.extend(preprocess_report.get("errors", []))

    inference_report = {
        "ran": False,
        "status": "SKIPPED",
        "warning": "inference skipped",
    }
    if skip_inference:
        warnings.append("inference skipped by --skip-inference")
    else:
        inference_report = _run_fast_path_smoke(
            engine_path=engine_path,
            width=width,
            height=height,
            imgsz=imgsz,
            batch_size=batch_size,
            preprocess_backend=preprocess_backend,
            device=device,
        )
        warnings.extend(inference_report.get("warnings", []))
        errors.extend(inference_report.get("errors", []))

    final_status = _final_status(errors, warnings)
    report = {
        "readiness_summary": {
            "status": readiness.get("status", {}).get("recommendation", "NOT_READY"),
            "cuda_available": readiness.get("cuda", {}).get("available", False),
            "engine": readiness.get("engine", {}),
        },
        "preprocess_smoke": preprocess_report,
        "inference_smoke": inference_report,
        "warnings": warnings,
        "errors": errors,
        "artifacts": artifacts,
        "final_status": final_status,
    }
    json_path = write_hardware_smoke_json(report, out_dir)
    md_path = write_hardware_smoke_markdown(report, out_dir)
    report["artifacts"].extend([str(json_path), str(md_path)])
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(format_hardware_smoke_markdown(report), encoding="utf-8")
    return report


def write_hardware_smoke_json(report: dict, output_dir: str | Path) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "hardware_smoke_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_hardware_smoke_markdown(report: dict, output_dir: str | Path) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "hardware_smoke_report.md"
    path.write_text(format_hardware_smoke_markdown(report), encoding="utf-8")
    return path


def format_hardware_smoke_markdown(report: dict) -> str:
    lines = [
        "# Hardware Smoke Test",
        "",
        f"Final status: **{report.get('final_status', 'NOT_READY')}**",
        "",
        "## Readiness",
        "",
        f"- status: {report.get('readiness_summary', {}).get('status', 'NOT_READY')}",
        f"- cuda_available: {report.get('readiness_summary', {}).get('cuda_available', False)}",
        "",
        "## Preprocess Smoke",
        "",
    ]
    prep = report.get("preprocess_smoke", {})
    lines.append(f"- backend: {prep.get('backend', '')}")
    lines.append(f"- status: {prep.get('status', '')}")
    if prep.get("packets_processed") is not None:
        lines.append(f"- packets_processed: {prep.get('packets_processed')}")
    if prep.get("bchw_compatible_packets") is not None:
        lines.append(f"- bchw_compatible_packets: {prep.get('bchw_compatible_packets')}")
    lines.extend(["", "## Inference Smoke", ""])
    inf = report.get("inference_smoke", {})
    lines.append(f"- ran: {inf.get('ran', False)}")
    lines.append(f"- status: {inf.get('status', '')}")
    if inf.get("preprocessing_path"):
        lines.append(f"- preprocessing_path: {inf.get('preprocessing_path')}")
    if inf.get("tensor_device"):
        lines.append(f"- tensor_device: {inf.get('tensor_device')}")
    if inf.get("tensor_shape"):
        lines.append(f"- tensor_shape: {inf.get('tensor_shape')}")
    lines.extend(["", "## Warnings", ""])
    if report.get("warnings"):
        lines.extend(f"- {w}" for w in report["warnings"])
    else:
        lines.append("- none")
    lines.extend(["", "## Errors", ""])
    if report.get("errors"):
        lines.extend(f"- {e}" for e in report["errors"])
    else:
        lines.append("- none")
    lines.extend(["", "## Artifacts", ""])
    for artifact in report.get("artifacts", []):
        lines.append(f"- {artifact}")
    return "\n".join(lines) + "\n"


def _run_preprocess_smoke(
    *,
    out_dir: Path,
    width: int,
    height: int,
    num_frames: int,
    imgsz: int,
    preprocess_backend: str,
    device: str,
) -> dict:
    compare_dir = out_dir / "preprocessing_compare"
    args = _parse_preprocess_args([
        "--width", str(int(width)),
        "--height", str(int(height)),
        "--num-frames", str(int(num_frames)),
        "--imgsz", str(int(imgsz)),
        "--device", _device_for_preprocess(device),
        "--native-backend", "cpu",
        "--output-dir", str(compare_dir),
    ])
    try:
        results = run_comparison(args)
        write_comparison_outputs(compare_dir, args, results)
        artifacts = [
            str(compare_dir / "preprocessing_compare.csv"),
            str(compare_dir / "preprocessing_compare_summary.json"),
            str(compare_dir / "preprocessing_compare_details.json"),
        ]
        compatible = sum(1 for r in results if r.bchw_compatible)
        errors = [r.to_dict() for r in results if r.native.error or r.unified.error]
        selected_errors = _filter_preprocess_errors(errors, preprocess_backend)
        return {
            "ran": True,
            "backend": preprocess_backend,
            "status": "OK" if not selected_errors else "FAIL",
            "packets_processed": len(results),
            "bchw_compatible_packets": compatible,
            "artifacts": artifacts,
            "errors": [json.dumps(e, sort_keys=True) for e in selected_errors],
            "warnings": [],
        }
    except Exception as exc:
        return {
            "ran": True,
            "backend": preprocess_backend,
            "status": "FAIL",
            "artifacts": [],
            "warnings": [],
            "errors": [f"preprocess smoke failed: {type(exc).__name__}: {exc}"],
        }


def _run_fast_path_smoke(
    *,
    engine_path: str | Path | None,
    width: int,
    height: int,
    imgsz: int,
    batch_size: int,
    preprocess_backend: str,
    device: str,
) -> dict:
    if not engine_path:
        return {
            "ran": False,
            "status": "SKIPPED",
            "warnings": ["engine path not provided; inference smoke skipped"],
            "errors": [],
        }
    engine = Path(engine_path)
    if not engine.exists():
        return {
            "ran": False,
            "status": "SKIPPED",
            "warnings": [f"engine path does not exist; inference smoke skipped: {engine}"],
            "errors": [],
        }
    try:
        import torch  # type: ignore
        if not torch.cuda.is_available():
            return {
                "ran": False,
                "status": "SKIPPED",
                "warnings": ["CUDA unavailable; inference smoke skipped"],
                "errors": [],
            }
    except Exception as exc:
        return {
            "ran": False,
            "status": "SKIPPED",
            "warnings": [f"torch/CUDA check failed; inference smoke skipped: {type(exc).__name__}: {exc}"],
            "errors": [],
        }
    try:
        from src.runtime_production.fast_path_config import FastPathConfig
        from src.runtime_production.fast_path_executor import FastPathExecutor

        use_unified = preprocess_backend == "unified"
        cfg = FastPathConfig(
            enabled=True,
            engine_path=str(engine),
            imgsz=int(imgsz),
            batch_size=int(batch_size),
            device="cuda" if device == "auto" else str(device),
            use_unified_image_processor=bool(use_unified),
            warmup_iterations=0,
        )
        executor = FastPathExecutor(cfg)
        if not executor.initialize():
            return {
                "ran": False,
                "status": "SKIPPED",
                "warnings": [f"fast path initialize failed: {executor.init_error}"],
                "errors": [],
            }
        raws = [np.zeros((int(height), int(width)), dtype=np.uint8) for _ in range(int(batch_size))]
        roles = list(cfg.required_roles[: int(batch_size)])
        result = executor.process_packet(raws, roles, ts_ns=1)
        if result.error:
            return {
                "ran": True,
                "status": "FAIL",
                "warnings": [],
                "errors": [result.error],
                "stage_ms": result.stage_ms,
            }
        return {
            "ran": True,
            "status": "OK",
            "preprocessing_path": result.preprocessing_path,
            "tensor_device": result.tensor_device,
            "tensor_shape": list(result.tensor_shape),
            "tensor_layout": result.tensor_layout,
            "fallback_used": result.fallback_used,
            "stage_ms": result.stage_ms,
            "warnings": [],
            "errors": [],
        }
    except Exception as exc:
        return {
            "ran": False,
            "status": "SKIPPED",
            "warnings": [f"fast path smoke skipped after exception: {type(exc).__name__}: {exc}"],
            "errors": [],
        }


def _filter_preprocess_errors(errors: list[dict], preprocess_backend: str) -> list[dict]:
    if preprocess_backend == "compare":
        return errors
    filtered = []
    for entry in errors:
        item = entry.get(preprocess_backend, {})
        if item.get("error"):
            filtered.append(entry)
    return filtered


def _device_for_preprocess(device: str) -> str:
    if str(device).lower() == "auto":
        return "cpu"
    return str(device)


def _final_status(errors: list[str], warnings: list[str]) -> str:
    if errors:
        return "NOT_READY"
    if warnings:
        return "READY_WITH_WARNINGS"
    return "READY"


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VolleyHub synthetic hardware smoke test.")
    parser.add_argument("--engine-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/hardware_smoke"))
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--preprocess-backend", choices=("native", "unified", "compare"), default="compare")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--skip-inference", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    report = run_hardware_smoke_test(
        output_dir=args.output_dir,
        engine_path=args.engine_path,
        width=int(args.width),
        height=int(args.height),
        num_frames=int(args.num_frames),
        imgsz=int(args.imgsz),
        batch_size=int(args.batch_size),
        preprocess_backend=str(args.preprocess_backend),
        device=str(args.device),
        skip_inference=bool(args.skip_inference),
    )
    print(f"Hardware smoke final status: {report['final_status']}")
    for warning in report.get("warnings", []):
        print(f"[WARN] {warning}")
    for error in report.get("errors", []):
        print(f"[FAIL] {error}")
    for artifact in report.get("artifacts", []):
        print(f"artifact: {artifact}")
    return 2 if report["final_status"] == "NOT_READY" else 0


if __name__ == "__main__":
    raise SystemExit(main())
