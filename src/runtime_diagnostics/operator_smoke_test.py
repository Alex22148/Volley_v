"""Operator smoke test: readiness + optional preprocessing comparison.

This tool is diagnostic-only. It does not change production defaults and does
not require CUDA, TensorRT, pypylon, or a model file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from src.runtime_benchmark.run_preprocessing_compare import (
    _parse_args as _parse_preprocess_args,
    run_comparison,
    write_comparison_outputs,
)
from src.runtime_diagnostics.runtime_readiness_check import (
    check_runtime_readiness,
    format_readiness_markdown,
    write_readiness_json,
    write_readiness_markdown,
)


def run_operator_smoke_test(
    *,
    output_dir: str | Path,
    engine_path: str | Path | None = None,
    run_preprocess_compare: bool = False,
    width: int = 128,
    height: int = 96,
    num_frames: int = 5,
    imgsz: int = 64,
    device: str = "cpu",
    native_backend: str = "cpu",
) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[str] = []
    readiness = check_runtime_readiness(engine_path)
    artifacts.append(str(write_readiness_json(readiness, out_dir)))
    artifacts.append(str(write_readiness_markdown(readiness, out_dir)))

    preprocess_summary = {
        "ran": False,
        "status": "SKIPPED",
        "warning": "preprocessing compare not requested",
    }
    if run_preprocess_compare:
        compare_dir = out_dir / "preprocessing_compare"
        compare_args = _parse_preprocess_args([
            "--width", str(int(width)),
            "--height", str(int(height)),
            "--num-frames", str(int(num_frames)),
            "--imgsz", str(int(imgsz)),
            "--device", str(device),
            "--native-backend", str(native_backend),
            "--output-dir", str(compare_dir),
        ])
        try:
            compare_results = run_comparison(compare_args)
            write_comparison_outputs(compare_dir, compare_args, compare_results)
            artifacts.extend([
                str(compare_dir / "preprocessing_compare.csv"),
                str(compare_dir / "preprocessing_compare_summary.json"),
                str(compare_dir / "preprocessing_compare_details.json"),
            ])
            compatible = sum(1 for r in compare_results if r.bchw_compatible)
            errors = [
                r.to_dict() for r in compare_results
                if r.native.error or r.unified.error
            ]
            preprocess_summary = {
                "ran": True,
                "status": "OK" if not errors else "FAIL",
                "packets_processed": len(compare_results),
                "bchw_compatible_packets": compatible,
                "errors": errors,
                "output_dir": str(compare_dir),
            }
        except Exception as exc:
            preprocess_summary = {
                "ran": True,
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
                "output_dir": str(compare_dir),
            }

    warnings = list(readiness.get("status", {}).get("warnings", []))
    errors = list(readiness.get("status", {}).get("errors", []))
    if preprocess_summary.get("status") == "FAIL":
        errors.append("preprocessing compare failed")
    elif preprocess_summary.get("status") == "SKIPPED":
        warnings.append(str(preprocess_summary.get("warning", "preprocessing compare skipped")))

    final_status = _final_status(errors, warnings)
    report = {
        "readiness_summary": {
            "status": readiness.get("status", {}).get("recommendation", "NOT_READY"),
            "warnings": readiness.get("status", {}).get("warnings", []),
            "errors": readiness.get("status", {}).get("errors", []),
        },
        "readiness": readiness,
        "preprocessing_compare_summary": preprocess_summary,
        "environment_warnings": warnings,
        "artifacts": artifacts,
        "final_status": final_status,
    }

    summary_json = write_operator_smoke_json(report, out_dir)
    summary_md = write_operator_smoke_markdown(report, out_dir)
    report["artifacts"].extend([str(summary_json), str(summary_md)])
    summary_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    summary_md.write_text(format_operator_smoke_markdown(report), encoding="utf-8")
    return report


def write_operator_smoke_json(report: dict, output_dir: str | Path) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "operator_smoke_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_operator_smoke_markdown(report: dict, output_dir: str | Path) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "operator_smoke_report.md"
    path.write_text(format_operator_smoke_markdown(report), encoding="utf-8")
    return path


def format_operator_smoke_markdown(report: dict) -> str:
    lines = [
        "# Operator Smoke Test",
        "",
        f"Final status: **{report.get('final_status', 'NOT_READY')}**",
        "",
        "## Readiness Summary",
        "",
        f"- status: {report.get('readiness_summary', {}).get('status', 'NOT_READY')}",
        f"- warnings: {len(report.get('readiness_summary', {}).get('warnings', []))}",
        f"- errors: {len(report.get('readiness_summary', {}).get('errors', []))}",
        "",
        "## Preprocessing Compare",
        "",
    ]
    pc = report.get("preprocessing_compare_summary", {})
    lines.append(f"- ran: {pc.get('ran', False)}")
    lines.append(f"- status: {pc.get('status', 'UNKNOWN')}")
    if pc.get("packets_processed") is not None:
        lines.append(f"- packets_processed: {pc.get('packets_processed')}")
    if pc.get("bchw_compatible_packets") is not None:
        lines.append(f"- bchw_compatible_packets: {pc.get('bchw_compatible_packets')}")
    if pc.get("error"):
        lines.append(f"- error: {pc.get('error')}")
    lines.extend(["", "## Environment Warnings", ""])
    warnings = report.get("environment_warnings", [])
    if warnings:
        lines.extend(f"- {w}" for w in warnings)
    else:
        lines.append("- none")
    lines.extend(["", "## Artifacts", ""])
    for artifact in report.get("artifacts", []):
        lines.append(f"- {artifact}")
    lines.extend(["", "## Full Readiness", ""])
    lines.append(format_readiness_markdown(report.get("readiness", {})))
    return "\n".join(lines)


def _final_status(errors: list[str], warnings: list[str]) -> str:
    if errors:
        return "NOT_READY"
    if warnings:
        return "READY_WITH_WARNINGS"
    return "READY"


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VolleyHub operator smoke test.")
    parser.add_argument("--engine-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/operator_smoke"))
    parser.add_argument("--run-preprocess-compare", action="store_true")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--native-backend", type=str, default="cpu")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    report = run_operator_smoke_test(
        output_dir=args.output_dir,
        engine_path=args.engine_path,
        run_preprocess_compare=bool(args.run_preprocess_compare),
        width=int(args.width),
        height=int(args.height),
        num_frames=int(args.num_frames),
        imgsz=int(args.imgsz),
        device=str(args.device),
        native_backend=str(args.native_backend),
    )
    print(f"Operator smoke final status: {report['final_status']}")
    for warning in report.get("environment_warnings", []):
        print(f"[WARN] {warning}")
    for artifact in report.get("artifacts", []):
        print(f"artifact: {artifact}")
    return 2 if report["final_status"] == "NOT_READY" else 0


if __name__ == "__main__":
    raise SystemExit(main())
