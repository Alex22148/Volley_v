"""Aggregate runtime diagnostics into one operator report.

The report groups readiness, operator smoke, hardware smoke, preprocessing
compare status, artifacts, and an operator checklist. It is diagnostic-only and
does not change production defaults.
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
from src.runtime_diagnostics.hardware_smoke_test import run_hardware_smoke_test
from src.runtime_diagnostics.operator_smoke_test import run_operator_smoke_test
from src.runtime_diagnostics.runtime_readiness_check import check_runtime_readiness


def run_diagnostic_report(
    *,
    output_dir: str | Path,
    engine_path: str | Path | None = None,
    run_preprocess_compare: bool = False,
    run_hardware_smoke: bool = False,
    skip_inference: bool = False,
    width: int = 128,
    height: int = 96,
    num_frames: int = 5,
    imgsz: int = 64,
    batch_size: int = 4,
    device: str = "cpu",
    native_backend: str = "cpu",
    preprocess_backend: str = "compare",
) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []
    checks: dict = {}

    readiness = check_runtime_readiness(engine_path)
    readiness_status = _status_from_readiness(readiness)
    checks["readiness_check"] = {
        "status": readiness_status,
        "summary": readiness.get("status", {}),
    }
    warnings.extend(readiness.get("status", {}).get("warnings", []))
    errors.extend(readiness.get("status", {}).get("errors", []))

    operator_dir = out_dir / "operator_smoke"
    operator = run_operator_smoke_test(
        output_dir=operator_dir,
        engine_path=engine_path,
        run_preprocess_compare=run_preprocess_compare,
        width=width,
        height=height,
        num_frames=num_frames,
        imgsz=imgsz,
        device=device,
        native_backend=native_backend,
    )
    checks["operator_smoke"] = {
        "status": _status_from_final(operator.get("final_status")),
        "summary": operator.get("readiness_summary", {}),
    }
    artifacts.extend(operator.get("artifacts", []))

    preprocessing_summary = {"ran": False, "status": "SKIPPED"}
    if run_preprocess_compare:
        preprocess_dir = out_dir / "preprocessing_compare"
        preprocessing_summary = _run_preprocessing_compare(
            preprocess_dir=preprocess_dir,
            width=width,
            height=height,
            num_frames=num_frames,
            imgsz=imgsz,
            device=device,
            native_backend=native_backend,
        )
        artifacts.extend(preprocessing_summary.get("artifacts", []))
        warnings.extend(preprocessing_summary.get("warnings", []))
        errors.extend(preprocessing_summary.get("errors", []))
    checks["preprocessing_compare"] = {
        "status": preprocessing_summary.get("status", "SKIPPED"),
        "summary": preprocessing_summary,
    }

    hardware_summary = {"ran": False, "status": "SKIPPED", "reason": "hardware smoke not requested"}
    if run_hardware_smoke:
        hardware_dir = out_dir / "hardware_smoke"
        hardware = run_hardware_smoke_test(
            output_dir=hardware_dir,
            engine_path=engine_path,
            width=width,
            height=height,
            num_frames=num_frames,
            imgsz=imgsz,
            batch_size=batch_size,
            preprocess_backend=preprocess_backend,
            device=device,
            skip_inference=skip_inference,
        )
        hardware_summary = {
            "ran": True,
            "status": _status_from_hardware(hardware),
            "summary": {
                "final_status": hardware.get("final_status"),
                "inference_smoke": hardware.get("inference_smoke", {}),
                "preprocess_smoke": hardware.get("preprocess_smoke", {}),
            },
        }
        artifacts.extend(hardware.get("artifacts", []))
        warnings.extend(hardware.get("warnings", []))
        errors.extend(hardware.get("errors", []))
    checks["hardware_smoke"] = {
        "status": hardware_summary.get("status", "SKIPPED"),
        "summary": hardware_summary,
    }

    final_status = _final_status(checks, warnings, errors)
    report = {
        "summary": {
            "checks_total": len(checks),
            "checks_passed": sum(1 for c in checks.values() if c.get("status") == "PASSED"),
            "checks_warned": sum(1 for c in checks.values() if c.get("status") == "PASSED_WITH_WARNINGS"),
            "checks_skipped": sum(1 for c in checks.values() if c.get("status") == "SKIPPED"),
            "checks_failed": sum(1 for c in checks.values() if c.get("status") == "FAILED"),
        },
        "final_status": final_status,
        "checks": checks,
        "warnings": _dedupe(warnings),
        "errors": _dedupe(errors),
        "artifacts": _dedupe(artifacts),
        "recommended_next_steps": _recommended_next_steps(checks, warnings, errors),
        "operator_checklist": _build_operator_checklist(readiness, checks),
    }
    json_path = write_diagnostic_json(report, out_dir)
    md_path = write_diagnostic_markdown(report, out_dir)
    report["artifacts"].extend([str(json_path), str(md_path)])
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(format_diagnostic_markdown(report), encoding="utf-8")
    return report


def write_diagnostic_json(report: dict, output_dir: str | Path) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "diagnostic_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_diagnostic_markdown(report: dict, output_dir: str | Path) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "diagnostic_report.md"
    path.write_text(format_diagnostic_markdown(report), encoding="utf-8")
    return path


def format_diagnostic_markdown(report: dict) -> str:
    lines = [
        "# VolleyHub Diagnostic Report",
        "",
        f"Final status: **{report.get('final_status', 'NOT_READY')}**",
        "",
        "## Checks",
        "",
        "| check | status |",
        "|---|---|",
    ]
    for name, item in report.get("checks", {}).items():
        lines.append(f"| {name} | {item.get('status', 'FAILED')} |")
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
    lines.extend(["", "## Operator Checklist", ""])
    for item in report.get("operator_checklist", []):
        lines.append(f"- [{item.get('status')}] {item.get('label')}")
    lines.extend(["", "## Recommended Next Steps", ""])
    for step in report.get("recommended_next_steps", []):
        lines.append(f"- {step}")
    lines.extend(["", "## Artifacts", ""])
    for artifact in report.get("artifacts", []):
        lines.append(f"- {artifact}")
    return "\n".join(lines) + "\n"


def _run_preprocessing_compare(
    *,
    preprocess_dir: Path,
    width: int,
    height: int,
    num_frames: int,
    imgsz: int,
    device: str,
    native_backend: str,
) -> dict:
    args = _parse_preprocess_args([
        "--width", str(int(width)),
        "--height", str(int(height)),
        "--num-frames", str(int(num_frames)),
        "--imgsz", str(int(imgsz)),
        "--device", str(device),
        "--native-backend", str(native_backend),
        "--output-dir", str(preprocess_dir),
    ])
    try:
        results = run_comparison(args)
        write_comparison_outputs(preprocess_dir, args, results)
        errors = [r.to_dict() for r in results if r.native.error or r.unified.error]
        return {
            "ran": True,
            "status": "FAILED" if errors else "PASSED",
            "packets_processed": len(results),
            "bchw_compatible_packets": sum(1 for r in results if r.bchw_compatible),
            "artifacts": [
                str(preprocess_dir / "preprocessing_compare.csv"),
                str(preprocess_dir / "preprocessing_compare_summary.json"),
                str(preprocess_dir / "preprocessing_compare_details.json"),
            ],
            "warnings": [],
            "errors": [json.dumps(e, sort_keys=True) for e in errors],
        }
    except Exception as exc:
        return {
            "ran": True,
            "status": "FAILED",
            "artifacts": [],
            "warnings": [],
            "errors": [f"preprocessing compare failed: {type(exc).__name__}: {exc}"],
        }


def _status_from_readiness(readiness: dict) -> str:
    value = readiness.get("status", {}).get("recommendation", "NOT_READY")
    if value == "READY":
        return "PASSED"
    if value == "READY_WITH_WARNINGS":
        return "PASSED_WITH_WARNINGS"
    return "FAILED"


def _status_from_final(value) -> str:
    if value == "READY":
        return "PASSED"
    if value == "READY_WITH_WARNINGS":
        return "PASSED_WITH_WARNINGS"
    return "FAILED"


def _status_from_hardware(hardware: dict) -> str:
    if hardware.get("final_status") == "NOT_READY":
        return "FAILED"
    inference = hardware.get("inference_smoke", {})
    if inference.get("status") == "SKIPPED":
        return "SKIPPED"
    if hardware.get("final_status") == "READY_WITH_WARNINGS":
        return "PASSED_WITH_WARNINGS"
    return "PASSED"


def _final_status(checks: dict, warnings: list[str], errors: list[str]) -> str:
    if errors or any(c.get("status") == "FAILED" for c in checks.values()):
        return "NOT_READY"
    if warnings or any(c.get("status") in ("PASSED_WITH_WARNINGS", "SKIPPED") for c in checks.values()):
        return "READY_WITH_WARNINGS"
    return "READY"


def _build_operator_checklist(readiness: dict, checks: dict) -> list[dict]:
    packages = readiness.get("packages", {})
    cuda = readiness.get("cuda", {})
    engine = readiness.get("engine", {})
    return [
        _check_item("Python environment OK", True),
        _check_item("OpenCV import OK", packages.get("cv2", {}).get("available")),
        _check_item("torch import OK", packages.get("torch", {}).get("available")),
        _check_item("CUDA available", cuda.get("available"), skipped_ok=True),
        _check_item("TensorRT available", packages.get("tensorrt", {}).get("available"), skipped_ok=True),
        _check_item("pypylon available", packages.get("pypylon", {}).get("available"), skipped_ok=True),
        _check_item("engine provided", engine.get("provided"), skipped_ok=True),
        _check_item("preprocessing compare completed", checks.get("preprocessing_compare", {}).get("status") == "PASSED", skipped_ok=True),
        _check_item("hardware inference completed", checks.get("hardware_smoke", {}).get("status") == "PASSED", skipped_ok=True),
    ]


def _check_item(label: str, ok, skipped_ok: bool = False) -> dict:
    if ok:
        status = "PASSED"
    elif skipped_ok:
        status = "SKIPPED"
    else:
        status = "FAILED"
    return {"label": label, "status": status}


def _recommended_next_steps(checks: dict, warnings: list[str], errors: list[str]) -> list[str]:
    if errors:
        return ["Resolve FAILED checks before production run."]
    steps = []
    if checks.get("hardware_smoke", {}).get("status") == "SKIPPED":
        steps.append("Run hardware smoke with CUDA and --engine-path to validate real inference.")
    if checks.get("preprocessing_compare", {}).get("status") == "SKIPPED":
        steps.append("Run with --run-preprocess-compare to verify native vs unified BCHW compatibility.")
    if warnings:
        steps.append("Review warnings and decide which are acceptable for the target environment.")
    if not steps:
        steps.append("Environment is ready for the next controlled production smoke.")
    return steps


def _dedupe(items: list[str]) -> list[str]:
    out = []
    seen = set()
    for item in items:
        text = str(item)
        if text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VolleyHub aggregate diagnostic report.")
    parser.add_argument("--engine-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/diagnostic_report"))
    parser.add_argument("--run-preprocess-compare", action="store_true")
    parser.add_argument("--run-hardware-smoke", action="store_true")
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--native-backend", type=str, default="cpu")
    parser.add_argument("--preprocess-backend", choices=("native", "unified", "compare"), default="compare")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    report = run_diagnostic_report(
        output_dir=args.output_dir,
        engine_path=args.engine_path,
        run_preprocess_compare=bool(args.run_preprocess_compare),
        run_hardware_smoke=bool(args.run_hardware_smoke),
        skip_inference=bool(args.skip_inference),
        width=int(args.width),
        height=int(args.height),
        num_frames=int(args.num_frames),
        imgsz=int(args.imgsz),
        batch_size=int(args.batch_size),
        device=str(args.device),
        native_backend=str(args.native_backend),
        preprocess_backend=str(args.preprocess_backend),
    )
    print(f"Diagnostic final status: {report['final_status']}")
    for name, item in report.get("checks", {}).items():
        print(f"{name}: {item.get('status')}")
    for artifact in report.get("artifacts", []):
        print(f"artifact: {artifact}")
    return 2 if report["final_status"] == "NOT_READY" else 0


if __name__ == "__main__":
    raise SystemExit(main())
