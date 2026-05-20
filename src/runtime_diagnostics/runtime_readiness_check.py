"""Runtime readiness check for operator smoke tests.

The check is intentionally diagnostic-only. Missing optional packages such as
CUDA, TensorRT, or pypylon are reported as warnings and never crash the tool.
"""
from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Optional


_RUNTIME_MODULES = (
    "src.runtime_gpu.gpu_image_processor",
    "src.runtime_gpu.gpu_color_converter",
    "src.runtime_production.fast_path_executor",
    "src.runtime_benchmark.run_preprocessing_compare",
)


def check_runtime_readiness(engine_path: str | Path | None = None) -> dict:
    report = {
        "python": _check_python(),
        "platform": _check_platform(),
        "packages": _check_packages(),
        "cuda": _check_cuda(),
        "modules": _check_modules(_RUNTIME_MODULES),
        "engine": _check_engine(engine_path),
        "status": {},
    }
    report["status"] = _build_status(report)
    return report


def write_readiness_json(report: dict, output_dir: str | Path) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "runtime_readiness.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_readiness_markdown(report: dict, output_dir: str | Path) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "runtime_readiness.md"
    path.write_text(format_readiness_markdown(report), encoding="utf-8")
    return path


def format_readiness_markdown(report: dict) -> str:
    status = report.get("status", {})
    lines = [
        "# Runtime Readiness Check",
        "",
        f"Status: **{status.get('recommendation', 'NOT_READY')}**",
        "",
        "## Python",
        "",
        f"- version: {report.get('python', {}).get('version', '')}",
        f"- executable: {report.get('python', {}).get('executable', '')}",
        "",
        "## Platform",
        "",
        f"- system: {report.get('platform', {}).get('system', '')}",
        f"- release: {report.get('platform', {}).get('release', '')}",
        f"- machine: {report.get('platform', {}).get('machine', '')}",
        "",
        "## Packages",
        "",
    ]
    for name, item in sorted(report.get("packages", {}).items()):
        lines.append(f"- {name}: {_status_text(item)}")
    lines.extend(["", "## CUDA", ""])
    cuda = report.get("cuda", {})
    lines.append(f"- torch available: {cuda.get('torch_available')}")
    lines.append(f"- cuda available: {cuda.get('available')}")
    if cuda.get("device_name"):
        lines.append(f"- device: {cuda.get('device_name')}")
    lines.extend(["", "## Modules", ""])
    for name, item in sorted(report.get("modules", {}).items()):
        lines.append(f"- {name}: {_status_text(item)}")
    lines.extend(["", "## Engine", ""])
    lines.append(f"- path: {report.get('engine', {}).get('path') or '(not provided)'}")
    lines.append(f"- status: {_status_text(report.get('engine', {}))}")
    lines.extend(["", "## Warnings", ""])
    warnings = status.get("warnings", [])
    if warnings:
        lines.extend(f"- {w}" for w in warnings)
    else:
        lines.append("- none")
    lines.extend(["", "## Errors", ""])
    errors = status.get("errors", [])
    if errors:
        lines.extend(f"- {e}" for e in errors)
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def _check_python() -> dict:
    return {
        "ok": True,
        "version": sys.version.replace("\n", " "),
        "version_info": list(sys.version_info[:3]),
        "executable": sys.executable,
    }


def _check_platform() -> dict:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "platform": platform.platform(),
    }


def _check_packages() -> dict:
    return {
        "numpy": _check_import("numpy"),
        "cv2": _check_import("cv2"),
        "torch": _check_import("torch"),
        "tensorrt": _check_import("tensorrt", optional=True),
        "pypylon": _check_import("pypylon", optional=True),
    }


def _check_cuda() -> dict:
    item = {
        "torch_available": False,
        "available": False,
        "device_count": 0,
        "device_name": "",
        "warning": "",
    }
    try:
        import torch  # type: ignore
    except Exception as exc:
        item["warning"] = f"torch import failed: {type(exc).__name__}: {exc}"
        return item
    item["torch_available"] = True
    try:
        item["available"] = bool(torch.cuda.is_available())
        if item["available"]:
            item["device_count"] = int(torch.cuda.device_count())
            item["device_name"] = str(torch.cuda.get_device_name(0))
        else:
            item["warning"] = "CUDA not available"
    except Exception as exc:
        item["warning"] = f"CUDA check failed: {type(exc).__name__}: {exc}"
    return item


def _check_modules(module_names) -> dict:
    return {name: _check_import(name) for name in module_names}


def _check_engine(engine_path: str | Path | None) -> dict:
    if not engine_path:
        return {
            "ok": True,
            "provided": False,
            "path": "",
            "exists": False,
            "size_bytes": 0,
            "warning": "engine path not provided",
        }
    path = Path(engine_path)
    item = {
        "ok": False,
        "provided": True,
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": 0,
        "warning": "",
        "error": "",
    }
    if not path.exists():
        item["warning"] = f"engine path does not exist: {path}"
        return item
    try:
        item["size_bytes"] = int(path.stat().st_size)
    except Exception as exc:
        item["warning"] = f"could not stat engine path: {type(exc).__name__}: {exc}"
        return item
    if item["size_bytes"] < 1024:
        item["warning"] = f"engine file is small: {item['size_bytes']} bytes"
    item["ok"] = True
    return item


def _check_import(module_name: str, optional: bool = False) -> dict:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        item = {
            "ok": bool(optional),
            "available": False,
            "optional": bool(optional),
            "warning": f"{module_name} unavailable: {type(exc).__name__}: {exc}",
        }
        if not optional:
            item["error"] = item["warning"]
        return item
    return {
        "ok": True,
        "available": True,
        "optional": bool(optional),
        "version": str(getattr(module, "__version__", "")),
    }


def _build_status(report: dict) -> dict:
    warnings = []
    errors = []
    packages = report.get("packages", {})
    for required in ("numpy", "cv2", "torch"):
        item = packages.get(required, {})
        if not item.get("available"):
            errors.append(item.get("error") or item.get("warning") or f"{required} unavailable")
    for optional in ("tensorrt", "pypylon"):
        item = packages.get(optional, {})
        if not item.get("available"):
            warnings.append(item.get("warning") or f"{optional} unavailable")
    cuda = report.get("cuda", {})
    if not cuda.get("available"):
        warnings.append(cuda.get("warning") or "CUDA not available")
    for name, item in report.get("modules", {}).items():
        if not item.get("available"):
            errors.append(item.get("error") or item.get("warning") or f"{name} import failed")
    engine = report.get("engine", {})
    if engine.get("warning"):
        warnings.append(engine["warning"])
    if engine.get("error"):
        errors.append(engine["error"])
    if errors:
        recommendation = "NOT_READY"
    elif warnings:
        recommendation = "READY_WITH_WARNINGS"
    else:
        recommendation = "READY"
    return {
        "recommendation": recommendation,
        "warnings": warnings,
        "errors": errors,
    }


def _status_text(item: dict) -> str:
    if item.get("available") or item.get("ok"):
        version = item.get("version")
        return f"OK ({version})" if version else "OK"
    if item.get("error"):
        return f"FAIL - {item['error']}"
    if item.get("warning"):
        return f"WARN - {item['warning']}"
    return "WARN"


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VolleyHub runtime readiness check.")
    parser.add_argument("--engine-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/runtime_readiness"))
    parser.add_argument("--json", action="store_true", help="Write JSON report.")
    parser.add_argument("--markdown", action="store_true", help="Write Markdown report.")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    report = check_runtime_readiness(args.engine_path)
    status = report["status"]["recommendation"]
    print(f"Runtime readiness: {status}")
    for warning in report["status"].get("warnings", []):
        print(f"[WARN] {warning}")
    for error in report["status"].get("errors", []):
        print(f"[FAIL] {error}")
    if args.json:
        print(f"wrote: {write_readiness_json(report, args.output_dir)}")
    if args.markdown:
        print(f"wrote: {write_readiness_markdown(report, args.output_dir)}")
    return 2 if status == "NOT_READY" else 0


if __name__ == "__main__":
    raise SystemExit(main())
