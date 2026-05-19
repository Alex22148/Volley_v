"""Environment diagnostics — runs FIRST. Never crashes; reports OK/WARN/FAIL."""
from __future__ import annotations

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


def _check_torch() -> dict:
    out: dict = {"name": "torch", "level": "FAIL", "detail": ""}
    try:
        import torch  # type: ignore
        out["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            out["level"] = "OK"
            out["cuda_available"] = True
            out["cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None) or ""
            out["gpu_name"] = torch.cuda.get_device_name(0)
            try:
                props = torch.cuda.get_device_properties(0)
                out["gpu_vram_gb"] = round(float(props.total_memory) / (1024 ** 3), 2)
            except Exception:
                out["gpu_vram_gb"] = None
            out["detail"] = f"torch {torch.__version__}, CUDA {out['cuda_version']}, GPU {out['gpu_name']}"
        else:
            out["level"] = "FAIL"
            out["cuda_available"] = False
            out["detail"] = "CUDA_UNAVAILABLE: benchmark cannot run in GPU mode."
    except Exception as exc:
        out["detail"] = f"TORCH_IMPORT_FAILED: {exc!r}"
    return out


def _check_tensorrt() -> dict:
    out: dict = {"name": "tensorrt", "level": "FAIL", "detail": ""}
    try:
        import tensorrt  # type: ignore
        out["level"] = "OK"
        out["tensorrt_version"] = getattr(tensorrt, "__version__", "?")
        out["detail"] = f"tensorrt {out['tensorrt_version']}"
    except Exception as exc:
        out["detail"] = f"TENSORRT_MISSING: install TensorRT matching CUDA version. ({exc!r})"
    return out


def _check_pypylon() -> dict:
    out: dict = {"name": "pypylon", "level": "WARN", "detail": ""}
    try:
        import pypylon  # type: ignore
        out["level"] = "OK"
        out["detail"] = "pypylon available — real Basler benchmark can run"
    except Exception as exc:
        out["detail"] = f"PYPYLON_MISSING: synthetic benchmark can run, real Basler benchmark cannot. ({exc!r})"
    return out


def _check_opencv() -> dict:
    out: dict = {"name": "opencv", "level": "FAIL", "detail": ""}
    try:
        import cv2  # type: ignore
        out["level"] = "OK"
        out["opencv_version"] = cv2.__version__
        out["detail"] = f"opencv {cv2.__version__}"
    except Exception as exc:
        out["detail"] = f"OPENCV_MISSING: pip install opencv-python ({exc!r})"
    return out


def _check_native_backend() -> dict:
    out: dict = {"name": "native_cuda_debayer", "level": "WARN", "detail": ""}
    try:
        from src.runtime_gpu.native_debayer import NativeCudaDebayer
        info = NativeCudaDebayer.describe_backend()
        if info.get("available"):
            out["level"] = "OK"
            out["detail"] = f"available, detail={info.get('detail')}"
            out["native_backend_detail"] = info.get("detail")
        else:
            out["level"] = "WARN"
            out["detail"] = (
                "NATIVE_BACKEND_MISSING: build/install native CUDA extension. "
                f"reason: {info.get('error')!r}"
            )
    except Exception as exc:
        out["detail"] = f"NATIVE_BACKEND_IMPORT_FAILED: {exc!r}"
    return out


def _check_engines() -> dict:
    out: dict = {"name": "engines", "level": "WARN", "detail": "", "engines": []}
    expected = [
        "best__fp16_640_b4_static__fp16__img640__b4__static.engine",
        "best__fp16_960_b4_static__fp16__img960__b4__static.engine",
        "best__fp16_1280_b4_static__fp16__img1280__b4__static.engine",
        "best__fp16_1088x1920_b4_static__fp16__img1088x1920__b4__static.engine",
    ]
    eng_dir = PACKAGE_ROOT / "artifacts" / "tensorrt_exports"
    found = 0
    for fname in expected:
        p = eng_dir / fname
        present = p.exists() and p.stat().st_size > 1024
        out["engines"].append({"file": fname, "present": present,
                               "size_bytes": p.stat().st_size if p.exists() else 0})
        if present:
            found += 1
    if found == len(expected):
        out["level"] = "OK"
        out["detail"] = f"all {len(expected)} engines present"
    elif found == 0:
        out["level"] = "WARN"
        out["detail"] = (
            "ENGINE_MISSING: no engines found. Run "
            "`python scripts\\export_or_check_engines.py --build-missing` "
            "or copy them in."
        )
    else:
        out["level"] = "WARN"
        out["detail"] = f"{found}/{len(expected)} engines present; missing variants will be skipped"
    return out


def _check_dirs() -> dict:
    needed = ["src", "scripts", "configs", "artifacts/tensorrt_exports", "reports", "logs"]
    missing = []
    for d in needed:
        p = PACKAGE_ROOT / d
        p.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            missing.append(d)
    return {
        "name": "directories",
        "level": "OK" if not missing else "WARN",
        "detail": "all required directories exist" if not missing else f"missing: {missing}",
    }


def main() -> int:
    checks = [
        {"name": "python", "level": "OK",
         "detail": f"{platform.python_version()} on {platform.platform()}"},
        _check_torch(),
        _check_tensorrt(),
        _check_pypylon(),
        _check_opencv(),
        _check_native_backend(),
        _check_engines(),
        _check_dirs(),
    ]

    overall = "OK"
    for c in checks:
        if c.get("level") == "FAIL":
            overall = "FAIL"
            break
        if c.get("level") == "WARN" and overall != "FAIL":
            overall = "WARN"

    out_dir = PACKAGE_ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "package_root": str(PACKAGE_ROOT),
        "overall": overall,
        "checks": checks,
    }
    (out_dir / "environment_check.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )

    md_lines = [
        "# Environment check",
        "",
        f"Generated: {payload['generated_utc']}  ",
        f"Package: `{PACKAGE_ROOT}`",
        "",
        f"## Overall: **{overall}**",
        "",
        "| component | level | detail |",
        "|-----------|-------|--------|",
    ]
    for c in checks:
        md_lines.append(f"| {c['name']} | {c['level']} | {c.get('detail', '')} |")
    (out_dir / "environment_check.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    color = {"OK": "\033[32m", "WARN": "\033[33m", "FAIL": "\033[31m"}
    reset = "\033[0m"
    print()
    print("=" * 60)
    print(f" Environment check — overall: {overall}")
    print("=" * 60)
    for c in checks:
        lvl = c.get("level", "OK")
        prefix = f"  [{lvl:<4}]"
        try:
            sys.stdout.write(f"{color.get(lvl, '')}{prefix}{reset} {c['name']:<22} {c.get('detail','')}\n")
        except Exception:
            print(f"{prefix} {c['name']:<22} {c.get('detail','')}")
    print()
    print(f"  reports/environment_check.json")
    print(f"  reports/environment_check.md")
    return 0 if overall != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
