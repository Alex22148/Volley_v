"""00_check_env.py â€” environment check for the lab benchmark.

WHAT THIS SCRIPT DOES:
    * Prints Python version, OS, torch / CUDA / TensorRT versions,
      GPU name + VRAM, whether the custom CUDA debayer can be loaded,
      and which engine files exist under engines/.
    * Writes results/environment.json and results/environment.md so
      the report can include this info verbatim.

WHAT THIS SCRIPT DOES NOT DO:
    * It does not run inference, debayer, or any benchmark loop.
    * It does not touch a camera, file dataset, or production code.
    * It does not create or modify engine files.
"""
from __future__ import annotations

import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path

# Make src/ importable when running as a script.
PKG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_ROOT))

from src.native_debayer import NativeCudaDebayer  # noqa: E402

# ----------------------------- static config (no argparse)
RESULTS_DIR = PKG_ROOT / "reports"
ENGINES_DIR = PKG_ROOT / "engines"
RUN_BENCHMARK_AFTER_CHECK = False


def _check(name: str, ok: bool, detail: str = "") -> dict:
    status = "OK" if ok else "FAIL"
    line = f"[{status}] {name}"
    if detail:
        line += f" â€” {detail}"
    print(line)
    return {"name": name, "status": status, "detail": detail}


def main() -> int:
    print("=" * 72)
    print("fastpath_lab_benchmark / 00_check_env.py")
    print("WHAT: environment check (Python, CUDA, TensorRT, GPU, engines).")
    print("WHAT NOT: no benchmarks are run here.")
    print("=" * 72)

    checks = []
    machine: dict = {}

    machine["python_version"] = sys.version.split()[0]
    machine["os"] = f"{platform.system()} {platform.release()} ({platform.machine()})"
    checks.append(_check("Python", sys.version_info >= (3, 9),
                         detail=machine["python_version"]))
    checks.append(_check("OS", True, detail=machine["os"]))

    # torch
    try:
        torch = importlib.import_module("torch")
        machine["torch_version"] = getattr(torch, "__version__", "?")
        cuda_ok = bool(torch.cuda.is_available())
        machine["cuda_available"] = cuda_ok
        machine["cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None) or ""
        if cuda_ok:
            machine["gpu_name"] = torch.cuda.get_device_name(0)
            try:
                props = torch.cuda.get_device_properties(0)
                machine["vram_gb"] = round(props.total_memory / (1024 ** 3), 2)
                machine["compute_capability"] = f"{props.major}.{props.minor}"
            except Exception as exc:
                machine["vram_gb"] = None
                machine["vram_error"] = str(exc)
        checks.append(_check("torch", True, detail=machine["torch_version"]))
        checks.append(_check("CUDA available", cuda_ok,
                             detail=machine.get("gpu_name", machine.get("cuda_version", "no"))))
    except Exception as exc:
        checks.append(_check("torch", False, detail=f"import failed: {exc!r}"))
        machine["torch_error"] = str(exc)

    # tensorrt
    try:
        trt = importlib.import_module("tensorrt")
        machine["trt_version"] = getattr(trt, "__version__", "?")
        checks.append(_check("tensorrt", True, detail=machine["trt_version"]))
    except Exception as exc:
        machine["trt_version"] = None
        machine["trt_error"] = str(exc)
        checks.append(_check("tensorrt", False, detail=f"import failed: {exc!r}"))

    # native CUDA debayer
    try:
        backend_info = NativeCudaDebayer.describe_backend()
        avail = bool(backend_info.get("available"))
        machine["native_cuda_debayer"] = backend_info
        detail = backend_info.get("detail") or backend_info.get("error") or "â€”"
        checks.append(_check("native CUDA debayer", avail, detail=detail))
        if avail:
            print("    (extension at: {})".format(backend_info.get("extension_path", "?")))
    except Exception as exc:
        machine["native_cuda_debayer_error"] = str(exc)
        checks.append(_check("native CUDA debayer", False, detail=f"error: {exc!r}"))

    # matplotlib (optional)
    try:
        mpl = importlib.import_module("matplotlib")
        machine["matplotlib_version"] = getattr(mpl, "__version__", "?")
        checks.append(_check("matplotlib (for charts)", True,
                             detail=machine["matplotlib_version"]))
    except Exception:
        machine["matplotlib_version"] = None
        checks.append(_check("matplotlib (for charts)", False,
                             detail="not installed â€” report will skip charts"))

    # engines
    engines_dir = Path(ENGINES_DIR)
    expected = [
        "best__fp16_640x640_b4.engine",
        "best__fp16_960x960_b4.engine",
        "best__fp16_1280x1280_b4.engine",
        "best__fp16_1088x1920_b4.engine",
    ]
    engine_info = {}
    for name in expected:
        p = engines_dir / name
        if p.exists():
            engine_info[name] = {"path": str(p), "size_bytes": p.stat().st_size}
            checks.append(_check(f"engine {name}", True,
                                 detail=f"{p.stat().st_size / (1024*1024):.1f} MiB"))
        else:
            engine_info[name] = {"path": str(p), "size_bytes": None}
            checks.append(_check(f"engine {name}", False, detail="missing"))
    machine["engines"] = engine_info

    payload = {"checks": checks, "machine": machine, "engines_dir": str(engines_dir)}

    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "environment.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")

    md_lines = ["# Environment", ""]
    md_lines.append("| key | value |")
    md_lines.append("|---|---|")
    for k, v in machine.items():
        if isinstance(v, dict):
            continue
        md_lines.append(f"| {k} | {v} |")
    md_lines.append("")
    md_lines.append("| check | status | detail |")
    md_lines.append("|---|---|---|")
    for c in checks:
        md_lines.append(f"| {c['name']} | {c['status']} | {c['detail']} |")
    (results_dir / "environment.md").write_text("\n".join(md_lines), encoding="utf-8")

    print()
    print(f"-> wrote {results_dir / 'environment.json'}")
    print(f"-> wrote {results_dir / 'environment.md'}")

    # exit code: non-zero only if critical things missing
    critical_failed = any(
        c["name"] in ("torch", "CUDA available", "tensorrt") and c["status"] == "FAIL"
        for c in checks
    )

    if RUN_BENCHMARK_AFTER_CHECK:
        benchmark_script = Path(__file__).resolve().parent / "08_final_benchmark.py"
        if critical_failed:
            print("[BENCHMARK] RUN_BENCHMARK_AFTER_CHECK=True, ale pomijam uruchomienie benchmarku (critical checks failed).")
        elif not benchmark_script.exists():
            print(f"[BENCHMARK] WARN: nie znaleziono skryptu benchmarku: {benchmark_script}")
        else:
            print("[BENCHMARK] RUN_BENCHMARK_AFTER_CHECK=True -> uruchamiam 08_final_benchmark.py ...")
            proc = subprocess.run([sys.executable, str(benchmark_script)])
            if proc.returncode != 0:
                print(f"[BENCHMARK] ERROR: 08_final_benchmark.py zakończył się kodem {proc.returncode}")
            else:
                print("[BENCHMARK] OK: benchmark uruchomiony po check_env.")

    return 1 if critical_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())


