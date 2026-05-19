import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def utc_stamp_for_path() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def prepare_run_directory(base_dir: str | Path, mode: str) -> dict:
    base = Path(base_dir).resolve()
    run_dir = base / "benchmark_runs" / f"{utc_stamp_for_path()}_{mode}"
    cases_dir = run_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_dir": run_dir,
        "cases_dir": cases_dir,
        "summary_csv": run_dir / "benchmark_summary.csv",
        "cases_index_json": run_dir / "cases_index.json",
        "manifest_json": run_dir / "benchmark_manifest.json",
    }


def write_json(path: str | Path, payload: dict | list) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_summary_csv(path: str | Path, rows: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["case_id"])
        return
    fields: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def detect_git_commit(cwd: str | Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=1.5,
            check=False,
        )
        if proc.returncode == 0:
            sha = proc.stdout.strip()
            return sha or None
    except Exception:
        return None
    return None


def collect_runtime_env() -> dict:
    out = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": None,
        "ultralytics_version": None,
        "cuda_available": None,
        "cuda_version": None,
        "gpu_name": None,
    }

    try:
        import torch  # type: ignore

        out["torch_version"] = str(getattr(torch, "__version__", None))
        out["cuda_available"] = bool(torch.cuda.is_available())
        out["cuda_version"] = str(getattr(torch.version, "cuda", None))
        if torch.cuda.is_available():
            try:
                out["gpu_name"] = str(torch.cuda.get_device_name(0))
            except Exception:
                pass
    except Exception:
        pass

    try:
        import ultralytics  # type: ignore

        out["ultralytics_version"] = str(getattr(ultralytics, "__version__", None))
    except Exception:
        pass

    return out

