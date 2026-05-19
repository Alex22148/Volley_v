"""Helpers for the runtime benchmark: env capture, GPU stats, JSON/CSV writers."""
from __future__ import annotations

import csv
import json
import platform
import socket
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def collect_runtime_env() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": socket.gethostname(),
    }
    try:
        import numpy as np
        info["numpy_version"] = np.__version__
    except Exception:
        info["numpy_version"] = None
    try:
        import cv2
        info["opencv_version"] = cv2.__version__
        info["opencv_cuda_devices"] = _cv2_cuda_count()
    except Exception:
        info["opencv_version"] = None
        info["opencv_cuda_devices"] = 0
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["torch_cuda_device_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["torch_cuda_total_memory_mb"] = float(props.total_memory) / (1024 * 1024)
        else:
            info["torch_cuda_device_name"] = None
    except Exception:
        info["torch_version"] = None
        info["torch_cuda_available"] = False
    try:
        import ultralytics
        info["ultralytics_version"] = ultralytics.__version__
    except Exception:
        info["ultralytics_version"] = None
    return info


def _cv2_cuda_count() -> int:
    try:
        import cv2
        cuda = getattr(cv2, "cuda", None)
        if cuda is None:
            return 0
        count = getattr(cuda, "getCudaEnabledDeviceCount", None)
        if count is None:
            return 0
        return int(count())
    except Exception:
        return 0


@dataclass(slots=True)
class GpuStatsSample:
    allocated_mb: float = 0.0
    reserved_mb: float = 0.0
    utilization_pct: Optional[float] = None
    timestamp: float = 0.0


def sample_gpu_stats() -> GpuStatsSample:
    sample = GpuStatsSample(timestamp=_now_seconds())
    try:
        import torch
        if torch.cuda.is_available():
            sample.allocated_mb = float(torch.cuda.memory_allocated()) / (1024 * 1024)
            sample.reserved_mb = float(torch.cuda.memory_reserved()) / (1024 * 1024)
    except Exception:
        pass
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        sample.utilization_pct = float(util.gpu)
    except Exception:
        sample.utilization_pct = None
    return sample


def _now_seconds() -> float:
    import time
    return float(time.time())


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return repr(obj)


def write_frames_csv(path: Path, rows: Iterable[Dict[str, Any]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})


def summarize_samples(samples: List[float]) -> Dict[str, float]:
    if not samples:
        return {"count": 0, "mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "stdev_ms": 0.0}
    import statistics
    sorted_s = sorted(samples)
    n = len(sorted_s)

    def _percentile(p: float) -> float:
        if n == 1:
            return sorted_s[0]
        k = (n - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, n - 1)
        return sorted_s[f] + (sorted_s[c] - sorted_s[f]) * (k - f)

    return {
        "count": n,
        "mean_ms": float(statistics.fmean(sorted_s)),
        "median_ms": float(statistics.median(sorted_s)),
        "p95_ms": float(_percentile(95)),
        "p99_ms": float(_percentile(99)),
        "min_ms": float(sorted_s[0]),
        "max_ms": float(sorted_s[-1]),
        "stdev_ms": float(statistics.pstdev(sorted_s)) if n > 1 else 0.0,
    }
