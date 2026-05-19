from typing import Iterable

import numpy as np


def safe_percentile(values: Iterable[float], q: float) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, q))


def summarize(values: Iterable[float]) -> dict:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "cv": 0.0,
        }
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    return {
        "count": int(arr.size),
        "mean": mean,
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "std": std,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "cv": float(std / mean) if abs(mean) > 1e-12 else 0.0,
    }


def slope_per_step(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size < 2:
        return 0.0
    x = np.arange(arr.size, dtype=float)
    if float(np.var(x)) < 1e-12:
        return 0.0
    return float(np.polyfit(x, arr, 1)[0])


def downsample(values: Iterable[float], max_points: int = 400) -> list[float]:
    vals = list(values)
    if max_points <= 0 or len(vals) <= max_points:
        return vals
    step = max(1, int(np.ceil(len(vals) / float(max_points))))
    reduced = vals[::step]
    if reduced and reduced[-1] != vals[-1]:
        reduced.append(vals[-1])
    return reduced
