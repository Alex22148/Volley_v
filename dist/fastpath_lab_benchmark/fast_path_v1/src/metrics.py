"""Tiny metrics module: percentiles + the benchmark verdict.

"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence


@dataclass(slots=True)
class StageStats:
    name: str
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    mean_ms: float
    stdev_ms: float
    count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "median_ms": float(self.median_ms),
            "p95_ms": float(self.p95_ms),
            "min_ms": float(self.min_ms),
            "max_ms": float(self.max_ms),
            "mean_ms": float(self.mean_ms),
            "stdev_ms": float(self.stdev_ms),
            "count": int(self.count),
        }


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return float(values[0])
    sorted_v = sorted(values)
    n = len(sorted_v)
    pos = q * (n - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_v[lo])
    frac = pos - lo
    return float(sorted_v[lo] * (1.0 - frac) + sorted_v[hi] * frac)


def stage_stats(name: str, samples: Sequence[float]) -> StageStats:
    if not samples:
        return StageStats(name=name, median_ms=float("nan"), p95_ms=float("nan"),
                          min_ms=float("nan"), max_ms=float("nan"),
                          mean_ms=float("nan"), stdev_ms=float("nan"), count=0)
    s = sorted(samples)
    n = len(s)
    median = percentile(s, 0.50)
    p95 = percentile(s, 0.95)
    mean = sum(s) / n
    var = sum((x - mean) ** 2 for x in s) / n
    return StageStats(
        name=name,
        median_ms=median,
        p95_ms=p95,
        min_ms=float(s[0]),
        max_ms=float(s[-1]),
        mean_ms=mean,
        stdev_ms=math.sqrt(var),
        count=n,
    )


# ------------------------------------------------------------- FPS verdicts

TARGET_FPS = 50.0
TARGET_PACKET_MS = 1000.0 / TARGET_FPS  # 20.0 ms


def packet_to_fps_per_camera(packet_ms: float) -> float:
    if packet_ms <= 0 or not math.isfinite(packet_ms):
        return float("nan")
    return 1000.0 / packet_ms


def verdict_from_packet_stats(
    packet_stats: StageStats,
    *,
    batch: int = 4,
    target_fps: float = TARGET_FPS,
) -> Dict[str, Any]:
    fps_median = packet_to_fps_per_camera(packet_stats.median_ms)
    fps_p95 = packet_to_fps_per_camera(packet_stats.p95_ms)
    pass_50 = bool(
        math.isfinite(fps_p95) and fps_p95 >= target_fps
    )
    target_packet_ms = 1000.0 / target_fps
    stability_margin = target_packet_ms - packet_stats.p95_ms
    return {
        "packet_ms_median": packet_stats.median_ms,
        "packet_ms_p95": packet_stats.p95_ms,
        "fps_per_camera_median": fps_median,
        "fps_per_camera_safe_p95": fps_p95,
        "total_images_per_second_safe_p95": (fps_p95 * batch) if math.isfinite(fps_p95) else float("nan"),
        "target_fps": target_fps,
        "target_packet_ms": target_packet_ms,
        "pass_50fps_safe_p95": pass_50,
        "stability_margin_ms": stability_margin,
        "batch": batch,
    }


def aggregate_run(
    *,
    packet_samples_ms: Sequence[float],
    stage_samples_ms: Optional[Dict[str, Sequence[float]]] = None,
    batch: int = 4,
    target_fps: float = TARGET_FPS,
    warmup_skipped: int = 0,
) -> Dict[str, Any]:
    packet_stats = stage_stats("packet_ms", packet_samples_ms)
    out: Dict[str, Any] = {
        "warmup_skipped": int(warmup_skipped),
        "iterations_kept": packet_stats.count,
        "packet_stats": packet_stats.to_dict(),
    }
    out["verdict"] = verdict_from_packet_stats(packet_stats, batch=batch, target_fps=target_fps)
    if stage_samples_ms:
        stages: Dict[str, Any] = {}
        for name, samples in stage_samples_ms.items():
            stages[name] = stage_stats(name, samples).to_dict()
        out["stage_stats"] = stages
    return out
