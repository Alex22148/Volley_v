"""Per-process timing probe for the live production pipeline.

Design constraints:
- OFF by default. Only active when env VOLLEYHUB_LIVE_PROBE=1 is set.
- Zero behavior change in production when active. Probe failures must
  never propagate to the caller; every public method is wrapped to
  swallow exceptions defensively.
- Rolling window per stage (default 200 samples) for stable p95.
- One concise log line every N frames (default 30), to stderr/logger and
  optionally to logs/live_probe_<tag>.log so the report generator can
  parse it.
- Per-process: each subprocess that calls get_probe(tag) gets its own
  probe instance keyed by tag in that process. No cross-process state.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional

_LOG = logging.getLogger("volleyhub.live_probe")

_DEFAULT_WINDOW = 200
_DEFAULT_EVERY = 30
_DEFAULT_LOG_DIR = "logs"


def _is_enabled() -> bool:
    val = os.environ.get("VOLLEYHUB_LIVE_PROBE", "")
    return str(val).strip() in ("1", "true", "True", "yes", "on")


def _read_int_env(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, "")).strip())
    except Exception:
        return int(default)


class _Probe:
    """Thread-safe per-process timing collector."""

    def __init__(
        self,
        tag: str,
        log_every: int = _DEFAULT_EVERY,
        window: int = _DEFAULT_WINDOW,
        log_file: Optional[Path] = None,
    ) -> None:
        self.tag = str(tag)
        self._log_every = max(1, int(log_every))
        self._window = max(10, int(window))
        self._lock = threading.Lock()
        self._stages: Dict[str, Deque[float]] = {}
        self._extras: Dict[str, Any] = {}
        self._frames_seen = 0
        self._drops = 0
        self._enabled = _is_enabled()
        self._log_file = log_file

    @property
    def enabled(self) -> bool:
        return self._enabled

    def record_packet(self, stage_ms: Optional[Dict[str, float]] = None, **extras: Any) -> None:
        """Record one logical packet. Does nothing if probe is disabled."""
        if not self._enabled:
            return
        try:
            with self._lock:
                if stage_ms:
                    for k, v in stage_ms.items():
                        if k not in self._stages:
                            self._stages[k] = deque(maxlen=self._window)
                        try:
                            self._stages[k].append(float(v))
                        except Exception:
                            continue
                if extras:
                    for k, v in extras.items():
                        self._extras[str(k)] = v
                self._frames_seen += 1
                if self._frames_seen % self._log_every == 0:
                    self._emit_summary_locked()
        except Exception:
            return

    def record_drop(self, reason: str = "") -> None:
        if not self._enabled:
            return
        try:
            with self._lock:
                self._drops += 1
                self._extras["last_drop_reason"] = str(reason)
        except Exception:
            return

    def time_block(self, stage_name: str) -> "_StageContext":
        """Context manager that times a code block as `stage_name`."""
        return _StageContext(self, str(stage_name))

    def _emit_summary_locked(self) -> None:
        try:
            line = self._format_summary_locked()
        except Exception as exc:
            line = f"[LIVE_PACKET_TIMING] tag={self.tag} format_error={exc!r}"
        try:
            _LOG.info("%s", line)
        except Exception:
            pass
        if self._log_file is not None:
            try:
                self._log_file.parent.mkdir(parents=True, exist_ok=True)
                with self._log_file.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                pass

    def _format_summary_locked(self) -> str:
        parts = [
            f"tag={self.tag}",
            f"frames={self._frames_seen}",
            f"window={min(self._frames_seen, self._window)}",
        ]
        # extras first (resolution, role, backend, fast-path flags…)
        for k in (
            "role", "resolution", "color_backend", "inference_backend",
            "batch", "yolo_enabled", "yolo_imgsz", "trt_engine", "n_roles_seen",
            "fast_path_enabled", "packet_roles_count",
            "zero_copy_to_inference", "gpu_roundtrip", "fallback_used",
            "color_output_location", "inference_input_location",
            "preview_enabled", "preview_fps_target", "preview_size",
            "preview_quality", "preview_role", "preview_latest_only",
            "preview_blocks_fast_path", "preview_queue_drops", "frames_published",
            "ts_spread_ms",
        ):
            if k in self._extras:
                parts.append(f"{k}={self._extras[k]}")
        # per-stage stats
        median_parts = []
        p95_parts = []
        for stage, samples in sorted(self._stages.items()):
            if not samples:
                continue
            sorted_s = sorted(samples)
            n = len(sorted_s)
            median = sorted_s[n // 2] if n % 2 == 1 else (sorted_s[n // 2 - 1] + sorted_s[n // 2]) / 2.0
            if n >= 5:
                k = (n - 1) * 0.95
                f = int(k)
                c = min(f + 1, n - 1)
                p95 = sorted_s[f] + (sorted_s[c] - sorted_s[f]) * (k - f)
            else:
                p95 = sorted_s[-1]
            median_parts.append(f"{stage}={median:.2f}")
            p95_parts.append(f"{stage}={p95:.2f}")
        if median_parts:
            parts.append("median_ms=" + ",".join(median_parts))
        if p95_parts:
            parts.append("p95_ms=" + ",".join(p95_parts))
        parts.append(f"drops={self._drops}")
        return "[LIVE_PACKET_TIMING] " + " ".join(parts)


class _NoopProbe:
    """Used when probe is disabled. All methods are zero-cost no-ops."""

    enabled = False

    def record_packet(self, *_args: Any, **_kwargs: Any) -> None:
        return

    def record_drop(self, *_args: Any, **_kwargs: Any) -> None:
        return

    def time_block(self, _stage_name: str) -> "_StageContext":
        return _StageContext(self, _stage_name)


class _StageContext:
    """Tiny context manager: with probe.time_block('debayer'): ..."""

    __slots__ = ("_probe", "_stage", "_t0")

    def __init__(self, probe: Any, stage_name: str) -> None:
        self._probe = probe
        self._stage = stage_name
        self._t0 = 0.0

    def __enter__(self) -> "_StageContext":
        if getattr(self._probe, "enabled", False):
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            if getattr(self._probe, "enabled", False) and self._t0 > 0:
                elapsed_ms = (time.perf_counter() - self._t0) * 1000.0
                self._probe.record_packet(stage_ms={self._stage: elapsed_ms})
        except Exception:
            return None


_PROBES: Dict[str, Any] = {}
_PROBES_LOCK = threading.Lock()
_NOOP_SINGLETON = _NoopProbe()


def get_probe(tag: str, log_every: Optional[int] = None) -> Any:
    """Return a process-local probe for the given tag, or a no-op if disabled.

    Safe to call from any code path; it never raises.
    """
    try:
        if not _is_enabled():
            return _NOOP_SINGLETON
        with _PROBES_LOCK:
            if tag not in _PROBES:
                every = log_every if log_every is not None else _read_int_env("VOLLEYHUB_LIVE_PROBE_EVERY", _DEFAULT_EVERY)
                window = _read_int_env("VOLLEYHUB_LIVE_PROBE_WINDOW", _DEFAULT_WINDOW)
                log_dir = Path(os.environ.get("VOLLEYHUB_LIVE_PROBE_LOGDIR", _DEFAULT_LOG_DIR))
                pid = os.getpid()
                log_file = log_dir / f"live_probe_{tag}_pid{pid}.log"
                _PROBES[tag] = _Probe(tag, log_every=every, window=window, log_file=log_file)
            return _PROBES[tag]
    except Exception:
        return _NOOP_SINGLETON


def reset_probes() -> None:
    """Clear all probes (used by tests)."""
    try:
        with _PROBES_LOCK:
            _PROBES.clear()
    except Exception:
        return
