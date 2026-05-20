"""Lightweight GUI preview worker for the production fast path.

Strict contract:
  - Runs in its OWN thread. Cannot block the fast-path inference loop.
  - Reads RAW Bayer from shared memory (just keys), no CUDA download.
  - NEVER does a full 1920x1080 CPU debayer. Decimates first, then debayers
    the small frame (or produces a fast grayscale-ish preview).
  - Latest-frame-only: if the GUI consumer falls behind, old frames drop.
  - Throttled to PREVIEW_MAX_FPS regardless of how fast frames arrive.

Configuration (env vars):
  VOLLEYHUB_PREVIEW_MAX_FPS    default 20
  VOLLEYHUB_PREVIEW_SIZE       default 512   (target width in pixels)
  VOLLEYHUB_PREVIEW_QUALITY    default low   (low | mid)
  VOLLEYHUB_PREVIEW_LATEST_ONLY default 1    (drop stale frames when busy)
  VOLLEYHUB_PREVIEW_ENABLED    default 1     (auto-on with fast path)
  VOLLEYHUB_PREVIEW_ROLE       default ""    (empty = use first ready role each tick)
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import numpy as np

_LOG = logging.getLogger(__name__)

_DEFAULT_FPS = 20.0
_DEFAULT_SIZE = 512


def _read_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "")
    if raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _read_int(name: str, default: int) -> int:
    try:
        raw = os.environ.get(name, "")
        if raw == "":
            return default
        return int(raw.strip())
    except Exception:
        return default


def _read_float(name: str, default: float) -> float:
    try:
        raw = os.environ.get(name, "")
        if raw == "":
            return default
        return float(raw.strip())
    except Exception:
        return default


@dataclass(slots=True)
class PreviewConfig:
    enabled: bool = True
    max_fps: float = _DEFAULT_FPS
    target_size: int = _DEFAULT_SIZE
    quality: str = "low"
    latest_only: bool = True
    bayer_pattern: str = "RG"
    preferred_role: str = ""

    def period_s(self) -> float:
        if self.max_fps <= 0:
            return 1.0
        return 1.0 / float(self.max_fps)


def load_preview_config_from_env(default_bayer: str = "RG") -> PreviewConfig:
    return PreviewConfig(
        enabled=_read_bool("VOLLEYHUB_PREVIEW_ENABLED", True),
        max_fps=_read_float("VOLLEYHUB_PREVIEW_MAX_FPS", _DEFAULT_FPS),
        target_size=_read_int("VOLLEYHUB_PREVIEW_SIZE", _DEFAULT_SIZE),
        quality=os.environ.get("VOLLEYHUB_PREVIEW_QUALITY", "low").strip().lower() or "low",
        latest_only=_read_bool("VOLLEYHUB_PREVIEW_LATEST_ONLY", True),
        bayer_pattern=os.environ.get("VOLLEYHUB_BAYER_PATTERN", default_bayer).strip().upper() or default_bayer,
        preferred_role=os.environ.get("VOLLEYHUB_PREVIEW_ROLE", "").strip().upper(),
    )


def cheap_preview_from_raw(raw_bayer: np.ndarray, target_size: int, quality: str = "low") -> np.ndarray:
    """Cheap CPU preview: avg-pool each 2x2 Bayer cell into one grayscale
    pixel, then resize to target_size width. Cost ≈ 1-2 ms for a 1920x1080
    input. Quality is intentionally low — color is approximate (greyscale
    fallback). For "mid" quality, use cv2.cvtColor on the half-resolution
    Bayer instead, which is ~5x more expensive but produces real color.
    """
    if raw_bayer.ndim != 2:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    h, w = raw_bayer.shape
    if h < 2 or w < 2:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    # Crop to even dims so the 2x2 block reshape is exact.
    h2 = (h // 2) * 2
    w2 = (w // 2) * 2
    raw = raw_bayer[:h2, :w2]

    if quality == "mid":
        # Real color path on half-resolution. ~5 ms for 1920x1080 -> 960x540.
        try:
            import cv2
            small = raw.copy()  # half-res Bayer (still mosaic)
            small = small[::2, ::2]  # quarter samples — actually one color
            # Better: use cv2.pyrDown? But pyrDown on Bayer breaks pattern too.
            # For "mid" we accept some artifacts.
            bgr = cv2.cvtColor(small, _cv2_bayer_code(small))
            bgr = _resize_to_target(bgr, target_size)
            return bgr
        except Exception:
            pass

    # Default low quality: fast decimation -> grayscale -> resize -> fake BGR.
    # This intentionally avoids a full-frame uint16 average over the Bayer
    # image, because that can be slower than the preview cadence on CPU-only
    # development machines. Preview is diagnostic only; inference uses the GPU
    # path and keeps full quality.
    half_gray = raw[0:h2:2, 0:w2:2]

    new_w = max(1, int(target_size))
    new_h = max(1, int(half_gray.shape[0] * (target_size / max(1, half_gray.shape[1]))))
    try:
        import cv2
        resized = cv2.resize(half_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        bgr = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        return bgr
    except Exception:
        # numpy-only fallback — nearest neighbor
        ys = (np.arange(new_h) * (half_gray.shape[0] / new_h)).astype(np.int64)
        xs = (np.arange(new_w) * (half_gray.shape[1] / new_w)).astype(np.int64)
        small = half_gray[ys[:, None], xs[None, :]]
        return np.repeat(small[..., None], 3, axis=-1)


def _resize_to_target(bgr: np.ndarray, target_size: int) -> np.ndarray:
    import cv2
    if bgr.ndim != 3:
        return bgr
    h, w = bgr.shape[:2]
    if w <= target_size:
        return bgr
    new_w = int(target_size)
    new_h = max(1, int(h * (target_size / max(1, w))))
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _cv2_bayer_code(_arr: np.ndarray) -> int:
    import cv2
    return cv2.COLOR_BAYER_BG2BGR  # default fallback if no pattern available


class PreviewWorker:
    """Background-thread preview producer.

    Reads RAW from shared memory, decimates+converts on CPU (cheap),
    stores latest frame for the GUI to poll. Never blocks anything else.
    """

    def __init__(
        self,
        config: PreviewConfig,
        smm: Any,
        read_role_to_key_map_fn: Callable[[], dict],
        roles: Tuple[str, ...],
        probe_factory: Optional[Callable[[str], Any]] = None,
    ) -> None:
        self.config = config
        self._smm = smm
        self._read_keys = read_role_to_key_map_fn
        self._roles = tuple(roles)
        self._probe_factory = probe_factory
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._latest_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_role: str = ""
        self._latest_ts_ns: int = 0
        self._frames_published: int = 0
        self._frames_dropped: int = 0
        self._last_publish_ms: float = 0.0
        self._published_in_window: int = 0

    # --- lifecycle ---

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        if not self.config.enabled or self.config.max_fps <= 0:
            _LOG.info("[PreviewWorker] disabled (enabled=%s, max_fps=%.1f)",
                      self.config.enabled, self.config.max_fps)
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._loop, name="preview-worker", daemon=True)
        self._thread.start()
        _LOG.info("[PreviewWorker] started (max_fps=%.1f, target_size=%d, quality=%s, latest_only=%s)",
                  self.config.max_fps, self.config.target_size, self.config.quality, self.config.latest_only)

    def stop(self, timeout_s: float = 1.0) -> None:
        self._stop_evt.set()
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=max(0.05, float(timeout_s)))

    @property
    def is_running(self) -> bool:
        return bool(self._thread is not None and self._thread.is_alive())

    # --- public API ---

    def get_latest(self) -> Optional[Tuple[np.ndarray, str, int]]:
        """Return (frame_bgr, role, ts_ns) or None. Returned array is a
        view of the worker's latest slot; callers should not mutate it.
        """
        with self._latest_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame, self._latest_role, self._latest_ts_ns

    def stats(self) -> dict:
        return {
            "running": self.is_running,
            "enabled": bool(self.config.enabled),
            "max_fps": float(self.config.max_fps),
            "target_size": int(self.config.target_size),
            "quality": str(self.config.quality),
            "latest_only": bool(self.config.latest_only),
            "frames_published": int(self._frames_published),
            "frames_dropped": int(self._frames_dropped),
            "last_publish_ms": float(self._last_publish_ms),
        }

    # --- internals ---

    def _pick_role(self, role_to_key: dict) -> Tuple[str, str]:
        # Prefer explicit role if set and ready.
        if self.config.preferred_role:
            k = str(role_to_key.get(self.config.preferred_role, "") or "")
            if k:
                return self.config.preferred_role, k
        # Otherwise pick the first ready role (rotate to spread load).
        for r in self._roles:
            k = str(role_to_key.get(r, "") or "")
            if k:
                return r, k
        return "", ""

    def _loop(self) -> None:
        period_s = self.config.period_s()
        next_deadline = time.perf_counter()
        target_size = int(self.config.target_size)
        quality = str(self.config.quality)

        probe = None
        if self._probe_factory is not None:
            try:
                probe = self._probe_factory("live_preview")
            except Exception:
                probe = None

        while not self._stop_evt.is_set():
            tick_start = time.perf_counter()
            # Throttle by self-paced deadline.
            if tick_start < next_deadline:
                self._stop_evt.wait(next_deadline - tick_start)
                if self._stop_evt.is_set():
                    break
                tick_start = time.perf_counter()
            next_deadline = tick_start + period_s

            try:
                role_to_key = self._read_keys() or {}
            except Exception:
                role_to_key = {}
            role, key = self._pick_role(role_to_key)
            if not key:
                self._frames_dropped += 1
                if probe is not None:
                    try:
                        probe.record_packet(
                            stage_ms={"preview_publish_ms": 0.0},
                            preview_enabled=True,
                            preview_fps_target=self.config.max_fps,
                            preview_size=target_size,
                            preview_blocks_fast_path=False,
                            drops=self._frames_dropped,
                            event="no_role_ready",
                        )
                    except Exception:
                        pass
                continue

            t_read = time.perf_counter()
            try:
                # SharedMemoryManager uses for-attempt-in-range(retries),
                # so retries must be >=1 or no read is attempted at all.
                # Some test doubles / older SharedMemoryManager variants do not
                # accept quiet_missing, therefore fall back to the legacy call.
                try:
                    read = self._smm.read_frame(key, retries=1, delay=0.0,
                                                quiet_missing=True)
                except TypeError:
                    read = self._smm.read_frame(key, retries=1, delay=0.0)
            except Exception:
                read = None
            if read is None:
                self._frames_dropped += 1
                if probe is not None:
                    try:
                        probe.record_packet(
                            stage_ms={"preview_publish_ms": 0.0},
                            preview_enabled=True,
                            preview_fps_target=self.config.max_fps,
                            preview_size=target_size,
                            preview_blocks_fast_path=False,
                            preview_queue_drops=self._frames_dropped,
                            event="read_miss",
                        )
                    except Exception:
                        pass
                continue
            raw_bayer, ts_ns = read[0], int(read[1] or 0)
            t_proc = time.perf_counter()

            try:
                bgr = cheap_preview_from_raw(raw_bayer, target_size, quality)
            except Exception as exc:
                _LOG.warning("[PreviewWorker] preview build failed: %s", exc)
                self._frames_dropped += 1
                continue

            t_done = time.perf_counter()
            publish_ms = (t_done - t_read) * 1000.0
            self._last_publish_ms = publish_ms

            # Latest-frame-only: replace existing slot under lock.
            with self._latest_lock:
                self._latest_frame = bgr
                self._latest_role = role
                self._latest_ts_ns = ts_ns
            self._frames_published += 1
            self._published_in_window += 1

            if probe is not None:
                try:
                    probe.record_packet(
                        stage_ms={
                            "preview_publish_ms": float(publish_ms),
                            "preview_read_ms": float((t_proc - t_read) * 1000.0),
                            "preview_build_ms": float((t_done - t_proc) * 1000.0),
                        },
                        preview_enabled=True,
                        preview_fps_target=self.config.max_fps,
                        preview_size=target_size,
                        preview_quality=quality,
                        preview_role=role,
                        preview_latest_only=self.config.latest_only,
                        preview_blocks_fast_path=False,  # by construction — separate thread
                        preview_queue_drops=self._frames_dropped,
                        frames_published=self._frames_published,
                    )
                except Exception:
                    pass
