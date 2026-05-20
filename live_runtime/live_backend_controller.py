from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

from storage.shared_memory_manager import get_shared_memory_manager
MIN_TRT_ENGINE_BYTES = 1024

# Diagnostic probe (off by default; activated by env VOLLEYHUB_LIVE_PROBE=1).
# Imported defensively — if the diagnostics package is missing or fails to
# load, production behaviour is unchanged.
try:
    from src.runtime_diagnostics.live_timing_probe import get_probe as _get_live_probe  # type: ignore
except Exception:
    def _get_live_probe(_tag: str):  # type: ignore[no-redef]
        class _Noop:
            enabled = False
            def record_packet(self, *_a, **_kw): return None
            def record_drop(self, *_a, **_kw): return None
        return _Noop()

# Production fast path (opt-in via VOLLEYHUB_FAST_PATH=1).
# Imports are guarded so production stays usable even if the new
# subsystem is missing or fails to load.
try:
    from src.runtime_production import (
        FastPathConfig as _FPConfig,
        FastPathExecutor as _FPExecutor,
        load_fast_path_config_from_env as _load_fp_cfg,
        PreviewWorker as _PreviewWorker,
        load_preview_config_from_env as _load_preview_cfg,
    )
    _FAST_PATH_IMPORT_OK = True
    _FAST_PATH_IMPORT_ERROR = None
except Exception as _fp_exc:  # pragma: no cover
    _FPConfig = None  # type: ignore[assignment]
    _FPExecutor = None  # type: ignore[assignment]
    _load_fp_cfg = None  # type: ignore[assignment]
    _PreviewWorker = None  # type: ignore[assignment]
    _load_preview_cfg = None  # type: ignore[assignment]
    _FAST_PATH_IMPORT_OK = False
    _FAST_PATH_IMPORT_ERROR = repr(_fp_exc)

def _now_ns() -> int:
    return time.time_ns()


class LiveBackendController:
    """
    Lightweight LIVE backend controller used by gui.py.
    Public API:
    - start(config)
    - stop()
    - shutdown(timeout_s=...)
    - update_config(config)
    - poll_events(max_items=...)
    """

    def __init__(self, shared_state: Optional[dict] = None, event_queue_max: int = 128) -> None:
        self.shared_state = shared_state if shared_state is not None else {}
        self._event_queue_max = max(16, int(event_queue_max))
        self._events: deque = deque()
        self._events_lock = threading.Lock()

        self._config_lock = threading.Lock()
        self._config: Dict[str, Any] = self._default_config()

        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._running = False
        self._state_lock = threading.Lock()

        self._counters = defaultdict(int)
        self._gauges = defaultdict(float)

        self._model = None
        self._model_key: Tuple[str, str] = ("", "")

        # Fast path state (opt-in; remains None when env var not set).
        self._fast_path_executor: Any = None
        self._fast_path_config: Any = None
        self._fast_path_init_attempted: bool = False
        self._fast_path_init_error: Optional[str] = None
        self._preview_worker: Any = None
        self._preview_config: Any = None

        self._tracks: Dict[int, Dict[str, Any]] = {}
        self._next_track_id = 1
        self._error_cooldown: Dict[str, float] = {}
        self._last_key_seen: str = ""
        self._last_ts_seen: int = -1
        self._last_ts_seen_by_role: Dict[str, int] = {}
        self._last_result_ts_ns: int = -1
        self._last_resolved_role: str = ""

    def start(self, config: Optional[Dict[str, Any]] = None) -> None:
        with self._state_lock:
            if self._running and self._thread is not None and self._thread.is_alive():
                if isinstance(config, dict):
                    self.update_config(config)
                return

            merged = self._default_config()
            if isinstance(config, dict):
                merged.update(config)
            with self._config_lock:
                self._config = merged
            self._tracks.clear()
            self._next_track_id = 1
            self._last_key_seen = ""
            self._last_ts_seen = -1
            self._last_result_ts_ns = -1
            self._last_resolved_role = ""
            self._counters.clear()
            self._gauges.clear()
            self._stop_evt.clear()
            self._thread = threading.Thread(target=self._run_loop, name="live-backend-loop", daemon=True)
            self._running = True
            self._thread.start()

        self._emit_status("running")

    def stop(self) -> None:
        thread = None
        with self._state_lock:
            if not self._running:
                return
            self._running = False
            self._stop_evt.set()
            thread = self._thread

        # Stop the preview worker first so its thread doesn't outlive LBC.
        try:
            if self._preview_worker is not None:
                self._preview_worker.stop(timeout_s=0.5)
                self._preview_worker = None
        except Exception:
            pass

        if thread is not None:
            thread.join(timeout=1.5)

        self._emit_status("stopped")

    def shutdown(self, timeout_s: float = 1.5) -> None:
        thread = None
        with self._state_lock:
            self._running = False
            self._stop_evt.set()
            thread = self._thread

        try:
            if self._preview_worker is not None:
                self._preview_worker.stop(timeout_s=0.5)
                self._preview_worker = None
        except Exception:
            pass

        if thread is not None:
            thread.join(timeout=max(0.1, float(timeout_s)))

        self._emit_status("shutdown")

    def update_config(self, config: Dict[str, Any]) -> None:
        if not isinstance(config, dict):
            return
        with self._config_lock:
            merged = dict(self._config)
            merged.update(config)
            self._config = merged
        self._emit_status("running" if self.is_running() else "stopped")

    def poll_events(self, max_items: int = 64) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        n = max(1, int(max_items))
        with self._events_lock:
            while self._events and len(out) < n:
                out.append(self._events.popleft())
        return out

    def is_running(self) -> bool:
        with self._state_lock:
            return bool(self._running and self._thread is not None and self._thread.is_alive())

    def _default_config(self) -> Dict[str, Any]:
        return {
            "roles": ["CENTER_L"],
            "selected_role": "CENTER_L",
            "poll_sleep_ms": 1.0,
            "publish_interval_ms": 80.0,
            "debayer_backend": "cpu",
            "bayer_pattern": "BG",
            "yolo_enabled": True,
            "yolo_backend": "ultralytics",
            "yolo_device": "cuda",
            "yolo_image_size": 640,
            "yolo_confidence": 0.25,
            "model_path": "best.pt",
            "yolo_trt_engine_path": "",
            "preview_enabled": False,
            "tracker_max_distance_px": 120.0,
            "tracker_max_missed": 5,
            "auto_start": False,
        }

    def _snapshot_config(self) -> Dict[str, Any]:
        with self._config_lock:
            return deepcopy(self._config)

    def _emit(self, ev_type: str, payload: Dict[str, Any]) -> None:
        event = {"type": str(ev_type), "payload": payload, "ts_ns": _now_ns()}
        with self._events_lock:
            if ev_type in ("live_result", "live_stats"):
                replaced = False
                for idx in range(len(self._events) - 1, -1, -1):
                    if isinstance(self._events[idx], dict) and self._events[idx].get("type") == ev_type:
                        self._events[idx] = event
                        replaced = True
                        self._counters[f"{ev_type}_overwrite"] += 1
                        break
                if replaced:
                    return

            if len(self._events) >= self._event_queue_max:
                self._events.popleft()
                self._counters["event_queue_drop_oldest"] += 1
            self._events.append(event)

    def _emit_status(self, state: str) -> None:
        self._emit(
            "live_status",
            {
                "state": str(state),
                "config": self._snapshot_config(),
            },
        )

    def _emit_error(self, where: str, error: str, cooldown_s: float = 1.0) -> None:
        key = f"{where}:{error}"
        now = time.monotonic()
        last = self._error_cooldown.get(key, 0.0)
        if (now - last) < cooldown_s:
            return
        self._error_cooldown[key] = now
        self._counters["errors"] += 1
        self._emit("live_error", {"where": where, "error": error})

    def _read_role_to_key_map(self) -> Dict[str, str]:
        try:
            bayer_map = self.shared_state.get("bayer_key")
        except Exception:
            bayer_map = None
        if bayer_map is None:
            return {}
        try:
            return dict(bayer_map)
        except Exception:
            return {}

    def _resolve_key(self, cfg: Dict[str, Any]) -> Tuple[str, str]:
        role_to_key = self._read_role_to_key_map()
        selected = str(cfg.get("selected_role", "") or "")
        if selected and selected in role_to_key:
            return selected, str(role_to_key[selected])
        if role_to_key:
            role, key = next(iter(role_to_key.items()))
            return str(role), str(key)
        roles = cfg.get("roles", []) or []
        fallback_role = str(roles[0] if roles else "CENTER_L")
        return fallback_role, ""

    def _convert_bayer(self, frame_bayer: np.ndarray, pattern: str) -> np.ndarray:
        if frame_bayer is None:
            return np.zeros((1, 1, 3), dtype=np.uint8)

        if frame_bayer.ndim == 3 and frame_bayer.shape[2] == 3:
            return frame_bayer

        if cv2 is None:
            if frame_bayer.ndim == 2:
                return np.stack([frame_bayer] * 3, axis=-1)
            return frame_bayer

        bayer_code_map = {
            "BG": cv2.COLOR_BAYER_BG2BGR,
            "GB": cv2.COLOR_BAYER_GB2BGR,
            "RG": cv2.COLOR_BAYER_RG2BGR,
            "GR": cv2.COLOR_BAYER_GR2BGR,
        }
        key = str(pattern or "BG").upper()
        code = bayer_code_map.get(key, cv2.COLOR_BAYER_BG2BGR)
        try:
            return cv2.cvtColor(frame_bayer, code)
        except Exception:
            try:
                return cv2.cvtColor(frame_bayer, cv2.COLOR_GRAY2BGR)
            except Exception:
                if frame_bayer.ndim == 2:
                    return np.stack([frame_bayer] * 3, axis=-1)
                return frame_bayer

    def _minimal_preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        if frame_bgr is None:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        if frame_bgr.dtype != np.uint8:
            frame_bgr = frame_bgr.astype(np.uint8, copy=False)
        if frame_bgr.ndim == 2:
            if cv2 is not None:
                return cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
            return np.stack([frame_bgr] * 3, axis=-1)
        if frame_bgr.ndim == 3 and frame_bgr.shape[2] == 3:
            return np.ascontiguousarray(frame_bgr) if not frame_bgr.flags["C_CONTIGUOUS"] else frame_bgr
        return np.ascontiguousarray(frame_bgr)

    def _validate_tensorrt_config(self, cfg: Dict[str, Any]) -> Tuple[bool, str]:
        backend = self._normalize_backend(cfg.get("yolo_backend", "ultralytics"))
        if backend != "tensorrt":
            return True, ""

        device = str(cfg.get("yolo_device", "cuda") or "cuda").strip().lower()
        if not device.startswith("cuda"):
            return False, "TensorRT requires yolo_device=cuda"

        engine_path = str(cfg.get("yolo_trt_engine_path", "") or "").strip()
        if not engine_path:
            return False, "TensorRT requires yolo_trt_engine_path"
        if not engine_path.lower().endswith(".engine"):
            return False, "TensorRT requires .engine file"
        if not os.path.isfile(engine_path):
            return False, f"TensorRT engine file not found: {engine_path}"
        try:
            size_b = int(os.path.getsize(engine_path))
        except Exception:
            size_b = 0
        if size_b < MIN_TRT_ENGINE_BYTES:
            return False, f"TensorRT engine file too small or invalid: {engine_path}"
        return True, ""

    def _normalize_backend(self, backend: str) -> str:
        b = str(backend or "").strip().lower()
        return b if b in ("ultralytics", "tensorrt") else "ultralytics"

    def _ensure_model(self, backend: str, model_path: str, device: str, trt_engine_path: str = "") -> Tuple[Any, str]:
        model_path = str(model_path or "").strip()
        trt_engine_path = str(trt_engine_path or "").strip()

        if backend == "tensorrt":
            if not trt_engine_path:
                raise RuntimeError("TensorRT backend requires yolo_trt_engine_path")
            model_path = trt_engine_path
            if not model_path.lower().endswith(".engine"):
                raise RuntimeError("TensorRT backend requires .engine file")
            if not os.path.isfile(model_path):
                raise RuntimeError(f"TensorRT engine file not found: {model_path}")
            try:
                size_b = int(os.path.getsize(model_path))
            except Exception:
                size_b = 0
            if size_b < MIN_TRT_ENGINE_BYTES:
                raise RuntimeError(f"TensorRT engine file too small or invalid: {model_path}")
            if not str(device or "").lower().startswith("cuda"):
                raise RuntimeError("TensorRT backend requires yolo_device=cuda")

        model_key = (backend, model_path)
        if self._model is not None and self._model_key == model_key:
            return self._model, model_path

        if YOLO is None:
            raise RuntimeError("ultralytics not available")

        self._model = YOLO(model_path)
        self._model_key = model_key
        return self._model, model_path

    def _run_inference(self, frame_bgr: np.ndarray, cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str, str]:
        yolo_enabled = bool(cfg.get("yolo_enabled", True))
        backend = self._normalize_backend(cfg.get("yolo_backend", "ultralytics"))
        trt_engine_path = str(cfg.get("yolo_trt_engine_path", "") or "").strip()
        used_trt_path = trt_engine_path if backend == "tensorrt" else ""
        model_path = str(cfg.get("model_path", "") or "").strip()

        if not yolo_enabled:
            return [], backend, used_trt_path
        ok_trt, trt_error = self._validate_tensorrt_config(cfg)
        if not ok_trt:
            self._counters["tensorrt_config_error"] += 1
            self._gauges["tensorrt_valid"] = 0.0
            self._emit_error("tensorrt_config", trt_error, cooldown_s=0.8)
            return [], backend, used_trt_path
        if backend == "tensorrt":
            self._gauges["tensorrt_valid"] = 1.0

        source_path = trt_engine_path if (backend == "tensorrt" and trt_engine_path) else model_path
        if not source_path:
            self._emit_error("inference", "missing model_path")
            return [], backend, used_trt_path
        if YOLO is None:
            self._emit_error("inference", "ultralytics not available")
            return [], backend, used_trt_path

        imgsz = int(cfg.get("yolo_image_size", 640) or 640)
        conf = float(cfg.get("yolo_confidence", 0.25) or 0.25)
        device = str(cfg.get("yolo_device", "cuda") or "cuda")

        try:
            model, used_trt = self._ensure_model(backend, source_path, device, trt_engine_path)
            results = model.predict(
                source=frame_bgr,
                imgsz=imgsz,
                conf=conf,
                device=device,
                verbose=False,
            )
        except Exception as exc:
            self._emit_error("inference", str(exc))
            return [], backend, used_trt_path

        detections: List[Dict[str, Any]] = []
        try:
            if not results:
                return detections, backend, used_trt
            boxes = getattr(results[0], "boxes", None)
            if boxes is None:
                return detections, backend, used_trt
            xyxy = boxes.xyxy.detach().cpu().numpy() if boxes.xyxy is not None else np.zeros((0, 4))
            confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.zeros((0,))
            cls = boxes.cls.detach().cpu().numpy() if boxes.cls is not None else np.zeros((0,))
            n = int(xyxy.shape[0])
            for i in range(n):
                x1, y1, x2, y2 = [float(v) for v in xyxy[i].tolist()]
                detections.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float(confs[i]) if i < len(confs) else 0.0,
                        "class_id": int(cls[i]) if i < len(cls) else 0,
                    }
                )
        except Exception as exc:
            self._emit_error("postprocess", str(exc))

        return detections, backend, used_trt

    def _update_tracks(self, detections: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        max_dist = float(cfg.get("tracker_max_distance_px", 120.0) or 120.0)
        max_missed = int(cfg.get("tracker_max_missed", 5) or 5)

        det_centers = []
        for det in detections:
            x1, y1, x2, y2 = det.get("bbox", [0, 0, 0, 0])
            cx = 0.5 * (float(x1) + float(x2))
            cy = 0.5 * (float(y1) + float(y2))
            det_centers.append((cx, cy))

        active_ids = set(self._tracks.keys())
        used_dets = set()

        for tid in list(self._tracks.keys()):
            tr = self._tracks[tid]
            best_i = -1
            best_d = 1e18
            for i, (cx, cy) in enumerate(det_centers):
                if i in used_dets:
                    continue
                d = (cx - tr["cx"]) ** 2 + (cy - tr["cy"]) ** 2
                if d < best_d:
                    best_d = d
                    best_i = i
            if best_i >= 0 and best_d <= (max_dist * max_dist):
                used_dets.add(best_i)
                cx, cy = det_centers[best_i]
                det = detections[best_i]
                tr["cx"] = cx
                tr["cy"] = cy
                tr["bbox"] = list(det.get("bbox", [0, 0, 0, 0]))
                tr["confidence"] = float(det.get("confidence", 0.0) or 0.0)
                tr["missed"] = 0
            else:
                tr["missed"] += 1
                if tr["missed"] > max_missed:
                    self._tracks.pop(tid, None)
                    active_ids.discard(tid)

        for i, det in enumerate(detections):
            if i in used_dets:
                continue
            cx, cy = det_centers[i]
            tid = self._next_track_id
            self._next_track_id += 1
            self._tracks[tid] = {
                "id": tid,
                "cx": cx,
                "cy": cy,
                "bbox": list(det.get("bbox", [0, 0, 0, 0])),
                "confidence": float(det.get("confidence", 0.0) or 0.0),
                "missed": 0,
            }

        tracks_out: List[Dict[str, Any]] = []
        for tid, tr in list(self._tracks.items()):
            if tr.get("missed", 0) > max_missed:
                self._tracks.pop(tid, None)
                continue
            tracks_out.append(
                {
                    "track_id": int(tid),
                    "bbox": list(tr.get("bbox", [0, 0, 0, 0])),
                    "confidence": float(tr.get("confidence", 0.0) or 0.0),
                }
            )
        return tracks_out

    def _publish_stats(self, role: str) -> None:
        self._counters["live_stats_emitted"] += 1
        payload = {
            "role": str(role),
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
        }
        self._emit("live_stats", payload)

    def _ensure_fast_path(self) -> None:
        """Lazy init of the fast path executor (only on first use).

        Failure paths are SAFE: any error leaves _fast_path_executor=None
        and the loop falls through to the existing slow path. The flag
        VOLLEYHUB_FAST_PATH=1 must be set to even attempt this.
        """
        if self._fast_path_init_attempted:
            return
        self._fast_path_init_attempted = True
        if not _FAST_PATH_IMPORT_OK or _load_fp_cfg is None or _FPExecutor is None:
            self._fast_path_init_error = _FAST_PATH_IMPORT_ERROR or "fast_path import failed"
            return
        try:
            cfg = _load_fp_cfg()
            self._fast_path_config = cfg
            if not cfg.enabled:
                self._fast_path_init_error = "VOLLEYHUB_FAST_PATH not set"
                return
            err = cfg.validate()
            if err:
                self._fast_path_init_error = f"config invalid: {err}"
                self._counters["fast_path_config_error"] += 1
                self._emit_error("fast_path_config", err, cooldown_s=2.0)
                return
            executor = _FPExecutor(cfg)
            if not executor.initialize():
                self._fast_path_init_error = executor.init_error or "executor init failed"
                self._counters["fast_path_init_error"] += 1
                self._emit_error("fast_path_init", str(self._fast_path_init_error), cooldown_s=2.0)
                return
            self._fast_path_executor = executor
            self._counters["fast_path_active"] += 1
            self._emit("live_status", {
                "state": "fast_path_active",
                "fast_path": executor.describe(),
            })
            # Lightweight preview tor — separate thread, never blocks fast path.
            try:
                if _load_preview_cfg is not None and _PreviewWorker is not None:
                    pv_cfg = _load_preview_cfg(default_bayer=str(cfg.bayer_pattern))
                    self._preview_config = pv_cfg
                    if pv_cfg.enabled and pv_cfg.max_fps > 0:
                        from storage.shared_memory_manager import get_shared_memory_manager as _get_smm
                        _smm_for_preview = _get_smm()
                        worker = _PreviewWorker(
                            config=pv_cfg,
                            smm=_smm_for_preview,
                            read_role_to_key_map_fn=self._read_role_to_key_map,
                            roles=tuple(cfg.required_roles),
                            probe_factory=_get_live_probe,
                        )
                        worker.start()
                        self._preview_worker = worker
                        self._emit("live_status", {
                            "state": "fast_path_preview_active",
                            "preview": worker.stats(),
                        })
            except Exception as pvexc:
                msg = f"preview worker init failed: {pvexc!r}"
                print(f"[LBC] {msg}")
                self._counters["preview_worker_init_error"] += 1
                self._emit_error("preview_worker_init", msg, cooldown_s=2.0)
        except Exception as exc:
            self._fast_path_init_error = f"fast_path init exception: {exc!r}"
            self._counters["fast_path_init_error"] += 1
            self._emit_error("fast_path_init", str(exc), cooldown_s=2.0)

    def _try_run_fast_path(self, smm, next_stats_t: float,
                           publish_interval_s: float) -> Tuple[bool, float]:
        """Attempt one fast-path iteration.

        Returns (handled, next_stats_t):
          handled=True  -> packet was processed via fast path; caller should NOT run slow path
          handled=False -> caller falls through to existing slow path
        """
        executor = self._fast_path_executor
        if executor is None or not executor.is_ready:
            return False, next_stats_t
        fp_cfg = self._fast_path_config

        # Snapshot bayer_key map; require all configured roles ready.
        role_to_key = self._read_role_to_key_map()
        required = tuple(fp_cfg.required_roles)
        keys: Dict[str, str] = {}
        for r in required:
            k = str(role_to_key.get(r, "") or "")
            if not k:
                self._counters["fast_path_skip_no_role_key"] += 1
                return False, next_stats_t
            keys[r] = k

        # Read all 4 RAW frames + timestamps.
        t_read = time.perf_counter()
        frames: List[np.ndarray] = []
        timestamps: List[int] = []
        for r in required:
            read = smm.read_frame(keys[r], retries=fp_cfg.frame_read_retries,
                                  delay=fp_cfg.frame_read_delay_s)
            if read is None:
                self._counters["fast_path_skip_read_miss"] += 1
                return False, next_stats_t
            frames.append(read[0])
            timestamps.append(int(read[1] or 0))
        sync_wait_ms = (time.perf_counter() - t_read) * 1000.0

        packet_ts_ns = max(timestamps) if timestamps else 0
        ts_spread_ms = (max(timestamps) - min(timestamps)) / 1e6 if timestamps else 0.0

        # Process the packet via the fast path executor.
        result = executor.process_packet(frames, list(required), packet_ts_ns)
        if result.error:
            self._counters["fast_path_packet_error"] += 1
            self._emit_error("fast_path_packet", str(result.error), cooldown_s=1.0)
            return False, next_stats_t

        # Build stage_ms with BOTH slow-path keys (for GUI overlay compat)
        # AND fast-path native keys (for probe / diagnostics).
        fast_stage = dict(result.stage_ms)  # color_ms, inference_ms, postprocess_ms, total_packet_ms, stack_ms
        color_ms = float(fast_stage.get("color_ms", 0.0))
        inf_ms = float(fast_stage.get("inference_ms", 0.0))
        post_ms = float(fast_stage.get("postprocess_ms", 0.0))
        total_ms = float(fast_stage.get("total_packet_ms", 0.0))
        gui_compat_stage_ms = {
            # Slow-path conventional keys consumed by ui/dir_gui/yolo_controls.py:
            "grab_ms": float(sync_wait_ms),    # frame-read wait is the closest analog
            "convert_ms": color_ms,            # debayer = color stage
            "preprocess_ms": 0.0,              # demosaic+resize+normalize IS preprocess in fast path
            "infer_ms": inf_ms,
            "infer_call_ms": inf_ms,
            "track_ms": post_ms,               # postprocess + per-role tracking
            "total_ms": total_ms,
            # Fast-path native keys preserved alongside (probe + future consumers)
            **{k: float(v) for k, v in fast_stage.items()},
            "sync_wait_ms": float(sync_wait_ms),
        }

        # Emit one live_result per role; throttled by publish_interval_s.
        now_mono = time.monotonic()
        publish_now = (now_mono - getattr(self, "_fast_path_last_publish_t", 0.0)) >= publish_interval_s
        if publish_now:
            self._fast_path_last_publish_t = now_mono
            engine_path = str(fp_cfg.engine_path)
            for i, role in enumerate(required):
                self._counters["live_result_emitted"] += 1
                self._emit("live_result", {
                    "role": str(role),
                    "frame_ts_ns": int(timestamps[i]),
                    "detections": result.detections_per_role[i],
                    "tracks": result.tracks_per_role[i],
                    "infer_backend": "tensorrt",
                    "trt_engine_path": engine_path,
                    "latency_total_ms": float(total_ms),
                    "frame_age_ms": 0.0,
                    "fast_path": True,
                    "stage_ms": dict(gui_compat_stage_ms),
                })
        else:
            self._counters["result_throttled"] += 1
        # Mirror counters that GUI's live_stats display reads. One inference
        # call processed `len(required)` frames; record both metrics so the
        # GUI overlay (`fr_ok`, `inf_ok`) stays meaningful in fast path.
        self._counters["frames_read_ok"] += len(required)
        self._counters["inference_calls_ok"] += 1
        self._counters["frames_processed"] += len(required)
        self._counters["fast_path_packets_processed"] += 1

        # Publish stats periodically.
        if now_mono >= next_stats_t:
            self._publish_stats(str(required[0]))
            next_stats_t = now_mono + publish_interval_s

        # Probe (one record per packet, with extras).
        try:
            probe = _get_live_probe("live_backend")
            if getattr(probe, "enabled", False):
                stage_ms = dict(result.stage_ms)
                stage_ms["sync_wait_ms"] = float(sync_wait_ms)
                first_frame = frames[0]
                resolution = (
                    f"{first_frame.shape[1]}x{first_frame.shape[0]}"
                    if isinstance(first_frame, np.ndarray) and first_frame.ndim >= 2 else "?"
                )
                probe.record_packet(
                    stage_ms=stage_ms,
                    role=",".join(required),
                    resolution=resolution,
                    color_backend=str(fp_cfg.color_backend),
                    inference_backend="tensorrt",
                    batch=int(fp_cfg.batch_size),
                    yolo_imgsz=int(fp_cfg.imgsz),
                    fast_path_enabled=True,
                    packet_roles_count=len(required),
                    zero_copy_to_inference=bool(result.zero_copy_to_inference),
                    gpu_roundtrip=bool(not result.zero_copy_to_inference),
                    color_output_location=result.color_output_location,
                    inference_input_location=result.inference_input_location,
                    fallback_used=str(result.fallback_used) if result.fallback_used else "false",
                    ts_spread_ms=round(float(ts_spread_ms), 3),
                )
        except Exception:
            pass

        return True, next_stats_t

    def _run_loop(self) -> None:
        smm = get_shared_memory_manager()
        next_stats_t = time.monotonic()
        last_publish_t = 0.0
        self._fast_path_last_publish_t = 0.0

        # Lazy fast-path setup; safe no-op when VOLLEYHUB_FAST_PATH not set.
        self._ensure_fast_path()
        if self._fast_path_executor is not None:
            print(f"[LBC] fast path active: {self._fast_path_executor.describe()}")
        elif self._fast_path_init_error and os.environ.get("VOLLEYHUB_FAST_PATH"):
            print(f"[LBC] fast path UNAVAILABLE -> CPU path: {self._fast_path_init_error}")

        while not self._stop_evt.is_set():
            self._counters["loop_iterations"] += 1
            self._gauges["loop_alive"] = 1.0

            cfg = self._snapshot_config()

            # ---- Fast path branch (opt-in; safe fallback to slow path) ----
            if self._fast_path_executor is not None:
                publish_interval_s = max(0.001, float(cfg.get("publish_interval_ms", 80.0) or 80.0) / 1000.0)
                handled, next_stats_t = self._try_run_fast_path(
                    smm, next_stats_t, publish_interval_s,
                )
                if handled:
                    poll_sleep_s = max(0.0005, float(cfg.get("poll_sleep_ms", 1.0) or 1.0) / 1000.0)
                    if not self._stop_evt.is_set():
                        time.sleep(poll_sleep_s)
                    continue
                # else: fall through to existing slow path
            # ---- end fast path branch ----
            ok_trt, trt_error = self._validate_tensorrt_config(cfg)
            if self._normalize_backend(cfg.get("yolo_backend", "ultralytics")) == "tensorrt" and not ok_trt:
                self._counters["tensorrt_preflight_error"] += 1
                self._gauges["tensorrt_valid"] = 0.0
                self._emit_error("tensorrt_preflight", trt_error, cooldown_s=1.0)
            poll_sleep_s = max(0.0005, float(cfg.get("poll_sleep_ms", 1.0) or 1.0) / 1000.0)
            publish_interval_s = max(0.03, float(cfg.get("publish_interval_ms", 80.0) or 80.0) / 1000.0)

            t0 = time.perf_counter()
            role, bayer_key = self._resolve_key(cfg)
            self._gauges["debug_has_bayer_key"] = 1.0 if bayer_key else 0.0
            self._last_resolved_role = role
            if not bayer_key:
                self._counters["no_bayer_key"] += 1
                self._gauges["debug_has_bayer_key"] = 0.0
                self._emit_error("resolve_key", f"no bayer_key for role={role}", cooldown_s=1.0)
                now = time.monotonic()
                if now >= next_stats_t:
                    self._publish_stats(role)
                    next_stats_t = now + publish_interval_s
                time.sleep(poll_sleep_s)
                continue

            # Single timed read — no duplicate call
            read_t0 = time.perf_counter()
            read = smm.read_frame(bayer_key, retries=1, delay=0.0005)
            read_t1 = time.perf_counter()
            grab_ms = (read_t1 - read_t0) * 1000.0
            self._gauges["grab_ms"] = grab_ms

            if read is None:
                self._counters["read_frame_none"] += 1
                self._emit_error("read_frame", f"read miss for role={role} key={bayer_key}", cooldown_s=1.0)
                now = time.monotonic()
                if now >= next_stats_t:
                    self._publish_stats(role)
                    next_stats_t = now + publish_interval_s
                time.sleep(poll_sleep_s)
                continue

            frame_bayer, ts_ns = read
            self._counters["frames_read_ok"] += 1
            ts_ns = int(ts_ns or 0)
            self._last_key_seen = str(bayer_key)
            last_ts = int(self._last_ts_seen_by_role.get(role, -1))
            # Stale-frame skip based only on timestamp
            if ts_ns > 0:
                if ts_ns <= last_ts:
                    self._counters["stale_frame_skip"] += 1
                    now = time.monotonic()
                    if now >= next_stats_t:
                        self._publish_stats(role)
                        next_stats_t = now + publish_interval_s
                    time.sleep(poll_sleep_s)
                    continue

            self._last_ts_seen_by_role[role] = ts_ns

            convert_t0 = time.perf_counter()
            frame_bgr = self._convert_bayer(frame_bayer, str(cfg.get("bayer_pattern", "BG") or "BG"))
            convert_t1 = time.perf_counter()
            convert_ms = (convert_t1 - convert_t0) * 1000.0
            self._gauges["convert_ms"] = convert_ms

            prep_t0 = time.perf_counter()
            frame_input = self._minimal_preprocess(frame_bgr)
            prep_t1 = time.perf_counter()
            preprocess_ms = (prep_t1 - prep_t0) * 1000.0
            self._gauges["preprocess_ms"] = preprocess_ms

            infer_t0 = time.perf_counter()
            detections, backend, used_trt_path = self._run_inference(frame_input, cfg)
            infer_t1 = time.perf_counter()
            infer_ms = (infer_t1 - infer_t0) * 1000.0
            self._gauges["infer_ms"] = infer_ms
            self._gauges["infer_call_ms"] = infer_ms
            self._counters["inference_calls_ok"] += 1

            track_t0 = time.perf_counter()
            tracks = self._update_tracks(detections, cfg)
            track_t1 = time.perf_counter()
            track_ms = (track_t1 - track_t0) * 1000.0
            self._gauges["track_ms"] = track_ms

            t1 = time.perf_counter()
            total_pipeline_ms = (t1 - t0) * 1000.0
            self._gauges["total_pipeline_ms"] = total_pipeline_ms
            self._gauges["latency_total_ms"] = total_pipeline_ms
            self._gauges["last_frame_ts_ns"] = float(ts_ns)

            # frame_age_ms is wall-clock age of the captured frame — separate from pipeline time
            now_ns = _now_ns()
            frame_age_ms = None
            now = time.monotonic()
            if (now - last_publish_t) < publish_interval_s:
                self._counters["result_throttled"] += 1
            else:
                last_publish_t = now
                self._last_result_ts_ns = ts_ns
                self._counters["live_result_emitted"] += 1
                self._emit(
                    "live_result",
                    {
                        "role": str(role),
                        "frame_ts_ns": int(ts_ns),
                        "detections": detections,
                        "tracks": tracks,
                        "infer_backend": str(backend),
                        "trt_engine_path": str(used_trt_path),
                        "latency_total_ms": float(total_pipeline_ms),
                        "frame_age_ms": float(0),
                        "stage_ms": {
                            "grab_ms": float(grab_ms),
                            "convert_ms": float(convert_ms),
                            "preprocess_ms": float(preprocess_ms),
                            "infer_ms": float(infer_ms),
                            "infer_call_ms": float(infer_ms),
                            "track_ms": float(track_ms),
                            "total_ms": float(total_pipeline_ms),
                        },
                    },
                )
            self._counters["frames_processed"] += 1
            now = time.monotonic()
            if now >= next_stats_t:
                self._publish_stats(role)
                next_stats_t = now + publish_interval_s

            # --- Diagnostic probe (no-op unless VOLLEYHUB_LIVE_PROBE=1) ---
            try:
                _probe = _get_live_probe("live_backend")
                if getattr(_probe, "enabled", False):
                    _probe.record_packet(
                        stage_ms={
                            "grab_ms": float(grab_ms),
                            "cpu_debayer_ms": float(convert_ms),
                            "preprocess_ms": float(preprocess_ms),
                            "inference_ms": float(infer_ms),
                            "postprocess_ms": float(track_ms),
                            "total_packet_ms": float(total_pipeline_ms),
                        },
                        role=str(role),
                        resolution=(f"{frame_bgr.shape[1]}x{frame_bgr.shape[0]}"
                                    if isinstance(frame_bgr, np.ndarray) and frame_bgr.ndim >= 2 else "?"),
                        color_backend=str(cfg.get("debayer_backend", "cpu") or "cpu"),
                        inference_backend=str(backend),
                        batch=1,
                        yolo_enabled=bool(cfg.get("yolo_enabled", True)),
                    )
            except Exception:
                pass
            # --- end diagnostic probe ---

            if total_pipeline_ms < (poll_sleep_s * 1000.0):
                time.sleep(max(0.0, poll_sleep_s - (t1 - t0)))


