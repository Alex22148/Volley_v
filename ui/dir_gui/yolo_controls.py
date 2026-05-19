"""
Auto-generated module extracted from ui/gui.py.
Do not edit manually unless you know what you are doing.
"""

import os
import re
import time
import webbrowser
import queue
import json
import csv
import warnings
from pathlib import Path
from collections import deque

import customtkinter as ctk
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from storage.shared_memory_manager import get_shared_memory_manager
from benchmarking.io_utils import (
    collect_runtime_env,
    detect_git_commit,
    prepare_run_directory,
    utc_now_iso,
    write_json,
    write_summary_csv,
)
from benchmarking.stats import downsample

from ui.style_volleyhub import (
    init_ctk_theme,
    BG_LIGHT,
    PANEL_BG,
    CARD_BG,
    CARD_INNER_BG,
    BORDER_COLOR,
    TEXT_MAIN,
    TEXT_DIM,
    ACCENT,
    ACCENT_HOVER,
    SUCCESS,
    DANGER,
    LIVE_BG,
    FONT_FAMILY,
)

def toggle_yolo(self):
    """Toggle YOLO detection on/off"""
    self.yolo_on = not self.yolo_on

    self._sync_operator_state_ui()

    if self.yolo_on:
        self._set_status("🎯 Detekcja piłki włączona", "success")
    else:
        self._set_status("🛑 Detekcja piłki wyłączona", "warning")

    if getattr(self, "_initing", False):
        return

    try:
        self.control_q.put(("yolo", self.yolo_on))
    except Exception as e:
        print(f"[GUI] ⚠️ Error toggling YOLO: {e}")

def choose_yolo_model(self):
    """Open file dialog to choose YOLO model"""
    from tkinter import filedialog

    model_path = filedialog.askopenfilename(
        title="Wybierz model YOLO",
        filetypes=[
            ("YOLO models", "*.pt *.onnx *.engine"),
            ("PyTorch models", "*.pt"),
            ("ONNX models", "*.onnx"),
            ("TensorRT models", "*.engine"),
            ("All files", "*.*")
        ]
    )

    if model_path:
        self.yolo_model_path = model_path
        # Show only filename in label
        import os
        filename = os.path.basename(model_path)
        self.yolo_model_label.configure(
            text=f"Model: {filename}",
            text_color=ACCENT
        )
        self._set_status(f"Wybrano model: {filename}", "accent")
        print(f"[GUI] Selected YOLO model: {model_path}")

def _pick_and_load_yolo_model(self):
    before = str(getattr(self, "yolo_model_path", "") or "")
    self.choose_yolo_model()
    after = str(getattr(self, "yolo_model_path", "") or "")
    if after and after != before:
        self.load_yolo_model()

def load_yolo_model(self):
    """Load the selected YOLO model"""
    if not self.yolo_model_path:
        self._set_status("Najpierw wybierz model YOLO", "warning")
        return

    try:
        self.yolo_load_btn.configure(text="Ładowanie...", state="disabled")
        self.control_q.put(("yolo_load_model", self.yolo_model_path))
        self._send_live_track_config_if_running()
        self._set_status(f"Ładowanie modelu YOLO...", "accent")
        print(f"[GUI] Loading YOLO model: {self.yolo_model_path}")

        # Re-enable button after 3 seconds
        self._after(3000, lambda: self.yolo_load_btn.configure(
            text="⚡ Wybierz i załaduj", state="normal"
        ))

    except Exception as e:
        print(f"[GUI] ⚠️ Error loading YOLO model: {e}")
        self._set_status("Błąd ładowania modelu", "danger")
        self.yolo_load_btn.configure(text="⚡ Wybierz i załaduj", state="normal")

def _normalize_live_infer_backend(self, backend: str) -> str:
    b = str(backend or "").strip().lower()
    return "tensorrt" if b == "tensorrt" else "ultralytics"

def _reset_live_compare_stats(self):
    self._live_compare_acc = {
        "ultralytics": {"sum_ms": 0.0, "count": 0},
        "tensorrt": {"sum_ms": 0.0, "count": 0},
    }
    self.live_compare_var.set("Porównanie infer: U=-- ms | TRT=-- ms")

def save_live_track_stats_snapshot(self):
    # Force a few polling cycles to get the freshest data from the backend
    for _ in range(3):
        try:
            self._poll_live_track_events()
        except Exception:
            pass
        time.sleep(0.05)

    # Debug diagnostics
    print(f"[LIVE_STATS_SNAPSHOT] live_track_on={self.live_track_on}")
    print(f"[LIVE_STATS_SNAPSHOT] live_track_last_result={self.live_track_last_result}")
    print(f"[LIVE_STATS_SNAPSHOT] live_track_last_stats={self.live_track_last_stats}")

    has_result = bool(self.live_track_last_result)
    has_stats = bool(self.live_track_last_stats)
    if not has_result and not has_stats:
        self._set_status(
            "Brak danych LIVE_TRACK do zapisania - uruchom LIVE_TRACK i odczekaj kilka sekund",
            "warning",
        )
        return

    def _avg(acc: dict):
        cnt = int((acc or {}).get("count", 0) or 0)
        if cnt <= 0:
            return None
        return float((acc or {}).get("sum_ms", 0.0) or 0.0) / float(cnt)

    try:
        u_acc = dict((self._live_compare_acc or {}).get("ultralytics", {}) or {})
        t_acc = dict((self._live_compare_acc or {}).get("tensorrt", {}) or {})
        u_avg = _avg(u_acc)
        t_avg = _avg(t_acc)

        counters = dict(((self.live_track_last_stats or {}).get("counters", {}) or {}))
        gauges = dict(((self.live_track_last_stats or {}).get("gauges", {}) or {}))
        last_result = dict(self.live_track_last_result or {})
        stage_ms = dict((last_result or {}).get("stage_ms", {}) or {})
        frame_age_ms = float(last_result.get("frame_age_ms", 0.0) or 0.0)

        snapshot = {
            "saved_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
            "saved_at_epoch_ns": time.time_ns(),
            "live_backend": str(getattr(self, "live_infer_backend", "ultralytics") or "ultralytics"),
            "trt_engine_path": str(getattr(self, "live_trt_engine_path", "") or ""),
            "comparison": {
                "ultralytics": {
                    "count": int(u_acc.get("count", 0) or 0),
                    "avg_infer_ms": u_avg,
                },
                "tensorrt": {
                    "count": int(t_acc.get("count", 0) or 0),
                    "avg_infer_ms": t_avg,
                },
            },
            "last_result": {
                "role": last_result.get("role"),
                "latency_total_ms": float(last_result.get("latency_total_ms", 0.0) or 0.0),
                "frame_age_ms": frame_age_ms,
                "infer_backend": last_result.get("infer_backend"),
                "trt_engine_path": last_result.get("trt_engine_path"),
                "detections_count": len(last_result.get("detections", []) or []),
                "tracks_count": len(last_result.get("tracks", []) or []),
                "stage_ms": stage_ms,
            },
            "runtime_stats": {
                "counters": counters,
                "gauges": gauges,
            },
        }

        print(f"[LIVE_STATS_SNAPSHOT] snapshot.last_result={snapshot.get('last_result')}")
        print(f"[LIVE_STATS_SNAPSHOT] snapshot.runtime_stats={snapshot.get('runtime_stats')}")

        base_dir = str(self.raw_dir or "").strip()
        if not base_dir or not os.path.isdir(base_dir):
            base_dir = str(Path.cwd())
        file_name = f"live_track_stats_{time.strftime('%Y%m%d_%H%M%S')}.json"
        out_path = filedialog.asksaveasfilename(
            title="Zapisz statystyki LIVE_TRACK",
            defaultextension=".json",
            initialdir=base_dir,
            initialfile=file_name,
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not out_path:
            return

        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)

        self._set_status(f"Zapisano LIVE stats: {out_file.name}", "success")
    except Exception as e:
        self._set_status("Błąd zapisu LIVE stats", "danger")
        print(f"[GUI] LIVE stats save failed: {e}")

def _apply_live_backend_ui(self):
    backend = self._normalize_live_infer_backend(getattr(self, "live_infer_backend", "ultralytics"))
    self.live_infer_backend = backend
    engine_path = str(getattr(self, "live_trt_engine_path", "") or "").strip()
    engine_name = os.path.basename(engine_path) if engine_path else "brak"
    if hasattr(self, "live_backend_var"):
        self.live_backend_var.set(f"LIVE backend: {backend} | engine: {engine_name}")
    if hasattr(self, "live_backend_btn"):
        label = "TensorRT" if backend == "tensorrt" else "Ultralytics"
        self.live_backend_btn.configure(text=f"Backend: {label}")
    if hasattr(self, "live_engine_btn"):
        if backend == "tensorrt":
            self.live_engine_btn.configure(state="normal", fg_color="#0f766e", hover_color="#0d5f5a")
        else:
            self.live_engine_btn.configure(state="normal", fg_color="#155e75", hover_color="#164e63")

def _push_live_backend_settings_to_control_plane(self):
    backend = self._normalize_live_infer_backend(getattr(self, "live_infer_backend", "ultralytics"))
    self.live_infer_backend = backend
    try:
        if self.shared_state is not None:
            self.shared_state["yolo_backend"] = backend
            self.shared_state["yolo_trt_engine_path"] = str(self.live_trt_engine_path or "")
        self.control_q.put(("yolo_set_backend", backend))
        if backend == "tensorrt":
            self.control_q.put(("yolo_set_tensorrt_engine", str(self.live_trt_engine_path or "")))
    except Exception:
        pass

def toggle_live_infer_backend(self):
    current = self._normalize_live_infer_backend(getattr(self, "live_infer_backend", "ultralytics"))
    target = "tensorrt" if current == "ultralytics" else "ultralytics"
    if target == "tensorrt":
        engine_path = str(getattr(self, "live_trt_engine_path", "") or "").strip()
        if not engine_path or not os.path.isfile(engine_path):
            self._set_status("Wybierz poprawny plik .engine przed włączeniem TensorRT", "warning")
            return
    self.live_infer_backend = target
    self._apply_live_backend_ui()
    self._push_live_backend_settings_to_control_plane()
    self._reset_live_compare_stats()
    self._send_live_track_config_if_running()
    self._set_status(f"LIVE backend: {target}", "accent")

def choose_live_trt_engine(self):
    path = filedialog.askopenfilename(
        title="Wybierz TensorRT engine dla LIVE_TRACK",
        filetypes=[("TensorRT engine", "*.engine"), ("All files", "*.*")],
    )
    if not path:
        return
    if not str(path).lower().endswith(".engine"):
        self._set_status("Wybrany plik nie ma rozszerzenia .engine", "warning")
        return
    if not os.path.isfile(path):
        self._set_status("Wybrany plik .engine nie istnieje", "warning")
        return
    try:
        if os.path.getsize(path) < 1024:
            self._set_status("Wybrany plik .engine jest zbyt maly", "warning")
            return
    except Exception:
        self._set_status("Nie udalo sie odczytac pliku .engine", "warning")
        return

    self.live_trt_engine_path = str(path)
    if self.shared_state is not None:
        try:
            self.shared_state["yolo_trt_engine_path"] = self.live_trt_engine_path
        except Exception:
            pass
    try:
        self.control_q.put(("yolo_set_tensorrt_engine", self.live_trt_engine_path))
    except Exception:
        pass
    self._apply_live_backend_ui()
    self._send_live_track_config_if_running()
    self._set_status(f"Ustawiono TensorRT engine: {os.path.basename(self.live_trt_engine_path)}", "success")

def _build_live_track_config(self):
    selected_role = self._get_shared_state_value("selected_role", self.roles[0] if self.roles else "CENTER_L")
    model_path = self._get_shared_state_value("yolo_model_path", self.yolo_model_path if self.yolo_model_path else "best.pt")
    backend = self._normalize_live_infer_backend(getattr(self, "live_infer_backend", "ultralytics"))
    engine_path = str(getattr(self, "live_trt_engine_path", "") or "").strip()
    if backend != "tensorrt":
        engine_path = ""

    device = str(self._get_shared_state_value("yolo_device", "cuda") or "cuda")
    if backend == "tensorrt":
        device = "cuda"

    cfg = {
        "roles": list(self.roles),
        "selected_role": str(selected_role or (self.roles[0] if self.roles else "CENTER_L")),
        "poll_sleep_ms": 1.0,
        "publish_interval_ms": 80.0,
        "debayer_backend": "cpu",
        "bayer_pattern": str(self._get_shared_state_value("yolo_bayer_pattern", "BG") or "BG"),
        "yolo_enabled": True,
        "yolo_backend": str(backend),
        "yolo_device": str(device),
        "yolo_image_size": int(self._get_shared_state_value("yolo_image_size", 640) or 640),
        "yolo_confidence": float(self._get_shared_state_value("yolo_confidence", 0.25) or 0.25),
        "model_path": str(model_path or ""),
        "yolo_trt_engine_path": str(engine_path),
        "yolo_trt_dynamic": bool(self._get_shared_state_value("yolo_trt_dynamic", False)),
        "yolo_trt_workspace_gb": float(self._get_shared_state_value("yolo_trt_workspace_gb", 2.0) or 2.0),
        "preview_enabled": False,
        "tracker_max_distance_px": 120.0,
        "tracker_max_missed": 5,
        "auto_start": False,
    }
    return cfg

def _send_live_track_config_if_running(self):
    try:
        if self.live_track_on and self.live_backend_ctrl is not None:
            self.live_backend_ctrl.update_config(self._build_live_track_config())
    except Exception:
        pass

def toggle_live_track(self):
    if self.live_backend_ctrl is None:
        self.live_track_status_var.set("LIVE_TRACK: niedostępny (brak kontrolera)")
        self._set_status("LIVE_TRACK niedostępny", "warning")
        return

    backend = self._normalize_live_infer_backend(getattr(self, "live_infer_backend", "ultralytics"))
    if not self.live_track_on and backend == "tensorrt":
        engine_path = str(getattr(self, "live_trt_engine_path", "") or "").strip()
        if not engine_path or not os.path.isfile(engine_path):
            self._set_status("TensorRT wymaga poprawnego pliku .engine", "warning")
            return
        if not engine_path.lower().endswith(".engine"):
            self._set_status("TensorRT wymaga pliku z rozszerzeniem .engine", "warning")
            return
        try:
            if os.path.getsize(engine_path) < 1024:
                self._set_status("TensorRT engine wygląda na niepoprawny lub za mały", "warning")
                return
        except Exception:
            self._set_status("Nie udało się odczytać rozmiaru TensorRT engine", "warning")
            return

    if not self.live_track_on:
        try:
            self._reset_live_compare_stats()
            self.live_backend_ctrl.start(self._build_live_track_config())
            self.live_track_on = True
            try:
                if self.shared_state is not None:
                    self.shared_state["live_track_enabled"] = True
            except Exception:
                pass
            self.live_track_btn.configure(text="Stop LIVE_TRACK", fg_color=DANGER, hover_color="#b91c1c")
            self.live_track_status_var.set("LIVE_TRACK: uruchomiony")
            self._set_status("LIVE_TRACK uruchomiony", "success")
        except Exception as e:
            self.live_track_on = False
            self.live_track_status_var.set(f"LIVE_TRACK: błąd startu ({e})")
            self._set_status("LIVE_TRACK błąd startu", "danger")
    else:
        try:
            self.live_backend_ctrl.stop()
        except Exception:
            pass
        self.live_track_on = False
        try:
            if self.shared_state is not None:
                self.shared_state["live_track_enabled"] = False
        except Exception:
            pass
        self.live_track_btn.configure(text="Start LIVE_TRACK", fg_color="#0ea5e9", hover_color="#0284c7")
        self.live_track_status_var.set("LIVE_TRACK: zatrzymany")
        self._set_status("LIVE_TRACK zatrzymany", "warning")

def _poll_live_track_events(self):
    if self.live_backend_ctrl is None:
        return
    try:
        events = self.live_backend_ctrl.poll_events(max_items=64)
    except Exception:
        return

    for ev in events:
        et = str(ev.get("type", "") or "")
        payload = ev.get("payload", {}) if isinstance(ev, dict) else {}

        if et == "live_status":
            state = str((payload or {}).get("state", "") or "")
            cfg = (payload or {}).get("config", {}) if isinstance(payload, dict) else {}
            if isinstance(cfg, dict):
                backend_cfg = self._normalize_live_infer_backend(cfg.get("yolo_backend", self.live_infer_backend))
                engine_cfg = str(cfg.get("yolo_trt_engine_path", self.live_trt_engine_path) or "").strip()
                self.live_infer_backend = backend_cfg
                if engine_cfg:
                    self.live_trt_engine_path = engine_cfg
                self._apply_live_backend_ui()
            if state:
                self.live_track_status_var.set(f"LIVE_TRACK: {state}")
                if state == "running":
                    self.live_track_on = True
                    try:
                        self.live_track_btn.configure(text="Stop LIVE_TRACK", fg_color=DANGER, hover_color="#b91c1c")
                    except Exception:
                        pass
                elif state in ("stopped", "shutdown"):
                    self.live_track_on = False
                    try:
                        self.live_track_btn.configure(text="Start LIVE_TRACK", fg_color="#0ea5e9", hover_color="#0284c7")
                    except Exception:
                        pass

        elif et == "live_result":
            self.live_track_last_result = payload if isinstance(payload, dict) else {}
            r = self.live_track_last_result or {}
            dets = len(r.get("detections", []) or [])
            tracks_count = len(r.get("tracks", []) or [])
            lat_ms = float(r.get("latency_total_ms", 0.0) or 0.0)
            frame_age_ms = float(r.get("frame_age_ms", 0.0) or 0.0)
            self._push_latency_sample(lat_ms)
            stage_ms = r.get("stage_ms", {}) or {}
            grab_ms = float(stage_ms.get("grab_ms", 0.0) or 0.0)
            convert_ms = float(stage_ms.get("convert_ms", 0.0) or 0.0)
            preprocess_ms = float(stage_ms.get("preprocess_ms", 0.0) or 0.0)
            infer_ms = float(stage_ms.get("infer_call_ms", stage_ms.get("infer_ms", 0.0)) or 0.0)
            track_ms = float(stage_ms.get("track_ms", 0.0) or 0.0)
            infer_backend = self._normalize_live_infer_backend(
                r.get("infer_backend", self.live_infer_backend)
            )
            self.live_infer_backend = infer_backend
            if infer_backend == "tensorrt":
                trt_path = str(r.get("trt_engine_path", "") or "").strip()
                if trt_path:
                    self.live_trt_engine_path = trt_path
            self._apply_live_backend_ui()
            acc = self._live_compare_acc.get(infer_backend)
            if isinstance(acc, dict):
                acc["sum_ms"] = float(acc.get("sum_ms", 0.0)) + infer_ms
                acc["count"] = int(acc.get("count", 0)) + 1
            u_cnt = int(self._live_compare_acc.get("ultralytics", {}).get("count", 0))
            t_cnt = int(self._live_compare_acc.get("tensorrt", {}).get("count", 0))
            u_avg = (
                float(self._live_compare_acc["ultralytics"]["sum_ms"]) / u_cnt
                if u_cnt > 0 else None
            )
            t_avg = (
                float(self._live_compare_acc["tensorrt"]["sum_ms"]) / t_cnt
                if t_cnt > 0 else None
            )
            u_txt = f"{u_avg:.1f}" if u_avg is not None else "--"
            t_txt = f"{t_avg:.1f}" if t_avg is not None else "--"
            self.live_compare_var.set(f"Porównanie infer: U={u_txt} ms | TRT={t_txt} ms")
            self.live_track_metrics_var.set(
                f"lat:{lat_ms:.1f} age:{frame_age_ms:.1f} grab:{grab_ms:.1f} "
                f"conv:{convert_ms:.1f} prep:{preprocess_ms:.1f} "
                f"inf:{infer_ms:.1f} trk:{track_ms:.1f} "
                f"det:{dets} trks:{tracks_count}"
            )

        elif et == "live_stats":
            self.live_track_last_stats = payload if isinstance(payload, dict) else {}
            counters = (self.live_track_last_stats or {}).get("counters", {}) or {}
            stale = int(counters.get("stale_frame_skip", 0) or 0)
            misses = int(counters.get("frame_read_miss", 0) or 0)
            res_ovr = int(counters.get("live_result_overwrite", 0) or 0)
            stats_ovr = int(counters.get("live_stats_overwrite", 0) or 0)
            loop_iter = int(counters.get("loop_iterations", 0) or 0)
            frames_ok = int(counters.get("frames_read_ok", 0) or 0)
            infer_ok = int(counters.get("inference_calls_ok", 0) or 0)
            result_emitted = int(counters.get("live_result_emitted", 0) or 0)
            qdrop = int(counters.get("event_queue_drop_oldest", 0) or 0)
            self.live_track_metrics_var.set(
                f"stale:{stale} miss:{misses} res_ovr:{res_ovr} stats_ovr:{stats_ovr} "
                f"loop:{loop_iter} fr_ok:{frames_ok} inf_ok:{infer_ok} "
                f"emitted:{result_emitted} qdrop:{qdrop}"
            )
        elif et == "live_error":
            msg = str((payload or {}).get("error", "") or "unknown error")
            where = str((payload or {}).get("where", "") or "live_backend")
            self.live_track_status_var.set(f"LIVE_TRACK error: {where}")
            self._set_status(f"LIVE_TRACK error: {msg}", "danger")
        elif et == "preview_ready":
            self._set_status("LIVE_TRACK preview ready", "success")

def on_yolo_conf_change(self, value):
    """Handle YOLO confidence threshold change"""
    confidence = float(value)
    self.yolo_conf_value.configure(text=f"{confidence:.2f}")

    # Don't send commands during initialization
    if getattr(self, "_initing", False):
        return

    try:
        self.control_q.put(("yolo_set_confidence", confidence))
        self._send_live_track_config_if_running()
        print(f"[GUI] YOLO confidence set to: {confidence:.2f}")
    except Exception as e:
        print(f"[GUI] ⚠️ Error setting YOLO confidence: {e}")

def _detect_cuda_preprocess_support(self) -> bool:
    try:
        import cv2

        return bool(
            hasattr(cv2, "cuda")
            and hasattr(cv2, "cuda_GpuMat")
            and hasattr(cv2.cuda, "demosaicing")
        )
    except Exception:
        return False

def _normalize_preprocess_backend(self, backend: str) -> str:
    b = str(backend or "").strip().lower()
    if b == "cuda" and bool(getattr(self, "_cuda_preprocess_supported", False)):
        return "cuda"
    return "cpu"

def _apply_yolo_preprocess_backend_ui(self):
    backend = self._normalize_preprocess_backend(getattr(self, "yolo_preprocess_backend", "cpu"))
    self.yolo_preprocess_backend = backend
    if not hasattr(self, "yolo_preprocess_btn") or not hasattr(self, "yolo_preprocess_label"):
        return

    if not bool(getattr(self, "_cuda_preprocess_supported", False)):
        self.yolo_preprocess_backend = "cpu"
        self.yolo_preprocess_label.configure(
            text="Preprocess YOLO: CPU (OpenCV bez CUDA demosaic)",
            text_color=TEXT_DIM,
        )
        self.yolo_preprocess_btn.configure(
            text="GPU niedostepne",
            fg_color="#4b5563",
            hover_color="#4b5563",
            state="disabled",
        )
        return

    if backend == "cuda":
        self.yolo_preprocess_label.configure(text="Preprocess YOLO: GPU (CUDA)", text_color=SUCCESS)
        self.yolo_preprocess_btn.configure(
            text="Przełącz na CPU",
            fg_color="#16a34a",
            hover_color="#15803d",
            state="normal",
        )
    else:
        self.yolo_preprocess_label.configure(text="Preprocess YOLO: CPU", text_color=TEXT_DIM)
        self.yolo_preprocess_btn.configure(
            text="Przełącz na GPU",
            fg_color="#6A5ACD",
            hover_color="#5A4FCF",
            state="normal",
        )

def toggle_yolo_preprocess_backend(self):
    if not bool(getattr(self, "_cuda_preprocess_supported", False)):
        self.yolo_preprocess_backend = "cpu"
        self._apply_yolo_preprocess_backend_ui()
        if not getattr(self, "_initing", False):
            try:
                self.control_q.put(("yolo_set_preprocess_backend", "cpu"))
            except Exception:
                pass
        self._set_status("CUDA preprocess niedostępny (OpenCV bez cv2.cuda.demosaicing)", "warning")
        return

    current = self._normalize_preprocess_backend(getattr(self, "yolo_preprocess_backend", "cpu"))
    target = "cuda" if current == "cpu" else "cpu"
    self.yolo_preprocess_backend = target
    self._apply_yolo_preprocess_backend_ui()

    if getattr(self, "_initing", False):
        return

    try:
        self.control_q.put(("yolo_set_preprocess_backend", target))
        self._send_live_track_config_if_running()
        self._set_status(f"YOLO preprocess backend: {target.upper()}", "accent")
        print(f"[GUI] YOLO preprocess backend set to: {target}")
    except Exception as e:
        print(f"[GUI] Error setting YOLO preprocess backend: {e}")
        self.yolo_preprocess_backend = current
        self._apply_yolo_preprocess_backend_ui()

def on_yolo_size_change(self, value):
    """Handle YOLO image size change"""
    size = int(float(value))
    valid_sizes = [320, 416, 512, 640, 736, 832, 1024, 1280]
    size = min(valid_sizes, key=lambda x: abs(x - size))

    self.yolo_size_value.configure(text=f"{size}px")

    if getattr(self, "_initing", False):
        return

    try:
        self.control_q.put(("yolo_set_image_size", size))
        self._send_live_track_config_if_running()
    except Exception as e:
        print(f"[GUI] ⚠️ Error setting YOLO image size: {e}")

def on_yolo_batch_change(self, value):
    """
    Suwak steruje łączną liczbą obrazów zbieranych do jednego wejścia YOLO.
    Dla 4 kamer:
      4  -> seq_len=1
      8  -> seq_len=2
      12 -> seq_len=3
      ...
    """
    batch_images = int(round(float(value) / 4.0) * 4)
    batch_images = max(4, batch_images)
    self.yolo_batch_value.configure(text=str(batch_images))

    if getattr(self, "_initing", False):
        return

    try:
        self.control_q.put(("yolo_set_input_batch_images", batch_images))
        self._send_live_track_config_if_running()
        print(f"[GUI] YOLO input batch images set to: {batch_images}")
    except Exception as e:
        print(f"[GUI] ⚠️ Error setting YOLO input batch images: {e}")

def debug_yolo(self):
    """Debug YOLO - sprawdź czy działa"""
    try:
        from vision.yolo_module import get_yolo_debug_info, get_ball_positions

        debug_info = get_yolo_debug_info()
        positions = get_ball_positions()

        msg = f"YOLO Debug:\n"
        msg += f"Enabled: {debug_info['enabled']}\n"
        msg += f"Model loaded: {debug_info['model_loaded']}\n"
        msg += f"Total frames processed: {debug_info['total_frames_processed']}\n"
        msg += f"Total detections: {debug_info['total_detections_made']}\n"
        msg += f"Avg time: {debug_info['detection_times_avg']:.1f}ms\n"
        msg += f"Active balls: {sum(len(dets) for dets in positions.values())}\n"
        msg += f"Per camera: {debug_info['last_detections_count']}"

        self._set_status(msg, "accent")
        print(f"[GUI] 🔍 {msg}")

    except Exception as e:
        self._set_status(f"YOLO Debug error: {e}", "danger")
