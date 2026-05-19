"""
Auto-generated module extracted from ui/gui.py.
Do not edit manually unless you know what you are doing.
"""

import os

import time
import queue
import json
import warnings
from pathlib import Path
import customtkinter as ctk
import numpy as np
import tkinter as tk
from tkinter import filedialog
from matplotlib.figure import Figure
from benchmarking.io_utils import (
    collect_runtime_env,
    detect_git_commit,
    prepare_run_directory,
    utc_now_iso,
    write_json,
    write_summary_csv,
)
from benchmarking.stats import downsample

def _choose_auto_bench_models_dir(self):
    path = filedialog.askdirectory(title="Wybierz folder z modelami YOLO")
    if path:
        self.ab_models_dir_var.set(path)

def _choose_auto_bench_model_file(self):
    path = filedialog.askopenfilename(
        title="Wybierz pojedynczy model YOLO",
        filetypes=[
            ("YOLO models", "*.pt *.onnx *.engine"),
            ("All files", "*.*"),
        ],
    )
    if path:
        self.ab_single_model_var.set(path)
        try:
            if not self.ab_models_dir_var.get().strip():
                self.ab_models_dir_var.set(os.path.dirname(path))
        except Exception:
            pass

def _normalize_auto_bench_source_mode(raw_mode: str) -> str:
    text = str(raw_mode or "").strip().lower()
    if ("jeden" in text) or ("single" in text) or ("pojedynczy" in text):
        return "single"
    return "folder"

def _normalize_auto_bench_backend_mode(raw_mode: str) -> str:
    text = str(raw_mode or "").strip().lower()
    if "ultralytics" in text:
        return "ultralytics"
    if ("tensorrt" in text) or ("engine" in text):
        return "tensorrt"
    return "auto"

def _on_auto_bench_model_source_change(self, _value=None):
    try:
        raw_mode = str(self.ab_model_source_var.get() or "")
    except Exception:
        raw_mode = "Wiele modeli (folder)"
    single_mode = (self._normalize_auto_bench_source_mode(raw_mode) == "single")
    folder_state = "disabled" if single_mode else "normal"
    single_state = "normal" if single_mode else "disabled"
    try:
        if hasattr(self, "ab_single_model_label"):
            self.ab_single_model_label.configure(
                text="Pojedynczy model (wymagany)" if single_mode else "Pojedynczy model (opcjonalny)"
            )
    except Exception:
        pass
    try:
        self.ab_models_dir_entry.configure(state=folder_state)
    except Exception:
        pass
    try:
        if self.ab_models_dir_btn is not None:
            self.ab_models_dir_btn.configure(state=folder_state)
    except Exception:
        pass
    try:
        self.ab_single_model_entry.configure(state=single_state)
    except Exception:
        pass
    try:
        if self.ab_single_model_btn is not None:
            self.ab_single_model_btn.configure(state=single_state)
    except Exception:
        pass
    self._update_auto_bench_plan_summary()

def _on_auto_bench_backend_mode_change(self, _value=None):
    mode = self._normalize_auto_bench_backend_mode(getattr(self, "ab_backend_mode_var", ctk.StringVar(value="auto")).get())
    hint = "Auto: .engine -> TensorRT, .pt/.onnx -> Ultralytics."
    if mode == "ultralytics":
        hint = "Ultralytics: uruchamiane będą tylko modele .pt lub .onnx."
    elif mode == "tensorrt":
        hint = "TensorRT: uruchamiane będą tylko modele .engine."
    try:
        if hasattr(self, "ab_backend_hint_var"):
            self.ab_backend_hint_var.set(hint)
    except Exception:
        pass
    self._update_auto_bench_plan_summary()

def _on_auto_bench_batch_mode_change(self, _value=None):
    batch_mode = str(getattr(self, "ab_batch_mode_var", ctk.StringVar(value="Zakres")).get() or "").strip().lower()
    is_list = "lista" in batch_mode
    range_state = "disabled" if is_list else "normal"
    list_state = "normal" if is_list else "disabled"
    for attr in ("ab_batch_from_combo", "ab_batch_to_combo", "ab_batch_step_combo"):
        try:
            getattr(self, attr).configure(state=range_state)
        except Exception:
            pass
    try:
        if hasattr(self, "ab_batch_list_combo"):
            self.ab_batch_list_combo.configure(state=list_state)
    except Exception:
        pass
    self._update_auto_bench_plan_summary()

def _on_auto_bench_write_mode_change(self, _value=None):
    modes = self._parse_modes_csv(getattr(self, "ab_modes_var", ctk.StringVar(value="Brak zapisu")).get())
    has_bin = any(m in ("bin", "bin+buffer") for m in modes)
    has_buffer = any(m in ("buffer", "bin+buffer") for m in modes)
    bin_state = "normal" if has_bin else "disabled"
    buf_state = "normal" if has_buffer else "disabled"
    for attr in ("ab_bin_s_entry",):
        try:
            getattr(self, attr).configure(state=bin_state)
        except Exception:
            pass
    for attr in ("ab_buffer_count_entry", "ab_buffer_pause_s_entry"):
        try:
            getattr(self, attr).configure(state=buf_state)
        except Exception:
            pass
    self._update_auto_bench_plan_summary()

def _update_auto_bench_plan_summary(self):
    try:
        source_mode_raw = str(getattr(self, "ab_model_source_var", ctk.StringVar(value="Wiele modeli (folder)")).get() or "")
        source_mode = self._normalize_auto_bench_source_mode(source_mode_raw)
        backend_mode = self._normalize_auto_bench_backend_mode(
            getattr(self, "ab_backend_mode_var", ctk.StringVar(value="Auto (wg modelu)")).get()
        )
        if source_mode == "single":
            model_count = 1 if str(getattr(self, "ab_single_model_var", ctk.StringVar(value="")).get() or "").strip() else 0
        else:
            models_dir = str(getattr(self, "ab_models_dir_var", ctk.StringVar(value="")).get() or "").strip()
            if models_dir and os.path.isdir(models_dir):
                model_count = len(
                    [
                        n for n in os.listdir(models_dir)
                        if os.path.isfile(os.path.join(models_dir, n)) and n.lower().endswith((".pt", ".onnx", ".engine"))
                    ]
                )
            else:
                model_count = 0
        imgsz_count = len(self._parse_int_csv(getattr(self, "ab_imgsz_var", ctk.StringVar(value="640")).get()) or [640])
        batch_mode = str(getattr(self, "ab_batch_mode_var", ctk.StringVar(value="Zakres")).get() or "").lower()
        if "lista" in batch_mode:
            batch_count = len(self._parse_int_csv(getattr(self, "ab_batch_var", ctk.StringVar(value="4")).get()) or [4])
        else:
            batch_count = len(
                self._build_int_range(
                    getattr(self, "ab_batch_from_var", ctk.StringVar(value="4")).get(),
                    getattr(self, "ab_batch_to_var", ctk.StringVar(value="4")).get(),
                    getattr(self, "ab_batch_step_var", ctk.StringVar(value="1")).get(),
                    fallback=[4],
                )
            )
        modes_count = len(self._parse_modes_csv(getattr(self, "ab_modes_var", ctk.StringVar(value="none")).get()))
        fps_count = 1
        total = max(0, model_count) * max(1, imgsz_count) * max(1, batch_count) * max(1, modes_count) * fps_count
        source_txt = "1 model" if source_mode == "single" else "folder modeli"
        if backend_mode == "ultralytics":
            backend_txt = "Ultralytics"
        elif backend_mode == "tensorrt":
            backend_txt = "TensorRT"
        else:
            backend_txt = "Auto"
        if hasattr(self, "ab_plan_var"):
            self.ab_plan_var.set(
                f"Plan: źródło={source_txt}, backend={backend_txt}, modele={model_count}, imgsz={imgsz_count}, batch={batch_count}, tryby={modes_count} -> case={total}"
            )
    except Exception:
        pass

def _parse_int_csv(raw: str) -> list[int]:
    out = []
    for tok in str(raw or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(float(tok)))
        except Exception:
            continue
    return [x for x in out if x > 0]

def _build_int_range(v_from: str, v_to: str, v_step: str, fallback: list[int]) -> list[int]:
    try:
        a = int(float(str(v_from).strip()))
        b = int(float(str(v_to).strip()))
        s = int(float(str(v_step).strip()))
    except Exception:
        return list(fallback)
    if s <= 0:
        return list(fallback)
    if b < a:
        a, b = b, a
    vals = list(range(a, b + 1, s))
    vals = [int(x) for x in vals if int(x) > 0]
    return vals or list(fallback)

def _parse_float_csv(raw: str) -> list[float]:
    out = []
    for tok in str(raw or "").split(","):
        tok = tok.strip().replace(",", ".")
        if not tok:
            continue
        try:
            out.append(float(tok))
        except Exception:
            continue
    return [x for x in out if x > 0]

def _fps_to_period_us(fps: float) -> float:
    fps = max(1.0, float(fps))
    return 1_000_000.0 / fps

def _safe_percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    try:
        return float(np.percentile(np.asarray(values, dtype=float), q))
    except Exception:
        arr = sorted(float(v) for v in values)
        idx = max(0, min(len(arr) - 1, int(round((q / 100.0) * (len(arr) - 1)))))
        return float(arr[idx])

def _safe_stats(values: list[float]) -> dict:
    if not values:
        return {
            "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0,
            "std": 0.0, "min": 0.0, "max": 0.0, "cv": 0.0,
        }
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    return {
        "mean": mean,
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "std": std,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "cv": float(std / mean) if abs(mean) > 1e-9 else 0.0,
    }

def _trend_slope(xs: list[float], ys: list[float]) -> float:
    if not xs or not ys or len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    try:
        x = np.asarray(xs, dtype=float)
        y = np.asarray(ys, dtype=float)
        x0 = float(x[0])
        x = x - x0
        if float(np.var(x)) < 1e-12:
            return 0.0
        # slope in y units / second
        return float(np.polyfit(x, y, 1)[0])
    except Exception:
        return 0.0

def _parse_modes_csv(raw: str) -> list[str]:
    allowed = {"none", "bin", "buffer", "bin+buffer"}
    aliases = {
        "wszystkiekombinacje": ["none", "bin", "buffer", "bin+buffer"],
        "all": ["none", "bin", "buffer", "bin+buffer"],
        "allcombinations": ["none", "bin", "buffer", "bin+buffer"],
        "brakzapisu": ["none"],
        "none": ["none"],
        "tylkobin": ["bin"],
        "bin": ["bin"],
        "tylkobufor": ["buffer"],
        "tylkobuffer": ["buffer"],
        "buffer": ["buffer"],
        "bin+bufor": ["bin+buffer"],
        "bin+buffer": ["bin+buffer"],
    }
    modes = []
    for tok in str(raw or "").split(","):
        t = tok.strip().lower().replace(" ", "")
        if t in aliases:
            for m in aliases[t]:
                if m not in modes:
                    modes.append(m)
            continue
        if t in allowed:
            modes.append(t)
    return modes or ["none"]

def _sample_system_metrics(self) -> dict:
    out = {}
    now = time.time()
    try:
        import psutil  # type: ignore
        out["cpu_percent"] = float(psutil.cpu_percent(interval=None))
        out["ram_percent"] = float(psutil.virtual_memory().percent)
        try:
            io = psutil.disk_io_counters()
            write_bytes = float(getattr(io, "write_bytes", 0.0) or 0.0)
            prev = getattr(self, "_sys_disk_prev", None)
            if prev is not None:
                prev_t, prev_wb = prev
                dt = max(1e-6, now - float(prev_t))
                out["disk_write_mb_s"] = max(0.0, (write_bytes - float(prev_wb)) / (1024.0 * 1024.0) / dt)
            self._sys_disk_prev = (now, write_bytes)
        except Exception:
            pass
    except Exception:
        pass

    try:
        import subprocess
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=0.4, check=False)
        if proc.returncode == 0 and proc.stdout.strip():
            line = proc.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpu_util = float(parts[0] or 0.0)
                gpu_mem_used = float(parts[1] or 0.0)
                gpu_mem_total = max(1.0, float(parts[2] or 1.0))
                gpu_temp = float(parts[3] or 0.0)
                gpu_power = float(parts[4] or 0.0)
                out["gpu_util_percent"] = gpu_util
                out["gpu_mem_used_mb"] = gpu_mem_used
                out["gpu_mem_total_mb"] = gpu_mem_total
                out["gpu_mem_percent"] = (gpu_mem_used / gpu_mem_total) * 100.0
                out["gpu_temp_c"] = gpu_temp
                out["gpu_power_w"] = gpu_power
    except Exception:
        pass

    return out

def _resolve_benchmark_output_base(self) -> Path:
    raw = str(self.ab_out_var.get() or "").strip() if hasattr(self, "ab_out_var") else ""
    if raw:
        p = Path(raw).expanduser()
        if p.suffix.lower() == ".csv":
            p = p.parent
        return p.resolve()
    if getattr(self, "session_dir", None):
        return Path(self.session_dir).resolve()
    return Path.cwd().resolve()

def _get_shared_state_value(self, key: str, default=None):
    try:
        if self.shared_state is not None:
            return self.shared_state.get(key, default)
    except Exception:
        pass
    return default

def _estimate_bbox_result_age_ms(self, now_ts: float | None = None):
    now_ts = float(now_ts if now_ts is not None else time.time())
    updated_candidates = []
    try:
        snap = self._get_shared_state_value("yolo_last_inference_snapshot", {}) or {}
        if isinstance(snap, dict):
            updated = snap.get("updated_at", None)
            if updated is not None:
                updated_candidates.append(float(updated))
    except Exception:
        pass
    try:
        batch_ts = self._get_shared_state_value("yolo_last_batch_timestamps", {}) or {}
        if isinstance(batch_ts, dict):
            updated = batch_ts.get("updated_at", None)
            if updated is not None:
                updated_candidates.append(float(updated))
    except Exception:
        pass
    if not updated_candidates:
        return None
    latest_updated = max(updated_candidates)
    if latest_updated <= 0:
        return None
    return max(0.0, (now_ts - latest_updated) * 1000.0)

def _start_auto_benchmark_legacy(self):
    # Backward-compat wrapper.
    self.start_auto_benchmark()

def _start_auto_benchmark_deprecated(self):
    # Backward-compat wrapper.
    self.start_auto_benchmark()

def start_auto_benchmark(self):
    if self._auto_bench_state and self._auto_bench_state.get("running"):
        self.ab_status_var.set("Auto benchmark: juz dziala")
        return

    source_mode = self._normalize_auto_bench_source_mode(getattr(self, "ab_model_source_var", ctk.StringVar(value="")).get())
    backend_mode = self._normalize_auto_bench_backend_mode(
        getattr(self, "ab_backend_mode_var", ctk.StringVar(value="Auto (wg modelu)")).get()
    )

    single_model = ""
    try:
        single_model = self.ab_single_model_var.get().strip()
    except Exception:
        pass

    candidate_models = []
    if source_mode == "single":
        if not (single_model and os.path.isfile(single_model) and single_model.lower().endswith((".pt", ".onnx", ".engine"))):
            self.ab_status_var.set("Auto benchmark: wybierz poprawny pojedynczy model")
            return
        candidate_models = [single_model]
        try:
            if not self.ab_models_dir_var.get().strip():
                self.ab_models_dir_var.set(os.path.dirname(single_model))
        except Exception:
            pass
    else:
        models_dir = self.ab_models_dir_var.get().strip()
        if not models_dir or not os.path.isdir(models_dir):
            self.ab_status_var.set("Auto benchmark: wskaz poprawny folder modeli")
            return
        for name in sorted(os.listdir(models_dir)):
            p = os.path.join(models_dir, name)
            if os.path.isfile(p) and name.lower().endswith((".pt", ".onnx", ".engine")):
                candidate_models.append(p)
        if not candidate_models:
            self.ab_status_var.set("Auto benchmark: brak modeli .pt/.onnx/.engine")
            return

    model_entries = []
    for model_path in candidate_models:
        ext = os.path.splitext(str(model_path))[1].lower()
        case_backend = None
        if backend_mode == "ultralytics":
            if ext in (".pt", ".onnx"):
                case_backend = "ultralytics"
        elif backend_mode == "tensorrt":
            if ext == ".engine":
                case_backend = "tensorrt"
        else:
            if ext == ".engine":
                case_backend = "tensorrt"
            elif ext in (".pt", ".onnx"):
                case_backend = "ultralytics"
        if case_backend is None:
            continue
        model_entries.append(
            {
                "model_path": str(model_path),
                "backend": case_backend,
                "engine_path": str(model_path) if case_backend == "tensorrt" else "",
            }
        )

    if not model_entries:
        if backend_mode == "ultralytics":
            self.ab_status_var.set("Auto benchmark: brak modeli .pt/.onnx dla wybranego backendu")
        elif backend_mode == "tensorrt":
            self.ab_status_var.set("Auto benchmark: brak modeli .engine dla wybranego backendu")
        else:
            self.ab_status_var.set("Auto benchmark: brak modeli zgodnych z konfiguracją")
        return

    imgsz_list = self._parse_int_csv(self.ab_imgsz_var.get()) or [640]
    batch_mode = str(getattr(self, "ab_batch_mode_var", ctk.StringVar(value="Zakres")).get() or "").strip().lower()
    if "lista" in batch_mode:
        batch_list = self._parse_int_csv(self.ab_batch_var.get()) or [4]
    else:
        batch_list = self._build_int_range(
            getattr(self, "ab_batch_from_var", ctk.StringVar(value="4")).get(),
            getattr(self, "ab_batch_to_var", ctk.StringVar(value="4")).get(),
            getattr(self, "ab_batch_step_var", ctk.StringVar(value="1")).get(),
            fallback=self._parse_int_csv(self.ab_batch_var.get()) or [4],
        )
    raw_cam_fps = self.ab_camera_fps_var.get() if hasattr(self, "ab_camera_fps_var") else "50"
    try:
        camera_fps = float(str(raw_cam_fps).strip().replace(",", "."))
    except Exception:
        self.ab_status_var.set("Auto benchmark: niepoprawny FPS (jedna liczba)")
        return
    if camera_fps <= 0:
        self.ab_status_var.set("Auto benchmark: FPS musi byc > 0")
        return
    camera_fps_list = [camera_fps]
    modes = self._parse_modes_csv(self.ab_modes_var.get())

    warmup_s = max(0.5, float(self.ab_warmup_s_var.get() or 4))
    measure_s = max(1.0, float(self.ab_measure_s_var.get() or 10))
    bin_s = max(1.0, float(self.ab_bin_s_var.get() or 10))
    buffer_count = max(0, int(float(self.ab_buffer_count_var.get() or 0)))
    buffer_pause_s = max(0.5, float(self.ab_buffer_pause_s_var.get() or 3))

    cases = []
    for entry in model_entries:
        for imgsz in imgsz_list:
            for b in batch_list:
                for fps in camera_fps_list:
                    for mode in modes:
                        cases.append({
                            "model": entry["model_path"],
                            "imgsz": imgsz,
                            "batch_images": b,
                            "camera_fps": float(fps),
                            "mode": mode,
                            "backend": entry["backend"],
                            "engine_path": entry["engine_path"],
                        })
    if not cases:
        self.ab_status_var.set("Auto benchmark: brak przypadkow testowych")
        return

    run_base = self._resolve_benchmark_output_base()
    run_paths = prepare_run_directory(run_base, mode="live")
    run_dir = str(run_paths["run_dir"])
    out_path = str(run_paths["summary_csv"])
    cases_dir = str(run_paths["cases_dir"])
    cases_index_path = str(run_paths["cases_index_json"])
    manifest_path = str(run_paths["manifest_json"])
    self.ab_out_var.set(out_path)

    self._auto_bench_results = []
    self._auto_bench_state = {
        "running": True,
        "cases": cases,
        "idx": -1,
        "phase": "next_case",
        "phase_t0": time.time(),
        "warmup_s": warmup_s,
        "measure_s": measure_s,
        "bin_s": bin_s,
        "buffer_count": buffer_count,
        "buffer_pause_s": buffer_pause_s,
        "recording": False,
        "next_buffer_ts": 0.0,
        "buffers_done": 0,
        "measure_samples": [],
        "router_baseline": None,
        "last_sys_sample_t": 0.0,
        "last_sys_sample": {},
        "case_started_t": 0.0,
        "out_path": out_path,
        "cases_dir": cases_dir,
        "cases_index_path": cases_index_path,
        "manifest_path": manifest_path,
        "run_dir": run_dir,
        "case_json_paths": [],
        "started_at_iso": utc_now_iso(),
        "started_at_epoch": time.time(),
        "benchmark_mode": "live",
        "parameter_grid": {
            "models": [str(e["model_path"]) for e in model_entries],
            "image_sizes": [int(x) for x in imgsz_list],
            "batch_images": [int(x) for x in batch_list],
            "camera_fps": [float(x) for x in camera_fps_list],
            "modes": [str(x) for x in modes],
            "backend_mode": backend_mode,
            "resolved_backends": sorted({str(e["backend"]) for e in model_entries}),
            "warmup_s": float(warmup_s),
            "measure_s": float(measure_s),
            "bin_s": float(bin_s),
            "buffer_count": int(buffer_count),
            "buffer_pause_s": float(buffer_pause_s),
        },
    }
    self._update_auto_bench_plan_summary()
    self.ab_status_var.set(f"Auto benchmark: start ({len(cases)} przypadkow)")
    self._set_status("Auto benchmark: start", "accent")
    self._after(50, self._auto_benchmark_tick)

def stop_auto_benchmark(self):
    if self._auto_bench_state:
        self._auto_bench_state["running"] = False
    self._auto_benchmark_reset_queues()
    self.ab_status_var.set("Auto benchmark: zatrzymywany...")

def _auto_benchmark_reset_queues(self):
    try:
        if self.control_q is not None:
            self.control_q.put(("yolo_reset_queues", True), timeout=0.2)
    except queue.Full:
        print("[GUI] auto bench queue reset skipped: control_q full")
    except (OSError, ValueError) as e:
        print(f"[GUI] auto bench queue reset failed: {e}")

def _auto_benchmark_tick(self):
    st = self._auto_bench_state
    if not st or not st.get("running"):
        if st and st.get("recording"):
            try:
                self.on_stop_record()
            except Exception:
                pass
        self._auto_bench_state = None
        self.ab_status_var.set("Auto benchmark: zatrzymany")
        return

    now = time.time()
    phase = st.get("phase")

    if phase == "next_case":
        st["idx"] += 1
        if st["idx"] >= len(st["cases"]):
            self._auto_benchmark_finish()
            return
        case = st["cases"][st["idx"]]
        try:
            cam_fps = float(case.get("camera_fps", 50.0))
            cam_period_us = self._fps_to_period_us(cam_fps)
        except (TypeError, ValueError):
            self.ab_status_var.set("Auto benchmark: błędny FPS w case")
            st["phase"] = "next_case"
            st["phase_t0"] = now
            self._after(200, self._auto_benchmark_tick)
            return
        try:
            if self.control_q is not None:
                case_backend = self._normalize_live_infer_backend(case.get("backend", "ultralytics"))
                case_engine_path = str(case.get("engine_path", "") or "")
                self.control_q.put(("yolo_set_backend", case_backend), timeout=0.2)
                if case_backend == "tensorrt":
                    self.control_q.put(("yolo_set_tensorrt_engine", case_engine_path), timeout=0.2)
                self.control_q.put(("camera_set_fps", cam_fps), timeout=0.2)
                self.control_q.put(("yolo_load_model", case["model"]), timeout=0.2)
                self.control_q.put(("yolo_set_image_size", int(case["imgsz"])), timeout=0.2)
                self.control_q.put(("yolo_set_input_batch_images", int(case["batch_images"])), timeout=0.2)
                self.control_q.put(("yolo", True), timeout=0.2)
                try:
                    if self.shared_state is not None:
                        self.shared_state["yolo_backend"] = case_backend
                        self.shared_state["yolo_trt_engine_path"] = case_engine_path if case_backend == "tensorrt" else ""
                except Exception:
                    pass
            try:
                self.ab_camera_period_var.set(f"{cam_period_us:.1f} us")
            except tk.TclError:
                pass
        except queue.Full:
            self.ab_status_var.set("Auto benchmark: control_q full, retry...")
            self._after(200, self._auto_benchmark_tick)
            return
        except (OSError, ValueError, TypeError) as e:
            print(f"[GUI][AUTO_BENCH] control setup failed: {e}")
            st["phase"] = "next_case"
            st["phase_t0"] = now
            self._after(200, self._auto_benchmark_tick)
            return

        if case["mode"] in ("bin", "bin+buffer"):
            try:
                self.on_start_record()
                st["recording"] = True
            except Exception:
                st["recording"] = False
        else:
            st["recording"] = False

        st["buffers_done"] = 0
        st["next_buffer_ts"] = now
        st["measure_samples"] = []
        st["router_baseline"] = None
        st["last_sys_sample_t"] = 0.0
        st["last_sys_sample"] = {}
        self._last_yolo_stats = None
        self._last_router_stats = None
        st["phase"] = "warmup"
        st["phase_t0"] = now
        st["case_started_t"] = now
        self.ab_status_var.set(
            f"Case {st['idx'] + 1}/{len(st['cases'])}: {os.path.basename(case['model'])} | "
            f"imgsz={case['imgsz']} batch={case['batch_images']} fps={float(case.get('camera_fps', 50.0)):.1f} "
            f"mode={case['mode']} | warmup..."
        )
        self._after(200, self._auto_benchmark_tick)
        return

    if phase == "warmup":
        if (now - st["phase_t0"]) >= st["warmup_s"]:
            st["phase"] = "measure"
            st["phase_t0"] = now
        self._after(200, self._auto_benchmark_tick)
        return

    if phase == "measure":
        case = st["cases"][st["idx"]]
        if (now - float(st.get("last_sys_sample_t", 0.0))) >= 1.0:
            st["last_sys_sample"] = self._sample_system_metrics()
            st["last_sys_sample_t"] = now
        if case["mode"] in ("buffer", "bin+buffer") and st["buffers_done"] < st["buffer_count"]:
            if now >= st["next_buffer_ts"]:
                try:
                    self.on_save_buffer()
                except Exception:
                    pass
                st["buffers_done"] += 1
                st["next_buffer_ts"] = now + st["buffer_pause_s"]

        stats = self._last_yolo_stats or {}
        if not (isinstance(stats, dict) and ("avg_infer_ms" in stats or "images_per_sec" in stats)):
            fallback_stats = self._get_shared_state_value("yolo_buffer_stats", None)
            if isinstance(fallback_stats, dict) and ("avg_infer_ms" in fallback_stats or "images_per_sec" in fallback_stats):
                stats = fallback_stats
        if isinstance(stats, dict) and ("avg_infer_ms" in stats or "images_per_sec" in stats):
            router = self._last_router_stats or {}
            det_counter = 0.0
            for det_key in ("total_detections", "detections_total", "total_detections_made", "detections"):
                if det_key in stats:
                    try:
                        det_counter = float(stats.get(det_key, 0) or 0)
                        break
                    except Exception:
                        pass
            bucket_counts = {}
            try:
                bc = router.get("bucket_fill_counts", {}) if isinstance(router, dict) else {}
                if isinstance(bc, dict):
                    bucket_counts = {str(k): int(v) for k, v in bc.items() if str(k).isdigit()}
            except Exception:
                bucket_counts = {}
            expected_roles = int(router.get("expected_roles", 4) or 4) if isinstance(router, dict) else 4
            if st.get("router_baseline") is None and isinstance(router, dict):
                st["router_baseline"] = {
                    "dropped": float(router.get("dropped", 0) or 0),
                    "dropped_incomplete": float(router.get("dropped_incomplete", 0) or 0),
                    "batch_q_full_count": float(router.get("batch_q_full_count", 0) or 0),
                    "steps_completed": float(router.get("steps_completed", 0) or 0),
                    "full_steps": float(router.get("full_steps", 0) or 0),
                }
            bbox_result_age_ms = self._estimate_bbox_result_age_ms(now)
            st["measure_samples"].append({
                "t": now,
                "avg_infer_ms": float(stats.get("avg_infer_ms", 0) or 0),
                "images_per_sec": float(stats.get("images_per_sec", 0) or 0),
                "steps_per_sec": float(stats.get("steps_per_sec", 0) or 0),
                "queue_wait_ms": float(stats.get("queue_wait_ms", 0) or 0),
                "batch_collect_ms": float(stats.get("batch_collect_ms", 0) or 0),
                "prepare_ms": float(stats.get("prepare_ms", 0) or 0),
                "forward_ms": float(stats.get("forward_ms", 0) or 0),
                "post_ms": float(stats.get("post_ms", 0) or 0),
                "e2e_pipeline_ms": float(stats.get("e2e_pipeline_ms", 0) or 0),
                "queue_wait_percent": float(stats.get("queue_wait_percent", 0) or 0),
                "prepare_percent": float(stats.get("prepare_percent", 0) or 0),
                "forward_percent": float(stats.get("forward_percent", 0) or 0),
                "post_percent": float(stats.get("post_percent", 0) or 0),
                "batches_per_sec": float(stats.get("batches_per_sec", 0) or 0),
                "target_util_percent": float(stats.get("target_util_percent", 0) or 0),
                "det_counter": det_counter,
                "drop_gap_frames": float(stats.get("drop_gap_frames", 0) or 0),
                "bbox_result_age_ms": (float(bbox_result_age_ms) if bbox_result_age_ms is not None else None),
                "worker_get_errors": float(stats.get("worker_get_errors", 0) or 0),
                "worker_infer_errors": float(stats.get("worker_infer_errors", 0) or 0),
                "worker_vis_errors": float(stats.get("worker_vis_errors", 0) or 0),
                "router_dropped": float(router.get("dropped", 0) or 0) if isinstance(router, dict) else 0.0,
                "router_dropped_incomplete": float(router.get("dropped_incomplete", 0) or 0) if isinstance(router, dict) else 0.0,
                "router_batch_q_full_count": float(router.get("batch_q_full_count", 0) or 0) if isinstance(router, dict) else 0.0,
                "router_steps_completed": float(router.get("steps_completed", 0) or 0) if isinstance(router, dict) else 0.0,
                "router_full_steps": float(router.get("full_steps", 0) or 0) if isinstance(router, dict) else 0.0,
                "router_pending_buckets": float(router.get("pending_buckets", 0) or 0) if isinstance(router, dict) else 0.0,
                "router_ready_steps": float(router.get("ready_steps", 0) or 0) if isinstance(router, dict) else 0.0,
                "router_yolo_raw_q": float(router.get("yolo_raw_q", 0) or 0) if isinstance(router, dict) else 0.0,
                "router_yolo_batch_q": float(router.get("yolo_batch_q", 0) or 0) if isinstance(router, dict) else 0.0,
                "router_oldest_pending_bucket_age_ms": float(router.get("oldest_pending_bucket_age_ms", 0) or 0) if isinstance(router, dict) else 0.0,
                "router_oldest_ready_step_age_ms": float(router.get("oldest_ready_step_age_ms", 0) or 0) if isinstance(router, dict) else 0.0,
                "router_expected_roles": float(expected_roles),
                "bucket_fill_1": float(bucket_counts.get("1", 0)),
                "bucket_fill_2": float(bucket_counts.get("2", 0)),
                "bucket_fill_3": float(bucket_counts.get("3", 0)),
                "bucket_fill_4": float(bucket_counts.get("4", 0)),
                "cpu_percent": float((st.get("last_sys_sample", {}) or {}).get("cpu_percent", 0.0) or 0.0),
                "ram_percent": float((st.get("last_sys_sample", {}) or {}).get("ram_percent", 0.0) or 0.0),
                "disk_write_mb_s": float((st.get("last_sys_sample", {}) or {}).get("disk_write_mb_s", 0.0) or 0.0),
                "gpu_util_percent": float((st.get("last_sys_sample", {}) or {}).get("gpu_util_percent", 0.0) or 0.0),
                "gpu_mem_percent": float((st.get("last_sys_sample", {}) or {}).get("gpu_mem_percent", 0.0) or 0.0),
                "gpu_temp_c": float((st.get("last_sys_sample", {}) or {}).get("gpu_temp_c", 0.0) or 0.0),
                "gpu_power_w": float((st.get("last_sys_sample", {}) or {}).get("gpu_power_w", 0.0) or 0.0),
                "configured_batch_size": float(stats.get("configured_batch_size", 0) or 0),
                "configured_input_batch_images": float(stats.get("configured_input_batch_images", 0) or 0),
                "configured_seq_len": float(stats.get("configured_seq_len", 0) or 0),
            })

        if st.get("recording") and (now - st["case_started_t"]) >= st["bin_s"]:
            try:
                self.on_stop_record()
            except Exception:
                pass
            st["recording"] = False

        if (now - st["phase_t0"]) >= st["measure_s"]:
            if st.get("recording"):
                try:
                    self.on_stop_record()
                except Exception:
                    pass
                st["recording"] = False

            samples = st.get("measure_samples", [])
            if samples:
                infer_vals = [float(s.get("avg_infer_ms", 0.0)) for s in samples]
                ips_vals = [float(s.get("images_per_sec", 0.0)) for s in samples]
                sps_vals = [float(s.get("steps_per_sec", 0.0)) for s in samples]
                qw_vals = [float(s.get("queue_wait_ms", 0.0)) for s in samples]
                collect_vals = [float(s.get("batch_collect_ms", 0.0)) for s in samples]
                prep_vals = [float(s.get("prepare_ms", 0.0)) for s in samples]
                fwd_vals = [float(s.get("forward_ms", 0.0)) for s in samples]
                post_vals = [float(s.get("post_ms", 0.0)) for s in samples]
                e2e_vals = [float(s.get("e2e_pipeline_ms", 0.0)) for s in samples]
                q_pct_vals = [float(s.get("queue_wait_percent", 0.0)) for s in samples]
                prep_pct_vals = [float(s.get("prepare_percent", 0.0)) for s in samples]
                fwd_pct_vals = [float(s.get("forward_percent", 0.0)) for s in samples]
                post_pct_vals = [float(s.get("post_percent", 0.0)) for s in samples]
                util_vals = [float(s.get("target_util_percent", 0.0)) for s in samples]
                det_vals = [float(s.get("det_counter", 0.0)) for s in samples]
                drop_gap_vals = [float(s.get("drop_gap_frames", 0.0)) for s in samples]
                bbox_age_vals = [
                    float(s.get("bbox_result_age_ms"))
                    for s in samples
                    if s.get("bbox_result_age_ms") is not None
                ]
                pending_bucket_vals = [float(s.get("router_pending_buckets", 0.0)) for s in samples]
                ready_steps_vals = [float(s.get("router_ready_steps", 0.0)) for s in samples]
                raw_q_vals = [float(s.get("router_yolo_raw_q", 0.0)) for s in samples]
                batch_q_vals = [float(s.get("router_yolo_batch_q", 0.0)) for s in samples]
                bucket1_vals = [float(s.get("bucket_fill_1", 0.0)) for s in samples]
                bucket2_vals = [float(s.get("bucket_fill_2", 0.0)) for s in samples]
                bucket3_vals = [float(s.get("bucket_fill_3", 0.0)) for s in samples]
                bucket4_vals = [float(s.get("bucket_fill_4", 0.0)) for s in samples]
                oldest_pending_age_vals = [float(s.get("router_oldest_pending_bucket_age_ms", 0.0)) for s in samples]
                oldest_ready_age_vals = [float(s.get("router_oldest_ready_step_age_ms", 0.0)) for s in samples]
                cpu_vals = [float(s.get("cpu_percent", 0.0)) for s in samples]
                ram_vals = [float(s.get("ram_percent", 0.0)) for s in samples]
                disk_vals = [float(s.get("disk_write_mb_s", 0.0)) for s in samples]
                gpu_util_vals = [float(s.get("gpu_util_percent", 0.0)) for s in samples]
                gpu_mem_vals = [float(s.get("gpu_mem_percent", 0.0)) for s in samples]
                gpu_temp_vals = [float(s.get("gpu_temp_c", 0.0)) for s in samples]
                gpu_power_vals = [float(s.get("gpu_power_w", 0.0)) for s in samples]
                infer_s = self._safe_stats(infer_vals)
                ips_s = self._safe_stats(ips_vals)
                sps_s = self._safe_stats(sps_vals)
                qw_s = self._safe_stats(qw_vals)
                collect_s = self._safe_stats(collect_vals)
                prep_s = self._safe_stats(prep_vals)
                fwd_s = self._safe_stats(fwd_vals)
                post_s = self._safe_stats(post_vals)
                e2e_s = self._safe_stats(e2e_vals)
                q_pct_s = self._safe_stats(q_pct_vals)
                prep_pct_s = self._safe_stats(prep_pct_vals)
                fwd_pct_s = self._safe_stats(fwd_pct_vals)
                post_pct_s = self._safe_stats(post_pct_vals)
                util_s = self._safe_stats(util_vals)
                drop_gap_s = self._safe_stats(drop_gap_vals)
                bbox_age_s = self._safe_stats(bbox_age_vals) if bbox_age_vals else None
                pending_bucket_s = self._safe_stats(pending_bucket_vals)
                ready_steps_s = self._safe_stats(ready_steps_vals)
                raw_q_s = self._safe_stats(raw_q_vals)
                batch_q_s = self._safe_stats(batch_q_vals)
                bucket1_s = self._safe_stats(bucket1_vals)
                bucket2_s = self._safe_stats(bucket2_vals)
                bucket3_s = self._safe_stats(bucket3_vals)
                bucket4_s = self._safe_stats(bucket4_vals)
                oldest_pending_age_s = self._safe_stats(oldest_pending_age_vals)
                oldest_ready_age_s = self._safe_stats(oldest_ready_age_vals)
                cpu_s = self._safe_stats(cpu_vals)
                ram_s = self._safe_stats(ram_vals)
                disk_s = self._safe_stats(disk_vals)
                gpu_util_s = self._safe_stats(gpu_util_vals)
                gpu_mem_s = self._safe_stats(gpu_mem_vals)
                gpu_temp_s = self._safe_stats(gpu_temp_vals)
                gpu_power_s = self._safe_stats(gpu_power_vals)

                baseline = st.get("router_baseline") or {}
                dropped_delta = max(0.0, float(samples[-1].get("router_dropped", 0.0)) - float(baseline.get("dropped", 0.0)))
                dropped_incomplete_delta = max(
                    0.0,
                    float(samples[-1].get("router_dropped_incomplete", 0.0)) - float(baseline.get("dropped_incomplete", 0.0)),
                )
                batch_q_full_delta = max(
                    0.0,
                    float(samples[-1].get("router_batch_q_full_count", 0.0)) - float(baseline.get("batch_q_full_count", 0.0)),
                )
                steps_completed_delta = max(
                    0.0,
                    float(samples[-1].get("router_steps_completed", 0.0)) - float(baseline.get("steps_completed", 0.0)),
                )
                full_steps_delta = max(
                    0.0,
                    float(samples[-1].get("router_full_steps", 0.0)) - float(baseline.get("full_steps", 0.0)),
                )
                bucket_fill_total_avg = bucket1_s["mean"] + bucket2_s["mean"] + bucket3_s["mean"] + bucket4_s["mean"]
                bucket_full_ratio = (bucket4_s["mean"] / bucket_fill_total_avg) if bucket_fill_total_avg > 1e-9 else 0.0
                get_errors_total = max(0.0, float(samples[-1].get("worker_get_errors", 0.0)) - float(samples[0].get("worker_get_errors", 0.0)))
                infer_errors_total = max(0.0, float(samples[-1].get("worker_infer_errors", 0.0)) - float(samples[0].get("worker_infer_errors", 0.0)))
                vis_errors_total = max(0.0, float(samples[-1].get("worker_vis_errors", 0.0)) - float(samples[0].get("worker_vis_errors", 0.0)))
                det_total = max(0.0, det_vals[-1] - det_vals[0]) if len(det_vals) >= 2 else 0.0
                duration_s = max(1e-6, float(samples[-1].get("t", now)) - float(samples[0].get("t", now)))
                det_per_sec = det_total / duration_s
                ts_vals = [float(s.get("t", 0.0)) for s in samples]
                ips_slope = self._trend_slope(ts_vals, ips_vals)
                infer_slope = self._trend_slope(ts_vals, infer_vals)
                queue_wait_slope = self._trend_slope(ts_vals, qw_vals)
                e2e_slope = self._trend_slope(ts_vals, e2e_vals)
                avg_infer = infer_s["mean"]
                avg_ips = ips_s["mean"]
                avg_sps = sps_s["mean"]
                avg_qw = qw_s["mean"]
            else:
                infer_s = ips_s = sps_s = qw_s = collect_s = prep_s = fwd_s = post_s = util_s = self._safe_stats([])
                e2e_s = q_pct_s = prep_pct_s = fwd_pct_s = post_pct_s = self._safe_stats([])
                pending_bucket_s = ready_steps_s = raw_q_s = batch_q_s = bucket1_s = bucket2_s = bucket3_s = bucket4_s = self._safe_stats([])
                oldest_pending_age_s = oldest_ready_age_s = self._safe_stats([])
                cpu_s = ram_s = disk_s = gpu_util_s = gpu_mem_s = gpu_temp_s = gpu_power_s = self._safe_stats([])
                dropped_delta = dropped_incomplete_delta = batch_q_full_delta = 0.0
                steps_completed_delta = full_steps_delta = 0.0
                bucket_full_ratio = 0.0
                get_errors_total = infer_errors_total = vis_errors_total = 0.0
                ips_slope = infer_slope = queue_wait_slope = e2e_slope = 0.0
                det_total = det_per_sec = 0.0
                avg_infer = avg_ips = avg_sps = avg_qw = 0.0
                drop_gap_s = self._safe_stats([])
                bbox_age_s = None

            case_id = f"case_{int(st.get('idx', 0)) + 1:04d}"
            backend_name = self._normalize_live_infer_backend(
                case.get("backend", self._get_shared_state_value("yolo_backend", "ultralytics"))
            )
            trt_engine_path = str(case.get("engine_path", "") or "")
            yolo_device = str(self._get_shared_state_value("yolo_device", "cuda") or "cuda")
            yolo_seq_len = int(self._get_shared_state_value("yolo_seq_len", 1) or 1)
            yolo_chunk_batch = int(self._get_shared_state_value("yolo_batch_size", 0) or 0)
            if yolo_seq_len <= 0 and samples:
                try:
                    yolo_seq_len = int(samples[-1].get("configured_seq_len", 1) or 1)
                except Exception:
                    yolo_seq_len = 1
            if yolo_chunk_batch <= 0 and samples:
                for key in ("configured_batch_size", "configured_input_batch_images"):
                    try:
                        candidate = int(samples[-1].get(key, 0) or 0)
                    except Exception:
                        candidate = 0
                    if candidate > 0:
                        yolo_chunk_batch = candidate
                        break
            if yolo_chunk_batch <= 0:
                yolo_chunk_batch = int(case.get("batch_images", 0) or 0)
            yolo_ball_class_id = int(self._get_shared_state_value("yolo_ball_class_id", 0) or 0)
            yolo_conf = float(self._get_shared_state_value("yolo_confidence", 0.25) or 0.25)
            bbox_mean = round(bbox_age_s["mean"], 3) if bbox_age_s is not None else None
            bbox_p50 = round(bbox_age_s["p50"], 3) if bbox_age_s is not None else None
            bbox_p95 = round(bbox_age_s["p95"], 3) if bbox_age_s is not None else None
            bbox_p99 = round(bbox_age_s["p99"], 3) if bbox_age_s is not None else None

            row_result = {
                "case_id": case_id,
                "benchmark_mode": "live",
                "backend": backend_name,
                "model_path": case["model"],
                "trt_engine_path": trt_engine_path,
                "device": yolo_device,
                "input_format": "live_camera_bayer_bg",
                "imgsz": int(case["imgsz"]),
                "seq_len": int(yolo_seq_len),
                "chunk_batch_size": int(yolo_chunk_batch),
                "images_per_call": int(case["batch_images"]),
                "object_scale": None,
                "repeats": int(len(samples)),
                "warmup": float(st["warmup_s"]),
                "ball_class_id": int(yolo_ball_class_id),
                "conf_threshold": float(yolo_conf),
                "model": case["model"],
                "model_name": os.path.basename(case["model"]),
                "batch_images": int(case["batch_images"]),
                "camera_fps": round(float(case.get("camera_fps", 50.0)), 3),
                "camera_period_us": round(self._fps_to_period_us(float(case.get("camera_fps", 50.0))), 3),
                "mode": case["mode"],
                "warmup_s": float(st["warmup_s"]),
                "measure_s": float(st["measure_s"]),
                "bin_s": float(st["bin_s"]),
                "buffer_count": int(st["buffer_count"]),
                "buffer_pause_s": float(st["buffer_pause_s"]),
                "avg_infer_ms": round(infer_s["mean"], 3),
                "infer_ms_p50": round(infer_s["p50"], 3),
                "infer_ms_p95": round(infer_s["p95"], 3),
                "infer_ms_p99": round(infer_s["p99"], 3),
                "infer_ms_std": round(infer_s["std"], 3),
                "infer_ms_cv": round(infer_s["cv"], 4),
                "images_per_sec": round(ips_s["mean"], 3),
                "images_per_sec_p50": round(ips_s["p50"], 3),
                "images_per_sec_p95": round(ips_s["p95"], 3),
                "images_per_sec_std": round(ips_s["std"], 3),
                "images_per_sec_cv": round(ips_s["cv"], 4),
                "steps_per_sec": round(sps_s["mean"], 3),
                "steps_per_sec_p50": round(sps_s["p50"], 3),
                "steps_per_sec_p95": round(sps_s["p95"], 3),
                "queue_wait_ms": round(qw_s["mean"], 3),
                "queue_wait_ms_p95": round(qw_s["p95"], 3),
                "queue_wait_ms_p99": round(qw_s["p99"], 3),
                "queue_wait_ms_std": round(qw_s["std"], 3),
                "batch_collect_ms": round(collect_s["mean"], 3),
                "prepare_ms": round(prep_s["mean"], 3),
                "forward_ms": round(fwd_s["mean"], 3),
                "post_ms": round(post_s["mean"], 3),
                "e2e_pipeline_ms": round(e2e_s["mean"], 3),
                "e2e_pipeline_ms_p95": round(e2e_s["p95"], 3),
                "e2e_pipeline_ms_p99": round(e2e_s["p99"], 3),
                "e2e_pipeline_ms_std": round(e2e_s["std"], 3),
                "e2e_pipeline_ms_cv": round(e2e_s["cv"], 4),
                "queue_wait_percent": round(q_pct_s["mean"], 3),
                "prepare_percent": round(prep_pct_s["mean"], 3),
                "forward_percent": round(fwd_pct_s["mean"], 3),
                "post_percent": round(post_pct_s["mean"], 3),
                "target_util_percent": round(util_s["mean"], 3),
                "queue_raw_q_avg": round(raw_q_s["mean"], 3),
                "queue_raw_q_p95": round(raw_q_s["p95"], 3),
                "queue_batch_q_avg": round(batch_q_s["mean"], 3),
                "queue_batch_q_p95": round(batch_q_s["p95"], 3),
                "pending_buckets_avg": round(pending_bucket_s["mean"], 3),
                "pending_buckets_p95": round(pending_bucket_s["p95"], 3),
                "ready_steps_avg": round(ready_steps_s["mean"], 3),
                "ready_steps_p95": round(ready_steps_s["p95"], 3),
                "oldest_pending_bucket_age_ms_avg": round(oldest_pending_age_s["mean"], 3),
                "oldest_pending_bucket_age_ms_p95": round(oldest_pending_age_s["p95"], 3),
                "oldest_ready_step_age_ms_avg": round(oldest_ready_age_s["mean"], 3),
                "oldest_ready_step_age_ms_p95": round(oldest_ready_age_s["p95"], 3),
                "bucket_fill_1_avg": round(bucket1_s["mean"], 3),
                "bucket_fill_2_avg": round(bucket2_s["mean"], 3),
                "bucket_fill_3_avg": round(bucket3_s["mean"], 3),
                "bucket_fill_4_avg": round(bucket4_s["mean"], 3),
                "bucket_full_ratio": round(bucket_full_ratio, 4),
                "drops_total": round(dropped_delta, 3),
                "drops_incomplete_total": round(dropped_incomplete_delta, 3),
                "batch_q_full_events": round(batch_q_full_delta, 3),
                "steps_completed_total": round(steps_completed_delta, 3),
                "full_steps_total": round(full_steps_delta, 3),
                "worker_get_errors": round(get_errors_total, 3),
                "worker_infer_errors": round(infer_errors_total, 3),
                "worker_vis_errors": round(vis_errors_total, 3),
                "cpu_percent_avg": round(cpu_s["mean"], 3),
                "cpu_percent_p95": round(cpu_s["p95"], 3),
                "ram_percent_avg": round(ram_s["mean"], 3),
                "ram_percent_p95": round(ram_s["p95"], 3),
                "disk_write_mb_s_avg": round(disk_s["mean"], 3),
                "disk_write_mb_s_p95": round(disk_s["p95"], 3),
                "gpu_util_percent_avg": round(gpu_util_s["mean"], 3),
                "gpu_util_percent_p95": round(gpu_util_s["p95"], 3),
                "gpu_mem_percent_avg": round(gpu_mem_s["mean"], 3),
                "gpu_mem_percent_p95": round(gpu_mem_s["p95"], 3),
                "gpu_temp_c_avg": round(gpu_temp_s["mean"], 3),
                "gpu_temp_c_p95": round(gpu_temp_s["p95"], 3),
                "gpu_power_w_avg": round(gpu_power_s["mean"], 3),
                "gpu_power_w_p95": round(gpu_power_s["p95"], 3),
                "images_per_sec_per_watt": round((ips_s["mean"] / gpu_power_s["mean"]) if gpu_power_s["mean"] > 1e-9 else 0.0, 6),
                "ips_slope_per_s": round(ips_slope, 6),
                "infer_slope_per_s": round(infer_slope, 6),
                "queue_wait_slope_per_s": round(queue_wait_slope, 6),
                "e2e_slope_per_s": round(e2e_slope, 6),
                "detections_total": round(det_total, 3),
                "detections_per_sec": round(det_per_sec, 3),
                "samples": len(samples),
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_ms_mean": round(infer_s["mean"], 3),
                "total_ms_p50": round(infer_s["p50"], 3),
                "total_ms_p95": round(infer_s["p95"], 3),
                "total_ms_p99": round(infer_s["p99"], 3),
                "prepare_ms_mean": round(prep_s["mean"], 3),
                "forward_ms_mean": round(fwd_s["mean"], 3),
                "post_ms_mean": round(post_s["mean"], 3),
                "fps_calls": round(self._safe_stats([float(s.get("batches_per_sec", 0.0)) for s in samples])["mean"], 6) if samples else 0.0,
                "fps_images": round(ips_s["mean"], 6),
                "detection_rate_frames": None,
                "mean_max_conf": None,
                "total_ms_std": round(infer_s["std"], 6),
                "total_ms_cv": round(infer_s["cv"], 6),
                "total_ms_slope": round(infer_slope, 6),
                "batch_collect_ms_mean": round(collect_s["mean"], 6),
                "queue_wait_ms_mean": round(qw_s["mean"], 6),
                "e2e_pipeline_ms_mean": round(e2e_s["mean"], 6),
                "e2e_pipeline_ms_p50": round(e2e_s["p50"], 6),
                "e2e_pipeline_ms_p95": round(e2e_s["p95"], 6),
                "e2e_pipeline_ms_p99": round(e2e_s["p99"], 6),
                "drop_gap_frames": round(drop_gap_s["max"], 6),
                "batch_q_full_count": round(batch_q_full_delta, 6),
                "pending_buckets_max": round(pending_bucket_s["max"], 6),
                "ready_steps_max": round(ready_steps_s["max"], 6),
                "yolo_raw_q_max": round(raw_q_s["max"], 6),
                "yolo_batch_q_max": round(batch_q_s["max"], 6),
                "cpu_percent_mean": round(cpu_s["mean"], 6),
                "ram_percent_mean": round(ram_s["mean"], 6),
                "gpu_percent_mean": round(gpu_util_s["mean"], 6),
                "gpu_mem_percent_mean": round(gpu_mem_s["mean"], 6),
                "bbox_result_age_ms_mean": bbox_mean,
                "bbox_result_age_ms_p50": bbox_p50,
                "bbox_result_age_ms_p95": bbox_p95,
                "bbox_result_age_ms_p99": bbox_p99,
            }
            self._auto_bench_results.append(row_result)
            try:
                row = self._auto_bench_results[-1] if self._auto_bench_results else {}
                cases_dir = str(st.get("cases_dir") or "")
                if cases_dir:
                    case_id = str(row.get("case_id", f"case_{int(st.get('idx', 0)) + 1:04d}"))
                    case_name = f"{case_id}.json"
                    case_path = os.path.join(cases_dir, case_name)
                    samples_full_path = os.path.join(cases_dir, f"{case_id}_samples_full.json")
                    batch_ts_path = os.path.join(cases_dir, f"{case_id}_yolo_last_batch_timestamps.json")
                    infer_snap_path = os.path.join(cases_dir, f"{case_id}_yolo_last_inference_snapshot.json")

                    raw_samples_payload = {"measure_samples": samples}
                    write_json(samples_full_path, raw_samples_payload)

                    batch_ts_snapshot = self._get_shared_state_value("yolo_last_batch_timestamps", None)
                    infer_snapshot = self._get_shared_state_value("yolo_last_inference_snapshot", None)
                    if isinstance(batch_ts_snapshot, dict):
                        write_json(batch_ts_path, batch_ts_snapshot)
                    else:
                        batch_ts_path = None
                    if isinstance(infer_snapshot, dict):
                        write_json(infer_snap_path, infer_snapshot)
                    else:
                        infer_snap_path = None

                    downsample_n = 450
                    sampled = {
                        "total_ms_samples": downsample([float(s.get("avg_infer_ms", 0.0)) for s in samples], downsample_n),
                        "prepare_ms_samples": downsample([float(s.get("prepare_ms", 0.0)) for s in samples], downsample_n),
                        "forward_ms_samples": downsample([float(s.get("forward_ms", 0.0)) for s in samples], downsample_n),
                        "post_ms_samples": downsample([float(s.get("post_ms", 0.0)) for s in samples], downsample_n),
                        "batch_collect_ms_samples": downsample([float(s.get("batch_collect_ms", 0.0)) for s in samples], downsample_n),
                        "queue_wait_ms_samples": downsample([float(s.get("queue_wait_ms", 0.0)) for s in samples], downsample_n),
                        "e2e_pipeline_ms_samples": downsample([float(s.get("e2e_pipeline_ms", 0.0)) for s in samples], downsample_n),
                        "bbox_result_age_ms_samples": downsample(
                            [float(s.get("bbox_result_age_ms")) for s in samples if s.get("bbox_result_age_ms") is not None],
                            downsample_n,
                        ),
                    }
                    payload = {
                        "schema": "benchmark_case_v2",
                        "generated_at": utc_now_iso(),
                        "case_id": case_id,
                        "benchmark_mode": "live",
                        "case_index": int(st.get("idx", 0)) + 1,
                        "case_total": len(st.get("cases", [])),
                        "configuration": {
                            "model": case.get("model"),
                            "imgsz": int(case.get("imgsz", 0)),
                            "batch_images": int(case.get("batch_images", 0)),
                            "camera_fps": float(case.get("camera_fps", 0.0)),
                            "mode": case.get("mode", "none"),
                            "backend": row.get("backend"),
                            "device": row.get("device"),
                            "trt_engine_path": row.get("trt_engine_path"),
                            "seq_len": row.get("seq_len"),
                            "chunk_batch_size": row.get("chunk_batch_size"),
                            "ball_class_id": row.get("ball_class_id"),
                            "conf_threshold": row.get("conf_threshold"),
                        },
                        "aggregates": row,
                        "samples": sampled,
                        "artifacts": {
                            "samples_full_path": os.path.relpath(samples_full_path, st.get("run_dir", cases_dir)),
                            "yolo_last_batch_timestamps_path": (
                                os.path.relpath(batch_ts_path, st.get("run_dir", cases_dir)) if batch_ts_path else None
                            ),
                            "yolo_last_inference_snapshot_path": (
                                os.path.relpath(infer_snap_path, st.get("run_dir", cases_dir)) if infer_snap_path else None
                            ),
                            "step_trace_path": None,
                            "frame_trace_path": None,
                        },
                    }
                    write_json(case_path, payload)
                    st.setdefault("case_json_paths", []).append(case_path)
            except Exception as e:
                print(f"[GUI][AUTO_BENCH] case json save failed: {e}")

            self.ab_status_var.set(
                f"Case {st['idx'] + 1}/{len(st['cases'])} done | img/s={avg_ips:.1f} ms={avg_infer:.1f}"
            )
            st["phase"] = "next_case"
            st["phase_t0"] = now

        self._after(200, self._auto_benchmark_tick)
        return

    self._after(250, self._auto_benchmark_tick)

def _auto_benchmark_finish(self):
    st = self._auto_bench_state or {}
    out_path = str(st.get("out_path") or "")
    run_dir = str(st.get("run_dir") or "")
    cases_dir = str(st.get("cases_dir") or "")
    index_path = str(st.get("cases_index_path") or "")
    manifest_path = str(st.get("manifest_path") or "")
    if not out_path or not cases_dir:
        run_paths = prepare_run_directory(self._resolve_benchmark_output_base(), mode="live")
        run_dir = str(run_paths["run_dir"])
        cases_dir = str(run_paths["cases_dir"])
        out_path = str(run_paths["summary_csv"])
        index_path = str(run_paths["cases_index_json"])
        manifest_path = str(run_paths["manifest_json"])

    try:
        Path(cases_dir).mkdir(parents=True, exist_ok=True)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        base_no_ext = str(Path(out_path).with_suffix(""))

        rows = []
        case_json_paths = list(st.get("case_json_paths") or [])
        if not case_json_paths:
            for name in sorted(os.listdir(cases_dir)):
                if name.lower().endswith(".json") and "_samples_full" not in name and "_yolo_last_" not in name:
                    case_json_paths.append(os.path.join(cases_dir, name))
        for p in case_json_paths:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                if isinstance(obj, dict):
                    if isinstance(obj.get("aggregates"), dict):
                        rows.append(obj["aggregates"])
                    elif isinstance(obj.get("result"), dict):
                        rows.append(obj["result"])
            except Exception:
                continue
        if not rows:
            rows = list(self._auto_bench_results)

        write_summary_csv(out_path, rows)

        cases_index = []
        for p in case_json_paths:
            try:
                obj = json.loads(Path(p).read_text(encoding="utf-8"))
                if not isinstance(obj, dict):
                    continue
                cases_index.append(
                    {
                        "case_id": obj.get("case_id"),
                        "configuration": obj.get("configuration", {}),
                        "case_json_path": os.path.relpath(p, run_dir or cases_dir),
                        "artifacts": obj.get("artifacts", {}),
                    }
                )
            except Exception:
                continue
        if not index_path:
            index_path = str(Path(base_no_ext + "_cases_index.json"))
        write_json(
            index_path,
            {
                "schema": "benchmark_cases_index_v1",
                "benchmark_mode": "live",
                "generated_at": utc_now_iso(),
                "summary_csv": os.path.relpath(out_path, run_dir or Path(out_path).parent),
                "cases_dir": os.path.relpath(cases_dir, run_dir or cases_dir),
                "cases_count": len(cases_index),
                "cases": cases_index,
            },
        )

        if not manifest_path:
            manifest_path = str(Path(base_no_ext + "_manifest.json"))
        write_json(
            manifest_path,
            {
                "schema": "benchmark_manifest_v1",
                "benchmark_mode": "live",
                "started_at": st.get("started_at_iso"),
                "ended_at": utc_now_iso(),
                "git_commit": detect_git_commit(Path(__file__).resolve().parent),
                "runtime": collect_runtime_env(),
                "parameter_grid": st.get("parameter_grid", {}),
                "cases_count": len(rows),
                "outputs": {
                    "summary_csv": os.path.relpath(out_path, run_dir or Path(out_path).parent),
                    "cases_index_json": os.path.relpath(index_path, run_dir or Path(index_path).parent),
                    "benchmark_manifest_json": os.path.relpath(manifest_path, run_dir or Path(manifest_path).parent),
                    "cases_dir": os.path.relpath(cases_dir, run_dir or cases_dir),
                },
            },
        )

        plots_dir = base_no_ext + "_plots"
        os.makedirs(plots_dir, exist_ok=True)

        if not rows:
            self.ab_status_var.set(f"Auto benchmark: zakończony (0 case), zapis: {out_path}")
            self._set_status("Auto benchmark zakończony (brak danych)", "warning")
            if self._auto_bench_state:
                self._auto_bench_state["running"] = False
            self._auto_bench_state = None
            return

        def _to_float(row: dict, key: str, default: float = 0.0) -> float:
            try:
                return float(row.get(key, default) or default)
            except Exception:
                return float(default)

        def _build_matrix(cur_rows: list[dict], value_key: str):
            imgsz_vals = sorted({int(_to_float(r, "imgsz", 0)) for r in cur_rows if int(_to_float(r, "imgsz", 0)) > 0})
            batch_vals = sorted({int(_to_float(r, "batch_images", 0)) for r in cur_rows if int(_to_float(r, "batch_images", 0)) > 0})
            if not imgsz_vals or not batch_vals:
                return None, None, None
            mat = np.full((len(imgsz_vals), len(batch_vals)), np.nan, dtype=float)
            for i, imgsz in enumerate(imgsz_vals):
                for j, bsz in enumerate(batch_vals):
                    vals = [
                        _to_float(r, value_key, np.nan)
                        for r in cur_rows
                        if int(_to_float(r, "imgsz", 0)) == imgsz and int(_to_float(r, "batch_images", 0)) == bsz
                    ]
                    vals = [v for v in vals if np.isfinite(v)]
                    if vals:
                        mat[i, j] = float(np.mean(vals))
            return mat, imgsz_vals, batch_vals

        def _save_heatmap(cur_rows: list[dict], value_key: str, title: str, file_name: str):
            mat, imgsz_vals, batch_vals = _build_matrix(cur_rows, value_key)
            if mat is None:
                return
            fig = Figure(figsize=(7.2, 4.8), dpi=120)
            ax = fig.add_subplot(111)
            cmap = "viridis" if "sec" in value_key else "plasma"
            im = ax.imshow(mat, cmap=cmap, aspect="auto")
            ax.set_title(title)
            ax.set_xlabel("batch_images")
            ax.set_ylabel("imgsz")
            ax.set_xticks(range(len(batch_vals)))
            ax.set_xticklabels([str(v) for v in batch_vals])
            ax.set_yticks(range(len(imgsz_vals)))
            ax.set_yticklabels([str(v) for v in imgsz_vals])
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    if np.isfinite(mat[i, j]):
                        ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", color="#FFFFFF", fontsize=7)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(os.path.join(plots_dir, file_name))

        if rows:
            _save_heatmap(rows, "images_per_sec", "images/s vs imgsz x batch", "heatmap_images_per_sec.png")
            _save_heatmap(rows, "avg_infer_ms", "avg infer ms vs imgsz x batch", "heatmap_avg_infer_ms.png")

            # Pareto-style scatter: throughput vs latency
            fig = Figure(figsize=(7.4, 5.0), dpi=120)
            ax = fig.add_subplot(111)
            mode_color = {
                "none": "#22c55e",
                "bin": "#3b82f6",
                "buffer": "#f59e0b",
                "bin+buffer": "#ef4444",
            }
            for r in rows:
                x = _to_float(r, "avg_infer_ms", 0.0)
                y = _to_float(r, "images_per_sec", 0.0)
                c = mode_color.get(str(r.get("mode", "none")), "#9ca3af")
                size = 42
                ax.scatter(x, y, c=c, s=size, alpha=0.75, edgecolors="none")
            ax.set_title("Pareto: throughput vs latency")
            ax.set_xlabel("avg_infer_ms (lower is better)")
            ax.set_ylabel("images_per_sec (higher is better)")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(os.path.join(plots_dir, "pareto_throughput_latency.png"))

            # Correlation matrix for numeric features
            corr_keys = [
                "imgsz", "batch_images", "camera_fps", "camera_period_us",
                "avg_infer_ms", "infer_ms_p95", "e2e_pipeline_ms", "e2e_pipeline_ms_p95",
                "images_per_sec", "steps_per_sec",
                "queue_wait_ms", "queue_wait_ms_p95", "forward_ms",
                "pending_buckets_avg", "bucket_full_ratio",
                "cpu_percent_avg", "ram_percent_avg",
                "gpu_util_percent_avg", "gpu_mem_percent_avg", "gpu_temp_c_avg", "gpu_power_w_avg",
                "detections_per_sec",
            ]
            usable_keys = [k for k in corr_keys if any(k in r for r in rows)]
            data = []
            for k in usable_keys:
                data.append([_to_float(r, k, 0.0) for r in rows])
            if len(data) >= 2 and len(rows) >= 2:
                arr = np.asarray(data, dtype=float)
                variances = np.var(arr, axis=1)
                keep_idx = [
                    i for i, v in enumerate(variances)
                    if np.isfinite(v) and float(v) > 1e-12
                ]
                if len(keep_idx) >= 2:
                    arr = arr[keep_idx, :]
                    corr_keys_used = [usable_keys[i] for i in keep_idx]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        corr = np.corrcoef(arr)
                    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

                    fig = Figure(figsize=(8.2, 6.0), dpi=120)
                    ax = fig.add_subplot(111)
                    im = ax.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
                    ax.set_title("Correlation matrix")
                    ax.set_xticks(range(len(corr_keys_used)))
                    ax.set_yticks(range(len(corr_keys_used)))
                    ax.set_xticklabels(corr_keys_used, rotation=45, ha="right", fontsize=8)
                    ax.set_yticklabels(corr_keys_used, fontsize=8)
                    for i in range(corr.shape[0]):
                        for j in range(corr.shape[1]):
                            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=6, color="#111827")
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    fig.tight_layout()
                    fig.savefig(os.path.join(plots_dir, "correlation_matrix.png"))

            # Queue/drop/bucket state overview
            q_rows = rows[:]
            if q_rows:
                labels = [
                    f"{(r.get('model_name') or os.path.basename(str(r.get('model', 'm'))))[:16]}|i{int(_to_float(r, 'imgsz', 0))}|b{int(_to_float(r, 'batch_images', 0))}|f{int(_to_float(r, 'camera_fps', 0))}|{r.get('mode', 'none')}"
                    for r in q_rows
                ]
                x = np.arange(len(q_rows), dtype=float)
                drops_vals = [_to_float(r, "drops_total", 0.0) for r in q_rows]
                drops_inc_vals = [_to_float(r, "drops_incomplete_total", 0.0) for r in q_rows]
                bq_full_vals = [_to_float(r, "batch_q_full_events", 0.0) for r in q_rows]
                raw_q_avg_vals = [_to_float(r, "queue_raw_q_avg", 0.0) for r in q_rows]
                batch_q_avg_vals = [_to_float(r, "queue_batch_q_avg", 0.0) for r in q_rows]
                pending_avg_vals = [_to_float(r, "pending_buckets_avg", 0.0) for r in q_rows]
                b1 = [_to_float(r, "bucket_fill_1_avg", 0.0) for r in q_rows]
                b2 = [_to_float(r, "bucket_fill_2_avg", 0.0) for r in q_rows]
                b3 = [_to_float(r, "bucket_fill_3_avg", 0.0) for r in q_rows]
                b4 = [_to_float(r, "bucket_fill_4_avg", 0.0) for r in q_rows]

                fig = Figure(figsize=(12.0, 8.0), dpi=120)
                ax1 = fig.add_subplot(221)
                ax2 = fig.add_subplot(222)
                ax3 = fig.add_subplot(223)
                ax4 = fig.add_subplot(224)

                ax1.bar(x - 0.25, drops_vals, width=0.25, color="#ef4444", label="drops_total")
                ax1.bar(x, drops_inc_vals, width=0.25, color="#f97316", label="drops_incomplete")
                ax1.bar(x + 0.25, bq_full_vals, width=0.25, color="#f59e0b", label="batch_q_full_events")
                ax1.set_title("Drops and queue full events")
                ax1.set_xticks(x)
                ax1.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
                ax1.grid(True, alpha=0.25)
                ax1.legend(fontsize=7)

                ax2.plot(x, raw_q_avg_vals, color="#3b82f6", marker="o", linewidth=1.6, label="raw_q_avg")
                ax2.plot(x, batch_q_avg_vals, color="#10b981", marker="o", linewidth=1.6, label="batch_q_avg")
                ax2.plot(x, pending_avg_vals, color="#a855f7", marker="o", linewidth=1.6, label="pending_buckets_avg")
                ax2.set_title("Queue occupancy state")
                ax2.set_xticks(x)
                ax2.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
                ax2.grid(True, alpha=0.25)
                ax2.legend(fontsize=7)

                ax3.bar(x, b1, width=0.65, color="#93c5fd", label="bucket 1/4")
                ax3.bar(x, b2, width=0.65, bottom=b1, color="#60a5fa", label="bucket 2/4")
                b12 = [b1[i] + b2[i] for i in range(len(b1))]
                ax3.bar(x, b3, width=0.65, bottom=b12, color="#3b82f6", label="bucket 3/4")
                b123 = [b12[i] + b3[i] for i in range(len(b1))]
                ax3.bar(x, b4, width=0.65, bottom=b123, color="#1d4ed8", label="bucket 4/4 full")
                ax3.set_title("Bucket fill profile (avg pending buckets)")
                ax3.set_xticks(x)
                ax3.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
                ax3.grid(True, alpha=0.25)
                ax3.legend(fontsize=7)

                ratios = [_to_float(r, "bucket_full_ratio", 0.0) * 100.0 for r in q_rows]
                ax4.bar(x, ratios, color="#22c55e")
                ax4.set_ylim(0, 100)
                ax4.set_title("Full bucket ratio (4/4)")
                ax4.set_ylabel("%")
                ax4.set_xticks(x)
                ax4.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
                ax4.grid(True, alpha=0.25)

                fig.tight_layout()
                fig.savefig(os.path.join(plots_dir, "queue_drop_bucket_overview.png"))

            # Latency breakdown percentages
            if rows:
                fig = Figure(figsize=(10.5, 5.6), dpi=120)
                ax = fig.add_subplot(111)
                labels = [
                    f"{(r.get('model_name') or os.path.basename(str(r.get('model', 'm'))))[:14]}|i{int(_to_float(r, 'imgsz', 0))}|b{int(_to_float(r, 'batch_images', 0))}|f{int(_to_float(r, 'camera_fps', 0))}"
                    for r in rows
                ]
                x = np.arange(len(rows), dtype=float)
                qv = [_to_float(r, "queue_wait_percent", 0.0) for r in rows]
                pv = [_to_float(r, "prepare_percent", 0.0) for r in rows]
                fv = [_to_float(r, "forward_percent", 0.0) for r in rows]
                pov = [_to_float(r, "post_percent", 0.0) for r in rows]
                ax.bar(x, qv, width=0.7, color="#f59e0b", label="queue_wait %")
                ax.bar(x, pv, width=0.7, bottom=qv, color="#3b82f6", label="prepare %")
                qp = [qv[i] + pv[i] for i in range(len(qv))]
                ax.bar(x, fv, width=0.7, bottom=qp, color="#10b981", label="forward %")
                qpf = [qp[i] + fv[i] for i in range(len(qv))]
                ax.bar(x, pov, width=0.7, bottom=qpf, color="#a855f7", label="post %")
                ax.set_title("Latency breakdown (%) by case")
                ax.set_ylabel("% of e2e_pipeline_ms")
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
                ax.grid(True, alpha=0.25)
                ax.legend(fontsize=8)
                fig.tight_layout()
                fig.savefig(os.path.join(plots_dir, "latency_breakdown_percent.png"))

            # Trend slopes over time in each case
            if rows:
                fig = Figure(figsize=(10.0, 5.2), dpi=120)
                ax = fig.add_subplot(111)
                labels = [
                    f"{(r.get('model_name') or os.path.basename(str(r.get('model', 'm'))))[:12]}|i{int(_to_float(r, 'imgsz', 0))}|b{int(_to_float(r, 'batch_images', 0))}|f{int(_to_float(r, 'camera_fps', 0))}"
                    for r in rows
                ]
                x = np.arange(len(rows), dtype=float)
                ips_sl = [_to_float(r, "ips_slope_per_s", 0.0) for r in rows]
                infer_sl = [_to_float(r, "infer_slope_per_s", 0.0) for r in rows]
                e2e_sl = [_to_float(r, "e2e_slope_per_s", 0.0) for r in rows]
                ax.axhline(0.0, color="#6b7280", linewidth=1.0)
                ax.plot(x, ips_sl, marker="o", color="#22c55e", label="ips slope /s")
                ax.plot(x, infer_sl, marker="o", color="#ef4444", label="infer slope /s")
                ax.plot(x, e2e_sl, marker="o", color="#f59e0b", label="e2e slope /s")
                ax.set_title("Time trend slopes (degradation detector)")
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
                ax.grid(True, alpha=0.25)
                ax.legend(fontsize=8)
                fig.tight_layout()
                fig.savefig(os.path.join(plots_dir, "trend_slopes.png"))

            # System utilization by case
            if rows:
                fig = Figure(figsize=(11.0, 6.4), dpi=120)
                ax1 = fig.add_subplot(211)
                ax2 = fig.add_subplot(212)
                labels = [
                    f"{(r.get('model_name') or os.path.basename(str(r.get('model', 'm'))))[:10]}|i{int(_to_float(r, 'imgsz', 0))}|b{int(_to_float(r, 'batch_images', 0))}|f{int(_to_float(r, 'camera_fps', 0))}"
                    for r in rows
                ]
                x = np.arange(len(rows), dtype=float)
                cpu = [_to_float(r, "cpu_percent_avg", 0.0) for r in rows]
                ram = [_to_float(r, "ram_percent_avg", 0.0) for r in rows]
                gpu_u = [_to_float(r, "gpu_util_percent_avg", 0.0) for r in rows]
                gpu_m = [_to_float(r, "gpu_mem_percent_avg", 0.0) for r in rows]
                gpu_t = [_to_float(r, "gpu_temp_c_avg", 0.0) for r in rows]
                gpu_p = [_to_float(r, "gpu_power_w_avg", 0.0) for r in rows]

                ax1.plot(x, cpu, marker="o", color="#3b82f6", label="cpu %")
                ax1.plot(x, ram, marker="o", color="#10b981", label="ram %")
                ax1.plot(x, gpu_u, marker="o", color="#f59e0b", label="gpu util %")
                ax1.plot(x, gpu_m, marker="o", color="#a855f7", label="gpu mem %")
                ax1.set_title("System utilization")
                ax1.set_xticks(x)
                ax1.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
                ax1.grid(True, alpha=0.25)
                ax1.legend(fontsize=8)

                ax2.plot(x, gpu_t, marker="o", color="#ef4444", label="gpu temp C")
                ax2.plot(x, gpu_p, marker="o", color="#f97316", label="gpu power W")
                ax2.set_title("Thermal and power")
                ax2.set_xticks(x)
                ax2.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
                ax2.grid(True, alpha=0.25)
                ax2.legend(fontsize=8)

                fig.tight_layout()
                fig.savefig(os.path.join(plots_dir, "system_utilization.png"))

            report_md = base_no_ext + "_report.md"
            with open(report_md, "w", encoding="utf-8") as f:
                f.write("# Benchmark Live Report\n\n")
                f.write("- This report contains raw measurements and artifacts only.\n")
                f.write("- No ranking, no best-case selection, no scoring.\n\n")
                f.write(f"- Cases tested: {len(rows)}\n")
                f.write(f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"- Summary CSV: `{os.path.basename(out_path)}`\n")
                f.write(f"- Cases index JSON: `{os.path.basename(index_path)}`\n")
                f.write(f"- Manifest JSON: `{os.path.basename(manifest_path)}`\n")
                f.write(f"- Plots dir: `{os.path.basename(plots_dir)}`\n")

        self.ab_status_var.set(f"Auto benchmark: zakończony ({len(rows)} case), zapis: {out_path}")
        self._set_status("Auto benchmark zakończony", "success")
    except Exception as e:
        self.ab_status_var.set(f"Auto benchmark: błąd zapisu wyników: {e}")
        self._set_status("Auto benchmark: błąd zapisu wyników", "danger")

    if self._auto_bench_state:
        self._auto_bench_state["running"] = False
    self._auto_bench_state = None

def on_apply_camera_fps(self):
    raw = ""
    try:
        raw = self.ab_camera_fps_var.get().strip()
    except Exception:
        raw = ""
    try:
        fps = float(str(raw).replace(",", "."))
    except Exception:
        self._set_status("Podaj poprawny FPS (jedna liczba)", "warning")
        return
    if fps <= 0:
        self._set_status("FPS musi być > 0", "warning")
        return
    self.ab_camera_fps_var.set(f"{fps:.2f}")
    period_us = self._fps_to_period_us(fps)
    try:
        self.control_q.put(("camera_set_fps", float(fps)))
        self.ab_camera_period_var.set(f"{period_us:.1f} us")
        self._set_status(f"Kamera FPS={fps:.2f} (periodTime={period_us:.1f} us)", "accent")
        self._update_auto_bench_plan_summary()
    except Exception as e:
        print(f"[GUI] ⚠️ camera_set_fps failed: {e}")
        self._set_status("Nie udało się ustawić FPS kamery", "danger")

def on_buffer_seconds_change(self, value):
    seconds = max(1, int(round(float(value))))
    self.buffer_sec_value.configure(text=str(seconds))
    try:
        self.btn_save_buffer.configure(text=f"💾 Save buffer ({seconds}s)")
    except Exception:
        pass

    if getattr(self, "_initing", False):
        return

    try:
        self.control_q.put(("buffer_set_seconds", seconds))
        print(f"[GUI] Buffer seconds set to: {seconds}")
    except Exception as e:
        print(f"[GUI] ⚠️ Error setting buffer seconds: {e}")
