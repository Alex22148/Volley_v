import queue
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

import customtkinter as ctk
from storage.shared_memory_manager import get_shared_memory_manager
from ui.style_volleyhub import (
    ACCENT,
    ACCENT_HOVER,
    BG_LIGHT,
    FONT_FAMILY,
    TEXT_DIM,

)

# noinspection PyProtectedMember
from ui.dir_gui.benchmark import (
    _auto_benchmark_finish,
    _auto_benchmark_reset_queues,
    _auto_benchmark_tick,
    _build_int_range,
    _choose_auto_bench_model_file,
    _choose_auto_bench_models_dir,
    _estimate_bbox_result_age_ms,
    _fps_to_period_us,
    _get_shared_state_value,
    _normalize_auto_bench_backend_mode,
    _normalize_auto_bench_source_mode,
    _on_auto_bench_backend_mode_change,
    _on_auto_bench_batch_mode_change,
    _on_auto_bench_model_source_change,
    _on_auto_bench_write_mode_change,
    _parse_float_csv,
    _parse_int_csv,
    _parse_modes_csv,
    _resolve_benchmark_output_base,
    _safe_percentile,
    _safe_stats,
    _sample_system_metrics,
    _start_auto_benchmark_deprecated,
    _start_auto_benchmark_legacy,
    _trend_slope,
    _update_auto_bench_plan_summary,
    on_apply_camera_fps,
    on_buffer_seconds_change,
    start_auto_benchmark,
    stop_auto_benchmark,
)
from ui.dir_gui.layout_operator import (
    _build_auto_benchmark_advanced_ui,
    _build_main_ui_operator,
    _ensure_auto_benchmark_vars,
    _on_dashboard_resize_preview_first,
    _open_benchmark_window,
    _run_benchmark_preset,
)
from ui.dir_gui.preview_panel import (
    _apply_yolo_point_overlay,
    _render_to_label,
    display_frame,
    on_preview_double_click,
    toggle_preview,
    toggle_preview_color_order,
    update_frames,
)
from ui.dir_gui.recording import (
    _append_stats_record,
    _count_session_files,
    _recording_stats_tick,
    on_pause_record,
    on_save_buffer,
    on_start_record,
    on_stop_record,
    on_take_shot,
)
from ui.dir_gui.remote import (
    _handle_remote_started,
    _open_remote_url,
    _poll_remote_url,
    toggle_remote_server,
)
from ui.dir_gui.session import (
    _apply_init_from_config,
    _get_initial_for_role,
    _on_main_configure,
    _show_main_ui,
    _wait_for_system_ready,
    create_new_session,
    ensure_valid_session_dir,
    on_choose_raw,
)
from ui.dir_gui.status_sync import (
    _append_perf_point,
    _draw_perf_chart,
    _insert_stat_text,
    _push_latency_sample,
    _rec_indicator_tick,
    _refresh_dashboard_metrics,
    _refresh_perf_chart,
    _refresh_system_health_ui,
    _set_status,
    _sync_operator_state_ui,
    _update_efficiency_bars,
    _update_stats,
)
from ui.dir_gui.yolo_controls import (
    _apply_live_backend_ui,
    _apply_yolo_preprocess_backend_ui,
    _build_live_track_config,
    _detect_cuda_preprocess_support,
    _normalize_live_infer_backend,
    _normalize_preprocess_backend,
    _pick_and_load_yolo_model,
    _poll_live_track_events,
    _push_live_backend_settings_to_control_plane,
    _reset_live_compare_stats,
    _send_live_track_config_if_running,
    choose_live_trt_engine,
    choose_yolo_model,
    debug_yolo,
    load_yolo_model,
    on_yolo_batch_change,
    on_yolo_conf_change,
    on_yolo_size_change,
    save_live_track_stats_snapshot,
    toggle_live_infer_backend,
    toggle_live_track,
    toggle_yolo,
    toggle_yolo_preprocess_backend,
)

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# --- CTk helper: zawsze CTkImage + referencja ---

class CaptureGUI(ctk.CTk):
    if TYPE_CHECKING:
        _build_main_ui_operator: Any
        _detect_cuda_preprocess_support: Any
        _ensure_auto_benchmark_vars: Any
        _get_shared_state_value: Any
        _on_main_configure: Any
        _poll_remote_url: Any
        _set_status: Any
        _wait_for_system_ready: Any
        ensure_valid_session_dir: Any
        toggle_preview: Any

    def __init__(self, live_q, stats_q, control_q, ready_evt, roles, init_settings,
             raw_q=None, snapshot_q=None, recording_event=None, snapshot_event=None,
             zoom_q=None, shared_state=None, yolo_vis_q=None):

        # --- argumenty / stan ---
        self.live_q = live_q
        self.zoom_q = zoom_q or queue.Queue(maxsize=8)
        self.stats_q = stats_q
        self.control_q = control_q
        self.ready_evt = ready_evt
        self.raw_q = raw_q
        self.roles = list(roles or [])
        self.snapshot_q = snapshot_q
        self.recording_event = recording_event
        self.snapshot_event = snapshot_event
        self.smm = get_shared_memory_manager()
        self.camera_labels = {}
        self.shared_state = shared_state
        self.yolo_vis_q = yolo_vis_q
        self.trajectory_window = None
        self.sliders: dict[str, dict[str, Any]] = {}
        self.session_dir: str | Path | None = None
        self._zoom_windows: dict[str, Any] = {}
        self._last_frame_time: dict[str, float] = {r: 0.0 for r in self.roles}
        self._update_frames_running = False
        if self.shared_state is not None:
            try:
                self.shared_state["preview_active"] = False
            except Exception:
                pass


        super().__init__()
        self.initial_settings = init_settings or {}
        self.fullres_frames = {r: None for r in self.roles}
        self._closing = False
        self._after_ids = set()
        self._initing = True

        # --- bezpieczne after ---
        def _safe_after(ms: int, fn: Any, *args: Any, **kwargs: Any):
            if self._closing or not self.winfo_exists():
                return None
            job_id: str | None = None

            def _wrapper():
                try:
                    if job_id in self._after_ids:
                        self._after_ids.discard(job_id)
                except Exception:
                    pass
                if self._closing or not self.winfo_exists():
                    return
                try:
                    fn(*args, **kwargs)
                except Exception:
                    pass

            job_id = self.after(ms=ms, func=_wrapper)
            self._after_ids.add(job_id)
            return job_id

        def _cancel_all_after() -> None:
            for aid in list(self._after_ids):
                try:
                    self.after_cancel(aid)
                except Exception:
                    pass
            self._after_ids.clear()

        self._after = _safe_after
        self._cancel_all_after = _cancel_all_after

        # statystyki
        self._preview_stats = {r: {"frames": 0, "last_t": time.time()} for r in self.roles}
        self._save_stats_buffer = {}
        self.eff_bars = {}
        self.eff_labels = {}

        self.title("VolleyHub Camera Control Panel")

        # --- AUTO ROZMIAR I POZYCJA GłÓWNEGO OKNA ---
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()

        target_w = 460

        main_w = max(target_w, int(sw * 0.23))
        main_w = min(main_w, int(sw * 0.35))

        main_h = int(sh * 0.8)

        x = int(sw * 0.01)
        y = int(sh * 0.02)

        self.geometry(f"{main_w}x{main_h}+{x}+{y}")
        # minimalny sensowny rozmiar
        self.minsize(target_w, int(sh * 0.5))

        self.configure(fg_color=BG_LIGHT)
        # === Stany ===
        self.recording = False
        self.preview_on = False
        self.remote_on = False
        self.raw_dir = self.initial_settings.get("path2save")
        self.record_start_time = None
        self._stats_timer_running = False

        self.preview_frames = {}

        # === Ekran ł‚adowania ===
        self.overlay = ctk.CTkFrame(self, fg_color=BG_LIGHT)
        self.overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.remote_url_var = ctk.StringVar(value="Remote: OFF")

        self.load_label = ctk.CTkLabel(self.overlay, text="🔧 Uruchamianie systemu...",
                                       text_color=ACCENT, font=(FONT_FAMILY, 26, "bold"))
        self.load_label.pack(pady=(300, 10))

        self.progress = ctk.CTkProgressBar(self.overlay, width=420, progress_color=ACCENT)
        self.progress.pack(pady=12)
        self.progress.set(0.0)

        self.load_sub = ctk.CTkLabel(self.overlay, text="Inicjalizacja...",
                                     text_color=TEXT_DIM, font=(FONT_FAMILY, 13))
        self.load_sub.pack(pady=(10, 0))

        # === Gł‚ówna rama ===
        self.main_frame = ctk.CTkFrame(self, fg_color=BG_LIGHT)
        self.yolo_on = False
        self.yolo_model_path = ""
        self.preview_input_color_order = "rgb"
        self.yolo_preprocess_backend = "cpu"

        self._cuda_preprocess_supported = self._detect_cuda_preprocess_support()

        self._last_yolo_stats = None
        self._last_router_stats = None

        self._perf_history_len = 60
        self._perf_history = deque(maxlen=self._perf_history_len)
        self._perf_markers = deque(maxlen=40)
        self._perf_next_idx = 0
        self._last_perf_cfg = None
        self._perf_chart_size = (220, 88)
        self._latency_samples = deque(maxlen=80)

        self._auto_bench_state = None
        self._auto_bench_results = []

        self.live_track_on = False
        self.live_track_last_result = None
        self.live_track_last_stats = None

        self.live_infer_backend = str(
            self._get_shared_state_value("yolo_backend", "ultralytics") or "ultralytics"
        ).strip().lower()
        if self.live_infer_backend not in ("ultralytics", "tensorrt"):
            self.live_infer_backend = "ultralytics"

        self.live_trt_engine_path = str(
            self._get_shared_state_value("yolo_trt_engine_path", "") or ""
        ).strip()

        self._live_compare_acc = {
            "ultralytics": {"sum_ms": 0.0, "count": 0},
            "tensorrt": {"sum_ms": 0.0, "count": 0},
        }

        self.header_status_var = ctk.StringVar(value="Kamery: --/-- | CPU: -- | GPU: -- | Remote: OFF")
        self.header_session_var = ctk.StringVar(value="Folder sesji: --")
        self.header_disk_var = ctk.StringVar(value="Wolne miejsce: --")
        self.header_hint_var = ctk.StringVar(value="Gotowy do startu sesji")

        self.remote_url_var = ctk.StringVar(value="Remote: OFF")

        self.session_elapsed_var = ctk.StringVar(value="Czas sesji: 00:00:00")
        self.buffer_fill_var = ctk.StringVar(value="Bufor: -- / -- s")
        self.left_disk_var = ctk.StringVar(value="Wolne miejsce: --")
        self.session_summary_var = ctk.StringVar(value="Podsumowanie sesji: brak danych")

        self.system_health_var = ctk.StringVar(value="⏳ Oczekiwanie na pomiary...")
        self.system_latency_var = ctk.StringVar(value="Opóźnienie: -- ms | trend: --")

        self.live_track_status_var = ctk.StringVar(value="LIVE_TRACK: zatrzymany")
        self.live_track_metrics_var = ctk.StringVar(value="latencja: -- ms | det: -- | track: -- | drop: --")
        self.live_backend_var = ctk.StringVar(value="LIVE backend: ultralytics")
        self.live_compare_var = ctk.StringVar(value="Porównanie infer: U=-- ms | TRT=-- ms")

        self.yolo_stats_var = ctk.StringVar(value="Śr. inferencja: -- ms | FPS: --")
        self.yolo_extra_stats_var = ctk.StringVar(value="Ostatnia inferencja: -- ms | batch: --")

        self.ab_status_var = ctk.StringVar(value="Auto benchmark: gotowy")
        self.ab_plan_var = ctk.StringVar(value="Plan: -")
        self._ensure_auto_benchmark_vars()
        self._build_main_ui()

        print("[GUI] ... GUI zainicjalizowane czekam na ready_evt z backendu")
        self._after(50, self._wait_for_system_ready)
        self._after(1000, self._poll_remote_url)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # aktualizacja geometrii do shared_state (dla dokowania preview)
        try:
            self.bind("<Configure>", self._on_main_configure)
        except Exception:
            pass

        # 🔚 Koniec inicjalizacji (od teraz eventy z suwaków mogą lecieć do backendu)
        self._initing = False

    def _build_main_ui(self):
        return self._build_main_ui_operator()




    @staticmethod
    def _clamp(v, lo, hi):
        try:
            v = float(str(v).replace(",", "."))
        except Exception:
            return None
        return max(lo, min(hi, v))

    def _quick_start_session(self):
        session_dir = self.ensure_valid_session_dir()
        if not session_dir:
            self._set_status("Nie udało się przygotować sesji", "danger")
            return
        if not self.preview_on:
            self.toggle_preview()
        if hasattr(self, "header_hint_var"):
            self.header_hint_var.set("Sesja aktywna")
        self._set_status("🚀 Session ready", "success")


    def _open_camera_assignment(self):
        try:
            from ui.assign_roles_gui import ask_camera_roles_gui
            choice = ask_camera_roles_gui()
            self._set_status(f"Camera assignment: {choice}", "accent")
        except Exception as e:
            self._set_status(f"Camera assignment error: {e}", "danger")


    # -------------------------------------------------------- --------------------------------------------------------
    @staticmethod
    def _make_btn(parent, text, cmd, color, height=30, font_size=11, corner_radius=8):
        """Fabryka przycisków akcji – kompaktowa wersja."""
        btn = ctk.CTkButton(
            parent,
            text=text,
            fg_color=color,
            hover_color=ACCENT_HOVER,
            text_color="#FFF",
            corner_radius=corner_radius,
            height=height,
            font=(FONT_FAMILY, font_size, "bold"),
            command=cmd,
        )
        return btn

    def on_generate_report(self):

        import subprocess, sys, os, webbrowser

        session_dir_raw = self.session_dir
        if session_dir_raw is None:
            print("[REPORT] Brak poprawnego folderu sesji:", session_dir_raw)
            self._set_status("Brak sesji najpierw coś nagraj.", "warning")
            return

        if isinstance(session_dir_raw, Path):
            session_dir = session_dir_raw
        elif isinstance(session_dir_raw, str):
            session_dir = Path(session_dir_raw)
        else:
            print("[REPORT] Brak poprawnego folderu sesji:", session_dir_raw)
            self._set_status("Brak sesji – najpierw coś nagraj.", "warning")
            return

        if not os.path.isdir(session_dir):
            print("[REPORT] Brak poprawnego folderu sesji:", session_dir_raw)
            self._set_status("Brak sesji – najpierw coś nagraj.", "warning")
            return
        stats_path = session_dir / "recording_stats.json"
        if not stats_path.exists():
            print("[REPORT] Nie ma recording_stats.json:", stats_path)
            self._set_status("Brak recording_stats.json dla tej sesji.", "warning")
            return

        # zakł‚adamy, łĽe recording_stats.py lełĽy obok gui.py / main.py
        script_path = Path(__file__).with_name("recording_stats.py")
        if not script_path.exists():
            print("[REPORT] Nie znaleziono recording_stats.py:", script_path)
            self._set_status("Brak pliku recording_stats.py obok aplikacji.", "danger")
            return

        print(f"[REPORT] Uruchamiam: {script_path} {stats_path}")
        try:
            subprocess.run(
                [sys.executable, str(script_path), str(stats_path)],
                check=False,
            )
        except Exception as e:
            print("[REPORT] Błąd uruchamiania recording_stats.py:", e)
            self._set_status("Błąd generowania raportu (zobacz konsolę).", "danger")
            return

        out_png = stats_path.with_name("recording_report.png")
        if out_png.exists():
            try:
                if os.name == "nt":
                    os.startfile(out_png)  # Windows
                else:
                    webbrowser.open(out_png.as_uri())
            except Exception as e:
                print("[REPORT] Nie udało się otworzyć PNG:", e)

            self._set_status(" Raport stabilności wygenerowany.", "success")
        else:
            print("[REPORT] Nie znaleziono:", out_png)
            self._set_status("Raport nie został‚ wygenerowany (brak PNG).", "warning")

    def on_expo_change(self, value, role):
        # 🔒 nie wysył‚aj komend do backendu podczas inicjalizacji widgetów
        if getattr(self, "_initing", False):
            s = self.sliders.get(role, {})
            if s and "exp_val" in s:
                s["exp_val"].configure(text=f"{float(value):.0f} µs")
            if s and "exp_entry" in s:
                s["exp_entry"].delete(0, "end"); s["exp_entry"].insert(0, f"{int(float(value))}")
            return
        try:
            self.control_q.put(("exposure", (role, float(value))))
            s = self.sliders.get(role, {})
            if s and "exp_val" in s:
                s["exp_val"].configure(text=f"{float(value):.0f} µs")
            if s and "exp_entry" in s:
                s["exp_entry"].delete(0, "end"); s["exp_entry"].insert(0, f"{int(float(value))}")
            self._set_status(f" {role}: {float(value):.0f} µs", "accent")
        except Exception as e:
            print(f"[GUI][ERROR] on_expo_change({role}) → {e}")

    def on_gain_change(self, value, role):
        if getattr(self, "_initing", False):
            s = self.sliders.get(role, {})
            if s and "gain_val" in s:
                s["gain_val"].configure(text=f"{float(value):.1f} dB")
            if s and "gain_entry" in s:
                s["gain_entry"].delete(0, "end"); s["gain_entry"].insert(0, f"{float(value):.1f}")
            return
        try:
            self.control_q.put(("gain", (role, float(value))))
            s = self.sliders.get(role, {})
            if s and "gain_val" in s:
                s["gain_val"].configure(text=f"{float(value):.1f} dB")
            if s and "gain_entry" in s:
                s["gain_entry"].delete(0, "end"); s["gain_entry"].insert(0, f"{float(value):.1f}")
            self._set_status(f" {role}: {float(value):.1f} dB", "accent")
        except Exception as e:
            print(f"[GUI][ERROR] on_gain_change({role}) → {e}")

    # --------------------------------------------------------
    # 🖥️ Preview i pętle GUI
    # --------------------------------------------------------

    def on_close_all(self):
        print("[GUI]  Zamykam system...")
        self._closing = True
        try:
            self._cancel_all_after()
        except Exception:
            pass
        try:
            self.control_q.put(("stop", True))
            self.control_q.put(("remote_stream", {"enabled": False}))
        except Exception:
            pass
        try:
            from core.utils_config import stop_evt
            stop_evt.set()
            print("[GUI]  stop_evt.set() zatrzymuję backend")
        except Exception as e:
            print(f"[GUI][] stop_evt.set() error: {e}")
        try:
            from streaming.server_stream import stop_webrtc_server
            stop_webrtc_server()
            print("[GUI]  stop_webrtc_server() ")
        except Exception as e:
            print(f"[GUI][] stop_webrtc_server() error: {e}")
        try:
            if hasattr(self, "_zoom_windows"):
                for r, win in list(self._zoom_windows.items()):
                    try:
                        if win.winfo_exists():
                            win.destroy()
                    except Exception:
                        pass
                self._zoom_windows.clear()
        except Exception:
            pass
        try:
            if self.winfo_exists():
                self.destroy()
        except Exception:
            pass
        try:
            from streaming.server_stream import kill_child_processes
            kill_child_processes()
            print("[GUI]  Wszystkie procesy potomne zakończone.")
        except Exception as e:
            print(f"[GUI][] cleanup error: {e}")

    def on_close(self):
        print("[GUI]  Zamykam aplikację...")
        self._closing = True
        try:
            self._cancel_all_after()
        except Exception:
            pass
        try:
            self.control_q.put(("remote_stream", {"enabled": False}))
            self.control_q.put(("stop", True))
        except Exception:
            pass
        try:
            from core.utils_config import stop_evt
            stop_evt.set()
        except Exception as e:
            print(f"[GUI]  stop_evt.set() error: {e}")
        try:
            from streaming.server_stream import stop_webrtc_server
            stop_webrtc_server()
        except Exception as e:
            print(f"[GUI]  stop_webrtc_server() error: {e}")
        try:
            if self.winfo_exists():
                self.withdraw()
                def _destroy_if_exists():
                    if self.winfo_exists():
                        self.destroy()
                self.after(ms=50, func=_destroy_if_exists)
        except Exception:
            pass

    # --------------------------------------------------------
    # Render i statystyki
    # --------------------------------------------------------

    def _ensure_savers_started(self):
        import threading
        import core.utils_config as utils_config
        from capture.grabber_module import saver_worker_bin
        from core.utils_config import BIN_BATCH_FRAMES, BIN_ROLL_EVERY

        runtime = getattr(utils_config, "_RUNTIME_PROCS", {})
        saver_threads = runtime.get("saver_threads")
        if saver_threads is None:
            saver_threads = {}
            runtime["saver_threads"] = saver_threads
            utils_config._RUNTIME_PROCS = runtime  # zapisz z powrotem

        per_role_q = runtime.get("per_role_q") or {}

        def _root_dir():
            return getattr(utils_config, "RAW_DIR", None)

        # uruchom po jednym saverze na rolę, jeśli jeszcze nie dział‚a
        for role in self.roles:
            t = saver_threads.get(role)
            if isinstance(t, threading.Thread) and t.is_alive():
                continue
            in_q = per_role_q.get(role)
            if in_q is None:
                continue  # backend jeszcze nie gotowy
            th = threading.Thread(
                target=saver_worker_bin,
                args=(role, in_q, _root_dir, BIN_BATCH_FRAMES, BIN_ROLL_EVERY,
                      self.recording_event, utils_config.stop_evt, self.stats_q),
                daemon=True,
                name=f"saver_{role}",
            )
            th.start()
            saver_threads[role] = th



CaptureGUI._sync_operator_state_ui = _sync_operator_state_ui
CaptureGUI._refresh_dashboard_metrics = _refresh_dashboard_metrics
CaptureGUI._push_latency_sample = _push_latency_sample
CaptureGUI._refresh_system_health_ui = _refresh_system_health_ui
CaptureGUI._update_efficiency_bars = _update_efficiency_bars
CaptureGUI._append_perf_point = _append_perf_point
CaptureGUI._draw_perf_chart = _draw_perf_chart
CaptureGUI._refresh_perf_chart = _refresh_perf_chart
CaptureGUI._update_stats = _update_stats
CaptureGUI._rec_indicator_tick = _rec_indicator_tick
CaptureGUI._insert_stat_text = _insert_stat_text
CaptureGUI._set_status = _set_status

#remote
CaptureGUI._poll_remote_url = _poll_remote_url
CaptureGUI.toggle_remote_server = toggle_remote_server
CaptureGUI._handle_remote_started = _handle_remote_started
CaptureGUI._open_remote_url = _open_remote_url

#recording
CaptureGUI.on_start_record = on_start_record
CaptureGUI.on_pause_record = on_pause_record
CaptureGUI.on_stop_record = on_stop_record
CaptureGUI.on_take_shot = on_take_shot
CaptureGUI.on_save_buffer = on_save_buffer
CaptureGUI._recording_stats_tick = _recording_stats_tick
CaptureGUI._append_stats_record = _append_stats_record
CaptureGUI._count_session_files = _count_session_files

#layout_operator
CaptureGUI._ensure_auto_benchmark_vars = _ensure_auto_benchmark_vars
CaptureGUI._build_main_ui_operator = _build_main_ui_operator
CaptureGUI._on_dashboard_resize_preview_first = _on_dashboard_resize_preview_first
CaptureGUI._build_auto_benchmark_advanced_ui = _build_auto_benchmark_advanced_ui
CaptureGUI._run_benchmark_preset = _run_benchmark_preset
CaptureGUI._open_benchmark_window = _open_benchmark_window

CaptureGUI._on_main_configure = _on_main_configure
CaptureGUI._get_initial_for_role = _get_initial_for_role
CaptureGUI._wait_for_system_ready = _wait_for_system_ready
CaptureGUI._show_main_ui = _show_main_ui
CaptureGUI._apply_init_from_config = _apply_init_from_config
CaptureGUI.ensure_valid_session_dir = ensure_valid_session_dir
CaptureGUI.create_new_session = create_new_session
CaptureGUI.on_choose_raw = on_choose_raw

#prev panel
CaptureGUI.toggle_preview = toggle_preview
CaptureGUI.update_frames = update_frames
CaptureGUI.display_frame = display_frame
CaptureGUI.toggle_preview_color_order = toggle_preview_color_order
CaptureGUI.on_preview_double_click = on_preview_double_click
CaptureGUI._render_to_label = _render_to_label
CaptureGUI._apply_yolo_point_overlay = _apply_yolo_point_overlay

# trajectory
from ui.dir_gui.trajectory import open_trajectory_window
CaptureGUI.open_trajectory_window = open_trajectory_window

#yolo
# yolo_controls
CaptureGUI.toggle_yolo = toggle_yolo
CaptureGUI.choose_yolo_model = choose_yolo_model
CaptureGUI._pick_and_load_yolo_model = _pick_and_load_yolo_model
CaptureGUI.load_yolo_model = load_yolo_model
CaptureGUI.on_yolo_conf_change = on_yolo_conf_change
CaptureGUI.on_yolo_size_change = on_yolo_size_change
CaptureGUI.on_yolo_batch_change = on_yolo_batch_change
CaptureGUI.debug_yolo = debug_yolo
CaptureGUI._normalize_live_infer_backend = _normalize_live_infer_backend
CaptureGUI._reset_live_compare_stats = _reset_live_compare_stats
CaptureGUI.save_live_track_stats_snapshot = save_live_track_stats_snapshot
CaptureGUI._apply_live_backend_ui = _apply_live_backend_ui
CaptureGUI._push_live_backend_settings_to_control_plane = _push_live_backend_settings_to_control_plane
CaptureGUI.toggle_live_infer_backend = toggle_live_infer_backend
CaptureGUI.choose_live_trt_engine = choose_live_trt_engine
CaptureGUI._build_live_track_config = _build_live_track_config
CaptureGUI._send_live_track_config_if_running = _send_live_track_config_if_running
CaptureGUI.toggle_live_track = toggle_live_track
CaptureGUI._poll_live_track_events = _poll_live_track_events
CaptureGUI._detect_cuda_preprocess_support = _detect_cuda_preprocess_support
CaptureGUI._normalize_preprocess_backend = _normalize_preprocess_backend
CaptureGUI._apply_yolo_preprocess_backend_ui = _apply_yolo_preprocess_backend_ui
CaptureGUI.toggle_yolo_preprocess_backend = toggle_yolo_preprocess_backend

# benchmark
CaptureGUI._choose_auto_bench_models_dir = _choose_auto_bench_models_dir
CaptureGUI._choose_auto_bench_model_file = _choose_auto_bench_model_file
CaptureGUI._on_auto_bench_model_source_change = _on_auto_bench_model_source_change
CaptureGUI._on_auto_bench_backend_mode_change = _on_auto_bench_backend_mode_change
CaptureGUI._on_auto_bench_batch_mode_change = _on_auto_bench_batch_mode_change
CaptureGUI._on_auto_bench_write_mode_change = _on_auto_bench_write_mode_change
CaptureGUI._update_auto_bench_plan_summary = _update_auto_bench_plan_summary
CaptureGUI._sample_system_metrics = _sample_system_metrics
CaptureGUI._resolve_benchmark_output_base = _resolve_benchmark_output_base
CaptureGUI._get_shared_state_value = _get_shared_state_value
CaptureGUI._estimate_bbox_result_age_ms = _estimate_bbox_result_age_ms
CaptureGUI._start_auto_benchmark_legacy = _start_auto_benchmark_legacy
CaptureGUI._start_auto_benchmark_deprecated = _start_auto_benchmark_deprecated
CaptureGUI.start_auto_benchmark = start_auto_benchmark
CaptureGUI.stop_auto_benchmark = stop_auto_benchmark
CaptureGUI._auto_benchmark_reset_queues = _auto_benchmark_reset_queues
CaptureGUI._auto_benchmark_tick = _auto_benchmark_tick
CaptureGUI._auto_benchmark_finish = _auto_benchmark_finish
CaptureGUI.on_apply_camera_fps = on_apply_camera_fps
CaptureGUI.on_buffer_seconds_change = on_buffer_seconds_change
CaptureGUI._normalize_auto_bench_source_mode = staticmethod(_normalize_auto_bench_source_mode)
CaptureGUI._normalize_auto_bench_backend_mode = staticmethod(_normalize_auto_bench_backend_mode)
CaptureGUI._parse_int_csv = staticmethod(_parse_int_csv)
CaptureGUI._build_int_range = staticmethod(_build_int_range)
CaptureGUI._parse_float_csv = staticmethod(_parse_float_csv)
CaptureGUI._fps_to_period_us = staticmethod(_fps_to_period_us)
CaptureGUI._safe_percentile = staticmethod(_safe_percentile)
CaptureGUI._safe_stats = staticmethod(_safe_stats)
CaptureGUI._trend_slope = staticmethod(_trend_slope)
CaptureGUI._parse_modes_csv = staticmethod(_parse_modes_csv)
