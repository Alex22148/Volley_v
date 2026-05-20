import os, re, time, webbrowser, queue, json, csv
import customtkinter as ctk
import warnings
from tkinter import filedialog
import tkinter as tk
from storage.shared_memory_manager import get_shared_memory_manager
from PIL import Image
import numpy as np
from pathlib import Path
print('korzystam z dobrego gui')
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
from collections import deque
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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
try:
    from live_runtime.live_backend_controller import LiveBackendController
except Exception:
    LiveBackendController = None

# ✅ Jeden punkt wejścia do stylu CTk
init_ctk_theme()


start = 0
full = 0
fcL, fCr, fL, fR = 0,0,0,0

# --- CTk helper: zawsze CTkImage + referencja ---


def set_ctk_image(widget, image_like, size=None):
    """Ustawia obraz na CTkLabel/CTkButton gwarantując CTkImage + referencję."""
    if isinstance(image_like, ctk.CTkImage):
        cimg = image_like
    elif isinstance(image_like, Image.Image):
        cimg = ctk.CTkImage(light_image=image_like, size=size or image_like.size)
    elif isinstance(image_like, np.ndarray):
        pil = Image.fromarray(image_like)
        cimg = ctk.CTkImage(light_image=pil, size=size or pil.size)
    else:
        return
    widget.configure(image=cimg, text="")
    widget.image = cimg  # trzymamy referencję


def _fix_mojibake_text(value: str) -> str:
    if not isinstance(value, str):
        return value
    s = value
    try:
        candidate = s.encode("cp1250", errors="ignore").decode("utf-8", errors="ignore")
        bad = s.count("â") + s.count("Ä") + s.count("Ĺ") + s.count("đ")
        bad_c = candidate.count("â") + candidate.count("Ä") + candidate.count("Ĺ") + candidate.count("đ")
        if candidate and bad_c <= bad:
            s = candidate
    except Exception:
        pass

    replacements = {
        "...": "...",
        "—": "-",
        "–": "-",
        "µ": "µ",
        "": "",
        "đź": "",
        "???": "[OK]",
        "❌": "[ERR]",
        "⚠️": "[WARN]",
        "ℹ️": "[INFO]",
        "▶": "[START]",
        "⏹": "[STOP]",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s



class CaptureGUI(ctk.CTk):
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
        self._last_frame_time = {r: 0.0 for r in self.roles}
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
        self._initing = True  # 🔴 Blokuje wysyłkę komend podczas inicjalizacji
        self.yolo_on = False
        self.preview_input_color_order = "rgb"  # "bgr" albo "rgb"
        self.yolo_preprocess_backend = "cpu"
        self._cuda_preprocess_supported = self._detect_cuda_preprocess_support()
        if not self._cuda_preprocess_supported and self.shared_state is not None:
            try:
                self.shared_state["yolo_preprocess_backend"] = "cpu"
            except Exception:
                pass

        # --- bezpieczne after ---
        def _safe_after(ms, fn, *args, **kwargs):
            if self._closing or not self.winfo_exists():
                return None
            job_id = None

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

            job_id = self.after(ms, _wrapper)
            self._after_ids.add(job_id)
            return job_id

        def _cancel_all_after():
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
        self._last_yolo_stats = None
        self._last_router_stats = None
        self.yolo_stats_var = ctk.StringVar(value="Śr. inferencja: -- ms | FPS: --")
        self.yolo_extra_stats_var = ctk.StringVar(value="Ostatnia inferencja: -- ms | batch: --")
        self._perf_history_len = 60
        self._perf_history = deque(maxlen=self._perf_history_len)  # items: {"i": int, "ips": float, "ms": float}
        self._perf_markers = deque(maxlen=40)  # items: {"i": int, "label": str}
        self._perf_next_idx = 0
        self._last_perf_cfg = None
        self._perf_chart_size = (220, 88)
        self._latency_samples = deque(maxlen=80)
        self._auto_bench_state = None
        self._auto_bench_results = []
        self.eff_bars = {}
        self.eff_labels = {}
        self.live_track_on = False
        self.live_track_last_result = None
        self.live_track_last_stats = None
        self.live_track_status_var = ctk.StringVar(value="LIVE_TRACK: zatrzymany")
        self.live_track_metrics_var = ctk.StringVar(value="latencja: -- ms | det: -- | track: -- | drop: --")
        self.live_backend_var = ctk.StringVar(value="LIVE backend: ultralytics")
        self.live_compare_var = ctk.StringVar(value="Porównanie infer: U=-- ms | TRT=-- ms")
        self.live_infer_backend = str(self._get_shared_state_value("yolo_backend", "ultralytics") or "ultralytics").strip().lower()
        if self.live_infer_backend not in ("ultralytics", "tensorrt"):
            self.live_infer_backend = "ultralytics"
        self.live_trt_engine_path = str(self._get_shared_state_value("yolo_trt_engine_path", "") or "").strip()
        self._live_compare_acc = {
            "ultralytics": {"sum_ms": 0.0, "count": 0},
            "tensorrt": {"sum_ms": 0.0, "count": 0},
        }
        try:
            self.live_backend_ctrl = LiveBackendController(shared_state=self.shared_state) if LiveBackendController else None
        except Exception:
            self.live_backend_ctrl = None

        self.title("VolleyHub Camera Control Panel")

        # --- AUTO ROZMIAR I POZYCJA GŁÓWNEGO OKNA ---
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()

        # Szerszy, wygodniejszy panel roboczy
        target_w = 1380
        main_w = max(target_w, int(sw * 0.78))
        main_w = min(main_w, int(sw * 0.94))

        main_h = int(sh * 0.90)

        x = int((sw - main_w) * 0.03)
        y = int(sh * 0.03)

        self.geometry(f"{main_w}x{main_h}+{x}+{y}")
        self.minsize(1180, int(sh * 0.70))

        self.configure(fg_color=BG_LIGHT)


        # === Stany ===
        self.recording = False
        self.preview_on = False
        self.remote_on = False
        self.raw_dir = self.initial_settings.get("path2save")
        self.record_start_time = None
        self._stats_timer_running = False

        self.preview_frames = {}
        self.img_labels = {}
        self.sliders = {}

        # === Ekran ładowania ===
        self.overlay = ctk.CTkFrame(self, fg_color=BG_LIGHT)
        self.overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.remote_url_var = ctk.StringVar(value="Remote: OFF")

        self.load_label = ctk.CTkLabel(self.overlay, text="🔧 Uruchamianie systemu...",
                                       text_color=ACCENT, font=(FONT_FAMILY, 26, "bold"))
        self.load_label.pack(pady=(300, 10))

        self.progress = ctk.CTkProgressBar(self.overlay, width=420, progress_color=ACCENT)
        self.progress.pack(pady=12)
        self.progress.set(0.0)

        self.load_sub = ctk.CTkLabel(self.overlay, text="Inicjalizacja komponentów...",
                                     text_color=TEXT_DIM, font=(FONT_FAMILY, 13))
        self.load_sub.pack(pady=(10, 0))

        # === Główna rama ===
        self.main_frame = ctk.CTkFrame(self, fg_color=BG_LIGHT)
        self._build_main_ui()
        self._apply_init_from_config()
        try:
            if self.shared_state is not None and hasattr(self, "ab_camera_fps_var") and hasattr(self, "ab_camera_period_var"):
                fps0 = float(self.shared_state.get("camera_target_fps", 50.0))
                period0 = float(self.shared_state.get("camera_period_us", self._fps_to_period_us(fps0)))
                self.ab_camera_fps_var.set(f"{fps0:.2f}")
                self.ab_camera_period_var.set(f"{period0:.1f} us")
        except Exception:
            pass

        print("[GUI] ✅ GUI zainicjalizowane — czekam na ready_evt z backendu")
        self._after(50, self._wait_for_system_ready)
        self._after(1000, self._poll_remote_url)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # aktualizacja geometrii do shared_state (dla dokowania preview)
        try:
            self.bind("<Configure>", self._on_main_configure)
        except Exception:
            pass
        
        # Dodaj informację o modelu YOLO z YAML
        print("[GUI] 🏀 YOLO Model: 1 klasa - 'ball' (ID: 0)")
        
        # 🔚 Koniec inicjalizacji (od teraz eventy z suwaków mogą lecieć do backendu)
        self._initing = False

    # --------------------------------------------------------
    # 🧠 Pomocnicze: odczyt startowych wartości z JSON
    # --------------------------------------------------------
    def _on_main_configure(self, event=None):
        """
        Zapisuje pozycję i rozmiar głównego okna do shared_state,
        aby proces preview mógł się 'dokleić' obok.
        """
        try:
            shared_state = getattr(self, "shared_state", None)
            if shared_state is None:
                return
            geom = {
                "x": self.winfo_x(),
                "y": self.winfo_y(),
                "w": self.winfo_width(),
                "h": self.winfo_height(),
            }
            shared_state["main_window_geometry"] = geom
        except Exception as e:
            print(f"[GUI] ⚠️ geometry share error: {e}")


    def _get_initial_for_role(self, role: str):
        """
        Zwraca (exp, gain) dla danej roli.
        Priorytety:
          1) init_settings['per_role'][role]['exposure'/'gain']
          2) init_settings['exposure_val'] / ['gain_val']
          3) None (bez zmiany slajdera)
        """
        cfg = self.initial_settings or {}
        per_role = cfg.get("per_role", {}) or {}
        role_cfg = per_role.get(role, {}) or {}
        exp = role_cfg.get("exposure")
        gain = role_cfg.get("gain")
        if exp is None:
            exp = cfg.get("exposure_val")
        if gain is None:
            gain = cfg.get("gain_val")
        try:
            exp = float(exp) if exp is not None else None
        except Exception:
            exp = None
        try:
            gain = float(gain) if gain is not None else None
        except Exception:
            gain = None
        return exp, gain

    # --------------------------------------------------------
    # 🕓 Czekanie na backend (progres + etapy)
    # --------------------------------------------------------
    def _poll_remote_url(self):
        try:
            if not hasattr(self, "shared_state"):
                return

            url = self.shared_state.get("remote_url", None)

            if url:
                # serwer online
                if not getattr(self, "_remote_shown", False):
                    self._remote_shown = True
                    self.remote_url_var.set(f"{url}")
                    self.remote_status.configure(
                        text="Server ONLINE ✅", text_color="#2ECC71"
                    )
            else:
                # serwer offline – wyczyść status jeśli wcześniej był ONLINE
                if getattr(self, "_remote_shown", False):
                    self._remote_shown = False
                    self.remote_url_var.set("Remote: OFF")
                    self.remote_status.configure(
                        text="Server wyłączony", text_color=TEXT_DIM
                    )
        except Exception as e:
            print(f"[GUI][⚠️] URL poll error: {e}")

        try:
            self._sync_operator_state_ui()
        except Exception:
            pass

        self._after(1000, self._poll_remote_url)


    def _wait_for_system_ready(self):
        if self.ready_evt.is_set():
            self.progress.set(1.0)
            self.load_label.configure(text="✅ System gotowy")
            if not self.winfo_exists():
                return
            self._after(400, self._show_main_ui)
            return

        try:
            msg = self.stats_q.get_nowait()
            if isinstance(msg, dict) and msg.get("type") == "camera_info":
                # info o kamerach obsługiwane w głównym GUI
                role = msg.get("role", "?")
                model = msg.get("model", "?")
                fw = msg.get("fw", "?")
                serial_number = msg.get("serial_number", "?")
                #print(f"[CAM] {role}: {model} | FW {fw} | S/N: {serial_number}")


            elif isinstance(msg, tuple) or (isinstance(msg, str) and any(x in msg.lower() for x in ["frame", "ts=", "grab", "capture"])):
                if not hasattr(self, "_warmup_phase"):
                    self._warmup_phase = 0
                self._warmup_phase = (self._warmup_phase + 1) % 4
                dots = "." * self._warmup_phase
                self.load_sub.configure(text=f" Warm-up kamer{dots}")
                self._warmup_shown = True
            elif isinstance(msg, str):
                self.load_sub.configure(text=msg)
                if not hasattr(self, "_seen_stages"):
                    self._seen_stages = set()
                if msg not in self._seen_stages:
                    self._seen_stages.add(msg)
                    total_stages = max(len(self._seen_stages), 6)
                    p = min(len(self._seen_stages) / total_stages, 1.0)
                    self.progress.set(p)
        except queue.Empty:
            if getattr(self, "_warmup_shown", False):
                if not hasattr(self, "_warmup_phase"):
                    self._warmup_phase = 0
                self._warmup_phase = (self._warmup_phase + 1) % 4
                dots = "." * self._warmup_phase
                self.load_sub.configure(text=f"Warm-up kamer{dots}")

        if not self.winfo_exists():
            return
        self._after(200, self._wait_for_system_ready)

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
    # --------------------------------------------------------
    # 🖥️ Główne UI
    # --------------------------------------------------------
    def _show_main_ui(self):
        self.overlay.place_forget()
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self._after(200, self._rec_indicator_tick)
        self._after(1000, self._update_stats)
        self._after(100, self._update_efficiency_bars)
        self._after(120, self.update_frames)

    def _build_main_ui(self):
        return self._build_main_ui_operator()

    def _build_main_ui_legacy_x(self):
        """Główne okno = panel sterowania + statystyki, responsywne przy resize."""
        # Spójne odstępy layoutu
        card_padx = 10
        card_pady = (4, 8)
        inner_padx = 10
        row_pady = (4, 6)

        # main_frame rośnie razem z oknem
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # boczny panel sterowania
        side = ctk.CTkFrame(self.main_frame, fg_color=PANEL_BG, corner_radius=14)
        side.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # układ grid w panelu bocznym
        side.grid_rowconfigure(0, weight=1)  # sterowanie / scroll
        side.grid_rowconfigure(1, weight=0)  # statystyki
        side.grid_rowconfigure(2, weight=0)  # przycisk Zamknij
        side.grid_columnconfigure(0, weight=1)

        # delikatna ramka wokół panelu
        side.configure(border_width=1, border_color=BORDER_COLOR)

        # szerokość do zawijania ścieżek
        side.update_idletasks()
        self._side_width = side.winfo_width() or max(560, int(self.winfo_screenwidth() * 0.34))

        def _on_side_configure(ev):
            self._side_width = ev.width
            try:
                if hasattr(self, "path_label") and self.path_label is not None:
                    self.path_label.configure(wraplength=self._side_width - 80)
            except Exception:
                pass

        side.bind("<Configure>", _on_side_configure)

        # === GÓRNY PANEL (STEROWANIE) ===
        # === GÓRNY PANEL (STEROWANIE) ===
        top = ctk.CTkScrollableFrame(side, fg_color=PANEL_BG)
        top.grid(row=0, column=0, sticky="nsew")
        top.grid_columnconfigure(0, weight=1)
        top._scrollbar.configure(width=10)

        # 🔹 Tytuł
        ctk.CTkLabel(
            top,
            text="🎬 Sterowanie",
            text_color=ACCENT,
            font=(FONT_FAMILY, 18, "bold"),
        ).pack(pady=(8, 8))

        # 🔹 Folder zapisu
        path_frame = ctk.CTkFrame(top, fg_color=CARD_BG, corner_radius=12)
        path_frame.pack(fill="x", padx=card_padx, pady=card_pady)

        ctk.CTkLabel(
            path_frame,
            text="📂 Folder zapisu:",
            text_color=TEXT_MAIN,
            font=(FONT_FAMILY, 14, "bold"),
        ).pack(anchor="w", padx=10, pady=(6, 0))

        self.path_label = ctk.CTkLabel(
            path_frame,
            text=self.raw_dir if self.raw_dir else "– nie wybrano –",
            text_color=TEXT_DIM,
            font=("Consolas", 11),
            anchor="w",
            justify="left",
            wraplength=self._side_width - 80,
        )
        self.path_label.pack(fill="x", padx=inner_padx, pady=(2, 4))

        if isinstance(self.initial_settings, dict) and self.initial_settings.get("path2save"):
            self.raw_dir = self.initial_settings["path2save"]
            self.path_label.configure(text=self.raw_dir)

        self.btn_choose_folder = self._make_btn(
            path_frame,
            "📁 Zmień folder",
            self.on_choose_raw,
            ACCENT,
        )
        self.btn_choose_folder.pack(pady=(4, 8), padx=inner_padx, fill="x")

        # 🔹 Status nagrywania
        rec_status_frame = ctk.CTkFrame(top, fg_color=CARD_BG, corner_radius=12)
        rec_status_frame.pack(fill="x", padx=card_padx, pady=row_pady)

        self.rec_led = ctk.CTkLabel(
            rec_status_frame,
            text="●",
            text_color="#B0B0B0",
            font=(FONT_FAMILY, 20, "bold"),
        )
        self.rec_led.pack(side="left", padx=(8,5), pady=6)

        self.rec_text = ctk.CTkLabel(
            rec_status_frame,
            text="Nagrywanie wyłączone",
            text_color=TEXT_DIM,
            font=(FONT_FAMILY, 13, "bold"),
        )
        self.rec_text.pack(side="left", padx=(3,8), pady=6)

        # 🔹 Sterowanie podglądem
        preview_frame = ctk.CTkFrame(top, fg_color=CARD_BG, corner_radius=12)
        preview_frame.pack(fill="x", padx=card_padx, pady=row_pady)

        self.preview_status = ctk.CTkLabel(
            preview_frame,
            text="Wyłącz podgląd",
            text_color=TEXT_DIM,
            font=(FONT_FAMILY, 13, "bold"),
        )
        self.preview_status.pack(pady=(4, 0))

        self.preview_btn = ctk.CTkButton(
            preview_frame,
            text="▶ Włącz podgląd",
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color="#FFFFFF",
            corner_radius=10,
            command=self.toggle_preview,
        )
        self.preview_btn.pack(pady=(4, 8), padx=inner_padx, fill="x")

        # 🔹 Zdalny serwer
        remote_frame = ctk.CTkFrame(top, fg_color=CARD_BG, corner_radius=12)
        remote_frame.pack(fill="x", padx=card_padx, pady=row_pady)

        self.remote_status = ctk.CTkLabel(
            remote_frame,
            text="Server wyłączony",
            text_color=TEXT_DIM,
            font=(FONT_FAMILY, 13, "bold"),
        )
        self.remote_status.pack(pady=(4, 0))

        self.remote_url_label = ctk.CTkLabel(
            remote_frame,
            textvariable=self.remote_url_var,
            text_color="#0077CC",
            font=(FONT_FAMILY, 12, "bold"),
        )
        self.remote_url_label.pack(pady=(2, 0))

        self.remote_url_label.bind("<Button-1>", lambda e: self._open_remote_url())
        self.remote_url_label.bind("<Enter>", lambda e: self.remote_url_label.configure(cursor="hand2"))
        self.remote_url_label.bind("<Leave>", lambda e: self.remote_url_label.configure(cursor=""))

        self.remote_qr_label = ctk.CTkLabel(remote_frame, text="")
        self.remote_qr_label.pack(pady=(2,2))

        self.remote_btn = self._make_btn(
            remote_frame,
            "🛰 Uruchom zdalną kontrolę",
            self.toggle_remote_server,
            "#6A5ACD",
        )
        self.remote_btn.pack(pady=(4, 8), padx=inner_padx, fill="x")

        # 🎯 YOLO - Detekcja piłki
        yolo_frame = ctk.CTkFrame(top, fg_color=CARD_BG, corner_radius=12)
        yolo_frame.pack(fill="x", padx=card_padx, pady=card_pady)

        ctk.CTkLabel(
            yolo_frame,
            text="🎯 Detekcja piłki (YOLO)",
            text_color=TEXT_MAIN,
            font=(FONT_FAMILY, 14, "bold"),
        ).pack(anchor="w", padx=8, pady=(6,3))

        self.yolo_status = ctk.CTkLabel(
            yolo_frame,
            text="YOLO wyłączony",
            text_color=TEXT_DIM,
            font=(FONT_FAMILY, 13, "bold"),
        )
        self.yolo_status.pack(pady=(0, ))
        
        # YOLO performance info
        self.yolo_perf_label = ctk.CTkLabel(
            yolo_frame,
            text="Wydajność: -- ms/klatka (-- FPS)",
            text_color=TEXT_DIM,
            font=(FONT_FAMILY, 11),
        )
        self.yolo_perf_label.pack(pady=(0, 2))

        # Usunięto mini-wykres wydajności (czarne okienko) na życzenie UX.
        self.yolo_perf_canvas = None

        # Model selection
        model_frame = ctk.CTkFrame(yolo_frame, fg_color="transparent")
        model_frame.pack(fill="x", padx=inner_padx, pady=(0, 4))
        
        ctk.CTkLabel(
            model_frame,
            text="Model YOLO:",
            text_color=TEXT_DIM,
            font=(FONT_FAMILY, 12),
        ).pack(anchor="w")
        
        model_btn_frame = ctk.CTkFrame(model_frame, fg_color="transparent")
        model_btn_frame.pack(fill="x", pady=(4, 0))
        model_btn_frame.grid_columnconfigure(0, weight=1)
        model_btn_frame.grid_columnconfigure(1, weight=0)
        
        self.yolo_model_btn = ctk.CTkButton(
            model_btn_frame,
            text="📁 Wybierz model (.pt)",
            fg_color="#6A5ACD",
            hover_color="#5A4FCF",
            text_color="#FFFFFF",
            corner_radius=8,
            height=28,
            command=self.choose_yolo_model,
        )
        self.yolo_model_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        
        self.yolo_load_btn = ctk.CTkButton(
            model_btn_frame,
            text="⚡ Załaduj",
            fg_color="#FF6B35",
            hover_color="#E55A2B",
            text_color="#FFFFFF",
            corner_radius=8,
            height=28,
            width=110,
            command=self.load_yolo_model,
        )
        self.yolo_load_btn.grid(row=0, column=1, sticky="ew")
        
        self.yolo_model_path = ""
        self.yolo_model_label = ctk.CTkLabel(
            model_frame,
            text="Brak wybranego modelu",
            text_color=TEXT_DIM,
            font=("Consolas", 10),
            anchor="w",
        )
        self.yolo_model_label.pack(fill="x", pady=(4, 0))

        # Confidence threshold
        conf_frame = ctk.CTkFrame(yolo_frame, fg_color="transparent")
        conf_frame.pack(fill="x", padx=inner_padx, pady=(4, 6))
        
        conf_label_frame = ctk.CTkFrame(conf_frame, fg_color="transparent")
        conf_label_frame.pack(fill="x")
        
        ctk.CTkLabel(
            conf_label_frame,
            text="Próg pewności:",
            text_color=TEXT_DIM,
            font=(FONT_FAMILY, 12),
        ).pack(side="left")
        
        self.yolo_conf_value = ctk.CTkLabel(
            conf_label_frame,
            text="0.25",
            text_color=ACCENT,
            font=(FONT_FAMILY, 12, "bold"),
        )
        self.yolo_conf_value.pack(side="right")
        
        self.yolo_conf_slider = ctk.CTkSlider(
            conf_frame,
            from_=0,
            to=1,
            number_of_steps=18,  # 0.1 to 1.0 in 0.05 steps
            command=self.on_yolo_conf_change,
        )
        self.yolo_conf_slider.pack(fill="x", pady=(4, 0))
        self.yolo_conf_slider.set(0.25)

        # Image size for detection
        size_frame = ctk.CTkFrame(yolo_frame, fg_color="transparent")
        size_frame.pack(fill="x", padx=inner_padx, pady=(4, 6))
        
        size_label_frame = ctk.CTkFrame(size_frame, fg_color="transparent")
        size_label_frame.pack(fill="x")
        
        ctk.CTkLabel(
            size_label_frame,
            text="Rozmiar detekcji:",
            text_color=TEXT_DIM,
            font=(FONT_FAMILY, 12),
        ).pack(side="left")
        
        self.yolo_size_value = ctk.CTkLabel(
            size_label_frame,
            text="640px",
            text_color=ACCENT,
            font=(FONT_FAMILY, 12, "bold"),
        )
        self.yolo_size_value.pack(side="right")
        
        self.yolo_size_slider = ctk.CTkSlider(
            size_frame,
            from_=320,
            to=1280,
            number_of_steps=12,  # 320, 416, 512, 640, 736, 832, 1024, 1280
            command=self.on_yolo_size_change,
        )
        self.yolo_size_slider.pack(fill="x", pady=(4, 0))
        self.yolo_size_slider.set(640)

        # Batch size for sequence batching
        batch_frame = ctk.CTkFrame(yolo_frame, fg_color="transparent")
        batch_frame.pack(fill="x", padx=inner_padx, pady=(4, 8))

        batch_label_frame = ctk.CTkFrame(batch_frame, fg_color="transparent")
        batch_label_frame.pack(fill="x")

        ctk.CTkLabel(
            batch_label_frame,
            text="Batch inferencji:",
            text_color=TEXT_DIM,
            font=(FONT_FAMILY, 12),
        ).pack(side="left")

        self.yolo_batch_value = ctk.CTkLabel(
            batch_label_frame,
            text="4",
            text_color=ACCENT,
            font=(FONT_FAMILY, 12, "bold"),
        )

        self.yolo_batch_value.pack(side="right")

        self.yolo_batch_slider = ctk.CTkSlider(
            batch_frame,
            from_=4,
            to=32,
            number_of_steps=7,  # 4,8,12,16,20,24,28,32
            command=self.on_yolo_batch_change,
        )
        self.yolo_batch_slider.pack(fill="x", pady=(4, 0))
        self.yolo_batch_slider.set(4)
        self.yolo_batch_value.configure(text="4")

        preprocess_frame = ctk.CTkFrame(yolo_frame, fg_color="transparent")
        preprocess_frame.pack(fill="x", padx=inner_padx, pady=(2, 6))
        preprocess_frame.grid_columnconfigure(0, weight=1)
        preprocess_frame.grid_columnconfigure(1, weight=0)

        self.yolo_preprocess_label = ctk.CTkLabel(
            preprocess_frame,
            text="Preprocess YOLO: CPU",
            text_color=TEXT_DIM,
            font=(FONT_FAMILY, 12, "bold"),
            anchor="w",
        )
        self.yolo_preprocess_label.grid(row=0, column=0, sticky="w")

        self.yolo_preprocess_btn = ctk.CTkButton(
            preprocess_frame,
            text="Przełącz na GPU",
            fg_color="#6A5ACD",
            hover_color="#5A4FCF",
            text_color="#FFFFFF",
            corner_radius=8,
            height=28,
            width=145,
            command=self.toggle_yolo_preprocess_backend,
        )
        self.yolo_preprocess_btn.grid(row=0, column=1, sticky="e")
        self._apply_yolo_preprocess_backend_ui()

        # Debug YOLO button
        self.yolo_debug_btn = ctk.CTkButton(
            yolo_frame,
            text="🔍 Debug YOLO",
            fg_color="#6A5ACD",
            hover_color="#5A4FCF",
            text_color="#FFFFFF",
            corner_radius=8,
            height=28,
            command=self.debug_yolo,
        )

        self.yolo_traj_btn = ctk.CTkButton(
            yolo_frame,
            text="Trajektorie",
            fg_color="#4A5568",
            hover_color="#3A4558",
            text_color="#FFFFFF",
            corner_radius=8,
            height=28,
            command=self.open_trajectory_window,
        )
        try:
            self.yolo_traj_btn.configure(text="Trajektorie")
        except Exception:
            pass
        self.yolo_traj_btn.pack(pady=(4, 4), padx=10, fill="x")

        try:
            self.yolo_debug_btn.configure(text="Debug YOLO")
        except Exception:
            pass
        self.yolo_debug_btn.pack(pady=(0, 4), padx=10, fill="x")

        self.yolo_btn = self._make_btn(
            yolo_frame,
            "🎯 Włącz detekcję piłki",
            self.toggle_yolo,
            "#FF6B35",
        )
        try:
            self.yolo_btn.configure(text="Włącz/Wyłącz detekcję piłki")
        except Exception:
            pass
        self.yolo_btn.pack(pady=(2, 10), padx=10, fill="x")

        live_track_frame = ctk.CTkFrame(yolo_frame, fg_color=CARD_INNER_BG, corner_radius=8)
        live_track_frame.pack(fill="x", padx=inner_padx, pady=(0, 8))
        ctk.CTkLabel(
            live_track_frame,
            text="LIVE_TRACK (minimal latency)",
            text_color=TEXT_MAIN,
            font=(FONT_FAMILY, 12, "bold"),
        ).pack(anchor="w", padx=8, pady=(8, 4))

        self.live_backend_label = ctk.CTkLabel(
            live_track_frame,
            textvariable=self.live_backend_var,
            text_color=TEXT_DIM,
            font=(FONT_FAMILY, 10, "bold"),
            anchor="w",
        )
        self.live_backend_label.pack(fill="x", padx=8, pady=(0, 4))

        backend_row = ctk.CTkFrame(live_track_frame, fg_color="transparent")
        backend_row.pack(fill="x", padx=8, pady=(0, 4))
        self.live_backend_btn = ctk.CTkButton(
            backend_row,
            text="Backend: Ultralytics",
            fg_color="#334155",
            hover_color="#1e293b",
            text_color="#FFFFFF",
            corner_radius=8,
            height=26,
            command=self.toggle_live_infer_backend,
        )
        self.live_backend_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.live_engine_btn = ctk.CTkButton(
            backend_row,
            text="Engine .engine",
            fg_color="#0f766e",
            hover_color="#0d5f5a",
            text_color="#FFFFFF",
            corner_radius=8,
            height=26,
            width=120,
            command=self.choose_live_trt_engine,
        )
        self.live_engine_btn.pack(side="right")

        self.live_track_status_label = ctk.CTkLabel(
            live_track_frame,
            textvariable=self.live_track_status_var,
            text_color=TEXT_DIM,
            font=(FONT_FAMILY, 11),
            anchor="w",
        )
        self.live_track_status_label.pack(fill="x", padx=8, pady=(0, 2))

        self.live_track_metrics_label = ctk.CTkLabel(
            live_track_frame,
            textvariable=self.live_track_metrics_var,
            text_color=TEXT_DIM,
            font=(FONT_FAMILY, 10),
            anchor="w",
        )
        self.live_track_metrics_label.pack(fill="x", padx=8, pady=(0, 6))
        self.live_compare_label = ctk.CTkLabel(
            live_track_frame,
            textvariable=self.live_compare_var,
            text_color=TEXT_DIM,
            font=(FONT_FAMILY, 10),
            anchor="w",
        )
        self.live_compare_label.pack(fill="x", padx=8, pady=(0, 6))

        self.live_save_stats_btn = ctk.CTkButton(
            live_track_frame,
            text="Zapisz LIVE stats",
            fg_color="#475569",
            hover_color="#334155",
            text_color="#FFFFFF",
            corner_radius=8,
            height=26,
            command=self.save_live_track_stats_snapshot,
        )
        self.live_save_stats_btn.pack(fill="x", padx=8, pady=(0, 6))

        self.live_track_btn = ctk.CTkButton(
            live_track_frame,
            text="Start LIVE_TRACK",
            fg_color="#0ea5e9",
            hover_color="#0284c7",
            text_color="#FFFFFF",
            corner_radius=8,
            height=28,
            command=self.toggle_live_track,
        )
        self.live_track_btn.pack(fill="x", padx=8, pady=(0, 8))
        self._apply_live_backend_ui()

        auto_bench_frame = ctk.CTkFrame(yolo_frame, fg_color=CARD_INNER_BG, corner_radius=8)
        auto_bench_frame.pack(fill="x", padx=inner_padx, pady=(2, 10))
        ctk.CTkLabel(
            auto_bench_frame,
            text="Auto Benchmark (live)",
            text_color=TEXT_MAIN,
            font=(FONT_FAMILY, 12, "bold"),
        ).pack(anchor="w", padx=8, pady=(8, 4))

        row_source = ctk.CTkFrame(auto_bench_frame, fg_color="transparent")
        row_source.pack(fill="x", padx=8, pady=(6, 4))
        ctk.CTkLabel(row_source, text="Źródło modeli", text_color=TEXT_DIM, font=(FONT_FAMILY, 11)).pack(side="left")
        self.ab_model_source_var = ctk.StringVar(value="Wiele modeli (folder)")
        self.ab_model_source_menu = ctk.CTkOptionMenu(
            row_source,
            variable=self.ab_model_source_var,
            values=["Folder modeli", "Jeden model"],
            width=150,
            height=24,
            command=self._on_auto_bench_model_source_change,
        )
        self.ab_model_source_menu.pack(side="right")
        try:
            row_source.winfo_children()[0].configure(text="Zrodlo modeli")
        except Exception:
            pass
        ctk.CTkLabel(auto_bench_frame, text="MODELE", text_color=ACCENT, font=(FONT_FAMILY, 10, "bold")).pack(anchor="w", padx=8, pady=(8, 2))

        row_models = ctk.CTkFrame(auto_bench_frame, fg_color="transparent")
        row_models.pack(fill="x", padx=8, pady=(4, 4))
        ctk.CTkLabel(row_models, text="Folder modeli", text_color=TEXT_DIM, font=(FONT_FAMILY, 11)).pack(side="left")
        self.ab_models_dir_var = ctk.StringVar(value="")
        self.ab_models_dir_btn = None  # set below when button is created
        ctk.CTkButton(
            row_models,
            text="📁",
            width=34,
            height=24,
            command=self._choose_auto_bench_models_dir,
        ).pack(side="right")

        try:
            self.ab_models_dir_btn = row_models.winfo_children()[-1]
            self.ab_models_dir_btn.configure(text="Wybierz")
        except Exception:
            self.ab_models_dir_btn = None

        self.ab_models_dir_entry = ctk.CTkEntry(auto_bench_frame, textvariable=self.ab_models_dir_var, height=24)
        self.ab_models_dir_entry.pack(fill="x", padx=8, pady=(0, 8))

        row_lists = ctk.CTkFrame(auto_bench_frame, fg_color="transparent")
        row_lists.pack(fill="x", padx=8, pady=(0, 6))
        row_lists.grid_columnconfigure((0, 1), weight=1)
        self.ab_imgsz_var = ctk.StringVar(value="640,960")
        self.ab_batch_var = ctk.StringVar(value="4,8,12")
        ctk.CTkLabel(row_lists, text="Rozdzielczości", text_color=TEXT_DIM, font=(FONT_FAMILY, 10)).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(row_lists, text="Batch img", text_color=TEXT_DIM, font=(FONT_FAMILY, 10)).grid(row=0, column=1, sticky="w")
        self.ab_imgsz_combo = ctk.CTkComboBox(
            row_lists,
            variable=self.ab_imgsz_var,
            values=["320", "416", "512", "640", "736", "832", "960", "1024", "1280", "640,960", "640,832,1024"],
            height=24,
        )
        self.ab_imgsz_combo.grid(row=1, column=0, sticky="ew", padx=(0, 3))
        self.ab_batch_list_combo = ctk.CTkComboBox(
            row_lists,
            variable=self.ab_batch_var,
            values=["4", "8", "12", "16", "20", "24", "28", "32", "4,8,12", "4,8,12,16"],
            height=24,
        )
        self.ab_batch_list_combo.grid(row=1, column=1, sticky="ew", padx=(3, 0))

        row_batch_mode = ctk.CTkFrame(auto_bench_frame, fg_color="transparent")
        row_batch_mode.pack(fill="x", padx=8, pady=(0, 6))
        ctk.CTkLabel(row_batch_mode, text="Tryb batch", text_color=TEXT_DIM, font=(FONT_FAMILY, 10)).pack(side="left")
        self.ab_batch_mode_var = ctk.StringVar(value="Zakres")
        self.ab_batch_mode_menu = ctk.CTkOptionMenu(
            row_batch_mode,
            variable=self.ab_batch_mode_var,
            values=["Zakres", "Lista"],
            width=120,
            height=24,
            command=self._on_auto_bench_batch_mode_change,
        )
        self.ab_batch_mode_menu.pack(side="right")
        ctk.CTkLabel(auto_bench_frame, text="PARAMETRY TESTU", text_color=ACCENT, font=(FONT_FAMILY, 10, "bold")).pack(anchor="w", padx=8, pady=(10, 2))

        row_single_model = ctk.CTkFrame(auto_bench_frame, fg_color="transparent")
        row_single_model.pack(fill="x", padx=8, pady=(4, 4))
        self.ab_single_model_label = ctk.CTkLabel(row_single_model, text="Pojedynczy model", text_color=TEXT_DIM, font=(FONT_FAMILY, 10))
        self.ab_single_model_label.pack(side="left")
        self.ab_single_model_var = ctk.StringVar(value="")
        self.ab_single_model_btn = None
        ctk.CTkButton(
            row_single_model,
            text="📄",
            width=34,
            height=24,
            command=self._choose_auto_bench_model_file,
        ).pack(side="right")
        try:
            self.ab_single_model_btn = row_single_model.winfo_children()[-1]
            self.ab_single_model_btn.configure(text="Wybierz")
        except Exception:
            self.ab_single_model_btn = None

        self.ab_single_model_entry = ctk.CTkEntry(auto_bench_frame, textvariable=self.ab_single_model_var, height=24)
        self.ab_single_model_entry.pack(fill="x", padx=8, pady=(0, 8))
        self._on_auto_bench_model_source_change()

        row_batch_range = ctk.CTkFrame(auto_bench_frame, fg_color="transparent")
        row_batch_range.pack(fill="x", padx=8, pady=(0, 6))
        row_batch_range.grid_columnconfigure((0, 1, 2), weight=1)
        ctk.CTkLabel(row_batch_range, text="Batch od", text_color=TEXT_DIM, font=(FONT_FAMILY, 10)).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(row_batch_range, text="Batch do", text_color=TEXT_DIM, font=(FONT_FAMILY, 10)).grid(row=0, column=1, sticky="w")
        ctk.CTkLabel(row_batch_range, text="Step", text_color=TEXT_DIM, font=(FONT_FAMILY, 10)).grid(row=0, column=2, sticky="w")
        self.ab_batch_from_var = ctk.StringVar(value="4")
        self.ab_batch_to_var = ctk.StringVar(value="16")
        self.ab_batch_step_var = ctk.StringVar(value="4")
        batch_values = ["4", "8", "12", "16", "20", "24", "28", "32"]
        self.ab_batch_from_combo = ctk.CTkComboBox(row_batch_range, variable=self.ab_batch_from_var, values=batch_values, height=24)
        self.ab_batch_to_combo = ctk.CTkComboBox(row_batch_range, variable=self.ab_batch_to_var, values=batch_values, height=24)
        self.ab_batch_step_combo = ctk.CTkComboBox(row_batch_range, variable=self.ab_batch_step_var, values=["1", "2", "4", "8"], height=24)
        self.ab_batch_from_combo.grid(row=1, column=0, sticky="ew", padx=(0, 3))
        self.ab_batch_to_combo.grid(row=1, column=1, sticky="ew", padx=(3, 3))
        self.ab_batch_step_combo.grid(row=1, column=2, sticky="ew", padx=(3, 0))

        row_modes = ctk.CTkFrame(auto_bench_frame, fg_color="transparent")
        row_modes.pack(fill="x", padx=8, pady=(2, 6))
        row_modes.grid_columnconfigure((0, 1), weight=1)
        self.ab_modes_var = ctk.StringVar(value="Wszystkie kombinacje")
        self.ab_measure_s_var = ctk.StringVar(value="10")
        ctk.CTkLabel(row_modes, text="Tryb zapisu", text_color=TEXT_DIM, font=(FONT_FAMILY, 10)).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(row_modes, text="Okno pomiaru [s]", text_color=TEXT_DIM, font=(FONT_FAMILY, 10)).grid(row=0, column=1, sticky="w")
        ctk.CTkComboBox(
            row_modes,
            variable=self.ab_modes_var,
            values=["Wszystkie kombinacje", "Brak zapisu", "Tylko BIN", "Tylko BUFOR", "BIN + BUFOR"],
            height=24,
            command=self._on_auto_bench_write_mode_change,
        ).grid(row=1, column=0, sticky="ew", padx=(0, 3))
        ctk.CTkEntry(row_modes, textvariable=self.ab_measure_s_var, height=24).grid(row=1, column=1, sticky="ew", padx=(3, 0))

        row_times = ctk.CTkFrame(auto_bench_frame, fg_color="transparent")
        row_times.pack(fill="x", padx=8, pady=(2, 6))
        row_times.grid_columnconfigure((0, 1, 2), weight=1)
        self.ab_warmup_s_var = ctk.StringVar(value="4")
        self.ab_bin_s_var = ctk.StringVar(value="10")
        self.ab_buffer_count_var = ctk.StringVar(value="2")
        ctk.CTkLabel(row_times, text="Warmup [s]", text_color=TEXT_DIM, font=(FONT_FAMILY, 10)).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(row_times, text="BIN [s]", text_color=TEXT_DIM, font=(FONT_FAMILY, 10)).grid(row=0, column=1, sticky="w")
        ctk.CTkLabel(row_times, text="Ile buforów", text_color=TEXT_DIM, font=(FONT_FAMILY, 10)).grid(row=0, column=2, sticky="w")
        ctk.CTkEntry(row_times, textvariable=self.ab_warmup_s_var, height=24).grid(row=1, column=0, sticky="ew", padx=(0, 3))
        self.ab_bin_s_entry = ctk.CTkEntry(row_times, textvariable=self.ab_bin_s_var, height=24)
        self.ab_buffer_count_entry = ctk.CTkEntry(row_times, textvariable=self.ab_buffer_count_var, height=24)
        self.ab_bin_s_entry.grid(row=1, column=1, sticky="ew", padx=(3, 3))
        self.ab_buffer_count_entry.grid(row=1, column=2, sticky="ew", padx=(3, 0))

        row_times2 = ctk.CTkFrame(auto_bench_frame, fg_color="transparent")
        row_times2.pack(fill="x", padx=8, pady=(2, 6))
        row_times2.grid_columnconfigure((0, 1), weight=1)
        self.ab_buffer_pause_s_var = ctk.StringVar(value="3")
        self.ab_out_var = ctk.StringVar(value="")
        ctk.CTkLabel(row_times2, text="Przerwa bufor [s]", text_color=TEXT_DIM, font=(FONT_FAMILY, 10)).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(row_times2, text="Plik wynikowy .csv", text_color=TEXT_DIM, font=(FONT_FAMILY, 10)).grid(row=0, column=1, sticky="w")
        self.ab_buffer_pause_s_entry = ctk.CTkEntry(row_times2, textvariable=self.ab_buffer_pause_s_var, height=24)
        self.ab_buffer_pause_s_entry.grid(row=1, column=0, sticky="ew", padx=(0, 3))
        ctk.CTkEntry(row_times2, textvariable=self.ab_out_var, height=24).grid(row=1, column=1, sticky="ew", padx=(3, 0))

        row_cam_fps = ctk.CTkFrame(auto_bench_frame, fg_color="transparent")
        row_cam_fps.pack(fill="x", padx=8, pady=(0, 8))
        row_cam_fps.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(row_cam_fps, text="FPS kamery (jedna liczba)", text_color=TEXT_DIM, font=(FONT_FAMILY, 10)).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(row_cam_fps, text="PeriodTime", text_color=TEXT_DIM, font=(FONT_FAMILY, 10)).grid(row=0, column=1, sticky="e")
        self.ab_camera_fps_var = ctk.StringVar(value="50")
        self.ab_camera_period_var = ctk.StringVar(value="20000.0 us")
        self.ab_camera_fps_entry = ctk.CTkEntry(row_cam_fps, textvariable=self.ab_camera_fps_var, height=24)
        self.ab_camera_fps_entry.grid(row=1, column=0, sticky="ew", padx=(0, 3))
        ctk.CTkLabel(row_cam_fps, textvariable=self.ab_camera_period_var, text_color=ACCENT, font=(FONT_FAMILY, 10, "bold")).grid(row=1, column=1, sticky="e")
        ctk.CTkButton(row_cam_fps, text="Zatwierdz FPS", width=108, height=24, command=self.on_apply_camera_fps).grid(row=1, column=2, padx=(6, 0))
        try:
            self.ab_camera_fps_entry.bind("<Return>", lambda _e: self.on_apply_camera_fps())
        except Exception:
            pass

        ctk.CTkLabel(auto_bench_frame, text="URUCHOMIENIE", text_color=ACCENT, font=(FONT_FAMILY, 10, "bold")).pack(anchor="w", padx=8, pady=(10, 2))
        row_btns = ctk.CTkFrame(auto_bench_frame, fg_color="transparent")
        row_btns.pack(fill="x", padx=8, pady=(2, 8))
        row_btns.grid_columnconfigure((0, 1), weight=1)
        self.ab_status_var = ctk.StringVar(value="Auto benchmark: gotowy")
        self.ab_start_btn = ctk.CTkButton(row_btns, text="▶ Start Auto Benchmark", command=self.start_auto_benchmark, height=28)
        self.ab_stop_btn = ctk.CTkButton(row_btns, text="■ Stop", command=self.stop_auto_benchmark, fg_color=DANGER, hover_color="#B91C1C", height=28)
        self.ab_start_btn.grid(row=0, column=0, sticky="ew", padx=(0, 3))
        self.ab_stop_btn.grid(row=0, column=1, sticky="ew", padx=(3, 0))
        try:
            self.ab_start_btn.configure(text="Start Auto Benchmark")
            self.ab_stop_btn.configure(text="Stop")
        except Exception:
            pass
        ctk.CTkLabel(
            auto_bench_frame,
            textvariable=self.ab_status_var,
            text_color=TEXT_DIM,
            font=(FONT_FAMILY, 10),
        ).pack(anchor="w", padx=8, pady=(2, 4))
        self.ab_plan_var = ctk.StringVar(value="Plan: -")
        ctk.CTkLabel(
            auto_bench_frame,
            textvariable=self.ab_plan_var,
            text_color=ACCENT,
            font=(FONT_FAMILY, 10),
        ).pack(anchor="w", padx=8, pady=(0, 8))
        self._on_auto_bench_batch_mode_change()
        self._on_auto_bench_write_mode_change()
        for _v in (
            self.ab_models_dir_var,
            self.ab_single_model_var,
            self.ab_imgsz_var,
            self.ab_batch_var,
            self.ab_batch_from_var,
            self.ab_batch_to_var,
            self.ab_batch_step_var,
            self.ab_modes_var,
            self.ab_camera_fps_var,
        ):
            try:
                _v.trace_add("write", lambda *_args: self._update_auto_bench_plan_summary())
            except Exception:
                pass
        try:
            self.ab_batch_mode_var.trace_add("write", lambda *_args: self._on_auto_bench_batch_mode_change())
        except Exception:
            pass
        try:
            self.ab_modes_var.trace_add("write", lambda *_args: self._on_auto_bench_write_mode_change())
        except Exception:
            pass

        # --- AKCJE NAGRYWANIA (karta 2x2) ---
        # --- REJESTRACJA I ZAPIS (kompakt) ---
        actions_card = ctk.CTkFrame(top, fg_color=CARD_BG, corner_radius=10)
        actions_card.pack(fill="x", padx=card_padx, pady=card_pady)

        ctk.CTkLabel(
            actions_card,
            text="Rejestracja i zapis",
            text_color=TEXT_MAIN,
            font=(FONT_FAMILY, 13, "bold"),
        ).pack(anchor="w", padx=10, pady=(6, 2))

        ctk.CTkLabel(
            actions_card,
            text=".bin = ciągły • bufor = ostatnie sekundy",
            text_color=TEXT_DIM,
            font=(FONT_FAMILY, 10),
        ).pack(anchor="w", padx=10, pady=(0, 4))

        actions_grid = ctk.CTkFrame(actions_card, fg_color="transparent")
        actions_grid.pack(fill="x", padx=inner_padx, pady=(0, 8))
        actions_grid.grid_columnconfigure((0, 1), weight=1)

        # rząd 1
        self.btn_start = self._make_btn(
            actions_grid, "● Start .bin", self.on_start_record, "#FFB26B", height=30, font_size=11
        )
        self.btn_stop = self._make_btn(
            actions_grid, "■ Stop .bin", self.on_stop_record, DANGER, height=30, font_size=11
        )
        self.btn_start.grid(row=0, column=0, padx=(0, 4), pady=3, sticky="ew")
        self.btn_stop.grid(row=0, column=1, padx=(4, 0), pady=3, sticky="ew")

        # rząd 2
        self.btn_save_buffer = self._make_btn(
            actions_grid, "💾 Bufor", self.on_save_buffer, "#9B59B6", height=30, font_size=11
        )
        self.btn_shot = self._make_btn(
            actions_grid, "📸 Zdjęcie", self.on_take_shot, SUCCESS, height=30, font_size=11
        )
        self.btn_save_buffer.grid(row=1, column=0, padx=(0, 4), pady=3, sticky="ew")
        self.btn_shot.grid(row=1, column=1, padx=(4, 0), pady=3, sticky="ew")
        buffer_cfg = ctk.CTkFrame(actions_grid, fg_color="transparent")
        buffer_cfg.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 2))

        ctk.CTkLabel(
            buffer_cfg,
            text="Bufor [s]",
            text_color=TEXT_DIM,
            font=(FONT_FAMILY, 11),
        ).pack(side="left")

        self.buffer_sec_value = ctk.CTkLabel(
            buffer_cfg,
            text="5",
            text_color=ACCENT,
            font=(FONT_FAMILY, 11, "bold"),
        )
        self.buffer_sec_value.pack(side="right")

        self.buffer_sec_slider = ctk.CTkSlider(
            buffer_cfg,
            from_=1,
            to=30,
            number_of_steps=29,
            command=self.on_buffer_seconds_change,
        )
        self.buffer_sec_slider.pack(fill="x", pady=(4, 0))
        self.buffer_sec_slider.set(5)


        # # === DOLNY PANEL (STATYSTYKI) ===
        # stats_frame = ctk.CTkFrame(side, fg_color=PANEL_BG)
        # stats_frame.grid(row=1, column=0, sticky="nsew")
        # stats_frame.grid_rowconfigure(0, weight=0)
        # stats_frame.grid_rowconfigure(1, weight=1)
        # stats_frame.grid_columnconfigure(0, weight=1)

        # --- Przycisk zamknięcia aplikacji na dole panelu ---
        exit_frame = ctk.CTkFrame(side, fg_color=PANEL_BG)
        exit_frame.grid(row=2, column=0, sticky="ew")

        self.btn_exit = ctk.CTkButton(
            exit_frame,
            text="❌ Zamknij aplikację",
            fg_color="#DDDDDD",
            hover_color="#CCCCCC",
            text_color="#555555",
            corner_radius=10,
            height=38,
            command=self.on_close_all,
        )
        self.btn_exit.pack(pady=(0, 10), padx=10, fill="x")

        # 🔹 Toast (status) – zostaje na main_frame
        self.status_toast = ctk.CTkLabel(
            self.main_frame,
            text="",
            fg_color="#333333",
            text_color="#FFFFFF",
            font=(FONT_FAMILY, 13, "bold"),
            corner_radius=10,
            padx=20,
            pady=10,
        )
        self.status_toast.place_forget()

    def _ensure_auto_benchmark_vars(self):
        if hasattr(self, "ab_model_source_var"):
            return
        self.ab_model_source_var = ctk.StringVar(value="Wiele modeli (folder)")
        self.ab_models_dir_var = ctk.StringVar(value="")
        self.ab_imgsz_var = ctk.StringVar(value="640,960")
        self.ab_batch_var = ctk.StringVar(value="4,8,12")
        self.ab_batch_mode_var = ctk.StringVar(value="Zakres")
        self.ab_single_model_var = ctk.StringVar(value="")
        self.ab_batch_from_var = ctk.StringVar(value="4")
        self.ab_batch_to_var = ctk.StringVar(value="16")
        self.ab_batch_step_var = ctk.StringVar(value="4")
        self.ab_modes_var = ctk.StringVar(value="Wszystkie kombinacje")
        self.ab_backend_mode_var = ctk.StringVar(value="Auto (wg modelu)")
        self.ab_measure_s_var = ctk.StringVar(value="10")
        self.ab_warmup_s_var = ctk.StringVar(value="4")
        self.ab_bin_s_var = ctk.StringVar(value="10")
        self.ab_buffer_count_var = ctk.StringVar(value="2")
        self.ab_buffer_pause_s_var = ctk.StringVar(value="3")
        self.ab_out_var = ctk.StringVar(value="")
        self.ab_camera_fps_var = ctk.StringVar(value="50")
        self.ab_camera_period_var = ctk.StringVar(value="20000.0 us")
        self.ab_status_var = ctk.StringVar(value="Auto benchmark: gotowy")
        self.ab_plan_var = ctk.StringVar(value="Plan: -")

    def _build_main_ui_operator_old(self):
        pad = 10
        inner = 8
        self._ensure_auto_benchmark_vars()

        self.main_frame.grid_rowconfigure(0, weight=0)
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(2, weight=0)
        self.main_frame.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(self.main_frame, fg_color=CARD_BG, corner_radius=12)
        header.grid(row=0, column=0, sticky="ew", padx=pad, pady=(pad, 6))
        header.grid_columnconfigure(0, weight=1)

        self.header_status_var = ctk.StringVar(value="Kamery: --/-- | CPU/GPU: -- | Remote: OFF")
        self.header_session_var = ctk.StringVar(value="Folder sesji: --")
        self.header_disk_var = ctk.StringVar(value="Wolne miejsce: --")
        self.header_hint_var = ctk.StringVar(value="Gotowy do startu sesji")

        h_left = ctk.CTkFrame(header, fg_color="transparent")
        h_left.grid(row=0, column=0, sticky="ew", padx=10, pady=8)
        ctk.CTkLabel(h_left, text="VolleyHub", text_color=TEXT_MAIN, font=(FONT_FAMILY, 16, "bold")).pack(anchor="w")
        ctk.CTkLabel(h_left, textvariable=self.header_status_var, text_color=TEXT_DIM).pack(anchor="w")
        ctk.CTkLabel(h_left, textvariable=self.header_session_var, text_color=TEXT_DIM).pack(anchor="w")
        ctk.CTkLabel(h_left, textvariable=self.header_disk_var, text_color=TEXT_DIM).pack(anchor="w")

        self.quick_start_btn = ctk.CTkButton(
            header,
            text="🚀 Start Session",
            fg_color="#0ea5e9",
            hover_color="#0284c7",
            command=self._quick_start_session,
            height=34,
            corner_radius=10,
            width=160,
        )
        self.quick_start_btn.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="e")
        ctk.CTkLabel(header, textvariable=self.header_hint_var, text_color=TEXT_DIM).grid(row=1, column=1, padx=(0, 10), pady=(0, 8), sticky="e")

        body = ctk.CTkFrame(self.main_frame, fg_color=BG_LIGHT)
        body.grid(row=1, column=0, sticky="nsew", padx=pad, pady=(0, 6))
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=28)
        body.grid_columnconfigure(1, weight=52)
        body.grid_columnconfigure(2, weight=20)

        left = ctk.CTkFrame(body, fg_color=PANEL_BG, corner_radius=12, border_width=1, border_color=BORDER_COLOR)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        session_card = ctk.CTkFrame(left, fg_color=CARD_BG, corner_radius=10)
        session_card.pack(fill="x", padx=inner, pady=(inner, 6))
        ctk.CTkLabel(session_card, text="Session i Recording", text_color=TEXT_MAIN, font=(FONT_FAMILY, 14, "bold")).pack(anchor="w", padx=10, pady=(8, 2))

        self.path_label = ctk.CTkLabel(
            session_card,
            text=self.raw_dir if self.raw_dir else "– nie wybrano –",
            text_color=TEXT_DIM,
            font=("Consolas", 10),
            justify="left",
            anchor="w",
            wraplength=320,
        )
        self.path_label.pack(fill="x", padx=10, pady=(2, 4))
        self.btn_choose_folder = ctk.CTkButton(session_card, text="📁 Zmień folder", command=self.on_choose_raw, fg_color=ACCENT, hover_color=ACCENT_HOVER, height=30)
        self.btn_choose_folder.pack(fill="x", padx=10, pady=(0, 8))

        self.rec_led = ctk.CTkLabel(session_card, text="●", text_color="#B0B0B0", font=(FONT_FAMILY, 18, "bold"))
        self.rec_led.pack(anchor="w", padx=10)
        self.rec_text = ctk.CTkLabel(session_card, text="Nagrywanie wyłączone", text_color=TEXT_DIM, font=(FONT_FAMILY, 12, "bold"), anchor="w")
        self.rec_text.pack(fill="x", padx=10, pady=(0, 6))

        self.session_elapsed_var = ctk.StringVar(value="Czas sesji: 00:00:00")
        self.buffer_fill_var = ctk.StringVar(value="Bufor: -- / -- s")
        self.left_disk_var = ctk.StringVar(value="Wolne miejsce: --")
        ctk.CTkLabel(session_card, textvariable=self.session_elapsed_var, text_color=TEXT_DIM).pack(anchor="w", padx=10)
        ctk.CTkLabel(session_card, textvariable=self.buffer_fill_var, text_color=TEXT_DIM).pack(anchor="w", padx=10)
        ctk.CTkLabel(session_card, textvariable=self.left_disk_var, text_color=TEXT_DIM).pack(anchor="w", padx=10, pady=(0, 8))

        rec_controls = ctk.CTkFrame(left, fg_color=CARD_BG, corner_radius=10)
        rec_controls.pack(fill="x", padx=inner, pady=(0, 6))
        rec_controls.grid_columnconfigure((0, 1), weight=1)
        self.btn_start = self._make_btn(rec_controls, "● Start recording", self.on_start_record, "#f97316", height=32, font_size=12)
        self.btn_pause = self._make_btn(rec_controls, "⏸ Pause", self.on_pause_record, "#f59e0b", height=32, font_size=12)
        self.btn_stop = self._make_btn(rec_controls, "■ Stop", self.on_stop_record, DANGER, height=32, font_size=12)
        self.btn_save_buffer = self._make_btn(rec_controls, "💾 Save buffer (5s)", self.on_save_buffer, "#7c3aed", height=32, font_size=12)
        self.btn_shot = self._make_btn(rec_controls, "📸 Snapshot", self.on_take_shot, SUCCESS, height=32, font_size=12)
        self.btn_start.grid(row=0, column=0, padx=(8, 4), pady=(8, 4), sticky="ew")
        self.btn_pause.grid(row=0, column=1, padx=(4, 8), pady=(8, 4), sticky="ew")
        self.btn_stop.grid(row=1, column=0, padx=(8, 4), pady=4, sticky="ew")
        self.btn_save_buffer.grid(row=1, column=1, padx=(4, 8), pady=4, sticky="ew")
        self.btn_shot.grid(row=2, column=0, columnspan=2, padx=8, pady=(4, 8), sticky="ew")

        buffer_card = ctk.CTkFrame(left, fg_color=CARD_BG, corner_radius=10)
        buffer_card.pack(fill="x", padx=inner, pady=(0, 8))
        ctk.CTkLabel(buffer_card, text="Długość bufora [s]", text_color=TEXT_DIM).pack(anchor="w", padx=10, pady=(8, 0))
        self.buffer_sec_value = ctk.CTkLabel(buffer_card, text="5", text_color=ACCENT, font=(FONT_FAMILY, 12, "bold"))
        self.buffer_sec_value.pack(anchor="e", padx=10)
        self.buffer_sec_slider = ctk.CTkSlider(buffer_card, from_=1, to=30, number_of_steps=29, command=self.on_buffer_seconds_change)
        self.buffer_sec_slider.pack(fill="x", padx=10, pady=(2, 8))
        self.buffer_sec_slider.set(5)

        center = ctk.CTkFrame(body, fg_color=PANEL_BG, corner_radius=12, border_width=1, border_color=BORDER_COLOR)
        center.grid(row=0, column=1, sticky="nsew", padx=6)
        center.grid_rowconfigure(1, weight=1)
        center.grid_rowconfigure(2, weight=0)
        center.grid_rowconfigure(3, weight=0)
        center.grid_columnconfigure(0, weight=1)

        preview_top = ctk.CTkFrame(center, fg_color=CARD_BG, corner_radius=10)
        preview_top.grid(row=0, column=0, sticky="ew", padx=inner, pady=(inner, 6))
        preview_top.grid_columnconfigure(0, weight=1)
        self.preview_status = ctk.CTkLabel(preview_top, text="Podgląd wyłączony", text_color=TEXT_DIM, font=(FONT_FAMILY, 12, "bold"), anchor="w")
        self.preview_status.grid(row=0, column=0, sticky="w", padx=10, pady=8)
        self.preview_btn = ctk.CTkButton(preview_top, text="▶ Włącz podgląd", command=self.toggle_preview, fg_color=ACCENT, hover_color=ACCENT_HOVER, height=30)
        self.preview_btn.grid(row=0, column=1, sticky="e", padx=10, pady=8)

        preview_wall = ctk.CTkFrame(center, fg_color=CARD_BG, corner_radius=10)
        preview_wall.grid(row=1, column=0, sticky="nsew", padx=inner, pady=(0, 6))
        preview_wall.grid_columnconfigure(0, weight=1)
        preview_wall.grid_columnconfigure(1, weight=1)
        preview_wall.grid_rowconfigure(0, weight=1)
        preview_wall.grid_rowconfigure(1, weight=1)

        self.camera_labels = {}
        self.img_labels = {}
        self.preview_frames = {}
        self.sliders = {}

        for i, role in enumerate(self.roles):
            row, col = divmod(i, 2)
            cam_card = ctk.CTkFrame(preview_wall, fg_color=PANEL_BG, corner_radius=10, border_width=1, border_color=BORDER_COLOR)
            cam_card.grid(row=row, column=col, sticky="nsew", padx=6, pady=6)
            cam_card.grid_rowconfigure(1, weight=1)
            cam_card.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(
                cam_card,
                text=role.upper(),
                text_color=TEXT_MAIN,
                font=(FONT_FAMILY, 12, "bold"),
            ).grid(row=0, column=0, sticky="w", padx=8, pady=(6, 2))

            preview_box = ctk.CTkFrame(cam_card, fg_color=CARD_INNER_BG, corner_radius=8, border_width=1, border_color=BORDER_COLOR)
            preview_box.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 4))
            self.preview_frames[role] = preview_box

            lbl = ctk.CTkLabel(
                preview_box,
                text=f"{role}\nOczekiwanie na obraz...",
                fg_color="transparent",
                text_color=TEXT_DIM,
                font=(FONT_FAMILY, 11),
            )
            lbl.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.96, relheight=0.96)
            lbl.bind("<Configure>", lambda e, r=role: (self.fullres_frames.get(r) is not None and self.display_frame(r, self.fullres_frames[r])))
            lbl.bind("<Double-Button-1>", lambda e, r=role: self.on_preview_double_click(r))
            self.camera_labels[role] = lbl
            self.img_labels[role] = lbl

            eff_wrap = ctk.CTkFrame(cam_card, fg_color="transparent")
            eff_wrap.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 2))
            eff_wrap.grid_columnconfigure(1, weight=1)
            eff_bar = ctk.CTkProgressBar(eff_wrap, width=90, fg_color="#334155", progress_color=ACCENT)
            eff_bar.set(0.0)
            eff_bar.grid(row=0, column=0, sticky="w", padx=(0, 6))
            eff_lbl = ctk.CTkLabel(eff_wrap, text="0% (0/0) b:0", text_color=TEXT_DIM, font=(FONT_FAMILY, 10))
            eff_lbl.grid(row=0, column=1, sticky="w")
            self.eff_bars[role] = eff_bar
            self.eff_labels[role] = eff_lbl

            ctrl = ctk.CTkFrame(cam_card, fg_color="transparent")
            ctrl.grid(row=3, column=0, sticky="ew", padx=8, pady=(0, 6))
            ctrl.grid_columnconfigure(1, weight=1)

            ctk.CTkLabel(ctrl, text="EXP", text_color=TEXT_DIM, font=(FONT_FAMILY, 10)).grid(row=0, column=0, sticky="w", padx=(0, 6))
            exp_slider = ctk.CTkSlider(ctrl, from_=100, to=20000, command=lambda v, r=role: self.on_expo_change(v, r), fg_color="#334155", progress_color=ACCENT, button_color=ACCENT, button_hover_color=ACCENT_HOVER)
            exp_slider.grid(row=0, column=1, sticky="ew")
            exp_val = ctk.CTkLabel(ctrl, text="1000 µs", text_color=TEXT_DIM, font=(FONT_FAMILY, 10))
            exp_val.grid(row=0, column=2, sticky="e", padx=(6, 0))
            exp_entry = ctk.CTkEntry(ctrl, width=80, placeholder_text="µs")
            exp_entry.grid(row=0, column=3, sticky="e", padx=(6, 0))
            exp_entry.bind("<Return>", lambda e, r=role, w=exp_entry: self._apply_from_entry(r, "exp", w))
            exp_entry.bind("<KP_Enter>", lambda e, r=role, w=exp_entry: self._apply_from_entry(r, "exp", w))

            ctk.CTkLabel(ctrl, text="GAIN", text_color=TEXT_DIM, font=(FONT_FAMILY, 10)).grid(row=1, column=0, sticky="w", padx=(0, 6), pady=(4, 0))
            gain_slider = ctk.CTkSlider(ctrl, from_=0, to=24, command=lambda v, r=role: self.on_gain_change(v, r), fg_color="#334155", progress_color=ACCENT, button_color=ACCENT, button_hover_color=ACCENT_HOVER)
            gain_slider.grid(row=1, column=1, sticky="ew", pady=(4, 0))
            gain_val = ctk.CTkLabel(ctrl, text="0.0 dB", text_color=TEXT_DIM, font=(FONT_FAMILY, 10))
            gain_val.grid(row=1, column=2, sticky="e", padx=(6, 0), pady=(4, 0))
            gain_entry = ctk.CTkEntry(ctrl, width=80, placeholder_text="dB")
            gain_entry.grid(row=1, column=3, sticky="e", padx=(6, 0), pady=(4, 0))
            gain_entry.bind("<Return>", lambda e, r=role, w=gain_entry: self._apply_from_entry(r, "gain", w))
            gain_entry.bind("<KP_Enter>", lambda e, r=role, w=gain_entry: self._apply_from_entry(r, "gain", w))

            self.sliders[role] = {
                "exp_slider": exp_slider,
                "exp_val": exp_val,
                "exp_entry": exp_entry,
                "gain_slider": gain_slider,
                "gain_val": gain_val,
                "gain_entry": gain_entry,
            }

        health_card = ctk.CTkFrame(center, fg_color=CARD_BG, corner_radius=10)
        health_card.grid(row=2, column=0, sticky="ew", padx=inner, pady=(0, 6))
        health_card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            health_card,
            text="Sygnał wydolności systemu",
            text_color=TEXT_MAIN,
            font=(FONT_FAMILY, 13, "bold"),
        ).pack(anchor="w", padx=10, pady=(10, 2))
        self.system_health_var = ctk.StringVar(value="⏳ Oczekiwanie na pomiary...")
        self.system_latency_var = ctk.StringVar(value="Opóźnienie: -- ms | trend: --")
        self.system_health_label = ctk.CTkLabel(
            health_card,
            textvariable=self.system_health_var,
            text_color=TEXT_DIM,
            font=(FONT_FAMILY, 12, "bold"),
            anchor="w",
        )
        self.system_health_label.pack(fill="x", padx=10, pady=(0, 2))
        self.system_latency_label = ctk.CTkLabel(
            health_card,
            textvariable=self.system_latency_var,
            text_color=TEXT_DIM,
            font=(FONT_FAMILY, 11),
            anchor="w",
        )
        self.system_latency_label.pack(fill="x", padx=10, pady=(0, 6))
        self.system_health_bar = ctk.CTkProgressBar(
            health_card,
            fg_color="#334155",
            progress_color="#22c55e",
            height=14,
        )
        self.system_health_bar.pack(fill="x", padx=10, pady=(0, 10))
        self.system_health_bar.set(0.0)

        ai_card = ctk.CTkFrame(center, fg_color=CARD_BG, corner_radius=10)
        ai_card.grid(row=3, column=0, sticky="ew", padx=inner, pady=(0, inner))
        ctk.CTkLabel(ai_card, text="Detekcja AI", text_color=TEXT_MAIN, font=(FONT_FAMILY, 13, "bold")).pack(anchor="w", padx=10, pady=(8, 4))

        self.ai_tabs = ctk.CTkTabview(ai_card)
        self.ai_tabs.pack(fill="x", padx=8, pady=(0, 8))
        tab_std = self.ai_tabs.add("Standard")
        tab_pro = self.ai_tabs.add("Pro")

        self.yolo_status = ctk.CTkLabel(tab_std, text="Detekcja wyłączona", text_color=TEXT_DIM, font=(FONT_FAMILY, 12, "bold"), anchor="w")
        self.yolo_status.pack(fill="x", padx=8, pady=(8, 2))
        self.yolo_perf_label = ctk.CTkLabel(tab_std, text="Szybkość: -- ms | -- img/s", text_color=TEXT_DIM, anchor="w")
        self.yolo_perf_label.pack(fill="x", padx=8, pady=(0, 2))
        self.yolo_perf_canvas = None

        self.yolo_model_path = ""
        self.yolo_model_label = ctk.CTkLabel(tab_std, text="Model: domyślny", text_color=TEXT_DIM, anchor="w")
        self.yolo_model_label.pack(fill="x", padx=8, pady=(0, 4))
        model_row = ctk.CTkFrame(tab_std, fg_color="transparent")
        model_row.pack(fill="x", padx=8, pady=(0, 4))
        self.yolo_model_btn = ctk.CTkButton(model_row, text="📁 Wybierz model", command=self.choose_yolo_model, height=28)
        self.yolo_model_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.yolo_load_btn = ctk.CTkButton(model_row, text="⚡ Wybierz i załaduj", command=self._pick_and_load_yolo_model, height=28, fg_color="#0f766e", hover_color="#0d5f5a")
        self.yolo_load_btn.pack(side="left", fill="x", expand=True, padx=(4, 0))

        conf_row = ctk.CTkFrame(tab_std, fg_color="transparent")
        conf_row.pack(fill="x", padx=8, pady=(2, 2))
        ctk.CTkLabel(conf_row, text="Czułość detekcji", text_color=TEXT_DIM).pack(side="left")
        self.yolo_conf_value = ctk.CTkLabel(conf_row, text="0.25", text_color=ACCENT, font=(FONT_FAMILY, 11, "bold"))
        self.yolo_conf_value.pack(side="right")
        self.yolo_conf_slider = ctk.CTkSlider(tab_std, from_=0, to=1, number_of_steps=18, command=self.on_yolo_conf_change)
        self.yolo_conf_slider.pack(fill="x", padx=8, pady=(0, 4))
        self.yolo_conf_slider.set(0.25)

        size_row = ctk.CTkFrame(tab_std, fg_color="transparent")
        size_row.pack(fill="x", padx=8, pady=(2, 2))
        ctk.CTkLabel(size_row, text="Rozmiar analizy", text_color=TEXT_DIM).pack(side="left")
        self.yolo_size_value = ctk.CTkLabel(size_row, text="640px", text_color=ACCENT, font=(FONT_FAMILY, 11, "bold"))
        self.yolo_size_value.pack(side="right")
        self.yolo_size_slider = ctk.CTkSlider(tab_std, from_=320, to=1280, number_of_steps=12, command=self.on_yolo_size_change)
        self.yolo_size_slider.pack(fill="x", padx=8, pady=(0, 4))
        self.yolo_size_slider.set(640)

        batch_row = ctk.CTkFrame(tab_std, fg_color="transparent")
        batch_row.pack(fill="x", padx=8, pady=(2, 2))
        ctk.CTkLabel(batch_row, text="Images processed at once", text_color=TEXT_DIM).pack(side="left")
        self.yolo_batch_value = ctk.CTkLabel(batch_row, text="4", text_color=ACCENT, font=(FONT_FAMILY, 11, "bold"))
        self.yolo_batch_value.pack(side="right")
        self.yolo_batch_slider = ctk.CTkSlider(tab_std, from_=4, to=32, number_of_steps=7, command=self.on_yolo_batch_change)
        self.yolo_batch_slider.pack(fill="x", padx=8, pady=(0, 6))
        self.yolo_batch_slider.set(4)

        std_btns = ctk.CTkFrame(tab_std, fg_color="transparent")
        std_btns.pack(fill="x", padx=8, pady=(2, 8))
        std_btns.grid_columnconfigure((0, 1), weight=1)
        self.yolo_toggle_btn = self._make_btn(
            std_btns,
            "🎯 Enable detection",
            self.toggle_yolo,
            "#FF6B35",
            height=30,
            font_size=11,
        )
        self.yolo_toggle_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.yolo_traj_btn = ctk.CTkButton(std_btns, text="📈 Show ball trajectory", command=self.open_trajectory_window, height=30, fg_color="#334155", hover_color="#1e293b")
        self.yolo_traj_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        self.live_backend_label = ctk.CTkLabel(tab_pro, textvariable=self.live_backend_var, text_color=TEXT_DIM, anchor="w")
        self.live_backend_label.pack(fill="x", padx=8, pady=(8, 4))
        pro_backend = ctk.CTkFrame(tab_pro, fg_color="transparent")
        pro_backend.pack(fill="x", padx=8, pady=(0, 4))
        self.live_backend_btn = ctk.CTkButton(pro_backend, text="Backend: Ultralytics", command=self.toggle_live_infer_backend, height=28)
        self.live_backend_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.live_engine_btn = ctk.CTkButton(pro_backend, text="Wybierz .engine", command=self.choose_live_trt_engine, height=28)
        self.live_engine_btn.pack(side="right")
        self.live_track_status_label = ctk.CTkLabel(tab_pro, textvariable=self.live_track_status_var, text_color=TEXT_DIM, anchor="w")
        self.live_track_status_label.pack(fill="x", padx=8, pady=(0, 2))
        self.live_track_metrics_label = ctk.CTkLabel(tab_pro, textvariable=self.live_track_metrics_var, text_color=TEXT_DIM, anchor="w")
        self.live_track_metrics_label.pack(fill="x", padx=8, pady=(0, 2))
        self.live_compare_label = ctk.CTkLabel(tab_pro, textvariable=self.live_compare_var, text_color=TEXT_DIM, anchor="w")
        self.live_compare_label.pack(fill="x", padx=8, pady=(0, 4))
        pro_btns = ctk.CTkFrame(tab_pro, fg_color="transparent")
        pro_btns.pack(fill="x", padx=8, pady=(2, 8))
        pro_btns.grid_columnconfigure((0, 1), weight=1)
        self.live_track_btn = ctk.CTkButton(pro_btns, text="Start LIVE_TRACK", command=self.toggle_live_track, fg_color="#0ea5e9", hover_color="#0284c7", height=30)
        self.live_track_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.live_save_stats_btn = ctk.CTkButton(pro_btns, text="Zapisz live stats", command=self.save_live_track_stats_snapshot, fg_color="#475569", hover_color="#334155", height=30)
        self.live_save_stats_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        pro_expert = ctk.CTkFrame(tab_pro, fg_color="transparent")
        pro_expert.pack(fill="x", padx=8, pady=(0, 8))
        self.yolo_preprocess_label = ctk.CTkLabel(pro_expert, text="Preprocess: CPU", text_color=TEXT_DIM, anchor="w")
        self.yolo_preprocess_label.pack(side="left", fill="x", expand=True)
        self.yolo_preprocess_btn = ctk.CTkButton(pro_expert, text="CPU/GPU", command=self.toggle_yolo_preprocess_backend, height=26, width=96)
        self.yolo_preprocess_btn.pack(side="left", padx=(6, 6))
        self.yolo_debug_btn = ctk.CTkButton(pro_expert, text="Debug", command=self.debug_yolo, height=26, width=84)
        self.yolo_debug_btn.pack(side="left")

        self._apply_live_backend_ui()
        self._apply_yolo_preprocess_backend_ui()

        right = ctk.CTkFrame(body, fg_color=PANEL_BG, corner_radius=12, border_width=1, border_color=BORDER_COLOR)
        right.grid(row=0, column=2, sticky="nsew", padx=(6, 0))
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(right, text="Status i feedback", text_color=TEXT_MAIN, font=(FONT_FAMILY, 13, "bold")).grid(row=0, column=0, sticky="w", padx=10, pady=(10, 6))
        self.stats_box = ctk.CTkTextbox(right, wrap="word", fg_color=CARD_BG, text_color=TEXT_MAIN)
        self.stats_box.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 6))
        self.stats_box.insert("end", "✅ System gotowy\n")
        self.session_summary_var = ctk.StringVar(value="Podsumowanie sesji: brak danych")
        ctk.CTkLabel(right, textvariable=self.session_summary_var, text_color=TEXT_DIM, justify="left", anchor="w").grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))

        bottom = ctk.CTkFrame(self.main_frame, fg_color=CARD_BG, corner_radius=12)
        bottom.grid(row=2, column=0, sticky="ew", padx=pad, pady=(0, pad))
        bottom.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)
        ctk.CTkButton(bottom, text="📷 Camera assignment", command=self._open_camera_assignment).grid(row=0, column=0, padx=6, pady=8, sticky="ew")
        self.remote_status = ctk.CTkLabel(bottom, text="Server wyłączony", text_color=TEXT_DIM)
        self.remote_status.grid(row=0, column=1, padx=6, pady=8, sticky="ew")
        self.remote_btn = ctk.CTkButton(bottom, text="🛰 Remote access", command=self.toggle_remote_server)
        self.remote_btn.grid(row=0, column=2, padx=6, pady=8, sticky="ew")
        ctk.CTkButton(bottom, text="📊 Benchmark", command=self._open_benchmark_window).grid(row=0, column=3, padx=6, pady=8, sticky="ew")
        self.btn_exit = ctk.CTkButton(bottom, text="❌ Exit", fg_color="#DDDDDD", hover_color="#CCCCCC", text_color="#555555", command=self.on_close_all)
        self.btn_exit.grid(row=0, column=4, padx=6, pady=8, sticky="ew")

        self.remote_url_label = ctk.CTkLabel(bottom, textvariable=self.remote_url_var, text_color="#0077CC", font=(FONT_FAMILY, 11, "bold"))
        self.remote_url_label.grid(row=1, column=0, columnspan=5, sticky="w", padx=10, pady=(0, 4))
        self.remote_url_label.bind("<Button-1>", lambda e: self._open_remote_url())
        self.remote_url_label.bind("<Enter>", lambda e: self.remote_url_label.configure(cursor="hand2"))
        self.remote_url_label.bind("<Leave>", lambda e: self.remote_url_label.configure(cursor=""))
        self.remote_qr_label = ctk.CTkLabel(bottom, text="")
        self.remote_qr_label.grid(row=2, column=0, columnspan=5, sticky="w", padx=10, pady=(0, 8))

        self.status_toast = ctk.CTkLabel(
            self.main_frame,
            text="",
            fg_color="#333333",
            text_color="#FFFFFF",
            font=(FONT_FAMILY, 13, "bold"),
            corner_radius=10,
            padx=20,
            pady=10,
        )
        self.status_toast.place_forget()
        self._dashboard_header = header
        self._dashboard_body = body
        self._dashboard_left = left
        self._dashboard_center = center
        self._dashboard_right = right
        self._dashboard_bottom = bottom
        self._dashboard_path_label = self.path_label
        self._dashboard_summary_label = self.session_summary_var
        try:
            self.main_frame.bind("<Configure>", self._on_dashboard_resize)
        except Exception:
            pass
        self._refresh_dashboard_metrics()

    def _on_dashboard_resize_preview_first(self, event=None):
        try:
            total_w = int(self.main_frame.winfo_width())
            total_h = int(self.main_frame.winfo_height())
            if total_w <= 10 or total_h <= 10:
                return

            body_w = max(500, total_w - 24)

            preview_w = int(body_w * 0.68)
            right_w = max(320, body_w - preview_w)

            if hasattr(self, "_dashboard_body"):
                self._dashboard_body.grid_columnconfigure(0, minsize=preview_w)
                self._dashboard_body.grid_columnconfigure(1, minsize=right_w)

            if hasattr(self, "_dashboard_path_label"):
                self._dashboard_path_label.configure(wraplength=max(220, right_w - 40))

            if hasattr(self, "stats_box"):
                self.stats_box.configure(width=max(240, right_w - 30))

            if hasattr(self, "quick_start_btn"):
                btn_w = max(140, min(180, int(total_w * 0.14)))
                self.quick_start_btn.configure(width=btn_w)

        except Exception:
            pass

    def _build_main_ui_operator(self):
        pad = 10
        inner = 8
        self._ensure_auto_benchmark_vars()

        # =========================================================
        # ROOT LAYOUT
        # =========================================================
        self.main_frame.grid_rowconfigure(0, weight=0)  # header
        self.main_frame.grid_rowconfigure(1, weight=1)  # content
        self.main_frame.grid_rowconfigure(2, weight=0)  # footer
        self.main_frame.grid_columnconfigure(0, weight=1)

        # =========================================================
        # HEADER — status + quick actions
        # =========================================================
        header = ctk.CTkFrame(self.main_frame, fg_color=CARD_BG, corner_radius=12)
        header.grid(row=0, column=0, sticky="ew", padx=pad, pady=(pad, 6))
        header.grid_columnconfigure(0, weight=1)
        header.grid_columnconfigure(1, weight=0)

        self.header_status_var = ctk.StringVar(value="Kamery: --/-- | CPU: -- | GPU: -- | Remote: OFF")
        self.header_session_var = ctk.StringVar(value="Folder sesji: --")
        self.header_disk_var = ctk.StringVar(value="Wolne miejsce: --")
        self.header_hint_var = ctk.StringVar(value="Gotowy do startu sesji")

        header_left = ctk.CTkFrame(header, fg_color="transparent")
        header_left.grid(row=0, column=0, sticky="ew", padx=10, pady=8)

        ctk.CTkLabel(
            header_left,
            text="VolleyHub — Operator",
            text_color=TEXT_MAIN,
            font=(FONT_FAMILY, 17, "bold"),
        ).pack(anchor="w")

        ctk.CTkLabel(
            header_left,
            textvariable=self.header_status_var,
            text_color=TEXT_DIM,
            font=(FONT_FAMILY, 12),
        ).pack(anchor="w")

        ctk.CTkLabel(
            header_left,
            textvariable=self.header_session_var,
            text_color=TEXT_DIM,
            font=(FONT_FAMILY, 12),
        ).pack(anchor="w")

        ctk.CTkLabel(
            header_left,
            textvariable=self.header_disk_var,
            text_color=TEXT_DIM,
            font=(FONT_FAMILY, 12),
        ).pack(anchor="w")

        header_right = ctk.CTkFrame(header, fg_color="transparent")
        header_right.grid(row=0, column=1, sticky="e", padx=10, pady=8)

        self.quick_start_btn = ctk.CTkButton(
            header_right,
            text="🚀 Start Session",
            fg_color="#0ea5e9",
            hover_color="#0284c7",
            command=self._quick_start_session,
            height=34,
            corner_radius=10,
            width=150,
        )
        self.quick_start_btn.pack(fill="x", pady=(0, 6))

        self.preview_btn = ctk.CTkButton(
            header_right,
            text="▶ Włącz podgląd",
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            command=self.toggle_preview,
            height=32,
            corner_radius=10,
            width=150,
        )
        self.preview_btn.pack(fill="x", pady=(0, 6))

        self.yolo_btn = ctk.CTkButton(
            header_right,
            text="🎯 Detekcja OFF",
            fg_color="#FF6B35",
            hover_color="#E55A2B",
            command=self.toggle_yolo,
            height=32,
            corner_radius=10,
            width=150,
        )
        self.yolo_btn.pack(fill="x", pady=(0, 4))

        ctk.CTkLabel(
            header_right,
            textvariable=self.header_hint_var,
            text_color=TEXT_DIM,
            font=(FONT_FAMILY, 11),
        ).pack(anchor="e", pady=(4, 0))

        # =========================================================
        # CONTENT — left preview / right contextual tabs
        # =========================================================
        body = ctk.CTkFrame(self.main_frame, fg_color=BG_LIGHT)
        body.grid(row=1, column=0, sticky="nsew", padx=pad, pady=(0, 6))
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=68)  # PREVIEW
        body.grid_columnconfigure(1, weight=32)  # RIGHT PANEL

        # ---------------------------------------------------------
        # PREVIEW AREA
        # ---------------------------------------------------------
        preview_area = ctk.CTkFrame(
            body,
            fg_color=PANEL_BG,
            corner_radius=12,
            border_width=1,
            border_color=BORDER_COLOR,
        )
        preview_area.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        preview_area.grid_rowconfigure(1, weight=1)
        preview_area.grid_columnconfigure(0, weight=1)

        preview_top = ctk.CTkFrame(preview_area, fg_color=CARD_BG, corner_radius=10)
        preview_top.grid(row=0, column=0, sticky="ew", padx=inner, pady=(inner, 6))
        preview_top.grid_columnconfigure(0, weight=1)
        preview_top.grid_columnconfigure(1, weight=0)

        self.preview_status = ctk.CTkLabel(
            preview_top,
            text="Podgląd wyłączony",
            text_color=TEXT_DIM,
            font=(FONT_FAMILY, 13, "bold"),
            anchor="w",
        )
        self.preview_status.grid(row=0, column=0, sticky="w", padx=10, pady=8)

        self.preview_hint_var = ctk.StringVar(value="Double click na kamerze = powiększenie")
        ctk.CTkLabel(
            preview_top,
            textvariable=self.preview_hint_var,
            text_color=TEXT_DIM,
            font=(FONT_FAMILY, 11),
        ).grid(row=0, column=1, sticky="e", padx=10, pady=8)

        preview_wall = ctk.CTkFrame(preview_area, fg_color=CARD_BG, corner_radius=10)
        preview_wall.grid(row=1, column=0, sticky="nsew", padx=inner, pady=(0, inner))
        preview_wall.grid_columnconfigure(0, weight=1)
        preview_wall.grid_columnconfigure(1, weight=1)
        preview_wall.grid_rowconfigure(0, weight=1)
        preview_wall.grid_rowconfigure(1, weight=1)

        self.camera_labels = {}
        self.img_labels = {}
        self.preview_frames = {}
        self.sliders = {}
        self.eff_bars = {}
        self.eff_labels = {}

        for i, role in enumerate(self.roles):
            row, col = divmod(i, 2)

            cam_card = ctk.CTkFrame(
                preview_wall,
                fg_color=PANEL_BG,
                corner_radius=10,
                border_width=1,
                border_color=BORDER_COLOR,
            )
            cam_card.grid(row=row, column=col, sticky="nsew", padx=6, pady=6)
            cam_card.grid_rowconfigure(1, weight=1)
            cam_card.grid_columnconfigure(0, weight=1)

            # top bar kamery
            cam_top = ctk.CTkFrame(cam_card, fg_color="transparent")
            cam_top.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 2))
            cam_top.grid_columnconfigure(1, weight=1)

            ctk.CTkLabel(
                cam_top,
                text=role.upper(),
                text_color=TEXT_MAIN,
                font=(FONT_FAMILY, 12, "bold"),
            ).grid(row=0, column=0, sticky="w")

            eff_wrap = ctk.CTkFrame(cam_top, fg_color="transparent")
            eff_wrap.grid(row=0, column=1, sticky="e")
            eff_wrap.grid_columnconfigure(1, weight=1)

            eff_bar = ctk.CTkProgressBar(
                eff_wrap,
                width=80,
                fg_color="#334155",
                progress_color=ACCENT,
            )
            eff_bar.set(0.0)
            eff_bar.grid(row=0, column=0, sticky="e", padx=(0, 6))

            eff_lbl = ctk.CTkLabel(
                eff_wrap,
                text="0% (0/0)",
                text_color=TEXT_DIM,
                font=(FONT_FAMILY, 10),
            )
            eff_lbl.grid(row=0, column=1, sticky="e")

            self.eff_bars[role] = eff_bar
            self.eff_labels[role] = eff_lbl

            # box preview — maksymalnie duży
            preview_box = ctk.CTkFrame(
                cam_card,
                fg_color=CARD_INNER_BG,
                corner_radius=8,
                border_width=1,
                border_color=BORDER_COLOR,
            )
            preview_box.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
            self.preview_frames[role] = preview_box

            lbl = ctk.CTkLabel(
                preview_box,
                text=f"{role}\nOczekiwanie na obraz...",
                fg_color="transparent",
                text_color=TEXT_DIM,
                font=(FONT_FAMILY, 12),
            )
            lbl.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.97, relheight=0.97)
            lbl.bind(
                "<Configure>",
                lambda e, r=role: (
                        self.fullres_frames.get(r) is not None and self.display_frame(r, self.fullres_frames[r])
                ),
            )
            lbl.bind("<Double-Button-1>", lambda e, r=role: self.on_preview_double_click(r))

            self.camera_labels[role] = lbl
            self.img_labels[role] = lbl

        # ---------------------------------------------------------
        # RIGHT CONTEXT PANEL
        # ---------------------------------------------------------
        right = ctk.CTkFrame(
            body,
            fg_color=PANEL_BG,
            corner_radius=12,
            border_width=1,
            border_color=BORDER_COLOR,
        )
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        right.grid_rowconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)

        self.operator_tabs = ctk.CTkTabview(right)
        self.operator_tabs.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        tab_session = self.operator_tabs.add("Session")
        tab_ai = self.operator_tabs.add("AI / YOLO")
        tab_cameras = self.operator_tabs.add("Cameras")
        tab_diag = self.operator_tabs.add("Diagnostics")

        # =========================================================
        # TAB: SESSION
        # =========================================================
        session_card = ctk.CTkFrame(tab_session, fg_color=CARD_BG, corner_radius=10)
        session_card.pack(fill="x", padx=6, pady=(6, 6))

        ctk.CTkLabel(
            session_card,
            text="Session i Recording",
            text_color=TEXT_MAIN,
            font=(FONT_FAMILY, 14, "bold"),
        ).pack(anchor="w", padx=10, pady=(8, 2))

        self.path_label = ctk.CTkLabel(
            session_card,
            text=self.raw_dir if self.raw_dir else "– nie wybrano –",
            text_color=TEXT_DIM,
            font=("Consolas", 10),
            justify="left",
            anchor="w",
            wraplength=340,
        )
        self.path_label.pack(fill="x", padx=10, pady=(2, 4))

        self.btn_choose_folder = ctk.CTkButton(
            session_card,
            text="📁 Zmień folder",
            command=self.on_choose_raw,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            height=30,
        )
        self.btn_choose_folder.pack(fill="x", padx=10, pady=(0, 8))

        self.rec_led = ctk.CTkLabel(
            session_card,
            text="●",
            text_color="#B0B0B0",
            font=(FONT_FAMILY, 18, "bold"),
        )
        self.rec_led.pack(anchor="w", padx=10)

        self.rec_text = ctk.CTkLabel(
            session_card,
            text="Nagrywanie wyłączone",
            text_color=TEXT_DIM,
            font=(FONT_FAMILY, 12, "bold"),
            anchor="w",
        )
        self.rec_text.pack(fill="x", padx=10, pady=(0, 6))

        self.session_elapsed_var = ctk.StringVar(value="Czas sesji: 00:00:00")
        self.buffer_fill_var = ctk.StringVar(value="Bufor: -- / -- s")
        self.left_disk_var = ctk.StringVar(value="Wolne miejsce: --")

        ctk.CTkLabel(session_card, textvariable=self.session_elapsed_var, text_color=TEXT_DIM).pack(anchor="w", padx=10)
        ctk.CTkLabel(session_card, textvariable=self.buffer_fill_var, text_color=TEXT_DIM).pack(anchor="w", padx=10)
        ctk.CTkLabel(session_card, textvariable=self.left_disk_var, text_color=TEXT_DIM).pack(anchor="w", padx=10,
                                                                                              pady=(0, 8))

        rec_controls = ctk.CTkFrame(tab_session, fg_color=CARD_BG, corner_radius=10)
        rec_controls.pack(fill="x", padx=6, pady=(0, 6))
        rec_controls.grid_columnconfigure((0, 1), weight=1)

        self.btn_start = self._make_btn(rec_controls, "● Start recording", self.on_start_record, "#f97316", height=32,
                                        font_size=12)
        self.btn_pause = self._make_btn(rec_controls, "⏸ Pause", self.on_pause_record, "#f59e0b", height=32,
                                        font_size=12)
        self.btn_stop = self._make_btn(rec_controls, "■ Stop", self.on_stop_record, DANGER, height=32, font_size=12)
        self.btn_save_buffer = self._make_btn(rec_controls, "💾 Save buffer (5s)", self.on_save_buffer, "#7c3aed",
                                              height=32, font_size=12)
        self.btn_shot = self._make_btn(rec_controls, "📸 Snapshot", self.on_take_shot, SUCCESS, height=32, font_size=12)

        self.btn_start.grid(row=0, column=0, padx=(8, 4), pady=(8, 4), sticky="ew")
        self.btn_pause.grid(row=0, column=1, padx=(4, 8), pady=(8, 4), sticky="ew")
        self.btn_stop.grid(row=1, column=0, padx=(8, 4), pady=4, sticky="ew")
        self.btn_save_buffer.grid(row=1, column=1, padx=(4, 8), pady=4, sticky="ew")
        self.btn_shot.grid(row=2, column=0, columnspan=2, padx=8, pady=(4, 8), sticky="ew")

        buffer_card = ctk.CTkFrame(tab_session, fg_color=CARD_BG, corner_radius=10)
        buffer_card.pack(fill="x", padx=6, pady=(0, 6))

        ctk.CTkLabel(
            buffer_card,
            text="Długość bufora [s]",
            text_color=TEXT_DIM,
        ).pack(anchor="w", padx=10, pady=(8, 0))

        self.buffer_sec_value = ctk.CTkLabel(
            buffer_card,
            text="5",
            text_color=ACCENT,
            font=(FONT_FAMILY, 12, "bold"),
        )
        self.buffer_sec_value.pack(anchor="e", padx=10)

        self.buffer_sec_slider = ctk.CTkSlider(
            buffer_card,
            from_=1,
            to=30,
            number_of_steps=29,
            command=self.on_buffer_seconds_change,
        )
        self.buffer_sec_slider.pack(fill="x", padx=10, pady=(2, 8))
        self.buffer_sec_slider.set(5)

        remote_card = ctk.CTkFrame(tab_session, fg_color=CARD_BG, corner_radius=10)
        remote_card.pack(fill="x", padx=6, pady=(0, 6))

        self.remote_status = ctk.CTkLabel(
            remote_card,
            text="Server wyłączony",
            text_color=TEXT_DIM,
            font=(FONT_FAMILY, 12, "bold"),
        )
        self.remote_status.pack(anchor="w", padx=10, pady=(8, 2))

        self.remote_btn = ctk.CTkButton(
            remote_card,
            text="🛰 Remote access",
            command=self.toggle_remote_server,
            height=30,
        )
        self.remote_btn.pack(fill="x", padx=10, pady=(0, 6))

        self.remote_url_label = ctk.CTkLabel(
            remote_card,
            textvariable=self.remote_url_var,
            text_color="#0077CC",
            font=(FONT_FAMILY, 11, "bold"),
        )
        self.remote_url_label.pack(anchor="w", padx=10, pady=(0, 4))
        self.remote_url_label.bind("<Button-1>", lambda e: self._open_remote_url())
        self.remote_url_label.bind("<Enter>", lambda e: self.remote_url_label.configure(cursor="hand2"))
        self.remote_url_label.bind("<Leave>", lambda e: self.remote_url_label.configure(cursor=""))

        self.remote_qr_label = ctk.CTkLabel(remote_card, text="")
        self.remote_qr_label.pack(anchor="w", padx=10, pady=(0, 8))

        # =========================================================
        # TAB: AI / YOLO
        # =========================================================
        ai_card = ctk.CTkFrame(tab_ai, fg_color=CARD_BG, corner_radius=10)
        ai_card.pack(fill="x", padx=6, pady=(6, 6))

        ctk.CTkLabel(
            ai_card,
            text="Detekcja AI",
            text_color=TEXT_MAIN,
            font=(FONT_FAMILY, 14, "bold"),
        ).pack(anchor="w", padx=10, pady=(8, 4))

        self.ai_tabs = ctk.CTkTabview(ai_card)
        self.ai_tabs.pack(fill="x", padx=8, pady=(0, 8))

        tab_std = self.ai_tabs.add("Standard")
        tab_pro = self.ai_tabs.add("Pro")

        self.yolo_status = ctk.CTkLabel(
            tab_std,
            text="Detekcja wyłączona",
            text_color=TEXT_DIM,
            font=(FONT_FAMILY, 12, "bold"),
            anchor="w",
        )
        self.yolo_status.pack(fill="x", padx=8, pady=(8, 2))

        self.yolo_perf_label = ctk.CTkLabel(
            tab_std,
            text="Szybkość: -- ms | -- img/s",
            text_color=TEXT_DIM,
            anchor="w",
        )
        self.yolo_perf_label.pack(fill="x", padx=8, pady=(0, 2))
        self.yolo_perf_canvas = None

        self.yolo_model_path = ""
        self.yolo_model_label = ctk.CTkLabel(
            tab_std,
            text="Model: domyślny",
            text_color=TEXT_DIM,
            anchor="w",
        )
        self.yolo_model_label.pack(fill="x", padx=8, pady=(0, 4))

        model_row = ctk.CTkFrame(tab_std, fg_color="transparent")
        model_row.pack(fill="x", padx=8, pady=(0, 4))
        self.yolo_model_btn = ctk.CTkButton(
            model_row,
            text="📁 Wybierz model",
            command=self.choose_yolo_model,
            height=28,
        )
        self.yolo_model_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))

        self.yolo_load_btn = ctk.CTkButton(
            model_row,
            text="⚡ Wybierz i załaduj",
            command=self._pick_and_load_yolo_model,
            height=28,
            fg_color="#0f766e",
            hover_color="#0d5f5a",
        )
        self.yolo_load_btn.pack(side="left", fill="x", expand=True, padx=(4, 0))

        conf_row = ctk.CTkFrame(tab_std, fg_color="transparent")
        conf_row.pack(fill="x", padx=8, pady=(2, 2))
        ctk.CTkLabel(conf_row, text="Czułość detekcji", text_color=TEXT_DIM).pack(side="left")
        self.yolo_conf_value = ctk.CTkLabel(
            conf_row,
            text="0.25",
            text_color=ACCENT,
            font=(FONT_FAMILY, 11, "bold"),
        )
        self.yolo_conf_value.pack(side="right")

        self.yolo_conf_slider = ctk.CTkSlider(
            tab_std,
            from_=0,
            to=1,
            number_of_steps=18,
            command=self.on_yolo_conf_change,
        )
        self.yolo_conf_slider.pack(fill="x", padx=8, pady=(0, 4))
        self.yolo_conf_slider.set(0.25)

        size_row = ctk.CTkFrame(tab_std, fg_color="transparent")
        size_row.pack(fill="x", padx=8, pady=(2, 2))
        ctk.CTkLabel(size_row, text="Rozmiar analizy", text_color=TEXT_DIM).pack(side="left")
        self.yolo_size_value = ctk.CTkLabel(
            size_row,
            text="640px",
            text_color=ACCENT,
            font=(FONT_FAMILY, 11, "bold"),
        )
        self.yolo_size_value.pack(side="right")

        self.yolo_size_slider = ctk.CTkSlider(
            tab_std,
            from_=320,
            to=1280,
            number_of_steps=12,
            command=self.on_yolo_size_change,
        )
        self.yolo_size_slider.pack(fill="x", padx=8, pady=(0, 4))
        self.yolo_size_slider.set(640)

        batch_row = ctk.CTkFrame(tab_std, fg_color="transparent")
        batch_row.pack(fill="x", padx=8, pady=(2, 2))
        ctk.CTkLabel(batch_row, text="Images processed at once", text_color=TEXT_DIM).pack(side="left")
        self.yolo_batch_value = ctk.CTkLabel(
            batch_row,
            text="4",
            text_color=ACCENT,
            font=(FONT_FAMILY, 11, "bold"),
        )
        self.yolo_batch_value.pack(side="right")

        self.yolo_batch_slider = ctk.CTkSlider(
            tab_std,
            from_=4,
            to=32,
            number_of_steps=7,
            command=self.on_yolo_batch_change,
        )
        self.yolo_batch_slider.pack(fill="x", padx=8, pady=(0, 6))
        self.yolo_batch_slider.set(4)

        std_btns = ctk.CTkFrame(tab_std, fg_color="transparent")
        std_btns.pack(fill="x", padx=8, pady=(2, 8))
        std_btns.grid_columnconfigure((0, 1), weight=1)

        self.yolo_btn.configure(text="🎯 Enable detection")
        self.yolo_btn.master = std_btns  # nie wpływa funkcjonalnie, tylko porządek referencji

        yolo_local_btn = self._make_btn(
            std_btns,
            "🎯 Enable detection",
            self.toggle_yolo,
            "#FF6B35",
            height=30,
            font_size=11,
        )
        yolo_local_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self.yolo_traj_btn = ctk.CTkButton(
            std_btns,
            text="📈 Show ball trajectory",
            command=self.open_trajectory_window,
            height=30,
            fg_color="#334155",
            hover_color="#1e293b",
        )
        self.yolo_traj_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        self.live_backend_label = ctk.CTkLabel(
            tab_pro,
            textvariable=self.live_backend_var,
            text_color=TEXT_DIM,
            anchor="w",
        )
        self.live_backend_label.pack(fill="x", padx=8, pady=(8, 4))

        pro_backend = ctk.CTkFrame(tab_pro, fg_color="transparent")
        pro_backend.pack(fill="x", padx=8, pady=(0, 4))

        self.live_backend_btn = ctk.CTkButton(
            pro_backend,
            text="Backend: Ultralytics",
            command=self.toggle_live_infer_backend,
            height=28,
        )
        self.live_backend_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))

        self.live_engine_btn = ctk.CTkButton(
            pro_backend,
            text="Wybierz .engine",
            command=self.choose_live_trt_engine,
            height=28,
        )
        self.live_engine_btn.pack(side="right")

        self.live_track_status_label = ctk.CTkLabel(
            tab_pro,
            textvariable=self.live_track_status_var,
            text_color=TEXT_DIM,
            anchor="w",
        )
        self.live_track_status_label.pack(fill="x", padx=8, pady=(0, 2))

        self.live_track_metrics_label = ctk.CTkLabel(
            tab_pro,
            textvariable=self.live_track_metrics_var,
            text_color=TEXT_DIM,
            anchor="w",
        )
        self.live_track_metrics_label.pack(fill="x", padx=8, pady=(0, 2))

        self.live_compare_label = ctk.CTkLabel(
            tab_pro,
            textvariable=self.live_compare_var,
            text_color=TEXT_DIM,
            anchor="w",
        )
        self.live_compare_label.pack(fill="x", padx=8, pady=(0, 4))

        pro_btns = ctk.CTkFrame(tab_pro, fg_color="transparent")
        pro_btns.pack(fill="x", padx=8, pady=(2, 8))
        pro_btns.grid_columnconfigure((0, 1), weight=1)

        self.live_track_btn = ctk.CTkButton(
            pro_btns,
            text="Start LIVE_TRACK",
            command=self.toggle_live_track,
            fg_color="#0ea5e9",
            hover_color="#0284c7",
            height=30,
        )
        self.live_track_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self.live_save_stats_btn = ctk.CTkButton(
            pro_btns,
            text="Zapisz live stats",
            command=self.save_live_track_stats_snapshot,
            fg_color="#475569",
            hover_color="#334155",
            height=30,
        )
        self.live_save_stats_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        pro_expert = ctk.CTkFrame(tab_pro, fg_color="transparent")
        pro_expert.pack(fill="x", padx=8, pady=(0, 8))

        self.yolo_preprocess_label = ctk.CTkLabel(
            pro_expert,
            text="Preprocess: CPU",
            text_color=TEXT_DIM,
            anchor="w",
        )
        self.yolo_preprocess_label.pack(side="left", fill="x", expand=True)

        self.yolo_preprocess_btn = ctk.CTkButton(
            pro_expert,
            text="CPU/GPU",
            command=self.toggle_yolo_preprocess_backend,
            height=26,
            width=96,
        )
        self.yolo_preprocess_btn.pack(side="left", padx=(6, 6))

        self.yolo_debug_btn = ctk.CTkButton(
            pro_expert,
            text="Debug",
            command=self.debug_yolo,
            height=26,
            width=84,
        )
        self.yolo_debug_btn.pack(side="left")

        self._apply_live_backend_ui()
        self._apply_yolo_preprocess_backend_ui()

        # =========================================================
        # TAB: CAMERAS
        # =========================================================
        cameras_info = ctk.CTkFrame(tab_cameras, fg_color=CARD_BG, corner_radius=10)
        cameras_info.pack(fill="both", expand=True, padx=6, pady=(6, 6))

        ctk.CTkLabel(
            cameras_info,
            text="Ustawienia kamer",
            text_color=TEXT_MAIN,
            font=(FONT_FAMILY, 14, "bold"),
        ).pack(anchor="w", padx=10, pady=(8, 6))

        ctk.CTkLabel(
            cameras_info,
            text="Tutaj sterowanie ekspozycją i gain dla każdej kamery. "
                 "Przeniesienie tych kontrolek z miniatur daje dużo większy preview.",
            text_color=TEXT_DIM,
            justify="left",
            wraplength=360,
        ).pack(anchor="w", padx=10, pady=(0, 8))

        cameras_scroll = ctk.CTkScrollableFrame(cameras_info, fg_color="transparent")
        cameras_scroll.pack(fill="both", expand=True, padx=6, pady=(0, 8))

        for role in self.roles:
            cam_ctrl = ctk.CTkFrame(cameras_scroll, fg_color=PANEL_BG, corner_radius=10)
            cam_ctrl.pack(fill="x", padx=4, pady=4)

            ctk.CTkLabel(
                cam_ctrl,
                text=role.upper(),
                text_color=TEXT_MAIN,
                font=(FONT_FAMILY, 12, "bold"),
            ).grid(row=0, column=0, columnspan=4, sticky="w", padx=8, pady=(8, 4))

            cam_ctrl.grid_columnconfigure(1, weight=1)

            ctk.CTkLabel(cam_ctrl, text="EXP", text_color=TEXT_DIM, font=(FONT_FAMILY, 10)).grid(row=1, column=0,
                                                                                                 sticky="w",
                                                                                                 padx=(8, 6))
            exp_slider = ctk.CTkSlider(
                cam_ctrl,
                from_=100,
                to=20000,
                command=lambda v, r=role: self.on_expo_change(v, r),
                fg_color="#334155",
                progress_color=ACCENT,
                button_color=ACCENT,
                button_hover_color=ACCENT_HOVER,
            )
            exp_slider.grid(row=1, column=1, sticky="ew", padx=(0, 6))
            exp_val = ctk.CTkLabel(cam_ctrl, text="1000 µs", text_color=TEXT_DIM, font=(FONT_FAMILY, 10))
            exp_val.grid(row=1, column=2, sticky="e", padx=(0, 6))
            exp_entry = ctk.CTkEntry(cam_ctrl, width=72, placeholder_text="µs")
            exp_entry.grid(row=1, column=3, sticky="e", padx=(0, 8))
            exp_entry.bind("<Return>", lambda e, r=role, w=exp_entry: self._apply_from_entry(r, "exp", w))
            exp_entry.bind("<KP_Enter>", lambda e, r=role, w=exp_entry: self._apply_from_entry(r, "exp", w))

            ctk.CTkLabel(cam_ctrl, text="GAIN", text_color=TEXT_DIM, font=(FONT_FAMILY, 10)).grid(row=2, column=0,
                                                                                                  sticky="w",
                                                                                                  padx=(8, 6),
                                                                                                  pady=(4, 8))
            gain_slider = ctk.CTkSlider(
                cam_ctrl,
                from_=0,
                to=24,
                command=lambda v, r=role: self.on_gain_change(v, r),
                fg_color="#334155",
                progress_color=ACCENT,
                button_color=ACCENT,
                button_hover_color=ACCENT_HOVER,
            )
            gain_slider.grid(row=2, column=1, sticky="ew", padx=(0, 6), pady=(4, 8))
            gain_val = ctk.CTkLabel(cam_ctrl, text="0.0 dB", text_color=TEXT_DIM, font=(FONT_FAMILY, 10))
            gain_val.grid(row=2, column=2, sticky="e", padx=(0, 6), pady=(4, 8))
            gain_entry = ctk.CTkEntry(cam_ctrl, width=72, placeholder_text="dB")
            gain_entry.grid(row=2, column=3, sticky="e", padx=(0, 8), pady=(4, 8))
            gain_entry.bind("<Return>", lambda e, r=role, w=gain_entry: self._apply_from_entry(r, "gain", w))
            gain_entry.bind("<KP_Enter>", lambda e, r=role, w=gain_entry: self._apply_from_entry(r, "gain", w))

            self.sliders[role] = {
                "exp_slider": exp_slider,
                "exp_val": exp_val,
                "exp_entry": exp_entry,
                "gain_slider": gain_slider,
                "gain_val": gain_val,
                "gain_entry": gain_entry,
            }

        # =========================================================
        # TAB: DIAGNOSTICS
        # =========================================================
        health_card = ctk.CTkFrame(tab_diag, fg_color=CARD_BG, corner_radius=10)
        health_card.pack(fill="x", padx=6, pady=(6, 6))

        ctk.CTkLabel(
            health_card,
            text="Sygnał wydolności systemu",
            text_color=TEXT_MAIN,
            font=(FONT_FAMILY, 13, "bold"),
        ).pack(anchor="w", padx=10, pady=(10, 2))

        self.system_health_var = ctk.StringVar(value="⏳ Oczekiwanie na pomiary...")
        self.system_latency_var = ctk.StringVar(value="Opóźnienie: -- ms | trend: --")

        self.system_health_label = ctk.CTkLabel(
            health_card,
            textvariable=self.system_health_var,
            text_color=TEXT_DIM,
            font=(FONT_FAMILY, 12, "bold"),
            anchor="w",
        )
        self.system_health_label.pack(fill="x", padx=10, pady=(0, 2))

        self.system_latency_label = ctk.CTkLabel(
            health_card,
            textvariable=self.system_latency_var,
            text_color=TEXT_DIM,
            font=(FONT_FAMILY, 11),
            anchor="w",
        )
        self.system_latency_label.pack(fill="x", padx=10, pady=(0, 6))

        self.system_health_bar = ctk.CTkProgressBar(
            health_card,
            fg_color="#334155",
            progress_color="#22c55e",
            height=14,
        )
        self.system_health_bar.pack(fill="x", padx=10, pady=(0, 10))
        self.system_health_bar.set(0.0)

        diag_logs = ctk.CTkFrame(tab_diag, fg_color=CARD_BG, corner_radius=10)
        diag_logs.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        ctk.CTkLabel(
            diag_logs,
            text="Status i feedback",
            text_color=TEXT_MAIN,
            font=(FONT_FAMILY, 13, "bold"),
        ).pack(anchor="w", padx=10, pady=(10, 6))

        self.stats_box = ctk.CTkTextbox(
            diag_logs,
            wrap="word",
            fg_color=PANEL_BG,
            text_color=TEXT_MAIN,
        )
        self.stats_box.pack(fill="both", expand=True, padx=10, pady=(0, 6))
        self.stats_box.insert("end", "✅ System gotowy\n")

        self.session_summary_var = ctk.StringVar(value="Podsumowanie sesji: brak danych")
        ctk.CTkLabel(
            diag_logs,
            textvariable=self.session_summary_var,
            text_color=TEXT_DIM,
            justify="left",
            anchor="w",
        ).pack(fill="x", padx=10, pady=(0, 10))

        # =========================================================
        # FOOTER — secondary actions only
        # =========================================================
        bottom = ctk.CTkFrame(self.main_frame, fg_color=CARD_BG, corner_radius=12)
        bottom.grid(row=2, column=0, sticky="ew", padx=pad, pady=(0, pad))
        bottom.grid_columnconfigure((0, 1, 2, 3), weight=1)

        ctk.CTkButton(
            bottom,
            text="📷 Camera assignment",
            command=self._open_camera_assignment,
        ).grid(row=0, column=0, padx=6, pady=8, sticky="ew")

        ctk.CTkButton(
            bottom,
            text="📊 Benchmark",
            command=self._open_benchmark_window,
        ).grid(row=0, column=1, padx=6, pady=8, sticky="ew")

        ctk.CTkButton(
            bottom,
            text="🛰 Remote",
            command=self.toggle_remote_server,
        ).grid(row=0, column=2, padx=6, pady=8, sticky="ew")

        self.btn_exit = ctk.CTkButton(
            bottom,
            text="❌ Exit",
            fg_color="#DDDDDD",
            hover_color="#CCCCCC",
            text_color="#555555",
            command=self.on_close_all,
        )
        self.btn_exit.grid(row=0, column=3, padx=6, pady=8, sticky="ew")

        # =========================================================
        # TOAST / REFS / RESIZE
        # =========================================================
        self.status_toast = ctk.CTkLabel(
            self.main_frame,
            text="",
            fg_color="#333333",
            text_color="#FFFFFF",
            font=(FONT_FAMILY, 13, "bold"),
            corner_radius=10,
            padx=20,
            pady=10,
        )
        self.status_toast.place_forget()

        self._dashboard_header = header
        self._dashboard_body = body
        self._dashboard_preview_area = preview_area
        self._dashboard_right = right
        self._dashboard_bottom = bottom
        self._dashboard_path_label = self.path_label
        self._dashboard_summary_label = self.session_summary_var

        try:
            self.main_frame.bind("<Configure>", self._on_dashboard_resize_preview_first)
        except Exception:
            pass

        self._refresh_dashboard_metrics()
        self._sync_operator_state_ui()


    def _on_dashboard_resize(self, event=None):
        try:
            total_w = int(self.main_frame.winfo_width())
            if total_w <= 10:
                return

            body_w = max(300, total_w - 24)
            left_w = int(body_w * 0.28)
            center_w = int(body_w * 0.52)
            right_w = max(220, body_w - left_w - center_w)

            if hasattr(self, "_dashboard_body"):
                self._dashboard_body.grid_columnconfigure(0, minsize=left_w)
                self._dashboard_body.grid_columnconfigure(1, minsize=center_w)
                self._dashboard_body.grid_columnconfigure(2, minsize=right_w)

            if hasattr(self, "_dashboard_path_label"):
                self._dashboard_path_label.configure(wraplength=max(160, left_w - 40))

            if hasattr(self, "stats_box"):
                self.stats_box.configure(width=max(180, right_w - 24))

            if hasattr(self, "quick_start_btn"):
                btn_w = max(140, int(total_w * 0.14))
                self.quick_start_btn.configure(width=btn_w)
        except Exception:
            pass

    def _quick_start_session(self):
        session_dir = self.ensure_valid_session_dir()
        if not session_dir:
            self._set_status("Nie udało się przygotować sesji", "danger")
            return
        if not self.preview_on:
            self.toggle_preview()
        self.header_hint_var.set("Sesja aktywna")
        self._set_status("🚀 Session ready", "success")

    def _open_camera_assignment(self):
        try:
            from ui.assign_roles_gui import ask_camera_roles_gui
            choice = ask_camera_roles_gui()
            self._set_status(f"Camera assignment: {choice}", "accent")
        except Exception as e:
            self._set_status(f"Camera assignment error: {e}", "danger")

    def _open_benchmark_window(self):
        if hasattr(self, "_bench_window") and self._bench_window is not None:
            try:
                if self._bench_window.winfo_exists():
                    self._bench_window.lift()
                    self._bench_window.focus_force()
                    return
            except Exception:
                pass

        win = ctk.CTkToplevel(self)
        self._bench_window = win
        win.title("Benchmark")
        win.geometry("980x760")
        win.grid_rowconfigure(1, weight=1)
        win.grid_columnconfigure(0, weight=1)

        top = ctk.CTkFrame(win, fg_color=CARD_BG, corner_radius=10)
        top.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
        top.grid_columnconfigure((0, 1, 2), weight=1)
        ctk.CTkButton(top, text="⚡ Quick Test", command=lambda: self._run_benchmark_preset("quick")).grid(row=0, column=0, sticky="ew", padx=6, pady=8)
        ctk.CTkButton(top, text="📊 Full Benchmark", command=lambda: self._run_benchmark_preset("full")).grid(row=0, column=1, sticky="ew", padx=6, pady=8)
        ctk.CTkButton(top, text="🔧 Advanced", command=lambda: tabs.set("Advanced")).grid(row=0, column=2, sticky="ew", padx=6, pady=8)

        tabs = ctk.CTkTabview(win)
        tabs.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        t_quick = tabs.add("Quick Test")
        t_full = tabs.add("Full Benchmark")
        t_adv = tabs.add("Advanced")

        ctk.CTkLabel(t_quick, text="Szybki test: jeden model, jedna rozdzielczość, krótki warmup.", text_color=TEXT_DIM).pack(anchor="w", padx=10, pady=(10, 6))
        ctk.CTkButton(t_quick, text="Start Quick Test", command=lambda: self._run_benchmark_preset("quick")).pack(fill="x", padx=10, pady=(0, 10))
        ctk.CTkLabel(t_full, text="Pełny benchmark z bezpiecznymi ustawieniami domyślnymi.", text_color=TEXT_DIM).pack(anchor="w", padx=10, pady=(10, 6))
        ctk.CTkButton(t_full, text="Start Full Benchmark", command=lambda: self._run_benchmark_preset("full")).pack(fill="x", padx=10, pady=(0, 10))

        adv = ctk.CTkScrollableFrame(t_adv, fg_color=PANEL_BG)
        adv.pack(fill="both", expand=True, padx=8, pady=8)
        self._build_auto_benchmark_advanced_ui(adv)
        self._update_auto_bench_plan_summary()

    def _run_benchmark_preset(self, mode: str):
        if mode == "quick":
            self.ab_model_source_var.set("Pojedynczy model")
            self.ab_backend_mode_var.set("Ultralytics (.pt/.onnx)")
            self.ab_imgsz_var.set("640")
            self.ab_batch_mode_var.set("Lista")
            self.ab_batch_var.set("4")
            self.ab_warmup_s_var.set("2")
            self.ab_measure_s_var.set("6")
            self.ab_modes_var.set("Brak zapisu")
        else:
            self.ab_model_source_var.set("Wiele modeli (folder)")
            self.ab_backend_mode_var.set("Auto (wg modelu)")
            self.ab_imgsz_var.set("640,960")
            self.ab_batch_mode_var.set("Zakres")
            self.ab_batch_from_var.set("4")
            self.ab_batch_to_var.set("16")
            self.ab_batch_step_var.set("4")
            self.ab_warmup_s_var.set("4")
            self.ab_measure_s_var.set("10")
            self.ab_modes_var.set("Wszystkie kombinacje")
        self._on_auto_bench_model_source_change()
        self._on_auto_bench_batch_mode_change()
        self._on_auto_bench_write_mode_change()
        self.start_auto_benchmark()
        self._set_status(f"Benchmark start ({mode})", "accent")

    def _build_auto_benchmark_advanced_ui(self, parent):
        for w in parent.winfo_children():
            w.destroy()
        ctk.CTkLabel(parent, text="Advanced Benchmark", text_color=TEXT_MAIN, font=(FONT_FAMILY, 13, "bold")).pack(anchor="w", padx=10, pady=(8, 2))
        ctk.CTkLabel(
            parent,
            text="Wybierz zakres testu, backend i parametry. Podsumowanie pokaże dokładnie, co zostanie uruchomione.",
            text_color=TEXT_DIM,
            wraplength=820,
            justify="left",
        ).pack(anchor="w", padx=10, pady=(0, 8))

        scope_card = ctk.CTkFrame(parent, fg_color=CARD_BG, corner_radius=10)
        scope_card.pack(fill="x", padx=8, pady=(0, 8))
        ctk.CTkLabel(scope_card, text="1) Zakres benchmarku", text_color=ACCENT, font=(FONT_FAMILY, 11, "bold")).pack(anchor="w", padx=10, pady=(8, 4))

        row_source = ctk.CTkFrame(scope_card, fg_color="transparent")
        row_source.pack(fill="x", padx=10, pady=(0, 4))
        ctk.CTkLabel(row_source, text="Tryb modeli", text_color=TEXT_DIM).pack(side="left")
        self.ab_model_source_menu = ctk.CTkOptionMenu(
            row_source,
            variable=self.ab_model_source_var,
            values=["Pojedynczy model", "Wiele modeli (folder)"],
            command=self._on_auto_bench_model_source_change,
            width=230,
        )
        self.ab_model_source_menu.pack(side="right")

        row_models = ctk.CTkFrame(scope_card, fg_color="transparent")
        row_models.pack(fill="x", padx=10, pady=(0, 4))
        ctk.CTkLabel(row_models, text="Folder modeli (.pt/.onnx/.engine)", text_color=TEXT_DIM).pack(side="left")
        self.ab_models_dir_btn = ctk.CTkButton(row_models, text="Wybierz", width=90, command=self._choose_auto_bench_models_dir)
        self.ab_models_dir_btn.pack(side="right")
        self.ab_models_dir_entry = ctk.CTkEntry(scope_card, textvariable=self.ab_models_dir_var, height=28, placeholder_text="np. C:/.../models")
        self.ab_models_dir_entry.pack(fill="x", padx=10, pady=(0, 6))

        row_single = ctk.CTkFrame(scope_card, fg_color="transparent")
        row_single.pack(fill="x", padx=10, pady=(0, 4))
        self.ab_single_model_label = ctk.CTkLabel(row_single, text="Pojedynczy model", text_color=TEXT_DIM)
        self.ab_single_model_label.pack(side="left")
        self.ab_single_model_btn = ctk.CTkButton(row_single, text="Wybierz", width=90, command=self._choose_auto_bench_model_file)
        self.ab_single_model_btn.pack(side="right")
        self.ab_single_model_entry = ctk.CTkEntry(scope_card, textvariable=self.ab_single_model_var, height=28, placeholder_text="np. best.pt / model.engine")
        self.ab_single_model_entry.pack(fill="x", padx=10, pady=(0, 8))

        backend_card = ctk.CTkFrame(parent, fg_color=CARD_BG, corner_radius=10)
        backend_card.pack(fill="x", padx=8, pady=(0, 8))
        ctk.CTkLabel(backend_card, text="2) Silnik inferencji", text_color=ACCENT, font=(FONT_FAMILY, 11, "bold")).pack(anchor="w", padx=10, pady=(8, 4))
        row_backend = ctk.CTkFrame(backend_card, fg_color="transparent")
        row_backend.pack(fill="x", padx=10, pady=(0, 4))
        ctk.CTkLabel(row_backend, text="Backend", text_color=TEXT_DIM).pack(side="left")
        self.ab_backend_mode_menu = ctk.CTkOptionMenu(
            row_backend,
            variable=self.ab_backend_mode_var,
            values=["Auto (wg modelu)", "Ultralytics (.pt/.onnx)", "TensorRT (.engine)"],
            command=self._on_auto_bench_backend_mode_change,
            width=230,
        )
        self.ab_backend_mode_menu.pack(side="right")
        self.ab_backend_hint_var = ctk.StringVar(value="")
        ctk.CTkLabel(backend_card, textvariable=self.ab_backend_hint_var, text_color=TEXT_DIM, wraplength=820, justify="left").pack(
            anchor="w", padx=10, pady=(0, 8)
        )

        params_card = ctk.CTkFrame(parent, fg_color=CARD_BG, corner_radius=10)
        params_card.pack(fill="x", padx=8, pady=(0, 8))
        ctk.CTkLabel(params_card, text="3) Parametry testu", text_color=ACCENT, font=(FONT_FAMILY, 11, "bold")).pack(anchor="w", padx=10, pady=(8, 4))

        row_lists = ctk.CTkFrame(params_card, fg_color="transparent")
        row_lists.pack(fill="x", padx=10, pady=(0, 4))
        row_lists.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkLabel(row_lists, text="Rozdzielczość wejścia (imgsz)", text_color=TEXT_DIM).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(row_lists, text="Batch obrazów (lista)", text_color=TEXT_DIM).grid(row=0, column=1, sticky="w")
        self.ab_imgsz_combo = ctk.CTkComboBox(
            row_lists,
            variable=self.ab_imgsz_var,
            values=["320", "416", "512", "640", "736", "832", "960", "1024", "1280", "640,960", "640,832,1024"],
        )
        self.ab_imgsz_combo.grid(row=1, column=0, sticky="ew", padx=(0, 4))
        self.ab_batch_list_combo = ctk.CTkComboBox(
            row_lists,
            variable=self.ab_batch_var,
            values=["2", "4", "8", "12", "16", "20", "24", "28", "32", "4,8,12", "4,8,12,16"],
        )
        self.ab_batch_list_combo.grid(row=1, column=1, sticky="ew", padx=(4, 0))

        row_batch_mode = ctk.CTkFrame(params_card, fg_color="transparent")
        row_batch_mode.pack(fill="x", padx=10, pady=(0, 4))
        ctk.CTkLabel(row_batch_mode, text="Tryb ustawiania batch", text_color=TEXT_DIM).pack(side="left")
        self.ab_batch_mode_menu = ctk.CTkOptionMenu(
            row_batch_mode,
            variable=self.ab_batch_mode_var,
            values=["Zakres", "Lista"],
            command=self._on_auto_bench_batch_mode_change,
            width=180,
        )
        self.ab_batch_mode_menu.pack(side="right")

        row_batch_range = ctk.CTkFrame(params_card, fg_color="transparent")
        row_batch_range.pack(fill="x", padx=10, pady=(0, 4))
        row_batch_range.grid_columnconfigure((0, 1, 2), weight=1)
        ctk.CTkLabel(row_batch_range, text="Od", text_color=TEXT_DIM).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(row_batch_range, text="Do", text_color=TEXT_DIM).grid(row=0, column=1, sticky="w")
        ctk.CTkLabel(row_batch_range, text="Krok", text_color=TEXT_DIM).grid(row=0, column=2, sticky="w")
        batch_values = [str(v) for v in range(1, 65)]
        self.ab_batch_from_combo = ctk.CTkComboBox(row_batch_range, variable=self.ab_batch_from_var, values=batch_values)
        self.ab_batch_to_combo = ctk.CTkComboBox(row_batch_range, variable=self.ab_batch_to_var, values=batch_values)
        self.ab_batch_step_combo = ctk.CTkComboBox(row_batch_range, variable=self.ab_batch_step_var, values=["1", "2", "4", "8"])
        self.ab_batch_from_combo.grid(row=1, column=0, sticky="ew", padx=(0, 4))
        self.ab_batch_to_combo.grid(row=1, column=1, sticky="ew", padx=4)
        self.ab_batch_step_combo.grid(row=1, column=2, sticky="ew", padx=(4, 0))

        row_times = ctk.CTkFrame(params_card, fg_color="transparent")
        row_times.pack(fill="x", padx=10, pady=(2, 8))
        row_times.grid_columnconfigure((0, 1, 2), weight=1)
        ctk.CTkLabel(row_times, text="Warmup [s]", text_color=TEXT_DIM).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(row_times, text="Pomiar [s]", text_color=TEXT_DIM).grid(row=0, column=1, sticky="w")
        ctk.CTkLabel(row_times, text="FPS kamery", text_color=TEXT_DIM).grid(row=0, column=2, sticky="w")
        ctk.CTkEntry(row_times, textvariable=self.ab_warmup_s_var, height=28).grid(row=1, column=0, sticky="ew", padx=(0, 4))
        ctk.CTkEntry(row_times, textvariable=self.ab_measure_s_var, height=28).grid(row=1, column=1, sticky="ew", padx=4)
        self.ab_camera_fps_entry = ctk.CTkEntry(row_times, textvariable=self.ab_camera_fps_var, height=28)
        self.ab_camera_fps_entry.grid(row=1, column=2, sticky="ew", padx=(4, 0))
        self.ab_camera_fps_entry.bind("<Return>", lambda _e: self.on_apply_camera_fps())

        write_card = ctk.CTkFrame(parent, fg_color=CARD_BG, corner_radius=10)
        write_card.pack(fill="x", padx=8, pady=(0, 8))
        ctk.CTkLabel(write_card, text="4) Tryb zapisu danych", text_color=ACCENT, font=(FONT_FAMILY, 11, "bold")).pack(anchor="w", padx=10, pady=(8, 4))
        row_write_mode = ctk.CTkFrame(write_card, fg_color="transparent")
        row_write_mode.pack(fill="x", padx=10, pady=(0, 4))
        ctk.CTkLabel(row_write_mode, text="Co zapisywać", text_color=TEXT_DIM).pack(side="left")
        self.ab_modes_menu = ctk.CTkOptionMenu(
            row_write_mode,
            variable=self.ab_modes_var,
            values=["Wszystkie kombinacje", "Brak zapisu", "Tylko BIN", "Tylko BUFOR", "BIN + BUFOR"],
            command=self._on_auto_bench_write_mode_change,
            width=210,
        )
        self.ab_modes_menu.pack(side="right")

        row_write_params = ctk.CTkFrame(write_card, fg_color="transparent")
        row_write_params.pack(fill="x", padx=10, pady=(0, 8))
        row_write_params.grid_columnconfigure((0, 1, 2), weight=1)
        ctk.CTkLabel(row_write_params, text="Długość BIN [s]", text_color=TEXT_DIM).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(row_write_params, text="Liczba zapisów bufora", text_color=TEXT_DIM).grid(row=0, column=1, sticky="w")
        ctk.CTkLabel(row_write_params, text="Przerwa bufora [s]", text_color=TEXT_DIM).grid(row=0, column=2, sticky="w")
        self.ab_bin_s_entry = ctk.CTkEntry(row_write_params, textvariable=self.ab_bin_s_var, height=28)
        self.ab_buffer_count_entry = ctk.CTkEntry(row_write_params, textvariable=self.ab_buffer_count_var, height=28)
        self.ab_buffer_pause_s_entry = ctk.CTkEntry(row_write_params, textvariable=self.ab_buffer_pause_s_var, height=28)
        self.ab_bin_s_entry.grid(row=1, column=0, sticky="ew", padx=(0, 4))
        self.ab_buffer_count_entry.grid(row=1, column=1, sticky="ew", padx=4)
        self.ab_buffer_pause_s_entry.grid(row=1, column=2, sticky="ew", padx=(4, 0))

        run_card = ctk.CTkFrame(parent, fg_color=CARD_BG, corner_radius=10)
        run_card.pack(fill="x", padx=8, pady=(0, 8))
        ctk.CTkLabel(run_card, text="5) Uruchomienie", text_color=ACCENT, font=(FONT_FAMILY, 11, "bold")).pack(anchor="w", padx=10, pady=(8, 4))
        row_buttons = ctk.CTkFrame(run_card, fg_color="transparent")
        row_buttons.pack(fill="x", padx=10, pady=(0, 6))
        row_buttons.grid_columnconfigure((0, 1), weight=1)
        self.ab_start_btn = ctk.CTkButton(row_buttons, text="▶ Start benchmark", command=self.start_auto_benchmark, height=30)
        self.ab_stop_btn = ctk.CTkButton(row_buttons, text="■ Stop", command=self.stop_auto_benchmark, fg_color=DANGER, hover_color="#B91C1C", height=30)
        self.ab_start_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.ab_stop_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        ctk.CTkLabel(run_card, textvariable=self.ab_status_var, text_color=TEXT_DIM, wraplength=820, justify="left").pack(anchor="w", padx=10, pady=(0, 2))
        ctk.CTkLabel(run_card, textvariable=self.ab_plan_var, text_color=ACCENT, wraplength=820, justify="left").pack(anchor="w", padx=10, pady=(0, 8))

        watch_vars = [
            self.ab_models_dir_var,
            self.ab_single_model_var,
            self.ab_imgsz_var,
            self.ab_batch_var,
            self.ab_batch_from_var,
            self.ab_batch_to_var,
            self.ab_batch_step_var,
            self.ab_modes_var,
            self.ab_camera_fps_var,
            self.ab_backend_mode_var,
        ]
        for _v in watch_vars:
            try:
                _v.trace_add("write", lambda *_args: self._update_auto_bench_plan_summary())
            except Exception:
                pass

        self._on_auto_bench_model_source_change()
        self._on_auto_bench_backend_mode_change()
        self._on_auto_bench_batch_mode_change()
        self._on_auto_bench_write_mode_change()

    def _refresh_dashboard_metrics(self):
        try:
            now = time.time()
            last = float(getattr(self, "_dashboard_last_refresh_ts", 0.0) or 0.0)
            if (now - last) < 1.2:
                return
            self._dashboard_last_refresh_ts = now

            import shutil
            roles_online = sum(1 for r in self.roles if (time.time() - float(self._last_frame_time.get(r, 0.0))) < 1.2)
            cpu_txt = "--"
            gpu_txt = "--"
            try:
                if isinstance(self._last_yolo_stats, dict):
                    cpu_txt = f"{float(self._last_yolo_stats.get('cpu_percent', 0.0) or 0.0):.0f}%"
                    gpu_raw = float(self._last_yolo_stats.get("gpu_util_percent", 0.0) or 0.0)
                    if gpu_raw > 0:
                        gpu_txt = f"{gpu_raw:.0f}%"
            except Exception:
                pass
            self.header_status_var.set(f"Kamery: {roles_online}/{len(self.roles)} online | CPU: {cpu_txt} | GPU: {gpu_txt} | Remote: {'ON' if self.remote_on else 'OFF'}")
            session_txt = str(self.raw_dir or "brak folderu")
            self.header_session_var.set(f"Folder sesji: {session_txt}")
            free_gb = getattr(self, "_dashboard_free_gb_cache", None)
            disk_cache_ts = float(getattr(self, "_dashboard_disk_cache_ts", 0.0) or 0.0)
            if free_gb is None or (now - disk_cache_ts) >= 5.0:
                disk_path = Path(self.raw_dir) if self.raw_dir and os.path.isdir(self.raw_dir) else Path.cwd()
                usage = shutil.disk_usage(str(disk_path))
                free_gb = usage.free / (1024 ** 3)
                self._dashboard_free_gb_cache = free_gb
                self._dashboard_disk_cache_ts = now
            self.header_disk_var.set(f"Wolne miejsce: {free_gb:.1f} GB")
            self.left_disk_var.set(f"Wolne miejsce: {free_gb:.1f} GB")
        except Exception:
            pass

    def _push_latency_sample(self, latency_ms: float):
        try:
            v = float(latency_ms)
        except Exception:
            return
        if not np.isfinite(v) or v < 0:
            return
        self._latency_samples.append((time.time(), v))
        self._refresh_system_health_ui()

    def _refresh_system_health_ui(self):
        if not hasattr(self, "system_health_var"):
            return
        samples = list(self._latency_samples)
        if len(samples) < 3:
            self.system_health_var.set("⏳ Oczekiwanie na pomiary...")
            self.system_latency_var.set("Opóźnienie: -- ms | trend: --")
            try:
                self.system_health_bar.set(0.0)
            except Exception:
                pass
            return

        vals = [float(v) for _t, v in samples]
        cur = vals[-1]
        avg = float(sum(vals) / len(vals))
        first_half = vals[: max(1, len(vals)//2)]
        second_half = vals[max(1, len(vals)//2):]
        trend_ms = float(sum(second_half) / len(second_half) - sum(first_half) / len(first_half))

        if avg < 120 and trend_ms < 8:
            state = "✅ System się wyrabia"
            col = "#22c55e"
            lvl = min(1.0, avg / 250.0)
        elif trend_ms >= 12:
            state = f"⚠ Opóźnienie rośnie (+{trend_ms:.1f} ms)"
            col = "#ef4444"
            lvl = min(1.0, avg / 120.0)
        else:
            state = f"⚠ Stałe opóźnienie (~{avg:.1f} ms)"
            col = "#f59e0b"
            lvl = min(1.0, avg / 300.0)

        self.system_health_var.set(state)
        self.system_latency_var.set(f"Opóźnienie: teraz {cur:.1f} ms | średnio {avg:.1f} ms | trend {trend_ms:+.1f} ms")
        try:
            self.system_health_label.configure(text_color=col)
            self.system_health_bar.configure(progress_color=col)
            self.system_health_bar.set(max(0.03, lvl))
        except Exception:
            pass

        self.trajectory_window = TrajectoryWindow(
            parent=self,
            roles=self.roles,
            yolo_vis_q=self.yolo_vis_q,
            stats_q=self.stats_q,
        )

    def open_trajectory_window(self):
        if getattr(self, "trajectory_window", None) is not None:
            try:
                if self.trajectory_window.winfo_exists():
                    self.trajectory_window.lift()
                    self.trajectory_window.focus_force()
                    return
            except Exception:
                pass

        self.trajectory_window = TrajectoryWindow(
            parent=self,
            roles=self.roles,
            yolo_vis_q=getattr(self, "yolo_vis_q", None),
            stats_q=self.stats_q,
        )

    def on_close(self):
        print("[GUI] 🔴 Zamykam aplikację...")
        self._closing = True
        try:
            self.stop_auto_benchmark()
        except Exception:
            pass
        try:
            if self.trajectory_window is not None and self.trajectory_window.winfo_exists():
                self.trajectory_window.destroy()
        except Exception:
            pass
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
            if self.live_backend_ctrl is not None:
                self.live_backend_ctrl.shutdown(timeout_s=1.5)
        except Exception:
            pass
        try:
            from core.utils_config import stop_evt
            stop_evt.set()
        except Exception as e:
            print(f"[GUI] ⚠️ stop_evt.set() error: {e}")
        try:
            from streaming.server_stream import stop_webrtc_server
            stop_webrtc_server()
        except Exception as e:
            print(f"[GUI] ⚠️ stop_webrtc_server() error: {e}")
        try:
            if self.winfo_exists():
                self.withdraw()
                self.destroy()
        except Exception:
            pass

    def _clamp(self, v, lo, hi):
        try:
            v = float(str(v).replace(",", "."))
        except Exception:
            return None
        return max(lo, min(hi, v))

    def _render_to_label(self, label, frame_rgb, keep_aspect: bool = True):
        if frame_rgb is None:
            return
        now = time.time()
        if not hasattr(label, "_last_query_ts"):
            label._last_query_ts = 0.0
        if (now - label._last_query_ts) > 0.3 or not hasattr(label, "_target_size"):
            tw = max(1, label.winfo_width())
            th = max(1, label.winfo_height())
            label._target_size = (tw, th)
            label._last_query_ts = now

        target_w, target_h = label._target_size
        fh, fw = frame_rgb.shape[:2]
        if keep_aspect:
            scale = min(target_w / max(1, fw), target_h / max(1, fh))
            new_w = max(1, int(fw * scale)); new_h = max(1, int(fh * scale))
        else:
            new_w, new_h = target_w, target_h

        if fw == new_w and fh == new_h:
            pil_img = Image.fromarray(frame_rgb)
        else:
            pil_img = Image.fromarray(frame_rgb).resize((new_w, new_h), Image.BILINEAR)


        img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(new_w, new_h))
        set_ctk_image(label, img)

    def _apply_yolo_point_overlay(self, role: str, frame_rgb: np.ndarray) -> np.ndarray:
        if frame_rgb is None or not isinstance(frame_rgb, np.ndarray):
            return frame_rgb

        snap = self._get_shared_state_value("yolo_last_inference_snapshot", None)
        if not hasattr(snap, "get"):
            return frame_rgb
        det_flat = snap.get("detections_flat", [])
        if not isinstance(det_flat, list) or not det_flat:
            return frame_rgb

        role_norm = str(role).strip().upper()
        latest_step = None
        for d in det_flat:
            if str(d.get("role", "")).strip().upper() != role_norm:
                continue
            sidx = int(d.get("step_idx", -1) or -1)
            if latest_step is None or sidx > latest_step:
                latest_step = sidx
        if latest_step is None:
            return frame_rgb

        try:
            import cv2
            out = frame_rgb.copy()
            h, w = out.shape[:2]
            class_id = int(self._get_shared_state_value("yolo_ball_class_id", 0) or 0)

            for d in det_flat:
                if str(d.get("role", "")).strip().upper() != role_norm:
                    continue
                if int(d.get("step_idx", -1) or -1) != int(latest_step):
                    continue

                fw = max(1, int(d.get("frame_w", w) or w))
                fh = max(1, int(d.get("frame_h", h) or h))
                bx = float(d.get("x", 0.0) or 0.0)
                by = float(d.get("y", 0.0) or 0.0)
                bw = float(d.get("width", 0.0) or 0.0)
                bh = float(d.get("height", 0.0) or 0.0)
                cx = float(d.get("center_x", bx + bw * 0.5) or (bx + bw * 0.5))
                cy = float(d.get("center_y", by + bh * 0.5) or (by + bh * 0.5))
                conf = float(d.get("confidence", 0.0) or 0.0)

                x1 = int(round(bx * w / fw))
                y1 = int(round(by * h / fh))
                x2 = int(round((bx + bw) * w / fw))
                y2 = int(round((by + bh) * h / fh))
                px = int(round(cx * w / fw))
                py = int(round(cy * h / fh))

                if x2 > x1 and y2 > y1:
                    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2, lineType=cv2.LINE_AA)

                if 0 <= px < w and 0 <= py < h:
                    cv2.circle(out, (px, py), 8, (0, 255, 0), -1, lineType=cv2.LINE_AA)
                    cv2.circle(out, (px, py), 10, (255, 255, 255), 2, lineType=cv2.LINE_AA)

                label_txt = f"{class_id} {conf:.2f}"
                label_size = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                lx1 = int(x1 if x2 > x1 else max(0, px - 10))
                ly1 = int(max(0, (y1 if y2 > y1 else py) - label_size[1] - 10))
                lx2 = int(lx1 + label_size[0])
                ly2 = int(y1 if y2 > y1 else py)
                cv2.rectangle(out, (lx1, ly1), (lx2, ly2), (0, 255, 0), -1, lineType=cv2.LINE_AA)
                cv2.putText(
                    out,
                    label_txt,
                    (lx1, max(0, ly2 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
            return out
        except Exception:
            return frame_rgb
    # --------------------------------------------------------
    # Inicjalizacja suwaków/etykiet z JSON
    # --------------------------------------------------------
    def _apply_init_from_config(self):
        cfg = self.initial_settings or {}

        # Ścieżka bazowa zapisu
        p = cfg.get("path2save")
        if p:
            self.raw_dir = p
            if hasattr(self, "path_label"):
                self.path_label.configure(text=p)

        # Per rola
        for r, s in (self.sliders or {}).items():
            exp, gain = self._get_initial_for_role(r)

            if exp is not None and "exp_slider" in s and "exp_val" in s:
                s["exp_slider"].set(float(exp))
                s["exp_val"].configure(text=f"{int(float(exp))} µs")
                if "exp_entry" in s:
                    s["exp_entry"].delete(0, "end")
                    s["exp_entry"].insert(0, f"{int(float(exp))}")

            if gain is not None and "gain_slider" in s and "gain_val" in s:
                s["gain_slider"].set(float(gain))
                s["gain_val"].configure(text=f"{float(gain):.1f} dB")
                if "gain_entry" in s:
                    s["gain_entry"].delete(0, "end")
                    s["gain_entry"].insert(0, f"{float(gain):.1f}")

    def _open_remote_url(self):
        """Otwiera w przeglądarce adres z self.remote_url_var, jeśli jest poprawny."""
        import webbrowser

        url = self.remote_url_var.get() or ""
        url = url.strip()
        if url.lower().startswith("http"):
            try:
                webbrowser.open(url)
                self._set_status(f"Otwieram: {url}", "accent")
            except Exception as e:
                print(f"[GUI] ⚠️ open url error: {e}")
                self._set_status("Nie udało się otworzyć linku.", "danger")
        else:
            self._set_status("Brak aktywnego adresu zdalnego podglądu.", "warning")


    # --------------------------------------------------------
    # 🎛️ Kontrolki
    # --------------------------------------------------------
    def _make_btnold(self, parent, text, cmd, color):
        """
        Tworzy przycisk i automatycznie dobiera pack/grid:
        - jeśli parent ma już dzieci w grid -> używamy grid
        - w przeciwnym wypadku -> pack
        """
        btn = ctk.CTkButton(
            parent,
            text=text,
            fg_color=color,
            hover_color=ACCENT_HOVER,
            text_color="#FFF",
            corner_radius=8,
            height=36,
            command=cmd,
        )

        use_grid = False
        try:
            # jeżeli parent ma jakiekolwiek "grid_slaves", to znaczy że w tym
            # kontenerze używany jest grid
            if parent.grid_slaves():
                use_grid = True
        except Exception:
            use_grid = False

        if use_grid:
            # rozciągamy przycisk na szerokość
            parent.grid_columnconfigure(0, weight=1)
            btn.grid(row=len(parent.grid_slaves()), column=0,
                     sticky="ew", padx=20, pady=6)
        else:
            btn.pack(pady=6, fill="x", padx=20)

        return btn

    # --------------------------------------------------------
    # 🎛️ Wspólny helper do przycisków akcji
    # --------------------------------------------------------
    def _make_btn(self, parent, text, cmd, color, height=30, font_size=11, corner_radius=8):
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


    def ensure_valid_session_dir(self):
        import os
        from pathlib import Path
        global fcL, fCr, fL, fR

        raw_base = getattr(self, "raw_dir", None)
        session_dir = getattr(self, "session_dir", None)

        if not raw_base or not os.path.isdir(raw_base):
            print("[SESSION] ⚠️ Brak poprawnego folderu raw_dir.")
            return None

        # jeśli mamy już sesję w raw_base → użyj jej
        if session_dir and os.path.isdir(session_dir):
            try:
                if Path(session_dir).resolve().is_relative_to(Path(raw_base).resolve()):
                    return session_dir
            except Exception:
                pass

        # brak poprawnej sesji → utwórz nową
        new_session_dir = self.create_new_session()
        fcL = fCr = fL = fR = 0
        return new_session_dir

    def _count_session_files(self):
        """
        Liczy pliki w katalogu bieżącej sesji.
        Aktualizuje globalne fcL, fCr, fL, fR i zwraca dict z licznikami.
        """
        import os
        from pathlib import Path
        global fcL, fCr, fL, fR

        session_dir = getattr(self, "session_dir", None)
        if not session_dir:
            return {}

        base = Path(session_dir)

        def _count(subdir: str) -> int:
            p = base / subdir
            try:
                return len(os.listdir(p))
            except FileNotFoundError:
                return 0
            except NotADirectoryError:
                return 0

        # konkretne cztery, o które pytasz
        fcL = _count("CENTER_L")
        fCr = _count("CENTER_R")
        fL  = _count("LEFT")
        fR  = _count("RIGHT")

        # dodatkowo per rola (jeśli katalog = nazwa roli)
        per_role = {}
        for role in self.roles:
            p = base / role
            try:
                per_role[role] = len(os.listdir(p))
            except (FileNotFoundError, NotADirectoryError):
                per_role[role] = 0

        counts = {
            "CENTER_L": fcL,
            "CENTER_R": fCr,
            "LEFT": fL,
            "RIGHT": fR,
        }
        counts.update(per_role)
        return counts

    def _append_stats_record(self, counts: dict, when: float | None = None):
        """
        Dopisuje pojedynczy rekord statystyk do recording_stats.json w katalogu sesji.
        """
        from pathlib import Path

        if when is None:
            when = time.time()

        session_dir = getattr(self, "session_dir", None) or self.ensure_valid_session_dir()
        if not session_dir:
            print("[REC] ⚠️ Brak session_dir – nie zapisuję recording_stats.")
            return

        stats_path = Path(session_dir) / "recording_stats.json"

        start_ts = getattr(self, "record_start_time", None)
        elapsed = float(when - start_ts) if start_ts is not None else None

        record = {
            "t_unix": when,
            "t_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(when)),
            "elapsed_s": elapsed,
            "files": counts,
        }

        # wczytaj istniejącą listę, jeśli jest
        data = []
        if stats_path.exists():
            try:
                with open(stats_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if not isinstance(data, list):
                        data = []
            except Exception:
                data = []

        data.append(record)

        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"[REC] Zapisano rekord do recording_stats.json (n={len(data)})")

    def _recording_stats_tick(self):
        """
        Co 30 s, jeśli nagrywanie jest włączone:
        - liczy pliki w katalogach sesji
        - dopisuje rekord do recording_stats.json
        - planuje kolejne wywołanie
        """
        if not self.winfo_exists():
            self._stats_timer_running = False
            return

        is_rec = False
        if getattr(self, "recording_event", None):
            try:
                is_rec = self.recording_event.is_set()
            except Exception:
                is_rec = False

        if not is_rec:
            # nagrywanie wyłączone → przestajemy działać
            self._stats_timer_running = False
            return

        counts = self._count_session_files()
        self._append_stats_record(counts)

        # zaplanuj kolejne wywołanie za 30 s
        try:
            if self.recording_event and self.recording_event.is_set():
                self._after(30_000, self._recording_stats_tick)
            else:
                self._stats_timer_running = False
        except Exception:
            self._stats_timer_running = False


    def on_preview_double_click(self, role):
        if self.zoom_q is None:
            self._set_status("Brak kolejki zoom_q.", "danger")
            return
        try:
            self.control_q.put(("set_zoom_role", role))
        except Exception:
            pass

        import cv2
        import queue as _queue
        if self.zoom_q is None:
            self.zoom_q = _queue.Queue(maxsize=8)

        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        max_w = int(screen_w * 0.8); max_h = int(screen_h * 0.8)

        if hasattr(self, "_zoom_windows") and role in getattr(self, "_zoom_windows", {}):
            win = self._zoom_windows[role]
            if win.winfo_exists():
                win.lift(); win.focus_force(); return
        else:
            self._zoom_windows = {}

        zoom_win = ctk.CTkToplevel(self)
        zoom_win.title(f"{role} — Zoom Preview")
        zoom_win.geometry(f"{max_w}x{max_h}")
        zoom_win.lift(); zoom_win.attributes("-topmost", True); zoom_win.focus_force()

        lbl = ctk.CTkLabel(zoom_win, text="⏳ czekam na klatkę...")
        lbl.pack(fill="both", expand=True)
        zoom_win.update_idletasks()
        self._zoom_windows[role] = zoom_win

        zoom_factor = 1.0
        zoom_center = [0.5, 0.5]

        def apply_zoom(new_zoom, cx, cy):
            nonlocal zoom_factor, zoom_center
            new_zoom = max(1.0, min(float(new_zoom), 12.0))
            if new_zoom == zoom_factor:
                return
            scale = new_zoom / zoom_factor
            zoom_center[0] = cx + (zoom_center[0] - cx) / scale
            zoom_center[1] = cy + (zoom_center[1] - cy) / scale
            zoom_center[0] = min(max(zoom_center[0], 0.0), 1.0)
            zoom_center[1] = min(max(zoom_center[1], 0.0), 1.0)
            zoom_factor = new_zoom

        def on_mouse_click(event):
            if lbl.winfo_width() <= 0 or lbl.winfo_height() <= 0:
                return
            zoom_center[0] = event.x / max(1, lbl.winfo_width())
            zoom_center[1] = event.y / max(1, lbl.winfo_height())

        def on_wheel_win(event):
            if lbl.winfo_width() <= 0 or lbl.winfo_height() <= 0:
                return
            cx = event.x / max(1, lbl.winfo_width())
            cy = event.y / max(1, lbl.winfo_height())
            step = 0.12 if getattr(event, "delta", 0) > 0 else -0.12
            apply_zoom(zoom_factor + step, cx, cy)

        def on_wheel_up(event):
            cx = event.x / max(1, lbl.winfo_width())
            cy = event.y / max(1, lbl.winfo_height())
            apply_zoom(zoom_factor + 0.12, cx, cy)

        def on_wheel_down(event):
            cx = event.x / max(1, lbl.winfo_width())
            cy = event.y / max(1, lbl.winfo_height())
            apply_zoom(zoom_factor - 0.12, cx, cy)

        lbl.bind("<Button-1>", on_mouse_click)
        lbl.bind("<MouseWheel>", on_wheel_win)  # Win/macOS
        lbl.bind("<Button-4>", on_wheel_up)    # Linux
        lbl.bind("<Button-5>", on_wheel_down)  # Linux

        def reset_zoom(_e=None):
            nonlocal zoom_factor, zoom_center
            zoom_factor = 1.0; zoom_center[:] = [0.5, 0.5]
        lbl.bind("<Double-Button-1>", reset_zoom)

        def render_zoom(frame_rgb):
            try:
                import cv2
                h, w = frame_rgb.shape[:2]
                cx = int(zoom_center[0] * w); cy = int(zoom_center[1] * h)
                half_w = max(1, int(w / (2.0 * zoom_factor)))
                half_h = max(1, int(h / (2.0 * zoom_factor)))
                x1, y1 = max(0, cx - half_w), max(0, cy - half_h)
                x2, y2 = min(w, cx + half_w), min(h, cy + half_h)
                if x2 <= x1 or y2 <= y1:
                    return None
                crop = frame_rgb[y1:y2, x1:x2]
                tw, th = max(2, lbl.winfo_width()), max(2, lbl.winfo_height())
                if crop.shape[1] != tw or crop.shape[0] != th:
                    crop = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)
                im = Image.fromarray(crop)
                return ctk.CTkImage(light_image=im, size=(tw, th))
            except Exception as e:
                print(f"[ZOOM] render_zoom error: {e}")
                return None

        zoom_after = {"id": None}

        def update_loop():
            if not zoom_win.winfo_exists() or not lbl.winfo_exists():
                return
            frame_rgb = None
            try:
                q = self.zoom_q; item = None
                while q is not None and not q.empty():
                    item = q.get_nowait()
                if item is not None:
                    try:
                        role2, payload, _ts = item
                    except Exception:
                        role2, _ts, payload = item
                    if role2 == role and payload is not None:
                        if isinstance(payload, str):
                            got = self.smm.read_frame(payload)
                            if got is not None:
                                frame_rgb, _ = got if isinstance(got, tuple) else (got, 0)
                        elif isinstance(payload, np.ndarray):
                            frame_rgb = payload
            except Exception as e:
                print(f"[ZOOM] queue read error: {e}")
            if frame_rgb is None:
                frame_rgb = self.fullres_frames.get(role)
            if frame_rgb is not None:
                img_ctk = render_zoom(frame_rgb)

                if img_ctk:
                    set_ctk_image(lbl, img_ctk)

            else:
                lbl.configure(text="⏳ brak klatek do zoomu (kolejka pusta)")
            zoom_after["id"] = zoom_win.after(16, update_loop)

        def on_zoom_close():
            aid = zoom_after.get("id")
            if aid:
                try:
                    zoom_win.after_cancel(aid)
                except Exception:
                    pass
                zoom_after["id"] = None
            try:
                if hasattr(self, "_zoom_windows"):
                    self._zoom_windows.pop(role, None)
            except Exception:
                pass
            try:
                if zoom_win.winfo_exists():
                    zoom_win.destroy()
            except Exception:
                pass

        zoom_win.protocol("WM_DELETE_WINDOW", on_zoom_close)
        update_loop()

    def on_start_record(self):
        if not self.raw_dir or not os.path.isdir(self.raw_dir):
            folder = filedialog.askdirectory(title="Wybierz folder do zapisu sesji")
            if not folder:
                self._set_status("❌ Nie wybrano folderu!", "danger")
                return
            self.raw_dir = folder

        session_dir = self.ensure_valid_session_dir()
        if not session_dir:
            print("[GUI] ❌ Nie można rozpocząć nagrywania — brak poprawnej ścieżki.")
            return

        import core.utils_config as utils_config
        utils_config.RAW_DIR = session_dir
        print(f"[GUI] 🎥 Aktywna sesja: {session_dir}")

        if self.control_q:
            try:
                self.control_q.put(("record", True))
            except Exception as e:
                print(f"[GUI] ⚠️ Nie udało się wysłać komendy START: {e}")
                return
        else:
            print("[GUI] ⚠️ control_q nie jest ustawione!")
            return

        self.record_start_time = time.time()
        self._set_status(f"⏺ Nagrywanie aktywne → {session_dir}", "accent")

        if not getattr(self, "_stats_timer_running", False):
            self._stats_timer_running = True
            self._recording_stats_tick()
        self.recording = True
        self._record_paused = False
        self._set_status(f"⏺ Nagrywanie aktywne → {session_dir}", "accent")

    def on_pause_record(self):
        if not getattr(self, "recording", False):
            self._set_status("Pause jest dostępne tylko podczas nagrywania", "warning")
            return

        paused = bool(getattr(self, "_record_paused", False))
        if paused:
            try:
                if self.control_q:
                    self.control_q.put(("record", True))
            except Exception:
                pass
            self._record_paused = False
            try:
                self.btn_pause.configure(text="⏸ Pause", fg_color="#f59e0b")
            except Exception:
                pass
            self._set_status("▶ Wznowiono nagrywanie", "success")
        else:
            try:
                if self.control_q:
                    self.control_q.put(("record", False))
            except Exception:
                pass
            self._record_paused = True
            try:
                self.btn_pause.configure(text="▶ Resume", fg_color="#0891b2")
            except Exception:
                pass
            self._set_status("⏸ Nagrywanie wstrzymane", "warning")

    def on_stop_record(self):
        if self.control_q:
            try:
                self.control_q.put(("record", False))
            except Exception as e:
                print(f"[GUI] ⚠️ Nie udało się wysłać komendy STOP: {e}")
        else:
            print("[GUI] ⚠️ control_q nie jest ustawione!")

        # Ostatni pomiar i statystyki
        counts = self._count_session_files()
        self._append_stats_record(counts)

        if self.record_start_time is not None:
            duration = time.time() - self.record_start_time
            print("liczba plików (koniec)", counts)
            print("czas nagrania [s]", duration)
            self.record_start_time = None

        self._stats_timer_running = False
        self.recording = False
        self._record_paused = False
        try:
            self.btn_pause.configure(text="⏸ Pause", fg_color="#f59e0b")
        except Exception:
            pass
        self._set_status("⏹ Nagrywanie zatrzymane", "accent")
        print("[GUI] ⏹ Stop recording (wysłano do backendu)")



    def on_take_shot(self):
        from pathlib import Path
        import cv2, time, queue

        # poproś backend o snapshot
        self.control_q.put(("snapshot", True))

        session_root = self.ensure_valid_session_dir()
        if not session_root:
            print("[SNAPSHOT] ❌ Nie można zapisać — brak poprawnej sesji.")
            return

        snap_dir = Path(session_root) / "snapshots"
        snap_dir.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        role_dirs = {r: (snap_dir / r) for r in self.roles}
        for d in role_dirs.values():
            d.mkdir(exist_ok=True)

        print(f"[SNAPSHOT] 📸 Oczekiwanie na klatki do {snap_dir}")

        frames_written = 0
        saved_roles = set()
        timeout_s = 2.5
        start_time = time.time()

        while len(saved_roles) < len(self.roles) and (time.time() - start_time) < timeout_s:
            try:
                item = self.snapshot_q.get(timeout=0.5)
            except queue.Empty:
                continue

            if not isinstance(item, (tuple, list)) or len(item) < 3:
                print(f"[SNAPSHOT] ⚠️ Zły format elementu snapshot_q: {item}")
                continue

            role = item[0]

            # Obsługa obu wariantów:
            # (role, frame, ts_ns) oraz (role, ts_ns, frame)
            if isinstance(item[1], np.ndarray):
                frame = item[1]
                ts_ns = item[2]
            elif isinstance(item[2], np.ndarray):
                ts_ns = item[1]
                frame = item[2]
            else:
                print(f"[SNAPSHOT] ⚠️ Nie znaleziono klatki ndarray w elemencie: {item}")
                continue

            if frame is None or role in saved_roles:
                continue

            if not isinstance(frame, np.ndarray):
                print(f"[SNAPSHOT] ⚠️ frame nie jest ndarray dla {role}: {type(frame)}")
                continue

            # --- przygotuj obraz BGR ---
            img_bgr = None

            if frame.ndim == 2:
                # surowy Bayer RG8 -> BGR
                try:
                    img_bgr = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)
                except Exception as e:
                    print(f"[SNAPSHOT] ⚠️ Bayer→BGR failed for {role}: {e}")
                    continue

            elif frame.ndim == 3 and frame.shape[2] == 3:
                # klatka już kolorowa
                img_bgr = frame

            else:
                print(f"[SNAPSHOT] ⚠️ Nieobsługiwany kształt klatki {role}: {getattr(frame, 'shape', '?')}")
                continue

            path = role_dirs[role] / f"{role}_{timestamp}.png"
            cv2.imwrite(str(path), img_bgr)
            print(f"[SNAPSHOT] ✅ zapisano {path}")
            frames_written += 1
            saved_roles.add(role)

        if frames_written == 0:
            print("[SNAPSHOT] ⚠️ Brak ramek w snapshot_q — jeszcze nie dotarły.")
        else:
            print(f"[SNAPSHOT] Zapisano {frames_written}/{len(self.roles)} kamer.")

        if hasattr(self, "snapshot_event") and self.snapshot_event:
            try:
                self.snapshot_event.clear()
            except Exception:
                pass
            try:
                if hasattr(self.snapshot_event, "sent_once"):
                    self.snapshot_event.sent_once.clear()
            except Exception:
                pass

    def on_save_buffer(self):
        """
        🎬 ZAPISUJE BUFOR - ostatnie 10 sekund z wszystkich kamer
        """
        try:
            self.btn_save_buffer.configure(state="disabled", text="💾 Zapisuję...")
            self._set_status("💾 Zapisuję bieżący bufor...", "accent")
            
            # Wyślij komendę do backendu
            self.control_q.put(("save_buffer", True))
            
            print("[GUI] 🎬 Wysłano żądanie zapisu bufora")
            
        except Exception as e:
            print(f"[GUI] ❌ Błąd zapisu bufora: {e}")
            self._set_status("❌ Błąd zapisu bufora", "danger")
            
        finally:
            # Przywróć przycisk po 3 sekundach
            self._after(3000, lambda: self.btn_save_buffer.configure(
                state="normal", text="💾 Zapisz bufor"
            ))


    def on_generate_report(self):
        """
        Generuje raport stabilności zapisu dla bieżącej sesji
        (recording_stats.json w folderze sesji) i otwiera PNG.
        """
        from pathlib import Path
        import subprocess, sys, os, webbrowser

        session_dir = getattr(self, "session_dir", None)
        if not session_dir or not os.path.isdir(session_dir):
            print("[REPORT] Brak poprawnego folderu sesji:", session_dir)
            self._set_status("Brak sesji – najpierw coś nagraj.", "warning")
            return

        session_dir = Path(session_dir)
        stats_path = session_dir / "recording_stats.json"
        if not stats_path.exists():
            print("[REPORT] Nie ma recording_stats.json:", stats_path)
            self._set_status("Brak recording_stats.json dla tej sesji.", "warning")
            return

        # zakładamy, że recording_stats.py leży obok gui.py / main.py
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

            self._set_status("📊 Raport stabilności wygenerowany.", "success")
        else:
            print("[REPORT] Nie znaleziono:", out_png)
            self._set_status("Raport nie został wygenerowany (brak PNG).", "warning")


    def create_new_session(self):
        from pathlib import Path
        import time as _t
        base_dir = Path(self.raw_dir or os.environ.get("RAW_DIR", "") or str(Path.home()))
        if not base_dir.exists():
            base_dir = Path.home()
        session_root = base_dir / "sessions"
        session_root.mkdir(parents=True, exist_ok=True)
        timestamp = _t.strftime("%Y%m%d_%H%M%S")
        session_dir = session_root / f"session_{timestamp}"
        session_dir.mkdir(parents=True, exist_ok=True)
        self.session_dir = str(session_dir)
        print(f"[SESSION] 🆕 Utworzono nową sesję: {session_dir}")
        return session_dir

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

    @staticmethod
    def _normalize_auto_bench_source_mode(raw_mode: str) -> str:
        text = str(raw_mode or "").strip().lower()
        if ("jeden" in text) or ("single" in text) or ("pojedynczy" in text):
            return "single"
        return "folder"

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def _fps_to_period_us(fps: float) -> float:
        fps = max(1.0, float(fps))
        return 1_000_000.0 / fps

    @staticmethod
    def _safe_percentile(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        try:
            return float(np.percentile(np.asarray(values, dtype=float), q))
        except Exception:
            arr = sorted(float(v) for v in values)
            idx = max(0, min(len(arr) - 1, int(round((q / 100.0) * (len(arr) - 1)))))
            return float(arr[idx])

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    def _apply_from_entry(self, role: str, kind: str, entry):
        try:
            txt = str(entry.get()).strip().replace(",", ".")
            if not txt:
                return
            val = float(txt)
        except Exception:
            return

        if kind == "exp":
            val = max(100.0, min(20000.0, val))
            s = self.sliders.get(role, {})
            if s.get("exp_slider"):
                s["exp_slider"].set(val)
            self.on_expo_change(val, role)
        elif kind == "gain":
            val = max(0.0, min(24.0, val))
            s = self.sliders.get(role, {})
            if s.get("gain_slider"):
                s["gain_slider"].set(val)
            self.on_gain_change(val, role)

    def on_expo_change(self, value, role):
        # 🔒 nie wysyłaj komend do backendu podczas inicjalizacji widgetów
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
            self._set_status(f"☀️ {role}: {float(value):.0f} µs", "accent")
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
            self._set_status(f"🎚 {role}: {float(value):.1f} dB", "accent")
        except Exception as e:
            print(f"[GUI][ERROR] on_gain_change({role}) → {e}")

    # --------------------------------------------------------
    # 🖥️ Preview i pętle GUI
    # --------------------------------------------------------
    def toggle_preview(self):
        self.preview_on = not self.preview_on

        self._sync_operator_state_ui()

        if self.preview_on:
            self._set_status("🎥 Podgląd aktywny", "success")
        else:
            self._set_status("🛑 Podgląd wyłączony", "danger")

        if getattr(self, "_initing", False):
            return

        try:
            self.control_q.put(("stream", self.preview_on))
        except Exception as e:
            print(f"[GUI] ⚠️ Error toggling preview: {e}")

    def _sync_operator_state_ui(self):
        """
        Synchronizuje teksty / kolory / hinty w całym operator GUI:
        - header
        - preview
        - recording
        - remote
        - YOLO
        """
        try:
            # ---------------------------
            # PREVIEW
            # ---------------------------
            preview_on = bool(getattr(self, "preview_on", False))
            if hasattr(self, "preview_btn"):
                if preview_on:
                    self.preview_btn.configure(
                        text="⏸ Wyłącz podgląd",
                        fg_color="#16a34a",
                        hover_color="#15803d",
                    )
                else:
                    self.preview_btn.configure(
                        text="▶ Włącz podgląd",
                        fg_color=ACCENT,
                        hover_color=ACCENT_HOVER,
                    )

            if hasattr(self, "preview_status"):
                self.preview_status.configure(
                    text="Podgląd aktywny" if preview_on else "Podgląd wyłączony",
                    text_color=SUCCESS if preview_on else TEXT_DIM,
                )

            # ---------------------------
            # RECORDING
            # ---------------------------
            recording_on = False
            try:
                if hasattr(self, "recording_event") and self.recording_event is not None:
                    recording_on = bool(self.recording_event.is_set())
                else:
                    recording_on = bool(getattr(self, "recording", False))
            except Exception:
                recording_on = bool(getattr(self, "recording", False))

            paused = bool(getattr(self, "_record_paused", False))

            if hasattr(self, "rec_text"):
                if recording_on and paused:
                    self.rec_text.configure(text="Nagrywanie: PAUZA", text_color="#f59e0b")
                elif recording_on:
                    self.rec_text.configure(text="Nagrywanie włączone", text_color="#E74C3C")
                else:
                    self.rec_text.configure(text="Nagrywanie wyłączone", text_color=TEXT_DIM)

            # ---------------------------
            # REMOTE
            # ---------------------------
            remote_on = bool(getattr(self, "remote_on", False))
            remote_url = ""
            try:
                remote_url = str(self.remote_url_var.get() or "")
            except Exception:
                remote_url = ""

            remote_live = remote_on or ("http" in remote_url.lower())

            if hasattr(self, "remote_btn"):
                if remote_live:
                    self.remote_btn.configure(
                        text="🛑 Zatrzymaj remote",
                        fg_color="#7c3aed",
                        hover_color="#6d28d9",
                    )
                else:
                    self.remote_btn.configure(
                        text="🛰 Remote access",
                        fg_color=ACCENT,
                        hover_color=ACCENT_HOVER,
                    )

            if hasattr(self, "remote_status"):
                if remote_live:
                    self.remote_status.configure(text="Server ONLINE ✅", text_color=SUCCESS)
                else:
                    self.remote_status.configure(text="Server wyłączony", text_color=TEXT_DIM)

            # ---------------------------
            # YOLO
            # ---------------------------
            yolo_on = bool(getattr(self, "yolo_on", False))

            if hasattr(self, "yolo_btn"):
                if yolo_on:
                    self.yolo_btn.configure(
                        text="🛑 Detekcja ON",
                        fg_color=DANGER,
                        hover_color="#b91c1c",
                    )
                else:
                    self.yolo_btn.configure(
                        text="🎯 Detekcja OFF",
                        fg_color="#FF6B35",
                        hover_color="#E55A2B",
                    )

            if hasattr(self, "yolo_status"):
                self.yolo_status.configure(
                    text="Detekcja aktywna ✅" if yolo_on else "Detekcja wyłączona",
                    text_color=SUCCESS if yolo_on else TEXT_DIM,
                )

            # opcjonalny drugi przycisk z tab_ai Standard, jeśli istnieje osobno
            if hasattr(self, "yolo_toggle_btn"):
                if yolo_on:
                    self.yolo_toggle_btn.configure(
                        text="🛑 Disable detection",
                        fg_color=DANGER,
                        hover_color="#b91c1c",
                    )
                else:
                    self.yolo_toggle_btn.configure(
                        text="🎯 Enable detection",
                        fg_color="#FF6B35",
                        hover_color="#E55A2B",
                    )

            # ---------------------------
            # HEADER HINT
            # ---------------------------
            hint = "Gotowy do startu sesji"
            if recording_on and paused:
                hint = "Sesja nagrywana — pauza"
            elif recording_on:
                hint = "Sesja nagrywana"
            elif preview_on and yolo_on:
                hint = "Podgląd + detekcja aktywne"
            elif preview_on:
                hint = "Podgląd aktywny"
            elif yolo_on:
                hint = "Detekcja aktywna"
            elif remote_live:
                hint = "Remote aktywny"

            if hasattr(self, "header_hint_var"):
                self.header_hint_var.set(hint)

            # ---------------------------
            # HEADER STATUS
            # ---------------------------
            roles_online = 0
            try:
                now = time.time()
                roles_online = sum(
                    1 for r in self.roles
                    if (now - float(self._last_frame_time.get(r, 0.0))) < 1.2
                )
            except Exception:
                roles_online = 0

            cpu_txt = "--"
            gpu_txt = "--"
            try:
                if isinstance(getattr(self, "_last_yolo_stats", None), dict):
                    cpu_txt = f"{float(self._last_yolo_stats.get('cpu_percent', 0.0) or 0.0):.0f}%"
                    gpu_raw = float(self._last_yolo_stats.get("gpu_util_percent", 0.0) or 0.0)
                    if gpu_raw > 0:
                        gpu_txt = f"{gpu_raw:.0f}%"
            except Exception:
                pass

            if hasattr(self, "header_status_var"):
                self.header_status_var.set(
                    f"Kamery: {roles_online}/{len(self.roles)} | "
                    f"Preview: {'ON' if preview_on else 'OFF'} | "
                    f"YOLO: {'ON' if yolo_on else 'OFF'} | "
                    f"CPU: {cpu_txt} | GPU: {gpu_txt} | "
                    f"Remote: {'ON' if remote_live else 'OFF'}"
                )

        except Exception as e:
            print(f"[GUI] _sync_operator_state_ui error: {e}")

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

    def toggle_remote_server(self):
        self.remote_on = not self.remote_on

        if self.remote_on:
            try:
                self.remote_status.configure(text="Server uruchamiany...", text_color="#555")
            except Exception:
                pass
            self._set_status("🛰 Uruchamianie remote access...", "accent")
        else:
            try:
                self._remote_shown = False
                self.remote_url_var.set("Remote: OFF")
                self.remote_status.configure(text="Server zatrzymywany...", text_color="#555")
            except Exception:
                pass
            self._set_status("🛑 Zatrzymywanie remote access...", "warning")

        self._sync_operator_state_ui()

        if getattr(self, "_initing", False):
            return

        try:
            self.control_q.put(("remote_stream", {"enabled": self.remote_on}))
            print(f"[GUI] remote_stream enabled={self.remote_on}")
        except Exception as e:
            print(f"[GUI] ⚠️ Error toggling remote server: {e}")

    def _handle_remote_started(self, msg):
        self._insert_stat_text(f"🛰 {msg}\n")
        self._set_status("🛰 Serwer gotowy – kliknij link w statystykach", "success")

    def on_choose_raw(self):
        folder = filedialog.askdirectory(title="Wybierz folder główny do zapisu RAW")
        if folder:
            self.raw_dir = folder
            self.path_label.configure(text=folder)
            self._set_status(f"Wybrano RAW DIR: {folder}", "accent")
            import core.utils_config as utils_config
            utils_config.RAW_DIR = folder
            os.environ["RAW_DIR"] = folder
            self.control_q.put(("set_path2save", folder))
            if hasattr(utils_config, "save_user_settings"):
                utils_config.save_user_settings()

    # --------------------------------------------------------
    # Zamknięcie
    # --------------------------------------------------------
    def on_close_all(self):
        print("[GUI] ❌ Zamykam system...")
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
            print("[GUI] 🛑 stop_evt.set() → zatrzymuję backend")
        except Exception as e:
            print(f"[GUI][⚠️] stop_evt.set() error: {e}")
        try:
            from streaming.server_stream import stop_webrtc_server
            stop_webrtc_server()
            print("[GUI] 📴 stop_webrtc_server() wywołane")
        except Exception as e:
            print(f"[GUI][⚠️] stop_webrtc_server() error: {e}")
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
            print("[GUI] 💀 Wszystkie procesy potomne zakończone.")
        except Exception as e:
            print(f"[GUI][⚠️] cleanup error: {e}")



    # --------------------------------------------------------
    # Render i statystyki
    # --------------------------------------------------------
    def update_frames(self):
        if self._closing or not self.winfo_exists():
            return
        if self.live_q is None:
            if not hasattr(self, "_live_q_missing_warned"):
                self._live_q_missing_warned = False
            if not self._live_q_missing_warned:
                self._live_q_missing_warned = True
                print("[GUI] live_q is None - preview disabled")
            self._after(500, self.update_frames)
            return

        frames_for_role = {}
        try:
            while True:
                item = self.live_q.get_nowait()
                if not (isinstance(item, (tuple, list)) and len(item) >= 2):
                    continue
                role = item[0]
                payload = item[1]
                ts_ns = item[2] if len(item) >= 3 else 0
                frames_for_role[role] = (payload, ts_ns)
        except queue.Empty:
            pass
        except Exception as e:
            print(f"[GUI] live_q error: {e}")

        for role, (payload, ts_ns) in frames_for_role.items():
            frame = None
            try:
                if isinstance(payload, str):
                    got = self.smm.read_frame(payload)
                    if got is None:
                        continue
                    if isinstance(got, tuple):
                        frame, _meta = got
                    else:
                        frame = got
                elif isinstance(payload, np.ndarray):
                    frame = payload
                else:
                    continue
            except Exception as e:
                print(f"[GUI] decode error for {role}: {e}")
                continue

            if frame is not None:
                self.display_frame(role, frame)

        now = time.time()
        timeout = 0.8
        for role, lbl in self.camera_labels.items():
            last = self._last_frame_time.get(role, 0.0)
            if last == 0.0:
                continue
            if (now - last) > timeout:
                try:
                    lbl.configure(
                        text=f"{role}\nOczekiwanie na obraz...",
                        image=None,
                        fg_color=LIVE_BG,
                    )
                    lbl.image = None
                except Exception:
                    pass
                self.fullres_frames[role] = None
                self._last_frame_time[role] = 0.0

        target_fps = 12.0
        interval_ms = int(1000 / target_fps)
        self._after(interval_ms, self.update_frames)

    def display_frame(self, role, frame):
        try:
            if frame is None or not isinstance(frame, np.ndarray):
                return
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                return

            label = self.camera_labels.get(role) or self.img_labels.get(role)
            if not label:
                print(f"[GUI] ❌ No label for role {role}")
                return

            render_frame = self._apply_yolo_point_overlay(role, frame)

            color_order = str(getattr(self, "preview_input_color_order", "bgr")).strip().lower()
            if color_order == "rgb":
                render_frame_rgb = np.ascontiguousarray(render_frame)
            else:
                render_frame_rgb = np.ascontiguousarray(render_frame[:, :, ::-1])

            self._render_to_label(label, render_frame_rgb)
            self._last_frame_time[role] = time.time()
            self.fullres_frames[role] = frame

        except Exception as e:
            print(f"[GUI] ❌ Error displaying frame for {role}: {e}")

    # ---------------------- podgląd z kolejki ----------------------
    def toggle_preview_color_order(self):
        cur = str(getattr(self, "preview_input_color_order", "bgr")).lower()
        self.preview_input_color_order = "rgb" if cur == "bgr" else "bgr"
        print(f"[GUI] preview_input_color_order = {self.preview_input_color_order}")
        try:
            self._set_status(f"Preview color order: {self.preview_input_color_order.upper()}", "accent")
        except Exception:
            pass

    def _update_efficiency_bars(self):
        for role, msg in self._save_stats_buffer.items():
            saved = msg.get("saved", 0)
            expected = msg.get("expected", 1)
            backlog = msg.get("backlog", 0)
            ratio = saved / expected if expected > 0 else 0.0
            percent = int(ratio * 100)
            bar = self.eff_bars.get(role)
            lbl = self.eff_labels.get(role)
            if bar and lbl:
                bar.set(ratio)
                lbl.configure(text=f"{percent}% ({saved}/{expected}) b:{backlog}")
        if self.winfo_exists():
            self._after(3000, self._update_efficiency_bars)


    def _append_perf_point(self, images_per_sec: float, avg_infer_ms: float, marker_label: str | None = None):
        try:
            idx = int(self._perf_next_idx)
            self._perf_next_idx = idx + 1
            self._perf_history.append({
                "i": idx,
                "ips": max(0.0, float(images_per_sec)),
                "ms": max(0.0, float(avg_infer_ms)),
            })
            if marker_label:
                self._perf_markers.append({"i": idx, "label": str(marker_label)})
            self._draw_perf_chart()
        except Exception:
            pass

    def _draw_perf_chart(self):
        canvas = getattr(self, "yolo_perf_canvas", None)
        if canvas is None:
            return
        try:
            w = max(120, int(canvas.winfo_width()))
            h = max(60, int(canvas.winfo_height()))
            canvas.delete("all")

            pad_l, pad_r, pad_t, pad_b = 8, 8, 8, 18
            x0, y0 = pad_l, pad_t
            x1, y1 = w - pad_r, h - pad_b
            pw = max(2, x1 - x0)
            ph = max(2, y1 - y0)

            canvas.create_rectangle(x0, y0, x1, y1, outline="#4B5563", width=1)

            n = len(self._perf_history)
            if n < 2:
                canvas.create_text(
                    w // 2,
                    h // 2,
                    text="czekam na dane...",
                    fill="#9CA3AF",
                    font=("Segoe UI", 9),
                )
                return

            hist = list(self._perf_history)[-n:]
            vals_ips = [float(it.get("ips", 0.0)) for it in hist]
            vals_ms = [float(it.get("ms", 0.0)) for it in hist]
            idxs = [int(it.get("i", i)) for i, it in enumerate(hist)]
            vmax = max(max(vals_ips), max(vals_ms), 1.0)
            vmin = min(min(vals_ips), min(vals_ms), 0.0)
            if abs(vmax - vmin) < 1e-6:
                vmax = vmin + 1.0

            def map_xy(i: int, v: float):
                x = x0 + (i / (n - 1)) * pw
                y = y1 - ((v - vmin) / (vmax - vmin)) * ph
                return x, y

            pts_ips = []
            pts_ms = []
            for i in range(n):
                xi, yi = map_xy(i, vals_ips[i])
                xm, ym = map_xy(i, vals_ms[i])
                pts_ips.extend((xi, yi))
                pts_ms.extend((xm, ym))

            x_map = {idxs[i]: (x0 + (i / (n - 1)) * pw) for i in range(n)}
            for mk in list(self._perf_markers):
                mi = int(mk.get("i", -1))
                mx = x_map.get(mi)
                if mx is None:
                    continue
                canvas.create_line(mx, y0, mx, y1, fill="#6B7280", width=1, dash=(2, 2))
                canvas.create_text(
                    mx + 2,
                    y0 + 2,
                    text=str(mk.get("label", ""))[:12],
                    fill="#9CA3AF",
                    anchor="nw",
                    font=("Segoe UI", 7),
                )

            canvas.create_line(*pts_ips, fill="#22C55E", width=2, smooth=True)
            canvas.create_line(*pts_ms, fill="#F59E0B", width=2, smooth=True)

            canvas.create_text(
                x0,
                h - 8,
                text=f"img/s {vals_ips[-1]:.0f}",
                fill="#22C55E",
                anchor="w",
                font=("Segoe UI", 8, "bold"),
            )
            canvas.create_text(
                x1,
                h - 8,
                text=f"ms/b {vals_ms[-1]:.1f}",
                fill="#F59E0B",
                anchor="e",
                font=("Segoe UI", 8, "bold"),
            )
        except Exception:
            pass

    def _refresh_perf_chart(self):
        if self._closing or not self.winfo_exists():
            return
        self._draw_perf_chart()
        self._after(500, self._refresh_perf_chart)



    def _update_stats(self):
        try:
            while not self.stats_q.empty():
                msg = self.stats_q.get_nowait()

                if isinstance(msg, str):
                    self._insert_stat_text(msg + "\n")
                    if "Remote Control UI available at:" in msg:
                        self._set_status("🛰 Serwer gotowy – kliknij link w statystykach", "success")
                    continue

                if isinstance(msg, dict):
                    msg_type = msg.get("type")
                    if msg_type is None and ("avg_infer_ms" in msg or "images_per_sec" in msg):
                        msg_type = "yolo_summary"
                    if msg_type is None and ("pending_buckets" in msg or "dropped_incomplete" in msg):
                        msg_type = "yolo_router_summary"

                    if msg_type == "save_status":
                        self._save_stats_buffer[msg["role"]] = msg
                        continue

                    elif msg_type == "yolo_summary":
                        try:
                            self._last_yolo_stats = dict(msg)
                            avg_time = float(msg.get("avg_infer_ms", 0) or 0)
                            e2e_ms = float(msg.get("e2e_pipeline_ms", 0) or 0)
                            if e2e_ms > 0:
                                self._push_latency_sample(e2e_ms)
                            elif avg_time > 0:
                                self._push_latency_sample(avg_time)
                            batches_per_sec = float(msg.get("batches_per_sec", 0) or 0)
                            images_per_sec = float(msg.get("images_per_sec", 0) or 0)
                            steps_per_sec = float(msg.get("steps_per_sec", 0) or 0)
                            target_steps_per_sec = float(msg.get("target_steps_per_sec", 50.0) or 50.0)
                            target_util_percent = float(msg.get("target_util_percent", 0) or 0)
                            configured_input_batch_images = int(msg.get("configured_input_batch_images", 0) or 0)

                            queue_wait_ms = float(msg.get("queue_wait_ms", 0) or 0)
                            batch_collect_ms = float(msg.get("batch_collect_ms", 0) or 0)
                            prepare_ms = float(msg.get("prepare_ms", 0) or 0)
                            forward_ms = float(msg.get("forward_ms", 0) or 0)
                            post_ms = float(msg.get("post_ms", 0) or 0)

                            last_steps_in_batch = int(msg.get("last_steps_in_batch", 0) or 0)
                            last_images_in_batch = int(msg.get("last_images_in_batch", 0) or 0)
                            last_batch_size_effective = int(msg.get("last_batch_size_effective", 0) or 0)
                            current_cfg = (
                                str(self.shared_state.get("yolo_backend", "ultralytics")) if self.shared_state is not None else "ultralytics",
                                int(self.shared_state.get("yolo_image_size", 0) or 0) if self.shared_state is not None else 0,
                                int(configured_input_batch_images),
                                int(last_steps_in_batch),
                            )
                            marker_label = None
                            if self._last_perf_cfg is not None and current_cfg != self._last_perf_cfg:
                                marker_label = f"{current_cfg[0]} i{current_cfg[1]} b{current_cfg[2]} s{current_cfg[3]}"
                            self._last_perf_cfg = current_cfg
                            self._append_perf_point(
                                images_per_sec=images_per_sec,
                                avg_infer_ms=avg_time,
                                marker_label=marker_label,
                            )

                            if hasattr(self, "yolo_perf_label"):
                                if avg_time > 0:
                                    perf_text = (
                                        f"YOLO: {avg_time:.1f} ms/batch | "
                                        f"Q:{queue_wait_ms:.0f} ms | "
                                        f"{steps_per_sec:.1f} kroków/s | "
                                        f"{images_per_sec:.0f} obrazów/s | "
                                        f"wejście: {configured_input_batch_images} img | "
                                        f"ostatnio: {last_images_in_batch} img"
                                    )

                                    color = (
                                        "#22c55e" if target_util_percent >= 90 else
                                        "#f59e0b" if target_util_percent >= 60 else
                                        "#ef4444"
                                    )

                                    self.yolo_perf_label.configure(
                                        text=perf_text,
                                        text_color=color,
                                    )

                                    if hasattr(self, "yolo_extra_stats_var"):
                                        self.yolo_extra_stats_var.set(
                                            f"collect: {batch_collect_ms:.1f} ms | prep: {prepare_ms:.1f} ms | "
                                            f"fwd: {forward_ms:.1f} ms | post: {post_ms:.1f} ms | b/s: {batches_per_sec:.2f}"
                                        )
                        except Exception:
                            pass
                    elif msg_type == "yolo_router_summary":
                        try:
                            self._last_router_stats = dict(msg)
                        except Exception:
                            pass
                    elif msg_type == "relay_stats":
                        continue

            try:
                self._poll_live_track_events()
            except Exception:
                pass

            try:
                self._refresh_dashboard_metrics()
            except Exception:
                pass

            try:
                backend = self._normalize_preprocess_backend(
                    self._get_shared_state_value("yolo_preprocess_backend", self.yolo_preprocess_backend)
                )
                if backend != self.yolo_preprocess_backend:
                    self.yolo_preprocess_backend = backend
                    self._apply_yolo_preprocess_backend_ui()
            except Exception:
                pass

            if self.winfo_exists():
                self._after(1000, self._update_stats)

        except Exception as e:
            print(f"[GUI][ERROR] _update_stats: {e}")
            if self.winfo_exists():
                self._after(500, self._update_stats)

    def _rec_indicator_tick(self):
        try:
            is_on = False
            if hasattr(self, "recording_event") and self.recording_event:
                try:
                    is_on = self.recording_event.is_set()
                except Exception:
                    is_on = bool(getattr(self, "recording", False))
            else:
                is_on = bool(getattr(self, "recording", False))
            if not hasattr(self, "_rec_blink"):
                self._rec_blink = False
            if is_on:
                self._rec_blink = not self._rec_blink
                led_color = "#E74C3C" if self._rec_blink else "#FFC7C2"
                self.rec_led.configure(text="●", text_color=led_color)
                paused = bool(getattr(self, "_record_paused", False))
                if paused:
                    self.rec_text.configure(text="Nagrywanie: PAUZA", text_color="#f59e0b")
                else:
                    self.rec_text.configure(text="Nagrywanie włączone", text_color="#E74C3C")
                interval = 450
            else:
                self.rec_led.configure(text="●", text_color="#B0B0B0")
                self.rec_text.configure(text="Nagrywanie wyłączone", text_color=TEXT_DIM)
                interval = 900

            if hasattr(self, "record_start_time") and self.record_start_time is not None:
                elapsed = max(0, int(time.time() - float(self.record_start_time)))
                hh = elapsed // 3600
                mm = (elapsed % 3600) // 60
                ss = elapsed % 60
                if hasattr(self, "session_elapsed_var"):
                    self.session_elapsed_var.set(f"Czas sesji: {hh:02d}:{mm:02d}:{ss:02d}")
            elif hasattr(self, "session_elapsed_var"):
                self.session_elapsed_var.set("Czas sesji: 00:00:00")

            if hasattr(self, "buffer_fill_var"):
                try:
                    cfg_sec = int(float(self.buffer_sec_value.cget("text") or 5))
                except Exception:
                    cfg_sec = 5
                self.buffer_fill_var.set(f"Bufor: {cfg_sec:.1f} / {cfg_sec:.1f} s")

            self._refresh_dashboard_metrics()
            self._sync_operator_state_ui()
        except Exception:
            interval = 900
        if self.winfo_exists():
            self._after(interval, self._rec_indicator_tick)

    def _insert_stat_text(self, text: str):
        text = _fix_mojibake_text(text)
        if not hasattr(self, "stats_box"):
            print("[STAT]", text); return
        url_match = re.search(r"https?://[^\s]+", text)
        if url_match:
            url = url_match.group(0)
            before, after = text.split(url, 1)
            self.stats_box.insert("end", before)
            start_index = self.stats_box.index("end-1c")
            self.stats_box.insert("end", url)
            end_index = self.stats_box.index("end-1c")
            self.stats_box.tag_add("link", start_index, end_index)
            self.stats_box.tag_config("link", foreground="#007BFF", underline=True)
            self.stats_box.tag_bind("link", "<Button-1>", lambda e, u=url: webbrowser.open(u))
            self.stats_box.tag_bind("link", "<Enter>", lambda e: self.stats_box.configure(cursor="hand2"))
            self.stats_box.tag_bind("link", "<Leave>", lambda e: self.stats_box.configure(cursor=""))
            self.stats_box.insert("end", after)
        else:
            self.stats_box.insert("end", text)
        self.stats_box.see("end")

    def _set_status(self, msg, color):
        colors = {
            "success": "#2ECC71",
            "danger": "#E74C3C",
            "warning": "#F1C40F",
            "accent": "#3498DB",
            "neutral": "#333333"
        }
        bg = colors.get(color, "#333333")
        self.status_toast.configure(text=msg, fg_color=bg)
        self.status_toast.place(relx=0.5, rely=1.05, anchor="s")
        try:
            self._insert_stat_text(f"{msg}\n")
        except Exception:
            pass
        try:
            if hasattr(self, "session_summary_var"):
                self.session_summary_var.set(f"Ostatnia akcja: {msg}")
        except Exception:
            pass

        def animate_up(step=0):
            if step <= 10:
                y = 1.05 - (step * 0.01)
                self.status_toast.place(relx=0.5, rely=y, anchor="s")
                if not self.winfo_exists():
                    return
                self._after(15, lambda: animate_up(step + 1))
            else:
                if not self.winfo_exists():
                    return
                self._after(300, fade_out)

        def fade_out(step=0):
            if step <= 10:
                y = 0.95 + (step * 0.01)
                self.status_toast.place(relx=0.5, rely=y, anchor="s")
                if not self.winfo_exists():
                    return
                self._after(15, lambda: fade_out(step + 1))
            else:
                self.status_toast.place_forget()

        animate_up()

class TrajectoryWindow(ctk.CTkToplevel):
    def __init__(self, parent, roles, yolo_vis_q, history_len=10, stats_q=None):
        self.stats_q = stats_q
        super().__init__(parent)

        self.title("YOLO trajectories")
        self.geometry("900x700")
        self.configure(fg_color=BG_LIGHT)

        self.roles = list(roles)
        self.yolo_vis_q = yolo_vis_q
        self.history_len = history_len
        self._closing = False
        self._after_id = None

        self.history = {
            role: deque(maxlen=history_len) for role in self.roles
        }

        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.axes = self.fig.subplots(2, 2)
        self.axes = self.axes.flatten()


        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._schedule_update()

    def _schedule_update(self, delay_ms=1000):
        if self._closing or not self.winfo_exists():
            return
        self._after_id = self.after(delay_ms, self._update_plot)

    def _on_close(self):
        self._closing = True
        if self._after_id:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        try:
            self.destroy()
        except Exception:
            pass

    def _consume_queue(self):
        if self.yolo_vis_q is None:
            return

        while True:
            try:
                item = self.yolo_vis_q.get_nowait()
            except Exception:
                break

            if not isinstance(item, dict):
                continue

            batch_id = item.get("batch_id")
            per_role_points = item.get("per_role_points", {})

            for role in self.roles:
                pts = per_role_points.get(role, [])
                self.history[role].append({
                    "batch_id": batch_id,
                    "points": pts,
                })

    def _update_plot(self):
        if self._closing or not self.winfo_exists():
            return

        self._consume_queue()

        cmap = [
            "#CBD5E1", "#94A3B8", "#64748B", "#475569", "#334155",
            "#1E293B", "#0F172A", "#2563EB", "#16A34A", "#DC2626"
        ]

        default_w = 4096
        default_h = 3000
        print(f'GUI UPDATE PLOT przed {time.time()}')

        for ax, role in zip(self.axes, self.roles):
            ax.clear()
            ax.set_title(role, fontsize=10)
            ax.grid(True, alpha=0.2)

            batches = list(self.history[role])
            if not batches:
                ax.set_xlim(0, default_w)
                ax.set_ylim(default_h, 0)   # odwrócona Y
                continue

            all_x = []
            all_y = []

            for i, batch in enumerate(batches):
                color = cmap[min(i, len(cmap) - 1)]
                pts = batch.get("points", [])

                if not pts:
                    continue

                xs = [p["x"] for p in pts]
                ys = [p["y"] for p in pts]

                all_x.extend(xs)
                all_y.extend(ys)

                # starsze batch'e cieńsze, nowsze wyraźniejsze
                lw = 0.8 if i < len(batches) - 1 else 1.8

                ax.plot(xs, ys, linewidth=lw, color=color)

                for j, p in enumerate(pts):
                    size = 12 + 8 * j
                    ax.scatter(p["x"], p["y"], s=size, color=color)

            if all_x and all_y:
                xmin, xmax = min(all_x), max(all_x)
                ymin, ymax = min(all_y), max(all_y)

                pad_x = max(10, int((xmax - xmin) * 0.2) if xmax > xmin else 20)
                pad_y = max(10, int((ymax - ymin) * 0.2) if ymax > ymin else 20)

                ax.set_xlim(xmin - pad_x, xmax + pad_x)
                ax.set_ylim(ymax + pad_y, ymin - pad_y)  # odwrócona Y
            else:
                ax.set_xlim(0, default_w)
                ax.set_ylim(default_h, 0)

            # podpis najnowszego batcha
            try:
                last_batch_id = batches[-1].get("batch_id")
                ax.text(
                    0.02, 0.98, f"batch {last_batch_id}",
                    transform=ax.transAxes,
                    fontsize=8,
                    va="top"
                )
            except Exception:
                pass

        self.fig.tight_layout()
        self.canvas.draw_idle()
        print(f'GUI UPDATE PLOT po {time.time()}')
        self._schedule_update(1000)
