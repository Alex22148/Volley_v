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
