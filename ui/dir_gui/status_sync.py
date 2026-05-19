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
        "Âµ": "µ",
        "Â": "",
        "đź": "",
        "???": "[OK]",
        "âťŚ": "[ERR]",
        "⚠️": "[WARN]",
        "â„ąď¸Ź": "[INFO]",
        "â–¶": "[START]",
        "âŹą": "[STOP]",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s

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
    if getattr(self, "_closing", False):
        return
    try:
        if not self.winfo_exists():
            return
    except Exception:
        return
    if not hasattr(self, "status_toast") or self.status_toast is None:
        return
    try:
        if not self.status_toast.winfo_exists():
            return
    except Exception:
        return
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
