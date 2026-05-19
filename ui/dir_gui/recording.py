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

import numpy as np
from tkinter import filedialog


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
