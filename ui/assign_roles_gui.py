import os, json
import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2, numpy as np
from typing import Dict, Optional


try:
    from pypylon import pylon
except Exception:
    pylon = None

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

ROLES = ["CENTER_L", "CENTER_R", "LEFT", "RIGHT"]

import os, sys

if getattr(sys, 'frozen', False):
    # š€ Aplikacja dziaĹ‚a jako .exe (PyInstaller)
    # Folder z plikiem EXE
    EXE_DIR = os.path.dirname(sys.executable)
    # Folder tymczasowy, w ktĂłrym PyInstaller rozpakowuje zasoby (np. layout.png)
    BASE_DIR = sys._MEIPASS
else:
    # Ť Tryb deweloperski (.py)
    EXE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    BASE_DIR = EXE_DIR

# ”ą JSON zapisujemy obok .exe
DEFAULT_JSON_PATH = os.path.join(EXE_DIR, "camera_roles.json")
print(DEFAULT_JSON_PATH)

# ”ą layout.png wczytujemy z zasobĂłw (czyli z folderu doĹ‚Ä...czonego w PyInstaller)
LAYOUT_PATH = os.path.join(BASE_DIR, "images//layout.png")

ROLE_POS_1280x720 = {
    "LEFT": (211, 219),
    "CENTER_L": (437, 600),
    "CENTER_R": (843, 600),
    "RIGHT": (1084, 450),
}


MARKER_RADIUS = 14


def load_layout_image():
    if os.path.exists(LAYOUT_PATH):
        try:
            return Image.open(LAYOUT_PATH).convert("RGB")
        except Exception:
            pass
    img = np.full((720, 1280, 3), 235, np.uint8)
    cv2.putText(img, "Brak layout.png", (50, 360),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
    return Image.fromarray(img)


def scale_role_positions(img_w, img_h):
    ref_w, ref_h = 1280, 720
    sx, sy = img_w / ref_w, img_h / ref_h
    return {r: (int(x * sx), int(y * sy)) for r, (x, y) in ROLE_POS_1280x720.items()}


def scan_cameras():
    if pylon is None:
        return []
    try:
        tl = pylon.TlFactory.GetInstance()
        devs = tl.EnumerateDevices()
        return [d.GetSerialNumber() for d in devs]
    except Exception:
        return []


def load_existing_mapping(path):
    if not os.path.exists(path):
        return {}
    try:
        data = json.load(open(path, "r", encoding="utf-8"))
        return {item["role"]: item["serial"] for item in data if "role" in item and "serial" in item}
    except Exception:
        return {}


ROLES = ["CENTER_L", "CENTER_R", "LEFT", "RIGHT"]

def save_mapping(path: str, role_to_serial: Dict[str, Optional[str]]) -> None:
    """Zapisuje w formacie listy obiektĂłw {serial, role} z zachowaniem kolejnoĹ›ci ROLES."""
    out = []
    for role in ROLES:
        serial = role_to_serial.get(role)
        if serial:
            out.append({"serial": str(serial), "role": role})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] Saved mapping to {path} (roles in canonical order)")
    try:
        import core.utils_config as utils_config
        if hasattr(utils_config, "stats_q"):
            utils_config.stats_q.put({"type": "save_ok", "filename": os.path.basename(path)})
    except Exception:
        pass


class AssignRolesApp(ctk.CTk):
    def __init__(self, json_path=DEFAULT_JSON_PATH):
        super().__init__()
        self.title("VolleyHub przypisywanie kamer")
        #self.geometry("1100x760")
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"{sw}x{sh}")
        self.json_path = json_path
        self.serials = []
        self.role_to_serial = {r: None for r in ROLES}
        self.active_role = None

        self.base_img = load_layout_image()
        self._build_ui()
        self._load_existing_mapping()
        self._refresh_serials()

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # --- GĂ“RNY PANEL ---
        top = ctk.CTkFrame(self, fg_color="#FFFFFF", corner_radius=12)
        top.grid(row=0, column=0, sticky="ew", padx=15, pady=(10, 4))
        for i in range(len(ROLES)):
            top.grid_columnconfigure(i, weight=1)

        self.dropdowns = {}
        for i, role in enumerate(ROLES):
            frame = ctk.CTkFrame(top, fg_color="#F8F9FB", corner_radius=10)
            frame.grid(row=0, column=i, padx=6, pady=6, sticky="ew")

            ctk.CTkLabel(frame, text=role, font=("Segoe UI", 13, "bold"),
                         text_color="#3498DB").pack(pady=(6, 2))

            dd = ctk.CTkOptionMenu(frame, values=['brak'],
                                   command=lambda v, r=role: self._on_change(r, v))
            dd.pack(padx=10, pady=(2, 6), fill="x")
            self.dropdowns[role] = dd

            btn = ctk.CTkButton(frame, text="Pokaż", height=28,
                                fg_color="#E8E9EE", text_color="#333",
                                hover_color="#D0D0D0",
                                command=lambda r=role: self._highlight_role(r))
            btn.pack(padx=10, pady=(0, 8), fill="x")

        # --- MAPA ---
        map_frame = ctk.CTkFrame(self, fg_color="#FFFFFF", corner_radius=12)
        map_frame.grid(row=1, column=0, sticky="nsew", padx=15, pady=(4, 10))
        map_frame.grid_columnconfigure(0, weight=1)
        map_frame.grid_rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(map_frame, bg="#F0F2F5", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.canvas.bind("<Configure>", lambda e: self._redraw_map())
        self.canvas.bind("<Button-1>", self._on_click)

        # --- PRZYCISKI DOLNE ---
        bottom = ctk.CTkFrame(self, fg_color="#FFFFFF", corner_radius=12)
        bottom.grid(row=2, column=0, sticky="ew", padx=15, pady=(0, 10))
        bottom.grid_columnconfigure((0, 1, 2, 3), weight=1)

        ctk.CTkButton(bottom, text="Odśwież kamery", command=self._refresh_serials).grid(row=0, column=0, padx=6, pady=10, sticky="ew")
        ctk.CTkButton(bottom, text="Auto-przypisz", command=self._auto_assign).grid(row=0, column=1, padx=6, pady=10, sticky="ew")
        ctk.CTkButton(bottom, text="Zapisz", fg_color="#2ECC71", hover_color="#27AE60",
                      command=self._save).grid(row=0, column=2, padx=6, pady=10, sticky="ew")
        ctk.CTkButton(bottom, text="ZZamknij", fg_color="#CCCCCC", hover_color="#BBBBBB",
                      command=self.destroy).grid(row=0, column=3, padx=6, pady=10, sticky="ew")

    # === LOGIKA ===
    def _load_existing_mapping(self):
        mapping = load_existing_mapping(self.json_path)
        for r in ROLES:
            self.role_to_serial[r] = mapping.get(r)
        self._refresh_dropdowns()

    def _refresh_serials(self):
        self.serials = scan_cameras()
        print("[DEBUG] detected serials:", self.serials)
        print("[DEBUG] loaded mapping:", self.role_to_serial)
        if not self.serials:
            messagebox.showwarning("Kamery", "Nie znaleziono kamer.")
        self._refresh_dropdowns()

    def _refresh_dropdowns(self):
        extra_serials = []
        for role in ROLES:
            s = self.role_to_serial.get(role)
            if s and s not in self.serials:
                extra_serials.append(s)

        values = ["brak"] + self.serials + extra_serials

        seen = set()
        values = [x for x in values if not (x in seen or seen.add(x))]

        for role, dd in self.dropdowns.items():
            dd.configure(values=values)
            current = self.role_to_serial.get(role)
            dd.set(current if current in values else "brak")

        self._redraw_map()

    def _on_change(self, role, value):
        self.role_to_serial[role] = None if value == "” brak ”" else value
        self.active_role = role
        self._redraw_map()

    def _highlight_role(self, role):
        self.active_role = role
        self._redraw_map()

    def _auto_assign(self):
        if not self.serials:
            self._refresh_serials()
        for i, role in enumerate(ROLES):
            self.role_to_serial[role] = self.serials[i] if i < len(self.serials) else None
        self._refresh_dropdowns()
        self.active_role = ROLES[0] if ROLES else None
        self._redraw_map()

    def _save(self):
        used = [s for s in self.role_to_serial.values() if s]
        if len(set(used)) != len(used):
            messagebox.showerror("Błąd", "Ten sam numer seryjny przypisano więcej niż jednej roli.")
            return
        save_mapping(self.json_path, self.role_to_serial)
        messagebox.showinfo("Zapisano", f"Zapisano do:\n{self.json_path}")

    # === MAPA ===
    def _redraw_map(self):
        self.canvas.delete("all")
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 50 or ch < 50:
            return

        img = self.base_img.copy()
        iw, ih = img.size

        # Skalowanie "contain" + obliczenie offsetu
        scale = min(cw / iw, ch / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        ox, oy = (cw - nw) // 2, (ch - nh) // 2

        img = img.resize((nw, nh), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(ox, oy, image=self._tk_img, anchor="nw")

        # Przeskaluj markery + uwzglÄ™dnij offset
        scaled = scale_role_positions(nw, nh)
        r = max(8, int(MARKER_RADIUS * min(nw / 1280, nh / 720)))

        for role, (x, y) in scaled.items():
            X, Y = ox + x, oy + y  # <” kluczowy moment: dodajemy offsety
            assigned = bool(self.role_to_serial.get(role))
            active = (role == self.active_role)
            color = "#2ECC71" if active else ("#888888" if assigned else "#DDDDDD")
            self.canvas.create_oval(X - r, Y - r, X + r, Y + r, fill=color, outline="#333", width=2)
            self.canvas.create_text(X, Y - (r + 12), text=role, fill="#222", font=("Segoe UI", 10, "bold"))

    def _on_click(self, e):
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        iw, ih = self.base_img.size
        scale = min(cw / iw, ch / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        ox, oy = (cw - nw) // 2, (ch - nh) // 2
        scaled_pos = scale_role_positions(nw, nh)
        best, best_d = None, 1e9
        for role, (x, y) in scaled_pos.items():
            X, Y = ox + x, oy + y
            d = (X - e.x) ** 2 + (Y - e.y) ** 2
            if d < best_d:
                best_d, best = d, role
        if best:
            self.active_role = best
            self._redraw_map()

import customtkinter as ctk

def ask_camera_roles_gui() -> str:
    """
    Nowoczesne CTk okno z fade-in i fade-out.
    Zwraca:
        "assign"   â†’ otwĂłrz przypisywanie kamer
        "existing" â†’ użyj istniejÄ...cego pliku
        "cancel"   â†’ zakoĹ„cz program
    """
    popup = ctk.CTk()
    popup.title("VolleyHub ˘ Camera Roles Setup")
    popup.geometry("440x260")
    popup.resizable(False, False)
    popup.configure(fg_color="#F7F8FA")
    popup.attributes("-alpha", 0.0)

    # === FADE-IN ===
    def fade_in():
        try:
            alpha = popup.attributes("-alpha")
            if alpha < 1.0:
                popup.attributes("-alpha", min(1.0, alpha + 0.05))
                popup.after(25, fade_in)
        except Exception:
            pass

    popup.after(100, fade_in)

    # === FADE-OUT ===
    def fade_out_and_close():
        try:
            alpha = popup.attributes("-alpha")
            if alpha > 0:
                popup.attributes("-alpha", max(0.0, alpha - 0.08))
                popup.after(25, fade_out_and_close)
            else:
                popup.destroy()
        except Exception:
            try:
                popup.destroy()
            except Exception:
                pass

    # === UI ===
    ctk.CTkLabel(
        popup,
        text="· Camera Roles Configuration",
        text_color="#3498DB",
        font=("Segoe UI", 18, "bold")
    ).pack(pady=(24, 8))

    ctk.CTkLabel(
        popup,
        text="Detected cameras.\nDo you want to assign roles now or use existing configuration?",
        text_color="#444",
        font=("Segoe UI", 13),
        justify="center"
    ).pack(pady=(0, 22))

    result = {"choice": None}

    def do_assign():
        result["choice"] = "assign"
        fade_out_and_close()

    def do_existing():
        result["choice"] = "existing"
        fade_out_and_close()

    def do_cancel():
        result["choice"] = "cancel"
        fade_out_and_close()

    btn_frame = ctk.CTkFrame(popup, fg_color="#F7F8FA")
    btn_frame.pack(pady=(0, 12), fill="x", expand=True)

    ctk.CTkButton(
        btn_frame,
        text="Assign now",
        fg_color="#3498DB",
        hover_color="#2E86DE",
        text_color="white",
        corner_radius=10,
        height=38,
        command=do_assign
    ).pack(pady=4, padx=40, fill="x")

    ctk.CTkButton(
        btn_frame,
        text="Use existing",
        fg_color="#AAAAAA",
        hover_color="#999999",
        text_color="white",
        corner_radius=10,
        height=38,
        command=do_existing
    ).pack(pady=4, padx=40, fill="x")

    ctk.CTkButton(
        btn_frame,
        text="”´ Cancel",
        fg_color="#E74C3C",
        hover_color="#C0392B",
        text_color="white",
        corner_radius=10,
        height=38,
        command=do_cancel
    ).pack(pady=4, padx=40, fill="x")

    popup.mainloop()
    return result["choice"]

# ===========================================================
# === FUNKCJA URUCHAMIAJÄ„CA GUI Z MAIN.PY ===================
# ===========================================================

# ===========================================================
# === FUNKCJA URUCHAMIAJÄ„CA Z MAIN.PY =======================
# ===========================================================

def run_assign_gui():
    """
    Uruchamia okno przypisywania ról kamer (Assign GUI)
    bez otwierania nowego procesu. DziaĹ‚a zarĂłwno w .py, jak i .exe.
    """
    import tkinter as tk

    print("[ASSIGN] ”Uruchamianie okna przypisywania ról kamer (inline)...")

    try:
        # ”ą Uruchom CTk popup z wyborami (Assign / Existing / Cancel)
        choice = ask_camera_roles_gui()
    except Exception as e:
        print(f"[ASSIGN]  Błąd podczas uruchamiania assign GUI: {e}")
        choice = "cancel"

    # §ą Zamknij domyĹ›lnego roota (jeśli zostaĹ‚)
    try:
        root = tk._default_root
        if root:
            root.update_idletasks()
            root.destroy()
    except Exception:
        pass

    print(f"[ASSIGN] Użytkownik wybrał: {choice}")
    return choice




