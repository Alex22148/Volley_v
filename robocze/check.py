# save as: visual_review_4cams_export.py
# pip install ultralytics pillow opencv-python

from __future__ import annotations
import threading
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def list_images(folder: Path) -> list[Path]:
    files = []
    for ext in IMG_EXTS:
        files.extend(folder.rglob(f"*{ext}"))
    files.sort()
    return files


def draw_detections_on_bgr(model: YOLO, img_bgr, conf: float, imgsz: int, cls_id: int | None):
    res = model.predict(img_bgr, conf=conf, imgsz=imgsz, verbose=False)[0]

    if cls_id is not None and res.boxes is not None and len(res.boxes) > 0:
        keep = (res.boxes.cls.int() == int(cls_id))
        res.boxes = res.boxes[keep]

    plotted_bgr = res.plot()  # BGR ndarray
    return plotted_bgr


def fit_to_box(img_rgb, max_w: int, max_h: int):
    h, w = img_rgb.shape[:2]
    scale = min(max_w / w, max_h / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    return resized


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Wizualna ocena detekcji piłki — 4 foldery + eksport")
        self.geometry("1400x950")

        # state
        self.root_dir: Path | None = None
        self.cam_dirs: list[Path] = []
        self.cam_images: list[list[Path]] = [[], [], [], []]
        self.index = 0
        self.step = tk.IntVar(value=1)

        self.model_path: Path | None = None
        self.model: YOLO | None = None

        self.conf = tk.DoubleVar(value=0.25)
        self.imgsz = tk.IntVar(value=640)
        self.cls_id = tk.StringVar(value="0")  # ball class id; empty = all

        self.status_text = tk.StringVar(value="Wybierz folder główny i model .pt")
        self.export_text = tk.StringVar(value="Eksport: —")

        # ratings
        self.ratings = {}  # key: (cam_idx, index) -> str
        self.ratings_path: Path | None = None

        # export range
        self.export_mode = tk.StringVar(value="current")  # "current" or "range"
        self.export_from = tk.StringVar(value="0")
        self.export_to = tk.StringVar(value="0")
        self.export_out_dir: Path | None = None
        self._export_running = False

        # UI
        self._build_ui()
        self._bind_keys()

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=8)

        ttk.Button(top, text="Wybierz folder główny (4 podfoldery)", command=self.pick_root).pack(side="left")
        ttk.Button(top, text="Wybierz model .pt", command=self.pick_model).pack(side="left", padx=(8, 0))

        ttk.Label(top, text="conf").pack(side="left", padx=(16, 4))
        ttk.Entry(top, textvariable=self.conf, width=6).pack(side="left")
        ttk.Label(top, text="imgsz").pack(side="left", padx=(10, 4))
        ttk.Entry(top, textvariable=self.imgsz, width=6).pack(side="left")

        ttk.Label(top, text="class_id (piłka)").pack(side="left", padx=(10, 4))
        ttk.Entry(top, textvariable=self.cls_id, width=6).pack(side="left")

        ttk.Label(top, text="krok").pack(side="left", padx=(16, 4))
        ttk.Spinbox(top, from_=1, to=999, textvariable=self.step, width=5).pack(side="left")

        ttk.Button(top, text="⟲ Odśwież bieżące", command=self.refresh_current).pack(side="right")

        nav = ttk.Frame(self)
        nav.pack(fill="x", padx=10, pady=(0, 8))

        ttk.Button(nav, text="◀ Poprzedni", command=self.prev).pack(side="left")
        ttk.Button(nav, text="Następny ▶", command=self.next).pack(side="left", padx=(8, 0))

        self.jump_var = tk.StringVar(value="0")
        ttk.Label(nav, text="Idź do indeksu").pack(side="left", padx=(16, 4))
        ttk.Entry(nav, textvariable=self.jump_var, width=8).pack(side="left")
        ttk.Button(nav, text="Skocz", command=self.jump).pack(side="left", padx=(6, 0))

        self.counter_lbl = ttk.Label(nav, text="—")
        self.counter_lbl.pack(side="right")

        ttk.Label(self, textvariable=self.status_text).pack(fill="x", padx=10, pady=(0, 4))
        ttk.Label(self, textvariable=self.export_text).pack(fill="x", padx=10, pady=(0, 8))

        # Export controls
        exp = ttk.LabelFrame(self, text="Eksport obrazów z naniesioną inferencją")
        exp.pack(fill="x", padx=10, pady=(0, 10))

        ttk.Radiobutton(exp, text="Bieżący indeks", variable=self.export_mode, value="current").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Radiobutton(exp, text="Zakres", variable=self.export_mode, value="range").grid(row=0, column=1, sticky="w", padx=8, pady=6)

        ttk.Label(exp, text="od").grid(row=0, column=2, sticky="e")
        ttk.Entry(exp, textvariable=self.export_from, width=8).grid(row=0, column=3, sticky="w", padx=(4, 12))

        ttk.Label(exp, text="do").grid(row=0, column=4, sticky="e")
        ttk.Entry(exp, textvariable=self.export_to, width=8).grid(row=0, column=5, sticky="w", padx=(4, 12))

        ttk.Button(exp, text="Wybierz folder wyjściowy…", command=self.pick_export_dir).grid(row=0, column=6, sticky="w", padx=8)
        ttk.Button(exp, text="Eksportuj anotacje…", command=self.export_annotated).grid(row=0, column=7, sticky="w", padx=8)

        exp.columnconfigure(8, weight=1)

        # grid 2x2
        grid = ttk.Frame(self)
        grid.pack(fill="both", expand=True, padx=10, pady=10)

        self.canvas = []
        self.tkimgs = [None, None, None, None]  # keep refs
        self.tile_labels = []

        for r in range(2):
            grid.rowconfigure(r, weight=1)
            for c in range(2):
                grid.columnconfigure(c, weight=1)
                i = r * 2 + c

                cell = ttk.Frame(grid, relief="ridge", padding=4)
                cell.grid(row=r, column=c, sticky="nsew", padx=6, pady=6)

                title = ttk.Label(cell, text=f"Kamera {i+1}: —")
                title.pack(anchor="w")
                self.tile_labels.append(title)

                cnv = tk.Canvas(cell, bg="black", highlightthickness=2)
                cnv.pack(fill="both", expand=True)
                cnv.bind("<Button-1>", lambda e, idx=i: self.toggle_rating(idx))
                self.canvas.append(cnv)

        # bottom help
        helpf = ttk.Frame(self)
        helpf.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Label(
            helpf,
            text="Skróty: ←/A poprzedni, →/D następny, R odśwież. Klik na kafelku: ? → OK → BAD → ?",
        ).pack(anchor="w")

    def _bind_keys(self):
        self.bind("<Left>", lambda e: self.prev())
        self.bind("<Right>", lambda e: self.next())
        self.bind("<Key-a>", lambda e: self.prev())
        self.bind("<Key-d>", lambda e: self.next())
        self.bind("<Key-r>", lambda e: self.refresh_current())

    def pick_root(self):
        p = filedialog.askdirectory(title="Wybierz folder główny (ma zawierać 4 podfoldery z obrazami)")
        if not p:
            return
        root = Path(p)
        subdirs = [d for d in root.iterdir() if d.is_dir()]
        print(f"subdirs {subdirs}")
        subdirs.sort()

        cam_dirs = []
        for d in subdirs:
            imgs = list_images(d)
            print(f"img {imgs}")
            if imgs:
                cam_dirs.append(d)
                print(len(cam_dirs))
            if len(cam_dirs) == 4:
                break

        if len(cam_dirs) != 4:
            messagebox.showerror(
                "Błąd",
                "Nie znalazłem 4 podfolderów z obrazami.\n"
                "Upewnij się, że folder główny zawiera 4 foldery (kamery) z plikami jpg/png itp.",
            )
            return

        self.root_dir = root
        self.cam_dirs = cam_dirs
        self.cam_images = [list_images(d) for d in cam_dirs]
        self.index = 0

        self.ratings_path = root / "ratings.jsonl"
        self.export_text.set("Eksport: (wybierz folder wyjściowy)")

        self.status_text.set(f"Folder: {root} | kamery: {[d.name for d in cam_dirs]}")
        self.export_from.set("0")
        self.export_to.set(str(max(0, self.max_len() - 1)))
        self.refresh_current()

    def pick_model(self):
        p = filedialog.askopenfilename(
            title="Wybierz model YOLO (.pt)",
            filetypes=[("PyTorch model", "*.pt"), ("Wszystkie pliki", "*.*")],
        )
        if not p:
            return
        self.model_path = Path(p)
        try:
            self.model = YOLO(str(self.model_path))
        except Exception as e:
            messagebox.showerror("Błąd modelu", f"Nie udało się wczytać modelu:\n{e}")
            self.model = None
            return

        self.status_text.set(f"Model: {self.model_path} | {self.status_text.get()}")
        self.refresh_current()

    def pick_export_dir(self):
        p = filedialog.askdirectory(title="Wybierz folder wyjściowy na anotacje (zostaną utworzone 4 podfoldery)")
        if not p:
            return
        self.export_out_dir = Path(p)
        cams = [d.name for d in self.cam_dirs] if self.cam_dirs else ["cam1", "cam2", "cam3", "cam4"]
        self.export_text.set(f"Eksport: {self.export_out_dir} | podfoldery: {cams}")

    def max_len(self) -> int:
        if not self.cam_images or not self.cam_images[0]:
            return 0
        return min(len(lst) for lst in self.cam_images)

    def _safe_get_imgpath(self, cam_idx: int, idx: int) -> Path | None:
        if cam_idx >= len(self.cam_images):
            return None
        lst = self.cam_images[cam_idx]
        if 0 <= idx < len(lst):
            return lst[idx]
        return None

    def next(self):
        if self.max_len() == 0:
            return
        self.index = min(self.index + int(self.step.get()), self.max_len() - 1)
        self.refresh_current()

    def prev(self):
        if self.max_len() == 0:
            return
        self.index = max(self.index - int(self.step.get()), 0)
        self.refresh_current()

    def jump(self):
        if self.max_len() == 0:
            return
        try:
            v = int(self.jump_var.get().strip())
        except Exception:
            return
        v = max(0, min(v, self.max_len() - 1))
        self.index = v
        self.refresh_current()

    def refresh_current(self):
        if not self.cam_dirs:
            self.status_text.set("Najpierw wybierz folder główny.")
            return

        self.counter_lbl.config(text=f"index: {self.index}/{max(0, self.max_len()-1)} | wspólny zakres: {self.max_len()}")

        for i in range(4):
            p = self._safe_get_imgpath(i, self.index)
            base = self.cam_dirs[i].name if i < len(self.cam_dirs) else f"cam{i+1}"
            if p:
                rate = self.ratings.get((i, self.index), "?")
                self.tile_labels[i].config(text=f"{base} | {p.name} | ocena: {rate}")
            else:
                self.tile_labels[i].config(text=f"{base} | brak pliku | ocena: ?")

        threading.Thread(target=self._render_four, daemon=True).start()

    def _render_four(self):
        sizes = []
        for cnv in self.canvas:
            w = cnv.winfo_width()
            h = cnv.winfo_height()
            if w < 50 or h < 50:
                w, h = 640, 360
            sizes.append((w, h))

        cls_txt = self.cls_id.get().strip()
        cls_id = int(cls_txt) if cls_txt != "" else None

        for i in range(4):
            img_path = self._safe_get_imgpath(i, self.index)
            wbox, hbox = sizes[i]
            if img_path is None:
                self._set_canvas_text(i, "BRAK OBRAZU")
                continue

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                self._set_canvas_text(i, "NIE MOŻNA WCZYTAĆ")
                continue

            if self.model is not None:
                try:
                    show_bgr = draw_detections_on_bgr(
                        self.model,
                        img_bgr,
                        conf=float(self.conf.get()),
                        imgsz=int(self.imgsz.get()),
                        cls_id=cls_id,
                    )
                except Exception as e:
                    show_bgr = img_bgr.copy()
                    cv2.putText(show_bgr, f"ERR: {e}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                show_bgr = img_bgr

            show_rgb = cv2.cvtColor(show_bgr, cv2.COLOR_BGR2RGB)
            show_rgb = fit_to_box(show_rgb, wbox, hbox)

            pil = Image.fromarray(show_rgb)
            tki = ImageTk.PhotoImage(pil)
            self.after(0, lambda idx=i, img=tki: self._set_canvas_image(idx, img))

    def _set_canvas_text(self, idx: int, text: str):
        def _do():
            cnv = self.canvas[idx]
            cnv.delete("all")
            cnv.create_text(20, 20, text=text, fill="white", anchor="nw", font=("Arial", 18, "bold"))
        self.after(0, _do)

    def _set_canvas_image(self, idx: int, tkimg: ImageTk.PhotoImage):
        cnv = self.canvas[idx]
        cnv.delete("all")
        self.tkimgs[idx] = tkimg
        cnv.create_image(0, 0, image=tkimg, anchor="nw")

        rate = self.ratings.get((idx, self.index), "?")
        cnv.create_rectangle(10, 10, 90, 44, fill="black", outline="white")
        cnv.create_text(50, 27, text=rate, fill="white", font=("Arial", 14, "bold"))

        if rate == "OK":
            cnv.config(highlightbackground="#2ecc71")
        elif rate == "BAD":
            cnv.config(highlightbackground="#e74c3c")
        else:
            cnv.config(highlightbackground="#aaaaaa")

    def toggle_rating(self, cam_idx: int):
        cur = self.ratings.get((cam_idx, self.index), "?")
        nxt = "OK" if cur == "?" else ("BAD" if cur == "OK" else "?")
        self.ratings[(cam_idx, self.index)] = nxt
        self._append_rating(cam_idx, self.index, nxt)
        self.refresh_current()

    def _append_rating(self, cam_idx: int, idx: int, rating: str):
        if self.ratings_path is None:
            return
        img_path = self._safe_get_imgpath(cam_idx, idx)
        payload = {
            "cam_idx": cam_idx,
            "cam_name": self.cam_dirs[cam_idx].name if cam_idx < len(self.cam_dirs) else f"cam{cam_idx+1}",
            "index": idx,
            "image": str(img_path) if img_path else None,
            "rating": rating,
        }
        try:
            with open(self.ratings_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # =========================
    # EXPORT
    # =========================
    def export_annotated(self):
        if self._export_running:
            return
        if not self.cam_dirs:
            messagebox.showerror("Błąd", "Najpierw wybierz folder główny z 4 podfolderami.")
            return
        if self.model is None:
            messagebox.showerror("Błąd", "Najpierw wybierz model .pt (bez modelu nie ma co eksportować).")
            return
        if self.export_out_dir is None:
            messagebox.showerror("Błąd", "Najpierw wybierz folder wyjściowy eksportu.")
            return

        mlen = self.max_len()
        if mlen <= 0:
            messagebox.showerror("Błąd", "Brak wspólnego zakresu obrazów.")
            return

        mode = self.export_mode.get()
        if mode == "current":
            start = end = self.index
        else:
            try:
                start = int(self.export_from.get().strip())
                end = int(self.export_to.get().strip())
            except Exception:
                messagebox.showerror("Błąd", "Zakres eksportu musi być liczbami całkowitymi.")
                return

        start = max(0, min(start, mlen - 1))
        end = max(0, min(end, mlen - 1))
        if end < start:
            start, end = end, start

        # create cam subfolders
        for d in self.cam_dirs[:4]:
            (self.export_out_dir / d.name).mkdir(parents=True, exist_ok=True)

        cls_txt = self.cls_id.get().strip()
        cls_id = int(cls_txt) if cls_txt != "" else None

        self._export_running = True
        self.export_text.set(f"Eksport: trwa... indeksy {start}-{end} do {self.export_out_dir}")
        threading.Thread(
            target=self._export_worker,
            args=(start, end, cls_id),
            daemon=True
        ).start()

    def _export_worker(self, start: int, end: int, cls_id: int | None):
        ok_count = 0
        fail_count = 0
        total = (end - start + 1) * 4

        for idx in range(start, end + 1):
            for cam_idx in range(4):
                img_path = self._safe_get_imgpath(cam_idx, idx)
                if img_path is None:
                    fail_count += 1
                    continue

                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    fail_count += 1
                    continue

                try:
                    plotted_bgr = draw_detections_on_bgr(
                        self.model,
                        img_bgr,
                        conf=float(self.conf.get()),
                        imgsz=int(self.imgsz.get()),
                        cls_id=cls_id,
                    )
                    out_path = (self.export_out_dir / self.cam_dirs[cam_idx].name / img_path.name)
                    cv2.imwrite(str(out_path), plotted_bgr)
                    ok_count += 1
                except Exception:
                    fail_count += 1

                # progress update
                done = ok_count + fail_count
                if done % 10 == 0 or done == total:
                    self.after(0, lambda d=done, t=total, ok=ok_count, fl=fail_count: self.export_text.set(
                        f"Eksport: {d}/{t} | OK={ok} FAIL={fl} | out={self.export_out_dir}"
                    ))

        self._export_running = False
        self.after(0, lambda: messagebox.showinfo(
            "Eksport zakończony",
            f"Zapisano: {ok_count}\nNieudane: {fail_count}\n\nFolder: {self.export_out_dir}"
        ))
        self.after(0, lambda: self.export_text.set(
            f"Eksport: gotowe | OK={ok_count} FAIL={fail_count} | out={self.export_out_dir}"
        ))


if __name__ == "__main__":
    app = App()
    app.mainloop()
