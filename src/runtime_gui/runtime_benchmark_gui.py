"""Simple Tkinter GUI for the two runtime studies.

This is a separate, parallel front-end. It does NOT touch gui.py, the
real grabber, or live_runtime/*. It calls the programmatic API in
src.runtime_benchmark.study_runner directly via a worker thread.

Run:
    python -m src.runtime_gui.runtime_benchmark_gui
"""
from __future__ import annotations

import logging
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional, Tuple

from src.runtime_benchmark.study_runner import (
    DEFAULT_CANDIDATE_RESOLUTIONS,
    StudyConfig,
    run_max_fps_study,
    run_max_resolution_study,
)

def _format_default_resolutions() -> str:
    return ", ".join(f"{w}x{h}" for (w, h) in DEFAULT_CANDIDATE_RESOLUTIONS)


def _parse_resolutions(text: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for token in text.replace(";", ",").split(","):
        token = token.strip().lower().replace(" ", "")
        if not token:
            continue
        if "x" not in token:
            raise ValueError(f"Resolution must look like '1280x960', got {token!r}")
        w_str, h_str = token.split("x", 1)
        w, h = int(w_str), int(h_str)
        if w <= 0 or h <= 0 or w % 2 != 0 or h % 2 != 0:
            raise ValueError(f"Resolution {token!r} must be positive and even.")
        out.append((w, h))
    if not out:
        raise ValueError("No resolutions provided.")
    return out


class RuntimeBenchmarkGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("VolleyHub Runtime Benchmark — Studies")
        self.root.geometry("980x720")

        self._study_var = tk.StringVar(value="A")
        self._pipeline_mode = tk.StringVar(value="auto")
        self._color_backend = tk.StringVar(value="auto")
        self._inference_backend = tk.StringVar(value="auto")
        self._bayer_pattern = tk.StringVar(value="RG")
        self._dtype = tk.StringVar(value="uint8")

        self._imgsz = tk.IntVar(value=640)
        self._batch_size = tk.IntVar(value=4)
        self._device = tk.StringVar(value="cuda")
        self._half = tk.BooleanVar(value=True)
        self._num_frames = tk.IntVar(value=200)
        self._warmup = tk.IntVar(value=10)
        self._dry_run = tk.BooleanVar(value=True)
        self._confidence = tk.DoubleVar(value=0.25)
        self._iou = tk.DoubleVar(value=0.45)
        self._max_det = tk.IntVar(value=300)
        self._ball_class_id = tk.IntVar(value=0)
        self._model_path = tk.StringVar(value="")

        self._target_ms = tk.DoubleVar(value=20.0)
        self._candidate_text = tk.StringVar(value=_format_default_resolutions())

        self._fixed_width = tk.IntVar(value=1280)
        self._fixed_height = tk.IntVar(value=960)

        self._log_queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._run_button: Optional[ttk.Button] = None
        self._output: Optional[tk.Text] = None

        self._build_layout()
        self._poll_log_queue()

    # --- layout ---------------------------------------------------------

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        # Top: study selection
        study_frame = ttk.LabelFrame(outer, text="Study", padding=8)
        study_frame.pack(fill="x", pady=(0, 8))
        ttk.Radiobutton(
            study_frame,
            text="A) Najwyższa rozdzielczość mieszcząca się w budżecie czasowym",
            variable=self._study_var, value="A", command=self._on_study_changed,
        ).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            study_frame,
            text="B) Maksymalny FPS dla zadanej rozdzielczości",
            variable=self._study_var, value="B", command=self._on_study_changed,
        ).grid(row=1, column=0, sticky="w")

        # Middle: two parameter blocks side by side
        params = ttk.Frame(outer)
        params.pack(fill="x", pady=(0, 8))
        params.columnconfigure(0, weight=1)
        params.columnconfigure(1, weight=1)

        # Left block: study-specific inputs
        self._study_block_a = ttk.LabelFrame(params, text="Study A — parametry", padding=8)
        ttk.Label(self._study_block_a, text="Budżet czasu / paczkę [ms]:").grid(row=0, column=0, sticky="w")
        ttk.Entry(self._study_block_a, textvariable=self._target_ms, width=10).grid(row=0, column=1, sticky="w")
        ttk.Label(self._study_block_a, text="(20 ms = 50 FPS)").grid(row=0, column=2, sticky="w", padx=(6, 0))
        ttk.Label(self._study_block_a, text="Kandydaci rozdzielczości (WxH, oddzielone przecinkami):").grid(row=1, column=0, columnspan=3, sticky="w", pady=(8, 0))
        self._candidate_entry = tk.Text(self._study_block_a, height=4, width=60, wrap="word")
        self._candidate_entry.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(2, 0))
        self._candidate_entry.insert("1.0", _format_default_resolutions())

        self._study_block_b = ttk.LabelFrame(params, text="Study B — parametry", padding=8)
        ttk.Label(self._study_block_b, text="Width:").grid(row=0, column=0, sticky="w")
        ttk.Entry(self._study_block_b, textvariable=self._fixed_width, width=10).grid(row=0, column=1, sticky="w")
        ttk.Label(self._study_block_b, text="Height:").grid(row=1, column=0, sticky="w")
        ttk.Entry(self._study_block_b, textvariable=self._fixed_height, width=10).grid(row=1, column=1, sticky="w")
        ttk.Label(
            self._study_block_b,
            text="(domyślnie 1280x960)",
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))

        # Common pipeline + inference block
        common = ttk.LabelFrame(outer, text="Pipeline + inferencja (wspólne)", padding=8)
        common.pack(fill="x", pady=(0, 8))

        row = 0
        ttk.Label(common, text="Pipeline mode:").grid(row=row, column=0, sticky="w")
        ttk.Combobox(common, textvariable=self._pipeline_mode,
                     values=("auto", "cpu", "gpu_roundtrip", "gpu_zero_copy"),
                     state="readonly", width=15).grid(row=row, column=1, sticky="w")
        ttk.Label(common, text="Color backend:").grid(row=row, column=2, sticky="w", padx=(12, 0))
        ttk.Combobox(common, textvariable=self._color_backend,
                     values=("auto", "cv2_cuda", "torch_gpu", "cpu"),
                     state="readonly", width=12).grid(row=row, column=3, sticky="w")
        ttk.Label(common, text="Inference backend:").grid(row=row, column=4, sticky="w", padx=(12, 0))
        ttk.Combobox(common, textvariable=self._inference_backend,
                     values=("auto", "ultralytics", "tensorrt", "dry_run"),
                     state="readonly", width=12).grid(row=row, column=5, sticky="w")

        row += 1
        ttk.Label(common, text="Bayer:").grid(row=row, column=0, sticky="w", pady=(6, 0))
        ttk.Combobox(common, textvariable=self._bayer_pattern,
                     values=("RG", "BG", "GR", "GB"),
                     state="readonly", width=6).grid(row=row, column=1, sticky="w", pady=(6, 0))
        ttk.Label(common, text="dtype:").grid(row=row, column=2, sticky="w", padx=(12, 0), pady=(6, 0))
        ttk.Combobox(common, textvariable=self._dtype,
                     values=("uint8", "uint16"),
                     state="readonly", width=8).grid(row=row, column=3, sticky="w", pady=(6, 0))
        ttk.Label(common, text="Device:").grid(row=row, column=4, sticky="w", padx=(12, 0), pady=(6, 0))
        ttk.Entry(common, textvariable=self._device, width=10).grid(row=row, column=5, sticky="w", pady=(6, 0))

        row += 1
        ttk.Label(common, text="imgsz (YOLO):").grid(row=row, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(common, textvariable=self._imgsz, width=8).grid(row=row, column=1, sticky="w", pady=(6, 0))
        ttk.Label(common, text="batch_size:").grid(row=row, column=2, sticky="w", padx=(12, 0), pady=(6, 0))
        ttk.Entry(common, textvariable=self._batch_size, width=8).grid(row=row, column=3, sticky="w", pady=(6, 0))
        ttk.Checkbutton(common, text="half (FP16)", variable=self._half).grid(row=row, column=4, sticky="w", padx=(12, 0), pady=(6, 0))
        ttk.Checkbutton(common, text="dry run (bez modelu)", variable=self._dry_run).grid(row=row, column=5, sticky="w", padx=(12, 0), pady=(6, 0))

        row += 1
        ttk.Label(common, text="num_frames:").grid(row=row, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(common, textvariable=self._num_frames, width=8).grid(row=row, column=1, sticky="w", pady=(6, 0))
        ttk.Label(common, text="warmup:").grid(row=row, column=2, sticky="w", padx=(12, 0), pady=(6, 0))
        ttk.Entry(common, textvariable=self._warmup, width=8).grid(row=row, column=3, sticky="w", pady=(6, 0))
        ttk.Label(common, text="confidence:").grid(row=row, column=4, sticky="w", padx=(12, 0), pady=(6, 0))
        ttk.Entry(common, textvariable=self._confidence, width=8).grid(row=row, column=5, sticky="w", pady=(6, 0))

        row += 1
        ttk.Label(common, text="iou:").grid(row=row, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(common, textvariable=self._iou, width=8).grid(row=row, column=1, sticky="w", pady=(6, 0))
        ttk.Label(common, text="max_det:").grid(row=row, column=2, sticky="w", padx=(12, 0), pady=(6, 0))
        ttk.Entry(common, textvariable=self._max_det, width=8).grid(row=row, column=3, sticky="w", pady=(6, 0))
        ttk.Label(common, text="ball_class_id:").grid(row=row, column=4, sticky="w", padx=(12, 0), pady=(6, 0))
        ttk.Entry(common, textvariable=self._ball_class_id, width=8).grid(row=row, column=5, sticky="w", pady=(6, 0))

        row += 1
        ttk.Label(common, text="Model (.pt/.onnx/.engine):").grid(row=row, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(common, textvariable=self._model_path, width=40).grid(row=row, column=1, columnspan=4, sticky="we", pady=(6, 0))
        ttk.Button(common, text="Browse...", command=self._browse_model).grid(row=row, column=5, sticky="w", pady=(6, 0))

        # Run button + clear log
        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=(0, 8))
        self._run_button = ttk.Button(controls, text="Uruchom", command=self._on_run_clicked)
        self._run_button.pack(side="left")
        ttk.Button(controls, text="Wyczyść log", command=self._clear_log).pack(side="left", padx=(8, 0))
        ttk.Button(controls, text="Zamknij", command=self.root.destroy).pack(side="right")

        # Output text area
        out_frame = ttk.LabelFrame(outer, text="Postęp / wyniki", padding=4)
        out_frame.pack(fill="both", expand=True)
        self._output = tk.Text(out_frame, height=20, wrap="word", state="disabled",
                               font=("Consolas", 10))
        scroll = ttk.Scrollbar(out_frame, orient="vertical", command=self._output.yview)
        self._output.configure(yscrollcommand=scroll.set)
        self._output.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        self._on_study_changed()

    # --- behavior --------------------------------------------------------

    def _on_study_changed(self) -> None:
        if self._study_var.get() == "A":
            self._study_block_a.grid(row=0, column=0, padx=(0, 6), sticky="nsew")
            self._study_block_b.grid_forget()
        else:
            self._study_block_a.grid_forget()
            self._study_block_b.grid(row=0, column=0, padx=(0, 6), sticky="nsew")

    def _browse_model(self) -> None:
        initial = self._model_path.get() or str(Path.cwd())
        path = filedialog.askopenfilename(
            title="Wybierz model YOLO",
            initialdir=str(Path(initial).parent if Path(initial).exists() else Path.cwd()),
            filetypes=[("YOLO model", "*.pt *.onnx *.engine"), ("All files", "*.*")],
        )
        if path:
            self._model_path.set(path)

    def _clear_log(self) -> None:
        if self._output is None:
            return
        self._output.configure(state="normal")
        self._output.delete("1.0", "end")
        self._output.configure(state="disabled")

    def _append_log(self, line: str) -> None:
        if self._output is None:
            return
        self._output.configure(state="normal")
        self._output.insert("end", line.rstrip() + "\n")
        self._output.see("end")
        self._output.configure(state="disabled")

    def _enqueue_log(self, line: str) -> None:
        self._log_queue.put(line)

    def _poll_log_queue(self) -> None:
        try:
            while True:
                msg = self._log_queue.get_nowait()
                if msg is None:
                    if self._run_button is not None:
                        self._run_button.configure(state="normal", text="Uruchom")
                    continue
                self._append_log(msg)
        except queue.Empty:
            pass
        self.root.after(80, self._poll_log_queue)

    # --- run -------------------------------------------------------------

    def _build_cfg(self) -> StudyConfig:
        return StudyConfig(
            pipeline_mode=self._pipeline_mode.get(),
            color_backend=self._color_backend.get(),
            inference_backend=self._inference_backend.get(),
            bayer_pattern=self._bayer_pattern.get(),
            dtype=self._dtype.get(),
            imgsz=int(self._imgsz.get()),
            batch_size=int(self._batch_size.get()),
            device=self._device.get().strip() or "cuda",
            half=bool(self._half.get()),
            num_frames=int(self._num_frames.get()),
            warmup_iterations=int(self._warmup.get()),
            model_path=(self._model_path.get().strip() or None),
            dry_run=bool(self._dry_run.get()),
            confidence=float(self._confidence.get()),
            iou=float(self._iou.get()),
            ball_class_id=int(self._ball_class_id.get()),
            max_det=int(self._max_det.get()),
        )

    def _on_run_clicked(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            messagebox.showinfo("Zajęte", "Pomiar już trwa, poczekaj na zakończenie.")
            return

        try:
            cfg = self._build_cfg()
            cfg.validate()
        except Exception as exc:
            messagebox.showerror("Błąd parametrów", str(exc))
            return

        if self._study_var.get() == "A":
            try:
                target_ms = float(self._target_ms.get())
                if target_ms <= 0:
                    raise ValueError("Budżet czasu musi być > 0 ms.")
                text = self._candidate_entry.get("1.0", "end").strip()
                candidates = _parse_resolutions(text)
            except Exception as exc:
                messagebox.showerror("Błąd parametrów Study A", str(exc))
                return

            def task() -> None:
                try:
                    self._enqueue_log(
                        f"== Study A: rozdzielczość przy <= {target_ms:.2f} ms "
                        f"({1000.0/target_ms:.1f} FPS) =="
                    )
                    run_max_resolution_study(
                        target_ms=target_ms,
                        candidates=candidates,
                        cfg=cfg,
                        progress=self._enqueue_log,
                    )
                except Exception as exc:
                    self._enqueue_log(f"BLAD: {exc!r}")
                finally:
                    self._enqueue_log("== Study A zakończone ==")
                    self._log_queue.put(None)
        else:
            try:
                w = int(self._fixed_width.get())
                h = int(self._fixed_height.get())
                if w <= 0 or h <= 0 or w % 2 != 0 or h % 2 != 0:
                    raise ValueError("Width/Height muszą być parzyste i > 0.")
            except Exception as exc:
                messagebox.showerror("Błąd parametrów Study B", str(exc))
                return

            def task() -> None:
                try:
                    self._enqueue_log(f"== Study B: max FPS przy {w}x{h} ==")
                    run_max_fps_study(
                        width=w,
                        height=h,
                        cfg=cfg,
                        progress=self._enqueue_log,
                    )
                except Exception as exc:
                    self._enqueue_log(f"BLAD: {exc!r}")
                finally:
                    self._enqueue_log("== Study B zakończone ==")
                    self._log_queue.put(None)

        if self._run_button is not None:
            self._run_button.configure(state="disabled", text="Trwa pomiar...")
        self._worker = threading.Thread(target=task, daemon=True)
        self._worker.start()


def main() -> int:
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    root = tk.Tk()
    RuntimeBenchmarkGUI(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
