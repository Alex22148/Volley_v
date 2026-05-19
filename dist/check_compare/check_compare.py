from __future__ import annotations

import csv
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from ultralytics import YOLO

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

PKG_ROOT = Path(__file__).resolve().parents[1]
ONNX_STATIC_DIR = PKG_ROOT / "fastpath_lab_benchmark" / "engines" / "onnx" / "static"
ENGINE_STATIC_DIR = PKG_ROOT / "fastpath_lab_benchmark" / "engines" / "static"

DEFAULT_BATCH = 4
DEFAULT_PRECISION = "fp16"
PROBLEM_VERDICTS = {"LOW_IOU", "MISSED_DETECTION", "FAILED", "EXTRA_DETECTION"}

RESOLUTION_PRESETS: dict[str, tuple[int, int]] = {
    "v640": (640, 640),
    "v960": (960, 960),
    "v1088x1920": (1088, 1920),
}

BACKENDS = ("pt", "onnx", "engine")
TILE_ORDER = ("original", "pt", "onnx", "engine")
TILE_NAMES = {"original": "ORYGINAŁ", "pt": "PT", "onnx": "ONNX", "engine": "ENGINE"}


# ============================================================
# Helpers
# ============================================================
def _fmt(v: Any, p: int = 3) -> str:
    try:
        if v is None:
            return "-"
        return f"{float(v):.{p}f}"
    except Exception:
        return "-"


def list_images(folder: Path) -> list[Path]:
    files: list[Path] = []
    for ext in IMG_EXTS:
        files.extend(folder.rglob(f"*{ext}"))
    files.sort()
    return files


def fit_to_box(img_rgb, max_w: int, max_h: int):
    h, w = img_rgb.shape[:2]
    if h <= 0 or w <= 0:
        return img_rgb
    scale = min(max_w / w, max_h / h)
    nw = max(1, int(w * max(scale, 1e-6)))
    nh = max(1, int(h * max(scale, 1e-6)))
    return cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)


def safe_filename(path: Path) -> str:
    return path.name.replace(" ", "_")


def engine_name_for(label: str, batch: int = DEFAULT_BATCH, precision: str = DEFAULT_PRECISION) -> str:
    return f"best__{precision}_{label}_b{batch}.engine"


def onnx_name_for(label: str, batch: int = DEFAULT_BATCH, precision: str = DEFAULT_PRECISION) -> str:
    return f"best__{precision}_{label}_b{batch}.onnx"


def resize_input_for_backend(img_bgr, hw: tuple[int, int]):
    target_h, target_w = int(hw[0]), int(hw[1])
    src_h, src_w = img_bgr.shape[:2]
    resized = cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return resized, {
        "source_hw": [int(src_h), int(src_w)],
        "target_hw": [int(target_h), int(target_w)],
        "scale_y": float(target_h / src_h) if src_h else None,
        "scale_x": float(target_w / src_w) if src_w else None,
        "resize_mode": "direct_resize_no_letterbox",
    }


def map_box_from_target_to_original(bbox_xyxy, resize_info: dict[str, Any]):
    if bbox_xyxy is None:
        return None
    sx = resize_info.get("scale_x")
    sy = resize_info.get("scale_y")
    if not sx or not sy:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    return [x1 / float(sx), y1 / float(sy), x2 / float(sx), y2 / float(sy)]


def crop_original_with_margin(original_bgr, bbox_xyxy_original, margin: int = 40):
    if original_bgr is None or bbox_xyxy_original is None:
        return None
    h, w = original_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy_original]
    x1 = max(0, x1 - int(margin))
    y1 = max(0, y1 - int(margin))
    x2 = min(w - 1, x2 + int(margin))
    y2 = min(h - 1, y2 + int(margin))
    if x2 <= x1 or y2 <= y1:
        return None
    return {
        "crop_bgr": original_bgr[y1:y2, x1:x2].copy(),
        "crop_box_original_xyxy": [int(x1), int(y1), int(x2), int(y2)],
    }


def select_problem_crop_box(gt_boxes, pred_boxes, verdict: str):
    gt = gt_boxes[0] if gt_boxes else None
    pred = pred_boxes[0] if pred_boxes else None
    if verdict in ("MISSED", "MISSED_DETECTION"):
        return gt
    if verdict in ("FALSE_POSITIVE", "EXTRA_DETECTION"):
        return pred
    if verdict in ("PARTIAL", "LOW_IOU"):
        if gt and pred:
            return [
                min(gt[0], pred[0]),
                min(gt[1], pred[1]),
                max(gt[2], pred[2]),
                max(gt[3], pred[3]),
            ]
        return gt or pred
    if verdict == "FAILED":
        return gt or pred
    return gt or pred


def bbox_iou_xyxy(a: list[float] | None, b: list[float] | None) -> float | None:
    if a is None or b is None:
        return None
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return None
    return float(inter / union)


def make_verdict(
    *,
    ref_detected: bool,
    test_detected: bool,
    iou: float | None,
    error: str | None,
    low_iou_threshold: float = 0.50,
) -> str:
    if error:
        return "FAILED"
    if ref_detected and not test_detected:
        return "MISSED_DETECTION"
    if not ref_detected and test_detected:
        return "EXTRA_DETECTION"
    if ref_detected and test_detected and iou is not None and iou < low_iou_threshold:
        return "LOW_IOU"
    return "OK"


def extract_best_detection(result) -> dict[str, Any]:
    out = {"detected": False, "num_detections": 0, "best_confidence": None, "bbox_xyxy": None, "class_id": None}
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return out
    try:
        confs = boxes.conf.detach().cpu().numpy()
        xyxy = boxes.xyxy.detach().cpu().numpy()
        cls = boxes.cls.detach().cpu().numpy()
        best_idx = int(confs.argmax())
        out["detected"] = True
        out["num_detections"] = int(len(xyxy))
        out["best_confidence"] = float(confs[best_idx])
        out["bbox_xyxy"] = [float(x) for x in xyxy[best_idx].tolist()]
        out["class_id"] = int(cls[best_idx])
    except Exception:
        pass
    return out


# ============================================================
# Detection
# ============================================================
def run_backend_detection(
    backend_name: str,
    model: YOLO | None,
    img_bgr,
    *,
    conf: float,
    imgsz: int | tuple[int, int],
    cls_id: int | None,
    batch: int = 1,
) -> dict[str, Any]:
    if model is None:
        plotted = img_bgr.copy()
        cv2.putText(plotted, f"{backend_name.upper()}: MODEL NOT LOADED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return {
            "backend": backend_name,
            "detected": False,
            "num_detections": 0,
            "best_confidence": None,
            "bbox_xyxy": None,
            "class_id": None,
            "total_ms": None,
            "batch_used": 0,
            "error": "model_not_loaded",
            "plotted_bgr": plotted,
        }

    if backend_name in ("onnx", "engine"):
        batch = max(1, int(batch))
        predict_input = [img_bgr for _ in range(batch)]
        batch_used = batch
    else:
        predict_input = img_bgr
        batch_used = 1

    t0 = time.perf_counter()
    try:
        res = model.predict(predict_input, conf=conf, imgsz=imgsz, verbose=False)[0]
        if cls_id is not None and res.boxes is not None and len(res.boxes) > 0:
            keep = res.boxes.cls.int() == int(cls_id)
            res.boxes = res.boxes[keep]
        total_ms = (time.perf_counter() - t0) * 1000.0
        det = extract_best_detection(res)
        return {
            "backend": backend_name,
            "detected": bool(det["detected"]),
            "num_detections": int(det["num_detections"]),
            "best_confidence": det["best_confidence"],
            "bbox_xyxy": det["bbox_xyxy"],
            "class_id": det["class_id"],
            "total_ms": float(total_ms),
            "batch_used": batch_used,
            "error": None,
            "plotted_bgr": res.plot(),
        }
    except Exception as exc:
        total_ms = (time.perf_counter() - t0) * 1000.0
        plotted = img_bgr.copy()
        cv2.putText(plotted, f"{backend_name.upper()} ERROR", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(plotted, str(exc)[:120], (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        return {
            "backend": backend_name,
            "detected": False,
            "num_detections": 0,
            "best_confidence": None,
            "bbox_xyxy": None,
            "class_id": None,
            "total_ms": float(total_ms),
            "batch_used": batch_used,
            "error": str(exc),
            "plotted_bgr": plotted,
        }


def compare_results(
    results: dict[str, dict[str, Any]],
    low_iou_threshold: float = 0.50,
) -> dict[str, dict[str, Any]]:
    ref = results.get("pt", {})
    ref_error = ref.get("error")
    ref_detected = bool(ref.get("detected"))
    ref_bbox = ref.get("bbox_xyxy")
    ref_conf = ref.get("best_confidence")
    reference_valid = ref_error is None and ref_detected and ref_bbox is not None

    comparisons: dict[str, dict[str, Any]] = {}
    for backend in ("onnx", "engine"):
        test = results.get(backend, {})
        test_error = test.get("error")
        test_detected = bool(test.get("detected"))
        test_bbox = test.get("bbox_xyxy")
        test_conf = test.get("best_confidence")
        if not reference_valid:
            comparisons[f"{backend}_vs_pt"] = {
                "comparison": f"{backend}_vs_pt",
                "matched": False,
                "iou": None,
                "conf_ref": ref_conf,
                "conf_test": test_conf,
                "confidence_diff": None,
                "missed_detection": False,
                "extra_detection": False,
                "verdict": "NO_REFERENCE",
                "reason": ref_error or "pt_not_detected_or_no_bbox",
            }
            continue
        iou = bbox_iou_xyxy(ref_bbox, test_bbox)
        confidence_diff = None
        if ref_conf is not None and test_conf is not None:
            confidence_diff = float(test_conf) - float(ref_conf)
        verdict = make_verdict(
            ref_detected=True,
            test_detected=test_detected,
            iou=iou,
            error=test_error,
            low_iou_threshold=low_iou_threshold,
        )
        comparisons[f"{backend}_vs_pt"] = {
            "comparison": f"{backend}_vs_pt",
            "matched": verdict == "OK",
            "iou": iou,
            "conf_ref": ref_conf,
            "conf_test": test_conf,
            "confidence_diff": confidence_diff,
            "missed_detection": bool(ref_detected and not test_detected),
            "extra_detection": bool((not ref_detected) and test_detected),
            "verdict": verdict,
            "reason": test_error,
        }
    return comparisons


def strip_image_arrays(results: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    clean: dict[str, dict[str, Any]] = {}
    for backend, data in results.items():
        item = dict(data)
        item.pop("plotted_bgr", None)
        clean[backend] = item
    return clean


# ============================================================
# GUI
# ============================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("check_compare - PT / ONNX / ENGINE")
        self.geometry("1500x980")

        self.images_dir: Path | None = None
        self.images: list[Path] = []
        self.index = 0
        self.step = tk.IntVar(value=1)

        self.model_paths: dict[str, Path | None] = {"pt": None, "onnx": None, "engine": None}
        self.models: dict[str, YOLO | None] = {"pt": None, "onnx": None, "engine": None}
        self.onnx_static_dir = ONNX_STATIC_DIR
        self.engine_static_dir = ENGINE_STATIC_DIR

        self.conf = tk.DoubleVar(value=0.25)
        self.cls_id = tk.StringVar(value="0")
        self.resolution_label = tk.StringVar(value="v640")
        self.batch = tk.IntVar(value=4)
        self.precision = tk.StringVar(value="fp16")
        self.low_iou_threshold = tk.DoubleVar(value=0.50)
        self.imgsz = tk.IntVar(value=640)

        self.status_text = tk.StringVar(value="Wybierz folder obrazów oraz modele PT / ONNX / ENGINE")
        self.models_text = tk.StringVar(value="Modele: PT=— | ONNX=— | ENGINE=—")
        self.verdict_text = tk.StringVar(value="Werdykt: —")
        self.error_text = tk.StringVar(value="Błędy: —")
        self.export_text = tk.StringVar(value="Eksport: —")
        self.zoom_all = tk.BooleanVar(value=False)

        self.export_mode = tk.StringVar(value="current")
        self.export_from = tk.StringVar(value="0")
        self.export_to = tk.StringVar(value="0")
        self.export_out_dir: Path | None = None
        self._export_running = False

        self.canvas: list[tk.Canvas] = []
        self.tile_labels: list[ttk.Label] = []
        self.tkimgs: list[ImageTk.PhotoImage | None] = [None, None, None, None]
        self.last_results: dict[str, dict[str, Any]] = {}
        self.last_comparisons: dict[str, dict[str, Any]] = {}
        self.problem_indices: set[int] = set()
        self.zoom_rects: dict[int, tuple[float, float, float, float] | None] = {0: None, 1: None, 2: None, 3: None}
        self.display_sizes: dict[int, tuple[int, int]] = {0: (0, 0), 1: (0, 0), 2: (0, 0), 3: (0, 0)}
        self.display_offsets: dict[int, tuple[int, int]] = {0: (0, 0), 1: (0, 0), 2: (0, 0), 3: (0, 0)}
        self.drag_start: dict[int, tuple[int, int] | None] = {0: None, 1: None, 2: None, 3: None}
        self.drag_rect_ids: dict[int, int | None] = {0: None, 1: None, 2: None, 3: None}
        self._rerender_after_id: str | None = None
        self._render_token = 0

        self._build_ui()
        self._bind_keys()
        self._update_models_text()

    # ------------------------------------------------------------
    # Config / helpers
    # ------------------------------------------------------------
    def selected_hw(self) -> tuple[int, int]:
        return RESOLUTION_PRESETS.get(self.resolution_label.get().strip(), RESOLUTION_PRESETS["v640"])

    def selected_imgsz_for_ultralytics(self) -> int | tuple[int, int]:
        h, w = self.selected_hw()
        return int(h) if h == w else (int(h), int(w))

    def sync_imgsz_var_with_resolution(self):
        self.imgsz.set(int(self.selected_hw()[0]))

    def _read_cls_id(self) -> int | None:
        txt = self.cls_id.get().strip()
        if txt == "":
            return None
        try:
            return int(txt)
        except Exception:
            return None

    def _safe_get_imgpath(self, idx: int) -> Path | None:
        return self.images[idx] if 0 <= idx < len(self.images) else None

    def max_len(self) -> int:
        return len(self.images)

    def _current_backend_config(self) -> tuple[str, int, str]:
        label = self.resolution_label.get().strip()
        try:
            batch = int(self.batch.get())
        except Exception:
            batch = DEFAULT_BATCH
            self.batch.set(batch)
        precision = self.precision.get().strip() or DEFAULT_PRECISION
        return label, batch, precision

    def _static_paths_info(self) -> dict[str, Any]:
        label, batch, precision = self._current_backend_config()
        onnx_name = onnx_name_for(label, batch=batch, precision=precision)
        engine_name = engine_name_for(label, batch=batch, precision=precision)
        onnx_path = self.onnx_static_dir / onnx_name
        engine_path = self.engine_static_dir / engine_name
        return {
            "label": label,
            "batch": batch,
            "precision": precision,
            "onnx_name": onnx_name,
            "engine_name": engine_name,
            "onnx_path": onnx_path,
            "engine_path": engine_path,
            "onnx_dir_exists": self.onnx_static_dir.exists(),
            "engine_dir_exists": self.engine_static_dir.exists(),
            "onnx_exists": onnx_path.exists(),
            "engine_exists": engine_path.exists(),
        }

    def _update_models_text(self):
        pt = self.model_paths["pt"].name if self.model_paths.get("pt") else "—"
        onnx = self.model_paths["onnx"].name if self.model_paths.get("onnx") else "—"
        engine = self.model_paths["engine"].name if self.model_paths.get("engine") else "—"
        self.models_text.set(f"Modele: PT={pt} | ONNX={onnx} | ENGINE={engine}")

    def _update_problem_cache(self, idx: int, comparisons: dict[str, dict[str, Any]]):
        onnx_v = comparisons.get("onnx_vs_pt", {}).get("verdict")
        engine_v = comparisons.get("engine_vs_pt", {}).get("verdict")
        if onnx_v in PROBLEM_VERDICTS or engine_v in PROBLEM_VERDICTS:
            self.problem_indices.add(idx)
        else:
            self.problem_indices.discard(idx)

    def _set_verdict_text(self, comparisons: dict[str, dict[str, Any]]):
        onnx_v = comparisons.get("onnx_vs_pt", {}).get("verdict", "—")
        eng_v = comparisons.get("engine_vs_pt", {}).get("verdict", "—")
        onnx_iou = comparisons.get("onnx_vs_pt", {}).get("iou")
        eng_iou = comparisons.get("engine_vs_pt", {}).get("iou")
        self.verdict_text.set(f"Werdykt: ONNX={onnx_v} IoU={self._fmt(onnx_iou)} | ENGINE={eng_v} IoU={self._fmt(eng_iou)}")

    @staticmethod
    def _fmt(v, p: int = 3):
        try:
            if v is None:
                return "-"
            return f"{float(v):.{p}f}"
        except Exception:
            return "-"

    def _normalize_rect(self, x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float] | None:
        nx1, nx2 = sorted([max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))])
        ny1, ny2 = sorted([max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))])
        if (nx2 - nx1) < 0.01 or (ny2 - ny1) < 0.01:
            return None
        return nx1, ny1, nx2, ny2

    def _zoom_targets(self, idx: int) -> list[int]:
        return [0, 1, 2, 3] if self.zoom_all.get() else [idx]

    def clear_zoom(self):
        for i in range(4):
            self.zoom_rects[i] = None
        self._rerender_last_results()

    def _apply_zoom_crop(self, idx: int, rgb):
        rect = self.zoom_rects.get(idx)
        if rect is None:
            return rgb
        h, w = rgb.shape[:2]
        x1 = max(0, min(w - 1, int(rect[0] * w)))
        y1 = max(0, min(h - 1, int(rect[1] * h)))
        x2 = max(1, min(w, int(rect[2] * w)))
        y2 = max(1, min(h, int(rect[3] * h)))
        if x2 <= x1 or y2 <= y1:
            return rgb
        return rgb[y1:y2, x1:x2]

    def _event_to_image_norm(self, idx: int, x: int, y: int) -> tuple[float, float] | None:
        dw, dh = self.display_sizes.get(idx, (0, 0))
        ox, oy = self.display_offsets.get(idx, (0, 0))
        if dw <= 1 or dh <= 1:
            return None
        rx = (float(x) - float(ox)) / float(dw)
        ry = (float(y) - float(oy)) / float(dh)
        if rx < 0.0 or rx > 1.0 or ry < 0.0 or ry > 1.0:
            return None
        return rx, ry

    def _view_norm_to_source_norm(self, idx: int, x: float, y: float) -> tuple[float, float]:
        base = self.zoom_rects.get(idx) or (0.0, 0.0, 1.0, 1.0)
        x1, y1, x2, y2 = base
        return x1 + x * (x2 - x1), y1 + y * (y2 - y1)

    def _rerender_last_results(self):
        if not self.last_results:
            return
        img_path = self._safe_get_imgpath(self.index)
        if img_path is None:
            return
        sizes: list[tuple[int, int]] = []
        for cnv in self.canvas:
            w, h = cnv.winfo_width(), cnv.winfo_height()
            if w < 50 or h < 50:
                w, h = 640, 360
            sizes.append((w, h))
        for idx, backend in enumerate(TILE_ORDER):
            plotted_bgr = self.last_results.get(backend, {}).get("plotted_bgr")
            if plotted_bgr is None:
                continue
            rgb = cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)
            rgb = self._apply_zoom_crop(idx, rgb)
            rgb = fit_to_box(rgb, sizes[idx][0], sizes[idx][1])
            tki = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.after(0, lambda i=idx, img=tki, b=backend, r=self.last_results[backend], name=img_path.name: self._update_tile(i, img, b, r, name))

    def _schedule_rerender_last_results(self):
        if self._rerender_after_id is not None:
            try:
                self.after_cancel(self._rerender_after_id)
            except Exception:
                pass
        self._rerender_after_id = self.after(40, self._rerender_last_results)

    def _on_canvas_press(self, idx: int, event):
        if self._event_to_image_norm(idx, int(event.x), int(event.y)) is None:
            self.drag_start[idx] = None
            return
        self.drag_start[idx] = (int(event.x), int(event.y))
        rect_id = self.drag_rect_ids.get(idx)
        if rect_id is not None:
            self.canvas[idx].delete(rect_id)
            self.drag_rect_ids[idx] = None

    def _on_canvas_motion(self, idx: int, event):
        start = self.drag_start.get(idx)
        if start is None:
            return
        ox, oy = self.display_offsets.get(idx, (0, 0))
        dw, dh = self.display_sizes.get(idx, (0, 0))
        if dw <= 1 or dh <= 1:
            return
        sx, sy = start
        ex, ey = int(event.x), int(event.y)
        sx = max(ox, min(ox + dw, sx))
        sy = max(oy, min(oy + dh, sy))
        ex = max(ox, min(ox + dw, ex))
        ey = max(oy, min(oy + dh, ey))
        cnv = self.canvas[idx]
        rect_id = self.drag_rect_ids.get(idx)
        if rect_id is None:
            rect_id = cnv.create_rectangle(sx, sy, ex, ey, outline="#00ffff", width=2, dash=(4, 2))
            self.drag_rect_ids[idx] = rect_id
        else:
            cnv.coords(rect_id, sx, sy, ex, ey)

    def _on_canvas_release(self, idx: int, event):
        start = self.drag_start.get(idx)
        self.drag_start[idx] = None
        rect_id = self.drag_rect_ids.get(idx)
        if rect_id is not None:
            self.canvas[idx].delete(rect_id)
            self.drag_rect_ids[idx] = None
        if start is None:
            return
        ex, ey = int(event.x), int(event.y)
        sx, sy = start
        ox, oy = self.display_offsets.get(idx, (0, 0))
        dw, dh = self.display_sizes.get(idx, (0, 0))
        if dw <= 1 or dh <= 1:
            return
        sx = max(ox, min(ox + dw, sx))
        sy = max(oy, min(oy + dh, sy))
        ex = max(ox, min(ox + dw, ex))
        ey = max(oy, min(oy + dh, ey))
        n1 = self._event_to_image_norm(idx, sx, sy)
        n2 = self._event_to_image_norm(idx, ex, ey)
        if n1 is None or n2 is None:
            return
        s1 = self._view_norm_to_source_norm(idx, n1[0], n1[1])
        s2 = self._view_norm_to_source_norm(idx, n2[0], n2[1])
        rect = self._normalize_rect(s1[0], s1[1], s2[0], s2[1])
        if rect is None:
            return
        for target in self._zoom_targets(idx):
            self.zoom_rects[target] = rect
        self._rerender_last_results()

    def _on_canvas_wheel(self, idx: int, event):
        point = self._event_to_image_norm(idx, int(event.x), int(event.y))
        if point is None:
            return
        if hasattr(event, "delta") and event.delta != 0:
            direction = 1 if event.delta > 0 else -1
        else:
            direction = 1 if getattr(event, "num", 0) == 4 else -1

        base = self.zoom_rects.get(idx) or (0.0, 0.0, 1.0, 1.0)
        x1, y1, x2, y2 = base
        bw, bh = max(0.01, x2 - x1), max(0.01, y2 - y1)
        cx, cy = self._view_norm_to_source_norm(idx, point[0], point[1])
        scale = 0.85 if direction > 0 else 1.18
        nw = max(0.03, min(1.0, bw * scale))
        nh = max(0.03, min(1.0, bh * scale))
        rel_x = (cx - x1) / bw
        rel_y = (cy - y1) / bh
        nx1 = cx - rel_x * nw
        ny1 = cy - rel_y * nh
        nx2 = nx1 + nw
        ny2 = ny1 + nh
        if nx1 < 0.0:
            nx2 -= nx1
            nx1 = 0.0
        if ny1 < 0.0:
            ny2 -= ny1
            ny1 = 0.0
        if nx2 > 1.0:
            shift = nx2 - 1.0
            nx1 -= shift
            nx2 = 1.0
        if ny2 > 1.0:
            shift = ny2 - 1.0
            ny1 -= shift
            ny2 = 1.0
        rect = self._normalize_rect(nx1, ny1, nx2, ny2)
        if rect is None:
            return
        for target in self._zoom_targets(idx):
            self.zoom_rects[target] = rect
        self._rerender_last_results()

    # ------------------------------------------------------------
    # Model loading / diagnostics
    # ------------------------------------------------------------
    def on_resolution_changed(self):
        self.sync_imgsz_var_with_resolution()
        self.autoload_static_backends()
        self.refresh_current()

    def on_backend_config_changed(self):
        self.autoload_static_backends()
        self.refresh_current()

    def autoload_static_backends(self):
        info = self._static_paths_info()
        messages: list[str] = []

        if info["onnx_exists"]:
            try:
                self.models["onnx"] = YOLO(str(info["onnx_path"]), task="detect")
                self.model_paths["onnx"] = info["onnx_path"]
                messages.append(f"ONNX loaded: {info['onnx_path']}")
            except Exception as exc:
                self.models["onnx"] = None
                self.model_paths["onnx"] = None
                messages.append(f"ONNX load error: {exc}")
                self._update_models_text()
        else:
            self.models["onnx"] = None
            self.model_paths["onnx"] = None
            messages.append(f"ONNX missing: {info['onnx_name']}")

        if info["engine_exists"]:
            try:
                self.models["engine"] = YOLO(str(info["engine_path"]), task="detect")
                self.model_paths["engine"] = info["engine_path"]
                messages.append(f"ENGINE loaded: {info['engine_path']}")
            except Exception as exc:
                self.models["engine"] = None
                self.model_paths["engine"] = None
                messages.append(f"ENGINE load error: {exc}")
                self._update_models_text()
        else:
            self.models["engine"] = None
            self.model_paths["engine"] = None
            messages.append(f"ENGINE missing: {info['engine_name']}")

        self._update_models_text()
        self.status_text.set(
            f"ONNX_STATIC_DIR={self.onnx_static_dir} (exists={info['onnx_dir_exists']}) | "
            f"ENGINE_STATIC_DIR={self.engine_static_dir} (exists={info['engine_dir_exists']}) | "
            f"search ONNX={info['onnx_name']} | ENGINE={info['engine_name']} | " + " | ".join(messages)
        )

    def show_paths_diagnostics(self):
        label = self.resolution_label.get().strip()
        try:
            batch = int(self.batch.get())
        except Exception:
            batch = DEFAULT_BATCH
        precision = self.precision.get().strip() or DEFAULT_PRECISION
        onnx_name = onnx_name_for(label, batch=batch, precision=precision)
        engine_name = engine_name_for(label, batch=batch, precision=precision)
        onnx_path = self.onnx_static_dir / onnx_name
        engine_path = self.engine_static_dir / engine_name
        msg = (
            f"PKG_ROOT:\n{PKG_ROOT}\n\n"
            f"ONNX_STATIC_DIR exists={self.onnx_static_dir.exists()}:\n{self.onnx_static_dir}\n\n"
            f"ENGINE_STATIC_DIR exists={self.engine_static_dir.exists()}:\n{self.engine_static_dir}\n\n"
            f"resolution={label}\n"
            f"batch={batch}\n"
            f"precision={precision}\n\n"
            f"ONNX searched:\n{onnx_path}\nexists={onnx_path.exists()}\n\n"
            f"ENGINE searched:\n{engine_path}\nexists={engine_path.exists()}"
        )
        messagebox.showinfo("Diagnostyka ścieżek", msg)

    def open_model_folder(self):
        folder = self.engine_static_dir
        if folder.exists():
            os.startfile(str(folder))
        else:
            messagebox.showwarning("Brak folderu", f"Nie istnieje:\n{folder}")

    def pick_onnx_folder(self):
        p = filedialog.askdirectory(title="Wskaż folder ONNX")
        if not p:
            return
        self.onnx_static_dir = Path(p)
        self.status_text.set(f"Ustawiono folder ONNX: {self.onnx_static_dir}")
        self.autoload_static_backends()
        self.refresh_current()

    def pick_engine_folder(self):
        p = filedialog.askdirectory(title="Wskaż folder ENGINE")
        if not p:
            return
        self.engine_static_dir = Path(p)
        self.status_text.set(f"Ustawiono folder ENGINE: {self.engine_static_dir}")
        self.autoload_static_backends()
        self.refresh_current()

    def open_export_folder(self):
        if self.export_out_dir and self.export_out_dir.exists():
            os.startfile(str(self.export_out_dir))
        else:
            messagebox.showwarning("Brak folderu", "Folder eksportu nie jest ustawiony lub nie istnieje.")

    # ------------------------------------------------------------
    # GUI
    # ------------------------------------------------------------
    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=8)

        ttk.Button(top, text="Folder obrazów", command=self.pick_images_folder).pack(side="left")
        ttk.Button(top, text="Model PT", command=lambda: self.pick_model("pt")).pack(side="left", padx=(8, 0))
        ttk.Button(top, text="Auto ONNX/ENGINE", command=self.autoload_static_backends).pack(side="left", padx=(8, 0))
        ttk.Button(top, text="Folder ONNX", command=self.pick_onnx_folder).pack(side="left", padx=(8, 0))
        ttk.Button(top, text="Folder ENGINE", command=self.pick_engine_folder).pack(side="left", padx=(8, 0))
        ttk.Button(top, text="Diagnoza ścieżek", command=self.show_paths_diagnostics).pack(side="left", padx=(8, 0))
        ttk.Button(top, text="Otwórz engines", command=self.open_model_folder).pack(side="left", padx=(8, 0))

        ttk.Label(top, text="conf").pack(side="left", padx=(18, 4))
        ttk.Entry(top, textvariable=self.conf, width=7).pack(side="left")
        ttk.Label(top, text="rozdzielczość").pack(side="left", padx=(10, 4))

        res_combo = ttk.Combobox(top, textvariable=self.resolution_label, values=list(RESOLUTION_PRESETS.keys()), width=12, state="readonly")
        res_combo.pack(side="left")
        res_combo.bind("<<ComboboxSelected>>", lambda _e: self.on_resolution_changed())

        ttk.Label(top, text="batch").pack(side="left", padx=(10, 4))
        batch_combo = ttk.Combobox(top, textvariable=self.batch, values=[4, 8, 12, 16, 20, 24, 28, 32, 64], width=6, state="readonly")
        batch_combo.pack(side="left")
        batch_combo.bind("<<ComboboxSelected>>", lambda _e: self.on_backend_config_changed())

        ttk.Label(top, text="precision").pack(side="left", padx=(10, 4))
        precision_combo = ttk.Combobox(top, textvariable=self.precision, values=["fp16", "fp32"], width=6, state="readonly")
        precision_combo.pack(side="left")
        precision_combo.bind("<<ComboboxSelected>>", lambda _e: self.on_backend_config_changed())

        ttk.Label(top, text="class_id").pack(side="left", padx=(10, 4))
        ttk.Entry(top, textvariable=self.cls_id, width=7).pack(side="left")
        ttk.Label(top, text="IoU min").pack(side="left", padx=(10, 4))
        ttk.Entry(top, textvariable=self.low_iou_threshold, width=6).pack(side="left")
        ttk.Label(top, text="krok").pack(side="left", padx=(14, 4))
        ttk.Spinbox(top, from_=1, to=999, textvariable=self.step, width=6).pack(side="left")
        ttk.Button(top, text="Odśwież", command=self.refresh_current).pack(side="right")

        nav = ttk.Frame(self)
        nav.pack(fill="x", padx=10, pady=(0, 8))
        ttk.Button(nav, text="Poprzedni", command=self.prev).pack(side="left")
        ttk.Button(nav, text="Następny", command=self.next).pack(side="left", padx=(8, 0))
        ttk.Button(nav, text="Poprzedni problem", command=self.prev_problem).pack(side="left", padx=(12, 0))
        ttk.Button(nav, text="Następny problem", command=self.next_problem).pack(side="left", padx=(8, 0))
        ttk.Checkbutton(nav, text="Zoom dla wszystkich", variable=self.zoom_all).pack(side="left", padx=(12, 0))
        ttk.Button(nav, text="Reset zoom", command=self.clear_zoom).pack(side="left", padx=(8, 0))

        self.jump_var = tk.StringVar(value="0")
        ttk.Label(nav, text="Idź do indeksu").pack(side="left", padx=(16, 4))
        ttk.Entry(nav, textvariable=self.jump_var, width=8).pack(side="left")
        ttk.Button(nav, text="Skocz", command=self.jump).pack(side="left", padx=(6, 0))
        self.counter_lbl = ttk.Label(nav, text="—")
        self.counter_lbl.pack(side="right")

        ttk.Label(self, textvariable=self.status_text).pack(fill="x", padx=10, pady=(0, 4))
        ttk.Label(self, textvariable=self.models_text).pack(fill="x", padx=10, pady=(0, 4))
        ttk.Label(self, textvariable=self.verdict_text).pack(fill="x", padx=10, pady=(0, 4))
        ttk.Label(self, textvariable=self.error_text).pack(fill="x", padx=10, pady=(0, 4))
        ttk.Label(self, textvariable=self.export_text).pack(fill="x", padx=10, pady=(0, 8))

        exp = ttk.LabelFrame(self, text="Eksport porównania")
        exp.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Radiobutton(exp, text="Bieżący obraz", variable=self.export_mode, value="current").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Radiobutton(exp, text="Zakres", variable=self.export_mode, value="range").grid(row=0, column=1, sticky="w", padx=8, pady=6)
        ttk.Label(exp, text="od").grid(row=0, column=2, sticky="e")
        ttk.Entry(exp, textvariable=self.export_from, width=8).grid(row=0, column=3, sticky="w", padx=(4, 12))
        ttk.Label(exp, text="do").grid(row=0, column=4, sticky="e")
        ttk.Entry(exp, textvariable=self.export_to, width=8).grid(row=0, column=5, sticky="w", padx=(4, 12))
        ttk.Button(exp, text="Folder wyjściowy...", command=self.pick_export_dir).grid(row=0, column=6, sticky="w", padx=8)
        ttk.Button(exp, text="Eksportuj", command=self.export_compare).grid(row=0, column=7, sticky="w", padx=8)
        ttk.Button(exp, text="Otwórz folder eksportu", command=self.open_export_folder).grid(row=0, column=8, sticky="w", padx=8)
        exp.columnconfigure(9, weight=1)

        grid = ttk.Frame(self)
        grid.pack(fill="both", expand=True, padx=10, pady=10)
        for r in range(2):
            grid.rowconfigure(r, weight=1)
            for c in range(2):
                grid.columnconfigure(c, weight=1)
                idx = r * 2 + c
                backend = TILE_ORDER[idx]
                cell = ttk.Frame(grid, relief="ridge", padding=4)
                cell.grid(row=r, column=c, sticky="nsew", padx=6, pady=6)
                title = ttk.Label(cell, text=f"{TILE_NAMES[backend]}: —")
                title.pack(anchor="w")
                self.tile_labels.append(title)
                cnv = tk.Canvas(cell, bg="#111111", highlightthickness=2, highlightbackground="#888888", width=640, height=360)
                cnv.pack(fill="both", expand=True)
                self.canvas.append(cnv)
                cnv.bind("<ButtonPress-1>", lambda e, i=idx: self._on_canvas_press(i, e))
                cnv.bind("<B1-Motion>", lambda e, i=idx: self._on_canvas_motion(i, e))
                cnv.bind("<ButtonRelease-1>", lambda e, i=idx: self._on_canvas_release(i, e))
                cnv.bind("<MouseWheel>", lambda e, i=idx: self._on_canvas_wheel(i, e))
                cnv.bind("<Button-4>", lambda e, i=idx: self._on_canvas_wheel(i, e))
                cnv.bind("<Button-5>", lambda e, i=idx: self._on_canvas_wheel(i, e))
                cnv.bind("<Configure>", lambda _e: self._schedule_rerender_last_results())

        helpf = ttk.Frame(self)
        helpf.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Label(
            helpf,
            text=(
                "Skróty: ←/A poprzedni, →/D następny, R odśwież. "
                "Po zmianie conf/class_id/IoU kliknij Odśwież lub naciśnij R. "
                "PT jest referencją. ONNX/ENGINE są dobierane automatycznie po resolution+batch+precision. "
                "Dla backendów statycznych bN aplikacja powiela bieżący obraz N razy i porównuje wynik [0]. "
                "Zoom: zaznacz prostokąt LPM, kółko myszy zmienia skalę."
            ),
        ).pack(anchor="w")

    def _bind_keys(self):
        self.bind("<Left>", lambda _e: self.prev())
        self.bind("<Right>", lambda _e: self.next())
        self.bind("<Key-a>", lambda _e: self.prev())
        self.bind("<Key-d>", lambda _e: self.next())
        self.bind("<Key-r>", lambda _e: self.refresh_current())

    # ------------------------------------------------------------
    # Pickers / navigation
    # ------------------------------------------------------------
    def pick_images_folder(self):
        p = filedialog.askdirectory(title="Wybierz folder z obrazami")
        if not p:
            return
        folder = Path(p)
        imgs = list_images(folder)
        if not imgs:
            messagebox.showerror("Błąd", "Nie znaleziono obrazów w wybranym folderze.")
            return
        self.images_dir = folder
        self.images = imgs
        self.index = 0
        self.export_from.set("0")
        self.export_to.set(str(max(0, self.max_len() - 1)))
        self.status_text.set(f"Folder obrazów: {folder} | obrazów: {len(imgs)}")
        self.autoload_static_backends()
        self.refresh_current()

    def pick_model(self, backend: str):
        filetypes = {
            "pt": [("PyTorch model", "*.pt"), ("Wszystkie pliki", "*.*")],
            "onnx": [("ONNX model", "*.onnx"), ("Wszystkie pliki", "*.*")],
            "engine": [("TensorRT engine", "*.engine"), ("Wszystkie pliki", "*.*")],
        }
        p = filedialog.askopenfilename(title=f"Wybierz model {backend.upper()}", filetypes=filetypes.get(backend, [("Wszystkie pliki", "*.*")]))
        if not p:
            return
        path = Path(p)
        try:
            model = YOLO(str(path), task="detect")
        except Exception as exc:
            self.models[backend] = None
            self.model_paths[backend] = None
            self._update_models_text()
            messagebox.showerror(f"Błąd modelu {backend.upper()}", f"Nie udało się wczytać modelu:\n\n{exc}")
            return
        self.models[backend] = model
        self.model_paths[backend] = path
        self._update_models_text()
        self.status_text.set(f"Załadowano {backend.upper()}: {path.name}")
        self.refresh_current()

    def pick_export_dir(self):
        p = filedialog.askdirectory(title="Wybierz folder wyjściowy eksportu")
        if not p:
            return
        self.export_out_dir = Path(p)
        self.export_text.set(f"Eksport: {self.export_out_dir}")

    def next(self):
        if self.max_len() == 0:
            return
        self.index = min(self.index + max(1, int(self.step.get())), self.max_len() - 1)
        self.refresh_current()

    def prev(self):
        if self.max_len() == 0:
            return
        self.index = max(self.index - max(1, int(self.step.get())), 0)
        self.refresh_current()

    def jump(self):
        if self.max_len() == 0:
            return
        try:
            idx = int(self.jump_var.get().strip())
        except Exception:
            return
        self.index = max(0, min(idx, self.max_len() - 1))
        self.refresh_current()

    def next_problem(self):
        if not self.problem_indices:
            self.status_text.set("Brak zapisanych problemów dla obejrzanych obrazów.")
            return
        ordered = sorted(self.problem_indices)
        for idx in ordered:
            if idx > self.index:
                self.index = idx
                self.refresh_current()
                return
        self.index = ordered[0]
        self.refresh_current()

    def prev_problem(self):
        if not self.problem_indices:
            self.status_text.set("Brak zapisanych problemów dla obejrzanych obrazów.")
            return
        ordered = sorted(self.problem_indices, reverse=True)
        for idx in ordered:
            if idx < self.index:
                self.index = idx
                self.refresh_current()
                return
        self.index = ordered[0]
        self.refresh_current()

    # ------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------
    def refresh_current(self):
        if not self.images:
            self.status_text.set("Najpierw wybierz folder z obrazami.")
            return
        self._render_token += 1
        token = self._render_token
        self.counter_lbl.config(text=f"index: {self.index}/{max(0, self.max_len() - 1)} | obrazów: {self.max_len()} | problemy: {len(self.problem_indices)}")
        img_path = self._safe_get_imgpath(self.index)
        if img_path is None:
            return
        for i, backend in enumerate(TILE_ORDER):
            self.tile_labels[i].config(text=f"{TILE_NAMES[backend]} | {img_path.name}")
        threading.Thread(target=self._render_compare_worker, args=(token,), daemon=True).start()

    def _set_all_canvas_text(self, text: str):
        for i in range(4):
            self._set_canvas_text(i, text)

    def _set_canvas_text(self, idx: int, text: str):
        def _do():
            cnv = self.canvas[idx]
            cnv.delete("all")
            cnv.create_text(20, 20, text=text, fill="white", anchor="nw", font=("Arial", 18, "bold"))

        self.after(0, _do)

    def _render_compare_worker(self, token: int):
        sizes: list[tuple[int, int]] = []
        for cnv in self.canvas:
            w, h = cnv.winfo_width(), cnv.winfo_height()
            if w < 50 or h < 50:
                w, h = 640, 360
            sizes.append((w, h))

        current_index = self.index
        img_path = self._safe_get_imgpath(current_index)
        if img_path is None:
            self._set_all_canvas_text("BRAK OBRAZU")
            return
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            self._set_all_canvas_text("NIE MOŻNA WCZYTAĆ OBRAZU")
            return

        input_bgr, resize_info = resize_input_for_backend(img_bgr, self.selected_hw())
        cls_id = self._read_cls_id()
        conf = float(self.conf.get())
        imgsz = self.selected_imgsz_for_ultralytics()
        try:
            batch = int(self.batch.get())
        except Exception:
            batch = DEFAULT_BATCH

        original = {
            "backend": "original",
            "detected": None,
            "num_detections": None,
            "best_confidence": None,
            "bbox_xyxy": None,
            "class_id": None,
            "total_ms": None,
            "batch_used": None,
            "error": None,
            "resize_info": resize_info,
            "plotted_bgr": input_bgr.copy(),
        }

        results: dict[str, dict[str, Any]] = {
            "original": original,
            "pt": run_backend_detection(
                "pt",
                self.models.get("pt"),
                input_bgr,
                conf=conf,
                imgsz=imgsz,
                cls_id=cls_id,
                batch=1,
            ),
            "onnx": run_backend_detection(
                "onnx",
                self.models.get("onnx"),
                input_bgr,
                conf=conf,
                imgsz=imgsz,
                cls_id=cls_id,
                batch=batch,
            ),
            "engine": run_backend_detection(
                "engine",
                self.models.get("engine"),
                input_bgr,
                conf=conf,
                imgsz=imgsz,
                cls_id=cls_id,
                batch=batch,
            ),
        }

        try:
            low_iou = float(self.low_iou_threshold.get())
        except Exception:
            low_iou = 0.50
        comparisons = compare_results(results, low_iou_threshold=low_iou)
        for comp_key, comp in comparisons.items():
            backend = comp_key.replace("_vs_pt", "")
            if backend in results:
                results[backend]["iou_vs_pt"] = comp.get("iou")
                results[backend]["verdict_vs_pt"] = comp.get("verdict")
        if results["pt"].get("error"):
            results["pt"]["verdict_vs_pt"] = "PT_ERROR"
            results["pt"]["iou_vs_pt"] = None
        elif results["pt"].get("detected") and results["pt"].get("bbox_xyxy") is not None:
            results["pt"]["verdict_vs_pt"] = "REFERENCE"
            results["pt"]["iou_vs_pt"] = 1.0
        else:
            results["pt"]["verdict_vs_pt"] = "NO_DETECTION"
            results["pt"]["iou_vs_pt"] = None

        if token != self._render_token:
            return
        self.last_results = results
        self.last_comparisons = comparisons
        self._update_problem_cache(current_index, comparisons)
        self.after(0, lambda: self._set_verdict_text(comparisons))
        errors = []
        for b in ("pt", "onnx", "engine"):
            err = results.get(b, {}).get("error")
            if err:
                errors.append(f"{b.upper()}: {str(err).replace(chr(10), ' ')[:180]}")
        self.after(0, lambda errs=errors: self.error_text.set("Błędy: " + " | ".join(errs) if errs else "Błędy: —"))

        for idx, backend in enumerate(TILE_ORDER):
            plotted_bgr = results[backend]["plotted_bgr"]
            rgb = cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)
            rgb = self._apply_zoom_crop(idx, rgb)
            rgb = fit_to_box(rgb, sizes[idx][0], sizes[idx][1])
            tki = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.after(0, lambda i=idx, img=tki, b=backend, r=results[backend], name=img_path.name: self._update_tile(i, img, b, r, name))

    def _update_tile(self, idx: int, tkimg: ImageTk.PhotoImage, backend: str, result: dict[str, Any], image_name: str):
        self.tkimgs[idx] = tkimg
        cnv = self.canvas[idx]
        cnv.delete("all")
        cnv_w = cnv.winfo_width()
        cnv_h = cnv.winfo_height()
        img_w = tkimg.width()
        img_h = tkimg.height()
        x = max(0, (cnv_w - img_w) // 2)
        y = max(0, (cnv_h - img_h) // 2)
        cnv.create_image(x, y, image=tkimg, anchor="nw")
        self.display_sizes[idx] = (int(tkimg.width()), int(tkimg.height()))
        self.display_offsets[idx] = (int(x), int(y))
        self._update_tile_label(idx, backend, result, image_name)
        self._update_tile_border(idx, backend, result)

    def _update_tile_label(self, idx: int, backend: str, result: dict[str, Any], image_name: str):
        name = TILE_NAMES[backend]
        if backend == "original":
            hw = result.get("resize_info", {}).get("target_hw")
            hw_txt = f" | resized={hw[0]}x{hw[1]}" if hw else ""
            self.tile_labels[idx].config(text=f"{name} | {image_name}{hw_txt}")
            return

        detected = result.get("detected")
        conf = result.get("best_confidence")
        total_ms = result.get("total_ms")
        batch_used = result.get("batch_used")
        iou = result.get("iou_vs_pt")
        verdict = result.get("verdict_vs_pt")
        error = result.get("error")

        txt = f"{name} | detected={detected}"
        if conf is not None:
            txt += f" | conf={float(conf):.3f}"
        if total_ms is not None:
            txt += f" | {float(total_ms):.2f} ms"
        if batch_used:
            txt += f" | batch={batch_used}"
        if backend != "pt" and iou is not None:
            txt += f" | IoU vs PT={float(iou):.3f}"
        if verdict:
            txt += f" | {verdict}"
        if error:
            err_short = str(error).replace("\n", " ")[:120]
            txt += f" | ERROR: {err_short}"
        self.tile_labels[idx].config(text=txt)

    def _update_tile_border(self, idx: int, backend: str, result: dict[str, Any]):
        cnv = self.canvas[idx]
        if backend == "original":
            cnv.config(highlightbackground="#888888")
            return
        verdict = result.get("verdict_vs_pt")
        if verdict in ("OK", "REFERENCE"):
            cnv.config(highlightbackground="#2ecc71")
        elif verdict in ("MISSED_DETECTION", "LOW_IOU", "FAILED"):
            cnv.config(highlightbackground="#e74c3c")
        elif verdict == "EXTRA_DETECTION":
            cnv.config(highlightbackground="#f39c12")
        else:
            cnv.config(highlightbackground="#aaaaaa")

    # ------------------------------------------------------------
    # Export
    # ------------------------------------------------------------
    def export_compare(self):
        if self._export_running:
            return
        if not self.images:
            messagebox.showerror("Błąd", "Najpierw wybierz folder obrazów.")
            return
        if self.export_out_dir is None:
            messagebox.showerror("Błąd", "Najpierw wybierz folder wyjściowy.")
            return
        if self.export_mode.get() == "current":
            start = end = self.index
        else:
            try:
                start = int(self.export_from.get().strip())
                end = int(self.export_to.get().strip())
            except Exception:
                messagebox.showerror("Błąd", "Zakres musi być liczbami całkowitymi.")
                return

        mlen = self.max_len()
        start = max(0, min(start, mlen - 1))
        end = max(0, min(end, mlen - 1))
        if end < start:
            start, end = end, start

        self.autoload_static_backends()
        self._export_running = True
        self.export_text.set(f"Eksport trwa: {start}-{end} → {self.export_out_dir}")
        threading.Thread(target=self._export_worker, args=(start, end), daemon=True).start()

    def _export_worker(self, start: int, end: int):
        out_dir = self.export_out_dir
        if out_dir is None:
            return
        for sub in ("original", "pt", "onnx", "engine", "compare_json", "problem_crops_original"):
            (out_dir / sub).mkdir(parents=True, exist_ok=True)

        summary_csv = out_dir / "backend_compare_summary.csv"
        pairs_csv = out_dir / "backend_compare_pairs.csv"
        results_json = out_dir / "backend_compare_results.json"
        manifest_json = out_dir / "backend_compare_manifest.json"

        all_results: list[dict[str, Any]] = []
        summary_rows: list[dict[str, Any]] = []
        pair_rows: list[dict[str, Any]] = []
        total = end - start + 1
        ok, fail = 0, 0

        cls_id = self._read_cls_id()
        conf = float(self.conf.get())
        imgsz = self.selected_imgsz_for_ultralytics()
        try:
            low_iou = float(self.low_iou_threshold.get())
        except Exception:
            low_iou = 0.50
        try:
            batch = int(self.batch.get())
        except Exception:
            batch = DEFAULT_BATCH

        for n, idx in enumerate(range(start, end + 1), start=1):
            img_path = self._safe_get_imgpath(idx)
            if img_path is None:
                fail += 1
                continue
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                fail += 1
                continue

            result_bundle = self._run_compare_for_export(
                idx=idx,
                img_path=img_path,
                img_bgr=img_bgr,
                conf=conf,
                imgsz=imgsz,
                cls_id=cls_id,
                batch=batch,
                low_iou_threshold=low_iou,
                out_dir=out_dir,
            )
            all_results.append(result_bundle)
            summary_rows.extend(result_bundle["summary_rows"])
            pair_rows.extend(result_bundle["pair_rows"])
            ok += 1

            pct = 100.0 * n / max(1, total)
            self.after(0, lambda done=n, t=total, okc=ok, fl=fail, p=pct: self.export_text.set(f"Eksport: {done}/{t} ({p:.1f}%) | OK={okc} FAIL={fl} | {out_dir}"))

        self._write_summary_csv(summary_csv, summary_rows)
        self._write_pairs_csv(pairs_csv, pair_rows)
        results_json.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest_json.write_text(
            json.dumps(
                {
                    "generated_at_epoch_ns": time.time_ns(),
                    "images_dir": str(self.images_dir) if self.images_dir else None,
                    "out_dir": str(out_dir),
                    "start": int(start),
                    "end": int(end),
                    "resolution_label": self.resolution_label.get(),
                    "target_hw": list(self.selected_hw()),
                    "batch": int(batch),
                    "precision": self.precision.get(),
                    "conf": conf,
                    "cls_id": cls_id,
                    "low_iou_threshold": low_iou,
                    "model_paths": {k: (str(v) if v else None) for k, v in self.model_paths.items()},
                    "ONNX_STATIC_DIR": str(self.onnx_static_dir),
                    "ENGINE_STATIC_DIR": str(self.engine_static_dir),
                    "ONNX_STATIC_DIR_DEFAULT": str(ONNX_STATIC_DIR),
                    "ENGINE_STATIC_DIR_DEFAULT": str(ENGINE_STATIC_DIR),
                    "backend_batch_mode": f"repeat_same_image_to_b{int(batch)}_for_onnx_engine",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        self._export_running = False
        self.after(
            0,
            lambda: self.export_text.set(f"Eksport gotowy | OK={ok} FAIL={fail} | {out_dir}"),
        )

        def _done_prompt():
            ans = messagebox.askyesno("Eksport zakończony", f"Obrazów OK: {ok}\nBłędy: {fail}\n\nFolder:\n{out_dir}\n\nCzy otworzyć folder?")
            if ans and out_dir.exists():
                os.startfile(str(out_dir))

        self.after(0, _done_prompt)

    def _run_compare_for_export(
        self,
        *,
        idx: int,
        img_path: Path,
        img_bgr,
        conf: float,
        imgsz: int | tuple[int, int],
        cls_id: int | None,
        batch: int,
        low_iou_threshold: float,
        out_dir: Path,
    ) -> dict[str, Any]:
        input_bgr, resize_info = resize_input_for_backend(img_bgr, self.selected_hw())
        original_bgr = img_bgr
        original = {
            "backend": "original",
            "detected": None,
            "num_detections": None,
            "best_confidence": None,
            "bbox_xyxy": None,
            "class_id": None,
            "total_ms": None,
            "batch_used": None,
            "error": None,
            "resize_info": resize_info,
            "plotted_bgr": input_bgr.copy(),
        }
        results: dict[str, dict[str, Any]] = {
            "original": original,
            "pt": run_backend_detection("pt", self.models.get("pt"), input_bgr, conf=conf, imgsz=imgsz, cls_id=cls_id, batch=1),
            "onnx": run_backend_detection("onnx", self.models.get("onnx"), input_bgr, conf=conf, imgsz=imgsz, cls_id=cls_id, batch=batch),
            "engine": run_backend_detection("engine", self.models.get("engine"), input_bgr, conf=conf, imgsz=imgsz, cls_id=cls_id, batch=batch),
        }
        comparisons = compare_results(results, low_iou_threshold=low_iou_threshold)
        for comp_key, comp in comparisons.items():
            backend = comp_key.replace("_vs_pt", "")
            if backend in results:
                results[backend]["iou_vs_pt"] = comp.get("iou")
                results[backend]["verdict_vs_pt"] = comp.get("verdict")
        if results["pt"].get("error"):
            results["pt"]["verdict_vs_pt"] = "PT_ERROR"
            results["pt"]["iou_vs_pt"] = None
        elif results["pt"].get("detected") and results["pt"].get("bbox_xyxy") is not None:
            results["pt"]["verdict_vs_pt"] = "REFERENCE"
            results["pt"]["iou_vs_pt"] = 1.0
        else:
            results["pt"]["verdict_vs_pt"] = "NO_DETECTION"
            results["pt"]["iou_vs_pt"] = None
        self._update_problem_cache(idx, comparisons)

        name = safe_filename(img_path)
        cv2.imwrite(str(out_dir / "original" / name), input_bgr)
        for backend in BACKENDS:
            plotted = results[backend].get("plotted_bgr")
            if plotted is not None:
                cv2.imwrite(str(out_dir / backend / name), plotted)

        problem_crops_original: list[dict[str, Any]] = []
        for backend in ("onnx", "engine"):
            verdict = str(results.get(backend, {}).get("verdict_vs_pt") or "")
            if verdict not in ("LOW_IOU", "MISSED_DETECTION", "FAILED", "EXTRA_DETECTION"):
                continue
            pt_box_original = map_box_from_target_to_original(results.get("pt", {}).get("bbox_xyxy"), resize_info)
            pred_box_original = map_box_from_target_to_original(results.get(backend, {}).get("bbox_xyxy"), resize_info)
            crop_box_original = select_problem_crop_box(
                [pt_box_original] if pt_box_original is not None else [],
                [pred_box_original] if pred_box_original is not None else [],
                verdict,
            )
            if crop_box_original is None:
                continue
            crop_pack = crop_original_with_margin(original_bgr, crop_box_original, margin=40)
            if crop_pack is None:
                continue
            crop_bgr = crop_pack["crop_bgr"]
            cx1, cy1, _, _ = crop_pack["crop_box_original_xyxy"]
            if pt_box_original is not None:
                x1, y1, x2, y2 = [int(round(v)) for v in pt_box_original]
                cv2.rectangle(crop_bgr, (x1 - cx1, y1 - cy1), (x2 - cx1, y2 - cy1), (0, 200, 0), 2)
            if pred_box_original is not None:
                x1, y1, x2, y2 = [int(round(v)) for v in pred_box_original]
                cv2.rectangle(crop_bgr, (x1 - cx1, y1 - cy1), (x2 - cx1, y2 - cy1), (255, 255, 0), 2)
            iou_val = comparisons.get(f"{backend}_vs_pt", {}).get("iou")
            conf_val = results.get(backend, {}).get("best_confidence")
            title = f"{backend.upper()} {verdict} IoU={_fmt(iou_val)} conf={_fmt(conf_val,2)}"
            cv2.putText(crop_bgr, title[:180], (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            crop_name = f"{backend.upper()}__{verdict}__{img_path.stem}__orig_crop.png"
            crop_path = out_dir / "problem_crops_original" / crop_name
            cv2.imwrite(str(crop_path), crop_bgr)
            problem_crops_original.append(
                {
                    "backend": backend,
                    "verdict": verdict,
                    "original_crop_path": str(crop_path),
                    "crop_box_original_xyxy": crop_pack["crop_box_original_xyxy"],
                    "source_hw": resize_info.get("source_hw"),
                    "target_hw": resize_info.get("target_hw"),
                    "scale_x": resize_info.get("scale_x"),
                    "scale_y": resize_info.get("scale_y"),
                }
            )

        clean_results = strip_image_arrays(results)
        one_json = {
            "image": str(img_path),
            "image_name": img_path.name,
            "params": {
                "conf": conf,
                "imgsz": imgsz,
                "cls_id": cls_id,
                "resolution_label": self.resolution_label.get(),
                "target_hw": list(self.selected_hw()),
                "batch": int(batch),
                "precision": self.precision.get(),
                "resize_mode": "direct_resize_no_letterbox",
                "backend_batch_mode": f"repeat_same_image_to_b{int(batch)}_for_onnx_engine",
                "low_iou_threshold": float(self.low_iou_threshold.get()) if hasattr(self, "low_iou_threshold") else 0.5,
            },
            "resize_info": resize_info,
            "models": {
                "pt": str(self.model_paths.get("pt")) if self.model_paths.get("pt") else None,
                "onnx": str(self.model_paths.get("onnx")) if self.model_paths.get("onnx") else None,
                "engine": str(self.model_paths.get("engine")) if self.model_paths.get("engine") else None,
            },
            "results": clean_results,
            "comparisons": comparisons,
            "problem_crops_original": problem_crops_original,
        }

        json_path = out_dir / "compare_json" / f"{img_path.stem}.json"
        json_path.write_text(json.dumps(one_json, ensure_ascii=False, indent=2), encoding="utf-8")

        summary_rows: list[dict[str, Any]] = []
        for backend in BACKENDS:
            r = clean_results[backend]
            bbox = r.get("bbox_xyxy") or [None, None, None, None]
            summary_rows.append(
                {
                    "image": img_path.name,
                    "backend": backend,
                    "detected": r.get("detected"),
                    "num_detections": r.get("num_detections"),
                    "best_confidence": r.get("best_confidence"),
                    "total_ms": r.get("total_ms"),
                    "batch_used": r.get("batch_used"),
                    "bbox_x1": bbox[0],
                    "bbox_y1": bbox[1],
                    "bbox_x2": bbox[2],
                    "bbox_y2": bbox[3],
                    "iou_vs_pt": r.get("iou_vs_pt"),
                    "verdict_vs_pt": r.get("verdict_vs_pt"),
                    "error": r.get("error"),
                }
            )

        pair_rows: list[dict[str, Any]] = []
        for comp_name, comp in comparisons.items():
            row = dict(comp)
            row["image"] = img_path.name
            row["comparison"] = comp_name
            pair_rows.append(row)

        return {"image": str(img_path), "json_path": str(json_path), "summary_rows": summary_rows, "pair_rows": pair_rows, "result": one_json}

    @staticmethod
    def _write_summary_csv(path: Path, rows: list[dict[str, Any]]):
        cols = [
            "image",
            "backend",
            "detected",
            "num_detections",
            "best_confidence",
            "total_ms",
            "batch_used",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "iou_vs_pt",
            "verdict_vs_pt",
            "error",
        ]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    @staticmethod
    def _write_pairs_csv(path: Path, rows: list[dict[str, Any]]):
        cols = ["image", "comparison", "matched", "iou", "conf_ref", "conf_test", "confidence_diff", "missed_detection", "extra_detection", "verdict", "reason"]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


if __name__ == "__main__":
    app = App()
    app.mainloop()
