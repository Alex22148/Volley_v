# save as: numeric_detection_audit_dashboard.py
# pip install ultralytics opencv-python pillow matplotlib

from __future__ import annotations

import csv
import json
import math
import os
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    from ultralytics import YOLO
except Exception:  # handled in GUI
    YOLO = None  # type: ignore


# ============================================================
# CONFIG
# ============================================================

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
BACKENDS = ("pt", "onnx", "engine")

PKG_ROOT = Path(__file__).resolve().parent
DEFAULT_ONNX_STATIC_DIR = PKG_ROOT / "fastpath_lab_benchmark" / "engines" / "onnx" / "static"
DEFAULT_ENGINE_STATIC_DIR = PKG_ROOT / "fastpath_lab_benchmark" / "engines" / "static"

RESOLUTION_PRESETS: dict[str, tuple[int, int]] = {
    "v640": (640, 640),
    "v960": (960, 960),
    "v1088x1920": (1088, 1920),
}
AVAILABLE_BATCHES = [1, 4, 8, 12, 16, 20, 24, 28, 32, 64]
AVAILABLE_PRECISIONS = ["fp16", "fp32"]

PROBLEM_VERDICTS = {"FAILED", "MISSED", "FALSE_POSITIVE", "PARTIAL", "LOW_IOU"}


# ============================================================
# SMALL HELPERS
# ============================================================

def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def fmt(v: Any, p: int = 3) -> str:
    try:
        if v is None:
            return "-"
        x = float(v)
        if math.isnan(x):
            return "-"
        return f"{x:.{p}f}"
    except Exception:
        return "-"


def list_images(folder: Path) -> list[Path]:
    files: list[Path] = []
    for ext in IMG_EXTS:
        files.extend(folder.rglob(f"*{ext}"))
    files.sort()
    return files


def resize_input_for_backend(img_bgr, hw: tuple[int, int]):
    target_h, target_w = int(hw[0]), int(hw[1])
    src_h, src_w = img_bgr.shape[:2]
    resized = cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return resized, {
        "source_hw": [int(src_h), int(src_w)],
        "target_hw": [target_h, target_w],
        "scale_y": float(target_h / src_h) if src_h else None,
        "scale_x": float(target_w / src_w) if src_w else None,
        "resize_mode": "direct_resize_no_letterbox",
    }


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
    return None if union <= 0 else float(inter / union)


def engine_name_for(label: str, batch: int, precision: str) -> str:
    return f"best__{precision}_{label}_b{batch}.engine"


def onnx_name_for(label: str, batch: int, precision: str) -> str:
    return f"best__{precision}_{label}_b{batch}.onnx"


# ============================================================
# LABEL HELPERS
# ============================================================

def yolo_xywhn_to_xyxy(class_id: int, xc: float, yc: float, w: float, h: float, target_hw: tuple[int, int]) -> dict[str, Any]:
    target_h, target_w = target_hw
    x1 = (xc - w / 2.0) * target_w
    y1 = (yc - h / 2.0) * target_h
    x2 = (xc + w / 2.0) * target_w
    y2 = (yc + h / 2.0) * target_h
    return {
        "class_id": int(class_id),
        "bbox_xyxy": [
            max(0.0, float(x1)),
            max(0.0, float(y1)),
            min(float(target_w - 1), float(x2)),
            min(float(target_h - 1), float(y2)),
        ],
    }


def load_yolo_labels(label_path: Path, target_hw: tuple[int, int], cls_id: int | None) -> tuple[list[dict[str, Any]], int]:
    if not label_path.exists():
        return [], 0
    gts: list[dict[str, Any]] = []
    warnings = 0
    text = label_path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return [], 0
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            warnings += 1
            continue
        try:
            c = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:5])
            if cls_id is not None and c != int(cls_id):
                continue
            gts.append(yolo_xywhn_to_xyxy(c, xc, yc, w, h, target_hw))
        except Exception:
            warnings += 1
    return gts, warnings


# ============================================================
# DETECTION
# ============================================================

def _extract_all_detections(result, cls_id: int | None) -> list[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    detections: list[dict[str, Any]] = []
    try:
        confs = boxes.conf.detach().cpu().numpy()
        xyxy = boxes.xyxy.detach().cpu().numpy()
        cls = boxes.cls.detach().cpu().numpy()
        for i in range(len(xyxy)):
            c = int(cls[i])
            if cls_id is not None and c != int(cls_id):
                continue
            detections.append({
                "class_id": c,
                "confidence": float(confs[i]),
                "bbox_xyxy": [float(x) for x in xyxy[i].tolist()],
            })
    except Exception:
        return []
    detections.sort(key=lambda d: float(d.get("confidence") or 0.0), reverse=True)
    return detections


def run_backend_detections_full(
    backend_name: str,
    model: Any,
    img_bgr,
    *,
    conf: float,
    iou_nms: float,
    imgsz: int | tuple[int, int],
    cls_id: int | None,
    batch: int = 1,
) -> dict[str, Any]:
    if model is None:
        return {
            "backend": backend_name,
            "detections": [],
            "detected": False,
            "num_detections": 0,
            "best_confidence": None,
            "total_ms": None,
            "total_ms_norm_per_image": None,
            "batch_used": 0,
            "error": "model_not_loaded",
        }

    try:
        batch = max(1, int(batch))
    except Exception:
        batch = 1

    if backend_name in ("onnx", "engine"):
        predict_input = [img_bgr for _ in range(batch)]
        batch_used = batch
    else:
        predict_input = img_bgr
        batch_used = 1

    t0 = time.perf_counter()
    try:
        results = model.predict(
            predict_input,
            conf=float(conf),
            iou=float(iou_nms),
            imgsz=imgsz,
            verbose=False,
        )
        total_ms = (time.perf_counter() - t0) * 1000.0
        if not results:
            return {
                "backend": backend_name,
                "detections": [],
                "detected": False,
                "num_detections": 0,
                "best_confidence": None,
                "total_ms": total_ms,
                "total_ms_norm_per_image": total_ms / batch_used if batch_used else None,
                "batch_used": batch_used,
                "error": "no_result",
            }
        detections = _extract_all_detections(results[0], cls_id)
        best_conf = detections[0]["confidence"] if detections else None
        return {
            "backend": backend_name,
            "detections": detections,
            "detected": bool(detections),
            "num_detections": len(detections),
            "best_confidence": best_conf,
            "total_ms": total_ms,
            "total_ms_norm_per_image": total_ms / batch_used if batch_used else None,
            "batch_used": batch_used,
            "error": None,
        }
    except Exception as exc:
        total_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "backend": backend_name,
            "detections": [],
            "detected": False,
            "num_detections": 0,
            "best_confidence": None,
            "total_ms": total_ms,
            "total_ms_norm_per_image": total_ms / batch_used if batch_used else None,
            "batch_used": batch_used,
            "error": str(exc),
        }


# ============================================================
# MATCHING / METRICS
# ============================================================

def match_detections_to_gt(preds: list[dict[str, Any]], gts: list[dict[str, Any]], iou_threshold: float) -> dict[str, Any]:
    candidates: list[tuple[float, int, int]] = []
    for pi, pred in enumerate(preds):
        for gi, gt in enumerate(gts):
            if int(pred.get("class_id", -1)) != int(gt.get("class_id", -2)):
                continue
            iou = bbox_iou_xyxy(pred.get("bbox_xyxy"), gt.get("bbox_xyxy"))
            if iou is not None:
                candidates.append((float(iou), pi, gi))
    candidates.sort(reverse=True, key=lambda x: x[0])

    used_pred: set[int] = set()
    used_gt: set[int] = set()
    matches: list[dict[str, Any]] = []

    for iou, pi, gi in candidates:
        if iou < iou_threshold:
            continue
        if pi in used_pred or gi in used_gt:
            continue
        used_pred.add(pi)
        used_gt.add(gi)
        pred = preds[pi]
        matches.append({
            "gt_index": gi,
            "pred_index": pi,
            "iou": float(iou),
            "confidence": pred.get("confidence"),
            "class_id": pred.get("class_id"),
        })

    unmatched_pred = [i for i in range(len(preds)) if i not in used_pred]
    unmatched_gt = [i for i in range(len(gts)) if i not in used_gt]
    return {
        "tp": len(matches),
        "fp": len(unmatched_pred),
        "fn": len(unmatched_gt),
        "matches": matches,
        "unmatched_gt": unmatched_gt,
        "unmatched_pred": unmatched_pred,
    }


def make_numeric_verdict(gt_count: int, pred_count: int, tp: int, fp: int, fn: int, error: str | None) -> str:
    if error:
        return "FAILED"
    if gt_count == 0 and pred_count == 0:
        return "NO_GT_NO_PRED"
    if fp == 0 and fn == 0:
        return "OK"
    if gt_count > 0 and tp == 0:
        return "MISSED"
    if gt_count == 0 and pred_count > 0:
        return "FALSE_POSITIVE"
    return "PARTIAL"


def build_per_image_row(image_name: str, backend: str, gt_count: int, detection_result: dict[str, Any], match: dict[str, Any]) -> dict[str, Any]:
    tp, fp, fn = int(match["tp"]), int(match["fp"]), int(match["fn"])
    pred_count = int(detection_result.get("num_detections") or 0)
    p = safe_div(tp, tp + fp)
    r = safe_div(tp, tp + fn)
    f1 = safe_div(2.0 * p * r, p + r)
    ious = [float(m["iou"]) for m in match.get("matches", [])]
    return {
        "image": image_name,
        "backend": backend,
        "gt_count": gt_count,
        "pred_count": pred_count,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision_image": p,
        "recall_image": r,
        "f1_image": f1,
        "best_iou": max(ious) if ious else None,
        "mean_iou": mean(ious),
        "best_confidence": detection_result.get("best_confidence"),
        "total_ms": detection_result.get("total_ms"),
        "total_ms_norm_per_image": detection_result.get("total_ms_norm_per_image"),
        "batch_used": detection_result.get("batch_used"),
        "error": detection_result.get("error"),
        "verdict": make_numeric_verdict(gt_count, pred_count, tp, fp, fn, detection_result.get("error")),
    }


def summarize_backend(rows: list[dict[str, Any]], backend: str) -> dict[str, Any]:
    br = [r for r in rows if r["backend"] == backend]
    times = [float(r["total_ms"]) for r in br if r.get("total_ms") is not None]
    times_norm = [float(r["total_ms_norm_per_image"]) for r in br if r.get("total_ms_norm_per_image") is not None]
    matched_ious = [float(r["best_iou"]) for r in br if r.get("best_iou") is not None]

    tp = sum(int(r["tp"]) for r in br)
    fp = sum(int(r["fp"]) for r in br)
    fn = sum(int(r["fn"]) for r in br)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    total_ms_mean = mean(times)
    total_ms_p95 = percentile(times, 95)

    conf_tp: list[float] = []
    conf_fp: list[float] = []
    # optional in this simple summary: best_conf is image-level, not match-level
    for r in br:
        if r.get("best_confidence") is not None:
            if int(r.get("tp") or 0) > 0:
                conf_tp.append(float(r["best_confidence"]))
            elif int(r.get("fp") or 0) > 0:
                conf_fp.append(float(r["best_confidence"]))

    images_count = len(br)
    error_count = sum(1 for r in br if r.get("error"))
    missed_images_count = sum(1 for r in br if r.get("verdict") == "MISSED")
    fp_images_count = sum(1 for r in br if r.get("verdict") == "FALSE_POSITIVE")
    partial_images_count = sum(1 for r in br if r.get("verdict") == "PARTIAL")
    perfect_images_count = sum(1 for r in br if r.get("verdict") in ("OK", "NO_GT_NO_PRED"))

    if images_count == 0:
        verdict = "NO_DATA"
    elif error_count > 0.10 * images_count:
        verdict = "FAIL_RUNTIME"
    elif recall < 0.80 or f1 < 0.80:
        verdict = "FAIL_QUALITY"
    elif recall < 0.90 or f1 < 0.90:
        verdict = "PASS_WITH_WARNINGS"
    else:
        verdict = "PASS"

    return {
        "backend": backend,
        "images_count": images_count,
        "ok_count": images_count - error_count,
        "error_count": error_count,
        "total_ms_mean": total_ms_mean,
        "total_ms_median": percentile(times, 50),
        "total_ms_p95": total_ms_p95,
        "total_ms_min": min(times) if times else None,
        "total_ms_max": max(times) if times else None,
        "fps_mean": 1000.0 / total_ms_mean if total_ms_mean else None,
        "fps_p95_safe": 1000.0 / total_ms_p95 if total_ms_p95 else None,
        "total_ms_norm_per_image_mean": mean(times_norm),
        "total_ms_norm_per_image_p95": percentile(times_norm, 95),
        "gt_count": sum(int(r["gt_count"]) for r in br),
        "pred_count": sum(int(r["pred_count"]) for r in br),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou_matched": mean(matched_ious),
        "median_iou_matched": percentile(matched_ious, 50),
        "p95_iou_matched": percentile(matched_ious, 95),
        "mean_confidence_tp": mean(conf_tp),
        "mean_confidence_fp": mean(conf_fp),
        "missed_images_count": missed_images_count,
        "false_positive_images_count": fp_images_count,
        "partial_images_count": partial_images_count,
        "perfect_images_count": perfect_images_count,
        "failed_images_count": error_count,
        "verdict": verdict,
    }


# ============================================================
# CHARTS / REPORT / OVERLAYS
# ============================================================

def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def generate_charts(summary_rows: list[dict[str, Any]], charts_dir: Path) -> list[str]:
    chart_files: list[str] = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return chart_files

    charts_dir.mkdir(parents=True, exist_ok=True)
    labels = [r["backend"].upper() for r in summary_rows]

    def bar(values: list[float], title: str, ylabel: str, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(labels, values)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(charts_dir / filename, dpi=160)
        plt.close(fig)
        chart_files.append(filename)

    bar([float(r.get("total_ms_p95") or 0) for r in summary_rows], "p95 czasu całkowitego", "ms", "chart_01_time_p95_backend.png")
    bar([float(r.get("fps_p95_safe") or 0) for r in summary_rows], "FPS safe z p95", "FPS", "chart_02_fps_safe_backend.png")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = list(range(len(labels)))
    width = 0.25
    ax.bar([i - width for i in x], [float(r.get("precision") or 0) for r in summary_rows], width, label="Precision")
    ax.bar(x, [float(r.get("recall") or 0) for r in summary_rows], width, label="Recall")
    ax.bar([i + width for i in x], [float(r.get("f1") or 0) for r in summary_rows], width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_title("Precision / Recall / F1")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(charts_dir / "chart_03_precision_recall_f1.png", dpi=160)
    plt.close(fig)
    chart_files.append("chart_03_precision_recall_f1.png")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar([i - width for i in x], [int(r.get("tp") or 0) for r in summary_rows], width, label="TP")
    ax.bar(x, [int(r.get("fp") or 0) for r in summary_rows], width, label="FP")
    ax.bar([i + width for i in x], [int(r.get("fn") or 0) for r in summary_rows], width, label="FN")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("TP / FP / FN")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(charts_dir / "chart_04_tp_fp_fn.png", dpi=160)
    plt.close(fig)
    chart_files.append("chart_04_tp_fp_fn.png")

    bar([float(r.get("mean_iou_matched") or 0) for r in summary_rows], "Średni IoU dopasowanych detekcji", "IoU", "chart_05_mean_iou.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    xs = [float(r.get("total_ms_p95") or 0) for r in summary_rows]
    ys = [float(r.get("f1") or 0) for r in summary_rows]
    ax.scatter(xs, ys)
    for x0, y0, label in zip(xs, ys, labels):
        ax.annotate(label, (x0, y0), xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("p95 total_ms [ms]")
    ax.set_ylabel("F1")
    ax.set_title("Speed vs quality")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(charts_dir / "chart_06_quality_vs_speed.png", dpi=160)
    plt.close(fig)
    chart_files.append("chart_06_quality_vs_speed.png")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar([i - width for i in x], [int(r.get("missed_images_count") or 0) for r in summary_rows], width, label="MISSED")
    ax.bar(x, [int(r.get("false_positive_images_count") or 0) for r in summary_rows], width, label="FP images")
    ax.bar([i + width for i in x], [int(r.get("partial_images_count") or 0) for r in summary_rows], width, label="PARTIAL")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Problematyczne obrazy")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(charts_dir / "chart_07_problem_images.png", dpi=160)
    plt.close(fig)
    chart_files.append("chart_07_problem_images.png")
    return chart_files


def draw_problem_overlay(img_bgr, gts: list[dict[str, Any]], preds: list[dict[str, Any]], title: str) -> Any:
    out = img_bgr.copy()
    for gt in gts:
        x1, y1, x2, y2 = map(int, gt["bbox_xyxy"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(out, f"GT c{gt['class_id']}", (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)
    for pred in preds:
        x1, y1, x2, y2 = map(int, pred["bbox_xyxy"])
        conf = float(pred.get("confidence") or 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 80, 255), 2)
        cv2.putText(out, f"P c{pred['class_id']} {conf:.2f}", (x1, min(out.shape[0] - 8, y2 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 255), 2)
    cv2.putText(out, title[:150], (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
    cv2.putText(out, title[:150], (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 1)
    return out


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


def write_markdown_report(report_path: Path, config: dict[str, Any], summary_rows: list[dict[str, Any]], chart_files: list[str], worst_rows: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    a = lines.append
    a("# Numeryczny audyt detekcji — dashboard report")
    a("")
    a("## 1. Cel analizy")
    a("Raport porównuje modele PT / ONNX / ENGINE pod względem szybkości inferencji oraz jakości detekcji względem referencyjnych labeli YOLO.")
    a("")
    a("## 2. Dane wejściowe i konfiguracja")
    for k, v in config.items():
        a(f"- **{k}**: `{v}`")
    a("")
    a("## 3. Jak czytać metryki")
    a("| Metryka | Znaczenie | Dobra wartość |")
    a("|---|---|---|")
    a("| p95 total_ms | czas, poniżej którego mieści się 95% predykcji | im mniej, tym lepiej; dla live ważniejsze niż średnia |")
    a("| FPS safe | 1000 / p95 total_ms | im więcej, tym lepiej |")
    a("| Precision | TP / (TP + FP), odporność na fałszywe detekcje | blisko 1.0 |")
    a("| Recall | TP / (TP + FN), zdolność niegubienia obiektu | dla piłki zwykle krytyczne; blisko 1.0 |")
    a("| F1 | kompromis precision i recall | blisko 1.0 |")
    a("| IoU | zgodność bboxa z referencją GT | >0.5 akceptowalne, >0.75 dobre |")
    a("| FN / MISSED | obiekt był w labelu, ale model go nie wykrył | jak najmniej |")
    a("| FP | model wykrył coś, czego nie ma w labelu | jak najmniej |")
    a("")
    a("## 4. Podsumowanie backendów")
    a("| Backend | p95 ms | FPS safe | Precision | Recall | F1 | TP | FP | FN | Mean IoU | Verdict |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in summary_rows:
        a(f"| {r['backend'].upper()} | {fmt(r.get('total_ms_p95'),2)} | {fmt(r.get('fps_p95_safe'),1)} | {fmt(r.get('precision'),3)} | {fmt(r.get('recall'),3)} | {fmt(r.get('f1'),3)} | {r.get('tp')} | {r.get('fp')} | {r.get('fn')} | {fmt(r.get('mean_iou_matched'),3)} | {r.get('verdict')} |")
    a("")
    a("## 5. Wykresy")
    if chart_files:
        for f in chart_files:
            a(f"![{f}](numeric_analysis_charts/{f})")
            a("")
    else:
        a("Wykresy pominięte — brak matplotlib albo błąd generowania wykresów.")
        a("")
    a("## 6. Najważniejsze wnioski automatyczne")
    if summary_rows:
        best_f1 = max(summary_rows, key=lambda r: float(r.get("f1") or 0))
        fastest = min(summary_rows, key=lambda r: float(r.get("total_ms_p95") or 1e18))
        a(f"- Najlepszy F1: **{best_f1['backend'].upper()}** = {fmt(best_f1.get('f1'),3)}.")
        a(f"- Najszybszy backend według p95: **{fastest['backend'].upper()}** = {fmt(fastest.get('total_ms_p95'),2)} ms.")
        pt = next((r for r in summary_rows if r["backend"] == "pt"), None)
        eng = next((r for r in summary_rows if r["backend"] == "engine"), None)
        if pt and eng and pt.get("total_ms_p95") and eng.get("total_ms_p95"):
            speedup = float(pt["total_ms_p95"]) / max(1e-9, float(eng["total_ms_p95"]))
            recall_drop = float(pt.get("recall") or 0) - float(eng.get("recall") or 0)
            a(f"- ENGINE speed-up vs PT według p95: **{speedup:.2f}x**.")
            a(f"- Różnica recall PT - ENGINE: **{recall_drop:.3f}**.")
    a("")
    a("## 7. Najgorsze przypadki")
    a("| Image | Backend | Verdict | GT | Pred | TP | FP | FN | Best IoU |")
    a("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for r in worst_rows[:30]:
        a(f"| {r.get('image')} | {r.get('backend')} | {r.get('verdict')} | {r.get('gt_count')} | {r.get('pred_count')} | {r.get('tp')} | {r.get('fp')} | {r.get('fn')} | {fmt(r.get('best_iou'),3)} |")
    a("")
    a("## 8. Pliki wyjściowe")
    a("- `numeric_analysis_results.json` — pełne wyniki.")
    a("- `numeric_analysis_per_image.csv` — wynik per obraz/backend.")
    a("- `numeric_analysis_summary.csv` — podsumowanie backendów.")
    a("- `numeric_analysis_charts/` — wykresy.")
    a("- `problem_overlays/` — obrazy problematyczne GT vs predykcje.")
    a("- `problem_crops_original/` — cropy problemów wycięte z oryginalnej rozdzielczości.")
    a("")
    a("## 9. Zoom z oryginału — analiza problematycznych przypadków")
    a("Overlay pokazuje obraz po resize/inferencji i jest dobry do szybkiego debugowania geometrii detekcji.")
    a("Crop original pokazuje wycinek z pełnej rozdzielczości źródła, więc lepiej oddaje małe obiekty, rozmycia, błędne labele i artefakty resize.")
    a("W praktyce oba widoki są komplementarne: overlay do kontekstu, crop original do jakościowej oceny szczegółu.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================
# GUI
# ============================================================

class NumericAuditDashboard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Numeric Detection Audit Dashboard — PT / ONNX / ENGINE")
        self.geometry("1460x920")

        self.images_dir: Path | None = None
        self.labels_dir: Path | None = None
        self.report_root: Path | None = None
        self.last_report_dir: Path | None = None

        self.models: dict[str, Any] = {"pt": None, "onnx": None, "engine": None}
        self.model_paths: dict[str, Path | None] = {"pt": None, "onnx": None, "engine": None}

        self.resolution_label = tk.StringVar(value="v640")
        self.batch = tk.IntVar(value=4)
        self.precision = tk.StringVar(value="fp16")
        self.conf = tk.DoubleVar(value=0.25)
        self.iou_nms = tk.DoubleVar(value=0.45)
        self.iou_match = tk.DoubleVar(value=0.50)
        self.cls_id = tk.StringVar(value="0")
        self.onnx_dir = tk.StringVar(value=str(DEFAULT_ONNX_STATIC_DIR))
        self.engine_dir = tk.StringVar(value=str(DEFAULT_ENGINE_STATIC_DIR))

        self.use_backend = {b: tk.BooleanVar(value=True) for b in BACKENDS}
        self.running = False
        self.stop_requested = False

        self.status_text = tk.StringVar(value="Wybierz images, labels, modele i folder raportu.")
        self.progress_text = tk.StringVar(value="Postęp: —")
        self.report_text = tk.StringVar(value="Raport: —")

        self._build_ui()

    # ---------- selection ----------
    def selected_hw(self) -> tuple[int, int]:
        return RESOLUTION_PRESETS.get(self.resolution_label.get(), RESOLUTION_PRESETS["v640"])

    def selected_imgsz(self) -> int | tuple[int, int]:
        h, w = self.selected_hw()
        return int(h) if h == w else (int(h), int(w))

    def selected_cls_id(self) -> int | None:
        txt = self.cls_id.get().strip()
        if not txt:
            return None
        try:
            return int(txt)
        except Exception:
            return None

    # ---------- UI ----------
    def _build_ui(self):
        root = ttk.Frame(self)
        root.pack(fill="both", expand=True, padx=10, pady=10)

        top = ttk.LabelFrame(root, text="Wejścia i modele")
        top.pack(fill="x", pady=(0, 8))

        ttk.Button(top, text="Folder images", command=self.pick_images).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Button(top, text="Folder labels", command=self.pick_labels).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ttk.Button(top, text="Folder raportu", command=self.pick_report_root).grid(row=0, column=2, padx=5, pady=5, sticky="w")
        ttk.Button(top, text="Model PT", command=lambda: self.pick_model("pt")).grid(row=0, column=3, padx=5, pady=5, sticky="w")
        ttk.Button(top, text="Model ONNX", command=lambda: self.pick_model("onnx")).grid(row=0, column=4, padx=5, pady=5, sticky="w")
        ttk.Button(top, text="Model ENGINE", command=lambda: self.pick_model("engine")).grid(row=0, column=5, padx=5, pady=5, sticky="w")
        ttk.Button(top, text="Auto ONNX/ENGINE", command=self.autoload_static_backends).grid(row=0, column=6, padx=5, pady=5, sticky="w")
        ttk.Button(top, text="Diagnoza", command=self.show_diagnostics).grid(row=0, column=7, padx=5, pady=5, sticky="w")

        cfg = ttk.LabelFrame(root, text="Konfiguracja analizy")
        cfg.pack(fill="x", pady=(0, 8))

        ttk.Label(cfg, text="resolution").grid(row=0, column=0, padx=5, pady=4)
        ttk.Combobox(cfg, textvariable=self.resolution_label, values=list(RESOLUTION_PRESETS.keys()), state="readonly", width=13).grid(row=0, column=1, padx=5, pady=4)
        ttk.Label(cfg, text="batch ONNX/ENGINE").grid(row=0, column=2, padx=5, pady=4)
        ttk.Combobox(cfg, textvariable=self.batch, values=AVAILABLE_BATCHES, state="readonly", width=8).grid(row=0, column=3, padx=5, pady=4)
        ttk.Label(cfg, text="precision").grid(row=0, column=4, padx=5, pady=4)
        ttk.Combobox(cfg, textvariable=self.precision, values=AVAILABLE_PRECISIONS, state="readonly", width=8).grid(row=0, column=5, padx=5, pady=4)
        ttk.Label(cfg, text="conf").grid(row=0, column=6, padx=5, pady=4)
        ttk.Entry(cfg, textvariable=self.conf, width=7).grid(row=0, column=7, padx=5, pady=4)
        ttk.Label(cfg, text="IoU NMS").grid(row=0, column=8, padx=5, pady=4)
        ttk.Entry(cfg, textvariable=self.iou_nms, width=7).grid(row=0, column=9, padx=5, pady=4)
        ttk.Label(cfg, text="IoU match GT").grid(row=0, column=10, padx=5, pady=4)
        ttk.Entry(cfg, textvariable=self.iou_match, width=7).grid(row=0, column=11, padx=5, pady=4)
        ttk.Label(cfg, text="class_id (puste=all)").grid(row=0, column=12, padx=5, pady=4)
        ttk.Entry(cfg, textvariable=self.cls_id, width=7).grid(row=0, column=13, padx=5, pady=4)

        for i, b in enumerate(BACKENDS):
            ttk.Checkbutton(cfg, text=b.upper(), variable=self.use_backend[b]).grid(row=1, column=i, padx=8, pady=4, sticky="w")

        controls = ttk.Frame(root)
        controls.pack(fill="x", pady=(0, 8))
        ttk.Button(controls, text="Uruchom analizę", command=self.start_analysis).pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="Stop", command=self.stop_analysis).pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="Otwórz raport", command=self.open_report).pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="Otwórz folder raportu", command=self.open_report_folder).pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="Otwórz cropy original", command=self.open_original_crops_folder).pack(side="left", padx=(0, 8))

        self.progress = ttk.Progressbar(root, orient="horizontal", mode="determinate")
        self.progress.pack(fill="x", pady=(0, 4))
        ttk.Label(root, textvariable=self.progress_text).pack(fill="x")
        ttk.Label(root, textvariable=self.status_text).pack(fill="x")
        ttk.Label(root, textvariable=self.report_text).pack(fill="x", pady=(0, 8))

        dashboard = ttk.Frame(root)
        dashboard.pack(fill="both", expand=True)
        left = ttk.Frame(dashboard)
        left.pack(side="left", fill="both", expand=True)
        right = ttk.LabelFrame(dashboard, text="Jak czytać wyniki")
        right.pack(side="right", fill="y", padx=(8, 0))

        cols = ("Backend", "Images", "Errors", "P95 ms", "FPS safe", "Precision", "Recall", "F1", "TP", "FP", "FN", "Mean IoU", "Missed", "FP imgs", "Verdict")
        self.tree = ttk.Treeview(left, columns=cols, show="headings", height=12)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=85, anchor="center")
        self.tree.pack(fill="both", expand=True)

        guide = tk.Text(right, width=48, height=24, wrap="word")
        guide.pack(fill="both", expand=True, padx=6, pady=6)
        guide.insert("end", self._metric_guide_text())
        guide.configure(state="disabled")

    def _metric_guide_text(self) -> str:
        return (
            "Najważniejsze metryki:\n\n"
            "p95 ms — czas, poniżej którego mieści się 95% pomiarów. Dla live ważniejsze niż średnia. Mniej = lepiej.\n\n"
            "FPS safe — 1000 / p95. Pokazuje bezpieczny FPS w gorszym ogonie czasów. Więcej = lepiej.\n\n"
            "Precision — ile predykcji było poprawnych. Niskie precision = dużo false positive.\n\n"
            "Recall — ile obiektów z labeli wykryto. Dla piłki bardzo ważne, bo missed detection przerywa trajektorię.\n\n"
            "F1 — kompromis precision i recall. Blisko 1.0 = dobrze.\n\n"
            "TP — poprawnie dopasowane detekcje. FP — fałszywe detekcje. FN — obiekty zgubione.\n\n"
            "Mean IoU — średnia zgodność bboxów z GT. >0.5 akceptowalne, >0.75 dobre.\n\n"
            "ONNX/ENGINE z batch bN: czas rzeczywisty dotyczy całego batcha powielonych obrazów; norm per image = total_ms / batch."
        )

    # ---------- pickers ----------
    def pick_images(self):
        p = filedialog.askdirectory(title="Wybierz folder images")
        if p:
            self.images_dir = Path(p)
            self.status_text.set(f"Images: {self.images_dir} | count={len(list_images(self.images_dir))}")

    def pick_labels(self):
        p = filedialog.askdirectory(title="Wybierz folder labels")
        if p:
            self.labels_dir = Path(p)
            self.status_text.set(f"Labels: {self.labels_dir}")

    def pick_report_root(self):
        p = filedialog.askdirectory(title="Wybierz folder raportu")
        if p:
            self.report_root = Path(p)
            self.report_text.set(f"Raport root: {self.report_root}")

    def pick_model(self, backend: str):
        ext = {"pt": "*.pt", "onnx": "*.onnx", "engine": "*.engine"}.get(backend, "*.*")
        p = filedialog.askopenfilename(title=f"Wybierz model {backend.upper()}", filetypes=[("Model", ext), ("All", "*.*")])
        if not p:
            return
        self.load_model(backend, Path(p))

    def load_model(self, backend: str, path: Path):
        if YOLO is None:
            messagebox.showerror("Brak ultralytics", "Nie można zaimportować ultralytics. Zainstaluj paczkę ultralytics.")
            return
        try:
            self.models[backend] = YOLO(str(path), task="detect")
            self.model_paths[backend] = path
            self.status_text.set(f"Załadowano {backend.upper()}: {path.name}")
        except Exception as exc:
            self.models[backend] = None
            self.model_paths[backend] = None
            messagebox.showerror(f"Błąd modelu {backend.upper()}", str(exc))

    def autoload_static_backends(self):
        label = self.resolution_label.get().strip()
        batch = int(self.batch.get())
        precision = self.precision.get().strip()
        onnx_dir = Path(self.onnx_dir.get())
        engine_dir = Path(self.engine_dir.get())
        onnx_path = onnx_dir / onnx_name_for(label, batch, precision)
        engine_path = engine_dir / engine_name_for(label, batch, precision)
        msgs = []
        if onnx_path.exists():
            self.load_model("onnx", onnx_path)
            msgs.append(f"ONNX={onnx_path.name}")
        else:
            self.models["onnx"] = None
            self.model_paths["onnx"] = None
            msgs.append(f"ONNX missing: {onnx_path.name}")
        if engine_path.exists():
            self.load_model("engine", engine_path)
            msgs.append(f"ENGINE={engine_path.name}")
        else:
            self.models["engine"] = None
            self.model_paths["engine"] = None
            msgs.append(f"ENGINE missing: {engine_path.name}")
        self.status_text.set(" | ".join(msgs))

    def show_diagnostics(self):
        label = self.resolution_label.get().strip()
        batch = int(self.batch.get())
        precision = self.precision.get().strip()
        onnx_dir = Path(self.onnx_dir.get())
        engine_dir = Path(self.engine_dir.get())
        onnx_path = onnx_dir / onnx_name_for(label, batch, precision)
        engine_path = engine_dir / engine_name_for(label, batch, precision)
        msg = (
            f"ONNX dir exists={onnx_dir.exists()}:\n{onnx_dir}\n\n"
            f"ENGINE dir exists={engine_dir.exists()}:\n{engine_dir}\n\n"
            f"resolution={label}\nbatch={batch}\nprecision={precision}\n\n"
            f"ONNX searched exists={onnx_path.exists()}:\n{onnx_path}\n\n"
            f"ENGINE searched exists={engine_path.exists()}:\n{engine_path}"
        )
        messagebox.showinfo("Diagnostyka ścieżek", msg)

    # ---------- run analysis ----------
    def start_analysis(self):
        if self.running:
            return
        if self.images_dir is None or not self.images_dir.exists():
            messagebox.showerror("Brak danych", "Wybierz poprawny folder images.")
            return
        if self.report_root is None:
            messagebox.showerror("Brak folderu raportu", "Wybierz folder raportu.")
            return
        if self.labels_dir is None or not self.labels_dir.exists():
            if not messagebox.askyesno("Brak labels", "Folder labels nie istnieje. Analizować jako GT=[]?"):
                return
        self.autoload_static_backends()
        images = list_images(self.images_dir)
        if not images:
            messagebox.showerror("Brak obrazów", "Folder images nie zawiera obrazów.")
            return
        active = [b for b in BACKENDS if self.use_backend[b].get()]
        if not active:
            messagebox.showerror("Brak backendów", "Zaznacz przynajmniej jeden backend.")
            return
        self.running = True
        self.stop_requested = False
        self.progress.configure(maximum=len(images) * len(active), value=0)
        self.tree.delete(*self.tree.get_children())
        threading.Thread(target=self._analysis_worker, args=(images, active), daemon=True).start()

    def stop_analysis(self):
        self.stop_requested = True
        self.status_text.set("Zatrzymywanie po bieżącej iteracji...")

    def _analysis_worker(self, images: list[Path], active: list[str]):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = (self.report_root or Path.cwd()) / f"numeric_audit_{timestamp}"
        charts_dir = report_dir / "numeric_analysis_charts"
        overlays_dir = report_dir / "problem_overlays"
        crops_original_dir = report_dir / "problem_crops_original"
        report_dir.mkdir(parents=True, exist_ok=True)
        overlays_dir.mkdir(parents=True, exist_ok=True)
        crops_original_dir.mkdir(parents=True, exist_ok=True)

        target_hw = self.selected_hw()
        imgsz = self.selected_imgsz()
        cls_id = self.selected_cls_id()
        conf = float(self.conf.get())
        iou_nms = float(self.iou_nms.get())
        iou_match = float(self.iou_match.get())
        batch = int(self.batch.get())

        per_image_rows: list[dict[str, Any]] = []
        all_results: list[dict[str, Any]] = []
        warnings = {"label_lines": 0, "missing_labels": 0, "image_read_errors": 0}
        done = 0

        for img_path in images:
            if self.stop_requested:
                break
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                warnings["image_read_errors"] += 1
                for backend in active:
                    row = {
                        "image": img_path.name, "backend": backend, "gt_count": 0, "pred_count": 0,
                        "tp": 0, "fp": 0, "fn": 0, "precision_image": 0, "recall_image": 0, "f1_image": 0,
                        "best_iou": None, "mean_iou": None, "best_confidence": None, "total_ms": None,
                        "total_ms_norm_per_image": None, "batch_used": 0, "error": "image_read_error", "verdict": "FAILED",
                    }
                    per_image_rows.append(row)
                    done += 1
                    self._update_progress(done, len(images) * len(active), backend, img_path.name, 1)
                continue

            input_bgr, resize_info = resize_input_for_backend(img_bgr, target_hw)
            label_path = (self.labels_dir / f"{img_path.stem}.txt") if self.labels_dir else Path("__missing__")
            if not label_path.exists():
                warnings["missing_labels"] += 1
            gts, warn_count = load_yolo_labels(label_path, target_hw, cls_id)
            warnings["label_lines"] += warn_count

            image_bundle = {"image": str(img_path), "resize_info": resize_info, "gt": gts, "backends": {}}
            for backend in active:
                if self.stop_requested:
                    break
                det = run_backend_detections_full(
                    backend,
                    self.models.get(backend),
                    input_bgr,
                    conf=conf,
                    iou_nms=iou_nms,
                    imgsz=imgsz,
                    cls_id=cls_id,
                    batch=batch if backend in ("onnx", "engine") else 1,
                )
                match = match_detections_to_gt(det.get("detections", []), gts, iou_match) if not det.get("error") else {"tp": 0, "fp": 0, "fn": len(gts), "matches": [], "unmatched_gt": list(range(len(gts))), "unmatched_pred": []}
                row = build_per_image_row(img_path.name, backend, len(gts), det, match)
                per_image_rows.append(row)
                image_bundle["backends"][backend] = {"detection": det, "match": match, "row": row}
                if row["verdict"] in PROBLEM_VERDICTS:
                    self._write_problem_overlay(overlays_dir, img_path, input_bgr, gts, det.get("detections", []), row)
                    crop_meta = self._write_problem_crop_original(
                        crops_original_dir=crops_original_dir,
                        img_path=img_path,
                        original_bgr=img_bgr,
                        resize_info=resize_info,
                        gts=gts,
                        preds=det.get("detections", []),
                        row=row,
                    )
                    image_bundle["backends"][backend]["problem_crop_original"] = crop_meta
                done += 1
                errors = sum(1 for r in per_image_rows if r.get("error"))
                self._update_progress(done, len(images) * len(active), backend, img_path.name, errors)
            all_results.append(image_bundle)

        summary_rows = [summarize_backend(per_image_rows, b) for b in active]
        worst_rows = sorted(
            [r for r in per_image_rows if r.get("verdict") in PROBLEM_VERDICTS],
            key=lambda r: (r.get("verdict") != "FAILED", r.get("verdict") != "MISSED", float(r.get("best_iou") or -1)),
        )[:30]

        chart_files = generate_charts(summary_rows, charts_dir)
        config = {
            "images_dir": str(self.images_dir),
            "labels_dir": str(self.labels_dir),
            "report_dir": str(report_dir),
            "resolution_label": self.resolution_label.get(),
            "target_hw": list(target_hw),
            "batch": batch,
            "precision": self.precision.get(),
            "conf": conf,
            "iou_nms": iou_nms,
            "iou_match_gt": iou_match,
            "class_id": cls_id,
            "backend_batch_mode": f"repeat_same_image_to_b{batch}_for_onnx_engine",
            "model_paths": {b: str(self.model_paths.get(b)) if self.model_paths.get(b) else None for b in BACKENDS},
            "warnings": warnings,
            "stopped_by_user": self.stop_requested,
        }

        payload = {"config": config, "summary_rows": summary_rows, "per_image_rows": per_image_rows, "warnings": warnings, "results": all_results}
        (report_dir / "numeric_analysis_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (report_dir / "numeric_analysis_manifest.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        write_csv(report_dir / "numeric_analysis_per_image.csv", per_image_rows, list(per_image_rows[0].keys()) if per_image_rows else [])
        write_csv(report_dir / "numeric_analysis_summary.csv", summary_rows, list(summary_rows[0].keys()) if summary_rows else [])
        write_markdown_report(report_dir / "numeric_analysis_report.md", config, summary_rows, chart_files, worst_rows)

        self.last_report_dir = report_dir
        self.after(0, lambda: self._finish_analysis(summary_rows, report_dir))

    def _write_problem_overlay(self, overlays_dir: Path, img_path: Path, input_bgr, gts: list[dict[str, Any]], preds: list[dict[str, Any]], row: dict[str, Any]) -> None:
        try:
            name = f"{row['backend'].upper()}__{row['verdict']}__{img_path.name}"
            overlay = draw_problem_overlay(input_bgr, gts, preds, f"{row['backend'].upper()} {row['verdict']} TP={row['tp']} FP={row['fp']} FN={row['fn']} IoU={fmt(row.get('best_iou'))}")
            cv2.imwrite(str(overlays_dir / name), overlay)
        except Exception:
            pass

    def _write_problem_crop_original(
        self,
        *,
        crops_original_dir: Path,
        img_path: Path,
        original_bgr,
        resize_info: dict[str, Any],
        gts: list[dict[str, Any]],
        preds: list[dict[str, Any]],
        row: dict[str, Any],
    ) -> dict[str, Any]:
        verdict = str(row.get("verdict") or "")
        gt_boxes_original = [map_box_from_target_to_original(g.get("bbox_xyxy"), resize_info) for g in gts]
        gt_boxes_original = [b for b in gt_boxes_original if b is not None]
        pred_pairs = []
        for p in preds:
            mapped = map_box_from_target_to_original(p.get("bbox_xyxy"), resize_info)
            if mapped is not None:
                pred_pairs.append((p, mapped))
        pred_boxes_original = [b for _, b in pred_pairs]

        crop_box = select_problem_crop_box(gt_boxes_original, pred_boxes_original, verdict)
        if crop_box is None:
            return {
                "original_crop_path": None,
                "crop_box_original_xyxy": None,
                "source_hw": resize_info.get("source_hw"),
                "target_hw": resize_info.get("target_hw"),
                "scale_x": resize_info.get("scale_x"),
                "scale_y": resize_info.get("scale_y"),
            }

        crop_pack = crop_original_with_margin(original_bgr, crop_box, margin=40)
        if crop_pack is None:
            return {
                "original_crop_path": None,
                "crop_box_original_xyxy": None,
                "source_hw": resize_info.get("source_hw"),
                "target_hw": resize_info.get("target_hw"),
                "scale_x": resize_info.get("scale_x"),
                "scale_y": resize_info.get("scale_y"),
            }

        crop_bgr = crop_pack["crop_bgr"]
        cx1, cy1, _, _ = crop_pack["crop_box_original_xyxy"]

        # GT boxes in crop-local coordinates (green)
        for box in gt_boxes_original:
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            x1 -= int(cx1)
            x2 -= int(cx1)
            y1 -= int(cy1)
            y2 -= int(cy1)
            cv2.rectangle(crop_bgr, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(crop_bgr, "GT", (x1, max(18, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)

        # Pred boxes in crop-local coordinates (red/cyan)
        for pred, box in pred_pairs:
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            x1 -= int(cx1)
            x2 -= int(cx1)
            y1 -= int(cy1)
            y2 -= int(cy1)
            conf = float(pred.get("confidence") or 0.0)
            cv2.rectangle(crop_bgr, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(crop_bgr, f"P {conf:.2f}", (x1, min(crop_bgr.shape[0] - 8, y2 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        title = f"{str(row.get('backend', '')).upper()} {verdict} IoU={fmt(row.get('best_iou'))} conf={fmt(row.get('best_confidence'),2)}"
        cv2.putText(crop_bgr, title[:180], (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        crop_name = f"{str(row.get('backend','')).upper()}__{verdict}__{img_path.stem}__orig_crop.png"
        crop_path = crops_original_dir / crop_name
        cv2.imwrite(str(crop_path), crop_bgr)

        return {
            "original_crop_path": str(crop_path),
            "crop_box_original_xyxy": crop_pack["crop_box_original_xyxy"],
            "source_hw": resize_info.get("source_hw"),
            "target_hw": resize_info.get("target_hw"),
            "scale_x": resize_info.get("scale_x"),
            "scale_y": resize_info.get("scale_y"),
        }

    def _update_progress(self, done: int, total: int, backend: str, image_name: str, errors: int) -> None:
        pct = 100.0 * done / max(1, total)
        self.after(0, lambda: self.progress.configure(value=done))
        self.after(0, lambda: self.progress_text.set(f"Analiza: {done}/{total} ({pct:.1f}%) | backend={backend.upper()} | errors={errors} | current={image_name}"))

    def _finish_analysis(self, summary_rows: list[dict[str, Any]], report_dir: Path):
        self.running = False
        self.progress_text.set("Analiza zakończona.")
        self.report_text.set(f"Raport: {report_dir}")
        self.tree.delete(*self.tree.get_children())
        for r in summary_rows:
            self.tree.insert("", "end", values=(
                r["backend"].upper(), r["images_count"], r["error_count"], fmt(r.get("total_ms_p95"), 2),
                fmt(r.get("fps_p95_safe"), 1), fmt(r.get("precision"), 3), fmt(r.get("recall"), 3), fmt(r.get("f1"), 3),
                r.get("tp"), r.get("fp"), r.get("fn"), fmt(r.get("mean_iou_matched"), 3),
                r.get("missed_images_count"), r.get("false_positive_images_count"), r.get("verdict"),
            ))
        open_it = messagebox.askyesno("Analiza zakończona", f"Raport zapisano w:\n{report_dir}\n\nOtworzyć folder?")
        if open_it:
            self.open_report_folder()

    # ---------- open ----------
    def open_report(self):
        if not self.last_report_dir:
            messagebox.showinfo("Brak raportu", "Najpierw uruchom analizę.")
            return
        path = self.last_report_dir / "numeric_analysis_report.md"
        if path.exists():
            os.startfile(str(path))

    def open_report_folder(self):
        if not self.last_report_dir:
            messagebox.showinfo("Brak raportu", "Najpierw uruchom analizę.")
            return
        os.startfile(str(self.last_report_dir))

    def open_original_crops_folder(self):
        if not self.last_report_dir:
            messagebox.showinfo("Brak raportu", "Najpierw uruchom analizę.")
            return
        p = self.last_report_dir / "problem_crops_original"
        if p.exists():
            os.startfile(str(p))
        else:
            messagebox.showinfo("Brak cropów", f"Folder nie istnieje:\n{p}")


if __name__ == "__main__":
    app = NumericAuditDashboard()
    app.mainloop()
