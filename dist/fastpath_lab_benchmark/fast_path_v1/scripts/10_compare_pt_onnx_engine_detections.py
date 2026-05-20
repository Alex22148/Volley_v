from __future__ import annotations

import csv
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
import re
import statistics
import time
from typing import Any

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("YOLO_AUTOINSTALL", "False")


def _prefer_cuda_12_runtime() -> None:
    if os.name != "nt":
        return

    cuda_root = Path(os.environ.get("CUDA_PATH_V12_8", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"))
    cuda_bin = cuda_root / "bin"
    if not cuda_bin.exists():
        return

    os.environ["PATH"] = str(cuda_bin) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(str(cuda_bin))
        except OSError:
            pass


_prefer_cuda_12_runtime()

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover
    YOLO = None
    _ULTRALYTICS_IMPORT_ERROR = repr(exc)
else:
    _ULTRALYTICS_IMPORT_ERROR = ""


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BACKENDS = ("pt", "onnx", "engine")
PAIR_KEYS = (("onnx", "pt"), ("engine", "pt"))
LOW_IOU_THRESHOLD = 0.5
LOW_CONF_DIFF_THRESHOLD = 0.2

COLORS = {
    "pt": (0, 220, 0),        # green
    "onnx": (255, 170, 0),    # orange
    "engine": (0, 170, 255),  # blue
}




def resolve_default_out_dir(script_path: Path) -> Path:
    reports_root = script_path.resolve().parent.parent / "reports" / "backend_compare"
    reports_root.mkdir(parents=True, exist_ok=True)
    candidates = sorted(
        [p for p in reports_root.glob("test_???") if p.is_dir() and p.name[5:].isdigit()],
        key=lambda p: int(p.name[5:]),
    )
    next_idx = 1 if not candidates else int(candidates[-1].name[5:]) + 1
    return reports_root / f"test_{next_idx:03d}"


def next_test_dir_under(parent: Path) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    candidates = sorted(
        [p for p in parent.glob("test_???") if p.is_dir() and p.name[5:].isdigit()],
        key=lambda p: int(p.name[5:]),
    )
    next_idx = 1 if not candidates else int(candidates[-1].name[5:]) + 1
    return parent / f"test_{next_idx:03d}"


def resolve_out_dir(script_path: Path, out_dir_arg: str) -> Path:
    # Domyślnie zawsze tworzymy nowy test_XXX.
    if not out_dir_arg:
        return resolve_default_out_dir(script_path)

    raw = Path(out_dir_arg).expanduser().resolve()
    # Jeśli użytkownik podał folder "backend_compare", też tworzymy nowy test_XXX.
    if raw.name == "backend_compare":
        return resolve_default_out_dir(script_path)
    # Jeśli podał konkretnie "test_XXX", tworzymy kolejny test obok.
    if re.fullmatch(r"test_\d{3}", raw.name):
        return next_test_dir_under(raw.parent)
    return raw


def list_images(images_path: Path) -> list[Path]:
    if not images_path.exists():
        raise FileNotFoundError(f"Nie znaleziono ścieżki obrazów: {images_path}")
    if images_path.is_file():
        if images_path.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"Nieobsługiwany format pliku: {images_path.suffix}")
        return [images_path]
    files = [p for p in sorted(images_path.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not files:
        raise FileNotFoundError(f"Brak obrazów w folderze: {images_path}")
    return files


def generate_synthetic_images(target_dir: Path, img_h: int, img_w: int, count: int = 12) -> list[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for i in range(count):
        canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        # Proste, deterministyczne sceny testowe (bez resize wejścia).
        cv2.rectangle(canvas, (20 + i * 3, 20 + i * 2), (img_w // 2, img_h // 2), (0, 180, 255), 3)
        cv2.circle(canvas, (img_w // 2 + i * 2, img_h // 2), max(12, min(img_h, img_w) // 12), (255, 255, 255), 2)
        cv2.line(canvas, (0, i * max(1, img_h // 20) % img_h), (img_w - 1, (i * max(1, img_h // 20) + img_h // 3) % img_h), (0, 255, 0), 2)
        label = f"synthetic_{img_h}x{img_w}_{i:03d}"
        cv2.putText(canvas, label, (12, max(24, min(img_h, img_w) // 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
        out_path = target_dir / f"{label}.jpg"
        cv2.imwrite(str(out_path), canvas)
        out.append(out_path)
    return out


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    s = sorted(values)
    pos = q * (len(s) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(s[lo])
    frac = pos - lo
    return float(s[lo] * (1.0 - frac) + s[hi] * frac)


def mean_or_none(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
    except Exception:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def iou_xyxy(a: list[float] | None, b: list[float] | None) -> float | None:
    if not a or not b:
        return None
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def create_model(
    path: Path,
    device: str,
    warmup_batch_size: int = 1,
    warmup_imgsz: int | tuple[int, int] = 64,
) -> tuple[Any | None, str]:
    if not path.exists():
        return None, f"Model nie istnieje: {path}"
    if YOLO is None:
        return None, f"Brak ultralytics: {_ULTRALYTICS_IMPORT_ERROR}"
    try:
        model = YOLO(str(path), task="detect")
        # warmup minimalny na pustym obrazie dla stabilizacji pierwszego pomiaru
        if isinstance(warmup_imgsz, tuple):
            dummy_h, dummy_w = int(warmup_imgsz[0]), int(warmup_imgsz[1])
        else:
            dummy_h = dummy_w = int(warmup_imgsz)
        dummy = np.zeros((dummy_h, dummy_w, 3), dtype=np.uint8)
        warmup_input: Any = dummy if warmup_batch_size <= 1 else [dummy.copy() for _ in range(warmup_batch_size)]
        model.predict(warmup_input, imgsz=warmup_imgsz, conf=0.01, iou=0.01, device=device, verbose=False)
        return model, ""
    except Exception as exc:
        return None, f"Nie udało się załadować modelu {path.name}: {type(exc).__name__}: {exc}"


def run_backend_on_image(
    model: Any | None,
    backend: str,
    image_path: Path,
    imgsz: int | tuple[int, int],
    conf_thres: float,
    iou_thres: float,
    device: str,
    batch_size: int = 1,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "image": image_path.name,
        "backend": backend.upper(),
        "detected": False,
        "num_detections": 0,
        "best_confidence": None,
        "bbox_xyxy": None,
        "preprocess_ms": None,
        "inference_ms": None,
        "postprocess_ms": None,
        "color_ms": None,
        "total_ms": None,
        "predict_ms": None,
        "e2e_ms": None,
        "preprocess_ms_packet": None,
        "inference_ms_packet": None,
        "postprocess_ms_packet": None,
        "total_ms_packet": None,
        "predict_ms_packet": None,
        "e2e_ms_packet": None,
        "batch_size": int(batch_size),
        "all_detections": [],
        "error": "",
    }
    if model is None:
        base["error"] = "Backend niedostępny."
        return base

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        base["error"] = f"Nie można odczytać obrazu: {image_path}"
        return base

    try:
        color_started = time.perf_counter()
        color_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        color_ms = (time.perf_counter() - color_started) * 1000.0
        base["color_ms"] = float(color_ms / max(1, int(batch_size)))

        started = time.perf_counter()
        infer_input: Any = img
        if batch_size > 1:
            infer_input = [color_img.copy() for _ in range(batch_size)]
        else:
            infer_input = color_img
        results = model.predict(
            infer_input,
            imgsz=imgsz,
            conf=conf_thres,
            iou=iou_thres,
            device=device,
            verbose=False,
        )
        predict_ms_packet = (time.perf_counter() - started) * 1000.0
        total_ms = predict_ms_packet

        if not results:
            bs = max(1, int(batch_size))
            base["predict_ms_packet"] = predict_ms_packet
            base["predict_ms"] = float(predict_ms_packet / bs)
            base["total_ms_packet"] = total_ms
            base["total_ms"] = float(total_ms / bs)
            base["e2e_ms_packet"] = float(color_ms + predict_ms_packet)
            base["e2e_ms"] = float((color_ms + predict_ms_packet) / bs)
            return base

        r = results[0]
        speed = getattr(r, "speed", {}) or {}
        pre_packet = safe_float(speed.get("preprocess"))
        inf_packet = safe_float(speed.get("inference"))
        post_packet = safe_float(speed.get("postprocess"))
        bs = max(1, int(batch_size))
        base["preprocess_ms_packet"] = pre_packet
        base["inference_ms_packet"] = inf_packet
        base["postprocess_ms_packet"] = post_packet
        base["predict_ms_packet"] = predict_ms_packet
        base["total_ms_packet"] = total_ms
        base["preprocess_ms"] = None if pre_packet is None else float(pre_packet / bs)
        base["inference_ms"] = None if inf_packet is None else float(inf_packet / bs)
        base["postprocess_ms"] = None if post_packet is None else float(post_packet / bs)
        base["predict_ms"] = float(predict_ms_packet / bs)
        base["total_ms"] = float(total_ms / bs)
        base["e2e_ms_packet"] = float(color_ms + predict_ms_packet)
        base["e2e_ms"] = float((color_ms + predict_ms_packet) / bs)

        boxes = getattr(r, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return base

        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
        clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.array(boxes.cls)

        detections: list[dict[str, Any]] = []
        for i in range(len(xyxy)):
            bbox = [float(x) for x in xyxy[i].tolist()]
            det = {
                "bbox_xyxy": bbox,
                "confidence": float(confs[i]),
                "class_id": int(clss[i]) if len(clss) > i else None,
            }
            detections.append(det)

        base["num_detections"] = len(detections)
        base["detected"] = len(detections) > 0
        base["all_detections"] = detections

        if detections:
            best = max(detections, key=lambda d: d["confidence"])
            base["best_confidence"] = float(best["confidence"])
            base["bbox_xyxy"] = list(best["bbox_xyxy"])
        return base
    except Exception as exc:
        base["error"] = f"{type(exc).__name__}: {exc}"
        return base


def _empty_backend_row(image_name: str, backend: str, batch_size: int, error: str = "") -> dict[str, Any]:
    return {
        "image": image_name,
        "backend": backend.upper(),
        "detected": False,
        "num_detections": 0,
        "best_confidence": None,
        "bbox_xyxy": None,
        "preprocess_ms": None,
        "inference_ms": None,
        "postprocess_ms": None,
        "color_ms": None,
        "total_ms": None,
        "predict_ms": None,
        "e2e_ms": None,
        "preprocess_ms_packet": None,
        "inference_ms_packet": None,
        "postprocess_ms_packet": None,
        "total_ms_packet": None,
        "predict_ms_packet": None,
        "e2e_ms_packet": None,
        "batch_size": int(batch_size),
        "all_detections": [],
        "error": error,
    }


def _row_from_yolo_result(
    *,
    image_name: str,
    backend: str,
    result: Any,
    batch_size: int,
    color_ms_packet: float,
    predict_ms_packet: float,
) -> dict[str, Any]:
    row = _empty_backend_row(image_name, backend, batch_size)
    bs = max(1, int(batch_size))
    speed = getattr(result, "speed", {}) or {}
    pre_packet = safe_float(speed.get("preprocess"))
    inf_packet = safe_float(speed.get("inference"))
    post_packet = safe_float(speed.get("postprocess"))

    row["color_ms"] = float(color_ms_packet / bs)
    row["preprocess_ms_packet"] = pre_packet
    row["inference_ms_packet"] = inf_packet
    row["postprocess_ms_packet"] = post_packet
    row["predict_ms_packet"] = predict_ms_packet
    row["total_ms_packet"] = predict_ms_packet
    row["preprocess_ms"] = None if pre_packet is None else float(pre_packet / bs)
    row["inference_ms"] = None if inf_packet is None else float(inf_packet / bs)
    row["postprocess_ms"] = None if post_packet is None else float(post_packet / bs)
    row["predict_ms"] = float(predict_ms_packet / bs)
    row["total_ms"] = float(predict_ms_packet / bs)
    row["e2e_ms_packet"] = float(color_ms_packet + predict_ms_packet)
    row["e2e_ms"] = float((color_ms_packet + predict_ms_packet) / bs)

    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None:
        return row

    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
    confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
    clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.array(boxes.cls)

    detections: list[dict[str, Any]] = []
    for i in range(len(xyxy)):
        det = {
            "bbox_xyxy": [float(x) for x in xyxy[i].tolist()],
            "confidence": float(confs[i]),
            "class_id": int(clss[i]) if len(clss) > i else None,
        }
        detections.append(det)

    row["num_detections"] = len(detections)
    row["detected"] = len(detections) > 0
    row["all_detections"] = detections
    if detections:
        best = max(detections, key=lambda d: d["confidence"])
        row["best_confidence"] = float(best["confidence"])
        row["bbox_xyxy"] = list(best["bbox_xyxy"])
    return row


def run_backend_on_images_batched(
    model: Any | None,
    backend: str,
    image_paths: list[Path],
    imgsz: int | tuple[int, int],
    conf_thres: float,
    iou_thres: float,
    device: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    if model is None:
        return [_empty_backend_row(p.name, backend, batch_size, "Backend niedostępny.") for p in image_paths]

    chunk_size = max(1, int(batch_size))
    rows: list[dict[str, Any]] = []
    for start_idx in range(0, len(image_paths), chunk_size):
        chunk_paths = image_paths[start_idx:start_idx + chunk_size]
        color_imgs: list[np.ndarray] = []
        valid_names: list[str] = []
        color_started = time.perf_counter()
        for image_path in chunk_paths:
            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if img is None:
                rows.append(_empty_backend_row(image_path.name, backend, batch_size, f"Nie można odczytać obrazu: {image_path}"))
                continue
            color_imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            valid_names.append(image_path.name)
        color_ms_packet = (time.perf_counter() - color_started) * 1000.0

        if not color_imgs:
            continue

        actual_count = len(color_imgs)
        while len(color_imgs) < chunk_size:
            color_imgs.append(color_imgs[-1].copy())

        try:
            started = time.perf_counter()
            results = model.predict(
                color_imgs,
                imgsz=imgsz,
                conf=conf_thres,
                iou=iou_thres,
                device=device,
                verbose=False,
            )
            predict_ms_packet = (time.perf_counter() - started) * 1000.0
            for name, result in zip(valid_names, list(results)[:actual_count]):
                rows.append(_row_from_yolo_result(
                    image_name=name,
                    backend=backend,
                    result=result,
                    batch_size=batch_size,
                    color_ms_packet=color_ms_packet,
                    predict_ms_packet=predict_ms_packet,
                ))
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            for name in valid_names:
                rows.append(_empty_backend_row(name, backend, batch_size, err))

    return rows


def compare_backend_vs_pt(
    image_name: str,
    backend_test_name: str,
    ref_row: dict[str, Any],
    test_row: dict[str, Any],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "image": image_name,
        "pair": f"{backend_test_name.upper()}_VS_PT",
        "matched": False,
        "iou": None,
        "conf_ref": ref_row.get("best_confidence"),
        "conf_test": test_row.get("best_confidence"),
        "confidence_diff": None,
        "missed_detection": False,
        "extra_detection": False,
        "verdict": "FAILED",
        "error": "",
    }

    if ref_row.get("error"):
        out["error"] = f"PT_FAILED: {ref_row['error']}"
        out["verdict"] = "FAILED"
        return out
    if test_row.get("error"):
        out["error"] = f"{backend_test_name.upper()}_FAILED: {test_row['error']}"
        out["verdict"] = "FAILED"
        return out

    ref_detected = bool(ref_row.get("detected"))
    test_detected = bool(test_row.get("detected"))

    if ref_detected and not test_detected:
        out["missed_detection"] = True
        out["verdict"] = "MISSED_DETECTION"
        return out
    if (not ref_detected) and test_detected:
        out["extra_detection"] = True
        out["verdict"] = "EXTRA_DETECTION"
        return out
    if (not ref_detected) and (not test_detected):
        out["matched"] = True
        out["verdict"] = "OK"
        return out

    ref_bbox = ref_row.get("bbox_xyxy")
    test_dets = test_row.get("all_detections", []) or []
    if not ref_bbox or not test_dets:
        out["verdict"] = "FAILED"
        out["error"] = "Brak danych bbox do porównania."
        return out

    best_iou = -1.0
    best_det: dict[str, Any] | None = None
    for det in test_dets:
        iou = iou_xyxy(ref_bbox, det.get("bbox_xyxy"))
        if iou is None:
            continue
        if iou > best_iou:
            best_iou = iou
            best_det = det

    if best_det is None:
        out["verdict"] = "FAILED"
        out["error"] = "Nie udało się obliczyć IoU."
        return out

    out["matched"] = True
    out["iou"] = float(best_iou)
    out["conf_test"] = float(best_det["confidence"])
    conf_ref = safe_float(out["conf_ref"])
    conf_test = safe_float(out["conf_test"])
    if conf_ref is not None and conf_test is not None:
        out["confidence_diff"] = float(conf_ref - conf_test)

    if out["iou"] < LOW_IOU_THRESHOLD:
        out["verdict"] = "LOW_IOU"
    elif out["confidence_diff"] is not None and float(out["confidence_diff"]) > LOW_CONF_DIFF_THRESHOLD:
        out["verdict"] = "LOW_CONFIDENCE"
    else:
        out["verdict"] = "OK"
    return out


def draw_overlay(
    image_path: Path,
    out_path: Path,
    pt_row: dict[str, Any],
    onnx_row: dict[str, Any],
    engine_row: dict[str, Any],
) -> None:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return
    canvas = img.copy()

    for row, key in [(pt_row, "pt"), (onnx_row, "onnx"), (engine_row, "engine")]:
        bbox = row.get("bbox_xyxy")
        conf = row.get("best_confidence")
        if bbox is None:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        color = COLORS[key]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{key.upper()} conf={conf:.3f}" if conf is not None else f"{key.upper()} conf=-"
        cv2.putText(canvas, label, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    legend_y = 24
    for key in ("pt", "onnx", "engine"):
        color = COLORS[key]
        cv2.putText(canvas, key.upper(), (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        legend_y += 22

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def build_backend_summary(rows: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    pair_by_backend = {
        "ONNX": [p for p in pairs if p["pair"] == "ONNX_VS_PT"],
        "ENGINE": [p for p in pairs if p["pair"] == "ENGINE_VS_PT"],
    }

    for backend in ("PT", "ONNX", "ENGINE"):
        b_rows = [r for r in rows if r["backend"] == backend]
        ok_rows = [r for r in b_rows if not r.get("error")]
        total_vals = [float(r["total_ms"]) for r in ok_rows if r.get("total_ms") is not None]
        pre_vals = [float(r["preprocess_ms"]) for r in ok_rows if r.get("preprocess_ms") is not None]
        inf_vals = [float(r["inference_ms"]) for r in ok_rows if r.get("inference_ms") is not None]
        post_vals = [float(r["postprocess_ms"]) for r in ok_rows if r.get("postprocess_ms") is not None]
        color_vals = [float(r["color_ms"]) for r in ok_rows if r.get("color_ms") is not None]
        predict_vals = [float(r["predict_ms"]) for r in ok_rows if r.get("predict_ms") is not None]
        e2e_vals = [float(r["e2e_ms"]) for r in ok_rows if r.get("e2e_ms") is not None]
        total_packet_vals = [float(r["total_ms_packet"]) for r in ok_rows if r.get("total_ms_packet") is not None]
        pre_packet_vals = [float(r["preprocess_ms_packet"]) for r in ok_rows if r.get("preprocess_ms_packet") is not None]
        inf_packet_vals = [float(r["inference_ms_packet"]) for r in ok_rows if r.get("inference_ms_packet") is not None]
        post_packet_vals = [float(r["postprocess_ms_packet"]) for r in ok_rows if r.get("postprocess_ms_packet") is not None]
        predict_packet_vals = [float(r["predict_ms_packet"]) for r in ok_rows if r.get("predict_ms_packet") is not None]
        e2e_packet_vals = [float(r["e2e_ms_packet"]) for r in ok_rows if r.get("e2e_ms_packet") is not None]
        conf_vals = [float(r["best_confidence"]) for r in ok_rows if r.get("best_confidence") is not None]
        detected_count = sum(1 for r in ok_rows if r.get("detected"))
        detection_rate = (100.0 * detected_count / len(ok_rows)) if ok_rows else 0.0

        iou_vals: list[float] = []
        missed = 0
        if backend in ("ONNX", "ENGINE"):
            p_rows = pair_by_backend[backend]
            iou_vals = [float(p["iou"]) for p in p_rows if p.get("iou") is not None]
            missed = sum(1 for p in p_rows if p.get("missed_detection"))

        summary.append(
            {
                "backend": backend,
                "mean_total_ms": mean_or_none(total_vals),
                "p95_total_ms": percentile(total_vals, 0.95),
                "mean_preprocess_ms": mean_or_none(pre_vals),
                "mean_inference_ms": mean_or_none(inf_vals),
                "mean_postprocess_ms": mean_or_none(post_vals),
                "mean_color_ms": mean_or_none(color_vals),
                "mean_predict_ms": mean_or_none(predict_vals),
                "mean_e2e_ms": mean_or_none(e2e_vals),
                "mean_total_ms_packet": mean_or_none(total_packet_vals),
                "p95_total_ms_packet": percentile(total_packet_vals, 0.95),
                "mean_preprocess_ms_packet": mean_or_none(pre_packet_vals),
                "p95_preprocess_ms_packet": percentile(pre_packet_vals, 0.95),
                "mean_inference_ms_packet": mean_or_none(inf_packet_vals),
                "p95_inference_ms_packet": percentile(inf_packet_vals, 0.95),
                "mean_postprocess_ms_packet": mean_or_none(post_packet_vals),
                "p95_postprocess_ms_packet": percentile(post_packet_vals, 0.95),
                "mean_predict_ms_packet": mean_or_none(predict_packet_vals),
                "p95_predict_ms_packet": percentile(predict_packet_vals, 0.95),
                "mean_e2e_ms_packet": mean_or_none(e2e_packet_vals),
                "p95_e2e_ms_packet": percentile(e2e_packet_vals, 0.95),
                "detection_rate_percent": detection_rate,
                "mean_best_confidence": mean_or_none(conf_vals),
                "mean_iou_vs_pt": mean_or_none(iou_vals) if backend in ("ONNX", "ENGINE") else None,
                "missed_detections_vs_pt": missed if backend in ("ONNX", "ENGINE") else None,
                "errors_count": sum(1 for r in b_rows if r.get("error")),
                "samples": len(b_rows),
            }
        )
    return summary


def backend_final_verdict(s: dict[str, Any]) -> str:
    if (s.get("errors_count") or 0) > 0:
        return "Błędy wykonania"
    backend = s["backend"]
    if backend == "PT":
        return "Referencja"
    mean_iou = safe_float(s.get("mean_iou_vs_pt"))
    missed = int(s.get("missed_detections_vs_pt") or 0)
    speed = safe_float(s.get("mean_total_ms"))
    if speed is None:
        return "Błędy wykonania"
    if mean_iou is not None and mean_iou >= 0.8 and missed == 0:
        return "Bardzo szybki i zgodny" if backend == "ENGINE" else "Dobra zgodność, umiarkowany zysk"
    if missed > 0:
        return "Szybki, ale gubi detekcje"
    if mean_iou is not None and mean_iou >= 0.6:
        return "Dobra zgodność, umiarkowany zysk"
    return "Nieopłacalny kompromis"


def ensure_float(v: Any, default: float = 0.0) -> float:
    x = safe_float(v)
    return default if x is None else float(x)


def save_charts(
    out_charts_dir: Path,
    summary: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
) -> None:
    out_charts_dir.mkdir(parents=True, exist_ok=True)

    backend_order = ["PT", "ONNX", "ENGINE"]
    summary_map = {s["backend"]: s for s in summary}
    labels = backend_order

    mean_total = [ensure_float(summary_map.get(b, {}).get("mean_total_ms")) for b in labels]
    p95_total = [ensure_float(summary_map.get(b, {}).get("p95_total_ms")) for b in labels]
    mean_pre = [ensure_float(summary_map.get(b, {}).get("mean_preprocess_ms")) for b in labels]
    mean_inf = [ensure_float(summary_map.get(b, {}).get("mean_inference_ms")) for b in labels]
    mean_post = [ensure_float(summary_map.get(b, {}).get("mean_postprocess_ms")) for b in labels]
    det_rate = [ensure_float(summary_map.get(b, {}).get("detection_rate_percent")) for b in labels]
    mean_conf = [ensure_float(summary_map.get(b, {}).get("mean_best_confidence")) for b in labels]

    iou_vals = [
        ensure_float(summary_map.get("ONNX", {}).get("mean_iou_vs_pt")),
        ensure_float(summary_map.get("ENGINE", {}).get("mean_iou_vs_pt")),
    ]
    missed_vals = [
        int(summary_map.get("ONNX", {}).get("missed_detections_vs_pt") or 0),
        int(summary_map.get("ENGINE", {}).get("missed_detections_vs_pt") or 0),
    ]

    # 1
    plt.figure(figsize=(8, 5))
    plt.bar(labels, mean_total, color=["#2ca02c", "#ff7f0e", "#1f77b4"])
    plt.ylabel("ms / image")
    plt.title("Średni czas całkowity per backend (per image)")
    plt.tight_layout()
    plt.savefig(out_charts_dir / "chart_01_sredni_czas_calkowity_backend.png", dpi=140)
    plt.close()

    # 2
    plt.figure(figsize=(8, 5))
    plt.bar(labels, p95_total, color=["#2ca02c", "#ff7f0e", "#1f77b4"])
    plt.ylabel("ms / image")
    plt.title("p95 czasu całkowitego per backend (per image)")
    plt.tight_layout()
    plt.savefig(out_charts_dir / "chart_02_p95_czas_calkowity_backend.png", dpi=140)
    plt.close()

    # 3
    plt.figure(figsize=(9, 5))
    x = np.arange(len(labels))
    plt.bar(x, mean_pre, label="preprocess_ms")
    plt.bar(x, mean_inf, bottom=mean_pre, label="inference_ms")
    bottoms = [a + b for a, b in zip(mean_pre, mean_inf)]
    plt.bar(x, mean_post, bottom=bottoms, label="postprocess_ms")
    plt.xticks(x, labels)
    plt.ylabel("ms / image")
    plt.title("Rozbicie czasu na etapy per backend (per image)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_charts_dir / "chart_03_rozbicie_czasu_backend.png", dpi=140)
    plt.close()

    # 4
    plt.figure(figsize=(8, 5))
    plt.bar(labels, det_rate, color=["#2ca02c", "#ff7f0e", "#1f77b4"])
    plt.ylim(0, 100)
    plt.ylabel("%")
    plt.title("Skuteczność detekcji per backend")
    plt.tight_layout()
    plt.savefig(out_charts_dir / "chart_04_skutecznosc_detekcji_backend.png", dpi=140)
    plt.close()

    # 5
    plt.figure(figsize=(8, 5))
    plt.bar(labels, mean_conf, color=["#2ca02c", "#ff7f0e", "#1f77b4"])
    plt.ylabel("confidence")
    plt.title("Średnie confidence per backend")
    plt.tight_layout()
    plt.savefig(out_charts_dir / "chart_05_confidence_backend.png", dpi=140)
    plt.close()

    # 6
    plt.figure(figsize=(8, 5))
    plt.bar(["ONNX vs PT", "ENGINE vs PT"], iou_vals, color=["#ff7f0e", "#1f77b4"])
    plt.ylim(0, 1.0)
    plt.ylabel("IoU")
    plt.title("Średnie IoU względem PT")
    plt.tight_layout()
    plt.savefig(out_charts_dir / "chart_06_sredni_iou_wzgledem_pt.png", dpi=140)
    plt.close()

    # 7
    plt.figure(figsize=(8, 5))
    plt.bar(["ONNX", "ENGINE"], missed_vals, color=["#ff7f0e", "#1f77b4"])
    plt.ylabel("Liczba missed_detection")
    plt.title("Liczba braków detekcji względem PT")
    plt.tight_layout()
    plt.savefig(out_charts_dir / "chart_07_missed_detections_vs_pt.png", dpi=140)
    plt.close()

    # 8
    plt.figure(figsize=(8, 6))
    speed = [ensure_float(summary_map.get("PT", {}).get("mean_total_ms")),
             ensure_float(summary_map.get("ONNX", {}).get("mean_total_ms")),
             ensure_float(summary_map.get("ENGINE", {}).get("mean_total_ms"))]
    iou_scatter = [1.0, iou_vals[0], iou_vals[1]]
    names = ["PT", "ONNX", "ENGINE"]
    colors = ["#2ca02c", "#ff7f0e", "#1f77b4"]
    for x, y, n, c in zip(speed, iou_scatter, names, colors):
        plt.scatter([x], [y], label=n, color=c, s=80)
        plt.text(x, y, f" {n}", va="bottom")
    plt.xlabel("Średni total_ms / image")
    plt.ylabel("Średni IoU vs PT")
    plt.title("Speed vs Accuracy")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_charts_dir / "chart_08_speed_vs_iou.png", dpi=140)
    plt.close()

    # 9
    plt.figure(figsize=(9, 6))
    pt_t = [float(r["total_ms"]) for r in rows if r["backend"] == "PT" and r.get("total_ms") is not None and not r.get("error")]
    onnx_t = [float(r["total_ms"]) for r in rows if r["backend"] == "ONNX" and r.get("total_ms") is not None and not r.get("error")]
    eng_t = [float(r["total_ms"]) for r in rows if r["backend"] == "ENGINE" and r.get("total_ms") is not None and not r.get("error")]
    data = [pt_t if pt_t else [0.0], onnx_t if onnx_t else [0.0], eng_t if eng_t else [0.0]]
    plt.boxplot(data, tick_labels=["PT", "ONNX", "ENGINE"])
    plt.ylabel("total_ms / image")
    plt.title("Rozkład czasów całkowitych (per image)")
    plt.tight_layout()
    plt.savefig(out_charts_dir / "chart_09_rozklad_total_ms_backend.png", dpi=140)
    plt.close()

    # 10
    problematic: list[tuple[str, float]] = []
    for p in pairs:
        if p.get("iou") is not None:
            problematic.append((f"{p['pair']} | {p['image']}", float(p["iou"])))
    problematic.sort(key=lambda x: x[1])
    top = problematic[:10]
    plt.figure(figsize=(12, 6))
    if top:
        labels_top = [t[0] for t in top]
        values_top = [t[1] for t in top]
        y = np.arange(len(labels_top))
        plt.barh(y, values_top, color="#d62728")
        plt.yticks(y, labels_top, fontsize=8)
        plt.xlabel("IoU")
        plt.title("Top problematyczne obrazy (najniższe IoU)")
        plt.gca().invert_yaxis()
    else:
        plt.text(0.5, 0.5, "Brak danych IoU", ha="center", va="center")
        plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(out_charts_dir / "chart_10_problematic_images.png", dpi=140)
    plt.close()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def build_all_cases_summary(case_records: list[dict[str, Any]], root_out_dir: Path) -> None:
    if not case_records:
        return

    rows: list[dict[str, Any]] = []
    for rec in case_records:
        meta = rec["meta"]
        batch = int(meta["batch_n"])
        for summary in rec["summary"]:
            mean_total = safe_float(summary.get("mean_total_ms"))
            p95_total = safe_float(summary.get("p95_total_ms"))
            mean_predict = safe_float(summary.get("mean_predict_ms"))
            mean_e2e = safe_float(summary.get("mean_e2e_ms"))
            mean_total_packet = safe_float(summary.get("mean_total_ms_packet"))
            p95_total_packet = safe_float(summary.get("p95_total_ms_packet"))
            mean_predict_packet = safe_float(summary.get("mean_predict_ms_packet"))
            mean_e2e_packet = safe_float(summary.get("mean_e2e_ms_packet"))
            rows.append({
                "variant": meta["variant"],
                "resolution": meta["imgsz"],
                "batch": batch,
                "backend": summary["backend"],
                "mean_total_ms_per_batch": mean_total_packet,
                "p95_total_ms_per_batch": p95_total_packet,
                "mean_total_ms_per_image": mean_total,
                "p95_total_ms_per_image": p95_total,
                "mean_predict_ms_per_batch": mean_predict_packet,
                "mean_predict_ms_per_image": mean_predict,
                "mean_e2e_ms_per_batch": mean_e2e_packet,
                "mean_e2e_ms_per_image": mean_e2e,
                "mean_color_ms_per_image": summary.get("mean_color_ms"),
                "mean_preprocess_ms_per_image": summary.get("mean_preprocess_ms"),
                "mean_inference_ms_per_image": summary.get("mean_inference_ms"),
                "mean_postprocess_ms_per_image": summary.get("mean_postprocess_ms"),
                "mean_preprocess_ms_per_batch": summary.get("mean_preprocess_ms_packet"),
                "mean_inference_ms_per_batch": summary.get("mean_inference_ms_packet"),
                "mean_postprocess_ms_per_batch": summary.get("mean_postprocess_ms_packet"),
                "p95_preprocess_ms_per_batch": summary.get("p95_preprocess_ms_packet"),
                "p95_inference_ms_per_batch": summary.get("p95_inference_ms_packet"),
                "p95_postprocess_ms_per_batch": summary.get("p95_postprocess_ms_packet"),
                "detection_rate_percent": summary.get("detection_rate_percent"),
                "mean_best_confidence": summary.get("mean_best_confidence"),
                "mean_iou_vs_pt": summary.get("mean_iou_vs_pt"),
                "missed_detections_vs_pt": summary.get("missed_detections_vs_pt"),
                "errors_count": summary.get("errors_count"),
                "samples": summary.get("samples"),
                "backend_verdict": summary.get("backend_verdict"),
                "case_dir": meta["output_dir"],
            })

    fieldnames = [
        "variant", "resolution", "batch", "backend",
        "mean_total_ms_per_batch", "p95_total_ms_per_batch",
        "mean_total_ms_per_image", "p95_total_ms_per_image",
        "mean_predict_ms_per_batch", "mean_predict_ms_per_image",
        "mean_e2e_ms_per_batch", "mean_e2e_ms_per_image",
        "mean_color_ms_per_image", "mean_preprocess_ms_per_image",
        "mean_inference_ms_per_image", "mean_postprocess_ms_per_image",
        "mean_preprocess_ms_per_batch", "mean_inference_ms_per_batch",
        "mean_postprocess_ms_per_batch", "p95_preprocess_ms_per_batch",
        "p95_inference_ms_per_batch", "p95_postprocess_ms_per_batch",
        "detection_rate_percent", "mean_best_confidence", "mean_iou_vs_pt",
        "missed_detections_vs_pt", "errors_count", "samples", "backend_verdict",
        "case_dir",
    ]
    csv_path = root_out_dir / "backend_compare_all_cases_summary.csv"
    write_csv_rows(csv_path, rows, fieldnames)

    labels = [f"{r['resolution']} b{r['batch']}" for r in rows if r["backend"] == "ENGINE"]
    backend_order = ["PT", "ONNX", "ENGINE"]
    x = np.arange(len(labels))
    width = 0.25
    chart_path = root_out_dir / "backend_compare_all_cases_total_ms.png"
    plt.figure(figsize=(max(12, len(labels) * 0.75), 6))
    for offset, backend in enumerate(backend_order):
        vals = []
        for label in labels:
            match = next(
                (r for r in rows if r["backend"] == backend and f"{r['resolution']} b{r['batch']}" == label),
                None,
            )
            vals.append(ensure_float(match.get("mean_total_ms_per_image") if match else None))
        plt.bar(x + (offset - 1) * width, vals, width, label=backend)
    plt.ylabel("mean total_ms / image")
    plt.title("Backend compare - zbiorczo po wariantach")
    plt.xticks(x, labels, rotation=65, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    plt.close()

    def fmt(v: Any, digits: int = 3) -> str:
        x = safe_float(v)
        return "-" if x is None else f"{x:.{digits}f}"

    report_path = root_out_dir / "backend_compare_all_cases_report.md"
    lines: list[str] = []
    a = lines.append
    a("# Backend Compare - raport zbiorczy")
    a("")
    a("Zbiera wszystkie `case_*` z jednego uruchomienia `10_compare_pt_onnx_engine_detections.py`.")
    a("")
    a("## Wykres zbiorczy")
    a("")
    a("![Backend compare all cases](backend_compare_all_cases_total_ms.png)")
    a("")
    a("## Tabela: batch vs image")
    a("")
    a("| Variant | Backend | Total mean / batch ms | Total p95 / batch ms | Total mean / image ms | Total p95 / image ms | IoU vs PT | Missed vs PT | Verdict |")
    a("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        a(
            f"| {r['variant']} | {r['backend']} | "
            f"{fmt(r['mean_total_ms_per_batch'])} | {fmt(r['p95_total_ms_per_batch'])} | "
            f"{fmt(r['mean_total_ms_per_image'])} | {fmt(r['p95_total_ms_per_image'])} | "
            f"{fmt(r['mean_iou_vs_pt'])} | "
            f"{'-' if r['missed_detections_vs_pt'] is None else r['missed_detections_vs_pt']} | "
            f"{r['backend_verdict']} |"
        )
    a("")
    a("## Rozbicie etapów per image")
    a("")
    a("| Variant | Backend | Color mean / image ms | Preprocess mean / image ms | Inference mean / image ms | Postprocess mean / image ms | Total mean / image ms |")
    a("|---|---|---:|---:|---:|---:|---:|")
    for r in rows:
        a(
            f"| {r['variant']} | {r['backend']} | "
            f"{fmt(r['mean_color_ms_per_image'])} | {fmt(r['mean_preprocess_ms_per_image'])} | "
            f"{fmt(r['mean_inference_ms_per_image'])} | {fmt(r['mean_postprocess_ms_per_image'])} | "
            f"{fmt(r['mean_total_ms_per_image'])} |"
        )
    a("")
    a("## Rozbicie etapów per batch")
    a("")
    a("| Variant | Backend | Preprocess mean / batch ms | Inference mean / batch ms | Postprocess mean / batch ms | Predict mean / batch ms | E2E mean / batch ms |")
    a("|---|---|---:|---:|---:|---:|---:|")
    for r in rows:
        a(
            f"| {r['variant']} | {r['backend']} | "
            f"{fmt(r['mean_preprocess_ms_per_batch'])} | {fmt(r['mean_inference_ms_per_batch'])} | "
            f"{fmt(r['mean_postprocess_ms_per_batch'])} | {fmt(r['mean_predict_ms_per_batch'])} | "
            f"{fmt(r['mean_e2e_ms_per_batch'])} |"
        )
    a("")
    a("## Pliki")
    a("")
    a("- `backend_compare_all_cases_summary.csv`")
    a("- `backend_compare_all_cases_report.md`")
    a("- `backend_compare_all_cases_total_ms.png`")
    a("- `case_*/backend_compare_report.md`")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[DONE] all-cases summary -> {csv_path}")
    print(f"[DONE] all-cases report  -> {report_path}")
    print(f"[DONE] all-cases chart   -> {chart_path}")


def build_report_md(
    out_dir: Path,
    params: dict[str, Any],
    summary: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> str:
    summary_map = {s["backend"]: s for s in summary}
    pt_mean = safe_float(summary_map.get("PT", {}).get("mean_total_ms"))
    eng_mean = safe_float(summary_map.get("ENGINE", {}).get("mean_total_ms"))
    onnx_mean = safe_float(summary_map.get("ONNX", {}).get("mean_total_ms"))

    fastest_backend = min(
        [s for s in summary if safe_float(s.get("mean_total_ms")) is not None],
        key=lambda s: float(s.get("mean_total_ms", 1e18)),
        default=None,
    )
    fastest_name = fastest_backend["backend"] if fastest_backend else "-"
    engine_speedup = (pt_mean / eng_mean) if (pt_mean and eng_mean and eng_mean > 0) else None

    engine_iou = safe_float(summary_map.get("ENGINE", {}).get("mean_iou_vs_pt"))
    engine_missed = int(summary_map.get("ENGINE", {}).get("missed_detections_vs_pt") or 0)
    onnx_missed = int(summary_map.get("ONNX", {}).get("missed_detections_vs_pt") or 0)

    engine_safe = (engine_iou is not None and engine_iou >= 0.7 and engine_missed == 0)
    onnx_safe = (safe_float(summary_map.get("ONNX", {}).get("mean_iou_vs_pt")) is not None and onnx_missed == 0)

    problem_rows = [p for p in pairs if p.get("iou") is not None]
    problem_rows.sort(key=lambda x: float(x["iou"]))
    top_problem = problem_rows[:10]

    lines: list[str] = []
    a = lines.append
    a("# Porównanie backendów PT / ONNX / ENGINE")
    a("")
    a("## 1. Cel porównania")
    a("")
    a("Celem testu jest porównanie backendów `.pt` (referencja), `.onnx` i `.engine` na tych samych obrazach pod kątem:")
    a("- czasu wykonania,")
    a("- wykryć obiektów,")
    a("- confidence,")
    a("- zgodności bboxów względem referencji `.pt`.")
    a("")
    a("## 2. Parametry testu")
    a("")
    a(f"- images: `{params['images']}`")
    a(f"- pt: `{params['pt']}`")
    a(f"- onnx: `{params['onnx']}`")
    a(f"- engine: `{params['engine']}`")
    a(f"- imgsz: `{params['imgsz']}`")
    a(f"- conf: `{params['conf']}`")
    a(f"- iou: `{params['iou']}`")
    a(f"- device: `{params['device_used']}`")
    a(f"- liczba obrazów: `{params['num_images']}`")
    a("")
    a("## 3. Jak czytać wyniki")
    a("")
    a("- `.pt` jest referencją.")
    a("- IoU (Intersection over Union) mierzy zgodność bboxów; im bliżej 1.0, tym lepiej.")
    a("- `missed_detection` oznacza, że backend testowy nie wykrył obiektu, który wykrył PT.")
    a("- `extra_detection` oznacza, że backend testowy wykrył obiekt, którego PT nie wykrył.")
    a("- Szybszy backend z gorszym IoU lub większą liczbą missed detections może być ryzykowny produkcyjnie.")
    a("")
    a("## 4. Podsumowanie ogólne")
    a("")
    a(f"- Najszybszy backend (średni total_ms): **{fastest_name}**")
    a(f"- ENGINE względem PT: **{engine_speedup:.2f}x szybciej**" if engine_speedup is not None else "- ENGINE względem PT: `-`")
    a(f"- Zgodność bboxów ENGINE vs PT (średni IoU): **{engine_iou:.3f}**" if engine_iou is not None else "- Zgodność bboxów ENGINE vs PT: `-`")
    a(f"- Missed detections: ONNX={onnx_missed}, ENGINE={engine_missed}")
    a(f"- Czy ENGINE bezpieczny do dalszego użycia: **{'TAK' if engine_safe else 'NIE / WYMAGA WERYFIKACJI'}**")
    a(f"- Czy ONNX bezpieczny do dalszego użycia: **{'TAK' if onnx_safe else 'NIE / WYMAGA WERYFIKACJI'}**")
    a("")
    a("## 5. Najważniejsze wnioski")
    a("")
    if engine_speedup is not None and engine_speedup > 1.0 and engine_safe:
        a("- ENGINE daje istotny zysk czasu i zachowuje dobrą zgodność z PT.")
    if engine_speedup is not None and engine_speedup > 1.0 and not engine_safe:
        a("- ENGINE jest szybszy, ale wymaga poprawy jakości detekcji (IoU/missed).")
    if onnx_mean is not None and pt_mean is not None and onnx_mean < pt_mean:
        a("- ONNX jest szybszy od PT, ale decyzja wdrożeniowa zależy od jakości bboxów.")
    if not top_problem:
        a("- Brak przypadków problematycznych z IoU (lub brak danych porównawczych).")
    a("")
    a("## 6. Wykresy")
    a("")
    for name, title in [
        ("chart_01_sredni_czas_calkowity_backend.png", "Średni czas całkowity"),
        ("chart_02_p95_czas_calkowity_backend.png", "p95 czasu całkowitego"),
        ("chart_03_rozbicie_czasu_backend.png", "Rozbicie czasu na etapy"),
        ("chart_04_skutecznosc_detekcji_backend.png", "Skuteczność detekcji"),
        ("chart_05_confidence_backend.png", "Średnie confidence"),
        ("chart_06_sredni_iou_wzgledem_pt.png", "Średnie IoU względem PT"),
        ("chart_07_missed_detections_vs_pt.png", "Missed detections względem PT"),
        ("chart_08_speed_vs_iou.png", "Speed vs IoU"),
        ("chart_09_rozklad_total_ms_backend.png", "Rozkład total_ms"),
        ("chart_10_problematic_images.png", "Top problematyczne obrazy"),
    ]:
        a(f"### {title}")
        a(f"![{title}](charts/{name})")
        a("")
    a("## 7. Tabela zbiorcza backendów")
    a("")
    a("| Backend | Średni total ms | p95 total ms | Skuteczność detekcji [%] | Średni confidence | Średni IoU vs PT | Missed detections | Werdykt |")
    a("|---|---:|---:|---:|---:|---:|---:|---|")
    for s in summary:
        verdict = backend_final_verdict(s)
        a(
            f"| {s['backend']} | "
            f"{'-' if s['mean_total_ms'] is None else f'{s['mean_total_ms']:.3f}'} | "
            f"{'-' if s['p95_total_ms'] is None else f'{s['p95_total_ms']:.3f}'} | "
            f"{s['detection_rate_percent']:.2f} | "
            f"{'-' if s['mean_best_confidence'] is None else f'{s['mean_best_confidence']:.3f}'} | "
            f"{'-' if s['mean_iou_vs_pt'] is None else f'{s['mean_iou_vs_pt']:.3f}'} | "
            f"{'-' if s['missed_detections_vs_pt'] is None else s['missed_detections_vs_pt']} | "
            f"{verdict} |"
        )
    a("")
    a("## 8. Problematyczne przypadki")
    a("")
    if not top_problem:
        a("Brak przypadków z obliczonym IoU.")
    else:
        a("| Pair | Obraz | IoU | Werdykt |")
        a("|---|---|---:|---|")
        for p in top_problem:
            a(f"| {p['pair']} | {p['image']} | {float(p['iou']):.3f} | {p['verdict']} |")
    a("")
    a("## 9. Pliki wyjściowe")
    a("")
    a("- `backend_compare_results.json`")
    a("- `backend_compare_summary.csv`")
    a("- `backend_compare_pairs.csv`")
    a("- `backend_compare_report.md`")
    a("- `charts/*.png`")
    a("- `overlays/*.jpg`")
    a("")
    return "\n".join(lines) + "\n"


def main() -> int:
    real_images = r""
    pt = r"C:\Users\UGB_a\PycharmProjects\VolleyHub_enterprice\dist\fastpath_lab_benchmark\fast_path_v1\model_base\best.pt"
    onnx_dir = r"C:\Users\UGB_a\PycharmProjects\VolleyHub_enterprice\dist\fastpath_lab_benchmark\fast_path_v1\engines\onnx"
    engine_dir = r"C:\Users\UGB_a\PycharmProjects\VolleyHub_enterprice\dist\fastpath_lab_benchmark\fast_path_v1\engines\static"
    conf = 0.25
    iou = 0.45
    device = "cuda"
    out_dir_arg = ""
    # Smoke/default mode for pipeline use. Full backend-quality sweeps with
    # hundreds of images should be a separate manual run, not part of 11_pipeline.
    synthetic_count = 150
    max_overlays_per_case = 8
    compare_matrix: dict[str, tuple[int, int, set[int]]] = {
        "v640": (640, 640, {4, 8, 12, 16,32}),
        "v960": (960, 960, {4, 8, 12, 16,32}),
        "v1088x1920": (1088, 1920, {4, 8, 12, 16}),
    }

    pt_path = Path(pt).expanduser().resolve()
    onnx_dir_path = Path(onnx_dir).expanduser().resolve()
    engine_dir_path = Path(engine_dir).expanduser().resolve()

    onnx_by_variant: dict[str, tuple[int, int, Path]] = {}
    for p in sorted(onnx_dir_path.rglob("*.onnx")):
        m = re.search(r"_v(\d+)(x\d+)?_b(\d+)\.onnx$", p.name.lower())
        if not m:
            continue
        h = int(m.group(1))
        w = int((m.group(2) or "").lstrip("x")) if m.group(2) else h
        variant = f"v{m.group(1)}{m.group(2) or ''}_b{m.group(3)}"
        onnx_by_variant[variant] = (h, w, p)

    model_cases: list[tuple[str, int, int, int, Path, Path]] = []
    for p in sorted(engine_dir_path.rglob("*.engine")):
        m = re.search(r"_v(\d+)(x\d+)?_b(\d+)\.engine$", p.name.lower())
        if not m:
            continue
        variant = f"v{m.group(1)}{m.group(2) or ''}_b{m.group(3)}"
        h = int(m.group(1))
        w = int((m.group(2) or "").lstrip("x")) if m.group(2) else h
        batch_n = int(m.group(3))
        allowed = compare_matrix.get(f"v{m.group(1)}{m.group(2) or ''}")
        if allowed is None or (h, w) != allowed[:2] or batch_n not in allowed[2]:
            continue
        if variant in onnx_by_variant and onnx_by_variant[variant][0] == h and onnx_by_variant[variant][1] == w:
            model_cases.append((variant, h, w, batch_n, onnx_by_variant[variant][2], p))

    if not model_cases:
        raise FileNotFoundError("Brak pasujących par ONNX/ENGINE o tym samym wariancie _v..._b...")
    _, imgsz, _, _, onnx_path, engine_path = model_cases[0]

    script_path = Path(__file__)
    root_out_dir = resolve_out_dir(script_path, out_dir_arg)
    real_images_path = Path(real_images).expanduser().resolve() if real_images else None
    if real_images_path is not None:
        image_files_fixed = list_images(real_images_path)
        print(f"[INFO] mode=real | images: {real_images_path} | count={len(image_files_fixed)}")
    else:
        image_files_fixed = []
        print("[INFO] mode=synthetic | images will be generated per case at target imgsz")

    total_predict_calls = len(model_cases) * synthetic_count * len(BACKENDS)
    print(
        "[INFO] compare scope | "
        f"cases={len(model_cases)} synthetic_count={synthetic_count} "
        f"backend_predict_calls~={total_predict_calls} overlays_per_case<={max_overlays_per_case}"
    )

    all_case_records: list[dict[str, Any]] = []

    for case_idx, (variant, case_h, case_w, case_batch_n, case_onnx_path, case_engine_path) in enumerate(model_cases, start=1):
        out_dir = root_out_dir / f"case_{case_idx:03d}_{variant}"
        charts_dir = out_dir / "charts"
        overlays_dir = out_dir / "overlays"
        out_dir.mkdir(parents=True, exist_ok=True)
        charts_dir.mkdir(parents=True, exist_ok=True)
        overlays_dir.mkdir(parents=True, exist_ok=True)
        synthetic_tmp_dir: Path | None = None
        if real_images_path is not None:
            image_files = image_files_fixed
            images_source = str(real_images_path)
        else:
            synthetic_tmp_dir = Path(tempfile.mkdtemp(prefix="fastpath_synth_"))
            image_files = generate_synthetic_images(synthetic_tmp_dir, case_h, case_w, count=synthetic_count)
            images_source = "synthetic:temp"

        requested_device = str(device).lower().strip()
        models: dict[str, Any | None] = {}
        model_errors: dict[str, str] = {}
        warmup_imgsz: int | tuple[int, int] = (case_h, case_w) if case_h != case_w else case_h

        for key, pth in [("pt", pt_path), ("onnx", case_onnx_path), ("engine", case_engine_path)]:
            warmup_batch_size = case_batch_n
            model, err = create_model(
                pth,
                requested_device,
                warmup_batch_size=warmup_batch_size,
                warmup_imgsz=warmup_imgsz,
            )
            models[key] = model
            model_errors[key] = err

        device_used = requested_device
        if requested_device.startswith("cuda"):
            retry_needed = any(models[k] is None for k in BACKENDS)
            if retry_needed:
                for key, pth in [("pt", pt_path), ("onnx", case_onnx_path), ("engine", case_engine_path)]:
                    if models[key] is None:
                        warmup_batch_size = case_batch_n
                        model, err = create_model(
                            pth,
                            "cpu",
                            warmup_batch_size=warmup_batch_size,
                            warmup_imgsz=warmup_imgsz,
                        )
                        if model is not None:
                            models[key] = model
                            model_errors[key] = f"{model_errors[key]} | fallback_cpu_ok"
                if all(models[k] is not None for k in BACKENDS):
                    device_used = "cpu"

        rows: list[dict[str, Any]] = []
        pairs: list[dict[str, Any]] = []
        infer_imgsz: int | tuple[int, int] = (case_h, case_w) if case_h != case_w else case_h
        batched_by_backend: dict[str, list[dict[str, Any]]] = {}
        for key in ("pt", "onnx", "engine"):
            print(f"[BATCH] {variant}: {key.upper()} predict chunks batch={case_batch_n} images={len(image_files)}")
            batched_by_backend[key] = run_backend_on_images_batched(
                models[key],
                key,
                image_files,
                infer_imgsz,
                conf,
                iou,
                device_used,
                batch_size=case_batch_n,
            )

        rows_by_backend = {
            key: {r["image"]: r for r in backend_rows}
            for key, backend_rows in batched_by_backend.items()
        }

        for idx, image_path in enumerate(image_files, start=1):
            result_pt = rows_by_backend["pt"].get(
                image_path.name,
                _empty_backend_row(image_path.name, "pt", case_batch_n, "Brak wyniku PT."),
            )
            result_onnx = rows_by_backend["onnx"].get(
                image_path.name,
                _empty_backend_row(image_path.name, "onnx", case_batch_n, "Brak wyniku ONNX."),
            )
            result_engine = rows_by_backend["engine"].get(
                image_path.name,
                _empty_backend_row(image_path.name, "engine", case_batch_n, "Brak wyniku ENGINE."),
            )
            if result_pt.get("error") and model_errors["pt"]:
                result_pt["error"] = f"{model_errors['pt']} | {result_pt['error']}"
            if result_onnx.get("error") and model_errors["onnx"]:
                result_onnx["error"] = f"{model_errors['onnx']} | {result_onnx['error']}"
            if result_engine.get("error") and model_errors["engine"]:
                result_engine["error"] = f"{model_errors['engine']} | {result_engine['error']}"
            rows.extend([result_pt, result_onnx, result_engine])
            pairs.extend([
                compare_backend_vs_pt(image_path.name, "onnx", result_pt, result_onnx),
                compare_backend_vs_pt(image_path.name, "engine", result_pt, result_engine),
            ])
            if idx <= max_overlays_per_case:
                draw_overlay(image_path, overlays_dir / f"frame_{idx:03d}_compare.jpg", result_pt, result_onnx, result_engine)
            if idx == 1 or idx % 8 == 0 or idx == len(image_files):
                print(f"[COMPARE] {variant}: {idx}/{len(image_files)} images")

        summary = build_backend_summary(rows, pairs)
        for s in summary:
            s["backend_verdict"] = backend_final_verdict(s)
        save_charts(charts_dir, summary, rows, pairs)
        summary_by_backend = {s["backend"]: s for s in summary}
        pt_ms = safe_float(summary_by_backend.get("PT", {}).get("mean_total_ms"))
        onnx_ms = safe_float(summary_by_backend.get("ONNX", {}).get("mean_total_ms"))
        engine_ms = safe_float(summary_by_backend.get("ENGINE", {}).get("mean_total_ms"))
        pt_pre = safe_float(summary_by_backend.get("PT", {}).get("mean_preprocess_ms"))
        pt_inf = safe_float(summary_by_backend.get("PT", {}).get("mean_inference_ms"))
        pt_post = safe_float(summary_by_backend.get("PT", {}).get("mean_postprocess_ms"))
        pt_color = safe_float(summary_by_backend.get("PT", {}).get("mean_color_ms"))
        onnx_pre = safe_float(summary_by_backend.get("ONNX", {}).get("mean_preprocess_ms"))
        onnx_inf = safe_float(summary_by_backend.get("ONNX", {}).get("mean_inference_ms"))
        onnx_post = safe_float(summary_by_backend.get("ONNX", {}).get("mean_postprocess_ms"))
        onnx_color = safe_float(summary_by_backend.get("ONNX", {}).get("mean_color_ms"))
        engine_pre = safe_float(summary_by_backend.get("ENGINE", {}).get("mean_preprocess_ms"))
        engine_inf = safe_float(summary_by_backend.get("ENGINE", {}).get("mean_inference_ms"))
        engine_post = safe_float(summary_by_backend.get("ENGINE", {}).get("mean_postprocess_ms"))
        engine_color = safe_float(summary_by_backend.get("ENGINE", {}).get("mean_color_ms"))
        onnx_speedup = (pt_ms / onnx_ms) if (pt_ms and onnx_ms and onnx_ms > 0) else None
        engine_speedup = (pt_ms / engine_ms) if (pt_ms and engine_ms and engine_ms > 0) else None
        def _fmt_ms(v: float | None) -> str:
            return "-" if v is None else f"{v:.3f}"

        def _fmt_x(v: float | None) -> str:
            return "-" if v is None else f"{v:.2f}x"

        print(
            "[TIME] "
            f"{variant} (imgsz={case_h}x{case_w}, b={case_batch_n}) | "
            f"PT={_fmt_ms(pt_ms)} ms, "
            f"ONNX={_fmt_ms(onnx_ms)} ms ({_fmt_x(onnx_speedup)}), "
            f"ENGINE={_fmt_ms(engine_ms)} ms ({_fmt_x(engine_speedup)})"
        )
        print(
            "[STAGE per_image_ms] "
            f"PT color/pre/inf/post={_fmt_ms(pt_color)}/{_fmt_ms(pt_pre)}/{_fmt_ms(pt_inf)}/{_fmt_ms(pt_post)} | "
            f"ONNX color/pre/inf/post={_fmt_ms(onnx_color)}/{_fmt_ms(onnx_pre)}/{_fmt_ms(onnx_inf)}/{_fmt_ms(onnx_post)} | "
            f"ENGINE color/pre/inf/post={_fmt_ms(engine_color)}/{_fmt_ms(engine_pre)}/{_fmt_ms(engine_inf)}/{_fmt_ms(engine_post)}"
        )
        def _mean_row(rows_in: list[dict[str, Any]], backend_name: str, key: str) -> float | None:
            vals = [float(r[key]) for r in rows_in if r.get("backend") == backend_name and r.get("error") == "" and r.get(key) is not None]
            return mean_or_none(vals)

        pt_predict = _mean_row(rows, "PT", "predict_ms")
        onnx_predict = _mean_row(rows, "ONNX", "predict_ms")
        engine_predict = _mean_row(rows, "ENGINE", "predict_ms")
        pt_e2e = _mean_row(rows, "PT", "e2e_ms")
        onnx_e2e = _mean_row(rows, "ONNX", "e2e_ms")
        engine_e2e = _mean_row(rows, "ENGINE", "e2e_ms")
        print(
            "[CLOCK per_image_ms] "
            f"PT predict/e2e={_fmt_ms(pt_predict)}/{_fmt_ms(pt_e2e)} | "
            f"ONNX predict/e2e={_fmt_ms(onnx_predict)}/{_fmt_ms(onnx_e2e)} | "
            f"ENGINE predict/e2e={_fmt_ms(engine_predict)}/{_fmt_ms(engine_e2e)}"
        )

        report_md = build_report_md(
            out_dir=out_dir,
            params={
                "images": images_source,
                "pt": str(pt_path),
                "onnx": str(case_onnx_path),
                "engine": str(case_engine_path),
                "imgsz": f"{case_h}x{case_w}" if case_h != case_w else case_h,
                "batch_n": case_batch_n,
                "conf": conf,
                "iou": iou,
                "device_requested": requested_device,
                "device_used": device_used,
                "num_images": len(image_files),
            },
            summary=summary,
            pairs=pairs,
            rows=rows,
        )
        meta = {
            "variant": variant,
            "images": images_source,
            "pt": str(pt_path),
            "onnx": str(case_onnx_path),
            "engine": str(case_engine_path),
            "imgsz": f"{case_h}x{case_w}" if case_h != case_w else case_h,
            "batch_n": case_batch_n,
            "conf": conf,
            "iou": iou,
            "device_requested": requested_device,
            "device_used": device_used,
            "num_images": len(image_files),
            "output_dir": str(out_dir),
            "model_load_errors": model_errors,
        }
        write_json(out_dir / "backend_compare_results.json", {
            "meta": meta,
            "per_image_backend": rows,
            "pairs_vs_pt": pairs,
            "backend_summary": summary,
        })
        all_case_records.append({
            "meta": meta,
            "summary": summary,
        })
        write_csv_rows(out_dir / "backend_compare_summary.csv", summary, [
            "backend", "mean_total_ms", "p95_total_ms", "mean_preprocess_ms", "mean_inference_ms",
            "mean_postprocess_ms", "mean_color_ms", "mean_predict_ms", "mean_e2e_ms",
            "mean_total_ms_packet", "p95_total_ms_packet",
            "mean_preprocess_ms_packet", "p95_preprocess_ms_packet",
            "mean_inference_ms_packet", "p95_inference_ms_packet",
            "mean_postprocess_ms_packet", "p95_postprocess_ms_packet",
            "mean_predict_ms_packet", "p95_predict_ms_packet",
            "mean_e2e_ms_packet", "p95_e2e_ms_packet",
            "detection_rate_percent", "mean_best_confidence", "mean_iou_vs_pt",
            "missed_detections_vs_pt", "errors_count", "samples", "backend_verdict",
        ])
        write_csv_rows(out_dir / "backend_compare_pairs.csv", pairs, [
            "image", "pair", "matched", "iou", "conf_ref", "conf_test", "confidence_diff",
            "missed_detection", "extra_detection", "verdict", "error",
        ])
        (out_dir / "backend_compare_report.md").write_text(report_md, encoding="utf-8")
        if synthetic_tmp_dir is not None and synthetic_tmp_dir.exists():
            shutil.rmtree(synthetic_tmp_dir, ignore_errors=True)
        print(f"[DONE] {variant} -> {out_dir}")

    build_all_cases_summary(all_case_records, root_out_dir)

    return 0

    script_path = Path(__file__)
    out_dir = resolve_out_dir(script_path, out_dir_arg)
    charts_dir = out_dir / "charts"
    overlays_dir = out_dir / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    image_files = list_images(images_path)

    requested_device = str(device).lower().strip()
    models: dict[str, Any | None] = {}
    model_errors: dict[str, str] = {}

    for key, pth in [("pt", pt_path), ("onnx", onnx_path), ("engine", engine_path)]:
        model, err = create_model(pth, requested_device)
        models[key] = model
        model_errors[key] = err

    # fallback device na cpu dla backendów, które nie wstały na cuda
    device_used = requested_device
    if requested_device.startswith("cuda"):
        retry_needed = any(models[k] is None for k in BACKENDS)
        if retry_needed:
            for key, pth in [("pt", pt_path), ("onnx", onnx_path), ("engine", engine_path)]:
                if models[key] is None:
                    model, err = create_model(pth, "cpu")
                    if model is not None:
                        models[key] = model
                        model_errors[key] = f"{model_errors[key]} | fallback_cpu_ok"
            if all(models[k] is not None for k in BACKENDS):
                device_used = "cpu"

    rows: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []

    for idx, image_path in enumerate(image_files, start=1):
        result_pt = run_backend_on_image(models["pt"], "pt", image_path, imgsz, conf, iou, device_used)
        result_onnx = run_backend_on_image(models["onnx"], "onnx", image_path, imgsz, conf, iou, device_used)
        result_engine = run_backend_on_image(models["engine"], "engine", image_path, imgsz, conf, iou, device_used)

        if result_pt.get("error") and model_errors["pt"]:
            result_pt["error"] = f"{model_errors['pt']} | {result_pt['error']}"
        if result_onnx.get("error") and model_errors["onnx"]:
            result_onnx["error"] = f"{model_errors['onnx']} | {result_onnx['error']}"
        if result_engine.get("error") and model_errors["engine"]:
            result_engine["error"] = f"{model_errors['engine']} | {result_engine['error']}"

        rows.extend([result_pt, result_onnx, result_engine])

        pair_onnx = compare_backend_vs_pt(image_path.name, "onnx", result_pt, result_onnx)
        pair_engine = compare_backend_vs_pt(image_path.name, "engine", result_pt, result_engine)
        pairs.extend([pair_onnx, pair_engine])

        overlay_path = overlays_dir / f"frame_{idx:03d}_compare.jpg"
        draw_overlay(image_path, overlay_path, result_pt, result_onnx, result_engine)

    summary = build_backend_summary(rows, pairs)
    for s in summary:
        s["backend_verdict"] = backend_final_verdict(s)

    save_charts(charts_dir, summary, rows, pairs)

    report_md = build_report_md(
        out_dir=out_dir,
        params={
            "images": str(images_path),
            "pt": str(pt_path),
            "onnx": str(onnx_path),
            "engine": str(engine_path),
            "imgsz": imgsz,
            "conf": conf,
            "iou": iou,
            "device_requested": requested_device,
            "device_used": device_used,
            "num_images": len(image_files),
        },
        summary=summary,
        pairs=pairs,
        rows=rows,
    )

    results_json = {
        "meta": {
            "images": str(images_path),
            "pt": str(pt_path),
            "onnx": str(onnx_path),
            "engine": str(engine_path),
            "imgsz": imgsz,
            "conf": conf,
            "iou": iou,
            "device_requested": requested_device,
            "device_used": device_used,
            "num_images": len(image_files),
            "output_dir": str(out_dir),
            "model_load_errors": model_errors,
        },
        "per_image_backend": rows,
        "pairs_vs_pt": pairs,
        "backend_summary": summary,
    }

    write_json(out_dir / "backend_compare_results.json", results_json)

    write_csv_rows(
        out_dir / "backend_compare_summary.csv",
        summary,
        [
            "backend",
            "mean_total_ms",
            "p95_total_ms",
            "mean_preprocess_ms",
            "mean_inference_ms",
            "mean_postprocess_ms",
            "detection_rate_percent",
            "mean_best_confidence",
            "mean_iou_vs_pt",
            "missed_detections_vs_pt",
            "errors_count",
            "samples",
            "backend_verdict",
        ],
    )

    write_csv_rows(
        out_dir / "backend_compare_pairs.csv",
        pairs,
        [
            "image",
            "pair",
            "matched",
            "iou",
            "conf_ref",
            "conf_test",
            "confidence_diff",
            "missed_detection",
            "extra_detection",
            "verdict",
            "error",
        ],
    )

    (out_dir / "backend_compare_report.md").write_text(report_md, encoding="utf-8")

    print(f"[DONE] output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
