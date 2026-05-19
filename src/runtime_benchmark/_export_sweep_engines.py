"""Export the TensorRT engines needed by the Full-HD inference-size sweep.

Each engine: batch=4, FP16, static shape. Skips engines that already
exist on disk under artifacts/tensorrt_exports/. Output filenames follow
the project's naming convention so the sweep runner can find them.
"""
from __future__ import annotations

import logging
import shutil
import sys
import time
from pathlib import Path
from typing import List, Tuple, Union

_LOG = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
PT_MODEL = REPO / "best.pt"
OUT_DIR = REPO / "artifacts" / "tensorrt_exports"

# Each entry: (config_label, imgsz: int|tuple, batch).
# imgsz can be int (square) or (H, W) tuple (rectangular).
SWEEP_CONFIGS: List[Tuple[str, Union[int, Tuple[int, int]], int]] = [
    ("fp16_640_b4_static",          640,             4),  # already exists; skip
    ("fp16_960_b4_static",          960,             4),
    ("fp16_1280_b4_static",         1280,            4),
    ("fp16_1088x1920_b4_static",    (1088, 1920),    4),
]


def _engine_target_path(stem: str, label: str, imgsz: Union[int, Tuple[int, int]], batch: int) -> Path:
    if isinstance(imgsz, tuple):
        h, w = int(imgsz[0]), int(imgsz[1])
        suffix = f"img{h}x{w}__b{batch}__static"
    else:
        suffix = f"img{int(imgsz)}__b{batch}__static"
    fname = f"{stem}__{label}__fp16__{suffix}.engine"
    return OUT_DIR / fname


def _resolve_exported_engine(pt_path: Path) -> Path:
    """Find the freshly-exported .engine that ultralytics drops next to the .pt."""
    candidates = sorted(pt_path.parent.glob("*.engine"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No .engine produced near {pt_path.parent}")
    return candidates[0]


def export_one(
    label: str,
    imgsz: Union[int, Tuple[int, int]],
    batch: int,
    *,
    half: bool = True,
    skip_if_exists: bool = True,
) -> dict:
    target = _engine_target_path(PT_MODEL.stem, label, imgsz, batch)
    started = time.perf_counter()
    if skip_if_exists and target.exists() and target.stat().st_size > 1024:
        _LOG.info("[EXPORT] skip (exists): %s", target.name)
        return {
            "label": label,
            "imgsz": imgsz,
            "batch": batch,
            "engine_path": str(target),
            "skipped_existing": True,
            "elapsed_s": 0.0,
            "error": None,
        }

    try:
        from ultralytics import YOLO
    except Exception as exc:
        return {"label": label, "engine_path": "", "error": f"ultralytics import failed: {exc}"}

    _LOG.info("[EXPORT] building %s  imgsz=%s batch=%d half=%s", label, imgsz, batch, half)
    try:
        model = YOLO(str(PT_MODEL))
        kwargs = dict(
            format="engine",
            imgsz=imgsz if not isinstance(imgsz, tuple) else list(imgsz),
            batch=batch,
            half=half,
            dynamic=False,
            simplify=True,
            device=0,
            verbose=False,
            workspace=4,
        )
        model.export(**kwargs)
        produced = _resolve_exported_engine(PT_MODEL)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        shutil.copy2(produced, target)
        _LOG.info("[EXPORT] saved -> %s (%.1f MB)", target.name, target.stat().st_size / 1024 / 1024)
        return {
            "label": label,
            "imgsz": imgsz,
            "batch": batch,
            "engine_path": str(target),
            "skipped_existing": False,
            "elapsed_s": time.perf_counter() - started,
            "error": None,
        }
    except Exception as exc:
        _LOG.exception("[EXPORT] failed for %s: %s", label, exc)
        return {
            "label": label,
            "imgsz": imgsz,
            "batch": batch,
            "engine_path": "",
            "skipped_existing": False,
            "elapsed_s": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main(argv: List[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if not PT_MODEL.exists():
        raise SystemExit(f"PT model not found: {PT_MODEL}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for label, imgsz, batch in SWEEP_CONFIGS:
        results.append(export_one(label, imgsz, batch))
    print()
    print("=" * 80)
    print("EXPORT SUMMARY")
    for r in results:
        status = "SKIP" if r.get("skipped_existing") else ("OK" if r.get("engine_path") and not r.get("error") else "FAIL")
        print(f"  [{status:>4}] {r['label']:<32} -> {r.get('engine_path') or '-'}  ({r.get('elapsed_s', 0):.1f}s)"
              + (f"  ERROR: {r['error']}" if r.get("error") else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
