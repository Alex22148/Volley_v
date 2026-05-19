"""Engine diagnostics + (optional) build for the portable benchmark.

`--check`         : list expected engines, report which are present
`--build-missing` : try to build missing engines from best.pt (must be in package root)
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import List, Tuple, Union

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
ENG_DIR = PACKAGE_ROOT / "artifacts" / "tensorrt_exports"

# label, imgsz (int or (H,W)), filename
EngineSpec = Tuple[str, Union[int, Tuple[int, int]], str]
SPECS: List[EngineSpec] = [
    ("fp16_640_b4_static",       640,             "best__fp16_640_b4_static__fp16__img640__b4__static.engine"),
    ("fp16_960_b4_static",       960,             "best__fp16_960_b4_static__fp16__img960__b4__static.engine"),
    ("fp16_1280_b4_static",      1280,            "best__fp16_1280_b4_static__fp16__img1280__b4__static.engine"),
    ("fp16_1088x1920_b4_static", (1088, 1920),    "best__fp16_1088x1920_b4_static__fp16__img1088x1920__b4__static.engine"),
]


def _check() -> int:
    print(f"engine dir: {ENG_DIR}")
    print()
    print(f"{'label':<32} {'present':<8} {'size (MB)':<10} {'expected_path'}")
    missing = 0
    for label, _imgsz, fname in SPECS:
        path = ENG_DIR / fname
        present = path.exists() and path.stat().st_size > 1024
        size_mb = (path.stat().st_size / (1024 * 1024)) if path.exists() else 0.0
        flag = "YES" if present else "NO"
        print(f"  {label:<30} {flag:<8} {size_mb:>8.2f}   {path.relative_to(PACKAGE_ROOT)}")
        if not present:
            missing += 1
    print()
    if missing:
        print(f"⚠ {missing} engine(s) missing.")
        print("  Options:")
        print("   1) python scripts\\export_or_check_engines.py --build-missing  (needs best.pt)")
        print("   2) copy the engine files into artifacts/tensorrt_exports/")
        print()
        print("  Note: TensorRT engines are GPU/driver/version specific. An engine")
        print("        built on another machine may NOT load here — rebuild locally.")
    else:
        print("✓ all engines present")
    return 0 if missing == 0 else 1


def _resolve_exported(pt_path: Path) -> Path:
    files = sorted(pt_path.parent.glob("*.engine"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"no .engine produced near {pt_path.parent}")
    return files[0]


def _build_one(label: str, imgsz: Union[int, Tuple[int, int]], fname: str,
               pt_path: Path, half: bool = True) -> bool:
    target = ENG_DIR / fname
    if target.exists() and target.stat().st_size > 1024:
        print(f"[skip] {label}: already exists at {target.name}")
        return True
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        print(f"[fail] {label}: ultralytics import failed: {exc!r}")
        return False
    print(f"[build] {label} imgsz={imgsz} batch=4 half={half}")
    started = time.perf_counter()
    try:
        model = YOLO(str(pt_path))
        kwargs = dict(
            format="engine",
            imgsz=imgsz if not isinstance(imgsz, tuple) else list(imgsz),
            batch=4,
            half=half,
            dynamic=False,
            simplify=True,
            device=0,
            verbose=False,
            workspace=4,
        )
        model.export(**kwargs)
    except Exception as exc:
        print(f"[fail] {label}: export raised: {type(exc).__name__}: {exc}")
        return False
    try:
        produced = _resolve_exported(pt_path)
        ENG_DIR.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        shutil.copy2(produced, target)
        elapsed = time.perf_counter() - started
        size_mb = target.stat().st_size / (1024 * 1024)
        print(f"[ok]   {label}: saved {target.name} ({size_mb:.1f} MB, {elapsed:.1f}s)")
        return True
    except Exception as exc:
        print(f"[fail] {label}: post-export copy failed: {exc!r}")
        return False


def _build_missing() -> int:
    pt_path = PACKAGE_ROOT / "best.pt"
    if not pt_path.exists():
        print(f"⚠ best.pt not found at {pt_path}.")
        print("  Drop your trained model into the package root and re-run.")
        return 2
    ENG_DIR.mkdir(parents=True, exist_ok=True)
    fail = 0
    for label, imgsz, fname in SPECS:
        ok = _build_one(label, imgsz, fname, pt_path)
        if not ok:
            fail += 1
    print()
    print("done." if fail == 0 else f"done with {fail} failure(s).")
    return 0 if fail == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="list expected engines + presence")
    parser.add_argument("--build-missing", action="store_true",
                        help="try to build missing engines (needs best.pt in package root)")
    args = parser.parse_args()
    if args.build_missing:
        return _build_missing()
    return _check()


if __name__ == "__main__":
    raise SystemExit(main())
