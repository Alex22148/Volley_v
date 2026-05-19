"""make_fastpath_lab_package.py — build dist/fastpath_lab_benchmark[.zip].

This script assembles the lab benchmark package by copying:
    - src/             (trt_runner, cuda_debayer, synthetic_raw, metrics,
                        report_builder, native_debayer/ with csrc)
    - scripts/         (00..05_*.py)
    - configs/         (benchmark_config.yaml, baseline_rtx2080super.json)
    - README.md, requirements.txt, RUN_ALL.ps1, CHECK_ENV.ps1
    - engines/         (only if --include-engines)

It explicitly does NOT copy:
    - GUI code, training code, dataset, runs/, reports/, logs/
    - live_backend_controller, shared_memory_manager, preview_worker
    - the ring buffer module
    - Basler / capture code
    - any Ultralytics weights (.pt) or dataset images

Usage:
    python tools/make_fastpath_lab_package.py --out dist
    python tools/make_fastpath_lab_package.py --out dist --include-engines
    python tools/make_fastpath_lab_package.py --out dist --include-engines --zip
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Iterable, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PACKAGE_DIR = REPO_ROOT / "dist" / "fastpath_lab_benchmark"   # template lives here
ENGINES_SRC = REPO_ROOT / "artifacts" / "tensorrt_exports"

ENGINE_MAP = {
    "best__fp16_640_b4_static.engine":
        "best__fp16_640_b4_static__fp16__img640__b4__static.engine",
    "best__fp16_960_b4_static.engine":
        "best__fp16_960_b4_static__fp16__img960__b4__static.engine",
    "best__fp16_1280_b4_static.engine":
        "best__fp16_1280_b4_static__fp16__img1280__b4__static.engine",
    "best__fp16_1088x1920_b4_static.engine":
        "best__fp16_1088x1920_b4_static__fp16__img1088x1920__b4__static.engine",
}

# Whitelist of files/directories to ship. Anything else is dropped.
WHITELIST_FILES = [
    "README.md",
    "requirements.txt",
    "RUN_ALL.ps1",
    "CHECK_ENV.ps1",
]
WHITELIST_DIRS = [
    "configs",
    "scripts",
    "src",
]
# What inside the directories above we keep.
KEEP_SRC_FILES = {
    "__init__.py",
    "trt_runner.py",
    "cuda_debayer.py",
    "synthetic_raw.py",
    "metrics.py",
    "report_builder.py",
}
KEEP_NATIVE_DEBAYER_FILES = {
    "__init__.py",
    "native_debayer.py",
}
KEEP_CSRC_FILES = {
    "debayer.cpp",
    "debayer_cuda.cu",
}
KEEP_SCRIPT_FILES = {
    "00_check_env.py",
    "01_benchmark_inference_only.py",
    "02_benchmark_color_only.py",
    "03_benchmark_full_synthetic_path.py",
    "04_resolution_sweep.py",
    "05_make_report.py",
    "06_run_single_test.py",
}
KEEP_CONFIG_FILES = {
    "benchmark_config.yaml",
    "baseline_rtx2080super.json",
    "single_test.json",
}
KEEP_DOCS_FILES = {
    "FASTPATH_LAB_USER_GUIDE.md",
    "FASTPATH_LAB_USER_GUIDE.html",
}


def _say(msg: str) -> None:
    print(f"[make_fastpath_lab] {msg}")


def _safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_whitelisted(src_root: Path, dst_root: Path) -> Iterable[Tuple[Path, Path]]:
    """Yield (src, dst) pairs for whitelisted files only."""
    for fname in WHITELIST_FILES:
        s = src_root / fname
        if s.exists():
            yield s, dst_root / fname

    # src/ tree
    src_src = src_root / "src"
    dst_src = dst_root / "src"
    for fname in KEEP_SRC_FILES:
        s = src_src / fname
        if s.exists():
            yield s, dst_src / fname
    nd_src = src_src / "native_debayer"
    nd_dst = dst_src / "native_debayer"
    for fname in KEEP_NATIVE_DEBAYER_FILES:
        s = nd_src / fname
        if s.exists():
            yield s, nd_dst / fname
    csrc_src = nd_src / "csrc"
    csrc_dst = nd_dst / "csrc"
    for fname in KEEP_CSRC_FILES:
        s = csrc_src / fname
        if s.exists():
            yield s, csrc_dst / fname

    # scripts/ tree
    for fname in KEEP_SCRIPT_FILES:
        s = src_root / "scripts" / fname
        if s.exists():
            yield s, dst_root / "scripts" / fname

    # configs/ tree
    for fname in KEEP_CONFIG_FILES:
        s = src_root / "configs" / fname
        if s.exists():
            yield s, dst_root / "configs" / fname

    # engines/README_ENGINES.md
    s = src_root / "engines" / "README_ENGINES.md"
    if s.exists():
        yield s, dst_root / "engines" / "README_ENGINES.md"

    # docs/ tree
    for fname in KEEP_DOCS_FILES:
        s = src_root / "docs" / fname
        if s.exists():
            yield s, dst_root / "docs" / fname


def build_package(out_root: Path, *, include_engines: bool, make_zip: bool) -> Path:
    if not SRC_PACKAGE_DIR.exists():
        raise FileNotFoundError(
            f"Source template missing: {SRC_PACKAGE_DIR}. "
            "Did you delete dist/fastpath_lab_benchmark/?"
        )

    out_root.mkdir(parents=True, exist_ok=True)
    pkg_dir = out_root / "fastpath_lab_benchmark"
    if pkg_dir.exists() and pkg_dir != SRC_PACKAGE_DIR:
        _say(f"removing existing {pkg_dir}")
        shutil.rmtree(pkg_dir)

    if pkg_dir == SRC_PACKAGE_DIR:
        _say("output target is the template directory — copying in place")
    else:
        _say(f"copying whitelisted files from {SRC_PACKAGE_DIR} -> {pkg_dir}")
        copied = 0
        for src, dst in _copy_whitelisted(SRC_PACKAGE_DIR, pkg_dir):
            _safe_copy(src, dst)
            copied += 1
        _say(f"copied {copied} files")

        # Empty results/ and reports/charts/
        (pkg_dir / "results").mkdir(parents=True, exist_ok=True)
        (pkg_dir / "reports" / "charts").mkdir(parents=True, exist_ok=True)
        (pkg_dir / "results" / ".gitkeep").write_text(
            "# JSON/CSV results land here.\n", encoding="utf-8")
        (pkg_dir / "reports" / ".gitkeep").write_text(
            "# Markdown/HTML/chart reports land here.\n", encoding="utf-8")

    # Engines
    eng_dst_dir = pkg_dir / "engines"
    eng_dst_dir.mkdir(parents=True, exist_ok=True)
    if include_engines:
        if not ENGINES_SRC.exists():
            _say(f"[WARN] {ENGINES_SRC} missing; cannot include engines.")
        else:
            for short, full in ENGINE_MAP.items():
                src_p = ENGINES_SRC / full
                if not src_p.exists():
                    _say(f"[WARN] engine not found: {src_p} (skipped)")
                    continue
                _safe_copy(src_p, eng_dst_dir / short)
                _say(f"engine: {short} ({src_p.stat().st_size / (1024 ** 2):.1f} MiB)")
    else:
        _say("--include-engines not set; engines/ kept empty.")

    # ZIP
    if make_zip:
        zip_path = out_root / "fastpath_lab_benchmark.zip"
        if zip_path.exists():
            zip_path.unlink()
        _say(f"writing zip: {zip_path}")
        skipped = 0
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in pkg_dir.rglob("*"):
                if not f.is_file():
                    continue
                rel = f.relative_to(out_root)
                parts = rel.parts
                # Skip generated artefacts so the tester gets a clean package.
                if "__pycache__" in parts:
                    skipped += 1
                    continue
                if len(parts) >= 2 and parts[1] in ("results", "reports"):
                    name = parts[-1]
                    if name in (".gitkeep", "README_ENGINES.md"):
                        pass
                    elif name.endswith((".json", ".csv", ".md", ".html", ".png", ".jpg")):
                        skipped += 1
                        continue
                zf.write(f, rel)
        _say(f"skipped {skipped} cache/result files; zip size: "
             f"{zip_path.stat().st_size / (1024 ** 2):.1f} MiB")
        return zip_path
    return pkg_dir


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(REPO_ROOT / "dist"),
                   help="output root (default: dist/)")
    p.add_argument("--include-engines", action="store_true",
                   help="copy 4 engine files from artifacts/tensorrt_exports/")
    p.add_argument("--zip", action="store_true",
                   help="also write fastpath_lab_benchmark.zip next to the folder")
    args = p.parse_args()
    out_root = Path(args.out).resolve()
    result = build_package(out_root,
                           include_engines=args.include_engines,
                           make_zip=args.zip)
    _say(f"done. result: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
