"""Build a portable fast-path benchmark package.

Copies templates + runtime modules into dist/volleyhub_fastpath_benchmark/
and (optionally) bundles TensorRT engines + zips the result.

Usage:
    python tools/make_portable_fastpath_package.py --include-engines --out dist/
    python tools/make_portable_fastpath_package.py --no-engines --out dist/

The resulting ZIP is at dist/volleyhub_fastpath_benchmark_windows_cuda.zip
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Iterable, List

REPO = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = REPO / "tools" / "_package_templates" / "volleyhub_fastpath_benchmark"
PACKAGE_NAME = "volleyhub_fastpath_benchmark"
ZIP_NAME = "volleyhub_fastpath_benchmark_windows_cuda.zip"

RUNTIME_PACKAGES = [
    "src/runtime_sources",
    "src/runtime_gpu",
    "src/runtime_inference",
    "src/runtime_production",
    "src/runtime_diagnostics",
]

EXTRA_FILES = [
    # SharedMemoryManager is consumed by the synthetic + basler grabbers.
    ("storage/__init__.py", "storage/__init__.py"),
    ("storage/shared_memory_manager.py", "storage/shared_memory_manager.py"),
]

ENGINE_DIR = REPO / "artifacts" / "tensorrt_exports"
EXPECTED_ENGINE_NAMES = [
    "best__fp16_640_b4_static__fp16__img640__b4__static.engine",
    "best__fp16_960_b4_static__fp16__img960__b4__static.engine",
    "best__fp16_1280_b4_static__fp16__img1280__b4__static.engine",
    "best__fp16_1088x1920_b4_static__fp16__img1088x1920__b4__static.engine",
]

_LOG = logging.getLogger("make_portable_fastpath_package")


def _copytree(src: Path, dst: Path) -> int:
    if not src.exists():
        _LOG.warning("source missing, skipped: %s", src)
        return 0
    n = 0
    for path in src.rglob("*"):
        if path.is_dir():
            continue
        # skip caches and build artefacts
        rel = path.relative_to(src)
        if any(part in {"__pycache__", ".pytest_cache", ".mypy_cache"} for part in rel.parts):
            continue
        if rel.suffix in {".pyc", ".pyo"}:
            continue
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        n += 1
    return n


def _copy_runtime(into: Path) -> None:
    # Copy `src/runtime_*` packages into `into/src/`
    src_root = into / "src"
    src_root.mkdir(parents=True, exist_ok=True)
    (src_root / "__init__.py").write_text(
        '"""Runtime modules for the portable fast-path benchmark."""\n', encoding="utf-8"
    )
    for rel in RUNTIME_PACKAGES:
        s = REPO / rel
        d = into / rel
        n = _copytree(s, d)
        _LOG.info("copied %d files: %s", n, rel)
    # Extra files (storage/ for SharedMemoryManager)
    for src_rel, dst_rel in EXTRA_FILES:
        s = REPO / src_rel
        if not s.exists():
            _LOG.warning("missing extra file: %s", src_rel)
            continue
        d = into / dst_rel
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)
        _LOG.info("copied extra: %s", src_rel)


def _copy_engines(into: Path, expected: Iterable[str]) -> List[dict]:
    eng_dst = into / "artifacts" / "tensorrt_exports"
    eng_dst.mkdir(parents=True, exist_ok=True)
    manifest: List[dict] = []
    for name in expected:
        src = ENGINE_DIR / name
        if src.exists():
            shutil.copy2(src, eng_dst / name)
            manifest.append({"engine": name, "bundled": True,
                             "size_bytes": src.stat().st_size})
            _LOG.info("bundled engine: %s (%.1f MB)", name, src.stat().st_size / 1e6)
        else:
            manifest.append({"engine": name, "bundled": False, "size_bytes": 0})
            _LOG.warning("engine NOT bundled (missing): %s", name)
    return manifest


def _ensure_dirs(into: Path) -> None:
    for d in ("reports", "logs", "artifacts/tensorrt_exports"):
        (into / d).mkdir(parents=True, exist_ok=True)
    # placeholder so empty dirs survive zipping
    keep = into / "reports" / ".gitkeep"
    keep.write_text("", encoding="utf-8")
    (into / "logs" / ".gitkeep").write_text("", encoding="utf-8")


def _build_zip(out_dir: Path, package_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / ZIP_NAME
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in package_dir.rglob("*"):
            if path.is_dir():
                continue
            arcname = Path(PACKAGE_NAME) / path.relative_to(package_dir)
            zf.write(path, arcname.as_posix())
    return zip_path


def _emit_manifest(package_dir: Path, manifest: List[dict]) -> None:
    import json
    (package_dir / "PACKAGE_MANIFEST.json").write_text(
        json.dumps({"engines": manifest}, indent=2), encoding="utf-8"
    )


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=REPO / "dist")
    parser.add_argument("--include-engines", action="store_true", default=True)
    parser.add_argument("--no-engines", dest="include_engines", action="store_false")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if not TEMPLATE_DIR.exists():
        raise SystemExit(f"template dir not found: {TEMPLATE_DIR}")

    out_dir: Path = args.out
    package_dir = out_dir / PACKAGE_NAME

    if package_dir.exists():
        _LOG.info("removing previous package dir: %s", package_dir)
        shutil.rmtree(package_dir, ignore_errors=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    _LOG.info("copying template -> %s", package_dir)
    _copytree(TEMPLATE_DIR, package_dir)
    _copy_runtime(package_dir)
    _ensure_dirs(package_dir)

    manifest: List[dict] = []
    if args.include_engines:
        manifest = _copy_engines(package_dir, EXPECTED_ENGINE_NAMES)
    else:
        _LOG.info("--no-engines: engines NOT bundled (will be checked/built on target machine)")
        manifest = [{"engine": n, "bundled": False, "size_bytes": 0}
                    for n in EXPECTED_ENGINE_NAMES]
    _emit_manifest(package_dir, manifest)

    _LOG.info("zipping...")
    zip_path = _build_zip(out_dir, package_dir)
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    _LOG.info("ZIP: %s (%.1f MB)", zip_path, size_mb)

    print()
    print("=" * 60)
    print(" Portable fast-path benchmark package built")
    print("=" * 60)
    print(f"  package dir: {package_dir}")
    print(f"  zip:         {zip_path}")
    print(f"  engines bundled: {sum(1 for m in manifest if m['bundled'])}/{len(manifest)}")
    print()
    print("On the target machine:")
    print("  1) unzip the file")
    print("  2) cd volleyhub_fastpath_benchmark")
    print("  3) .\\RUN_ME_FIRST.ps1")
    print("  4) .\\run_synthetic_sweep.ps1   (or .\\run_real_basler_sweep.ps1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
