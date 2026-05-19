from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import time
import traceback
from typing import Any

from ultralytics import YOLO


# ============================================================
# KONFIGURACJA
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[3]

PT_MODEL_PATH = REPO_ROOT / "best.pt"
ONNX_OUTPUT_DIR = REPO_ROOT / "engines" / "onnx" / "static"

DEVICE = 0
SKIP_IF_EXISTS = True
MIN_ONNX_BYTES = 1024

# Nazwy muszą pasować do export2engine.py:
# best__<cfg_name>.engine  <->  best__<cfg_name>.onnx
USE_FP16_NAMING = True

INPUT_MATRIX: dict[str, list[list[int]]] = {
    "v640": [[640, 640], [4, 8, 12]],
    "v960": [[960, 960], [4, 8, 12]],
    "v1088x1920": [[1088, 1920], [1, 4, 8]],
}


def build_onnx_export_configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    precision = "fp16" if USE_FP16_NAMING else "fp32"

    for label, matrix in INPUT_MATRIX.items():
        if len(matrix) != 2:
            raise ValueError(f"Invalid matrix for {label}: {matrix}")
        hw, batches = matrix
        if len(hw) != 2:
            raise ValueError(f"Invalid resolution for {label}: {hw}")
        imgsz = [int(hw[0]), int(hw[1])]

        for batch in batches:
            configs.append(
                {
                    "name": f"{precision}_{label}_b{int(batch)}",
                    "imgsz": imgsz,
                    "batch": int(batch),
                    "dynamic": False,
                    "simplify": True,
                }
            )
    return configs


EXPORT_CONFIGS = build_onnx_export_configs()


@dataclass(slots=True)
class ExportResult:
    source_pt: str
    config_name: str
    imgsz: int | list[int]
    batch: int
    dynamic: bool
    simplify: bool
    onnx_saved: bool
    onnx_path: str
    elapsed_s: float
    error: str = ""


def normalize_imgsz(value: Any) -> int | list[int]:
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"imgsz list/tuple must have 2 values, got: {value}")
        return [int(value[0]), int(value[1])]
    return int(value)


def safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    shutil.copy2(src, dst)


def validate_onnx_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"ONNX file not found: {path}")
    size = path.stat().st_size
    if size < MIN_ONNX_BYTES:
        raise ValueError(f"ONNX file too small or invalid: {path} | size={size}")


def resolve_exported_onnx_path(pt_path: Path, exported_value: Any) -> Path:
    if exported_value:
        candidate = Path(str(exported_value))
        if candidate.exists() and candidate.suffix.lower() == ".onnx":
            return candidate

    default_onnx = pt_path.with_suffix(".onnx")
    if default_onnx.exists():
        return default_onnx

    nearby_onnx = sorted(
        pt_path.parent.glob("*.onnx"),
        key=lambda p: p.stat().st_mtime,
    )
    if nearby_onnx:
        return nearby_onnx[-1]

    raise FileNotFoundError(f"Export did not produce ONNX near: {pt_path.parent}")


def cleanup_temp_onnx(pt_path: Path, exported_onnx: Path | None = None) -> None:
    candidates = [pt_path.with_suffix(".onnx")]
    if exported_onnx is not None:
        candidates.append(exported_onnx)

    for p in candidates:
        try:
            if p.exists():
                p.unlink()
        except Exception as exc:
            print(f"[WARN] Could not remove temp ONNX: {p} | {exc}")


def save_manifest(results: list[ExportResult], output_dir: Path, device: int | str) -> Path:
    manifest_path = output_dir / "export_manifest_onnx_v1.json"
    payload = {
        "saved_at_epoch_ns": time.time_ns(),
        "device": device,
        "source_pt": str(PT_MODEL_PATH),
        "variants": {
            "inputs_matrix": INPUT_MATRIX,
            "precision_naming": "fp16" if USE_FP16_NAMING else "fp32",
            "dynamic": False,
            "format": "onnx",
        },
        "results": [asdict(r) for r in results],
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def export_one_variant(
    model: YOLO,
    pt_path: Path,
    output_dir: Path,
    config: dict[str, Any],
    device: int | str,
    skip_if_exists: bool,
) -> ExportResult:
    started = time.perf_counter()

    stem = pt_path.stem
    cfg_name = str(config["name"])
    imgsz = normalize_imgsz(config["imgsz"])
    batch = int(config["batch"])
    dynamic = bool(config.get("dynamic", False))
    simplify = bool(config.get("simplify", True))

    target_onnx = output_dir / f"{stem}__{cfg_name}.onnx"

    if skip_if_exists and target_onnx.exists():
        try:
            validate_onnx_file(target_onnx)
            print("-" * 100)
            print("[SKIP] Existing valid ONNX:")
            print(f"       {target_onnx}")
            return ExportResult(
                source_pt=str(pt_path),
                config_name=cfg_name,
                imgsz=imgsz,
                batch=batch,
                dynamic=dynamic,
                simplify=simplify,
                onnx_saved=True,
                onnx_path=str(target_onnx),
                elapsed_s=time.perf_counter() - started,
                error="skipped_existing",
            )
        except Exception as exc:
            print("-" * 100)
            print("[WARN] Existing ONNX invalid, re-exporting:")
            print(f"       {target_onnx}")
            print(f"       reason: {type(exc).__name__}: {exc}")

    onnx_saved = False
    onnx_path_str = ""
    error_msg = ""

    try:
        print("-" * 100)
        print(f"[EXPORT ONNX] PT      : {pt_path}")
        print(f"[EXPORT ONNX] CONFIG  : {cfg_name}")
        print(f"[EXPORT ONNX] imgsz   : {imgsz}")
        print(f"[EXPORT ONNX] batch   : {batch}")
        print(f"[EXPORT ONNX] dynamic : {dynamic}")
        print(f"[EXPORT ONNX] device  : {device}")
        print(f"[EXPORT ONNX] output  : {target_onnx}")

        kwargs = dict(
            format="onnx",
            imgsz=imgsz,
            batch=batch,
            device=device,
            dynamic=dynamic,
            simplify=simplify,
            verbose=False,
        )
        exported_value = model.export(**kwargs)
        exported_onnx = resolve_exported_onnx_path(pt_path, exported_value)
        safe_copy(exported_onnx, target_onnx)
        validate_onnx_file(target_onnx)
        cleanup_temp_onnx(pt_path, exported_onnx)

        onnx_saved = True
        onnx_path_str = str(target_onnx)
        print("[EXPORT ONNX] ONNX saved:")
        print(f"              {target_onnx}")

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        print(f"[EXPORT ONNX] ERROR: {error_msg}")
        print(traceback.format_exc())
        try:
            cleanup_temp_onnx(pt_path)
        except Exception:
            pass

    return ExportResult(
        source_pt=str(pt_path),
        config_name=cfg_name,
        imgsz=imgsz,
        batch=batch,
        dynamic=dynamic,
        simplify=simplify,
        onnx_saved=onnx_saved,
        onnx_path=onnx_path_str,
        elapsed_s=time.perf_counter() - started,
        error=error_msg,
    )


def main() -> None:
    pt_path = PT_MODEL_PATH.expanduser().resolve()
    output_dir = ONNX_OUTPUT_DIR.expanduser().resolve()

    if not pt_path.exists():
        raise FileNotFoundError(f"Missing .pt model: {pt_path}")
    if pt_path.suffix.lower() != ".pt":
        raise ValueError(f"Input model must be .pt, got: {pt_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    device = int(DEVICE)

    print("=" * 100)
    print("[EXPORT] ONE PT -> MANY ONNX / V1")
    print(f"[EXPORT] source PT      : {pt_path}")
    print(f"[EXPORT] onnx output dir: {output_dir}")
    print(f"[EXPORT] naming profile : {'fp16' if USE_FP16_NAMING else 'fp32'}")
    print(f"[EXPORT] device         : {device}")
    print(f"[EXPORT] skip existing  : {SKIP_IF_EXISTS}")
    print(f"[EXPORT] configs count  : {len(EXPORT_CONFIGS)}")
    print()
    print("[EXPORT] Planned ONNX variants:")
    for cfg in EXPORT_CONFIGS:
        print(f"  - {cfg['name']} | imgsz={cfg['imgsz']} | batch={cfg['batch']}")
    print("=" * 100)

    model = YOLO(str(pt_path))
    results: list[ExportResult] = []
    for cfg in EXPORT_CONFIGS:
        results.append(
            export_one_variant(
                model=model,
                pt_path=pt_path,
                output_dir=output_dir,
                config=cfg,
                device=device,
                skip_if_exists=SKIP_IF_EXISTS,
            )
        )

    manifest = save_manifest(results, output_dir, device)
    ok = sum(1 for r in results if r.onnx_saved)
    fail = len(results) - ok
    skipped = sum(1 for r in results if r.error == "skipped_existing")

    print()
    print("=" * 100)
    print("[EXPORT ONNX] SUMMARY")
    print(f"[EXPORT ONNX] success          : {ok}")
    print(f"[EXPORT ONNX] skipped existing : {skipped}")
    print(f"[EXPORT ONNX] fail             : {fail}")
    print(f"[EXPORT ONNX] manifest         : {manifest}")

    if fail:
        print()
        print("[EXPORT ONNX] FAILED CONFIGS:")
        for r in results:
            if not r.onnx_saved:
                print(f"  - {r.config_name}: {r.error}")

    print("[EXPORT ONNX] Done.")


if __name__ == "__main__":
    main()

