from __future__ import annotations

from pathlib import Path
from ultralytics import YOLO
from dataclasses import dataclass, asdict
import traceback
import shutil
import json
import time
from typing import Any


# ============================================================
# KONFIGURACJA — TU ZMIENIASZ ŚCIEŻKI
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[3]

PT_MODEL_PATH = REPO_ROOT / "best.pt"

ENGINE_OUTPUT_DIR = REPO_ROOT / "engines" / "int8"

ONNX_OUTPUT_DIR = ENGINE_OUTPUT_DIR.parent / "onnx" / "int8"

# Dataset do kalibracji INT8.
# Musi wskazywać na data.yaml używany przy treningu / walidacji.
CALIB_DATA_YAML = Path(
    r"E:\models_runs\test4\complementary_20260411_114516__restructured_v1\experiments\exp_001\data.yaml"
)

DEVICE = 0

SKIP_IF_EXISTS = True
WORKSPACE_GB = 2
MIN_ENGINE_BYTES = 1024

# Ile datasetu użyć do kalibracji INT8.
# 0.25 = 25% danych. Dla szybkiego testu OK.
# Dla finalnej wersji można dać 0.5 albo 1.0.
INT8_CALIB_FRACTION = 0.25


# ============================================================
# WARIANTY EKSPORTU INT8
# format: "label": [[height, width], [batch_1, batch_2, ...]]
# ============================================================

INPUT_MATRIX: dict[str, list[list[int]]] = {
    "v640": [[640, 640], [4, 8, 12]],
    "v960": [[960, 960], [4, 8, 12]],
    "v1088x1920": [[1088, 1920], [1, 4, 8]],
}


def build_static_int8_export_configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []

    for label, matrix in INPUT_MATRIX.items():
        if len(matrix) != 2:
            raise ValueError(f"Invalid matrix for {label}: expected [[H,W],[batches...]], got: {matrix}")

        hw, batches = matrix
        if len(hw) != 2:
            raise ValueError(f"Invalid resolution for {label}: expected [H,W], got: {hw}")

        imgsz_value = [int(hw[0]), int(hw[1])]

        for batch in batches:
            configs.append(
                {
                    "name": f"int8_{label}_b{int(batch)}",
                    "imgsz": imgsz_value,
                    "batch": int(batch),
                    "half": False,
                    "dynamic": False,
                    "simplify": True,
                    "workspace": WORKSPACE_GB,
                    "int8": True,
                    "save_onnx": False,
                    "calib_data": CALIB_DATA_YAML,
                    "calib_fraction": INT8_CALIB_FRACTION,
                }
            )

    return configs


EXPORT_CONFIGS = build_static_int8_export_configs()


@dataclass(slots=True)
class ExportResult:
    source_pt: str
    config_name: str
    imgsz: int | list[int]
    batch: int
    half: bool
    dynamic: bool
    simplify: bool
    workspace: float | int | None
    int8: bool
    calib_data: str
    calib_fraction: float
    onnx_saved: bool
    engine_saved: bool
    onnx_path: str
    engine_path: str
    elapsed_s: float
    error: str = ""


def safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        dst.unlink()

    shutil.copy2(src, dst)


def validate_engine_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Engine file not found: {path}")

    size = path.stat().st_size
    if size < MIN_ENGINE_BYTES:
        raise ValueError(f"Engine file too small or invalid: {path} | size={size} bytes")


def normalize_imgsz(value: Any) -> int | list[int]:
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"imgsz list/tuple must have 2 values, got: {value}")
        return [int(value[0]), int(value[1])]

    return int(value)


def resolve_exported_engine_path(pt_path: Path, exported_value: Any) -> Path:
    """
    Ultralytics czasem zwraca ścieżkę do engine,
    a czasem zapisuje plik obok .pt.
    """
    if exported_value:
        candidate = Path(str(exported_value))
        if candidate.exists() and candidate.suffix.lower() in {".engine", ".trt", ".rt"}:
            return candidate

    default_engine = pt_path.with_suffix(".engine")
    if default_engine.exists():
        return default_engine

    nearby_engines = sorted(
        pt_path.parent.glob("*.engine"),
        key=lambda p: p.stat().st_mtime,
    )
    if nearby_engines:
        return nearby_engines[-1]

    nearby_trt = sorted(
        pt_path.parent.glob("*.trt"),
        key=lambda p: p.stat().st_mtime,
    )
    if nearby_trt:
        return nearby_trt[-1]

    nearby_rt = sorted(
        pt_path.parent.glob("*.rt"),
        key=lambda p: p.stat().st_mtime,
    )
    if nearby_rt:
        return nearby_rt[-1]

    raise FileNotFoundError(f"Export did not produce an engine-like file near: {pt_path.parent}")


def resolve_exported_onnx_path(pt_path: Path) -> Path:
    default_onnx = pt_path.with_suffix(".onnx")
    if default_onnx.exists():
        return default_onnx

    nearby_onnx = sorted(
        pt_path.parent.glob("*.onnx"),
        key=lambda p: p.stat().st_mtime,
    )
    if nearby_onnx:
        return nearby_onnx[-1]

    raise FileNotFoundError(f"Export did not produce an ONNX file near: {pt_path.parent}")


def cleanup_temp_export_files(pt_path: Path, exported_engine_path: Path | None = None) -> None:
    """
    Sprząta tymczasowe pliki generowane przez Ultralytics obok .pt:
    best.onnx / best.engine.
    Dzięki temu kolejny wariant nie złapie starego pliku.
    """
    temp_candidates = [
        pt_path.with_suffix(".onnx"),
        pt_path.with_suffix(".engine"),
    ]

    if exported_engine_path is not None:
        temp_candidates.append(exported_engine_path)

    for p in temp_candidates:
        try:
            if p.exists():
                p.unlink()
        except Exception as exc:
            print(f"[WARN] Could not remove temporary export file: {p} | {exc}")


def save_manifest(results: list[ExportResult], output_dir: Path, device: int | str) -> Path:
    manifest_path = output_dir / "export_manifest_int8_v1.json"

    payload = {
        "saved_at_epoch_ns": time.time_ns(),
        "device": device,
        "source_pt": str(PT_MODEL_PATH),
        "calib_data_yaml": str(CALIB_DATA_YAML),
        "calib_fraction": INT8_CALIB_FRACTION,
        "variants": {
            "inputs_matrix": INPUT_MATRIX,
            "precision": "int8",
            "dynamic": False,
            "mode": "static",
            "workspace_gb": WORKSPACE_GB,
        },
        "results": [asdict(r) for r in results],
    }

    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return manifest_path


def export_one_variant(
    model: YOLO,
    pt_path: Path,
    output_dir: Path,
    onnx_output_dir: Path,
    config: dict[str, Any],
    device: int | str,
    skip_if_exists: bool,
) -> ExportResult:
    started = time.perf_counter()

    stem = pt_path.stem
    cfg_name = str(config["name"])

    imgsz = normalize_imgsz(config["imgsz"])
    batch = int(config["batch"])
    half = bool(config.get("half", False))
    dynamic = bool(config.get("dynamic", False))
    simplify = bool(config.get("simplify", True))
    workspace = config.get("workspace", None)
    int8 = bool(config.get("int8", True))
    save_onnx = bool(config.get("save_onnx", False))

    calib_data = Path(config.get("calib_data", CALIB_DATA_YAML)).resolve()
    calib_fraction = float(config.get("calib_fraction", INT8_CALIB_FRACTION))

    target_engine = output_dir / f"{stem}__{cfg_name}.engine"
    target_onnx = onnx_output_dir / f"{stem}__{cfg_name}.onnx"

    if skip_if_exists and target_engine.exists():
        try:
            validate_engine_file(target_engine)

            print("-" * 100)
            print("[SKIP] Existing valid INT8 engine:")
            print(f"       {target_engine}")

            return ExportResult(
                source_pt=str(pt_path),
                config_name=cfg_name,
                imgsz=imgsz,
                batch=batch,
                half=half,
                dynamic=dynamic,
                simplify=simplify,
                workspace=workspace,
                int8=int8,
                calib_data=str(calib_data),
                calib_fraction=calib_fraction,
                onnx_saved=target_onnx.exists(),
                engine_saved=True,
                onnx_path=str(target_onnx) if target_onnx.exists() else "",
                engine_path=str(target_engine),
                elapsed_s=time.perf_counter() - started,
                error="skipped_existing",
            )
        except Exception as exc:
            print("-" * 100)
            print("[WARN] Existing engine invalid, re-exporting:")
            print(f"       {target_engine}")
            print(f"       reason: {type(exc).__name__}: {exc}")

    onnx_saved = False
    engine_saved = False
    onnx_path_str = ""
    engine_path_str = ""
    error_msg = ""

    try:
        if not calib_data.exists():
            raise FileNotFoundError(f"Missing INT8 calibration data.yaml: {calib_data}")

        print("-" * 100)
        print(f"[EXPORT INT8] PT       : {pt_path}")
        print(f"[EXPORT INT8] CONFIG   : {cfg_name}")
        print(f"[EXPORT INT8] imgsz    : {imgsz}")
        print(f"[EXPORT INT8] batch    : {batch}")
        print(f"[EXPORT INT8] half     : {half}")
        print(f"[EXPORT INT8] int8     : {int8}")
        print(f"[EXPORT INT8] dynamic  : {dynamic}")
        print(f"[EXPORT INT8] device   : {device}")
        print(f"[EXPORT INT8] calib    : {calib_data}")
        print(f"[EXPORT INT8] fraction : {calib_fraction}")
        print(f"[EXPORT INT8] output   : {target_engine}")

        # ========================================================
        # Optional ONNX export
        # ========================================================
        if save_onnx and not (skip_if_exists and target_onnx.exists()):
            onnx_kwargs = dict(
                format="onnx",
                imgsz=imgsz,
                batch=batch,
                device=device,
                dynamic=dynamic,
                simplify=simplify,
                verbose=False,
            )

            print("[EXPORT INT8] Exporting ONNX...")
            model.export(**onnx_kwargs)

            exported_onnx_path = resolve_exported_onnx_path(pt_path)
            safe_copy(exported_onnx_path, target_onnx)

            onnx_saved = True
            onnx_path_str = str(target_onnx)

            print("[EXPORT INT8] ONNX saved:")
            print(f"              {target_onnx}")

        elif target_onnx.exists():
            onnx_saved = True
            onnx_path_str = str(target_onnx)

            print("[SKIP] Existing ONNX:")
            print(f"       {target_onnx}")

        # ========================================================
        # TensorRT INT8 engine export
        # ========================================================
        engine_kwargs = dict(
            format="engine",
            imgsz=imgsz,
            batch=batch,
            device=device,
            half=half,
            dynamic=dynamic,
            simplify=simplify,
            verbose=False,
            int8=True,
            data=str(calib_data),
            fraction=calib_fraction,
        )

        if workspace is not None:
            engine_kwargs["workspace"] = workspace

        print("[EXPORT INT8] Exporting TensorRT INT8 engine...")
        exported_value = model.export(**engine_kwargs)

        exported_engine_path = resolve_exported_engine_path(pt_path, exported_value)

        safe_copy(exported_engine_path, target_engine)
        validate_engine_file(target_engine)

        engine_saved = True
        engine_path_str = str(target_engine)

        print("[EXPORT INT8] ENGINE saved:")
        print(f"              {target_engine}")
        print(f"[EXPORT INT8] ENGINE size: {target_engine.stat().st_size / (1024 * 1024):.2f} MiB")

        cleanup_temp_export_files(pt_path, exported_engine_path)

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"

        print(f"[EXPORT INT8] ERROR: {error_msg}")
        print(traceback.format_exc())

        try:
            cleanup_temp_export_files(pt_path)
        except Exception:
            pass

    return ExportResult(
        source_pt=str(pt_path),
        config_name=cfg_name,
        imgsz=imgsz,
        batch=batch,
        half=half,
        dynamic=dynamic,
        simplify=simplify,
        workspace=workspace,
        int8=int8,
        calib_data=str(calib_data),
        calib_fraction=calib_fraction,
        onnx_saved=onnx_saved,
        engine_saved=engine_saved,
        onnx_path=onnx_path_str,
        engine_path=engine_path_str,
        elapsed_s=time.perf_counter() - started,
        error=error_msg,
    )


def main() -> None:
    pt_path = PT_MODEL_PATH.expanduser().resolve()
    output_dir = ENGINE_OUTPUT_DIR.expanduser().resolve()
    onnx_output_dir = ONNX_OUTPUT_DIR.expanduser().resolve()
    calib_data = CALIB_DATA_YAML.expanduser().resolve()

    if not pt_path.exists():
        raise FileNotFoundError(f"Missing .pt model: {pt_path}")

    if pt_path.suffix.lower() != ".pt":
        raise ValueError(f"Input model must be .pt, got: {pt_path}")

    if not calib_data.exists():
        raise FileNotFoundError(f"Missing calibration data.yaml: {calib_data}")

    if str(DEVICE).lower() == "cpu":
        raise ValueError("TensorRT INT8 engine export wymaga CUDA. Ustaw DEVICE = 0, nie CPU.")

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_output_dir.mkdir(parents=True, exist_ok=True)

    device = int(DEVICE)

    configs = []
    for cfg in EXPORT_CONFIGS:
        cfg_copy = dict(cfg)
        cfg_copy["workspace"] = WORKSPACE_GB
        cfg_copy["calib_data"] = calib_data
        cfg_copy["calib_fraction"] = INT8_CALIB_FRACTION
        configs.append(cfg_copy)

    print("=" * 100)
    print("[EXPORT] ONE PT -> MANY STATIC INT8 ENGINES / V1")
    print(f"[EXPORT] source PT       : {pt_path}")
    print(f"[EXPORT] output dir      : {output_dir}")
    print(f"[EXPORT] onnx dir        : {onnx_output_dir}")
    print(f"[EXPORT] calib data.yaml : {calib_data}")
    print(f"[EXPORT] calib fraction  : {INT8_CALIB_FRACTION}")
    print(f"[EXPORT] device          : {device}")
    print(f"[EXPORT] skip existing   : {SKIP_IF_EXISTS}")
    print(f"[EXPORT] workspace GB    : {WORKSPACE_GB}")
    print(f"[EXPORT] configs count   : {len(configs)}")
    print()
    print("[EXPORT] Planned INT8 variants:")

    for cfg in configs:
        print(
            f"  - {cfg['name']} | imgsz={cfg['imgsz']} | batch={cfg['batch']} | "
            f"int8={cfg['int8']} | half={cfg['half']} | dynamic={cfg['dynamic']}"
        )

    print("=" * 100)

    model = YOLO(str(pt_path))

    all_results: list[ExportResult] = []

    for config in configs:
        result = export_one_variant(
            model=model,
            pt_path=pt_path,
            output_dir=output_dir,
            onnx_output_dir=onnx_output_dir,
            config=config,
            device=device,
            skip_if_exists=SKIP_IF_EXISTS,
        )
        all_results.append(result)

    manifest_path = save_manifest(all_results, output_dir, device)

    ok_count = sum(1 for r in all_results if r.engine_saved)
    fail_count = len(all_results) - ok_count
    skipped_count = sum(1 for r in all_results if r.error == "skipped_existing")

    print()
    print("=" * 100)
    print("[EXPORT INT8] SUMMARY")
    print(f"[EXPORT INT8] success          : {ok_count}")
    print(f"[EXPORT INT8] skipped existing : {skipped_count}")
    print(f"[EXPORT INT8] fail             : {fail_count}")
    print(f"[EXPORT INT8] manifest         : {manifest_path}")

    if fail_count:
        print()
        print("[EXPORT INT8] FAILED CONFIGS:")
        for r in all_results:
            if not r.engine_saved:
                print(f"  - {r.config_name}: {r.error}")

    print("[EXPORT INT8] Done.")


if __name__ == "__main__":
    main()
