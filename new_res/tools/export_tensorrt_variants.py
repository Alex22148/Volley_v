from __future__ import annotations

from pathlib import Path
from ultralytics import YOLO
from dataclasses import dataclass, asdict
from typing import Any
import re
import traceback
import shutil
import json
import time
import yaml


DIR_RUNS = Path(r"E:\models_runs\test4\complementary_20260411_114516__restructured_v1\runs")
OUTPUT_ROOT_DIR = Path(r"E:\models_runs\test4\tensor_export")
ONNX_OUTPUT_ROOT_DIR = OUTPUT_ROOT_DIR.parent / "onnx"
TRIAL_DIR_PATTERN = re.compile(r"^exp_\d+__.+__trial_\d+$")

DEFAULT_DEVICE = 0
SKIP_IF_EXISTS = True
MIN_ENGINE_BYTES = 1024


@dataclass(slots=True)
class ExportResult:
    model_tag: str
    source_pt: str
    source_args_yaml: str
    config_name: str
    imgsz: int
    batch: int
    half: bool
    dynamic: bool
    simplify: bool
    workspace: float | int | None
    int8: bool
    device: str
    onnx_saved: bool
    engine_saved: bool
    onnx_path: str
    engine_path: str
    elapsed_s: float
    error: str = ""


def collect_pt_models(dir_runs: Path) -> list[Path]:
    pt_models: list[Path] = []
    for folder_path in dir_runs.iterdir():
        if not folder_path.is_dir():
            continue
        if not TRIAL_DIR_PATTERN.match(folder_path.name):
            continue

        pt_path = folder_path / "ultralytics" / "weights" / "best.pt"
        if pt_path.exists():
            pt_models.append(pt_path)

    return sorted(pt_models)


def safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    shutil.copy2(src, dst)


def resolve_exported_engine_path(pt_path: Path, exported_value: Any) -> Path:
    if exported_value:
        candidate = Path(str(exported_value))
        if candidate.exists() and candidate.suffix.lower() in {".engine", ".trt", ".rt"}:
            return candidate

    default_engine = pt_path.with_suffix(".engine")
    if default_engine.exists():
        return default_engine

    nearby_engines = sorted(pt_path.parent.glob("*.engine"), key=lambda p: p.stat().st_mtime)
    if nearby_engines:
        return nearby_engines[-1]

    nearby_trt = sorted(pt_path.parent.glob("*.trt"), key=lambda p: p.stat().st_mtime)
    if nearby_trt:
        return nearby_trt[-1]

    nearby_rt = sorted(pt_path.parent.glob("*.rt"), key=lambda p: p.stat().st_mtime)
    if nearby_rt:
        return nearby_rt[-1]

    raise FileNotFoundError(f"Export did not produce an engine-like file near: {pt_path.parent}")


def resolve_exported_onnx_path(pt_path: Path) -> Path:
    default_onnx = pt_path.with_suffix(".onnx")
    if default_onnx.exists():
        return default_onnx

    nearby_onnx = sorted(pt_path.parent.glob("*.onnx"), key=lambda p: p.stat().st_mtime)
    if nearby_onnx:
        return nearby_onnx[-1]

    raise FileNotFoundError(f"Export did not produce an ONNX file near: {pt_path.parent}")


def validate_engine_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Engine file not found: {path}")
    if path.stat().st_size < MIN_ENGINE_BYTES:
        raise ValueError(f"Engine file too small or invalid: {path}")


def resolve_args_yaml_for_pt(pt_path: Path) -> Path:
    candidates = [
        pt_path.parents[1] / "args.yaml",  # .../ultralytics/args.yaml
        pt_path.parents[2] / "args.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"args.yaml not found for model: {pt_path}")


def load_args_yaml(args_yaml_path: Path) -> dict[str, Any]:
    with open(args_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"args.yaml must contain a YAML mapping: {args_yaml_path}")

    return data


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    try:
        return bool(value)
    except Exception:
        return bool(default)


def _normalize_device(value: Any, default: int = 0) -> int | str:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    if text.isdigit():
        return int(text)
    return text


def build_export_configs_from_args_yaml(
    args_yaml: dict[str, Any],
    *,
    include_train_imgsz: bool = True,
    include_512: bool = True,
    include_640: bool = True,
    include_dynamic_variant: bool = True,
    include_batch4_variant: bool = True,
    prefer_half: bool = True,
    default_workspace: int | float = 4,
) -> list[dict[str, Any]]:
    train_imgsz = _as_int(args_yaml.get("imgsz", 640), 640)
    train_simplify = _as_bool(args_yaml.get("simplify", True), True)
    train_int8 = _as_bool(args_yaml.get("int8", False), False)
    train_device = _normalize_device(args_yaml.get("device", DEFAULT_DEVICE), DEFAULT_DEVICE)

    imgsz_candidates: list[int] = []
    if include_512:
        imgsz_candidates.append(512)
    if include_640:
        imgsz_candidates.append(640)
    if include_train_imgsz:
        imgsz_candidates.append(train_imgsz)

    seen: set[int] = set()
    ordered_imgsz: list[int] = []
    for v in imgsz_candidates:
        if v not in seen:
            seen.add(v)
            ordered_imgsz.append(v)

    configs: list[dict[str, Any]] = []

    for imgsz in ordered_imgsz:
        configs.append(
            {
                "name": f"fp16_{imgsz}_b1_static",
                "imgsz": imgsz,
                "batch": 1,
                "half": bool(prefer_half),
                "dynamic": False,
                "simplify": train_simplify,
                "workspace": default_workspace,
                "int8": False,
                "device": train_device,
                "save_onnx": imgsz in {640, train_imgsz},
            }
        )

    if include_dynamic_variant:
        configs.append(
            {
                "name": f"fp16_{train_imgsz}_b1_dynamic",
                "imgsz": train_imgsz,
                "batch": 1,
                "half": bool(prefer_half),
                "dynamic": True,
                "simplify": train_simplify,
                "workspace": default_workspace,
                "int8": False,
                "device": train_device,
                "save_onnx": False,
            }
        )

    if include_batch4_variant:
        configs.append(
            {
                "name": f"fp16_{train_imgsz}_b4_static",
                "imgsz": train_imgsz,
                "batch": 4,
                "half": bool(prefer_half),
                "dynamic": False,
                "simplify": train_simplify,
                "workspace": default_workspace,
                "int8": False,
                "device": train_device,
                "save_onnx": False,
            }
        )

    if train_int8:
        configs.append(
            {
                "name": f"int8_{train_imgsz}_b1_dynamic",
                "imgsz": train_imgsz,
                "batch": 1,
                "half": False,
                "dynamic": True,
                "simplify": train_simplify,
                "workspace": default_workspace,
                "int8": True,
                "device": train_device,
                "save_onnx": False,
            }
        )

    return configs


def build_export_configs_for_pt(pt_path: Path) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    args_yaml_path = resolve_args_yaml_for_pt(pt_path)
    args_yaml = load_args_yaml(args_yaml_path)
    export_configs = build_export_configs_from_args_yaml(args_yaml)
    return args_yaml_path, args_yaml, export_configs


def save_manifest(results: list[ExportResult]) -> Path:
    manifest_path = OUTPUT_ROOT_DIR / "export_manifest.json"
    payload = {
        "saved_at_epoch_ns": time.time_ns(),
        "default_device": DEFAULT_DEVICE,
        "results": [asdict(r) for r in results],
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def export_one_variant(model: YOLO, pt_path: Path, args_yaml_path: Path, config: dict[str, Any]) -> ExportResult:
    started = time.perf_counter()

    model_tag = pt_path.parents[2].name
    out_dir_engine = OUTPUT_ROOT_DIR / model_tag
    out_dir_onnx = ONNX_OUTPUT_ROOT_DIR / model_tag
    out_dir_engine.mkdir(parents=True, exist_ok=True)
    out_dir_onnx.mkdir(parents=True, exist_ok=True)

    stem = pt_path.stem
    cfg_name = str(config["name"])
    imgsz = int(config["imgsz"])
    batch = int(config["batch"])
    half = bool(config.get("half", True))
    dynamic = bool(config.get("dynamic", False))
    simplify = bool(config.get("simplify", True))
    workspace = config.get("workspace", None)
    int8 = bool(config.get("int8", False))
    save_onnx = bool(config.get("save_onnx", False))
    device = config.get("device", DEFAULT_DEVICE)

    target_engine = out_dir_engine / f"{stem}__{cfg_name}.engine"
    target_onnx = out_dir_onnx / f"{stem}__{cfg_name}.onnx"

    if SKIP_IF_EXISTS and target_engine.exists():
        try:
            validate_engine_file(target_engine)
            return ExportResult(
                model_tag=model_tag,
                source_pt=str(pt_path),
                source_args_yaml=str(args_yaml_path),
                config_name=cfg_name,
                imgsz=imgsz,
                batch=batch,
                half=half,
                dynamic=dynamic,
                simplify=simplify,
                workspace=workspace,
                int8=int8,
                device=str(device),
                onnx_saved=target_onnx.exists(),
                engine_saved=True,
                onnx_path=str(target_onnx) if target_onnx.exists() else "",
                engine_path=str(target_engine),
                elapsed_s=time.perf_counter() - started,
                error="skipped_existing",
            )
        except Exception:
            pass

    onnx_saved = False
    engine_saved = False
    onnx_path_str = ""
    engine_path_str = ""
    error_msg = ""

    try:
        print("-" * 100)
        print(f"[EXPORT] model={pt_path}")
        print(
            f"[EXPORT] config={cfg_name} "
            f"imgsz={imgsz} batch={batch} half={half} dynamic={dynamic} "
            f"simplify={simplify} int8={int8} device={device} workspace={workspace}"
        )

        if save_onnx and not (SKIP_IF_EXISTS and target_onnx.exists()):
            onnx_kwargs = dict(
                format="onnx",
                imgsz=imgsz,
                batch=batch,
                device=device,
                dynamic=dynamic,
                simplify=simplify,
                verbose=False,
            )
            model.export(**onnx_kwargs)
            exported_onnx_path = resolve_exported_onnx_path(pt_path)
            safe_copy(exported_onnx_path, target_onnx)
            onnx_saved = True
            onnx_path_str = str(target_onnx)
            print(f"[EXPORT] ONNX saved: {target_onnx}")
        elif target_onnx.exists():
            onnx_saved = True
            onnx_path_str = str(target_onnx)
            print(f"[EXPORT] ONNX reused: {target_onnx}")

        engine_kwargs = dict(
            format="engine",
            imgsz=imgsz,
            batch=batch,
            device=device,
            half=half,
            dynamic=dynamic,
            simplify=simplify,
            verbose=False,
            int8=int8,
        )
        if workspace is not None:
            engine_kwargs["workspace"] = workspace

        exported_value = model.export(**engine_kwargs)
        exported_engine_path = resolve_exported_engine_path(pt_path, exported_value)

        if target_engine.exists():
            target_engine.unlink()

        safe_copy(exported_engine_path, target_engine)
        validate_engine_file(target_engine)

        engine_saved = True
        engine_path_str = str(target_engine)
        print(f"[EXPORT] ENGINE saved: {target_engine}")

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        print(f"[EXPORT] ERROR: {error_msg}")
        print(traceback.format_exc())

    return ExportResult(
        model_tag=model_tag,
        source_pt=str(pt_path),
        source_args_yaml=str(args_yaml_path),
        config_name=cfg_name,
        imgsz=imgsz,
        batch=batch,
        half=half,
        dynamic=dynamic,
        simplify=simplify,
        workspace=workspace,
        int8=int8,
        device=str(device),
        onnx_saved=onnx_saved,
        engine_saved=engine_saved,
        onnx_path=onnx_path_str,
        engine_path=engine_path_str,
        elapsed_s=time.perf_counter() - started,
        error=error_msg,
    )


def export_one_pt_model(pt_path: Path) -> list[ExportResult]:
    model = YOLO(str(pt_path))
    args_yaml_path, args_yaml, export_configs = build_export_configs_for_pt(pt_path)
    results: list[ExportResult] = []

    print("=" * 100)
    print(f"[EXPORT] Source PT: {pt_path}")
    print(f"[EXPORT] args.yaml : {args_yaml_path}")
    print(f"[EXPORT] train imgsz={args_yaml.get('imgsz')} batch={args_yaml.get('batch')} "
          f"device={args_yaml.get('device')} dynamic={args_yaml.get('dynamic')} "
          f"simplify={args_yaml.get('simplify')} int8={args_yaml.get('int8')}")
    print(f"[EXPORT] Engine output root: {OUTPUT_ROOT_DIR}")
    print(f"[EXPORT] ONNX output root: {ONNX_OUTPUT_ROOT_DIR}")
    print("[EXPORT] Auto configs:")
    for cfg in export_configs:
        print(f"  - {cfg}")

    for config in export_configs:
        results.append(export_one_variant(model, pt_path, args_yaml_path, config))

    return results


def main() -> None:
    if not DIR_RUNS.exists():
        raise FileNotFoundError(f"Missing runs directory: {DIR_RUNS}")

    OUTPUT_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    ONNX_OUTPUT_ROOT_DIR.mkdir(parents=True, exist_ok=True)

    pt_models = collect_pt_models(DIR_RUNS)
    if not pt_models:
        raise FileNotFoundError(f"No models found in trial folders under: {DIR_RUNS}")

    print(f"[EXPORT] Found {len(pt_models)} model(s):")
    for pt in pt_models:
        print(f"  - {pt}")
    print()

    all_results: list[ExportResult] = []
    for pt in pt_models:
        all_results.extend(export_one_pt_model(pt))

    manifest_path = save_manifest(all_results)

    print()
    print("=" * 100)
    print("[EXPORT] SUMMARY")
    ok_count = sum(1 for r in all_results if r.engine_saved)
    fail_count = len(all_results) - ok_count
    print(f"[EXPORT] success={ok_count} fail={fail_count}")
    print(f"[EXPORT] manifest={manifest_path}")
    print("[EXPORT] Done.")


if __name__ == "__main__":
    main()