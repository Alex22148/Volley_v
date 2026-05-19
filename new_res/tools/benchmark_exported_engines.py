from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from vision_runtime.core.pipeline_runner import OfflinePipelineRunner, RunnerConfig


# =========================================
# USTAWIENIA
# =========================================

BASE_DIR = Path(r"C:\Users\Hyperbook\python_project\VolleyHub_K\new_res")
INPUT_DIR = BASE_DIR / "sample_frames"

# Możesz wskazać:
# - cały root z engine: E:\models_runs\test4\tensor_export
# - albo jeden konkretny folder triala
ENGINES_DIR = Path(r"E:\models_runs\test4\tensor_export")

BENCHMARK_OUTPUT_DIR = BASE_DIR / "artifacts" / "tensorrt_benchmark"

REFERENCE_PT = Path(r"C:\Users\Hyperbook\python_project\VolleyHub_K\best.pt")
RUN_ULTRALYTICS_REFERENCE = False

DEFAULT_DEVICE = "cuda"
CONFIDENCE = 0.25
DEBAYER_BACKEND = "cpu_opencv"
BAYER_PATTERN = "BG"
SAVE_OVERLAYS = False
SAVE_JSON = True

# Jeśli chcesz ograniczyć benchmark tylko do części engine:
ENGINE_NAME_CONTAINS: Optional[str] = None
ONLY_OK_FROM_MANIFEST = True


# =========================================
# TYPY
# =========================================

@dataclass(slots=True)
class BenchmarkRow:
    name: str
    backend: str
    model_path: str
    engine_path: str
    config_name: str
    image_size: int
    batch: int
    device: str
    frames_total: int
    frames_processed: int
    detections_total: int
    tracks_total: int
    avg_load_ms: float
    avg_convert_ms: float
    avg_preprocess_ms: float
    avg_infer_ms: float
    avg_track_ms: float
    avg_total_ms: float
    estimated_fps: float
    elapsed_s: float
    ok: bool
    error: str = ""


# =========================================
# HELPERY
# =========================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def normalize_runtime_device(value: Any, default: str = "cuda") -> str:
    if value is None:
        return default

    if isinstance(value, int):
        return "cuda"

    text = str(value).strip().lower()
    if text.isdigit():
        return "cuda"
    if text.startswith("cuda"):
        return "cuda"
    if text == "gpu":
        return "cuda"
    if text == "cpu":
        return "cpu"

    return default

def compute_fps(avg_total_ms: float) -> float:
    if avg_total_ms <= 0:
        return 0.0
    return 1000.0 / avg_total_ms


def safe_stem(path: Path) -> str:
    parent = path.parent.name.replace(" ", "_")
    stem = path.stem.replace(" ", "_")
    return f"{parent}__{stem}"


def load_manifest_for_root(engines_dir: Path) -> Optional[dict[str, Any]]:
    candidates = [
        engines_dir / "export_manifest.json",
        engines_dir.parent / "export_manifest.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return None


def build_manifest_index(manifest: Optional[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    if not manifest:
        return index

    for row in manifest.get("results", []) or []:
        engine_path = str((row or {}).get("engine_path", "") or "").strip()
        if not engine_path:
            continue
        index[str(Path(engine_path).resolve())] = dict(row)

    return index


def list_engine_files(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Engine directory not found: {root}")
    return sorted(root.rglob("*.engine"))


def parse_imgsz_from_engine_name(path: Path) -> int:
    name = path.stem.lower()

    # szukamy wzorca typu fp16_512_b1_static
    parts = name.split("__")
    for part in parts:
        tokens = part.split("_")
        for token in tokens:
            if token.isdigit():
                value = int(token)
                if 64 <= value <= 4096:
                    return value

    # fallback: szukaj dowolnej liczby w nazwie
    import re
    matches = re.findall(r"(?<!\d)(\d{3,4})(?!\d)", name)
    for m in matches:
        value = int(m)
        if 64 <= value <= 4096:
            return value

    return 640


def parse_batch_from_engine_name(path: Path) -> int:
    import re
    match = re.search(r"_b(\d+)", path.stem.lower())
    if match:
        return max(1, int(match.group(1)))
    return 1


def resolve_engine_runtime_params(engine_path: Path, manifest_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    resolved_path = str(engine_path.resolve())
    row = manifest_index.get(resolved_path)

    if row:
        return {
            "config_name": str(row.get("config_name", engine_path.stem)),
            "image_size": int(row.get("imgsz", 640) or 640),
            "batch": int(row.get("batch", 1) or 1),
            "device": normalize_runtime_device(row.get("device", DEFAULT_DEVICE), str(DEFAULT_DEVICE)),
            "ok_from_manifest": bool(row.get("engine_saved", False)),
        }

    return {
        "config_name": engine_path.stem,
        "image_size": parse_imgsz_from_engine_name(engine_path),
        "batch": parse_batch_from_engine_name(engine_path),
        "device": normalize_runtime_device(DEFAULT_DEVICE, str(DEFAULT_DEVICE)),
        "ok_from_manifest": True,
    }


def build_runner_config_for_engine(
    engine_path: Path,
    output_dir: Path,
    *,
    image_size: int,
    device: str,
) -> RunnerConfig:
    return RunnerConfig(
        input_dir=INPUT_DIR,
        output_dir=output_dir,
        detector_backend="tensorrt",
        model_path=str(REFERENCE_PT),
        trt_engine_path=str(engine_path),
        device=device,
        image_size=image_size,
        confidence=CONFIDENCE,
        debayer_backend=DEBAYER_BACKEND,
        bayer_pattern=BAYER_PATTERN,
        save_overlays=SAVE_OVERLAYS,
        save_json=SAVE_JSON,
    )


def build_runner_config_for_ultralytics(output_dir: Path, *, image_size: int) -> RunnerConfig:
    return RunnerConfig(
        input_dir=INPUT_DIR,
        output_dir=output_dir,
        detector_backend="ultralytics",
        model_path=str(REFERENCE_PT),
        trt_engine_path="",
        device=str(DEFAULT_DEVICE),
        image_size=image_size,
        confidence=CONFIDENCE,
        debayer_backend=DEBAYER_BACKEND,
        bayer_pattern=BAYER_PATTERN,
        save_overlays=SAVE_OVERLAYS,
        save_json=SAVE_JSON,
    )


def run_one(
    config: RunnerConfig,
    *,
    name: str,
    backend: str,
    config_name: str,
    image_size: int,
    batch: int,
    device: str,
    engine_path: str = "",
) -> BenchmarkRow:
    started = time.perf_counter()

    try:
        runner = OfflinePipelineRunner(config)
        summary = runner.run()
        elapsed_s = time.perf_counter() - started

        avg_total_ms = float(summary.get("avg_total_ms", 0.0) or 0.0)

        return BenchmarkRow(
            name=name,
            backend=backend,
            model_path=str(config.model_path),
            engine_path=str(engine_path),
            config_name=str(config_name),
            image_size=int(image_size),
            batch=int(batch),
            device=str(device),
            frames_total=int(summary.get("frames_total", 0) or 0),
            frames_processed=int(summary.get("frames_processed", 0) or 0),
            detections_total=int(summary.get("detections_total", 0) or 0),
            tracks_total=int(summary.get("tracks_total", 0) or 0),
            avg_load_ms=float(summary.get("avg_load_ms", 0.0) or 0.0),
            avg_convert_ms=float(summary.get("avg_convert_ms", 0.0) or 0.0),
            avg_preprocess_ms=float(summary.get("avg_preprocess_ms", 0.0) or 0.0),
            avg_infer_ms=float(summary.get("avg_infer_ms", 0.0) or 0.0),
            avg_track_ms=float(summary.get("avg_track_ms", 0.0) or 0.0),
            avg_total_ms=avg_total_ms,
            estimated_fps=compute_fps(avg_total_ms),
            elapsed_s=elapsed_s,
            ok=True,
            error="",
        )

    except Exception as exc:
        elapsed_s = time.perf_counter() - started
        return BenchmarkRow(
            name=name,
            backend=backend,
            model_path=str(config.model_path),
            engine_path=str(engine_path),
            config_name=str(config_name),
            image_size=int(image_size),
            batch=int(batch),
            device=str(device),
            frames_total=0,
            frames_processed=0,
            detections_total=0,
            tracks_total=0,
            avg_load_ms=0.0,
            avg_convert_ms=0.0,
            avg_preprocess_ms=0.0,
            avg_infer_ms=0.0,
            avg_track_ms=0.0,
            avg_total_ms=0.0,
            estimated_fps=0.0,
            elapsed_s=elapsed_s,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def save_comparison_json(rows: List[BenchmarkRow], out_path: Path) -> None:
    payload = {
        "saved_at_epoch_ns": time.time_ns(),
        "input_dir": str(INPUT_DIR),
        "engines_dir": str(ENGINES_DIR),
        "rows": [asdict(r) for r in rows],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_comparison_csv(rows: List[BenchmarkRow], out_path: Path) -> None:
    fieldnames = [
        "name",
        "backend",
        "model_path",
        "engine_path",
        "config_name",
        "image_size",
        "batch",
        "device",
        "frames_total",
        "frames_processed",
        "detections_total",
        "tracks_total",
        "avg_load_ms",
        "avg_convert_ms",
        "avg_preprocess_ms",
        "avg_infer_ms",
        "avg_track_ms",
        "avg_total_ms",
        "estimated_fps",
        "elapsed_s",
        "ok",
        "error",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def print_table(rows: List[BenchmarkRow]) -> None:
    print("=" * 168)
    print(
        f"{'NAME':<34} {'BACKEND':<12} {'OK':<5} {'IMG':>6} {'B':>4} "
        f"{'INF(ms)':>10} {'TOT(ms)':>10} {'FPS':>10} "
        f"{'LOAD':>8} {'CONV':>8} {'PRE':>8} {'TRACK':>8}"
    )
    print("=" * 168)
    for row in rows:
        print(
            f"{row.name:<34} {row.backend:<12} {str(row.ok):<5} {row.image_size:>6} {row.batch:>4} "
            f"{row.avg_infer_ms:>10.3f} {row.avg_total_ms:>10.3f} {row.estimated_fps:>10.2f} "
            f"{row.avg_load_ms:>8.3f} {row.avg_convert_ms:>8.3f} {row.avg_preprocess_ms:>8.3f} {row.avg_track_ms:>8.3f}"
        )
        if row.error:
            print(f"  ERROR: {row.error}")
    print("=" * 168)


def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input image folder not found: {INPUT_DIR}")
    if not REFERENCE_PT.exists():
        raise FileNotFoundError(f"Reference .pt not found: {REFERENCE_PT}")

    ensure_dir(BENCHMARK_OUTPUT_DIR)
    rows: List[BenchmarkRow] = []

    manifest = load_manifest_for_root(ENGINES_DIR)
    manifest_index = build_manifest_index(manifest)

    engine_files = list_engine_files(ENGINES_DIR)

    if ENGINE_NAME_CONTAINS:
        engine_files = [p for p in engine_files if ENGINE_NAME_CONTAINS.lower() in str(p).lower()]

    if manifest_index and ONLY_OK_FROM_MANIFEST:
        engine_files = [p for p in engine_files if manifest_index.get(str(p.resolve()), {}).get("engine_saved", False)]

    if RUN_ULTRALYTICS_REFERENCE:
        if manifest_index:
            # referencję puść na najczęstszym imgsz z engine albo 640
            image_sizes = []
            for p in engine_files:
                params = resolve_engine_runtime_params(p, manifest_index)
                image_sizes.append(int(params["image_size"]))
            ref_imgsz = image_sizes[0] if image_sizes else 640
        else:
            ref_imgsz = 640

        ref_output_dir = BENCHMARK_OUTPUT_DIR / f"ultralytics_reference_{ref_imgsz}"
        ensure_dir(ref_output_dir)
        print(f"[BENCH] Running ultralytics reference at imgsz={ref_imgsz}...")
        ref_config = build_runner_config_for_ultralytics(ref_output_dir, image_size=ref_imgsz)
        rows.append(
            run_one(
                ref_config,
                name=f"ultralytics_reference_{ref_imgsz}",
                backend="ultralytics",
                config_name=f"ultralytics_reference_{ref_imgsz}",
                image_size=ref_imgsz,
                batch=1,
                device=str(DEFAULT_DEVICE),
                engine_path="",
            )
        )

    if not engine_files:
        print("[BENCH] No TensorRT files found (*.engine).")
    else:
        for engine_path in engine_files:
            params = resolve_engine_runtime_params(engine_path, manifest_index)
            run_name = safe_stem(engine_path)
            run_output_dir = BENCHMARK_OUTPUT_DIR / run_name
            ensure_dir(run_output_dir)

            print(
                f"[BENCH] Running engine: {engine_path.name} | "
                f"imgsz={params['image_size']} batch={params['batch']} device={params['device']}"
            )

            config = build_runner_config_for_engine(
                engine_path,
                run_output_dir,
                image_size=int(params["image_size"]),
                device=str(params["device"]),
            )
            rows.append(
                run_one(
                    config,
                    name=run_name,
                    backend="tensorrt",
                    config_name=str(params["config_name"]),
                    image_size=int(params["image_size"]),
                    batch=int(params["batch"]),
                    device=str(params["device"]),
                    engine_path=str(engine_path),
                )
            )

    rows.sort(
        key=lambda r: (
            not r.ok,
            r.avg_infer_ms if r.ok else float("inf"),
            r.avg_total_ms if r.ok else float("inf"),
        )
    )

    comparison_json = BENCHMARK_OUTPUT_DIR / "comparison.json"
    comparison_csv = BENCHMARK_OUTPUT_DIR / "comparison.csv"

    save_comparison_json(rows, comparison_json)
    save_comparison_csv(rows, comparison_csv)
    print_table(rows)

    print(f"[BENCH] comparison.json -> {comparison_json}")
    print(f"[BENCH] comparison.csv  -> {comparison_csv}")


if __name__ == "__main__":
    main()