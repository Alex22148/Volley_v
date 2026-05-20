"""08_final_benchmark.py - final benchmark without resize.

Założenia:
    * wejściowy obraz ma już docelowy rozmiar inferencji (brak resize),
    * mierzymy czasy etapów: color processing, inference, packet total,
    * robimy sweep po rozdzielczościach i batchach,
    * dla batch > NORMALIZE_TO_BATCH raportujemy też czasy znormalizowane
      do jednej paczki referencyjnej.
"""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

VENV_SCRIPTS = Path(sys.executable).resolve().parent
os.environ["PATH"] = str(VENV_SCRIPTS) + os.pathsep + os.environ.get("PATH", "")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def _ensure_msvc_build_env() -> None:
    if os.name != "nt" or shutil.which("cl"):
        return

    install_root = ""
    program_files_x86 = os.environ.get("ProgramFiles(x86)")
    if program_files_x86:
        vswhere = Path(program_files_x86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
        if vswhere.exists():
            try:
                install_root = subprocess.check_output(
                    [
                        str(vswhere),
                        "-latest",
                        "-products",
                        "*",
                        "-requires",
                        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                        "-property",
                        "installationPath",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            except Exception:
                install_root = ""

    vcvars64 = Path(install_root) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat" if install_root else None
    if vcvars64 is None or not vcvars64.exists():
        program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        candidates = sorted(
            Path(program_files).glob(
                "Microsoft Visual Studio/2022/*/VC/Auxiliary/Build/vcvars64.bat"
            ),
            reverse=True,
        )
        vcvars64 = candidates[0] if candidates else None

    if vcvars64 is None or not vcvars64.exists():
        return

    if not vcvars64.exists():
        return

    try:
        env_dump = subprocess.check_output(
            f'"{vcvars64}" >nul && set',
            shell=True,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return

    for line in env_dump.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key] = value


_ensure_msvc_build_env()

PKG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_ROOT))

from src.cuda_debayer import CudaDebayer, CudaDebayerConfig  # noqa: E402
from src.synthetic_raw import SyntheticRawSource, SyntheticRawConfig  # noqa: E402
from src.trt_runner import TrtRunner  # noqa: E402
from src.metrics import aggregate_run  # noqa: E402


# ----------------------------- static config (no argparse)
ENGINES_DIR = PKG_ROOT.parent / "engines" / "static"
if not ENGINES_DIR.exists():
    ENGINES_DIR = PKG_ROOT / "engines" / "static"
REPORTS_ROOT = PKG_ROOT / "reports"
BAYER_PATTERN = "RG"
DURATION_S = 15.0
WARMUP = 10
CREATE_REPORT = False
CHECK_ENV_BEFORE_RUN = False
# Normalizacja wyników: batch=4 oznacza "jedna paczka referencyjna" (4 kamery).
NORMALIZE_TO_BATCH = 4

INPUT_MATRIX: dict[str, list[list[int]]] = {
    "v640": [[640, 640], [4, 8, 12,16]],
    "v960": [[960, 960], [4, 8, 12,16]],
    "v1088x1920": [[1088, 1920], [4, 8, 12,16]],
}


def _resolve_test_results_dir(reports_root: Path) -> Path:
    candidates = sorted(
        [p for p in reports_root.glob("test_???") if p.is_dir() and p.name[5:].isdigit()],
        key=lambda p: int(p.name[5:]),
    )
    if candidates:
        return candidates[-1]
    return reports_root / "test_001"


def _engine_candidates(label: str, batch: int) -> list[str]:
    return [
        f"best__fp16_{label}_b{batch}.engine",
        f"best__fp32_{label}_b{batch}.engine",
        f"best__{label}_b{batch}.engine",
    ]


def _resolve_engine_path(engines_dir: Path, label: str, batch: int) -> Path | None:
    for name in _engine_candidates(label, batch):
        p = engines_dir / name
        if p.exists():
            return p
    return None


def _normalize_ms(ms_value: float, batch: int, normalize_to_batch: int) -> float:
    if batch <= 0 or normalize_to_batch <= 0:
        return ms_value
    return ms_value * (normalize_to_batch / float(batch))


def _stats_with_normalization(stage: dict[str, Any], batch: int) -> dict[str, float]:
    median_ms = float(stage.get("median_ms", float("nan")))
    p95_ms = float(stage.get("p95_ms", float("nan")))
    return {
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "median_ms_norm": _normalize_ms(median_ms, batch, NORMALIZE_TO_BATCH),
        "p95_ms_norm": _normalize_ms(p95_ms, batch, NORMALIZE_TO_BATCH),
    }


def _r3(v: float) -> float:
    return round(float(v), 3)

def _progress_bar(
    current: float,
    total: float,
    *,
    width: int = 32,
    prefix: str = "",
    suffix: str = "",
) -> str:
    if total <= 0:
        frac = 1.0
    else:
        frac = max(0.0, min(1.0, float(current) / float(total)))

    filled = int(round(width * frac))
    bar = "█" * filled + "░" * (width - filled)
    percent = frac * 100.0

    return f"{prefix}[{bar}] {percent:6.2f}% {suffix}"

def postprocess_nms_dummy(outputs: Any) -> tuple[int, str]:
    """
    Placeholder pod realny postprocess/NMS.

    Na razie nie wykonuje dekodowania bboxów ani NMS, ale utrzymuje pełną
    strukturę metryk benchmarku:
        - postprocess_nms_ms
        - packet_without_nms_ms
        - packet_with_nms_ms
        - nms_overhead_ms
        - nms_detections_count

    Później tę funkcję można podmienić na realny decode + NMS.
    """
    return 0, "disabled_dummy"


def _zero_stats() -> dict[str, float]:
    return {
        "median_ms": 0.0,
        "p95_ms": 0.0,
        "median_ms_norm": 0.0,
        "p95_ms_norm": 0.0,
    }


def _detection_count_stats(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {
            "median": 0.0,
            "mean": 0.0,
            "max": 0,
        }

    arr = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "max": int(np.max(arr)),
    }

def run_one(
            engine_path: Path,
            hw: tuple[int, int],
            batch: int,
            *,
            progress_label: str = "",
    ) -> dict[str, Any]:

    in_h, in_w = int(hw[0]), int(hw[1])
    cap_h, cap_w = in_h, in_w  # brak resize: capture = inference

    import torch

    runner = TrtRunner(str(engine_path))
    info = runner.info()
    eng_b, _eng_c, eng_h, eng_w = info.input_shape
    if eng_b != batch:
        raise RuntimeError(f"engine batch={eng_b} != expected batch={batch}")
    if (eng_h, eng_w) != (in_h, in_w):
        raise RuntimeError(f"engine input {eng_h}x{eng_w} != expected {in_h}x{in_w}")

    src = SyntheticRawSource(SyntheticRawConfig(
        capture_hw=(cap_h, cap_w),
        batch=batch,
        bayer_pattern=BAYER_PATTERN,
    ))
    deb = CudaDebayer(CudaDebayerConfig(
        bayer_pattern=BAYER_PATTERN,
        output_hw=(in_h, in_w),  # brak resize (input already correct size)
        normalize_01=True,
        half_precision=info.is_fp16,
        prefer_native=True,
    ))

    # warmup
    # warmup
    warmup_n = max(0, int(WARMUP))
    for i in range(warmup_n):
        raw_batch = src.next_batch()
        color = deb.convert(raw_batch)
        runner.infer(color.tensor)

        msg = _progress_bar(
            i + 1,
            warmup_n,
            prefix=f"\r[WARMUP] {progress_label} ",
            suffix=f"{i + 1}/{warmup_n}",
        )
        print(msg, end="", flush=True)

    if warmup_n:
        print()

    torch.cuda.synchronize()

    samples_batch_build_ms: list[float] = []
    samples_color_debayer_ms: list[float] = []
    samples_color_ms: list[float] = []
    samples_inference_ms: list[float] = []

    # pełny zestaw metryk czasowych
    samples_color_debayer_plus_inference_ms: list[float] = []
    samples_color_batch_inference_ms: list[float] = []
    samples_color_plus_inference_ms: list[float] = []
    samples_postprocess_nms_ms: list[float] = []
    samples_packet_without_nms_ms: list[float] = []
    samples_packet_with_nms_ms: list[float] = []
    samples_nms_overhead_ms: list[float] = []
    samples_nms_detections_count: list[int] = []

    backend_used_set: set[str] = set()
    nms_mode_used_set: set[str] = set()

    start = time.perf_counter()
    end_at = start + float(DURATION_S)
    iterations = 0

    while time.perf_counter() < end_at:
        t_packet_start = time.perf_counter()

        # 0) batch build / capture packet assembly.
        # In this lab this is SyntheticRawSource.next_batch(), so it is a CPU-side
        # proxy for preparing one raw Bayer batch before the GPU color path.
        t_batch_build = time.perf_counter()
        raw_batch = src.next_batch()
        batch_build_ms = (time.perf_counter() - t_batch_build) * 1000.0

        # 1) color / debayer / normalize: RAW Bayer -> RGB BCHW tensor on CUDA.
        color = deb.convert(raw_batch)
        color_debayer_ms = float(color.timings_ms["total_color_ms"])

        # 2) TensorRT inference
        outputs, inf_ms = runner.timed_infer(color.tensor)

        # ważne: bez synchronizacji mierzymy tylko enqueue, nie realny koniec GPU
        torch.cuda.synchronize()
        packet_without_nms_ms = (time.perf_counter() - t_packet_start) * 1000.0

        color_debayer_plus_inference_ms = color_debayer_ms + float(inf_ms)
        color_batch_inference_ms = batch_build_ms + color_debayer_plus_inference_ms

        # kompatybilnie: stara nazwa zostaje, ale nowe CSV ma jawne kolumny:
        # batch_build_ms, color_debayer_ms, inference_ms, color_batch_inference_ms.
        color_plus_inference_ms = color_debayer_plus_inference_ms

        # 3) postprocess / NMS
        t_nms_start = time.perf_counter()

        detections_count, nms_mode = postprocess_nms_dummy(outputs)

        # jeżeli później NMS będzie GPU, ta synchronizacja nadal jest poprawna
        torch.cuda.synchronize()
        postprocess_nms_ms = (time.perf_counter() - t_nms_start) * 1000.0

        # 4) pełny packet z NMS
        packet_with_nms_ms = (time.perf_counter() - t_packet_start) * 1000.0
        nms_overhead_ms = packet_with_nms_ms - packet_without_nms_ms

        samples_batch_build_ms.append(float(batch_build_ms))
        samples_color_debayer_ms.append(float(color_debayer_ms))
        samples_color_ms.append(float(color_debayer_ms))
        samples_inference_ms.append(float(inf_ms))
        samples_color_debayer_plus_inference_ms.append(float(color_debayer_plus_inference_ms))
        samples_color_batch_inference_ms.append(float(color_batch_inference_ms))
        samples_color_plus_inference_ms.append(float(color_plus_inference_ms))
        samples_postprocess_nms_ms.append(float(postprocess_nms_ms))
        samples_packet_without_nms_ms.append(float(packet_without_nms_ms))
        samples_packet_with_nms_ms.append(float(packet_with_nms_ms))
        samples_nms_overhead_ms.append(float(nms_overhead_ms))
        samples_nms_detections_count.append(int(detections_count))

        backend_used_set.add(color.backend_used)
        iterations += 1

        now = time.perf_counter()
        elapsed_now = now - start

        if iterations == 1 or iterations % 10 == 0 or now >= end_at:
            msg = _progress_bar(
                elapsed_now,
                DURATION_S,
                prefix=f"\r[MEASURE] {progress_label} ",
                suffix=f"{elapsed_now:5.1f}/{DURATION_S:.1f}s | iter={iterations}",
            )
            print(msg, end="", flush=True)

    print()

    elapsed_s = time.perf_counter() - start
    agg = aggregate_run(
        # jako główny packet raportujemy pełną ścieżkę z NMS,
        # nawet jeśli NMS jest teraz dummy
        packet_samples_ms=samples_packet_with_nms_ms,
        stage_samples_ms={
            "batch_build_ms": samples_batch_build_ms,
            "color_debayer_ms": samples_color_debayer_ms,
            "color_ms": samples_color_ms,
            "inference_ms": samples_inference_ms,
            "color_debayer_plus_inference_ms": samples_color_debayer_plus_inference_ms,
            "color_batch_inference_ms": samples_color_batch_inference_ms,
            "color_plus_inference_ms": samples_color_plus_inference_ms,
            "postprocess_nms_ms": samples_postprocess_nms_ms,
            "packet_without_nms_ms": samples_packet_without_nms_ms,
            "packet_with_nms_ms": samples_packet_with_nms_ms,
            "nms_overhead_ms": samples_nms_overhead_ms,
        },
        batch=batch,
    )

    batch_build_stats = _stats_with_normalization(agg["stage_stats"]["batch_build_ms"], batch)
    color_debayer_stats = _stats_with_normalization(agg["stage_stats"]["color_debayer_ms"], batch)
    color_stats = _stats_with_normalization(agg["stage_stats"]["color_ms"], batch)
    inf_stats = _stats_with_normalization(agg["stage_stats"]["inference_ms"], batch)
    color_debayer_plus_inf_stats = _stats_with_normalization(
        agg["stage_stats"]["color_debayer_plus_inference_ms"],
        batch,
    )
    color_batch_inf_stats = _stats_with_normalization(
        agg["stage_stats"]["color_batch_inference_ms"],
        batch,
    )
    color_plus_inf_stats = _stats_with_normalization(
        agg["stage_stats"]["color_plus_inference_ms"],
        batch,
    )
    post_nms_stats = _stats_with_normalization(
        agg["stage_stats"]["postprocess_nms_ms"],
        batch,
    )
    pkt_without_nms_stats = _stats_with_normalization(
        agg["stage_stats"]["packet_without_nms_ms"],
        batch,
    )
    pkt_with_nms_stats = _stats_with_normalization(
        agg["stage_stats"]["packet_with_nms_ms"],
        batch,
    )
    nms_overhead_stats = _stats_with_normalization(
        agg["stage_stats"]["nms_overhead_ms"],
        batch,
    )

    detections_stats = _detection_count_stats(samples_nms_detections_count)

    return {
        "engine": str(engine_path),
        "input_shape_hw": [in_h, in_w],
        "capture_shape_hw": [cap_h, cap_w],
        "batch": batch,
        "duration_s": DURATION_S,
        "warmup": WARMUP,
        "normalize_to_batch": NORMALIZE_TO_BATCH,
        "iterations": iterations,
        "elapsed_s": elapsed_s,
        "backends_used": sorted(backend_used_set),
        "nms_modes_used": sorted(nms_mode_used_set),
        "nms_mode": ",".join(sorted(nms_mode_used_set)) if nms_mode_used_set else "unknown",
        "nms_detections_count": detections_stats,

        # packet_ms zostaje jako kompatybilność wsteczna,
        # ale nowe raporty powinny używać packet_without_nms_ms / packet_with_nms_ms
        "packet_ms_legacy_alias": "packet_without_nms_ms",

        "aggregate": agg,
        "normalized": {
            "batch_build_ms": batch_build_stats,
            "color_debayer_ms": color_debayer_stats,
            "color_ms": color_stats,
            "inference_ms": inf_stats,
            "color_debayer_plus_inference_ms": color_debayer_plus_inf_stats,
            "color_batch_inference_ms": color_batch_inf_stats,
            "color_plus_inference_ms": color_plus_inf_stats,
            "postprocess_nms_ms": post_nms_stats,
            "packet_without_nms_ms": pkt_without_nms_stats,
            "packet_with_nms_ms": pkt_with_nms_stats,
            "nms_overhead_ms": nms_overhead_stats,

            # alias kompatybilności dla starego raportu
            "packet_ms": pkt_without_nms_stats,
        },
    }


def main() -> int:
    if CHECK_ENV_BEFORE_RUN:
        check_env_script = Path(__file__).resolve().parent / "00_check_env.py"
        if not check_env_script.exists():
            print(f"[ENV] ERROR: nie znaleziono skryptu check_env: {check_env_script}")
            return 2
        print("[ENV] CHECK_ENV_BEFORE_RUN=True -> uruchamiam 00_check_env.py ...")
        env_proc = subprocess.run([sys.executable, str(check_env_script)])
        if env_proc.returncode != 0:
            print(f"[ENV] ERROR: środowisko niezgodne (exit code={env_proc.returncode}). Benchmark przerwany.")
            return env_proc.returncode
        print("[ENV] OK: środowisko zgodne, kontynuuję benchmark.")

    results_dir = _resolve_test_results_dir(REPORTS_ROOT)
    print("=" * 88)
    print("fastpath_lab_benchmark / 08_final_benchmark.py")
    print("WHAT: color + inference + packet benchmark without resize.")
    print(f"INPUT: engines_dir={ENGINES_DIR} duration={DURATION_S}s warmup={WARMUP}")
    print(f"OUT  : results_dir={results_dir}")
    print(f"NORM : batch metrics normalized to batch={NORMALIZE_TO_BATCH}")
    print("=" * 88)
    total_cases = sum(len(matrix[1]) for matrix in INPUT_MATRIX.values())
    case_idx = 0
    rows: list[dict[str, Any]] = []

    for label, matrix in INPUT_MATRIX.items():
        hw, batches = matrix
        h, w = int(hw[0]), int(hw[1])
        for batch in [int(b) for b in batches]:
            case_idx += 1
            case_prefix = f"[CASE {case_idx}/{total_cases}] {label} {h}x{w} b{batch}"
            engine_path = _resolve_engine_path(ENGINES_DIR, label, batch)
            if engine_path is None:
                rows.append({
                    "label": label,
                    "shape_hw": [h, w],
                    "batch": batch,
                    "skipped": True,
                    "reason": "engine_missing",
                    "engine_candidates": _engine_candidates(label, batch),
                })
                print(f"{case_prefix} | SKIP: engine missing")
                continue

            print()
            print("=" * 88)
            print(f"{case_prefix}")
            print(f"ENGINE: {engine_path.name}")
            print("=" * 88)
            try:
                result = run_one(
                    engine_path,
                    (h, w),
                    batch,
                    progress_label=f"{label} b{batch}",
                )
                result["label"] = label
                result["skipped"] = False
                rows.append(result)
            except Exception as exc:
                rows.append({
                    "label": label,
                    "shape_hw": [h, w],
                    "batch": batch,
                    "skipped": True,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "engine": str(engine_path),
                })
                print(f"{case_prefix} | FAIL: {type(exc).__name__}: {exc}")

    out = {
        "what": "final_benchmark_no_resize",
        "generated_at_epoch_ns": time.time_ns(),
        "engines_dir": str(ENGINES_DIR),
        "duration_s": DURATION_S,
        "warmup": WARMUP,
        "normalize_to_batch": NORMALIZE_TO_BATCH,
        "input_matrix": INPUT_MATRIX,
        "rows": rows,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "final_benchmark_08.json"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    csv_path = results_dir / "final_benchmark_08.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "label", "shape_h", "shape_w", "batch", "engine", "skipped", "reason",

            "batch_build_ms_median", "batch_build_ms_p95",
            "color_debayer_ms_median", "color_debayer_ms_p95",
            "color_ms_median", "color_ms_p95",
            "inference_ms_median", "inference_ms_p95",
            "color_debayer_plus_inference_ms_median", "color_debayer_plus_inference_ms_p95",
            "color_batch_inference_ms_median", "color_batch_inference_ms_p95",
            "color_plus_inference_ms_median", "color_plus_inference_ms_p95",
            "postprocess_nms_ms_median", "postprocess_nms_ms_p95",
            "packet_without_nms_ms_median", "packet_without_nms_ms_p95",
            "packet_with_nms_ms_median", "packet_with_nms_ms_p95",
            "nms_overhead_ms_median", "nms_overhead_ms_p95",

            "batch_build_ms_median_norm_b4", "batch_build_ms_p95_norm_b4",
            "color_debayer_ms_median_norm_b4", "color_debayer_ms_p95_norm_b4",
            "color_ms_median_norm_b4", "color_ms_p95_norm_b4",
            "inference_ms_median_norm_b4", "inference_ms_p95_norm_b4",
            "color_debayer_plus_inference_ms_median_norm_b4", "color_debayer_plus_inference_ms_p95_norm_b4",
            "color_batch_inference_ms_median_norm_b4", "color_batch_inference_ms_p95_norm_b4",
            "color_plus_inference_ms_median_norm_b4", "color_plus_inference_ms_p95_norm_b4",
            "postprocess_nms_ms_median_norm_b4", "postprocess_nms_ms_p95_norm_b4",
            "packet_without_nms_ms_median_norm_b4", "packet_without_nms_ms_p95_norm_b4",
            "packet_with_nms_ms_median_norm_b4", "packet_with_nms_ms_p95_norm_b4",
            "nms_overhead_ms_median_norm_b4", "nms_overhead_ms_p95_norm_b4",

            "fps_without_nms_p95_norm_b4",
            "fps_with_nms_p95_norm_b4",

            "nms_mode",
            "nms_detections_count_median",
            "nms_detections_count_max",

            "fps_per_camera_median",
            "fps_per_camera_safe_p95",
        ])
        for r in rows:
            if r.get("skipped"):
                w.writerow([
                    r.get("label"),
                    (r.get("shape_hw") or [None, None])[0],
                    (r.get("shape_hw") or [None, None])[1],
                    r.get("batch"),
                    r.get("engine", ""),
                    True,
                    r.get("reason", ""),
                    *[""] * 51,
                ])
                continue

            agg = r["aggregate"]
            stage = agg["stage_stats"]
            verdict = agg["verdict"]
            norm = r["normalized"]
            in_h, in_w = r["input_shape_hw"]

            packet_without_nms_norm_p95 = float(norm["packet_without_nms_ms"]["p95_ms_norm"])
            packet_with_nms_norm_p95 = float(norm["packet_with_nms_ms"]["p95_ms_norm"])

            fps_without_nms_p95_norm_b4 = (
                1000.0 / packet_without_nms_norm_p95
                if packet_without_nms_norm_p95 > 0
                else float("nan")
            )
            fps_with_nms_p95_norm_b4 = (
                1000.0 / packet_with_nms_norm_p95
                if packet_with_nms_norm_p95 > 0
                else float("nan")
            )

            det_stats = r.get("nms_detections_count", {})

            w.writerow([
                r.get("label"),
                in_h,
                in_w,
                r["batch"],
                r["engine"],
                False,
                "",

                stage["batch_build_ms"]["median_ms"], stage["batch_build_ms"]["p95_ms"],
                stage["color_debayer_ms"]["median_ms"], stage["color_debayer_ms"]["p95_ms"],
                stage["color_ms"]["median_ms"], stage["color_ms"]["p95_ms"],
                stage["inference_ms"]["median_ms"], stage["inference_ms"]["p95_ms"],
                stage["color_debayer_plus_inference_ms"]["median_ms"], stage["color_debayer_plus_inference_ms"]["p95_ms"],
                stage["color_batch_inference_ms"]["median_ms"], stage["color_batch_inference_ms"]["p95_ms"],
                stage["color_plus_inference_ms"]["median_ms"], stage["color_plus_inference_ms"]["p95_ms"],
                stage["postprocess_nms_ms"]["median_ms"], stage["postprocess_nms_ms"]["p95_ms"],
                stage["packet_without_nms_ms"]["median_ms"], stage["packet_without_nms_ms"]["p95_ms"],
                stage["packet_with_nms_ms"]["median_ms"], stage["packet_with_nms_ms"]["p95_ms"],
                stage["nms_overhead_ms"]["median_ms"], stage["nms_overhead_ms"]["p95_ms"],

                norm["batch_build_ms"]["median_ms_norm"], norm["batch_build_ms"]["p95_ms_norm"],
                norm["color_debayer_ms"]["median_ms_norm"], norm["color_debayer_ms"]["p95_ms_norm"],
                norm["color_ms"]["median_ms_norm"], norm["color_ms"]["p95_ms_norm"],
                norm["inference_ms"]["median_ms_norm"], norm["inference_ms"]["p95_ms_norm"],
                norm["color_debayer_plus_inference_ms"]["median_ms_norm"], norm["color_debayer_plus_inference_ms"]["p95_ms_norm"],
                norm["color_batch_inference_ms"]["median_ms_norm"], norm["color_batch_inference_ms"]["p95_ms_norm"],
                norm["color_plus_inference_ms"]["median_ms_norm"], norm["color_plus_inference_ms"]["p95_ms_norm"],
                norm["postprocess_nms_ms"]["median_ms_norm"], norm["postprocess_nms_ms"]["p95_ms_norm"],
                norm["packet_without_nms_ms"]["median_ms_norm"], norm["packet_without_nms_ms"]["p95_ms_norm"],
                norm["packet_with_nms_ms"]["median_ms_norm"], norm["packet_with_nms_ms"]["p95_ms_norm"],
                norm["nms_overhead_ms"]["median_ms_norm"], norm["nms_overhead_ms"]["p95_ms_norm"],

                fps_without_nms_p95_norm_b4,
                fps_with_nms_p95_norm_b4,

                r.get("nms_mode", ""),
                det_stats.get("median", 0),
                det_stats.get("max", 0),

                verdict["fps_per_camera_median"],
                verdict["fps_per_camera_safe_p95"],
            ])

    # CSV stricte pod porównanie batchy:
    # - czasy na cały batch,
    # - czasy przeliczone na "pojedynczy pakiet b4" (b4=1, b8=2, b12=3).
    compare_csv = results_dir / "final_benchmark_08_compare.csv"
    with compare_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "resolution",
            "batch",
            "packets_equivalent_vs_b4",

            "batch_build_batch_ms_p95",
            "color_debayer_batch_ms_p95",
            "color_batch_ms_p95",
            "inference_batch_ms_p95",
            "color_debayer_plus_inference_batch_ms_p95",
            "color_batch_inference_batch_ms_p95",
            "color_plus_inference_batch_ms_p95",
            "postprocess_nms_batch_ms_p95",
            "packet_without_nms_batch_ms_p95",
            "packet_with_nms_batch_ms_p95",
            "nms_overhead_batch_ms_p95",

            "batch_build_per_packet_b4_ms_p95",
            "color_debayer_per_packet_b4_ms_p95",
            "color_per_packet_b4_ms_p95",
            "inference_per_packet_b4_ms_p95",
            "color_debayer_plus_inference_per_packet_b4_ms_p95",
            "color_batch_inference_per_packet_b4_ms_p95",
            "color_plus_inference_per_packet_b4_ms_p95",
            "postprocess_nms_per_packet_b4_ms_p95",
            "packet_without_nms_per_packet_b4_ms_p95",
            "packet_with_nms_per_packet_b4_ms_p95",
            "nms_overhead_per_packet_b4_ms_p95",

            "fps_without_nms_p95_norm_b4",
            "fps_with_nms_p95_norm_b4",

            "nms_mode",
        ])
        for r in sorted(
            [x for x in rows if not x.get("skipped")],
            key=lambda x: (int(x["input_shape_hw"][0]) * int(x["input_shape_hw"][1]), int(x["batch"])),
        ):
            batch = int(r["batch"])
            packets_equiv = float(batch) / float(NORMALIZE_TO_BATCH)
            shape_label = f"{int(r['input_shape_hw'][0])}x{int(r['input_shape_hw'][1])}"
            st = r["aggregate"]["stage_stats"]
            vd = r["aggregate"]["verdict"]
            nm = r["normalized"]
            st = r["aggregate"]["stage_stats"]
            nm = r["normalized"]

            without_norm = float(nm["packet_without_nms_ms"]["p95_ms_norm"])
            with_norm = float(nm["packet_with_nms_ms"]["p95_ms_norm"])

            fps_without = 1000.0 / without_norm if without_norm > 0 else float("nan")
            fps_with = 1000.0 / with_norm if with_norm > 0 else float("nan")

            w.writerow([
                shape_label,
                batch,
                _r3(packets_equiv),

                _r3(float(st["batch_build_ms"]["p95_ms"])),
                _r3(float(st["color_debayer_ms"]["p95_ms"])),
                _r3(float(st["color_ms"]["p95_ms"])),
                _r3(float(st["inference_ms"]["p95_ms"])),
                _r3(float(st["color_debayer_plus_inference_ms"]["p95_ms"])),
                _r3(float(st["color_batch_inference_ms"]["p95_ms"])),
                _r3(float(st["color_plus_inference_ms"]["p95_ms"])),
                _r3(float(st["postprocess_nms_ms"]["p95_ms"])),
                _r3(float(st["packet_without_nms_ms"]["p95_ms"])),
                _r3(float(st["packet_with_nms_ms"]["p95_ms"])),
                _r3(float(st["nms_overhead_ms"]["p95_ms"])),

                _r3(float(nm["batch_build_ms"]["p95_ms_norm"])),
                _r3(float(nm["color_debayer_ms"]["p95_ms_norm"])),
                _r3(float(nm["color_ms"]["p95_ms_norm"])),
                _r3(float(nm["inference_ms"]["p95_ms_norm"])),
                _r3(float(nm["color_debayer_plus_inference_ms"]["p95_ms_norm"])),
                _r3(float(nm["color_batch_inference_ms"]["p95_ms_norm"])),
                _r3(float(nm["color_plus_inference_ms"]["p95_ms_norm"])),
                _r3(float(nm["postprocess_nms_ms"]["p95_ms_norm"])),
                _r3(float(nm["packet_without_nms_ms"]["p95_ms_norm"])),
                _r3(float(nm["packet_with_nms_ms"]["p95_ms_norm"])),
                _r3(float(nm["nms_overhead_ms"]["p95_ms_norm"])),

                _r3(fps_without),
                _r3(fps_with),

                r.get("nms_mode", ""),
            ])

    print()
    print("=" * 88)
    print("SUMMARY")
    for r in rows:
        if r.get("skipped"):
            print(f"  - {r.get('label')} b{r.get('batch')}: SKIP ({r.get('reason')})")
            continue
        norm_without = r["normalized"]["packet_without_nms_ms"]["p95_ms_norm"]
        norm_with = r["normalized"]["packet_with_nms_ms"]["p95_ms_norm"]
        norm_nms = r["normalized"]["nms_overhead_ms"]["p95_ms_norm"]
        norm_batch_color_infer = r["normalized"]["color_batch_inference_ms"]["p95_ms_norm"]

        print(
            f"  - {r.get('label')} b{r.get('batch')}: "
            f"batch+color+infer_p95_norm_b4={norm_batch_color_infer:.2f} ms | "
            f"without_nms_p95_norm_b4={norm_without:.2f} ms | "
            f"with_nms_p95_norm_b4={norm_with:.2f} ms | "
            f"nms_overhead_norm_b4={norm_nms:.2f} ms | "
            f"nms_mode={r.get('nms_mode', '')}"

        )

    print(f"\n-> wrote {json_path}")
    print(f"-> wrote {csv_path}")
    print(f"-> wrote {compare_csv}")

    if CREATE_REPORT:
        report_script = Path(__file__).resolve().parent / "09_make_final_report.py"
        if report_script.exists():
            print("\n[REPORT] CREATE_REPORT=True -> uruchamiam 09_make_final_report.py ...")
            proc = subprocess.run([sys.executable, str(report_script)])
            if proc.returncode != 0:
                print(f"[REPORT] ERROR: 09_make_final_report.py zakończył się kodem {proc.returncode}")
            else:
                print("[REPORT] OK: raport wygenerowany po wynikach.")
        else:
            print(f"[REPORT] WARN: nie znaleziono skryptu raportowego: {report_script}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
