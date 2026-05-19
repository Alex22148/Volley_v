"""02_benchmark_color_only.py â€” GPU debayer + resize + normalize only.

WHAT THIS SCRIPT MEASURES:
    Bayer RAW -> RGB float tensor (B, 3, H, W) on CUDA.
    The pipeline is: numpy uint8 Bayer -> CUDA upload -> debayer ->
    normalize -> float16. The output stays on the GPU.
    Raw input is generated directly at target shape, so resize is skipped.

WHAT THIS SCRIPT DOES NOT MEASURE:
    * TensorRT inference.
    * Camera grab.
    * GUI / ring buffer / IPC.
    * Postprocess / NMS.

INPUT:
    --output-shape  H W  (target shape; raw is generated at same shape)
    --batch              batch size (default 4)
    --duration-s         duration after warmup (default 30)
    --warmup             warmup iterations (default 10)
    --bayer-pattern      RG / BG / GR / GB (default RG)
    --backend            auto|native|torch (default auto)

OUTPUT:
    results/color_only_<HxW>.json
    results/color_only_<HxW>.csv

The result records `output_location = cuda` and `gpu_roundtrip = false`
(the tensor never leaves the GPU).
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_ROOT))

from src.cuda_debayer import CudaDebayer, CudaDebayerConfig  # noqa: E402
from src.synthetic_raw import SyntheticRawSource, SyntheticRawConfig  # noqa: E402
from src.metrics import aggregate_run  # noqa: E402

# ----------------------------- static config (no argparse)
OUTPUT_SHAPE = (640, 640)  # (H, W)
BATCH = 4
DURATION_S = 30.0
WARMUP = 10
BAYER_PATTERN = "RG"  # RG/BG/GR/GB
BACKEND = "auto"      # auto/native/torch
RESULTS_DIR = PKG_ROOT / "reports"


def _parse_cli(argv: list[str]) -> dict:
    cfg = {
        "output_shape": [int(OUTPUT_SHAPE[0]), int(OUTPUT_SHAPE[1])],
        "batch": int(BATCH),
        "duration_s": float(DURATION_S),
        "warmup": int(WARMUP),
        "bayer_pattern": str(BAYER_PATTERN),
        "backend": str(BACKEND),
        "results_dir": str(RESULTS_DIR),
    }
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--output-shape":
            cfg["output_shape"] = [int(argv[i + 1]), int(argv[i + 2])]; i += 3; continue
        if a == "--batch":
            cfg["batch"] = int(argv[i + 1]); i += 2; continue
        if a == "--duration-s":
            cfg["duration_s"] = float(argv[i + 1]); i += 2; continue
        if a == "--warmup":
            cfg["warmup"] = int(argv[i + 1]); i += 2; continue
        if a == "--bayer-pattern":
            cfg["bayer_pattern"] = argv[i + 1]; i += 2; continue
        if a == "--backend":
            cfg["backend"] = argv[i + 1]; i += 2; continue
        if a == "--results-dir":
            cfg["results_dir"] = argv[i + 1]; i += 2; continue
        raise SystemExit(f"Unknown argument: {a}")
    return cfg


def main() -> int:
    cfg = _parse_cli(sys.argv[1:])
    out_h, out_w = int(cfg["output_shape"][0]), int(cfg["output_shape"][1])
    cap_h, cap_w = out_h, out_w
    batch = int(cfg["batch"])
    duration_s = float(cfg["duration_s"])
    warmup = int(cfg["warmup"])
    bayer_pattern = str(cfg["bayer_pattern"]).upper()
    backend = str(cfg["backend"]).lower()
    results_dir = Path(cfg["results_dir"])

    print("=" * 72)
    print("fastpath_lab_benchmark / 02_benchmark_color_only.py")
    print("WHAT: measures GPU debayer + normalize on a synthetic Bayer batch.")
    print("WHAT NOT: no TensorRT inference, no camera, no GUI.")
    print(f"INPUT:  capture={cap_h}x{cap_w}  output={out_h}x{out_w}  batch={batch}")
    print(f"        duration={duration_s}s  warmup={warmup}  "
          f"pattern={bayer_pattern}  backend={backend}")
    print("RESIZE: disabled (capture shape == output shape).")
    print("OUTPUT LOCATION: CUDA (no host roundtrip).")
    print("=" * 72)

    import torch

    src = SyntheticRawSource(SyntheticRawConfig(
        capture_hw=(cap_h, cap_w), batch=batch,
        bayer_pattern=bayer_pattern,
    ))

    prefer_native = backend != "torch"
    cfg = CudaDebayerConfig(
        bayer_pattern=bayer_pattern,
        output_hw=(out_h, out_w),
        normalize_01=True,
        half_precision=True,
        prefer_native=prefer_native,
    )
    deb = CudaDebayer(cfg)
    if backend == "native" and deb._native is None:
        raise SystemExit("Native CUDA debayer requested but unavailable. "
                         "Run scripts/00_check_env.py for diagnostics.")
    status = deb.backend_status()
    print(f"backend: native_available={status['native_available']} "
          f"info={status['native_info'].get('detail') or status['native_info'].get('error')}")

    print(f"warmup: {warmup} iterations...")
    warmup_batch = src.next_batch()
    for _ in range(max(0, warmup)):
        deb.convert(warmup_batch)
    torch.cuda.synchronize()

    samples_color_ms: list[float] = []
    samples_packet_ms: list[float] = []
    backend_used_set: set = set()
    start = time.perf_counter()
    end_at = start + duration_s
    it = 0
    print(f"running for {duration_s:.1f}s ...")
    while time.perf_counter() < end_at:
        raw = src.next_batch()
        t_pkt = time.perf_counter()
        result = deb.convert(raw)
        torch.cuda.synchronize()
        pkt_ms = (time.perf_counter() - t_pkt) * 1000.0
        samples_color_ms.append(result.timings_ms["total_color_ms"])
        samples_packet_ms.append(pkt_ms)
        backend_used_set.add(result.backend_used)
        it += 1
        if it % 200 == 0:
            print(f"  it={it} last_color_ms={result.timings_ms['total_color_ms']:.2f} "
                  f"backend={result.backend_used}")

    elapsed = time.perf_counter() - start
    print(f"done. iterations={it} elapsed={elapsed:.2f}s")

    aggregate = aggregate_run(
        packet_samples_ms=samples_packet_ms,
        stage_samples_ms={"color_ms": samples_color_ms},
        batch=batch,
    )

    print("\n--- summary ---")
    cs = aggregate["stage_stats"]["color_ms"]
    print(f"color_ms median = {cs['median_ms']:.2f}")
    print(f"color_ms p95    = {cs['p95_ms']:.2f}")
    print(f"FPS per camera safe p95 = {aggregate['verdict']['fps_per_camera_safe_p95']:.2f} "
          "(color-only, no inference)")

    out_payload = {
        "what": "color_only",
        "backend_used": sorted(backend_used_set),
        "capture_shape": {"h": cap_h, "w": cap_w},
        "inference_input_shape": {"h": out_h, "w": out_w},
        "resize_applied": False,
        "batch": batch,
        "duration_s": duration_s,
        "warmup": warmup,
        "bayer_pattern": bayer_pattern,
        "output_location": "cuda",
        "gpu_roundtrip": False,
        "iterations": it,
        "elapsed_s": elapsed,
        "backend_status": status,
        "aggregate": aggregate,
    }

    label = f"{out_h}x{out_w}_b{batch}"
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / f"color_only_{label}.json"
    json_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    csv_path = results_dir / f"color_only_{label}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "color_ms", "packet_ms"])
        for i, (c, pkt) in enumerate(zip(samples_color_ms, samples_packet_ms)):
            w.writerow([i, f"{c:.4f}", f"{pkt:.4f}"])

    print(f"\n-> wrote {json_path}")
    print(f"-> wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


