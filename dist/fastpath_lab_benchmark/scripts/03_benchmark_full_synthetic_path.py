"""03_benchmark_full_synthetic_path.py â€” synthetic RAW -> color -> TRT.

WHAT THIS SCRIPT MEASURES:
    The whole synthetic GPU fast path, end to end:
        synthetic Bayer batch (numpy uint8)
            -> CUDA upload
            -> native CUDA debayer (or torch fallback)
            -> bilinear resize + normalize
            -> TensorRT engine
            -> trivial postprocess (just reading output shapes).
    Output tensor stays on the GPU and is fed zero-copy into the engine.

WHAT THIS SCRIPT DOES NOT MEASURE:
    * Basler camera grab.
    * GUI, ring buffer, preview worker.
    * Hardware sync, shared memory, or any IPC.
    * Realistic NMS / detection scoring loop.

INPUT:
    --engine        path to .engine
    --capture-shape H W (raw Bayer size, default 1080 1920)
    --input-shape   H W (engine input size; must match engine)
    --batch         batch size (must match engine)
    --duration-s    duration after warmup
    --warmup        warmup iterations
    --bayer-pattern RG/BG/GR/GB

OUTPUT:
    results/full_synthetic_<HxW>_b<BATCH>.json
    results/full_synthetic_<HxW>_b<BATCH>.csv
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
from src.trt_runner import TrtRunner  # noqa: E402
from src.metrics import aggregate_run  # noqa: E402

# ----------------------------- static config (no argparse)
ENGINE = PKG_ROOT / "engines" / "best__fp16_640x640_b4.engine"
CAPTURE_SHAPE = (1080, 1920)  # (H, W)
INPUT_SHAPE = (640, 640)      # (H, W)
BATCH = 4
DURATION_S = 30.0
WARMUP = 10
BAYER_PATTERN = "RG"
RESULTS_DIR = PKG_ROOT / "reports"


def _parse_cli(argv: list[str]) -> dict:
    cfg = {
        "engine": str(ENGINE),
        "capture_shape": [int(CAPTURE_SHAPE[0]), int(CAPTURE_SHAPE[1])],
        "input_shape": [int(INPUT_SHAPE[0]), int(INPUT_SHAPE[1])],
        "batch": int(BATCH),
        "duration_s": float(DURATION_S),
        "warmup": int(WARMUP),
        "bayer_pattern": str(BAYER_PATTERN),
        "results_dir": str(RESULTS_DIR),
    }
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--engine":
            cfg["engine"] = argv[i + 1]; i += 2; continue
        if arg == "--capture-shape":
            cfg["capture_shape"] = [int(argv[i + 1]), int(argv[i + 2])]; i += 3; continue
        if arg == "--input-shape":
            cfg["input_shape"] = [int(argv[i + 1]), int(argv[i + 2])]; i += 3; continue
        if arg == "--batch":
            cfg["batch"] = int(argv[i + 1]); i += 2; continue
        if arg == "--duration-s":
            cfg["duration_s"] = float(argv[i + 1]); i += 2; continue
        if arg == "--warmup":
            cfg["warmup"] = int(argv[i + 1]); i += 2; continue
        if arg == "--bayer-pattern":
            cfg["bayer_pattern"] = argv[i + 1]; i += 2; continue
        if arg == "--results-dir":
            cfg["results_dir"] = argv[i + 1]; i += 2; continue
        raise SystemExit(f"Unknown argument: {arg}")
    return cfg


def main() -> int:
    args = _parse_cli(sys.argv[1:])
    cap_h, cap_w = args["capture_shape"]
    in_h, in_w = args["input_shape"]
    batch = int(args["batch"])
    duration_s = float(args["duration_s"])
    warmup = int(args["warmup"])
    bayer_pattern = str(args["bayer_pattern"]).upper()
    results_dir = Path(args["results_dir"])
    engine = str(args["engine"])

    print("=" * 72)
    print("fastpath_lab_benchmark / 03_benchmark_full_synthetic_path.py")
    print("WHAT: synthetic Bayer -> GPU debayer -> resize -> TRT, end to end.")
    print("WHAT NOT: no Basler grab, no GUI, no ring buffer, no IPC.")
    print(f"INPUT:  engine={engine}")
    print(f"        capture={cap_h}x{cap_w}  inference={in_h}x{in_w}  batch={batch}")
    print(f"        duration={duration_s}s  warmup={warmup}  "
          f"pattern={bayer_pattern}")
    print("FPS:    fps_per_camera = 1000 / packet_ms (batch=4 = 1 packet).")
    print("=" * 72)

    import torch

    runner = TrtRunner(engine)
    info = runner.info()
    eng_b, _eng_c, eng_h, eng_w = info.input_shape
    if eng_b != batch:
        raise SystemExit(f"engine batch={eng_b} != batch {batch}")
    if (eng_h, eng_w) != (in_h, in_w):
        raise SystemExit(
            f"engine input {eng_h}x{eng_w} != --input-shape {in_h}x{in_w}")
    print(f"engine: input={info.input_shape} dtype={info.input_dtype}")

    src = SyntheticRawSource(SyntheticRawConfig(
        capture_hw=(cap_h, cap_w), batch=batch,
        bayer_pattern=bayer_pattern,
    ))
    deb = CudaDebayer(CudaDebayerConfig(
        bayer_pattern=bayer_pattern,
        output_hw=(in_h, in_w),
        normalize_01=True,
        half_precision=info.is_fp16,
        prefer_native=True,
    ))
    backend_status = deb.backend_status()
    print(f"debayer backend: native_available={backend_status['native_available']}")

    print(f"warmup: {warmup} iterations...")
    warm_raw = src.next_batch()
    for _ in range(max(0, warmup)):
        color = deb.convert(warm_raw)
        runner.infer(color.tensor)
    torch.cuda.synchronize()

    samples_color_ms: list[float] = []
    samples_inference_ms: list[float] = []
    samples_postprocess_ms: list[float] = []
    samples_packet_ms: list[float] = []
    backend_used_set: set = set()
    zero_copy_seen = True

    print(f"running for {duration_s:.1f}s ...")
    start = time.perf_counter()
    end_at = start + duration_s
    it = 0
    while time.perf_counter() < end_at:
        raw = src.next_batch()
        t_pkt = time.perf_counter()
        color = deb.convert(raw)
        outputs, inf_ms = runner.timed_infer(color.tensor)
        # Trivial postprocess: shape check (kept inside packet so we report it)
        t_post = time.perf_counter()
        for buf in outputs.values():
            _ = buf.shape  # no-op; we don't do NMS in this lab benchmark
        post_ms = (time.perf_counter() - t_post) * 1000.0
        torch.cuda.synchronize()
        pkt_ms = (time.perf_counter() - t_pkt) * 1000.0

        samples_color_ms.append(color.timings_ms["total_color_ms"])
        samples_inference_ms.append(inf_ms)
        samples_postprocess_ms.append(post_ms)
        samples_packet_ms.append(pkt_ms)
        backend_used_set.add(color.backend_used)
        if not color.tensor.is_cuda:
            zero_copy_seen = False
        it += 1
        if it % 200 == 0:
            print(f"  it={it} color_ms={color.timings_ms['total_color_ms']:.2f} "
                  f"inf_ms={inf_ms:.2f} pkt_ms={pkt_ms:.2f}")

    elapsed = time.perf_counter() - start
    print(f"done. iterations={it} elapsed={elapsed:.2f}s")

    aggregate = aggregate_run(
        packet_samples_ms=samples_packet_ms,
        stage_samples_ms={
            "color_ms": samples_color_ms,
            "inference_ms": samples_inference_ms,
            "postprocess_ms": samples_postprocess_ms,
        },
        batch=batch,
    )

    print("\n--- summary ---")
    cs = aggregate["stage_stats"]["color_ms"]
    isr = aggregate["stage_stats"]["inference_ms"]
    print(f"color_ms     median = {cs['median_ms']:.2f}  p95 = {cs['p95_ms']:.2f}")
    print(f"inference_ms median = {isr['median_ms']:.2f}  p95 = {isr['p95_ms']:.2f}")
    print(f"packet_ms    median = {aggregate['verdict']['packet_ms_median']:.2f}  "
          f"p95 = {aggregate['verdict']['packet_ms_p95']:.2f}")
    print(f"FPS per camera median   = {aggregate['verdict']['fps_per_camera_median']:.2f}")
    print(f"FPS per camera safe p95 = {aggregate['verdict']['fps_per_camera_safe_p95']:.2f}")
    print(f"50 FPS safe p95? = "
          f"{'PASS' if aggregate['verdict']['pass_50fps_safe_p95'] else 'FAIL'}")

    out_payload = {
        "what": "full_synthetic",
        "engine": str(Path(engine).resolve()),
        "engine_info": info.to_dict(),
        "capture_shape": {"h": cap_h, "w": cap_w},
        "inference_input_shape": {"h": in_h, "w": in_w},
        "batch": batch,
        "duration_s": duration_s,
        "warmup": warmup,
        "bayer_pattern": bayer_pattern,
        "backends_used": sorted(backend_used_set),
        "zero_copy_to_inference": zero_copy_seen,
        "fallback_used": "torch_gpu_fallback" in backend_used_set,
        "iterations": it,
        "elapsed_s": elapsed,
        "aggregate": aggregate,
    }

    label = f"{in_h}x{in_w}_b{batch}"
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / f"full_synthetic_{label}.json"
    json_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    csv_path = results_dir / f"full_synthetic_{label}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "color_ms", "inference_ms", "postprocess_ms", "packet_ms"])
        for i in range(it):
            w.writerow([i,
                        f"{samples_color_ms[i]:.4f}",
                        f"{samples_inference_ms[i]:.4f}",
                        f"{samples_postprocess_ms[i]:.4f}",
                        f"{samples_packet_ms[i]:.4f}"])
    print(f"\n-> wrote {json_path}")
    print(f"-> wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


