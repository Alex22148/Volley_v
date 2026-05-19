"""01_benchmark_inference_only.py â€” TensorRT inference only.

WHAT THIS SCRIPT MEASURES:
    Just the TensorRT forward pass. A random CUDA tensor is created once
    and re-used. The only timer is a CUDA event around execute_async_v3.

WHAT THIS SCRIPT DOES NOT MEASURE:
    * Camera grab.
    * Bayer debayer or color conversion.
    * Resize / normalize on the GPU.
    * Host -> device upload (the tensor lives on the GPU already).
    * NMS / postprocessing.
    * Any production code, GUI, ring buffer, or IPC.

INPUT:
    --engine        path to a .engine file
    --input-shape   H W (must match the engine; sanity-checked)
    --batch         batch size (must match the engine; sanity-checked)
    --duration-s    how long to run after warmup
    --warmup        warmup iterations (skipped from stats)

OUTPUT:
    results/inference_only_<HxW>.json
    results/inference_only_<HxW>.csv

FPS DEFINITION:
    fps_per_camera = 1000 / inference_ms.
    With batch=4, each iteration is one packet = four synchronised
    camera frames. So one packet/s = one frame/s/camera.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_ROOT))

from src.trt_runner import TrtRunner  # noqa: E402
from src.metrics import aggregate_run  # noqa: E402

# ----------------------------- static config (no argparse)
ENGINE = PKG_ROOT / "engines" / "best__fp16_640x640_b4.engine"
INPUT_SHAPE = (640, 640)  # (H, W)
BATCH = 4
DURATION_S = 30.0
WARMUP = 10
RESULTS_DIR = PKG_ROOT / "reports"


def _parse_cli(argv: list[str]) -> dict:
    cfg = {
        "engine": str(ENGINE),
        "input_shape": [int(INPUT_SHAPE[0]), int(INPUT_SHAPE[1])],
        "batch": int(BATCH),
        "duration_s": float(DURATION_S),
        "warmup": int(WARMUP),
        "results_dir": str(RESULTS_DIR),
    }
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--engine":
            cfg["engine"] = argv[i + 1]; i += 2; continue
        if a == "--input-shape":
            cfg["input_shape"] = [int(argv[i + 1]), int(argv[i + 2])]; i += 3; continue
        if a == "--batch":
            cfg["batch"] = int(argv[i + 1]); i += 2; continue
        if a == "--duration-s":
            cfg["duration_s"] = float(argv[i + 1]); i += 2; continue
        if a == "--warmup":
            cfg["warmup"] = int(argv[i + 1]); i += 2; continue
        if a == "--results-dir":
            cfg["results_dir"] = argv[i + 1]; i += 2; continue
        raise SystemExit(f"Unknown argument: {a}")
    return cfg


def main() -> int:
    cfg = _parse_cli(sys.argv[1:])
    engine = Path(cfg["engine"])
    input_shape = (int(cfg["input_shape"][0]), int(cfg["input_shape"][1]))
    batch = int(cfg["batch"])
    duration_s = float(cfg["duration_s"])
    warmup = int(cfg["warmup"])
    results_dir = Path(cfg["results_dir"])

    print("=" * 72)
    print("fastpath_lab_benchmark / 01_benchmark_inference_only.py")
    print("WHAT: measures TensorRT engine forward time on a random CUDA tensor.")
    print("WHAT NOT: no camera grab, debayer, color convert, host->device,")
    print("          or postprocess in this number.")
    print(f"INPUT:  engine={engine}  input={list(input_shape)}  batch={batch}")
    print(f"        duration={duration_s}s  warmup={warmup}")
    print("FPS:    fps_per_camera = 1000 / inference_ms (batch=4 = 1 packet).")
    print("=" * 72)

    import torch

    runner = TrtRunner(str(engine))
    info = runner.info()
    eng_b, _eng_c, eng_h, eng_w = info.input_shape
    if eng_b != batch:
        raise SystemExit(
            f"engine batch={eng_b} but BATCH={batch}; use a matching engine."
        )
    if (eng_h, eng_w) != input_shape:
        raise SystemExit(
            f"engine input HxW = {eng_h}x{eng_w} but INPUT_SHAPE "
            f"{input_shape[0]}x{input_shape[1]}; use a matching engine."
        )
    print(f"engine: input={info.input_shape} dtype={info.input_dtype} "
          f"outputs={list(info.output_shapes.keys())}")

    # Pre-allocate a single random input on CUDA â€” we re-use it.
    x = torch.rand(info.input_shape, dtype=torch.float32, device="cuda")
    if info.is_fp16:
        x = x.to(torch.float16)

    print(f"warmup: {warmup} iterations...")
    for _ in range(max(0, warmup)):
        runner.infer(x)
    torch.cuda.synchronize()

    samples_inference_ms: list[float] = []
    samples_packet_ms: list[float] = []
    start = time.perf_counter()
    end_at = start + duration_s
    it = 0
    print(f"running for {duration_s:.1f}s ...")
    while time.perf_counter() < end_at:
        t_pkt = time.perf_counter()
        _, inf_ms = runner.timed_infer(x)
        torch.cuda.synchronize()
        pkt_ms = (time.perf_counter() - t_pkt) * 1000.0
        samples_inference_ms.append(inf_ms)
        samples_packet_ms.append(pkt_ms)
        it += 1
        if it % 200 == 0:
            print(f"  it={it} last_inference_ms={inf_ms:.2f} "
                  f"last_packet_ms={pkt_ms:.2f}")

    elapsed = time.perf_counter() - start
    print(f"done. iterations={it} elapsed={elapsed:.2f}s "
          f"(actual fps_packets={it / elapsed:.2f})")

    aggregate = aggregate_run(
        packet_samples_ms=samples_packet_ms,
        stage_samples_ms={"inference_ms": samples_inference_ms},
        batch=batch,
    )

    print("\n--- summary ---")
    print(f"inference_ms median = {aggregate['stage_stats']['inference_ms']['median_ms']:.2f}")
    print(f"inference_ms p95    = {aggregate['stage_stats']['inference_ms']['p95_ms']:.2f}")
    print(f"packet_ms    median = {aggregate['verdict']['packet_ms_median']:.2f}")
    print(f"packet_ms    p95    = {aggregate['verdict']['packet_ms_p95']:.2f}")
    print(f"FPS per camera median        = {aggregate['verdict']['fps_per_camera_median']:.2f}")
    print(f"FPS per camera safe p95      = {aggregate['verdict']['fps_per_camera_safe_p95']:.2f}")
    print(f"50 FPS safe p95?             = "
          f"{'PASS' if aggregate['verdict']['pass_50fps_safe_p95'] else 'FAIL'}")

    out_payload = {
        "what": "inference_only",
        "engine": str(engine.resolve()),
        "engine_info": info.to_dict(),
        "inference_input_shape": {"h": eng_h, "w": eng_w},
        "batch": batch,
        "duration_s": duration_s,
        "warmup": warmup,
        "iterations": it,
        "elapsed_s": elapsed,
        "aggregate": aggregate,
    }

    label = f"{eng_h}x{eng_w}_b{batch}"
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / f"inference_only_{label}.json"
    json_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")

    csv_path = results_dir / f"inference_only_{label}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "inference_ms", "packet_ms"])
        for i, (inf, pkt) in enumerate(zip(samples_inference_ms, samples_packet_ms)):
            w.writerow([i, f"{inf:.4f}", f"{pkt:.4f}"])

    print(f"\n-> wrote {json_path}")
    print(f"-> wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


