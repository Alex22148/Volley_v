"""06_run_single_test.py â€” one-shot lab benchmark driven by a config file.

WHAT THIS SCRIPT DOES:
    Reads `configs/single_test.json` (or any --config path you pass)
    and runs ONE benchmark with those parameters. The config controls:

        engine             : path to a .engine file
        capture_shape_hw   : synthetic Bayer size [H, W]
        inference_shape_hw : engine input [H, W]
        batch              : engine batch size
        active_cameras     : 1..batch â€” how many slots carry real content;
                             slots above this are dark blanks. Lets you
                             simulate 1 / 2 / 3 / 4 of 4 cameras connected.
        target_fps         : desired sustained FPS per camera. 0 = no
                             pacing (free-run). 50 = aim for 20 ms / packet.
        duration_s         : measurement window after warmup.
        warmup             : warmup iterations.
        bayer_pattern      : RG / BG / GR / GB.

WHAT THIS SCRIPT DOES NOT DO:
    * No Basler grab, no GUI, no ring buffer, no IPC.
    * No NMS / detection postprocess (trivial shape check only).
    * No multi-resolution sweep â€” for that use 04_resolution_sweep.py.

OUTPUT:
    results/single_test_<label>.json   (label = "<inferenceHxW>_b<batch>_<act>cam_<fps>fps")
    results/single_test_<label>.csv
    Console: clear summary with the verdict "did we hit target FPS?".
"""
from __future__ import annotations

import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

PKG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_ROOT))

from src.cuda_debayer import CudaDebayer, CudaDebayerConfig  # noqa: E402
from src.synthetic_raw import SyntheticRawSource, SyntheticRawConfig  # noqa: E402
from src.trt_runner import TrtRunner  # noqa: E402
from src.metrics import aggregate_run  # noqa: E402

# ----------------------------- static config (no argparse)
CONFIG_PATH = PKG_ROOT / "configs" / "single_test.json"
RESULTS_DIR = PKG_ROOT / "reports"


def _infer_hw_from_engine_name(engine_value: str) -> tuple[int, int] | None:
    name = Path(engine_value).name.lower()

    # np. "...1088x1920..." -> (1088, 1920)
    m = re.search(r"(?<!\d)(\d{3,4})x(\d{3,4})(?!\d)", name)
    if m:
        return int(m.group(1)), int(m.group(2))

    # mapowanie labeli z nazw eksportu, np. "v640", "img960"
    if "v640" in name or "img640" in name:
        return 640, 640
    if "v960" in name or "img960" in name:
        return 960, 960
    if "v1280" in name or "img1280" in name:
        return 1280, 1280
    if "v512" in name or "img512" in name:
        return 512, 512

    # fallback: dowolne samotne "640"/"960"/"1280"/"512" w nazwie
    m_sq = re.search(r"(?<!\d)(512|640|960|1280)(?!\d)", name)
    if m_sq:
        v = int(m_sq.group(1))
        return v, v

    return None


def _load_config(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"config not found: {path}")
    raw = path.read_text(encoding="utf-8")
    cfg = json.loads(raw)
    # drop _help / _fields / etc. â€” they're documentation
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def _validate(cfg: dict) -> None:
    required = ["engine",
                "batch", "active_cameras", "target_fps", "duration_s",
                "warmup", "bayer_pattern"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise SystemExit(f"config missing fields: {missing}")
    if not (1 <= int(cfg["active_cameras"]) <= int(cfg["batch"])):
        raise SystemExit(
            f"active_cameras={cfg['active_cameras']} must be in [1, batch={cfg['batch']}]"
        )
    if cfg["bayer_pattern"] not in ("RG", "BG", "GR", "GB"):
        raise SystemExit(f"bayer_pattern={cfg['bayer_pattern']} must be RG/BG/GR/GB")
    if bool(cfg.get("disable_resize", False)) and "capture_shape_hw" in cfg and "inference_shape_hw" in cfg:
        cap_h, cap_w = [int(v) for v in cfg["capture_shape_hw"]]
        inf_h, inf_w = [int(v) for v in cfg["inference_shape_hw"]]
        if (cap_h, cap_w) != (inf_h, inf_w):
            raise SystemExit(
                "disable_resize=true wymaga capture_shape_hw == inference_shape_hw "
                f"(got capture={cap_h}x{cap_w}, inference={inf_h}x{inf_w})"
            )


def main() -> int:
    cfg_path = Path(CONFIG_PATH)
    cfg = _load_config(cfg_path)
    inferred_hw = _infer_hw_from_engine_name(str(cfg.get("engine", "")))

    if "inference_shape_hw" not in cfg:
        if inferred_hw is None:
            raise SystemExit(
                "Brak inference_shape_hw i nie udało się wywnioskować rozdzielczości z nazwy engine. "
                "Uzupełnij inference_shape_hw albo nazwę engine z rozmiarem (np. 640, 960, 1088x1920)."
            )
        cfg["inference_shape_hw"] = [inferred_hw[0], inferred_hw[1]]

    if bool(cfg.get("disable_resize", False)) and "capture_shape_hw" not in cfg:
        cfg["capture_shape_hw"] = list(cfg["inference_shape_hw"])

    if "capture_shape_hw" not in cfg:
        # domyślny capture jak wcześniej (sensor-like), jeśli test ma mierzyć resize
        cfg["capture_shape_hw"] = [1080, 1920]

    _validate(cfg)

    engine_path = (PKG_ROOT / cfg["engine"]) if not Path(cfg["engine"]).is_absolute() \
        else Path(cfg["engine"])
    cap_h, cap_w = [int(v) for v in cfg["capture_shape_hw"]]
    in_h, in_w = [int(v) for v in cfg["inference_shape_hw"]]
    batch = int(cfg["batch"])
    active = int(cfg["active_cameras"])
    target_fps = float(cfg["target_fps"])
    duration_s = float(cfg["duration_s"])
    warmup = int(cfg["warmup"])
    pattern = cfg["bayer_pattern"]
    disable_resize = bool(cfg.get("disable_resize", False))
    period_ms = (1000.0 / target_fps) if target_fps > 0 else None

    bar = "=" * 72
    sep = "-" * 64

    print(bar)
    print("fastpath_lab_benchmark / 06_run_single_test.py")
    print(bar)
    print()
    print("1. WHAT THIS TEST MEASURES")
    print("   " + sep)
    print("   * GPU debayer (custom CUDA kernel; torch fallback only if the")
    print("     native kernel cannot be built on this machine).")
    print("   * Bilinear resize + normalize to the engine input size.")
    print("   * TensorRT engine forward pass on a CUDA tensor (zero-copy: the")
    print("     debayer output feeds directly into the engine).")
    print("   * Wall-clock cost of one packet, including any pacing wait.")
    print()
    print("2. WHAT THIS TEST DOES NOT MEASURE")
    print("   " + sep)
    print("   * Basler / pypylon camera grab.")
    print("   * GUI, ring buffer, preview worker, shared memory, IPC.")
    print("   * Hardware synchronisation between four real cameras.")
    print("   * NMS or any real detection postprocessing.")
    print("   * CPU-side synthetic generation: ONE Bayer batch is built up-front")
    print("     and re-used in the loop so the timing reflects the GPU path")
    print("     only (numpy mosaicing at 1080x1920x4 would otherwise dominate).")
    print()
    print("3. HOW FPS IS CALCULATED")
    print("   " + sep)
    print(f"   * One \"packet\" = one TensorRT call with batch={batch} images,")
    print(f"     conceptually {batch} synchronised camera frames.")
    print("   * fps_per_camera = 1000 / packet_ms. Because each packet")
    print("     contains one frame per camera, packets/s = FPS per camera.")
    print("   * uncapped median FPS = 1000 / median(packet_ms)")
    print("       -> peak throughput on this machine, no pacing limit.")
    print("   * uncapped safe p95 FPS = 1000 / p95(packet_ms)")
    print("       -> pessimistic: sustained in at least 95% of packets.")
    print("   * actual sustained FPS = iterations / elapsed")
    print("       -> end-to-end count, INCLUDING any pacing sleeps.")
    print(f"   * total images/s = fps_per_camera * batch = fps_per_camera * {batch}.")
    print()
    print("   Pacing (target_fps):")
    print("   * target_fps > 0  : pacing mode. Each iteration waits until")
    print("                       1000/target_fps ms have elapsed since the")
    print("                       previous one. \"overruns\" are iterations where")
    print("                       the GPU work exceeded the budget. Use this to")
    print("                       simulate \"frames arrive at exactly N FPS\".")
    print("   * target_fps = 0  : free-run, no sleeps. Reports the maximum")
    print("                       throughput the GPU fast path can sustain.")
    print("   * target_hit      : YES when actual FPS >= 98% of target AND")
    print("                       overrun ratio < 5%.")
    print()
    print("4. ACTIVE_CAMERAS VS BATCH")
    print("   " + sep)
    print("   * `batch` is FIXED by the .engine (static-shape build). For a")
    print(f"     batch={batch} engine every call processes {batch} images â€” no")
    print("     exception, no partial batches.")
    print("   * `active_cameras` (1..batch) is a SCENARIO knob, not a GPU")
    print("     workload knob. It chooses how many of the batch slots carry")
    print("     real synthetic content; the remaining slots are dark blank")
    print("     frames (so the test mimics 1/4, 2/4, 3/4, 4/4 cameras connected).")
    print("   * Therefore: in this engine,")
    print(f"       active_cameras=1 has the SAME GPU cost as active_cameras={batch}.")
    print("     The knob is useful for *interpreting* a run (\"would 1 of 4")
    print("     cameras leave us 50 FPS headroom?\") but it does NOT shorten")
    print("     a static-batch TRT call.")
    print("   * To genuinely shrink the workload for fewer cameras, ship a")
    print("     smaller-batch engine (e.g. batch=1) and set both batch and")
    print(f"     active_cameras accordingly. See engines/README_ENGINES.md.")
    print()
    print(bar)
    print("CONFIG SOURCE")
    print(f"  file               = {cfg_path}")
    print(f"  engine             = {engine_path}")
    print(f"  capture_shape_hw   = {cap_h}x{cap_w}")
    print(f"  inference_shape_hw = {in_h}x{in_w}")
    if inferred_hw is not None:
        print(f"  inferred_from_name = {inferred_hw[0]}x{inferred_hw[1]}")
    print(f"  batch              = {batch}")
    print(f"  active_cameras     = {active}/{batch}"
          f"   (blank slots = {max(0, batch - active)})")
    if period_ms is None:
        print(f"  target_fps         = 0 (free-run, no pacing)")
    else:
        print(f"  target_fps         = {target_fps:.1f}  -> period = {period_ms:.2f} ms")
    print(f"  duration_s         = {duration_s:.1f}")
    print(f"  warmup             = {warmup}")
    print(f"  bayer_pattern      = {pattern}")
    print(f"  disable_resize     = {disable_resize}")
    print(bar)

    import torch

    runner = TrtRunner(str(engine_path))
    info = runner.info()
    eng_b, _eng_c, eng_h, eng_w = info.input_shape
    if eng_b != batch:
        raise SystemExit(f"engine batch={eng_b} != config batch={batch}")
    if (eng_h, eng_w) != (in_h, in_w):
        raise SystemExit(
            f"engine input {eng_h}x{eng_w} != config inference_shape_hw {in_h}x{in_w}")

    synth_cfg_in = cfg.get("synthetic_source") or {}
    src = SyntheticRawSource(SyntheticRawConfig(
        capture_hw=(cap_h, cap_w),
        batch=batch,
        bayer_pattern=pattern,
        dtype=str(synth_cfg_in.get("dtype", "uint8")),
        noise_sigma=float(synth_cfg_in.get("noise_sigma", 0.0)),
        ball_enabled=bool(synth_cfg_in.get("ball_enabled", True)),
        ball_radius_px=int(synth_cfg_in.get("ball_radius_px", 32)),
        ball_speed_px_per_frame=float(synth_cfg_in.get("ball_speed_px_per_frame", 24.0)),
        base_intensity=int(synth_cfg_in.get("base_intensity", 90)),
        background_amplitude=int(synth_cfg_in.get("background_amplitude", 30)),
        ball_intensity=int(synth_cfg_in.get("ball_intensity", 220)),
        seed=int(synth_cfg_in.get("seed", 1234)),
    ))
    deb = CudaDebayer(CudaDebayerConfig(
        bayer_pattern=pattern,
        output_hw=(cap_h, cap_w) if disable_resize else (in_h, in_w),
        normalize_01=True,
        half_precision=info.is_fp16,
        prefer_native=True,
    ))
    backend_status = deb.backend_status()
    print(f"debayer backend : native_available={backend_status['native_available']}")
    print(f"engine          : input={info.input_shape} dtype={info.input_dtype}")

    # Build one synthetic batch up-front and reuse it in the measurement
    # loop. We do this so the per-iteration time reflects the GPU fast
    # path only â€” not the CPU-side synthetic Bayer rendering, which is
    # not what the tester is trying to measure here. (The CPU cost is
    # not zero: at 1080x1920x4 frames the synthetic source takes ~15 ms
    # per call, which would dominate the loop and mask the real GPU
    # throughput.)
    blank_dtype = np.uint16 if str(synth_cfg_in.get("dtype", "uint8")) == "uint16" \
        else np.uint8
    blank_frame = np.full((cap_h, cap_w), fill_value=16, dtype=blank_dtype)
    fixed_raw = src.next_batch()
    if active < batch:
        for i in range(active, batch):
            fixed_raw[i] = blank_frame

    def get_batch() -> np.ndarray:
        return fixed_raw

    # ------------------------------------------------------------- warmup
    print(f"warmup: {warmup} iterations...")
    for _ in range(max(0, warmup)):
        color = deb.convert(get_batch())
        runner.infer(color.tensor)
    torch.cuda.synchronize()

    # ----------------------------------------------------------- measure
    samples_color_ms: list[float] = []
    samples_inference_ms: list[float] = []
    samples_packet_ms: list[float] = []
    samples_period_ms: list[float] = []
    backend_used_set: set = set()
    overruns = 0

    print(f"running for {duration_s:.1f}s ...")
    start = time.perf_counter()
    end_at = start + duration_s
    if period_ms is not None:
        next_due = start + period_ms / 1000.0
    else:
        next_due = None
    last_pkt_t = start
    it = 0
    while time.perf_counter() < end_at:
        raw = get_batch()
        t_pkt = time.perf_counter()
        color = deb.convert(raw)
        outputs, inf_ms = runner.timed_infer(color.tensor)
        for buf in outputs.values():
            _ = buf.shape
        torch.cuda.synchronize()
        pkt_ms = (time.perf_counter() - t_pkt) * 1000.0

        samples_color_ms.append(color.timings_ms["total_color_ms"])
        samples_inference_ms.append(inf_ms)
        samples_packet_ms.append(pkt_ms)
        samples_period_ms.append((t_pkt - last_pkt_t) * 1000.0)
        last_pkt_t = t_pkt
        backend_used_set.add(color.backend_used)

        # pacing
        if next_due is not None:
            now = time.perf_counter()
            slack = next_due - now
            if slack > 0:
                time.sleep(slack)
            else:
                overruns += 1
            next_due += period_ms / 1000.0  # type: ignore[operator]

        it += 1
        if it % 200 == 0:
            print(f"  it={it} color_ms={color.timings_ms['total_color_ms']:.2f} "
                  f"inf_ms={inf_ms:.2f} pkt_ms={pkt_ms:.2f} overruns={overruns}")

    elapsed = time.perf_counter() - start
    actual_fps = it / elapsed if elapsed > 0 else 0.0

    aggregate = aggregate_run(
        packet_samples_ms=samples_packet_ms,
        stage_samples_ms={
            "color_ms": samples_color_ms,
            "inference_ms": samples_inference_ms,
        },
        batch=batch,
        target_fps=target_fps if target_fps > 0 else 50.0,
    )

    # Verdict against the *config*'s target_fps (not the canonical 50).
    target_hit: bool | None = None
    if target_fps > 0:
        # We "hit target" if the *actual sustained* FPS reaches the target
        # AND the overrun ratio is low (< 5%).
        overrun_ratio = overruns / max(1, it)
        target_hit = bool(actual_fps >= target_fps * 0.98 and overrun_ratio < 0.05)

    print("\n--- summary ---")
    cs = aggregate["stage_stats"]["color_ms"]
    isr = aggregate["stage_stats"]["inference_ms"]
    print(f"iterations              = {it} in {elapsed:.2f}s")
    print(f"actual sustained FPS    = {actual_fps:.2f}  (per camera, packets/s)")
    if target_fps > 0:
        print(f"target FPS (config)     = {target_fps:.2f}  "
              f"-> overruns = {overruns}/{it} ({100*overruns/max(1,it):.1f}%)")
        print(f"target HIT?             = {'YES' if target_hit else 'NO'}")
    print(f"color_ms     median = {cs['median_ms']:.2f}  p95 = {cs['p95_ms']:.2f}")
    print(f"inference_ms median = {isr['median_ms']:.2f}  p95 = {isr['p95_ms']:.2f}")
    print(f"packet_ms    median = {aggregate['verdict']['packet_ms_median']:.2f}  "
          f"p95 = {aggregate['verdict']['packet_ms_p95']:.2f}")
    print(f"FPS per camera (uncapped median)   = {aggregate['verdict']['fps_per_camera_median']:.2f}")
    print(f"FPS per camera (uncapped safe p95) = {aggregate['verdict']['fps_per_camera_safe_p95']:.2f}")

    payload = {
        "what": "single_test",
        "config_path": str(cfg_path),
        "config": cfg,
        "engine_info": info.to_dict(),
        "capture_shape": {"h": cap_h, "w": cap_w},
        "inference_input_shape": {"h": in_h, "w": in_w},
        "batch": batch,
        "active_cameras": active,
        "target_fps": target_fps,
        "target_period_ms": period_ms,
        "duration_s": duration_s,
        "warmup": warmup,
        "bayer_pattern": pattern,
        "backends_used": sorted(backend_used_set),
        "iterations": it,
        "elapsed_s": elapsed,
        "actual_sustained_fps_per_camera": actual_fps,
        "overruns": overruns,
        "overrun_ratio": overruns / max(1, it),
        "target_hit": target_hit,
        "aggregate": aggregate,
    }

    fps_label = f"{int(target_fps)}fps" if target_fps > 0 else "freerun"
    label = f"{in_h}x{in_w}_b{batch}_{active}cam_{fps_label}"
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / f"single_test_{label}.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_path = results_dir / f"single_test_{label}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "color_ms", "inference_ms", "packet_ms", "period_ms"])
        for i in range(it):
            w.writerow([
                i,
                f"{samples_color_ms[i]:.4f}",
                f"{samples_inference_ms[i]:.4f}",
                f"{samples_packet_ms[i]:.4f}",
                f"{samples_period_ms[i]:.4f}",
            ])
    print(f"\n-> wrote {json_path}")
    print(f"-> wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


