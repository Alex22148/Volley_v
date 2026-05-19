"""Generate reports/live_pipeline_timing_report.md.

Reads the existing benchmark JSONs (study_native_cpu, study at 2464x2056
in gpu_zero_copy mode from the max_resolution sweep) and any live probe
log files in logs/live_probe_*.log to assemble a single comparison
report.
"""
from __future__ import annotations

import json
import logging
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

_LINE_PAT = re.compile(r"\[LIVE_PACKET_TIMING\]\s+(.*)$")
_KV_PAT = re.compile(r"(\w+)=([^\s]+)")


def _parse_kv(text: str) -> Dict[str, str]:
    return {m.group(1): m.group(2) for m in _KV_PAT.finditer(text)}


def _parse_stage_block(block: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for entry in block.split(","):
        if "=" not in entry:
            continue
        k, v = entry.split("=", 1)
        try:
            out[k] = float(v)
        except Exception:
            continue
    return out


def parse_live_probe_log(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _LINE_PAT.search(line)
        if not m:
            continue
        body = m.group(1)
        # Strip out median_ms=... and p95_ms=... blocks first (they contain commas)
        median_block = ""
        p95_block = ""
        m_med = re.search(r"median_ms=([^\s]+(?:,[^\s]+)*)", body)
        if m_med:
            median_block = m_med.group(1)
            body = body.replace(m_med.group(0), "")
        m_p95 = re.search(r"p95_ms=([^\s]+(?:,[^\s]+)*)", body)
        if m_p95:
            p95_block = m_p95.group(1)
            body = body.replace(m_p95.group(0), "")
        kv: Dict[str, Any] = {}
        for k_, v_ in _parse_kv(body).items():
            kv[k_] = v_
        kv["median_ms_by_stage"] = _parse_stage_block(median_block) if median_block else {}
        kv["p95_ms_by_stage"] = _parse_stage_block(p95_block) if p95_block else {}
        rows.append(kv)
    return rows


def aggregate_live_logs(log_dir: Path) -> Dict[str, Dict]:
    """Group probe lines by tag, return aggregated medians+p95 across runs."""
    out: Dict[str, Dict] = {}
    if not log_dir.exists():
        return out
    for p in sorted(log_dir.glob("live_probe_*.log")):
        rows = parse_live_probe_log(p)
        if not rows:
            continue
        # tag is in each row; use first
        tag = rows[0].get("tag", p.stem)
        # take the last row per file as the most "settled" reading
        last = rows[-1]
        bucket = out.setdefault(tag, {"runs": [], "files": []})
        bucket["files"].append(str(p))
        bucket["runs"].append(last)
    # collapse multiple runs per tag into mean of medians
    for tag, bucket in out.items():
        runs = bucket["runs"]
        # gather union of stages
        stage_med: Dict[str, List[float]] = {}
        stage_p95: Dict[str, List[float]] = {}
        for r in runs:
            for k, v in (r.get("median_ms_by_stage") or {}).items():
                stage_med.setdefault(k, []).append(float(v))
            for k, v in (r.get("p95_ms_by_stage") or {}).items():
                stage_p95.setdefault(k, []).append(float(v))
        out[tag]["median_ms"] = {k: statistics.fmean(vs) for k, vs in stage_med.items()}
        out[tag]["p95_ms"] = {k: max(vs) for k, vs in stage_p95.items()}
        out[tag]["last_extras"] = {k: v for k, v in (runs[-1].items())
                                   if k not in ("median_ms_by_stage", "p95_ms_by_stage")}
    return out


def _load_json(path: Path) -> Optional[Dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _benchmark_native_cpu(repo: Path) -> Optional[Dict]:
    j = _load_json(repo / "reports" / "study_native_cpu.json")
    if not j:
        return None
    m = j.get("measurement") or {}
    return {
        "median_compute_ms": m.get("median_compute_ms"),
        "p95_compute_ms": m.get("p95_compute_ms"),
        "median_color_ms": m.get("median_color_ms"),
        "median_inference_ms": m.get("median_inference_ms"),
        "median_color_download_ms": m.get("median_color_download_ms"),
        "median_inference_upload_ms": m.get("median_inference_upload_ms"),
        "fps_median": m.get("max_fps_from_median"),
        "fps_p95": m.get("max_fps_from_p95"),
        "color_output_location": m.get("color_output_location"),
        "inference_input_location": m.get("inference_input_location"),
    }


def _benchmark_gpu_zero_copy_native(repo: Path) -> Optional[Dict]:
    j = _load_json(repo / "reports" / "study_max_resolution_20ms.json")
    if not j:
        return None
    for c in j.get("candidates", []):
        if c.get("width") == 2464 and c.get("height") == 2056 and not c.get("error"):
            return {
                "median_compute_ms": c.get("median_compute_ms"),
                "p95_compute_ms": c.get("p95_compute_ms"),
                "median_color_ms": c.get("median_color_ms"),
                "median_inference_ms": c.get("median_inference_ms"),
                "median_color_download_ms": c.get("median_color_download_ms"),
                "median_inference_upload_ms": c.get("median_inference_upload_ms"),
                "fps_median": c.get("max_fps_from_median"),
                "fps_p95": c.get("max_fps_from_p95"),
                "color_output_location": c.get("color_output_location"),
                "inference_input_location": c.get("inference_input_location"),
            }
    return None


def _benchmark_gpu_roundtrip_native(repo: Path) -> Optional[Dict]:
    j = _load_json(repo / "reports" / "study_native_roundtrip.json")
    if not j:
        return None
    m = j.get("measurement") or {}
    return {
        "median_compute_ms": m.get("median_compute_ms"),
        "p95_compute_ms": m.get("p95_compute_ms"),
        "median_color_ms": m.get("median_color_ms"),
        "median_inference_ms": m.get("median_inference_ms"),
        "median_color_download_ms": m.get("median_color_download_ms"),
        "median_inference_upload_ms": m.get("median_inference_upload_ms"),
        "fps_median": m.get("max_fps_from_median"),
        "fps_p95": m.get("max_fps_from_p95"),
        "color_output_location": m.get("color_output_location"),
        "inference_input_location": m.get("inference_input_location"),
    }


def _fmt_ms(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.2f}"
    except Exception:
        return "—"


def _fmt_fps(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.1f}"
    except Exception:
        return "—"


def build_report(repo: Path) -> str:
    cpu = _benchmark_native_cpu(repo) or {}
    rt = _benchmark_gpu_roundtrip_native(repo) or {}
    zc = _benchmark_gpu_zero_copy_native(repo) or {}

    log_dir = repo / "logs"
    live = aggregate_live_logs(log_dir)
    live_back = live.get("live_backend") or {}
    live_ringbuf = live.get("ring_buffer") or {}

    lines: List[str] = []
    a = lines.append
    a("# Live Pipeline Timing Report")
    a("")
    a("Comparison of three pipeline shapes for the 4-camera VolleyHub system at native 2464×2056.")
    a("Benchmarks use the synthetic source + identical TRT FP16 b4@640 engine.")
    a("Live data is sourced from the diagnostic probe (`VOLLEYHUB_LIVE_PROBE=1`), which collects per-stage timings from the existing production timers without changing pipeline behaviour.")
    a("")
    a("## Headline numbers (native 2464×2056, RTX 2080 Super)")
    a("")
    a("| Tor                       | median compute | p95     | median color | median inference | FPS median | FPS p95 |")
    a("|---------------------------|-----------------|---------|---------------|-------------------|-------------|----------|")
    a(f"| benchmark_cpu_equivalent   | {_fmt_ms(cpu.get('median_compute_ms'))} ms | {_fmt_ms(cpu.get('p95_compute_ms'))} ms | {_fmt_ms(cpu.get('median_color_ms'))} ms | {_fmt_ms(cpu.get('median_inference_ms'))} ms | {_fmt_fps(cpu.get('fps_median'))} | {_fmt_fps(cpu.get('fps_p95'))} |")
    a(f"| benchmark_gpu_roundtrip    | {_fmt_ms(rt.get('median_compute_ms'))} ms | {_fmt_ms(rt.get('p95_compute_ms'))} ms | {_fmt_ms(rt.get('median_color_ms'))} ms | {_fmt_ms(rt.get('median_inference_ms'))} ms | {_fmt_fps(rt.get('fps_median'))} | {_fmt_fps(rt.get('fps_p95'))} |")
    a(f"| benchmark_gpu_zero_copy    | {_fmt_ms(zc.get('median_compute_ms'))} ms | {_fmt_ms(zc.get('p95_compute_ms'))} ms | {_fmt_ms(zc.get('median_color_ms'))} ms | {_fmt_ms(zc.get('median_inference_ms'))} ms | {_fmt_fps(zc.get('fps_median'))} | {_fmt_fps(zc.get('fps_p95'))} |")
    if live_back:
        med = live_back.get("median_ms", {})
        p95 = live_back.get("p95_ms", {})
        extras = live_back.get("last_extras", {})
        total_med = med.get("total_packet_ms")
        total_p95 = p95.get("total_packet_ms")
        col_med = med.get("cpu_debayer_ms")
        inf_med = med.get("inference_ms")
        fps_med = (1000.0 / total_med) if total_med else None
        fps_p95 = (1000.0 / total_p95) if total_p95 else None
        a(f"| **real_live_pipeline**     | **{_fmt_ms(total_med)} ms** | **{_fmt_ms(total_p95)} ms** | {_fmt_ms(col_med)} ms | {_fmt_ms(inf_med)} ms | **{_fmt_fps(fps_med)}** | **{_fmt_fps(fps_p95)}** |")
        a("")
        a(f"Live extras: role={extras.get('role','?')}, color_backend={extras.get('color_backend','?')}, "
          f"inference_backend={extras.get('inference_backend','?')}, batch={extras.get('batch','?')}, "
          f"resolution={extras.get('resolution','?')}, frames_window={extras.get('frames','?')}.")
    else:
        a("| **real_live_pipeline**     | _to be filled — run with VOLLEYHUB_LIVE_PROBE=1 and re-run this generator_ |  |  |  |  |  |")
        a("")
        a("> No `logs/live_probe_*.log` files found yet. Activate probe and run the live system, then re-run this generator. See the activation section below.")

    a("")
    a("## Architectural deltas (why the three rows differ)")
    a("")
    a("All three benchmark rows use the SAME hardware, SAME TRT engine, SAME native 2464×2056 frames. Only the data path differs.")
    a("")
    a("| Stage                 | benchmark_cpu_equivalent | benchmark_gpu_roundtrip | benchmark_gpu_zero_copy | real_live_pipeline                              |")
    a("|-----------------------|---------------------------|--------------------------|--------------------------|-------------------------------------------------|")
    a("| debayer               | CPU `cv2.cvtColor`        | GPU torch (download CPU)| GPU torch (kept on CUDA) | CPU `cv2.cvtColor` (production default)          |")
    a("| inference input       | numpy BGR HWC uint8       | numpy BGR HWC uint8     | torch BCHW float on cuda | numpy BGR HWC uint8                              |")
    a("| inference batch       | 4 (TRT b4 engine)         | 4                       | 4                        | **1** (production processes one role per loop) |")
    a("| process model         | single-process            | single-process          | single-process           | **multi-process IPC** (grabber→color→YOLO)      |")
    a("| inference call format | `model.predict(numpy)`    | `model.predict(numpy)`  | `model.predict(tensor)`  | `model.predict(numpy)` per single frame         |")
    a("")
    a("## Per-stage breakdown (live data)")
    a("")
    if live_back and live_back.get("median_ms"):
        a("Median (rolling window) per stage in `live_backend` probe tag:")
        a("")
        a("| stage                | median ms | p95 ms |")
        a("|----------------------|-----------|--------|")
        for k in sorted(live_back["median_ms"].keys()):
            med = live_back["median_ms"].get(k)
            p95v = live_back["p95_ms"].get(k)
            a(f"| {k:<20} | {_fmt_ms(med)}   | {_fmt_ms(p95v)} |")
    else:
        a("_Live data not yet collected._ Once `logs/live_probe_live_backend_*.log` exists, this section auto-fills with stage breakdowns: `grab_ms`, `cpu_debayer_ms`, `preprocess_ms`, `inference_ms`, `postprocess_ms`, `total_packet_ms`.")
    a("")
    if live_ringbuf and live_ringbuf.get("median_ms"):
        a("Ring buffer push:")
        a("")
        med = live_ringbuf["median_ms"].get("buffer_push_ms")
        p95v = live_ringbuf["p95_ms"].get("buffer_push_ms")
        a(f"- `buffer_push_ms` median = {_fmt_ms(med)} ms, p95 = {_fmt_ms(p95v)} ms")
    a("")
    a("## How to populate the live row")
    a("")
    a("```powershell")
    a("# 1. activate the probe before launching production")
    a("$env:VOLLEYHUB_LIVE_PROBE = \"1\"")
    a("$env:VOLLEYHUB_LIVE_PROBE_EVERY = \"30\"   # log line every 30 packets")
    a("$env:VOLLEYHUB_LIVE_PROBE_LOGDIR = \"logs\"")
    a("")
    a("# 2. start the live pipeline as usual (gui.py / main.py)")
    a("python main.py")
    a("# ... let the live system run for a minute under realistic load")
    a("")
    a("# 3. regenerate this report (auto-collates probe logs)")
    a("python -m src.runtime_diagnostics.build_live_pipeline_report")
    a("")
    a("# 4. to disable the probe, just unset the env var on next launch")
    a("$env:VOLLEYHUB_LIVE_PROBE = \"\"")
    a("```")
    a("")
    a("## Decision pending")
    a("")
    a("Pick from these once live data is available:")
    a("")
    a("- **A)** Reduce source resolution (1920×1080 fits in 20 ms in benchmark gpu_zero_copy; production CPU path likely needs further measurement).")
    a("- **B)** Build the GPU debayer adapter (architecture note already plans it as `future_cuda_npp`). Expected gain: ~14 ms → ~3-5 ms at native res.")
    a("- **C)** Switch live inference to gpu_zero_copy (BCHW float on CUDA → TRT). Requires a ~b1@native engine and a tensor-input path in `live_backend_controller._run_inference`.")
    a("- **D)** Lower `publish_interval_ms` only matters if GUI publish is the bottleneck — probe will tell us.")
    a("- **E)** Switch from 1-frame-per-loop to a true 4-frame batched inference (use `b4@native` engine). Requires changes to `_resolve_key` to gather all 4 roles per iteration.")
    a("")
    a("Choosing without live data risks optimising the wrong stage.")
    return "\n".join(lines) + "\n"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    repo = Path(__file__).resolve().parents[2]
    out_path = repo / "reports" / "live_pipeline_timing_report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = build_report(repo)
    out_path.write_text(text, encoding="utf-8")
    print(f"wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
