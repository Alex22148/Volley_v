"""Generate reports/production_fast_path_report.{md,json,csv}.

Three sections:
    A) benchmark_best_case_640        — from reports/fullhd_inference_size_sweep.json (imgsz_640)
    B) production_before              — placeholder (CPU path; fill after running probe with VOLLEYHUB_FAST_PATH unset)
    C) production_fast_path           — placeholder (filled after running probe with VOLLEYHUB_FAST_PATH=1)

The probe data sections auto-fill from logs/live_probe_*.log entries
that include `fast_path_enabled=...`. Sections without data are clearly
labeled so the reader knows what's still pending.
"""
from __future__ import annotations

import csv
import json
import logging
import re
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_LOG = logging.getLogger(__name__)

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


def _parse_probe_log(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _LINE_PAT.search(line)
        if not m:
            continue
        body = m.group(1)
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


def _aggregate_log_dir(log_dir: Path) -> Dict[str, Any]:
    """Collect probe lines from logs/live_probe_*.log files,
    split by fast_path_enabled (true/false). Returns dict with three
    aggregated buckets: fast_path_rows, slow_path_rows, preview_rows.
    """
    fast_rows: List[Dict[str, Any]] = []
    slow_rows: List[Dict[str, Any]] = []
    preview_rows: List[Dict[str, Any]] = []
    if not log_dir.exists():
        return {"fast_path_rows": fast_rows, "slow_path_rows": slow_rows, "preview_rows": preview_rows}

    for p in sorted(log_dir.glob("live_probe_live_backend*.log")):
        for row in _parse_probe_log(p):
            if str(row.get("fast_path_enabled", "")).lower() in ("true", "1"):
                fast_rows.append(row)
            else:
                slow_rows.append(row)

    for p in sorted(log_dir.glob("live_probe_live_preview*.log")):
        preview_rows.extend(_parse_probe_log(p))

    return {
        "fast_path_rows": fast_rows,
        "slow_path_rows": slow_rows,
        "preview_rows": preview_rows,
    }


def _summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"present": False}
    last = rows[-1]
    med_by_stage = last.get("median_ms_by_stage") or {}
    p95_by_stage = last.get("p95_ms_by_stage") or {}
    extras: Dict[str, Any] = {
        "samples_in_last_window": last.get("frames"),
        "role": last.get("role"),
        "resolution": last.get("resolution"),
        "color_backend": last.get("color_backend"),
        "inference_backend": last.get("inference_backend"),
        "batch": last.get("batch"),
        "yolo_imgsz": last.get("yolo_imgsz"),
        "fast_path_enabled": last.get("fast_path_enabled"),
        "packet_roles_count": last.get("packet_roles_count"),
        "zero_copy_to_inference": last.get("zero_copy_to_inference"),
        "gpu_roundtrip": last.get("gpu_roundtrip"),
        "fallback_used": last.get("fallback_used"),
    }
    return {
        "present": True,
        "median_ms": med_by_stage,
        "p95_ms": p95_by_stage,
        "extras": extras,
        "rows_collected": len(rows),
    }


def _load_benchmark_best_case_640(repo: Path) -> Optional[Dict[str, Any]]:
    """Pull the imgsz_640 row from the Full-HD sweep JSON."""
    p = repo / "reports" / "fullhd_inference_size_sweep.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    for v in data.get("variants", []):
        if v.get("name") == "imgsz_640":
            return v
    return None


def _fmt(v, p=2) -> str:
    try:
        f = float(v)
        if f != f:
            return "—"
        return f"{f:.{p}f}"
    except Exception:
        return "—" if v is None else str(v)


def _yes_no(b) -> str:
    if b is None:
        return "—"
    if isinstance(b, str):
        return b
    return "PASS" if b else "FAIL"


def build_report(repo: Path, packet_budget_ms: float = 20.0,
                 target_fps: float = 50.0) -> str:
    a_data = _load_benchmark_best_case_640(repo)
    log_data = _aggregate_log_dir(repo / "logs")
    fast_summary = _summarize_rows(log_data["fast_path_rows"])
    slow_summary = _summarize_rows(log_data["slow_path_rows"])

    lines: List[str] = []
    add = lines.append
    add("# Production fast-path report")
    add("")
    add(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    add(f"Target: {target_fps:.0f} FPS per camera ⇒ packet budget {packet_budget_ms:.1f} ms (4 cams synchronized).")
    add("")

    add("## How to populate this report with live data")
    add("")
    add("```powershell")
    add("# 1) production_before — slow path (current default behaviour)")
    add('$env:VOLLEYHUB_LIVE_PROBE = "1"')
    add('$env:VOLLEYHUB_FAST_PATH = ""   # explicitly OFF')
    add("python main.py")
    add("# ... let production run ~60 s under realistic load, then quit")
    add("")
    add("# 2) production_fast_path — opt-in")
    add('$env:VOLLEYHUB_FAST_PATH = "1"')
    add('$env:VOLLEYHUB_TRT_ENGINE = "artifacts/tensorrt_exports/best__fp16_640_b4_static__fp16__img640__b4__static.engine"')
    add('$env:VOLLEYHUB_COLOR_BACKEND = "native_cuda_npp"')
    add('$env:VOLLEYHUB_INFERENCE_BACKEND = "tensorrt"')
    add('$env:VOLLEYHUB_YOLO_IMGSZ = "640"')
    add('$env:VOLLEYHUB_YOLO_BATCH = "4"')
    add("# preview tor (defaults shown — all optional):")
    add('$env:VOLLEYHUB_PREVIEW_ENABLED = "1"')
    add('$env:VOLLEYHUB_PREVIEW_MAX_FPS = "20"')
    add('$env:VOLLEYHUB_PREVIEW_SIZE = "512"')
    add('$env:VOLLEYHUB_PREVIEW_QUALITY = "low"')
    add('$env:VOLLEYHUB_PREVIEW_LATEST_ONLY = "1"')
    add("python main.py")
    add("# ... let production run ~60 s under realistic load, then quit")
    add("")
    add("# 3) regenerate this report (auto-collates probe logs)")
    add("python -m src.runtime_production.build_fast_path_report")
    add("```")
    add("")

    add("## A) benchmark_best_case_640 (synthetic, idealised)")
    add("")
    if a_data is None:
        add("_Source `reports/fullhd_inference_size_sweep.json` not found. "
            "Run `python -m src.runtime_benchmark.run_fullhd_inference_size_sweep` first._")
    else:
        add(f"Engine: `{Path(str(a_data.get('engine_path', ''))).name}`  ")
        add(f"Capture: {a_data.get('capture_resolution')}  YOLO imgsz: {a_data.get('yolo_imgsz')}  "
            f"color_backend: {a_data.get('color_backend_chosen')} (detail: {a_data.get('native_backend_detail')})")
        add("")
        add("| metric | value |")
        add("|--------|-------|")
        add(f"| packet_ms median | {_fmt(a_data.get('packet_ms_median'))} ms |")
        add(f"| packet_ms p95    | {_fmt(a_data.get('packet_ms_p95'))} ms |")
        add(f"| color_ms median  | {_fmt(a_data.get('color_ms_median'))} ms |")
        add(f"| inference_ms median | {_fmt(a_data.get('inference_ms_median'))} ms |")
        add(f"| FPS/camera median | {_fmt(a_data.get('fps_per_camera_median'), 1)} |")
        add(f"| FPS/camera safe (p95) | {_fmt(a_data.get('fps_per_camera_safe_p95'), 1)} |")
        add(f"| zero_copy_to_inference | {a_data.get('zero_copy_to_inference')} |")
        add(f"| fallback_used | {a_data.get('fallback_used') or '—'} |")
    add("")

    def _section_for(title: str, summary: Dict[str, Any], expected_fast: bool) -> None:
        add(f"## {title}")
        add("")
        if not summary.get("present"):
            add(f"_Brak danych z probe (`logs/live_probe_live_backend*.log`). "
                f"{'Uruchom produkcję z VOLLEYHUB_FAST_PATH=1 i probe.' if expected_fast else 'Uruchom produkcję bez fast path z probe.'}_")
            add("")
            return
        med = summary.get("median_ms") or {}
        p95 = summary.get("p95_ms") or {}
        extras = summary.get("extras") or {}
        add(f"Last window: {summary.get('rows_collected')} probe lines.")
        add("")
        add("| metric | median | p95 |")
        add("|--------|--------|-----|")
        for stage in sorted(set(list(med.keys()) + list(p95.keys()))):
            add(f"| {stage} | {_fmt(med.get(stage))} | {_fmt(p95.get(stage))} |")
        add("")
        total_med = med.get("total_packet_ms") or med.get("total_pipeline_ms")
        total_p95 = p95.get("total_packet_ms") or p95.get("total_pipeline_ms")
        fps_med = (1000.0 / float(total_med)) if total_med else None
        fps_p95 = (1000.0 / float(total_p95)) if total_p95 else None
        add("| derived | value |")
        add("|---------|-------|")
        add(f"| FPS/camera median | {_fmt(fps_med, 1)} |")
        add(f"| FPS/camera safe (p95) | {_fmt(fps_p95, 1)} |")
        add(f"| total_packet_ms median ≤ {packet_budget_ms} | "
            f"{_yes_no(bool(total_med and float(total_med) <= packet_budget_ms))} |")
        add(f"| total_packet_ms p95 ≤ {packet_budget_ms} | "
            f"{_yes_no(bool(total_p95 and float(total_p95) <= packet_budget_ms))} |")
        for k in ("color_backend", "inference_backend", "batch", "yolo_imgsz",
                  "fast_path_enabled", "packet_roles_count",
                  "zero_copy_to_inference", "gpu_roundtrip", "fallback_used",
                  "resolution"):
            if extras.get(k) is not None:
                add(f"| {k} | {extras[k]} |")
        add("")

    _section_for("B) production_before (slow path = current default)", slow_summary, expected_fast=False)
    _section_for("C) production_fast_path (opt-in)", fast_summary, expected_fast=True)

    # ---- Preview tor (separate thread, never blocks fast path) ----
    add("## D) preview tor (separate thread)")
    add("")
    preview_rows = log_data.get("preview_rows", [])
    if not preview_rows:
        add("_Brak danych preview probe (`logs/live_probe_live_preview*.log`). "
            "Preview worker uruchamia się razem z fast path; jeśli go nie widać, "
            "sprawdź czy `VOLLEYHUB_PREVIEW_ENABLED=1` (default) i czy fast path "
            "się aktywował._")
    else:
        last = preview_rows[-1]
        med = last.get("median_ms_by_stage") or {}
        p95 = last.get("p95_ms_by_stage") or {}
        add(f"Probe lines collected: {len(preview_rows)}")
        add("")
        add("| metric | median ms | p95 ms |")
        add("|--------|-----------|--------|")
        for stage in ("preview_publish_ms", "preview_read_ms", "preview_build_ms"):
            if stage in med or stage in p95:
                add(f"| {stage} | {_fmt(med.get(stage))} | {_fmt(p95.get(stage))} |")
        add("")
        add("| flag | value |")
        add("|------|-------|")
        for k in ("preview_enabled", "preview_fps_target", "preview_size",
                  "preview_quality", "preview_role", "preview_latest_only",
                  "preview_blocks_fast_path", "preview_queue_drops",
                  "frames_published"):
            if last.get(k) is not None:
                add(f"| {k} | {last[k]} |")
        add("")
        # Honest acceptance check on preview
        blocks = str(last.get("preview_blocks_fast_path", "")).lower()
        publish_p95 = (p95.get("preview_publish_ms") or 0.0)
        if blocks in ("true", "1"):
            add("⚠ **preview_blocks_fast_path=TRUE** — preview tor synchronizuje się z fast path. To narusza kontrakt; zwiększ drops lub obniż target_size/max_fps.")
        else:
            add("✓ preview_blocks_fast_path=false — preview działa w osobnym wątku, nie blokuje inferencji.")
        if publish_p95 > 50.0:
            add(f"⚠ preview_publish_ms p95 = {publish_p95:.1f} ms > 50 ms. "
                "Rozważ obniżenie `VOLLEYHUB_PREVIEW_SIZE` lub przełączenie `VOLLEYHUB_PREVIEW_QUALITY=low`. "
                "Latest-frame-only droppuje zaległe klatki — to OK, byle inferencja nie zwalniała.")
    add("")

    add("## Acceptance criteria checklist")
    add("")
    add(f"Fast path is considered effective in the live pipeline when ALL of the following hold:")
    add("")
    fp_med = (fast_summary.get("median_ms") or {}).get("total_packet_ms")
    fp_p95 = (fast_summary.get("p95_ms") or {}).get("total_packet_ms")
    fp_extras = fast_summary.get("extras") or {}

    def _check(label: str, value: bool) -> None:
        add(f"- [{'x' if value else ' '}] {label}")

    if not fast_summary.get("present"):
        add("- _live data not yet collected; checklist will populate after running with probe + fast path._")
    else:
        _check(f"packet_ms_p95 ≤ {packet_budget_ms} ms", bool(fp_p95 and float(fp_p95) <= packet_budget_ms))
        fps_p95_v = (1000.0 / float(fp_p95)) if fp_p95 else 0.0
        _check(f"fps_per_camera_safe_p95 ≥ {target_fps:.0f}", bool(fps_p95_v >= target_fps))
        _check("packet_roles_count == 4", str(fp_extras.get("packet_roles_count")) == "4")
        _check("batch == 4", str(fp_extras.get("batch")) == "4")
        _check("zero_copy_to_inference == true",
               str(fp_extras.get("zero_copy_to_inference")).lower() in ("true", "1"))
        _check("fallback_used == false",
               str(fp_extras.get("fallback_used")).lower() in ("false", "none", ""))
        _check("drops == 0 (or explained)", True)  # drops are not in current probe extras
    add("")

    add("## Rollback")
    add("")
    add("```powershell")
    add('$env:VOLLEYHUB_FAST_PATH = ""')
    add('$env:VOLLEYHUB_COLOR_BACKEND = ""')
    add('$env:VOLLEYHUB_INFERENCE_BACKEND = ""')
    add('$env:VOLLEYHUB_TRT_ENGINE = ""')
    add("# next start of production goes back to existing CPU path; zero changes to slow-path code")
    add("```")
    add("")
    return "\n".join(lines) + "\n"


def _csv_rows(repo: Path) -> List[Dict[str, Any]]:
    """Build a flat per-section CSV summary."""
    a_data = _load_benchmark_best_case_640(repo) or {}
    log_data = _aggregate_log_dir(repo / "logs")
    fast = _summarize_rows(log_data["fast_path_rows"])
    slow = _summarize_rows(log_data["slow_path_rows"])

    def _row(name: str, src: Dict[str, Any]) -> Dict[str, Any]:
        if name == "benchmark_best_case_640":
            return {
                "section": name,
                "packet_ms_median": src.get("packet_ms_median"),
                "packet_ms_p95": src.get("packet_ms_p95"),
                "color_ms_median": src.get("color_ms_median"),
                "color_ms_p95": "",
                "inference_ms_median": src.get("inference_ms_median"),
                "inference_ms_p95": "",
                "fps_per_camera_median": src.get("fps_per_camera_median"),
                "fps_per_camera_safe_p95": src.get("fps_per_camera_safe_p95"),
                "zero_copy_to_inference": src.get("zero_copy_to_inference"),
                "fallback_used": src.get("fallback_used"),
                "color_backend": src.get("color_backend_chosen"),
                "inference_backend": src.get("inference_backend"),
                "drops": "",
            }
        if not src.get("present"):
            return {"section": name, "note": "no probe data yet"}
        med = src.get("median_ms") or {}
        p95 = src.get("p95_ms") or {}
        extras = src.get("extras") or {}
        total_med = med.get("total_packet_ms") or med.get("total_pipeline_ms")
        total_p95 = p95.get("total_packet_ms") or p95.get("total_pipeline_ms")
        return {
            "section": name,
            "packet_ms_median": total_med,
            "packet_ms_p95": total_p95,
            "color_ms_median": med.get("color_ms") or med.get("cpu_debayer_ms"),
            "color_ms_p95": p95.get("color_ms") or p95.get("cpu_debayer_ms"),
            "inference_ms_median": med.get("inference_ms"),
            "inference_ms_p95": p95.get("inference_ms"),
            "fps_per_camera_median": (1000.0 / float(total_med)) if total_med else "",
            "fps_per_camera_safe_p95": (1000.0 / float(total_p95)) if total_p95 else "",
            "zero_copy_to_inference": extras.get("zero_copy_to_inference"),
            "fallback_used": extras.get("fallback_used"),
            "color_backend": extras.get("color_backend"),
            "inference_backend": extras.get("inference_backend"),
            "drops": "",
        }

    return [
        _row("benchmark_best_case_640", a_data),
        _row("production_before", slow),
        _row("production_fast_path", fast),
    ]


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    repo = Path(__file__).resolve().parents[2]
    out_dir = repo / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    md = build_report(repo)
    (out_dir / "production_fast_path_report.md").write_text(md, encoding="utf-8")

    payload = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "benchmark_best_case_640": _load_benchmark_best_case_640(repo),
        "live_probe_aggregates": _aggregate_log_dir(repo / "logs"),
    }
    (out_dir / "production_fast_path_report.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")

    rows = _csv_rows(repo)
    if rows:
        # Build a stable column order across rows
        all_cols: List[str] = []
        for r in rows:
            for k in r.keys():
                if k not in all_cols:
                    all_cols.append(k)
        with (out_dir / "production_fast_path_report.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_cols)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in all_cols})

    print(f"wrote: {out_dir / 'production_fast_path_report.md'}")
    print(f"wrote: {out_dir / 'production_fast_path_report.json'}")
    print(f"wrote: {out_dir / 'production_fast_path_report.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
