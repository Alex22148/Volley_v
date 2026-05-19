"""Build the final HTML + Markdown report with charts from results.json."""
from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import matplotlib  # type: ignore
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore

PACKET_BUDGET_MS = 20.0
TARGET_FPS = 50.0


# ----------------------------------------------------------- helpers

def _f(v, p: int = 2) -> str:
    if v is None:
        return "—"
    try:
        x = float(v)
        if x != x:
            return "—"
        return f"{x:.{p}f}"
    except Exception:
        return str(v)


def _shape_area(label: str) -> int:
    if "x" not in label:
        return 0
    a, b = label.split("x", 1)
    try:
        return int(a) * int(b)
    except Exception:
        return 0


def _ok_rows(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in results
            if r.get("status") == "PASS"
            and r.get("total_packet_ms_p95") is not None]


def _full_hd_row(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for r in rows:
        if r.get("inference_input_shape") in ("1088x1920", "1920x1088"):
            return r
    return None


def _largest_passing(rows: List[Dict[str, Any]], metric: str) -> Optional[Dict[str, Any]]:
    passing = [r for r in rows
               if r.get(metric) is not None
               and float(r.get(metric, 1e9)) <= PACKET_BUDGET_MS]
    if not passing:
        return None
    return max(passing, key=lambda r: _shape_area(str(r.get("inference_input_shape", ""))))


def _bottleneck(row: Dict[str, Any]) -> str:
    stages = {
        "color": float(row.get("color_ms_median") or 0.0),
        "inference": float(row.get("inference_ms_median") or 0.0),
        "postprocess": float(row.get("postprocess_ms_median") or 0.0),
        "stack": float(row.get("stack_ms_median") or 0.0),
        "sync_wait": float(row.get("sync_wait_ms_median") or 0.0),
        "resize_or_letterbox": float(row.get("resize_or_letterbox_ms_median") or 0.0),
    }
    if not any(stages.values()):
        return "—"
    return max(stages, key=stages.get)


def _label_color(passes: bool) -> str:
    return "#16a34a" if passes else "#dc2626"


# ----------------------------------------------------------- charts

def _annotate(ax, bars, fmt: str = "{:.1f}", offset: float = 0.5):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + offset, fmt.format(h),
                ha="center", va="bottom", fontsize=9)


def chart_fps(rows: List[Dict[str, Any]], out_path: Path) -> None:
    rows = sorted(rows, key=lambda r: _shape_area(str(r.get("inference_input_shape", ""))))
    labels = [str(r.get("inference_input_shape", "?")) for r in rows]
    fps_med = [float(r.get("fps_per_camera_median") or 0.0) for r in rows]
    fps_p95 = [float(r.get("fps_per_camera_safe_p95") or 0.0) for r in rows]
    x = list(range(len(labels)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar([i - width / 2 for i in x], fps_med, width=width,
                color="#2563eb", label="FPS per camera (median)")
    b2 = ax.bar([i + width / 2 for i in x], fps_p95, width=width,
                color="#0d9488", label="FPS per camera (safe = p95)")
    _annotate(ax, b1, fmt="{:.1f}", offset=0.7)
    _annotate(ax, b2, fmt="{:.1f}", offset=0.7)
    ax.axhline(y=TARGET_FPS, color="#dc2626", linestyle="--", linewidth=2,
               label=f"Target = {TARGET_FPS:.0f} FPS")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("FPS per camera (4 cams synchronized)", fontsize=11)
    ax.set_xlabel("YOLO inference input size (H×W)", fontsize=11)
    ax.set_title("FPS per camera vs inference resolution", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ymax = max(max(fps_med, default=0), max(fps_p95, default=0), TARGET_FPS) * 1.18
    ax.set_ylim(0, ymax)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


def chart_packet_ms(rows: List[Dict[str, Any]], out_path: Path) -> None:
    rows = sorted(rows, key=lambda r: _shape_area(str(r.get("inference_input_shape", ""))))
    labels = [str(r.get("inference_input_shape", "?")) for r in rows]
    med = [float(r.get("total_packet_ms_median") or 0.0) for r in rows]
    p95 = [float(r.get("total_packet_ms_p95") or 0.0) for r in rows]
    x = list(range(len(labels)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar([i - width / 2 for i in x], med, width=width,
                color="#7c3aed", label="packet_ms (median)")
    b2 = ax.bar([i + width / 2 for i in x], p95, width=width,
                color="#db2777", label="packet_ms (p95)")
    _annotate(ax, b1); _annotate(ax, b2)
    ax.axhline(y=PACKET_BUDGET_MS, color="#16a34a", linestyle="--", linewidth=2,
               label=f"Budget 50 FPS = {PACKET_BUDGET_MS:.0f} ms / packet")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("packet_ms (4 synced cameras)", fontsize=11)
    ax.set_xlabel("YOLO inference input size (H×W)", fontsize=11)
    ax.set_title("Packet latency vs inference resolution", fontsize=12)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ymax = max(max(med, default=0), max(p95, default=0), PACKET_BUDGET_MS) * 1.18
    ax.set_ylim(0, ymax)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


def chart_pass_fail(rows: List[Dict[str, Any]], out_path: Path) -> None:
    rows = sorted(rows, key=lambda r: _shape_area(str(r.get("inference_input_shape", ""))))
    labels = [str(r.get("inference_input_shape", "?")) for r in rows]
    p95 = [float(r.get("total_packet_ms_p95") or 0.0) for r in rows]
    colors = [_label_color(v <= PACKET_BUDGET_MS and v > 0) for v in p95]
    x = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(x, p95, color=colors, width=0.6)
    for b, v in zip(bars, p95):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}",
                ha="center", va="bottom", fontsize=10)
    ax.axhline(y=PACKET_BUDGET_MS, color="#0ea5e9", linestyle="--", linewidth=2,
               label=f"Budget 50 FPS = {PACKET_BUDGET_MS:.0f} ms")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("packet_ms p95 (ms)", fontsize=11)
    ax.set_title("PASS / FAIL 50 FPS — green = PASS p95, red = FAIL", fontsize=12)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


def chart_stages(rows: List[Dict[str, Any]], out_path: Path) -> None:
    rows = sorted(rows, key=lambda r: _shape_area(str(r.get("inference_input_shape", ""))))
    labels = [str(r.get("inference_input_shape", "?")) for r in rows]
    stages = ["color_ms_median", "stack_ms_median", "sync_wait_ms_median",
              "inference_ms_median", "postprocess_ms_median"]
    colors = ["#0ea5e9", "#10b981", "#a855f7", "#f59e0b", "#ef4444"]
    legend = ["color (debayer+resize+norm)", "stack", "sync_wait", "inference (TRT)", "postprocess"]
    bottoms = [0.0] * len(rows)
    fig, ax = plt.subplots(figsize=(10, 6))
    for stage, color, name in zip(stages, colors, legend):
        vals = [float(r.get(stage) or 0.0) for r in rows]
        ax.bar(labels, vals, bottom=bottoms, color=color, label=name, width=0.55)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    for i, total in enumerate(bottoms):
        ax.text(i, total + 0.4, f"{total:.1f} ms", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.axhline(y=PACKET_BUDGET_MS, color="#16a34a", linestyle="--", linewidth=2,
               label=f"Budget 50 FPS = {PACKET_BUDGET_MS:.0f} ms")
    ax.set_ylabel("median ms")
    ax.set_xlabel("YOLO inference input size (H×W)")
    ax.set_title("Stage breakdown per inference resolution", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


def chart_baseline(rows: List[Dict[str, Any]], baseline: dict, out_path: Path) -> None:
    rows = sorted(rows, key=lambda r: _shape_area(str(r.get("inference_input_shape", ""))))
    labels = [str(r.get("inference_input_shape", "?")) for r in rows]
    cur_p95 = [float(r.get("total_packet_ms_p95") or 0.0) for r in rows]
    base_p95 = []
    for label in labels:
        match = next((b for b in baseline.get("results", [])
                      if str(b.get("name")) == label
                      or str(b.get("inference_input_shape")) == label), None)
        base_p95.append(float(match.get("packet_ms_p95")) if match else 0.0)

    x = list(range(len(labels))); width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar([i - width / 2 for i in x], cur_p95, width=width,
                color="#2563eb", label="this machine (p95)")
    b2 = ax.bar([i + width / 2 for i in x], base_p95, width=width,
                color="#9ca3af", label=f"baseline {baseline.get('baseline_name', '')} (p95)")
    _annotate(ax, b1); _annotate(ax, b2)
    ax.axhline(y=PACKET_BUDGET_MS, color="#16a34a", linestyle="--", linewidth=2,
               label=f"Budget = {PACKET_BUDGET_MS:.0f} ms")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("packet_ms p95")
    ax.set_xlabel("YOLO inference input size (H×W)")
    ax.set_title("This machine vs baseline (lower is better)", fontsize=12)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


def chart_pixels_vs_inference(rows: List[Dict[str, Any]], out_path: Path) -> None:
    rows = sorted(rows, key=lambda r: float(r.get("inference_pixels_mpx") or 0.0))
    px = [float(r.get("inference_pixels_mpx") or 0.0) for r in rows]
    inf = [float(r.get("inference_ms_median") or 0.0) for r in rows]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(px, inf, marker="o", linewidth=2, color="#f59e0b", label="inference_ms median")
    for xi, yi, lbl in zip(px, inf, [str(r.get("inference_input_shape")) for r in rows]):
        ax.annotate(lbl, (xi, yi), textcoords="offset points", xytext=(7, 6), fontsize=9)
    ax.set_xlabel("inference input pixels (Mpx)")
    ax.set_ylabel("inference_ms (median)")
    ax.set_title("Inference time vs input pixel count", fontsize=12)
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


# ----------------------------------------------------------- markdown

def _build_md(payload: dict, baseline: dict, ok_rows: List[Dict[str, Any]],
              all_rows: List[Dict[str, Any]], chart_dir: Path) -> str:
    fhd = _full_hd_row(ok_rows)
    lp_med = _largest_passing(ok_rows, "total_packet_ms_median")
    lp_p95 = _largest_passing(ok_rows, "total_packet_ms_p95")
    machine = payload.get("machine") or {}
    src_type = payload.get("source_type", "?")

    L: List[str] = []
    a = L.append
    a("# Fast-path benchmark — comparison report")
    a("")
    a(f"Generated: {payload.get('generated_utc', '')}  ")
    a(f"Source: **{src_type}**  |  Machine: `{machine.get('hostname')}`  |  GPU: `{machine.get('gpu_name')}` "
      f"({machine.get('gpu_vram_gb')} GB)  |  CUDA: `{machine.get('cuda_version_from_torch')}`  |  "
      f"torch `{machine.get('torch_version')}`")
    a("")

    # ---- A. Executive summary ----
    a("## A. Executive summary")
    a("")
    if fhd:
        a(f"### Full HD-like inference (1088×1920)")
        a("")
        a(f"- median FPS per camera: **{_f(fhd.get('fps_per_camera_median'), 1)}**")
        a(f"- safe FPS p95 per camera: **{_f(fhd.get('fps_per_camera_safe_p95'), 1)}**")
        a(f"- packet_ms median = {_f(fhd.get('total_packet_ms_median'))} ms / "
          f"p95 = {_f(fhd.get('total_packet_ms_p95'))} ms")
        a(f"- 50 FPS median: **{'PASS' if fhd.get('pass_50fps_median') else 'FAIL'}**, "
          f"safe p95: **{'PASS' if fhd.get('pass_50fps_safe_p95') else 'FAIL'}**")
    else:
        a("### Full HD-like inference (1088×1920) — _no successful run for this variant_")
    a("")

    a("### Largest inference resolution hitting 50 FPS")
    a("")
    if lp_med:
        a(f"- 50 FPS **median** ≤ 20 ms: **{lp_med.get('inference_input_shape')}** "
          f"(median {_f(lp_med.get('total_packet_ms_median'))} ms)")
    else:
        a("- 50 FPS median: **none — no variant fits the budget at median**")
    if lp_p95:
        a(f"- 50 FPS **safe (p95)** ≤ 20 ms: **{lp_p95.get('inference_input_shape')}** "
          f"(p95 {_f(lp_p95.get('total_packet_ms_p95'))} ms, "
          f"margin +{_f(lp_p95.get('stability_margin_ms'))} ms)")
    else:
        a("- 50 FPS safe p95: **none — no variant fits the budget at p95**")
    a("")

    rec = lp_p95 or lp_med
    if rec:
        a(f"**Recommended production preset:** `{rec.get('inference_input_shape')}` "
          f"(p95 = {_f(rec.get('total_packet_ms_p95'))} ms, "
          f"FPS safe = {_f(rec.get('fps_per_camera_safe_p95'), 1)})")
    a("")
    a("### Practical presets")
    a("")
    presets = []
    by_label = {str(r.get("inference_input_shape")): r for r in ok_rows}

    def _preset(label: str, target_label: str) -> str:
        r = by_label.get(target_label)
        if r and r.get("pass_50fps_safe_p95"):
            return f"- **{label}** = `{target_label}` ✅ PASS safe p95"
        if r and r.get("pass_50fps_median"):
            return (f"- **{label}** = `{target_label}` ⚠ PASS median only "
                    f"(p95 {_f(r.get('total_packet_ms_p95'))} ms exceeds budget)")
        if r:
            return f"- **{label}** = `{target_label}` ❌ FAIL ({_f(r.get('fps_per_camera_safe_p95'), 1)} FPS safe)"
        return f"- **{label}** = `{target_label}` — _no data_"

    a(_preset("LIVE_50FPS", "640x640"))
    a(_preset("QUALITY_40FPS", "960x960"))
    a(_preset("FULLHD_30FPS", "1088x1920"))
    a("")
    if rec:
        bn = _bottleneck(rec)
        a(f"**Main bottleneck for the recommended preset (`{rec.get('inference_input_shape')}`): `{bn}`**")
    a("")

    # ---- B. Hardware comparison ----
    a("## B. Hardware comparison vs baseline")
    a("")
    base_name = baseline.get("baseline_name", "baseline")
    a(f"Baseline: **{base_name}**")
    a("")
    a("| inference_input_shape | this p95 (ms) | baseline p95 (ms) | this safe FPS | baseline safe FPS | speedup vs baseline | margin vs 20 ms | 50 FPS |")
    a("|-----------------------|----------------|--------------------|----------------|-------------------|----------------------|------------------|--------|")
    for r in sorted(all_rows, key=lambda x: _shape_area(str(x.get("inference_input_shape", "")))):
        label = str(r.get("inference_input_shape"))
        match = next((b for b in baseline.get("results", []) if b.get("name") == label
                      or str(b.get("inference_input_shape")) == label), None)
        base_p = float(match.get("packet_ms_p95") or 0.0) if match else 0.0
        cur_p = r.get("total_packet_ms_p95")
        cur_p_f = float(cur_p) if cur_p is not None else 0.0
        speedup = (base_p / cur_p_f) if (base_p and cur_p_f) else None
        margin = r.get("stability_margin_ms")
        passes = "✅ PASS" if r.get("pass_50fps_safe_p95") else "❌ FAIL"
        if r.get("status") != "PASS":
            passes = f"_{r.get('status')}_"
        base_safe_fps = float(match.get("fps_per_camera_safe_p95") or 0.0) if match else 0.0
        a(f"| {label} | {_f(cur_p)} | {_f(base_p)} | "
          f"{_f(r.get('fps_per_camera_safe_p95'), 1)} | {_f(base_safe_fps, 1)} | "
          f"{('×' + _f(speedup, 2)) if speedup else '—'} | "
          f"{_f(margin, 2)} | {passes} |")
    a("")

    # ---- C. Resolution sweep ----
    a("## C. Resolution sweep")
    a("")
    a("| imgsz | pixels (Mpx) | packet_ms median | packet_ms p95 | FPS/cam median | FPS/cam safe p95 | total imgs/s safe | PASS median | PASS p95 | margin |")
    a("|-------|---------------|-------------------|----------------|-----------------|-------------------|---------------------|--------------|----------|--------|")
    for r in sorted(all_rows, key=lambda x: _shape_area(str(x.get("inference_input_shape", "")))):
        a(f"| {r.get('inference_input_shape')} | {_f(r.get('inference_pixels_mpx'), 3)} | "
          f"{_f(r.get('total_packet_ms_median'))} | {_f(r.get('total_packet_ms_p95'))} | "
          f"{_f(r.get('fps_per_camera_median'), 1)} | {_f(r.get('fps_per_camera_safe_p95'), 1)} | "
          f"{_f(r.get('total_images_per_second_safe_p95'), 1)} | "
          f"{'✅' if r.get('pass_50fps_median') else '❌'} | "
          f"{'✅' if r.get('pass_50fps_safe_p95') else '❌'} | "
          f"{_f(r.get('stability_margin_ms'))} ms |")
    a("")

    # ---- D. Stage breakdown ----
    a("## D. Stage breakdown")
    a("")
    a("| imgsz | color | resize | inference | postprocess | stack | sync_wait | total | bottleneck |")
    a("|-------|-------|--------|-----------|-------------|-------|------------|-------|------------|")
    for r in sorted(all_rows, key=lambda x: _shape_area(str(x.get("inference_input_shape", "")))):
        a(f"| {r.get('inference_input_shape')} | {_f(r.get('color_ms_median'))} | "
          f"{_f(r.get('resize_or_letterbox_ms_median'))} | {_f(r.get('inference_ms_median'))} | "
          f"{_f(r.get('postprocess_ms_median'))} | {_f(r.get('stack_ms_median'))} | "
          f"{_f(r.get('sync_wait_ms_median'))} | {_f(r.get('total_packet_ms_median'))} | "
          f"{_bottleneck(r)} |")
    a("")

    # ---- E. Decision answer ----
    a("## E. Decision answer")
    a("")
    a("### 1. Full HD-like inference (1088×1920)")
    if fhd:
        a(f"- ~{_f(fhd.get('fps_per_camera_median'), 1)} FPS median per camera")
        a(f"- ~{_f(fhd.get('fps_per_camera_safe_p95'), 1)} FPS safe p95 per camera")
        a(f"- 50 FPS: **{'PASS' if fhd.get('pass_50fps_safe_p95') else 'FAIL'}**")
    else:
        a("- _no data_")
    a("")
    a("### 2. Stable 50 FPS per camera")
    a(f"- largest imgsz at p95 ≤ 20 ms: **{lp_p95.get('inference_input_shape') if lp_p95 else 'none'}**")
    a(f"- largest imgsz at median ≤ 20 ms: **{lp_med.get('inference_input_shape') if lp_med else 'none'}**")
    a(f"- recommended production: **{(lp_p95 or lp_med or {}).get('inference_input_shape', 'none')}**")
    a("")
    a("### 3. Practical presets")
    a(_preset("LIVE_50FPS", "640x640"))
    a(_preset("QUALITY_40FPS", "960x960"))
    a(_preset("FULLHD_30FPS", "1088x1920"))
    a("")

    # ---- F. Environment ----
    a("## F. Environment diagnostics")
    a("")
    a("| field | value |")
    a("|-------|-------|")
    for k in ("os", "python_version", "cuda_available", "torch_version",
              "cuda_version_from_torch", "tensorrt_available", "gpu_name",
              "gpu_vram_gb", "pypylon_available", "native_backend_available"):
        a(f"| {k} | {machine.get(k)} |")
    a("")

    # ---- G. Errors / warnings ----
    a("## G. Errors / warnings")
    a("")
    any_err = False
    for r in all_rows:
        if r.get("status") != "PASS" or r.get("errors"):
            any_err = True
            a(f"### {r.get('inference_input_shape')} — status: **{r.get('status')}**")
            for e in (r.get("errors") or []):
                a(f"  - {e}")
            if r.get("engine_compatibility_warning"):
                a(f"  - engine_compatibility_warning: {r.get('engine_compatibility_warning')}")
    if not any_err:
        a("_No errors or warnings recorded._")
    a("")

    # ---- charts ----
    a("## Charts")
    a("")
    for fname, caption in [
        ("chart_1_fps.png", "FPS per camera vs inference resolution"),
        ("chart_2_packet_ms.png", "Packet latency median/p95 vs inference resolution"),
        ("chart_3_pass_fail.png", "PASS/FAIL 50 FPS bar chart"),
        ("chart_4_stages.png", "Stage breakdown per resolution"),
        ("chart_5_baseline.png", "Comparison vs baseline"),
        ("chart_6_pixels_vs_inference.png", "Inference time vs input pixels"),
    ]:
        if (chart_dir / fname).exists():
            a(f"### {caption}")
            a("")
            a(f"![{caption}](final_charts/{fname})")
            a("")

    return "\n".join(L) + "\n"


def _build_html(md_body: str, chart_dir: Path) -> str:
    # very simple md→html (preserves headings, tables converted minimally)
    body = html.escape(md_body)
    return ("<!doctype html><html><head><meta charset='utf-8'>"
            "<title>Fast-path benchmark report</title>"
            "<style>body{font-family:sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;}"
            "pre{background:#f3f4f6;padding:8px;}img{max-width:100%;}</style></head><body>"
            f"<pre>{body}</pre></body></html>")


# ----------------------------------------------------------- main

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=PACKAGE_ROOT / "reports" / "results.json")
    parser.add_argument("--baseline", type=Path,
                        default=PACKAGE_ROOT / "configs" / "baseline_rtx2080super.json")
    parser.add_argument("--out-md", type=Path,
                        default=PACKAGE_ROOT / "reports" / "fastpath_comparison_report.md")
    parser.add_argument("--out-html", type=Path,
                        default=PACKAGE_ROOT / "reports" / "fastpath_comparison_report.html")
    parser.add_argument("--out-csv", type=Path,
                        default=PACKAGE_ROOT / "reports" / "results.csv")
    args = parser.parse_args(argv)

    if not args.input.exists():
        raise SystemExit(f"results json not found: {args.input}")
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    baseline = (json.loads(args.baseline.read_text(encoding="utf-8"))
                if args.baseline.exists() else {"baseline_name": "n/a", "results": []})

    rows: List[Dict[str, Any]] = list(payload.get("results", []))
    ok = _ok_rows(rows)

    chart_dir = args.out_md.parent / "final_charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    if ok:
        chart_fps(ok, chart_dir / "chart_1_fps.png")
        chart_packet_ms(ok, chart_dir / "chart_2_packet_ms.png")
        chart_pass_fail(ok, chart_dir / "chart_3_pass_fail.png")
        chart_stages(ok, chart_dir / "chart_4_stages.png")
        chart_baseline(ok, baseline, chart_dir / "chart_5_baseline.png")
        chart_pixels_vs_inference(ok, chart_dir / "chart_6_pixels_vs_inference.png")

    md = _build_md(payload, baseline, ok, rows, chart_dir)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md, encoding="utf-8")
    args.out_html.write_text(_build_html(md, chart_dir), encoding="utf-8")

    # CSV (re-emit for completeness)
    if rows:
        keys: List[str] = []
        for r in rows:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        with args.out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in keys})

    print(f"wrote: {args.out_md}")
    print(f"wrote: {args.out_html}")
    print(f"wrote: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
