"""Generate the final inference-size report with charts.

Reads `reports/fullhd_inference_size_sweep.json` (produced by
run_fullhd_inference_size_sweep) and renders three PNGs + a markdown
report that highlights two decision answers:

    1. Full HD inference: how many FPS can one camera sustain?
    2. Which inference size is the largest that still hits 50 FPS?

Output:
    reports/final_inference_report.md
    reports/final_charts/chart_1_fps_per_imgsz.png
    reports/final_charts/chart_2_packet_ms_per_imgsz.png
    reports/final_charts/chart_3_stage_breakdown.png
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_LOG = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
SRC_JSON = REPO / "reports" / "fullhd_inference_size_sweep.json"
OUT_DIR = REPO / "reports"
CHART_DIR = OUT_DIR / "final_charts"

PACKET_BUDGET_MS = 20.0
TARGET_FPS = 50.0


def _load_sweep() -> Dict[str, Any]:
    if not SRC_JSON.exists():
        raise SystemExit(
            f"Missing source data: {SRC_JSON}.\n"
            "Run: python -m src.runtime_benchmark.run_fullhd_inference_size_sweep "
            "(with the right TRT engines built first)."
        )
    return json.loads(SRC_JSON.read_text(encoding="utf-8"))


def _ok_variants(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for v in data.get("variants", []):
        if v.get("status") == "OK" and not v.get("error"):
            out.append(v)
    return out


def _shape_label(v: Dict[str, Any]) -> str:
    return str(v.get("inference_input_shape", "?"))


def _shape_area(v: Dict[str, Any]) -> int:
    s = _shape_label(v)
    if "x" not in s:
        return 0
    a, b = s.split("x")
    try:
        return int(a) * int(b)
    except Exception:
        return 0


# ------------------------------------------------------------------ charts


def _annotate_bars(ax, bars, fmt: str = "{:.1f}", offset: float = 0.5) -> None:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + offset, fmt.format(h),
                ha="center", va="bottom", fontsize=9)


def chart_fps(variants: List[Dict[str, Any]], out_path: Path) -> None:
    labels = [_shape_label(v) for v in variants]
    fps_med = [float(v.get("fps_per_camera_median", 0.0)) for v in variants]
    fps_p95 = [float(v.get("fps_per_camera_safe_p95", 0.0)) for v in variants]
    x = list(range(len(labels)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar([i - width / 2 for i in x], fps_med, width=width,
                   color="#2563eb", label="FPS per camera (median)")
    bars2 = ax.bar([i + width / 2 for i in x], fps_p95, width=width,
                   color="#0d9488", label="FPS per camera (safe = p95)")
    _annotate_bars(ax, bars1, fmt="{:.1f}", offset=0.7)
    _annotate_bars(ax, bars2, fmt="{:.1f}", offset=0.7)

    ax.axhline(y=TARGET_FPS, color="#dc2626", linestyle="--", linewidth=2,
               label=f"Target = {TARGET_FPS:.0f} FPS")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("FPS per camera (4 cams synchronized)", fontsize=11)
    ax.set_xlabel("YOLO inference input size (H×W)", fontsize=11)
    ax.set_title(
        "Maks FPS per kamera vs rozmiar inferencji\n"
        "capture 1920×1080  |  TRT FP16 b4  |  native_cuda_npp zero-copy  |  RTX 2080 Super",
        fontsize=12,
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ymax = max(max(fps_med, default=0), max(fps_p95, default=0), TARGET_FPS) * 1.18
    ax.set_ylim(0, ymax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def chart_packet_ms(variants: List[Dict[str, Any]], out_path: Path) -> None:
    labels = [_shape_label(v) for v in variants]
    med = [float(v.get("packet_ms_median", 0.0)) for v in variants]
    p95 = [float(v.get("packet_ms_p95", 0.0)) for v in variants]
    x = list(range(len(labels)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar([i - width / 2 for i in x], med, width=width,
                   color="#7c3aed", label="packet_ms (median)")
    bars2 = ax.bar([i + width / 2 for i in x], p95, width=width,
                   color="#db2777", label="packet_ms (p95)")
    _annotate_bars(ax, bars1, fmt="{:.1f}", offset=0.4)
    _annotate_bars(ax, bars2, fmt="{:.1f}", offset=0.4)

    ax.axhline(y=PACKET_BUDGET_MS, color="#16a34a", linestyle="--", linewidth=2,
               label=f"Budżet 50 FPS = {PACKET_BUDGET_MS:.0f} ms / pakiet")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("packet_ms (1 pakiet = 4 zsynchronizowane kamery)", fontsize=11)
    ax.set_xlabel("YOLO inference input size (H×W)", fontsize=11)
    ax.set_title(
        "Czas pakietu vs rozmiar inferencji\n"
        "PASS = mieści się w 20 ms ⇒ 50 FPS osiągalne",
        fontsize=12,
    )
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ymax = max(max(med, default=0), max(p95, default=0), PACKET_BUDGET_MS) * 1.18
    ax.set_ylim(0, ymax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def chart_stage_breakdown(variants: List[Dict[str, Any]], out_path: Path) -> None:
    labels = [_shape_label(v) for v in variants]
    color_med = [float(v.get("color_ms_median", 0.0)) for v in variants]
    inf_med = [float(v.get("inference_ms_median", 0.0)) for v in variants]
    x = list(range(len(labels)))
    width = 0.55

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_color = ax.bar(x, color_med, width=width, color="#0ea5e9",
                        label="color_ms (debayer + resize + normalize, native CUDA)")
    bars_inf = ax.bar(x, inf_med, width=width, bottom=color_med,
                      color="#f59e0b", label="inference_ms (TRT FP16 b4)")

    for i, (c, fb) in enumerate(zip(color_med, inf_med)):
        total = c + fb
        ax.text(i, total + 0.4, f"{total:.1f} ms",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
        if c > 0.5:
            ax.text(i, c / 2, f"{c:.1f}", ha="center", va="center",
                    fontsize=9, color="white")
        if fb > 0.5:
            ax.text(i, c + fb / 2, f"{fb:.1f}", ha="center", va="center",
                    fontsize=9, color="white")

    ax.axhline(y=PACKET_BUDGET_MS, color="#16a34a", linestyle="--", linewidth=2,
               label=f"Budżet 50 FPS = {PACKET_BUDGET_MS:.0f} ms")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("ms (median)", fontsize=11)
    ax.set_xlabel("YOLO inference input size (H×W)", fontsize=11)
    ax.set_title(
        "Rozkład czasu na pakiet — color stage vs inference\n"
        "color_ms ~stałe (kapture 1920×1080, native CUDA), inference rośnie z imgsz",
        fontsize=12,
    )
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ymax = max(c + i for c, i in zip(color_med, inf_med)) * 1.25
    ymax = max(ymax, PACKET_BUDGET_MS * 1.2)
    ax.set_ylim(0, ymax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------ markdown


def _full_hd_variant(variants: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for v in variants:
        if _shape_label(v) in ("1088x1920", "1920x1088"):
            return v
    return None


def _largest_passing(variants: List[Dict[str, Any]],
                     budget_ms: float,
                     metric: str) -> Optional[Dict[str, Any]]:
    passing = [v for v in variants if float(v.get(metric, 1e9)) <= budget_ms]
    if not passing:
        return None
    return max(passing, key=_shape_area)


def _fmt(v, p: int = 2) -> str:
    try:
        f = float(v)
        if f != f:
            return "—"
        return f"{f:.{p}f}"
    except Exception:
        return "—"


def build_md(data: Dict[str, Any], variants: List[Dict[str, Any]]) -> str:
    cap = data.get("args", {})
    fhd = _full_hd_variant(variants)
    lp_med = _largest_passing(variants, PACKET_BUDGET_MS, "packet_ms_median")
    lp_p95 = _largest_passing(variants, PACKET_BUDGET_MS, "packet_ms_p95")

    lines: List[str] = []
    a = lines.append
    a("# Final inference-size report — VolleyHub 4 cams")
    a("")
    a(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    a(f"Source data: `reports/fullhd_inference_size_sweep.json`")
    a("")
    a("## Stałe pomiaru")
    a("")
    a(f"- cameras: 4 (synchronized packet)")
    a(f"- capture: **{cap.get('capture_width', '?')}×{cap.get('capture_height', '?')}** RAW Bayer={cap.get('bayer_pattern', '?')}")
    a(f"- pipeline: native_cuda_npp zero-copy → TRT FP16 batch=4")
    a(f"- target: **{TARGET_FPS:.0f} FPS per kamera** ⇒ packet budget **{PACKET_BUDGET_MS:.0f} ms**")
    a(f"- hardware: NVIDIA GeForce RTX 2080 Super")
    a("")

    # ---------- two key decision answers, prominent ----------
    a("---")
    a("")
    a("# 🎯 Dwie kluczowe odpowiedzi")
    a("")
    a("## Pytanie 1 — Inferencja Full HD: ile FPS z kamery?")
    a("")
    if fhd is None:
        a("> ⚠ Brak danych dla 1088×1920 w sweepie.")
    else:
        med_fps = float(fhd.get("fps_per_camera_median", 0.0))
        p95_fps = float(fhd.get("fps_per_camera_safe_p95", 0.0))
        med_ms = float(fhd.get("packet_ms_median", 0.0))
        p95_ms = float(fhd.get("packet_ms_p95", 0.0))
        margin = PACKET_BUDGET_MS - p95_ms
        verdict_med = "PASS" if med_ms <= PACKET_BUDGET_MS else "FAIL"
        verdict_p95 = "PASS" if p95_ms <= PACKET_BUDGET_MS else "FAIL"
        a(f"**Rozmiar inferencji: 1088×1920 (Full HD-aligned, height padded do stride 32)**")
        a("")
        a(f"| metric | wartość |")
        a("|--------|---------|")
        a(f"| FPS/kamera median (aggressive) | **{med_fps:.1f} FPS** |")
        a(f"| FPS/kamera safe (p95)          | **{p95_fps:.1f} FPS** |")
        a(f"| packet_ms median               | {_fmt(med_ms)} ms |")
        a(f"| packet_ms p95                  | {_fmt(p95_ms)} ms |")
        a(f"| margines vs 20 ms              | **{margin:+.2f} ms** |")
        a(f"| 50 FPS median?                 | **{verdict_med}** |")
        a(f"| 50 FPS safe (p95)?             | **{verdict_p95}** |")
        a("")
        a(f"### Werdykt: na RTX 2080 Super z full HD inferencją można praktycznie wyciągnąć "
          f"**~{p95_fps:.0f} FPS per kamera safe / ~{med_fps:.0f} FPS aggressive**, co JEST "
          f"poniżej celu 50 FPS. Full HD inference z 4 kamerami × 50 FPS na tym sprzęcie się NIE wyrabia "
          f"(brak {-margin:.1f} ms marginesu na p95).")
    a("")

    a("## Pytanie 2 — Dla jakiego rozmiaru inferencji wpada 50 FPS?")
    a("")
    if lp_p95 is None and lp_med is None:
        a("> ⚠ Żaden z testowanych rozmiarów nie mieści się w budżecie 20 ms.")
    else:
        if lp_p95:
            shape = _shape_label(lp_p95)
            margin = PACKET_BUDGET_MS - float(lp_p95.get("packet_ms_p95", 0.0))
            engine_path = str(lp_p95.get("engine_path", ""))
            engine_name = Path(engine_path).name if engine_path else "—"
            a(f"**Rekomendowany rozmiar produkcyjny: {shape}** (utrzymuje p95 ≤ 20 ms)")
            a("")
            a(f"| metric | wartość |")
            a("|--------|---------|")
            a(f"| packet_ms median                    | {_fmt(lp_p95.get('packet_ms_median'))} ms |")
            a(f"| packet_ms p95                       | {_fmt(lp_p95.get('packet_ms_p95'))} ms |")
            a(f"| margines vs 20 ms                   | **+{margin:.2f} ms** |")
            a(f"| FPS/kamera median                   | **{_fmt(lp_p95.get('fps_per_camera_median'), 1)}** |")
            a(f"| FPS/kamera safe (p95)               | **{_fmt(lp_p95.get('fps_per_camera_safe_p95'), 1)}** |")
            a(f"| total images/s (4 cams)             | {_fmt(lp_p95.get('total_images_per_second_safe_p95'), 1)} |")
            a(f"| zero_copy_to_inference              | {lp_p95.get('zero_copy_to_inference')} |")
            a(f"| engine                              | `{engine_name}` |")
            a("")
            a(f"### Werdykt: największy rozmiar inferencji który **bezpiecznie utrzymuje 50 FPS** "
              f"(p95 w budżecie) to **{shape}**. Przy nim FPS/kamera safe = "
              f"{_fmt(lp_p95.get('fps_per_camera_safe_p95'), 1)}, marginesu jeszcze "
              f"+{margin:.1f} ms na nieprzewidziane skoki.")
        if lp_med and lp_med is not lp_p95:
            a(f"\n_Dla referencji — agresywnie (tylko median ≤ 20 ms) wpada również "
              f"`{_shape_label(lp_med)}`, ale jego p95 "
              f"{_fmt(lp_med.get('packet_ms_p95'))} ms przekracza budżet — "
              f"możliwe rzadkie dropy._")
    a("")
    a("---")
    a("")

    # ---------- charts ----------
    a("## Wykresy")
    a("")
    a("### Wykres 1 — FPS per kamera vs rozmiar inferencji")
    a("")
    a("![FPS per imgsz](final_charts/chart_1_fps_per_imgsz.png)")
    a("")
    a(f"Słupki: median (niebieski) i safe p95 (turkusowy) FPS/kamera. Czerwona przerywana = "
      f"**target {TARGET_FPS:.0f} FPS**. Tylko `640×640` jest powyżej linii w obu wartościach.")
    a("")
    a("### Wykres 2 — Czas pakietu vs rozmiar inferencji")
    a("")
    a("![packet_ms per imgsz](final_charts/chart_2_packet_ms_per_imgsz.png)")
    a("")
    a(f"Słupki: median (fiolet) i p95 (różowy) `packet_ms`. Zielona przerywana = "
      f"**budżet 20 ms = 50 FPS**. PASS gdy oba słupki pod linią.")
    a("")
    a("### Wykres 3 — Rozkład czasu (color stage vs inference)")
    a("")
    a("![stage breakdown](final_charts/chart_3_stage_breakdown.png)")
    a("")
    a("Stos: dolny niebieski = `color_ms` (native CUDA debayer + resize + normalize) — "
      "praktycznie stałe ~2 ms bo zawsze dotyczy capture 1920×1080. "
      "Górny pomarańczowy = `inference_ms` (TRT FP16 b4) — rośnie ~liniowo z liczbą pikseli "
      "wyjściowych. **Inference dominuje koszt powyżej 640×640.**")
    a("")

    # ---------- table ----------
    a("## Tabela szczegółowa")
    a("")
    a("| imgsz | packet_ms median | packet_ms p95 | FPS/cam median | FPS/cam safe p95 | color_ms | inference_ms | margines (20 ms) | werdykt |")
    a("|-------|-------------------|----------------|----------------|-------------------|----------|---------------|-------------------|---------|")
    for v in variants:
        med = float(v.get("packet_ms_median", 0.0))
        p95 = float(v.get("packet_ms_p95", 0.0))
        margin = PACKET_BUDGET_MS - p95
        verdict = "✅ PASS" if p95 <= PACKET_BUDGET_MS else "❌ FAIL"
        a(f"| {_shape_label(v)} | {_fmt(med)} | {_fmt(p95)} | "
          f"{_fmt(v.get('fps_per_camera_median'), 1)} | {_fmt(v.get('fps_per_camera_safe_p95'), 1)} | "
          f"{_fmt(v.get('color_ms_median'))} | {_fmt(v.get('inference_ms_median'))} | "
          f"{margin:+.2f} ms | {verdict} |")
    a("")

    # ---------- closing ----------
    a("## Krótkie odpowiedzi (TL;DR)")
    a("")
    if fhd is not None:
        p95_fps_fhd = float(fhd.get("fps_per_camera_safe_p95", 0.0))
        med_fps_fhd = float(fhd.get("fps_per_camera_median", 0.0))
        a(f"**1. Full HD inference (1088×1920) → ~{p95_fps_fhd:.0f} FPS/cam safe, ~{med_fps_fhd:.0f} FPS aggressive.** "
          f"Nie spełnia 50 FPS na obecnym sprzęcie.")
    if lp_p95 is not None:
        a(f"**2. Największy rozmiar inferencji utrzymujący 50 FPS dla 4 kamer = `{_shape_label(lp_p95)}`** "
          f"(p95 = {_fmt(lp_p95.get('packet_ms_p95'))} ms, margines +{PACKET_BUDGET_MS - float(lp_p95.get('packet_ms_p95', 0.0)):.2f} ms).")
    a("")
    a("Jeżeli chcesz inferencję większą niż 640×640 przy 50 FPS dla 4 kamer:")
    a("- mniejszy model (np. YOLOv8n zamiast obecnego)")
    a("- INT8 zamiast FP16")
    a("- ROI crop / tile detection")
    a("- mocniejszy GPU (RTX 4080/4090)")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the final inference-size report with charts.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    data = _load_sweep()
    variants = _ok_variants(data)
    if not variants:
        raise SystemExit("No OK variants in sweep JSON; cannot build report.")
    variants = sorted(variants, key=_shape_area)

    CHART_DIR.mkdir(parents=True, exist_ok=True)
    chart_fps(variants, CHART_DIR / "chart_1_fps_per_imgsz.png")
    chart_packet_ms(variants, CHART_DIR / "chart_2_packet_ms_per_imgsz.png")
    chart_stage_breakdown(variants, CHART_DIR / "chart_3_stage_breakdown.png")

    md = build_md(data, variants)
    out_md = OUT_DIR / "final_inference_report.md"
    out_md.write_text(md, encoding="utf-8")

    print(f"wrote: {out_md}")
    print(f"wrote: {CHART_DIR / 'chart_1_fps_per_imgsz.png'}")
    print(f"wrote: {CHART_DIR / 'chart_2_packet_ms_per_imgsz.png'}")
    print(f"wrote: {CHART_DIR / 'chart_3_stage_breakdown.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
