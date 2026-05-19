"""09_make_final_report.py - raport wyników final_benchmark_08.

Zachowuje dotychczasowe pliki:
    - final_benchmark_08_report.md
    - final_benchmark_08_summary.csv

Dodaje nowe pliki:
    - final_benchmark_08_parametry_opisowe.json
    - final_benchmark_08_raport_opisowy.md
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PKG_ROOT = Path(__file__).resolve().parent.parent
REPORTS_ROOT = PKG_ROOT / "reports"

TARGET_FPS = 50.0
TARGET_PACKET_MS = 1000.0 / TARGET_FPS


def _resolve_test_results_dir(reports_root: Path) -> Path:
    candidates = sorted(
        [p for p in reports_root.glob("test_???") if p.is_dir() and p.name[5:].isdigit()],
        key=lambda p: int(p.name[5:]),
    )
    if candidates:
        return candidates[-1]
    return reports_root / "test_001"


def _f(v: Any, p: int = 2) -> str:
    try:
        x = float(v)
        if math.isnan(x):
            return "-"
        return f"{x:.{p}f}"
    except Exception:
        return "-"


def _ok_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if not r.get("skipped")]


def _shape(hw: list[int] | tuple[int, int]) -> str:
    return f"{int(hw[0])}x{int(hw[1])}"


def _existing_or_dash(value: Any, p: int = 2) -> str:
    if value is None:
        return "-"
    try:
        v = float(value)
    except Exception:
        return "-"
    if math.isnan(v):
        return "-"
    return f"{v:.{p}f}"


def _get_stage_p95(row: dict[str, Any], *keys: str) -> float | None:
    stage = row.get("aggregate", {}).get("stage_stats", {})
    for key in keys:
        item = stage.get(key)
        if isinstance(item, dict) and item.get("p95_ms") is not None:
            try:
                return float(item["p95_ms"])
            except Exception:
                pass
    return None


def _get_norm_p95(row: dict[str, Any], *keys: str) -> float | None:
    norm = row.get("normalized", {})
    for key in keys:
        item = norm.get(key)
        if isinstance(item, dict):
            if item.get("p95_ms_norm") is not None:
                try:
                    return float(item["p95_ms_norm"])
                except Exception:
                    pass
            if item.get("p95_ms") is not None:
                try:
                    return float(item["p95_ms"])
                except Exception:
                    pass
    return None


def _get_stage_median(row: dict[str, Any], *keys: str) -> float | None:
    stage = row.get("aggregate", {}).get("stage_stats", {})
    for key in keys:
        item = stage.get(key)
        if isinstance(item, dict) and item.get("median_ms") is not None:
            try:
                return float(item["median_ms"])
            except Exception:
                pass
    return None


def _metric_norm(row: dict[str, Any], metric: str) -> float | None:
    """
    Skrót do pobierania p95 norm dla wykresów.
    Obsługuje nowy i stary JSON.
    """
    if metric == "packet_without_nms_ms":
        return _get_norm_p95(row, "packet_without_nms_ms", "czas_paczki_bez_nms_ms", "packet_ms")

    if metric == "packet_with_nms_ms":
        return _get_norm_p95(row, "packet_with_nms_ms", "czas_paczki_z_nms_ms", "packet_ms")

    if metric == "postprocess_nms_ms":
        return _get_norm_p95(row, "postprocess_nms_ms", "czas_postprocessingu_nms_ms")

    if metric == "nms_overhead_ms":
        return _nms_overhead_norm_p95(row)

    return _get_norm_p95(row, metric)


def _metric_stage(row: dict[str, Any], metric: str) -> float | None:
    if metric == "packet_without_nms_ms":
        return _get_stage_p95(row, "packet_without_nms_ms", "czas_paczki_bez_nms_ms", "packet_ms")

    if metric == "packet_with_nms_ms":
        return _get_stage_p95(row, "packet_with_nms_ms", "czas_paczki_z_nms_ms", "packet_ms")

    if metric == "postprocess_nms_ms":
        return _get_stage_p95(row, "postprocess_nms_ms", "czas_postprocessingu_nms_ms")

    if metric == "nms_overhead_ms":
        val = _get_stage_p95(row, "nms_overhead_ms", "narzut_nms_ms")
        if val is not None:
            return val
        with_nms = _metric_stage(row, "packet_with_nms_ms")
        no_nms = _metric_stage(row, "packet_without_nms_ms")
        if with_nms is not None and no_nms is not None:
            return with_nms - no_nms
        return None

    return _get_stage_p95(row, metric)


def _interpret_budget(value_ms: float | None) -> str:
    if value_ms is None:
        return "-"
    if value_ms < 15.0:
        return "bardzo dobry zapas"
    if value_ms <= 20.0:
        return "mieści się w celu 50 FPS"
    if value_ms <= 25.0:
        return "wariant graniczny"
    return "za wolno dla celu 50 FPS"


def _interpret_nms_overhead(value_ms: float | None) -> str:
    if value_ms is None:
        return "-"
    if value_ms < 2.0:
        return "bardzo mały koszt NMS"
    if value_ms <= 5.0:
        return "akceptowalny koszt NMS"
    if value_ms <= 10.0:
        return "NMS zaczyna być istotny"
    return "NMS jest bottleneckiem"


def _packet_with_nms_norm_p95(row: dict[str, Any]) -> float | None:
    # Preferencja: metryka packet_with_nms w normalized.
    val = _get_norm_p95(row, "packet_with_nms_ms", "czas_paczki_z_nms_ms")
    if val is not None:
        return val
    # Fallback: stary benchmark - packet_ms.
    return _get_norm_p95(row, "packet_ms")


def _packet_no_nms_norm_p95(row: dict[str, Any]) -> float | None:
    return _get_norm_p95(row, "packet_no_nms_ms", "czas_paczki_bez_nms_ms", "packet_ms")


def _nms_overhead_norm_p95(row: dict[str, Any]) -> float | None:
    val = _get_norm_p95(row, "nms_overhead_ms", "narzut_nms_ms")
    if val is not None:
        return val
    with_nms = _get_norm_p95(row, "packet_with_nms_ms", "czas_paczki_z_nms_ms")
    no_nms = _get_norm_p95(row, "packet_no_nms_ms", "czas_paczki_bez_nms_ms")
    if with_nms is not None and no_nms is not None:
        return with_nms - no_nms
    return None


def _best_global_for_explanatory_report(rows_ok: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows_ok:
        return None
    ranked: list[tuple[float, dict[str, Any]]] = []
    for row in rows_ok:
        score = _packet_with_nms_norm_p95(row)
        if score is None:
            score = 1e18
        ranked.append((float(score), row))
    ranked.sort(key=lambda x: x[0])
    return ranked[0][1] if ranked else None


def _bottleneck(row: dict[str, Any], normalized: bool = False) -> str:
    if normalized:
        color = _get_norm_p95(row, "color_ms") or 0.0
        inf = _get_norm_p95(row, "inference_ms") or 0.0
    else:
        color = _get_stage_p95(row, "color_ms") or 0.0
        inf = _get_stage_p95(row, "inference_ms") or 0.0
    return "inference" if inf >= color else "color"


def _best_by_shape(rows_ok: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for r in rows_ok:
        key = _shape(r["input_shape_hw"])
        cur = best.get(key)
        cur_val = _packet_with_nms_norm_p95(cur) if cur else None
        new_val = _packet_with_nms_norm_p95(r)
        cur_cmp = 1e18 if cur_val is None else float(cur_val)
        new_cmp = 1e18 if new_val is None else float(new_val)
        if cur is None or new_cmp < cur_cmp:
            best[key] = r
    return best

def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        if math.isnan(x):
            return default
        return x
    except Exception:
        return default


def _chart_label(row: dict[str, Any]) -> str:
    return f"{_shape(row['input_shape_hw'])} b{row['batch']}"


def _group_rows_by_resolution(rows_ok: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for r in rows_ok:
        key = _shape(r["input_shape_hw"])
        grouped.setdefault(key, []).append(r)

    for key in grouped:
        grouped[key].sort(key=lambda x: int(x["batch"]))

    return grouped


def build_charts(payload: dict[str, Any], rows: list[dict[str, Any]], reports_dir: Path) -> Path:
    """
    Generuje wykresy PNG do final_benchmark_08_charts/.
    Wykresy działają także na starym JSON, bez realnego NMS.
    """
    rows_ok = _ok_rows(rows)
    charts_dir = reports_dir / "final_benchmark_08_charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    if not rows_ok:
        return charts_dir

    grouped = _group_rows_by_resolution(rows_ok)

    # ------------------------------------------------------------
    # 01 packet without vs with NMS norm
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    for res, items in grouped.items():
        batches = [int(r["batch"]) for r in items]
        without = [_safe_float(_metric_norm(r, "packet_without_nms_ms"), np.nan) for r in items]
        withn = [_safe_float(_metric_norm(r, "packet_with_nms_ms"), np.nan) for r in items]

        ax.plot(batches, without, marker="o", linestyle="--", label=f"{res} bez NMS")
        ax.plot(batches, withn, marker="o", linestyle="-", label=f"{res} z NMS")

    ax.axhline(TARGET_PACKET_MS, linestyle=":", linewidth=1.5, label="budżet 50 FPS / 20 ms")
    ax.set_title("Packet p95 norm — bez NMS vs z NMS")
    ax.set_xlabel("Batch")
    ax.set_ylabel("ms / paczka 4-kamerowa")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(charts_dir / "chart_01_packet_without_vs_with_nms_norm.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------
    # 02 breakdown norm b4
    # ------------------------------------------------------------
    labels = [_chart_label(r) for r in rows_ok]
    color_vals = [_safe_float(_metric_norm(r, "color_ms")) for r in rows_ok]
    inf_vals = [_safe_float(_metric_norm(r, "inference_ms")) for r in rows_ok]
    nms_vals = [_safe_float(_metric_norm(r, "postprocess_nms_ms")) for r in rows_ok]

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.55), 6))
    ax.bar(x, color_vals, label="color/debayer")
    ax.bar(x, inf_vals, bottom=color_vals, label="inference")
    bottom2 = np.asarray(color_vals) + np.asarray(inf_vals)
    ax.bar(x, nms_vals, bottom=bottom2, label="postprocess/NMS")

    ax.axhline(TARGET_PACKET_MS, linestyle=":", linewidth=1.5, label="20 ms")
    ax.set_title("Rozbicie czasu p95 norm do paczki 4-kamerowej")
    ax.set_ylabel("ms / paczka 4-kamerowa")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=65, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(charts_dir / "chart_02_breakdown_norm_b4.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------
    # 03 NMS overhead norm
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    for res, items in grouped.items():
        batches = [int(r["batch"]) for r in items]
        overhead = [_safe_float(_metric_norm(r, "nms_overhead_ms"), np.nan) for r in items]
        ax.plot(batches, overhead, marker="o", label=res)

    ax.axhline(10.0, linestyle=":", linewidth=1.5, label="NMS bottleneck 10 ms")
    ax.set_title("Narzut NMS p95 norm do paczki 4-kamerowej")
    ax.set_xlabel("Batch")
    ax.set_ylabel("ms / paczka 4-kamerowa")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(charts_dir / "chart_03_nms_overhead_norm_b4.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------
    # 04 FPS without vs with NMS
    # ------------------------------------------------------------
    labels = [_chart_label(r) for r in rows_ok]
    fps_without = []
    fps_with = []

    for r in rows_ok:
        without = _metric_norm(r, "packet_without_nms_ms")
        withn = _metric_norm(r, "packet_with_nms_ms")
        fps_without.append(1000.0 / without if without and without > 0 else 0.0)
        fps_with.append(1000.0 / withn if withn and withn > 0 else 0.0)

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.55), 6))
    ax.bar(x - width / 2, fps_without, width, label="FPS bez NMS")
    ax.bar(x + width / 2, fps_with, width, label="FPS z NMS")
    ax.axhline(TARGET_FPS, linestyle=":", linewidth=1.5, label="cel 50 FPS")
    ax.set_title("FPS p95 norm — bez NMS vs z NMS")
    ax.set_ylabel("FPS")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=65, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(charts_dir / "chart_04_fps_without_vs_with_nms.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------
    # 05 best batch per resolution
    # ------------------------------------------------------------
    best_shape = _best_by_shape(rows_ok)
    res_labels = []
    best_batches = []
    best_times = []

    for res in sorted(best_shape.keys(), key=lambda s: int(s.split("x")[0]) * int(s.split("x")[1])):
        r = best_shape[res]
        res_labels.append(res)
        best_batches.append(int(r["batch"]))
        best_times.append(_safe_float(_packet_with_nms_norm_p95(r)))

    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(res_labels))
    ax1.bar(x, best_times, label="packet z NMS p95 norm")
    ax1.axhline(TARGET_PACKET_MS, linestyle=":", linewidth=1.5, label="20 ms")
    ax1.set_title("Najlepszy batch dla każdej rozdzielczości")
    ax1.set_ylabel("ms / paczka 4-kamerowa")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{r}\nb{b}" for r, b in zip(res_labels, best_batches)])
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(charts_dir / "chart_05_best_batch_per_resolution.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------
    # 06 bottleneck share
    # ------------------------------------------------------------
    labels = [_chart_label(r) for r in rows_ok]
    color_vals = np.asarray([_safe_float(_metric_norm(r, "color_ms")) for r in rows_ok])
    inf_vals = np.asarray([_safe_float(_metric_norm(r, "inference_ms")) for r in rows_ok])
    nms_vals = np.asarray([_safe_float(_metric_norm(r, "postprocess_nms_ms")) for r in rows_ok])

    total = color_vals + inf_vals + nms_vals
    total[total == 0] = 1.0

    color_pct = color_vals / total * 100.0
    inf_pct = inf_vals / total * 100.0
    nms_pct = nms_vals / total * 100.0

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.55), 6))
    ax.bar(x, color_pct, label="color")
    ax.bar(x, inf_pct, bottom=color_pct, label="inference")
    ax.bar(x, nms_pct, bottom=color_pct + inf_pct, label="NMS")
    ax.set_title("Udział procentowy etapów w czasie p95 norm")
    ax.set_ylabel("%")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=65, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(charts_dir / "chart_06_bottleneck_share.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------
    # 07 latency vs throughput
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    xs = [_safe_float(_metric_stage(r, "packet_with_nms_ms")) for r in rows_ok]
    ys = [_safe_float(_metric_norm(r, "packet_with_nms_ms")) for r in rows_ok]

    ax.scatter(xs, ys)
    for x0, y0, r in zip(xs, ys, rows_ok):
        ax.annotate(_chart_label(r), (x0, y0), fontsize=7, xytext=(4, 4), textcoords="offset points")

    ax.axhline(TARGET_PACKET_MS, linestyle=":", linewidth=1.5, label="20 ms norm")
    ax.set_title("Latency vs throughput")
    ax.set_xlabel("rzeczywisty packet z NMS p95 dla całego batcha [ms]")
    ax.set_ylabel("packet z NMS p95 norm paczka4 [ms]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(charts_dir / "chart_07_latency_vs_throughput.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------
    # 08 heatmap packet with NMS norm
    # ------------------------------------------------------------
    resolutions = sorted(grouped.keys(), key=lambda s: int(s.split("x")[0]) * int(s.split("x")[1]))
    batches = sorted({int(r["batch"]) for r in rows_ok})

    matrix = np.full((len(resolutions), len(batches)), np.nan, dtype=np.float64)
    for i, res in enumerate(resolutions):
        for r in grouped[res]:
            j = batches.index(int(r["batch"]))
            matrix[i, j] = _safe_float(_packet_with_nms_norm_p95(r), np.nan)

    fig, ax = plt.subplots(figsize=(max(8, len(batches) * 0.7), 4.8))
    im = ax.imshow(matrix, aspect="auto")
    ax.set_title("Heatmapa: packet z NMS p95 norm paczka4")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Rozdzielczość")
    ax.set_xticks(np.arange(len(batches)))
    ax.set_xticklabels([str(b) for b in batches])
    ax.set_yticks(np.arange(len(resolutions)))
    ax.set_yticklabels(resolutions)

    for i in range(len(resolutions)):
        for j in range(len(batches)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="ms")
    fig.tight_layout()
    fig.savefig(charts_dir / "chart_08_heatmap_packet_with_nms_norm.png", dpi=160)
    plt.close(fig)

    return charts_dir


def build_md(payload: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    # Dotychczasowy skrócony raport techniczny (zachowany).
    rows_ok = _ok_rows(rows)
    best_shape = _best_by_shape(rows_ok)
    best_global = _best_global_for_explanatory_report(rows_ok)
    normalize_to_batch = int(payload.get("normalize_to_batch", 4))

    lines: list[str] = []
    a = lines.append
    a("# Final Benchmark 08 - Report")
    a("")
    a("## Scope")
    a("")
    a("- Pipeline mierzony bez resize (capture_shape == inference_shape).")
    a("- Czasy etapów: color processing, inference, packet total.")
    a(f"- Normalizacja porównawcza: packet/color/inference do batch={normalize_to_batch}.")
    a(f"- Cel referencyjny: {TARGET_FPS:.0f} FPS => {TARGET_PACKET_MS:.2f} ms/packet.")
    a("")

    a("## Executive Summary")
    a("")
    if best_global is None:
        a("Brak poprawnych wyników (wszystkie warianty skipped/fail).")
    else:
        gp95 = _packet_with_nms_norm_p95(best_global)
        gfps = (1000.0 / gp95) if (gp95 is not None and gp95 > 0) else None
        gshape = _shape(best_global["input_shape_hw"])
        gbatch = int(best_global["batch"])
        gmargin = (TARGET_PACKET_MS - gp95) if gp95 is not None else None
        a(f"- Najlepszy wariant globalnie (po normalizacji do b{normalize_to_batch}): `{gshape} b{gbatch}`.")
        a(f"- Packet p95 norm: **{_existing_or_dash(gp95)} ms** (FPS ~ **{_existing_or_dash(gfps, 1)}**).")
        a(f"- Margines do celu 50 FPS: **{_existing_or_dash(gmargin)} ms**.")
        a(f"- Dominujący bottleneck (p95 norm): **{_bottleneck(best_global, normalized=True)}**.")
    a("")

    a("## Best Batch Per Resolution")
    a("")
    a("| Resolution | Best batch | Packet p95 norm (ms) | FPS norm | Margin vs 20ms | Bottleneck |")
    a("|---|---:|---:|---:|---:|---|")
    for shape_key in sorted(best_shape.keys(), key=lambda s: int(s.split("x")[0]) * int(s.split("x")[1])):
        r = best_shape[shape_key]
        p95n = _packet_with_nms_norm_p95(r)
        fpsn = (1000.0 / p95n) if (p95n is not None and p95n > 0) else None
        margin = (TARGET_PACKET_MS - p95n) if p95n is not None else None
        a(f"| {shape_key} | {r['batch']} | {_existing_or_dash(p95n)} | {_existing_or_dash(fpsn, 1)} | "
          f"{_existing_or_dash(margin)} | {_bottleneck(r, normalized=True)} |")
    a("")

    a("## Full Table")
    a("")
    a("| Res | Batch | Color p95 | Inf p95 | Packet p95 | Color p95 norm | Inf p95 norm | Packet p95 norm | FPS p95 norm |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(rows_ok, key=lambda x: (int(x["input_shape_hw"][0]) * int(x["input_shape_hw"][1]), int(x["batch"]))):
        p95n = _packet_with_nms_norm_p95(r)
        fpsn = (1000.0 / p95n) if (p95n is not None and p95n > 0) else None
        a(
            f"| {_shape(r['input_shape_hw'])} | {r['batch']} | "
            f"{_existing_or_dash(_get_stage_p95(r, 'color_ms'))} | {_existing_or_dash(_get_stage_p95(r, 'inference_ms'))} | "
            f"{_existing_or_dash(_get_stage_p95(r, 'packet_with_nms_ms', 'czas_paczki_z_nms_ms', 'packet_ms'))} | "
            f"{_existing_or_dash(_get_norm_p95(r, 'color_ms'))} | {_existing_or_dash(_get_norm_p95(r, 'inference_ms'))} | "
            f"{_existing_or_dash(p95n)} | {_existing_or_dash(fpsn, 1)} |"
        )
    a("")

    skipped = [r for r in rows if r.get("skipped")]
    a("## Skipped / Errors")
    a("")
    if not skipped:
        a("Brak.")
    else:
        for r in skipped:
            shape_hw = r.get("shape_hw") or r.get("input_shape_hw") or ["?", "?"]
            a(f"- {_shape(shape_hw)} b{r.get('batch')}: {r.get('reason')}")
    a("")
    return "\n".join(lines) + "\n"


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    rows_ok = _ok_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "resolution",
            "batch",
            "color_p95_ms",
            "inference_p95_ms",
            "packet_p95_ms",
            "color_p95_ms_norm",
            "inference_p95_ms_norm",
            "packet_p95_ms_norm",
            "fps_p95_norm",
            "bottleneck_raw",
            "bottleneck_norm",
        ])
        for r in sorted(rows_ok, key=lambda x: (int(x["input_shape_hw"][0]) * int(x["input_shape_hw"][1]), int(x["batch"]))):
            p95n = _packet_with_nms_norm_p95(r)
            fpsn = (1000.0 / p95n) if (p95n is not None and p95n > 0) else None
            w.writerow([
                _shape(r["input_shape_hw"]),
                r["batch"],
                _get_stage_p95(r, "color_ms"),
                _get_stage_p95(r, "inference_ms"),
                _get_stage_p95(r, "packet_with_nms_ms", "czas_paczki_z_nms_ms", "packet_ms"),
                _get_norm_p95(r, "color_ms"),
                _get_norm_p95(r, "inference_ms"),
                p95n,
                fpsn,
                _bottleneck(r, normalized=False),
                _bottleneck(r, normalized=True),
            ])


def build_metric_dictionary_json(payload: dict[str, Any], reports_dir: Path) -> Path:
    out_path = reports_dir / "final_benchmark_08_parametry_opisowe.json"
    dictionary = {
        "nazwa_testu": "final_benchmark_08",
        "folder_testu": str(reports_dir),
        "opis": "Benchmark ścieżki bez resize: RAW Bayer -> debayer/color -> TensorRT inference, opcjonalnie postprocessing/NMS.",
        "parametry_globalne": {
            "czas_pomiaru_s": payload.get("duration_s"),
            "warmup_iteracje": payload.get("warmup"),
            "normalizacja_do_batcha": payload.get("normalize_to_batch", 4),
            "znaczenie_batcha_referencyjnego": "batch=4 oznacza jedną paczkę 4-kamerową (jedna zsynchronizowana chwila z 4 kamer).",
            "cel_fps": TARGET_FPS,
            "budzet_ms_na_paczke_4_kamer": TARGET_PACKET_MS,
        },
        "definicje_pojec": {
            "batch": "Liczba obrazów przetwarzanych jednocześnie przez silnik.",
            "paczka_4_kamerowa": "Referencyjna jednostka czasu odpowiadająca 4 kamerom w tej samej chwili.",
            "normalizacja_do_paczki4": "Przeliczenie czasu batcha do ekwiwalentu paczki 4-kamerowej: czas_norm = czas_batcha * 4 / batch.",
            "p95": "Percentyl 95; 95% próbek ma czas <= p95.",
            "mediana": "Percentyl 50; środkowa wartość rozkładu.",
        },
        "definicje_metryk": {
            "color_ms": "Sam etap przygotowania obrazu: RAW Bayer -> color/debayer/normalizacja.",
            "inference_ms": "Sam czas inferencji TensorRT.",
            "color_plus_inference_ms": "Czas color + inference, czyli szybki tor bez NMS.",
            "postprocess_nms_ms": "Czas postprocessingu i NMS. Jeśli NMS jest dummy, wartość może być bliska zeru.",
            "packet_without_nms_ms": "Pełny czas paczki bez NMS, do końca inferencji.",
            "packet_with_nms_ms": "Pełny czas paczki z NMS, najbliższy realnej detekcji.",
            "nms_overhead_ms": "Różnica: packet_with_nms_ms - packet_without_nms_ms.",
            "nms_detections_count": "Liczba detekcji po NMS, jeśli realny NMS jest dostępny.",
            "nms_mode": "Tryb NMS, np. disabled_dummy, torchvision_nms, custom, trt_plugin.",
            "p95": "Percentyl 95; 95% próbek ma czas <= p95.",
            "median": "Mediana; typowy czas środkowy.",
            "norm_b4": "Czas przeliczony do paczki referencyjnej batch=4.",
            "fps_without_nms_p95_norm_b4": "FPS liczony z packet_without_nms_ms p95 norm b4.",
            "fps_with_nms_p95_norm_b4": "FPS liczony z packet_with_nms_ms p95 norm b4.",
        },

        "progi_interpretacyjne": {
            "czas_paczki_z_nms_p95_norm_paczka4_ms": {
                "<15": "bardzo dobry zapas",
                "15-20": "mieści się w celu 50 FPS",
                "20-25": "wariant graniczny",
                ">25": "za wolno dla celu 50 FPS",
            },
            "narzut_nms_p95_norm_paczka4_ms": {
                "<2": "bardzo mały koszt NMS",
                "2-5": "akceptowalny koszt NMS",
                "5-10": "NMS zaczyna być istotny",
                ">10": "NMS jest bottleneckiem",
            },
        },
    }
    out_path.write_text(json.dumps(dictionary, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def build_explanatory_md(payload: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    rows_ok = _ok_rows(rows)
    best = _best_global_for_explanatory_report(rows_ok)

    lines: list[str] = []
    a = lines.append
    a("# Final Benchmark 08 — Raport opisowy")
    a("")
    a("## 1. Co mierzy ten benchmark?")
    a("")
    a("Benchmark mierzy ścieżkę bez resize:")
    a("")
    a("`RAW Bayer -> color processing / debayer -> TensorRT inference`")
    a("")
    a("Opcjonalnie może obejmować pełną ścieżkę:")
    a("")
    a("`RAW Bayer -> color processing / debayer -> TensorRT inference -> postprocessing -> NMS -> finalne detekcje`")
    a("")

    a("## 2. Co oznacza batch?")
    a("")
    a("`batch=4` oznacza jedną referencyjną paczkę 4-kamerową, czyli jedną zsynchronizowaną chwilę z 4 kamer.")
    a("")
    a("| Batch | Liczba paczek 4-kamerowych |")
    a("|---:|---:|")
    for b, p in [(4, 1), (8, 2), (12, 3), (16, 4), (24, 6), (32, 8), (64, 16)]:
        a(f"| {b} | {p} |")
    a("")

    a("## 3. Jakie czasy są mierzone?")
    a("")
    a("| Metryka | Znaczenie | Jak interpretować |")
    a("|---|---|---|")
    a("| `color_ms` | Samo przygotowanie obrazu / debayer / normalizacja | Wysoko = preprocessing jest bottleneckiem |")
    a("| `inference_ms` | Sama inferencja TensorRT | Wysoko = model albo engine jest bottleneckiem |")
    a("| `color_plus_inference_ms` | Color + inference | Szybki tor bez NMS |")
    a("| `postprocess_nms_ms` | Postprocess i NMS | Wysoko = problem po stronie dekodowania/NMS |")
    a("| `packet_without_nms_ms` | Cały packet bez NMS | Czas do końca inferencji |")
    a("| `packet_with_nms_ms` | Cały packet z NMS | Najbliżej realnej detekcji |")
    a("| `nms_overhead_ms` | Różnica z NMS minus bez NMS | Koszt pełnego postprocessingu |")
    a("")

    a("## 4. Czas paczki bez NMS i z NMS")
    a("")
    a("- czas paczki bez NMS = color processing + inference")
    a("- czas paczki z NMS = color processing + inference + postprocessing + NMS")
    a("- narzut NMS = czas paczki z NMS - czas paczki bez NMS")
    a("")

    a("## 5. Jak czytać p95?")
    a("")
    a("`p95` to percentyl 95. Dla systemu live to kluczowa metryka stabilności, bo pokazuje zachowanie gorszego ogona opóźnień.")
    a("")

    a("## 6. Normalizacja do paczki 4-kamerowej")
    a("")
    a("Wzór:")
    a("")
    a("`czas_norm = czas_batcha * 4 / batch`")
    a("")
    a("Normalizacja służy do porównania throughputu między batchami. Latency nadal trzeba czytać też z czasu rzeczywistego całego batcha.")
    a("")

    a("## 7. Progi interpretacyjne")
    a("")
    a("Dla `czas_paczki_z_nms_p95_norm_paczka4_ms`:")
    a("- < 15 ms: bardzo dobry zapas")
    a("- 15–20 ms: mieści się w celu 50 FPS")
    a("- 20–25 ms: wariant graniczny")
    a("- > 25 ms: za wolno dla celu 50 FPS")
    a("")
    a("Dla `narzut_nms_p95_norm_paczka4_ms`:")
    a("- < 2 ms: bardzo mały koszt NMS")
    a("- 2–5 ms: akceptowalny koszt NMS")
    a("- 5–10 ms: NMS zaczyna być istotny")
    a("- > 10 ms: NMS jest bottleneckiem")
    a("")

    a("## 8. Wykresy — szybka interpretacja")
    a("")
    a("Poniższe wykresy pokazują wizualnie różnicę między samym przygotowaniem obrazu, inferencją, packetem bez NMS, packetem z NMS oraz narzutem NMS.")
    a("")
    a("![Packet bez NMS vs z NMS](final_benchmark_08_charts/chart_01_packet_without_vs_with_nms_norm.png)")
    a("")
    a("![Rozbicie czasu](final_benchmark_08_charts/chart_02_breakdown_norm_b4.png)")
    a("")
    a("![Narzut NMS](final_benchmark_08_charts/chart_03_nms_overhead_norm_b4.png)")
    a("")
    a("![FPS bez NMS vs z NMS](final_benchmark_08_charts/chart_04_fps_without_vs_with_nms.png)")
    a("")
    a("![Najlepszy batch per rozdzielczość](final_benchmark_08_charts/chart_05_best_batch_per_resolution.png)")
    a("")
    a("![Udział bottlenecków](final_benchmark_08_charts/chart_06_bottleneck_share.png)")
    a("")
    a("![Latency vs throughput](final_benchmark_08_charts/chart_07_latency_vs_throughput.png)")
    a("")
    a("![Heatmapa packet z NMS](final_benchmark_08_charts/chart_08_heatmap_packet_with_nms_norm.png)")
    a("")

    a("## 9. Najlepszy wariant globalny")
    a("")
    if best is None:
        a("Brak poprawnych wyników.")
    else:
        res = _shape(best["input_shape_hw"])
        batch = int(best["batch"])
        no_nms = _packet_no_nms_norm_p95(best)
        with_nms = _packet_with_nms_norm_p95(best)
        overhead = _nms_overhead_norm_p95(best)
        a(f"- rozdzielczość: `{res}`")
        a(f"- batch: `{batch}`")
        a(f"- czas paczki bez NMS p95 norm paczka4: `{_existing_or_dash(no_nms, 3)} ms`")
        a(f"- czas paczki z NMS p95 norm paczka4: `{_existing_or_dash(with_nms, 3)} ms`")
        a(f"- narzut NMS p95 norm paczka4: `{_existing_or_dash(overhead, 3)} ms`")
        a(f"- interpretacja budżetu: **{_interpret_budget(with_nms)}**")
        a(f"- interpretacja NMS: **{_interpret_nms_overhead(overhead)}**")
    a("")

    a("## 8. Tabela wyników — czasy rzeczywiste dla całego batcha")
    a("")
    a("| Rozdzielczość | Batch | Paczek 4-kam. | Kolor p95 | Inferencja p95 | NMS/postprocess p95 | Paczka bez NMS p95 | Paczka z NMS p95 | Narzut NMS p95 |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(rows_ok, key=lambda x: (int(x["input_shape_hw"][0]) * int(x["input_shape_hw"][1]), int(x["batch"]))):
        batch = int(r["batch"])
        eq = batch / 4.0
        color = _get_stage_p95(r, "color_ms")
        inf = _get_stage_p95(r, "inference_ms")
        post = _get_stage_p95(r, "postprocess_ms", "czas_postprocessingu_nms_ms")
        no_nms = _get_stage_p95(r, "packet_no_nms_ms", "czas_paczki_bez_nms_ms")
        with_nms = _get_stage_p95(r, "packet_with_nms_ms", "czas_paczki_z_nms_ms", "packet_ms")
        overhead = _get_stage_p95(r, "nms_overhead_ms", "narzut_nms_ms")
        if overhead is None and no_nms is not None and with_nms is not None:
            overhead = with_nms - no_nms
        a(
            f"| {_shape(r['input_shape_hw'])} | {batch} | {_existing_or_dash(eq, 3)} | "
            f"{_existing_or_dash(color, 3)} | {_existing_or_dash(inf, 3)} | {_existing_or_dash(post, 3)} | "
            f"{_existing_or_dash(no_nms, 3)} | {_existing_or_dash(with_nms, 3)} | {_existing_or_dash(overhead, 3)} |"
        )
    a("")

    a("## 9. Tabela wyników — przeliczenie do paczki 4-kamerowej")
    a("")
    a("| Rozdzielczość | Batch | Paczek 4-kam. | Paczka bez NMS p95 norm | Paczka z NMS p95 norm | Narzut NMS p95 norm | Interpretacja |")
    a("|---|---:|---:|---:|---:|---:|---|")
    for r in sorted(rows_ok, key=lambda x: (int(x["input_shape_hw"][0]) * int(x["input_shape_hw"][1]), int(x["batch"]))):
        batch = int(r["batch"])
        eq = batch / 4.0
        no_nms_n = _packet_no_nms_norm_p95(r)
        with_nms_n = _packet_with_nms_norm_p95(r)
        overhead_n = _nms_overhead_norm_p95(r)
        interp = _interpret_budget(with_nms_n)
        a(
            f"| {_shape(r['input_shape_hw'])} | {batch} | {_existing_or_dash(eq, 3)} | "
            f"{_existing_or_dash(no_nms_n, 3)} | {_existing_or_dash(with_nms_n, 3)} | "
            f"{_existing_or_dash(overhead_n, 3)} | {interp} |"
        )
    a("")

    a("## 10. Jak wyciągać wnioski?")
    a("")
    a("- Jeśli czas bez NMS jest niski, ale czas z NMS wysoki, problemem jest postprocessing/NMS.")
    a("- Jeśli oba czasy są wysokie, problem zaczyna się już w przygotowaniu obrazu albo inferencji.")
    a("- Jeśli większy batch poprawia czas bez NMS, ale pogarsza czas z NMS, NMS skaluje się nieliniowo.")
    a("- Do throughputu patrz na wartości `norm paczka4`.")
    a("- Do latency patrz też na rzeczywisty czas całego batcha i liczbę paczek 4-kamerowych w batchu.")
    a("")

    a("## 11. Pominięte warianty i błędy")
    a("")
    skipped = [r for r in rows if r.get("skipped")]
    if not skipped:
        a("Brak.")
    else:
        for r in skipped:
            shape_hw = r.get("shape_hw") or r.get("input_shape_hw") or ["?", "?"]
            a(f"- {_shape(shape_hw)} b{r.get('batch')}: {r.get('reason')}")
    a("")
    return "\n".join(lines) + "\n"


def main() -> int:
    reports_dir = _resolve_test_results_dir(REPORTS_ROOT)
    input_json = reports_dir / "final_benchmark_08.json"
    out_md = reports_dir / "final_benchmark_08_report.md"
    out_csv = reports_dir / "final_benchmark_08_summary.csv"
    out_desc_json = reports_dir / "final_benchmark_08_parametry_opisowe.json"
    out_desc_md = reports_dir / "final_benchmark_08_raport_opisowy.md"

    if not input_json.exists():
        raise SystemExit(f"Missing input json: {input_json}")

    payload = json.loads(input_json.read_text(encoding="utf-8"))
    rows = list(payload.get("rows", []))

    reports_dir.mkdir(parents=True, exist_ok=True)

    charts_dir = build_charts(payload, rows, reports_dir)

    md = build_md(payload, rows)
    out_md.write_text(md, encoding="utf-8")
    write_csv(rows, out_csv)

    dict_path = build_metric_dictionary_json(payload, reports_dir)
    desc_md = build_explanatory_md(payload, rows)
    out_desc_md.write_text(desc_md, encoding="utf-8")

    print(f"wrote: {out_md}")
    print(f"wrote: {out_csv}")
    print(f"wrote: {dict_path}")
    print(f"wrote: {out_desc_md}")
    print(f"wrote charts: {charts_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
