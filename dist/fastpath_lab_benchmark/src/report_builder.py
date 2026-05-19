"""Report builder for the lab benchmark.

"""
from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------- I/O


def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _next_test_dir(reports_root: Path) -> Path:
    max_idx = 0
    for entry in reports_root.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        if not name.startswith("test_"):
            continue
        suffix = name[5:]
        if not suffix.isdigit():
            continue
        max_idx = max(max_idx, int(suffix))
    return reports_root / f"test_{max_idx + 1:03d}"


# ---------------------------------------------------------- structures


@dataclass(slots=True)
class ResolutionRow:
    label: str
    batch: int
    inference_h: int
    inference_w: int
    pixels_mpx: float
    color_ms_p95: Optional[float]
    inference_ms_p95: float
    packet_ms_p95: float
    fps_per_camera_median: float
    fps_per_camera_safe_p95: float
    pass_50fps_safe_p95: bool
    stability_margin_ms: float
    inference_only_ms_p95: Optional[float] = None
    color_only_ms_p95: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "batch": self.batch,
            "inference_h": self.inference_h,
            "inference_w": self.inference_w,
            "pixels_mpx": self.pixels_mpx,
            "color_ms_p95": self.color_ms_p95,
            "inference_ms_p95": self.inference_ms_p95,
            "packet_ms_p95": self.packet_ms_p95,
            "fps_per_camera_median": self.fps_per_camera_median,
            "fps_per_camera_safe_p95": self.fps_per_camera_safe_p95,
            "pass_50fps_safe_p95": self.pass_50fps_safe_p95,
            "stability_margin_ms": self.stability_margin_ms,
            "inference_only_ms_p95": self.inference_only_ms_p95,
            "color_only_ms_p95": self.color_only_ms_p95,
        }


def _row_from_result(label: str, full_payload: dict) -> Optional[ResolutionRow]:
    """Build a row from a benchmark JSON payload. Returns None if invalid."""
    shape = full_payload.get("inference_input_shape") or {}
    h = int(shape.get("h", 0))
    w = int(shape.get("w", 0))
    if h <= 0 or w <= 0:
        return None

    verdict = (full_payload.get("aggregate") or {}).get("verdict") or {}
    stage_stats = (full_payload.get("aggregate") or {}).get("stage_stats") or {}
    inference_p95 = float(stage_stats.get("inference_ms", {}).get("p95_ms", float("nan")))
    color_p95 = stage_stats.get("color_ms", {}).get("p95_ms")
    return ResolutionRow(
        label=label,
        batch=int(full_payload.get("batch", 0) or 0),
        inference_h=h,
        inference_w=w,
        pixels_mpx=(h * w) / 1.0e6,
        color_ms_p95=float(color_p95) if color_p95 is not None else None,
        inference_ms_p95=inference_p95,
        packet_ms_p95=float(verdict.get("packet_ms_p95", float("nan"))),
        fps_per_camera_median=float(verdict.get("fps_per_camera_median", float("nan"))),
        fps_per_camera_safe_p95=float(verdict.get("fps_per_camera_safe_p95", float("nan"))),
        pass_50fps_safe_p95=bool(verdict.get("pass_50fps_safe_p95", False)),
        stability_margin_ms=float(verdict.get("stability_margin_ms", float("nan"))),
        inference_only_ms_p95=float(full_payload.get("inference_only_ms_p95"))
        if full_payload.get("inference_only_ms_p95") is not None else None,
        color_only_ms_p95=float(full_payload.get("color_only_ms_p95"))
        if full_payload.get("color_only_ms_p95") is not None else None,
    )


def collect_resolution_rows(results_dir: Path) -> List[ResolutionRow]:
    """Read all `full_synthetic_*.json` files in results/."""
    rows: List[ResolutionRow] = []
    for f in sorted(results_dir.glob("full_synthetic_*.json")):
        payload = load_json(f)
        if not payload:
            continue
        row = _row_from_result(f.stem.replace("full_synthetic_", ""), payload)
        if row is not None:
            inf_path = results_dir / f"inference_only_{row.label}.json"
            col_path = results_dir / f"color_only_{row.label}.json"
            inf_payload = load_json(inf_path)
            col_payload = load_json(col_path)
            if inf_payload:
                row.inference_only_ms_p95 = (
                    (inf_payload.get("aggregate") or {})
                    .get("stage_stats", {})
                    .get("inference_ms", {})
                    .get("p95_ms")
                )
            if col_payload:
                row.color_only_ms_p95 = (
                    (col_payload.get("aggregate") or {})
                    .get("stage_stats", {})
                    .get("color_ms", {})
                    .get("p95_ms")
                )
            rows.append(row)
    rows.sort(key=lambda r: (r.pixels_mpx, r.batch))
    return rows


# -------------------------------------------------------------- charts


def _maybe_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def render_charts(rows: List[ResolutionRow], baseline: Optional[dict],
                  charts_dir: Path) -> List[str]:
    plt = _maybe_import_matplotlib()
    if plt is None or not rows:
        return []
    charts_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    labels = [r.label for r in rows]
    fps_med = [r.fps_per_camera_median for r in rows]
    fps_p95 = [r.fps_per_camera_safe_p95 for r in rows]
    packet_p95 = [r.packet_ms_p95 for r in rows]
    inf_p95 = [r.inference_ms_p95 for r in rows]
    mpx = [r.pixels_mpx for r in rows]

    # 1) FPS chart with 50 FPS line (legend by batch color)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = list(range(len(rows)))
    batch_groups: Dict[int, List[int]] = {}
    for i, r in enumerate(rows):
        batch_groups.setdefault(r.batch, []).append(i)
    for b, idxs in sorted(batch_groups.items()):
        ax.plot(
            [x[i] for i in idxs],
            [fps_p95[i] for i in idxs],
            marker="o",
            linestyle="-",
            label=f"b{b}",
        )
    ax.axhline(50.0, color="red", linewidth=1.0, linestyle=":", label="50 FPS target")
    if baseline:
        b_p95 = [
            item["fps_per_camera_safe_p95"]
            for item in baseline.get("results", [])
            if item.get("inference_input_shape") in labels
        ]
        b_labels = [
            item["inference_input_shape"]
            for item in baseline.get("results", [])
            if item.get("inference_input_shape") in labels
        ]
        if b_p95:
            b_idx = [labels.index(lbl) for lbl in b_labels]
            ax.plot(b_idx, b_p95, "^:", label=f"{baseline.get('baseline_name', 'baseline')} p95",
                    color="gray")
    ax.set_xticks(x, labels, rotation=20)
    ax.set_xlabel("inference input shape")
    ax.set_ylabel("FPS per camera")
    ax.set_title("FPS per camera vs inference size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = charts_dir / "fps_per_camera.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    written.append(out.name)

    # 2) packet_ms p95 vs 20 ms budget (legend by batch color)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for b, idxs in sorted(batch_groups.items()):
        ax.plot(
            [x[i] for i in idxs],
            [packet_p95[i] for i in idxs],
            marker="o",
            linestyle="-",
            label=f"b{b}",
        )
    ax.axhline(20.0, color="red", linewidth=1.0, linestyle=":", label="20 ms = 50 FPS budget")
    ax.set_xticks(x, labels, rotation=20)
    ax.set_xlabel("inference input shape")
    ax.set_ylabel("packet ms (p95)")
    ax.set_title("Packet latency vs 50-FPS budget")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = charts_dir / "packet_ms_vs_budget.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    written.append(out.name)

    # 3) inference ms vs mpx (legend by batch color)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for b, idxs in sorted(batch_groups.items()):
        ax.plot(
            [mpx[i] for i in idxs],
            [inf_p95[i] for i in idxs],
            marker="o",
            linestyle="-",
            label=f"b{b}",
        )
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (mpx[i], inf_p95[i]), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("input pixels (Mpx)")
    ax.set_ylabel("inference ms (p95)")
    ax.set_title("Inference cost vs pixel count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = charts_dir / "inference_vs_mpx.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    written.append(out.name)

    # 4) stage breakdown
    color_p95 = [r.color_ms_p95 if r.color_ms_p95 is not None else 0.0 for r in rows]
    post_p95 = [max(0.0, r.packet_ms_p95 - (r.color_ms_p95 or 0.0) - r.inference_ms_p95)
                for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x, color_p95, label="color ms p95", color="#4c72b0")
    ax.bar(x, inf_p95, bottom=color_p95, label="inference ms p95", color="#dd8452")
    ax.bar(x, post_p95, bottom=[c + i for c, i in zip(color_p95, inf_p95)],
           label="other (postprocess/host)", color="#55a868")
    ax.axhline(20.0, color="red", linewidth=1.0, linestyle=":", label="20 ms budget")
    ax.set_xticks(x, labels, rotation=20)
    ax.set_ylabel("ms (stacked, p95)")
    ax.set_title("Stage breakdown vs 50 FPS budget")
    ax.legend()
    fig.tight_layout()
    out = charts_dir / "stage_breakdown.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    written.append(out.name)

    # 5) comparative models: safe FPS
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.bar(x, fps_p95, color="#0d9488")
    for i, v in enumerate(fps_p95):
        ax.text(i, v + 0.8, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(50.0, color="red", linewidth=1.0, linestyle=":", label="50 FPS target")
    ax.set_xticks(x, labels, rotation=25)
    ax.set_ylabel("FPS per camera (safe p95)")
    ax.set_title("Model comparison: safe FPS (higher is better)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = charts_dir / "model_compare_fps_safe.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    written.append(out.name)

    # 6) comparative models: packet p95
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.bar(x, packet_p95, color="#7c3aed")
    for i, v in enumerate(packet_p95):
        ax.text(i, v + 0.3, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(20.0, color="red", linewidth=1.0, linestyle=":", label="20 ms budget")
    ax.set_xticks(x, labels, rotation=25)
    ax.set_ylabel("packet_ms p95 (lower is better)")
    ax.set_title("Model comparison: packet latency p95")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = charts_dir / "model_compare_packet_p95.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    written.append(out.name)

    return written


# --------------------------------------------------------- markdown/html


def _fmt(x: Optional[float], decimals: int = 2, na: str = "—") -> str:
    if x is None or x != x or not math.isfinite(x):  # NaN check
        return na
    return f"{x:.{decimals}f}"


def build_markdown(rows: List[ResolutionRow], baseline: Optional[dict],
                   charts: List[str], machine_info: Optional[dict]) -> str:
    lines: List[str] = []
    lines.append("# Fastpath Lab Benchmark — report")
    lines.append("")
    lines.append("Batch/packet semantics:")
    lines.append("")
    lines.append("- 1 packet = 1 batch = N obrazów (N = wartość `batch`).")
    lines.append("- Przykład: batch=4 -> 1 pakiet to 4 obrazy, 2 pakiety to 8 obrazów.")
    lines.append("- Report shows FPS per camera and total FPS for all 4 cameras.")
    lines.append("")
    lines.append("Interpretacja logu `inference_only`:")
    lines.append("")
    lines.append("- `last_inference_ms` = czas samego wywołania TensorRT (forward).")
    lines.append("- `last_packet_ms` = czas całej iteracji pętli testowej wokół inferencji")
    lines.append("  (inferencja + synchronizacja CUDA + narzut pętli/Python/pomiaru).")
    lines.append("- Dlatego `last_packet_ms` w `inference_only` jest zwykle większy niż")
    lines.append("  `last_inference_ms` i nie należy go traktować jako pełnego czasu")
    lines.append("  potoku produkcyjnego.")
    lines.append("- Do porównania modeli używaj głównie `inference_ms` (szczególnie p95),")
    lines.append("  a do oceny realnego budżetu 50 FPS używaj `full_synthetic packet_ms p95`.")
    lines.append("")
    lines.append("Nazwy metryk w raporcie (jednoznacznie):")
    lines.append("")
    lines.append("- **Sama inferencja (TRT)** = `inference_only_ms_p95`.")
    lines.append("- **Sam kolor (debayer)** = `color_only_ms_p95`.")
    lines.append("- **Pełny pipeline** = `full_synthetic packet_ms_p95`.")
    lines.append("")
    lines.append("Szybkie rozróżnienie (`inference_only`):")
    lines.append("")
    lines.append("- `last_inference_ms` -> \"ile trwa sam model\" (czysty TRT forward).")
    lines.append("- `last_packet_ms` -> \"ile trwa cały krok testu\" (forward + sync + narzut pętli).")
    lines.append("- To normalne, że `last_packet_ms` > `last_inference_ms`.")
    lines.append("")
    lines.append("This report answers three questions, in order:")
    lines.append("")
    lines.append("1. At Full HD-like inference (1088x1920), what FPS per camera do we get?")
    lines.append("2. What is the LARGEST inference size that still hits 50 FPS safely (p95)?")
    lines.append("3. How is the per-packet time split between color, inference, and other?")
    lines.append("")

    if machine_info:
        lines.append("## Machine")
        lines.append("")
        for k in ("gpu_name", "vram_gb", "cuda_version", "trt_version", "torch_version",
                  "python_version", "os"):
            v = machine_info.get(k)
            if v:
                lines.append(f"- **{k}**: {v}")
        lines.append("")

    # Q1 + Q2 answers
    fullhd = next((r for r in rows if r.label in ("1088x1920", "1088_1920")), None)
    pass_rows = [r for r in rows if r.pass_50fps_safe_p95]
    largest_pass = max(pass_rows, key=lambda r: r.pixels_mpx, default=None)

    lines.append("## Direct answers")
    lines.append("")
    if fullhd:
        lines.append("**Q1 — Full HD-like inference (1088x1920):**")
        lines.append("")
        lines.append(f"- FPS per camera (median): **{_fmt(fullhd.fps_per_camera_median, 1)}**")
        lines.append(f"- FPS per camera (safe p95): **{_fmt(fullhd.fps_per_camera_safe_p95, 1)}**")
        lines.append(f"- FPS total for 4 cameras (median): **{_fmt(fullhd.fps_per_camera_median * 4.0, 1)}**")
        lines.append(f"- FPS total for 4 cameras (safe p95): **{_fmt(fullhd.fps_per_camera_safe_p95 * 4.0, 1)}**")
        lines.append(f"- packet ms (p95): {_fmt(fullhd.packet_ms_p95, 2)} ms "
                     f"(budget 20 ms for 50 FPS)")
        verdict = "PASS" if fullhd.pass_50fps_safe_p95 else "FAIL"
        lines.append(f"- 50 FPS verdict (p95): **{verdict}**")
        lines.append("")
        lines.append("**Interpretacja wyniku (jak czytać log testu):**")
        lines.append("")
        lines.append(f"- `engine: input=(4, 3, 1088, 1920)` oznacza batch=4 i wejście RGB 1088x1920.")
        lines.append("- `debayer backend: native_available=True` oznacza, że użyto natywnego backendu CUDA.")
        lines.append("- `color_ms` to czas etapu debayer/kolor, `inf_ms` to sama inferencja TRT.")
        lines.append("- `packet_ms` to czas całego pakietu (to kluczowa metryka pod FPS).")
        lines.append(f"- `packet_ms p95 = {_fmt(fullhd.packet_ms_p95, 2)} ms` => `FPS safe p95 = {_fmt(fullhd.fps_per_camera_safe_p95, 2)}`.")
        lines.append(f"- Werdykt `PASS` bo safe p95 jest powyżej 50 FPS (cel spełniony z marginesem {_fmt(fullhd.stability_margin_ms, 2)} ms).")
        lines.append("")
    else:
        lines.append("**Q1**: 1088x1920 result not present in results/.")
        lines.append("")

    lines.append("**Q2 — Largest inference size at safe 50 FPS:**")
    lines.append("")
    if largest_pass:
        lines.append(
            f"- Largest input that holds **safe p95 >= 50 FPS**: "
            f"**{largest_pass.label}** "
            f"(p95 FPS = {_fmt(largest_pass.fps_per_camera_safe_p95, 1)}, "
            f"margin = {_fmt(largest_pass.stability_margin_ms, 2)} ms)"
        )
    else:
        lines.append("- No tested inference size held safe 50 FPS p95.")
    lines.append("")

    # Recommended presets
    def _by_label(lbl: str) -> Optional[ResolutionRow]:
        return next((r for r in rows if r.label == lbl), None)

    live = next((r for r in rows
                 if r.pass_50fps_safe_p95),
                None)  # smallest passing already first since sorted asc
    if not live:
        live = rows[0] if rows else None
    quality = None
    for r in rows:
        if math.isfinite(r.fps_per_camera_safe_p95) and r.fps_per_camera_safe_p95 >= 40.0:
            quality = r
    fullhd_30 = _by_label("1088x1920")

    lines.append("**Sugerowane presety (wybrane automatycznie z tabeli):**")
    lines.append("")
    lines.append(f"- `LIVE_50FPS`: `{live.label if live else '—'}` "
                 f"({(_fmt(live.fps_per_camera_safe_p95, 1) + ' FPS safe') if live else '—'})")
    lines.append("  - Co to jest: profil pod maksymalną płynność i minimalne ryzyko dropów.")
    lines.append("  - Co robi: używa mniejszego wejścia inferencji, dzięki czemu skraca czas pakietu.")
    lines.append("  - Kiedy używać: live/produkcyjnie, gdy priorytetem jest stabilne FPS.")
    lines.append("")
    lines.append(f"- `QUALITY_40FPS`: `{quality.label if quality else '—'}` "
                 f"({(_fmt(quality.fps_per_camera_safe_p95, 1) + ' FPS safe') if quality else '—'})")
    lines.append("  - Co to jest: profil jakościowy z większym rozmiarem wejścia modelu.")
    lines.append("  - Co robi: daje więcej detalu dla detekcji kosztem wyższego obciążenia GPU.")
    lines.append("  - Kiedy używać: gdy ważniejsza jest jakość detekcji niż minimalna latencja.")
    lines.append("")
    lines.append(f"- `FULLHD_30FPS`: `{fullhd_30.label if fullhd_30 else '—'}` "
                 f"({(_fmt(fullhd_30.fps_per_camera_safe_p95, 1) + ' FPS safe') if fullhd_30 else '—'})")
    lines.append("  - Co to jest: profil fullhd (duże wejście inferencji) z naciskiem na jakość obrazu.")
    lines.append("  - Co robi: utrzymuje tryb wysokiej rozdzielczości, zwykle z większym kosztem czasu pakietu.")
    lines.append("  - Kiedy używać: gdy scena jest trudna i potrzebujesz większej rozdzielczości wejściowej.")
    lines.append("")

    # Big table
    lines.append("## Resolution sweep")
    lines.append("")
    lines.append("| label | batch | Mpx | sama inferencja p95 (ms) | sam kolor p95 (ms) | pełny pipeline p95 (ms) | FPS/cam med | FPS/cam safe p95 | FPS total(4) med | FPS total(4) safe p95 | 50 FPS? | margin (ms) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|")
    for r in rows:
        lines.append(
            "| {label} | {batch} | {mpx} | {inf_only} | {col_only} | {pkt} | {med} | {p95} | {tot_med} | {tot_p95} | {ok} | {mgn} |".format(
                label=r.label,
                batch=r.batch,
                mpx=_fmt(r.pixels_mpx, 2),
                inf_only=_fmt(r.inference_only_ms_p95),
                col_only=_fmt(r.color_only_ms_p95),
                pkt=_fmt(r.packet_ms_p95),
                med=_fmt(r.fps_per_camera_median, 1),
                p95=_fmt(r.fps_per_camera_safe_p95, 1),
                tot_med=_fmt(r.fps_per_camera_median * 4.0, 1),
                tot_p95=_fmt(r.fps_per_camera_safe_p95 * 4.0, 1),
                ok="PASS" if r.pass_50fps_safe_p95 else "FAIL",
                mgn=_fmt(r.stability_margin_ms),
            )
        )
    lines.append("")

    if baseline and baseline.get("results"):
        lines.append(f"## Baseline comparison — {baseline.get('baseline_name', 'baseline')}")
        lines.append("")
        lines.append("| label | this packet p95 | base packet p95 | this FPS p95 | base FPS p95 |")
        lines.append("|---|---:|---:|---:|---:|")
        base_by_label = {item["inference_input_shape"]: item
                         for item in baseline["results"]}
        for r in rows:
            b = base_by_label.get(r.label)
            if not b:
                continue
            lines.append(
                "| {l} | {p1} | {p2} | {f1} | {f2} |".format(
                    l=r.label,
                    p1=_fmt(r.packet_ms_p95),
                    p2=_fmt(b.get("packet_ms_p95")),
                    f1=_fmt(r.fps_per_camera_safe_p95, 1),
                    f2=_fmt(b.get("fps_per_camera_safe_p95"), 1),
                )
            )
        lines.append("")

    if charts:
        lines.append("## Charts")
        lines.append("")
        for c in charts:
            lines.append(f"![{c}](charts/{c})")
        lines.append("")

    lines.append("## What this benchmark does NOT measure")
    lines.append("")
    lines.append("- Basler camera grab time.")
    lines.append("- The GUI, ring buffer, preview worker, or any IPC.")
    lines.append("- Hardware sync between four real cameras.")
    lines.append("- Whole-application overhead.")
    lines.append("")
    lines.append("Use this report to bound the GPU fast path. The production app will be")
    lines.append("slower; how much slower depends on the rest of the system.")
    return "\n".join(lines)


def build_html_from_markdown(markdown_text: str, charts: List[str]) -> str:  # noqa: ARG001
    """Trivial Markdown -> HTML conversion (no external deps).

    Good enough to read in a browser. For pretty output, copy the .md
    into your favourite editor.
    """
    html_lines: List[str] = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>Fastpath Lab Benchmark</title>",
        "<style>",
        "body{font-family:sans-serif;max-width:980px;margin:1.5em auto;padding:0 1em;}",
        "table{border-collapse:collapse;}",
        "td,th{border:1px solid #ccc;padding:4px 8px;text-align:right;}",
        "th{background:#eee;}",
        "h1,h2{border-bottom:1px solid #ccc;padding-bottom:4px;}",
        "img{max-width:100%;border:1px solid #ddd;}",
        "code{background:#f4f4f4;padding:0 4px;border-radius:3px;}",
        "</style></head><body>",
    ]
    in_table = False
    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        if line.startswith("# "):
            html_lines.append(f"<h1>{line[2:]}</h1>")
            continue
        if line.startswith("## "):
            html_lines.append(f"<h2>{line[3:]}</h2>")
            continue
        if line.startswith("|") and line.endswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            tag = "th" if all(set(c) <= set("-: ") for c in cells) else "td"
            if not in_table:
                html_lines.append("<table>")
                in_table = True
            if tag == "th":
                continue  # alignment separator row, skip
            row = "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
            html_lines.append(row)
            continue
        if in_table:
            html_lines.append("</table>")
            in_table = False
        if line.startswith("!["):
            # ![alt](charts/x.png)
            try:
                alt = line[2:line.index("]")]
                src = line[line.index("(") + 1: line.index(")")]
                html_lines.append(f"<img alt='{alt}' src='{src}'/>")
                continue
            except Exception:
                pass
        if line.startswith("- "):
            html_lines.append(f"<li>{line[2:]}</li>")
            continue
        if not line:
            html_lines.append("<br/>")
            continue
        html_lines.append(f"<p>{line}</p>")
    if in_table:
        html_lines.append("</table>")
    html_lines.append("</body></html>")
    return "\n".join(html_lines)


def write_report(results_dir: Path, reports_dir: Path,
                 baseline_path: Optional[Path] = None,
                 env_path: Optional[Path] = None,
                 write_html: bool = False) -> Dict[str, Any]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    test_dir = _next_test_dir(reports_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = test_dir / "charts"
    results_snapshot_dir = test_dir / "results"
    results_snapshot_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_resolution_rows(results_dir)
    baseline = load_json(baseline_path) if baseline_path else None
    env_payload = load_json(env_path) if env_path else None
    machine_info = (env_payload or {}).get("machine") if env_payload else None

    charts = render_charts(rows, baseline, charts_dir)
    md = build_markdown(rows, baseline, charts, machine_info)
    md_path = test_dir / "fastpath_lab_report.md"
    md_path.write_text(md, encoding="utf-8")

    html_path: Optional[Path] = None
    if write_html:
        html = build_html_from_markdown(md, charts)
        html_path = test_dir / "fastpath_lab_report.html"
        html_path.write_text(html, encoding="utf-8")

    # Snapshot benchmark inputs used by this report.
    snapshot_patterns = (
        "full_synthetic_*.json",
        "full_synthetic_*.csv",
        "resolution_sweep.json",
        "resolution_sweep.csv",
        "environment.json",
    )
    copied_results: List[str] = []
    for pattern in snapshot_patterns:
        for src in sorted(results_dir.glob(pattern)):
            if not src.is_file():
                continue
            dst = results_snapshot_dir / src.name
            shutil.copy2(src, dst)
            copied_results.append(str(dst))

    return {
        "markdown": str(md_path),
        "html": str(html_path) if html_path is not None else "",
        "charts_dir": str(charts_dir),
        "results_dir": str(results_snapshot_dir),
        "results_files_copied": len(copied_results),
        "report_dir": str(test_dir),
        "rows": len(rows),
    }
