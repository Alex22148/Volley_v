#!/usr/bin/env python
"""
VolleyHub – Recording Stability Report

Użycie:
    python recording_report.py path/do/recording_stats.json

Wygeneruje pliki:
    - recording_report.png  (wykresy)
    - recording_report.txt  (tekstowe podsumowanie)
w tym samym katalogu co plik JSON.
"""

import json
import statistics
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt

# --- KONFIGURACJA -----------------------------------------------------------

# Docelowy FPS – dostosuj do swojej konfiguracji (np. 50 dla 4x50fps)
TARGET_FPS = 50.0

# Ile KLATEK reprezentuje 1 wpis w "files"
# U Ciebie: 1 record = 128 klatek
FRAMES_PER_RECORD = 512


# --- Wczytanie i przygotowanie danych --------------------------------------


def load_stats(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("Plik recording_stats.json powinien zawierać listę rekordów.")

    # posortuj po czasie (na wszelki wypadek)
    data.sort(key=lambda r: float(r.get("t_unix", 0.0)))

    # policz czas relatywny, jeśli nie ma
    t0 = float(data[0].get("t_unix", 0.0))
    for rec in data:
        if rec.get("elapsed_s") is None:
            rec["elapsed_s"] = float(rec.get("t_unix", 0.0)) - t0
    return data


def extract_roles(data: List[Dict[str, Any]]) -> List[str]:
    roles = set()
    for rec in data:
        files = rec.get("files") or {}
        roles.update(files.keys())
    return sorted(roles)


# --- Analiza: rekordy -> klatki -> FPS -------------------------------------


def compute_intervals(data: List[Dict[str, Any]], roles: List[str]):
    """
    Zwraca:
      - intervals: lista odcinków czasu z wyliczonym FPS/efektywnością
      - summary: metryki globalne per rola

    Uwaga: 1 wpis w 'files' = FRAMES_PER_RECORD klatek.
    """
    intervals: List[Dict[str, Any]] = []

    # per-odcinek czasu (kolejne rekordy)
    for prev, cur in zip(data, data[1:]):
        t0 = float(prev["elapsed_s"])
        t1 = float(cur["elapsed_s"])
        dt = max(1e-6, t1 - t0)
        seg = {"t0": t0, "t1": t1, "dt": dt, "per_role": {}}

        for r in roles:
            f0 = (prev.get("files") or {}).get(r, 0)
            f1 = (cur.get("files") or {}).get(r, f0)
            df_records = max(0, int(f1) - int(f0))        # przyrost rekordów
            frames = df_records * FRAMES_PER_RECORD       # przyrost KLATEK
            fps = frames / dt                             # realny FPS
            eff = fps / TARGET_FPS if TARGET_FPS > 0 else 0.0

            seg["per_role"][r] = {
                "df_records": df_records,
                "frames": frames,
                "fps": fps,
                "eff": eff,
            }

        intervals.append(seg)

    # podsumowanie globalne
    total_duration = max(
        1e-6, float(data[-1]["elapsed_s"]) - float(data[0]["elapsed_s"])
    )
    summary: Dict[str, Dict[str, float]] = {}

    for r in roles:
        f_start = (data[0].get("files") or {}).get(r, 0)
        f_end = (data[-1].get("files") or {}).get(r, 0)
        total_records = max(0, int(f_end) - int(f_start))
        total_frames = total_records * FRAMES_PER_RECORD
        avg_fps = total_frames / total_duration

        fps_samples = [seg["per_role"][r]["fps"] for seg in intervals if seg["dt"] > 0]
        if fps_samples:
            mean_fps = sum(fps_samples) / len(fps_samples)
            std_fps = (
                statistics.pstdev(fps_samples) if len(fps_samples) > 1 else 0.0
            )
            jitter = std_fps / mean_fps if mean_fps > 0 else 0.0
        else:
            mean_fps = 0.0
            std_fps = 0.0
            jitter = 1.0

        efficiency = avg_fps / TARGET_FPS if TARGET_FPS > 0 else 0.0
        efficiency = max(0.0, min(1.2, efficiency))  # lekki margines powyżej 100%

        # Stabilność = kombinacja efektywności i "płynności" (małego jittera)
        smoothness = max(0.0, 1.0 - min(jitter, 2.0) / 2.0)
        stability_score = 0.7 * min(efficiency, 1.0) + 0.3 * smoothness

        summary[r] = {
            "total_records": total_records,
            "total_frames": total_frames,
            "duration_s": total_duration,
            "avg_fps": avg_fps,
            "efficiency": efficiency,
            "mean_fps": mean_fps,
            "std_fps": std_fps,
            "jitter": jitter,
            "smoothness": smoothness,
            "stability_score": stability_score,
        }

    return intervals, summary


# --- Marketingowa klasyfikacja ----------------------------------------------


def classify_session(score: float) -> str:
    """Marketingowa klasyfikacja stabilności – bez emoji, przyjazna dla czcionek."""
    if score >= 0.9:
        return "[5/5] Bardzo stabilny zapis"
    if score >= 0.8:
        return "[4/5] Stabilny zapis (drobne wahania)"
    if score >= 0.65:
        return "[3/5] Umiarkowana stabilność (widoczne dropy)"
    if score >= 0.5:
        return "[2/5] Niestabilny zapis (częste straty klatek)"
    return "[1/5] Krytycznie niestabilny zapis"



# --- Rysowanie wykresów -----------------------------------------------------


def make_plot(data: List[Dict[str, Any]],
              roles,
              intervals,
              summary,
              out_png: Path):
    from matplotlib.gridspec import GridSpec

    # Brandowe kolory VolleyHub (delikatny "marketingowy" styl)
    palette = {
        "CENTER_L": "#4B83F5",
        "CENTER_R": "#2ECC71",
        "LEFT": "#F39C12",
        "RIGHT": "#E74C3C",
    }
    extra_colors = ["#9B59B6", "#1ABC9C", "#34495E", "#E67E22"]

    # ---- globalny score do podpisu ----
    global_scores = [v["stability_score"] for v in summary.values()] or [0.0]
    global_score = sum(global_scores) / len(global_scores)
    global_label = classify_session(global_score)

    fig = plt.figure(figsize=(12, 7), dpi=120)
    fig.patch.set_facecolor("#F8F9FB")

    # siatka 2x2: góra – wykresy, dół – słupki + tekst
    gs = GridSpec(2, 2, height_ratios=[1.0, 1.3], figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])  # liczba rekordów
    ax2 = fig.add_subplot(gs[0, 1])  # FPS
    ax3 = fig.add_subplot(gs[1, 0])  # słupki
    ax4 = fig.add_subplot(gs[1, 1])  # blok tekstowy

    # --- PANEL 1: Cumulative records ---
    ax1.set_title("Czas nagrania vs liczba rekordów", fontsize=11, weight="bold")
    t = [float(r["elapsed_s"]) for r in data]
    files_per_role = {r: [(rec.get("files") or {}).get(r, 0) for rec in data]
                      for r in roles}

    for i, r in enumerate(roles):
        color = palette.get(r, extra_colors[i % len(extra_colors)])
        ax1.plot(t, files_per_role[r], label=r, color=color, linewidth=2)

    ax1.set_xlabel("Czas od startu [s]")
    ax1.set_ylabel("Liczba rekordów (plików)")
    ax1.grid(True, alpha=0.2)
    ax1.legend(frameon=False, fontsize=8)
    ax1.text(
        0.01,
        0.97,
        f"1 rekord = {FRAMES_PER_RECORD:.0f} klatek",
        transform=ax1.transAxes,
        va="top",
        fontsize=8,
        color="#555555",
    )

    # --- PANEL 2: Instantaneous FPS ---
    ax2.set_title("FPS w czasie (z przyrostu rekordów)", fontsize=11, weight="bold")

    if intervals:
        mid_ts = [(seg["t0"] + seg["t1"]) / 2.0 for seg in intervals]
        for i, r in enumerate(roles):
            color = palette.get(r, extra_colors[i % len(extra_colors)])
            fps_vals = [seg["per_role"][r]["fps"] for seg in intervals]
            ax2.plot(mid_ts, fps_vals, label=r, color=color,
                     linewidth=1.8, marker="o")

        total_duration = float(data[-1]["elapsed_s"]) - float(data[0]["elapsed_s"])
        if total_duration > 0:
            ax2.set_xlim(0, total_duration * 1.02)

    if TARGET_FPS > 0:
        ax2.axhline(
            TARGET_FPS,
            color="#95A5A6",
            linestyle="--",
            linewidth=1,
            label=f"Cel FPS = {TARGET_FPS:.0f}",
        )

    ax2.set_xlabel("Czas od startu [s]")
    ax2.set_ylabel("FPS (realne klatki/s)")
    ax2.grid(True, alpha=0.2)
    ax2.legend(frameon=False, fontsize=8)

    # --- PANEL 3: Marketingowy "score" (słupki) ---
    ax3.set_title("Stabilność zapisu per kamera", fontsize=12, weight="bold")
    bars_roles = list(roles)
    scores = [summary[r]["stability_score"] * 100 for r in bars_roles]
    effs = [summary[r]["efficiency"] * 100 for r in bars_roles]

    x = range(len(bars_roles))
    width = 0.35

    for i, r in enumerate(bars_roles):
        color = palette.get(r, extra_colors[i % len(extra_colors)])
        ax3.bar(i - width / 2, effs[i], width=width, color=color, alpha=0.6)
        ax3.bar(i + width / 2, scores[i], width=width, color=color, alpha=0.9)

    for i, r in enumerate(bars_roles):
        ax3.text(
            i - width / 2,
            effs[i] - 15,
            f"{effs[i]:.0f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#555555",
        )
        ax3.text(
            i + width / 2,
            scores[i] - 15,
            f"{scores[i]:.0f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#111111",
        )

    ax3.set_xticks(list(x))
    ax3.set_xticklabels(bars_roles)
    ax3.set_ylabel("Procent / score")
    ax3.grid(axis="y", alpha=0.15)

    # --- PANEL 4: Tekstowe podsumowanie (to co w .txt) ---
    ax4.axis("off")
    ax4.set_title("Podsumowanie liczbowe", fontsize=11, weight="bold", loc="left")

    y = 0.95
    ax4.text(
        0.0,
        y,
        f"Ocena ogólna: {global_label}",
        transform=ax4.transAxes,
        fontsize=10,
        weight="bold",
        va="top",
    )
    y -= 0.10
    ax4.text(
        0.0,
        y,
        f"Globalny score stabilności: {global_score*100:.1f}%",
        transform=ax4.transAxes,
        fontsize=9,
        va="top",
    )
    y -= 0.12
    ax4.text(
        0.0,
        y,
        f"Założenie: 1 wpis 'files' = {FRAMES_PER_RECORD:.0f} klatek, cel FPS = {TARGET_FPS:.0f}",
        transform=ax4.transAxes,
        fontsize=8,
        va="top",
        color="#555555",
    )
    y -= 0.12
    ax4.text(
        0.0,
        y,
        "Kamery:",
        transform=ax4.transAxes,
        fontsize=9,
        weight="bold",
        va="top",
    )
    y -= 0.08

    for r in bars_roles:
        m = summary[r]
        line1 = (
            f"{r}: {m['avg_fps']:.1f} FPS "
            f"({m['efficiency']*100:.0f}% celu)"
        )
        line2 = (
            f"     jitter {m['jitter']*100:.1f}%, "
            f"score {m['stability_score']*100:.1f}%"
        )
        ax4.text(
            0.0,
            y,
            line1,
            transform=ax4.transAxes,
            fontsize=8,
            va="top",
        )
        y -= 0.07
        ax4.text(
            0.0,
            y,
            line2,
            transform=ax4.transAxes,
            fontsize=8,
            va="top",
            color="#555555",
        )
        y -= 0.09

    # --- tytuł globalny i zapis ---
    fig.suptitle(
        "VolleyHub – Raport stabilności zapisu",
        fontsize=14,
        weight="bold",
        color="#2C3E50",
    )
    fig.tight_layout(rect=[0.02, 0.04, 0.98, 0.93])

    fig.savefig(out_png, facecolor=fig.get_facecolor())
    plt.close(fig)



# --- Raport tekstowy --------------------------------------------------------


def write_text_report(summary: Dict[str, Any], out_txt: Path):
    lines: List[str] = []
    lines.append("VolleyHub – Raport stabilności zapisu")
    lines.append("=" * 50)
    lines.append("")
    lines.append(f"Założenie: 1 wpis 'files' = {FRAMES_PER_RECORD:.0f} klatek")
    lines.append(f"Docelowy FPS: {TARGET_FPS:.0f}")
    lines.append("")

    global_scores = [v["stability_score"] for v in summary.values()] or [0.0]
    global_score = sum(global_scores) / len(global_scores)
    lines.append(f"Ocena ogólna: {classify_session(global_score)}")
    lines.append(f"Globalny score stabilności: {global_score*100:.1f}%")
    lines.append("")

    for r, m in summary.items():
        lines.append(f"[{r}]")
        lines.append(f"  Liczba rekordów (pliki): {m['total_records']}")
        lines.append(f"  Liczba klatek (szac.): {m['total_frames']:.0f}")
        lines.append(f"  Czas nagrania: {m['duration_s']:.1f} s")
        lines.append(f"  Średni FPS: {m['avg_fps']:.2f}")
        lines.append(f"  Efektywność (vs cel): {m['efficiency']*100:.1f}%")
        lines.append(f"  Jitter FPS: {m['jitter']*100:.1f}%")
        lines.append(f"  Score stabilności: {m['stability_score']*100:.1f}%")
        lines.append("")

    out_txt.write_text("\n".join(lines), encoding="utf-8")

# --- main -------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Generuje marketingowy raport stabilności zapisu "
            "z pliku recording_stats.json"
        )
    )
    parser.add_argument(
        "json_path",
        nargs="?",
        default="recording_stats.json",
        help="Ścieżka do pliku recording_stats.json (domyślnie: ./recording_stats.json)",
    )

    args = parser.parse_args()
    json_path = Path(args.json_path)
    if not json_path.exists():
        raise SystemExit(f"Nie znaleziono pliku: {json_path}")

    data = load_stats(json_path)
    roles = extract_roles(data)
    intervals, summary = compute_intervals(data, roles)

    out_png = json_path.with_name("recording_report.png")
    out_txt = json_path.with_name("recording_report.txt")

    make_plot(data, roles, intervals, summary, out_png)
    write_text_report(summary, out_txt)

    print(f"✅ Raport graficzny zapisano jako: {out_png}")
    print(f"✅ Raport tekstowy zapisano jako: {out_txt}")


if __name__ == "__main__":
    main()
