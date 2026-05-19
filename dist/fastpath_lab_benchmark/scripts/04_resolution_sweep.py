"""04_resolution_sweep.py â€” sweep full synthetic path across resolutions.

WHAT THIS SCRIPT DOES:
    Runs scripts/03_benchmark_full_synthetic_path.py for every inference
    size we care about (640x640, 960x960, 1280x1280, 1088x1920) and
    rolls up the verdicts into a single table.

WHAT IT DOES NOT DO:
    * It does not rebuild engines. If an engine is missing, that row
      is marked as skipped.
    * It does not touch cameras, GUI, ring buffer, or production code.

INPUT:
    --engines-dir   default: engines/
    --batches       comma-separated batch sizes (default: 4)
    --duration-s    per-resolution duration (default 30)
    --warmup        per-resolution warmup (default 10)

OUTPUT:
    results/resolution_sweep.json
    results/resolution_sweep.csv
    (also: refreshed inference_only/color_only/full_synthetic per combo)
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = PKG_ROOT / "scripts"


# Default sweep: (label, capture HW, inference HW)
DEFAULT_SWEEP = [
    ("v640", (640, 640), (640, 640)),
    ("v960", (960, 960), (960, 960)),
    ("1280x1280", (1280, 1920), (1280, 1920)),
    ("v1088x1920", (1088, 1920), (1088, 1920)),
]

# ----------------------------- static config (no argparse)
ENGINES_DIR = PKG_ROOT / "engines/static"
BATCHES = [4,8,12]
DURATION_S = 40.0
WARMUP = 10
BAYER_PATTERN = "RG"
RESULTS_DIR = PKG_ROOT / "reports"


def _engine_candidates(label: str, batch: int) -> list[str]:
    # Canonical naming used in this benchmark package.
    return [f"best__fp16_{label}_b{batch}.engine"]


def _resolve_engine_path(engines_dir: Path, label: str, batch: int) -> Path | None:
    for name in _engine_candidates(label, batch):
        p = engines_dir / name
        if p.exists():
            return p
    return None


def _parse_batches(raw: str) -> list[int]:
    out: list[int] = []
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        val = int(part)
        if val <= 0:
            raise ValueError("batch must be > 0")
        out.append(val)
    if not out:
        raise ValueError("empty batch list")
    return sorted(set(out))


def main() -> int:
    batches = sorted(set(int(x) for x in BATCHES if int(x) > 0))
    if not batches:
        raise SystemExit("BATCHES cannot be empty.")

    print("=" * 72)
    print("fastpath_lab_benchmark / 04_resolution_sweep.py")
    print("WHAT: runs the full synthetic path for 640, 960, 1280, 1088x1920.")
    print("WHAT NOT: does not export engines, does not touch hardware.")
    print(f"INPUT:  engines_dir={ENGINES_DIR}  duration={DURATION_S}s")
    print(f"        warmup={WARMUP}  pattern={BAYER_PATTERN}  batches={batches}")
    print("=" * 72)

    rows: list[dict] = []
    engines_dir = Path(ENGINES_DIR)
    for batch in batches:
        for label, cap_hw, in_hw in DEFAULT_SWEEP:
            eng_path = _resolve_engine_path(engines_dir, label, batch)
            row = {
                "engine": str(eng_path) if eng_path is not None else "",
                "label": label,
                "batch": batch,
                "capture_shape": list(cap_hw),
                "inference_input_shape": list(in_hw),
            }
            if eng_path is None:
                row["skipped"] = True
                row["reason"] = "engine_missing"
                print(f"[SKIP] {label} b{batch}: engine missing. Tried: "
                      f"{', '.join(_engine_candidates(label, batch))}")
                rows.append(row)
                continue

            cmd_inf = [
                sys.executable,
                str(SCRIPTS / "01_benchmark_inference_only.py"),
                "--engine", str(eng_path),
                "--input-shape", str(in_hw[0]), str(in_hw[1]),
                "--batch", str(batch),
                "--duration-s", str(DURATION_S),
                "--warmup", str(WARMUP),
                "--results-dir", str(RESULTS_DIR),
            ]
            print(f"\n[RUN ] inference-only {label} b{batch}  ->  {' '.join(cmd_inf[1:])}")
            rc_inf = subprocess.run(cmd_inf).returncode
            row["inference_only_rc"] = rc_inf
            if rc_inf != 0:
                row["skipped"] = True
                row["reason"] = f"inference_only_failed_rc={rc_inf}"
                rows.append(row)
                continue

            cmd_col = [
                sys.executable,
                str(SCRIPTS / "02_benchmark_color_only.py"),
                "--output-shape", str(in_hw[0]), str(in_hw[1]),
                "--batch", str(batch),
                "--duration-s", str(DURATION_S),
                "--warmup", str(WARMUP),
                "--bayer-pattern", BAYER_PATTERN,
                "--results-dir", str(RESULTS_DIR),
            ]
            print(f"[RUN ] color-only     {label} b{batch}  ->  {' '.join(cmd_col[1:])}")
            rc_col = subprocess.run(cmd_col).returncode
            row["color_only_rc"] = rc_col
            if rc_col != 0:
                row["skipped"] = True
                row["reason"] = f"color_only_failed_rc={rc_col}"
                rows.append(row)
                continue

            cmd = [
                sys.executable,
                str(SCRIPTS / "03_benchmark_full_synthetic_path.py"),
                "--engine", str(eng_path),
                "--capture-shape", str(cap_hw[0]), str(cap_hw[1]),
                "--input-shape", str(in_hw[0]), str(in_hw[1]),
                "--batch", str(batch),
                "--duration-s", str(DURATION_S),
                "--warmup", str(WARMUP),
                "--bayer-pattern", BAYER_PATTERN,
                "--results-dir", str(RESULTS_DIR),
            ]
            print(f"\n[RUN ] {label} b{batch}  ->  {' '.join(cmd[1:])}")
            result = subprocess.run(cmd)
            row["full_synthetic_rc"] = result.returncode
            if result.returncode != 0:
                row["skipped"] = True
                row["reason"] = f"subprocess_failed_rc={result.returncode}"
                rows.append(row)
                continue

            inf_json_path = Path(RESULTS_DIR) / f"inference_only_{label}_b{batch}.json"
            col_json_path = Path(RESULTS_DIR) / f"color_only_{label}_b{batch}.json"
            json_path = Path(RESULTS_DIR) / f"full_synthetic_{label}_b{batch}.json"
            try:
                inf_payload = json.loads(inf_json_path.read_text(encoding="utf-8"))
                col_payload = json.loads(col_json_path.read_text(encoding="utf-8"))
                payload = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception as exc:
                row["skipped"] = True
                row["reason"] = f"read_results_failed: {exc!r}"
                rows.append(row)
                continue

            inf_stage = (inf_payload.get("aggregate") or {}).get("stage_stats", {}).get("inference_ms", {})
            col_stage = (col_payload.get("aggregate") or {}).get("stage_stats", {}).get("color_ms", {})
            verdict = payload["aggregate"]["verdict"]
            stages = payload["aggregate"].get("stage_stats", {})
            row.update({
                "skipped": False,
                "pixels_mpx": (in_hw[0] * in_hw[1]) / 1.0e6,
                "inference_only_ms_p95": inf_stage.get("p95_ms"),
                "color_only_ms_p95": col_stage.get("p95_ms"),
                "color_ms_p95": stages.get("color_ms", {}).get("p95_ms"),
                "inference_ms_p95": stages.get("inference_ms", {}).get("p95_ms"),
                "packet_ms_median": verdict.get("packet_ms_median"),
                "packet_ms_p95": verdict.get("packet_ms_p95"),
                "fps_per_camera_median": verdict.get("fps_per_camera_median"),
                "fps_per_camera_safe_p95": verdict.get("fps_per_camera_safe_p95"),
                "pass_50fps_safe_p95": verdict.get("pass_50fps_safe_p95"),
                "stability_margin_ms": verdict.get("stability_margin_ms"),
            })
            rows.append(row)

    print("\n=== sweep summary ===")
    print("{:>11}  {:>5}  {:>5}  {:>9}  {:>9}  {:>9}  {:>5}".format(
        "label", "batch", "mpx", "pkt_p95", "fps_med", "fps_p95", "50fps"))
    for r in rows:
        if r.get("skipped"):
            print(f"{r['label']:>11}  b{r.get('batch', '?'):>2}  [skipped: {r.get('reason')}]")
            continue
        print("{:>11}  {:>5}  {:>5.2f}  {:>9.2f}  {:>9.2f}  {:>9.2f}  {:>5}".format(
            r["label"], r["batch"], r["pixels_mpx"], r["packet_ms_p95"],
            r["fps_per_camera_median"], r["fps_per_camera_safe_p95"],
            "PASS" if r["pass_50fps_safe_p95"] else "FAIL"))

    out = {
        "what": "resolution_sweep",
        "batches": batches,
        "duration_s": DURATION_S,
        "warmup": WARMUP,
        "rows": rows,
    }
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "resolution_sweep.json"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    csv_path = results_dir / "resolution_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "label", "batch", "pixels_mpx", "color_ms_p95", "inference_ms_p95",
            "inference_only_ms_p95", "color_only_ms_p95",
            "packet_ms_median", "packet_ms_p95",
            "fps_per_camera_median", "fps_per_camera_safe_p95",
            "pass_50fps_safe_p95", "stability_margin_ms", "skipped", "reason",
        ])
        for r in rows:
            w.writerow([
                r.get("label"), r.get("batch"), r.get("pixels_mpx"), r.get("color_ms_p95"),
                r.get("inference_ms_p95"), r.get("inference_only_ms_p95"),
                r.get("color_only_ms_p95"), r.get("packet_ms_median"),
                r.get("packet_ms_p95"), r.get("fps_per_camera_median"),
                r.get("fps_per_camera_safe_p95"), r.get("pass_50fps_safe_p95"),
                r.get("stability_margin_ms"), r.get("skipped"), r.get("reason"),
            ])
    print(f"\n-> wrote {json_path}")
    print(f"-> wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
