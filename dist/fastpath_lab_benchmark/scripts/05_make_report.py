"""05_make_report.py â€” build the lab benchmark report.

WHAT THIS SCRIPT DOES:
    Reads every results/full_synthetic_*.json that the sweep scripts
    produced, plus results/environment.json (if present), plus
    configs/baseline_rtx2080super.json (if present), and writes:

        reports/fastpath_lab_report.md
        reports/fastpath_lab_report.html
        reports/charts/*.png   (only if matplotlib is installed)

    The report directly answers:
        Q1. At Full HD-like inference (1088x1920), what FPS per camera?
        Q2. What is the largest inference size that hits safe 50 FPS?
        Q3. How is the per-packet cost split (color/inference/other)?

WHAT IT DOES NOT DO:
    * No new benchmark runs. It only aggregates what is already in
      results/.
    * No comparison to the production app's runtime â€” this is the GPU
      fast-path bound only.
"""
from __future__ import annotations

import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_ROOT))

from src.report_builder import write_report  # noqa: E402

# ----------------------------- static config (no argparse)
RESULTS_DIR = PKG_ROOT / "reports"
REPORTS_DIR = PKG_ROOT / "reports"
BASELINE_PATH = PKG_ROOT / "configs" / "baseline_rtx2080super.json"
ENV_PATH = PKG_ROOT / "reports" / "environment.json"
WITH_HTML = False


def main() -> int:
    print("=" * 72)
    print("fastpath_lab_benchmark / 05_make_report.py")
    print("WHAT: aggregates results/ into reports/ (Markdown + HTML + charts).")
    print("WHAT NOT: does not run any new benchmarks.")
    print("=" * 72)

    out = write_report(
        results_dir=Path(RESULTS_DIR),
        reports_dir=Path(REPORTS_DIR),
        baseline_path=Path(BASELINE_PATH) if Path(BASELINE_PATH).exists() else None,
        env_path=Path(ENV_PATH) if Path(ENV_PATH).exists() else None,
        write_html=bool(WITH_HTML),
    )

    print(f"\nrows used    : {out['rows']}")
    print(f"markdown out : {out['markdown']}")
    if out["html"]:
        print(f"html out     : {out['html']}")
    print(f"charts dir   : {out['charts_dir']}")
    print(f"results dir  : {out['results_dir']}")
    if int(out["rows"]) == 0:
        print("\n[WARN] No resolution rows found. Did you run "
              "scripts/04_resolution_sweep.py first?")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


