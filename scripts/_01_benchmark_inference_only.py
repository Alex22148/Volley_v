from __future__ import annotations

import runpy
from pathlib import Path


def main() -> int:
    target = (
        Path(__file__).resolve().parents[1]
        / "dist"
        / "fastpath_lab_benchmark"
        / "scripts"
        / "01_benchmark_inference_only.py"
    )
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

