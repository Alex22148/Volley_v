"""Convenience: run only one variant from the sweep config (by name)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("synthetic", "basler"), default="synthetic")
    parser.add_argument("--name", required=True, help="variant name from configs/benchmark_config.yaml")
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--camera-roles", type=Path, default=PACKAGE_ROOT / "configs" / "camera_roles.json")
    args = parser.parse_args()

    # Filter the YAML in-memory and dispatch into run_fastpath_sweep.main
    import yaml  # type: ignore
    cfg_path = PACKAGE_ROOT / "configs" / "benchmark_config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    sweep = [e for e in cfg.get("sweep", []) if str(e.get("name")) == args.name]
    if not sweep:
        print(f"no sweep entry named {args.name!r}; available: "
              f"{[e.get('name') for e in cfg.get('sweep', [])]}")
        return 2

    # Write a temp config and pass it via env (simplest path).
    tmp = PACKAGE_ROOT / "configs" / "_single_case_config.yaml"
    cfg["sweep"] = sweep
    tmp.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    argv = [
        "--mode", args.mode,
        "--duration-s", str(args.duration_s),
        "--config", str(tmp),
    ]
    if args.mode == "basler":
        argv += ["--camera-roles", str(args.camera_roles)]
    from scripts1.run_fastpath_sweep import main as run_main
    return run_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
