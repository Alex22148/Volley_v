from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time

# ----------------------------- static config (no argparse)
# Preset:
#   "smoke"   -> check_env + benchmark + compare + report
#   "full"    -> check_env + export + benchmark + compare + report
#   "custom"  -> użyj flag RUN_*
PIPELINE_MODE = "smoke"

RUN_CHECK_ENV = True
RUN_EXPORT_STATIC = False
RUN_EXPORT_INT8 = False
RUN_BENCHMARK = True
RUN_COMPARE_BACKENDS = True
RUN_REPORT = True


def run_step(name: str, cmd: list[str]) -> tuple[bool, float]:
    print(f"\n[STEP] {name}")
    print(f"[CMD ] {' '.join(cmd)}")
    started = time.perf_counter()
    proc = subprocess.run(cmd)
    elapsed = time.perf_counter() - started
    ok = proc.returncode == 0
    print(f"[STEP] {name}: {'OK' if ok else 'FAIL'} | {elapsed:.1f}s | code={proc.returncode}")
    return ok, elapsed


def main() -> int:
    scripts_dir = Path(__file__).resolve().parent
    py = sys.executable

    if PIPELINE_MODE == "smoke":
        do_check_env = True
        do_export = False
        do_export_int8 = False
        do_benchmark = True
        do_compare = True
        do_report = True
    elif PIPELINE_MODE == "full":
        do_check_env = True
        do_export = True
        do_export_int8 = False
        do_benchmark = True
        do_compare = True
        do_report = True
    else:
        do_check_env = bool(RUN_CHECK_ENV)
        do_export = bool(RUN_EXPORT_STATIC)
        do_export_int8 = bool(RUN_EXPORT_INT8)
        do_benchmark = bool(RUN_BENCHMARK)
        do_compare = bool(RUN_COMPARE_BACKENDS)
        do_report = bool(RUN_REPORT)

    steps: list[tuple[str, list[str], bool]] = []
    if do_check_env:
        steps.append(("check_env", [py, str(scripts_dir / "00_check_env.py")], True))
    if do_export:
        steps.append(("export_static_engine", [py, str(scripts_dir / "export2engine.py")], False))
        steps.append(("export_static_onnx", [py, str(scripts_dir / "export2onnx.py")], False))
    if do_export_int8:
        steps.append(
            ("export_int8_engine", [py, str(scripts_dir / "export_one_pt_to_many_engines_int8_v1.py")], False)
        )
    if do_benchmark:
        steps.append(("benchmark_08", [py, str(scripts_dir / "08_final_benchmark.py")], False))
    if do_compare:
        steps.append(("compare_10", [py, str(scripts_dir / "10_compare_pt_onnx_engine_detections.py")], False))
    if do_report:
        steps.append(("report_09", [py, str(scripts_dir / "09_make_final_report.py")], False))

    if not steps:
        print("[PIPELINE] Nic do uruchomienia. Podaj flagi albo użyj --smoke.")
        return 2

    print("=" * 88)
    print("fastpath_lab_benchmark / 11_run_pipeline.py")
    print(f"Plan: {', '.join(s[0] for s in steps)}")
    print("=" * 88)

    failed = []
    total_started = time.perf_counter()

    for name, cmd, hard_gate in steps:
        ok, elapsed = run_step(name, cmd)
        if not ok:
            failed.append((name, elapsed))
            # check_env jest bramką: jeśli fail -> stop.
            if hard_gate:
                print("[PIPELINE] Przerwano: check_env nie przeszedł.")
                break
            # dla pozostałych kroków też przerywamy, bo następne zależą od poprzednich.
            print("[PIPELINE] Przerwano: krok nie przeszedł.")
            break

    total_elapsed = time.perf_counter() - total_started
    success = len(failed) == 0

    print("\n" + "=" * 88)
    print("PIPELINE SUMMARY")
    if success:
        print("STATUS: PASS")
    else:
        print("STATUS: FAIL")
        for name, elapsed in failed:
            print(f" - failed step: {name} ({elapsed:.1f}s)")
    print(f"TOTAL: {total_elapsed:.1f}s")
    print("=" * 88)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
