# Operator Smoke Diagnostics

This document describes the recommended diagnostic commands for checking a
VolleyHub runtime environment before a controlled production run.

The diagnostics do not change production defaults. The production fast path
continues to use the native preprocessing path unless explicitly configured
otherwise. The unified image processor can be enabled for production testing
with:

```powershell
$env:VOLLEYHUB_USE_UNIFIED_IMAGE_PROCESSOR="1"
```

## Basic Readiness Check

Use this first when validating a workstation or deployment environment:

```powershell
.venv\Scripts\python.exe -m src.runtime_diagnostics.runtime_readiness_check --json --markdown --output-dir reports/runtime_readiness
```

This checks Python, platform details, numpy, OpenCV, torch, CUDA availability,
TensorRT availability, pypylon availability, key runtime imports, and an
optional engine path.

## Diagnostic Report With Preprocessing Compare

Use this for the normal operator report on a machine without requiring CUDA,
TensorRT, cameras, or a model:

```powershell
.venv\Scripts\python.exe -m src.runtime_diagnostics.diagnostic_report --output-dir reports/diagnostic_report --run-preprocess-compare --width 128 --height 96 --num-frames 5 --imgsz 64 --device cpu --native-backend cpu
```

This runs the readiness check, operator smoke, and native-vs-unified
preprocessing comparison on small synthetic frames.

## Hardware Smoke Without Inference

Use this when you want the hardware smoke report structure but do not have an
engine available yet:

```powershell
.venv\Scripts\python.exe -m src.runtime_diagnostics.hardware_smoke_test --output-dir reports/hardware_smoke_no_engine --skip-inference --preprocess-backend compare --width 128 --height 96 --num-frames 5 --imgsz 64 --batch-size 4 --device cpu
```

This validates synthetic preprocessing only. Missing engine or skipped
inference is reported as a warning or skipped check, not as a hardware failure.

## Hardware Smoke With Engine

Use this only when CUDA and a TensorRT engine are available:

```powershell
.venv\Scripts\python.exe -m src.runtime_diagnostics.hardware_smoke_test --engine-path path\to\model.engine --output-dir reports/hardware_smoke_engine --preprocess-backend native --width 128 --height 96 --num-frames 5 --imgsz 64 --batch-size 4 --device auto
```

For a unified preprocessing experiment, keep production defaults unchanged and
run diagnostics with:

```powershell
.venv\Scripts\python.exe -m src.runtime_diagnostics.hardware_smoke_test --engine-path path\to\model.engine --output-dir reports/hardware_smoke_unified --preprocess-backend unified --width 128 --height 96 --num-frames 5 --imgsz 64 --batch-size 4 --device auto
```

## Status Interpretation

Top-level environment status:

- `READY`: required checks passed and no warnings were reported.
- `READY_WITH_WARNINGS`: required checks passed, but optional or environment
  warnings were reported.
- `NOT_READY`: one or more required checks failed.

Individual check status:

- `PASSED`: check completed successfully.
- `PASSED_WITH_WARNINGS`: check completed, but warnings should be reviewed.
- `SKIPPED`: check was intentionally skipped or could not run because optional
  hardware or an engine was not available.
- `FAILED`: check failed and should be addressed before production use.

Important distinction:

```text
hardware SKIPPED is not the same as hardware FAILED
```

For example, if no engine path is provided or CUDA is unavailable, hardware
inference can be skipped while the rest of the diagnostics still pass with
warnings.

## Typical Report Locations

Readiness check:

```text
reports/runtime_readiness/runtime_readiness.json
reports/runtime_readiness/runtime_readiness.md
```

Operator smoke:

```text
reports/operator_smoke/operator_smoke_report.json
reports/operator_smoke/operator_smoke_report.md
```

Hardware smoke:

```text
reports/hardware_smoke/hardware_smoke_report.json
reports/hardware_smoke/hardware_smoke_report.md
```

Aggregate diagnostic report:

```text
reports/diagnostic_report/diagnostic_report.json
reports/diagnostic_report/diagnostic_report.md
```

Generated files under `reports/` are local artifacts and should not be added to
the repository.
