# GPU / build environment notes

This benchmark needs a working CUDA toolchain. Common stumbling blocks below.

## torch + CUDA

Install `torch` from the wheel index that matches your CUDA toolkit:

```powershell
# CUDA 12.1
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
# CUDA 12.4
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# CUDA 11.8
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
```

Verify:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## TensorRT

TensorRT comes either:
- bundled with the torch CUDA wheel (recent versions), OR
- as a separate install (`pip install tensorrt`).

`ultralytics.YOLO('best.engine')` will tell you immediately if TRT is missing.

## native CUDA debayer extension (custom_cuda_kernel)

This is JIT-compiled from `src/runtime_gpu/native_debayer/csrc/*` on first use.

Requirements:
- CUDA Toolkit (`nvcc` on PATH)
- Visual Studio Build Tools 2019 or 2022 (`cl.exe` on PATH at first build)
- `ninja` (`pip install ninja`)

On Windows the easiest way to ensure `cl.exe` and `nvcc` are visible is to launch
PowerShell from a "x64 Native Tools Command Prompt for VS …" or run the build via
`scripts/export_or_check_engines.py --build-native` which auto-locates `vcvars64.bat`.

## TensorRT engine portability

A `.engine` file is **not portable** between machines in general:

- different GPU architectures (sm_75 vs sm_86 vs sm_89 …) require rebuilds,
- TensorRT minor version mismatches break loads,
- driver mismatches can also break loads.

Strategy in this package:
1. Try to load engines that ship in `artifacts/tensorrt_exports/`.
2. On failure, mark `engine_compatibility_warning = "..."`.
3. Rebuild on the target machine via `scripts/export_or_check_engines.py --build-missing`
   (this requires `best.pt` to be present in the package root).

## pypylon (only for real Basler mode)

```powershell
pip install pypylon
```

You also need the Basler pylon SDK installed system-wide. Without pypylon, only
the synthetic sweep runs; the real-basler sweep exits cleanly with `PYPYLON_MISSING`.
