# Native CUDA debayer (`native_cuda_npp` backend)

Diagnostic / opt-in backend for Bayer → BGR/RGB conversion on GPU. The
slot is named `native_cuda_npp` for forward compatibility, but the
**current implementation is a custom bilinear demosaic CUDA kernel**.
NPP (`nppiCFAToBGR_8u_C1C3R` and friends) is documented as future work;
the runtime always reports the actually-running implementation in
`describe_backend()['detail']`.

## Status

| Field                   | Value                                                      |
|-------------------------|------------------------------------------------------------|
| Backend slot name       | `native_cuda_npp`                                          |
| `native_backend_detail` | `custom_cuda_kernel`                                       |
| Algorithm               | Bilinear demosaic, 4 Bayer patterns (RG/BG/GR/GB), uint8 input |
| Output                  | `torch.Tensor` (B, 3, H, W) on CUDA, optional resize/normalize/half |
| Build                   | JIT via `torch.utils.cpp_extension` or AOT `setup.py`      |
| Production wired?       | **No** — opt-in only via benchmark; live pipeline unchanged |

## Requirements (Windows)

- CUDA Toolkit (verified with **CUDA 12.9** nvcc + torch 2.5.1+cu121, sm_75 RTX 2080 Super).
  - Different toolkit version is fine as long as nvcc can target your GPU's compute capability.
- Visual Studio Build Tools 2019 or 2022 with the C++ workload (so `cl.exe` is available).
- Python 3.12, `torch` with CUDA, `ninja` (`pip install ninja`).
- `nvcc` and `cl.exe` reachable on PATH **at first build** — easiest way is to use the helper batch
  file below which auto-locates `vcvars64.bat`.

## Build options

### Option 1 — JIT (default, transparent)

The Python adapter calls `torch.utils.cpp_extension.load(...)` at first import. On Windows the
build step needs `cl.exe`, so the helper batch script activates the right Visual Studio environment
automatically:

```powershell
# from project root; first build only:
src\runtime_gpu\native_debayer\_build_jit.bat
```

The compiled `.pyd` is cached under `%LOCALAPPDATA%\torch_extensions\...`. Subsequent imports load
the cached binary directly without re-running the build, so you do **not** need `cl.exe` on PATH
after the first build.

### Option 2 — AOT (pre-built)

```powershell
cd src\runtime_gpu\native_debayer
python setup.py build_ext --inplace
```

Drops `volleyhub_native_debayer*.pyd` next to the source. The Python adapter prefers this over the
JIT cache when present.

## How to verify it works

```powershell
python -c "from src.runtime_gpu.native_debayer import NativeCudaDebayer; import json; print(json.dumps(NativeCudaDebayer.describe_backend(), indent=2, default=str))"
```

Expected output (relevant fields):

```
"available": true,
"detail": "custom_cuda_kernel",
"extension_path": "C:\\Users\\<you>\\AppData\\Local\\torch_extensions\\...\\volleyhub_native_debayer.pyd",
"cuda_device_name": "NVIDIA GeForce RTX 2080 Super",
"torch_version": "2.5.1+cu121",
"cuda_version": "12.1"
```

If `"available": false`, check `"error"` — typically it'll say either *Ninja missing*, *CUDA not
available*, or a build-toolchain mismatch. Re-run `_build_jit.bat` from a clean shell.

## Run the comparison benchmark

```powershell
python -m src.runtime_benchmark.run_native_debayer_compare `
  --width 2464 --height 2056 --bayer-pattern RG `
  --imgsz 640 --batch-size 4 --num-frames 200 --warmup 20 `
  --inference-backend tensorrt `
  --model-path "artifacts/tensorrt_exports/best__fp16_640_b4_static__fp16__img640__b4__static.engine"
```

Outputs:

- `reports/native_debayer_benchmark.md` — human-readable comparison + honest assessment.
- `reports/native_debayer_benchmark.json` — full structured run (variants, config, native backend info).
- `reports/native_debayer_benchmark.csv` — flat per-variant rows for spreadsheets.

## Production activation (DEFERRED)

The native backend is **not** wired into the live pipeline yet. The wiring point is documented for
the next iteration:

```powershell
# Future: opt-in via env var. Currently a no-op — live pipeline always uses CPU debayer.
$env:VOLLEYHUB_COLOR_BACKEND = "native_cuda_npp"
```

Plan when activation is approved:

1. Read `VOLLEYHUB_COLOR_BACKEND` in `live_runtime/live_backend_controller._convert_bayer`.
2. If set to `native_cuda_npp` AND `NativeCudaDebayer.is_available()` AND we hold all 4 frames in
   one tick, route through `NativeCudaDebayer.debayer(...)`.
3. If anything fails (extension missing, CUDA error, single-frame path), log a warning ONCE and
   silently fall back to the existing `cv2.cvtColor` CPU path. **Default behaviour remains CPU.**

## Bayer pattern naming convention

This project uses **first-cell-letter** naming (e.g. `"RG"` means R is at the top-left of the 2×2
mosaic cell). OpenCV uses an inverted convention internally:

| Project pattern | Physical layout (top-left)      | Equivalent cv2 code            |
|-----------------|----------------------------------|--------------------------------|
| `RG`            | R G / G B (R at (0,0))            | `cv2.COLOR_BAYER_BG2BGR`       |
| `BG`            | B G / G R                         | `cv2.COLOR_BAYER_RG2BGR`       |
| `GR`            | G R / B G                         | `cv2.COLOR_BAYER_GB2BGR`       |
| `GB`            | G B / R G                         | `cv2.COLOR_BAYER_GR2BGR`       |

Production `vision/yolo_module.py` defaults to `bayer_code = COLOR_BAYER_BG2BGR`, which corresponds
to project pattern `"RG"` — that is what to pass to `NativeCudaDebayer` for the live cameras.

## Fallback behaviour

`GpuColorConverter` keeps the existing fallback ladder:

- `--color-backend native_cuda_npp` → if extension unavailable, log warning and **fall back to
  `cpu`** (recorded in `fallback_warnings`).
- `--color-backend auto` → never picks `native_cuda_npp` automatically. The default ladder
  (`cv2_cuda → torch_gpu → cpu`) is preserved to avoid changing default production behaviour.

## Future work — NPP

The current kernel is a textbook bilinear demosaic. Switching to NPP could squeeze additional
performance:

- `nppiCFAToBGR_8u_C1C3R(pSrc, srcStep, srcSize, srcRect, pDst, dstStep, eGrid, eInterpolation)`
- Requires linking `nppc`, `nppial`, `nppidei` (CUDA Toolkit ships them).
- Need to detect the right `eGrid` for our 4 patterns (`NPPI_BAYER_RGGB`, `BGGR`, `GRBG`, `GBRG`).
- Add a runtime selector: prefer NPP if linkable, else fall back to the custom kernel.
- Update `NativeBackendInfo.detail` accordingly (`"NPP"` vs `"custom_cuda_kernel"`).

This is **not** implemented in the current build. Do NOT claim NPP is in use until the runtime
reports `detail == "NPP"`.

## Layout

```
src/runtime_gpu/native_debayer/
  __init__.py
  native_debayer.py          # NativeCudaDebayer adapter + lazy loader
  csrc/
    debayer.cpp              # pybind11 bindings
    debayer_cuda.cu          # bilinear demosaic kernel (4 patterns)
  setup.py                   # AOT build option
  _build_jit.bat             # Windows helper: vcvars64 + JIT build
  README.md                  # this file
```
