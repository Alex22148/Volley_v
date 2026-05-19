# fastpath_lab_benchmark

> **For step-by-step usage, see [docs/FASTPATH_LAB_USER_GUIDE.md](docs/FASTPATH_LAB_USER_GUIDE.md)**
> (also available as [docs/FASTPATH_LAB_USER_GUIDE.html](docs/FASTPATH_LAB_USER_GUIDE.html)).
> The guide is written for an external tester who does not know the
> VolleyHub_K architecture â€” it explains every script in order, the
> inputs/outputs, how to interpret each metric, and how to answer the
> two key questions (Full HD-like FPS / largest input at stable 50 FPS).

A **minimal, transparent lab benchmark** for the GPU fast path
(synthetic Bayer -> CUDA debayer -> resize -> TensorRT).

> This is NOT a portable production package. It does NOT model the
> camera grabber, GUI, ring buffer, preview worker, or shared-memory
> IPC. It is a single-purpose tool for one job: **measure the upper
> bound of the GPU fast path on this machine** and answer specific
> questions about FPS at different inference resolutions.

If you are looking for the production-shaped portable package, see the
sibling `volleyhub_fastpath_benchmark/` directory.

---

## 1. What this benchmark measures

There are three independent benchmarks. Each can be run on its own.

| Script | What it measures |
|---|---|
| `scripts/01_benchmark_inference_only.py` | Only the TensorRT forward pass on a pre-allocated CUDA tensor. No debayer, no resize, no host roundtrip. |
| `scripts/02_benchmark_color_only.py`     | Only the GPU debayer + bilinear resize + normalize. No TensorRT. |
| `scripts/03_benchmark_full_synthetic_path.py` | The full synthetic path: Bayer batch -> debayer -> resize -> TensorRT. Output stays on the GPU. |
| `scripts/04_resolution_sweep.py`         | Runs (3) for 640x640, 960x960, 1280x1280, 1088x1920 in turn. |
| `scripts/05_make_report.py`              | Aggregates `results/` into an HTML + Markdown report with charts. |
| `scripts/06_run_single_test.py`          | One-shot benchmark driven by `configs/single_test.json` (one engine, one resolution, one batch, N active cameras, target FPS). |
| `scripts/00_check_env.py`                | Prints what is and is not available on this machine. |

## 2. What this benchmark does NOT measure

* No Basler camera grab â€” input is generated synthetically.
* No GUI, no preview worker, no display path.
* No ring buffer, no producer/consumer queue, no shared memory.
* No hardware sync between four real cameras.
* No NMS or real detection scoring (postprocess is a no-op shape check).
* No production application code is imported by any script.

This means: a 50 FPS PASS verdict here means the **GPU fast path** can
sustain 50 FPS, not that the whole production app can. The production
app will be slower; how much slower depends on the rest of the system.

---

## 3. Install

You need:

* Windows or Linux with an NVIDIA GPU + recent driver
* CUDA Toolkit (a host C++ compiler is required for the JIT debayer)
* Python 3.10 or newer
* `torch` (CUDA build), `tensorrt`, `numpy`, `matplotlib`

The simplest install on a clean machine:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1
pip install tensorrt==10.16.*
pip install -r requirements.txt
```

(Adjust the CUDA suffix `cu121` to match your CUDA Toolkit.)

If you already have torch + tensorrt installed:

```powershell
pip install -r requirements.txt
```

---

## 4. Run

```powershell
.\CHECK_ENV.ps1                     # quick sanity check
.\RUN_ALL.ps1                       # full sweep + report (about 8 minutes)
.\RUN_ALL.ps1 -DurationS 15 -Warmup 5   # faster sweep
```

Or run individual scripts:

```powershell
python scripts\00_check_env.py
python scripts\01_benchmark_inference_only.py `
        --engine engines\best__fp16_640_b4_static.engine `
        --input-shape 640 640 --batch 4 --duration-s 30
python scripts\02_benchmark_color_only.py `
        --capture-shape 1080 1920 --output-shape 640 640 --batch 4 --duration-s 30
python scripts\03_benchmark_full_synthetic_path.py `
        --engine engines\best__fp16_640_b4_static.engine `
        --capture-shape 1080 1920 --input-shape 640 640 --batch 4 --duration-s 30
python scripts\04_resolution_sweep.py --duration-s 30
python scripts\05_make_report.py

# One-shot single test (all parameters in one file):
#   edit configs\single_test.json   (engine, capture, inference, batch,
#                                    active_cameras, target_fps, ...)
python scripts\06_run_single_test.py --config configs\single_test.json
```

### Szybkie scenariusze (copy/paste)

1. **Szybki smoke test** (check env -> benchmark 08 -> report 09):

```powershell
python scripts\11_run_pipeline.py --smoke
```

2. **Pełny benchmark** (z opcjonalnym exportem static, potem benchmark i raport):

```powershell
python scripts\11_run_pipeline.py --check-env --export --benchmark --report
```

3. **Compare backendów** (`.pt` vs `.onnx` vs `.engine`):

```powershell
python scripts\10_compare_pt_onnx_engine_detections.py `
  --images data\sample_images `
  --pt model_base\best.pt `
  --onnx engines\onnx\static\best__fp16_v640_b4.onnx `
  --engine engines\static\best__fp16_v640_b4.engine `
  --imgsz 640 --conf 0.25 --iou 0.45 --device cuda
```

### configs/single_test.json â€” single experimental run

Edit the file and run `scripts/06_run_single_test.py` to test ONE
configuration. Useful experimental knobs:

| field                  | meaning |
|---|---|
| `engine`               | path to a `.engine` file under `engines/` |
| `capture_shape_hw`     | synthetic Bayer size `[H, W]` (e.g. `[1080, 1920]`) |
| `inference_shape_hw`   | engine input `[H, W]` â€” must match the engine |
| `batch`                | engine batch â€” must match the engine |
| `active_cameras`       | 1..batch â€” how many slots carry real synthetic content; remaining slots are dark blanks (simulates 1/4, 2/4, 3/4, 4/4 cameras connected) |
| `target_fps`           | desired sustained FPS per camera; 0 = free-run (max throughput) |
| `duration_s`           | measurement window after warmup |
| `warmup`               | warmup iterations (discarded from stats) |
| `bayer_pattern`        | RG / BG / GR / GB |
| `synthetic_source.*`   | noise, ball, intensity controls for the synthetic generator |

Output lands in `results/single_test_<HxW>_b<batch>_<active>cam_<fps>fps.{json,csv}`,
including the verdict `target_hit` (did we sustain `target_fps` with under 5% overruns).

Outputs land in `results/` (per-iteration CSV + summary JSON) and
`reports/` (Markdown + HTML + PNG charts).

---

## 5. How to read the report

Some definitions used throughout:

* **packet**: one TensorRT call with batch = 4. Conceptually, four
  synchronised camera frames.
* **fps_per_camera**: packets per second. Equal to frames per second
  per single camera because each packet contains one frame from each
  camera. Formula: `1000 / packet_ms`.
* **fps_per_camera_safe_p95**: `1000 / packet_ms_p95`. Pessimistic
  number: this FPS is met in 95% of packets.
* **50 FPS target**: the production target is 50 FPS per camera = a
  20 ms budget per packet.
* **Full HD-like**: a 1088 x 1920 inference input (height x width).
  Close to the 1080 x 1920 raw capture size.

### Questions the report answers

1. **At Full HD-like inference (1088 x 1920)**:
   - FPS per camera (median, safe p95)
   - 50 FPS verdict (PASS / FAIL)

2. **For stable 50 FPS**:
   - The largest inference input that holds safe p95 >= 50 FPS.
   - The largest inference input that holds median >= 50 FPS.

3. **Suggested presets**:
   - `LIVE_50FPS` â€” smallest input that still holds safe p95 >= 50 FPS.
   - `QUALITY_40FPS` â€” largest input with safe p95 >= 40 FPS.
   - `FULLHD_30FPS` â€” the Full HD-like row, regardless of FPS.

### Conclusions you may NOT draw from this report

> "The production application runs at 50 FPS on this machine."

You may not. This benchmark answers an **upper-bound** question about
the GPU fast path. The production app adds camera grab time, IPC,
synchronization, and GUI rendering, all of which cost real time.

What the report DOES support:

> "The GPU fast path can / cannot sustain X FPS at inference Y x Z on
> this machine."

That is exactly what the tester asked for.

---

## 6. Directory layout

```
fastpath_lab_benchmark/
  README.md                    # this file
  requirements.txt
  CHECK_ENV.ps1                # 30-second environment check
  RUN_ALL.ps1                  # full sweep + report
  docs/
    FASTPATH_LAB_USER_GUIDE.md     # full step-by-step manual (Polish)
    FASTPATH_LAB_USER_GUIDE.html
  configs/
    benchmark_config.yaml          # defaults consumed by RUN_ALL.ps1
    baseline_rtx2080super.json     # reference numbers for comparison
    single_test.json               # one-shot test parameters
  engines/
    README_ENGINES.md
    best__fp16_640_b4_static.engine
    best__fp16_960_b4_static.engine
    best__fp16_1280_b4_static.engine
    best__fp16_1088x1920_b4_static.engine
  scripts/
    00_check_env.py
    01_benchmark_inference_only.py
    02_benchmark_color_only.py
    03_benchmark_full_synthetic_path.py
    04_resolution_sweep.py
    05_make_report.py
    06_run_single_test.py
  src/
    trt_runner.py              # TRT wrapper (no Ultralytics)
    cuda_debayer.py            # GPU Bayer -> RGB float
    synthetic_raw.py           # Bayer batch generator
    metrics.py                 # percentiles + verdict
    report_builder.py          # Markdown + HTML + charts
    native_debayer/            # custom CUDA bilinear demosaic kernel
      native_debayer.py
      csrc/
        debayer.cpp
        debayer_cuda.cu
  results/                     # JSON + CSV per run
  reports/                     # Markdown + HTML + PNG charts
```

Every Python file in `src/` is short by design (200-300 lines) so a
tester can read it before trusting a number it produces. No hidden
behaviour, no production glue.


