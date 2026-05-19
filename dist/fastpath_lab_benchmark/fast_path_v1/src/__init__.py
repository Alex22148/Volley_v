"""fastpath_lab_benchmark.src — minimal building blocks for the lab benchmark.

Each module is intentionally small and self-contained so a tester can
read it end-to-end before trusting a number from the report:

    trt_runner.py      : TensorRT engine loader + execute_async_v3 wrapper.
    cuda_debayer.py    : Bayer -> RGB float tensor on CUDA (native or torch).
    synthetic_raw.py   : Deterministic Bayer batches with a moving ball.
    metrics.py         : Percentiles + the 50 FPS verdict.
    report_builder.py  : Markdown + HTML + matplotlib charts.
    native_debayer/    : Custom CUDA bilinear demosaic kernel (JIT-built).

There are no production imports here. Nothing from this folder touches
the live runtime, the GUI, the ring buffer, or the Basler camera path.
"""
