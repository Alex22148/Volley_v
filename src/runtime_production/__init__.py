"""Production fast-path subsystem (opt-in via VOLLEYHUB_FAST_PATH=1).

OFF by default. When the env var is set AND the engine + native debayer
are available AND all 4 camera roles publish frames in shared memory,
the fast path executes a single batched packet:
    RAW (B=4, H, W) uint8
      -> NativeCudaDebayer (custom_cuda_kernel)
      -> torch.Tensor (B=4, 3, imgsz, imgsz) RGB float on CUDA
      -> model.predict(tensor)  [TensorRT b4 static engine]
      -> 4 detection lists, one per camera role
Otherwise the live pipeline silently runs the existing slow path.
"""
from .fast_path_config import FastPathConfig, load_fast_path_config_from_env
from .fast_path_executor import FastPathExecutor, FastPathPacketResult
from .preview_worker import (
    PreviewConfig,
    PreviewWorker,
    cheap_preview_from_raw,
    load_preview_config_from_env,
)

__all__ = [
    "FastPathConfig",
    "FastPathExecutor",
    "FastPathPacketResult",
    "PreviewConfig",
    "PreviewWorker",
    "cheap_preview_from_raw",
    "load_fast_path_config_from_env",
    "load_preview_config_from_env",
]
