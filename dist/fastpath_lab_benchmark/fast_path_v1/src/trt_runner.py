"""Minimal TensorRT runner for the lab benchmark.

What this module is:
    A small, single-file wrapper around the TensorRT Python API that
    loads a .engine file, allocates input/output GPU buffers using
    torch tensors, and runs `execute_async_v3` on a batch of images.

What this module is NOT:
    - It is not Ultralytics. There is no .predict() and no NMS done
      inside the engine. Output post-processing is the caller's job
      (we time it as zero ms for "inference only" benchmarks).
    - It is not the production YoloInferenceEngine. That class wraps
      Ultralytics and has its own preprocessing path.

The goal of this file is to be readable end-to-end in five minutes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass(slots=True)
class TrtEngineInfo:
    engine_path: str
    trt_version: str
    input_name: str
    output_names: List[str]
    input_shape: Tuple[int, int, int, int]   # (B, C, H, W)
    input_dtype: str                          # "float16" or "float32"
    is_fp16: bool
    output_shapes: Dict[str, Tuple[int, ...]]
    engine_size_bytes: int

    def to_dict(self) -> dict:
        return {
            "engine_path": self.engine_path,
            "trt_version": self.trt_version,
            "input_name": self.input_name,
            "output_names": list(self.output_names),
            "input_shape": list(self.input_shape),
            "input_dtype": self.input_dtype,
            "is_fp16": bool(self.is_fp16),
            "output_shapes": {k: list(v) for k, v in self.output_shapes.items()},
            "engine_size_bytes": int(self.engine_size_bytes),
        }


class TrtRunner:
    """Load a TensorRT .engine and run batched inference on CUDA tensors.

    Usage:
        runner = TrtRunner("engines/best__fp16_640_b4_static.engine")
        x = torch.randn(4, 3, 640, 640, dtype=torch.float16, device="cuda")
        y = runner.infer(x)                  # dict of output tensors on CUDA
        ms = runner.timed_infer(x)           # (outputs, inference_ms)
    """

    def __init__(self, engine_path: str, device: str = "cuda") -> None:
        self.engine_path = str(engine_path)
        self.device = str(device)
        self._import_runtime()
        self._load_engine()
        self._allocate_io_buffers()

    # ------------------------------------------------------------ loaders

    def _import_runtime(self) -> None:
        import tensorrt as trt
        import torch
        self._trt = trt
        self._torch = torch
        self._trt_version = trt.__version__
        self._logger = trt.Logger(trt.Logger.WARNING)

    def _load_engine(self) -> None:
        trt = self._trt
        path = Path(self.engine_path)
        if not path.exists():
            raise FileNotFoundError(f"Engine file not found: {path}")
        engine_bytes = path.read_bytes()
        if len(engine_bytes) < 1024:
            raise RuntimeError(
                f"Engine file is suspiciously small ({len(engine_bytes)} B). "
                "Was the export interrupted?"
            )

        # Ultralytics exports the engine with a length-prefixed JSON metadata
        # header. Skip it transparently — also accept raw TRT engines.
        engine_payload, ult_metadata = _strip_ultralytics_header(engine_bytes)

        runtime = trt.Runtime(self._logger)
        engine = runtime.deserialize_cuda_engine(engine_payload)
        if engine is None:
            raise RuntimeError(f"Failed to deserialize TRT engine: {path}")
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("Failed to create TRT execution context")

        self._runtime = runtime
        self._engine = engine
        self._context = context
        self._engine_size = len(engine_bytes)
        self._ultralytics_metadata = ult_metadata

    def _allocate_io_buffers(self) -> None:
        trt = self._trt
        torch = self._torch
        engine = self._engine
        context = self._context

        # Enumerate I/O tensors (TRT 10+ API)
        io_names: List[str] = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        input_names = [n for n in io_names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        output_names = [n for n in io_names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        if len(input_names) != 1:
            raise RuntimeError(f"Expected exactly 1 input, found: {input_names}")

        input_name = input_names[0]
        input_shape = tuple(engine.get_tensor_shape(input_name))
        if any(d < 0 for d in input_shape):
            # Static-shape engines only (b4 static). Dynamic engines need set_input_shape.
            raise RuntimeError(
                f"Dynamic input shape {input_shape} not supported in this lab runner. "
                "Use a static-shape (.engine) file."
            )
        input_dtype_trt = engine.get_tensor_dtype(input_name)
        torch_dtype = _trt_to_torch_dtype(trt, torch, input_dtype_trt)

        self._input_name = input_name
        self._input_shape = tuple(int(x) for x in input_shape)
        self._input_torch_dtype = torch_dtype
        self._is_fp16 = (torch_dtype == torch.float16)

        # Persistent input/output GPU tensors (we re-use these across iterations)
        self._input_buffer = torch.empty(self._input_shape, dtype=torch_dtype, device=self.device)
        context.set_tensor_address(input_name, int(self._input_buffer.data_ptr()))

        self._output_names = output_names
        self._output_buffers: Dict[str, "torch.Tensor"] = {}
        self._output_shapes: Dict[str, Tuple[int, ...]] = {}
        for name in output_names:
            shape = tuple(int(x) for x in engine.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                raise RuntimeError(f"Output {name} has dynamic shape {shape}, unsupported here.")
            dtype = _trt_to_torch_dtype(trt, torch, engine.get_tensor_dtype(name))
            buf = torch.empty(shape, dtype=dtype, device=self.device)
            self._output_buffers[name] = buf
            self._output_shapes[name] = shape
            context.set_tensor_address(name, int(buf.data_ptr()))

        # CUDA stream + event for timing
        self._stream = torch.cuda.Stream(device=self.device)
        self._evt_start = torch.cuda.Event(enable_timing=True)
        self._evt_end = torch.cuda.Event(enable_timing=True)

    # ---------------------------------------------------------- inference

    def infer(self, input_tensor) -> Dict[str, "torch.Tensor"]:
        """Run inference. Returns a dict of output tensors on CUDA."""
        outputs, _ = self.timed_infer(input_tensor)
        return outputs

    def timed_infer(self, input_tensor) -> Tuple[Dict[str, "torch.Tensor"], float]:
        """Run inference and return (outputs, inference_ms).

        inference_ms is measured with CUDA events: includes ONLY the
        async kernel time on the GPU, not the host->device copy.
        """
        torch = self._torch
        if input_tensor.shape != self._input_buffer.shape:
            raise ValueError(
                f"Input shape {tuple(input_tensor.shape)} != engine input {self._input_shape}"
            )
        if input_tensor.dtype != self._input_buffer.dtype:
            input_tensor = input_tensor.to(self._input_buffer.dtype)
        if not input_tensor.is_cuda:
            input_tensor = input_tensor.to(self.device, non_blocking=True)
        if not input_tensor.is_contiguous():
            input_tensor = input_tensor.contiguous()

        # Copy into our persistent input buffer (zero-copy when caller already used it)
        if input_tensor.data_ptr() != self._input_buffer.data_ptr():
            self._input_buffer.copy_(input_tensor, non_blocking=True)

        self._evt_start.record()
        ok = self._context.execute_async_v3(stream_handle=self._stream.cuda_stream)
        if not ok:
            raise RuntimeError("TRT execute_async_v3 returned False")
        self._evt_end.record()
        self._evt_end.synchronize()
        inference_ms = float(self._evt_start.elapsed_time(self._evt_end))

        # Return references to the persistent output buffers
        outputs = {name: buf for name, buf in self._output_buffers.items()}
        return outputs, inference_ms

    # -------------------------------------------------------- diagnostics

    def info(self) -> TrtEngineInfo:
        torch = self._torch
        return TrtEngineInfo(
            engine_path=self.engine_path,
            trt_version=self._trt_version,
            input_name=self._input_name,
            output_names=list(self._output_names),
            input_shape=self._input_shape,  # type: ignore[arg-type]
            input_dtype=str(self._input_torch_dtype).replace("torch.", ""),
            is_fp16=self._is_fp16,
            output_shapes={k: tuple(v) for k, v in self._output_shapes.items()},
            engine_size_bytes=self._engine_size,
        )

    def warmup(self, iterations: int = 5) -> None:
        torch = self._torch
        x = torch.zeros(self._input_shape, dtype=self._input_torch_dtype, device=self.device)
        for _ in range(max(0, int(iterations))):
            self.infer(x)
        torch.cuda.synchronize()


def _strip_ultralytics_header(engine_bytes: bytes):
    """Detect and strip an Ultralytics .engine length-prefixed JSON header.

    Ultralytics writes:
        [ uint32 little-endian length ][ JSON bytes ][ raw TRT engine ]

    If the first 4 bytes look like a sensible length (< 1 MiB) and the
    JSON parses, we return the payload after the JSON. Otherwise we
    return the bytes unchanged.
    """
    import json
    if len(engine_bytes) < 8:
        return engine_bytes, None
    raw_len = int.from_bytes(engine_bytes[:4], "little", signed=False)
    if raw_len <= 0 or raw_len > 1024 * 1024 or raw_len + 4 >= len(engine_bytes):
        return engine_bytes, None
    head = engine_bytes[4:4 + raw_len]
    try:
        meta = json.loads(head.decode("utf-8"))
    except Exception:
        return engine_bytes, None
    if not isinstance(meta, dict):
        return engine_bytes, None
    return engine_bytes[4 + raw_len:], meta


def _trt_to_torch_dtype(trt, torch, dt):
    """Map a TRT DataType to a torch dtype."""
    mapping = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT8: torch.int8,
        trt.DataType.INT32: torch.int32,
        trt.DataType.BOOL: torch.bool,
    }
    # Newer TRT versions add more (BF16, UINT8, FP8...) — best-effort
    for attr_name, torch_name in (("BF16", "bfloat16"), ("UINT8", "uint8"), ("FP8", "float8_e4m3fn")):
        attr = getattr(trt.DataType, attr_name, None)
        if attr is not None and hasattr(torch, torch_name):
            mapping.setdefault(attr, getattr(torch, torch_name))
    if dt not in mapping:
        raise RuntimeError(f"Unsupported TRT dtype: {dt!r}")
    return mapping[dt]
