"""Python adapter for the native CUDA debayer extension.

Loads the C++/CUDA extension lazily. If the build fails (missing CUDA
Toolkit, missing MSVC, NVCC mismatch, ...), the adapter records the
failure reason, reports `is_available() -> False`, and falls back
gracefully — callers must always check `is_available()` before using
debayer().

The class is independent of OpenCV, pypylon, and the production
pipeline. It is wired into GpuColorConverter as the `native_cuda_npp`
backend, OFF by default.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

_LOG = logging.getLogger(__name__)

_PATTERN_INDEX = {"RG": 0, "BG": 1, "GR": 2, "GB": 3}
_FORMAT_INDEX = {"RGB": 0, "BGR": 1}


@dataclass(slots=True)
class NativeBackendInfo:
    available: bool
    detail: str = ""               # e.g. "custom_cuda_kernel" or "NPP" once added
    error: Optional[str] = None
    extension_path: Optional[str] = None
    cuda_device_name: Optional[str] = None
    torch_version: Optional[str] = None
    cuda_version: Optional[str] = None
    fallback_used: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": bool(self.available),
            "detail": self.detail,
            "error": self.error,
            "extension_path": self.extension_path,
            "cuda_device_name": self.cuda_device_name,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "fallback_used": self.fallback_used,
            "extra": dict(self.extra),
        }


_LOAD_LOCK = threading.Lock()
_EXTENSION = None
_LOAD_INFO: Optional[NativeBackendInfo] = None


def _find_cached_extension() -> Optional[Path]:
    """Look for a pre-built extension in the torch_extensions cache.

    Returns the path to volleyhub_native_debayer.{pyd,so} if found.
    """
    candidate_roots: List[Path] = []
    appdata_local = os.environ.get("LOCALAPPDATA")
    if appdata_local:
        candidate_roots.append(Path(appdata_local) / "torch_extensions")
    home = Path(os.path.expanduser("~"))
    candidate_roots.append(home / "AppData" / "Local" / "torch_extensions")
    candidate_roots.append(home / ".cache" / "torch_extensions")

    for root in candidate_roots:
        if not root.exists():
            continue
        # Search recursively, but stay shallow
        for ext_name in ("volleyhub_native_debayer.pyd", "volleyhub_native_debayer.so"):
            for found in root.rglob(ext_name):
                if found.is_file():
                    return found
    return None


def _try_load_extension() -> Tuple[Any, NativeBackendInfo]:
    """Idempotent JIT load. Returns (extension or None, info)."""
    global _EXTENSION, _LOAD_INFO
    if _LOAD_INFO is not None:
        return _EXTENSION, _LOAD_INFO

    info = NativeBackendInfo(available=False, detail="")

    try:
        import torch  # noqa: F401
        info.torch_version = torch.__version__
        if torch.cuda.is_available():
            info.cuda_device_name = torch.cuda.get_device_name(0)
            _torch_version_mod = getattr(torch, "version", None)
            info.cuda_version = (getattr(_torch_version_mod, "cuda", None) if _torch_version_mod else None) or ""
        else:
            info.error = "torch.cuda is not available"
            _LOAD_INFO = info
            return None, info
    except Exception as exc:
        info.error = f"torch import failed: {exc!r}"
        _LOAD_INFO = info
        return None, info

    # 1) Try to import a pre-built extension first (system-installed).
    try:
        import volleyhub_native_debayer as _prebuilt  # type: ignore
        info.available = True
        info.detail = "custom_cuda_kernel"
        info.extension_path = getattr(_prebuilt, "__file__", "")
        info.extra["load_path"] = "prebuilt"
        _EXTENSION = _prebuilt
        _LOAD_INFO = info
        _LOG.info("[NativeCudaDebayer] loaded prebuilt extension at %s", info.extension_path)
        return _EXTENSION, info
    except Exception:
        pass

    # 2) Try to import a cached .pyd/.so from torch_extensions cache directly.
    #    This avoids torch.utils.cpp_extension.load(), which always
    #    verifies the build toolchain even when the artifact is cached.
    try:
        cached = _find_cached_extension()
        if cached is not None:
            import importlib.util
            spec = importlib.util.spec_from_file_location("volleyhub_native_debayer", str(cached))
            if spec is not None and spec.loader is not None:
                _cached_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(_cached_mod)
                info.available = True
                info.detail = "custom_cuda_kernel"
                info.extension_path = str(cached)
                info.extra["load_path"] = "cached_pyd"
                _EXTENSION = _cached_mod
                _LOAD_INFO = info
                _LOG.info("[NativeCudaDebayer] loaded cached extension at %s", cached)
                return _EXTENSION, info
    except Exception as exc:
        info.extra["cache_load_error"] = repr(exc)

    # 3) Fall back to JIT compile via torch.utils.cpp_extension.
    try:
        from torch.utils.cpp_extension import load
    except Exception as exc:
        info.error = f"torch.utils.cpp_extension unavailable: {exc!r}"
        _LOAD_INFO = info
        return None, info

    csrc = Path(__file__).parent / "csrc"
    sources = [csrc / "debayer.cpp", csrc / "debayer_cuda.cu"]
    missing = [str(p) for p in sources if not p.exists()]
    if missing:
        info.error = f"missing source files: {missing}"
        _LOAD_INFO = info
        return None, info

    try:
        ext = load(
            name="volleyhub_native_debayer",
            sources=[str(p) for p in sources],
            extra_cflags=["/O2"] if os.name == "nt" else ["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    except Exception as exc:
        info.error = f"JIT build failed: {exc!r}"
        info.extra["load_path"] = "jit_failed"
        _LOAD_INFO = info
        _LOG.warning("[NativeCudaDebayer] JIT build failed: %s", exc)
        return None, info

    info.available = True
    info.detail = "custom_cuda_kernel"
    info.extension_path = getattr(ext, "__file__", "")
    info.extra["load_path"] = "jit"
    _EXTENSION = ext
    _LOAD_INFO = info
    _LOG.info("[NativeCudaDebayer] JIT-built extension loaded at %s", info.extension_path)
    return _EXTENSION, info


class NativeCudaDebayer:
    """Adapter around the CUDA bilinear demosaic extension.

    Construction is cheap; the extension is lazily loaded on the first
    call to is_available() / debayer(). Callers MUST check is_available()
    first if they need a hard guarantee.
    """

    def __init__(self, device: str = "cuda") -> None:
        self.device = str(device)
        with _LOAD_LOCK:
            self._ext, self._info = _try_load_extension()

    @staticmethod
    def is_available() -> bool:
        with _LOAD_LOCK:
            ext, info = _try_load_extension()
            return bool(ext is not None and info.available)

    @staticmethod
    def describe_backend() -> Dict[str, Any]:
        with _LOAD_LOCK:
            _, info = _try_load_extension()
            return info.to_dict()

    def debayer(
        self,
        raw_batch: Union[np.ndarray, "Any", List[np.ndarray]],
        bayer_pattern: str,
        output_format: str = "RGB",
        resize_to: Optional[Tuple[int, int]] = None,
        normalize: bool = False,
        output_layout: str = "BCHW",
        device: str = "cuda",
        half: bool = False,
    ) -> "Any":
        """Run native CUDA debayer on a batch of RAW frames.

        Parameters
        ----------
        raw_batch
            Either an ndarray of shape (B, H, W) uint8, a CUDA torch tensor
            (B, H, W) uint8, or a Python list of (H, W) uint8 frames.
        bayer_pattern
            One of "RG" / "BG" / "GR" / "GB".
        output_format
            "RGB" or "BGR".
        resize_to
            Optional (H, W) target. Resize is performed on GPU via
            torch.nn.functional.interpolate (bilinear).
        normalize
            If True, divide by 255 to produce float in [0, 1].
        output_layout
            "BCHW" (default, ML-friendly) or "HWC" (returns first frame
            HWC for visual debug only).
        device
            Target CUDA device.
        half
            If True, return float16. Implies normalize=True.

        Returns
        -------
        torch.Tensor on CUDA. Shape:
            BCHW: (B, 3, H_out, W_out)
            HWC : (H_out, W_out, 3)  — first frame only.
        """
        if not self.is_available():
            raise RuntimeError(
                "NativeCudaDebayer extension is not available. "
                f"Reason: {self._info.error if self._info else 'unknown'}"
            )
        try:
            import torch
        except Exception as exc:
            raise RuntimeError(f"torch is required: {exc}") from exc

        pattern_key = str(bayer_pattern).upper()
        if pattern_key not in _PATTERN_INDEX:
            raise ValueError(f"Unsupported bayer pattern: {bayer_pattern!r}")
        fmt_key = str(output_format).upper()
        if fmt_key not in _FORMAT_INDEX:
            raise ValueError(f"Unsupported output format: {output_format!r}")

        raw_tensor = self._to_uint8_cuda_tensor(raw_batch, device)
        if raw_tensor.dim() != 3:
            raise ValueError(f"raw must be (B, H, W); got {tuple(raw_tensor.shape)}")

        if half and not normalize:
            normalize = True  # half implies normalized float

        # Run the CUDA kernel — output is (B, 3, H, W) float32, RGB or BGR.
        out = self._ext.bilinear_demosaic(
            raw_tensor,
            _PATTERN_INDEX[pattern_key],
            _FORMAT_INDEX[fmt_key],
            bool(normalize),
        )

        if resize_to is not None:
            t_h, t_w = int(resize_to[0]), int(resize_to[1])
            if t_h > 0 and t_w > 0 and (t_h != out.shape[2] or t_w != out.shape[3]):
                out = torch.nn.functional.interpolate(
                    out, size=(t_h, t_w), mode="bilinear", align_corners=False
                )

        if half:
            out = out.to(torch.float16)

        if output_layout.upper() == "HWC":
            return out[0].permute(1, 2, 0).contiguous()
        return out.contiguous()

    def benchmark(
        self,
        raw_batch: Union[np.ndarray, "Any", List[np.ndarray]],
        bayer_pattern: str,
        warmup: int = 5,
        iterations: int = 50,
        **debayer_kwargs: Any,
    ) -> Dict[str, Any]:
        if not self.is_available():
            return {"available": False, "reason": self._info.error if self._info else "unknown"}
        try:
            import torch
        except Exception as exc:
            return {"available": False, "reason": f"torch unavailable: {exc}"}

        for _ in range(max(0, int(warmup))):
            try:
                self.debayer(raw_batch, bayer_pattern, **debayer_kwargs)
            except Exception as exc:
                return {"available": False, "reason": f"warmup failed: {exc}"}
        torch.cuda.synchronize()

        samples: List[float] = []
        for _ in range(max(1, int(iterations))):
            t0 = time.perf_counter()
            _ = self.debayer(raw_batch, bayer_pattern, **debayer_kwargs)
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1000.0)

        sorted_s = sorted(samples)
        n = len(sorted_s)
        median = sorted_s[n // 2] if n % 2 == 1 else (sorted_s[n // 2 - 1] + sorted_s[n // 2]) / 2.0
        p95 = sorted_s[max(0, int(round(0.95 * (n - 1))))]
        return {
            "available": True,
            "iterations": n,
            "median_ms": float(median),
            "p95_ms": float(p95),
            "min_ms": float(sorted_s[0]),
            "max_ms": float(sorted_s[-1]),
            "mean_ms": float(sum(samples) / n),
            "backend_info": self._info.to_dict() if self._info else {},
        }

    def _to_uint8_cuda_tensor(self, raw_batch: Any, device: str) -> "Any":
        import torch

        if isinstance(raw_batch, list):
            arr = np.stack([np.ascontiguousarray(f) for f in raw_batch], axis=0)
            tensor = torch.from_numpy(arr.astype(np.uint8, copy=False)).to(device, non_blocking=True)
        elif isinstance(raw_batch, np.ndarray):
            arr = raw_batch
            if arr.ndim == 2:
                arr = arr[None, ...]
            if arr.dtype == np.uint16:
                arr = (arr >> 8).astype(np.uint8, copy=False)
            elif arr.dtype != np.uint8:
                arr = arr.astype(np.uint8, copy=False)
            tensor = torch.from_numpy(np.ascontiguousarray(arr)).to(device, non_blocking=True)
        elif isinstance(raw_batch, torch.Tensor):
            tensor = raw_batch
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(0)
            if tensor.dtype != torch.uint8:
                tensor = tensor.to(torch.uint8)
            if str(tensor.device) != str(torch.device(device)):
                tensor = tensor.to(device, non_blocking=True)
            tensor = tensor.contiguous()
        else:
            raise TypeError(f"Unsupported raw_batch type: {type(raw_batch)}")
        return tensor
