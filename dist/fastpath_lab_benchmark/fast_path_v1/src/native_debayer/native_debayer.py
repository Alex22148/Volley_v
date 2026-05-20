"""Native CUDA debayer adapter (standalone version for the lab benchmark).

Loads a custom C++/CUDA extension that does bilinear demosaicing on the
GPU. The first call triggers a JIT build (requires CUDA Toolkit + a host
compiler); subsequent calls reuse the cached .pyd/.so. If the build
fails, is_available() returns False and the caller must use a fallback.

This file is intentionally short and self-contained — no production
imports. The kernel sources live under ./csrc/.
"""
from __future__ import annotations

import logging
import os
import threading
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
    detail: str = ""
    error: Optional[str] = None
    extension_path: Optional[str] = None
    cuda_device_name: Optional[str] = None
    torch_version: Optional[str] = None
    cuda_version: Optional[str] = None
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
            "extra": dict(self.extra),
        }


_LOAD_LOCK = threading.Lock()
_EXTENSION = None
_LOAD_INFO: Optional[NativeBackendInfo] = None


def _find_cached_extension() -> Optional[Path]:
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
        for ext_name in ("volleyhub_native_debayer.pyd", "volleyhub_native_debayer.so"):
            for found in root.rglob(ext_name):
                if found.is_file():
                    return found
    return None


def _try_load_extension() -> Tuple[Any, NativeBackendInfo]:
    global _EXTENSION, _LOAD_INFO
    if _LOAD_INFO is not None:
        return _EXTENSION, _LOAD_INFO

    info = NativeBackendInfo(available=False, detail="")

    try:
        import torch  # noqa: F401
        info.torch_version = torch.__version__
        if torch.cuda.is_available():
            info.cuda_device_name = torch.cuda.get_device_name(0)
            info.cuda_version = getattr(getattr(torch, "version", None), "cuda", None) or ""
        else:
            info.error = "torch.cuda is not available"
            _LOAD_INFO = info
            return None, info
    except Exception as exc:
        info.error = f"torch import failed: {exc!r}"
        _LOAD_INFO = info
        return None, info

    # 1) prebuilt (pip-installed) extension
    try:
        import volleyhub_native_debayer as _prebuilt  # type: ignore
        info.available = True
        info.detail = "custom_cuda_kernel"
        info.extension_path = getattr(_prebuilt, "__file__", "")
        info.extra["load_path"] = "prebuilt"
        _EXTENSION = _prebuilt
        _LOAD_INFO = info
        _LOG.info("[NativeCudaDebayer] prebuilt loaded at %s", info.extension_path)
        return _EXTENSION, info
    except Exception:
        pass

    # 2) cached .pyd/.so from torch_extensions
    try:
        cached = _find_cached_extension()
        if cached is not None:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "volleyhub_native_debayer", str(cached)
            )
            if spec is not None and spec.loader is not None:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                info.available = True
                info.detail = "custom_cuda_kernel"
                info.extension_path = str(cached)
                info.extra["load_path"] = "cached_pyd"
                _EXTENSION = mod
                _LOAD_INFO = info
                _LOG.info("[NativeCudaDebayer] cached loaded at %s", cached)
                return _EXTENSION, info
    except Exception as exc:
        info.extra["cache_load_error"] = repr(exc)

    # 3) JIT compile via torch.utils.cpp_extension
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
            extra_cflags=["/O2", "/Zc:preprocessor"] if os.name == "nt" else ["-O3"],
            extra_cuda_cflags=(
                ["-O3", "--use_fast_math", "-Xcompiler", "/Zc:preprocessor"]
                if os.name == "nt"
                else ["-O3", "--use_fast_math"]
            ),
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
    _LOG.info("[NativeCudaDebayer] JIT-built at %s", info.extension_path)
    return _EXTENSION, info


class NativeCudaDebayer:
    """Bilinear Bayer -> RGB/BGR demosaic on GPU.

    Cheap to construct. Extension is loaded lazily on first call.
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
        """Demosaic a batch of RAW Bayer frames.

        raw_batch: ndarray (B,H,W) uint8/uint16, list of (H,W), or CUDA tensor.
        Returns torch.Tensor on CUDA:
            BCHW: (B, 3, H_out, W_out)
            HWC : (H_out, W_out, 3) — first frame only.
        """
        if not self.is_available():
            raise RuntimeError(
                "NativeCudaDebayer extension is not available. "
                f"Reason: {self._info.error if self._info else 'unknown'}"
            )
        import torch

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
            normalize = True

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
