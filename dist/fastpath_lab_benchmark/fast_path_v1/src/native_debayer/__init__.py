"""Native CUDA debayer (standalone, no production dependencies).

This subpackage wraps a small C++/CUDA extension that performs bilinear
demosaicing of an RG/BG/GR/GB Bayer batch directly on the GPU. It is
identical in behaviour to the production runtime kernel but ships here
without any VolleyHub-specific glue code.

Public API:
    NativeCudaDebayer().debayer(raw_batch, bayer_pattern, ...) -> torch.Tensor
    NativeCudaDebayer.is_available() -> bool
    NativeCudaDebayer.describe_backend() -> dict
"""
from .native_debayer import NativeCudaDebayer, NativeBackendInfo

__all__ = ["NativeCudaDebayer", "NativeBackendInfo"]
