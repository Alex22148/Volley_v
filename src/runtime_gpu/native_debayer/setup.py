"""AOT build script for the native debayer extension.

Use this if you prefer to pre-compile rather than rely on torch's JIT
loader (the default in native_debayer.py). On Windows you typically need
to run this from a Visual Studio Developer Command Prompt with the CUDA
Toolkit on PATH.

Build:
    python setup.py build_ext --inplace

The resulting .pyd / .so lands in this directory and is picked up
automatically by NativeCudaDebayer when JIT load fails.
"""
from __future__ import annotations

from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


_ROOT = Path(__file__).resolve().parent
_CSRC = _ROOT / "csrc"


def _sources() -> list:
    return [str(_CSRC / "debayer.cpp"), str(_CSRC / "debayer_cuda.cu")]


setup(
    name="volleyhub_native_debayer",
    version="0.1.0",
    description="VolleyHub native CUDA bilinear Bayer demosaic",
    ext_modules=[
        CUDAExtension(
            name="volleyhub_native_debayer",
            sources=_sources(),
            extra_compile_args={
                "cxx": ["/O2"],
                "nvcc": ["-O3", "--use_fast_math", "-Xcompiler=/O2"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
