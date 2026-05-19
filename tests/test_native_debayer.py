"""Sanity tests for the native CUDA debayer adapter.

Skips cleanly if the extension is unavailable (no CUDA, no MSVC, etc.).
Compares native output against cv2.cvtColor for all 4 Bayer patterns.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.runtime_gpu.native_debayer import NativeCudaDebayer


_PATTERN_TO_CV_BGR = {}
try:
    import cv2
    # OpenCV uses an inverted naming convention compared to the
    # "first-cell-letter" scheme used by _bgr_to_bayer and NativeCudaDebayer.
    # Mapping: our "RG" (R at top-left) corresponds to cv2.COLOR_BAYER_BG2BGR.
    _PATTERN_TO_CV_BGR = {
        "RG": cv2.COLOR_BAYER_BG2BGR,
        "BG": cv2.COLOR_BAYER_RG2BGR,
        "GR": cv2.COLOR_BAYER_GB2BGR,
        "GB": cv2.COLOR_BAYER_GR2BGR,
    }
except Exception:  # pragma: no cover
    cv2 = None


_HAS_NATIVE = NativeCudaDebayer.is_available()
pytestmark = pytest.mark.skipif(
    not _HAS_NATIVE,
    reason="NativeCudaDebayer extension is not available (CUDA/MSVC needed for build).",
)


def _make_synthetic_bayer(h: int = 96, w: int = 128, seed: int = 1234) -> np.ndarray:
    """Build a synthetic Bayer mosaic from a known BGR canvas, RGGB pattern."""
    rng = np.random.default_rng(seed)
    bgr = (rng.integers(0, 256, size=(h, w, 3), dtype=np.int32)).astype(np.uint8)
    # Simulate "RG" mosaic: top-left=R, top-right=G, bottom-left=G, bottom-right=B
    raw = np.empty((h, w), dtype=np.uint8)
    raw[0::2, 0::2] = bgr[0::2, 0::2, 2]   # R
    raw[0::2, 1::2] = bgr[0::2, 1::2, 1]   # G
    raw[1::2, 0::2] = bgr[1::2, 0::2, 1]   # G
    raw[1::2, 1::2] = bgr[1::2, 1::2, 0]   # B
    return raw


def test_describe_backend_reports_custom_kernel() -> None:
    info = NativeCudaDebayer.describe_backend()
    assert info["available"] is True
    assert info["detail"] == "custom_cuda_kernel", (
        "Extension must self-identify as custom_cuda_kernel. "
        "NPP path is documented as future work."
    )
    assert info["error"] is None


def test_native_returns_tensor_on_cuda_with_correct_shape() -> None:
    raw = _make_synthetic_bayer()
    batch = np.stack([raw, raw, raw, raw], axis=0)
    debayer = NativeCudaDebayer(device="cuda")

    out = debayer.debayer(
        batch,
        bayer_pattern="RG",
        output_format="RGB",
        normalize=True,
        output_layout="BCHW",
    )
    assert out.is_cuda
    assert tuple(out.shape) == (4, 3, raw.shape[0], raw.shape[1])
    assert out.dtype.is_floating_point
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0 + 1e-3


def test_native_resize_takes_effect() -> None:
    raw = _make_synthetic_bayer(h=96, w=128)
    batch = np.stack([raw] * 4, axis=0)
    debayer = NativeCudaDebayer(device="cuda")

    out = debayer.debayer(
        batch,
        bayer_pattern="RG",
        output_format="RGB",
        normalize=True,
        resize_to=(64, 64),
        output_layout="BCHW",
    )
    assert tuple(out.shape) == (4, 3, 64, 64)


@pytest.mark.parametrize("pattern", ["RG", "BG", "GR", "GB"])
def test_native_matches_cv2_within_tolerance(pattern: str) -> None:
    """MAE between native bilinear demosaic and cv2.cvtColor on smooth input.

    Both implementations interpolate; on smooth gradients they should
    agree closely. cv2 uses Malvar/Hamilton-Adams (frequency-aware) and
    native uses simple bilinear, so on high-frequency content they
    diverge — that's expected and fine for benchmark purposes.
    """
    import torch
    if cv2 is None:
        pytest.skip("opencv unavailable")

    h, w = 64, 96
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    bgr = np.stack(
        [
            (xx / max(1, w - 1) * 200 + 30).astype(np.uint8),  # B
            (yy / max(1, h - 1) * 180 + 50).astype(np.uint8),  # G
            ((xx + yy) / max(1, h + w - 2) * 200 + 20).astype(np.uint8),  # R
        ],
        axis=-1,
    )

    from src.runtime_sources.synthetic_raw_4cam_source import _bgr_to_bayer
    raw = _bgr_to_bayer(bgr, pattern, dtype="uint8")

    cv_bgr = cv2.cvtColor(raw, _PATTERN_TO_CV_BGR[pattern])

    debayer = NativeCudaDebayer(device="cuda")
    out = debayer.debayer(
        raw,
        bayer_pattern=pattern,
        output_format="BGR",
        normalize=False,
        output_layout="BCHW",
    )
    native_bgr = out.clamp(0, 255).to(torch.uint8)
    native_bgr_np = native_bgr[0].permute(1, 2, 0).contiguous().detach().cpu().numpy()

    # Trim borders 4 px (cv2 reflect vs native clamp differ at edges).
    cv_inner = cv_bgr[4:-4, 4:-4].astype(np.int32)
    native_inner = native_bgr_np[4:-4, 4:-4].astype(np.int32)
    mae = float(np.mean(np.abs(cv_inner - native_inner)))
    assert mae < 5.0, (
        f"native vs cv2 MAE={mae:.3f} too high for pattern {pattern} on smooth gradient"
    )


def test_benchmark_returns_timings() -> None:
    raw = _make_synthetic_bayer(h=128, w=128)
    batch = np.stack([raw] * 4, axis=0)
    debayer = NativeCudaDebayer(device="cuda")
    stats = debayer.benchmark(
        batch,
        bayer_pattern="RG",
        warmup=2,
        iterations=5,
        output_format="RGB",
        normalize=True,
        resize_to=(64, 64),
        output_layout="BCHW",
    )
    assert stats["available"] is True
    assert stats["iterations"] == 5
    assert stats["median_ms"] >= 0.0
    assert stats["p95_ms"] >= stats["median_ms"]
    assert stats["backend_info"]["detail"] == "custom_cuda_kernel"
