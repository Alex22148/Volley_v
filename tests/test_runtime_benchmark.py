"""Sanity tests for the runtime benchmark / simulator tor.

These tests exercise the diagnostic path only — they never touch the
production grabber, GUI, or live_runtime modules. They are intentionally
lightweight (small frames) to run on CI without a GPU or a model file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure repo root is on sys.path when invoked via `pytest` from the project root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.runtime_sources.base_frame_source import CAMERA_ROLES
from src.runtime_sources.synthetic_raw_4cam_source import (
    SyntheticRaw4CamSource,
    SyntheticSourceConfig,
)
from src.runtime_gpu.gpu_color_converter import (
    ColorConverterConfig,
    GpuColorConverter,
)
from src.runtime_inference.yolo_inference_engine import (
    InferenceConfig,
    YoloInferenceEngine,
)
from src.runtime_benchmark._runtime_metrics import (
    summarize_samples,
    write_frames_csv,
    write_json,
)


def _small_source(num_frames: int = 3) -> SyntheticRaw4CamSource:
    return SyntheticRaw4CamSource(
        SyntheticSourceConfig(
            mode="generated",
            width=128,
            height=96,
            fps=60.0,
            num_frames=num_frames,
            dtype="uint8",
            bayer_pattern="RG",
            ball_enabled=True,
            ball_radius_px=8,
        )
    )


def test_synthetic_source_yields_four_synchronized_frames() -> None:
    src = _small_source(num_frames=2)
    packets = list(src.iter_packets())
    assert len(packets) == 2
    for pkt in packets:
        assert len(pkt.frames) == 4
        assert tuple(f.role for f in pkt.frames) == CAMERA_ROLES
        assert len({f.timestamp_ns for f in pkt.frames}) == 1


def test_synthetic_raw_shape_matches_config() -> None:
    src = _small_source()
    pkt = next(iter(src.iter_packets()))
    for frame in pkt.frames:
        assert frame.image.ndim == 2
        assert frame.image.shape == (96, 128)
        assert frame.image.dtype == np.uint8
        assert frame.bayer_pattern == "RG"


def test_color_converter_returns_three_channel_bgr() -> None:
    src = _small_source()
    pkt = next(iter(src.iter_packets()))
    converter = GpuColorConverter(ColorConverterConfig(backend="cpu", target_image_size=0))
    result = converter.convert_packet(pkt)
    assert len(result.color_frames) == 4
    for frame in result.color_frames:
        assert frame.image.ndim == 3
        assert frame.image.shape[2] == 3
        assert frame.color_format == "BGR"
    assert result.timings.total_color_ms >= 0.0
    assert result.timings.backend == "cpu"
    assert result.timings.output_location == "cpu"
    assert result.timings.download_from_gpu_ms == 0.0
    assert result.gpu_batch is None


def test_zero_copy_mode_requires_target_image_size() -> None:
    with pytest.raises(ValueError):
        GpuColorConverter(
            ColorConverterConfig(
                backend="torch_gpu",
                output_mode="torch_cuda_bchw_rgb_norm",
                target_image_size=0,
            )
        )


def test_zero_copy_mode_falls_back_when_torch_unavailable() -> None:
    # On a CPU-only host this should warn and fall back to numpy output;
    # the result should still be usable for inference (color_frames populated).
    converter = GpuColorConverter(
        ColorConverterConfig(
            backend="cpu",
            output_mode="torch_cuda_bchw_rgb_norm",
            target_image_size=64,
        )
    )
    src = _small_source()
    pkt = next(iter(src.iter_packets()))
    result = converter.convert_packet(pkt)
    # cpu backend cannot honor zero-copy; we must get either gpu_batch or
    # a clean numpy fallback with a recorded warning.
    if result.gpu_batch is None:
        assert len(result.color_frames) == 4
        assert result.fallback_warnings, "expected a fallback warning"


def test_color_converter_resize_targets_imgsz() -> None:
    src = _small_source()
    pkt = next(iter(src.iter_packets()))
    converter = GpuColorConverter(ColorConverterConfig(backend="cpu", target_image_size=64))
    result = converter.convert_packet(pkt)
    for frame in result.color_frames:
        assert frame.image.shape[:2] == (64, 64)


def test_inference_engine_dry_run_no_model() -> None:
    engine = YoloInferenceEngine(
        InferenceConfig(model_path=None, dry_run=True, batch_size=4, image_size=64)
    )
    assert engine.is_dry_run is True
    src = _small_source()
    pkt = next(iter(src.iter_packets()))
    converter = GpuColorConverter(ColorConverterConfig(backend="cpu", target_image_size=64))
    result = converter.convert_packet(pkt)
    out = engine.infer_color_frames(result.color_frames)
    assert len(out.results) == 4
    for r in out.results:
        assert r.num_detections == 0
        assert r.backend == "dry_run"
    # Telemetry: even in dry-run the timings dict carries the new keys.
    sample_timing = out.results[0].timing_ms
    for key in (
        "preprocess_ms",
        "upload_to_gpu_ms",
        "inference_ms",
        "input_location",
        "zero_copy_to_inference",
        "gpu_debayer_only_not_full_gpu_pipeline",
    ):
        assert key in sample_timing


def test_inference_engine_requires_model_when_not_dry_run() -> None:
    with pytest.raises(ValueError):
        YoloInferenceEngine(InferenceConfig(model_path=None, dry_run=False))


def test_summarize_samples_handles_empty_and_small() -> None:
    empty = summarize_samples([])
    assert empty["count"] == 0
    one = summarize_samples([12.0])
    assert one["count"] == 1
    assert one["mean_ms"] == pytest.approx(12.0)
    assert one["p95_ms"] == pytest.approx(12.0)


def test_benchmark_writes_json_and_csv(tmp_path: Path) -> None:
    payload = {"hello": "world", "n": 3}
    out_json = tmp_path / "benchmark_summary.json"
    write_json(out_json, payload)
    assert out_json.exists()
    loaded = json.loads(out_json.read_text(encoding="utf-8"))
    assert loaded["hello"] == "world"

    rows = [
        {"packet_id": i, "color_total_ms": 1.5 * i} for i in range(3)
    ]
    out_csv = tmp_path / "benchmark_frames.csv"
    write_frames_csv(out_csv, rows, ["packet_id", "color_total_ms"])
    assert out_csv.exists()
    text = out_csv.read_text(encoding="utf-8")
    assert "packet_id,color_total_ms" in text
    assert text.count("\n") >= 4  # header + 3 rows + trailing


def test_synthetic_source_invalid_dimensions_rejected() -> None:
    with pytest.raises(ValueError):
        SyntheticRaw4CamSource(
            SyntheticSourceConfig(
                mode="generated",
                width=129,  # odd, must be rejected
                height=96,
                num_frames=1,
            )
        )


def test_folder_mode_requires_path() -> None:
    with pytest.raises(ValueError):
        SyntheticRaw4CamSource(
            SyntheticSourceConfig(mode="folder", width=64, height=64, num_frames=1)
        )


def test_zero_copy_full_gpu_pipeline_when_cuda_available() -> None:
    """Verifies the zero-copy contract end-to-end when CUDA is present."""
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available on this host")
    except Exception:
        pytest.skip("torch not available")

    src = _small_source()
    pkt = next(iter(src.iter_packets()))
    converter = GpuColorConverter(
        ColorConverterConfig(
            backend="torch_gpu",
            output_mode="torch_cuda_bchw_rgb_norm",
            target_image_size=64,
            normalize_01=True,
            half_precision=False,
        )
    )
    result = converter.convert_packet(pkt)
    assert result.gpu_batch is not None, "gpu_zero_copy mode must produce a GPU batch"
    assert result.timings.output_location == "cuda"
    assert result.timings.download_from_gpu_ms == 0.0
    tensor = result.gpu_batch.tensor
    assert tensor.is_cuda, "tensor must live on CUDA"
    assert tuple(tensor.shape) == (4, 3, 64, 64)
    assert result.gpu_batch.color_format == "RGB"

    # Pass the GPU batch through the engine in dry-run mode and verify
    # the engine reports zero-copy without an upload.
    engine = YoloInferenceEngine(
        InferenceConfig(model_path=None, dry_run=True, image_size=64)
    )
    out = engine.infer_gpu_batch(result.gpu_batch)
    assert len(out.results) == 4
    assert out.timings.input_location == "cuda"
    assert out.timings.zero_copy_to_inference is True
    assert out.timings.upload_to_gpu_ms == 0.0
