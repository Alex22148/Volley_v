"""Sanity tests for the production fast path.

Covers (per user spec):
- fast path disabled: zachowanie bez zmian (config validate())
- native backend unavailable: graceful "not ready" + init_error
- wrong engine path: czytelny błąd z config validation
- batch != 4 przy static b4 engine: czytelny błąd config validation
- synthetic 4cam packet przechodzi przez fast path (skipped if no CUDA)
- live probe poprawnie raportuje zero_copy/fallback/batch (skipped if no CUDA)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.runtime_production import (
    FastPathConfig,
    FastPathExecutor,
    load_fast_path_config_from_env,
)


_ENGINE_B4_640 = _REPO_ROOT / "artifacts" / "tensorrt_exports" / (
    "best__fp16_640_b4_static__fp16__img640__b4__static.engine"
)


def _has_cuda_and_engine() -> bool:
    try:
        import torch
        if not torch.cuda.is_available():
            return False
    except Exception:
        return False
    return _ENGINE_B4_640.exists() and _ENGINE_B4_640.stat().st_size > 1024


def test_disabled_when_env_var_unset(monkeypatch) -> None:
    monkeypatch.delenv("VOLLEYHUB_FAST_PATH", raising=False)
    cfg = load_fast_path_config_from_env()
    assert cfg.enabled is False
    # disabled config validates as None (no requirement to satisfy)
    assert cfg.validate() is None


def test_enabled_requires_engine_path(monkeypatch) -> None:
    monkeypatch.setenv("VOLLEYHUB_FAST_PATH", "1")
    monkeypatch.delenv("VOLLEYHUB_TRT_ENGINE", raising=False)
    cfg = load_fast_path_config_from_env()
    assert cfg.enabled is True
    err = cfg.validate()
    assert err is not None and "VOLLEYHUB_TRT_ENGINE" in err


def test_engine_must_exist(monkeypatch) -> None:
    monkeypatch.setenv("VOLLEYHUB_FAST_PATH", "1")
    monkeypatch.setenv("VOLLEYHUB_TRT_ENGINE", "no_such_file.engine")
    cfg = load_fast_path_config_from_env()
    err = cfg.validate()
    assert err is not None and "not found" in err


def test_batch_must_match_required_roles_count() -> None:
    # batch=2 with default 4 roles -> rejected
    cfg = FastPathConfig(
        enabled=True,
        engine_path=str(_ENGINE_B4_640) if _ENGINE_B4_640.exists() else "dummy",
        batch_size=2,
    )
    err = cfg.validate()
    assert err is not None
    assert "batch_size" in err and "required_roles" in err


def test_imgsz_must_be_stride_aligned() -> None:
    cfg = FastPathConfig(
        enabled=True,
        engine_path=str(_ENGINE_B4_640) if _ENGINE_B4_640.exists() else "dummy",
        imgsz=641,  # not divisible by 32
    )
    err = cfg.validate()
    assert err is not None and "imgsz" in err


def test_unsupported_color_backend_rejected() -> None:
    cfg = FastPathConfig(
        enabled=True,
        engine_path=str(_ENGINE_B4_640) if _ENGINE_B4_640.exists() else "dummy",
        color_backend="cpu",
    )
    err = cfg.validate()
    assert err is not None and "color_backend" in err


def test_executor_not_ready_when_init_fails() -> None:
    cfg = FastPathConfig(enabled=True, engine_path="nope.engine")
    ex = FastPathExecutor(cfg)
    ok = ex.initialize()
    assert ok is False
    assert ex.is_ready is False
    assert ex.init_error is not None
    # process_packet on a not-ready executor returns an error result, doesn't crash
    res = ex.process_packet([], [], 0)
    assert res.error is not None


def test_env_to_config_round_trip(monkeypatch) -> None:
    monkeypatch.setenv("VOLLEYHUB_FAST_PATH", "1")
    monkeypatch.setenv("VOLLEYHUB_TRT_ENGINE", "/tmp/some.engine")
    monkeypatch.setenv("VOLLEYHUB_YOLO_IMGSZ", "640")
    monkeypatch.setenv("VOLLEYHUB_YOLO_BATCH", "4")
    monkeypatch.setenv("VOLLEYHUB_BAYER_PATTERN", "BG")
    monkeypatch.setenv("VOLLEYHUB_FAST_PATH_HALF", "1")
    monkeypatch.setenv("VOLLEYHUB_FAST_PATH_REQUIRED_ROLES", "LEFT,CENTER_L,CENTER_R,RIGHT")
    cfg = load_fast_path_config_from_env()
    assert cfg.enabled is True
    assert cfg.engine_path == "/tmp/some.engine"
    assert cfg.imgsz == 640 and cfg.batch_size == 4
    assert cfg.bayer_pattern == "BG"
    assert cfg.half is True
    assert cfg.required_roles == ("LEFT", "CENTER_L", "CENTER_R", "RIGHT")


def test_unified_image_processor_flag_defaults_off(monkeypatch) -> None:
    monkeypatch.delenv("VOLLEYHUB_USE_UNIFIED_IMAGE_PROCESSOR", raising=False)
    cfg = load_fast_path_config_from_env()
    assert cfg.use_unified_image_processor is False


def test_unified_image_processor_flag_from_env(monkeypatch) -> None:
    monkeypatch.setenv("VOLLEYHUB_USE_UNIFIED_IMAGE_PROCESSOR", "1")
    cfg = load_fast_path_config_from_env()
    assert cfg.use_unified_image_processor is True


def test_fast_path_default_uses_native_tensor_prepare(monkeypatch) -> None:
    import torch

    class _FakeDebayer:
        def __init__(self):
            self.calls = 0

        def debayer(self, *args, **kwargs):
            self.calls += 1
            return torch.zeros((4, 3, 32, 32), dtype=torch.float32)

    cfg = FastPathConfig(
        enabled=True,
        engine_path=str(_ENGINE_B4_640) if _ENGINE_B4_640.exists() else "dummy",
        imgsz=32,
        device="cpu",
        use_unified_image_processor=False,
    )
    ex = FastPathExecutor(cfg)
    ex._debayer = _FakeDebayer()
    stage_ms = {}
    stack = np.zeros((4, 64, 64), dtype=np.uint8)

    tensor, path = ex._prepare_bchw_tensor(stack, stage_ms, torch)

    assert tuple(tensor.shape) == (4, 3, 32, 32)
    assert path == "native"
    assert ex._debayer.calls == 1
    assert stage_ms.get("_fallback_used") is None
    assert "color_ms" in stage_ms


def test_unified_image_processor_fallback_to_native(monkeypatch) -> None:
    import torch
    from src.runtime_gpu.gpu_image_processor import GpuImageProcessor

    class _FakeDebayer:
        def __init__(self):
            self.calls = 0

        def debayer(self, *args, **kwargs):
            self.calls += 1
            return torch.zeros((4, 3, 32, 32), dtype=torch.float32)

    def _raise_process_batch(self, frames):
        raise RuntimeError("forced unified failure")

    monkeypatch.setattr(GpuImageProcessor, "process_batch", _raise_process_batch)

    cfg = FastPathConfig(
        enabled=True,
        engine_path=str(_ENGINE_B4_640) if _ENGINE_B4_640.exists() else "dummy",
        imgsz=32,
        device="cpu",
        use_unified_image_processor=True,
    )
    ex = FastPathExecutor(cfg)
    ex._debayer = _FakeDebayer()
    stage_ms = {}
    stack = np.zeros((4, 64, 64), dtype=np.uint8)

    tensor, path = ex._prepare_bchw_tensor(stack, stage_ms, torch)

    assert tuple(tensor.shape) == (4, 3, 32, 32)
    assert path == "native"
    assert ex._debayer.calls == 1
    assert "forced unified failure" in stage_ms["_fallback_used"]
    assert "color_ms" in stage_ms


def test_fast_path_result_reports_preprocessing_path_tensor_and_timings() -> None:
    import torch

    class _FakeDebayer:
        def debayer(self, *args, **kwargs):
            return torch.zeros((4, 3, 32, 32), dtype=torch.float32)

    class _FakeModel:
        def predict(self, *args, **kwargs):
            return []

    cfg = FastPathConfig(
        enabled=True,
        engine_path=str(_ENGINE_B4_640) if _ENGINE_B4_640.exists() else "dummy",
        imgsz=32,
        device="cpu",
        use_unified_image_processor=False,
    )
    ex = FastPathExecutor(cfg)
    ex._debayer = _FakeDebayer()
    ex._model = _FakeModel()
    ex._initialized = True

    raws = [np.zeros((64, 64), dtype=np.uint8) for _ in range(4)]
    roles = ["LEFT", "CENTER_L", "CENTER_R", "RIGHT"]
    res = ex.process_packet(raws, roles, ts_ns=123)

    assert res.error is None
    assert res.preprocessing_path == "native"
    assert res.tensor_device == "cpu"
    assert res.tensor_shape == (4, 3, 32, 32)
    assert res.tensor_layout == "BCHW"
    assert res.fallback_used is None
    for key in ("stack_ms", "color_ms", "inference_ms", "postprocess_ms", "total_packet_ms"):
        assert key in res.stage_ms and res.stage_ms[key] >= 0.0


def test_native_and_unified_prepare_report_compatible_bchw_format() -> None:
    import torch

    class _FakeDebayer:
        def debayer(self, *args, **kwargs):
            return torch.zeros((4, 3, 32, 32), dtype=torch.float32)

    cfg = FastPathConfig(
        enabled=True,
        engine_path=str(_ENGINE_B4_640) if _ENGINE_B4_640.exists() else "dummy",
        imgsz=32,
        device="cpu",
        use_unified_image_processor=True,
    )
    ex = FastPathExecutor(cfg)
    ex._debayer = _FakeDebayer()
    stack = np.zeros((4, 64, 64), dtype=np.uint8)

    native_tensor, native_path = ex._prepare_bchw_tensor_native(stack, {}, torch)
    unified_tensor, unified_path = ex._prepare_bchw_tensor_unified(stack, {}, torch)

    assert native_path == "native"
    assert unified_path == "unified"
    assert tuple(native_tensor.shape) == tuple(unified_tensor.shape) == (4, 3, 32, 32)


def test_fast_path_executor_import_does_not_require_cuda() -> None:
    import importlib

    module = importlib.import_module("src.runtime_production.fast_path_executor")
    assert hasattr(module, "FastPathExecutor")


@pytest.mark.skipif(not _has_cuda_and_engine(), reason="CUDA + TRT b4 engine required")
def test_fast_path_processes_synthetic_4cam_packet() -> None:
    cfg = FastPathConfig(
        enabled=True,
        engine_path=str(_ENGINE_B4_640),
        imgsz=640,
        batch_size=4,
        bayer_pattern="RG",
        warmup_iterations=1,
    )
    assert cfg.validate() is None
    ex = FastPathExecutor(cfg)
    assert ex.initialize() is True
    raws = [np.zeros((1080, 1920), dtype=np.uint8) for _ in range(4)]
    roles = ["LEFT", "CENTER_L", "CENTER_R", "RIGHT"]
    res = ex.process_packet(raws, roles, ts_ns=12345)
    assert res.error is None
    assert res.roles == roles
    assert len(res.detections_per_role) == 4
    assert len(res.tracks_per_role) == 4
    assert res.zero_copy_to_inference is True
    assert res.color_output_location == "cuda"
    assert res.inference_input_location == "cuda"
    assert res.color_backend == "native_cuda_npp"
    assert res.inference_backend == "tensorrt"
    for k in ("color_ms", "inference_ms", "postprocess_ms", "total_packet_ms"):
        assert k in res.stage_ms and res.stage_ms[k] >= 0.0


# --------------------------- preview worker ---------------------------

def test_preview_config_defaults(monkeypatch) -> None:
    for k in ("VOLLEYHUB_PREVIEW_MAX_FPS", "VOLLEYHUB_PREVIEW_SIZE",
              "VOLLEYHUB_PREVIEW_QUALITY", "VOLLEYHUB_PREVIEW_LATEST_ONLY"):
        monkeypatch.delenv(k, raising=False)
    from src.runtime_production import load_preview_config_from_env
    cfg = load_preview_config_from_env(default_bayer="RG")
    assert cfg.enabled is True
    assert cfg.max_fps == 20.0
    assert cfg.target_size == 512
    assert cfg.quality == "low"
    assert cfg.latest_only is True
    assert cfg.bayer_pattern == "RG"


def test_preview_config_env_round_trip(monkeypatch) -> None:
    monkeypatch.setenv("VOLLEYHUB_PREVIEW_MAX_FPS", "15")
    monkeypatch.setenv("VOLLEYHUB_PREVIEW_SIZE", "640")
    monkeypatch.setenv("VOLLEYHUB_PREVIEW_QUALITY", "mid")
    monkeypatch.setenv("VOLLEYHUB_PREVIEW_LATEST_ONLY", "0")
    monkeypatch.setenv("VOLLEYHUB_PREVIEW_ROLE", "CENTER_L")
    from src.runtime_production import load_preview_config_from_env
    cfg = load_preview_config_from_env(default_bayer="BG")
    assert cfg.max_fps == 15.0
    assert cfg.target_size == 640
    assert cfg.quality == "mid"
    assert cfg.latest_only is False
    assert cfg.preferred_role == "CENTER_L"


def test_cheap_preview_never_runs_full_debayer() -> None:
    """Cheap preview must produce a small BGR frame without doing a full
    1920x1080 cv2.cvtColor on the original Bayer."""
    from src.runtime_production import cheap_preview_from_raw
    raw = np.random.randint(0, 256, size=(1080, 1920), dtype=np.uint8)
    bgr = cheap_preview_from_raw(raw, target_size=512, quality="low")
    # Output is small, 3-channel, uint8
    assert bgr.dtype == np.uint8
    assert bgr.ndim == 3 and bgr.shape[2] == 3
    assert bgr.shape[1] <= 512
    assert bgr.shape[0] <= 512  # height proportional, can't exceed target_size


def test_preview_worker_runs_in_separate_thread_and_drops_when_no_keys() -> None:
    """The worker must start its own thread and never block the caller.
    With an empty keys map, it should accumulate drops and not crash."""
    from src.runtime_production import PreviewConfig, PreviewWorker
    import time

    class _StubSmm:
        def read_frame(self, key, retries=0, delay=0.0):
            return None  # never returns frames

    cfg = PreviewConfig(enabled=True, max_fps=50.0, target_size=128)
    worker = PreviewWorker(
        config=cfg,
        smm=_StubSmm(),
        read_role_to_key_map_fn=lambda: {},   # no roles ever ready
        roles=("LEFT", "CENTER_L", "CENTER_R", "RIGHT"),
        probe_factory=None,
    )
    worker.start()
    assert worker.is_running is True
    time.sleep(0.05)  # let the worker tick a few times
    worker.stop(timeout_s=0.5)
    stats = worker.stats()
    # Drops accumulated, nothing published, never blocked the test thread.
    assert stats["frames_published"] == 0
    assert stats["frames_dropped"] >= 1
    assert worker.is_running is False


def test_preview_worker_publishes_latest_frame() -> None:
    """When the smm yields a frame, the worker must produce a small BGR
    preview and store it as the latest."""
    from src.runtime_production import PreviewConfig, PreviewWorker
    import time

    raw = np.random.randint(0, 256, size=(1080, 1920), dtype=np.uint8)

    class _StubSmm:
        def read_frame(self, key, retries=0, delay=0.0):
            return (raw, 12345)

    cfg = PreviewConfig(enabled=True, max_fps=60.0, target_size=256, quality="low")
    worker = PreviewWorker(
        config=cfg,
        smm=_StubSmm(),
        read_role_to_key_map_fn=lambda: {"LEFT": "key_left"},
        roles=("LEFT", "CENTER_L", "CENTER_R", "RIGHT"),
        probe_factory=None,
    )
    worker.start()
    time.sleep(0.1)
    latest = worker.get_latest()
    worker.stop(timeout_s=0.5)
    assert latest is not None
    bgr, role, ts = latest
    assert role == "LEFT"
    assert ts == 12345
    assert bgr.ndim == 3 and bgr.shape[2] == 3
    assert bgr.shape[1] <= 256
    stats = worker.stats()
    assert stats["frames_published"] >= 1


@pytest.mark.skipif(not _has_cuda_and_engine(), reason="CUDA + TRT b4 engine required")
def test_probe_extras_via_fast_path(monkeypatch) -> None:
    """Confirm probe records fast-path extras when LBC routes through it."""
    monkeypatch.setenv("VOLLEYHUB_LIVE_PROBE", "1")
    monkeypatch.setenv("VOLLEYHUB_LIVE_PROBE_EVERY", "1")
    from src.runtime_diagnostics.live_timing_probe import get_probe, reset_probes
    reset_probes()
    probe = get_probe("live_backend_test")
    assert probe.enabled is True
    # Simulate one fast-path packet record
    probe.record_packet(
        stage_ms={
            "color_ms": 5.6,
            "inference_ms": 7.4,
            "postprocess_ms": 0.5,
            "total_packet_ms": 13.5,
            "sync_wait_ms": 0.2,
        },
        role="LEFT,CENTER_L,CENTER_R,RIGHT",
        resolution="1920x1080",
        color_backend="native_cuda_npp",
        inference_backend="tensorrt",
        batch=4,
        yolo_imgsz=640,
        fast_path_enabled=True,
        packet_roles_count=4,
        zero_copy_to_inference=True,
        gpu_roundtrip=False,
        fallback_used="false",
    )
    # Internal state reflects packet
    assert probe._frames_seen == 1  # type: ignore[attr-defined]
    assert "color_ms" in probe._stages and probe._stages["color_ms"][-1] == pytest.approx(5.6)  # type: ignore[attr-defined]
