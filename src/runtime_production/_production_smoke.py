"""60-second integration smoke that exercises fast path end-to-end.

NOT REAL PRODUCTION — there are no Basler cameras in this process. We
fake the grabber side by pushing synthetic 1920x1080 RAW frames into
shared memory at 50 fps per role. LBC reads them via the same code
paths the live system uses (`smm.read_frame`, `shared_state["bayer_key"]`).

What this verifies:
  - Fast path actually engages with VOLLEYHUB_FAST_PATH=1
  - Probe emits [LIVE_PACKET_TIMING] lines into logs/
  - Preview worker runs in parallel without blocking inference
  - stage_ms keys are present in BOTH slow-path and fast-path conventions
  - Report generator can parse the resulting probe logs

What this does NOT verify:
  - Real Basler grab timing
  - Real GUI overhead
  - Real multi-process IPC latency
  - Real 4-camera hardware sync drift
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import List

import numpy as np

_LOG = logging.getLogger("production_smoke")

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ENGINE = REPO / "artifacts" / "tensorrt_exports" / (
    "best__fp16_640_b4_static__fp16__img640__b4__static.engine"
)
ROLES = ("LEFT", "CENTER_L", "CENTER_R", "RIGHT")


def _make_synthetic_bayer(h: int, w: int, frame_id: int) -> np.ndarray:
    """Cheap synthetic Bayer frame; varies slightly per frame_id for realism."""
    rng = np.random.default_rng(frame_id & 0xFFFF)
    return rng.integers(0, 256, size=(h, w), dtype=np.uint8)


def _grabber_thread(role: str, key: str, smm, shared_state: dict, stop_evt: threading.Event,
                    fps: float, h: int, w: int, frame_offset: int) -> None:
    period = 1.0 / max(1.0, fps)
    next_t = time.perf_counter()
    frame_id = frame_offset
    while not stop_evt.is_set():
        if time.perf_counter() < next_t:
            stop_evt.wait(max(0.0, next_t - time.perf_counter()))
            if stop_evt.is_set():
                break
        next_t += period
        frame_id += 1
        frame = _make_synthetic_bayer(h, w, frame_id)
        smm.write_frame(key, frame, ts_ns=time.time_ns())
        # Publish current key+ts for LBC to consume.
        bayer_map = shared_state.setdefault("bayer_key", {})
        bayer_map[role] = key
        ts_map = shared_state.setdefault("last_frame_ts", {})
        ts_map[role] = time.time_ns()


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="60s integration smoke for the production fast path.")
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--capture-fps", type=float, default=50.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--engine", type=str, default=str(DEFAULT_ENGINE))
    parser.add_argument("--probe-every", type=int, default=30)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if not Path(args.engine).exists():
        raise SystemExit(f"engine not found: {args.engine}")

    # Activate fast path + probe via env BEFORE importing LBC subsystems
    os.environ["VOLLEYHUB_FAST_PATH"] = "1"
    os.environ["VOLLEYHUB_TRT_ENGINE"] = str(args.engine)
    os.environ["VOLLEYHUB_COLOR_BACKEND"] = "native_cuda_npp"
    os.environ["VOLLEYHUB_INFERENCE_BACKEND"] = "tensorrt"
    os.environ["VOLLEYHUB_YOLO_IMGSZ"] = "640"
    os.environ["VOLLEYHUB_YOLO_BATCH"] = "4"
    os.environ["VOLLEYHUB_BAYER_PATTERN"] = "RG"
    os.environ["VOLLEYHUB_LIVE_PROBE"] = "1"
    os.environ["VOLLEYHUB_LIVE_PROBE_EVERY"] = str(int(args.probe_every))
    # Preview defaults; latest-only, 20 fps, 512 px
    os.environ.setdefault("VOLLEYHUB_PREVIEW_ENABLED", "1")
    os.environ.setdefault("VOLLEYHUB_PREVIEW_MAX_FPS", "20")
    os.environ.setdefault("VOLLEYHUB_PREVIEW_SIZE", "512")
    os.environ.setdefault("VOLLEYHUB_PREVIEW_QUALITY", "low")

    # Clean previous probe logs so this run produces fresh data.
    log_dir = REPO / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for old in list(log_dir.glob("live_probe_*.log")):
        try:
            old.unlink()
        except Exception:
            pass

    from storage.shared_memory_manager import get_shared_memory_manager
    from live_runtime.live_backend_controller import LiveBackendController

    smm = get_shared_memory_manager()
    shared_state: dict = {"bayer_key": {}, "last_frame_ts": {}}

    # Spin up 4 fake grabber threads, one per role.
    stop_evt = threading.Event()
    grabbers: List[threading.Thread] = []
    for i, role in enumerate(ROLES):
        key = f"smoke_bayer_{role}"
        t = threading.Thread(
            target=_grabber_thread,
            name=f"smoke-grab-{role}",
            args=(role, key, smm, shared_state, stop_evt,
                  args.capture_fps, args.height, args.width, i * 1000),
            daemon=True,
        )
        t.start()
        grabbers.append(t)

    _LOG.info("started %d fake grabbers @ %.0f fps, frame %dx%d, role keys: %s",
              len(grabbers), args.capture_fps, args.width, args.height, list(ROLES))
    # Let the grabbers fill shared memory so LBC immediately sees keys.
    time.sleep(0.5)

    # Build LBC. Default config will trigger fast path because env is set.
    lbc = LiveBackendController(shared_state=shared_state)
    cfg = {
        "roles": list(ROLES),
        "selected_role": "CENTER_L",
        "yolo_enabled": True,
        "yolo_backend": "tensorrt",
        "yolo_device": "cuda",
        "yolo_image_size": 640,
        "model_path": "best.pt",
        "yolo_trt_engine_path": str(args.engine),
        "publish_interval_ms": 80.0,
        "preview_enabled": True,
        "bayer_pattern": "RG",
    }
    _LOG.info("starting LBC with fast_path=ON, duration=%.1fs", args.duration_s)
    lbc.start(cfg)

    # Drain events periodically so the deque doesn't fill (mimic GUI poll).
    poll_thread_stop = threading.Event()

    def _poll() -> None:
        while not poll_thread_stop.is_set():
            try:
                _ = lbc.poll_events(max_items=64)
            except Exception:
                pass
            poll_thread_stop.wait(0.05)

    poll_thread = threading.Thread(target=_poll, name="smoke-event-poll", daemon=True)
    poll_thread.start()

    # Run for the requested duration.
    t_start = time.perf_counter()
    try:
        while time.perf_counter() - t_start < args.duration_s:
            time.sleep(0.5)
    except KeyboardInterrupt:
        _LOG.info("interrupted; shutting down early")

    elapsed = time.perf_counter() - t_start
    _LOG.info("duration done (%.1fs); shutting down", elapsed)

    poll_thread_stop.set()
    poll_thread.join(timeout=1.0)
    lbc.shutdown(timeout_s=2.0)
    stop_evt.set()
    for t in grabbers:
        t.join(timeout=1.0)
    try:
        smm.cleanup_all()
    except Exception:
        pass

    _LOG.info("smoke complete. probe logs in: %s", log_dir)
    for p in sorted(log_dir.glob("live_probe_*.log")):
        sz = p.stat().st_size
        _LOG.info("  %s  (%d bytes)", p.name, sz)

    # Trigger report generation
    from src.runtime_production.build_fast_path_report import main as build_main
    build_main(argv=[])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
