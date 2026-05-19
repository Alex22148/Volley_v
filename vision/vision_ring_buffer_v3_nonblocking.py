import numpy as np
import threading
import queue
import time
from pathlib import Path

# Diagnostic probe (off by default; activated by env VOLLEYHUB_LIVE_PROBE=1).
try:
    from src.runtime_diagnostics.live_timing_probe import get_probe as _get_live_probe  # type: ignore
except Exception:
    def _get_live_probe(_tag: str):  # type: ignore[no-redef]
        class _Noop:
            enabled = False
            def record_packet(self, *_a, **_kw): return None
        return _Noop()


class DumpWorker(threading.Thread):

    def __init__(self, queue):
        super().__init__(daemon=True)
        self.queue = queue

    def run(self):

        while True:

            item = self.queue.get()

            if item is None:
                break

            buffer_ref, timestamps_ref, start, count, path = item

            self.dump(buffer_ref, timestamps_ref, start, count, path)

    def dump(self, buffer, timestamps, start, count, path):

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        filename = path / f"vision_dump_{time.time_ns()}.bin"

        with open(filename, "wb", buffering=1024 * 1024 * 64) as f:

            for i in range(count):

                idx = (start + i) % buffer.shape[0]

                ts = np.uint64(timestamps[idx])
                f.write(ts.tobytes())

                f.write(buffer[idx].tobytes())

        print(f"[DUMP DONE] {filename} | frames: {count}")


# ============================================================


class VisionRingBuffer:

    def __init__(self, width, height, cameras, fps, seconds):

        self.capacity = fps * seconds

        self.buffer = np.empty(
            (self.capacity, cameras, height, width),
            dtype=np.uint8
        )

        self.timestamps = np.zeros(self.capacity, dtype=np.uint64)

        self.index = 0
        self.full = False

        self.lock = threading.Lock()

        self.dump_queue = queue.Queue()
        self.dump_worker = DumpWorker(self.dump_queue)
        self.dump_worker.start()

    # -----------------------------------------------------

    def push(self, frames):

        ts = time.time_ns()

        # Diagnostic probe (no-op unless VOLLEYHUB_LIVE_PROBE=1)
        try:
            _probe = _get_live_probe("ring_buffer")
            _probe_enabled = bool(getattr(_probe, "enabled", False))
            _push_t0 = time.perf_counter() if _probe_enabled else 0.0
        except Exception:
            _probe_enabled = False
            _probe = None
            _push_t0 = 0.0

        with self.lock:

            idx = self.index

            for cam in range(len(frames)):
                self.buffer[idx, cam][:] = frames[cam]

            self.timestamps[idx] = ts

            self.index += 1

            if self.index >= self.capacity:
                self.index = 0
                self.full = True

        if _probe_enabled and _probe is not None:
            try:
                _probe.record_packet(
                    stage_ms={"buffer_push_ms": (time.perf_counter() - _push_t0) * 1000.0},
                    n_roles_seen=int(len(frames)),
                )
            except Exception:
                pass

    # -----------------------------------------------------

    def dump_async(self, path="dump", require_full=True):

        # 🔴 KRÓTKI LOCK – tylko indeksy
        with self.lock:

            if require_full and not self.full:
                print("[WARNING] buffer not full yet")
                return

            if not self.full:
                start = 0
                count = self.index
            else:
                start = self.index
                count = self.capacity

        # 🔴 ZERO BLOKADY – reszta poza lockiem
        self.dump_queue.put(
            (self.buffer, self.timestamps, start, count, path)
        )

    # -----------------------------------------------------

    def stop(self):
        self.dump_queue.put(None)