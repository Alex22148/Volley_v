# utils_config.py
import os, time, threading
from queue import Queue
from multiprocessing import Event, Value

zoom_role_var = None

# Optymalizacja FPS
OPTIMAL_FPS_CONFIG = {
    "preview_gui": 12,      # 12 FPS dla GUI (wystarczająco płynnie)
    "remote_stream": 8,     # 8 FPS dla HTTP stream
    "snapshot": 1,          # 1 FPS (tylko na żądanie)
    "recording": 30,        # 30 FPS dla zapisu (pełna jakość)
}

# Limity kolejek
UNIVERSAL_QUEUE_SIZES = {
    "bayer_input": 8,       # Mniejsze kolejki = mniejsze opóźnienie
    "preview_output": 4,
    "remote_output": 3,
    "snapshot_output": 2
}

# ================= GLOBAL SETTINGS =================
live_q = None  # zostanie ustawione w main.py

RAW_DIR = os.environ.get("RAW_DIR", os.path.join(os.getcwd(), "sessions"))

RETRIEVE_TIMEOUT_MS = 50
BIN_BATCH_FRAMES = 16
BIN_ROLL_EVERY = 512
PER_ROLE_QUEUE_MAX = 25
FPS_ASSUMED = 50.0

# Warm-up
WARMUP_MAX_S = 6.0
WARMUP_MIN_FRAMES = 50  # ~2s @50fps

# =============== GLOBAL EVENTS ===============
recording_enabled = threading.Event()
streaming_enabled = threading.Event()
remote_stream_enabled = threading.Event()
stop_evt = threading.Event()
sync_ready_event = threading.Event()
CALIB_READY = threading.Event()

# =============== OFFSETS ===============
OFFSETS = {}
OFFSETS_LOCK = threading.Lock()

# =============== IO STATS ===============
IO_WRITE_INTERVAL_S = 0.5
io_lock = threading.Lock()
io_stats = {
    "roles": {},
    "ts": int(time.time())
}

# =============== QUEUES ===============
sync_in_q = Queue(maxsize=256)

STATS_QUEUE_MAX_raw = 10_000
STATS_QUEUE_MAX_fps = 2000
STATS_QUEUE_MAX_sync = 5000



import queue

def try_put(q, item):
    """Bezpiecznie dodaje element do kolejki, nadpisując najstarszy przy przepełnieniu."""
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            _ = q.get_nowait()
            q.put_nowait(item)
        except queue.Empty:
            pass



# ===========================================================
# === RUNTIME STATE (uzupełniane dynamicznie przez main.py) =
# ===========================================================

_RUNTIME_PROCS = {
    # uzupełniane podczas startu backendu w main.py:
    # {
    #   "cams": [...],
    #   "stream_source": <StreamSource>,
    #   "queues": [...],
    #   "stop_evt": <threading.Event>
    # }
}


