# grabber_module.py
import json
import queue
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
from pypylon import pylon
from storage.shared_memory_manager import get_shared_memory_manager
from streaming.server_stream import get_selected_role

# đźŽ¬ BUFOR - Import VisionRingBuffer
try:
    from vision.vision_ring_buffer_v3_nonblocking import VisionRingBuffer
    BUFFER_AVAILABLE = True
    print("[BUFFER] ... VisionRingBuffer dostÄ™pny")
except ImportError as e:
    BUFFER_AVAILABLE = False
    print(f"[BUFFER] âš ď¸Ź VisionRingBuffer niedostÄ™pny: {e}")
    VisionRingBuffer = None

# Globalny bufor dla wszystkich kamer (jak w demo)
_GLOBAL_MULTI_BUFFER = None
_GLOBAL_MULTI_BUFFER_LOCK = threading.Lock()
_BUFFER_FRAMES = {}
_BUFFER_META = {}
_BUFFER_TRACE = deque(maxlen=5000)  # per full 4-camera set
_BUFFER_SET_COUNTER = 0

_TS_LOG_LOCK = threading.Lock()
_TS_LOG_FH = None
_TS_LOG_PATH = None




def grabber(cam_idx, cam, role, serial,
            raw_q,
            universal_bayer_q,
            stats_q,
            stop_evt,
            streaming_enabled,
            remote_stream_enabled,
            recording_event, snapshot_event, snapshot_raw_q, yolo_raw_q,
            *args, **kwargs):
    """
    Back-compat: akceptuj dodatkowe kwargs (np. shared_state),
    ĹĽeby nie wysypywaÄ‡ się gdy wywoĹ‚ujÄ…cy przekaĹĽe nowe pola.
    """
    import time
    import queue
    import threading
    import numpy as np
    import json
    from pathlib import Path
    from pypylon import pylon

    shared_state = kwargs.get("shared_state", None)

    # mapa ostatnich timestampĂłw ramek per rola (do watchdoga)
    last_ts_map = None
    try:
        if shared_state is not None:
            last_ts_map = shared_state.get("last_frame_ts")
            if last_ts_map is None:
                shared_state["last_frame_ts"] = {}
                last_ts_map = shared_state["last_frame_ts"]
    except Exception:
        last_ts_map = None

    # SharedMemoryManager do klatek Bayer/RGB
    smm = get_shared_memory_manager()

    # mapa ostatniego klucza Bayer per rola
    bayer_map = None
    try:
        if shared_state is not None:
            bayer_map = shared_state.get("bayer_key")
            if bayer_map is None:
                shared_state["bayer_key"] = {}
                bayer_map = shared_state["bayer_key"]
    except Exception:
        bayer_map = None

    # upewnij się, ĹĽe mamy Event dla tej roli
    if role not in streaming_enabled:
        streaming_enabled[role] = threading.Event()

    frame_counter = 0
    last_universal_time = 0.0
    target_universal_fps = 12.0
    skip_counter = 0
    log_interval_s = 4.0
    last_grabber_yolo_log_t = 0.0
    last_grabber_preview_log_t = 0.0

    ts_debug_every = 0
    frame_idx_local = 0

    global _TS_LOG_FH, _TS_LOG_PATH

    try:
        with _TS_LOG_LOCK:
            if ts_debug_every and _TS_LOG_FH is None:
                log_dir = Path("../logs") / "camera_timestamps"
                log_dir.mkdir(parents=True, exist_ok=True)
                _TS_LOG_PATH = log_dir / "all_cameras_timestamps.jsonl"
                _TS_LOG_FH = open(_TS_LOG_PATH, "a", encoding="utf-8", buffering=1)
                print(f"[TS_LOG] shared -> {_TS_LOG_PATH}")
    except Exception as e:
        print(f"[TS_LOG] âš ď¸Ź shared open failed: {e}")
        _TS_LOG_FH = None

    # bandwidth (PC-side)
    bw_bytes = 0
    bw_frames = 0
    bw_t0 = time.perf_counter()

    trace_enabled = False
    trace_refresh_t = 0.0

    def pipeline_logs_enabled() -> bool:
        try:
            return bool(shared_state.get("verbose_pipeline_logs", False)) if shared_state is not None else False
        except Exception:
            return False

    while not stop_evt.is_set():
        try:
            if not cam.IsGrabbing():
                time.sleep(0.01)
                continue
        except Exception:
            time.sleep(0.01)
            continue

        gr = None
        try:
            gr = cam.RetrieveResult(50, pylon.TimeoutHandling_Return)
            if not gr or not gr.GrabSucceeded():
                continue

            # --- bandwidth (PC-side): ile realnie odebraliĹ›my ---
            try:
                payload = int(gr.GetPayloadSize())
            except Exception:
                payload = 0

            if payload <= 0:
                try:
                    payload = int(gr.Array.nbytes)
                except Exception:
                    payload = 0

            bw_bytes += payload
            bw_frames += 1

            nowp = time.perf_counter()
            dt = nowp - bw_t0
            if dt >= 1.0:
                pc_bps = bw_bytes / dt if dt > 0 else 0.0
                pc_mbps = (pc_bps * 8) / 1e6
                fps = bw_frames / dt if dt > 0 else 0.0

                try:
                    stats_q.put_nowait({
                        "type": "bandwidth",
                        "role": role,
                        "serial": serial,
                        "pc_bps": pc_bps,
                        "pc_mbps": pc_mbps,
                        "pc_MBps": pc_bps / 1e6,
                        "fps": fps,
                    })
                except Exception:
                    pass

                bw_bytes = 0
                bw_frames = 0
                bw_t0 = nowp

            # --- timestamp z chunka kamery ---
            from capture.basler_lib import grab_ts_ns_from_chunk, ns_per_tick_for
            ts_ns = grab_ts_ns_from_chunk(gr, ns_per_tick_fallback=ns_per_tick_for(cam))
            ts_source = "camera_chunk"
            if ts_ns is None:
                ts_ns = time.time_ns()
                ts_source = "system_time"

            # zapisz do shared_state ostatni ts tej roli
            try:
                if last_ts_map is not None:
                    last_ts_map[role] = int(ts_ns)
            except Exception:
                pass

            # --- surowa klatka Bayer/mono ---
            arr = gr.Array
            frame_idx_local += 1

            if ts_debug_every and _TS_LOG_FH is not None and (frame_idx_local % ts_debug_every == 0):
                try:
                    rec = {
                        "role": role,
                        "serial": serial,
                        "cam_idx": cam_idx,
                        "frame_idx_local": frame_idx_local,
                        "ts_ns": int(ts_ns),
                        "ts_source": ts_source,
                        "pc_time_ns": time.time_ns(),
                        "payload_bytes": int(payload),
                    }
                    line = json.dumps(rec, ensure_ascii=False) + "\n"
                    with _TS_LOG_LOCK:
                        _TS_LOG_FH.write(line)
                except Exception as e:
                    if frame_idx_local % 200 == 0:
                        print(f"[TS_LOG] âš ď¸Ź write failed for {role}: {e}")

            try:
                if not arr.flags.c_contiguous:
                    arr = np.ascontiguousarray(arr)
            except Exception:
                arr = np.ascontiguousarray(np.asarray(arr))

            # =========================================================
            # 1) Globalny bufor 4-kamerowy
            # =========================================================
            if BUFFER_AVAILABLE:
                try:
                    global _GLOBAL_MULTI_BUFFER, _GLOBAL_MULTI_BUFFER_LOCK, _BUFFER_FRAMES, _BUFFER_META, _BUFFER_TRACE, _BUFFER_SET_COUNTER
                    now_trace = time.time()
                    if shared_state is not None and now_trace >= trace_refresh_t:
                        try:
                            trace_enabled = bool(shared_state.get("buffer_trace_enabled", False))
                        except Exception:
                            trace_enabled = False
                        trace_refresh_t = now_trace + 0.5

                    if _GLOBAL_MULTI_BUFFER is None:
                        with _GLOBAL_MULTI_BUFFER_LOCK:
                            if _GLOBAL_MULTI_BUFFER is None:
                                height, width = arr.shape[:2]
                                buffer_seconds = 5
                                try:
                                    if shared_state is not None:
                                        buffer_seconds = int(shared_state.get("buffer_dt_s", 5))
                                except Exception:
                                    pass

                                _GLOBAL_MULTI_BUFFER = VisionRingBuffer(
                                    width=width,
                                    height=height,
                                    cameras=4,
                                    fps=50,
                                    seconds=max(1, buffer_seconds),
                                )
                                cap = int(getattr(_GLOBAL_MULTI_BUFFER, "capacity", max(1, buffer_seconds * 50)))
                                _BUFFER_TRACE = deque(maxlen=max(128, cap + 32))
                                _BUFFER_SET_COUNTER = 0
                                print(f"[BUFFER] ... VisionRingBuffer init: {max(1, buffer_seconds)}s")

                    expected_roles = ["CENTER_L", "CENTER_R", "LEFT", "RIGHT"]

                    with _GLOBAL_MULTI_BUFFER_LOCK:
                        _BUFFER_FRAMES[role] = arr.copy()
                        if trace_enabled:
                            _BUFFER_META[role] = (
                                int(ts_ns),
                                int(frame_idx_local),
                                int(time.time_ns()),
                            )

                        if all(r in _BUFFER_FRAMES for r in expected_roles):
                            frames_to_push = [
                                _BUFFER_FRAMES["CENTER_L"],
                                _BUFFER_FRAMES["CENTER_R"],
                                _BUFFER_FRAMES["LEFT"],
                                _BUFFER_FRAMES["RIGHT"],
                            ]
                            _GLOBAL_MULTI_BUFFER.push(frames_to_push)
                            if trace_enabled:
                                try:
                                    set_meta = {}
                                    ts_vals = []
                                    for r in expected_roles:
                                        m = _BUFFER_META.get(r)
                                        if isinstance(m, tuple) and len(m) >= 3:
                                            ts_v = int(m[0])
                                            set_meta[r] = {
                                                "ts_ns": ts_v,
                                                "frame_idx_local": int(m[1]),
                                                "captured_at_unix_ns": int(m[2]),
                                            }
                                            ts_vals.append(ts_v)
                                    _BUFFER_SET_COUNTER += 1
                                    _BUFFER_TRACE.append({
                                        "set_idx": int(_BUFFER_SET_COUNTER),
                                        "pushed_at_unix_ns": int(time.time_ns()),
                                        "per_role": set_meta,
                                        "ts_min_ns": int(min(ts_vals)) if ts_vals else None,
                                        "ts_max_ns": int(max(ts_vals)) if ts_vals else None,
                                        "ts_span_ms": float(((max(ts_vals) - min(ts_vals)) / 1e6) if ts_vals else 0.0),
                                    })
                                except Exception:
                                    pass
                            _BUFFER_FRAMES.clear()
                            if trace_enabled:
                                _BUFFER_META.clear()

                except Exception as e:
                    if frame_idx_local % 200 == 0:
                        print(f"[BUFFER] âš ď¸Ź Błąd bufora {role}: {e}")

            # =========================================================
            # 2) Surowy tor sync / zapis
            # =========================================================
            try:
                raw_q.put_nowait((role, ts_ns, arr))
            except queue.Full:
                pass

            # =========================================================
            # 3) Legacy YOLO path
            # LIVE_TRACK does not consume yolo_raw_q anymore.
            # Freshest-frame LIVE backend reads directly from shared_state["bayer_key"].
            # Keep this path only for legacy modules that still depend on yolo_raw_q.
            # =========================================================
            try:
                yolo_active = bool(shared_state.get("yolo_enabled", False)) if shared_state is not None else False
                live_track_active = bool(
                    shared_state.get("live_track_enabled", False)) if shared_state is not None else False
            except Exception:
                yolo_active = False
                live_track_active = False

            if yolo_active and (not live_track_active) and yolo_raw_q is not None:
                try:
                    yolo_key = f"{role}_yolo_raw_{ts_ns}"
                    ok_yolo = smm.write_frame(yolo_key, arr, ts_ns=ts_ns)
                    if ok_yolo:
                        try:
                            yolo_raw_q.put_nowait((role, yolo_key, ts_ns))
                        except queue.Full:
                            try:
                                _ = yolo_raw_q.get_nowait()  # latest-ish fallback
                            except Exception:
                                pass
                            try:
                                yolo_raw_q.put_nowait((role, yolo_key, ts_ns))
                            except Exception:
                                pass

                        _now_log = time.time()
                        if pipeline_logs_enabled() and (_now_log - last_grabber_yolo_log_t) >= log_interval_s:
                            print(f"[GRABBER_YOLO] pushed role={role} ts={ts_ns}")
                            last_grabber_yolo_log_t = _now_log
                except Exception as e:
                    _now_log = time.time()
                    if pipeline_logs_enabled() and (_now_log - last_grabber_yolo_log_t) >= log_interval_s:
                        print(f"[GRABBER_YOLO] push failed role={role}: {e}")
                        last_grabber_yolo_log_t = _now_log

            # =========================================================
            # 4) Snapshot (jednorazowo)
            # =========================================================
            if snapshot_event.is_set():
                if not hasattr(snapshot_event, "sent_once"):
                    snapshot_event.sent_once = {}

                if not snapshot_event.sent_once.get(role, False):
                    try:
                        snapshot_raw_q.put_nowait((role, ts_ns, arr.copy()))
                        snapshot_event.sent_once[role] = True
                    except queue.Full:
                        pass

            # =========================================================
            # 4.5) LIVE_TRACK freshest-frame source
            # Always publish latest Bayer frame key for LIVE_TRACK,
            # independent from preview/remote state.
            # =========================================================
            try:
                live_track_active = bool(
                    shared_state.get("live_track_enabled", False)) if shared_state is not None else False
            except Exception:
                live_track_active = False

            if live_track_active:
                try:
                    live_bayer_key = f"{role}_live_bayer_{ts_ns}"
                    ok_live = smm.write_frame(live_bayer_key, arr, ts_ns=ts_ns)
                    if ok_live:
                        try:
                            if bayer_map is not None:
                                bayer_map[role] = live_bayer_key
                        except Exception:
                            pass
                        if pipeline_logs_enabled() and (time.time() - last_grabber_preview_log_t) >= log_interval_s:
                            print(f"[GRABBER_LIVE] role={role} key={live_bayer_key} ts={ts_ns}")
                            last_grabber_preview_log_t = time.time()
                except Exception as e:
                    if pipeline_logs_enabled() and (time.time() - last_grabber_preview_log_t) >= log_interval_s:
                        print(f"[GRABBER_LIVE] push failed role={role}: {e}")
                        last_grabber_preview_log_t = time.time()

            # =========================================================
            # 5) Tor preview / remote -> universal_color_processor
            # =========================================================
            now = time.time()

            try:
                preview_active = streaming_enabled[role].is_set()
            except Exception:
                preview_active = False

            try:
                selected_role = get_selected_role()
            except Exception:
                selected_role = None

            remote_active = bool(remote_stream_enabled.is_set() and selected_role == role)

            # YOLO ma wĹ‚asny tor przez yolo_raw_q -> nie dublujemy go tu
            universal_active = (preview_active or remote_active)

            send_for_preview_remote = False
            skip_counter += 1
            should_send_preview_remote = (skip_counter % 4 == 0)

            if universal_active:
                if should_send_preview_remote and (now - last_universal_time) >= (1.0 / target_universal_fps):
                    send_for_preview_remote = True

            if universal_active and send_for_preview_remote:
                try:
                    bayer_key = f"{role}_bayer_universal_{ts_ns}"
                    ok_preview = smm.write_frame(bayer_key, arr, ts_ns=ts_ns)

                    if ok_preview:
                        try:
                            if bayer_map is not None:
                                bayer_map[role] = bayer_key
                        except Exception:
                            pass

                        try:
                            universal_bayer_q.put_nowait((role, bayer_key, ts_ns))
                            if pipeline_logs_enabled() and (now - last_grabber_preview_log_t) >= log_interval_s:
                                print(
                                    f"[GRABBER_PREVIEW] role={role} preview_active={preview_active} remote_active={remote_active} pushed ts={ts_ns}")
                                last_grabber_preview_log_t = now
                            last_universal_time = now
                            frame_counter += 1

                            if pipeline_logs_enabled() and (now - last_grabber_preview_log_t) >= log_interval_s:
                                print(f"[GRABBER_PREVIEW] pushed role={role} ts={ts_ns}")
                                last_grabber_preview_log_t = now
                        except queue.Full:
                            pass

                except Exception as e:
                    if pipeline_logs_enabled() and (now - last_grabber_preview_log_t) >= log_interval_s:
                        print(f"[GRABBER_PREVIEW] push failed role={role}: {e}")
                        last_grabber_preview_log_t = now

        finally:
            try:
                if gr is not None:
                    gr.Release()
            except Exception:
                pass

    try:
        with _TS_LOG_LOCK:
            if _TS_LOG_FH is not None:
                _TS_LOG_FH.flush()
    except Exception:
        pass


def preview_worker(preview_q_rgb, streaming_enabled, stop_evt, live_q):
    import time
    import queue

    print("[PREVIEW] ▶️ Worker started - waiting for frames...")

    last_sent_per_role = {}
    target_fps = 15

    if not isinstance(preview_q_rgb, dict):
        print("[PREVIEW] âťŚ preview_q_rgb powinno byÄ‡ sĹ‚ownikiem {role: Queue}")
        return

    while not stop_evt.is_set():
        for role, q in preview_q_rgb.items():
            try:
                if not streaming_enabled[role].is_set():
                    continue
            except Exception as e:
                print(f"[PREVIEW][{role}] âš ď¸Ź streaming_enabled access error: {e}")
                continue

            latest = None
            try:
                while True:
                    latest = q.get_nowait()
            except queue.Empty:
                pass

            if latest is None:
                continue

            role2, ts_ns, frame_rgb = latest
            if role2 != role:
                print(f"[PREVIEW][{role}] âš ď¸Ź role mismatch: {role2} vs {role}")
                continue

            now = time.time()
            last_sent = last_sent_per_role.get(role, 0)
            if now - last_sent < 1.0 / target_fps:
                continue
            last_sent_per_role[role] = now

            try:
                while not live_q.empty():
                    _ = live_q.get_nowait()

                live_q.put_nowait((role, frame_rgb, ts_ns))
                time.sleep(0.002)

            except queue.Full:
                print(f"[PREVIEW][{role}] âš ď¸Ź live_q full - skipping frame")

        time.sleep(0.005)  # niewielka pauza, ĹĽeby nie zajeĹĽdĹĽaÄ‡ CPU

    print("[PREVIEW] âŹą Worker exiting cleanly.")

def async_sync_worker(file_ref, stop_evt):
    while not stop_evt.is_set():
        time.sleep(2)
        try:
            if file_ref[0]:
                file_ref[0].flush()
        except Exception:
            pass


def saver_worker_bin(role, in_q, root_dir, batch_frames, roll_every,
                     recording_event, stop_evt, stats_q=None, expected_fps=50.0,
                     resume_dir=None):  # đź†• dodany argument resume_dir

    import os, time, numpy as np, queue
    from pathlib import Path
    from typing import Optional
    from datetime import datetime
    import core.utils_config as utils_config
    recording_event = recording_event or utils_config.recording_event
    print(f"[SAVER][{role}] start")

    FSYNC_SECS  = float(os.environ.get("VH_FSYNC_SECS", "2.0"))
    REPORT_EVERY = 3.0

    f = None
    file_idx = 0
    frames_in_file = 0
    session_dir: Optional[Path] = None
    recording_was_active = False
    draining_after_stop = False

    saved_frames_total = 0
    start_time = time.time()
    last_report = 0.0
    bytes_since_sync = 0
    last_sync_ts = time.time()
    frames_seen_total = 0

    HEADER_TAG = b'BFRM'
    HEADER_FIXED_BYTES = 4 + 4 + 4 + 4 + 8 + 4

    def current_root():
        return root_dir() if callable(root_dir) else root_dir

    def resolve_session_dir() -> Path:
        """đź§  Jeśli resume_dir istnieje â€“ kontynuuj w nim."""
        if resume_dir is not None and Path(resume_dir).exists():
            print(f"[SAVER][{role}] ▶️ KontynuujÄ™ zapis w {resume_dir}")
            return Path(resume_dir)

        root = current_root()
        if root is None:
            base = Path.cwd() / "sessions"
        else:
            base = Path(str(root))

        if base.name.lower().startswith("session_"):
            sess = base
        else:
            sessions_root = base if base.name.lower() == "sessions" else (base / "sessions")
            sessions_root.mkdir(parents=True, exist_ok=True)
            sess = sessions_root / f"session_{datetime.now():%Y%m%d_%H%M%S}"

        sess.mkdir(parents=True, exist_ok=True)
        (sess / "snapshots").mkdir(parents=True, exist_ok=True)

        # đź†• zapisz w utils_config do wykorzystania przy reconnect
        utils_config._CURRENT_SESSION_DIR = str(sess)
        return sess

    def ensure_role_folder(role: str, sess: Path) -> Path:
        p = Path(sess) / role
        p.mkdir(parents=True, exist_ok=True)
        return p

    def open_new_file(role: str, ts_ns: int, idx: int, sess: Optional[Path]):
        nonlocal session_dir
        if sess is None:
            session_dir = resolve_session_dir()
            sess = session_dir
        ts = datetime.fromtimestamp(ts_ns / 1e9)
        role_dir = ensure_role_folder(role, sess)
        filename = f"{role}_{ts:%Y%m%d_%H%M%S}_{idx:04d}.bin"
        return open(role_dir / filename, "ab", buffering=(1 << 20))

    def do_flush_sync():
        nonlocal bytes_since_sync, last_sync_ts
        try:
            if f:
                f.flush()
        finally:
            bytes_since_sync = 0
            last_sync_ts = time.time()

    prev_rec_state = False

    while not stop_evt.is_set():
        current_rec = recording_event.is_set()

        # ▶️ START nagrywania
        if current_rec and not prev_rec_state:
            print(f"[SAVER][{role}] ▶️ Recording START - otwieram nowy plik")
            try:
                if f:
                    f.flush(); f.close()
            except Exception:
                pass
            f = None
            start_time = time.time()
            saved_frames_total = 0
            frames_in_file = 0
            bytes_since_sync = 0
            last_sync_ts = time.time()
            frames_seen_total = 0

            # đź§  Ustal katalog sesji
            session_dir = resolve_session_dir()
            try:
                role_dir = ensure_role_folder(role, session_dir)
                existing_bins = sorted(role_dir.glob(f"{role}_*.bin"))
                if existing_bins:
                    last_name = existing_bins[-1].name
                    base_idx = int(last_name.rsplit("_", 1)[-1].split(".")[0]) + 1
                else:
                    base_idx = 0
            except Exception:
                base_idx = 0

            file_idx = base_idx
            recording_was_active = True
            draining_after_stop = False

        # âŹą STOP nagrywania
        if (not current_rec) and prev_rec_state and recording_was_active and not draining_after_stop:
            print(f"[SAVER][{role}] âŹą Stop - czekam aĹĽ kolejka się opróżni")
            draining_after_stop = True

        prev_rec_state = current_rec

        if not current_rec and not recording_was_active:
            time.sleep(0.05)
            continue

        if not current_rec and recording_was_active and draining_after_stop:
            if in_q.empty():
                print(f"[SAVER][{role}] ... Queue drained - closing file.")
                try:
                    if f:
                        do_flush_sync(); f.close()
                except Exception:
                    pass
                f = None
                recording_was_active = False
                draining_after_stop = False
            continue

        # OdbiĂłr ramek
        try:
            role_in, ts_ns, frame = in_q.get(timeout=0.2)
        except queue.Empty:
            now = time.time()
            if f and (now - last_sync_ts) >= FSYNC_SECS and bytes_since_sync > 0:
                f.flush(); bytes_since_sync = 0; last_sync_ts = now
            continue

        if role_in != role:
            continue

        frames_seen_total += 1

        # Nowy plik po roll_every
        if f is None or frames_in_file >= roll_every:
            try:
                if f:
                    f.flush(); f.close()
            except Exception:
                pass
            f = open_new_file(role, ts_ns, file_idx, session_dir)
            file_idx += 1
            frames_in_file = 0

        if not frame.flags.c_contiguous:
            frame = np.ascontiguousarray(frame)

        h, w = frame.shape[:2]
        c = 1 if frame.ndim == 2 else frame.shape[2]
        mv = memoryview(frame)
        payload_nbytes = mv.nbytes

        f.write(HEADER_TAG)
        f.write((w).to_bytes(4, 'little'))
        f.write((h).to_bytes(4, 'little'))
        f.write((c).to_bytes(4, 'little'))
        f.write(int(ts_ns).to_bytes(8, 'little', signed=False))
        f.write((payload_nbytes).to_bytes(4, 'little'))
        f.write(mv)

        frames_in_file += 1
        saved_frames_total += 1
        bytes_since_sync += (HEADER_FIXED_BYTES + payload_nbytes)

        now = time.time()
        if (now - last_sync_ts) >= FSYNC_SECS:
            f.flush(); bytes_since_sync = 0; last_sync_ts = now

        if stats_q and (now - last_report) >= REPORT_EVERY:
            try:
                backlog = in_q.qsize()
            except Exception:
                backlog = -1
            expected = saved_frames_total + (backlog if backlog >= 0 else 0)
            elapsed = max(now - start_time, 1e-3)
            real_fps = saved_frames_total / elapsed
            dropped = max(0, frames_seen_total - saved_frames_total)
            try:
                stats_q.put_nowait({
                    "type": "save_status",
                    "role": role,
                    "saved": saved_frames_total,
                    "received": frames_seen_total,
                    "dropped": dropped,
                    "expected": expected,
                    "backlog": backlog,
                    "fps": real_fps,
                })
            except queue.Full:
                pass
            last_report = now

    if f:
        try:
            f.flush(); f.close()
        except Exception:
            pass

    print(f"[SAVER][{role}] ... exit, total saved frames: {saved_frames_total}")


def reconfigure_ring_buffer(seconds: int, width: int | None = None, height: int | None = None):
    global _GLOBAL_MULTI_BUFFER, _GLOBAL_MULTI_BUFFER_LOCK, _BUFFER_FRAMES, _BUFFER_META, _BUFFER_TRACE, _BUFFER_SET_COUNTER

    if not BUFFER_AVAILABLE:
        return False, "VisionRingBuffer niedostÄ™pny"

    seconds = max(1, int(seconds))

    try:
        with _GLOBAL_MULTI_BUFFER_LOCK:
            if _GLOBAL_MULTI_BUFFER is None:
                return False, "Bufor nie zostaĹ‚ jeszcze zainicjalizowany"

            width = int(width or _GLOBAL_MULTI_BUFFER.width)
            height = int(height or _GLOBAL_MULTI_BUFFER.height)
            fps = int(getattr(_GLOBAL_MULTI_BUFFER, "fps", 50))
            cameras = int(getattr(_GLOBAL_MULTI_BUFFER, "cameras", 4))

            _GLOBAL_MULTI_BUFFER = VisionRingBuffer(
                width=width,
                height=height,
                cameras=cameras,
                fps=fps,
                seconds=seconds,
            )
            _BUFFER_FRAMES.clear()
            _BUFFER_META.clear()
            cap = int(getattr(_GLOBAL_MULTI_BUFFER, "capacity", max(1, seconds * fps)))
            _BUFFER_TRACE = deque(maxlen=max(128, cap + 32))
            _BUFFER_SET_COUNTER = 0

        print(f"[BUFFER] đź” Reconfigured buffer to {seconds}s")
        return True, f"Buffer reconfigured to {seconds}s"
    except Exception as e:
        return False, f"Buffer reconfigure failed: {e}"

# 🎯 FUNKCJE BUFORA - Zapis bufora na ĹĽÄ…danie
def save_buffer_to_disk(output_dir="buffer_recordings", role=None):
    """
    Zapisuje bufor do dysku (wszystkie 4 kamery razem jak w demo)
    
    Args:
        output_dir: Folder docelowy
        role: Ignorowany (bufor jest wspĂłlny dla wszystkich kamer)
    
    Returns:
        tuple: (success: bool, message: str, files_saved: list)
    """
    global _GLOBAL_MULTI_BUFFER
    
    if not BUFFER_AVAILABLE:
        return False, "VisionRingBuffer niedostÄ™pny", []
    
    if _GLOBAL_MULTI_BUFFER is None:
        return False, "Bufor nie zostaĹ‚ jeszcze zainicjalizowany", []
    
    from pathlib import Path
    import time
    
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Sprawdź status bufora
        status = {
            'index': _GLOBAL_MULTI_BUFFER.index,
            'capacity': _GLOBAL_MULTI_BUFFER.capacity,
            'full': _GLOBAL_MULTI_BUFFER.full
        }
        
        if status['index'] == 0 and not status['full']:
            return False, "Bufor pusty - brak klatek do zapisu", []
        
        # Zapisz bufor asynchronicznie
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        buffer_output = output_path / f"multi_camera_buffer_{timestamp}"
        
        _GLOBAL_MULTI_BUFFER.dump_async(str(buffer_output), require_full=False)
        
        frames_count = status['capacity'] if status['full'] else status['index']
        total_frames = frames_count * 4  # 4 kamery
        
        message = f"Zapisano bufor: {frames_count} zestawĂłw klatek ({total_frames} klatek Ĺ‚Ä…cznie)"
        saved_files = [str(buffer_output)]
        
        print(f"[BUFFER] đź’ľ {message} â†’ {buffer_output}")
        
        return True, message, saved_files
            
    except Exception as e:
        return False, f"Błąd zapisu bufora: {e}", []


def get_buffer_status(role=None):
    """
    Zwraca status bufora (wspĂłlnego dla wszystkich kamer)
    
    Args:
        role: Ignorowany (bufor jest wspĂłlny)
    
    Returns:
        dict: Status bufora
    """
    global _GLOBAL_MULTI_BUFFER
    
    if not BUFFER_AVAILABLE:
        return {"available": False, "error": "VisionRingBuffer niedostÄ™pny"}
    
    try:
        if _GLOBAL_MULTI_BUFFER is None:
            return {
                "available": True, 
                "initialized": False,
                "error": "Bufor nie zostaĹ‚ jeszcze zainicjalizowany (czekam na pierwsze klatki)"
            }
        
        return {
            "available": True,
            "initialized": True,
            "index": _GLOBAL_MULTI_BUFFER.index,
            "capacity": _GLOBAL_MULTI_BUFFER.capacity,
            "full": _GLOBAL_MULTI_BUFFER.full,
            "frames_count": _GLOBAL_MULTI_BUFFER.capacity if _GLOBAL_MULTI_BUFFER.full else _GLOBAL_MULTI_BUFFER.index,
            "cameras": 4,
            "total_frames": (_GLOBAL_MULTI_BUFFER.capacity if _GLOBAL_MULTI_BUFFER.full else _GLOBAL_MULTI_BUFFER.index) * 4
        }
            
    except Exception as e:
        return {"available": True, "error": f"Błąd statusu: {e}"}


def get_buffer_trace_snapshot(max_sets: int | None = None):
    """
    Zwraca historiÄ™ metryk per-zestaw 4 klatek z bufora.
    KaĹĽdy element odpowiada jednemu push() do VisionRingBuffer.
    """
    global _BUFFER_TRACE, _GLOBAL_MULTI_BUFFER
    try:
        trace = list(_BUFFER_TRACE)
        if max_sets is not None:
            n = max(0, int(max_sets))
            if n > 0:
                trace = trace[-n:]
        frames_count = None
        capacity = None
        full = None
        try:
            if _GLOBAL_MULTI_BUFFER is not None:
                capacity = int(_GLOBAL_MULTI_BUFFER.capacity)
                full = bool(_GLOBAL_MULTI_BUFFER.full)
                idx = int(_GLOBAL_MULTI_BUFFER.index)
                frames_count = capacity if full else idx
        except Exception:
            pass
        return {
            "ok": True,
            "frames_count": frames_count,
            "capacity": capacity,
            "full": full,
            "sets": trace,
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "sets": []}


def cleanup_buffers():
    """
    CzyĹ›ci bufor (wywoĹ‚aj przy zamkniÄ™ciu aplikacji)
    """
    global _GLOBAL_MULTI_BUFFER, _BUFFER_FRAMES, _BUFFER_META, _BUFFER_TRACE, _BUFFER_SET_COUNTER
    
    if not BUFFER_AVAILABLE:
        return
    
    try:
        if _GLOBAL_MULTI_BUFFER is not None:
            try:
                _GLOBAL_MULTI_BUFFER.stop()
                print("[BUFFER]  Zatrzymano globalny bufor")
            except Exception as e:
                print(f"[BUFFER] âš ď¸Ź Błąd zatrzymywania bufora: {e}")
        
        _GLOBAL_MULTI_BUFFER = None
        _BUFFER_FRAMES.clear()
        _BUFFER_META.clear()
        _BUFFER_TRACE.clear()
        _BUFFER_SET_COUNTER = 0
        print("[BUFFER] 🧹 Bufor wyczyszczony")
        
    except Exception as e:
        print(f"[BUFFER] âš ď¸Ź Błąd czyszczenia bufora: {e}")

