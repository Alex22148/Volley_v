# -*- coding: utf-8 -*-
# main.py
from pathlib import Path
from core.log_setup import setup_logging
import json
import os
import sys
from collections import deque
from storage.session_archive import SessionArchiveManager
from datetime import datetime
# ===========================================================
# App dir + logging
# ===========================================================

def _app_dir():
    return os.path.dirname(sys.executable) if getattr(sys, "frozen", False) \
           else os.path.dirname(os.path.abspath(__file__))

def _truthy(x: str | None) -> bool:
    return str(x or "").strip().lower() in {"1", "true", "yes", "y", "on"}

APP_DIR = _app_dir()
LOG_PATH = setup_logging(Path(APP_DIR) / "logs", level=20)
if _truthy(os.environ.get("VOLLEYHUB_CONSOLE")):
    print(f"[LOG] zapis do: {LOG_PATH}")

# --------- stdout/stderr w trybie frozen ----------

if getattr(sys, "frozen", False):
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", buffering=1, encoding="utf-8", errors="replace")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", buffering=1, encoding="utf-8", errors="replace")
else:
    try:
        stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
        if callable(stdout_reconfigure):
            stdout_reconfigure(encoding="utf-8", errors="replace")
        stderr_reconfigure = getattr(sys.stderr, "reconfigure", None)
        if callable(stderr_reconfigure):
            stderr_reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Pylon DLLs on Windows (tylko gdy folder istnieje)
if os.name == "nt" and hasattr(os, "add_dll_directory"):
    _pylon_dir = r"C:\Program Files\Basler\pylon 8\Runtime\x64"
    if os.path.isdir(_pylon_dir):
        try:
            os.add_dll_directory(_pylon_dir)
        except Exception:
            pass

# Run from app dir (not MEIPASS)
try:
    os.chdir(APP_DIR)
except Exception:
    pass

# Global env hints
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# (opcjonalnie) wyłącz OpenCL i zredukuj wątki OpenCV
try:
    import cv2
    cv2.setNumThreads(1)

    try:
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass
except Exception:
    pass

# ===========================================================
# Imports that don't spawn processes yet
# ===========================================================
import time  # noqa: E402
import threading  # noqa: E402
import queue  # noqa: E402
import multiprocessing as mp  # noqa: E402

from capture.basler_lib import (  # noqa: E402
    cam_config,
    load_roles,
    wait_for_all_ptp_ready,
    configure_periodic_signal_trigger)

from ui.assign_roles_gui import ask_camera_roles_gui  # noqa: E402
import core.utils_config as utils_config  # noqa: E402

_YOLO_STEP_TRACE_LOCK = threading.Lock()
_YOLO_STEP_TRACE = deque(maxlen=12000)
_CONTROL_WORKER_PRINTED_IDS = False


def push_yolo_step_trace(record: dict):
    try:
        with _YOLO_STEP_TRACE_LOCK:
            _YOLO_STEP_TRACE.append(record)
    except Exception:
        pass


def get_yolo_step_trace_snapshot(max_steps: int | None = None):
    try:
        with _YOLO_STEP_TRACE_LOCK:
            arr = list(_YOLO_STEP_TRACE)
        if max_steps is not None:
            n = max(0, int(max_steps))
            if n > 0:
                arr = arr[-n:]
        return {"ok": True, "steps": arr}
    except Exception as e:
        return {"ok": False, "error": str(e), "steps": []}

# ===========================================================
# Helpers
# ===========================================================

def clear_queue(q):
    cleared = 0
    try:
        while True:
            _ = q.get_nowait()
            if hasattr(q, "task_done"):
                q.task_done()
            cleared += 1
    except Exception:
        pass
    if cleared:
        print(f"[SYNC] Wyczyściłem {cleared} starych ramek po kalibracji.")

def calibrate_ptp_offsets(cams, roles, stats_q=None):
    """
    Przy PTP + Scheduled Action Commands nie korygujemy timestampów software'owo.
    Ten krok zostaje tylko jako znacznik diagnostyczny.
    """
    print("[SYNC] PTP + Scheduled Action Commands aktywne brak software'owej korekcji OFFSETS.")
    if stats_q:
        try:
            stats_q.put("PTP + Scheduled Action Commands aktywne brak korekcji OFFSETS")
        except Exception:
            pass

def control_worker(control_q, stop_evt, cams, roles, streaming_enabled, stats_q,
                   recording_event, snapshot_event, zoom_role_var,shared_state,archive_mgr=None, yolo_batch_q=None, yolo_raw_q=None):
    """Minimal, latency-friendly control loop. Accepts (cmd, val) tuples."""
    from core.config_menager import save_config
    global _CONTROL_WORKER_PRINTED_IDS

    while not stop_evt.is_set():
        if not _CONTROL_WORKER_PRINTED_IDS:
            print(f"[DEBUG] [control_worker] recording_event id={id(recording_event)} stop_evt id={id(stop_evt)}")
            _CONTROL_WORKER_PRINTED_IDS = True

        try:
            msg = control_q.get(timeout=0.1)
        except queue.Empty:
            continue
        except Exception:
            continue

        if not isinstance(msg, (tuple, list)) or len(msg) < 2:
            print(f"[CTRL] ⚠️ Zly format wiadomości: {msg}")
            continue

        cmd, val = msg[0], msg[1]

        if cmd == "stream":
            on = bool(val)
            for r in streaming_enabled:
                (streaming_enabled[r].set() if on else streaming_enabled[r].clear())
            print(f"[CTRL] Stream {'ON' if on else 'OFF'} dla wszystkich ról")
            stats_q.put(f"Stream {'ON' if on else 'OFF'} dla wszystkich ról")

        elif cmd == "yolo_set_batch_size":
            try:
                batch_size = max(1, int(val))
                shared_state["yolo_batch_size"] = batch_size

                from vision.yolo_module import set_yolo_batch_size
                set_yolo_batch_size(batch_size)

                stats_q.put(f" YOLO infer batch size: {batch_size}")
            except Exception as e:
                print(f"[CTRL] ⚠️ YOLO batch size set failed: {e}")
                stats_q.put(f"⚠️ YOLO batch size set failed: {e}")


        elif cmd == "yolo_set_input_batch_images":
            try:
                batch_images = max(1, int(val))
                num_roles = len(roles) if roles else 4
                seq_len = max(1, batch_images // max(1, num_roles))

                shared_state["yolo_input_batch_images"] = batch_images
                shared_state["yolo_seq_len"] = seq_len
                try:
                    shared_state["yolo_generation"] = int(shared_state.get("yolo_generation", 0)) + 1
                except Exception:
                    pass

                stats_q.put(f" YOLO input batch images: {batch_images} -> seq_len={seq_len}")
            except Exception as e:
                stats_q.put(f"⚠️ YOLO input batch images set failed: {e}")

        elif cmd == "yolo_set_seq_len":
            try:
                seq_len = max(1, int(val))
                shared_state["yolo_seq_len"] = seq_len
                try:
                    shared_state["yolo_generation"] = int(shared_state.get("yolo_generation", 0)) + 1
                except Exception:
                    pass

                stats_q.put(f" YOLO seq_len: {seq_len}")
            except Exception as e:
                stats_q.put(f"⚠️ YOLO seq_len set failed: {e}")

        elif cmd == "set_zoom_role":
            try:
                new_role = (val or "").strip() if isinstance(val, str) else ""
                zoom_role_var.value = new_role
                try:
                    shared_state["zoom_role"] = new_role
                except Exception:
                    pass
                stats_q.put("Zmieniono źródło zoomu na: {new_role or '—(OFF)—'}")
            except Exception as e:
                print(f"[CTRL] ⚠️ Nie udało się zmienić źródła zoomu: {e}")

        elif cmd == "record":
            try:
                rec_on = bool(val)

                if rec_on:
                    recording_event.set()

                    session_dir = getattr(utils_config, "RAW_DIR", None)
                    if not session_dir:
                        raise RuntimeError("RAW_DIR/session_dir nieustawione przed START")

                    config = {
                        "session_started_at": datetime.now().isoformat(timespec="milliseconds"),
                        "roles": list(roles),
                        "serial_by_role": dict(shared_state.get("serial_by_role", {})) if shared_state else {},
                        "model_path": shared_state.get("yolo_model_path") if shared_state else None,
                        "model_variant": shared_state.get("yolo_model_variant") if shared_state else None,
                        "image_size": int(shared_state.get("yolo_image_size", 640)) if shared_state else 640,
                        "batch_size": int(shared_state.get("yolo_batch_size", 8)) if shared_state else 8,
                        "confidence": float(shared_state.get("yolo_confidence", 0.5)) if shared_state else 0.5,
                        "device": shared_state.get("yolo_device", "cuda") if shared_state else "cuda",
                        "backend": shared_state.get("yolo_backend", "ultralytics") if shared_state else "ultralytics",
                        "trt_engine_path": shared_state.get("yolo_trt_engine_path") if shared_state else None,
                        "trt_dynamic": bool(shared_state.get("yolo_trt_dynamic", False)) if shared_state else False,
                        "trt_workspace_gb": float(shared_state.get("yolo_trt_workspace_gb", 2.0)) if shared_state else 2.0,
                        "buffer_dt_s": int(shared_state.get("buffer_dt_s", 5)) if shared_state else 5,
                    }

                    if archive_mgr is not None:
                        archive_mgr.start_session(session_dir=session_dir, config=config)
                        try:
                            shared_state["analysis_session_dir"] = str(session_dir)
                        except Exception:
                            pass

                    print(f"[CTRL] Record ON | session={session_dir}")

                else:
                    recording_event.clear()
                    print("[CTRL] Record OFF")

                    session_dir = None
                    try:
                        session_dir = shared_state.get("analysis_session_dir") if shared_state else None
                    except Exception:
                        session_dir = None

                    buffer_ok = None
                    buffer_msg = None
                    if session_dir:
                        try:
                            from capture.grabber_module import save_buffer_to_disk
                            buffer_ok, buffer_msg, _ = save_buffer_to_disk(str(Path(session_dir) / "buffer"))
                            print(f"[CTRL] auto-save buffer: ok={buffer_ok} msg={buffer_msg}")
                        except Exception as e:
                            buffer_ok = False
                            buffer_msg = str(e)
                            print(f"[CTRL]  auto-save buffer failed: {e}")

                    if archive_mgr is not None:
                        archive_mgr.stop_session(
                            reason="record_stop",
                            extra_summary={
                                "buffer_saved": buffer_ok,
                                "buffer_message": buffer_msg,
                            },
                        )

                    try:
                        if shared_state is not None:
                            shared_state["analysis_session_dir"] = None
                    except Exception:
                        pass

            except Exception as e:
                print(f"[CTRL]  record handling failed: {e}")

        elif cmd in ("exposure", "set_exposure"):
            try:
                role, v = val
                serial_for_role = None
                try:
                    serial_for_role = shared_state.get("serial_by_role", {}).get(role)
                except Exception:
                    pass
                for cam, r in zip(cams, roles):
                    if r == role:
                        cam.ExposureTime.SetValue(v)
                        print(f' expo = {v}')
                try:
                    payload = {"per_role": {role: {"exposure": float(v)}}}
                    if serial_for_role:
                        payload["per_camera"] = {serial_for_role: {"exposure": float(v)}}
                    save_config(payload)
                except Exception as e:
                    print(f"[CTRL] ⚠️ save_config exposure failed: {e}")
            except Exception as e:
                print(f"[CTRL] ⚠️ Exposure set failed: {e}")

        elif cmd in ("gain", "set_gain"):
            try:
                role, v = val
                serial_for_role = None
                try:
                    serial_for_role = shared_state.get("serial_by_role", {}).get(role)
                except Exception:
                    pass
                for cam, r in zip(cams, roles):
                    if r == role:
                        cam.Gain.SetValue(v)
                        print(f' gain = {v}')
                try:
                    payload = {"per_role": {role: {"gain": float(v)}}}
                    if serial_for_role:
                        payload["per_camera"] = {serial_for_role: {"gain": float(v)}}
                    save_config(payload)
                except Exception as e:
                    print(f"[CTRL] ⚠️ save_config gain failed: {e}")
            except Exception as e:
                print(f"[CTRL] ⚠️ Gain set failed: {e}")

        elif cmd in ("save_path", "set_path2save"):
            try:
                new_path = str(val)
                utils_config.RAW_DIR = new_path
                save_config({"path2save": new_path})
                print(f"[CTRL] Zmieniono folder zapisu na: {new_path}")
            except Exception as e:
                print(f"[CTRL] ⚠️ Nie udało się zapisać path2save: {e}")

        elif cmd == "camera_set_fps":
            try:
                from pypylon import pylon
                fps = max(1.0, float(val))
                period_us = 1_000_000.0 / fps
                if shared_state is not None:
                    shared_state["camera_target_fps"] = float(fps)
                    shared_state["camera_period_us"] = float(period_us)

                for cam, role in zip(cams, roles):
                    was_grabbing = False
                    try:
                        was_grabbing = bool(cam.IsGrabbing())
                    except Exception:
                        was_grabbing = False
                    try:
                        if was_grabbing:
                            cam.StopGrabbing()
                    except Exception:
                        pass
                    configure_periodic_signal_trigger(cam, role, period_us=period_us)
                    try:
                        if was_grabbing:
                            cam.StartGrabbing(pylon.GrabStrategy_OneByOne)
                    except Exception as e:
                        print(f"[CTRL] ⚠️ camera_set_fps restart failed for {role}: {e}")

                msg_fps = f"Camera FPS set: {fps:.2f} -> periodTime={period_us:.1f} us"
                print(f"[CTRL] {msg_fps}")
                try:
                    stats_q.put(msg_fps)
                except Exception:
                    pass
            except Exception as e:
                print(f"[CTRL] camera_set_fps failed: {e}")
                try:
                    stats_q.put(f"camera_set_fps failed: {e}")
                except Exception:
                    pass

        elif cmd == "buffer_set_seconds":
            try:
                seconds = max(1, int(val))
                shared_state["buffer_dt_s"] = seconds
                reconfigured = False
                reconfig_msg = ""
                status_after = {}
                try:
                    from capture.grabber_module import reconfigure_ring_buffer, get_buffer_status

                    width = None
                    height = None
                    try:
                        if cams:
                            width = int(cams[0].Width.GetValue())
                            height = int(cams[0].Height.GetValue())
                    except Exception:
                        width = None
                        height = None

                    ok, msg = reconfigure_ring_buffer(seconds, width=width, height=height)
                    reconfigured = bool(ok)
                    reconfig_msg = str(msg)
                    try:
                        status_after = get_buffer_status()
                    except Exception:
                        status_after = {}
                except Exception as e:
                    reconfig_msg = f"reconfigure exception: {e}"

                if reconfigured:
                    cap = status_after.get("capacity")
                    idx = status_after.get("index")
                    full = status_after.get("full")
                    msg = f"Buffer seconds={seconds} | reconfigured | capacity={cap} index={idx} full={full}"
                    print(f"[CTRL] {msg}")
                    stats_q.put(msg)
                else:
                    msg = f"Buffer seconds={seconds} | pending reconfigure ({reconfig_msg})"
                    print(f"[CTRL] {msg}")
                    stats_q.put(msg)
            except Exception as e:
                print(f"[CTRL] buffer_set_seconds failed: {e}")
                stats_q.put(f"buffer_set_seconds failed: {e}")

        elif cmd == "snapshot":
            if not snapshot_event.is_set():
                snapshot_event.set()
                print("[CTRL] Snapshot triggered")

        elif cmd == "save_buffer":
            # 🎬 NOWA KOMENDA - Zapisz bufor
            import json
            import time
            try:
                from capture.grabber_module import save_buffer_to_disk, get_buffer_status, get_buffer_trace_snapshot
                
                # Sprawdź status buforów
                status = get_buffer_status()
                if not status.get("available", False):
                    stats_q.put("bufor niedostępny")
                    print("[CTRL] save_buffer: bufor niedostępny")
                    continue
                
                if "error" in status:
                    stats_q.put(f"Błąd bufora: {status['error']}")
                    print(f"[CTRL]  save_buffer error: {status['error']}")
                    continue
                
                # Zapisz bufor
                save_t0 = time.time()
                success, message, files = save_buffer_to_disk("buffer_recordings")
                save_t1 = time.time()
                buffer_stats = {}
                try:
                    buffer_stats = dict(shared_state.get("yolo_buffer_stats", {})) if shared_state else {}
                except Exception:
                    buffer_stats = {}

                buffer_status = {}
                buffer_trace = {}
                yolo_step_trace = {}
                try:
                    from capture.grabber_module import get_buffer_status
                    buffer_status = get_buffer_status()
                except Exception as e:
                    buffer_status = {"error": str(e)}
                try:
                    frames_count = int(buffer_status.get("frames_count", 0) or 0)
                    buffer_trace = get_buffer_trace_snapshot(max_sets=frames_count if frames_count > 0 else None)
                    yolo_step_trace = get_yolo_step_trace_snapshot(max_steps=frames_count if frames_count > 0 else None)
                except Exception as e:
                    buffer_trace = {"ok": False, "error": str(e), "sets": []}
                    yolo_step_trace = {"ok": False, "error": str(e), "steps": []}

                if success and files:
                    try:
                        buffer_dir = Path(files[0])
                        click_id = int(time.time() * 1000)
                        stats_path = buffer_dir.with_name(buffer_dir.name + f"_stats_{click_id}.json")

                        yolo_router_stats = {}
                        yolo_last_batch_ts = {}
                        yolo_last_inference_snapshot = {}
                        fixed_params = {}
                        camera_params_current = {}
                        try:
                            if shared_state is not None:
                                yolo_router_stats = dict(shared_state.get("yolo_router_stats_last", {}))
                                yolo_last_batch_ts = dict(shared_state.get("yolo_last_batch_timestamps", {}))
                                yolo_last_inference_snapshot = dict(shared_state.get("yolo_last_inference_snapshot", {}))
                                fixed_params = {
                                    "roles": list(shared_state.get("roles", [])),
                                    "serial_by_role": dict(shared_state.get("serial_by_role", {})),
                                    "camera_target_fps": float(shared_state.get("camera_target_fps", 0.0) or 0.0),
                                    "camera_period_us": float(shared_state.get("camera_period_us", 0.0) or 0.0),
                                    "yolo_model_path": shared_state.get("yolo_model_path"),
                                    "yolo_backend": shared_state.get("yolo_backend"),
                                    "yolo_device": shared_state.get("yolo_device"),
                                    "yolo_preprocess_backend": shared_state.get("yolo_preprocess_backend"),
                                    "yolo_image_size": int(shared_state.get("yolo_image_size", 0) or 0),
                                    "yolo_confidence": float(shared_state.get("yolo_confidence", 0.0) or 0.0),
                                    "yolo_input_batch_images": int(shared_state.get("yolo_input_batch_images", 0) or 0),
                                    "yolo_seq_len": int(shared_state.get("yolo_seq_len", 0) or 0),
                                    "yolo_batch_size": int(shared_state.get("yolo_batch_size", 0) or 0),
                                    "yolo_bayer_pattern": shared_state.get("yolo_bayer_pattern"),
                                    "buffer_dt_s": int(shared_state.get("buffer_dt_s", 0) or 0),
                                }
                        except Exception:
                            pass

                        try:
                            from pypylon import pylon
                            for cam, role_name in zip(cams, roles):
                                role_key = str(role_name)
                                info = {}
                                try:
                                    info["is_grabbing"] = bool(cam.IsGrabbing())
                                except Exception:
                                    info["is_grabbing"] = None
                                try:
                                    info["width"] = int(cam.Width.GetValue())
                                    info["height"] = int(cam.Height.GetValue())
                                except Exception:
                                    pass
                                try:
                                    info["exposure_us"] = float(cam.ExposureTime.GetValue())
                                except Exception:
                                    pass
                                try:
                                    info["gain_db"] = float(cam.Gain.GetValue())
                                except Exception:
                                    pass
                                try:
                                    info["pixel_format"] = str(cam.PixelFormat.GetValue())
                                except Exception:
                                    pass
                                try:
                                    info["acquisition_fps"] = float(cam.ResultingFrameRateAbs.GetValue())
                                except Exception:
                                    pass
                                camera_params_current[role_key] = info
                        except Exception:
                            pass

                        target_fps = float(fixed_params.get("camera_target_fps", 0.0) or 0.0)
                        target_step_period_ms = (1000.0 / target_fps) if target_fps > 0 else 0.0
                        infer_ms = float((yolo_last_batch_ts.get("infer_ms", 0.0) or 0.0))
                        prepare_ms = float((yolo_last_batch_ts.get("prepare_ms", 0.0) or 0.0))
                        forward_ms = float((yolo_last_batch_ts.get("forward_ms", 0.0) or 0.0))
                        post_ms = float((yolo_last_batch_ts.get("post_ms", 0.0) or 0.0))
                        queue_wait_ms = float((yolo_last_batch_ts.get("queue_wait_ms", 0.0) or 0.0))
                        batch_collect_ms = float((yolo_last_batch_ts.get("batch_collect_ms", 0.0) or 0.0))
                        e2e_pipeline_ms = float(batch_collect_ms + queue_wait_ms + infer_ms)
                        save_duration_ms = float((save_t1 - save_t0) * 1000.0)
                        max_overhead_for_live_ms = max(0.0, target_step_period_ms - e2e_pipeline_ms)
                        live_ok_now = bool(e2e_pipeline_ms <= target_step_period_ms) if target_step_period_ms > 0 else None

                        step_trace_steps = []
                        if isinstance(yolo_step_trace, dict):
                            steps_arr = yolo_step_trace.get("steps", [])
                            if isinstance(steps_arr, list):
                                step_trace_steps = steps_arr

                        payload = {
                            "schema": "buffer_event_report_v3",
                            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "event": {
                                "message": message,
                                "save_duration_ms": save_duration_ms,
                                "paths": {
                                    "buffer_output": str(buffer_dir),
                                    "stats_json": str(stats_path),
                                },
                            },
                            "camera": {
                                "target_fps": target_fps,
                                "target_period_us": float(fixed_params.get("camera_period_us", 0.0) or 0.0),
                                "roles": fixed_params.get("roles", []),
                                "serial_by_role": fixed_params.get("serial_by_role", {}),
                                "params_current": camera_params_current,
                            },
                            "buffer": {
                                "seconds": int(fixed_params.get("buffer_dt_s", 0) or 0),
                                "trace_enabled": bool(shared_state.get("buffer_trace_enabled", False)) if shared_state is not None else False,
                                "status": buffer_status,
                                "frame_trace": buffer_trace,  # jedyne źródło timestampów wszystkich zapisanych klatek
                            },
                            "pipeline": {
                                "router_stats_last": yolo_router_stats,
                                "latest_batch_timestamps": yolo_last_batch_ts,
                                "live_budget": {
                                    "target_step_period_ms": target_step_period_ms,
                                    "current_e2e_pipeline_ms": e2e_pipeline_ms,
                                    "max_extra_overhead_ms_for_live": max_overhead_for_live_ms,
                                    "live_ok_now": live_ok_now,
                                    "components_ms": {
                                        "batch_collect_ms": batch_collect_ms,
                                        "queue_wait_ms": queue_wait_ms,
                                        "prepare_ms": prepare_ms,
                                        "forward_ms": forward_ms,
                                        "post_ms": post_ms,
                                        "infer_ms": infer_ms,
                                    },
                                },
                            },
                            "inference": {
                                "model": {
                                    "path": fixed_params.get("yolo_model_path"),
                                    "backend": fixed_params.get("yolo_backend"),
                                    "device": fixed_params.get("yolo_device"),
                                    "preprocess_backend": fixed_params.get("yolo_preprocess_backend"),
                                    "image_size": int(fixed_params.get("yolo_image_size", 0) or 0),
                                    "confidence": float(fixed_params.get("yolo_confidence", 0.0) or 0.0),
                                    "input_batch_images": int(fixed_params.get("yolo_input_batch_images", 0) or 0),
                                    "seq_len": int(fixed_params.get("yolo_seq_len", 0) or 0),
                                    "batch_size": int(fixed_params.get("yolo_batch_size", 0) or 0),
                                    "bayer_pattern": fixed_params.get("yolo_bayer_pattern"),
                                },
                                "stats_last": buffer_stats,
                                "step_trace": {
                                    "ok": bool(yolo_step_trace.get("ok", False)) if isinstance(yolo_step_trace, dict) else False,
                                    "steps_count": len(step_trace_steps),
                                    "steps": step_trace_steps,  # per-step timestamps + detections_flat + timings_ms
                                },
                                "last_snapshot": yolo_last_inference_snapshot,
                            },
                        }

                        with open(stats_path, "w", encoding="utf-8") as f:
                            json.dump(payload, f, ensure_ascii=False, indent=2)

                        print(f"[CTRL] zapisano statystyki bufora: {stats_path}")
                    except Exception as e:
                        print(f"[CTRL] ⚠️ nie udało się zapisać stats bufora: {e}")
                if success:
                    stats_q.put(f"{message}")
                    print(f"[CTRL] save_buffer: {message}")
                    for f in files:
                        print(f"[CTRL] Zapisano: {f}")
                else:
                    stats_q.put(f"{message}")
                    print(f"[CTRL]  save_buffer failed: {message}")
                    
            except Exception as e:
                error_msg = f"Błąd zapisu bufora: {e}"
                stats_q.put(f"{error_msg}")
                print(f"[CTRL] save_buffer exception: {e}")

        elif cmd == "yolo_load_model":
            # Load custom YOLO model
            try:
                model_path = str(val)
                from vision.yolo_module import load_custom_yolo_model
                
                success = load_custom_yolo_model(model_path)
                if success:
                    print(f"[CTRL]  Custom YOLO model loaded: {model_path}")
                    stats_q.put(f" Custom YOLO model loaded: {model_path}")
                    shared_state["yolo_model_path"] = model_path
                else:
                    print(f"[CTRL]  Failed to load YOLO model: {model_path}")
                    stats_q.put(f" Failed to load YOLO model: {model_path}")
                    
            except Exception as e:
                print(f"[CTRL] ⚠️ YOLO model load failed: {e}")
                stats_q.put(f"⚠️ YOLO model load failed: {e}")

        elif cmd == "yolo_set_class_id":
            # Set YOLO ball class ID
            try:
                class_id = int(val)
                from vision.yolo_module import set_yolo_ball_class_id
                
                set_yolo_ball_class_id(class_id)
                print(f"[CTRL]  YOLO ball class ID set to: {class_id}")
                stats_q.put(f" YOLO ball class ID: {class_id}")
                shared_state["yolo_ball_class_id"] = class_id
                
            except Exception as e:
                print(f"[CTRL] ⚠️ YOLO class ID set failed: {e}")
                stats_q.put(f"⚠️ YOLO class ID set failed: {e}")

        elif cmd == "yolo_set_confidence":
            # Set YOLO confidence threshold
            try:
                confidence = float(val)
                from vision.yolo_module import set_yolo_confidence
                
                set_yolo_confidence(confidence)
                print(f"[CTRL]  YOLO confidence set to: {confidence}")
                stats_q.put(f" YOLO confidence: {confidence}")
                shared_state["yolo_confidence"] = confidence
                
            except Exception as e:
                print(f"[CTRL] ⚠️ YOLO confidence set failed: {e}")
                stats_q.put(f"⚠️ YOLO confidence set failed: {e}")
        
        elif cmd == "yolo_set_image_size":
            # Set YOLO image size
            try:
                size = int(val)
                from vision.yolo_module import set_yolo_image_size
                
                set_yolo_image_size(size)
                print(f"[CTRL] YOLO image size set to: {size}")
                stats_q.put(f"YOLO image size: {size}")
                shared_state["yolo_image_size"] = size
                
            except Exception as e:
                print(f"[CTRL] ⚠️ YOLO image size set failed: {e}")
                stats_q.put(f"⚠️ YOLO image size set failed: {e}")
        
        elif cmd == "yolo_set_device":
            # Set YOLO device (cpu/cuda)
            try:
                device = str(val)
                from vision.yolo_module import set_yolo_device
                
                if set_yolo_device(device):
                    print(f"[CTRL]  YOLO device set to: {device}")
                    stats_q.put(f" YOLO device: {device}")
                    shared_state["yolo_device"] = device
                else:
                    print(f"[CTRL]  Failed to set YOLO device to: {device}")
                    stats_q.put(f" Failed to set YOLO device: {device}")
                
            except Exception as e:
                print(f"[CTRL] ⚠️ YOLO device set failed: {e}")
                stats_q.put(f"⚠️ YOLO device set failed: {e}")

        elif cmd == "yolo_set_preprocess_backend":
            # Set YOLO preprocess backend (cpu/cuda) for demosaic stage.
            try:
                backend = str(val).strip().lower()
                if backend not in ("cpu", "cuda"):
                    backend = "cpu"
                from vision.yolo_module import set_yolo_preprocess_backend

                if set_yolo_preprocess_backend(backend):
                    shared_state["yolo_preprocess_backend"] = backend
                    print(f"[CTRL] YOLO preprocess backend set to: {backend}")
                    stats_q.put(f"YOLO preprocess backend: {backend}")
                else:
                    if backend == "cuda" and set_yolo_preprocess_backend("cpu"):
                        shared_state["yolo_preprocess_backend"] = "cpu"
                        print("[CTRL] YOLO preprocess backend fallback: cuda -> cpu")
                        stats_q.put("YOLO preprocess backend fallback: cuda -> cpu")
                    else:
                        print(f"[CTRL] Failed to set YOLO preprocess backend: {backend}")
                        stats_q.put(f"Failed to set YOLO preprocess backend: {backend}")
            except Exception as e:
                print(f"[CTRL] YOLO preprocess backend set failed: {e}")
                stats_q.put(f"YOLO preprocess backend set failed: {e}")



        elif cmd == "yolo_set_backend":
            try:
                backend = str(val).strip().lower()
                from vision.yolo_module import set_yolo_backend

                if set_yolo_backend(backend):
                    shared_state["yolo_backend"] = backend
                    print(f"[CTRL] YOLO backend set to: {backend}")
                    stats_q.put(f"YOLO backend: {backend}")
                else:
                    print(f"[CTRL] Failed to set YOLO backend: {backend}")
                    stats_q.put(f"Failed to set YOLO backend: {backend}")
            except Exception as e:
                print(f"[CTRL] YOLO backend set failed: {e}")
                stats_q.put(f"YOLO backend set failed: {e}")

        elif cmd == "yolo_set_tensorrt_engine":
            try:
                engine_path = str(val or "").strip()
                from vision.yolo_module import set_yolo_tensorrt_engine

                if set_yolo_tensorrt_engine(engine_path):
                    shared_state["yolo_trt_engine_path"] = engine_path or None
                    print(f"[CTRL] YOLO TensorRT engine path: {engine_path or '(cleared)'}")
                    stats_q.put(f"YOLO TensorRT engine path: {engine_path or '(cleared)'}")
                else:
                    print(f"[CTRL] Failed to set TensorRT engine path: {engine_path}")
                    stats_q.put(f"Failed to set TensorRT engine path: {engine_path}")
            except Exception as e:
                print(f"[CTRL] TensorRT engine path set failed: {e}")
                stats_q.put(f"TensorRT engine path set failed: {e}")

        elif cmd == "yolo_set_trt_dynamic":
            try:
                dynamic = bool(val)
                from vision.yolo_module import configure_yolo_tensorrt

                configure_yolo_tensorrt(dynamic=dynamic, workspace_gb=None)
                shared_state["yolo_trt_dynamic"] = dynamic
                print(f"[CTRL] YOLO TensorRT dynamic shapes: {dynamic}")
                stats_q.put(f"YOLO TensorRT dynamic shapes: {dynamic}")
            except Exception as e:
                print(f"[CTRL] YOLO TensorRT dynamic set failed: {e}")
                stats_q.put(f"YOLO TensorRT dynamic set failed: {e}")

        elif cmd == "yolo_set_trt_workspace_gb":
            try:
                workspace_gb = max(0.5, float(val))
                from vision.yolo_module import configure_yolo_tensorrt

                configure_yolo_tensorrt(dynamic=None, workspace_gb=workspace_gb)
                shared_state["yolo_trt_workspace_gb"] = workspace_gb
                print(f"[CTRL] YOLO TensorRT workspace_gb: {workspace_gb}")
                stats_q.put(f"YOLO TensorRT workspace_gb: {workspace_gb}")
            except Exception as e:
                print(f"[CTRL] YOLO TensorRT workspace set failed: {e}")
                stats_q.put(f"YOLO TensorRT workspace set failed: {e}")

        elif cmd == "yolo_reset_queues":
            cleared_batch = 0
            cleared_raw = 0
            try:
                if yolo_batch_q is not None:
                    while True:
                        _ = yolo_batch_q.get_nowait()
                        cleared_batch += 1
            except queue.Empty:
                pass
            except Exception as e:
                print(f"[CTRL] ⚠️ yolo_reset_queues batch clear failed: {e}")

            try:
                if yolo_raw_q is not None:
                    while True:
                        _ = yolo_raw_q.get_nowait()
                        cleared_raw += 1
            except queue.Empty:
                pass
            except Exception as e:
                print(f"[CTRL] ⚠️ yolo_reset_queues raw clear failed: {e}")

            try:
                if shared_state is not None:
                    shared_state["yolo_generation"] = int(shared_state.get("yolo_generation", 0)) + 1
            except Exception:
                pass

            msg_reset = f" YOLO queue reset | batch={cleared_batch} raw={cleared_raw}"
            print(f"[CTRL] {msg_reset}")
            try:
                stats_q.put(msg_reset)
            except Exception:
                pass

        elif cmd == "yolo_set_bayer_pattern":
            # Bayer pattern is fixed to BG for this project.
            try:
                pattern = "BG"
                from vision.yolo_module import set_yolo_bayer_pattern

                if set_yolo_bayer_pattern(pattern):
                    print("[CTRL] YOLO Bayer pattern fixed to: BG")
                    stats_q.put("YOLO Bayer pattern fixed: BG")
                    shared_state["yolo_bayer_pattern"] = pattern
                else:
                    print("[CTRL] Failed to set fixed YOLO Bayer pattern: BG")
                    stats_q.put("Failed fixed YOLO Bayer pattern: BG")
            except Exception as e:
                print(f"[CTRL] YOLO Bayer pattern set failed: {e}")
                stats_q.put(f"YOLO Bayer pattern set failed: {e}")

        elif cmd == "demo_set_context":
            try:
                payload = dict(val or {})
                if "test_name" in payload:
                    shared_state["demo_test_name"] = payload.get("test_name")
                if "operator_name" in payload:
                    shared_state["demo_operator_name"] = payload.get("operator_name")
                print(f"[CTRL] 🧪 demo context updated: {payload}")
            except Exception as e:
                print(f"[CTRL] ⚠️ demo_set_context failed: {e}")


        elif cmd == "yolo":

            try:

                yolo_enabled = bool(val)

                shared_state["yolo_enabled"] = yolo_enabled
                try:
                    shared_state["yolo_generation"] = int(shared_state.get("yolo_generation", 0)) + 1
                except Exception:
                    pass

                from vision.yolo_module import (
                    enable_yolo,
                    disable_yolo,
                    set_yolo_backend,
                    set_yolo_tensorrt_engine,
                    configure_yolo_tensorrt,
                    set_yolo_preprocess_backend,
                )

                if yolo_enabled:
                    try:
                        backend = str(shared_state.get("yolo_backend", "ultralytics")) if shared_state is not None else "ultralytics"
                        trt_engine_path = str(shared_state.get("yolo_trt_engine_path", "") or "") if shared_state is not None else ""
                        trt_dynamic = bool(shared_state.get("yolo_trt_dynamic", False)) if shared_state is not None else False
                        trt_workspace_gb = float(shared_state.get("yolo_trt_workspace_gb", 2.0)) if shared_state is not None else 2.0
                        preprocess_backend = str(shared_state.get("yolo_preprocess_backend", "cpu")) if shared_state is not None else "cpu"
                        preprocess_backend = preprocess_backend.strip().lower()
                        if preprocess_backend not in ("cpu", "cuda"):
                            preprocess_backend = "cpu"
                        set_yolo_backend(backend)
                        configure_yolo_tensorrt(dynamic=trt_dynamic, workspace_gb=trt_workspace_gb)
                        if trt_engine_path:
                            set_yolo_tensorrt_engine(trt_engine_path)
                        if not set_yolo_preprocess_backend(preprocess_backend):
                            if preprocess_backend == "cuda" and set_yolo_preprocess_backend("cpu"):
                                preprocess_backend = "cpu"
                                if shared_state is not None:
                                    shared_state["yolo_preprocess_backend"] = "cpu"
                                print("[CTRL] YOLO preprocess backend fallback on enable: cuda -> cpu")
                    except Exception as e:
                        print(f"[CTRL] YOLO backend runtime config failed: {e}")

                    success = enable_yolo()

                    if success:

                        try:
                            from vision.yolo_module import (
                                set_yolo_confidence,
                                set_yolo_image_size,
                                set_yolo_batch_size,
                                set_yolo_ball_class_id,
                                set_yolo_bayer_pattern,
                                set_yolo_backend,
                                set_yolo_tensorrt_engine,
                                configure_yolo_tensorrt,
                                set_yolo_preprocess_backend,
                            )

                            conf = float(shared_state.get("yolo_confidence", 0.25)) if shared_state is not None else 0.25
                            imgsz = int(shared_state.get("yolo_image_size", 640)) if shared_state is not None else 640
                            bsz = int(shared_state.get("yolo_batch_size", 8)) if shared_state is not None else 8
                            cls_id = int(shared_state.get("yolo_ball_class_id", 0)) if shared_state is not None else 0
                            backend = str(shared_state.get("yolo_backend", "ultralytics")) if shared_state is not None else "ultralytics"
                            trt_engine_path = str(shared_state.get("yolo_trt_engine_path", "") or "") if shared_state is not None else ""
                            trt_dynamic = bool(shared_state.get("yolo_trt_dynamic", False)) if shared_state is not None else False
                            trt_workspace_gb = float(shared_state.get("yolo_trt_workspace_gb", 2.0)) if shared_state is not None else 2.0
                            preprocess_backend = str(shared_state.get("yolo_preprocess_backend", "cpu")) if shared_state is not None else "cpu"
                            preprocess_backend = preprocess_backend.strip().lower()
                            if preprocess_backend not in ("cpu", "cuda"):
                                preprocess_backend = "cpu"

                            set_yolo_backend(backend)
                            configure_yolo_tensorrt(dynamic=trt_dynamic, workspace_gb=trt_workspace_gb)
                            if trt_engine_path:
                                set_yolo_tensorrt_engine(trt_engine_path)
                            set_yolo_confidence(conf)
                            set_yolo_image_size(imgsz)
                            set_yolo_batch_size(bsz)
                            set_yolo_ball_class_id(cls_id)
                            set_yolo_bayer_pattern("BG")
                            if not set_yolo_preprocess_backend(preprocess_backend):
                                if preprocess_backend == "cuda" and set_yolo_preprocess_backend("cpu"):
                                    preprocess_backend = "cpu"
                                    if shared_state is not None:
                                        shared_state["yolo_preprocess_backend"] = "cpu"
                                    print("[CTRL] YOLO preprocess backend fallback after enable: cuda -> cpu")
                            if shared_state is not None:
                                shared_state["yolo_bayer_pattern"] = "BG"
                        except Exception as e:
                            print(f"[CTRL] ⚠️ apply YOLO runtime config failed: {e}")

                        print("[CTRL]  YOLO detection enabled")

                        stats_q.put(" YOLO detection enabled")

                    else:

                        print("[CTRL] ⚠️ YOLO enable failed")

                        stats_q.put("⚠️ YOLO enable failed")

                        shared_state["yolo_enabled"] = False

                else:
                    disable_yolo()

                    print("[CTRL]  YOLO detection disabled")
                    stats_q.put(" YOLO detection disabled")

                    cleared_batch = 0
                    cleared_raw = 0

                    if yolo_batch_q is not None:
                        try:
                            while True:
                                _ = yolo_batch_q.get_nowait()
                                cleared_batch += 1
                        except queue.Empty:
                            pass
                        except Exception as e:
                            print(f"[CTRL] ⚠️ clearing yolo_batch_q failed: {e}")

                    if yolo_raw_q is not None:
                        try:
                            while True:
                                _ = yolo_raw_q.get_nowait()
                                cleared_raw += 1
                        except queue.Empty:
                            pass
                        except Exception as e:
                            print(f"[CTRL] ⚠️ clearing yolo_raw_q failed: {e}")

                    print(f"[CTRL]  Cleared YOLO queues | batch={cleared_batch} raw={cleared_raw}")
                    stats_q.put(f" Cleared YOLO queues | batch={cleared_batch} raw={cleared_raw}")


            except Exception as e:

                print(f"[CTRL] ⚠️ YOLO toggle failed: {e}")

                stats_q.put(f"⚠️ YOLO toggle failed: {e}")

        elif cmd == "remote_stream":
            want_enabled = bool(val.get("enabled")) if isinstance(val, dict) else bool(val)

            if want_enabled:
                try:
                    from streaming.server_stream import start_webrtc_server, set_selected_role
                    url = start_webrtc_server(port=8765,
                                              shared_state=shared_state,
                                              control_q=control_q)

                    if url:
                        # start się udał
                        utils_config.remote_stream_enabled.set()
                        sel = zoom_role_var.value if hasattr(zoom_role_var, "value") else str(zoom_role_var)
                        set_selected_role(sel)
                        shared_state["selected_role"] = sel
                    else:
                        # start nie wyszedł – upewnij się, że wszystko wygląda na OFF
                        utils_config.remote_stream_enabled.clear()
                        shared_state["remote_url"] = None
                        print("[CTRL] WebRTC start returned no URL ().")

                except Exception as e:
                    utils_config.remote_stream_enabled.clear()
                    shared_state["remote_url"] = None
                    print(f"[CTRL]  WebRTC start failed: {e}")

            else:
                try:
                    from streaming.server_stream import stop_webrtc_server
                    stop_webrtc_server()
                except Exception as e:
                    print(f"[CTRL] ⚠️ stop_webrtc_server failed: {e}")

                utils_config.remote_stream_enabled.clear()
                shared_state["remote_url"] = None


        elif cmd == "stop":
            stop_evt.set()
            print("[CTRL] stop")
            break

        else:
            print(f"[CTRL] ⚠️ Nieznana komenda: {cmd}")

    print("[CTRL] .")

# ===========================================================
# Backend initializer
# ===========================================================

def backend_initializer(live_q, stats_q, control_q, ready_evt,
                        recording_event, preview_q_rgb, snapshot_raw_q, raw_q,
                        zoom_role_var, shared_state, zoom_q, simulate,
                        yolo_vis_q=None,archive_mgr=None,yolo_raw_q=None):
    import core.utils_config as utils_config
    from core.utils_config import (
        PER_ROLE_QUEUE_MAX,
    )


    import multiprocessing as mp
    import threading
    import time
    from queue import Queue
    import queue
    from capture.grabber_module import grabber
    from demo_session_logger import init_demo_logger
    from pathlib import Path
    from capture.color_convert_process import universal_color_processor
    try:
        from pypylon import pylon
    except Exception as e:
        stats_q.put(f"⚠️ pypylon import failed: {e} ")
        simulate = True
        pylon = None  # type: ignore
    from storage.shared_memory_manager import get_shared_memory_manager

    stop_evt = utils_config.stop_evt
    recording_event = getattr(utils_config, "recording_event", recording_event)
    snapshot_event = getattr(utils_config, "snapshot_event", mp.Event())
    remote_stream_enabled = getattr(utils_config, "remote_stream_enabled", mp.Event())
    streaming_enabled_raw = getattr(utils_config, "streaming_enabled", {})
    if isinstance(streaming_enabled_raw, dict):
        streaming_enabled_map = streaming_enabled_raw
    else:
        streaming_enabled_map = {}
    streaming_enabled = streaming_enabled_map

    if pylon is None:
        simulate = True

    if yolo_raw_q is None:
        yolo_raw_q = queue.Queue(maxsize=512)
    #==============

    demo_sessions_dir = Path("logs") / "demo_sessions"
    init_demo_logger(demo_sessions_dir)
    #===============
    # === KONFIGURACJA WYJŚĆ / PIPELINE'U ===
    processing_config = {
        "preview":  {"scale": 0.2, "fps": 12, "color_format": "BGR", "send_every_n": 1},
        "remote":   {"scale": 0.8, "fps": 10, "color_format": "RGB", "send_every_n": 1},
        "snapshot": {"scale": 1.00, "fps": 1,  "color_format": "BGR", "send_every_n": 1},
        "zoom":     {"scale": 1.00, "fps": 12, "color_format": "BGR", "send_every_n": 1},
        "yolo":    {"scale": 1.0, "fps": 50, "color_format": "RGB", "send_every_n": 1},

    }

    output_queues = {
        "preview": preview_q_rgb,
        "snapshot": snapshot_raw_q,
        "zoom": zoom_q,

    }

    # === SKANOWANIE KAMER ===
    stats_q.put("🔍 Skanowanie kamer Basler...")
    if simulate:
        devs = []
    else:
        try:
            if pylon is None:
                raise RuntimeError("pypylon is unavailable")
            tl = pylon.TlFactory.GetInstance()
            devs = tl.EnumerateDevices()
        except Exception as e:
            stats_q.put(f"⚠️ EnumerateDevices failed: {e} — SYMULACJA.")
            devs = []
            simulate = True

    # === KOLEJKA SUROWEGO BAYER (klucze do SHM) ===
    universal_bayer_q = mp.Queue(maxsize=64)

    # === UNIWERSALNY PROCES KOLORU ===
    universal_proc = mp.Process(
        target=universal_color_processor,
        args=(universal_bayer_q, output_queues, processing_config, stop_evt, stats_q, remote_stream_enabled, shared_state)
    )
    universal_proc.daemon = True
    universal_proc.start()
    stats_q.put("🎨 Uruchamianie procesora wizji...")

    def relay_preview_to_live_keys():
        """
        Zbiera najnowsze klatki z preview_q_rgb (per rola),
        wyrzuca stare i przepycha tylko najświeższe do live_q.
        """
        import time as _time
        import queue

        last_stats_t = _time.time()
        forwarded = 0  # licznik tylko dla live_q (główne GUI)
        dropped_preview = 0

        while not stop_evt.is_set():
            newest_per_role = {}

            # 1. Opróżnij wszystkie kolejki preview dla każdej roli
            for role, q in preview_q_rgb.items():
                latest = None
                try:
                    while True:
                        item = q.get_nowait()
                        latest = item
                except queue.Empty:
                    pass

                if latest is None:
                    continue

                # latest może być (role2, key, ts_ns) albo np. sam "key"
                if isinstance(latest, tuple) and len(latest) >= 2:
                    role2 = latest[0]
                    key = latest[1]
                    ts_ns = latest[2] if len(latest) >= 3 else _time.time_ns()
                else:
                    role2 = role
                    key = latest
                    ts_ns = _time.time_ns()

                newest_per_role[role2] = (role2, key, ts_ns)

            # 2. Jeśli nie ma nowych klatek – chwila przerwy
            if not newest_per_role:
                _time.sleep(0.003)
            else:
                # 3. Do live_q wrzucamy po 1, NAJNOWSZEJ klatce na rolę
                for role2, triple in newest_per_role.items():
                    # --- główne GUI (lokalne live_q) ---
                    try:
                        if live_q.full():
                            try:
                                _ = live_q.get_nowait()
                                dropped_preview += 1
                            except queue.Empty:
                                pass
                        live_q.put_nowait(triple)
                        forwarded += 1
                    except queue.Full:
                        dropped_preview += 1

            # 4. (opcjonalnie) statystyki do GUI / logów
            now = _time.time()
            if stats_q is not None and (now - last_stats_t) >= 1.0:
                try:
                    msg = {
                        "type": "relay_stats",
                        "forwarded": forwarded,
                        "dropped_preview": dropped_preview,
                        "live_q_size": live_q.qsize() if hasattr(live_q, "qsize") else -1,
                    }
                    stats_q.put_nowait(msg)
                except Exception:
                    pass
                last_stats_t = now

    threading.Thread(target=relay_preview_to_live_keys, daemon=True).start()

    # === SYMULACJA (gdy brak kamer) ===
    def sim_grabber(role):
        import time as _time
        import numpy as np
        smm = get_shared_memory_manager()
        t = 0
        h,w = 640,360
        while not stop_evt.is_set():
            try:
                preview_evt = streaming_enabled_map.get(role)
                preview_active = bool(preview_evt is not None and preview_evt.is_set())
            except Exception:
                preview_active = False
            try:
                selected_role = shared_state.get("selected_role") if shared_state else None
            except Exception:
                selected_role = None
            remote_active = bool(remote_stream_enabled.is_set() and selected_role == role)
            if not (preview_active or remote_active):
                _time.sleep(0.03)  # idle – praktycznie zero CPU
                continue
            t += 1
            y, x = np.indices((h, w), dtype=np.int32)
            base = ((x + t * 3) // 4 + (y * 2) // 5) & 0xFF
            base[h // 2 - 4:h // 2 + 4, w // 2 - 50:w // 2 + 50] = (t * 5) & 0xFF
            bayer = base.astype(np.uint8, copy=False)

            ts_ns = int(_time.time() * 1e9)
            key = f"{role}_bayer_universal_{ts_ns}"
            ok = smm.write_frame(key, bayer, ts_ns=ts_ns)
            if ok:
                try:
                    universal_bayer_q.put_nowait((role, key, ts_ns))
                except queue.Full:
                    try:
                        _ = universal_bayer_q.get_nowait()
                        universal_bayer_q.put_nowait((role, key, ts_ns))
                    except Exception:
                        pass

            if snapshot_event.is_set():
                try:
                    snapshot_raw_q.put_nowait((role, bayer.copy(), ts_ns))
                except queue.Full:
                    pass
            _time.sleep(1 / 12)

    if (simulate or not devs) and os.getenv("VOLLEYHUB_SIMULATE","0")=="1":
        stats_q.put("⚠️ Brak wykrytych kamer — włączam SYMULACJĘ.")
        roles = list(shared_state.get("roles", ["CENTER_L", "CENTER_R", "LEFT", "RIGHT"]))
        for r in roles:
            threading.Thread(target=sim_grabber, args=(r,), daemon=True).start()
        for r in roles:
            evt = streaming_enabled_map.get(r)
            if evt is not None:
                evt.clear()
        utils_config._RUNTIME_PROCS = {"stop_evt": stop_evt, "proc_color": {"universal": universal_proc}}
        ready_evt.set()
        return

    # === REAL CAMS ===
    def init_cameras():
        if pylon is None:
            raise RuntimeError("pypylon is unavailable")
        role_by_serial = load_roles(devs)
        cams, roles_local, serials = [], [], []
        for i, di in enumerate(devs):
            ser = di.GetSerialNumber()
            role = role_by_serial.get(ser, f"CAM{i}")
            stats_q.put(f"Tworzę kamerę {role} ({ser})...")
            cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(di))
            try:
                cam.Open()
                time.sleep(0.3)
                cam_config(ser, cam,stats_q)
                params = shared_state.get("initial_camera_params", {})
                per_role = params.get("per_role", {})
                per_cam  = params.get("per_camera", {})
                # if exp_cfg is not None:
                #     try:
                        # try:
                        #     mn, mx = cam.ExposureTime.GetMin(), cam.ExposureTime.GetMax()
                        # except Exception:
                        #     mn, mx = 10.0, 1_000_000.0
                    #     cam.ExposureTime.SetValue(max(mn, min(mx, float(exp_cfg))))
                    # except Exception as e:
                    #     stats_q.put(f"⚠️ ExposureTime={exp_cfg} fail: {e}")

                # Gain
                gain_cfg = per_cam.get(ser, {}).get("gain") \
                           or per_role.get(role, {}).get("gain") \
                           or params.get("gain_val")
                if gain_cfg is not None:
                    try:
                        try:
                            mn, mx = cam.Gain.GetMin(), cam.Gain.GetMax()
                        except Exception:
                            mn, mx = 0.0, 24.0
                        cam.Gain.SetValue(max(mn, min(mx, float(gain_cfg))))
                    except Exception as e:
                        stats_q.put(f"⚠️ Gain={gain_cfg} fail: {e}")

                cams.append(cam)
                roles_local.append(role)
                serials.append(ser)


                stats_q.put({
                    "type": "camera_info",
                    "role": role,
                    "model": di.GetModelName(),
                    "fw": di.GetDeviceVersion(),
                    "serial_number": ser,
                })
            except Exception as e:
                stats_q.put(f" Błąd kamery {ser}: {e}")
                try:
                    cam.Close()
                except Exception:
                    pass
                continue
        return cams, roles_local, serials

    cams, roles, serials = init_cameras()
    try:
        skip_ptp_wait = bool(shared_state.get("skip_ptp_wait", False)) if shared_state is not None else False
    except Exception:
        skip_ptp_wait = False
    if skip_ptp_wait:
        ptp_ready = False
        stats_q.put("[INIT] ⚠️ Pomijam czekanie na PTP (skip_ptp_wait=True)")
    else:
        ptp_ready = wait_for_all_ptp_ready(cams, max_wait=60)
    for cam, serial in zip(cams, serials):
        try:
            period_us = float(shared_state.get("camera_period_us", 20000.0))
        except Exception:
            period_us = 20000.0
        configure_periodic_signal_trigger(
            cam,
            serial,
            period_us=period_us,
        )


    try:
        shared_state["serial_by_role"] = dict(zip(roles, serials))
        shared_state["role_by_serial"] = dict(zip(serials, roles))
        shared_state["roles"] = roles
        if shared_state.get("bayer_key") is None:
            manager_obj = getattr(utils_config, "_MANAGER", None)
            if manager_obj is not None and hasattr(manager_obj, "dict"):
                shared_state["bayer_key"] = manager_obj.dict()
            else:
                shared_state["bayer_key"] = {}
    except Exception:
        pass

    if not cams:
        raise RuntimeError("Brak poprawnie skonfigurowanych kamer")

    # --- PTP + Periodic Signal ---
    try:
        for cam, role in zip(cams, roles):

            stats_q.put(f"[INIT] ??? Periodic Signal configured: {role}")
        print("[INIT] >>> PRZED wait_for_all_ptp_ready()")
        stats_q.put("[INIT] >>> PRZED wait_for_all_ptp_ready()")
        stats_q.put("[INIT] ⏳ Czekam na PTP ready...")

        stats_q.put(f"[INIT] <<< PO wait_for_all_ptp_ready(): {ptp_ready}")

        if not ptp_ready and not skip_ptp_wait:
            stats_q.put("[INIT]  PTP nie jest gotowe")
            raise RuntimeError(
                "PTP nie ustabilizowało się na wszystkich kamerach - przerywam start Periodic Signal"
            )

        if ptp_ready:
            stats_q.put("[INIT] ??? PTP ready")
        else:
            stats_q.put("[INIT] ⚠️ Start bez PTP ready (tryb skip)")

        if pylon is None:
            raise RuntimeError("pypylon is unavailable")
        for cam, role in zip(cams, roles):
            cam.StartGrabbing(pylon.GrabStrategy_OneByOne)
            stats_q.put(f"[INIT] ▶️StartGrabbing: {role}")

    except Exception as e:
        stats_q.put(f"[INIT]  Periodic Signal config failed: {e}")
        raise


    # diagnostyka PTP
    #calibrate_ptp_offsets(cams, roles, stats_q)

    # === URUCHOM GRABBERY ===
    per_role_q = {r: Queue(maxsize=PER_ROLE_QUEUE_MAX) for r in roles}
    save_raw_q = queue.Queue(maxsize=256)
    yolo_batch_q_maxsize = 4
    try:
        if shared_state is not None:
            yolo_batch_q_maxsize = int(shared_state.get("yolo_batch_q_maxsize", 4) or 4)
    except Exception:
        yolo_batch_q_maxsize = 4
    yolo_batch_q = queue.Queue(maxsize=max(4, yolo_batch_q_maxsize))
    if yolo_raw_q is None:
        yolo_raw_q_maxsize = 512
        try:
            if shared_state is not None:
                yolo_raw_q_maxsize = int(shared_state.get("yolo_raw_q_maxsize", 512) or 512)
        except Exception:
            yolo_raw_q_maxsize = 512
        yolo_raw_q = mp.Queue(maxsize=max(16, yolo_raw_q_maxsize))


    grabber_threads = {}
    for i, (cam, role, serial) in enumerate(zip(cams, roles, serials)):
        th = threading.Thread(
            target=grabber,
            kwargs=dict(
                cam_idx=i,
                cam=cam,
                role=role,
                serial=serial,
                raw_q=raw_q,
                universal_bayer_q=universal_bayer_q,
                stats_q=stats_q,
                stop_evt=stop_evt,
                streaming_enabled=streaming_enabled,
                remote_stream_enabled=remote_stream_enabled,
                recording_event=recording_event,
                snapshot_event=snapshot_event,
                snapshot_raw_q=snapshot_raw_q,
                shared_state=shared_state,yolo_raw_q=yolo_raw_q,
            ),
            daemon=True,
            name=f"grabber_{role}",
        )
        th.start()
        grabber_threads[role] = th
        print(f"[DEBUG] [backend_initializer] recording_event id={id(recording_event)} stop_evt id={id(stop_evt)}")

    stats_q.put("Kamery uruchomione")


    # === Wątki pomocnicze ===

    # === Wątki pomocnicze ===
    def start_helper_threads():
        from vision.yolo_module import process_batch_with_yolo, is_yolo_enabled

        def raw_fanout_router():
            """
            Czyta wspólne raw_q i rozdziela klatki warunkowo:
              - do save_raw_q tylko gdy recording_event jest ON
              - do yolo_raw_q tylko gdy YOLO jest ON
            Gdy oba OFF -> tylko czyści raw_q na bieżąco, bez dalszego przetwarzania.
            """
            forwarded = 0
            save_forwarded = 0
            dropped_save = 0

            while not stop_evt.is_set():
                try:
                    item = raw_q.get(timeout=0.10)
                except queue.Empty:
                    continue
                except (OSError, EOFError, BrokenPipeError):
                    print("[RAW_FANOUT] queue handle closed → exiting")
                    break
                except Exception as e:
                    print(f"[RAW_FANOUT] get() error: {e}")
                    continue

                if not isinstance(item, (tuple, list)) or len(item) < 3:
                    continue

                forwarded += 1

                save_on = False

                try:
                    save_on = bool(recording_event.is_set())
                except Exception:
                    save_on = False

                # -------- tor zapisu --------
                if save_on:
                    try:
                        if save_raw_q.full():
                            try:
                                _ = save_raw_q.get_nowait()
                                dropped_save += 1
                            except Exception:
                                pass
                        save_raw_q.put_nowait(item)
                        save_forwarded += 1
                    except Exception as e:
                        dropped_save += 1
                        print(f"[RAW_FANOUT] save_raw_q put error: {e}")


        def raw_router_save():
            """
            Prosty tor zapisu:
              save_raw_q -> per_role_q[role]
            Bez batchowania, bez YOLO.
            """
            processed = 0
            dropped = 0
            last_log_t = time.time()

            while not stop_evt.is_set():
                try:
                    item = save_raw_q.get(timeout=0.10)
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"[RAW_SAVE] get() error: {e}")
                    continue

                if not isinstance(item, (tuple, list)) or len(item) < 3:
                    continue

                role, ts_ns, frame = item[0], item[1], item[2]
                q = per_role_q.get(role)
                if q is None:
                    continue

                try:
                    q.put_nowait((role, ts_ns, frame))
                    processed += 1
                except queue.Full:
                    try:
                        _ = q.get_nowait()
                        q.put_nowait((role, ts_ns, frame))
                        processed += 1
                    except Exception:
                        dropped += 1
                except Exception as e:
                    dropped += 1
                    print(f"[RAW_SAVE] put() error for {role}: {e}")

                now = time.time()
                if now - last_log_t >= 2.0:
                    print(
                        f"[RAW_SAVE] processed={processed}, dropped={dropped}, "
                        f"save_raw_q={save_raw_q.qsize() if hasattr(save_raw_q, 'qsize') else 'NA'}"
                    )
                    last_log_t = now




        def router_yolo_from_processed():
            from storage.shared_memory_manager import get_shared_memory_manager
            from vision.yolo_module import is_yolo_enabled

            smm = get_shared_memory_manager()

            expected_roles = list(roles)

            FRAME_PERIOD_NS = 20_000_000

            # MA BYĆ
            STEP_TIMEOUT_MS = 400.0  # nie trzymaj starych bucketów
            MATCH_TOL_NS = 5_000_000
            MAX_PENDING_BUCKETS = 64  # mały bufor pending dla live
            seq_len = 2  # LIVE: zawsze 1 krok czasowy
            last_seq_len = seq_len
            ingress_per_role = {r: 0 for r in expected_roles}
            missing_per_role = {r: 0 for r in expected_roles}

            full_steps_built = 0


            sync_state = "WARMUP_SYNC"
            sync_ready = False

            sync_total_full_steps = 0
            sync_good_streak = 0
            sync_bad_streak = 0
            missing_streak = 0

            READY_MIN_FULL_STEPS = 6
            READY_MAX_SPAN_MS = 0.5

            LOST_BAD_STREAK = 100
            LOST_MISSING_STREAK = 300

            pending_steps = {}
            sequence_buffer = []
            batch_collect_t0 = None

            produced = 0
            dropped = 0
            dropped_incomplete = 0
            last_log_t = time.time()
            router_log_interval_s = 4.0

            # diagnostyka
            steps_completed_total = 0
            batches_enqueued_total = 0
            frames_enqueued_total = 0
            batch_q_full_count = 0
            batch_q_drop_oldest_count = 0
            batch_q_drop_newest_count = 0
            enqueue_block_wait_ms_total = 0.0
            enqueue_block_wait_events = 0

            def pipeline_logs_enabled() -> bool:
                try:
                    return bool(shared_state.get("verbose_pipeline_logs", False)) if shared_state is not None else False
                except Exception:
                    return False

            def get_batch_queue_policy() -> str:
                try:
                    raw = str(shared_state.get("yolo_batch_queue_policy", "drop_oldest")) if shared_state is not None else "drop_oldest"
                except Exception:
                    raw = "drop_oldest"
                policy = raw.strip().lower()
                aliases = {
                    "latest": "drop_oldest",
                    "latest_only": "drop_oldest",
                    "live_drop_oldest": "drop_oldest",
                    "drop_oldest": "drop_oldest",
                    "drop_newest": "drop_newest",
                    "fifo": "blocking_fifo",
                    "blocking_fifo": "blocking_fifo",
                }
                return aliases.get(policy, "drop_oldest")

            def enqueue_yolo_batch(payload, frames_to_enqueue: int) -> bool:
                nonlocal produced, batches_enqueued_total, frames_enqueued_total
                nonlocal batch_q_full_count, batch_q_drop_oldest_count, batch_q_drop_newest_count
                nonlocal dropped, enqueue_block_wait_ms_total, enqueue_block_wait_events

                policy = get_batch_queue_policy()

                if policy == "blocking_fifo":
                    blocked_t0 = time.perf_counter()
                    enqueued = False
                    while not enqueued and not stop_evt.is_set():
                        try:
                            yolo_batch_q.put(payload, timeout=0.2)
                            produced += 1
                            batches_enqueued_total += 1
                            frames_enqueued_total += int(frames_to_enqueue)
                            enqueued = True
                        except queue.Full:
                            batch_q_full_count += 1
                            if pipeline_logs_enabled() and batch_q_full_count % 20 == 0:
                                print("[YOLO_ROUTER] yolo_batch_q full -> waiting (blocking_fifo)")
                    if enqueued:
                        blocked_ms = max(0.0, (time.perf_counter() - blocked_t0) * 1000.0)
                        if blocked_ms > 0.01:
                            enqueue_block_wait_ms_total += blocked_ms
                            enqueue_block_wait_events += 1
                    return enqueued

                if policy == "drop_newest":
                    try:
                        yolo_batch_q.put_nowait(payload)
                        produced += 1
                        batches_enqueued_total += 1
                        frames_enqueued_total += int(frames_to_enqueue)
                        return True
                    except queue.Full:
                        batch_q_full_count += 1
                        batch_q_drop_newest_count += 1
                        dropped += 1
                        return False

                # Default/live policy: keep the freshest data.
                # If queue is full, evict one oldest batch and insert current batch.
                try:
                    yolo_batch_q.put_nowait(payload)
                    produced += 1
                    batches_enqueued_total += 1
                    frames_enqueued_total += int(frames_to_enqueue)
                    return True
                except queue.Full:
                    batch_q_full_count += 1
                    try:
                        _ = yolo_batch_q.get_nowait()
                        batch_q_drop_oldest_count += 1
                        dropped += 1
                    except Exception:
                        pass
                    try:
                        yolo_batch_q.put_nowait(payload)
                        produced += 1
                        batches_enqueued_total += 1
                        frames_enqueued_total += int(frames_to_enqueue)
                        return True
                    except Exception:
                        batch_q_drop_newest_count += 1
                        dropped += 1
                        return False

            def make_bucket(ts_ns: int) -> int:
                return int(((ts_ns + FRAME_PERIOD_NS // 2) // FRAME_PERIOD_NS) * FRAME_PERIOD_NS)

            def slot_ref_ts(slot):
                ts_vals = list(slot["ts"].values())
                if not ts_vals:
                    return None
                return int(sum(ts_vals) / len(ts_vals))

            def choose_bucket_for_frame(role: str, ts_ns: int) -> int:
                nominal = make_bucket(ts_ns)

                candidates = [
                    nominal - FRAME_PERIOD_NS,
                    nominal,
                    nominal + FRAME_PERIOD_NS,
                ]

                best_bucket = None
                best_delta = None

                for bucket_ts in candidates:
                    slot = pending_steps.get(bucket_ts)
                    if slot is None:
                        continue
                    if role in slot["frames"]:
                        continue

                    ref_ts = slot_ref_ts(slot)
                    if ref_ts is None:
                        continue

                    delta = abs(int(ts_ns) - ref_ts)
                    if delta <= MATCH_TOL_NS:
                        if best_delta is None or delta < best_delta:
                            best_delta = delta
                            best_bucket = bucket_ts

                return best_bucket if best_bucket is not None else nominal

            def flush_yolo_batch_queue():
                cleared = 0
                try:
                    while True:
                        _ = yolo_batch_q.get_nowait()
                        cleared += 1
                except Exception:
                    pass
                return cleared

            def reset_after_sync_ready():
                nonlocal sequence_buffer, pending_steps, batch_collect_t0
                sequence_buffer.clear()
                pending_steps.clear()
                batch_collect_t0 = None
                cleared = flush_yolo_batch_queue()
                if pipeline_logs_enabled():
                    print(f"[SYNC_STATE] reset buffers after SYNC_OK | cleared_yolo_batch_q={cleared}")


            stats_last_t = time.time()
            while not stop_evt.is_set():
                empty_polls = 0
                try:
                    current_seq_len = max(1, int(shared_state.get("yolo_seq_len", 1))) if shared_state else 1
                except Exception:
                    current_seq_len = last_seq_len

                if current_seq_len != last_seq_len:
                    seq_len = current_seq_len
                    last_seq_len = current_seq_len
                    sequence_buffer.clear()
                    batch_collect_t0 = None
                    if pipeline_logs_enabled():
                        print(f"[YOLO_ROUTER] seq_len changed to {seq_len} -> cleared sequence_buffer")
                try:
                    yolo_on = bool(is_yolo_enabled())
                except Exception:
                    yolo_on = False

                if not yolo_on:
                    pending_steps.clear()
                    sequence_buffer.clear()
                    batch_collect_t0 = None

                    try:
                        while True:
                            _ = yolo_raw_q.get_nowait()
                    except Exception:
                        pass

                    time.sleep(0.05)
                    continue

                try:
                    item = yolo_raw_q.get(timeout=0.05)
                    items_to_route = [item]

                    # FIFO: zachowujemy pełną ciągłość ramek (bez latest-only drain)
                    items_to_route = [item]

                    try:
                        yolo_raw_q_size = yolo_raw_q.qsize() if hasattr(yolo_raw_q, "qsize") else -1
                    except Exception:
                        yolo_raw_q_size = -1

                    try:
                        yolo_batch_q_size = yolo_batch_q.qsize() if hasattr(yolo_batch_q, "qsize") else -1
                    except Exception:
                        yolo_batch_q_size = -1
                    now_stats = time.time()
                    produced += 1
                    if (now_stats - stats_last_t) >= router_log_interval_s:
                        bucket_fill_counts = {str(k): 0 for k in range(1, len(expected_roles) + 1)}
                        oldest_pending_bucket_age_ms = 0.0
                        oldest_ready_step_age_ms = 0.0
                        try:
                            for _bts, _slot in pending_steps.items():
                                lvl = len((_slot or {}).get("frames", {}))
                                if 1 <= lvl <= len(expected_roles):
                                    bucket_fill_counts[str(lvl)] += 1
                            if pending_steps:
                                oldest_pending_created_t = min(
                                    float((_slot or {}).get("created_t", now_stats)) for _slot in pending_steps.values()
                                )
                                oldest_pending_bucket_age_ms = max(0.0, (now_stats - oldest_pending_created_t) * 1000.0)
                            if sequence_buffer:
                                oldest_ready_t = min(float((_s or {}).get("ready_t", now_stats)) for _s in sequence_buffer)
                                oldest_ready_step_age_ms = max(0.0, (now_stats - oldest_ready_t) * 1000.0)
                        except Exception:
                            pass
                        if pipeline_logs_enabled():
                            print(
                                f"[YOLO_ROUTER] "
                                f"sync_state={sync_state} sync_ready={sync_ready} "
                                f"steps_completed={steps_completed_total} "
                                f"batches_enqueued={batches_enqueued_total} "
                                f"frames_enqueued={frames_enqueued_total} "
                                f"dropped={dropped} dropped_incomplete={dropped_incomplete} "
                                f"drop_oldest={batch_q_drop_oldest_count} drop_newest={batch_q_drop_newest_count} "
                                f"pending_buckets={len(pending_steps)} ready_steps={len(sequence_buffer)} "
                                f"yolo_raw_q={yolo_raw_q_size} yolo_batch_q={yolo_batch_q_size}"
                            )
                        try:
                            enqueue_block_wait_ms_avg = (
                                float(enqueue_block_wait_ms_total / enqueue_block_wait_events)
                                if enqueue_block_wait_events > 0 else 0.0
                            )
                            router_snapshot = {
                                "type": "yolo_router_summary",
                                "sync_state": sync_state,
                                "sync_ready": bool(sync_ready),
                                "steps_completed": int(steps_completed_total),
                                "batches_enqueued": int(batches_enqueued_total),
                                "frames_enqueued": int(frames_enqueued_total),
                                "dropped": int(dropped),
                                "dropped_incomplete": int(dropped_incomplete),
                                "batch_q_full_count": int(batch_q_full_count),
                                "batch_q_drop_oldest_count": int(batch_q_drop_oldest_count),
                                "batch_q_drop_newest_count": int(batch_q_drop_newest_count),
                                "batch_queue_policy": get_batch_queue_policy(),
                                "enqueue_block_wait_events": int(enqueue_block_wait_events),
                                "enqueue_block_wait_ms_avg": float(enqueue_block_wait_ms_avg),
                                "pending_buckets": int(len(pending_steps)),
                                "ready_steps": int(len(sequence_buffer)),
                                "yolo_raw_q": int(yolo_raw_q_size),
                                "yolo_batch_q": int(yolo_batch_q_size),
                                "full_steps": int(full_steps_built),
                                "expected_roles": int(len(expected_roles)),
                                "bucket_fill_counts": bucket_fill_counts,
                                "oldest_pending_bucket_age_ms": float(oldest_pending_bucket_age_ms),
                                "oldest_ready_step_age_ms": float(oldest_ready_step_age_ms),
                                "ingress_per_role": dict(ingress_per_role),
                                "missing_per_role": dict(missing_per_role),
                                "ts": now_stats,
                            }
                            stats_q.put_nowait(router_snapshot)
                            try:
                                if shared_state is not None:
                                    shared_state["yolo_router_stats_last"] = dict(router_snapshot)
                            except Exception:
                                pass
                        except Exception:
                            pass
                        stats_last_t = now_stats

                except queue.Empty:
                    empty_polls += 1
                    if pipeline_logs_enabled() and empty_polls % 100 == 0:
                        print(f"[YOLO_ROUTER] empty polls={empty_polls}")
                    items_to_route = []
                except Exception as e:
                    if pipeline_logs_enabled():
                        print(f"[YOLO_ROUTER] get() error: {e}")
                    items_to_route = []

                now_t = time.time()

                for routed_item in items_to_route:
                    if not isinstance(routed_item, (tuple, list)) or len(routed_item) < 3:
                        continue
                    role, frame_key, ts_ns = routed_item[0], routed_item[1], routed_item[2]

                    if role in expected_roles:
                        data = smm.read_frame(frame_key)
                        ingress_per_role[role] += 1
                        if data is not None:
                            frame = data[0] if isinstance(data, tuple) else data
                            bucket_ts = choose_bucket_for_frame(role, int(ts_ns))

                            slot = pending_steps.get(bucket_ts)
                            if slot is None:
                                slot = {
                                    "frames": {},
                                    "ts": {},
                                    "created_t": now_t,
                                    "updated_t": now_t,
                                }
                                pending_steps[bucket_ts] = slot

                            slot["frames"][role] = frame
                            slot["ts"][role] = int(ts_ns)
                            slot["updated_t"] = now_t

                if len(pending_steps) > MAX_PENDING_BUCKETS:
                    over = len(pending_steps) - MAX_PENDING_BUCKETS
                    for bucket_ts in sorted(pending_steps.keys())[:over]:
                        slot = pending_steps.pop(bucket_ts, None)
                        if slot is None:
                            continue
                        if len(slot["frames"]) < len(expected_roles):
                            dropped_incomplete += 1

                ready_bucket_keys = []
                for bucket_ts, slot in pending_steps.items():
                    if all(r in slot["frames"] for r in expected_roles):
                        ready_bucket_keys.append(bucket_ts)

                ready_bucket_keys.sort()

                for bucket_ts in ready_bucket_keys:
                    full_steps_built += 1
                    steps_completed_total += 1
                    slot = pending_steps.pop(bucket_ts)

                    step_frames = [slot["frames"][r] for r in expected_roles]
                    step_ts = [slot["ts"][r] for r in expected_roles]

                    ts_span_ns = max(step_ts) - min(step_ts) if step_ts else 0
                    ts_span_ms = ts_span_ns / 1e6

                    sync_total_full_steps += 1
                    missing_streak = 0

                    if ts_span_ms <= READY_MAX_SPAN_MS:
                        sync_good_streak += 1
                        sync_bad_streak = 0
                    else:
                        sync_bad_streak += 1
                        sync_good_streak = 0

                    if (
                            sync_state == "WARMUP_SYNC"
                            and sync_total_full_steps >= READY_MIN_FULL_STEPS
                            and sync_good_streak >= READY_MIN_FULL_STEPS
                    ):
                        sync_state = "SYNC_OK"
                        sync_ready = True
                        pending_steps.clear()
                        sequence_buffer.clear()
                        batch_collect_t0 = None
                        print(
                            f"[SYNC_STATE] -> SYNC_OK | "
                            f"full_steps={sync_total_full_steps} "
                            f"good_streak={sync_good_streak} "
                            f"span_ms={ts_span_ms:.6f}"
                        )

                        reset_after_sync_ready()

                        try:
                            if shared_state is not None:
                                shared_state["ptp_sync_ready"] = True
                                shared_state["ptp_sync_state"] = "SYNC_OK"
                                shared_state["ptp_sync_span_ms"] = float(ts_span_ms)
                        except Exception:
                            pass

                        continue

                    elif sync_state == "SYNC_OK" and sync_bad_streak >= LOST_BAD_STREAK:
                        sync_state = "SYNC_LOST"
                        sync_ready = False
                        sequence_buffer.clear()
                        batch_collect_t0 = None
                        print(
                            f"[SYNC_STATE] -> SYNC_LOST | "
                            f"bad_streak={sync_bad_streak} "
                            f"span_ms={ts_span_ms:.6f}"
                        )

                    elif sync_state == "SYNC_LOST" and sync_good_streak >= READY_MIN_FULL_STEPS:
                        sync_state = "SYNC_OK"
                        sync_ready = True
                        print(
                            f"[SYNC_STATE] -> SYNC_OK (recovered) | "
                            f"good_streak={sync_good_streak} "
                            f"span_ms={ts_span_ms:.6f}"
                        )

                        try:
                            if shared_state is not None:
                                shared_state["ptp_sync_ready"] = True
                                shared_state["ptp_sync_state"] = "SYNC_OK"
                                shared_state["ptp_sync_span_ms"] = float(ts_span_ms)
                        except Exception:
                            pass

                        continue

                    try:
                        if shared_state is not None:
                            shared_state["ptp_sync_ready"] = sync_ready
                            shared_state["ptp_sync_state"] = sync_state
                            shared_state["ptp_sync_span_ms"] = float(ts_span_ms)
                    except Exception:
                        pass

                    if not sync_ready:
                        continue

                    sequence_buffer.append({
                        "frames": step_frames,
                        "ts": step_ts,
                        "bucket_ts": bucket_ts,
                        "ts_span_ns": ts_span_ns,
                        "created_t": float(slot.get("created_t", now_t)),
                        "ready_t": float(now_t),
                    })

                    if batch_collect_t0 is None:
                        batch_collect_t0 = now_t

                stale_bucket_keys = []
                for bucket_ts, slot in pending_steps.items():
                    age_ms = (now_t - slot["created_t"]) * 1000.0
                    if age_ms >= STEP_TIMEOUT_MS:
                        stale_bucket_keys.append(bucket_ts)

                for bucket_ts in stale_bucket_keys:
                    slot = pending_steps.pop(bucket_ts, None)
                    if slot is None:
                        continue

                    missing = [r for r in expected_roles if r not in slot["frames"]]
                    for r in missing:
                        missing_per_role[r] += 1
                    dropped_incomplete += 1

                    if sync_state == "SYNC_OK":
                        missing_streak += 1
                        if missing_streak >= LOST_MISSING_STREAK:
                            sync_state = "SYNC_LOST"
                            sync_ready = False
                            sequence_buffer.clear()
                            batch_collect_t0 = None
                            print(
                                f"[SYNC_STATE] -> SYNC_LOST | "
                                f"missing_streak={missing_streak} missing={missing}"
                            )

                    try:
                        if shared_state is not None:
                            shared_state["ptp_sync_ready"] = sync_ready
                            shared_state["ptp_sync_state"] = sync_state
                    except Exception:
                        pass

                if len(sequence_buffer) >= seq_len:
                    chunk = sequence_buffer[:seq_len]
                    sequence_buffer = sequence_buffer[seq_len:]

                    frames_by_step = [step["frames"] for step in chunk]
                    ts_by_step = [step["ts"] for step in chunk]

                    batch_collect_ms = (time.time() - batch_collect_t0) * 1000.0 if batch_collect_t0 else 0.0
                    enqueue_ts = time.time()

                    payload = {
                        "frames_by_step": frames_by_step,
                        "roles": expected_roles,
                        "ts_by_step": ts_by_step,
                        "batch_collect_ms": batch_collect_ms,
                        "enqueue_ts": enqueue_ts,
                        "yolo_generation": int(
                            shared_state.get("yolo_generation", 0)) if shared_state is not None else 0,
                    }

                    _ = enqueue_yolo_batch(
                        payload=payload,
                        frames_to_enqueue=len(frames_by_step) * len(expected_roles),
                    )

                batch_collect_t0 = now_t if sequence_buffer else None

                if now_t - last_log_t >= router_log_interval_s:
                    bucket_fill_counts = {str(k): 0 for k in range(1, len(expected_roles) + 1)}
                    oldest_pending_bucket_age_ms = 0.0
                    oldest_ready_step_age_ms = 0.0
                    try:
                        for _bts, _slot in pending_steps.items():
                            lvl = len((_slot or {}).get("frames", {}))
                            if 1 <= lvl <= len(expected_roles):
                                bucket_fill_counts[str(lvl)] += 1
                        if pending_steps:
                            oldest_pending_created_t = min(
                                float((_slot or {}).get("created_t", now_t)) for _slot in pending_steps.values()
                            )
                            oldest_pending_bucket_age_ms = max(0.0, (now_t - oldest_pending_created_t) * 1000.0)
                        if sequence_buffer:
                            oldest_ready_t = min(float((_s or {}).get("ready_t", now_t)) for _s in sequence_buffer)
                            oldest_ready_step_age_ms = max(0.0, (now_t - oldest_ready_t) * 1000.0)
                    except Exception:
                        pass

                    try:
                        yolo_raw_q_size = yolo_raw_q.qsize() if hasattr(yolo_raw_q, "qsize") else -1
                    except Exception:
                        yolo_raw_q_size = -1

                    try:
                        yolo_batch_q_size = yolo_batch_q.qsize() if hasattr(yolo_batch_q, "qsize") else -1
                    except Exception:
                        yolo_batch_q_size = -1

                    if pipeline_logs_enabled():
                        print(
                            f"[YOLO_ROUTER] "
                            f"sync_state={sync_state} sync_ready={sync_ready} "
                                f"steps_completed={steps_completed_total} "
                                f"batches_enqueued={batches_enqueued_total} "
                                f"frames_enqueued={frames_enqueued_total} "
                                f"produced={produced} dropped={dropped} dropped_incomplete={dropped_incomplete} "
                                f"batch_q_full_count={batch_q_full_count} "
                                f"drop_oldest={batch_q_drop_oldest_count} drop_newest={batch_q_drop_newest_count} "
                                f"pending_buckets={len(pending_steps)} ready_steps={len(sequence_buffer)} "
                                f"yolo_raw_q={yolo_raw_q_size} yolo_batch_q={yolo_batch_q_size} "
                                f"full_steps={full_steps_built} "
                                f"ingress={ingress_per_role}, missing={missing_per_role}"
                        )
                    try:
                        enqueue_block_wait_ms_avg = (
                            float(enqueue_block_wait_ms_total / enqueue_block_wait_events)
                            if enqueue_block_wait_events > 0 else 0.0
                        )
                        router_snapshot = {
                            "type": "yolo_router_summary",
                            "sync_state": sync_state,
                            "sync_ready": bool(sync_ready),
                            "steps_completed": int(steps_completed_total),
                            "batches_enqueued": int(batches_enqueued_total),
                            "frames_enqueued": int(frames_enqueued_total),
                            "produced": int(produced),
                            "dropped": int(dropped),
                            "dropped_incomplete": int(dropped_incomplete),
                            "batch_q_full_count": int(batch_q_full_count),
                            "batch_q_drop_oldest_count": int(batch_q_drop_oldest_count),
                            "batch_q_drop_newest_count": int(batch_q_drop_newest_count),
                            "batch_queue_policy": get_batch_queue_policy(),
                            "enqueue_block_wait_events": int(enqueue_block_wait_events),
                            "enqueue_block_wait_ms_avg": float(enqueue_block_wait_ms_avg),
                            "pending_buckets": int(len(pending_steps)),
                            "ready_steps": int(len(sequence_buffer)),
                            "yolo_raw_q": int(yolo_raw_q_size),
                            "yolo_batch_q": int(yolo_batch_q_size),
                            "full_steps": int(full_steps_built),
                            "expected_roles": int(len(expected_roles)),
                            "bucket_fill_counts": bucket_fill_counts,
                            "oldest_pending_bucket_age_ms": float(oldest_pending_bucket_age_ms),
                            "oldest_ready_step_age_ms": float(oldest_ready_step_age_ms),
                            "ingress_per_role": dict(ingress_per_role),
                            "missing_per_role": dict(missing_per_role),
                            "ts": now_t,
                        }
                        stats_q.put_nowait(router_snapshot)
                        try:
                            if shared_state is not None:
                                shared_state["yolo_router_stats_last"] = dict(router_snapshot)
                        except Exception:
                            pass
                    except Exception:
                        pass
                    last_log_t = now_t

        import numpy as np

        ROLE_TO_CAM = {
            "CENTER_L": 0,
            "CENTER_R": 1,
            "LEFT": 2,
            "RIGHT": 3,
        }

        def build_3d_input_from_yolo_detections(detections_by_step, roles_local, max_objects=5):
            """
            Buduje pts_bat i conf_bat WYŁĄCZNIE z realnych detekcji YOLO.

            detections_by_step:
                [
                  {
                    "CENTER_L": [det1, det2, ...],
                    "CENTER_R": [...],
                    "LEFT": [...],
                    "RIGHT": [...]
                  },
                  ...
                ]

            Zwraca:
                pts_bat  -> (N, M, max_objects, 2)
                conf_bat -> (N, M, max_objects)
            """
            N = len(detections_by_step)
            M = len(roles_local)

            pts_bat = np.full((N, M, max_objects, 2), np.nan, dtype=np.float32)
            conf_bat = np.full((N, M, max_objects), np.nan, dtype=np.float32)

            for step_idx, det_step in enumerate(detections_by_step):
                for role in roles_local:
                    cam_idx = ROLE_TO_CAM[role]
                    dets = det_step.get(role, [])

                    if not dets:
                        continue

                    dets_sorted = sorted(
                        dets,
                        key=lambda d: float(d.get("confidence", 0.0)),
                        reverse=True
                    )

                    keep = dets_sorted[:max_objects]

                    for obj_idx, det in enumerate(keep):
                        cx = det.get("center_x", None)
                        cy = det.get("center_y", None)
                        conf = det.get("confidence", None)

                        if cx is None or cy is None or conf is None:
                            continue

                        pts_bat[step_idx, cam_idx, obj_idx, 0] = float(cx)
                        pts_bat[step_idx, cam_idx, obj_idx, 1] = float(cy)
                        conf_bat[step_idx, cam_idx, obj_idx] = float(conf)

            return pts_bat, conf_bat

        def yolo_batch_worker_old():
            # Legacy shim kept only for backward compatibility.
            # Always delegate to the active worker implementation.
            # Prevents accidental execution of stale legacy code paths.
            """

            from demo_session_logger import get_demo_logger
            import time
            import queue
            import numpy as np
            import p3D

            batch_counter = 0
            # Frequent summary updates for live benchmark sampling.
            report_every_s = 1.0
            sleep_when_off_s = 0.05
            last_report_t = time.time()

            # --- nowe liczniki diagnostyczne ---
            batches_processed_total = 0
            frames_processed_total = 0
            frames_expected_total = 0
            worker_last_dequeue_ts = 0.0

            # agregacja okresowa
            yolo_in_frames = 0
            yolo_valid_frames = 0
            yolo_proc_frames = 0
            yolo_detected = 0
            yolo_infer_ms_sum = 0.0
            yolo_infer_count = 0
            yolo_batches = 0
            worker_get_errors = 0
            worker_infer_errors = 0
            worker_vis_errors = 0
            worker_stale_skips = 0
            worker_timing_log_interval_s = 4.0
            last_worker_timing_log_t = 0.0

            # # kalibracja 3D ładujemy raz
            # try:
            #     (
            #         _cfg_json,
            #         P_all,
            #         K_all,
            #         dist_all,
            #         C_all,
            #         pair_transforms,
            #         _conf_json,
            #         _pts_json,
            #     ) = p3D.load_data_from_json("dane_testowe_3D_fixed.json", as_numpy=True)
            #     print("[YOLO_3D] ??? Calibration/config loaded once")
            # except Exception as e:
            #     print(f"[YOLO_3D]  calibration load error: {e}")
            #     P_all = K_all = dist_all = C_all = pair_transforms = None

            while not stop_evt.is_set():
                try:
                    yolo_on = bool(is_yolo_enabled())
                except Exception:
                    yolo_on = False

                if not yolo_on:
                    time.sleep(sleep_when_off_s)
                    continue

                try:
                    # worker: bierz tylko najnowszy batch
                    item = yolo_batch_q.get(timeout=0.2)
                    worker_last_dequeue_ts = time.time()

                    #     print(f"[YOLO_WORKER] latest-only: skipped stale batches={stale_drained}")
                    worker_start_ts = time.time()
                    enqueue_ts = float(item.get("enqueue_ts", worker_start_ts) or worker_start_ts)
                    queue_wait_ms = (worker_start_ts - enqueue_ts) * 1000.0
                    batch_collect_ms = float(item.get("batch_collect_ms", 0.0) or 0.0)

                except queue.Empty:
                    now = time.time()
                    if (now - last_report_t >= report_every_s) and (yolo_in_frames > 0 or yolo_proc_frames > 0):
                        avg_infer_ms = (yolo_infer_ms_sum / yolo_infer_count) if yolo_infer_count > 0 else 0.0
                        fps = 1000.0 / avg_infer_ms if avg_infer_ms > 0 else 0.0
                        try:
                            current_yolo_q = yolo_batch_q.qsize()
                        except Exception:
                            current_yolo_q = -1

                        drop_gap_frames = frames_expected_total - frames_processed_total
                        num_roles = len(roles_local) if roles_local else 4
                        batches_per_sec = 1000.0 / avg_infer_ms if avg_infer_ms > 0 else 0.0
                        images_per_sec = (
                                                     yolo_proc_frames / yolo_infer_ms_sum) * 1000.0 if yolo_infer_ms_sum > 0 else 0.0
                        steps_per_sec = images_per_sec / num_roles if num_roles > 0 else 0.0

                        try:
                            target_steps_per_sec = float(
                                shared_state.get("yolo_target_hz", 50.0)) if shared_state is not None else 50.0
                        except Exception:
                            target_steps_per_sec = 50.0

                        target_util_percent = (
                                    steps_per_sec / target_steps_per_sec * 100.0) if target_steps_per_sec > 0 else 0.0

                        last_images_per_sec = (frames_processed / infer_ms) * 1000.0 if infer_ms > 0 else 0.0
                        last_steps_per_sec = last_images_per_sec / num_roles if num_roles > 0 else 0.0

                        try:
                            if shared_state is not None:
                                shared_state["yolo_buffer_stats"] = {
                                    "batch_id": int(batch_counter),
                                    "batches": int(yolo_batches),


                                    "in_frames": int(yolo_in_frames),
                                    "valid_frames": int(yolo_valid_frames),
                                    "proc_frames": int(yolo_proc_frames),
                                    "detected": int(yolo_detected),

                                    "valid_percent": float(
                                        (yolo_valid_frames / yolo_in_frames * 100.0) if yolo_in_frames > 0 else 0.0
                                    ),
                                    "proc_percent": float(
                                        (yolo_proc_frames / yolo_in_frames * 100.0) if yolo_in_frames > 0 else 0.0
                                    ),

                                    "avg_infer_ms": float(avg_infer_ms),

                                    # stare pole zostawione dla kompatybilności
                                    "fps": float(batches_per_sec),

                                    # nowe, czytelniejsze metryki
                                    "batches_per_sec": float(batches_per_sec),
                                    "images_per_sec": float(images_per_sec),
                                    "steps_per_sec": float(steps_per_sec),
                                    "target_steps_per_sec": float(target_steps_per_sec),
                                    "target_util_percent": float(target_util_percent),

                                    "last_batch_collect_ms": float(batch_collect_ms),
                                    "last_queue_wait_ms": float(queue_wait_ms),
                                    "last_prepare_ms": float(prepare_ms),
                                    "last_forward_ms": float(forward_ms),
                                    "last_post_ms": float(post_ms),
                                    "last_infer_ms": float(infer_ms),
                                    "last_infer_ms_wall": float(infer_ms_wall),

                                    "last_expected_frames": int(expected_frames),
                                    "last_frames_valid": int(frames_valid),
                                    "last_frames_processed": int(frames_processed),
                                    "last_total_detected": int(total_detected),

                                    "last_images_per_sec": float(last_images_per_sec),
                                    "last_steps_per_sec": float(last_steps_per_sec),
                                    "last_steps_in_batch": int(len(frames_by_step)),
                                    "last_images_in_batch": int(expected_frames),

                                    "last_per_role_counts": dict(detections_per_role),
                                    "last_per_step_counts": list(detections_per_step),
                                    "configured_batch_size": int(
                                        shared_state.get("yolo_batch_size", 0)) if shared_state is not None else 0,
                                    "configured_seq_len": int(
                                        shared_state.get("yolo_seq_len", 0)) if shared_state is not None else 0,
                                    "last_batch_size_effective": int(
                                        min(expected_frames, int(shared_state.get("yolo_batch_size",
                                                                                  0)))) if shared_state is not None else int(
                                        expected_frames),

                                    "valid_2d_points": int(valid_2d_points),
                                    "points3D": int(valid_points3d),
                                    "updated_at": time.time(),

                                }
                        except Exception as e:
                            if pipeline_logs_enabled():
                                print(f"[YOLO_WORKER] shared_state yolo_buffer_stats update failed: {e}")

                        try:
                            stats_q.put_nowait({
                                "type": "yolo_summary",
                                "batch_id": batch_counter,
                                "batches": yolo_batches,
                                "in_frames": yolo_in_frames,
                                "valid_frames": yolo_valid_frames,
                                "proc_frames": yolo_proc_frames,
                                "detected": yolo_detected,
                                "avg_infer_ms": avg_infer_ms,
                                "fps": fps,
                                "infer_ms_wall": round(infer_ms_wall, 2),
                                "batch_collect_ms": round(batch_collect_ms, 2),
                                "queue_wait_ms": round(queue_wait_ms, 2),
                                "prepare_ms": round(prepare_ms, 2),
                                "forward_ms": round(forward_ms, 2),
                                "post_ms": round(post_ms, 2),
                                "batches_per_sec": batches_per_sec,
                                "images_per_sec": images_per_sec,
                                "steps_per_sec": steps_per_sec,
                                "target_steps_per_sec": target_steps_per_sec,
                                "target_util_percent": target_util_percent,
                                "last_steps_in_batch": int(len(frames_by_step)),
                                "last_images_in_batch": int(expected_frames),
                                "last_batch_size_effective": int(
                                    min(expected_frames, int(shared_state.get("yolo_batch_size", 0)))
                                ) if shared_state is not None else int(expected_frames),
                                "per_role_counts": detections_per_role,
                                "per_step_counts": detections_per_step,
                                "configured_input_batch_images": int(
                                    shared_state.get("yolo_input_batch_images", expected_frames)),
                                "valid_2d_points": valid_2d_points,
                                "points3D": valid_points3d,
                            })
                        except Exception:
                                pass


                        yolo_in_frames = 0
                        yolo_valid_frames = 0
                        yolo_proc_frames = 0
                        yolo_detected = 0
                        yolo_infer_ms_sum = 0.0
                        yolo_infer_count = 0
                        yolo_batches = 0
                        last_report_t = now

                    continue

                except Exception as e:
                    worker_get_errors += 1
                    print(f"[YOLO_WORKER] get error: {e}")
                    continue

                if not isinstance(item, dict):
                    continue
                try:
                    current_generation = int(shared_state.get("yolo_generation", 0)) if shared_state is not None else 0
                except Exception:
                    current_generation = 0

                item_generation = int(item.get("yolo_generation", current_generation) or current_generation)

                if item_generation != current_generation:
                    print(
                        f"[YOLO_WORKER] stale batch skipped | "
                        f"item_generation={item_generation} current_generation={current_generation}"
                    )
                    continue

                frames_by_step = item.get("frames_by_step", [])
                roles_local = item.get("roles", [])
                ts_by_step = item.get("ts_by_step", [])

                if not frames_by_step or not roles_local:
                    continue

                batch_counter += 1
                yolo_batches += 1
                batches_processed_total += 1

                expected_frames = len(frames_by_step) * len(roles_local)
                frames_expected_total += int(expected_frames)

                # =========================
                # 1) YOLO inference
                # =========================
                try:
                    t_infer0 = time.time()
                    result = process_batch_with_yolo(frames_by_step, roles_local)
                    t_infer1 = time.time()
                    infer_ms_wall = (t_infer1 - t_infer0) * 1000.0

                    detections_by_step = result.get("detections", [])
                    meta = result.get("meta", {})

                    _timing_now = time.time()
                    if (_timing_now - last_worker_timing_log_t) >= worker_timing_log_interval_s:
                        print(
                            f"[YOLO_WORKER_TIMING] "
                            f"batch_collect={batch_collect_ms:.2f} ms | "
                            f"queue_wait={queue_wait_ms:.2f} ms | "
                            f"prepare={float(meta.get('prepare_ms', 0.0)):.2f} ms | "
                            f"forward={float(meta.get('forward_ms', 0.0)):.2f} ms | "
                            f"post={float(meta.get('post_ms', 0.0)):.2f} ms | "
                            f"total={float(meta.get('infer_ms', 0.0)):.2f} ms"
                        )
                        last_worker_timing_log_t = _timing_now

                    infer_ms_meta = meta.get("infer_ms", None)
                    infer_ms = float(infer_ms_meta) if infer_ms_meta is not None else float(infer_ms_wall)

                    frames_valid = int(meta.get("frames_valid", expected_frames))
                    frames_processed = int(meta.get("frames_processed", expected_frames))
                    prepare_ms = float(meta.get("prepare_ms", 0.0) or 0.0)
                    forward_ms = float(meta.get("forward_ms", 0.0) or 0.0)
                    post_ms = float(meta.get("post_ms", 0.0) or 0.0)

                    yolo_in_frames += int(expected_frames)
                    yolo_valid_frames += frames_valid
                    yolo_proc_frames += frames_processed
                    yolo_infer_ms_sum += infer_ms
                    yolo_infer_count += 1
                    frames_processed_total += int(frames_processed)
                    avg_infer_ms = (yolo_infer_ms_sum / yolo_infer_count) if yolo_infer_count > 0 else 0.0
                    fps = 1000.0 / avg_infer_ms if avg_infer_ms > 0 else 0.0

                    try:
                        stats_q.put_nowait({
                            "type": "yolo_summary",
                            "batches_processed_total": int(batches_processed_total),
                            "frames_expected_total": int(frames_expected_total),
                            "frames_processed_total": int(frames_processed_total),
                            "drop_gap_frames": int(drop_gap_frames),
                            "yolo_q_size": int(current_yolo_q),
                            "batch_id": int(batch_counter),
                            "batches": int(yolo_batches),
                            "in_frames": int(yolo_in_frames),
                            "valid_frames": int(yolo_valid_frames),
                            "proc_frames": int(yolo_proc_frames),
                            "detected": int(yolo_detected),
                            "valid_percent": float(
                                (yolo_valid_frames / yolo_in_frames * 100.0) if yolo_in_frames > 0 else 0.0),
                            "proc_percent": float(
                                (yolo_proc_frames / yolo_in_frames * 100.0) if yolo_in_frames > 0 else 0.0),
                            "avg_infer_ms": float(avg_infer_ms),
                            "fps": float(fps),

                            "last_batch_collect_ms": float(batch_collect_ms),
                            "last_queue_wait_ms": float(queue_wait_ms),
                            "last_prepare_ms": float(prepare_ms),
                            "last_forward_ms": float(forward_ms),
                            "last_post_ms": float(post_ms),
                            "last_infer_ms": float(infer_ms),
                            "last_infer_ms_wall": float(infer_ms_wall),

                            "last_expected_frames": int(expected_frames),
                            "last_frames_valid": int(frames_valid),
                            "last_frames_processed": int(frames_processed),
                            "last_total_detected": int(total_detected),

                            "last_per_role_counts": dict(detections_per_role),
                            "last_per_step_counts": list(detections_per_step),
                            "infer_ms_wall": round(infer_ms_wall, 2),
                            "batch_collect_ms": round(batch_collect_ms, 2),
                            "queue_wait_ms": round(queue_wait_ms, 2),
                            "prepare_ms": round(prepare_ms, 2),
                            "forward_ms": round(forward_ms, 2),
                            "post_ms": round(post_ms, 2),
                            "configured_input_batch_images": int(
                                shared_state.get("yolo_input_batch_images", expected_frames)),
                            "per_role_counts": detections_per_role,
                            "per_step_counts": detections_per_step,
                            "valid_2d_points": valid_2d_points,
                            "points3D": valid_points3d,
                        })
                    except Exception:
                        pass

                except Exception as e:
                    print(f"[YOLO_WORKER] inference error: {e}")
                    continue

                # =========================
                # 2) Zlicz detekcje YOLO
                # =========================
                total_detected = 0
                detections_per_role = {r: 0 for r in roles_local}
                detections_per_step = []
                per_role_points = {r: [] for r in roles_local}
                latest_centers_by_role: dict[str, dict | None] = {r: None for r in roles_local}
                raw_best_by_role: dict[str, dict | None] = {r: None for r in roles_local}
                latest_step_idx_by_role = {r: -1 for r in roles_local}
                raw_points_latest_step_by_role = {r: [] for r in roles_local}

                for step_idx, det_step in enumerate(detections_by_step):
                    step_info = {}

                    for r in roles_local:
                        dets = det_step.get(r, [])
                        n = len(dets)

                        step_info[r] = n
                        detections_per_role[r] += n
                        total_detected += n
                        for det in dets:
                            per_role_points[r].append({
                                "x": det["center_x"],
                                "y": det["center_y"],
                                "confidence": det["confidence"],
                                "step_idx": step_idx,
                            })

                        if dets:
                            best_det = max(dets, key=lambda d: float(d.get("confidence", 0.0)))
                            latest_centers_by_role[r] = {
                                "x": float(best_det.get("center_x", 0.0)),
                                "y": float(best_det.get("center_y", 0.0)),
                                "confidence": float(best_det.get("confidence", 0.0)),
                                "src_w": int(best_det.get("frame_w", 1920) or 1920),
                                "src_h": int(best_det.get("frame_h", 1080) or 1080),
                            }

                        if dets:
                            best_det = max(dets, key=lambda d: float(d.get("confidence", 0.0)))
                            latest_centers_by_role[r] = {
                                "x": float(best_det.get("center_x", 0.0)),
                                "y": float(best_det.get("center_y", 0.0)),
                                "confidence": float(best_det.get("confidence", 0.0)),
                                "src_w": int(best_det.get("frame_w", 1920) or 1920),
                                "src_h": int(best_det.get("frame_h", 1080) or 1080),
                            }

                    detections_per_step.append(step_info)

                yolo_detected += int(total_detected)
                print(
                    f"[YOLO_WORKER_STATS] "
                    f"batches_processed_total={batches_processed_total} "
                    f"frames_expected_total={frames_expected_total} "
                    f"frames_processed_total={frames_processed_total} "
                    f"drop_gap_frames={drop_gap_frames} "
                    f"yolo_q={current_yolo_q} "
                    f"avg_infer_ms={avg_infer_ms:.2f}"
                )

                # =========================
                # 3) 3D tylko z realnych detekcji YOLO
                # =========================
                selection_3D = []
                analyze_3D = []
                info_time_3d = {}
                valid_2d_points = 0
                valid_points3d = 0

                # try:
                #     if P_all is None:
                #         raise RuntimeError("Calibration not loaded")
                #
                #     pts_bat, conf_bat = build_3d_input_from_yolo_detections(
                #         detections_by_step=detections_by_step,
                #         roles_local=roles_local,
                #         O=configuration_est3D["O"],
                #     )
                #
                #     valid_2d_points = int(np.isfinite(pts_bat[..., 0]).sum())
                #
                #     if valid_2d_points == 0:
                #         print("[YOLO_3D] no valid YOLO 2D points -> skipping 3D")
                #     else:
                #         selection_3D, analyze_3D, info_time_3d = estiamte3D_work(
                #             DATA=pts_bat,
                #             configuration=configuration_est3D,
                #             P_all=P_all,
                #             K_all=K_all,
                #             dist_all=dist_all,
                #             C_all=C_all,
                #             pair_transforms=pair_transforms,
                #             work=False,
                #             conf_bat=conf_bat,
                #         )
                #
                #         valid_points3d = sum(
                #             1 for p in selection_3D
                #             if isinstance(p, (list, tuple)) and len(p) == 3
                #             and all(np.isfinite(v) for v in p)
                #         )

                #         print(
                #             f"[YOLO_3D] steps={len(detections_by_step)} | "
                #             f"valid_2d={valid_2d_points} | "
                #             f"points3D={valid_points3d} | "
                #             f"undistort={info_time_3d.get('undistort', 0):.3f} ms | "
                #             f"get3D={info_time_3d.get('get3D', 0):.3f} ms | "
                #             f"analyze_3D={info_time_3d.get('analyze_3D', 0):.3f} ms | "
                #             f"selection_3D={info_time_3d.get('selection_3D', 0):.3f} ms"
                #         )
                #
                # except Exception as e:
                #     print(f"[YOLO_3D] error: {e}")
                #     selection_3D = []
                #     analyze_3D = []
                #     info_time_3d = {}
                #     valid_points3d = 0

                # =========================
                # 4) trajektorie / wizualizacja
                # =========================
                try:
                    if yolo_vis_q is not None:
                        if yolo_vis_q.full():
                            try:
                                _ = yolo_vis_q.get_nowait()
                            except Exception:
                                pass

                        yolo_vis_q.put_nowait({
                            "batch_id": batch_counter,
                            "per_role_points": per_role_points,
                            "latest_centers_by_role": latest_centers_by_role,
                            "selection_3D": selection_3D,
                        })
                except Exception as e:
                    print(f"[YOLO_WORKER] yolo_vis_q put error: {e}")

                # =========================
                # 5) statystyki
                # =========================
            """
            return yolo_batch_worker()

        def yolo_batch_worker():
            import time
            import queue

            batch_counter = 0
            # Frequent summary updates for live benchmark sampling.
            report_every_s = 1.0
            sleep_when_off_s = 0.05
            last_report_t = time.time()

            # --- liczniki diagnostyczne całkowite ---
            batches_processed_total = 0
            frames_processed_total = 0
            frames_expected_total = 0

            # --- agregacja okresowa ---
            yolo_in_frames = 0
            yolo_valid_frames = 0
            yolo_proc_frames = 0
            yolo_detected = 0
            yolo_infer_ms_sum = 0.0
            yolo_infer_count = 0
            yolo_batches = 0

            # --- ostatnie znane wartości do raportowania, nawet gdy kolejka chwilowo pusta ---
            worker_timing_log_interval_s = 4.0
            last_worker_timing_log_t = 0.0
            last_batch_collect_ms = 0.0
            last_queue_wait_ms = 0.0
            last_prepare_ms = 0.0
            last_forward_ms = 0.0
            last_post_ms = 0.0
            last_infer_ms = 0.0
            last_infer_ms_wall = 0.0

            last_expected_frames = 0
            last_frames_valid = 0
            last_frames_processed = 0
            last_total_detected = 0
            last_detections_per_role = {}
            last_detections_per_step = []
            last_valid_2d_points = 0
            last_valid_points3d = 0
            last_steps_in_batch = 0
            last_images_in_batch = 0
            last_e2e_pipeline_ms = 0.0
            last_queue_wait_percent = 0.0
            last_prepare_percent = 0.0
            last_forward_percent = 0.0
            last_post_percent = 0.0
            worker_get_errors = 0
            worker_infer_errors = 0
            worker_vis_errors = 0
            worker_stale_skips = 0
            point_state_by_role = {}

            def pipeline_logs_enabled() -> bool:
                try:
                    return bool(shared_state.get("verbose_pipeline_logs", False)) if shared_state is not None else False
                except Exception:
                    return False

            def _tracker_cfg():
                cfg = {
                    "ema_alpha": 0.35,
                    "ttl_ms": 220.0,
                    "gate_px": 180.0,
                    "conf_on": 0.45,
                    "conf_off": 0.30,
                }
                if shared_state is None:
                    return cfg
                try:
                    cfg["ema_alpha"] = float(shared_state.get("yolo_point_ema_alpha", cfg["ema_alpha"]) or cfg["ema_alpha"])
                    cfg["ttl_ms"] = float(shared_state.get("yolo_point_ttl_ms", cfg["ttl_ms"]) or cfg["ttl_ms"])
                    cfg["gate_px"] = float(shared_state.get("yolo_point_gate_px", cfg["gate_px"]) or cfg["gate_px"])
                    cfg["conf_on"] = float(shared_state.get("yolo_point_conf_on", cfg["conf_on"]) or cfg["conf_on"])
                    cfg["conf_off"] = float(shared_state.get("yolo_point_conf_off", cfg["conf_off"]) or cfg["conf_off"])
                except Exception:
                    pass
                cfg["ema_alpha"] = max(0.01, min(1.0, cfg["ema_alpha"]))
                cfg["ttl_ms"] = max(30.0, cfg["ttl_ms"])
                cfg["gate_px"] = max(10.0, cfg["gate_px"])
                cfg["conf_on"] = max(0.0, min(1.0, cfg["conf_on"]))
                cfg["conf_off"] = max(0.0, min(cfg["conf_on"], cfg["conf_off"]))
                return cfg

            def _tracker_update(role_name, candidate, now_ts):
                cfg = _tracker_cfg()
                ttl_s = cfg["ttl_ms"] / 1000.0
                state = point_state_by_role.get(role_name)

                if candidate is not None:
                    min_conf = cfg["conf_off"] if state is not None else cfg["conf_on"]
                    if float(candidate.get("confidence", 0.0) or 0.0) < min_conf:
                        candidate = None

                if candidate is None:
                    if state is None:
                        return None
                    age_s = now_ts - float(state.get("last_seen_ts", 0.0) or 0.0)
                    if age_s > ttl_s:
                        point_state_by_role.pop(role_name, None)
                        return None
                    return {
                        "x": float(state["x"]),
                        "y": float(state["y"]),
                        "confidence": float(state.get("confidence", 0.0) or 0.0),
                        "src_w": int(state.get("src_w", 1920) or 1920),
                        "src_h": int(state.get("src_h", 1080) or 1080),
                        "fresh": False,
                        "age_ms": float(age_s * 1000.0),
                    }

                cx = float(candidate.get("x", 0.0) or 0.0)
                cy = float(candidate.get("y", 0.0) or 0.0)
                conf = float(candidate.get("confidence", 0.0) or 0.0)
                src_w = int(candidate.get("src_w", 1920) or 1920)
                src_h = int(candidate.get("src_h", 1080) or 1080)

                if state is not None:
                    dx = cx - float(state.get("x", cx))
                    dy = cy - float(state.get("y", cy))
                    dist = float((dx * dx + dy * dy) ** 0.5)
                    age_s = now_ts - float(state.get("last_seen_ts", 0.0) or 0.0)
                    if dist > cfg["gate_px"] and age_s <= ttl_s:
                        return {
                            "x": float(state["x"]),
                            "y": float(state["y"]),
                            "confidence": float(state.get("confidence", 0.0) or 0.0),
                            "src_w": int(state.get("src_w", src_w) or src_w),
                            "src_h": int(state.get("src_h", src_h) or src_h),
                            "fresh": False,
                            "age_ms": float(age_s * 1000.0),
                        }
                    a = cfg["ema_alpha"]
                    cx = (1.0 - a) * float(state.get("x", cx)) + a * cx
                    cy = (1.0 - a) * float(state.get("y", cy)) + a * cy

                point_state_by_role[role_name] = {
                    "x": float(cx),
                    "y": float(cy),
                    "confidence": float(conf),
                    "src_w": int(src_w),
                    "src_h": int(src_h),
                    "last_seen_ts": float(now_ts),
                }
                return {
                    "x": float(cx),
                    "y": float(cy),
                    "confidence": float(conf),
                    "src_w": int(src_w),
                    "src_h": int(src_h),
                    "fresh": True,
                    "age_ms": 0.0,
                }

            def emit_periodic_summary(now_ts: float) -> None:
                nonlocal yolo_in_frames, yolo_valid_frames, yolo_proc_frames, yolo_detected
                nonlocal yolo_infer_ms_sum, yolo_infer_count, yolo_batches, last_report_t

                if (now_ts - last_report_t) < report_every_s:
                    return
                if yolo_in_frames <= 0 and yolo_proc_frames <= 0:
                    return

                avg_infer_ms = (yolo_infer_ms_sum / yolo_infer_count) if yolo_infer_count > 0 else 0.0
                fps = 1000.0 / avg_infer_ms if avg_infer_ms > 0 else 0.0

                try:
                    current_yolo_q = yolo_batch_q.qsize()
                except Exception:
                    current_yolo_q = -1

                drop_gap_frames = frames_expected_total - frames_processed_total
                num_roles = len(last_detections_per_role) if last_detections_per_role else 4
                batches_per_sec = 1000.0 / avg_infer_ms if avg_infer_ms > 0 else 0.0
                images_per_sec = (yolo_proc_frames / yolo_infer_ms_sum) * 1000.0 if yolo_infer_ms_sum > 0 else 0.0
                steps_per_sec = images_per_sec / num_roles if num_roles > 0 else 0.0

                try:
                    target_steps_per_sec = float(shared_state.get("yolo_target_hz", 50.0)) if shared_state is not None else 50.0
                except Exception:
                    target_steps_per_sec = 50.0

                target_util_percent = (steps_per_sec / target_steps_per_sec * 100.0) if target_steps_per_sec > 0 else 0.0
                last_images_per_sec = (last_frames_processed / last_infer_ms) * 1000.0 if last_infer_ms > 0 else 0.0
                last_steps_per_sec = last_images_per_sec / num_roles if num_roles > 0 else 0.0

                configured_batch_size = int(shared_state.get("yolo_batch_size", 0)) if shared_state is not None else 0
                configured_seq_len = int(shared_state.get("yolo_seq_len", 0)) if shared_state is not None else 0
                configured_input_batch_images = int(shared_state.get("yolo_input_batch_images", last_expected_frames)) if shared_state is not None else int(last_expected_frames)
                last_batch_size_effective = int(min(last_expected_frames, configured_batch_size)) if configured_batch_size > 0 else int(last_expected_frames)

                payload = {
                    "batch_id": int(batch_counter),
                    "batches": int(yolo_batches),
                    "batches_processed_total": int(batches_processed_total),
                    "frames_expected_total": int(frames_expected_total),
                    "frames_processed_total": int(frames_processed_total),
                    "drop_gap_frames": int(drop_gap_frames),
                    "yolo_q_size": int(current_yolo_q),
                    "in_frames": int(yolo_in_frames),
                    "valid_frames": int(yolo_valid_frames),
                    "proc_frames": int(yolo_proc_frames),
                    "detected": int(yolo_detected),
                    "valid_percent": float((yolo_valid_frames / yolo_in_frames * 100.0) if yolo_in_frames > 0 else 0.0),
                    "proc_percent": float((yolo_proc_frames / yolo_in_frames * 100.0) if yolo_in_frames > 0 else 0.0),
                    "avg_infer_ms": float(avg_infer_ms),
                    "fps": float(fps),
                    "batches_per_sec": float(batches_per_sec),
                    "images_per_sec": float(images_per_sec),
                    "steps_per_sec": float(steps_per_sec),
                    "target_steps_per_sec": float(target_steps_per_sec),
                    "target_util_percent": float(target_util_percent),
                    "infer_ms_wall": round(last_infer_ms_wall, 2),
                    "batch_collect_ms": round(last_batch_collect_ms, 2),
                    "queue_wait_ms": round(last_queue_wait_ms, 2),
                    "prepare_ms": round(last_prepare_ms, 2),
                    "forward_ms": round(last_forward_ms, 2),
                    "post_ms": round(last_post_ms, 2),
                    "e2e_pipeline_ms": round(last_e2e_pipeline_ms, 2),
                    "queue_wait_percent": round(last_queue_wait_percent, 2),
                    "prepare_percent": round(last_prepare_percent, 2),
                    "forward_percent": round(last_forward_percent, 2),
                    "post_percent": round(last_post_percent, 2),
                    "last_images_per_sec": float(last_images_per_sec),
                    "last_steps_per_sec": float(last_steps_per_sec),
                    "last_steps_in_batch": int(last_steps_in_batch),
                    "last_images_in_batch": int(last_images_in_batch),
                    "last_batch_size_effective": int(last_batch_size_effective),
                    "last_expected_frames": int(last_expected_frames),
                    "last_frames_valid": int(last_frames_valid),
                    "last_frames_processed": int(last_frames_processed),
                    "last_total_detected": int(last_total_detected),
                    "per_role_counts": dict(last_detections_per_role),
                    "per_step_counts": list(last_detections_per_step),
                    "configured_input_batch_images": int(configured_input_batch_images),
                    "configured_batch_size": int(configured_batch_size),
                    "configured_seq_len": int(configured_seq_len),
                    "valid_2d_points": int(last_valid_2d_points),
                    "points3D": int(last_valid_points3d),
                    "worker_get_errors": int(worker_get_errors),
                    "worker_infer_errors": int(worker_infer_errors),
                    "worker_vis_errors": int(worker_vis_errors),
                    "worker_stale_skips": int(worker_stale_skips),
                }

                try:
                    if shared_state is not None:
                        shared_state["yolo_buffer_stats"] = dict(payload, updated_at=time.time())
                except Exception as e:
                    if pipeline_logs_enabled():
                        print(f"[YOLO_WORKER] shared_state yolo_buffer_stats update failed: {e}")

                try:
                    stats_q.put_nowait(dict(payload, type="yolo_summary"))
                except Exception:
                    pass

                if pipeline_logs_enabled():
                    print(
                        f"[YOLO_WORKER_STATS] "
                        f"batches_processed_total={batches_processed_total} "
                        f"frames_expected_total={frames_expected_total} "
                        f"frames_processed_total={frames_processed_total} "
                        f"drop_gap_frames={drop_gap_frames} "
                        f"yolo_q={current_yolo_q} "
                        f"avg_infer_ms={avg_infer_ms:.2f}"
                    )

                yolo_in_frames = 0
                yolo_valid_frames = 0
                yolo_proc_frames = 0
                yolo_detected = 0
                yolo_infer_ms_sum = 0.0
                yolo_infer_count = 0
                yolo_batches = 0
                last_report_t = now_ts

            # # kalibracja 3D ładujemy raz
            # try:
            #     (
            #         _cfg_json,
            #         P_all,
            #         K_all,
            #         dist_all,
            #         C_all,
            #         pair_transforms,
            #         _conf_json,
            #         _pts_json,
            #     ) = p3D.load_data_from_json("dane_testowe_3D_fixed.json", as_numpy=True)
            #     print("[YOLO_3D] ??? Calibration/config loaded once")
            # except Exception as e:
            #     print(f"[YOLO_3D]  calibration load error: {e}")
            #     P_all = K_all = dist_all = C_all = pair_transforms = None

            while not stop_evt.is_set():
                try:
                    yolo_on = bool(is_yolo_enabled())
                except Exception:
                    yolo_on = False

                if not yolo_on:
                    time.sleep(sleep_when_off_s)
                    continue

                # =========================================================
                # 0) Pobranie batcha
                # =========================================================
                try:
                    item = yolo_batch_q.get(timeout=0.2)

                    worker_start_ts = time.time()
                    enqueue_ts = float(item.get("enqueue_ts", worker_start_ts) or worker_start_ts)
                    queue_wait_ms = (worker_start_ts - enqueue_ts) * 1000.0
                    batch_collect_ms = float(item.get("batch_collect_ms", 0.0) or 0.0)

                except queue.Empty:
                    emit_periodic_summary(time.time())
                    continue

                except Exception as e:
                    if pipeline_logs_enabled():
                        print(f"[YOLO_WORKER] get error: {e}")
                    continue

                if not isinstance(item, dict):
                    continue

                try:
                    max_queue_wait_ms = float(shared_state.get("yolo_max_queue_wait_ms", 0.0)) if shared_state is not None else 0.0
                except Exception:
                    max_queue_wait_ms = 0.0
                if max_queue_wait_ms > 0.0 and queue_wait_ms > max_queue_wait_ms:
                    worker_stale_skips += 1
                    if pipeline_logs_enabled() and worker_stale_skips % 20 == 0:
                        print(
                            f"[YOLO_WORKER] stale skip: queue_wait_ms={queue_wait_ms:.2f} > {max_queue_wait_ms:.2f} "
                            f"(skips={worker_stale_skips})"
                        )
                    continue

                frames_by_step = item.get("frames_by_step", [])
                roles_local = item.get("roles", [])
                ts_by_step = item.get("ts_by_step", [])

                if not frames_by_step or not roles_local:
                    continue

                batch_counter += 1
                yolo_batches += 1
                batches_processed_total += 1

                expected_frames = len(frames_by_step) * len(roles_local)
                frames_expected_total += int(expected_frames)

                # domyślne wartości na wypadek błędu
                detections_by_step = []
                infer_ms = 0.0
                infer_ms_wall = 0.0
                frames_valid = 0
                frames_processed = 0
                prepare_ms = 0.0
                forward_ms = 0.0
                post_ms = 0.0

                total_detected = 0
                detections_per_role = {r: 0 for r in roles_local}
                detections_per_step = []
                per_role_points = {r: [] for r in roles_local}
                latest_centers_by_role: dict[str, dict | None] = {r: None for r in roles_local}
                raw_best_by_role: dict[str, dict | None] = {r: None for r in roles_local}
                latest_step_idx_by_role = {r: -1 for r in roles_local}
                raw_points_latest_step_by_role = {r: [] for r in roles_local}

                selection_3D = []
                valid_2d_points = 0
                valid_points3d = 0

                # =========================================================
                # 1) YOLO inference
                # =========================================================
                try:
                    t_infer0 = time.time()
                    result = process_batch_with_yolo(frames_by_step, roles_local)
                    t_infer1 = time.time()
                    infer_ms_wall = (t_infer1 - t_infer0) * 1000.0

                    detections_by_step = result.get("detections", [])
                    meta = result.get("meta", {})

                    infer_ms_meta = meta.get("infer_ms", None)
                    infer_ms = float(infer_ms_meta) if infer_ms_meta is not None else float(infer_ms_wall)

                    frames_valid = int(meta.get("frames_valid", expected_frames))
                    frames_processed = int(meta.get("frames_processed", expected_frames))
                    prepare_ms = float(meta.get("prepare_ms", 0.0) or 0.0)
                    forward_ms = float(meta.get("forward_ms", 0.0) or 0.0)
                    post_ms = float(meta.get("post_ms", 0.0) or 0.0)
                    try:
                        ts_flat = []
                        per_step_span_ms = []
                        if isinstance(ts_by_step, list):
                            for step_ts in ts_by_step:
                                if isinstance(step_ts, (list, tuple)) and step_ts:
                                    vals = [int(x) for x in step_ts if x is not None]
                                    if vals:
                                        ts_flat.extend(vals)
                                        per_step_span_ms.append((max(vals) - min(vals)) / 1e6)
                        camera_timestamps_last_ns = {}
                        try:
                            if ts_by_step and roles_local:
                                last_step_ts = ts_by_step[-1] if isinstance(ts_by_step[-1], (list, tuple)) else []
                                for ridx, rname in enumerate(roles_local):
                                    if ridx < len(last_step_ts):
                                        camera_timestamps_last_ns[str(rname)] = int(last_step_ts[ridx])
                        except Exception:
                            camera_timestamps_last_ns = {}
                        ts_snapshot = {
                            "updated_at": time.time(),
                            "steps_in_batch": int(len(ts_by_step) if isinstance(ts_by_step, list) else 0),
                            "roles_count": int(len(roles_local)),
                            "camera_timestamps_last_ns": camera_timestamps_last_ns,
                            "source_ts_min_ns": int(min(ts_flat)) if ts_flat else None,
                            "source_ts_max_ns": int(max(ts_flat)) if ts_flat else None,
                            "source_ts_span_ms": float(((max(ts_flat) - min(ts_flat)) / 1e6) if ts_flat else 0.0),
                            "per_step_span_ms": [float(x) for x in per_step_span_ms],
                            "per_step_span_ms_avg": float(sum(per_step_span_ms) / len(per_step_span_ms)) if per_step_span_ms else 0.0,
                            "batch_collect_ms": float(batch_collect_ms),
                            "queue_wait_ms": float(queue_wait_ms),
                            "prepare_ms": float(prepare_ms),
                            "forward_ms": float(forward_ms),
                            "post_ms": float(post_ms),
                            "infer_ms": float(infer_ms),
                            "frames_expected": int(expected_frames),
                            "frames_valid": int(frames_valid),
                            "frames_processed": int(frames_processed),
                            "configured_image_size": int(shared_state.get("yolo_image_size", 0)) if shared_state is not None else 0,
                            "configured_input_batch_images": int(shared_state.get("yolo_input_batch_images", 0)) if shared_state is not None else 0,
                            "configured_seq_len": int(shared_state.get("yolo_seq_len", 0)) if shared_state is not None else 0,
                        }
                        if shared_state is not None:
                            shared_state["yolo_last_batch_timestamps"] = ts_snapshot
                    except Exception:
                        pass

                    _timing_now = time.time()
                    if pipeline_logs_enabled() and (_timing_now - last_worker_timing_log_t) >= worker_timing_log_interval_s:
                        print(
                            f"[YOLO_WORKER_TIMING] "
                            f"batch_collect={batch_collect_ms:.2f} ms | "
                            f"queue_wait={queue_wait_ms:.2f} ms | "
                            f"prepare={prepare_ms:.2f} ms | "
                            f"forward={forward_ms:.2f} ms | "
                            f"post={post_ms:.2f} ms | "
                            f"total={infer_ms:.2f} ms"
                        )
                        last_worker_timing_log_t = _timing_now

                    yolo_in_frames += int(expected_frames)
                    yolo_valid_frames += frames_valid
                    yolo_proc_frames += frames_processed
                    yolo_infer_ms_sum += infer_ms
                    yolo_infer_count += 1
                    frames_processed_total += int(frames_processed)
                    e2e_pipeline_ms = max(0.0, float(batch_collect_ms + queue_wait_ms + infer_ms))
                    last_e2e_pipeline_ms = float(e2e_pipeline_ms)
                    denom = max(1e-9, float(last_e2e_pipeline_ms))
                    last_queue_wait_percent = float(queue_wait_ms / denom * 100.0)
                    last_prepare_percent = float(prepare_ms / denom * 100.0)
                    last_forward_percent = float(forward_ms / denom * 100.0)
                    last_post_percent = float(post_ms / denom * 100.0)

                except Exception as e:
                    worker_infer_errors += 1
                    if pipeline_logs_enabled():
                        print(f"[YOLO_WORKER] inference error: {e}")
                    continue

                # =========================================================
                # 2) Zlicz detekcje YOLO
                # =========================================================
                for step_idx, det_step in enumerate(detections_by_step):
                    step_info = {}

                    for r in roles_local:
                        dets = det_step.get(r, [])
                        n = len(dets)

                        step_info[r] = n
                        detections_per_role[r] += n
                        total_detected += n

                        for det in dets:
                            per_role_points[r].append({
                                "x": det["center_x"],
                                "y": det["center_y"],
                                "confidence": det["confidence"],
                                "step_idx": step_idx,
                            })
                        if dets:
                            if step_idx > int(latest_step_idx_by_role.get(r, -1)):
                                latest_step_idx_by_role[r] = int(step_idx)
                                raw_points_latest_step_by_role[r] = []
                            if step_idx == int(latest_step_idx_by_role.get(r, -1)):
                                raw_points_latest_step_by_role[r] = [
                                    {
                                        "x": float(d.get("center_x", 0.0)),
                                        "y": float(d.get("center_y", 0.0)),
                                        "confidence": float(d.get("confidence", 0.0)),
                                        "bx": float(d.get("x", 0.0)),
                                        "by": float(d.get("y", 0.0)),
                                        "bw": float(d.get("width", 0.0)),
                                        "bh": float(d.get("height", 0.0)),
                                        "src_w": int(d.get("frame_w", 1920) or 1920),
                                        "src_h": int(d.get("frame_h", 1080) or 1080),
                                    }
                                    for d in dets
                                ]

                        if dets:
                            best_det = max(dets, key=lambda d: float(d.get("confidence", 0.0)))
                            raw_candidate = {
                                "x": float(best_det.get("center_x", 0.0)),
                                "y": float(best_det.get("center_y", 0.0)),
                                "confidence": float(best_det.get("confidence", 0.0)),
                                "src_w": int(best_det.get("frame_w", 1920) or 1920),
                                "src_h": int(best_det.get("frame_h", 1080) or 1080),
                                "step_idx": int(step_idx),
                            }
                            prev = raw_best_by_role.get(r)
                            if (
                                prev is None
                                or int(raw_candidate["step_idx"]) > int(prev.get("step_idx", -1))
                                or (
                                    int(raw_candidate["step_idx"]) == int(prev.get("step_idx", -1))
                                    and float(raw_candidate["confidence"]) >= float(prev.get("confidence", 0.0))
                                )
                            ):
                                raw_best_by_role[r] = raw_candidate

                    detections_per_step.append(step_info)

                yolo_detected += int(total_detected)

                now_track_ts = time.time()
                for r in roles_local:
                    latest_centers_by_role[r] = _tracker_update(r, raw_best_by_role.get(r), now_track_ts)
                try:
                    if shared_state is not None:
                        shared_state["yolo_overlay_points"] = dict(latest_centers_by_role)
                        points_all = {r: list(raw_points_latest_step_by_role.get(r, [])) for r in roles_local}
                        shared_state["yolo_overlay_points_all"] = points_all
                        shared_state["yolo_overlay_counts_latest_step"] = {
                            r: int(len(points_all.get(r, []))) for r in roles_local
                        }
                except Exception:
                    pass

                # Snapshot ostatniego wyniku inferencji do raportu save_buffer
                try:
                    if shared_state is not None:
                        flat_detections = []
                        for step_idx, det_step in enumerate(detections_by_step):
                            if not isinstance(det_step, dict):
                                continue
                            for role_name in roles_local:
                                dets = det_step.get(role_name, [])
                                if not isinstance(dets, list):
                                    continue
                                for d in dets[:8]:
                                    flat_detections.append({
                                        "step_idx": int(step_idx),
                                        "role": str(role_name),
                                        "center_x": int(d.get("center_x", 0) or 0),
                                        "center_y": int(d.get("center_y", 0) or 0),
                                        "x": int(d.get("x", 0) or 0),
                                        "y": int(d.get("y", 0) or 0),
                                        "width": int(d.get("width", 0) or 0),
                                        "height": int(d.get("height", 0) or 0),
                                        "confidence": float(d.get("confidence", 0.0) or 0.0),
                                        "frame_w": int(d.get("frame_w", 0) or 0),
                                        "frame_h": int(d.get("frame_h", 0) or 0),
                                    })

                        shared_state["yolo_last_inference_snapshot"] = {
                            "updated_at": time.time(),
                            "batch_id": int(batch_counter),
                            "steps_in_batch": int(len(frames_by_step)),
                            "roles": list(roles_local),
                            "expected_frames": int(expected_frames),
                            "frames_valid": int(frames_valid),
                            "frames_processed": int(frames_processed),
                            "objects_detected_total": int(total_detected),
                            "detections_per_role": dict(detections_per_role),
                            "detections_per_step": list(detections_per_step),
                            "latest_centers_by_role": dict(latest_centers_by_role),
                            "detections_flat": flat_detections,
                            "timings_ms": {
                                "batch_collect_ms": float(batch_collect_ms),
                                "queue_wait_ms": float(queue_wait_ms),
                                "prepare_ms": float(prepare_ms),
                                "forward_ms": float(forward_ms),
                                "post_ms": float(post_ms),
                                "infer_ms": float(infer_ms),
                            },
                        }
                except Exception:
                    pass

                # Pełny trace per-step (dla porównań całego okna bufora)
                try:
                    step_timings = {
                        "batch_collect_ms": float(batch_collect_ms),
                        "queue_wait_ms": float(queue_wait_ms),
                        "prepare_ms": float(prepare_ms),
                        "forward_ms": float(forward_ms),
                        "post_ms": float(post_ms),
                        "infer_ms": float(infer_ms),
                    }
                    for step_idx, det_step in enumerate(detections_by_step):
                        step_ts_map = {}
                        try:
                            if step_idx < len(ts_by_step):
                                step_ts = ts_by_step[step_idx]
                                if isinstance(step_ts, (list, tuple)):
                                    for ridx, rname in enumerate(roles_local):
                                        if ridx < len(step_ts):
                                            step_ts_map[str(rname)] = int(step_ts[ridx])
                        except Exception:
                            step_ts_map = {}

                        dets_flat = []
                        if isinstance(det_step, dict):
                            for role_name in roles_local:
                                dets = det_step.get(role_name, [])
                                if not isinstance(dets, list):
                                    continue
                                for d in dets:
                                    dets_flat.append({
                                        "step_idx": int(step_idx),
                                        "role": str(role_name),
                                        "center_x": int(d.get("center_x", 0) or 0),
                                        "center_y": int(d.get("center_y", 0) or 0),
                                        "x": int(d.get("x", 0) or 0),
                                        "y": int(d.get("y", 0) or 0),
                                        "width": int(d.get("width", 0) or 0),
                                        "height": int(d.get("height", 0) or 0),
                                        "confidence": float(d.get("confidence", 0.0) or 0.0),
                                        "frame_w": int(d.get("frame_w", 0) or 0),
                                        "frame_h": int(d.get("frame_h", 0) or 0),
                                    })

                        push_yolo_step_trace({
                            "trace_unix_ns": int(time.time_ns()),
                            "batch_id": int(batch_counter),
                            "step_idx_in_batch": int(step_idx),
                            "roles": list(roles_local),
                            "camera_timestamps_ns": step_ts_map,
                            "detections_flat": dets_flat,
                            "timings_ms": dict(step_timings),
                            "configured_image_size": int(shared_state.get("yolo_image_size", 0)) if shared_state is not None else 0,
                            "configured_input_batch_images": int(shared_state.get("yolo_input_batch_images", 0)) if shared_state is not None else 0,
                            "configured_seq_len": int(shared_state.get("yolo_seq_len", 0)) if shared_state is not None else 0,
                        })
                except Exception:
                    pass

                # =========================================================
                # 3) 3D tylko z realnych detekcji YOLO
                # =========================================================
                # try:
                #     if P_all is None:
                #         raise RuntimeError("Calibration not loaded")
                #
                #     pts_bat, conf_bat = build_3d_input_from_yolo_detections(
                #         detections_by_step=detections_by_step,
                #         roles_local=roles_local,
                #         O=configuration_est3D["O"],
                #     )
                #
                #     valid_2d_points = int(np.isfinite(pts_bat[..., 0]).sum())
                #
                #     if valid_2d_points == 0:
                #         print("[YOLO_3D] no valid YOLO 2D points -> skipping 3D")
                #     else:
                #         selection_3D, analyze_3D, info_time_3d = estiamte3D_work(
                #             DATA=pts_bat,
                #             configuration=configuration_est3D,
                #             P_all=P_all,
                #             K_all=K_all,
                #             dist_all=dist_all,
                #             C_all=C_all,
                #             pair_transforms=pair_transforms,
                #             work=False,
                #             conf_bat=conf_bat,
                #         )
                #
                #         valid_points3d = sum(
                #             1 for p in selection_3D
                #             if isinstance(p, (list, tuple)) and len(p) == 3
                #             and all(np.isfinite(v) for v in p)
                #         )
                #
                # except Exception as e:
                #     print(f"[YOLO_3D] error: {e}")
                #     selection_3D = []
                #     analyze_3D = []
                #     info_time_3d = {}
                #     valid_points3d = 0

                # =========================================================
                # 4) trajektorie / wizualizacja
                # =========================================================
                try:
                    if yolo_vis_q is not None:
                        if yolo_vis_q.full():
                            try:
                                _ = yolo_vis_q.get_nowait()
                            except Exception:
                                pass

                        yolo_vis_q.put_nowait({
                            "batch_id": batch_counter,
                            "per_role_points": per_role_points,
                            "latest_centers_by_role": latest_centers_by_role,
                            "selection_3D": selection_3D,
                        })
                except Exception as e:
                    worker_vis_errors += 1
                    if pipeline_logs_enabled():
                        print(f"[YOLO_WORKER] yolo_vis_q put error: {e}")

                # =========================================================
                # 5) zapis ostatnich statystyk
                # =========================================================
                last_batch_collect_ms = batch_collect_ms
                last_queue_wait_ms = queue_wait_ms
                last_prepare_ms = prepare_ms
                last_forward_ms = forward_ms
                last_post_ms = post_ms
                last_infer_ms = infer_ms
                last_infer_ms_wall = infer_ms_wall

                last_expected_frames = expected_frames
                last_frames_valid = frames_valid
                last_frames_processed = frames_processed
                last_total_detected = total_detected
                last_detections_per_role = dict(detections_per_role)
                last_detections_per_step = list(detections_per_step)
                last_valid_2d_points = valid_2d_points
                last_valid_points3d = valid_points3d
                last_steps_in_batch = len(frames_by_step)
                last_images_in_batch = expected_frames
                emit_periodic_summary(time.time())

        threads = []
        # --------------------------------------------------------
        # Saver startuje dopiero po rozpoczęciu nagrywania
        # --------------------------------------------------------
        def start_saver_threads():
            """Startuje saver_worker_bin dla każdej roli – tylko raz, gdy nagrywanie ruszy."""
            import core.utils_config as utils_config
            from capture.grabber_module import saver_worker_bin
            from core.utils_config import BIN_BATCH_FRAMES, BIN_ROLL_EVERY

            def _root_dir():
                return getattr(utils_config, "RAW_DIR", None)

            runtime = getattr(utils_config, "_RUNTIME_PROCS", {})
            saver_threads = runtime.get("saver_threads")
            if saver_threads is None:
                saver_threads = {}
                runtime["saver_threads"] = saver_threads
                utils_config._RUNTIME_PROCS = runtime

            for role in roles:
                t = saver_threads.get(role)
                if t is not None and t.is_alive():
                    continue  # już działa

                in_q = per_role_q.get(role)
                if in_q is None:
                    continue  # backend jeszcze nie gotowy

                th = threading.Thread(
                    target=saver_worker_bin,
                    kwargs=dict(
                        role=role,
                        in_q=in_q,
                        root_dir=_root_dir,
                        batch_frames=BIN_BATCH_FRAMES,
                        roll_every=BIN_ROLL_EVERY,
                        recording_event=recording_event,
                        stop_evt=stop_evt,
                        stats_q=stats_q,   # ważne: statystyki trafiają tu
                    ),
                    daemon=True,
                    name=f"saver_{role}",
                )
                th.start()
                saver_threads[role] = th
                print(f"[SAVER] ▶️uruchomiono saver_worker dla {role}")

        def monitor_record_event():
            """Czeka aż recording_event zostanie ustawiony i wtedy startuje savery."""
            last_state = False
            while not stop_evt.is_set():
                now = recording_event.is_set()
                if now and not last_state:
                    print("[SAVER] ▶️wykryto uruchomienie nagrywania – start saver_worker_bin")
                    start_saver_threads()
                last_state = now
                time.sleep(0.1)

        # Start obserwatora nagrywania (osobny wątek, nie w liście threads)
        threading.Thread(target=monitor_record_event, daemon=True).start()
        print("[MAIN] monitor_record_event uruchomiony")
        threading.Thread(
            target=raw_fanout_router,
            name="raw_fanout_router",
            daemon=True,
        ).start()
        print("[MAIN] raw_fanout_router uruchomiony")

        threading.Thread(
            target=raw_router_save,
            name="raw_router_save",
            daemon=True,
        ).start()
        print("[MAIN] raw_router_save uruchomiony")

        threading.Thread(
            target=router_yolo_from_processed,
            name="router_yolo_from_processed",
            daemon=True,
        ).start()
        print("[MAIN] router_yolo_from_processed uruchomiony")

        threading.Thread(
            target=yolo_batch_worker,
            name="yolo_batch_worker",
            daemon=True,
        ).start()
        print("[MAIN] yolo_batch_worker uruchomiony")

        utils_config._RUNTIME_PROCS = {
            "stop_evt": stop_evt,
            "proc_color": {"universal": universal_proc},
            "per_role_q": per_role_q,
            "universal_bayer_q": universal_bayer_q,
            "save_raw_q": save_raw_q,

            "yolo_batch_q": yolo_batch_q,
            "saver_threads": {},
            "grabber_threads": grabber_threads,
        }

        # === WATCHDOG HOT-PLUG: znika / pojawia się numer seryjny ===
        def camera_hotplug_watchdog():

            def wait_for_single_ptp_ready(cam, role, max_wait=20.0, poll_s=0.5):
                deadline = time.time() + max_wait

                while time.time() < deadline and not stop_evt.is_set():
                    try:
                        status = None
                        servo = None

                        try:
                            status = cam.PtpStatus.GetValue()
                        except Exception:
                            try:
                                status = cam.BslPtpStatus.GetValue()
                            except Exception:
                                status = None

                        try:
                            servo = cam.PtpServoStatus.GetValue()
                        except Exception:
                            try:
                                servo = cam.BslPtpServoStatus.GetValue()
                            except Exception:
                                servo = None

                        print(f"[WATCHDOG][PTP] {role}: status={status} servo={servo}")

                        # dopuszczamy Slave/Master; servo bywa Unknown chwilowo
                        if status in ("Slave", "Master"):
                            return True

                    except Exception:
                        pass

                    time.sleep(poll_s)

                return False

            if simulate:
                # w trybie symulacji nie ma sensu
                return
            if pylon is None:
                return

            CHECK_EVERY_S = 5.0

            try:
                tl_local = pylon.TlFactory.GetInstance()
            except Exception as e:
                try:
                    stats_q.put(f"[WATCHDOG]  TlFactory.GetInstance() failed: {e}")
                except Exception:
                    print(f"[WATCHDOG] TlFactory failed: {e}")
                return

            # mapa statusów kamer do pokazania w GUI
            import multiprocessing as mp
            import core.utils_config as utils_config

            try:
                if "cam_status" not in shared_state:
                    try:
                        # jeśli utils_config._MANAGER istnieje — użyj go
                        manager = getattr(utils_config, "_MANAGER", None)
                        if manager is None:
                            manager = mp.Manager()
                            setattr(utils_config, "_MANAGER", manager)
                        shared_state["cam_status"] = manager.dict()
                    except Exception as e:
                        print(f"[WATCHDOG] ⚠️ Nie udało się utworzyć manager.dict(): {e}")
                        shared_state["cam_status"] = {}
                cam_status = shared_state.get("cam_status", {})
            except Exception as e:
                print(f"[WATCHDOG] ⚠️ cam_status init fail: {e}")
                cam_status = {}

            # pamiętamy, które seriale były już zgubione
            missing_serials = set()
            missing_counts = {}
            MISSING_THRESHOLD = 3

            while not stop_evt.is_set():
                time.sleep(CHECK_EVERY_S)

                # 1) aktualna lista urządzeń
                try:
                    devs_now = tl_local.EnumerateDevices()
                    if not devs_now:
                        msg = "[WATCHDOG] ⚠️ EnumerateDevices zwróciło pustą listę."
                        try:
                            stats_q.put(msg)
                        except Exception:
                            print(msg)
                        continue

                    online_serials = {d.GetSerialNumber() for d in devs_now}
                except Exception as e:
                    msg = f"[WATCHDOG] ⚠️ EnumerateDevices failed: {e}"
                    try:
                        stats_q.put(msg)
                    except Exception:
                        print(msg)
                    continue

                # 2) przechodzimy po znanych rolach / serialach
                for idx, (role, serial) in enumerate(zip(roles, serials)):
                    if not serial:
                        continue

                    # --- SERIAL ZNIKNĄŁ Z SYSTEMU ---
                    if serial not in online_serials:
                        missing_counts[serial] = missing_counts.get(serial, 0) + 1

                        if missing_counts[serial] < MISSING_THRESHOLD:
                            if cam_status is not None:
                                cam_status[role] = "unstable"
                            continue

                        if serial not in missing_serials:
                            missing_serials.add(serial)
                            msg = f"[WATCHDOG] ⚠️ Kamera {role} (S/N {serial}) zniknęła z systemu."
                            try:
                                stats_q.put(msg)
                            except Exception:
                                print(msg)

                        if cam_status is not None:
                            cam_status[role] = "disconnected"

                        try:
                            cam = cams[idx]
                            if cam is not None:
                                try:
                                    if cam.IsGrabbing():
                                        cam.StopGrabbing()
                                except Exception:
                                    pass
                                try:
                                    if cam.IsOpen():
                                        cam.Close()
                                except Exception:
                                    pass
                        except Exception:
                            pass

                        cams[idx] = None
                        try:
                            grabber_threads.pop(role, None)
                        except Exception:
                            pass
                        continue
                    else:
                        missing_counts[serial] = 0

                    # --- SERIAL JEST ONLINE ---
                    cam = None
                    try:
                        cam = cams[idx]
                    except Exception:
                        cam = None

                    if cam is not None:
                        try:
                            if cam.IsOpen():
                                if cam_status is not None:
                                    cam_status[role] = "ok"
                                missing_serials.discard(serial)
                                continue
                        except Exception:
                            pass

                    if cam_status is not None:
                        cam_status[role] = "reconnecting"

                    dev = None
                    for di in devs_now:
                        if di.GetSerialNumber() == serial:
                            dev = di
                            break
                    if dev is None:
                        continue

                    try:
                        new_cam = pylon.InstantCamera(tl_local.CreateDevice(dev))
                        new_cam.Open()
                        time.sleep(0.3)

                        try:
                            cam_config(serial, new_cam,stats_q)

                        except Exception as e:
                            msg = f"[WATCHDOG] ⚠️ cam_config retry dla {role} ({serial}) nieudany: {e}"
                            try:
                                stats_q.put(msg)
                            except Exception:
                                print(msg)

                        cam_status[role] = "reconnecting"
                        params = shared_state.get("initial_camera_params", {}) if shared_state else {}
                        per_role = params.get("per_role", {})
                        per_cam = params.get("per_camera", {})

                        exp_cfg = per_cam.get(serial, {}).get("exposure") \
                                  or per_role.get(role, {}).get("exposure") \
                                  or params.get("exposure_val")
                        if exp_cfg is not None:
                            try:
                                try:
                                    mn, mx = new_cam.ExposureTime.GetMin(), new_cam.ExposureTime.GetMax()
                                except Exception:
                                    mn, mx = 10.0, 1_000_000.0
                                new_cam.ExposureTime.SetValue(max(mn, min(mx, float(exp_cfg))))
                            except Exception as e:
                                try:
                                    stats_q.put(f"[WATCHDOG] ⚠️ ExposureTime(reconnect)={exp_cfg} fail: {e}")
                                except Exception:
                                    print("[WATCHDOG] ExposureTime(reconnect) fail:", e)


                        gain_cfg = per_cam.get(serial, {}).get("gain") \
                                   or per_role.get(role, {}).get("gain") \
                                   or params.get("gain_val")
                        if gain_cfg is not None:
                            try:
                                try:
                                    mn, mx = new_cam.Gain.GetMin(), new_cam.Gain.GetMax()
                                except Exception:
                                    mn, mx = 0.0, 24.0
                                new_cam.Gain.SetValue(max(mn, min(mx, float(gain_cfg))))
                            except Exception as e:
                                try:
                                    stats_q.put(f"[WATCHDOG] ⚠️ Gain(reconnect)={gain_cfg} fail: {e}")
                                except Exception:
                                    print("[WATCHDOG] Gain(reconnect) fail:", e)

                        cams[idx] = new_cam
                        missing_serials.discard(serial)
                        missing_counts[serial] = 0

                        # NIE ruszamy innych kamer — dołączamy tylko tę jedną
                        if not wait_for_single_ptp_ready(new_cam, role, max_wait=20.0):
                            raise RuntimeError(f"PTP nie ustabilizowało się dla kamery {role} po reconnect")

                        configure_periodic_signal_trigger(
                            new_cam,
                            serial,
                            period_us=float(shared_state.get("camera_period_us", 20000.0)),
                        )

                        try:
                            if not new_cam.IsGrabbing():
                                new_cam.StartGrabbing(pylon.GrabStrategy_OneByOne)
                        except Exception as e:
                            stats_q.put(f"[WATCHDOG]  StartGrabbing after reconnect failed for {role}: {e}")
                            raise
                        else:
                            th = threading.Thread(
                                target=grabber,
                                kwargs=dict(
                                    cam_idx=idx,
                                    cam=new_cam,
                                    role=role,
                                    serial=serial,
                                    raw_q=raw_q,
                                    universal_bayer_q=universal_bayer_q,
                                    stats_q=stats_q,
                                    stop_evt=stop_evt,
                                    streaming_enabled=streaming_enabled,
                                    remote_stream_enabled=remote_stream_enabled,
                                    recording_event=recording_event,
                                    snapshot_event=snapshot_event,
                                    snapshot_raw_q=snapshot_raw_q,
                                    shared_state=shared_state,
                                ),
                                daemon=True,
                                name=f"grabber_{role}",
                            )
                            th.start()
                            grabber_threads[role] = th

                        msg = f"[WATCHDOG] ??? Kamera {role} (S/N {serial}) ponownie podłączona i uruchomiona."
                        try:
                            stats_q.put(msg)
                        except Exception:
                            print(msg)
                        # --- [NOWOŚĆ] Jeśli trwa nagrywanie, uruchom saver_worker_bin dla tej kamery ---
                        if recording_event.is_set():
                            try:
                                from capture.grabber_module import saver_worker_bin
                                import core.utils_config as utils_config
                                from core.utils_config import BIN_BATCH_FRAMES, BIN_ROLL_EVERY

                                saver_threads = getattr(utils_config, "_RUNTIME_PROCS", {}).get("saver_threads", {})
                                if saver_threads is None:
                                    saver_threads = {}

                                in_q = per_role_q.get(role)
                                current_session = getattr(utils_config, "_CURRENT_SESSION_DIR", None)

                                if in_q is not None:
                                    th = threading.Thread(
                                        target=saver_worker_bin,
                                        kwargs=dict(
                                            role=role,
                                            in_q=in_q,
                                            root_dir=lambda: getattr(utils_config, "RAW_DIR", None),
                                            batch_frames=BIN_BATCH_FRAMES,
                                            roll_every=BIN_ROLL_EVERY,
                                            recording_event=recording_event,
                                            stop_evt=stop_evt,
                                            stats_q=stats_q,
                                            resume_dir=current_session,  # 🆕
                                        ),
                                        daemon=True,
                                        name=f"saver_{role}_reconnected",
                                    )
                                    th.start()
                                    saver_threads[role] = th
                                    utils_config._RUNTIME_PROCS["saver_threads"] = saver_threads
                                    msg = f"[WATCHDOG] 💾 Saver wznowiony dla kamery {role} po reconnect (kontynuacja sesji)."
                                    try:
                                        stats_q.put(msg)
                                    except Exception:
                                        print(msg)
                            except Exception as e:
                                msg = f"[WATCHDOG] ⚠️ Nie udało się wznowić saver_worker dla {role}: {e}"
                                try:
                                    stats_q.put(msg)
                                except Exception:
                                    print(msg)

                        if cam_status is not None:
                            cam_status[role] = "ok"

                    except Exception as e:
                        msg = f"[WATCHDOG]  Reconnect kamery {role} (S/N {serial}) nieudany: {e}"
                        try:
                            stats_q.put(msg)
                        except Exception:
                            print(msg)
                        continue


        # 🔄 Watchdog hot-plug kamer
        threading.Thread(
            target=camera_hotplug_watchdog,
            name="camera_hotplug_watchdog",
            daemon=True,
        ).start()

        # Control thread
        threads.append(threading.Thread(
            target=control_worker,
            args=(control_q, stop_evt, cams, roles, streaming_enabled, stats_q,
                  recording_event, snapshot_event, zoom_role_var, shared_state,archive_mgr, yolo_batch_q, yolo_raw_q

                  ),
            daemon=True,
        ))

        for t in threads:
            t.start()
        return threads

    start_helper_threads()
    stats_q.put("🔧 Uruchomiono wątki pomocnicze...")

    utils_config._RUNTIME_PROCS = {
        "stop_evt": stop_evt,
        "proc_color": {"universal": universal_proc},
        "per_role_q": per_role_q,
        "universal_bayer_q": universal_bayer_q,
        "saver_threads": {},
        "grabber_threads": grabber_threads,
    }

    ready_evt.set()

# ===========================================================
# Graceful shutdown
# ===========================================================

def _close_mp_queue(q):
    if q is None:
        return
    try:
        if hasattr(q, "cancel_join_thread"):
            q.cancel_join_thread()
    except Exception:
        pass
    try:
        if hasattr(q, "close"):
            q.close()
    except Exception:
        pass
    try:
        while hasattr(q, "empty") and not q.empty():
            _ = q.get_nowait()
    except Exception:
        pass

def graceful_shutdown(cams=None, router_p=None, yolo_procs=None, sorter_p=None,
                      stop_evt=None, stop_evt_mp=None, queues=None, procs=None, gui_root=None):
    print("\n[SHUTDOWN]  Zatrzymuję wszystkie procesy i kolejki...")

    if procs is None:
        procs = {}

    color_proc = procs.get("color_proc") if isinstance(procs, dict) else None

    try:
        if stop_evt:
            stop_evt.set()
        if stop_evt_mp:
            stop_evt_mp.set()
    except Exception:
        pass

    process_list = [p for p in [router_p, sorter_p, color_proc] if p] + (yolo_procs or [])
    try:
        from streaming.server_stream import stop_webrtc_server, kill_child_processes
        stop_webrtc_server()
        kill_child_processes()
    except Exception:
        pass

    # 🎬 Wyczyść bufory przed zamknięciem
    try:
        from capture.grabber_module import cleanup_buffers
        cleanup_buffers()
    except Exception as e:
        print(f"[SHUTDOWN] ⚠️ cleanup_buffers error: {e}")

    for p in process_list:
        try:
            if p and p.is_alive():
                print(f"[SHUTDOWN] Terminating {getattr(p, 'name', p)} (pid={p.pid})")
                p.terminate()
                p.join(timeout=2.0)
        except Exception as e:
            print(f"[SHUTDOWN] ⚠️ Could not terminate {p}: {e}")

    for name, p in procs.items():
        if p and p not in process_list:
            try:
                if hasattr(p, "is_alive") and p.is_alive():
                    print(f"[SHUTDOWN] 🧩 Terminating {name} (pid={p.pid})")
                    p.terminate()
                    p.join(timeout=2.0)
            except Exception as e:
                print(f"[SHUTDOWN] ⚠️ Could not terminate {name}: {e}")

    if queues:
        q_list = queues if isinstance(queues, (list, tuple)) else [queues]
        for q in q_list:
            try:
                while not q.empty():
                    _ = q.get_nowait()
            except Exception:
                pass

    if cams:
        for cam in cams:
            try:
                if hasattr(cam, "IsGrabbing") and cam.IsGrabbing():
                    cam.StopGrabbing()
                cam.Close()
            except Exception:
                pass

    if gui_root:
        try:
            gui_root.quit()
            gui_root.destroy()
        except Exception:
            pass

    try:
        from storage.shared_memory_manager import get_shared_memory_manager
        smm = get_shared_memory_manager()
        shutdown_fn = getattr(smm, "shutdown", None)
        close_fn = getattr(smm, "close", None)
        if callable(shutdown_fn):
            shutdown_fn()
        elif callable(close_fn):
            close_fn()
    except Exception:
        pass

    print("[SHUTDOWN]")

# ===========================================================
# Main
# ===========================================================

def main():
    import signal
    mp.freeze_support()

    # --- Manager i wspólny stan ---
    manager = mp.Manager()
    setattr(utils_config, "_MANAGER", manager)

    shared_state = manager.dict()
    shared_state["remote_url"] = None
    shared_state["selected_role"] = "CENTER_L"
    shared_state["current_key"] = manager.dict()
    roles = ["CENTER_L", "CENTER_R", "LEFT", "RIGHT"]
    shared_state["roles"] = roles
    shared_state["demo_test_name"] = "demo_default"
    shared_state["demo_operator_name"] = "operator_default"
    shared_state["yolo_generation"] = 0
    shared_state["yolo_bayer_pattern"] = "BG"
    shared_state["yolo_device"] = "cuda"
    shared_state["yolo_backend"] = "ultralytics"
    shared_state["yolo_preprocess_backend"] = "cpu"
    shared_state["yolo_trt_engine_path"] = None
    shared_state["yolo_trt_dynamic"] = False
    shared_state["yolo_trt_workspace_gb"] = 2.0
    shared_state["yolo_batch_queue_policy"] = "drop_oldest"
    shared_state["yolo_batch_q_maxsize"] = 4
    shared_state["yolo_raw_q_maxsize"] = 512
    shared_state["yolo_max_queue_wait_ms"] = 60.0
    shared_state["skip_ptp_wait"] = False
    shared_state["camera_target_fps"] = 50.0
    shared_state["camera_period_us"] = 20000.0
    shared_state["buffer_trace_enabled"] = True
    shared_state["verbose_pipeline_logs"] = False
    try:
        trt_cfg_path = Path(__file__).resolve().parent / "tensorrt_runtime.json"
        if trt_cfg_path.exists():
            trt_cfg = json.loads(trt_cfg_path.read_text(encoding="utf-8"))
            shared_state["yolo_backend"] = str(trt_cfg.get("backend", shared_state["yolo_backend"])).strip().lower()
            shared_state["yolo_device"] = str(trt_cfg.get("device", shared_state.get("yolo_device", "cuda"))).strip().lower()
            shared_state["yolo_preprocess_backend"] = str(
                trt_cfg.get("preprocess_backend", shared_state.get("yolo_preprocess_backend", "cpu"))
            ).strip().lower()
            engine_path = str(trt_cfg.get("engine_path", "") or "").strip()
            shared_state["yolo_trt_engine_path"] = engine_path or None
            shared_state["yolo_trt_dynamic"] = bool(trt_cfg.get("dynamic_shapes", False))
            shared_state["yolo_trt_workspace_gb"] = max(0.5, float(trt_cfg.get("workspace_gb", 2.0)))
            print(f"[MAIN] TensorRT runtime config loaded from: {trt_cfg_path}")
    except Exception as e:
        print(f"[MAIN] TensorRT runtime config load failed: {e}")

    # Zdarzenia
    stop_evt = mp.Event()
    recording_event = mp.Event()
    snapshot_event = mp.Event()
    remote_stream_enabled = mp.Event()
    remote_stream_enabled.clear()
    streaming_enabled = {r: mp.Event() for r in roles}
    for r in roles:
        streaming_enabled[r].clear()

    # Wstrzyknięcie do utils_config
    utils_config.stop_evt = stop_evt
    setattr(utils_config, "recording_event", recording_event)
    setattr(utils_config, "snapshot_event", snapshot_event)
    setattr(utils_config, "remote_stream_enabled", remote_stream_enabled)
    setattr(utils_config, "streaming_enabled", streaming_enabled)
    # Kolejki
    stats_q_mp = mp.Queue(maxsize=256)    # szybciej niż Manager().Queue
    stats_q_gui = queue.Queue()           # lokalna do GUI
    raw_q = queue.Queue(maxsize=128)      # thread-only, zero pickling
    live_q = queue.Queue(maxsize=64)      # thread-only; przekazujemy KLUCZE do SHM
    zoom_q = mp.Queue(maxsize=2)
    control_q = mp.Queue(maxsize=64)
    utils_config.live_q = live_q
    setattr(utils_config, "control_q", control_q)
    snapshot_raw_q = mp.Queue(maxsize=len(roles) * 2)
    yolo_vis_q = mp.Queue(maxsize=32)
    yolo_raw_q_maxsize = int(shared_state.get("yolo_raw_q_maxsize", 512) or 512)
    yolo_raw_q = queue.Queue(maxsize=max(16, yolo_raw_q_maxsize))
    try:
        num_roles = len(roles) if roles else 4
        if "yolo_input_batch_images" not in shared_state:
            shared_state["yolo_input_batch_images"] = num_roles
        if "yolo_seq_len" not in shared_state:
            shared_state["yolo_seq_len"] = 1
    except Exception:
        pass



    def _sigterm_handler(signum, frame):
        try:
            utils_config.stop_evt.set()
        except Exception:
            pass
        try:
            control_q.put(("remote_stream", {"enabled": False}))
            control_q.put(("stop", True))
        except Exception:
            pass

    signal.signal(signal.SIGINT, _sigterm_handler)
    signal.signal(signal.SIGTERM, _sigterm_handler)

    preview_queue_maxsize = 6
    preview_q_rgb = {r: mp.Queue(maxsize=preview_queue_maxsize) for r in roles}

    ready_evt = threading.Event()

    # Przepinanie logów z backendu do GUI (bez blokad)
    def copy_stats():
        """
        Kopiuje wszystkie statystyki z kolejki stats_q_mp
        do stats_q_gui (główne GUI).

        Z zabezpieczeniem przed przepełnieniem kolejki.
        """
        msg = None
        while True:
            # 1) Pobierz wiadomość z kolejki głównej stats_q_mp
            try:
                msg = stats_q_mp.get(timeout=1.0)
            except Exception:
                if stop_evt.is_set():
                    break
                continue

            # 2) Przekaż do głównego GUI
            try:
                stats_q_gui.put_nowait(msg)
            except queue.Full:
                # Jeśli pełna – zrzuć najstarszą i dopiero wstaw
                try:
                    _ = stats_q_gui.get_nowait()
                    stats_q_gui.put_nowait(msg)
                except Exception:
                    pass
            except Exception:
                pass

            # 3) sprawdź stop_evt
            if stop_evt.is_set():
                break

    # uruchomienie wątku
    threading.Thread(target=copy_stats, daemon=True).start()

    print("[MAIN] Uruchamianie GUI i backendu...")

    from core.config_menager import load_config, set_config_path
    APP_CONFIG_PATH = os.path.join(APP_DIR, "app_config.json")
    set_config_path(APP_CONFIG_PATH)
    cfg = load_config()

    init_settings = {
        "path2save": cfg.get("path2save"),
        "gain_val": cfg.get("gain_val"),
        "exposure_val": cfg.get("exposure_val"),
        "per_role": cfg.get("per_role", {}),
    }

    shared_state["initial_camera_params"] = {
        "gain_val": cfg.get("gain_val"),
        "exposure_val": cfg.get("exposure_val"),
        "per_role": cfg.get("per_role", {}),
        "per_camera": cfg.get("per_camera", {}),
    }

    force_simulate = _truthy(os.environ.get("VOLLEYHUB_SIMULATE"))
    if force_simulate:
        print("[MAIN] VOLLEYHUB_SIMULATE=1 - starting in SIMULATION mode.")
        devices = []
        simulate = True
    else:
        try:
            from pypylon import pylon
            tl = pylon.TlFactory.GetInstance()
            devs = tl.EnumerateDevices()
            devices = [d.GetSerialNumber() for d in devs]
            simulate = False
        except Exception as e:
            print(f"[MAIN] ⚠️ pypylon unavailable: {e} SYMULACJA.")
            devices = []
            simulate = True

    if not devices and not force_simulate:
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showwarning("No Cameras", "No cameras detected. Starting in SIMULATION mode.")
            root.destroy()
        except Exception:
            pass
        simulate = True

    # (Tylko jeśli są kamery) wybór/ładowanie ról
    if not simulate:
        THIS_DIR = os.path.dirname(os.path.abspath(__file__))
        ROLES_PATH = os.path.join(THIS_DIR, "camera_roles.json")
        choice = ask_camera_roles_gui()
        if choice == "cancel":
            print("[MAIN] User cancelled startup.")
            sys.exit(0)
        elif choice == "assign":
            try:
                from ui.assign_roles_gui import AssignRolesApp
                app_assign = AssignRolesApp()
                app_assign.mainloop()
            except Exception as e:
                print(f"[MAIN] Assign GUI failed: {e}")
        elif choice == "existing":
            print("[MAIN] Using existing camera_roles.json configuration.")
        else:
            print("[MAIN] ⚠️ No choice made, exiting.")
            sys.exit(0)
        if not os.path.exists(ROLES_PATH):
            try:
                import tkinter as tk
                from tkinter import messagebox
                root = tk.Tk()
                root.withdraw()
                messagebox.showerror("Missing configuration", "camera_roles.json not found. Assign roles first.")
                root.destroy()
            except Exception:
                pass
            sys.exit(1)

    # --- Ustawienia startowe ---
    if not simulate:
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            skip_ptp_wait = bool(messagebox.askyesno(
                "PTP Startup",
                "pomiń czekanie na PTP przed startem?\n\n"
                "TAK = szybszy start, ale  gorsza synchronizacja.\n"
                "NIE = czekaj na PTP."
            ))
            root.destroy()
        except Exception:
            skip_ptp_wait = False
        shared_state["skip_ptp_wait"] = skip_ptp_wait
        print(f"[MAIN] skip_ptp_wait={skip_ptp_wait}")

    # GUI
    from ui._backup_gui_extract.gui import CaptureGUI
    app = CaptureGUI(
        live_q=live_q,
        stats_q=stats_q_gui,
        control_q=control_q,
        ready_evt=ready_evt,
        roles=roles,
        init_settings=init_settings,
        recording_event=recording_event,
        snapshot_event=snapshot_event,
        snapshot_q=snapshot_raw_q,
        zoom_q=zoom_q,
        shared_state=shared_state,
        yolo_vis_q=yolo_vis_q,
    )
    app.shared_state = shared_state
    app.after(50, app._wait_for_system_ready)

    # Backend (wątek)
    from storage.shared_memory_manager import get_shared_memory_manager
    get_shared_memory_manager()
    archive_mgr = SessionArchiveManager()

    try:
        shared_state["buffer_dt_s"] = int(shared_state.get("buffer_dt_s", 5))
    except Exception:
        shared_state["buffer_dt_s"] = 5


    print("[MAIN] ✓ SharedMemoryManager zainicjalizowany globalnie")
    backend_thread = threading.Thread(target=backend_initializer, args=(
        live_q, stats_q_mp, control_q, ready_evt, recording_event,
        preview_q_rgb, snapshot_raw_q, raw_q, manager.Value('s', "CENTER_L"),
        shared_state, zoom_q, simulate, yolo_vis_q,archive_mgr,yolo_raw_q
    ), daemon=True)
    backend_thread.start()
    print("[MAIN] ✓ backend uruchomiony")
    print("[MAIN] Uruchamiam GUI (mainloop)...")
    app.mainloop()

    # === Shutdown path ===
    try:
        stop_evt.set()

    except Exception:
        pass
    try:
        control_q.put(("remote_stream", {"enabled": False}))
        control_q.put(("stop", True))
    except Exception:
        pass

    _close_mp_queue(raw_q)
    _close_mp_queue(live_q)
    _close_mp_queue(zoom_q)
    _close_mp_queue(control_q)
    _close_mp_queue(snapshot_raw_q)
    for _q in (preview_q_rgb.values() if isinstance(preview_q_rgb, dict) else []):
        _close_mp_queue(_q)

    procs = getattr(utils_config, "_RUNTIME_PROCS", {})
    graceful_shutdown(
        cams=procs.get("cams"),
        router_p=procs.get("router"),
        sorter_p=procs.get("sorter"),
        stop_evt=procs.get("stop_evt"),
        stop_evt_mp=procs.get("stop_evt_mp"),
        queues=procs.get("queues"),
        gui_root=app,
    )

    if backend_thread.is_alive():
        print("[MAIN]  backend_thread...")
        backend_thread.join(timeout=2.0)

    try:
        manager_obj = getattr(utils_config, "_MANAGER", None)
        if manager_obj is not None and hasattr(manager_obj, "shutdown"):
            manager_obj.shutdown()
            print("[MAIN] Manager shutdown")
    except Exception as e:
        print(f"[MAIN][⚠️] Manager shutdown error: {e}")

    try:
        from streaming.server_stream import stop_webrtc_server, kill_child_processes
        stop_webrtc_server()
        kill_child_processes()
    except Exception as e:
        print(f"[MAIN][⚠️] kill_child_processes error: {e}")

    for child in list(mp.active_children()):
        try:
            if child.is_alive():
                print(f"[MAIN] terminate {getattr(child, 'name', child)} (pid={child.pid})")
                child.terminate()
                child.join(timeout=1.0)
                if child.is_alive():
                    print(f"[MAIN] kill {getattr(child, 'name', child)} (pid={child.pid})")
                    child.kill()
                    child.join(timeout=1.0)
        except Exception as e:
            print(f"[MAIN][⚠️] Nie  {getattr(child, 'name', child)}: {e}")

    try:
        sys.exit(0)
    except SystemExit:
        time.sleep(0.2)
        os._exit(0)

# ===========================================================
# Entrypoint (set start method only here)
# ===========================================================

if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
