import json
import os
import struct
import time

import numpy as np
from pypylon import pylon

from ui.assign_roles_gui import DEFAULT_JSON_PATH

role_path = DEFAULT_JSON_PATH


# exposure_us = data["cam_exposure_us"]
width = 1920
height = 1080

def _node(nm, name):
    try:
        return nm.GetNode(name)
    except Exception:
        return None

def _set_bool(node, val: bool):
    if node:
        node.SetValue(bool(val))
        return True
    return False

def _set_enum(node, entry_str: str):
    if node:
        try:
            node.FromString(entry_str)
            return True
        except Exception:
            entry = node.GetEntryByName(entry_str)
            if entry:
                node.SetIntValue(entry.GetValue())
                return True
    return False

#=====orders


def _get_ptp_status(cam):
    nm = cam.GetNodeMap()
    for name in ("PtpStatus", "GevIEEE1588Status", "BslPtpStatus"):
        try:
            n = _node(nm, name)
            if n is not None:
                return str(n.ToString() if hasattr(n, "ToString") else n.GetValue())
        except Exception:
            pass
    return None

def get_gige_tl():
    """
    Zwraca transport layer GigE, o ile wrapper pypylon go udostępnia.
    """
    tl_factory = pylon.TlFactory.GetInstance()

    # najprostsza ścieżka: wrapper ma CreateTl
    for name in ("BaslerGigE", "GEV", "GigE"):
        try:
            return tl_factory.CreateTl(name)
        except Exception:
            pass

    # fallback: jeśli wrapper nie wspiera CreateTl w ten sposób
    raise RuntimeError("Nie udało się uzyskać GigE transport layer z pypylon")

def latch_timestamp_ns(cam) -> int:
    nm = cam.GetNodeMap()

    # command
    for cmd_name in ("TimestampLatch", "GevTimestampControlLatch"):
        try:
            cmd = _node(nm, cmd_name)
            if cmd is not None:
                cmd.Execute()
                break
        except Exception:
            pass

    # value
    for val_name in ("TimestampLatchValue", "GevTimestampValue", "GevTimestampControlLatchValue"):
        try:
            n = _node(nm, val_name)
            if n is not None:
                return int(n.GetValue())
        except Exception:
            pass

    raise RuntimeError("Nie udało się odczytać timestamp latch z kamery")


def _set_value_if_exists(nm, name, value):
    n = _node(nm, name)
    if n is None:
        return False
    try:
        if isinstance(value, str):
            n.FromString(value)
        else:
            n.SetValue(value)
        return True
    except Exception:
        return False


def configure_periodic_signal_trigger(cam, serial, period_us: float = 20000.0):

    # 1. Upewnij się, że nie grabujemy
    if cam.IsGrabbing():
        cam.StopGrabbing()

    cam.TriggerMode.SetValue("Off")
    cam.AcquisitionFrameRateEnable.SetValue(False)

    timeout = 10
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            cam.BslPeriodicSignalSelector.SetValue("PeriodicSignal1")
            print(f"[{serial}] PtpClock ustawiony jako źródło.")
            break
        except Exception:
            time.sleep(0.5)

    # 4. Ustawienie parametrów sygnału i aktywacja
    try:
        cam.BslPeriodicSignalPeriod.SetValue(float(period_us))
        cam.BslPeriodicSignalDelay.SetValue(0.0)

        cam.TriggerSelector.SetValue("FrameStart")
        cam.TriggerSource.SetValue("PeriodicSignal1")
        cam.TriggerMode.SetValue("On")
        print(f"[{serial}] Synchronizacja PeriodicSignal1 aktywna.")
    except Exception as e:
        print(f"[{serial}] błąd końcowy: {e}")


def _get_ptp_servo_status(cam):
    nm = cam.GetNodeMap()
    for name in ("PtpServoStatus", "BslPtpServoStatus"):
        try:
            n = _node(nm, name)
            if n is not None:
                return str(n.ToString() if hasattr(n, "ToString") else n.GetValue())
        except Exception:
            pass
    return None

def wait_for_all_ptp_ready(cams, max_wait=60):
    for _ in range(max_wait):
        statuses = []
        servo_states = []

        for cam in cams:

            try:
                try:
                    cam.PtpDataSetLatch.Execute()
                except Exception:
                    pass

                st = _get_ptp_status(cam)
                sv = _get_ptp_servo_status(cam)
            except Exception:
                st, sv = None, None

            statuses.append(st)
            servo_states.append(sv)

        print(f"[PTP] statuses={statuses} servo={servo_states}")

        masters = 0
        ok = True

        for st, sv in zip(statuses, servo_states):
            if st == "Master":
                masters += 1
                continue

            if st == "Slave" and (sv == "Locked" or sv is None):
                continue

            ok = False
            break

        if ok and masters == 1:
            print("[PTP]  Wszystkie kamery gotowe")
            return True

        time.sleep(2)

    print("[PTP] Timeout waiting for all cameras to lock PTP")
    return False


def cam_config(serial: str, cam: pylon.InstantCamera, stats_q=None):
    nm = cam.GetNodeMap()
    print("==============camconfig===============")

    # reset starego stanu
    for name, value in (
        ("PtpEnable", False),
        ("GevIEEE1588", False),
    ):
        try:
            n = _node(nm, name)
            if n is not None:
                n.SetValue(value)
                print(f"[{serial}] {name} -> {value}")
        except Exception:
            pass

    time.sleep(0.2)

    wn, hn = cam.Width, cam.Height
    aligned_w = wn.GetMin() + ((min(width, wn.GetMax()) - wn.GetMin()) // wn.GetInc()) * wn.GetInc()
    aligned_h = hn.GetMin() + ((min(height, hn.GetMax()) - hn.GetMin()) // hn.GetInc()) * hn.GetInc()

    cam.Width.SetValue(aligned_w)
    cam.Height.SetValue(aligned_h)
    cam.PixelFormat.SetValue("BayerRG8")
    cam.ExposureMode.SetValue("Timed")
    try:
        cam.GevSCPD.SetValue(4000)
    except Exception:
        pass

    try:
        cam.GevSCPSPacketSize.SetValue(8000)
    except Exception:
        pass
    _set_bool(_node(nm, "ChunkModeActive"), True)

    selector_set = False
    for sel_name in ("BslChunkTimestampSelector", "ChunkTimestampSelector", "TimestampSelector"):
        if _set_enum(_node(nm, sel_name), "FrameStart"):
            print(f"[{serial}] {sel_name}=FrameStart")
            selector_set = True
            break

    if not selector_set:
        print(f"[{serial}] WARNING: nie udało się ustawić selector FrameStart")

    n_chunk_sel = _node(nm, "ChunkSelector")
    n_chunk_en = _node(nm, "ChunkEnable")

    if _set_enum(n_chunk_sel, "Timestamp") and _set_bool(n_chunk_en, True):
        print(f"[{serial}] ChunkSelector=Timestamp + ChunkEnable=ON")
    else:
        for alt in ("ChunkTimestampEnable", "BslChunkTimestampEnable"):
            if _set_bool(_node(nm, alt), True):
                print(f"[{serial}] {alt}=ON")
                break

    # PTP włączamy, ale gotowość sprawdzimy później zbiorczo
    ptp_ok = False
    for ptp_name in ("PtpEnable", "GevIEEE1588"):
        try:
            node = _node(nm, ptp_name)
            if node is not None:
                node.SetValue(True)
                print(f"[{serial}] {ptp_name}: Enabled")
                ptp_ok = True
                break
        except Exception as e:
            print(f"[{serial}] {ptp_name}: fail ({e})")

    if not ptp_ok:
        print(f"[{serial}] PTP: not available")

    try:
        cam.GainSelector.SetValue("All")
    except Exception:
        pass

def load_roles(dev_infos):
    roles = {}
    if os.path.isfile(role_path):
        with open(role_path, "r", encoding="utf-8") as f:
            arr = json.load(f)
        for it in arr:
            s = str(it.get("serial", "")).strip()
            r = str(it.get("role", "")).strip() or None
            if s and r:
                roles[s] = r
    out = {}
    for i, di in enumerate(dev_infos):
        ser = di.GetSerialNumber()
        out[ser] = roles.get(ser, f"CAM{i}")
    return out

def make_bgr_converter():
    conv = pylon.ImageFormatConverter()
    conv.OutputPixelFormat = pylon.PixelType_BGR8packed
    conv.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    return conv

def ns_per_tick_for(cam) -> float:
    """GigE: GevTimestampTickFrequency → ns/tick; fallback 1.0."""
    try:
        freq = float(cam.GevTimestampTickFrequency.GetValue())
        if freq > 0:
            return 1e9 / freq
    except Exception:
        pass
    return 1.0

def write_binary_batch(file_path: str, frames: list[np.ndarray], frame_ids: list[int], timestamps: list[int]):
    """Format rekordów: <Q ts_ns><I frame_id><raw BGR bytes> * N (append)."""
    with open(file_path, "ab", buffering=0) as f:
        for i in range(len(frames)):
            f.write(struct.pack("<QI", int(timestamps[i]), int(frame_ids[i])))
            f.write(frames[i].tobytes(order="C"))

def grab_ts_ns(gr, ns_per_tick: float) -> int:
    """Preferuj TimeStamp w tickach; inaczej perf_counter_ns."""
    try:
        ts_ticks = int(getattr(gr, "TimeStamp"))
        if ts_ticks:
            return int(ts_ticks * ns_per_tick)
    except Exception:
        pass
    return time.perf_counter_ns()

def grab_ts_ns_from_chunk(gr, ns_per_tick_fallback: float | None = None) -> int | None:
    """
    Spróbuj odczytać hardware'owy ChunkTimestamp z GrabResult.
    Zwraca ns (int) lub None gdy niedostępny.
    """
    ts_ns = None
    try:
        # najprościej: wiele modeli wystawia właściwość bezpośrednio
        if hasattr(gr, "ChunkTimestamp"):
            # bywa .Value lub bezpośrednio int; wspieramy oba warianty
            val = getattr(gr, "ChunkTimestamp")
            ts = int(val.Value if hasattr(val, "Value") else int(val))
            # Niektóre kamery zwracają ticki – jeśli znasz ns_per_tick, przelicz:
            if ns_per_tick_fallback and ts < 1e12:
                ts_ns = int(ts * ns_per_tick_fallback)
            else:
                # część kamer zwraca już ns – zwykle wielkości 1e18 to przesada; wyczuj skalę
                ts_ns = int(ts)
            return ts_ns
    except Exception:
        pass

    # Alternatywa: node map chunków
    try:
        node_map = gr.GetChunkDataNodeMap()
        # W API pypylon często wystarczy tak:
        n = node_map.GetNode("ChunkTimestamp")
        if n and n.IsReadable():
            ts = int(n.GetValue())
            if ns_per_tick_fallback and ts < 1e12:
                ts_ns = int(ts * ns_per_tick_fallback)
            else:
                ts_ns = int(ts)
            return ts_ns
    except Exception:
        pass

    return None

def quick_ts_probe(cam, n=5):
    print("[TS probe] start")
    for _ in range(n):
        gr = cam.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
        if not gr or not gr.GrabSucceeded():
            print("  grab fail"); continue
        ts = grab_ts_ns_from_chunk(gr, ns_per_tick_fallback=ns_per_tick_for(cam))
        print(f"  ChunkTimestamp ns: {ts}")
        gr.Release()
    print("[TS probe] done")

def init_all_cameras():
    print("[INIT] Wykrywanie kamer Basler...")

    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()

    if not devices:
        raise RuntimeError("Nie znaleziono żadnych kamer Basler!")

    print(f"[INIT] Wykryto {len(devices)} kamer.")

    serial_roles = load_roles(devices)

    cams = []
    roles = []

    for di in devices:
        serial = di.GetSerialNumber()
        role = serial_roles.get(serial, f"CAM_{serial}")
        cam = pylon.InstantCamera(tl_factory.CreateDevice(di))
        cam.Open()

        print(f"[INIT] Konfiguracja kamery {serial} ({role}) ...")
        try:
            cam_config(serial, cam)
            cams.append(cam)
            roles.append(role)
        except Exception as e:
            print(f"[INIT] nie udało się skonfigurować kamery {serial}: {e}")
            cam.Close()

    if not cams:
        raise RuntimeError("nie udało się skonfigurować żadnej kamery")

    # 1. poczekaj na pełne PTP po otwarciu wszystkich kamer
    if not wait_for_all_ptp_ready(cams, max_wait=30):
        print("[INIT] PTP nie ustabilizował na wszystkich kamerach")
    return cams, roles

def init_one_cameras(serial):
    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()

    if not devices:
        raise RuntimeError("Nie znaleziono żadnych kamer Basler!")

    chosen = None
    for di in devices:
        if di.GetSerialNumber() == str(serial):
            chosen = di
            break

    if chosen is None:
        raise RuntimeError(f"Nie znaleziono kamery o serialu {serial}")

    roles_map = load_roles(devices)
    role = roles_map.get(str(serial), f"CAM_{serial}")

    cam = pylon.InstantCamera(tl_factory.CreateDevice(chosen))
    cam.Open()

    cam_config(serial, cam)

    return cam, role



