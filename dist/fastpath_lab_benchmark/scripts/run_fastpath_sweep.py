"""Drive the 4-variant fast-path sweep and write reports/results.json.

Modes:
  --mode synthetic : pure-numpy 4-cam Bayer source (no Basler hardware)
  --mode basler    : real Basler 4-cam via pypylon + camera_roles.json

For each variant in configs/benchmark_config.yaml:
  - locate engine in artifacts/tensorrt_exports/
  - if engine missing → status=ENGINE_MISSING (no traceback)
  - else: run native_cuda_npp + TRT b4 packet loop for `duration_s`
  - record per-stage timings, derive pass/fail vs 50 FPS budget

Output:
  reports/results.json — structured, baseline-comparable
  reports/results.csv  — flat per-variant rows
  logs/live_probe_*.log — probe lines (when probe is on)
"""
from __future__ import annotations

import sys
# Make emoji-laden prints from third-party modules safe on Windows cp1250 consoles.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import argparse
import csv
import json
import logging
import os
import platform
import socket
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import numpy as np
import yaml  # type: ignore

_LOG = logging.getLogger("fastpath_sweep")


# ---------------------------------------------------------------- helpers

def _load_config() -> dict:
    cfg_path = PACKAGE_ROOT / "configs" / "benchmark_config.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"missing config: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _percentile(samples: List[float], p: float) -> float:
    if not samples:
        return 0.0
    s = sorted(samples)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    f_idx = int(k)
    c_idx = min(f_idx + 1, len(s) - 1)
    return s[f_idx] + (s[c_idx] - s[f_idx]) * (k - f_idx)


def _median(samples: List[float]) -> float:
    if not samples:
        return 0.0
    s = sorted(samples)
    n = len(s)
    return s[n // 2] if n % 2 == 1 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _machine_info() -> dict:
    info = {
        "machine_id": socket.gethostname(),
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "cuda_available": False,
        "torch_version": None,
        "cuda_version_from_torch": None,
        "gpu_name": None,
        "gpu_vram_gb": None,
        "tensorrt_available": False,
        "pypylon_available": False,
        "native_backend_available": False,
    }
    try:
        import torch  # type: ignore
        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["cuda_version_from_torch"] = getattr(getattr(torch, "version", None), "cuda", None)
            info["gpu_name"] = torch.cuda.get_device_name(0)
            try:
                props = torch.cuda.get_device_properties(0)
                info["gpu_vram_gb"] = round(float(props.total_memory) / (1024 ** 3), 2)
            except Exception:
                pass
    except Exception:
        pass
    try:
        import tensorrt  # type: ignore  # noqa: F401
        info["tensorrt_available"] = True
    except Exception:
        pass
    try:
        import pypylon  # type: ignore  # noqa: F401
        info["pypylon_available"] = True
    except Exception:
        pass
    try:
        from src.runtime_gpu.native_debayer import NativeCudaDebayer
        info["native_backend_available"] = bool(NativeCudaDebayer.is_available())
    except Exception:
        pass
    return info


# ---------------------------------------------------------------- sources

class _SyntheticSource:
    """4-cam shared-memory grabber emulator. Writes synthetic 1920x1080 RAW to SMM."""

    def __init__(self, capture_w: int, capture_h: int, target_fps: float = 50.0,
                 bayer_pattern: str = "RG"):
        from storage.shared_memory_manager import get_shared_memory_manager  # type: ignore
        self.smm = get_shared_memory_manager()
        self.capture_w = int(capture_w)
        self.capture_h = int(capture_h)
        self.target_fps = float(target_fps)
        self.bayer_pattern = bayer_pattern
        self.shared_state: dict = {"bayer_key": {}, "last_frame_ts": {}}
        self.roles = ("LEFT", "CENTER_L", "CENTER_R", "RIGHT")
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []

    def _grab(self, role: str, key: str, frame_offset: int) -> None:
        period = 1.0 / max(1.0, self.target_fps)
        next_t = time.perf_counter()
        frame_id = frame_offset
        while not self._stop.is_set():
            if time.perf_counter() < next_t:
                self._stop.wait(max(0.0, next_t - time.perf_counter()))
                if self._stop.is_set():
                    break
            next_t += period
            frame_id += 1
            rng = np.random.default_rng(frame_id & 0xFFFF)
            frame = rng.integers(0, 256, size=(self.capture_h, self.capture_w), dtype=np.uint8)
            self.smm.write_frame(key, frame, ts_ns=time.time_ns())
            self.shared_state["bayer_key"][role] = key
            self.shared_state["last_frame_ts"][role] = time.time_ns()

    def start(self) -> None:
        self._stop.clear()
        for i, role in enumerate(self.roles):
            key = f"fastpath_smoke_{role}"
            t = threading.Thread(target=self._grab, name=f"grab-{role}",
                                 args=(role, key, i * 1000), daemon=True)
            t.start()
            self._threads.append(t)
        time.sleep(0.5)  # let SMM keys appear

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=1.0)
        try:
            self.smm.cleanup_all()
        except Exception:
            pass


class _BaslerSource:
    """Real Basler 4-cam grabber. Reads camera_roles.json, opens 4 cameras."""

    def __init__(self, roles_path: Path, capture_w: int, capture_h: int,
                 bayer_pattern: str = "RG"):
        self.capture_w = int(capture_w)
        self.capture_h = int(capture_h)
        self.bayer_pattern = bayer_pattern
        self.shared_state: dict = {"bayer_key": {}, "last_frame_ts": {}}
        self.roles = ("LEFT", "CENTER_L", "CENTER_R", "RIGHT")
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self._roles_path = roles_path
        self._cameras: list = []
        self._open()

    def _open(self) -> None:
        try:
            from pypylon import pylon  # type: ignore
        except Exception as exc:
            raise SystemExit(f"PYPYLON_MISSING: {exc!r}")
        if not self._roles_path.exists():
            raise SystemExit(f"camera_roles file missing: {self._roles_path}")
        with self._roles_path.open("r", encoding="utf-8") as f:
            roles_cfg = json.load(f)
        wanted: Dict[str, str] = {}
        for entry in roles_cfg:
            role = str(entry.get("role", "")).upper()
            serial = str(entry.get("serial", ""))
            if role and serial:
                wanted[role] = serial
        for r in self.roles:
            if r not in wanted:
                raise SystemExit(f"camera_roles missing role={r}")

        tlf = pylon.TlFactory.GetInstance()
        devs = tlf.EnumerateDevices()
        by_serial = {d.GetSerialNumber(): d for d in devs if d.GetSerialNumber()}
        opened: Dict[str, Any] = {}
        for role, serial in wanted.items():
            dev = by_serial.get(serial)
            if dev is None:
                raise SystemExit(f"BASLER_DEVICE_MISSING: serial={serial} not found "
                                 f"(role={role}). Available: {list(by_serial.keys())}")
            cam = pylon.InstantCamera(tlf.CreateDevice(dev))
            cam.Open()
            try:
                cam.GetNodeMap().GetNode("Width").SetValue(self.capture_w)
                cam.GetNodeMap().GetNode("Height").SetValue(self.capture_h)
            except Exception:
                pass
            try:
                cam.GetNodeMap().GetNode("PixelFormat").FromString(f"BayerRG8")
            except Exception:
                pass
            cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            opened[role] = cam
        self._cameras_by_role = opened
        self._pylon = pylon

    def _grab(self, role: str, key: str) -> None:
        from storage.shared_memory_manager import get_shared_memory_manager  # type: ignore
        smm = get_shared_memory_manager()
        cam = self._cameras_by_role[role]
        pylon = self._pylon
        while not self._stop.is_set():
            try:
                grab = cam.RetrieveResult(500, pylon.TimeoutHandling_ThrowException)
            except Exception:
                continue
            try:
                if not grab.GrabSucceeded():
                    continue
                arr = grab.GetArray()
                smm.write_frame(key, arr, ts_ns=time.time_ns())
                self.shared_state["bayer_key"][role] = key
                self.shared_state["last_frame_ts"][role] = time.time_ns()
            finally:
                grab.Release()

    def start(self) -> None:
        self._stop.clear()
        for role in self.roles:
            key = f"fastpath_basler_{role}"
            t = threading.Thread(target=self._grab, name=f"basler-{role}",
                                 args=(role, key), daemon=True)
            t.start()
            self._threads.append(t)
        time.sleep(0.5)

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=1.0)
        for cam in self._cameras_by_role.values():
            try:
                cam.StopGrabbing()
            except Exception:
                pass
            try:
                cam.Close()
            except Exception:
                pass


# ---------------------------------------------------------------- one variant

def _run_variant(name: str, imgsz: Union[int, List[int]], engine_path: Path,
                 source, runtime_cfg: dict, target_cfg: dict,
                 duration_s: float, source_type: str) -> Dict[str, Any]:
    machine = _machine_info()
    capture_w = int(target_cfg.get("capture_resolution", [1920, 1080])[0])
    capture_h = int(target_cfg.get("capture_resolution", [1920, 1080])[1])
    batch = int(target_cfg.get("batch", 4))
    pkt_budget = float(target_cfg.get("packet_budget_ms", 20.0))

    if isinstance(imgsz, list) and len(imgsz) == 2:
        imgsz_value = (int(imgsz[0]), int(imgsz[1]))
        infer_h, infer_w = imgsz_value
    else:
        v = int(imgsz)
        imgsz_value = v
        infer_h = infer_w = v

    base_row: Dict[str, Any] = {
        "run_id": f"{name}__{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "name": name,
        "source_type": source_type,
        "machine_id": machine.get("machine_id"),
        "hostname": machine.get("hostname"),
        "os": machine.get("os"),
        "python_version": machine.get("python_version"),
        "cuda_available": machine.get("cuda_available"),
        "torch_version": machine.get("torch_version"),
        "cuda_version_from_torch": machine.get("cuda_version_from_torch"),
        "gpu_name": machine.get("gpu_name"),
        "gpu_vram_gb": machine.get("gpu_vram_gb"),
        "tensorrt_available": machine.get("tensorrt_available"),
        "pypylon_available": machine.get("pypylon_available"),
        "native_backend_available": machine.get("native_backend_available"),
        "capture_resolution": f"{capture_w}x{capture_h}",
        "inference_input_shape": f"{infer_h}x{infer_w}",
        "inference_pixels_mpx": round(infer_h * infer_w / 1e6, 4),
        "batch": batch,
        "packet_roles_count": 4,
        "color_backend": runtime_cfg.get("color_backend", "native_cuda_npp"),
        "native_backend_detail": None,
        "inference_backend": runtime_cfg.get("inference_backend", "tensorrt"),
        "engine_path": str(engine_path),
        "engine_reused": False,
        "engine_built_on_this_machine": False,
        "engine_compatibility_warning": "",
        "engine_missing": False,
        "expected_input_shape": f"{infer_h}x{infer_w}",
        "expected_batch": batch,
        "zero_copy_to_inference": False,
        "gpu_roundtrip": False,
        "fallback_used": "",
        "color_output_location": "",
        "inference_input_location": "",
        "drops": 0,
        "errors": [],
        "status": "PASS",
    }

    # Engine missing → ENGINE_MISSING (no traceback)
    if not engine_path.exists():
        base_row["status"] = "ENGINE_MISSING"
        base_row["engine_missing"] = True
        base_row["errors"].append(f"engine not found: {engine_path.name}")
        return _empty_metrics(base_row, pkt_budget)

    # Backend / TRT preflight
    if not machine.get("cuda_available"):
        base_row["status"] = "BACKEND_MISSING"
        base_row["errors"].append("CUDA_UNAVAILABLE")
        return _empty_metrics(base_row, pkt_budget)
    if not machine.get("native_backend_available"):
        base_row["status"] = "BACKEND_MISSING"
        base_row["errors"].append("NATIVE_BACKEND_MISSING")
        return _empty_metrics(base_row, pkt_budget)

    # Build the executor
    try:
        from src.runtime_production import FastPathConfig, FastPathExecutor
        from src.runtime_gpu.native_debayer import NativeCudaDebayer
    except Exception as exc:
        base_row["status"] = "ERROR"
        base_row["errors"].append(f"runtime import failed: {exc!r}")
        return _empty_metrics(base_row, pkt_budget)

    cfg = FastPathConfig(
        enabled=True,
        engine_path=str(engine_path),
        color_backend="native_cuda_npp",
        inference_backend="tensorrt",
        imgsz=imgsz_value,  # int (square) or (H, W) tuple
        batch_size=batch,
        bayer_pattern=str(runtime_cfg.get("bayer_pattern", "RG")),
        half=bool(runtime_cfg.get("half", True)),
        device="cuda",
        warmup_iterations=int(runtime_cfg.get("warmup_packets", 30)) // 10 + 1,
        required_roles=("LEFT", "CENTER_L", "CENTER_R", "RIGHT"),
    )

    err = cfg.validate()
    if err:
        base_row["status"] = "ERROR"
        base_row["errors"].append(f"config invalid: {err}")
        return _empty_metrics(base_row, pkt_budget)

    executor = FastPathExecutor(cfg)
    try:
        ok = executor.initialize()
    except Exception as exc:
        base_row["status"] = "ERROR"
        base_row["errors"].append(f"executor init exception: {exc!r}")
        return _empty_metrics(base_row, pkt_budget)
    if not ok:
        msg = executor.init_error or "executor init failed"
        # Engine-load specific messages get a nicer status
        if "engine" in msg.lower() or "trt" in msg.lower() or "tensorrt" in msg.lower():
            base_row["status"] = "ENGINE_MISSING"
            base_row["engine_compatibility_warning"] = msg
        else:
            base_row["status"] = "BACKEND_MISSING"
        base_row["errors"].append(msg)
        return _empty_metrics(base_row, pkt_budget)

    base_row["engine_reused"] = engine_path.exists()
    info = NativeCudaDebayer.describe_backend()
    base_row["native_backend_detail"] = info.get("detail")

    # Ensure the source has populated SMM keys for our 4 roles.
    smm_module = sys.modules.get("storage.shared_memory_manager")
    if smm_module is None:
        from storage.shared_memory_manager import get_shared_memory_manager  # type: ignore
        smm = get_shared_memory_manager()
    else:
        smm = smm_module.get_shared_memory_manager()  # type: ignore[attr-defined]

    # Sample loop — drive packets manually via the source's shared_state.
    color_samples: List[float] = []
    inference_samples: List[float] = []
    postprocess_samples: List[float] = []
    stack_samples: List[float] = []
    sync_wait_samples: List[float] = []
    resize_norm_samples: List[float] = []
    total_samples: List[float] = []
    drops = 0
    started = time.perf_counter()
    warmup = int(runtime_cfg.get("warmup_packets", 30))
    pkts = 0
    zero_copy_seen = False
    color_out_loc = "?"
    inf_in_loc = "?"

    try:
        while time.perf_counter() - started < duration_s:
            t_iter = time.perf_counter()
            keys_map = {r: source.shared_state.get("bayer_key", {}).get(r, "")
                        for r in cfg.required_roles}
            if not all(keys_map.values()):
                drops += 1
                time.sleep(0.005)
                continue
            t_read = time.perf_counter()
            frames = []
            ts_list = []
            ok_all = True
            for r in cfg.required_roles:
                read = smm.read_frame(keys_map[r], retries=1, delay=0.0005)
                if read is None:
                    ok_all = False
                    break
                frames.append(read[0])
                ts_list.append(int(read[1] or 0))
            sync_wait_ms = (time.perf_counter() - t_read) * 1000.0
            if not ok_all:
                drops += 1
                continue

            res = executor.process_packet(frames, list(cfg.required_roles), max(ts_list))
            if res.error:
                drops += 1
                base_row["errors"].append(res.error)
                continue

            pkts += 1
            if pkts <= warmup:
                continue
            stage_ms = res.stage_ms
            color_samples.append(float(stage_ms.get("color_ms", 0.0)))
            inference_samples.append(float(stage_ms.get("inference_ms", 0.0)))
            postprocess_samples.append(float(stage_ms.get("postprocess_ms", 0.0)))
            stack_samples.append(float(stage_ms.get("stack_ms", 0.0)))
            sync_wait_samples.append(float(sync_wait_ms))
            resize_norm_samples.append(0.0)  # bundled into color_ms in our adapter
            total_samples.append(float(stage_ms.get("total_packet_ms", 0.0)))
            zero_copy_seen = zero_copy_seen or bool(res.zero_copy_to_inference)
            color_out_loc = res.color_output_location or color_out_loc
            inf_in_loc = res.inference_input_location or inf_in_loc
    except Exception as exc:
        base_row["errors"].append(f"runtime exception: {exc!r}")
        traceback.print_exc()

    if not total_samples:
        base_row["status"] = "ERROR"
        base_row["errors"].append("no measured packets")
        return _empty_metrics(base_row, pkt_budget)

    med = _median(total_samples)
    p95 = _percentile(total_samples, 95.0)
    fps_med = (1000.0 / med) if med > 0 else 0.0
    fps_p95 = (1000.0 / p95) if p95 > 0 else 0.0

    base_row.update({
        "color_ms_median": _median(color_samples),
        "color_ms_p95": _percentile(color_samples, 95.0),
        "resize_or_letterbox_ms_median": _median(resize_norm_samples),
        "resize_or_letterbox_ms_p95": _percentile(resize_norm_samples, 95.0),
        "inference_ms_median": _median(inference_samples),
        "inference_ms_p95": _percentile(inference_samples, 95.0),
        "postprocess_ms_median": _median(postprocess_samples),
        "postprocess_ms_p95": _percentile(postprocess_samples, 95.0),
        "stack_ms_median": _median(stack_samples),
        "stack_ms_p95": _percentile(stack_samples, 95.0),
        "sync_wait_ms_median": _median(sync_wait_samples),
        "sync_wait_ms_p95": _percentile(sync_wait_samples, 95.0),
        "total_packet_ms_median": med,
        "total_packet_ms_p95": p95,
        "fps_per_camera_median": fps_med,
        "fps_per_camera_safe_p95": fps_p95,
        "total_images_per_second_median": fps_med * 4.0,
        "total_images_per_second_safe_p95": fps_p95 * 4.0,
        "pass_50fps_median": med <= pkt_budget,
        "pass_50fps_safe_p95": p95 <= pkt_budget,
        "stability_margin_ms": pkt_budget - p95,
        "zero_copy_to_inference": zero_copy_seen,
        "gpu_roundtrip": not zero_copy_seen,
        "fallback_used": "false" if zero_copy_seen else "true",
        "color_output_location": color_out_loc,
        "inference_input_location": inf_in_loc,
        "drops": drops,
        "samples": len(total_samples),
    })
    return base_row


def _empty_metrics(row: Dict[str, Any], pkt_budget: float) -> Dict[str, Any]:
    row.update({
        "color_ms_median": None,
        "color_ms_p95": None,
        "resize_or_letterbox_ms_median": None,
        "resize_or_letterbox_ms_p95": None,
        "inference_ms_median": None,
        "inference_ms_p95": None,
        "postprocess_ms_median": None,
        "postprocess_ms_p95": None,
        "stack_ms_median": None,
        "stack_ms_p95": None,
        "sync_wait_ms_median": None,
        "sync_wait_ms_p95": None,
        "total_packet_ms_median": None,
        "total_packet_ms_p95": None,
        "fps_per_camera_median": 0.0,
        "fps_per_camera_safe_p95": 0.0,
        "total_images_per_second_median": 0.0,
        "total_images_per_second_safe_p95": 0.0,
        "pass_50fps_median": False,
        "pass_50fps_safe_p95": False,
        "stability_margin_ms": -pkt_budget,
        "samples": 0,
    })
    return row


# ---------------------------------------------------------------- main

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fast-path benchmark sweep")
    parser.add_argument("--mode", choices=("synthetic", "basler"), default="synthetic")
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--camera-roles", type=Path, default=PACKAGE_ROOT / "configs" / "camera_roles.json")
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "configs" / "benchmark_config.yaml")
    parser.add_argument("--probe", action="store_true", default=True)
    parser.add_argument("--no-probe", dest="probe", action="store_false")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    cfg = _load_config()
    target = cfg.get("target", {})
    runtime = cfg.get("runtime", {})
    duration_s = float(args.duration_s if args.duration_s is not None
                       else runtime.get("duration_s", 60.0))
    sweep = cfg.get("sweep", [])
    if not sweep:
        raise SystemExit("config has no sweep entries")

    if args.probe:
        os.environ.setdefault("VOLLEYHUB_LIVE_PROBE", "1")
        os.environ.setdefault("VOLLEYHUB_LIVE_PROBE_EVERY",
                              str(int(runtime.get("report_every", 30))))
        os.environ.setdefault("VOLLEYHUB_LIVE_PROBE_LOGDIR", str(PACKAGE_ROOT / "logs"))

    capture_w = int(target.get("capture_resolution", [1920, 1080])[0])
    capture_h = int(target.get("capture_resolution", [1920, 1080])[1])
    bayer_pattern = str(runtime.get("bayer_pattern", "RG"))

    # Set up source.
    if args.mode == "synthetic":
        src = _SyntheticSource(capture_w, capture_h, target_fps=50.0, bayer_pattern=bayer_pattern)
        source_type = "synthetic"
    else:
        try:
            import pypylon  # type: ignore  # noqa: F401
        except Exception:
            print("PYPYLON_MISSING — install pypylon + pylon SDK to run --mode basler.")
            return 2
        src = _BaslerSource(args.camera_roles, capture_w, capture_h, bayer_pattern=bayer_pattern)
        source_type = "basler"

    src.start()
    _LOG.info("source started: %s (%dx%d)", source_type, capture_w, capture_h)

    rows: List[Dict[str, Any]] = []
    try:
        for entry in sweep:
            name = str(entry.get("name", "?"))
            imgsz = entry.get("inference_input_shape", 640)
            engine_rel = str(entry.get("engine_path", ""))
            engine_path = (PACKAGE_ROOT / engine_rel).resolve()
            _LOG.info("=== variant %s | imgsz=%s | engine=%s",
                      name, imgsz, engine_path.name)
            row = _run_variant(name, imgsz, engine_path, src,
                               runtime_cfg=runtime, target_cfg=target,
                               duration_s=duration_s, source_type=source_type)
            rows.append(row)
            _LOG.info("=== variant %s done. status=%s packet_p95=%s",
                      name, row.get("status"),
                      f"{row.get('total_packet_ms_p95'):.2f} ms" if row.get('total_packet_ms_p95') else "—")
    finally:
        try:
            src.stop()
        except Exception:
            pass

    out_dir = PACKAGE_ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_type": source_type,
        "package_root": str(PACKAGE_ROOT),
        "config": cfg,
        "machine": _machine_info(),
        "results": rows,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    _write_csv(out_dir / "results.csv", rows)
    _LOG.info("wrote: %s", out_dir / "results.json")
    _LOG.info("wrote: %s", out_dir / "results.csv")
    return 0


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in keys})


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
