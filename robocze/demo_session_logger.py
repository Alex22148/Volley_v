# demo_session_logger.py
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


def _now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _safe_name(text: str | None, fallback: str = "unnamed") -> str:
    s = str(text or "").strip()
    if not s:
        s = fallback
    s = re.sub(r'[<>:"/\\|?*]+', "_", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("._") or fallback


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _read_last_results_row(results_csv_path: Path) -> dict[str, Any] | None:
    try:
        if not results_csv_path.exists():
            return None
        with results_csv_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return None
        return rows[-1]
    except Exception:
        return None


def resolve_model_artifacts(model_path: str | Path | None) -> dict[str, Any]:
    """
    Dla:
      .../train_xxx/ultralytics/weights/best.pt
    zwraca:
      run_dir     = .../train_xxx
      meta_json   = .../train_xxx/meta.json
      args_yaml   = .../train_xxx/args.yaml
      results_csv = .../train_xxx/results.csv
      run_name    = train_xxx
    """
    out = {
        "model_path": str(model_path) if model_path else None,
        "run_dir": None,
        "run_name": None,
        "meta_json_path": None,
        "args_yaml_path": None,
        "results_csv_path": None,
        "meta_json": None,
        "results_last_row": None,
    }

    if not model_path:
        return out

    p = Path(model_path)
    out["model_path"] = str(p)

    try:
        run_dir = p.parent.parent.parent
        if not run_dir.exists():
            return out

        out["run_dir"] = str(run_dir)
        out["run_name"] = run_dir.name

        meta_json = run_dir / "meta.json"
        args_yaml = run_dir / "args.yaml"
        results_csv = run_dir / "results.csv"

        out["meta_json_path"] = str(meta_json)
        out["args_yaml_path"] = str(args_yaml)
        out["results_csv_path"] = str(results_csv)

        out["meta_json"] = _load_json_if_exists(meta_json)
        out["results_last_row"] = _read_last_results_row(results_csv)

    except Exception:
        pass

    return out


class DemoSessionLogger:
    def __init__(self, sessions_dir: str | Path):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._current_file: Path | None = None
        self._session_data: dict[str, Any] | None = None

    @property
    def current_file(self) -> Path | None:
        return self._current_file

    def start_session(
        self,
        *,
        model_path: str | None,
        confidence: float | None,
        imgsz: int | None,
        device: str | None,
        active_roles: list[str] | None,
        sync_state: str | None,
        recordings_path: str | None,
        test_name: str | None,
        operator_name: str | None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        with self._lock:
            model_info = resolve_model_artifacts(model_path)
            run_name = _safe_name(model_info.get("run_name"), fallback="unknown_run")
            test_part = _safe_name(test_name, fallback="demo")
            ts_part = _now_stamp()

            filename = f"{run_name}__{test_part}__{ts_part}.json"
            self._current_file = self.sessions_dir / filename

            self._session_data = {
                "session_id": ts_part,
                "started_at": _now_iso(),
                "stopped_at": None,
                "status": "running",
                "context": {
                    "test_name": test_name,
                    "operator_name": operator_name,
                    "recordings_path": recordings_path,
                },
                "model": model_info,
                "settings_at_start": {
                    "confidence_threshold": confidence,
                    "imgsz": imgsz,
                    "device": device,
                    "active_roles": list(active_roles or []),
                    "sync_state": sync_state,
                },
                "pipeline_summary": {
                    "batches": 0,
                    "frames_in": 0,
                    "frames_valid": 0,
                    "frames_processed": 0,
                    "detections_total": 0,
                    "valid_2d_points_total": 0,
                    "points3D_total": 0,
                    "avg_prepare_ms": 0.0,
                    "avg_forward_ms": 0.0,
                    "avg_post_ms": 0.0,
                    "avg_infer_total_ms": 0.0,
                    "avg_batch_collect_ms": 0.0,
                    "avg_queue_wait_ms": 0.0,
                    "avg_get3D_ms": 0.0,
                    "avg_analyze3D_ms": 0.0,
                    "avg_selection3D_ms": 0.0,
                    "_timing_samples": 0,
                    "_timing3d_samples": 0,
                },
                "batches": [],
                "extra": extra or {},
            }

            self._flush_locked()
            return self._current_file

    def log_batch(
        self,
        *,
        roles: list[str],
        batch_id: int,
        imgsz: int | None,
        device: str | None,
        confidence: float | None,
        sync_state: str | None,
        detected: int,
        in_frames: int,
        valid_frames: int,
        processed_frames: int,
        batch_collect_ms: float | None,
        queue_wait_ms: float | None,
        prepare_ms: float | None,
        forward_ms: float | None,
        post_ms: float | None,
        infer_total_ms: float | None,
        valid_2d_points: int,
        points3D: int,
        per_role_counts: dict[str, Any] | None,
        per_step_counts: list[dict[str, Any]] | None,
        timing_3d: dict[str, Any] | None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            if self._session_data is None or self._current_file is None:
                return

            batch = {
                "ts": _now_iso(),
                "batch_id": batch_id,
                "roles": list(roles or []),
                "settings": {
                    "imgsz": imgsz,
                    "device": device,
                    "confidence_threshold": confidence,
                    "sync_state": sync_state,
                },
                "counts": {
                    "detected": detected,
                    "in_frames": in_frames,
                    "valid_frames": valid_frames,
                    "processed_frames": processed_frames,
                    "valid_2d_points": valid_2d_points,
                    "points3D": points3D,
                    "per_role_counts": per_role_counts or {},
                    "per_step_counts": per_step_counts or [],
                },
                "timings_ms": {
                    "batch_collect_ms": batch_collect_ms,
                    "queue_wait_ms": queue_wait_ms,
                    "prepare_ms": prepare_ms,
                    "forward_ms": forward_ms,
                    "post_ms": post_ms,
                    "infer_total_ms": infer_total_ms,
                    "undistort_ms": (timing_3d or {}).get("undistort"),
                    "get3D_ms": (timing_3d or {}).get("get3D"),
                    "analyze_3D_ms": (timing_3d or {}).get("analyze_3D"),
                    "selection_3D_ms": (timing_3d or {}).get("selection_3D"),
                },
                "extra": extra or {},
            }

            self._session_data["batches"].append(batch)
            self._update_summary_locked(batch)
            self._flush_locked()

    def stop_session(self, *, reason: str = "stop_detection") -> None:
        with self._lock:
            if self._session_data is None:
                return

            self._session_data["stopped_at"] = _now_iso()
            self._session_data["status"] = "stopped"
            self._session_data["stop_reason"] = reason

            summary = self._session_data.get("pipeline_summary", {})
            summary.pop("_timing_samples", None)
            summary.pop("_timing3d_samples", None)

            self._flush_locked()
            self._current_file = None
            self._session_data = None

    def _update_summary_locked(self, batch: dict[str, Any]) -> None:
        s = self._session_data["pipeline_summary"]
        c = batch["counts"]
        t = batch["timings_ms"]

        s["batches"] += 1
        s["frames_in"] += int(c.get("in_frames") or 0)
        s["frames_valid"] += int(c.get("valid_frames") or 0)
        s["frames_processed"] += int(c.get("processed_frames") or 0)
        s["detections_total"] += int(c.get("detected") or 0)
        s["valid_2d_points_total"] += int(c.get("valid_2d_points") or 0)
        s["points3D_total"] += int(c.get("points3D") or 0)

        sample_count = s.get("_timing_samples", 0) + 1
        s["_timing_samples"] = sample_count

        for src_key, dst_key in [
            ("prepare_ms", "avg_prepare_ms"),
            ("forward_ms", "avg_forward_ms"),
            ("post_ms", "avg_post_ms"),
            ("infer_total_ms", "avg_infer_total_ms"),
            ("batch_collect_ms", "avg_batch_collect_ms"),
            ("queue_wait_ms", "avg_queue_wait_ms"),
        ]:
            val = t.get(src_key)
            if val is None:
                val = 0.0
            prev = float(s.get(dst_key, 0.0))
            s[dst_key] = prev + (float(val) - prev) / sample_count

        timing3d_count = s.get("_timing3d_samples", 0) + 1
        s["_timing3d_samples"] = timing3d_count

        for src_key, dst_key in [
            ("get3D_ms", "avg_get3D_ms"),
            ("analyze_3D_ms", "avg_analyze3D_ms"),
            ("selection_3D_ms", "avg_selection3D_ms"),
        ]:
            val = t.get(src_key)
            if val is None:
                val = 0.0
            prev = float(s.get(dst_key, 0.0))
            s[dst_key] = prev + (float(val) - prev) / timing3d_count

    def _flush_locked(self) -> None:
        if self._current_file is None or self._session_data is None:
            return
        self._current_file.parent.mkdir(parents=True, exist_ok=True)
        self._current_file.write_text(
            json.dumps(self._session_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


_demo_logger: DemoSessionLogger | None = None


def init_demo_logger(sessions_dir: str | Path) -> DemoSessionLogger:
    global _demo_logger
    _demo_logger = DemoSessionLogger(sessions_dir)
    return _demo_logger


def get_demo_logger() -> DemoSessionLogger | None:
    return _demo_logger
