from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


def _safe_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


@dataclass
class SessionArchiveManager:
    session_dir: Optional[Path] = None
    started_at_wall: Optional[float] = None
    config: Dict[str, Any] = field(default_factory=dict)
    data_jsonl_path: Optional[Path] = None
    summary_path: Optional[Path] = None
    _data_fh: Any = None
    _records_written: int = 0
    _last_batch_id: Optional[int] = None

    def is_active(self) -> bool:
        return self.session_dir is not None and self._data_fh is not None

    def start_session(self, session_dir: str | Path, config: Dict[str, Any]) -> Path:
        self.stop_session(reason="restart_before_start")

        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.started_at_wall = time.time()
        self.config = dict(config or {})
        self.config.setdefault("session_started_at", _now_iso())

        self.data_jsonl_path = self.session_dir / "data.jsonl"
        self.summary_path = self.session_dir / "summary.json"
        self._data_fh = open(self.data_jsonl_path, "a", encoding="utf-8", buffering=1)
        self._records_written = 0
        self._last_batch_id = None

        self.write_config()
        return self.session_dir

    def write_config(self) -> None:
        if self.session_dir is None:
            return
        cfg_path = self.session_dir / "config_meas.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)

    def update_config(self, patch: Dict[str, Any]) -> None:
        if not patch:
            return
        self.config.update(patch)
        self.write_config()

    def append_batch_record(self, record: Dict[str, Any]) -> None:
        if self._data_fh is None:
            return
        row = dict(record or {})
        row.setdefault("ts_wall", _now_iso())
        self._last_batch_id = row.get("batch_id", self._last_batch_id)
        self._data_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._records_written += 1

    def stop_session(
        self,
        reason: str = "stop",
        extra_summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.session_dir is None:
            return

        duration_s = 0.0
        if self.started_at_wall is not None:
            duration_s = max(0.0, time.time() - self.started_at_wall)

        if self._data_fh is not None:
            try:
                self._data_fh.flush()
                self._data_fh.close()
            except Exception:
                pass
            self._data_fh = None

        summary = {
            "session_dir": str(self.session_dir),
            "session_started_at": self.config.get("session_started_at"),
            "session_stopped_at": _now_iso(),
            "reason": reason,
            "duration_s": round(duration_s, 3),
            "records_written": int(self._records_written),
            "last_batch_id": self._last_batch_id,
            "config": self.config,
        }
        if extra_summary:
            summary.update(extra_summary)

        if self.summary_path is not None:
            with open(self.summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

        self.session_dir = None
        self.started_at_wall = None
        self.config = {}
        self.data_jsonl_path = None
        self.summary_path = None
        self._records_written = 0
        self._last_batch_id = None