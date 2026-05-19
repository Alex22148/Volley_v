from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(slots=True)
class RuntimeStats:
    samples_load_ms: List[float] = field(default_factory=list)
    samples_convert_ms: List[float] = field(default_factory=list)
    samples_preprocess_ms: List[float] = field(default_factory=list)
    samples_infer_ms: List[float] = field(default_factory=list)
    samples_track_ms: List[float] = field(default_factory=list)
    samples_total_ms: List[float] = field(default_factory=list)
    counters: Dict[str, int] = field(default_factory=dict)
    gauges: Dict[str, float] = field(default_factory=dict)

    def add_sample(self, name: str, value_ms: float) -> None:
        raise NotImplementedError("Implement stats aggregation here.")

    def snapshot(self) -> dict:
        raise NotImplementedError("Implement stats snapshot here.")
