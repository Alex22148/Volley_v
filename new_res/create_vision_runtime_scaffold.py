# file: create_vision_runtime_scaffold.py
from __future__ import annotations

from pathlib import Path


TARGET_ROOT = Path(r"C:\Users\Hyperbook\python_project\VolleyHub_K\new_res")
OVERWRITE_EXISTING = False


README_CONTENT = """# Vision Runtime Scaffold

Scaffold for:
- offline folder runner
- detector adapter: ultralytics / tensorrt
- debayer adapter: cpu_opencv / future_cuda_npp
- minimal preprocess
- tracker runtime
- runtime stats
- later: live backend integration

Next steps:
1. Fill ipc/messages.py
2. Fill adapters/detector_adapter.py
3. Fill adapters/debayer_adapter.py
4. Fill core/preprocess.py
5. Fill core/pipeline_runner.py
6. Fill apps/run_folder_inference.py
"""

GITIGNORE_CONTENT = """__pycache__/
*.pyc
*.pyo
*.pyd
.venv/
venv/
.env
.idea/
.vscode/
build/
dist/
artifacts/
outputs/
logs/
*.engine
"""

CONFIG_EXAMPLE = """{
  "input_dir": "sample_frames",
  "output_dir": "outputs",
  "detector_backend": "ultralytics",
  "model_path": "best.pt",
  "trt_engine_path": "",
  "device": "cuda",
  "image_size": 640,
  "confidence": 0.25,
  "debayer_backend": "cpu_opencv",
  "bayer_pattern": "BG",
  "save_overlays": true,
  "save_json": true
}
"""

INIT_PY = '"""Package marker."""\n'

MESSAGES_PY = '''from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class FramePacket:
    frame_id: int
    source_path: str
    timestamp_host_ns: int
    timestamp_source_ns: Optional[int]
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DetectionPacket:
    bbox_xyxy: List[float]
    score: float
    class_id: int
    label: str = ""


@dataclass(slots=True)
class TrackPacket:
    track_id: int
    bbox_xyxy: List[float]
    score: float
    class_id: int
    label: str = ""


@dataclass(slots=True)
class RuntimeStatsSnapshot:
    frames_seen: int = 0
    frames_processed: int = 0
    avg_load_ms: float = 0.0
    avg_convert_ms: float = 0.0
    avg_preprocess_ms: float = 0.0
    avg_infer_ms: float = 0.0
    avg_track_ms: float = 0.0
    avg_total_ms: float = 0.0
    counters: Dict[str, int] = field(default_factory=dict)
    gauges: Dict[str, float] = field(default_factory=dict)
'''

DEBAYER_ADAPTER_PY = '''from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Protocol

import numpy as np


DebayerBackendName = Literal["cpu_opencv", "future_cuda_npp"]


@dataclass(slots=True)
class DebayerConfig:
    backend: DebayerBackendName = "cpu_opencv"
    bayer_pattern: Literal["RG", "BG", "GR", "GB"] = "BG"
    output_format: Literal["BGR"] = "BGR"
    library_path: Optional[str] = None


class DebayerAdapter(Protocol):
    def debayer(self, frame_bayer: np.ndarray) -> np.ndarray:
        ...

    def close(self) -> None:
        ...


class CpuOpenCVDebayerAdapter:
    def __init__(self, config: DebayerConfig) -> None:
        self.config = config

    def debayer(self, frame_bayer: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Implement CPU OpenCV debayer here.")

    def close(self) -> None:
        return


class FutureCudaNppDebayerAdapter:
    def __init__(self, config: DebayerConfig) -> None:
        self.config = config

    def debayer(self, frame_bayer: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Attach future C++/CUDA/NPP debayer here.")

    def close(self) -> None:
        return


def build_debayer_adapter(config: DebayerConfig) -> DebayerAdapter:
    if config.backend == "cpu_opencv":
        return CpuOpenCVDebayerAdapter(config)
    if config.backend == "future_cuda_npp":
        return FutureCudaNppDebayerAdapter(config)
    raise ValueError(f"Unsupported debayer backend: {config.backend}")
'''

DETECTOR_ADAPTER_PY = '''from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Protocol

import numpy as np

from vision_runtime.ipc.messages import DetectionPacket


DetectorBackendName = Literal["ultralytics", "tensorrt"]


@dataclass(slots=True)
class DetectorConfig:
    backend: DetectorBackendName = "ultralytics"
    model_path: str = "best.pt"
    trt_engine_path: str = ""
    device: str = "cuda"
    image_size: int = 640
    confidence: float = 0.25


class DetectorAdapter(Protocol):
    def warmup(self) -> None:
        ...

    def infer(self, frame_bgr: np.ndarray) -> List[DetectionPacket]:
        ...

    def close(self) -> None:
        ...


class UltralyticsDetectorAdapter:
    def __init__(self, config: DetectorConfig) -> None:
        self.config = config

    def warmup(self) -> None:
        return

    def infer(self, frame_bgr: np.ndarray) -> List[DetectionPacket]:
        raise NotImplementedError("Implement Ultralytics detector adapter here.")

    def close(self) -> None:
        return


class TensorRtDetectorAdapter:
    def __init__(self, config: DetectorConfig) -> None:
        self.config = config

    def warmup(self) -> None:
        return

    def infer(self, frame_bgr: np.ndarray) -> List[DetectionPacket]:
        raise NotImplementedError("Implement TensorRT detector adapter here.")

    def close(self) -> None:
        return


def build_detector_adapter(config: DetectorConfig) -> DetectorAdapter:
    if config.backend == "ultralytics":
        return UltralyticsDetectorAdapter(config)
    if config.backend == "tensorrt":
        return TensorRtDetectorAdapter(config)
    raise ValueError(f"Unsupported detector backend: {config.backend}")
'''

IMAGE_SOURCE_PY = '''from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List

import time


@dataclass(slots=True)
class ImageItem:
    frame_id: int
    path: Path
    timestamp_host_ns: int


class FolderImageSource:
    def __init__(self, input_dir: Path, patterns: List[str] | None = None) -> None:
        self.input_dir = input_dir
        self.patterns = patterns or ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]

    def list_files(self) -> List[Path]:
        files: List[Path] = []
        for pattern in self.patterns:
            files.extend(self.input_dir.glob(pattern))
        return sorted(set(files))

    def iter_items(self) -> Iterator[ImageItem]:
        for idx, path in enumerate(self.list_files(), start=1):
            yield ImageItem(
                frame_id=idx,
                path=path,
                timestamp_host_ns=time.time_ns(),
            )
'''

PREPROCESS_PY = '''from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np


@dataclass(slots=True)
class PreprocessConfig:
    image_size: int = 640
    input_color: Literal["BGR", "RGB"] = "BGR"
    normalize_01: bool = True


def minimal_preprocess(frame_bgr: np.ndarray, config: PreprocessConfig) -> Tuple[np.ndarray, dict]:
    raise NotImplementedError("Implement minimal preprocess here.")
'''

TRACKER_RUNTIME_PY = '''from __future__ import annotations

from dataclasses import dataclass
from typing import List

from vision_runtime.ipc.messages import DetectionPacket, TrackPacket


@dataclass(slots=True)
class TrackerConfig:
    max_age_frames: int = 8
    iou_threshold: float = 0.3


class SimpleTrackerRuntime:
    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self._next_track_id = 1

    def reset(self) -> None:
        self._next_track_id = 1

    def update(self, detections: List[DetectionPacket]) -> List[TrackPacket]:
        raise NotImplementedError("Implement tracker update here.")
'''

RUNTIME_STATS_PY = '''from __future__ import annotations

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
'''

PIPELINE_RUNNER_PY = '''from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class RunnerConfig:
    input_dir: Path
    output_dir: Path
    detector_backend: str = "ultralytics"
    model_path: str = "best.pt"
    trt_engine_path: str = ""
    device: str = "cuda"
    image_size: int = 640
    confidence: float = 0.25
    debayer_backend: str = "cpu_opencv"
    bayer_pattern: str = "BG"
    save_overlays: bool = True
    save_json: bool = True


class OfflinePipelineRunner:
    def __init__(self, config: RunnerConfig) -> None:
        self.config = config

    def run(self) -> dict:
        raise NotImplementedError("Implement offline folder pipeline runner here.")
'''

RUN_FOLDER_INFERENCE_PY = '''from __future__ import annotations

from pathlib import Path

from vision_runtime.core.pipeline_runner import OfflinePipelineRunner, RunnerConfig


INPUT_DIR = Path("sample_frames")
OUTPUT_DIR = Path("outputs")
DETECTOR_BACKEND = "ultralytics"   # "ultralytics" or "tensorrt"
MODEL_PATH = "best.pt"
TRT_ENGINE_PATH = ""
DEVICE = "cuda"
IMAGE_SIZE = 640
CONFIDENCE = 0.25
DEBAYER_BACKEND = "cpu_opencv"     # "cpu_opencv" or "future_cuda_npp"
BAYER_PATTERN = "BG"
SAVE_OVERLAYS = True
SAVE_JSON = True


def main() -> None:
    config = RunnerConfig(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        detector_backend=DETECTOR_BACKEND,
        model_path=MODEL_PATH,
        trt_engine_path=TRT_ENGINE_PATH,
        device=DEVICE,
        image_size=IMAGE_SIZE,
        confidence=CONFIDENCE,
        debayer_backend=DEBAYER_BACKEND,
        bayer_pattern=BAYER_PATTERN,
        save_overlays=SAVE_OVERLAYS,
        save_json=SAVE_JSON,
    )
    runner = OfflinePipelineRunner(config)
    summary = runner.run()
    print(summary)


if __name__ == "__main__":
    main()
'''

LIVE_BACKEND_BRIDGE_PY = '''from __future__ import annotations

"""
Placeholder bridge for future reuse of the same adapters inside the live backend.
Keep this module thin. Live-specific threading / IPC should stay outside the core runtime.
"""
'''

TEST_SMOKE_PY = '''from __future__ import annotations

from pathlib import Path


def test_scaffold_imports() -> None:
    import vision_runtime  # noqa: F401
    import vision_runtime.adapters  # noqa: F401
    import vision_runtime.core  # noqa: F401
    import vision_runtime.apps  # noqa: F401
    assert Path(__file__).exists()
'''


FILES = {
    "README.md": README_CONTENT,
    ".gitignore": GITIGNORE_CONTENT,
    "configs/example_offline_config.json": CONFIG_EXAMPLE,
    "vision_runtime/__init__.py": INIT_PY,
    "vision_runtime/adapters/__init__.py": INIT_PY,
    "vision_runtime/core/__init__.py": INIT_PY,
    "vision_runtime/apps/__init__.py": INIT_PY,
    "vision_runtime/ipc/__init__.py": INIT_PY,
    "vision_runtime/bridges/__init__.py": INIT_PY,
    "tests/__init__.py": INIT_PY,
    "vision_runtime/ipc/messages.py": MESSAGES_PY,
    "vision_runtime/adapters/debayer_adapter.py": DEBAYER_ADAPTER_PY,
    "vision_runtime/adapters/detector_adapter.py": DETECTOR_ADAPTER_PY,
    "vision_runtime/adapters/image_source.py": IMAGE_SOURCE_PY,
    "vision_runtime/core/preprocess.py": PREPROCESS_PY,
    "vision_runtime/core/tracker_runtime.py": TRACKER_RUNTIME_PY,
    "vision_runtime/core/runtime_stats.py": RUNTIME_STATS_PY,
    "vision_runtime/core/pipeline_runner.py": PIPELINE_RUNNER_PY,
    "vision_runtime/apps/run_folder_inference.py": RUN_FOLDER_INFERENCE_PY,
    "vision_runtime/bridges/live_backend_bridge.py": LIVE_BACKEND_BRIDGE_PY,
    "tests/test_smoke.py": TEST_SMOKE_PY,
}


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_file(path: Path, content: str, overwrite: bool) -> str:
    if path.exists() and not overwrite:
        return f"SKIP  {path}"
    ensure_parent(path)
    path.write_text(content, encoding="utf-8")
    return f"WRITE {path}"


def main() -> None:
    root = TARGET_ROOT.resolve()
    root.mkdir(parents=True, exist_ok=True)

    print(f"Creating scaffold in: {root}")
    print(f"Overwrite existing: {OVERWRITE_EXISTING}")

    for rel_path, content in FILES.items():
        abs_path = root / rel_path
        print(write_file(abs_path, content, OVERWRITE_EXISTING))

    print("\\nDone.")
    print("Now open the generated structure and start filling the modules.")


if __name__ == "__main__":
    main()