# file: vision_runtime/apps/run_folder_inference.py
from __future__ import annotations

import json
from pathlib import Path

from vision_runtime.core.pipeline_runner import OfflinePipelineRunner, RunnerConfig


BASE_DIR = Path(__file__).resolve().parents[2]

# Ustaw tutaj folder z obrazami testowymi
INPUT_DIR = BASE_DIR / "sample_frames"

# Tu zapiszą się overlaye i summary.json
OUTPUT_DIR = BASE_DIR / "outputs"

# "ultralytics" albo "tensorrt"
DETECTOR_BACKEND = "ultralytics"

# Dla ultralytics
MODEL_PATH = str(BASE_DIR / "best.pt")

# Dla TensorRT
TRT_ENGINE_PATH = str(BASE_DIR / "model.engine")

DEVICE = "cuda"
IMAGE_SIZE = 640
CONFIDENCE = 0.25

# "cpu_opencv" albo "future_cuda_npp"
DEBAYER_BACKEND = "cpu_opencv"
BAYER_PATTERN = "BG"

SAVE_OVERLAYS = True
SAVE_JSON = True


def _validate_config(config: RunnerConfig) -> None:
    if not config.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {config.input_dir}")

    if not config.input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {config.input_dir}")

    backend = str(config.detector_backend).strip().lower()
    if backend not in {"ultralytics", "tensorrt"}:
        raise ValueError(f"Unsupported detector backend: {config.detector_backend}")

    if backend == "ultralytics":
        model_path = Path(config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Ultralytics model file not found: {model_path}")

    if backend == "tensorrt":
        engine_path = Path(config.trt_engine_path)
        if not engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine file not found: {engine_path}")
        if engine_path.suffix.lower() != ".engine":
            raise ValueError(f"TensorRT engine must end with .engine: {engine_path}")

    if int(config.image_size) <= 0:
        raise ValueError(f"image_size must be > 0, got {config.image_size}")

    if not (0.0 <= float(config.confidence) <= 1.0):
        raise ValueError(f"confidence must be in [0, 1], got {config.confidence}")

    if str(config.debayer_backend).strip().lower() not in {"cpu_opencv", "future_cuda_npp"}:
        raise ValueError(f"Unsupported debayer backend: {config.debayer_backend}")

    if str(config.bayer_pattern).strip().upper() not in {"RG", "BG", "GR", "GB"}:
        raise ValueError(f"Unsupported Bayer pattern: {config.bayer_pattern}")


def _print_header(config: RunnerConfig) -> None:
    print("=" * 72)
    print("VISION RUNTIME OFFLINE RUNNER")
    print("=" * 72)
    print(f"Input dir         : {config.input_dir}")
    print(f"Output dir        : {config.output_dir}")
    print(f"Detector backend  : {config.detector_backend}")
    print(f"Model path        : {config.model_path}")
    print(f"TensorRT engine   : {config.trt_engine_path}")
    print(f"Device            : {config.device}")
    print(f"Image size        : {config.image_size}")
    print(f"Confidence        : {config.confidence}")
    print(f"Debayer backend   : {config.debayer_backend}")
    print(f"Bayer pattern     : {config.bayer_pattern}")
    print(f"Save overlays     : {config.save_overlays}")
    print(f"Save JSON         : {config.save_json}")
    print("=" * 72)


def _print_summary(summary: dict) -> None:
    print("\nRUN SUMMARY")
    print("-" * 72)
    print(f"frames_total      : {summary.get('frames_total', 0)}")
    print(f"frames_processed  : {summary.get('frames_processed', 0)}")
    print(f"detections_total  : {summary.get('detections_total', 0)}")
    print(f"tracks_total      : {summary.get('tracks_total', 0)}")
    print(f"avg_load_ms       : {summary.get('avg_load_ms', 0.0):.3f}")
    print(f"avg_convert_ms    : {summary.get('avg_convert_ms', 0.0):.3f}")
    print(f"avg_preprocess_ms : {summary.get('avg_preprocess_ms', 0.0):.3f}")
    print(f"avg_infer_ms      : {summary.get('avg_infer_ms', 0.0):.3f}")
    print(f"avg_track_ms      : {summary.get('avg_track_ms', 0.0):.3f}")
    print(f"avg_total_ms      : {summary.get('avg_total_ms', 0.0):.3f}")
    print("-" * 72)

    if summary.get("frames_processed", 0):
        fps = 1000.0 / float(summary.get("avg_total_ms", 0.0)) if float(summary.get("avg_total_ms", 0.0)) > 0 else 0.0
        print(f"estimated_fps     : {fps:.2f}")
        print("-" * 72)

    output_dir = summary.get("config", {}).get("output_dir")
    if output_dir:
        print(f"outputs saved in  : {output_dir}")


def _write_console_summary(config: RunnerConfig, summary: dict) -> None:
    console_summary_path = Path(config.output_dir) / "console_summary.json"
    console_summary_path.parent.mkdir(parents=True, exist_ok=True)
    console_summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"console_summary   : {console_summary_path}")


def build_config() -> RunnerConfig:
    return RunnerConfig(
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


def main() -> None:
    config = build_config()
    _validate_config(config)
    _print_header(config)

    runner = OfflinePipelineRunner(config)
    summary = runner.run()

    _print_summary(summary)
    _write_console_summary(config, summary)


if __name__ == "__main__":
    main()