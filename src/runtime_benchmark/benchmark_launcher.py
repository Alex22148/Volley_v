"""Subprocess launcher for runtime benchmark entrypoints.

The main GUI should not import benchmark execution logic directly.  This
module keeps that boundary small: it discovers an available runtime benchmark
entrypoint, builds a ``python -m ...`` command, and starts it without blocking
the caller.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence


DEFAULT_ENTRYPOINTS: tuple[str, ...] = (
    "src.runtime_benchmark.run_4cam_raw_inference_benchmark",
)


class BenchmarkLaunchError(RuntimeError):
    """Raised when no runtime benchmark entrypoint can be launched."""


@dataclass(frozen=True)
class BenchmarkLaunchResult:
    process: subprocess.Popen
    command: list[str]
    entrypoint: str


def find_benchmark_entrypoint(
    candidates: Sequence[str] = DEFAULT_ENTRYPOINTS,
) -> Optional[str]:
    for module_name in candidates:
        if importlib.util.find_spec(module_name) is not None:
            return module_name
    return None


def _append_option(command: list[str], flag: str, value: object | None) -> None:
    if value is None:
        return
    command.extend([flag, str(value)])


def build_benchmark_command(
    *,
    entrypoint: str | None = None,
    python_executable: str | None = None,
    source: str = "synthetic",
    model_path: str | Path | None = None,
    output_dir: str | Path,
    width: int = 2464,
    height: int = 2056,
    fps: float = 77.0,
    num_frames: int = 1000,
    batch_size: int = 4,
    imgsz: int = 640,
    device: str = "cpu",
    color_backend: str = "cpu",
    inference_backend: str = "dry_run",
    pipeline_mode: str = "cpu",
    dry_run: bool = True,
    extra_args: Iterable[str] = (),
) -> list[str]:
    module_name = entrypoint or find_benchmark_entrypoint()
    if module_name is None:
        raise BenchmarkLaunchError(
            "No runtime benchmark entrypoint found in src.runtime_benchmark"
        )

    command = [python_executable or sys.executable, "-m", module_name]
    _append_option(command, "--source", source)
    _append_option(command, "--model-path", model_path)
    _append_option(command, "--width", int(width))
    _append_option(command, "--height", int(height))
    _append_option(command, "--fps", float(fps))
    _append_option(command, "--num-frames", int(num_frames))
    _append_option(command, "--batch-size", int(batch_size))
    _append_option(command, "--imgsz", int(imgsz))
    _append_option(command, "--device", device)
    _append_option(command, "--color-backend", color_backend)
    _append_option(command, "--inference-backend", inference_backend)
    _append_option(command, "--pipeline-mode", pipeline_mode)
    _append_option(command, "--output-dir", output_dir)
    if dry_run:
        command.append("--dry-run")
    command.extend(str(arg) for arg in extra_args)
    return command


def launch_benchmark(
    *,
    cwd: str | Path | None = None,
    stdout=None,
    stderr=None,
    creationflags: int = 0,
    **command_options,
) -> BenchmarkLaunchResult:
    entrypoint = command_options.get("entrypoint") or find_benchmark_entrypoint()
    if entrypoint is None:
        raise BenchmarkLaunchError(
            "No runtime benchmark entrypoint found in src.runtime_benchmark"
        )
    command_options["entrypoint"] = entrypoint
    command = build_benchmark_command(**command_options)
    process = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd is not None else None,
        stdout=stdout,
        stderr=stderr,
        creationflags=creationflags,
    )
    return BenchmarkLaunchResult(
        process=process,
        command=command,
        entrypoint=entrypoint,
    )
