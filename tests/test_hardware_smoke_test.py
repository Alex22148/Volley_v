import json
from pathlib import Path


def test_hardware_smoke_module_imports():
    import importlib

    module = importlib.import_module("src.runtime_diagnostics.hardware_smoke_test")
    assert hasattr(module, "run_hardware_smoke_test")


def test_hardware_smoke_without_engine_does_not_raise(tmp_path: Path):
    from src.runtime_diagnostics.hardware_smoke_test import run_hardware_smoke_test

    report = run_hardware_smoke_test(
        output_dir=tmp_path,
        skip_inference=False,
        preprocess_backend="compare",
        width=64,
        height=64,
        num_frames=1,
        imgsz=32,
        device="cpu",
    )

    assert isinstance(report, dict)
    assert report["inference_smoke"]["status"] == "SKIPPED"
    assert any("engine path not provided" in w for w in report["warnings"])


def test_hardware_smoke_missing_engine_reports_warning(tmp_path: Path):
    from src.runtime_diagnostics.hardware_smoke_test import run_hardware_smoke_test

    report = run_hardware_smoke_test(
        output_dir=tmp_path,
        engine_path=tmp_path / "missing.engine",
        skip_inference=False,
        preprocess_backend="native",
        width=64,
        height=64,
        num_frames=1,
        imgsz=32,
        device="cpu",
    )

    assert report["inference_smoke"]["status"] == "SKIPPED"
    assert any("does not exist" in w for w in report["warnings"])


def test_hardware_smoke_skip_inference_cpu_synthetic(tmp_path: Path):
    from src.runtime_diagnostics.hardware_smoke_test import run_hardware_smoke_test

    report = run_hardware_smoke_test(
        output_dir=tmp_path,
        skip_inference=True,
        preprocess_backend="compare",
        width=64,
        height=64,
        num_frames=1,
        imgsz=32,
        device="cpu",
    )

    assert report["inference_smoke"]["status"] == "SKIPPED"
    assert report["preprocess_smoke"]["status"] == "OK"
    assert report["preprocess_smoke"]["bchw_compatible_packets"] == 1


def test_hardware_smoke_preprocess_backend_modes(tmp_path: Path):
    from src.runtime_diagnostics.hardware_smoke_test import run_hardware_smoke_test

    for mode in ("native", "unified", "compare"):
        report = run_hardware_smoke_test(
            output_dir=tmp_path / mode,
            skip_inference=True,
            preprocess_backend=mode,
            width=64,
            height=64,
            num_frames=1,
            imgsz=32,
            device="cpu",
        )
        assert report["preprocess_smoke"]["backend"] == mode
        assert report["preprocess_smoke"]["status"] == "OK"


def test_hardware_smoke_exports_json_and_markdown(tmp_path: Path):
    from src.runtime_diagnostics.hardware_smoke_test import run_hardware_smoke_test

    report = run_hardware_smoke_test(
        output_dir=tmp_path,
        skip_inference=True,
        preprocess_backend="compare",
        width=64,
        height=64,
        num_frames=1,
        imgsz=32,
        device="cpu",
    )
    json_path = tmp_path / "hardware_smoke_report.json"
    md_path = tmp_path / "hardware_smoke_report.md"

    assert json_path.exists()
    assert md_path.exists()
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["final_status"] == report["final_status"]
    assert "# Hardware Smoke Test" in md_path.read_text(encoding="utf-8")
