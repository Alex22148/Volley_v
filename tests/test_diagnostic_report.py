import json
from pathlib import Path


def test_diagnostic_report_module_imports():
    import importlib

    module = importlib.import_module("src.runtime_diagnostics.diagnostic_report")
    assert hasattr(module, "run_diagnostic_report")


def test_diagnostic_report_without_engine_does_not_raise(tmp_path: Path):
    from src.runtime_diagnostics.diagnostic_report import run_diagnostic_report

    report = run_diagnostic_report(output_dir=tmp_path)

    assert isinstance(report, dict)
    for key in ("final_status", "checks", "warnings", "artifacts", "operator_checklist"):
        assert key in report


def test_hardware_smoke_without_engine_is_not_failed(tmp_path: Path):
    from src.runtime_diagnostics.diagnostic_report import run_diagnostic_report

    report = run_diagnostic_report(
        output_dir=tmp_path,
        run_hardware_smoke=True,
        skip_inference=True,
        width=64,
        height=64,
        num_frames=1,
        imgsz=32,
        device="cpu",
    )

    assert report["checks"]["hardware_smoke"]["status"] in ("SKIPPED", "PASSED_WITH_WARNINGS", "PASSED")
    assert report["checks"]["hardware_smoke"]["status"] != "FAILED"


def test_diagnostic_report_exports_json_and_markdown(tmp_path: Path):
    from src.runtime_diagnostics.diagnostic_report import run_diagnostic_report

    report = run_diagnostic_report(output_dir=tmp_path)
    json_path = tmp_path / "diagnostic_report.json"
    md_path = tmp_path / "diagnostic_report.md"

    assert json_path.exists()
    assert md_path.exists()
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["final_status"] == report["final_status"]
    assert "# VolleyHub Diagnostic Report" in md_path.read_text(encoding="utf-8")


def test_diagnostic_report_with_preprocess_compare_cpu(tmp_path: Path):
    from src.runtime_diagnostics.diagnostic_report import run_diagnostic_report

    report = run_diagnostic_report(
        output_dir=tmp_path,
        run_preprocess_compare=True,
        width=64,
        height=64,
        num_frames=1,
        imgsz=32,
        device="cpu",
        native_backend="cpu",
    )

    assert report["checks"]["preprocessing_compare"]["status"] == "PASSED"


def test_diagnostic_report_with_hardware_smoke_skip_inference(tmp_path: Path):
    from src.runtime_diagnostics.diagnostic_report import run_diagnostic_report

    report = run_diagnostic_report(
        output_dir=tmp_path,
        run_hardware_smoke=True,
        skip_inference=True,
        preprocess_backend="compare",
        width=64,
        height=64,
        num_frames=1,
        imgsz=32,
        batch_size=4,
        device="cpu",
        native_backend="cpu",
    )

    assert report["checks"]["hardware_smoke"]["status"] == "SKIPPED"
    assert report["final_status"] in ("READY_WITH_WARNINGS", "READY")
