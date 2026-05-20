import json
from pathlib import Path


def test_operator_smoke_module_imports():
    import importlib

    module = importlib.import_module("src.runtime_diagnostics.operator_smoke_test")
    assert hasattr(module, "run_operator_smoke_test")


def test_operator_smoke_without_preprocess_compare(tmp_path: Path):
    from src.runtime_diagnostics.operator_smoke_test import run_operator_smoke_test

    report = run_operator_smoke_test(output_dir=tmp_path, run_preprocess_compare=False)

    assert isinstance(report, dict)
    assert report["preprocessing_compare_summary"]["ran"] is False
    assert report["final_status"] in ("READY", "READY_WITH_WARNINGS", "NOT_READY")
    assert (tmp_path / "operator_smoke_report.json").exists()
    assert (tmp_path / "operator_smoke_report.md").exists()


def test_operator_smoke_with_cpu_preprocess_compare(tmp_path: Path):
    from src.runtime_diagnostics.operator_smoke_test import run_operator_smoke_test

    report = run_operator_smoke_test(
        output_dir=tmp_path,
        run_preprocess_compare=True,
        width=64,
        height=64,
        num_frames=1,
        imgsz=32,
        device="cpu",
        native_backend="cpu",
    )

    compare = report["preprocessing_compare_summary"]
    assert compare["ran"] is True
    assert compare["status"] == "OK"
    assert compare["packets_processed"] == 1
    assert compare["bchw_compatible_packets"] == 1
    assert (tmp_path / "preprocessing_compare" / "preprocessing_compare_summary.json").exists()


def test_operator_smoke_missing_engine_is_warning(tmp_path: Path):
    from src.runtime_diagnostics.operator_smoke_test import run_operator_smoke_test

    missing = tmp_path / "missing.engine"
    report = run_operator_smoke_test(
        output_dir=tmp_path,
        engine_path=missing,
        run_preprocess_compare=False,
    )

    warnings = report["environment_warnings"]
    assert any("does not exist" in w for w in warnings)
    assert report["final_status"] in ("READY_WITH_WARNINGS", "NOT_READY")


def test_operator_smoke_exports_json_and_markdown(tmp_path: Path):
    from src.runtime_diagnostics.operator_smoke_test import run_operator_smoke_test

    report = run_operator_smoke_test(output_dir=tmp_path, run_preprocess_compare=False)
    json_path = tmp_path / "operator_smoke_report.json"
    md_path = tmp_path / "operator_smoke_report.md"

    assert json_path.exists()
    assert md_path.exists()
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["final_status"] == report["final_status"]
    assert "# Operator Smoke Test" in md_path.read_text(encoding="utf-8")
