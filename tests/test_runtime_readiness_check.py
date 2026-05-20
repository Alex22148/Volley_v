import json
from pathlib import Path


def test_runtime_readiness_module_imports():
    import importlib

    module = importlib.import_module("src.runtime_diagnostics.runtime_readiness_check")
    assert hasattr(module, "check_runtime_readiness")


def test_check_runtime_readiness_returns_expected_sections():
    from src.runtime_diagnostics.runtime_readiness_check import check_runtime_readiness

    report = check_runtime_readiness()

    assert isinstance(report, dict)
    for key in ("python", "packages", "cuda", "modules", "engine", "status"):
        assert key in report
    assert "recommendation" in report["status"]


def test_missing_engine_path_is_warning_not_error():
    from src.runtime_diagnostics.runtime_readiness_check import check_runtime_readiness

    report = check_runtime_readiness()

    assert report["engine"]["provided"] is False
    assert report["engine"]["warning"]
    assert "error" not in report["engine"] or not report["engine"]["error"]


def test_nonexistent_engine_path_is_warning_not_exception(tmp_path: Path):
    from src.runtime_diagnostics.runtime_readiness_check import check_runtime_readiness

    missing = tmp_path / "missing.engine"
    report = check_runtime_readiness(missing)

    assert report["engine"]["provided"] is True
    assert report["engine"]["exists"] is False
    assert "does not exist" in report["engine"]["warning"]
    assert report["status"]["recommendation"] in ("READY_WITH_WARNINGS", "NOT_READY")


def test_readiness_json_and_markdown_exports(tmp_path: Path):
    from src.runtime_diagnostics.runtime_readiness_check import (
        check_runtime_readiness,
        write_readiness_json,
        write_readiness_markdown,
    )

    report = check_runtime_readiness()
    json_path = write_readiness_json(report, tmp_path)
    md_path = write_readiness_markdown(report, tmp_path)

    assert json_path.exists()
    assert md_path.exists()
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert "status" in loaded
    md = md_path.read_text(encoding="utf-8")
    assert "# Runtime Readiness Check" in md
    assert "Status:" in md
