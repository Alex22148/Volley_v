from __future__ import annotations

from pathlib import Path


def test_scaffold_imports() -> None:
    import vision_runtime  # noqa: F401
    import vision_runtime.adapters  # noqa: F401
    import vision_runtime.core  # noqa: F401
    import vision_runtime.apps  # noqa: F401
    assert Path(__file__).exists()
