from __future__ import annotations

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
        self.patterns = patterns or [
            "*.png",
            "*.jpg",
            "*.jpeg",
            "*.bmp",
            "*.tif",
            "*.tiff",
            "*.webp",
        ]

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
