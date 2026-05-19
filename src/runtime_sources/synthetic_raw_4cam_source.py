"""Synthetic 4-camera RAW/Bayer source for the runtime benchmark.

Three modes:
  * generated: pure-numpy synthetic frames with a moving virtual ball
  * folder:    deterministic replay of *.npy / *.bin / *.png from disk
  * replay:    same as folder but with FPS pacing

This source intentionally does not depend on pypylon, OpenCV CUDA, or any
production-side grabber. It produces RawFrame objects that respect the
neutral data contract in base_frame_source.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .base_frame_source import (
    BaseFrameSource,
    CAMERA_ROLES,
    FrameBatch4Cam,
    RawFrame,
    SUPPORTED_BAYER_PATTERNS,
    SUPPORTED_RAW_DTYPES,
)

_LOG = logging.getLogger(__name__)

_VALID_MODES: Tuple[str, ...] = ("generated", "folder", "replay")
_RAW_FILE_EXTS: Tuple[str, ...] = (".npy", ".bin", ".raw", ".png", ".tif", ".tiff", ".bmp")


@dataclass(slots=True)
class SyntheticSourceConfig:
    """Configuration for SyntheticRaw4CamSource."""

    mode: str = "generated"
    width: int = 2464
    height: int = 2056
    fps: float = 77.0
    num_frames: int = 1000
    dtype: str = "uint8"
    bayer_pattern: str = "RG"
    noise_sigma: float = 0.0
    ball_enabled: bool = True
    ball_radius_px: int = 32
    ball_speed_px_per_frame: float = 24.0
    folder_path: Optional[Path] = None
    replicate_single_to_4cams: bool = True
    seed: int = 1234
    camera_serials: Tuple[str, str, str, str] = ("EMU-LEFT", "EMU-CENTER_L", "EMU-CENTER_R", "EMU-RIGHT")
    extra_metadata: dict = field(default_factory=dict)

    def validate(self) -> None:
        if self.mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got {self.mode!r}")
        if self.dtype not in SUPPORTED_RAW_DTYPES:
            raise ValueError(f"dtype must be one of {SUPPORTED_RAW_DTYPES}, got {self.dtype!r}")
        if self.bayer_pattern not in SUPPORTED_BAYER_PATTERNS:
            raise ValueError(
                f"bayer_pattern must be one of {SUPPORTED_BAYER_PATTERNS}, got {self.bayer_pattern!r}"
            )
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width/height must be > 0")
        if self.width % 2 != 0 or self.height % 2 != 0:
            raise ValueError("width/height must be even (Bayer mosaic requirement)")
        if self.num_frames <= 0:
            raise ValueError("num_frames must be > 0")
        if self.fps <= 0:
            raise ValueError("fps must be > 0")
        if self.mode in ("folder", "replay") and not self.folder_path:
            raise ValueError(f"mode={self.mode} requires folder_path")
        if len(self.camera_serials) != 4:
            raise ValueError("camera_serials must have exactly 4 entries")


def _bgr_to_bayer(frame_bgr: np.ndarray, pattern: str, dtype: str) -> np.ndarray:
    """Mosaic a BGR frame into a Bayer mosaic following the requested pattern.

    Pattern naming follows OpenCV BayerXX2BGR convention, where the two
    letters are the colors at (row=1, col=0) and (row=0, col=1) of the 2x2
    repeating cell in the source image expected by cv2.cvtColor.
    """
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 BGR, got shape={frame_bgr.shape}")
    b = frame_bgr[..., 0]
    g = frame_bgr[..., 1]
    r = frame_bgr[..., 2]
    h, w = frame_bgr.shape[:2]
    out = np.empty((h, w), dtype=np.uint8)

    if pattern == "RG":
        out[0::2, 0::2] = r[0::2, 0::2]
        out[0::2, 1::2] = g[0::2, 1::2]
        out[1::2, 0::2] = g[1::2, 0::2]
        out[1::2, 1::2] = b[1::2, 1::2]
    elif pattern == "BG":
        out[0::2, 0::2] = b[0::2, 0::2]
        out[0::2, 1::2] = g[0::2, 1::2]
        out[1::2, 0::2] = g[1::2, 0::2]
        out[1::2, 1::2] = r[1::2, 1::2]
    elif pattern == "GR":
        out[0::2, 0::2] = g[0::2, 0::2]
        out[0::2, 1::2] = r[0::2, 1::2]
        out[1::2, 0::2] = b[1::2, 0::2]
        out[1::2, 1::2] = g[1::2, 1::2]
    elif pattern == "GB":
        out[0::2, 0::2] = g[0::2, 0::2]
        out[0::2, 1::2] = b[0::2, 1::2]
        out[1::2, 0::2] = r[1::2, 0::2]
        out[1::2, 1::2] = g[1::2, 1::2]
    else:
        raise ValueError(f"Unsupported bayer pattern: {pattern}")

    if dtype == "uint16":
        return (out.astype(np.uint16) << 8)
    return out


def _draw_ball_bgr(
    canvas: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
    color: Tuple[int, int, int],
) -> None:
    """Soft-edge filled circle, no OpenCV dependency in the source path."""
    if radius <= 0:
        return
    h, w = canvas.shape[:2]
    x0 = max(0, cx - radius)
    x1 = min(w, cx + radius + 1)
    y0 = max(0, cy - radius)
    y1 = min(h, cy + radius + 1)
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    mask = dist2 <= radius * radius
    sub = canvas[y0:y1, x0:x1]
    sub[mask] = color


class SyntheticRaw4CamSource(BaseFrameSource):
    """Generates synchronized 4-camera Bayer packets without real cameras."""

    def __init__(self, config: SyntheticSourceConfig) -> None:
        config.validate()
        self.config = config
        self._rng = np.random.default_rng(config.seed)
        self._frame_id = 0
        self._packet_id = 0
        self._role_offsets_px: Tuple[int, int, int, int] = (-180, -60, 60, 180)
        self._folder_packets: Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = None
        if config.mode in ("folder", "replay"):
            self._folder_packets = self._preload_folder()

    @property
    def description(self) -> str:
        c = self.config
        return (
            f"SyntheticRaw4CamSource(mode={c.mode}, {c.width}x{c.height}, "
            f"{c.dtype}, bayer={c.bayer_pattern}, fps={c.fps})"
        )

    def expected_total_packets(self) -> Optional[int]:
        return int(self.config.num_frames)

    def iter_packets(self) -> Iterator[FrameBatch4Cam]:
        if self.config.mode == "generated":
            yield from self._iter_generated()
        elif self.config.mode == "folder":
            yield from self._iter_folder(pace=False)
        elif self.config.mode == "replay":
            yield from self._iter_folder(pace=True)
        else:
            raise RuntimeError(f"unreachable mode={self.config.mode}")

    def _iter_generated(self) -> Iterator[FrameBatch4Cam]:
        for _ in range(self.config.num_frames):
            packet = self._build_generated_packet()
            self._packet_id += 1
            self._frame_id += 1
            yield packet

    def _iter_folder(self, pace: bool) -> Iterator[FrameBatch4Cam]:
        if not self._folder_packets:
            raise RuntimeError("folder/replay mode but no preloaded packets")
        period = 1.0 / float(self.config.fps)
        next_deadline = time.perf_counter()
        for _ in range(self.config.num_frames):
            quad = self._folder_packets[self._packet_id % len(self._folder_packets)]
            packet = self._build_folder_packet(quad)
            self._packet_id += 1
            self._frame_id += 1
            if pace:
                next_deadline += period
                slack = next_deadline - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
            yield packet

    def _build_generated_packet(self) -> FrameBatch4Cam:
        cfg = self.config
        ts_ns = time.time_ns()
        bg_value = 32 if cfg.dtype == "uint8" else 32
        frames: List[RawFrame] = []
        ball_x, ball_y = self._ball_position()
        for role_idx, role in enumerate(CAMERA_ROLES):
            bgr = np.full((cfg.height, cfg.width, 3), bg_value, dtype=np.uint8)
            self._draw_static_pattern(bgr, role_idx)
            if cfg.ball_enabled:
                cx = int(ball_x + self._role_offsets_px[role_idx])
                cy = int(ball_y)
                _draw_ball_bgr(bgr, cx, cy, cfg.ball_radius_px, (255, 255, 255))
            if cfg.noise_sigma > 0:
                noise = self._rng.normal(0.0, float(cfg.noise_sigma), bgr.shape)
                bgr = np.clip(bgr.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
            raw = _bgr_to_bayer(bgr, cfg.bayer_pattern, cfg.dtype)
            frames.append(
                RawFrame(
                    camera_id=cfg.camera_serials[role_idx],
                    role=role,
                    frame_id=self._frame_id,
                    timestamp_ns=ts_ns,
                    image=raw,
                    width=cfg.width,
                    height=cfg.height,
                    bayer_pattern=cfg.bayer_pattern,
                    dtype=cfg.dtype,
                )
            )
        packet = FrameBatch4Cam(
            packet_id=self._packet_id,
            timestamp_ns=ts_ns,
            frames=tuple(frames),  # type: ignore[arg-type]
            metadata={"source_mode": cfg.mode, **dict(cfg.extra_metadata)},
        )
        packet.validate()
        return packet

    def _build_folder_packet(
        self, quad: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> FrameBatch4Cam:
        cfg = self.config
        ts_ns = time.time_ns()
        frames: List[RawFrame] = []
        for role_idx, role in enumerate(CAMERA_ROLES):
            raw = quad[role_idx]
            if raw.shape != (cfg.height, cfg.width):
                raise ValueError(
                    f"Folder frame for role={role} has shape {raw.shape}, "
                    f"expected ({cfg.height}, {cfg.width})"
                )
            frames.append(
                RawFrame(
                    camera_id=cfg.camera_serials[role_idx],
                    role=role,
                    frame_id=self._frame_id,
                    timestamp_ns=ts_ns,
                    image=raw,
                    width=cfg.width,
                    height=cfg.height,
                    bayer_pattern=cfg.bayer_pattern,
                    dtype=cfg.dtype,
                )
            )
        packet = FrameBatch4Cam(
            packet_id=self._packet_id,
            timestamp_ns=ts_ns,
            frames=tuple(frames),  # type: ignore[arg-type]
            metadata={"source_mode": cfg.mode, **dict(cfg.extra_metadata)},
        )
        packet.validate()
        return packet

    def _ball_position(self) -> Tuple[int, int]:
        cfg = self.config
        f = self._frame_id
        # Lissajous-ish trajectory keeps the ball on-screen with margin.
        margin = max(cfg.ball_radius_px * 2, 64)
        amp_x = (cfg.width - 2 * margin) // 2
        amp_y = (cfg.height - 2 * margin) // 2
        cx0 = cfg.width // 2
        cy0 = cfg.height // 2
        phase_x = (f * cfg.ball_speed_px_per_frame) / max(1.0, float(amp_x))
        phase_y = (f * cfg.ball_speed_px_per_frame * 0.6) / max(1.0, float(amp_y))
        return (
            int(cx0 + amp_x * np.sin(phase_x)),
            int(cy0 + amp_y * np.sin(phase_y)),
        )

    def _draw_static_pattern(self, canvas: np.ndarray, role_idx: int) -> None:
        w = canvas.shape[1]
        stripe_w = max(8, w // 64)
        # subtle vertical stripes per role so frames are visually distinct
        for x in range(0, w, stripe_w * 4):
            canvas[:, x : x + stripe_w] = (24 + role_idx * 6,) * 3

    def _preload_folder(self) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        cfg = self.config
        if cfg.folder_path is None:
            raise RuntimeError("folder_path is None for folder/replay mode")
        folder = Path(cfg.folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder does not exist: {folder}")
        files = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in _RAW_FILE_EXTS)
        if not files:
            raise RuntimeError(f"No raw files found in {folder}")
        grouped = self._group_by_role(files)
        if grouped:
            _LOG.info("[SyntheticSource] folder mode: %d role-grouped packets", len(grouped))
            return grouped
        if not cfg.replicate_single_to_4cams:
            raise RuntimeError(
                f"Could not group files by role and replicate_single_to_4cams=False: {folder}"
            )
        _LOG.warning(
            "[SyntheticSource] folder mode: replicating single image across 4 cams (%d source files)",
            len(files),
        )
        return [tuple(self._load_raw(f) for _ in range(4)) for f in files]  # type: ignore[misc]

    def _group_by_role(
        self, files: Sequence[Path]
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        buckets: dict = {}
        for f in files:
            stem = f.stem
            for role in CAMERA_ROLES:
                if stem.endswith(f"_{role}") or f"_{role}_" in stem or stem.startswith(f"{role}_"):
                    key = stem.replace(f"_{role}", "").replace(f"{role}_", "")
                    buckets.setdefault(key, {})[role] = f
                    break
        full: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for key in sorted(buckets):
            roles = buckets[key]
            if all(role in roles for role in CAMERA_ROLES):
                full.append(tuple(self._load_raw(roles[role]) for role in CAMERA_ROLES))  # type: ignore[arg-type]
        return full

    def _load_raw(self, path: Path) -> np.ndarray:
        cfg = self.config
        suffix = path.suffix.lower()
        if suffix == ".npy":
            arr = np.load(str(path))
        elif suffix in (".bin", ".raw"):
            np_dtype = np.uint8 if cfg.dtype == "uint8" else np.uint16
            arr = np.fromfile(str(path), dtype=np_dtype).reshape((cfg.height, cfg.width))
        else:
            try:
                import cv2  # local import to keep this module test-importable without cv2
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    f"Loading {suffix} requires opencv-python"
                ) from exc
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"Failed to read image: {path}")
            if img.shape != (cfg.height, cfg.width):
                img = cv2.resize(img, (cfg.width, cfg.height), interpolation=cv2.INTER_AREA)
            arr = img
        if cfg.dtype == "uint16" and arr.dtype != np.uint16:
            arr = (arr.astype(np.uint16) << 8)
        if cfg.dtype == "uint8" and arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        return np.ascontiguousarray(arr)
