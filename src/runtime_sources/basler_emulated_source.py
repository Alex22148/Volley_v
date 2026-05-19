"""Basler-emulated 4-camera RAW source (pypylon Camera Emulator).

This adapter is a parallel, isolated path next to SyntheticRaw4CamSource.
It uses pypylon's emulated cameras (PYLON_CAMEMU=4) to push frames
through the actual driver/SDK code, which gives more realistic timing
than a pure-numpy generator without touching the production grabber in
capture/.

Imports of pypylon are deferred; if pypylon is unavailable, instantiation
raises a clear RuntimeError so tests/dry-runs that don't need this source
are unaffected.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

import numpy as np

from .base_frame_source import (
    BaseFrameSource,
    CAMERA_ROLES,
    FrameBatch4Cam,
    RawFrame,
    SUPPORTED_BAYER_PATTERNS,
)

_LOG = logging.getLogger(__name__)


@dataclass(slots=True)
class BaslerEmulatedConfig:
    """Configuration for BaslerEmulatedSource.

    Note: the pylon emulator delivers Mono8 / Mono16 frames by default.
    bayer_pattern here describes what we *label* the synthetic frames as,
    so that downstream debayer code follows the same path as on real
    cameras. The pixel content is not a real Bayer mosaic; this is purely
    a timing-realistic carrier.
    """

    width: int = 2464
    height: int = 2056
    fps: float = 77.0
    num_frames: int = 1000
    bayer_pattern: str = "RG"
    dtype: str = "uint8"
    pixel_format: str = "Mono8"
    test_image_selector: str = "Testimage1"
    grab_timeout_ms: int = 1000
    extra_metadata: dict = field(default_factory=dict)

    def validate(self) -> None:
        if self.bayer_pattern not in SUPPORTED_BAYER_PATTERNS:
            raise ValueError(
                f"bayer_pattern must be one of {SUPPORTED_BAYER_PATTERNS}, got {self.bayer_pattern!r}"
            )
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width/height must be > 0")
        if self.num_frames <= 0:
            raise ValueError("num_frames must be > 0")
        if self.fps <= 0:
            raise ValueError("fps must be > 0")


class BaslerEmulatedSource(BaseFrameSource):
    """Synchronized 4-camera RAW source backed by pypylon's emulator."""

    def __init__(self, config: BaslerEmulatedConfig) -> None:
        config.validate()
        self.config = config
        self._packet_id = 0
        self._frame_id = 0
        self._array = None
        self._cameras: List = []
        self._init_pylon()

    @property
    def description(self) -> str:
        c = self.config
        return (
            f"BaslerEmulatedSource(PYLON_CAMEMU=4, {c.width}x{c.height}, "
            f"pixel_format={c.pixel_format}, fps={c.fps})"
        )

    def expected_total_packets(self) -> Optional[int]:
        return int(self.config.num_frames)

    def _init_pylon(self) -> None:
        os.environ.setdefault("PYLON_CAMEMU", "4")
        try:
            from pypylon import pylon  # type: ignore
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "pypylon not available — BaslerEmulatedSource requires "
                "pypylon and the pylon SDK installed on the host."
            ) from exc

        tlf = pylon.TlFactory.GetInstance()
        devs = tlf.EnumerateDevices()
        emu_devs = [d for d in devs if "Emu" in (d.GetDeviceClass() or "") or "emulated" in (d.GetModelName() or "").lower()]
        if len(emu_devs) < 4:
            raise RuntimeError(
                f"Expected >=4 emulated devices (PYLON_CAMEMU=4), found {len(emu_devs)}. "
                "Set PYLON_CAMEMU=4 in the environment before launching."
            )

        array = pylon.InstantCameraArray(4)
        for idx in range(4):
            array[idx].Attach(tlf.CreateDevice(emu_devs[idx]))
        array.Open()

        for idx in range(4):
            cam = array[idx]
            for node_name, value in (
                ("Width", int(self.config.width)),
                ("Height", int(self.config.height)),
                ("PixelFormat", str(self.config.pixel_format)),
                ("TestImageSelector", str(self.config.test_image_selector)),
                ("AcquisitionFrameRateAbs", float(self.config.fps)),
            ):
                self._safe_set(cam, node_name, value)

        array.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self._array = array
        self._cameras = [array[i] for i in range(4)]
        _LOG.info(
            "[BaslerEmulatedSource] %d emulated cameras opened (%dx%d %s)",
            len(self._cameras),
            self.config.width,
            self.config.height,
            self.config.pixel_format,
        )

    def _safe_set(self, cam, node_name: str, value) -> None:
        try:
            node = cam.GetNodeMap().GetNode(node_name)
            if node is None:
                return
            if hasattr(node, "FromString"):
                node.FromString(str(value))
            elif hasattr(node, "SetValue"):
                node.SetValue(value)
        except Exception as exc:
            _LOG.debug("[BaslerEmulatedSource] node %s set skipped: %s", node_name, exc)

    def iter_packets(self) -> Iterator[FrameBatch4Cam]:
        if self._array is None:
            raise RuntimeError("BaslerEmulatedSource not initialized")
        try:
            from pypylon import pylon  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("pypylon disappeared mid-run") from exc

        timeout = int(self.config.grab_timeout_ms)
        for _ in range(self.config.num_frames):
            ts_ns = time.time_ns()
            quad: List[np.ndarray] = []
            for cam in self._cameras:
                grab = cam.RetrieveResult(timeout, pylon.TimeoutHandling_ThrowException)
                try:
                    if not grab.GrabSucceeded():
                        raise RuntimeError(
                            f"Grab failed: {grab.ErrorCode} {grab.ErrorDescription}"
                        )
                    arr = grab.GetArray()
                    if arr.shape != (self.config.height, self.config.width):
                        raise RuntimeError(
                            f"Unexpected emu frame shape: {arr.shape}"
                        )
                    quad.append(np.ascontiguousarray(arr))
                finally:
                    grab.Release()

            frames: List[RawFrame] = []
            for role_idx, role in enumerate(CAMERA_ROLES):
                frames.append(
                    RawFrame(
                        camera_id=f"EMU-{role}",
                        role=role,
                        frame_id=self._frame_id,
                        timestamp_ns=ts_ns,
                        image=quad[role_idx],
                        width=self.config.width,
                        height=self.config.height,
                        bayer_pattern=self.config.bayer_pattern,
                        dtype=self.config.dtype,
                    )
                )
            packet = FrameBatch4Cam(
                packet_id=self._packet_id,
                timestamp_ns=ts_ns,
                frames=tuple(frames),  # type: ignore[arg-type]
                metadata={"source_mode": "basler_emu", **dict(self.config.extra_metadata)},
            )
            packet.validate()
            self._packet_id += 1
            self._frame_id += 1
            yield packet

    def close(self) -> None:
        if self._array is not None:
            try:
                self._array.StopGrabbing()
            except Exception:
                pass
            try:
                self._array.Close()
            except Exception:
                pass
            self._array = None
            self._cameras = []
