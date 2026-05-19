"""Synthetic RAW/Bayer batch generator.

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(slots=True)
class SyntheticRawConfig:
    """Configuration of the synthetic Bayer source.

    Field names match the production `SyntheticSourceConfig` where
    possible so a tester who already knows the production source can
    map values 1:1 without surprise.
    """
    capture_hw: Tuple[int, int] = (1080, 1920)        # (H, W) of the raw Bayer frame
    batch: int = 4
    bayer_pattern: str = "RG"
    dtype: str = "uint8"                              # "uint8" or "uint16"
    base_intensity: int = 90                          # neutral background brightness
    background_amplitude: int = 30                    # color gradient strength
    ball_enabled: bool = True
    ball_radius_px: int = 32
    ball_speed_px_per_frame: float = 24.0             # >0 advances the ball each call
    ball_intensity: int = 220
    noise_sigma: float = 0.0                          # additive Gaussian noise (sigma in 0..255)
    seed: int = 1234


class SyntheticRawSource:
    """Generate (B, H, W) uint8 Bayer batches with a moving ball pattern.

    Usage:
        src = SyntheticRawSource(SyntheticRawConfig(capture_hw=(1080, 1920), batch=4))
        for k in range(N):
            raw = src.next_batch()       # np.ndarray (4, 1080, 1920) uint8
    """

    def __init__(self, config: SyntheticRawConfig | None = None) -> None:
        self.config = config or SyntheticRawConfig()
        if self.config.capture_hw[0] % 2 or self.config.capture_hw[1] % 2:
            raise ValueError("capture_hw must be even (Bayer requirement)")
        if self.config.dtype not in ("uint8", "uint16"):
            raise ValueError(f"unsupported dtype: {self.config.dtype}")
        self._frame_index = 0
        self._rng = np.random.default_rng(self.config.seed)
        self._bg = self._build_background()

    # ---------------------------------------------------- background image

    def _build_background(self) -> np.ndarray:
        """Pre-compute a B-G-R gradient background — color so debayer is testable."""
        H, W = self.config.capture_hw
        yy, xx = np.mgrid[0:H, 0:W]
        bg_r = (self.config.base_intensity
                + self.config.background_amplitude
                * np.sin(xx / max(1.0, W) * 2 * np.pi))
        bg_g = (self.config.base_intensity
                + self.config.background_amplitude
                * np.sin(yy / max(1.0, H) * 2 * np.pi + 1.0))
        bg_b = (self.config.base_intensity
                + self.config.background_amplitude
                * np.cos((xx + yy) / max(1.0, H + W) * 4 * np.pi))
        bg_rgb = np.stack([bg_r, bg_g, bg_b], axis=-1)
        return np.clip(bg_rgb, 0, 255).astype(np.uint8)

    # -------------------------------------------------------- ball drawing

    def _draw_ball(self, rgb: np.ndarray, cx: float, cy: float) -> np.ndarray:
        """Render a soft-edge filled circle into an HWC RGB image."""
        H, W = rgb.shape[:2]
        r = max(2, int(self.config.ball_radius_px))
        x0 = max(0, int(cx) - r - 2)
        x1 = min(W, int(cx) + r + 2)
        y0 = max(0, int(cy) - r - 2)
        y1 = min(H, int(cy) + r + 2)
        if x0 >= x1 or y0 >= y1:
            return rgb
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        edge = (r ** 2) - d2
        mask = np.clip(edge / (2.0 * r), 0.0, 1.0)
        patch = rgb[y0:y1, x0:x1].astype(np.float32)
        ball = np.array([self.config.ball_intensity] * 3, dtype=np.float32)
        for c in range(3):
            patch[..., c] = patch[..., c] * (1.0 - mask) + ball[c] * mask
        rgb[y0:y1, x0:x1] = patch.astype(np.uint8)
        return rgb

    # ------------------------------------------------------ Bayer mosaicing

    @staticmethod
    def _rgb_to_bayer(rgb: np.ndarray, pattern: str) -> np.ndarray:
        """HWC uint8 RGB -> HW uint8 Bayer for the given pattern."""
        H, W, _ = rgb.shape
        bayer = np.zeros((H, W), dtype=np.uint8)
        R = rgb[..., 0]
        G = rgb[..., 1]
        B = rgb[..., 2]
        pat = pattern.upper()
        # row-even / row-odd, col-even / col-odd
        if pat == "RG":
            bayer[0::2, 0::2] = R[0::2, 0::2]
            bayer[0::2, 1::2] = G[0::2, 1::2]
            bayer[1::2, 0::2] = G[1::2, 0::2]
            bayer[1::2, 1::2] = B[1::2, 1::2]
        elif pat == "BG":
            bayer[0::2, 0::2] = B[0::2, 0::2]
            bayer[0::2, 1::2] = G[0::2, 1::2]
            bayer[1::2, 0::2] = G[1::2, 0::2]
            bayer[1::2, 1::2] = R[1::2, 1::2]
        elif pat == "GR":
            bayer[0::2, 0::2] = G[0::2, 0::2]
            bayer[0::2, 1::2] = R[0::2, 1::2]
            bayer[1::2, 0::2] = B[1::2, 0::2]
            bayer[1::2, 1::2] = G[1::2, 1::2]
        elif pat == "GB":
            bayer[0::2, 0::2] = G[0::2, 0::2]
            bayer[0::2, 1::2] = B[0::2, 1::2]
            bayer[1::2, 0::2] = R[1::2, 0::2]
            bayer[1::2, 1::2] = G[1::2, 1::2]
        else:
            raise ValueError(f"Unsupported Bayer pattern: {pattern}")
        return bayer

    # ------------------------------------------------------------ public API

    def next_batch(self) -> np.ndarray:
        """Return a single (B, H, W) Bayer batch (uint8 or uint16 per config)."""
        H, W = self.config.capture_hw
        out_dtype = np.uint8 if self.config.dtype == "uint8" else np.uint16
        out = np.empty((self.config.batch, H, W), dtype=out_dtype)
        # Convert pixel-per-frame speed to a Lissajous-step parameter so the
        # ball traverses ~ball_speed_px_per_frame pixels of arc per frame.
        step = (float(self.config.ball_speed_px_per_frame) /
                max(1.0, 0.5 * float(min(H, W))))
        for i in range(self.config.batch):
            t = (self._frame_index + i) * step
            cx = (0.5 + 0.4 * np.sin(t)) * W
            cy = (0.5 + 0.35 * np.cos(t * 1.3)) * H
            rgb = self._bg.copy()
            if self.config.ball_enabled:
                rgb = self._draw_ball(rgb, cx, cy)
            if self.config.noise_sigma > 0.0:
                noise = self._rng.normal(
                    loc=0.0, scale=float(self.config.noise_sigma),
                    size=rgb.shape,
                ).astype(np.float32)
                rgb = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            bayer8 = self._rgb_to_bayer(rgb, self.config.bayer_pattern)
            if out_dtype is np.uint16:
                out[i] = (bayer8.astype(np.uint16) << 8)
            else:
                out[i] = bayer8
        self._frame_index += self.config.batch
        return out
