"""Tiny synthetic clips for pipeline tests (no real footage / weights needed)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

DARK = (20, 20, 20)
BRIGHT = (230, 230, 230)


def make_clip(
    path: Path | str,
    n_frames: int = 12,
    size: tuple[int, int] = (64, 48),
    fps: int = 30,
    cut_at: int | None = None,
    colors: tuple[tuple[int, int, int], tuple[int, int, int]] = (DARK, BRIGHT),
) -> Path:
    """Write a solid-color BGR clip (MJPG/AVI — portable and readable everywhere).

    With ``cut_at`` set, frames ``[0, cut_at)`` are ``colors[0]`` and frames
    ``[cut_at, n)`` are ``colors[1]`` — one hard scene cut for the detector.
    """
    path = Path(path)
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("could not open VideoWriter (MJPG)")
    try:
        for i in range(n_frames):
            color = colors[1] if (cut_at is not None and i >= cut_at) else colors[0]
            writer.write(np.full((h, w, 3), color, dtype=np.uint8))
    finally:
        writer.release()
    return path
