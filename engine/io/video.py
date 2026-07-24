"""Sequential frame iteration over a clip.

Uses OpenCV ``grab()``/``retrieve()`` rather than seeking with
``CAP_PROP_POS_FRAMES`` (unreliable on some encodes) — the same decode approach
proven in ``extract_frames.py::extract_clip``, copied here so the engine imports
nothing from the training scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoInfo:
    fps: float
    width: int
    height: int
    frame_count: int


def _open(video_path: Path | str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    return cap


def probe(video_path: Path | str) -> VideoInfo:
    cap = _open(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return VideoInfo(fps=fps, width=width, height=height, frame_count=frame_count)
    finally:
        cap.release()


def iter_frames(
    video_path: Path | str,
    stride: int = 1,
    max_frames: int | None = None,
    start_index: int = 0,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield ``(frame_index, timestamp_s, frame_bgr)``.

    ``stride`` retrieves every Nth frame (grabbing the rest cheaply);
    ``max_frames`` caps how many frames are yielded. ``start_index`` skips ahead
    to a native frame index by grabbing (not decoding) everything before it — the
    cheap resume path: pass the frame *after* the last one already processed.
    ``frame_index`` is always the native (unstrided) index, so it is a stable
    anchor across a resume.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if start_index < 0:
        raise ValueError("start_index must be >= 0")
    cap = _open(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 30.0
    yielded = 0
    frame_index = 0
    try:
        while True:
            if max_frames is not None and yielded >= max_frames:
                break
            if not cap.grab():
                break
            if frame_index >= start_index and frame_index % stride == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                yield frame_index, round(frame_index / fps, 3), frame
                yielded += 1
            frame_index += 1
    finally:
        cap.release()
