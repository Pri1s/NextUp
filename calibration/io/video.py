"""Frame sources for inspection.

A ``FrameSource`` yields BGR frames by index and knows how many exist. Two
implementations ship: one over a video file, one over a list of still images
(the intake path for crowd/replay/advertisement negatives). Tests inject a third
backed by in-memory arrays, so nothing under test needs a real decoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

import cv2
import numpy as np

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass(frozen=True, slots=True)
class SourceInfo:
    """What a frame source is, for provenance."""

    source_id: str
    kind: str  # "video" | "images" | "memory"
    frame_count: int
    fps: float | None
    width: int
    height: int


class FrameSource(Protocol):
    """Random access to frames of one clip or image set."""

    @property
    def info(self) -> SourceInfo: ...

    def frame_at(self, index: int) -> np.ndarray:
        """Return the BGR frame at ``index``; raise ``IndexError`` if absent."""
        ...

    def timestamp_of(self, index: int) -> float:
        """Seconds into the source, or ``0.0`` when there is no time base."""
        ...

    def close(self) -> None: ...


class VideoFrameSource:
    """Frames from a video file, by index.

    Seeks with ``CAP_PROP_POS_FRAMES`` and falls back to sequential reads when
    the container's index is unreliable — some codecs land on the nearest
    keyframe instead of the requested frame.
    """

    def __init__(self, path: Path | str):
        self._path = Path(path)
        if not self._path.is_file():
            raise FileNotFoundError(f"video not found: {self._path}")
        self._capture = cv2.VideoCapture(str(self._path))
        if not self._capture.isOpened():
            raise RuntimeError(f"could not open video: {self._path}")

        frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        self._info = SourceInfo(
            source_id=self._path.name,
            kind="video",
            frame_count=max(frame_count, 0),
            fps=fps if fps > 0 else None,
            width=int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        self._cursor = 0

    @property
    def path(self) -> Path:
        return self._path

    @property
    def info(self) -> SourceInfo:
        return self._info

    def frame_at(self, index: int) -> np.ndarray:
        if index < 0:
            raise IndexError(f"negative frame index {index}")
        if index != self._cursor:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            self._cursor = index
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise IndexError(f"could not read frame {index} of {self._path.name}")
        self._cursor = index + 1
        return frame

    def timestamp_of(self, index: int) -> float:
        fps = self._info.fps
        return index / fps if fps else 0.0

    def close(self) -> None:
        self._capture.release()


class ImageFrameSource:
    """Still images treated as pseudo-frames, indexed by sorted filename.

    Used to feed negatives (crowd, replay wipes, advertisement boards) through
    the same inspection path as video frames.
    """

    def __init__(self, paths: Iterable[Path | str], source_id: str = "images"):
        self._paths = sorted(Path(p) for p in paths)
        if not self._paths:
            raise ValueError("no images supplied")
        missing = [p for p in self._paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"images not found: {missing}")

        first = cv2.imread(str(self._paths[0]))
        if first is None:
            raise RuntimeError(f"could not decode image: {self._paths[0]}")
        self._info = SourceInfo(
            source_id=source_id,
            kind="images",
            frame_count=len(self._paths),
            fps=None,
            width=int(first.shape[1]),
            height=int(first.shape[0]),
        )

    @property
    def paths(self) -> list[Path]:
        return list(self._paths)

    @property
    def info(self) -> SourceInfo:
        return self._info

    def frame_at(self, index: int) -> np.ndarray:
        if not 0 <= index < len(self._paths):
            raise IndexError(f"image index {index} out of range")
        frame = cv2.imread(str(self._paths[index]))
        if frame is None:
            raise RuntimeError(f"could not decode image: {self._paths[index]}")
        return frame

    def timestamp_of(self, index: int) -> float:
        return 0.0

    def close(self) -> None:
        return None


def resolve_image_paths(spec: str) -> list[Path]:
    """Expand a directory or glob into a sorted list of image paths."""
    candidate = Path(spec)
    if candidate.is_dir():
        found = [p for p in candidate.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
    else:
        found = [
            p
            for p in Path(candidate.parent or ".").glob(candidate.name)
            if p.suffix.lower() in IMAGE_SUFFIXES
        ]
    if not found:
        raise FileNotFoundError(f"no images matched {spec!r}")
    return sorted(found)
