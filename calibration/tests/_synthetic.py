"""Synthetic fixtures.

The whole tool is exercised through the ``KeypointDetector`` protocol, so these
fakes stand in for a 400 MB checkpoint. That is not only about speed: a planted
detector lets a test state the *expected* answer, which is impossible against a
real model whose behaviour is the very thing under investigation.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from contracts.calibration_types import (
    ModelProvenance,
    RawKeypointFrame,
    RawKeypointObservation,
)
from calibration.io.video import SourceInfo

DIGEST = "0" * 64
KEYPOINT_COUNT = 18
WIDTH, HEIGHT = 1280, 720


def checkerboard(width: int = WIDTH, height: int = HEIGHT, seed: int = 0) -> np.ndarray:
    """A textured frame, so crops of different points are visibly different."""
    rng = np.random.default_rng(seed)
    frame = rng.integers(30, 90, size=(height, width, 3), dtype=np.uint8)
    frame[:: 40, :, :] = 200
    frame[:, :: 40, :] = 180
    return frame


class ArrayFrameSource:
    """A ``FrameSource`` over in-memory frames."""

    def __init__(self, frames: Sequence[np.ndarray], source_id: str = "synthetic.mp4", fps: float = 30.0):
        self._frames = list(frames)
        self._info = SourceInfo(
            source_id=source_id,
            kind="memory",
            frame_count=len(self._frames),
            fps=fps,
            width=int(self._frames[0].shape[1]),
            height=int(self._frames[0].shape[0]),
        )

    @property
    def info(self) -> SourceInfo:
        return self._info

    def frame_at(self, index: int) -> np.ndarray:
        if not 0 <= index < len(self._frames):
            raise IndexError(index)
        return self._frames[index]

    def timestamp_of(self, index: int) -> float:
        return index / (self._info.fps or 1.0)

    def close(self) -> None:
        return None


def default_slot_positions(count: int = KEYPOINT_COUNT) -> list[tuple[float, float]]:
    """Distinct, well-separated positions — one per slot."""
    positions = []
    for i in range(count):
        column, row = divmod(i, 6)
        positions.append((120.0 + column * 380.0, 110.0 + row * 100.0))
    return positions


class FakeDetector:
    """Returns whatever a planted callback says, in contract form.

    ``predict`` receives ``(frame_bgr, frame_index)`` and returns a list of
    ``(x, y, confidence)`` per slot, or ``None`` to simulate no detection.
    """

    def __init__(
        self,
        predict: Callable[[np.ndarray, int], list[tuple[float, float, float]] | None],
        keypoint_count: int = KEYPOINT_COUNT,
        instance_count: int = 1,
        device: str | None = "cpu",
        imgsz: int = 640,
    ):
        self._predict = predict
        self._keypoint_count = keypoint_count
        self._instance_count = instance_count
        self._device = device
        self._imgsz = imgsz
        self.calls: list[tuple[int, tuple[int, int]]] = []

    @property
    def keypoint_count(self) -> int:
        return self._keypoint_count

    def provenance(self) -> ModelProvenance:
        return ModelProvenance(
            weights_path="/fake/weights.pt",
            weights_sha256=DIGEST,
            keypoint_count=self._keypoint_count,
            task="pose",
            class_names={0: "basketball"},
            train_imgsz=640,
            train_fliplr=0.5,
            train_data="/fake/data.yaml",
            library_versions={"ultralytics": "fake"},
            device=self._device,
            inference_imgsz=self._imgsz,
            inference_conf=0.25,
        )

    def detect(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
        timestamp_s: float,
        source_id: str,
    ) -> RawKeypointFrame:
        height, width = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
        self.calls.append((frame_index, (width, height)))
        points = self._predict(frame_bgr, frame_index)

        if points is None:
            return RawKeypointFrame(
                frame_index=frame_index,
                timestamp_s=timestamp_s,
                source_id=source_id,
                image_width=width,
                image_height=height,
                detection_state="no_detection",
                instance_count=0,
                instance_confidences=(),
                selected_instance=None,
                observations=(),
            )

        return RawKeypointFrame(
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            source_id=source_id,
            image_width=width,
            image_height=height,
            detection_state="detected",
            instance_count=self._instance_count,
            instance_confidences=tuple(0.9 for _ in range(self._instance_count)),
            selected_instance=0,
            observations=tuple(
                RawKeypointObservation(
                    slot_index=i,
                    x=x,
                    y=y,
                    confidence=c,
                    clamped=x <= 1.0 or y <= 1.0 or x >= width - 2 or y >= height - 2,
                )
                for i, (x, y, c) in enumerate(points)
            ),
        )


def stable_detector(
    positions: Sequence[tuple[float, float]] | None = None,
    confidence: float = 0.9,
    **kwargs,
) -> FakeDetector:
    """Every slot in the same place on every frame, above gate."""
    positions = list(positions or default_slot_positions())

    def predict(_frame, _index):
        return [(x, y, confidence) for x, y in positions]

    return FakeDetector(predict, keypoint_count=len(positions), **kwargs)


def no_detection_detector(**kwargs) -> FakeDetector:
    """Never detects anything — the no-court case."""
    return FakeDetector(lambda _frame, _index: None, **kwargs)
