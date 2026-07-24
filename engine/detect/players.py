"""Thin wrapper over an Ultralytics YOLO detector for players/referees/ball.

Returns framework-neutral ``Detection`` rows so downstream code (tracking,
teams, analytics — all consequent) never touches Ultralytics result objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DETECT_CONF = 0.25
DETECT_IMGSZ = 1280


@dataclass
class Detection:
    xyxy: tuple[float, float, float, float]  # pixel box (x1, y1, x2, y2)
    cls: int
    name: str
    conf: float
    track_id: int | None = None  # set by the tracker; None before/without tracking

    def foot_point(self) -> tuple[float, float]:
        """Bottom-center of the box — the player's court contact point, the
        input the homography projects onto the court plane."""
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, y2)


class Detector:
    def __init__(
        self,
        weights: Path | str,
        conf: float = DETECT_CONF,
        imgsz: int = DETECT_IMGSZ,
        device: str | None = None,
    ):
        weights = Path(weights)
        if not weights.is_file():
            raise FileNotFoundError(f"Detector weights not found: {weights}")
        from ultralytics import YOLO

        self._model = YOLO(str(weights))
        self._names = self._model.names
        self._conf = conf
        self._imgsz = imgsz
        self._device = device

    def _rows(self, result) -> list[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        detections = []
        for i in range(len(xyxy)):
            class_id = int(cls[i])
            detections.append(
                Detection(
                    xyxy=tuple(float(v) for v in xyxy[i]),
                    cls=class_id,
                    name=str(self._names.get(class_id, class_id)) if isinstance(self._names, dict) else str(class_id),
                    conf=float(conf[i]),
                )
            )
        return detections

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        result = self._model.predict(
            frame_bgr, conf=self._conf, imgsz=self._imgsz, device=self._device, verbose=False
        )[0]
        return self._rows(result)

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        """One batched ``predict`` over N frames — throughput path for the pipeline."""
        if not frames:
            return []
        results = self._model.predict(
            frames, conf=self._conf, imgsz=self._imgsz, device=self._device, verbose=False
        )
        return [self._rows(result) for result in results]
