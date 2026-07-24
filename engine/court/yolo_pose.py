"""A CourtModel backed by an Ultralytics YOLO pose checkpoint.

Wraps any ``.pt`` pose model and routes its output through a
``KeypointAdapter`` into canonical order. The single-frame inference logic
mirrors the prefill path already proven in ``serve.py::predict_keypoints``
(confidence gating, highest-box-confidence instance selection, per-keypoint
visibility) — kept as an independent copy so the engine imports nothing from the
training scripts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .adapters import KeypointAdapter
from .base import CourtModel

# Match serve.py's prefill inference settings.
PREDICT_CONF = 0.25
PREDICT_KEYPOINT_CONF = 0.25
PREDICT_IMGSZ = 960


def source_keypoints(result, keypoint_conf: float = PREDICT_KEYPOINT_CONF) -> np.ndarray:
    """Extract one instance's native keypoints as an ``(K, 3)`` ``(x, y, v)`` array.

    Mirrors ``serve.py``: if several instances are detected, keep the one with
    the highest box confidence; a keypoint is visible (``v == 2``) when its
    confidence clears ``keypoint_conf``, otherwise it is dropped (``v == 0``).
    Returns an empty ``(0, 3)`` array when nothing is detected.
    """
    keypoints = getattr(result, "keypoints", None)
    xy_tensor = getattr(keypoints, "xy", None)
    if xy_tensor is None or len(xy_tensor) == 0:
        return np.zeros((0, 3), dtype=float)

    xy = xy_tensor.cpu().numpy()
    conf = getattr(keypoints, "conf", None)
    conf = conf.cpu().numpy() if conf is not None else None
    box_conf = getattr(getattr(result, "boxes", None), "conf", None)

    instance = 0
    if box_conf is not None and len(box_conf) > 1:
        instance = int(box_conf.cpu().numpy().argmax())

    rows = []
    for i, (x, y) in enumerate(xy[instance]):
        point_conf = float(conf[instance][i]) if conf is not None else 1.0
        visible = 2 if point_conf >= keypoint_conf else 0
        rows.append([float(x) if visible else 0.0, float(y) if visible else 0.0, visible])
    return np.asarray(rows, dtype=float)


class YoloPoseCourtModel(CourtModel):
    def __init__(
        self,
        weights: Path | str,
        adapter: KeypointAdapter,
        imgsz: int = PREDICT_IMGSZ,
        conf: float = PREDICT_CONF,
        keypoint_conf: float = PREDICT_KEYPOINT_CONF,
        device: str | None = None,
    ):
        weights = Path(weights)
        if not weights.is_file():
            raise FileNotFoundError(f"Court model weights not found: {weights}")
        from ultralytics import YOLO

        self._model = YOLO(str(weights))
        self._adapter = adapter
        self._imgsz = imgsz
        self._conf = conf
        self._keypoint_conf = keypoint_conf
        self._device = device

    @property
    def kpt_shape(self):
        return getattr(self._model.model, "kpt_shape", None)

    def predict_source(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Native (K, 3) keypoints before adaptation — used by --inspect."""
        result = self._model.predict(
            frame_bgr, conf=self._conf, imgsz=self._imgsz, device=self._device, verbose=False
        )[0]
        return source_keypoints(result, self._keypoint_conf)

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        source = self.predict_source(frame_bgr)
        if source.shape[0] == 0:
            return self.empty()
        return self._adapter.to_canonical(source)
