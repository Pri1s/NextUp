"""Ultralytics YOLO pose checkpoint as a ``KeypointDetector``.

Deliberately thin: run the model, read the native slots, hand back a
``RawKeypointFrame``. No gating, no adaptation, no court semantics — Phase 0 is
about seeing what the model actually emits, so nothing here filters it.

Two behaviours the reference pipeline got wrong are handled explicitly:

*Instance selection.* The reference assumed instance 0. This records every
instance's box confidence and selects the most confident one, so multi-instance
frames are visible in the record rather than silently resolved.

*Absent points.* The reference wrote ``(0, 0)``. This keeps whatever the model
predicted and flags border-saturated points as ``clamped`` — the observed
behaviour on real footage is a slot pinned to ``x = 0.0``, which is a clamped
prediction, not a sentinel, and the two must not be confused.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from contracts.calibration_types import (
    CLAMP_EPSILON_PX,
    ModelProvenance,
    RawKeypointFrame,
    RawKeypointObservation,
)

from ..io.provenance import sha256_file

DEFAULT_IMGSZ = 640  # matches the checkpoint's training imgsz
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.7
DEFAULT_MAX_DET = 10


class YoloPoseDetector:
    """A pose checkpoint loaded once and run per frame."""

    def __init__(
        self,
        weights: Path | str,
        imgsz: int = DEFAULT_IMGSZ,
        conf: float = DEFAULT_CONF,
        iou: float = DEFAULT_IOU,
        max_det: int = DEFAULT_MAX_DET,
        device: str | None = None,
    ):
        self._weights = Path(weights)
        if not self._weights.is_file():
            raise FileNotFoundError(f"weights not found: {self._weights}")

        from ultralytics import YOLO  # imported lazily so tests need no torch

        self._model = YOLO(str(self._weights))
        self._imgsz = imgsz
        self._conf = conf
        self._iou = iou
        self._max_det = max_det
        self._device = device
        self._weights_sha256: str | None = None

        kpt_shape = getattr(self._model.model, "kpt_shape", None)
        if not kpt_shape:
            raise ValueError(f"{self._weights.name} is not a pose model (no kpt_shape)")
        self._keypoint_count = int(kpt_shape[0])

    @property
    def keypoint_count(self) -> int:
        return self._keypoint_count

    @property
    def imgsz(self) -> int:
        return self._imgsz

    def provenance(self) -> ModelProvenance:
        if self._weights_sha256 is None:
            self._weights_sha256 = sha256_file(self._weights)

        train_args = getattr(self._model, "ckpt", {}).get("train_args", {}) or {}
        versions: dict[str, str] = {}
        for module_name, key in (("ultralytics", "ultralytics"), ("torch", "torch")):
            try:
                versions[key] = str(getattr(__import__(module_name), "__version__", "unknown"))
            except Exception:  # noqa: BLE001
                versions[key] = "not-installed"

        return ModelProvenance(
            weights_path=str(self._weights.resolve()),
            weights_sha256=self._weights_sha256,
            keypoint_count=self._keypoint_count,
            task=str(getattr(self._model, "task", "pose")),
            class_names=dict(getattr(self._model, "names", {}) or {}),
            train_imgsz=train_args.get("imgsz"),
            train_fliplr=train_args.get("fliplr"),
            train_data=train_args.get("data"),
            library_versions=versions,
            device=self._device,
            inference_imgsz=self._imgsz,
            inference_conf=self._conf,
        )

    def detect(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
        timestamp_s: float,
        source_id: str,
    ) -> RawKeypointFrame:
        height, width = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
        result = self._model.predict(
            frame_bgr,
            imgsz=self._imgsz,
            conf=self._conf,
            iou=self._iou,
            max_det=self._max_det,
            device=self._device,
            verbose=False,
        )[0]

        boxes = getattr(result, "boxes", None)
        box_confidences: tuple[float, ...] = ()
        if boxes is not None and len(boxes) > 0 and boxes.conf is not None:
            box_confidences = tuple(float(c) for c in boxes.conf.cpu().numpy())

        keypoints = getattr(result, "keypoints", None)
        xy_tensor = getattr(keypoints, "xy", None)
        if xy_tensor is None or len(xy_tensor) == 0:
            return RawKeypointFrame(
                frame_index=frame_index,
                timestamp_s=timestamp_s,
                source_id=source_id,
                image_width=width,
                image_height=height,
                detection_state="no_detection",
                instance_count=len(box_confidences),
                instance_confidences=box_confidences,
                selected_instance=None,
                observations=(),
            )

        xy = xy_tensor.cpu().numpy()
        conf_tensor = getattr(keypoints, "conf", None)
        confidences = conf_tensor.cpu().numpy() if conf_tensor is not None else None

        instance_count = int(len(xy))
        selected = 0
        if box_confidences and len(box_confidences) == instance_count:
            selected = int(np.argmax(box_confidences))

        slot_confidences = (
            None if confidences is None else [float(c) for c in confidences[selected]]
        )
        observations = tuple(
            _observation(slot_index, float(x), float(y), slot_confidences, width, height)
            for slot_index, (x, y) in enumerate(xy[selected])
        )

        return RawKeypointFrame(
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            source_id=source_id,
            image_width=width,
            image_height=height,
            detection_state="detected",
            instance_count=instance_count if instance_count else len(box_confidences),
            instance_confidences=box_confidences,
            selected_instance=selected,
            observations=observations,
        )


def _observation(
    slot_index: int,
    x: float,
    y: float,
    slot_confidences: list[float] | None,
    width: int,
    height: int,
) -> RawKeypointObservation:
    confidence = 1.0 if slot_confidences is None else slot_confidences[slot_index]
    return RawKeypointObservation(
        slot_index=slot_index,
        x=x,
        y=y,
        confidence=min(max(confidence, 0.0), 1.0),
        clamped=is_clamped(x, y, width, height),
    )


def is_clamped(x: float, y: float, width: int, height: int) -> bool:
    """True when a prediction sits on the frame border.

    A landmark genuinely on the edge and a prediction saturated against the edge
    look identical in the coordinates alone; flagging them lets the reviewer tell
    the difference from the overlay.
    """
    return (
        x <= CLAMP_EPSILON_PX
        or y <= CLAMP_EPSILON_PX
        or x >= width - 1 - CLAMP_EPSILON_PX
        or y >= height - 1 - CLAMP_EPSILON_PX
    )
