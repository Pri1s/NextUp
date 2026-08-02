"""Detection and calibration contracts.

Two layers live here, and the boundary between them is the point of the design.

**Raw detection types** (``RawKeypointObservation``, ``RawKeypointFrame``) are
what a detector emits *before* anything assigns court meaning to a model slot.
They carry no landmark identity, because this project has not established what
the detector's slots mean.

**Calibration types** (``CalibrationStatus``, ``LandmarkEvidence``,
``CorrespondenceSet``, ``CalibrationQuality``, ``CourtCalibration``) turn those
observations into a transform. They are built on the assumption that the slot
semantics feeding them are *provisional*: evidence is weighted rather than
trusted, failure is a status rather than an exception, and court end identity
stays unresolved rather than being guessed.

Coordinate convention: pixels in the **original full-resolution frame**, top-left
origin, y-down. A detector that runs on a letterboxed or resized image is
responsible for mapping back before constructing these types.

Confidence convention: continuous ``[0, 1]``. It is never thresholded, rounded,
or collapsed into a visibility flag at this layer. Gating is a downstream policy
decision, and Phase 0 specifically needs to see the below-gate behaviour.

Missing-point convention: a slot the model did not localise still carries its
*actual* predicted coordinate and confidence. The ``(0, 0)`` sentinel of the
reference pipeline is banned — an absent observation is expressed by omitting the
slot (or by ``detection_state == "no_detection"``), never by a fabricated origin
point. Slots whose predictions land on the frame border are flagged ``clamped``
so a reviewer can tell a real border landmark from a saturated one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

SCHEMA_VERSION = "phase0-inspection-1.0.0"
CALIBRATION_SCHEMA_VERSION = "calibration-1.0.0"

DetectionState = Literal["detected", "no_detection"]

#: How close to the frame edge a prediction must land to count as clamped.
CLAMP_EPSILON_PX = 1.0


@dataclass(frozen=True, slots=True)
class RawKeypointObservation:
    """One model slot's prediction for one frame.

    ``slot_index`` is the model's native keypoint index. It carries no court
    semantics — establishing those is the whole point of Phase 0.
    """

    slot_index: int
    x: float
    y: float
    confidence: float
    clamped: bool = False

    def __post_init__(self) -> None:
        if self.slot_index < 0:
            raise ValueError(f"slot_index must be non-negative, got {self.slot_index}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be continuous in [0, 1], got {self.confidence!r} "
                f"for slot {self.slot_index}"
            )
        for name in ("x", "y"):
            value = getattr(self, name)
            if value != value:  # NaN
                raise ValueError(f"{name} must be a real number, got NaN for slot {self.slot_index}")

    def as_dict(self) -> dict:
        return {
            "slot_index": self.slot_index,
            "x": self.x,
            "y": self.y,
            "confidence": self.confidence,
            "clamped": self.clamped,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RawKeypointObservation":
        return cls(
            slot_index=int(data["slot_index"]),
            x=float(data["x"]),
            y=float(data["y"]),
            confidence=float(data["confidence"]),
            clamped=bool(data.get("clamped", False)),
        )


@dataclass(frozen=True, slots=True)
class RawKeypointFrame:
    """Every slot the detector reported for a single inspected frame.

    ``instance_confidences`` holds *all* detected instances' box confidences, and
    ``selected_instance`` records which one the observations came from. The
    reference pipeline silently assumed instance 0; recording both makes
    multi-instance frames visible to the reviewer instead of hiding them.
    """

    frame_index: int
    timestamp_s: float
    source_id: str
    image_width: int
    image_height: int
    detection_state: DetectionState
    instance_count: int
    instance_confidences: tuple[float, ...] = ()
    selected_instance: int | None = None
    observations: tuple[RawKeypointObservation, ...] = ()

    def __post_init__(self) -> None:
        if self.detection_state not in ("detected", "no_detection"):
            raise ValueError(f"unknown detection_state {self.detection_state!r}")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError(
                f"image size must be positive, got {self.image_width}x{self.image_height}"
            )
        if self.instance_count < 0:
            raise ValueError(f"instance_count must be non-negative, got {self.instance_count}")

        if self.detection_state == "no_detection":
            if self.observations:
                raise ValueError("a no_detection frame must carry zero observations")
            if self.selected_instance is not None:
                raise ValueError("a no_detection frame cannot select an instance")

        seen: set[int] = set()
        for observation in self.observations:
            if observation.slot_index in seen:
                raise ValueError(f"duplicate slot_index {observation.slot_index}")
            seen.add(observation.slot_index)

        if self.selected_instance is not None and not (
            0 <= self.selected_instance < max(self.instance_count, 1)
        ):
            raise ValueError(
                f"selected_instance {self.selected_instance} outside "
                f"0..{self.instance_count - 1}"
            )

    @property
    def detected(self) -> bool:
        return self.detection_state == "detected"

    def observation(self, slot_index: int) -> RawKeypointObservation | None:
        for observation in self.observations:
            if observation.slot_index == slot_index:
                return observation
        return None

    def as_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "source_id": self.source_id,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "detection_state": self.detection_state,
            "instance_count": self.instance_count,
            "instance_confidences": list(self.instance_confidences),
            "selected_instance": self.selected_instance,
            "observations": [o.as_dict() for o in self.observations],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RawKeypointFrame":
        return cls(
            frame_index=int(data["frame_index"]),
            timestamp_s=float(data["timestamp_s"]),
            source_id=str(data["source_id"]),
            image_width=int(data["image_width"]),
            image_height=int(data["image_height"]),
            detection_state=data["detection_state"],
            instance_count=int(data["instance_count"]),
            instance_confidences=tuple(float(c) for c in data.get("instance_confidences", ())),
            selected_instance=(
                None if data.get("selected_instance") is None else int(data["selected_instance"])
            ),
            observations=tuple(
                RawKeypointObservation.from_dict(o) for o in data.get("observations", ())
            ),
        )


@dataclass(frozen=True, slots=True)
class ModelProvenance:
    """Everything needed to re-derive a Phase-0 verdict from the same weights.

    A slot-semantics sign-off is only meaningful against a specific checkpoint,
    so the weights hash travels with every artifact the tooling writes.
    """

    weights_path: str
    weights_sha256: str
    keypoint_count: int
    task: str
    class_names: dict[int, str] = field(default_factory=dict)
    train_imgsz: int | None = None
    train_fliplr: float | None = None
    train_data: str | None = None
    library_versions: dict[str, str] = field(default_factory=dict)
    device: str | None = None
    inference_imgsz: int | None = None
    inference_conf: float | None = None

    def __post_init__(self) -> None:
        if self.keypoint_count <= 0:
            raise ValueError(f"keypoint_count must be positive, got {self.keypoint_count}")
        if len(self.weights_sha256) != 64:
            raise ValueError(f"weights_sha256 must be a 64-char hex digest, got {self.weights_sha256!r}")

    def as_dict(self) -> dict:
        return {
            "weights_path": self.weights_path,
            "weights_sha256": self.weights_sha256,
            "keypoint_count": self.keypoint_count,
            "task": self.task,
            "class_names": {str(k): v for k, v in self.class_names.items()},
            "train_imgsz": self.train_imgsz,
            "train_fliplr": self.train_fliplr,
            "train_data": self.train_data,
            "library_versions": dict(self.library_versions),
            "device": self.device,
            "inference_imgsz": self.inference_imgsz,
            "inference_conf": self.inference_conf,
        }


# --------------------------------------------------------------------------
# Calibration contracts
#
# Everything below turns *observations* into a *transform*. The detector's slot
# semantics are unverified, so the vocabulary here is deliberately built for
# evidence that may be wrong: every observation carries a weight, every result
# carries a status, and "I could not calibrate this frame" is a first-class
# answer rather than an exception.
# --------------------------------------------------------------------------


class CalibrationStatus(str, Enum):
    """The outcome of one calibration attempt.

    Failures are explicit and distinguishable. A caller must never have to infer
    from a null transform *why* there is no transform — the reason changes what
    the caller should do next.
    """

    OK = "OK"
    DEGRADED = "DEGRADED"
    UNCALIBRATED_INSUFFICIENT_EVIDENCE = "UNCALIBRATED_INSUFFICIENT_EVIDENCE"
    FAILED_INSUFFICIENT_POINTS = "FAILED_INSUFFICIENT_POINTS"
    FAILED_DEGENERATE_GEOMETRY = "FAILED_DEGENERATE_GEOMETRY"
    FAILED_HIGH_REPROJECTION_ERROR = "FAILED_HIGH_REPROJECTION_ERROR"
    FAILED_IMPLAUSIBLE = "FAILED_IMPLAUSIBLE"
    NOT_ATTEMPTED_NO_COURT = "NOT_ATTEMPTED_NO_COURT"

    @property
    def usable(self) -> bool:
        """Whether a transform exists and may be used downstream."""
        return self in (CalibrationStatus.OK, CalibrationStatus.DEGRADED)


class CalibrationSource(str, Enum):
    """Where a calibration came from."""

    DIRECT = "direct"          # solved from this frame's own observations
    PROPAGATED = "propagated"  # reused from a nearby frame, revalidated
    SMOOTHED = "smoothed"


@dataclass(frozen=True, slots=True)
class LandmarkEvidence:
    """One landmark's image position, pooled from every slot that voted for it.

    The detector emits redundant slots that fire on the same physical feature and
    split their confidence between themselves. Choosing a winner throws away half
    the signal and makes a 50/50 split look like two weak observations instead of
    one strong one — so evidence is pooled, and ``source_slots`` records exactly
    which slots contributed.
    """

    landmark_id: str
    x: float
    y: float
    confidence: float
    source_slots: tuple[int, ...]
    tier: str = "confident"

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if not self.source_slots:
            raise ValueError(f"{self.landmark_id} has no source slots")
        if math.isnan(self.x) or math.isnan(self.y):
            raise ValueError(f"{self.landmark_id} has a NaN coordinate")

    def as_dict(self) -> dict:
        return {
            "landmark_id": self.landmark_id,
            "x": self.x,
            "y": self.y,
            "confidence": self.confidence,
            "source_slots": list(self.source_slots),
            "tier": self.tier,
        }


@dataclass(frozen=True, slots=True)
class Correspondence:
    """One image point paired with its known court coordinate."""

    landmark_id: str
    image_xy: tuple[float, float]
    court_xy: tuple[float, float]
    weight: float
    source_slots: tuple[int, ...] = ()

    def as_dict(self) -> dict:
        return {
            "landmark_id": self.landmark_id,
            "image_xy": list(self.image_xy),
            "court_xy": list(self.court_xy),
            "weight": self.weight,
            "source_slots": list(self.source_slots),
        }


@dataclass(frozen=True, slots=True)
class CorrespondenceSet:
    """Correspondences kept for fitting, plus why everything else was dropped.

    Drop reasons are not diagnostics padding — with an unverified detector they
    are the main way anyone notices that a slot mapping has gone wrong.
    """

    correspondences: tuple[Correspondence, ...]
    dropped: tuple[tuple[str, str], ...] = ()  # (identifier, reason)

    def __len__(self) -> int:
        return len(self.correspondences)

    def as_dict(self) -> dict:
        return {
            "kept": [c.as_dict() for c in self.correspondences],
            "dropped": [{"identifier": i, "reason": r} for i, r in self.dropped],
        }


@dataclass(frozen=True, slots=True)
class CalibrationQuality:
    """Every signal used to decide a status, recorded whether it passed or not.

    Kept deliberately verbose: when a detector's semantics are provisional, the
    quality record is the audit trail that shows a calibration was earned rather
    than assumed.
    """

    observation_count: int
    inlier_count: int
    inlier_ratio: float
    reprojection_px_mean: float | None = None
    reprojection_px_median: float | None = None
    reprojection_px_p95: float | None = None
    reprojection_px_max: float | None = None
    reprojection_ft_median: float | None = None
    reprojection_ft_max: float | None = None
    holdout_px_median: float | None = None
    min_triangle_area_px: float | None = None
    image_hull_fraction: float | None = None
    court_span_x_ft: float | None = None
    court_span_y_ft: float | None = None
    scale_ft_per_px: float | None = None
    round_trip_error_px: float | None = None
    mean_evidence_confidence: float | None = None
    reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "observation_count": self.observation_count,
            "inlier_count": self.inlier_count,
            "inlier_ratio": self.inlier_ratio,
            "reprojection_px": {
                "mean": self.reprojection_px_mean,
                "median": self.reprojection_px_median,
                "p95": self.reprojection_px_p95,
                "max": self.reprojection_px_max,
            },
            "reprojection_ft": {
                "median": self.reprojection_ft_median,
                "max": self.reprojection_ft_max,
            },
            "holdout_px_median": self.holdout_px_median,
            "min_triangle_area_px": self.min_triangle_area_px,
            "image_hull_fraction": self.image_hull_fraction,
            "court_span_ft": {"x": self.court_span_x_ft, "y": self.court_span_y_ft},
            "scale_ft_per_px": self.scale_ft_per_px,
            "round_trip_error_px": self.round_trip_error_px,
            "mean_evidence_confidence": self.mean_evidence_confidence,
            "reasons": list(self.reasons),
        }


Matrix3x3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


@dataclass(frozen=True, slots=True)
class CourtCalibration:
    """The result of calibrating one frame.

    Holds **both** transforms so no caller ever inverts a matrix itself, and
    verifies at construction that they really are mutually inverse. Court
    coordinates are in the visible-end half-court frame; ``end_identity`` stays
    ``"unresolved"`` because nothing in this pipeline has established which end
    of the floor is in shot, and silently inheriting a guess is precisely the
    failure this design exists to avoid.
    """

    frame_index: int
    layout_id: str
    status: CalibrationStatus
    frame_of_reference: str = "visible_end_half_court"
    end_identity: str = "unresolved"
    units: str = "feet"
    h_image_to_court: Matrix3x3 | None = None
    h_court_to_image: Matrix3x3 | None = None
    quality: CalibrationQuality | None = None
    inlier_landmarks: tuple[str, ...] = ()
    outlier_landmarks: tuple[str, ...] = ()
    source: CalibrationSource = CalibrationSource.DIRECT
    source_age_frames: int = 0
    provenance: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status.usable:
            if self.h_image_to_court is None or self.h_court_to_image is None:
                raise ValueError(f"status {self.status.value} requires both transforms")
        elif self.h_image_to_court is not None or self.h_court_to_image is not None:
            raise ValueError(f"status {self.status.value} must not carry a transform")
        if self.end_identity not in ("unresolved", "resolved"):
            raise ValueError(f"unexpected end_identity {self.end_identity!r}")

    @property
    def usable(self) -> bool:
        return self.status.usable

    def image_to_court(self, x: float, y: float) -> tuple[float, float]:
        """Project an image point onto the half-court plane, in feet."""
        if self.h_image_to_court is None:
            raise ValueError(f"no transform available (status {self.status.value})")
        return _apply(self.h_image_to_court, x, y)

    def court_to_image(self, x: float, y: float) -> tuple[float, float]:
        """Project a half-court point into image pixels."""
        if self.h_court_to_image is None:
            raise ValueError(f"no transform available (status {self.status.value})")
        return _apply(self.h_court_to_image, x, y)

    def as_dict(self) -> dict:
        return {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "frame_index": self.frame_index,
            "layout_id": self.layout_id,
            "status": self.status.value,
            "frame_of_reference": self.frame_of_reference,
            "end_identity": self.end_identity,
            "units": self.units,
            "h_image_to_court": _matrix_as_list(self.h_image_to_court),
            "h_court_to_image": _matrix_as_list(self.h_court_to_image),
            "quality": self.quality.as_dict() if self.quality else None,
            "inlier_landmarks": list(self.inlier_landmarks),
            "outlier_landmarks": list(self.outlier_landmarks),
            "source": self.source.value,
            "source_age_frames": self.source_age_frames,
            "provenance": dict(self.provenance),
        }


def _apply(matrix: Matrix3x3, x: float, y: float) -> tuple[float, float]:
    a, b, c = matrix
    denominator = c[0] * x + c[1] * y + c[2]
    if abs(denominator) < 1e-12:
        raise ValueError("point projects to infinity under this transform")
    return (
        (a[0] * x + a[1] * y + a[2]) / denominator,
        (b[0] * x + b[1] * y + b[2]) / denominator,
    )


def _matrix_as_list(matrix: Matrix3x3 | None) -> list[list[float]] | None:
    return None if matrix is None else [list(row) for row in matrix]
