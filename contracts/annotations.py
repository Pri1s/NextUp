"""Court-marking annotation vocabulary.

The benchmark these types describe exists to answer one question the calibration
engine cannot answer about itself: *does a fitted transform describe the floor?*
Every numeric gate in ``calibration/quality.py`` measures a fit against its own
evidence, and with five landmarks against a homography's eight degrees of freedom
that is cheap to satisfy while the projection drifts feet away from the paint. An
annotation is a claim about where paint actually is, made without reference to any
model output.

Three conventions are load-bearing.

**Painted centerline.** A coordinate names the middle of the painted stripe, not a
stripe edge and not the rule-book measurement line. This is what a ridge detector
finds and what an annotator can actually see. The layout's dimensions remain
rule-book values, so a constant per-marking offset of up to half a line width
survives; it is recorded rather than absorbed, because a systematic offset that
nobody wrote down is indistinguishable from a calibration error.

**A segment endpoint is not a court landmark.** Where visible paint stops is a
fact about occlusion and framing, not about the court. Sampled points along a
marking therefore constrain only distance *perpendicular* to that marking;
position along it stays unknown. The type system enforces this: ``MarkingFeature``
runs and ``JunctionPoint`` landmarks are different types, and a junction is
admissible only where both of its crossing markings were themselves annotated.

**Coordinate space is explicit.** Annotators work in magnified crops, where a
tool -- never the annotator's own arithmetic -- maps back to the frame. A crop-space
annotation carries a ``crop_id`` on every point and cannot be scored; a frame-space
one carries none and can. ``coordinate_space`` says which, and validation refuses
the mixture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from .markings import LINE_CONVENTION, FeatureKind, MarkingFeature

ANNOTATION_SCHEMA_VERSION = "court-annotation-1.0.0"

#: What a coordinate means. Re-exported for annotation API compatibility.
CoordinateSpace = Literal["crop", "frame"]

#: A polyline needs two points to exist. An arc needs enough to be distinguishable
#: from a straight run -- three can be fitted by any line with noise, five cannot.
MIN_POLYLINE_POINTS = 2
MIN_ARC_POINTS = 5

#: A homography maps straight lines to straight lines, so a straight court marking
#: is straight in the image no matter the camera. Deviation beyond this is a
#: mislabelled feature or points sampled off two different markings.
STRAIGHTNESS_TOLERANCE_PX = 3.0


#: Families held out of hybrid fitting and used only for validation (§4.3, §8.3).
#: Named here so the benchmark and the fitter cannot disagree about which they are.
HELD_OUT_FAMILIES = frozenset(
    {
        MarkingFeature.FREE_THROW_CIRCLE_FAR_HALF,
        MarkingFeature.FREE_THROW_CIRCLE_NEAR_HALF,
        MarkingFeature.RESTRICTED_AREA_ARC,
    }
)


class JunctionPoint(str, Enum):
    """A visible crossing of two identified markings -- the only admissible landmark.

    Every member maps to exactly two ``MarkingFeature`` runs. That mapping is what
    makes "an endpoint is not a landmark" checkable rather than merely documented:
    a junction whose constituents were not annotated is rejected.
    """

    BASELINE_SIDELINE_FAR = "baseline_sideline_far"
    BASELINE_SIDELINE_NEAR = "baseline_sideline_near"
    THREE_POINT_BASELINE_FAR = "three_point_baseline_far"
    THREE_POINT_BASELINE_NEAR = "three_point_baseline_near"
    LANE_BASELINE_FAR = "lane_baseline_far"
    LANE_BASELINE_NEAR = "lane_baseline_near"
    LANE_FREE_THROW_FAR = "lane_free_throw_far"
    LANE_FREE_THROW_NEAR = "lane_free_throw_near"
    MIDCOURT_SIDELINE_FAR = "midcourt_sideline_far"
    MIDCOURT_SIDELINE_NEAR = "midcourt_sideline_near"

    @property
    def crossing(self) -> tuple[MarkingFeature, MarkingFeature]:
        return _JUNCTION_CROSSINGS[self]


_JUNCTION_CROSSINGS: dict[JunctionPoint, tuple[MarkingFeature, MarkingFeature]] = {
    JunctionPoint.BASELINE_SIDELINE_FAR: (MarkingFeature.BASELINE, MarkingFeature.SIDELINE_FAR),
    JunctionPoint.BASELINE_SIDELINE_NEAR: (MarkingFeature.BASELINE, MarkingFeature.SIDELINE_NEAR),
    JunctionPoint.THREE_POINT_BASELINE_FAR: (
        MarkingFeature.BASELINE,
        MarkingFeature.THREE_POINT_CORNER_FAR,
    ),
    JunctionPoint.THREE_POINT_BASELINE_NEAR: (
        MarkingFeature.BASELINE,
        MarkingFeature.THREE_POINT_CORNER_NEAR,
    ),
    JunctionPoint.LANE_BASELINE_FAR: (MarkingFeature.BASELINE, MarkingFeature.LANE_EDGE_FAR),
    JunctionPoint.LANE_BASELINE_NEAR: (MarkingFeature.BASELINE, MarkingFeature.LANE_EDGE_NEAR),
    JunctionPoint.LANE_FREE_THROW_FAR: (
        MarkingFeature.FREE_THROW_LINE,
        MarkingFeature.LANE_EDGE_FAR,
    ),
    JunctionPoint.LANE_FREE_THROW_NEAR: (
        MarkingFeature.FREE_THROW_LINE,
        MarkingFeature.LANE_EDGE_NEAR,
    ),
    JunctionPoint.MIDCOURT_SIDELINE_FAR: (
        MarkingFeature.MIDCOURT_LINE,
        MarkingFeature.SIDELINE_FAR,
    ),
    JunctionPoint.MIDCOURT_SIDELINE_NEAR: (
        MarkingFeature.MIDCOURT_LINE,
        MarkingFeature.SIDELINE_NEAR,
    ),
}


class Visibility(str, Enum):
    """How well the paint could be seen, per feature."""

    CLEAR = "clear"
    FAINT = "faint"
    PARTIALLY_OCCLUDED = "partially_occluded"


class SkipReason(str, Enum):
    """Why a feature carries no annotation.

    An enum rather than free text: "skipped because ambiguous" and "skipped because
    outside the frame" mean opposite things for coverage statistics, and prose does
    not aggregate.
    """

    NOT_VISIBLE = "not_visible"
    OUT_OF_FRAME = "out_of_frame"
    FULLY_OCCLUDED = "fully_occluded"
    TOO_FAINT = "too_faint"
    AMBIGUOUS_IDENTITY = "ambiguous_identity"


@dataclass(frozen=True, slots=True)
class SamplePoint:
    """One observed centerline coordinate.

    ``crop_id`` is present exactly when the annotation is in crop space. Points of
    one feature routinely come from different crops, so crop-space coordinates
    share no common frame and no geometry may be computed until they are resolved.
    """

    x: float
    y: float
    crop_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("x", "y"):
            value = getattr(self, name)
            if math.isnan(value) or math.isinf(value):
                raise ValueError(f"{name} must be a real number, got {value!r}")

    def as_dict(self) -> dict:
        payload: dict = {"x": self.x, "y": self.y}
        if self.crop_id is not None:
            payload["crop_id"] = self.crop_id
        return payload

    @classmethod
    def from_dict(cls, data: dict) -> "SamplePoint":
        return cls(x=float(data["x"]), y=float(data["y"]), crop_id=data.get("crop_id"))


@dataclass(frozen=True, slots=True)
class AnnotatedFeature:
    """A traced run of one painted marking.

    ``points`` holds only coordinates the annotator actually observed -- nothing is
    interpolated across a gap, because an invented point is indistinguishable from
    an observed one once written down. Where paint disappears mid-run,
    ``occluded_after`` records the index it disappeared past, so a consumer can
    tell a deliberate gap from sparse sampling.
    """

    feature: MarkingFeature
    points: tuple[SamplePoint, ...]
    visibility: Visibility = Visibility.CLEAR
    uncertainty_px: float = 1.0
    occluded_after: tuple[int, ...] = ()
    notes: str = ""

    @property
    def kind(self) -> FeatureKind:
        return self.feature.kind

    def __post_init__(self) -> None:
        if self.uncertainty_px < 0:
            raise ValueError(f"uncertainty_px must be non-negative, got {self.uncertainty_px}")

    def as_dict(self) -> dict:
        return {
            "feature": self.feature.value,
            "kind": self.kind,
            "points": [p.as_dict() for p in self.points],
            "visibility": self.visibility.value,
            "uncertainty_px": self.uncertainty_px,
            "occluded_after": list(self.occluded_after),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AnnotatedFeature":
        return cls(
            feature=MarkingFeature(data["feature"]),
            points=tuple(SamplePoint.from_dict(p) for p in data["points"]),
            visibility=Visibility(data.get("visibility", Visibility.CLEAR.value)),
            uncertainty_px=float(data.get("uncertainty_px", 1.0)),
            occluded_after=tuple(int(i) for i in data.get("occluded_after", ())),
            notes=str(data.get("notes", "")),
        )


@dataclass(frozen=True, slots=True)
class AnnotatedJunction:
    """A visible crossing, and the only thing that may become a point landmark."""

    junction: JunctionPoint
    point: SamplePoint
    uncertainty_px: float = 1.0
    notes: str = ""

    def __post_init__(self) -> None:
        if self.uncertainty_px < 0:
            raise ValueError(f"uncertainty_px must be non-negative, got {self.uncertainty_px}")

    def as_dict(self) -> dict:
        return {
            "junction": self.junction.value,
            "point": self.point.as_dict(),
            "uncertainty_px": self.uncertainty_px,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AnnotatedJunction":
        return cls(
            junction=JunctionPoint(data["junction"]),
            point=SamplePoint.from_dict(data["point"]),
            uncertainty_px=float(data.get("uncertainty_px", 1.0)),
            notes=str(data.get("notes", "")),
        )


@dataclass(frozen=True, slots=True)
class SkippedFeature:
    """A feature deliberately not annotated, and why.

    Recorded rather than omitted: a missing feature could mean "not there" or
    "annotator missed it", and coverage statistics need to tell those apart.
    """

    feature: MarkingFeature
    reason: SkipReason
    notes: str = ""

    def as_dict(self) -> dict:
        return {"feature": self.feature.value, "reason": self.reason.value, "notes": self.notes}

    @classmethod
    def from_dict(cls, data: dict) -> "SkippedFeature":
        return cls(
            feature=MarkingFeature(data["feature"]),
            reason=SkipReason(data["reason"]),
            notes=str(data.get("notes", "")),
        )


@dataclass(frozen=True, slots=True)
class FrameAnnotation:
    """One annotator's pass over one frame.

    ``image_sha256`` pins the exact bytes annotated, so two passes can be proven to
    have seen the same image rather than assumed to. ``prompt_version`` pins the
    instructions that produced it: a prompt change invalidates prior annotations,
    and without the field that invalidation is silent.
    """

    frame_id: str
    clip: str
    frame_index: int
    image_width: int
    image_height: int
    image_sha256: str
    annotator_id: str
    pass_id: str
    prompt_version: str
    coordinate_space: CoordinateSpace = "crop"
    features: tuple[AnnotatedFeature, ...] = ()
    junctions: tuple[AnnotatedJunction, ...] = ()
    skipped: tuple[SkippedFeature, ...] = ()
    notes: str = ""
    schema_version: str = ANNOTATION_SCHEMA_VERSION
    line_convention: str = LINE_CONVENTION

    def __post_init__(self) -> None:
        if self.coordinate_space not in ("crop", "frame"):
            raise ValueError(f"unknown coordinate_space {self.coordinate_space!r}")
        if self.line_convention != LINE_CONVENTION:
            raise ValueError(
                f"annotation declares line convention {self.line_convention!r}, "
                f"but this schema defines {LINE_CONVENTION!r}"
            )
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError(
                f"image size must be positive, got {self.image_width}x{self.image_height}"
            )

    def feature(self, feature: MarkingFeature) -> AnnotatedFeature | None:
        for item in self.features:
            if item.feature is feature:
                return item
        return None

    @property
    def annotated_features(self) -> frozenset[MarkingFeature]:
        return frozenset(item.feature for item in self.features)

    def as_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "line_convention": self.line_convention,
            "frame_id": self.frame_id,
            "clip": self.clip,
            "frame_index": self.frame_index,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "image_sha256": self.image_sha256,
            "annotator_id": self.annotator_id,
            "pass_id": self.pass_id,
            "prompt_version": self.prompt_version,
            "coordinate_space": self.coordinate_space,
            "features": [f.as_dict() for f in self.features],
            "junctions": [j.as_dict() for j in self.junctions],
            "skipped": [s.as_dict() for s in self.skipped],
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FrameAnnotation":
        schema = data.get("schema_version")
        if schema != ANNOTATION_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported annotation schema {schema!r}, expected {ANNOTATION_SCHEMA_VERSION!r}"
            )
        return cls(
            frame_id=str(data["frame_id"]),
            clip=str(data["clip"]),
            frame_index=int(data["frame_index"]),
            image_width=int(data["image_width"]),
            image_height=int(data["image_height"]),
            image_sha256=str(data["image_sha256"]),
            annotator_id=str(data["annotator_id"]),
            pass_id=str(data["pass_id"]),
            prompt_version=str(data["prompt_version"]),
            coordinate_space=data.get("coordinate_space", "crop"),
            features=tuple(AnnotatedFeature.from_dict(f) for f in data.get("features", ())),
            junctions=tuple(AnnotatedJunction.from_dict(j) for j in data.get("junctions", ())),
            skipped=tuple(SkippedFeature.from_dict(s) for s in data.get("skipped", ())),
            notes=str(data.get("notes", "")),
            line_convention=str(data.get("line_convention", LINE_CONVENTION)),
        )


def _perpendicular_deviation_px(points: tuple[SamplePoint, ...]) -> float:
    """Largest distance from a sample to the chord through the first and last.

    A homography maps lines to lines, so a straight painted marking is straight in
    the image regardless of camera pose. Deviation is therefore evidence that the
    samples came from two different markings, not evidence of perspective.
    """
    if len(points) < 3:
        return 0.0
    x0, y0 = points[0].x, points[0].y
    x1, y1 = points[-1].x, points[-1].y
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return 0.0
    worst = 0.0
    for point in points[1:-1]:
        cross = abs(dx * (point.y - y0) - dy * (point.x - x0))
        worst = max(worst, cross / length)
    return worst


def _turn_signs(points: tuple[SamplePoint, ...]) -> list[float]:
    """Signed turn at each interior sample -- the cross product of adjacent steps."""
    signs = []
    for a, b, c in zip(points, points[1:], points[2:]):
        ux, uy = b.x - a.x, b.y - a.y
        vx, vy = c.x - b.x, c.y - b.y
        signs.append(ux * vy - uy * vx)
    return signs


def completeness_problems(
    annotation: FrameAnnotation,
    expected: frozenset[MarkingFeature] | None = None,
) -> list[str]:
    """Return features with no explicit annotated-or-skipped disposition.

    Structural validation cannot infer whether an omitted feature was invisible or
    simply forgotten. Annotation passes must therefore account for the complete
    vocabulary. Callers evaluating a synthetic scene may pass its visible subset
    as ``expected`` as an additional, truth-aware check.
    """
    if expected is None:
        expected = frozenset(MarkingFeature)
    accounted = annotation.annotated_features | frozenset(
        entry.feature for entry in annotation.skipped
    )
    missing = sorted(expected - accounted, key=lambda feature: feature.value)
    if not missing:
        return []
    return [
        "incomplete annotation: no annotated-or-skipped disposition for "
        + ", ".join(feature.value for feature in missing)
    ]


def validate(annotation: FrameAnnotation, *, require_complete: bool = False) -> list[str]:
    """Structural and geometric checks. Empty list means healthy.

    Geometric checks run only in frame space. Crop-space points from different
    crops live in unrelated coordinate frames, so straightness and curvature are
    meaningless until resolution -- silently computing them anyway would produce
    confident nonsense.
    """
    problems: list[str] = []
    in_frame_space = annotation.coordinate_space == "frame"

    seen_features: set[MarkingFeature] = set()
    for item in annotation.features:
        name = item.feature.value
        if item.feature in seen_features:
            problems.append(f"{name}: annotated more than once")
        seen_features.add(item.feature)

        minimum = MIN_ARC_POINTS if item.kind == "arc" else MIN_POLYLINE_POINTS
        if len(item.points) < minimum:
            problems.append(
                f"{name}: {len(item.points)} points, a {item.kind} needs at least {minimum}"
            )

        for index in item.occluded_after:
            if not 0 <= index < len(item.points) - 1:
                problems.append(
                    f"{name}: occluded_after index {index} does not lie between two samples"
                )

        for position, point in enumerate(item.points):
            has_crop = point.crop_id is not None
            if in_frame_space and has_crop:
                problems.append(f"{name}[{position}]: frame-space point still carries a crop_id")
            if not in_frame_space and not has_crop:
                problems.append(f"{name}[{position}]: crop-space point has no crop_id")
            if in_frame_space and not (
                0 <= point.x <= annotation.image_width and 0 <= point.y <= annotation.image_height
            ):
                problems.append(
                    f"{name}[{position}]: ({point.x:.1f}, {point.y:.1f}) is outside the image"
                )

        if in_frame_space and len(item.points) >= 3:
            if item.kind == "polyline":
                deviation = _perpendicular_deviation_px(item.points)
                if deviation > STRAIGHTNESS_TOLERANCE_PX:
                    problems.append(
                        f"{name}: bends {deviation:.1f} px off straight; a projected straight "
                        f"marking stays straight, so these samples likely span two markings"
                    )
            else:
                signs = [s for s in _turn_signs(item.points) if abs(s) > 1e-9]
                if signs and not (all(s > 0 for s in signs) or all(s < 0 for s in signs)):
                    problems.append(
                        f"{name}: curvature changes direction; a projected circular arc "
                        f"has no inflection, so the samples are out of order or mixed"
                    )

    skipped_features: set[MarkingFeature] = set()
    for entry in annotation.skipped:
        if entry.feature in skipped_features:
            problems.append(f"{entry.feature.value}: skipped more than once")
        skipped_features.add(entry.feature)
        if entry.feature in seen_features:
            problems.append(f"{entry.feature.value}: both annotated and skipped")

    seen_junctions: set[JunctionPoint] = set()
    for junction in annotation.junctions:
        name = junction.junction.value
        if junction.junction in seen_junctions:
            problems.append(f"{name}: annotated more than once")
        seen_junctions.add(junction.junction)

        missing = [f.value for f in junction.junction.crossing if f not in seen_features]
        if missing:
            problems.append(
                f"{name}: is a landmark only where both crossing markings were traced; "
                f"missing {', '.join(sorted(missing))}"
            )

        has_crop = junction.point.crop_id is not None
        if in_frame_space and has_crop:
            problems.append(f"{name}: frame-space point still carries a crop_id")
        if not in_frame_space and not has_crop:
            problems.append(f"{name}: crop-space point has no crop_id")
        if in_frame_space and not (
            0 <= junction.point.x <= annotation.image_width
            and 0 <= junction.point.y <= annotation.image_height
        ):
            problems.append(f"{name}: outside the image")

    if require_complete:
        problems.extend(completeness_problems(annotation))

    return problems
