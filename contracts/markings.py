"""Shared painted-court marking vocabulary and analytic primitives.

This module is deliberately stdlib-only.  Layouts, annotations, visualization,
benchmarking, and the future hybrid fitter all use these feature identities and
centerline primitives; consumers may convert sampled tuples to NumPy themselves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Literal, TypeAlias

Point2D: TypeAlias = tuple[float, float]
FeatureKind = Literal["polyline", "arc"]
LINE_CONVENTION = "painted_centerline"


class MarkingFamily(str, Enum):
    BOUNDARY = "boundary"
    LANE = "lane"
    FREE_THROW_CIRCLE = "free_throw_circle"
    THREE_POINT = "three_point"
    RESTRICTED_AREA = "restricted_area"
    MIDCOURT = "midcourt"


class MarkingFeature(str, Enum):
    """Atomic painted runs, named relative to the visible end/camera."""

    BASELINE = "baseline"
    SIDELINE_FAR = "sideline_far"
    SIDELINE_NEAR = "sideline_near"
    LANE_EDGE_FAR = "lane_edge_far"
    LANE_EDGE_NEAR = "lane_edge_near"
    FREE_THROW_LINE = "free_throw_line"
    FREE_THROW_CIRCLE_FAR_HALF = "free_throw_circle_far_half"
    FREE_THROW_CIRCLE_NEAR_HALF = "free_throw_circle_near_half"
    THREE_POINT_CORNER_FAR = "three_point_corner_far"
    THREE_POINT_CORNER_NEAR = "three_point_corner_near"
    THREE_POINT_ARC = "three_point_arc"
    RESTRICTED_AREA_ARC = "restricted_area_arc"
    MIDCOURT_LINE = "midcourt_line"
    CENTER_CIRCLE = "center_circle"

    @property
    def kind(self) -> FeatureKind:
        return "arc" if self in _CURVED_FEATURES else "polyline"

    @property
    def family(self) -> MarkingFamily:
        return _FEATURE_FAMILIES[self]


_CURVED_FEATURES = frozenset(
    {
        MarkingFeature.FREE_THROW_CIRCLE_FAR_HALF,
        MarkingFeature.FREE_THROW_CIRCLE_NEAR_HALF,
        MarkingFeature.THREE_POINT_ARC,
        MarkingFeature.RESTRICTED_AREA_ARC,
        MarkingFeature.CENTER_CIRCLE,
    }
)

_FEATURE_FAMILIES: dict[MarkingFeature, MarkingFamily] = {
    MarkingFeature.BASELINE: MarkingFamily.BOUNDARY,
    MarkingFeature.SIDELINE_FAR: MarkingFamily.BOUNDARY,
    MarkingFeature.SIDELINE_NEAR: MarkingFamily.BOUNDARY,
    MarkingFeature.LANE_EDGE_FAR: MarkingFamily.LANE,
    MarkingFeature.LANE_EDGE_NEAR: MarkingFamily.LANE,
    MarkingFeature.FREE_THROW_LINE: MarkingFamily.LANE,
    MarkingFeature.FREE_THROW_CIRCLE_FAR_HALF: MarkingFamily.FREE_THROW_CIRCLE,
    MarkingFeature.FREE_THROW_CIRCLE_NEAR_HALF: MarkingFamily.FREE_THROW_CIRCLE,
    MarkingFeature.THREE_POINT_CORNER_FAR: MarkingFamily.THREE_POINT,
    MarkingFeature.THREE_POINT_CORNER_NEAR: MarkingFamily.THREE_POINT,
    MarkingFeature.THREE_POINT_ARC: MarkingFamily.THREE_POINT,
    MarkingFeature.RESTRICTED_AREA_ARC: MarkingFamily.RESTRICTED_AREA,
    MarkingFeature.MIDCOURT_LINE: MarkingFamily.MIDCOURT,
    MarkingFeature.CENTER_CIRCLE: MarkingFamily.MIDCOURT,
}


def _point(value: object, name: str) -> Point2D:
    try:
        x, y = value  # type: ignore[misc]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain exactly two coordinates") from error
    result = (float(x), float(y))
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain finite coordinates, got {result!r}")
    return result


@dataclass(frozen=True, slots=True)
class StraightMarking:
    feature: MarkingFeature
    start: Point2D
    end: Point2D
    width_ft: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", _point(self.start, "start"))
        object.__setattr__(self, "end", _point(self.end, "end"))
        if self.feature.kind != "polyline":
            raise ValueError(f"{self.feature.value} is curved, not a straight marking")
        if not math.isfinite(self.width_ft) or self.width_ft <= 0:
            raise ValueError(f"marking width must be positive and finite, got {self.width_ft!r}")
        if math.dist(self.start, self.end) <= 1e-12:
            raise ValueError(f"{self.feature.value} has coincident endpoints")

    @property
    def family(self) -> MarkingFamily:
        return self.feature.family

    @property
    def closed(self) -> bool:
        return False

    def sample(self, count: int = 2) -> tuple[Point2D, ...]:
        if count < 2:
            raise ValueError("a straight marking needs at least two samples")
        x0, y0 = self.start
        x1, y1 = self.end
        return tuple(
            (x0 + (x1 - x0) * i / (count - 1), y0 + (y1 - y0) * i / (count - 1))
            for i in range(count)
        )

    def as_dict(self) -> dict:
        return {
            "primitive": "straight",
            "feature": self.feature.value,
            "family": self.family.value,
            "start": list(self.start),
            "end": list(self.end),
            "width_ft": self.width_ft,
            "closed": False,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StraightMarking":
        feature = MarkingFeature(data["feature"])
        item = cls(
            feature=feature,
            start=_point(data["start"], "start"),
            end=_point(data["end"], "end"),
            width_ft=float(data["width_ft"]),
        )
        if data.get("primitive") not in (None, "straight"):
            raise ValueError(f"unexpected primitive {data.get('primitive')!r}")
        if data.get("family", item.family.value) != item.family.value:
            raise ValueError(f"wrong family for {feature.value}")
        if data.get("closed", False):
            raise ValueError("a straight marking cannot be closed")
        return item


@dataclass(frozen=True, slots=True)
class ArcMarking:
    feature: MarkingFeature
    center: Point2D
    radius_ft: float
    start_angle_rad: float
    end_angle_rad: float
    width_ft: float
    closed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _point(self.center, "center"))
        if self.feature.kind != "arc":
            raise ValueError(f"{self.feature.value} is straight, not an arc marking")
        for name in ("radius_ft", "start_angle_rad", "end_angle_rad", "width_ft"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if self.radius_ft <= 0 or self.width_ft <= 0:
            raise ValueError("arc radius and width must be positive")
        span = self.end_angle_rad - self.start_angle_rad
        if span <= 0 or span > 2.0 * math.pi + 1e-9:
            raise ValueError(f"arc span must be in (0, 2pi], got {span}")
        full_circle = math.isclose(span, 2.0 * math.pi, abs_tol=1e-9)
        if self.closed != full_circle:
            raise ValueError("closed is true exactly for a full circular primitive")

    @property
    def family(self) -> MarkingFamily:
        return self.feature.family

    def sample(self, count: int = 72) -> tuple[Point2D, ...]:
        minimum = 3 if self.closed else 2
        if count < minimum:
            raise ValueError(f"this arc needs at least {minimum} samples")
        cx, cy = self.center
        return tuple(
            (
                cx + self.radius_ft * math.cos(
                    self.start_angle_rad
                    + (self.end_angle_rad - self.start_angle_rad) * i / (count - 1)
                ),
                cy + self.radius_ft * math.sin(
                    self.start_angle_rad
                    + (self.end_angle_rad - self.start_angle_rad) * i / (count - 1)
                ),
            )
            for i in range(count)
        )

    def as_dict(self) -> dict:
        return {
            "primitive": "arc",
            "feature": self.feature.value,
            "family": self.family.value,
            "center": list(self.center),
            "radius_ft": self.radius_ft,
            "start_angle_rad": self.start_angle_rad,
            "end_angle_rad": self.end_angle_rad,
            "width_ft": self.width_ft,
            "closed": self.closed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ArcMarking":
        feature = MarkingFeature(data["feature"])
        item = cls(
            feature=feature,
            center=_point(data["center"], "center"),
            radius_ft=float(data["radius_ft"]),
            start_angle_rad=float(data["start_angle_rad"]),
            end_angle_rad=float(data["end_angle_rad"]),
            width_ft=float(data["width_ft"]),
            closed=bool(data.get("closed", False)),
        )
        if data.get("primitive") not in (None, "arc"):
            raise ValueError(f"unexpected primitive {data.get('primitive')!r}")
        if data.get("family", item.family.value) != item.family.value:
            raise ValueError(f"wrong family for {feature.value}")
        return item


MarkingPrimitive: TypeAlias = StraightMarking | ArcMarking


def marking_from_dict(data: dict) -> MarkingPrimitive:
    primitive = data.get("primitive")
    if primitive == "straight":
        return StraightMarking.from_dict(data)
    if primitive == "arc":
        return ArcMarking.from_dict(data)
    raise ValueError(f"unknown marking primitive {primitive!r}")
