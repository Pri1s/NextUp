"""End-agnostic, painted-centerline NBA half-court geometry.

The origin is the intersection of the visible baseline and far-sideline painted
centerlines. +X runs toward midcourt and +Y toward the near sideline. Published
rule-book dimensions are retained with their edge conventions and converted to
painted centerlines exactly once by :func:`build_layout`.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .markings import (
    LINE_CONVENTION,
    ArcMarking,
    MarkingFeature,
    MarkingPrimitive,
    StraightMarking,
)

LAYOUT_SCHEMA_VERSION = "half-court-layout-2.0.0"


class HalfCourtLandmark(str, Enum):
    BASELINE_SIDELINE_FAR = "baseline_sideline_far"
    BASELINE_SIDELINE_NEAR = "baseline_sideline_near"
    THREE_POINT_BASELINE_FAR = "three_point_baseline_far"
    THREE_POINT_BASELINE_NEAR = "three_point_baseline_near"
    LANE_BASELINE_FAR = "lane_baseline_far"
    LANE_BASELINE_NEAR = "lane_baseline_near"
    LANE_FREE_THROW_FAR = "lane_free_throw_far"
    LANE_FREE_THROW_NEAR = "lane_free_throw_near"
    FREE_THROW_CIRCLE_APEX = "free_throw_circle_apex"
    MIDCOURT_SIDELINE_FAR = "midcourt_sideline_far"
    MIDCOURT_SIDELINE_NEAR = "midcourt_sideline_near"

    @property
    def sideline_mirror(self) -> "HalfCourtLandmark | None":
        return _SIDELINE_MIRRORS.get(self)


_SIDELINE_MIRRORS: dict[HalfCourtLandmark, HalfCourtLandmark] = {}
for _a, _b in (
    (HalfCourtLandmark.BASELINE_SIDELINE_FAR, HalfCourtLandmark.BASELINE_SIDELINE_NEAR),
    (HalfCourtLandmark.THREE_POINT_BASELINE_FAR, HalfCourtLandmark.THREE_POINT_BASELINE_NEAR),
    (HalfCourtLandmark.LANE_BASELINE_FAR, HalfCourtLandmark.LANE_BASELINE_NEAR),
    (HalfCourtLandmark.LANE_FREE_THROW_FAR, HalfCourtLandmark.LANE_FREE_THROW_NEAR),
    (HalfCourtLandmark.MIDCOURT_SIDELINE_FAR, HalfCourtLandmark.MIDCOURT_SIDELINE_NEAR),
):
    _SIDELINE_MIRRORS[_a] = _b
    _SIDELINE_MIRRORS[_b] = _a


class MeasurementReference(str, Enum):
    INSIDE_EDGE = "inside_edge"
    OUTSIDE_EDGE = "outside_edge"
    CENTERLINE = "centerline"
    STRIPE_WIDTH = "stripe_width"


_UNIT_TO_FEET = {"feet": 1.0, "inches": 1.0 / 12.0}


@dataclass(frozen=True, slots=True)
class SourceMeasurement:
    """One published value and its audited conversion to a centerline value.

    ``centerline_offset_half_widths`` records the signed number of half-stripe
    widths needed for this particular diagram dimension. It makes semantic
    direction explicit (inside/outside alone is insufficient for a gap versus a
    radius) while still deriving the physical offset from the layout line width.
    """

    value: float
    unit: str
    reference: MeasurementReference
    centerline_offset_half_widths: int
    diagram_label: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.value) or self.value <= 0:
            raise ValueError(f"source measurement must be positive and finite, got {self.value!r}")
        if self.unit not in _UNIT_TO_FEET:
            raise ValueError(f"unsupported source unit {self.unit!r}")
        if not isinstance(self.centerline_offset_half_widths, int):
            raise ValueError("centerline_offset_half_widths must be an integer")
        if abs(self.centerline_offset_half_widths) > 2:
            raise ValueError("centerline offset may be at most one full stripe width")
        if not self.diagram_label:
            raise ValueError("source measurement needs its diagram label")

    @property
    def raw_ft(self) -> float:
        return self.value * _UNIT_TO_FEET[self.unit]

    def centerline_ft(self, line_width_ft: float) -> float:
        return self.raw_ft + self.centerline_offset_half_widths * line_width_ft / 2.0

    def as_dict(self) -> dict:
        return {
            "value": self.value,
            "unit": self.unit,
            "reference": self.reference.value,
            "centerline_offset_half_widths": self.centerline_offset_half_widths,
            "diagram_label": self.diagram_label,
            "raw_ft": self.raw_ft,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SourceMeasurement":
        required = {
            "value", "unit", "reference", "centerline_offset_half_widths", "diagram_label"
        }
        missing = required - set(data)
        unknown = set(data) - required
        if missing or unknown:
            raise ValueError(
                f"measurement fields mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        return cls(
            value=float(data["value"]),
            unit=str(data["unit"]),
            reference=MeasurementReference(data["reference"]),
            centerline_offset_half_widths=int(data["centerline_offset_half_widths"]),
            diagram_label=str(data["diagram_label"]),
        )


_MEASUREMENT_KEYS = frozenset(
    {
        "line_width",
        "half_length",
        "width",
        "lane_width",
        "free_throw_distance",
        "basket_from_baseline",
        "three_point_corner_inset",
        "three_point_radius",
        "free_throw_circle_radius",
        "center_circle_radius",
        "restricted_area_radius",
    }
)


@dataclass(frozen=True, slots=True)
class HalfCourtLayout:
    layout_id: str
    rule_set: str
    source: str
    source_url: str
    source_measurements: dict[str, SourceMeasurement]
    half_length: float
    width: float
    lane_width: float
    free_throw_distance: float
    basket_from_baseline: float
    three_point_corner_inset: float
    three_point_radius: float
    free_throw_circle_radius: float
    center_circle_radius: float
    restricted_area_radius: float
    line_width: float
    landmarks: dict[HalfCourtLandmark, tuple[float, float]] = field(default_factory=dict)
    markings: dict[MarkingFeature, MarkingPrimitive] = field(default_factory=dict)
    units: str = "feet"
    line_convention: str = LINE_CONVENTION

    def __post_init__(self) -> None:
        if self.units != "feet":
            raise ValueError(f"layouts normalise to feet, got {self.units!r}")
        if self.line_convention != LINE_CONVENTION:
            raise ValueError(f"unsupported line convention {self.line_convention!r}")
        if set(self.source_measurements) != _MEASUREMENT_KEYS:
            raise ValueError("layout source measurements are incomplete or contain unknown keys")
        for name in (
            "half_length", "width", "lane_width", "free_throw_distance",
            "basket_from_baseline", "three_point_corner_inset", "three_point_radius",
            "free_throw_circle_radius", "center_circle_radius", "restricted_area_radius",
            "line_width",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite, got {value!r}")
        for landmark, (x, y) in self.landmarks.items():
            if not 0.0 <= x <= self.half_length:
                raise ValueError(f"{landmark.value} x={x} outside 0..{self.half_length}")
            if not 0.0 <= y <= self.width:
                raise ValueError(f"{landmark.value} y={y} outside 0..{self.width}")
        for feature, primitive in self.markings.items():
            if primitive.feature is not feature:
                raise ValueError(f"marking key {feature.value} disagrees with primitive feature")

    @property
    def basket_center(self) -> tuple[float, float]:
        return (self.basket_from_baseline, self.width / 2.0)

    def coordinate(self, landmark: HalfCourtLandmark) -> tuple[float, float]:
        try:
            return self.landmarks[landmark]
        except KeyError as error:
            raise KeyError(f"{landmark.value} is absent from layout {self.layout_id}") from error

    def has(self, landmark: HalfCourtLandmark) -> bool:
        return landmark in self.landmarks

    def marking(self, feature: MarkingFeature) -> MarkingPrimitive:
        try:
            return self.markings[feature]
        except KeyError as error:
            raise KeyError(f"{feature.value} is absent from layout {self.layout_id}") from error

    def sample_marking(self, feature: MarkingFeature, count: int | None = None) -> tuple[tuple[float, float], ...]:
        primitive = self.marking(feature)
        if count is None:
            count = 2 if isinstance(primitive, StraightMarking) else (
                81 if feature is MarkingFeature.THREE_POINT_ARC else
                73 if primitive.closed else 41
            )
        return primitive.sample(count)

    def _geometry_payload(self) -> dict:
        return {
            "layout_id": self.layout_id,
            "schema_version": LAYOUT_SCHEMA_VERSION,
            "rule_set": self.rule_set,
            "source": {"title": self.source, "url": self.source_url},
            "units": self.units,
            "line_convention": self.line_convention,
            "source_measurements": {
                key: self.source_measurements[key].as_dict() for key in sorted(self.source_measurements)
            },
            "derived": {
                "half_length": self.half_length,
                "width": self.width,
                "lane_width": self.lane_width,
                "free_throw_distance": self.free_throw_distance,
                "basket_from_baseline": self.basket_from_baseline,
                "three_point_corner_inset": self.three_point_corner_inset,
                "three_point_radius": self.three_point_radius,
                "free_throw_circle_radius": self.free_throw_circle_radius,
                "center_circle_radius": self.center_circle_radius,
                "restricted_area_radius": self.restricted_area_radius,
                "line_width": self.line_width,
            },
            "landmarks": {
                key.value: list(value)
                for key, value in sorted(self.landmarks.items(), key=lambda item: item[0].value)
            },
            "markings": {
                key.value: value.as_dict()
                for key, value in sorted(self.markings.items(), key=lambda item: item[0].value)
            },
        }

    def content_hash(self) -> str:
        payload = json.dumps(self._geometry_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict:
        payload = self._geometry_payload()
        payload["content_hash"] = self.content_hash()
        return payload


def _centerline_measurements(
    *, half_length: float, width: float, lane_width: float,
    free_throw_distance: float, basket_from_baseline: float,
    three_point_corner_inset: float, three_point_radius: float,
    free_throw_circle_radius: float, center_circle_radius: float,
    restricted_area_radius: float, line_width: float,
) -> dict[str, SourceMeasurement]:
    values = locals()
    result = {}
    for name in _MEASUREMENT_KEYS:
        result[name] = SourceMeasurement(
            value=float(values[name]), unit="feet",
            reference=(MeasurementReference.STRIPE_WIDTH if name == "line_width" else MeasurementReference.CENTERLINE),
            centerline_offset_half_widths=0,
            diagram_label=f"test centerline {name}",
        )
    return result


def build_layout(
    layout_id: str,
    rule_set: str,
    source: str,
    half_length: float | None = None,
    width: float | None = None,
    lane_width: float | None = None,
    free_throw_distance: float | None = None,
    basket_from_baseline: float | None = None,
    three_point_corner_inset: float | None = None,
    free_throw_circle_radius: float | None = None,
    three_point_radius: float | None = None,
    center_circle_radius: float | None = None,
    restricted_area_radius: float | None = None,
    line_width: float | None = None,
    *,
    source_url: str = "",
    source_measurements: dict[str, SourceMeasurement] | None = None,
) -> HalfCourtLayout:
    """Derive landmarks and every painted-centerline primitive from source data."""
    if source_measurements is None:
        supplied = {
            "half_length": half_length, "width": width, "lane_width": lane_width,
            "free_throw_distance": free_throw_distance,
            "basket_from_baseline": basket_from_baseline,
            "three_point_corner_inset": three_point_corner_inset,
            "three_point_radius": three_point_radius,
            "free_throw_circle_radius": free_throw_circle_radius,
            "center_circle_radius": center_circle_radius,
            "restricted_area_radius": restricted_area_radius,
            "line_width": line_width,
        }
        missing = sorted(key for key, value in supplied.items() if value is None)
        if missing:
            raise ValueError(f"missing centerline layout dimensions: {', '.join(missing)}")
        source_measurements = _centerline_measurements(**supplied)  # type: ignore[arg-type]
    elif set(source_measurements) != _MEASUREMENT_KEYS:
        missing = sorted(_MEASUREMENT_KEYS - set(source_measurements))
        unknown = sorted(set(source_measurements) - _MEASUREMENT_KEYS)
        raise ValueError(f"measurement keys mismatch; missing={missing}, unknown={unknown}")

    line_spec = source_measurements["line_width"]
    if line_spec.reference is not MeasurementReference.STRIPE_WIDTH:
        raise ValueError("line_width must declare stripe_width reference")
    line_width_ft = line_spec.raw_ft
    derived = {
        name: spec.centerline_ft(line_width_ft)
        for name, spec in source_measurements.items() if name != "line_width"
    }
    length = derived["half_length"]
    court_width = derived["width"]
    lane = derived["lane_width"]
    free_throw = derived["free_throw_distance"]
    basket_x = derived["basket_from_baseline"]
    corner = derived["three_point_corner_inset"]
    three_radius = derived["three_point_radius"]
    free_radius = derived["free_throw_circle_radius"]
    center_radius = derived["center_circle_radius"]
    restricted_radius = derived["restricted_area_radius"]

    centre_y = court_width / 2.0
    lane_far = centre_y - lane / 2.0
    lane_near = centre_y + lane / 2.0
    dy = centre_y - corner
    if three_radius <= abs(dy):
        raise ValueError("three-point radius cannot reach the corner straight")
    break_x = basket_x + math.sqrt(three_radius**2 - dy**2)
    arc_start = math.atan2(corner - centre_y, break_x - basket_x)
    arc_end = math.atan2(court_width - corner - centre_y, break_x - basket_x)

    markings: dict[MarkingFeature, MarkingPrimitive] = {
        MarkingFeature.BASELINE: StraightMarking(MarkingFeature.BASELINE, (0.0, 0.0), (0.0, court_width), line_width_ft),
        MarkingFeature.SIDELINE_FAR: StraightMarking(MarkingFeature.SIDELINE_FAR, (0.0, 0.0), (length, 0.0), line_width_ft),
        MarkingFeature.SIDELINE_NEAR: StraightMarking(MarkingFeature.SIDELINE_NEAR, (0.0, court_width), (length, court_width), line_width_ft),
        MarkingFeature.LANE_EDGE_FAR: StraightMarking(MarkingFeature.LANE_EDGE_FAR, (0.0, lane_far), (free_throw, lane_far), line_width_ft),
        MarkingFeature.LANE_EDGE_NEAR: StraightMarking(MarkingFeature.LANE_EDGE_NEAR, (0.0, lane_near), (free_throw, lane_near), line_width_ft),
        MarkingFeature.FREE_THROW_LINE: StraightMarking(MarkingFeature.FREE_THROW_LINE, (free_throw, lane_far), (free_throw, lane_near), line_width_ft),
        MarkingFeature.MIDCOURT_LINE: StraightMarking(MarkingFeature.MIDCOURT_LINE, (length, 0.0), (length, court_width), line_width_ft),
        MarkingFeature.THREE_POINT_CORNER_FAR: StraightMarking(MarkingFeature.THREE_POINT_CORNER_FAR, (0.0, corner), (break_x, corner), line_width_ft),
        MarkingFeature.THREE_POINT_CORNER_NEAR: StraightMarking(MarkingFeature.THREE_POINT_CORNER_NEAR, (0.0, court_width - corner), (break_x, court_width - corner), line_width_ft),
        MarkingFeature.THREE_POINT_ARC: ArcMarking(MarkingFeature.THREE_POINT_ARC, (basket_x, centre_y), three_radius, arc_start, arc_end, line_width_ft),
        MarkingFeature.FREE_THROW_CIRCLE_FAR_HALF: ArcMarking(MarkingFeature.FREE_THROW_CIRCLE_FAR_HALF, (free_throw, centre_y), free_radius, -math.pi / 2.0, math.pi / 2.0, line_width_ft),
        MarkingFeature.FREE_THROW_CIRCLE_NEAR_HALF: ArcMarking(MarkingFeature.FREE_THROW_CIRCLE_NEAR_HALF, (free_throw, centre_y), free_radius, math.pi / 2.0, 3.0 * math.pi / 2.0, line_width_ft),
        MarkingFeature.RESTRICTED_AREA_ARC: ArcMarking(MarkingFeature.RESTRICTED_AREA_ARC, (basket_x, centre_y), restricted_radius, -math.pi / 2.0, math.pi / 2.0, line_width_ft),
        MarkingFeature.CENTER_CIRCLE: ArcMarking(MarkingFeature.CENTER_CIRCLE, (length, centre_y), center_radius, 0.0, 2.0 * math.pi, line_width_ft, closed=True),
    }

    landmarks = {
        HalfCourtLandmark.BASELINE_SIDELINE_FAR: (0.0, 0.0),
        HalfCourtLandmark.BASELINE_SIDELINE_NEAR: (0.0, court_width),
        HalfCourtLandmark.THREE_POINT_BASELINE_FAR: (0.0, corner),
        HalfCourtLandmark.THREE_POINT_BASELINE_NEAR: (0.0, court_width - corner),
        HalfCourtLandmark.LANE_BASELINE_FAR: (0.0, lane_far),
        HalfCourtLandmark.LANE_BASELINE_NEAR: (0.0, lane_near),
        HalfCourtLandmark.LANE_FREE_THROW_FAR: (free_throw, lane_far),
        HalfCourtLandmark.LANE_FREE_THROW_NEAR: (free_throw, lane_near),
        HalfCourtLandmark.FREE_THROW_CIRCLE_APEX: (free_throw + free_radius, centre_y),
        HalfCourtLandmark.MIDCOURT_SIDELINE_FAR: (length, 0.0),
        HalfCourtLandmark.MIDCOURT_SIDELINE_NEAR: (length, court_width),
    }

    layout = HalfCourtLayout(
        layout_id=layout_id, rule_set=rule_set, source=source, source_url=source_url,
        source_measurements=dict(source_measurements), half_length=length, width=court_width,
        lane_width=lane, free_throw_distance=free_throw, basket_from_baseline=basket_x,
        three_point_corner_inset=corner, three_point_radius=three_radius,
        free_throw_circle_radius=free_radius, center_circle_radius=center_radius,
        restricted_area_radius=restricted_radius, line_width=line_width_ft,
        landmarks=landmarks, markings=markings,
    )
    problems = validate_layout(layout)
    if problems:
        raise ValueError("invalid derived layout: " + "; ".join(problems))
    return layout


def load_layout(path: Path | str) -> HalfCourtLayout:
    with open(path, encoding="utf-8") as handle:
        profile = json.load(handle)
    schema = profile.get("schema_version")
    if schema != LAYOUT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported layout schema {schema!r}; expected {LAYOUT_SCHEMA_VERSION!r}. "
            "Layout v1 geometry used unconverted rule-book edges and must be migrated."
        )
    required = {
        "layout_id", "schema_version", "rule_set", "source", "units",
        "frame_of_reference", "line_convention", "measurements",
    }
    optional = {"notes"}
    missing = required - set(profile)
    unknown = set(profile) - required - optional
    if missing or unknown:
        raise ValueError(f"layout profile fields mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}")
    if profile["units"] != "feet":
        raise ValueError("layout profile canonical units must be feet")
    if profile["frame_of_reference"] != "visible_end_half_court":
        raise ValueError("unsupported layout frame_of_reference")
    if profile["line_convention"] != LINE_CONVENTION:
        raise ValueError("layout profile must use painted_centerline")
    source = profile["source"]
    if set(source) != {"title", "url"}:
        raise ValueError("layout source must contain exactly title and url")
    measurements = {
        key: SourceMeasurement.from_dict(value)
        for key, value in profile["measurements"].items()
    }
    return build_layout(
        layout_id=str(profile["layout_id"]), rule_set=str(profile["rule_set"]),
        source=str(source["title"]), source_url=str(source["url"]),
        source_measurements=measurements,
    )


LAYOUT_DIR = Path(__file__).resolve().parent / "layouts"


def available_layouts() -> list[str]:
    return sorted(path.stem for path in LAYOUT_DIR.glob("*.json")) if LAYOUT_DIR.is_dir() else []


def load_registered_layout(layout_id: str) -> HalfCourtLayout:
    path = LAYOUT_DIR / f"{layout_id}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"unknown layout {layout_id!r}; available: {', '.join(available_layouts()) or 'none'}"
        )
    return load_layout(path)


def _close(a: tuple[float, float], b: tuple[float, float], tolerance: float = 1e-9) -> bool:
    return math.dist(a, b) <= tolerance


def validate_layout(layout: HalfCourtLayout) -> list[str]:
    problems: list[str] = []
    centre_y = layout.width / 2.0

    if set(layout.markings) != set(MarkingFeature):
        missing = sorted(feature.value for feature in set(MarkingFeature) - set(layout.markings))
        extra = sorted(feature.value for feature in set(layout.markings) - set(MarkingFeature))
        problems.append(f"marking vocabulary mismatch; missing={missing}, extra={extra}")

    for landmark, mirror in _SIDELINE_MIRRORS.items():
        if not (layout.has(landmark) and layout.has(mirror)):
            continue
        x_a, y_a = layout.coordinate(landmark)
        x_b, y_b = layout.coordinate(mirror)
        if abs(x_a - x_b) > 1e-9:
            problems.append(f"{landmark.value}/{mirror.value} differ in x ({x_a} vs {x_b})")
        if abs(y_a + y_b - 2.0 * centre_y) > 1e-9:
            problems.append(f"{landmark.value}/{mirror.value} are not symmetric about y={centre_y}")

    if layout.free_throw_distance >= layout.half_length:
        problems.append("free-throw line is at or beyond midcourt")
    if layout.lane_width >= layout.width:
        problems.append("lane is wider than the court")
    if not 0 < layout.basket_from_baseline < layout.half_length:
        problems.append("basket centre is outside the half court")

    for feature, primitive in layout.markings.items():
        if primitive.width_ft <= 0:
            problems.append(f"{feature.value} has non-positive width")
        if not math.isclose(primitive.width_ft, layout.line_width, abs_tol=1e-12):
            problems.append(f"{feature.value} width disagrees with layout line width")
        for x, y in layout.sample_marking(feature):
            if not (math.isfinite(x) and math.isfinite(y)):
                problems.append(f"{feature.value} has non-finite geometry")
                break
            if not 0.0 <= y <= layout.width:
                problems.append(f"{feature.value} leaves the court width at y={y}")
                break
            if feature is not MarkingFeature.CENTER_CIRCLE and not 0.0 <= x <= layout.half_length:
                problems.append(f"{feature.value} leaves the half-court depth at x={x}")
                break

    if set(layout.markings) == set(MarkingFeature):
        far_corner = layout.marking(MarkingFeature.THREE_POINT_CORNER_FAR)
        near_corner = layout.marking(MarkingFeature.THREE_POINT_CORNER_NEAR)
        arc = layout.marking(MarkingFeature.THREE_POINT_ARC)
        assert isinstance(far_corner, StraightMarking)
        assert isinstance(near_corner, StraightMarking)
        assert isinstance(arc, ArcMarking)
        arc_points = arc.sample(2)
        if not _close(far_corner.end, arc_points[0]):
            problems.append("far three-point corner does not join the arc")
        if not _close(near_corner.end, arc_points[-1]):
            problems.append("near three-point corner does not join the arc")
        if not math.isclose(arc.radius_ft, layout.three_point_radius, abs_tol=1e-12):
            problems.append("three-point primitive radius disagrees with layout")

        free_far = layout.marking(MarkingFeature.FREE_THROW_CIRCLE_FAR_HALF)
        free_near = layout.marking(MarkingFeature.FREE_THROW_CIRCLE_NEAR_HALF)
        center = layout.marking(MarkingFeature.CENTER_CIRCLE)
        restricted = layout.marking(MarkingFeature.RESTRICTED_AREA_ARC)
        for feature, primitive, radius in (
            (MarkingFeature.FREE_THROW_CIRCLE_FAR_HALF, free_far, layout.free_throw_circle_radius),
            (MarkingFeature.FREE_THROW_CIRCLE_NEAR_HALF, free_near, layout.free_throw_circle_radius),
            (MarkingFeature.CENTER_CIRCLE, center, layout.center_circle_radius),
            (MarkingFeature.RESTRICTED_AREA_ARC, restricted, layout.restricted_area_radius),
        ):
            if not isinstance(primitive, ArcMarking) or not math.isclose(primitive.radius_ft, radius, abs_tol=1e-12):
                problems.append(f"{feature.value} radius disagrees with layout")

        if isinstance(free_far, ArcMarking) and isinstance(free_near, ArcMarking):
            far_ends = free_far.sample(2)
            near_ends = free_near.sample(2)
            if not (_close(far_ends[0], near_ends[-1]) and _close(far_ends[-1], near_ends[0])):
                problems.append("free-throw circle halves do not share exact endpoints")
            if not (_close(free_far.center, (layout.free_throw_distance, centre_y)) and
                    _close(free_near.center, (layout.free_throw_distance, centre_y))):
                problems.append("free-throw circle center disagrees with free-throw line")
        if isinstance(center, ArcMarking) and not _close(center.center, (layout.half_length, centre_y)):
            problems.append("center circle is not centered on midcourt")
        if isinstance(restricted, ArcMarking) and not _close(restricted.center, layout.basket_center):
            problems.append("restricted-area arc is not centered on the basket")

        expected_landmarks = {
            HalfCourtLandmark.BASELINE_SIDELINE_FAR: layout.marking(MarkingFeature.BASELINE).start,
            HalfCourtLandmark.BASELINE_SIDELINE_NEAR: layout.marking(MarkingFeature.BASELINE).end,
            HalfCourtLandmark.THREE_POINT_BASELINE_FAR: far_corner.start,
            HalfCourtLandmark.THREE_POINT_BASELINE_NEAR: near_corner.start,
            HalfCourtLandmark.LANE_BASELINE_FAR: layout.marking(MarkingFeature.LANE_EDGE_FAR).start,
            HalfCourtLandmark.LANE_BASELINE_NEAR: layout.marking(MarkingFeature.LANE_EDGE_NEAR).start,
            HalfCourtLandmark.LANE_FREE_THROW_FAR: layout.marking(MarkingFeature.LANE_EDGE_FAR).end,
            HalfCourtLandmark.LANE_FREE_THROW_NEAR: layout.marking(MarkingFeature.LANE_EDGE_NEAR).end,
            HalfCourtLandmark.MIDCOURT_SIDELINE_FAR: layout.marking(MarkingFeature.MIDCOURT_LINE).start,
            HalfCourtLandmark.MIDCOURT_SIDELINE_NEAR: layout.marking(MarkingFeature.MIDCOURT_LINE).end,
        }
        for landmark, expected in expected_landmarks.items():
            if not layout.has(landmark) or not _close(layout.coordinate(landmark), expected):
                problems.append(f"{landmark.value} disagrees with marking topology")
        apex = (layout.free_throw_distance + layout.free_throw_circle_radius, centre_y)
        if not layout.has(HalfCourtLandmark.FREE_THROW_CIRCLE_APEX) or not _close(
            layout.coordinate(HalfCourtLandmark.FREE_THROW_CIRCLE_APEX), apex
        ):
            problems.append("free-throw circle apex disagrees with primitive")

    return problems
