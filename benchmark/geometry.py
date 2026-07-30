"""Court marking geometry, and the one place its scalars are named.

Two calibration-relevant dimensions -- the three-point radius and the restricted-area
radius -- currently live as default arguments on ``calibration.viz.court_markings``
rather than as layout data. Milestone 2 moves them into the layout layer so that
rendering and fitting consume one source. Until then, every benchmark consumer
reads them from here, so the migration touches one module instead of several.

The benchmark also needs something the visualisation never did: for a point on a
projected marking, *how far down the court is it?* Error at 40 feet and error at 5
feet are the difference between a transform that describes the floor and one that
merely fits its own evidence, and averaging them together hides exactly the effect
this benchmark was built to measure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from contracts.annotations import MarkingFeature
from contracts.court_layout import HalfCourtLayout

from calibration.viz import court_markings as _engine_markings

#: Milestone 2 relocates both into ``HalfCourtLayout``.
THREE_POINT_RADIUS_FT = 23.75
RESTRICTED_AREA_RADIUS_FT = 4.0

#: Depth bands in feet from the visible baseline. The first is the span the current
#: confident evidence actually covers; everything beyond it is extrapolation.
DEPTH_BANDS: tuple[tuple[str, float, float], ...] = (
    ("0-19ft", 0.0, 19.0),
    ("19-30ft", 19.0, 30.0),
    ("30-47ft", 30.0, 47.0),
)


def depth_band(court_x: float) -> str:
    """Name the band a court-space x falls in; the last band absorbs overshoot."""
    for name, low, high in DEPTH_BANDS:
        if low <= court_x < high:
            return name
    return DEPTH_BANDS[-1][0] if court_x >= DEPTH_BANDS[-1][1] else DEPTH_BANDS[0][0]


@dataclass(frozen=True, slots=True)
class MarkingPolyline:
    """One court marking as an ordered court-space polyline, in feet."""

    feature: MarkingFeature
    points: np.ndarray  # (N, 2) float64, half-court feet

    @property
    def closed(self) -> bool:
        return self.feature is MarkingFeature.CENTER_CIRCLE


def _arc(cx: float, cy: float, radius: float, start: float, end: float, count: int) -> np.ndarray:
    angles = np.linspace(start, end, count)
    return np.stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)], axis=1)


def marking_polylines(layout: HalfCourtLayout) -> dict[MarkingFeature, MarkingPolyline]:
    """Every annotatable marking in half-court feet, keyed by annotation vocabulary.

    ``calibration.viz.court_markings`` groups by *drawing* convenience -- the whole
    three-point line is one polyline, the free-throw circle is a full circle. The
    benchmark needs the annotation vocabulary instead, where the corner straights
    are separable from the arc (only the arc carries depth beyond 19 ft, §4.1) and
    the circle's midcourt-side half is a distinct held-out family (§4.3).
    """
    width = layout.width
    centre_y = width / 2.0
    lane_far = centre_y - layout.lane_width / 2.0
    lane_near = centre_y + layout.lane_width / 2.0
    free_throw = layout.free_throw_distance
    basket_x = layout.basket_from_baseline
    inset = layout.three_point_corner_inset
    circle_r = layout.free_throw_circle_radius

    # Where the three-point arc meets the corner straights.
    dy = centre_y - inset
    break_x = basket_x + math.sqrt(max(THREE_POINT_RADIUS_FT**2 - dy**2, 0.0))
    arc_start = math.atan2(inset - centre_y, break_x - basket_x)
    arc_end = math.atan2(width - inset - centre_y, break_x - basket_x)

    polylines: dict[MarkingFeature, np.ndarray] = {
        MarkingFeature.BASELINE: np.array([[0.0, 0.0], [0.0, width]]),
        MarkingFeature.SIDELINE_FAR: np.array([[0.0, 0.0], [layout.half_length, 0.0]]),
        MarkingFeature.SIDELINE_NEAR: np.array([[0.0, width], [layout.half_length, width]]),
        MarkingFeature.LANE_EDGE_FAR: np.array([[0.0, lane_far], [free_throw, lane_far]]),
        MarkingFeature.LANE_EDGE_NEAR: np.array([[0.0, lane_near], [free_throw, lane_near]]),
        MarkingFeature.FREE_THROW_LINE: np.array(
            [[free_throw, lane_far], [free_throw, lane_near]]
        ),
        MarkingFeature.MIDCOURT_LINE: np.array(
            [[layout.half_length, 0.0], [layout.half_length, width]]
        ),
        MarkingFeature.THREE_POINT_CORNER_FAR: np.array([[0.0, inset], [break_x, inset]]),
        MarkingFeature.THREE_POINT_CORNER_NEAR: np.array(
            [[0.0, width - inset], [break_x, width - inset]]
        ),
        MarkingFeature.THREE_POINT_ARC: _arc(
            basket_x, centre_y, THREE_POINT_RADIUS_FT, arc_start, arc_end, 80
        ),
        # Midcourt-side half: the semicircle reaching away from the baseline.
        MarkingFeature.FREE_THROW_CIRCLE_FAR_HALF: _arc(
            free_throw, centre_y, circle_r, -math.pi / 2, math.pi / 2, 40
        ),
        MarkingFeature.FREE_THROW_CIRCLE_NEAR_HALF: _arc(
            free_throw, centre_y, circle_r, math.pi / 2, 3 * math.pi / 2, 40
        ),
        MarkingFeature.RESTRICTED_AREA_ARC: _arc(
            basket_x, centre_y, RESTRICTED_AREA_RADIUS_FT, -math.pi / 2, math.pi / 2, 40
        ),
        MarkingFeature.CENTER_CIRCLE: _arc(
            layout.half_length, centre_y, 6.0, 0.0, 2 * math.pi, 72
        ),
    }

    return {
        feature: MarkingPolyline(feature, np.asarray(points, dtype=np.float64))
        for feature, points in polylines.items()
    }


def depth_span(polyline: MarkingPolyline) -> tuple[float, float]:
    """Minimum and maximum distance from the baseline this marking covers, in feet."""
    xs = polyline.points[:, 0]
    return float(np.min(xs)), float(np.max(xs))


def _consistency_problems(layout: HalfCourtLayout) -> list[str]:
    """Cross-check this module against the engine's own rendering geometry.

    The two exist for different reasons and are grouped differently, but they
    describe the same floor. If they ever disagree the benchmark would be scoring
    against a court the engine does not draw, so the disagreement must be loud.
    """
    problems: list[str] = []
    engine = _engine_markings(
        layout,
        three_point_radius=THREE_POINT_RADIUS_FT,
        restricted_radius=RESTRICTED_AREA_RADIUS_FT,
    )
    ours = marking_polylines(layout)

    engine_lane = np.asarray(engine["lane"], dtype=np.float64)
    for feature, expected in (
        (MarkingFeature.LANE_EDGE_FAR, engine_lane[:2]),
        (MarkingFeature.LANE_EDGE_NEAR, engine_lane[2:][::-1]),
    ):
        got = ours[feature].points
        if not np.allclose(np.sort(got, axis=0), np.sort(expected, axis=0), atol=1e-9):
            problems.append(f"{feature.value} disagrees with calibration.viz lane geometry")

    engine_three = np.asarray(engine["three_point"], dtype=np.float64)
    arc = ours[MarkingFeature.THREE_POINT_ARC].points
    for point in (arc[0], arc[-1], arc[len(arc) // 2]):
        distances = np.linalg.norm(engine_three - point, axis=1)
        if float(np.min(distances)) > 1e-6:
            problems.append("three_point_arc leaves the engine's three-point polyline")
            break

    return problems


def validate_geometry(layout: HalfCourtLayout) -> list[str]:
    """Structural checks on the derived markings. Empty list means healthy."""
    problems = _consistency_problems(layout)
    polylines = marking_polylines(layout)

    for feature in MarkingFeature:
        if feature not in polylines:
            problems.append(f"{feature.value} has no court geometry")
            continue
        points = polylines[feature].points
        if not np.all(np.isfinite(points)):
            problems.append(f"{feature.value} contains non-finite coordinates")
        if feature.kind == "arc" and len(points) < 5:
            problems.append(f"{feature.value} is an arc with only {len(points)} points")

    arc_low, arc_high = depth_span(polylines[MarkingFeature.THREE_POINT_ARC])
    if arc_high <= 19.0:
        problems.append(
            f"three-point arc reaches only {arc_high:.1f} ft; it is the benchmark's "
            f"main source of evidence beyond the current 19 ft span"
        )
    if arc_low < 0.0:
        problems.append("three-point arc crosses the baseline")

    return problems
