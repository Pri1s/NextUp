"""Benchmark adapters over authoritative layout-owned marking geometry.

Depth bands and NumPy conversion are benchmark concerns. Court dimensions,
centerlines, widths, radii, topology, and sampling all belong to HalfCourtLayout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from contracts.markings import MarkingFeature
from contracts.court_layout import HalfCourtLayout, validate_layout

DEPTH_BANDS: tuple[tuple[str, float, float], ...] = (
    ("0-19ft", 0.0, 19.0),
    ("19-30ft", 19.0, 30.0),
    ("30-47ft", 30.0, 47.0),
)


def depth_band(court_x: float) -> str:
    for name, low, high in DEPTH_BANDS:
        if low <= court_x < high:
            return name
    return DEPTH_BANDS[-1][0] if court_x >= DEPTH_BANDS[-1][1] else DEPTH_BANDS[0][0]


@dataclass(frozen=True, slots=True)
class MarkingPolyline:
    feature: MarkingFeature
    points: np.ndarray
    width_ft: float
    closed: bool


def marking_polylines(layout: HalfCourtLayout) -> dict[MarkingFeature, MarkingPolyline]:
    """Sample each layout primitive into the benchmark's NumPy representation."""
    return {
        feature: MarkingPolyline(
            feature=feature,
            points=np.asarray(layout.sample_marking(feature), dtype=np.float64),
            width_ft=primitive.width_ft,
            closed=primitive.closed,
        )
        for feature, primitive in layout.markings.items()
    }


def depth_span(polyline: MarkingPolyline) -> tuple[float, float]:
    xs = polyline.points[:, 0]
    return float(np.min(xs)), float(np.max(xs))


def validate_geometry(layout: HalfCourtLayout) -> list[str]:
    """Benchmark-facing checks over the one authoritative geometry source."""
    problems = list(validate_layout(layout))
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
        if polylines[feature].width_ft <= 0:
            problems.append(f"{feature.value} has non-positive paint width")

    if MarkingFeature.THREE_POINT_ARC in polylines:
        arc_low, arc_high = depth_span(polylines[MarkingFeature.THREE_POINT_ARC])
        if arc_high <= 19.0:
            problems.append(
                f"three-point arc reaches only {arc_high:.1f} ft; expected depth evidence"
            )
        if arc_low < 0.0:
            problems.append("three-point arc crosses the baseline")
    return problems
