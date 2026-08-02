"""The feature definitions an annotator reads.

Written for someone who has never seen this court and cannot ask a follow-up
question. Two kinds of content earn their place:

*Identity* -- what the marking is, in rule-book terms, with its dimensions derived
from the layout so the prose cannot drift away from the geometry the benchmark
scores against.

*Discrimination* -- what it is confusable with, and how to tell. This is the part
that prevents the failure mode the whole benchmark is built to avoid. An annotator
who traces a floor logo's curve as the three-point arc produces a label that looks
perfectly well-formed, passes every structural check, and silently corrupts the
comparison. Half of these entries exist to name the specific wrong thing nearby.
"""

from __future__ import annotations

import numpy as np

from contracts.annotations import (
    HELD_OUT_FAMILIES,
    LINE_CONVENTION,
    JunctionPoint,
    MarkingFeature,
)
from contracts.court_layout import HalfCourtLayout

from .geometry import depth_span, marking_polylines

#: Bumped whenever wording changes in a way that could alter what an annotator
#: produces. Recorded in every annotation file, because a prompt change silently
#: invalidates earlier labels otherwise.
GLOSSARY_VERSION = "glossary-1.0.0"

_DESCRIPTIONS: dict[MarkingFeature, tuple[str, str]] = {
    MarkingFeature.BASELINE: (
        "The end line under the basket, running the full width of the court.",
        "It is often cropped by the broadcast frame or hidden behind photographers, "
        "stanchion padding and player legs. Do not confuse it with the edge of the "
        "wooden floor, with an advertising border painted outside the court, or with "
        "the shadow line where the floor meets the surround.",
    ),
    MarkingFeature.SIDELINE_FAR: (
        "The long boundary on the side away from the camera.",
        "Frequently confused with the scorer's-table edge, bench markings, or a "
        "sponsor stripe running parallel just outside it. If two parallel long lines "
        "are visible near the top of the frame, the court boundary is the inner one.",
    ),
    MarkingFeature.SIDELINE_NEAR: (
        "The long boundary on the side toward the camera.",
        "Usually the most cropped marking in a broadcast frame. Annotate only the "
        "portion actually visible; do not extend it to where you believe it goes.",
    ),
    MarkingFeature.LANE_EDGE_FAR: (
        "The lane (paint) boundary on the far-sideline side, from baseline to "
        "free-throw line.",
        "Runs parallel to the far lane edge on the opposite side of the paint. Tell "
        "them apart by which side of the basket they are on. Lane block marks and "
        "hash marks sit *on* this line -- trace through them, they are not separate "
        "markings.",
    ),
    MarkingFeature.LANE_EDGE_NEAR: (
        "The lane (paint) boundary on the near-sideline side, from baseline to "
        "free-throw line.",
        "As above. The painted lane interior often has a strong colour edge that is "
        "not the line itself; the marking is the white stripe, not the colour "
        "boundary, and the two can sit a few pixels apart.",
    ),
    MarkingFeature.FREE_THROW_LINE: (
        "The line at the top of the lane that the shooter stands behind.",
        "It spans only the lane width, not the whole court. Its ends are junctions "
        "with the lane edges. The free-throw circle touches its midpoint -- do not "
        "let the circle's curve pull your samples off the straight line.",
    ),
    MarkingFeature.FREE_THROW_CIRCLE_FAR_HALF: (
        "The half of the free-throw circle on the midcourt side of the free-throw "
        "line -- the semicircle bulging away from the basket.",
        "Solid on an NBA floor. Distinguish it from the near half, which is dashed "
        "and sits inside the paint. If you cannot tell which half you are on, use "
        "the free-throw line: this half is on the side away from the basket.",
    ),
    MarkingFeature.FREE_THROW_CIRCLE_NEAR_HALF: (
        "The half of the free-throw circle inside the lane, on the basket side.",
        "Dashed on an NBA floor, so it is a sequence of short segments rather than a "
        "continuous curve. Sample only where paint is actually present; the gaps are "
        "occlusion of a sort and belong in occluded_after.",
    ),
    MarkingFeature.THREE_POINT_CORNER_FAR: (
        "The straight three-point segment in the far corner, running from the "
        "baseline until it meets the arc.",
        "Parallel to the far sideline and close to it -- the gap is small and the two "
        "are easy to swap. The three-point segment is the *inner* of the pair. It "
        "ends where it curves; the curve itself is three_point_arc.",
    ),
    MarkingFeature.THREE_POINT_CORNER_NEAR: (
        "The straight three-point segment in the near corner.",
        "As above, and usually the more cropped of the two.",
    ),
    MarkingFeature.THREE_POINT_ARC: (
        "The curved portion of the three-point line, between the two corner "
        "segments.",
        "The highest-value marking in this benchmark: it is the only one reaching "
        "well beyond the free-throw line. Confusable with the outer edge of a large "
        "centre-court or key logo, whose curve can be similar. The three-point arc is "
        "continuous, uniform in width, and the same colour as the other court lines; "
        "a logo edge usually is not. Where a logo overlaps it, sample only where you "
        "can see the line itself.",
    ),
    MarkingFeature.RESTRICTED_AREA_ARC: (
        "The small arc under the basket marking the restricted area.",
        "Small, close to the baseline, and very often occluded by players. Do not "
        "confuse it with the near half of the free-throw circle -- the restricted arc "
        "is much closer to the baseline and much tighter.",
    ),
    MarkingFeature.MIDCOURT_LINE: (
        "The halfway line, crossing the full width of the court.",
        "Visible only in transition or wide frames. Centre-court logos create strong "
        "false edges near it. Never annotate a midcourt line you inferred from the "
        "logo's symmetry; if the line itself is not visible, skip it.",
    ),
    MarkingFeature.CENTER_CIRCLE: (
        "The circle at centre court, straddling the midcourt line.",
        "Almost always overlapping a painted logo, which is the dominant source of "
        "false curves in these clips. Annotate only unambiguous painted line.",
    ),
}

_PROCEDURE = f"""\
## How to place a point

Coordinates mean the **{LINE_CONVENTION.replace('_', ' ')}**: the middle of the
painted stripe, halfway between its two edges. Not an edge, and not the rule-book
measurement line. Where a stripe is 4 px wide in the image, the correct answer sits
2 px inside either edge.

Work only in magnified crops. Request a crop centred on the part of the marking you
are sampling, read the coordinate off the crop's labelled grid, and record it in
**crop pixels** together with that crop's id. A tool converts crop coordinates back
to frame coordinates -- never do that arithmetic yourself, and never estimate a
coordinate from the full frame.

## What not to do

**Do not invent a point.** If paint is hidden behind a player, sample up to where it
disappears, record the gap, and resume on the far side. An interpolated point is
indistinguishable from an observed one once it is written down.

**Do not treat the end of visible paint as a landmark.** Where a marking runs out of
the frame or behind a body tells you about the camera, not about the court. Only a
visible *crossing* of two markings you have both traced is a landmark.

**Skip rather than guess.** Every feature you do not annotate needs a recorded
reason. "Ambiguous" is a perfectly good answer and is far more useful than a
confident wrong label -- a skipped feature costs coverage, a wrong one corrupts the
benchmark.
"""


def _dimensions(feature: MarkingFeature, layout: HalfCourtLayout) -> str:
    """Rule-book geometry, derived from the layout rather than typed in."""
    polylines = marking_polylines(layout)
    low, high = depth_span(polylines[feature])
    span = f"{low:.1f}-{high:.1f} ft from the baseline"

    extra = {
        MarkingFeature.THREE_POINT_ARC: (
            f"centerline radius {layout.three_point_radius:.3f} ft about the basket"
        ),
        MarkingFeature.RESTRICTED_AREA_ARC: (
            f"centerline radius {layout.restricted_area_radius:.3f} ft about the basket"
        ),
        MarkingFeature.FREE_THROW_CIRCLE_FAR_HALF: (
            f"centerline radius {layout.free_throw_circle_radius:.3f} ft "
            "about the free-throw line midpoint"
        ),
        MarkingFeature.FREE_THROW_CIRCLE_NEAR_HALF: (
            f"centerline radius {layout.free_throw_circle_radius:.3f} ft "
            "about the free-throw line midpoint"
        ),
        MarkingFeature.FREE_THROW_LINE: f"{layout.lane_width:.3f} ft centerline length",
        MarkingFeature.LANE_EDGE_FAR: f"lane centerlines are {layout.lane_width:.3f} ft apart",
        MarkingFeature.LANE_EDGE_NEAR: f"lane centerlines are {layout.lane_width:.3f} ft apart",
        MarkingFeature.THREE_POINT_CORNER_FAR: (
            f"centerline {layout.three_point_corner_inset:.3f} ft from sideline centerline"
        ),
        MarkingFeature.THREE_POINT_CORNER_NEAR: (
            f"centerline {layout.three_point_corner_inset:.3f} ft from sideline centerline"
        ),
    }.get(feature)

    return f"{span}; {extra}" if extra else span


def render_markdown(layout: HalfCourtLayout) -> str:
    """The full glossary, as the annotator receives it."""
    lines = [
        "# Court marking glossary",
        "",
        f"Layout `{layout.layout_id}` ({layout.rule_set}), "
        f"{layout.half_length} x {layout.width} ft half court. "
        f"Glossary version `{GLOSSARY_VERSION}`.",
        "",
        "Sides are named relative to the camera: **far** is the side away from you, "
        "**near** is the side toward you. Nothing here says which end of the court "
        "this is, and you are not being asked to decide.",
        "",
        _PROCEDURE,
        "",
        "## Markings",
        "",
    ]

    for feature in MarkingFeature:
        identity, discrimination = _DESCRIPTIONS[feature]
        lines.append(f"### `{feature.value}`")
        lines.append("")
        lines.append(f"*{feature.kind}* &mdash; {_dimensions(feature, layout)}")
        if feature in HELD_OUT_FAMILIES:
            lines.append("")
            lines.append(
                "> Held-out validation family. Annotate it exactly as carefully as the "
                "rest; it is deliberately never used for fitting, which is what makes "
                "it able to test a fit."
            )
        lines.append("")
        lines.append(identity)
        lines.append("")
        lines.append(f"**Telling it apart.** {discrimination}")
        lines.append("")

    lines += [
        "## Junctions",
        "",
        "A junction is a landmark only where you have traced **both** crossing "
        "markings and can see them meet. If either is missing, or the crossing is "
        "off-frame or occluded, do not record the junction.",
        "",
        "| junction | crossing of |",
        "| --- | --- |",
    ]
    for junction in JunctionPoint:
        a, b = junction.crossing
        lines.append(f"| `{junction.value}` | `{a.value}` and `{b.value}` |")
    lines.append("")

    return "\n".join(lines)


def feature_catalog(layout: HalfCourtLayout) -> list[dict]:
    """Structured glossary content for interactive labeling tools.

    The Markdown glossary and the browser labeler must teach the same identities
    and discrimination rules. Keeping the catalog here prevents a UI-only copy
    from quietly drifting away from the benchmark instructions.
    """
    catalog = []
    for feature in MarkingFeature:
        identity, discrimination = _DESCRIPTIONS[feature]
        catalog.append(
            {
                "id": feature.value,
                "kind": feature.kind,
                "dimensions": _dimensions(feature, layout),
                "description": identity,
                "discrimination": discrimination,
                "held_out": feature in HELD_OUT_FAMILIES,
                "minimum_samples": 6 if feature.kind == "arc" else 4,
            }
        )
    return catalog


def junction_catalog() -> list[dict]:
    """Structured junction vocabulary for interactive labeling tools."""
    return [
        {
            "id": junction.value,
            "crossing": [feature.value for feature in junction.crossing],
        }
        for junction in JunctionPoint
    ]


def reference_court(layout: HalfCourtLayout) -> dict:
    """Authoritative top-down marking geometry for the human labeler.

    The UI diagram is explanatory, but it must still depict the same court the
    scorer uses. Sending the derived polylines avoids maintaining a hand-drawn
    second definition of the three-point and circle geometry.
    """
    polylines = marking_polylines(layout)
    label_anchors = {}
    for feature, polyline in polylines.items():
        points = polyline.points
        segment_lengths = np.linalg.norm(points[1:] - points[:-1], axis=1)
        target = float(np.sum(segment_lengths)) / 2.0
        cumulative = np.cumsum(segment_lengths)
        segment_index = int(np.searchsorted(cumulative, target, side="left"))
        distance_before = (
            0.0 if segment_index == 0 else float(cumulative[segment_index - 1])
        )
        segment_length = float(segment_lengths[segment_index])
        if segment_length <= 1e-12:
            anchor = points[segment_index]
        else:
            amount = (target - distance_before) / segment_length
            anchor = (
                points[segment_index] * (1.0 - amount)
                + points[segment_index + 1] * amount
            )
        label_anchors[feature.value] = anchor.tolist()

    return {
        "half_length": layout.half_length,
        "width": layout.width,
        "basket_center": list(layout.basket_center),
        "markings": {
            feature.value: polyline.points.tolist()
            for feature, polyline in polylines.items()
        },
        "label_anchors": label_anchors,
    }


def validate_glossary() -> list[str]:
    """Every feature must be described. A missing entry is a silent instruction gap."""
    problems = []
    for feature in MarkingFeature:
        if feature not in _DESCRIPTIONS:
            problems.append(f"{feature.value} has no glossary entry")
            continue
        identity, discrimination = _DESCRIPTIONS[feature]
        if not identity.strip():
            problems.append(f"{feature.value} has no identity description")
        if not discrimination.strip():
            problems.append(f"{feature.value} has no discrimination guidance")
    return problems
