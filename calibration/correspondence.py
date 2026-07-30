"""Pair pooled evidence with known court coordinates.

The only place where model-derived observations meet court geometry. Everything
upstream is court-agnostic; everything downstream is model-agnostic.

Drops are recorded with reasons rather than filtered silently. With a detector
whose semantics are provisional, the drop log is often the first sign that a slot
mapping has stopped holding.
"""

from __future__ import annotations

from typing import Sequence

from contracts.calibration_types import Correspondence, CorrespondenceSet, LandmarkEvidence
from contracts.court_layout import HalfCourtLandmark, HalfCourtLayout


def build_correspondences(
    evidence: Sequence[LandmarkEvidence],
    layout: HalfCourtLayout,
    min_confidence: float = 0.25,
) -> CorrespondenceSet:
    """Match evidence to layout coordinates.

    Weights are the pooled confidences themselves, so a landmark two slots agree
    strongly on outranks one a single slot barely saw.
    """
    kept: list[Correspondence] = []
    dropped: list[tuple[str, str]] = []

    for item in evidence:
        if item.confidence < min_confidence:
            dropped.append((item.landmark_id, f"below_confidence:{item.confidence:.3f}"))
            continue
        try:
            landmark = HalfCourtLandmark(item.landmark_id)
        except ValueError:
            dropped.append((item.landmark_id, "unknown_landmark"))
            continue
        if not layout.has(landmark):
            dropped.append((item.landmark_id, f"absent_from_layout:{layout.layout_id}"))
            continue

        kept.append(
            Correspondence(
                landmark_id=item.landmark_id,
                image_xy=(item.x, item.y),
                court_xy=layout.coordinate(landmark),
                weight=item.confidence,
                source_slots=item.source_slots,
            )
        )

    return CorrespondenceSet(correspondences=tuple(kept), dropped=tuple(dropped))
