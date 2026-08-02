"""Residual construction for the shadow hybrid refiner.

The important detail in this module is that a marking sample contributes only
the distance normal to its assigned primitive.  Its location along the
primitive is unknown, so the projected foot is an ICP auxiliary variable and
is frozen for one optimizer outer iteration.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from contracts.calibration_types import Correspondence
from contracts.court_layout import HalfCourtLayout
from contracts.hybrid_types import (
    AssignmentStatus,
    MarkingAssignment,
    ShadowMarkingFrame,
)
from contracts.markings import MarkingFeature

from ..estimator import project
from ..markings.extractor import diagonal_scale
from ..polyline import nearest_on_polyline, polyline_length
from .control_points import homography_from_control_points


@dataclass(frozen=True, slots=True)
class MarkingObservation:
    """Immutable fitted samples for one primitive/evidence fragment."""

    feature: MarkingFeature
    evidence_id: str
    image_xy: np.ndarray
    sigma_px: np.ndarray
    confidence: np.ndarray
    foot_court_xy: np.ndarray
    #: Endpoints of the template segment the foot landed on.  The local
    #: direction is taken from the whole segment, never from the foot to the
    #: next vertex: that secant is systematically rotated on a curve, so its
    #: error is directional rather than noise, and it collapses to zero length
    #: whenever the foot lands near a vertex.
    segment_start_court_xy: np.ndarray
    segment_end_court_xy: np.ndarray
    weight: np.ndarray
    depth_ft: np.ndarray


def _sample_count(feature: MarkingFeature, config) -> int:
    return int(getattr(config, "arc_samples", 401) if feature.kind == "arc"
               else getattr(config, "straight_samples", 201))


def _template(feature: MarkingFeature, layout: HalfCourtLayout, config) -> np.ndarray:
    return np.asarray(layout.marking(feature).sample(_sample_count(feature, config)), dtype=np.float64)


def _observation_from_samples(
    feature: MarkingFeature,
    evidence_id: str,
    image_xy: np.ndarray,
    sigma_px: np.ndarray,
    confidence: np.ndarray,
    layout: HalfCourtLayout,
    matrix: np.ndarray,
    config,
) -> MarkingObservation:
    court_template = _template(feature, layout, config)
    image_template = project(matrix, court_template)
    if image_template is None or not np.all(np.isfinite(image_template)):
        raise ValueError(f"template projection failed for {feature.value}")
    nearest = nearest_on_polyline(image_xy, image_template)
    foot = nearest.interpolate(court_template)
    start_index = np.clip(nearest.segment, 0, len(court_template) - 2)
    return MarkingObservation(
        feature=feature,
        evidence_id=str(evidence_id),
        image_xy=np.asarray(image_xy, dtype=np.float64).copy(),
        sigma_px=np.asarray(sigma_px, dtype=np.float64).copy(),
        confidence=np.asarray(confidence, dtype=np.float64).copy(),
        foot_court_xy=foot.copy(),
        segment_start_court_xy=court_template[start_index].copy(),
        segment_end_court_xy=court_template[start_index + 1].copy(),
        weight=np.zeros(len(image_xy), dtype=np.float64),
        depth_ft=foot[:, 0].copy(),
    )


def _freeze(observation: MarkingObservation, matrix: np.ndarray, layout: HalfCourtLayout, config) -> MarkingObservation:
    return _observation_from_samples(
        observation.feature,
        observation.evidence_id,
        observation.image_xy,
        observation.sigma_px,
        observation.confidence,
        layout,
        matrix,
        config,
    )


def _territory_weights(observations: list[MarkingObservation], budget: float, matrix: np.ndarray, layout: HalfCourtLayout, config) -> np.ndarray:
    """Return weights proportional to occupied projected-template territory.

    One-pixel bins are used instead of sample counts.  If a detector changes
    its resampling spacing, the same physical territory receives the same
    total weight and only the number of pieces changes.
    """
    if not observations:
        return np.zeros(0, dtype=np.float64)
    feature = observations[0].feature
    template = _template(feature, layout, config)
    projected = project(matrix, template)
    if projected is None:
        return np.zeros(sum(len(item.image_xy) for item in observations), dtype=np.float64)
    lengths = np.linalg.norm(np.diff(projected, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    all_arc = np.concatenate([
        nearest_on_polyline(item.image_xy, projected).arc_length for item in observations
    ])
    bin_size = max(float(getattr(config, "territory_bin_px", 1.0)), 1e-9)
    bins = np.floor(all_arc / bin_size).astype(np.int64)
    unique, counts = np.unique(bins, return_counts=True)
    territory = np.zeros(len(all_arc), dtype=np.float64)
    for item_bin, count in zip(unique, counts):
        start = float(item_bin) * bin_size
        end = min(start + bin_size, float(cumulative[-1]))
        share = max(end - start, 0.0) / float(count)
        territory[bins == item_bin] = share
    if not np.any(territory > 0):
        territory.fill(1.0)
    return budget * territory / float(np.sum(territory))


def rebalance(observations, config, image_shape) -> tuple[MarkingObservation, ...]:
    """Apply primitive, family, then global marking budgets."""
    observations = tuple(observations)
    if not observations:
        return ()
    matrix = getattr(config, "_matrix_for_rebalance", None)
    layout = getattr(config, "_layout_for_rebalance", None)
    if matrix is None or layout is None:
        # Observations normally arrive already frozen and this fallback keeps
        # the public helper useful for tests that only exercise its arithmetic.
        grouped: dict[MarkingFeature, list[MarkingObservation]] = {}
        for item in observations:
            grouped.setdefault(item.feature, []).append(item)
        result = []
        for feature in sorted(grouped, key=lambda value: value.value):
            group = grouped[feature]
            length = max(float(np.ptp(np.concatenate([item.depth_ft for item in group]))), 0.0)
            reference = float(getattr(config, "reference_length_px", 120.0)) * diagonal_scale(image_shape)
            budget = float(getattr(config, "beta", 1.0)) * min(1.0, length / max(reference, 1e-9))
            total = sum(float(np.sum(item.weight)) for item in group)
            for item in group:
                w = item.weight.copy()
                if total > 0:
                    w *= budget / total
                result.append(replace(item, weight=w))
        return tuple(sorted(result, key=lambda item: (item.feature.value, item.evidence_id)))

    grouped: dict[MarkingFeature, list[MarkingObservation]] = {}
    for item in observations:
        grouped.setdefault(item.feature, []).append(item)
    raw: dict[MarkingFeature, tuple[MarkingObservation, ...]] = {}
    reference = float(getattr(config, "reference_length_px", 120.0)) * diagonal_scale(image_shape)
    beta = float(getattr(config, "beta", 1.0))
    for feature in sorted(grouped, key=lambda value: value.value):
        group = sorted(grouped[feature], key=lambda item: item.evidence_id)
        template = _template(feature, layout, config)
        projected = project(matrix, template)
        if projected is None:
            support = 0.0
        else:
            arcs = np.concatenate([nearest_on_polyline(item.image_xy, projected).arc_length for item in group])
            # The occupied extent is a stable approximation of union territory;
            # it is intentionally independent of the number of samples.
            support = float(np.ptp(arcs)) if len(arcs) > 1 else 0.0
        budget = beta * min(1.0, support / max(reference, 1e-9))
        weights = _territory_weights(group, budget, matrix, layout, config)
        cursor = 0
        updated = []
        for item in group:
            count = len(item.image_xy)
            updated.append(replace(item, weight=weights[cursor:cursor + count]))
            cursor += count
        raw[feature] = tuple(updated)

    # Family cap.
    family_totals: dict[object, float] = {}
    for feature, group in raw.items():
        family_totals[feature.family] = family_totals.get(feature.family, 0.0) + sum(float(np.sum(i.weight)) for i in group)
    fam_cap = float(getattr(config, "family_cap", 2.0))
    for feature, group in list(raw.items()):
        scale = min(1.0, fam_cap / family_totals[feature.family]) if family_totals[feature.family] > 0 else 0.0
        raw[feature] = tuple(replace(item, weight=item.weight * scale) for item in group)

    total = sum(float(np.sum(item.weight)) for group in raw.values() for item in group)
    glob_cap = float(getattr(config, "global_cap", 4.0))
    scale = min(1.0, glob_cap / total) if total > 0 else 0.0
    return tuple(
        replace(item, weight=item.weight * scale)
        for feature in sorted(raw, key=lambda value: value.value)
        for item in raw[feature]
    )


def collect_observations(shadow: ShadowMarkingFrame, layout: HalfCourtLayout, matrix, config, image_shape) -> tuple[MarkingObservation, ...]:
    """Collect only accepted, non-held-out evidence in deterministic order."""
    evidence = {item.evidence_id: item for item in shadow.evidence}
    held_out = set(getattr(config, "held_out_features", ()))
    assignments = sorted(shadow.assignments, key=lambda item: (
        item.selected_feature.value if item.selected_feature is not None else "",
        item.evidence_id,
    ))
    observations = []
    for assignment in assignments:
        if assignment.status is not AssignmentStatus.ACCEPTED or assignment.selected_feature is None:
            continue
        if assignment.selected_feature in held_out:
            continue
        item = evidence[assignment.evidence_id]
        samples = tuple(item.samples)
        image_xy = np.asarray([(sample.x, sample.y) for sample in samples], dtype=np.float64)
        sigma = np.asarray([
            np.clip(sample.uncertainty_px,
                    float(getattr(config, "marking_sigma_min_px", 0.5)),
                    float(getattr(config, "marking_sigma_max_px", 4.0)))
            for sample in samples
        ], dtype=np.float64)
        confidence = np.asarray([sample.confidence for sample in samples], dtype=np.float64)
        observations.append(_observation_from_samples(
            assignment.selected_feature, item.evidence_id, image_xy, sigma, confidence,
            layout, np.asarray(matrix, dtype=np.float64), config,
        ))
    # Store the context privately on a shallow config proxy only for rebalance;
    # frozen dataclass configs cannot be mutated. The actual refiner calls the
    # context-aware helper below.
    return rebalance_with_context(observations, config, image_shape, layout, np.asarray(matrix, dtype=np.float64))


def rebalance_with_context(observations, config, image_shape, layout, matrix) -> tuple[MarkingObservation, ...]:
    """Context-bearing variant used internally by collection and ICP freezing."""
    class _Context:
        def __getattr__(self, name):
            return getattr(config, name)
    context = _Context()
    context._layout_for_rebalance = layout
    context._matrix_for_rebalance = matrix
    return rebalance(observations, context, image_shape)


def freeze_observations(observations, matrix, layout, config, image_shape) -> tuple[MarkingObservation, ...]:
    frozen = [_freeze(item, np.asarray(matrix, dtype=np.float64), layout, config) for item in observations]
    return rebalance_with_context(frozen, config, image_shape, layout, np.asarray(matrix, dtype=np.float64))


def _homography(theta: np.ndarray, layout: HalfCourtLayout) -> np.ndarray:
    from .control_points import control_court_points
    matrix = homography_from_control_points(control_court_points(layout), np.asarray(theta).reshape(4, 2))
    if matrix is None:
        raise ValueError("control points do not define a homography")
    return matrix


def marking_residuals(theta, observations, layout) -> np.ndarray:
    matrix = _homography(np.asarray(theta, dtype=np.float64), layout)
    values = []
    for item in observations:
        # The point comes from the interpolated foot; the direction comes from
        # the whole template segment.  Separating them is what makes the
        # residual purely perpendicular: a template edge has a fixed, non-zero
        # length wherever the foot happens to sit along it.
        a = project(matrix, item.foot_court_xy)
        start = project(matrix, item.segment_start_court_xy)
        end = project(matrix, item.segment_end_court_xy)
        if a is None or start is None or end is None:
            raise ValueError(f"marking projection failed for {item.feature.value}")
        direction = end - start
        length = np.linalg.norm(direction, axis=1)
        if np.any(length <= 1e-12):
            raise ValueError(f"degenerate template segment for {item.feature.value}")
        unit = direction / length[:, None]
        normal = np.stack((-unit[:, 1], unit[:, 0]), axis=1)
        values.extend(np.einsum("ij,ij->i", normal, item.image_xy - a))
    return np.asarray(values, dtype=np.float64)


def landmark_residuals(theta, correspondences, layout) -> np.ndarray:
    matrix = _homography(np.asarray(theta, dtype=np.float64), layout)
    court = np.asarray([item.court_xy for item in correspondences], dtype=np.float64)
    image = np.asarray([item.image_xy for item in correspondences], dtype=np.float64)
    projected = project(matrix, court)
    if projected is None:
        raise ValueError("landmark projection failed")
    return (projected - image).reshape(-1)


def measure_feature_residuals(shadow, layout: HalfCourtLayout, matrix, config) -> dict:
    """Freshly measure every accepted fragment against a candidate transform.

    Returns ``{feature: [(distance_px, depth_ft), ...]}``.

    Two properties make this a *measurement* rather than a restatement of the
    optimizer's objective, and both matter for the records it feeds:

    - the foot points are re-derived against ``matrix``. Reusing the ones the
      refiner froze would report the quantity it was minimising, flattering the
      candidate exactly where it is wrong;
    - the distance is the clamped point-to-polyline distance, and the depth is
      read from the *court* polyline at the same segment and local position, so
      band membership cannot shift with the transform under test.

    This is the only implementation. An earlier second one measured the
    perpendicular component to a single segment instead, which agrees in a
    segment's interior and diverges at a clamped endpoint -- two numbers for one
    quantity, free to drift apart.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    evidence = {item.evidence_id: item for item in shadow.evidence}
    rows: dict[MarkingFeature, list[tuple[float, float]]] = {}
    for assignment in sorted(shadow.assignments, key=lambda item: item.evidence_id):
        feature = assignment.selected_feature
        if assignment.status is not AssignmentStatus.ACCEPTED or feature is None:
            continue
        template = _template(feature, layout, config)
        projected = project(matrix, template)
        if projected is None:
            continue
        item = evidence[assignment.evidence_id]
        image = np.asarray([(sample.x, sample.y) for sample in item.samples], dtype=np.float64)
        nearest = nearest_on_polyline(image, projected)
        court_foot = nearest.interpolate(template)
        rows.setdefault(feature, []).extend(
            (float(distance), float(depth))
            for distance, depth in zip(nearest.distances, court_foot[:, 0])
        )
    return rows


def control_points_from_matrix(matrix, layout):
    from .control_points import control_points_from_homography
    return control_points_from_homography(np.asarray(matrix, dtype=np.float64), layout)

