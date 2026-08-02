"""Model-seeded, deterministic association of unlabeled fragments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from contracts.court_layout import HalfCourtLayout
from contracts.hybrid_types import (
    AssignmentAlternative, AssignmentStatus, AssociationDiagnostics, MarkingAssignment,
    ShadowFamilySupport, ShadowFeatureSupport, UnlabeledMarkingEvidence,
)
from contracts.markings import ArcMarking, MarkingFamily, MarkingFeature

from calibration.estimator import project
from .extractor import corridor_width_px, diagonal_scale

ACCEPTED_FEATURES = frozenset({
    MarkingFeature.LANE_EDGE_FAR, MarkingFeature.LANE_EDGE_NEAR,
    MarkingFeature.FREE_THROW_LINE, MarkingFeature.THREE_POINT_CORNER_FAR,
    MarkingFeature.THREE_POINT_CORNER_NEAR, MarkingFeature.THREE_POINT_ARC,
    MarkingFeature.FREE_THROW_CIRCLE_FAR_HALF,
})


@dataclass(frozen=True, slots=True)
class AssociationConfig:
    min_corridor_fraction: float = 0.80
    straight_tangent_deg: float = 12.0
    arc_tangent_deg: float = 18.0
    min_width_ratio: float = 0.5
    max_width_ratio: float = 2.5
    min_uniqueness_margin: float = 0.20
    min_inlier_distance_px: float = 12.0
    arc_support_px: float = 60.0
    lane_join_px: float = 12.0
    circle_support_px: float = 40.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_corridor_fraction <= 1.0:
            raise ValueError("min_corridor_fraction must be in [0, 1]")
        if not 0.0 < self.min_uniqueness_margin <= 1.0:
            raise ValueError("min_uniqueness_margin must be in (0, 1]")
        if self.min_width_ratio <= 0 or self.max_width_ratio < self.min_width_ratio:
            raise ValueError("width ratios must be positive and ordered")
        for name in (
            "straight_tangent_deg", "arc_tangent_deg", "min_inlier_distance_px",
            "arc_support_px", "lane_join_px", "circle_support_px",
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")

    def as_dict(self) -> dict:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class AssociationResult:
    assignments: tuple[MarkingAssignment, ...]
    feature_support: tuple[ShadowFeatureSupport, ...]
    family_support: tuple[ShadowFamilySupport, ...]


@dataclass(frozen=True, slots=True)
class _Candidate:
    feature: MarkingFeature
    score: float
    distance_px: float
    tangent_error_deg: float
    evidence_coverage: float
    template_coverage: float
    width_ratio: float
    support_px: float
    visible_template_px: float
    corridor_fraction: float
    turning: float


class MarkingAssociator:
    """Compare every fragment against all 14 layout primitives as competitors."""

    def __init__(
        self,
        layout: HalfCourtLayout,
        h_court_to_image: np.ndarray | tuple,
        h_image_to_court: np.ndarray | tuple | None = None,
        config: AssociationConfig | None = None,
        image_shape: tuple[int, int] | tuple[int, int, int] = (720, 1280),
    ):
        self.layout = layout
        self.h_court_to_image = np.asarray(h_court_to_image, dtype=np.float64)
        self.h_image_to_court = (
            np.linalg.inv(self.h_court_to_image)
            if h_image_to_court is None else np.asarray(h_image_to_court, dtype=np.float64)
        )
        self.config = config or AssociationConfig()
        self.image_shape = image_shape
        self.pixel_scale = diagonal_scale(image_shape)
        self._templates = self._build_templates()

    def associate(self, evidence: Iterable[UnlabeledMarkingEvidence]) -> AssociationResult:
        items = tuple(sorted(evidence, key=lambda item: item.evidence_id))
        assignments: list[MarkingAssignment] = []
        selected: dict[MarkingFeature, list[tuple[str, _Candidate]]] = {feature: [] for feature in MarkingFeature}
        for item in items:
            candidates = sorted((self._score(item, feature) for feature in MarkingFeature), key=lambda c: (-c.score, c.feature.value))
            best = candidates[0] if candidates else None
            second = candidates[1] if len(candidates) > 1 else None
            margin = (best.score - second.score) if best and second else 1.0
            diagnostics = AssociationDiagnostics(
                distance_px=best.distance_px if best else 0.0,
                tangent_error_deg=best.tangent_error_deg if best else 90.0,
                coverage_fraction=best.evidence_coverage if best else 0.0,
                uniqueness_margin=max(0.0, margin),
            )
            alternatives = tuple(
                AssignmentAlternative(candidate.feature, min(1.0, max(0.0, candidate.score)))
                for candidate in candidates[:5]
            )
            if best is None:
                assignments.append(MarkingAssignment(
                    item.evidence_id, AssignmentStatus.REJECTED, None, (), diagnostics,
                    ("no_templates",),
                ))
                continue
            reason_codes: list[str] = []
            acceptable = best.feature in ACCEPTED_FEATURES
            if best.feature not in ACCEPTED_FEATURES:
                reason_codes.append("out_of_scope_decoy")
            if best.corridor_fraction < self.config.min_corridor_fraction:
                acceptable = False
                reason_codes.append("corridor_support")
            limit = self.config.arc_tangent_deg if best.feature.kind == "arc" else self.config.straight_tangent_deg
            if best.tangent_error_deg > limit:
                acceptable = False
                reason_codes.append("tangent_error")
            if not self.config.min_width_ratio <= best.width_ratio <= self.config.max_width_ratio:
                acceptable = False
                reason_codes.append("stripe_width")
            if best.distance_px > self.config.min_inlier_distance_px * self.pixel_scale:
                acceptable = False
                reason_codes.append("template_distance")
            if margin < self.config.min_uniqueness_margin:
                assignments.append(MarkingAssignment(
                    item.evidence_id, AssignmentStatus.AMBIGUOUS, None, alternatives, diagnostics,
                    tuple(sorted(set(reason_codes + ["low_margin"]))),
                ))
                continue
            if acceptable:
                assignments.append(MarkingAssignment(
                    item.evidence_id, AssignmentStatus.ACCEPTED, best.feature, alternatives,
                    diagnostics, tuple(sorted(set(reason_codes + ["unique"]))),
                ))
                selected[best.feature].append((item.evidence_id, best))
            else:
                assignments.append(MarkingAssignment(
                    item.evidence_id, AssignmentStatus.REJECTED, None, alternatives, diagnostics,
                    tuple(sorted(set(reason_codes))) or ("association_gates",),
                ))

        return AssociationResult(
            assignments=tuple(assignments),
            feature_support=self._feature_support(assignments, selected),
            family_support=self._family_support(assignments, selected),
        )

    def _build_templates(self) -> dict[MarkingFeature, tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
        templates = {}
        for feature in MarkingFeature:
            primitive = self.layout.marking(feature)
            count = 181 if feature.kind == "arc" else 81
            court = np.asarray(primitive.sample(count), dtype=np.float64)
            image = project(self.h_court_to_image, court)
            if image is None or not np.all(np.isfinite(image)):
                image = np.full_like(court, np.nan)
            tangents = _polyline_tangents(image)
            widths = np.array([_projected_width(self.h_court_to_image, point, primitive.width_ft) for point in court])
            visible_length = _visible_length(image)
            templates[feature] = image, tangents, widths, visible_length
        return templates

    def _score(self, evidence: UnlabeledMarkingEvidence, feature: MarkingFeature) -> _Candidate:
        image, tangents, widths, visible_length = self._templates[feature]
        query = np.array([[sample.x, sample.y] for sample in evidence.samples], dtype=np.float64)
        model_tangent = np.array([sample.tangent_xy or (1.0, 0.0) for sample in evidence.samples], dtype=np.float64)
        distances, indices = _nearest(query, image)
        valid = np.isfinite(distances)
        if not np.any(valid):
            return _Candidate(feature, 0.0, 1e9, 90.0, 0.0, 0.0, 0.0, 0.0, visible_length, 0.0, 0.0)
        matched_tangent = tangents[indices]
        tangent_error = np.degrees(np.arccos(np.clip(np.abs(np.sum(model_tangent * matched_tangent, axis=1)), 0.0, 1.0)))
        observed_width = np.array([sample.observed_width_px or np.nan for sample in evidence.samples])
        projected_width = widths[indices]
        ratios = observed_width / np.maximum(projected_width, 1e-6)
        ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
        width_ratio = float(np.median(ratios)) if len(ratios) else 1.0
        corridor_width = _corridor_width(self.layout, self.h_image_to_court, query, self.image_shape)
        inliers = distances <= np.maximum(corridor_width, self.config.min_inlier_distance_px * self.pixel_scale)
        evidence_coverage = float(np.mean(inliers))
        template_coverage = _template_coverage(query, image, self.config.min_inlier_distance_px * self.pixel_scale)
        support = _polyline_length(query[inliers]) if np.sum(inliers) > 1 else 0.0
        limit = self.config.arc_tangent_deg if feature.kind == "arc" else self.config.straight_tangent_deg
        distance_quality = math.exp(-float(np.median(distances)) / max(1.0, 8.0 * self.pixel_scale))
        tangent_quality = max(0.0, 1.0 - float(np.median(tangent_error)) / limit)
        width_quality = max(0.0, 1.0 - abs(math.log(max(width_ratio, 1e-6))) / math.log(self.config.max_width_ratio / self.config.min_width_ratio))
        extent_quality = min(1.0, support / max(12.0 * self.pixel_scale, 1.0))
        purity = evidence_coverage
        score = 0.24 * distance_quality + 0.18 * tangent_quality + 0.16 * width_quality + 0.16 * evidence_coverage + 0.12 * template_coverage + 0.08 * extent_quality + 0.06 * purity
        return _Candidate(
            feature, float(min(1.0, max(0.0, score))), float(np.median(distances)),
            float(np.median(tangent_error)), evidence_coverage, template_coverage,
            width_ratio, support, visible_length, evidence_coverage, _turning(query),
        )

    def _feature_support(self, assignments, selected):
        result = []
        for feature in MarkingFeature:
            accepted = selected[feature]
            rejected = sum(1 for item in assignments if item.status is AssignmentStatus.REJECTED and _mentions(item, feature))
            ambiguous = sum(1 for item in assignments if item.status is AssignmentStatus.AMBIGUOUS and _mentions(item, feature))
            if accepted:
                evidence_coverage = float(np.mean([candidate.evidence_coverage for _, candidate in accepted]))
                template_coverage = float(np.mean([candidate.template_coverage for _, candidate in accepted]))
            else:
                evidence_coverage = template_coverage = 0.0
            result.append(ShadowFeatureSupport(
                feature=feature, accepted_fragment_count=len(accepted),
                rejected_fragment_count=rejected, ambiguous_fragment_count=ambiguous,
                evidence_to_template_coverage=evidence_coverage,
                template_to_evidence_coverage=template_coverage,
                selected_evidence_ids=tuple(item[0] for item in accepted),
                reasons=() if accepted else ("no_accepted_support",),
            ))
        return tuple(result)

    def _family_support(self, assignments, selected):
        result = []
        scale = self.pixel_scale
        for family in MarkingFamily:
            members = [feature for feature in MarkingFeature if feature.family is family]
            accepted = [(evidence_id, candidate) for feature in members for evidence_id, candidate in selected[feature]]
            support = sum(candidate.support_px for _, candidate in accepted)
            visible = sum(self._templates[feature][3] for feature in members)
            ev_cov = float(np.mean([c.evidence_coverage for _, c in accepted])) if accepted else 0.0
            reverse = float(np.mean([c.template_coverage for _, c in accepted])) if accepted else 0.0
            reasons: list[str] = []
            ready = False
            if family is MarkingFamily.LANE:
                have_line = bool(selected[MarkingFeature.FREE_THROW_LINE])
                have_edge = bool(selected[MarkingFeature.LANE_EDGE_FAR] or selected[MarkingFeature.LANE_EDGE_NEAR])
                ready = have_line and have_edge and self._lane_join_ok(selected)
                if not have_line: reasons.append("free_throw_line_missing")
                if not have_edge: reasons.append("lane_edge_missing")
                if not self._lane_join_ok(selected): reasons.append("lane_join")
            elif family is MarkingFamily.THREE_POINT:
                arc = selected[MarkingFeature.THREE_POINT_ARC]
                corner = selected[MarkingFeature.THREE_POINT_CORNER_FAR] or selected[MarkingFeature.THREE_POINT_CORNER_NEAR]
                arc_support = sum(c.support_px for _, c in arc)
                arc_visible = self._templates[MarkingFeature.THREE_POINT_ARC][3]
                ready = bool(arc) and arc_support >= max(self.config.arc_support_px * scale, 0.15 * arc_visible) and bool(corner) and self._three_point_join_ok(selected)
                if not arc: reasons.append("three_point_arc_missing")
                if not corner: reasons.append("corner_straight_missing")
                if arc_support < max(self.config.arc_support_px * scale, 0.15 * arc_visible): reasons.append("arc_support")
                if not self._three_point_join_ok(selected): reasons.append("corner_join")
            elif family is MarkingFamily.FREE_THROW_CIRCLE:
                half = selected[MarkingFeature.FREE_THROW_CIRCLE_FAR_HALF]
                circle_support = sum(c.support_px for _, c in half)
                visible_half = self._templates[MarkingFeature.FREE_THROW_CIRCLE_FAR_HALF][3]
                sweep = max((c.turning for _, c in half), default=0.0)
                ready = bool(half) and circle_support >= max(self.config.circle_support_px * scale, 0.15 * visible_half) and sweep >= 0.25
                if not half: reasons.append("held_out_half_missing")
                if sweep < 0.25: reasons.append("insufficient_tangent_sweep")
            else:
                reasons.append("family_out_of_scope")
            result.append(ShadowFamilySupport(
                family=family, ready=ready, accepted_fragment_count=len(accepted),
                support_px=support, visible_template_px=visible,
                evidence_to_template_coverage=ev_cov,
                template_to_evidence_coverage=reverse,
                reasons=tuple(reasons) if not ready else (),
            ))
        return tuple(result)

    def _lane_join_ok(self, selected) -> bool:
        line = selected[MarkingFeature.FREE_THROW_LINE]
        edges = selected[MarkingFeature.LANE_EDGE_FAR] + selected[MarkingFeature.LANE_EDGE_NEAR]
        if not line or not edges:
            return False
        line_image = self._templates[MarkingFeature.FREE_THROW_LINE][0]
        edge_images = [self._templates[item.feature][0] for item in (self.layout.marking(MarkingFeature.LANE_EDGE_FAR), self.layout.marking(MarkingFeature.LANE_EDGE_NEAR))]
        join = line_image[0]
        return any(np.min(np.linalg.norm(edge - join, axis=1)) <= self.config.lane_join_px * self.pixel_scale for edge in edge_images)

    def _three_point_join_ok(self, selected) -> bool:
        arc = self._templates[MarkingFeature.THREE_POINT_ARC][0]
        for feature in (MarkingFeature.THREE_POINT_CORNER_FAR, MarkingFeature.THREE_POINT_CORNER_NEAR):
            if selected[feature]:
                line = self._templates[feature][0]
                if np.min(np.linalg.norm(line[-1] - arc[[0, -1]], axis=1)) <= self.config.lane_join_px * self.pixel_scale:
                    return True
        return False


def _polyline_tangents(points: np.ndarray) -> np.ndarray:
    tangent = np.gradient(points, axis=0)
    length = np.linalg.norm(tangent, axis=1, keepdims=True)
    return tangent / np.maximum(length, 1e-9)


def _nearest(query: np.ndarray, template: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.all(np.isfinite(template), axis=1)
    distances = np.full(len(query), np.inf)
    indices = np.zeros(len(query), dtype=np.int64)
    if np.any(valid):
        source = template[valid]
        delta = query[:, None, :] - source[None, :, :]
        raw = np.linalg.norm(delta, axis=2)
        local = np.argmin(raw, axis=1)
        distances = raw[np.arange(len(query)), local]
        indices = np.flatnonzero(valid)[local]
    return distances, indices


def _projected_width(matrix: np.ndarray, point: np.ndarray, width_ft: float) -> float:
    epsilon = max(width_ft * 0.5, 1e-4)
    a = project(matrix, np.asarray([point - [0, epsilon], point + [0, epsilon]]))
    if a is None or not np.all(np.isfinite(a)):
        return 1.0
    return max(0.5, float(np.linalg.norm(a[1] - a[0])))


def _visible_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    keep = np.all(np.isfinite(points), axis=1)
    points = points[keep]
    return _polyline_length(points)


def _polyline_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) > 1 else 0.0


def _template_coverage(query: np.ndarray, template: np.ndarray, tolerance: float) -> float:
    if len(template) == 0:
        return 0.0
    distances, _ = _nearest(template, query)
    return float(np.mean(distances <= tolerance))


def _turning(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    vectors = np.diff(points, axis=0)
    angles = np.unwrap(np.arctan2(vectors[:, 1], vectors[:, 0]))
    return float(np.ptp(angles))


def _corridor_width(layout, inverse, query, image_shape):
    # Convert distance outside the convex hull of authoritative landmarks to
    # feet, then back to a diagonal-scaled pixel corridor.  This lets near and
    # far court templates retain a narrow corridor while tolerating projected
    # uncertainty outside the seed hull.
    landmarks = np.asarray(list(layout.landmarks.values()), dtype=np.float64)
    court = project(inverse, query)
    if court is None:
        return np.full(len(query), 12.0)
    hull = _convex_hull(landmarks)
    outside = np.array([_outside_distance(point, hull) for point in court])
    return np.asarray([corridor_width_px(float(distance), image_shape) for distance in outside])


def _convex_hull(points: np.ndarray) -> np.ndarray:
    points = sorted(map(tuple, points))
    if len(points) <= 1:
        return np.asarray(points, dtype=np.float64)
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0: lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0: upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _outside_distance(point: np.ndarray, hull: np.ndarray) -> float:
    if len(hull) < 3:
        return 0.0
    inside = False
    for a, b in zip(hull, np.vstack((hull[1:], hull[:1]))):
        if ((a[1] > point[1]) != (b[1] > point[1])) and point[0] < (b[0] - a[0]) * (point[1] - a[1]) / (b[1] - a[1] + 1e-12) + a[0]:
            inside = not inside
    if inside:
        return 0.0
    distances = []
    for a, b in zip(hull, np.vstack((hull[1:], hull[:1]))):
        vector = b - a
        fraction = np.clip(float(np.dot(point - a, vector)) / max(float(np.dot(vector, vector)), 1e-12), 0.0, 1.0)
        distances.append(float(np.linalg.norm(point - (a + fraction * vector))))
    return min(distances)


def _mentions(assignment, feature):
    return any(item.feature is feature for item in assignment.alternatives)
