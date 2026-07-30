"""Quality measurement and status assignment.

A homography that fits its own input points perfectly can still be nonsense. Four
points strung along the baseline admit a family of solutions that all reproject
beautifully and place midcourt in the crowd. So a fit is not accepted because it
converged — it is accepted because it survives a set of independent checks:

*Reprojection* — does it explain the points it was fitted to?
*Held-out* — does it explain a point it was **not** fitted to? This is the check
that catches a systematically wrong slot mapping, because a wrong map can still
be self-consistent.
*Degeneracy* — are the points spread over real area, or nearly collinear?
*Plausibility* — does the implied scale and court extent make physical sense?

Thresholds live in ``CalibrationPolicy``, not scattered through the code, so
tuning is a data decision rather than an edit.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from contracts.calibration_types import (
    CalibrationQuality,
    CalibrationStatus,
    Correspondence,
    Matrix3x3,
)
from contracts.court_layout import HalfCourtLayout

from .estimator import project


@dataclass(frozen=True, slots=True)
class CalibrationPolicy:
    """Every threshold that separates OK from DEGRADED from rejected.

    Defaults are deliberately conservative. With provisional slot semantics the
    cost of a confident-but-wrong calibration is far higher than the cost of an
    honest ``UNCALIBRATED`` frame.
    """

    preferred_min_points: int = 5
    absolute_min_points: int = 4
    min_pooled_confidence: float = 0.25
    min_mean_confidence: float = 0.35
    ransac_threshold_px: float = 8.0
    max_reprojection_px_median: float = 6.0
    max_reprojection_px_p95: float = 14.0
    max_holdout_px_median: float = 25.0
    min_triangle_area_px: float = 900.0
    #: Points that must lie *off* the best-supported line through the court-side
    #: evidence. Three collinear baseline points plus two others fit beautifully
    #: and still leave the court's length under-constrained.
    min_offline_court_points: int = 3
    collinear_tolerance_ft: float = 0.5
    min_inlier_ratio: float = 0.6
    min_court_span_x_ft: float = 8.0
    min_court_span_y_ft: float = 8.0
    scale_ft_per_px_range: tuple[float, float] = (0.005, 0.5)
    round_trip_tolerance_px: float = 1.0
    degraded_below_points: int = 5

    def as_dict(self) -> dict:
        return {
            "preferred_min_points": self.preferred_min_points,
            "absolute_min_points": self.absolute_min_points,
            "min_pooled_confidence": self.min_pooled_confidence,
            "min_mean_confidence": self.min_mean_confidence,
            "ransac_threshold_px": self.ransac_threshold_px,
            "max_reprojection_px_median": self.max_reprojection_px_median,
            "max_reprojection_px_p95": self.max_reprojection_px_p95,
            "max_holdout_px_median": self.max_holdout_px_median,
            "min_triangle_area_px": self.min_triangle_area_px,
            "min_inlier_ratio": self.min_inlier_ratio,
            "min_court_span_ft": [self.min_court_span_x_ft, self.min_court_span_y_ft],
            "scale_ft_per_px_range": list(self.scale_ft_per_px_range),
            "round_trip_tolerance_px": self.round_trip_tolerance_px,
        }


def max_triangle_area_px(points: np.ndarray) -> float:
    """Largest triangle spanned by any three points.

    Near-collinear evidence produces a tiny value here even when the points are
    far apart, which is exactly the configuration that yields a well-fitting,
    physically absurd homography.
    """
    if len(points) < 3:
        return 0.0
    best = 0.0
    for a, b, c in combinations(range(len(points)), 3):
        area = 0.5 * abs(
            (points[b][0] - points[a][0]) * (points[c][1] - points[a][1])
            - (points[c][0] - points[a][0]) * (points[b][1] - points[a][1])
        )
        best = max(best, float(area))
    return best


def offline_court_points(court_points: np.ndarray, tolerance_ft: float = 0.5) -> int:
    """How many points lie off the best-supported line through the evidence.

    Reprojection error cannot see this. Five points of which three are collinear
    fit a homography exactly and leave the direction across that line supported
    by only two observations — so the court's length is effectively extrapolated,
    and the far half of the projection drifts while every residual stays tiny.
    """
    if len(court_points) < 3:
        return 0
    best_on_line = 2
    for i, j in combinations(range(len(court_points)), 2):
        a, b = court_points[i], court_points[j]
        direction = b - a
        length = float(np.linalg.norm(direction))
        if length < 1e-9:
            continue
        normal = np.array([-direction[1], direction[0]]) / length
        distances = np.abs((court_points - a) @ normal)
        best_on_line = max(best_on_line, int(np.sum(distances <= tolerance_ft)))
    return len(court_points) - best_on_line


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(round(fraction * len(ordered) + 0.5))))
    return ordered[rank - 1]


def holdout_error_px(
    correspondences: tuple[Correspondence, ...],
    policy: CalibrationPolicy,
) -> float | None:
    """Median leave-one-out reprojection error.

    Refit without each point in turn and measure how far the fit misses it. A
    self-consistent but wrong slot mapping fits its own points and fails here,
    which is the whole reason this check exists.
    """
    if len(correspondences) <= policy.absolute_min_points:
        return None

    from .estimator import estimate_homography  # local import avoids a cycle

    errors: list[float] = []
    for index in range(len(correspondences)):
        subset = correspondences[:index] + correspondences[index + 1 :]
        fit = estimate_homography(subset, policy.ransac_threshold_px)
        if not fit.solved or fit.h_court_to_image is None:
            continue
        held = correspondences[index]
        predicted = project(
            np.array(fit.h_court_to_image), np.array([held.court_xy], dtype=np.float64)
        )
        if predicted is None:
            continue
        errors.append(float(np.linalg.norm(predicted[0] - np.array(held.image_xy))))

    return statistics.median(errors) if errors else None


def evaluate(
    correspondences: tuple[Correspondence, ...],
    inlier_landmarks: tuple[str, ...],
    h_court_to_image: Matrix3x3,
    h_image_to_court: Matrix3x3,
    layout: HalfCourtLayout,
    image_size: tuple[int, int],
    policy: CalibrationPolicy,
    round_trip_error_px: float | None,
    mean_confidence: float | None,
) -> tuple[CalibrationStatus, CalibrationQuality]:
    """Measure a fit and decide its status."""
    reasons: list[str] = []
    inlier_set = set(inlier_landmarks)
    inliers = tuple(c for c in correspondences if c.landmark_id in inlier_set)

    court_points = np.array([c.court_xy for c in inliers], dtype=np.float64)
    image_points = np.array([c.image_xy for c in inliers], dtype=np.float64)
    matrix = np.array(h_court_to_image, dtype=np.float64)

    projected = project(matrix, court_points)
    if projected is None:
        return CalibrationStatus.FAILED_DEGENERATE_GEOMETRY, CalibrationQuality(
            observation_count=len(correspondences),
            inlier_count=len(inliers),
            inlier_ratio=len(inliers) / max(len(correspondences), 1),
            reasons=("projection_to_infinity",),
        )

    residuals = [float(v) for v in np.linalg.norm(projected - image_points, axis=1)]
    scale_ft_per_px = _scale_ft_per_px(h_image_to_court, image_size)
    residuals_ft = [r * scale_ft_per_px for r in residuals] if scale_ft_per_px else []

    triangle_area = max_triangle_area_px(image_points)
    span_x = float(court_points[:, 0].max() - court_points[:, 0].min()) if len(court_points) else 0.0
    span_y = float(court_points[:, 1].max() - court_points[:, 1].min()) if len(court_points) else 0.0
    holdout = holdout_error_px(inliers, policy)
    hull_fraction = _hull_fraction(image_points, image_size)

    quality = CalibrationQuality(
        observation_count=len(correspondences),
        inlier_count=len(inliers),
        inlier_ratio=len(inliers) / max(len(correspondences), 1),
        reprojection_px_mean=statistics.fmean(residuals) if residuals else None,
        reprojection_px_median=statistics.median(residuals) if residuals else None,
        reprojection_px_p95=_percentile(residuals, 0.95) if residuals else None,
        reprojection_px_max=max(residuals) if residuals else None,
        reprojection_ft_median=statistics.median(residuals_ft) if residuals_ft else None,
        reprojection_ft_max=max(residuals_ft) if residuals_ft else None,
        holdout_px_median=holdout,
        min_triangle_area_px=triangle_area,
        image_hull_fraction=hull_fraction,
        court_span_x_ft=span_x,
        court_span_y_ft=span_y,
        scale_ft_per_px=scale_ft_per_px,
        round_trip_error_px=round_trip_error_px,
        mean_evidence_confidence=mean_confidence,
        reasons=(),
    )

    # --- hard rejections -------------------------------------------------
    if len(inliers) < policy.absolute_min_points:
        reasons.append(f"inliers:{len(inliers)}")
        return CalibrationStatus.FAILED_INSUFFICIENT_POINTS, _with(quality, reasons)

    if triangle_area < policy.min_triangle_area_px:
        reasons.append(f"near_collinear:triangle_area={triangle_area:.0f}px2")
        return CalibrationStatus.FAILED_DEGENERATE_GEOMETRY, _with(quality, reasons)

    if span_x < policy.min_court_span_x_ft or span_y < policy.min_court_span_y_ft:
        reasons.append(f"court_span_too_small:{span_x:.1f}x{span_y:.1f}ft")
        return CalibrationStatus.FAILED_DEGENERATE_GEOMETRY, _with(quality, reasons)

    if quality.reprojection_px_median is not None and (
        quality.reprojection_px_median > policy.max_reprojection_px_median
        or (quality.reprojection_px_p95 or 0.0) > policy.max_reprojection_px_p95
    ):
        reasons.append(
            f"reprojection:median={quality.reprojection_px_median:.1f}px "
            f"p95={quality.reprojection_px_p95:.1f}px"
        )
        return CalibrationStatus.FAILED_HIGH_REPROJECTION_ERROR, _with(quality, reasons)

    if holdout is not None and holdout > policy.max_holdout_px_median:
        reasons.append(f"holdout:{holdout:.1f}px")
        return CalibrationStatus.FAILED_HIGH_REPROJECTION_ERROR, _with(quality, reasons)

    implausible = _plausibility_failures(
        h_court_to_image, h_image_to_court, layout, image_size, scale_ft_per_px, policy
    )
    if implausible:
        return CalibrationStatus.FAILED_IMPLAUSIBLE, _with(quality, implausible)

    # --- degradations ----------------------------------------------------
    offline = offline_court_points(court_points, policy.collinear_tolerance_ft)
    if offline < policy.min_offline_court_points:
        # Not a rejection: the fit is usable near the evidence and only degrades
        # as it extrapolates away from it. Callers that need the far half of the
        # court should treat this reason as disqualifying.
        reasons.append(f"weak_conditioning:only_{offline}_points_off_dominant_line")

    if len(inliers) < policy.degraded_below_points:
        reasons.append(f"few_points:{len(inliers)}")
    if len(inliers) == policy.absolute_min_points:
        # Four points determine a homography exactly, so its reprojection error is
        # zero by construction and says nothing about correctness. Flag it rather
        # than let a perfect-looking number be mistaken for evidence.
        reasons.append("reprojection_uninformative:exactly_minimal_sample")
    if quality.inlier_ratio < policy.min_inlier_ratio:
        reasons.append(f"low_inlier_ratio:{quality.inlier_ratio:.2f}")
    if mean_confidence is not None and mean_confidence < policy.min_mean_confidence:
        reasons.append(f"low_mean_confidence:{mean_confidence:.2f}")
    if holdout is None:
        reasons.append("holdout_unavailable")

    status = CalibrationStatus.DEGRADED if reasons else CalibrationStatus.OK
    return status, _with(quality, reasons)


def _with(quality: CalibrationQuality, reasons: list[str]) -> CalibrationQuality:
    from dataclasses import replace

    return replace(quality, reasons=tuple(reasons))


def _scale_ft_per_px(h_image_to_court: Matrix3x3, image_size: tuple[int, int]) -> float | None:
    """Court feet spanned by one pixel at the image centre."""
    width, height = image_size
    centre = np.array([[width / 2.0, height / 2.0]], dtype=np.float64)
    stepped = np.array([[width / 2.0 + 1.0, height / 2.0]], dtype=np.float64)
    matrix = np.array(h_image_to_court, dtype=np.float64)
    a = project(matrix, centre)
    b = project(matrix, stepped)
    if a is None or b is None:
        return None
    return float(np.linalg.norm(b[0] - a[0]))


def _hull_fraction(image_points: np.ndarray, image_size: tuple[int, int]) -> float | None:
    """Fraction of the frame covered by the evidence's bounding box.

    Cheap proxy for spatial coverage: evidence bunched into one corner
    constrains the far side of the frame not at all.
    """
    if len(image_points) < 3:
        return None
    width, height = image_size
    box = (image_points[:, 0].max() - image_points[:, 0].min()) * (
        image_points[:, 1].max() - image_points[:, 1].min()
    )
    return float(box / (width * height)) if width and height else None


def _plausibility_failures(
    h_court_to_image: Matrix3x3,
    h_image_to_court: Matrix3x3,
    layout: HalfCourtLayout,
    image_size: tuple[int, int],
    scale_ft_per_px: float | None,
    policy: CalibrationPolicy,
) -> list[str]:
    """Physical sanity checks that reprojection error cannot catch."""
    failures: list[str] = []

    if scale_ft_per_px is None:
        failures.append("scale_undefined")
    else:
        low, high = policy.scale_ft_per_px_range
        if not low <= scale_ft_per_px <= high:
            failures.append(f"implausible_scale:{scale_ft_per_px:.4f}ft/px")

    # The half-court's four corners must stay convex and correctly wound in the
    # image. A 'solution' that folds the court inside out reprojects fine.
    corners = np.array(
        [
            [0.0, 0.0],
            [layout.half_length, 0.0],
            [layout.half_length, layout.width],
            [0.0, layout.width],
        ],
        dtype=np.float64,
    )
    projected = project(np.array(h_court_to_image, dtype=np.float64), corners)
    if projected is None:
        failures.append("court_corners_project_to_infinity")
        return failures

    if not _is_convex(projected):
        failures.append("court_quad_not_convex")

    return failures


def _is_convex(quad: np.ndarray) -> bool:
    signs = []
    for i in range(4):
        a, b, c = quad[i], quad[(i + 1) % 4], quad[(i + 2) % 4]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        signs.append(cross > 0)
    return all(signs) or not any(signs)
