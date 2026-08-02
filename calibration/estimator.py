"""Robust homography estimation.

Solved **court → image**, not the other way round. Measurement noise lives in
image pixels, so a RANSAC threshold expressed there means something physical: "a
landmark may be eight pixels off". The same threshold in court feet would be
wildly non-uniform across the frame, tight near the camera and meaningless at the
far baseline.

MAGSAC++ is preferred because it does not need a hard inlier/outlier cut, which
matters when the evidence feeding it comes from a detector whose slot semantics
are provisional and whose error distribution is therefore not well characterised.
Plain RANSAC is the fallback where the USAC solver is unavailable.

Both transforms are computed once and checked against each other. Callers get
``image_to_court`` and ``court_to_image`` and never invert a matrix themselves.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from contracts.calibration_types import Correspondence, Matrix3x3

#: Absolute minimum for a homography. Four points is geometry; it is not evidence.
ABSOLUTE_MIN_POINTS = 4


@dataclass(frozen=True, slots=True)
class HomographyFit:
    """A solved transform pair, or a reason there is none."""

    h_court_to_image: Matrix3x3 | None
    h_image_to_court: Matrix3x3 | None
    inlier_landmarks: tuple[str, ...]
    outlier_landmarks: tuple[str, ...]
    round_trip_error_px: float | None
    solver: str
    failure: str | None = None

    @property
    def solved(self) -> bool:
        return self.h_court_to_image is not None


def _as_matrix(array: np.ndarray) -> Matrix3x3:
    normalised = array / array[2, 2] if abs(array[2, 2]) > 1e-12 else array
    return tuple(tuple(float(v) for v in row) for row in normalised)  # type: ignore[return-value]


def estimate_homography(
    correspondences: tuple[Correspondence, ...],
    ransac_threshold_px: float = 8.0,
    confidence: float = 0.999,
    max_iters: int = 10000,
    round_trip_tolerance_px: float = 1.0,
) -> HomographyFit:
    """Fit court → image robustly, then invert and verify."""
    if len(correspondences) < ABSOLUTE_MIN_POINTS:
        return HomographyFit(
            None, None, (), (), None, "none",
            failure=f"insufficient_points:{len(correspondences)}",
        )

    court_points = np.array([c.court_xy for c in correspondences], dtype=np.float64)
    image_points = np.array([c.image_xy for c in correspondences], dtype=np.float64)

    solver = "USAC_MAGSAC"
    matrix, mask = None, None
    try:
        matrix, mask = cv2.findHomography(
            court_points, image_points, cv2.USAC_MAGSAC,
            ransac_threshold_px, maxIters=max_iters, confidence=confidence,
        )
    except (cv2.error, AttributeError):
        matrix = None

    if matrix is None:
        solver = "RANSAC"
        try:
            matrix, mask = cv2.findHomography(
                court_points, image_points, cv2.RANSAC,
                ransac_threshold_px, maxIters=max_iters, confidence=confidence,
            )
        except cv2.error:
            matrix = None

    if matrix is None or not np.all(np.isfinite(matrix)):
        return HomographyFit(None, None, (), (), None, solver, failure="solver_failed")

    if mask is None:
        inlier_flags = np.ones(len(correspondences), dtype=bool)
    else:
        inlier_flags = mask.ravel().astype(bool)

    inliers = tuple(c.landmark_id for c, keep in zip(correspondences, inlier_flags) if keep)
    outliers = tuple(c.landmark_id for c, keep in zip(correspondences, inlier_flags) if not keep)

    if len(inliers) < ABSOLUTE_MIN_POINTS:
        return HomographyFit(
            None, None, inliers, outliers, None, solver,
            failure=f"insufficient_inliers:{len(inliers)}",
        )

    # Always refit on the full inlier set. A RANSAC/MAGSAC result is a model from
    # a *minimal* sample, so it fits four points exactly and dumps the residual
    # onto everything else — which makes the median reprojection error read as
    # near-zero no matter how noisy the evidence was. Least squares over all
    # inliers both improves the estimate and makes the quality metrics honest.
    refined, _ = cv2.findHomography(court_points[inlier_flags], image_points[inlier_flags], 0)
    if refined is not None and np.all(np.isfinite(refined)):
        matrix = refined

    determinant = float(np.linalg.det(matrix))
    if abs(determinant) < 1e-12:
        return HomographyFit(
            None, None, inliers, outliers, None, solver, failure="singular_matrix"
        )

    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return HomographyFit(
            None, None, inliers, outliers, None, solver, failure="non_invertible"
        )

    round_trip = _round_trip_error(matrix, inverse, court_points[inlier_flags])
    if round_trip is None or round_trip > round_trip_tolerance_px:
        return HomographyFit(
            None, None, inliers, outliers, round_trip, solver,
            failure=f"round_trip_error:{round_trip}",
        )

    return HomographyFit(
        h_court_to_image=_as_matrix(matrix),
        h_image_to_court=_as_matrix(inverse),
        inlier_landmarks=inliers,
        outlier_landmarks=outliers,
        round_trip_error_px=round_trip,
        solver=solver,
    )


def _round_trip_error(matrix: np.ndarray, inverse: np.ndarray, court_points: np.ndarray) -> float | None:
    """Max court→image→court→image drift, in pixels.

    A transform pair that does not compose to the identity is unusable no matter
    how good its reprojection error looks.
    """
    if len(court_points) == 0:
        return None
    projected = project(matrix, court_points)
    if projected is None:
        return None
    recovered = project(inverse, projected)
    if recovered is None:
        return None
    reprojected = project(matrix, recovered)
    if reprojected is None:
        return None
    return float(np.max(np.linalg.norm(reprojected - projected, axis=1)))


def project(matrix: np.ndarray, points: np.ndarray) -> np.ndarray | None:
    """Apply a 3x3 homography to an ``(N, 2)`` array."""
    if len(points) == 0:
        return points
    homogeneous = np.hstack([points, np.ones((len(points), 1))])
    transformed = homogeneous @ np.asarray(matrix).T
    w = transformed[:, 2]
    if np.any(np.abs(w) < 1e-12):
        return None
    return transformed[:, :2] / w[:, None]


def matrix_to_array(matrix: Matrix3x3) -> np.ndarray:
    return np.array(matrix, dtype=np.float64)
