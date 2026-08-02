"""Bounded projected Levenberg--Marquardt for the hybrid candidate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from contracts.calibration_types import Correspondence
from contracts.court_layout import HalfCourtLayout

from ..estimator import project
from .control_points import (
    clip_to_box,
    control_court_points,
    control_points_from_homography,
    homography_from_control_points,
    is_convex,
    local_pixel_scale,
    probe_court_points,
)
from .residuals import (
    MarkingObservation,
    freeze_observations,
    landmark_residuals,
    marking_residuals,
)


@dataclass(frozen=True, slots=True)
class RefineResult:
    theta: np.ndarray
    h_court_to_image: np.ndarray | None
    h_image_to_court: np.ndarray | None
    round_trip_error_px: float | None
    cost: float
    outer_iterations: int
    inner_iterations: int
    converged: bool
    bounds_active: int
    convexity_backtracks: int
    failure: str | None


def _huber(value: np.ndarray, delta: float) -> np.ndarray:
    absolute = np.abs(value)
    return np.where(absolute <= delta, 0.5 * value * value, delta * absolute - 0.5 * delta * delta)


def _huber_weights(value: np.ndarray, delta: float) -> np.ndarray:
    absolute = np.abs(value)
    return np.where(absolute <= delta, 1.0, delta / np.maximum(absolute, 1e-12))


def _cauchy(value: np.ndarray, delta: float) -> np.ndarray:
    return 0.5 * delta * delta * np.log1p((value / delta) ** 2)


def _cauchy_weights(value: np.ndarray, delta: float) -> np.ndarray:
    return 1.0 / (1.0 + (value / delta) ** 2)


def central_difference_jacobian(function, theta, step: float) -> np.ndarray:
    """Central-difference Jacobian with the specified absolute scale rule."""
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    base = np.asarray(function(theta), dtype=np.float64).reshape(-1)
    result = np.empty((len(base), len(theta)), dtype=np.float64)
    for index in range(len(theta)):
        h = float(step) * max(1.0, abs(float(theta[index])))
        plus = theta.copy(); plus[index] += h
        minus = theta.copy(); minus[index] -= h
        result[:, index] = (np.asarray(function(plus)) - np.asarray(function(minus))) / (2.0 * h)
    return result


def _round_trip(matrix: np.ndarray, inverse: np.ndarray, court: np.ndarray) -> float | None:
    projected = project(matrix, court)
    if projected is None:
        return None
    recovered = project(inverse, projected)
    if recovered is None:
        return None
    returned = project(matrix, recovered)
    if returned is None:
        return None
    return float(np.max(np.linalg.norm(returned - projected, axis=1))) if len(projected) else 0.0


def _matrix(theta: np.ndarray, layout: HalfCourtLayout) -> np.ndarray:
    matrix = homography_from_control_points(control_court_points(layout), np.asarray(theta).reshape(4, 2))
    if matrix is None:
        raise ValueError("control points do not define a homography")
    return np.asarray(matrix, dtype=np.float64)


def _terms(theta, correspondences, observations, layout, config, landmark_weights=None, marking_irls=None):
    landmark_raw = landmark_residuals(theta, correspondences, layout)
    landmark_sigma = float(getattr(config, "landmark_sigma_px", 2.0))
    landmark_norm = landmark_raw / max(landmark_sigma, 1e-12)
    landmark_norm_pairs = landmark_norm.reshape(-1, 2) if len(landmark_norm) else np.zeros((0, 2))
    pair_norm = np.linalg.norm(landmark_norm_pairs, axis=1)
    if landmark_weights is None:
        landmark_weights = _huber_weights(pair_norm, float(getattr(config, "landmark_huber_delta", 2.0)))

    marking_raw = marking_residuals(theta, observations, layout)
    sigma = np.concatenate([item.sigma_px for item in observations]) if observations else np.zeros(0)
    normalized_marking = marking_raw / np.maximum(sigma, 1e-12)
    marking_delta = float(getattr(config, "marking_cauchy_delta", 1.5))
    marking_weights = np.concatenate([item.weight for item in observations]) if observations else np.zeros(0)
    if marking_irls is None:
        marking_irls = _cauchy_weights(normalized_marking, marking_delta)

    prior_weight = float(getattr(config, "prior_weight", 0.25))
    matrix = _matrix(np.asarray(theta), layout)
    # The caller replaces theta0 below; this field is attached by the closure.
    return matrix, landmark_raw, landmark_norm, landmark_norm_pairs, pair_norm, landmark_weights, normalized_marking, marking_weights, marking_irls


def _objective(theta, correspondences, observations, layout, config, theta0, landmark_weights, marking_irls=None):
    matrix, landmark_raw, landmark_norm, _, pair_norm, landmark_weights, normalized_marking, marking_weights, marking_irls = _terms(
        theta, correspondences, observations, layout, config, landmark_weights, marking_irls
    )
    cost = float(np.sum(_huber(pair_norm, float(getattr(config, "landmark_huber_delta", 2.0)))))
    delta = float(getattr(config, "marking_cauchy_delta", 1.5))
    cost += float(np.sum(marking_weights * _cauchy(normalized_marking, delta)))
    scales = local_pixel_scale(_matrix(np.asarray(theta0), layout), control_court_points(layout))
    offset = (np.asarray(theta).reshape(4, 2) - np.asarray(theta0).reshape(4, 2))
    if np.all(np.isfinite(scales)):
        cost += float(getattr(config, "prior_weight", 0.25)) * float(np.sum((offset / np.maximum(scales[:, None], 1e-12)) ** 2))
    return cost


def _weighted_residual(theta, correspondences, observations, layout, config, theta0, landmark_weights, marking_irls):
    matrix, landmark_raw, landmark_norm, _, _, landmark_weights, normalized_marking, marking_weights, marking_irls = _terms(
        theta, correspondences, observations, layout, config, landmark_weights, marking_irls
    )
    landmark_vector = landmark_norm.copy()
    if len(landmark_vector):
        landmark_vector = landmark_vector.reshape(-1, 2) * np.sqrt(landmark_weights)[:, None]
        landmark_vector = landmark_vector.reshape(-1)
    marking_vector = normalized_marking * np.sqrt(marking_weights * marking_irls)
    scales = local_pixel_scale(_matrix(np.asarray(theta0), layout), control_court_points(layout))
    offset = np.asarray(theta).reshape(4, 2) - np.asarray(theta0).reshape(4, 2)
    prior = np.sqrt(float(getattr(config, "prior_weight", 0.25))) * offset / np.maximum(scales[:, None], 1e-12)
    return np.concatenate((landmark_vector, marking_vector, prior.reshape(-1)))


def refine(theta0, correspondences: tuple[Correspondence, ...], observations: tuple[MarkingObservation, ...], layout: HalfCourtLayout, config, image_shape) -> RefineResult:
    """Refine from a keypoint seed within a per-control-point physical box."""
    theta0 = np.asarray(theta0, dtype=np.float64).reshape(8).copy()
    correspondences = tuple(correspondences)
    observations = tuple(observations)
    if not np.all(np.isfinite(theta0)):
        return RefineResult(theta0, None, None, None, float("inf"), 0, 0, False, 0, 0, "nonfinite_seed")
    try:
        seed = _matrix(theta0, layout)
        scales = local_pixel_scale(seed, control_court_points(layout))
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
            raise ValueError("invalid local pixel scale")
        radius = float(getattr(config, "max_control_shift_ft", 1.5)) * scales
    except Exception as error:
        return RefineResult(theta0, None, None, None, float("inf"), 0, 0, False, 0, 0, f"{type(error).__name__}:{str(error)[:120]}")

    # A zero marking set is an explicit invariant: the prior's unique minimum
    # is the seed, and returning before any numerical solve preserves exactness.
    if not observations:
        inverse = np.linalg.inv(seed)
        return RefineResult(theta0.copy(), seed.copy(), inverse / inverse[2, 2], _round_trip(seed, inverse, control_court_points(layout)), 0.0, 0, 0, True, 0, 0, None)

    theta = theta0.copy()
    mu = float(getattr(config, "lm_lambda0", 1e-3))
    total_inner = 0
    outer_done = 0
    bounds_active = 0
    convexity_backtracks = 0
    # Three separate facts, deliberately not collapsed into one flag.  A small
    # probe shift between outer iterations means the ICP feet stopped moving,
    # which happens both when the solve has arrived and when it is wedged.
    # Reporting a wedged solve as converged would let the hybrid.converged gate
    # pass on a transform the optimizer gave up on.
    inner_converged = False
    icp_settled = False
    stalled = False
    failure = None
    previous_matrix = seed
    cost = float("inf")

    for outer in range(int(getattr(config, "max_outer", 5))):
        current_matrix = _matrix(theta, layout)
        observations = freeze_observations(observations, current_matrix, layout, config, image_shape)
        # Damping restarts with the objective. Re-freezing the feet and the IRLS
        # weights defines a *new* least-squares problem, so inherited damping
        # describes a function that no longer exists. Carrying it over let one
        # hard iteration inflate mu and permanently cripple the rest: the solve
        # then found no descent step, exited as a stall, and reported the point
        # it happened to be standing on -- 1.80 px from a planted transform it
        # had exact evidence for.
        mu = float(getattr(config, "lm_lambda0", 1e-3))
        # Weights are intentionally frozen for all inner iterations in this outer loop.
        raw_landmark = landmark_residuals(theta, correspondences, layout)
        pairs = raw_landmark.reshape(-1, 2) / max(float(getattr(config, "landmark_sigma_px", 2.0)), 1e-12)
        landmark_weights = _huber_weights(np.linalg.norm(pairs, axis=1), float(getattr(config, "landmark_huber_delta", 2.0)))
        raw_marking = marking_residuals(theta, observations, layout)
        sigma = np.concatenate([item.sigma_px for item in observations]) if observations else np.zeros(0)
        marking_irls = _cauchy_weights(
            raw_marking / np.maximum(sigma, 1e-12),
            float(getattr(config, "marking_cauchy_delta", 1.5)),
        )
        cost = _objective(theta, correspondences, observations, layout, config, theta0, landmark_weights, marking_irls)
        outer_done = outer + 1
        accepted_this_outer = False
        inner_converged = False
        for _inner in range(int(getattr(config, "max_inner", 25))):
            total_inner += 1
            residual_function = lambda value: _weighted_residual(value, correspondences, observations, layout, config, theta0, landmark_weights, marking_irls)
            residual = residual_function(theta)
            jacobian = central_difference_jacobian(residual_function, theta, float(getattr(config, "jacobian_step", 1e-4)))
            normal = jacobian.T @ jacobian
            diagonal = np.diag(normal).copy()
            diagonal[diagonal <= 1e-12] = 1.0
            try:
                delta = np.linalg.solve(normal + mu * np.diag(diagonal), -(jacobian.T @ residual))
            except np.linalg.LinAlgError:
                mu *= 10.0
                continue
            if not np.all(np.isfinite(delta)):
                mu *= 10.0
                continue
            trial, active = clip_to_box(theta + delta, theta0, radius)
            bounds_active = max(bounds_active, active)
            if not is_convex(trial.reshape(4, 2)):
                convexity_backtracks += 1
                mu *= 10.0
                continue
            trial_cost = _objective(trial, correspondences, observations, layout, config, theta0, landmark_weights, marking_irls)
            if not np.isfinite(trial_cost) or trial_cost >= cost:
                mu *= 10.0
                continue
            change = cost - trial_cost
            actual_delta = trial - theta
            theta = trial
            cost = trial_cost
            accepted_this_outer = True
            mu = max(mu / 10.0, 1e-9)
            if (
                np.max(np.abs(actual_delta)) <= float(getattr(config, "step_tolerance", 1e-7)) * (1.0 + np.max(np.abs(theta)))
                or change <= float(getattr(config, "cost_tolerance", 1e-10)) * (1.0 + cost)
            ):
                inner_converged = True
                break
        new_matrix = _matrix(theta, layout)
        probes = probe_court_points(layout)
        before = project(previous_matrix, probes)
        after = project(new_matrix, probes)
        probe_shift = float(np.max(np.linalg.norm(after - before, axis=1))) if before is not None and after is not None else float("inf")
        previous_matrix = new_matrix
        # Deliberately *not* an early break. ICP progress is not monotonic per
        # iteration: the solve can crawl for one round and then move again, so
        # "barely moved this round" does not mean "arrived". Breaking on it made
        # the result chaotic -- on a planted transform, prior=0.001 stopped at
        # 1.70 px while prior=0.0003 reached 0.008 px, purely on which side of
        # the threshold one iteration happened to land. Run until the optimizer
        # genuinely stalls or the outer budget is spent, then judge convergence
        # from the fixed point actually reached.
        icp_settled = probe_shift <= float(getattr(config, "outer_tolerance_px", 1e-3))
        if not accepted_this_outer:
            stalled = True
            break

    # Converged means the transform reached a fixed point: the last outer
    # iteration left it where it found it, and the inner solve ended for a
    # reason rather than by exhausting its budget. A stall counts -- an LM that
    # cannot find a descent step after damping is at a minimum to numerical
    # precision, which is ordinary termination, not failure. What is *not*
    # convergence is running out of outer budget while still moving, and that
    # is exactly the case icp_settled excludes.
    converged = bool(icp_settled and (inner_converged or stalled))

    try:
        matrix = _matrix(theta, layout)
        inverse = np.linalg.inv(matrix)
        inverse = inverse / inverse[2, 2]
        round_trip = _round_trip(matrix, inverse, control_court_points(layout))
    except Exception as error:
        return RefineResult(theta, None, None, None, cost, outer_done, total_inner, False, bounds_active, convexity_backtracks, f"{type(error).__name__}:{str(error)[:120]}")
    return RefineResult(theta, matrix, inverse, round_trip, cost, outer_done, total_inner, converged, bounds_active, convexity_backtracks, failure)
