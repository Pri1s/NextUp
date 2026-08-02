"""Measurements and gates for the two shadow candidates."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from contracts.calibration_types import Correspondence, CourtCalibration
from contracts.markings import MarkingFamily, MarkingFeature
from contracts.hybrid_types import (
    AssignmentStatus, CandidateQuality, CandidateSource, CandidateTransform,
    GateResult, MarkingAssignment, PrimitiveQuality, PrimitiveStability,
    ProbeDisplacement, ShadowMarkingFrame, TransformDifference,
)

from ..estimator import _as_matrix, project
from ..quality import CalibrationPolicy, _scale_ft_per_px, plausibility_failures
from ..polyline import nearest_on_polyline
from .control_points import (
    control_court_points, control_points_from_homography, local_pixel_scale,
    probe_displacement, probe_court_points,
)
from .refine import RefineResult, refine
from .residuals import MarkingObservation, collect_observations, measure_feature_residuals


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(float(item) for item in values)
    index = max(0, min(len(ordered) - 1, int(round(fraction * len(ordered) + 0.5)) - 1))
    return ordered[index]


def _primitive_quality(shadow, layout, matrix, config, fitted_features):
    measurements = measure_feature_residuals(shadow, layout, matrix, config)
    features = set(measurements) | set(config.held_out_features)
    rows = []
    for feature in sorted(features, key=lambda item: item.value):
        values = measurements.get(feature, [])
        distances = [item[0] for item in values]
        depth = [item[1] for item in values]
        if not depth:
            template = np.asarray(layout.marking(feature).sample(2 if feature.kind != "arc" else 41), dtype=np.float64)
            depth = [float(value) for value in template[:, 0]]
        rows.append(PrimitiveQuality(
            feature=feature, fitted=feature in fitted_features,
            sample_count=len(distances),
            residual_px_median=float(np.median(distances)) if distances else None,
            residual_px_p95=_percentile(distances, .95),
            depth_min_ft=min(depth), depth_max_ft=max(depth),
        ))
    return tuple(rows), measurements


def _candidate_gates(source, matrix, inverse, calibration, correspondences, layout, config, model_median, model_p95, max_shift, max_depth, supported, held_median, leave_one, difference, marking_median, round_trip=None):
    policy = CalibrationPolicy()
    scale = _scale_ft_per_px(inverse, (calibration.provenance.get("image_width", 1280), calibration.provenance.get("image_height", 720)))
    if scale is None:
        scale = float("inf")
    image_size = (int(calibration.provenance.get("image_width", 1280)), int(calibration.provenance.get("image_height", 720)))
    span_x = float(max((item.court_xy[0] for item in correspondences), default=0) - min((item.court_xy[0] for item in correspondences), default=0))
    span_y = float(max((item.court_xy[1] for item in correspondences), default=0) - min((item.court_xy[1] for item in correspondences), default=0))
    physical = plausibility_failures(_as_matrix(matrix), _as_matrix(inverse), layout, image_size, scale, policy)
    gates = [
        GateResult("candidate.model_point_median_px", model_median <= config.gates.model_point_median_px, model_median, config.gates.model_point_median_px, "px", "model median exceeds policy" if model_median > config.gates.model_point_median_px else ""),
        GateResult("candidate.model_point_p95_px", model_p95 <= config.gates.model_point_p95_px, model_p95, config.gates.model_point_p95_px, "px", "model p95 exceeds policy" if model_p95 > config.gates.model_point_p95_px else ""),
        GateResult("candidate.model_point_max_shift_px", max_shift <= config.gates.model_point_max_shift_px, max_shift, config.gates.model_point_max_shift_px, "px", "model point shift exceeds policy" if max_shift > config.gates.model_point_max_shift_px else ""),
        GateResult("candidate.round_trip_px", (calibration.quality.round_trip_error_px if round_trip is None and source is CandidateSource.KEYPOINT and calibration.quality and calibration.quality.round_trip_error_px is not None else (0.0 if round_trip is None else round_trip)) <= config.gates.round_trip_px, (calibration.quality.round_trip_error_px if round_trip is None and source is CandidateSource.KEYPOINT and calibration.quality and calibration.quality.round_trip_error_px is not None else (0.0 if round_trip is None else round_trip)), config.gates.round_trip_px, "px", "round trip exceeds policy" if ((calibration.quality.round_trip_error_px if round_trip is None and source is CandidateSource.KEYPOINT and calibration.quality and calibration.quality.round_trip_error_px is not None else (0.0 if round_trip is None else round_trip)) > config.gates.round_trip_px) else ""),
        GateResult("candidate.scale_plausible", config.gates.scale_ft_per_px_range[0] <= scale <= config.gates.scale_ft_per_px_range[1], scale, config.gates.scale_ft_per_px_range[1], "ft/px", "scale outside policy" if not config.gates.scale_ft_per_px_range[0] <= scale <= config.gates.scale_ft_per_px_range[1] else ""),
        GateResult("candidate.court_quad_convex", "court_quad_not_convex" not in physical, None, None, "", "court quad is not convex" if "court_quad_not_convex" in physical else ""),
        GateResult("candidate.court_span_ft", span_x >= config.gates.min_court_span_x_ft and span_y >= config.gates.min_court_span_y_ft, min(span_x, span_y), min(config.gates.min_court_span_x_ft, config.gates.min_court_span_y_ft), "ft", "inlier court span is too small" if span_x < config.gates.min_court_span_x_ft or span_y < config.gates.min_court_span_y_ft else ""),
    ]
    return gates


def _model_errors(matrix, correspondences, baseline_matrix):
    court = np.asarray([item.court_xy for item in correspondences], dtype=np.float64)
    image = np.asarray([item.image_xy for item in correspondences], dtype=np.float64)
    projected = project(matrix, court)
    baseline = project(baseline_matrix, court)
    if projected is None:
        return [], float("inf")
    errors = np.linalg.norm(projected - image, axis=1)
    shift = np.linalg.norm(projected - baseline, axis=1) if baseline is not None else np.zeros(len(errors))
    return [float(item) for item in errors], float(np.max(shift)) if len(shift) else 0.0


def baseline_quality(transform, calibration, layout, config, primitives):
    if not transform.solved:
        return CandidateQuality(transform.candidate_id, CandidateSource.KEYPOINT, False, None, None, (), (), 0.0, None, None, (), ("baseline_transform_missing",))
    matrix = np.asarray(transform.h_court_to_image); inverse = np.asarray(transform.h_image_to_court)
    inliers = tuple()
    errors = []
    if calibration.quality:
        errors = [0.0] * calibration.quality.inlier_count
    gates = _candidate_gates(CandidateSource.KEYPOINT, matrix, inverse, calibration, inliers, layout, config, float(np.median(errors)) if errors else 0.0, _percentile(errors,.95) or 0.0, 0.0, 19.0, (), None, None, None, None)
    reasons = tuple(gate.gate_id for gate in gates if not gate.passed)
    return CandidateQuality(transform.candidate_id, CandidateSource.KEYPOINT, not reasons, float(np.median(errors)) if errors else 0.0, _percentile(errors,.95) or 0.0, tuple(primitives), (), 19.0, None, None, tuple(gates), reasons)


def build_diagnostics(shadow, calibration, correspondences, observations, result: RefineResult, layout, config, image_shape, baseline_id, challenger_id):
    baseline_matrix = np.asarray(calibration.h_court_to_image, dtype=np.float64)
    baseline_inverse = np.asarray(calibration.h_image_to_court, dtype=np.float64)
    challenger_matrix = np.asarray(result.h_court_to_image, dtype=np.float64)
    challenger_inverse = np.asarray(result.h_image_to_court, dtype=np.float64)
    fitted = tuple(sorted({item.feature for item in observations}, key=lambda item: item.value))
    held_out = tuple(sorted(config.held_out_features, key=lambda item: item.value))
    primitive_rows, measurements = _primitive_quality(shadow, layout, challenger_matrix, config, set(fitted))
    baseline_rows, _ = _primitive_quality(shadow, layout, baseline_matrix, config, set())
    baseline_errors, _ = _model_errors(baseline_matrix, correspondences, baseline_matrix)
    challenger_errors, max_shift = _model_errors(challenger_matrix, correspondences, baseline_matrix)
    baseline_transform = CandidateTransform(baseline_id, CandidateSource.KEYPOINT, layout.layout_id, layout.content_hash(), _as_matrix(baseline_matrix), _as_matrix(baseline_inverse), calibration.quality.round_trip_error_px if calibration.quality else 0.0, "keypoint")
    challenger_transform = CandidateTransform(challenger_id, CandidateSource.HYBRID, layout.layout_id, layout.content_hash(), _as_matrix(challenger_matrix), _as_matrix(challenger_inverse), result.round_trip_error_px, "projected_lm")
    held_values = [value for feature in held_out for value, _ in measure_feature_residuals(shadow, layout, challenger_matrix, config).get(feature, [])]
    held_baseline = [value for feature in held_out for value, _ in measure_feature_residuals(shadow, layout, baseline_matrix, config).get(feature, [])]
    held_median = float(np.median(held_values)) if held_values else None
    held_p95 = _percentile(held_values, .95)
    marking_values = [value for feature in fitted for value, _ in measurements.get(feature, [])]
    marking_median = float(np.median(marking_values)) if marking_values else 0.0
    families_ready = {item.family for item in shadow.family_support if item.ready}
    supported = tuple(sorted((family for family in families_ready if family in {feature.family for feature in fitted}), key=lambda item: item.value))
    inlier_depth = max((item.court_xy[0] for item in correspondences), default=0.0)
    max_depth = max([inlier_depth] + [row.depth_max_ft for row in primitive_rows if row.fitted])

    stability = []
    for feature in fitted:
        reduced = tuple(item for item in observations if item.feature is not feature)
        theta0 = control_points_from_homography(baseline_matrix, layout)
        refit = refine(theta0.reshape(-1), correspondences, reduced, layout, config, image_shape)
        if refit.h_court_to_image is None:
            shift = float("inf")
            ok = False
        else:
            probe_a = project(challenger_matrix, probe_court_points(layout)); probe_b = project(refit.h_court_to_image, probe_court_points(layout))
            shift = float(np.max(np.linalg.norm(probe_b - probe_a, axis=1)))
            ok = refit.converged
        stability.append(PrimitiveStability(feature, shift, ok))
    leave_one = max((item.shift_px for item in stability), default=0.0)
    diff = probe_displacement(baseline_matrix, challenger_matrix, layout, config.depth_bands)
    distances = np.asarray(diff["total"], dtype=np.float64)
    scale_base = _scale_ft_per_px(baseline_inverse, image_shape)
    scale_hybrid = _scale_ft_per_px(challenger_inverse, image_shape)
    ratio = scale_hybrid / scale_base if scale_base and scale_hybrid else 1.0
    base_controls = control_points_from_homography(baseline_matrix, layout); hybrid_controls = control_points_from_homography(challenger_matrix, layout)
    scales = local_pixel_scale(baseline_matrix, control_court_points(layout))
    control_shift = float(np.max(np.linalg.norm(hybrid_controls - base_controls, axis=1) / np.maximum(scales, 1e-12))) if base_controls is not None and hybrid_controls is not None else float("inf")
    bands = tuple(ProbeDisplacement(name, int(np.sum(np.asarray(diff["bands"][name]).size)), float(np.median(diff["bands"][name])) if len(diff["bands"][name]) else 0.0, float(np.max(diff["bands"][name])) if len(diff["bands"][name]) else 0.0) for name, _, _ in config.depth_bands)
    difference = TransformDifference(baseline_id, challenger_id, len(distances), float(np.median(distances)), _percentile(list(distances), .95) or 0.0, float(np.max(distances)) if len(distances) else 0.0, control_shift, result.bounds_active, ratio, bands)
    common = _candidate_gates(CandidateSource.KEYPOINT, baseline_matrix, baseline_inverse, calibration, correspondences, layout, config, float(np.median(baseline_errors)) if baseline_errors else 0.0, _percentile(baseline_errors,.95) or 0.0, 0.0, 19.0, (), None, None, difference, None)
    base_reasons = tuple(gate.gate_id for gate in common if not gate.passed)
    baseline_quality_record = CandidateQuality(baseline_id, CandidateSource.KEYPOINT, not base_reasons, float(np.median(baseline_errors)) if baseline_errors else 0.0, _percentile(baseline_errors,.95) or 0.0, baseline_rows, (), max(inlier_depth, 19.0), float(np.median(held_baseline)) if held_baseline else None, None, tuple(common), base_reasons)
    candidate_gates = _candidate_gates(CandidateSource.HYBRID, challenger_matrix, challenger_inverse, calibration, correspondences, layout, config, float(np.median(challenger_errors)) if challenger_errors else 0.0, _percentile(challenger_errors,.95) or 0.0, max_shift, max_depth, supported, held_median, leave_one, difference, marking_median, result.round_trip_error_px)
    ambiguous = sum(1 for assignment in shadow.assignments if assignment.status is AssignmentStatus.AMBIGUOUS and assignment.alternatives and assignment.alternatives[0].feature in fitted)
    hybrid_gates = [
        GateResult("hybrid.depth_beyond_19ft", max_depth >= config.gates.depth_beyond_19ft, max_depth, config.gates.depth_beyond_19ft, "ft", "insufficient fitted depth" if max_depth < config.gates.depth_beyond_19ft else ""),
        GateResult("hybrid.independent_families", len(supported) >= config.gates.independent_families, len(supported), config.gates.independent_families, "families", "insufficient independent families" if len(supported) < config.gates.independent_families else ""),
        GateResult("hybrid.no_ambiguous_fitted", ambiguous == config.gates.no_ambiguous_fitted, ambiguous, config.gates.no_ambiguous_fitted, "fragments", "ambiguous fitted assignment" if ambiguous else ""),
        GateResult("hybrid.held_out_available", len(held_values) >= config.gates.held_out_available, len(held_values), config.gates.held_out_available, "samples", "held-out evidence unavailable" if len(held_values) < config.gates.held_out_available else ""),
        GateResult("hybrid.held_out_median_px", held_median is not None and held_median <= config.gates.held_out_median_px, held_median or 0.0, config.gates.held_out_median_px, "px", "held-out median exceeds policy" if held_median is None or held_median > config.gates.held_out_median_px else ""),
        GateResult("hybrid.held_out_p95_px", held_p95 is not None and held_p95 <= config.gates.held_out_p95_px, held_p95 or 0.0, config.gates.held_out_p95_px, "px", "held-out p95 exceeds policy" if held_p95 is None or held_p95 > config.gates.held_out_p95_px else ""),
        GateResult("hybrid.leave_one_primitive_shift_px", leave_one <= config.gates.leave_one_primitive_shift_px, leave_one, config.gates.leave_one_primitive_shift_px, "px", "primitive leave-one-out shift exceeds policy" if leave_one > config.gates.leave_one_primitive_shift_px else ""),
        GateResult("hybrid.control_point_shift_ft", control_shift <= config.gates.control_point_shift_ft, control_shift, config.gates.control_point_shift_ft, "ft", "control point shift exceeds policy" if control_shift > config.gates.control_point_shift_ft else ""),
        GateResult("hybrid.bounds_inactive", result.bounds_active == config.gates.bounds_inactive, result.bounds_active, config.gates.bounds_inactive, "corners", "optimizer touched a bound" if result.bounds_active else ""),
        GateResult("hybrid.probe_shift_px", difference.probe_px_max <= config.gates.probe_shift_px, difference.probe_px_max, config.gates.probe_shift_px, "px", "probe displacement exceeds policy" if difference.probe_px_max > config.gates.probe_shift_px else ""),
        GateResult("hybrid.scale_ratio", config.gates.scale_ratio_range[0] <= ratio <= config.gates.scale_ratio_range[1], ratio, config.gates.scale_ratio_range[1], "ratio", "scale ratio outside policy" if not config.gates.scale_ratio_range[0] <= ratio <= config.gates.scale_ratio_range[1] else ""),
        GateResult("hybrid.converged", result.converged == config.gates.converged, float(result.converged), 1.0, "bool", "optimizer did not converge" if not result.converged else ""),
        GateResult("hybrid.marking_residual_median_px", marking_median <= config.gates.marking_residual_median_px, marking_median, config.gates.marking_residual_median_px, "px", "fitted marking residual is too large" if marking_median > config.gates.marking_residual_median_px else ""),
        GateResult("hybrid.validation_advantage_px", (float(np.median(held_baseline)) - held_median) > config.gates.validation_advantage_px if held_baseline and held_median is not None else False, (float(np.median(held_baseline)) - held_median) if held_baseline and held_median is not None else 0.0, config.gates.validation_advantage_px, "px", "challenger has no held-out advantage"),
    ]
    all_gates = tuple(common + hybrid_gates)
    hybrid_reasons = tuple(gate.gate_id for gate in all_gates if not gate.passed)
    challenger_quality = CandidateQuality(challenger_id, CandidateSource.HYBRID, not hybrid_reasons, float(np.median(challenger_errors)) if challenger_errors else 0.0, _percentile(challenger_errors,.95) or 0.0, primitive_rows, supported, max_depth, held_median, leave_one, all_gates, hybrid_reasons)
    return (baseline_transform, challenger_transform), (baseline_quality_record, challenger_quality), difference, tuple(stability), fitted, held_out
