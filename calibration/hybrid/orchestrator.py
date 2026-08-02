"""Configuration and top-level orchestration for hybrid shadow refinement."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field

import numpy as np

from contracts.annotations import HELD_OUT_FAMILIES
from contracts.calibration_types import Correspondence, CourtCalibration
from contracts.court_layout import HalfCourtLayout
from contracts.hybrid_types import (
    CandidateQuality, CandidateSource, CandidateTransform, HybridFrameStatus, HybridRefinementFrame,
    SelectionDecision, SelectionOutcome, ShadowMarkingFrame,
)
from contracts.markings import MarkingFeature

from ..quality import CalibrationPolicy
from .control_points import DEFAULT_DEPTH_BANDS, control_points_from_homography
from .residuals import collect_observations
from .refine import refine


@dataclass(frozen=True, slots=True)
class HybridGatePolicy:
    model_point_median_px: float = CalibrationPolicy().max_reprojection_px_median
    model_point_p95_px: float = CalibrationPolicy().max_reprojection_px_p95
    model_point_max_shift_px: float = 4.0
    round_trip_px: float = CalibrationPolicy().round_trip_tolerance_px
    scale_ft_per_px_range: tuple[float, float] = CalibrationPolicy().scale_ft_per_px_range
    min_court_span_x_ft: float = CalibrationPolicy().min_court_span_x_ft
    min_court_span_y_ft: float = CalibrationPolicy().min_court_span_y_ft
    depth_beyond_19ft: float = 24.0
    independent_families: int = 2
    no_ambiguous_fitted: int = 0
    held_out_available: int = 12
    held_out_median_px: float = 4.0
    held_out_p95_px: float = 10.0
    leave_one_primitive_shift_px: float = 6.0
    control_point_shift_ft: float = 1.5
    bounds_inactive: int = 0
    probe_shift_px: float = 60.0
    scale_ratio_range: tuple[float, float] = (0.9, 1.1)
    converged: bool = True
    marking_residual_median_px: float = 3.0
    validation_advantage_px: float = 0.0

    def as_dict(self) -> dict:
        return {name: (list(value) if isinstance(value, tuple) else value)
                for name, value in self.__dataclass_fields__.items()
                for value in [getattr(self, name)]}


@dataclass(frozen=True, slots=True)
class HybridRefinerConfig:
    landmark_sigma_px: float = 2.0
    landmark_huber_delta: float = 2.0
    marking_cauchy_delta: float = 1.5
    marking_sigma_min_px: float = 0.5
    marking_sigma_max_px: float = 4.0
    beta: float = 1.0
    reference_length_px: float = 120.0
    family_cap: float = 2.0
    global_cap: float = 4.0
    # A tie-breaker for flat directions, not a pull toward the seed. The prior
    # measures corner offsets in court feet, and at ~7 px/ft a weight of 0.25
    # outweighed a zero-residual data fit: planted recovery sat 5.97 px from a
    # transform it had exact evidence for. With the outer loop fixed, residual
    # bias on that probe is linear in this weight -- 0.01 -> 0.178 px,
    # 0.003 -> 0.057 px, 0.001 -> 0.021 px, 0.0 -> 0.003 px -- so this is set an
    # order of magnitude below the 0.05 px acceptance bar. In a direction the
    # data does not constrain, curvature is ~0 and any positive weight decides
    # the answer, so a small value regularises just as well as a large one.
    # The "no markings implies the seed is the minimum" invariant is carried
    # structurally by the early return in refine(), not by this term.
    prior_weight: float = 0.001
    arc_samples: int = 401
    straight_samples: int = 201
    territory_bin_px: float = 1.0
    max_control_shift_ft: float = 1.5
    # ICP needs room to re-seat the foot points after a large seed correction.
    # At max_outer=5 with a 0.01 px outer tolerance the loop stopped while still
    # 1.84 px from a planted transform; letting it run reaches 0.0026 px.
    max_outer: int = 12
    max_inner: int = 25
    lm_lambda0: float = 1e-3
    jacobian_step: float = 1e-4
    step_tolerance: float = 1e-7
    cost_tolerance: float = 1e-10
    outer_tolerance_px: float = 1e-3
    held_out_features: tuple[MarkingFeature, ...] = (MarkingFeature.FREE_THROW_CIRCLE_FAR_HALF,)
    min_fitted_features: int = 2
    depth_bands: tuple[tuple[str, float, float], ...] = DEFAULT_DEPTH_BANDS
    gates: HybridGatePolicy = field(default_factory=HybridGatePolicy)

    def __post_init__(self) -> None:
        if not set(self.held_out_features) <= set(HELD_OUT_FAMILIES):
            raise ValueError("held_out_features must be a subset of HELD_OUT_FAMILIES")
        if self.marking_sigma_min_px <= 0 or self.marking_sigma_max_px < self.marking_sigma_min_px:
            raise ValueError("marking sigma bounds are invalid")
        if self.max_control_shift_ft <= 0 or self.max_outer < 0 or self.max_inner < 0:
            raise ValueError("optimizer bounds are invalid")

    def as_dict(self) -> dict:
        result = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if name == "held_out_features":
                value = [item.value for item in value]
            elif name == "depth_bands":
                value = [list(item) for item in value]
            elif name == "gates":
                value = value.as_dict()
            result[name] = value
        return result

    def content_hash(self) -> str:
        encoded = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


class HybridProvenanceError(ValueError):
    pass


class HybridOrchestrator:
    def __init__(self, layout: HalfCourtLayout, config: HybridRefinerConfig | None = None):
        self.layout = layout
        self.config = config or HybridRefinerConfig()

    @property
    def config_hash(self) -> str:
        return self.config.content_hash()

    def _validate_provenance(self, shadow: ShadowMarkingFrame, calibration: CourtCalibration) -> None:
        expected_layout = self.layout.content_hash()
        if shadow.layout_hash != expected_layout:
            raise HybridProvenanceError("shadow layout_hash does not match refinement layout")
        if shadow.layout_id != self.layout.layout_id or calibration.layout_id != self.layout.layout_id:
            raise HybridProvenanceError("layout_id mismatch before hybrid fitting")
        provenance = calibration.provenance or {}
        image_hash = provenance.get("image_sha256")
        if image_hash is not None and image_hash != shadow.image_sha256:
            raise HybridProvenanceError("image_sha256 disagreement before hybrid fitting")
        provenance_frame = provenance.get("frame_id")
        if provenance_frame is not None and str(provenance_frame) != shadow.frame_id:
            raise HybridProvenanceError("frame_id mismatch before hybrid fitting")
        trailing = shadow.frame_id.rsplit("_", 1)[-1]
        if trailing.isdigit() and int(trailing) != calibration.frame_index:
            raise HybridProvenanceError("frame_id mismatch before hybrid fitting")

    def refine_frame(self, shadow: ShadowMarkingFrame, calibration: CourtCalibration,
                     correspondences: tuple[Correspondence, ...], *, image_shape: tuple[int, int]) -> HybridRefinementFrame:
        self._validate_provenance(shadow, calibration)
        started = time.perf_counter()
        baseline_id = f"{shadow.frame_id}:keypoint"
        challenger_id = f"{shadow.frame_id}:hybrid:{self.config_hash[:12]}"
        try:
            if not shadow.eligible:
                return self._abstained(shadow, calibration, baseline_id, shadow.eligibility_reasons)
            if not calibration.usable or calibration.h_court_to_image is None or calibration.h_image_to_court is None:
                return self._abstained(shadow, calibration, baseline_id, ("baseline_not_usable",))
            inliers = tuple(sorted(
                (item for item in correspondences if item.landmark_id in set(calibration.inlier_landmarks)),
                key=lambda item: item.landmark_id,
            ))
            observations = collect_observations(
                shadow, self.layout, np.asarray(calibration.h_court_to_image, dtype=np.float64),
                self.config, image_shape,
            )
            if not observations or len({item.feature for item in observations}) < self.config.min_fitted_features:
                return self._abstained(shadow, calibration, baseline_id, ("no_fittable_evidence",))
            theta0 = control_points_from_homography(np.asarray(calibration.h_court_to_image), self.layout)
            if theta0 is None:
                raise ValueError("baseline control points are not finite")
            result = refine(theta0.reshape(-1), inliers, observations, self.layout, self.config, image_shape)
            if result.h_court_to_image is None or result.h_image_to_court is None:
                raise ValueError(result.failure or "refiner did not produce a transform")
            from .diagnostics import build_diagnostics
            transforms, qualities, difference, stability, fitted, held_out = build_diagnostics(
                shadow, calibration, inliers, observations, result, self.layout, self.config, image_shape,
                baseline_id, challenger_id,
            )
            selection = SelectionDecision(
                frame_id=shadow.frame_id, baseline_candidate_id=baseline_id,
                challenger_candidate_id=challenger_id, selected_candidate_id=baseline_id,
                outcome=SelectionOutcome.KEEP_BASELINE,
                gates=qualities[1].gates,
                reasons=("shadow_mode_no_promotion",),
            )
            return HybridRefinementFrame(
                frame_id=shadow.frame_id, image_sha256=shadow.image_sha256, status=HybridFrameStatus.OK,
                eligible=True, eligibility_reasons=(), layout_id=self.layout.layout_id,
                layout_hash=self.layout.content_hash(), shadow_config_hash=shadow.config_hash,
                refiner_config_hash=self.config_hash, baseline_candidate_id=baseline_id,
                challenger_candidate_id=challenger_id, transforms=transforms, candidates=qualities,
                difference=difference, selection=selection, stability=stability,
                fitted_features=fitted, held_out_features=held_out,
                outer_iterations=result.outer_iterations, inner_iterations=result.inner_iterations,
                converged=result.converged, refine_ms=0.0, diagnostics_ms=0.0,
            )
        except Exception as error:
            reason = f"{type(error).__name__}:{str(error).replace(chr(10), ' ')[:120]}"
            return self._failed(shadow, baseline_id, reason)

    def _baseline_transform(self, shadow, calibration, candidate_id):
        from calibration.estimator import _as_matrix
        if calibration.h_court_to_image is None or calibration.h_image_to_court is None:
            return CandidateTransform(candidate_id, CandidateSource.KEYPOINT, self.layout.layout_id, self.layout.content_hash(), None, None, None, "keypoint", "baseline_not_usable")
        return CandidateTransform(candidate_id, CandidateSource.KEYPOINT, self.layout.layout_id, self.layout.content_hash(),
                                  _as_matrix(np.asarray(calibration.h_court_to_image)), _as_matrix(np.asarray(calibration.h_image_to_court)),
                                  0.0 if calibration.quality is None or calibration.quality.round_trip_error_px is None else calibration.quality.round_trip_error_px,
                                  "keypoint")

    def _abstained(self, shadow, calibration, baseline_id, reasons):
        from .diagnostics import baseline_quality
        transform = self._baseline_transform(shadow, calibration, baseline_id)
        quality = baseline_quality(transform, calibration, self.layout, self.config, ())
        return HybridRefinementFrame(
            frame_id=shadow.frame_id, image_sha256=shadow.image_sha256, status=HybridFrameStatus.ABSTAINED,
            eligible=shadow.eligible, eligibility_reasons=tuple(reasons), layout_id=self.layout.layout_id,
            layout_hash=self.layout.content_hash(), shadow_config_hash=shadow.config_hash,
            refiner_config_hash=self.config_hash, baseline_candidate_id=baseline_id, challenger_candidate_id=None,
            transforms=(transform,), candidates=(quality,), selection=None,
        )

    def _failed(self, shadow, baseline_id, reason):
        transform = CandidateTransform(baseline_id, CandidateSource.KEYPOINT, self.layout.layout_id, self.layout.content_hash(), None, None, None, "keypoint", reason)
        quality = CandidateQuality(
            candidate_id=baseline_id, source=CandidateSource.KEYPOINT, usable=False,
            model_point_px_median=None, model_point_px_p95=None, primitives=(),
            supported_families=(), max_depth_ft=0.0, held_out_px_median=None,
            leave_one_primitive_shift_px=None, gates=(), reasons=(reason,),
        )
        return HybridRefinementFrame(
            frame_id=shadow.frame_id, image_sha256=shadow.image_sha256, status=HybridFrameStatus.REFINEMENT_FAILED,
            eligible=shadow.eligible, eligibility_reasons=(), layout_id=self.layout.layout_id,
            layout_hash=self.layout.content_hash(), shadow_config_hash=shadow.config_hash,
            refiner_config_hash=self.config_hash, baseline_candidate_id=baseline_id, challenger_candidate_id=None,
            transforms=(transform,), candidates=(quality,), failure_reasons=(reason,),
        )
