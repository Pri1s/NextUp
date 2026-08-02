"""The public entry point: one frame in, one explicit answer out.

`FrameCalibrator.calibrate` never raises on bad input and never returns a
transform it could not justify. Every path ends in a `CalibrationStatus`, and the
statuses are distinguishable because they call for different responses:

``NOT_ATTEMPTED_NO_COURT``               the detector saw nothing; not a failure
``UNCALIBRATED_INSUFFICIENT_EVIDENCE``   court is visible but the evidence is too
                                         thin or too uncertain to fit — the
                                         honest answer for a provisional detector
``FAILED_*``                             a fit existed and was rejected, with the
                                         specific check that rejected it
``DEGRADED`` / ``OK``                    usable, with quality attached

Downstream code should branch on ``calibration.usable``, and anything that needs
to explain itself reads ``quality.reasons``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean

from contracts.calibration_types import (
    CalibrationSource,
    CalibrationStatus,
    CourtCalibration,
    RawKeypointFrame,
)
from contracts.court_layout import HalfCourtLayout

from .adapters.evidence import EvidenceMap, pool_evidence
from .correspondence import build_correspondences
from .estimator import estimate_homography
from .quality import CalibrationPolicy, evaluate


@dataclass(frozen=True, slots=True)
class CalibrationAttempt:
    """A calibration plus the working that produced it."""

    calibration: CourtCalibration
    evidence: tuple = ()
    dropped_evidence: tuple = ()
    dropped_correspondences: tuple = ()

    def as_dict(self) -> dict:
        return {
            "calibration": self.calibration.as_dict(),
            "evidence": [e.as_dict() for e in self.evidence],
            "dropped_evidence": [{"identifier": i, "reason": r} for i, r in self.dropped_evidence],
            "dropped_correspondences": [
                {"identifier": i, "reason": r} for i, r in self.dropped_correspondences
            ],
        }


@dataclass
class FrameCalibrator:
    """Calibrates single frames against one layout and one evidence map."""

    layout: HalfCourtLayout
    evidence_map: EvidenceMap
    policy: CalibrationPolicy = field(default_factory=CalibrationPolicy)
    tiers: tuple[str, ...] = ("confident",)

    def calibrate(self, detection: RawKeypointFrame) -> CalibrationAttempt:
        provenance = {
            "layout_id": self.layout.layout_id,
            "layout_hash": self.layout.content_hash(),
            "evidence_map_id": self.evidence_map.map_id,
            "evidence_map_status": self.evidence_map.status,
            "weights_sha256": self.evidence_map.weights_sha256,
            "tiers": list(self.tiers),
            "policy": self.policy.as_dict(),
            "slot_semantics": "provisional_unverified",
        }

        def result(status: CalibrationStatus, reasons=(), **extra) -> CalibrationAttempt:
            from contracts.calibration_types import CalibrationQuality

            quality = extra.pop("quality", None)
            if quality is None and reasons:
                quality = CalibrationQuality(
                    observation_count=extra.pop("observation_count", 0),
                    inlier_count=0,
                    inlier_ratio=0.0,
                    reasons=tuple(reasons),
                )
            return CalibrationAttempt(
                calibration=CourtCalibration(
                    frame_index=detection.frame_index,
                    layout_id=self.layout.layout_id,
                    status=status,
                    quality=quality,
                    provenance=provenance,
                    source=CalibrationSource.DIRECT,
                ),
                **extra,
            )

        if not detection.detected:
            return result(CalibrationStatus.NOT_ATTEMPTED_NO_COURT, ["no_detection"])

        evidence, dropped_evidence = pool_evidence(
            detection,
            self.evidence_map,
            tiers=self.tiers,
            min_confidence=self.policy.min_pooled_confidence,
        )

        if len(evidence) < self.policy.absolute_min_points:
            return result(
                CalibrationStatus.UNCALIBRATED_INSUFFICIENT_EVIDENCE,
                [f"pooled_evidence:{len(evidence)}<{self.policy.absolute_min_points}"],
                observation_count=len(evidence),
                evidence=evidence,
                dropped_evidence=dropped_evidence,
            )

        correspondence_set = build_correspondences(
            evidence, self.layout, self.policy.min_pooled_confidence
        )
        correspondences = correspondence_set.correspondences

        if len(correspondences) < self.policy.absolute_min_points:
            return result(
                CalibrationStatus.UNCALIBRATED_INSUFFICIENT_EVIDENCE,
                [f"correspondences:{len(correspondences)}<{self.policy.absolute_min_points}"],
                observation_count=len(correspondences),
                evidence=evidence,
                dropped_evidence=dropped_evidence,
                dropped_correspondences=correspondence_set.dropped,
            )

        fit = estimate_homography(
            correspondences,
            ransac_threshold_px=self.policy.ransac_threshold_px,
            round_trip_tolerance_px=self.policy.round_trip_tolerance_px,
        )

        if not fit.solved or fit.h_court_to_image is None or fit.h_image_to_court is None:
            failure = fit.failure or "solver_failed"
            status = (
                CalibrationStatus.FAILED_INSUFFICIENT_POINTS
                if failure.startswith("insufficient")
                else CalibrationStatus.FAILED_DEGENERATE_GEOMETRY
            )
            return result(
                status,
                [failure],
                observation_count=len(correspondences),
                evidence=evidence,
                dropped_evidence=dropped_evidence,
                dropped_correspondences=correspondence_set.dropped,
            )

        status, quality = evaluate(
            correspondences=correspondences,
            inlier_landmarks=fit.inlier_landmarks,
            h_court_to_image=fit.h_court_to_image,
            h_image_to_court=fit.h_image_to_court,
            layout=self.layout,
            image_size=(detection.image_width, detection.image_height),
            policy=self.policy,
            round_trip_error_px=fit.round_trip_error_px,
            mean_confidence=fmean([e.confidence for e in evidence]) if evidence else None,
        )

        usable = status.usable
        calibration = CourtCalibration(
            frame_index=detection.frame_index,
            layout_id=self.layout.layout_id,
            status=status,
            h_image_to_court=fit.h_image_to_court if usable else None,
            h_court_to_image=fit.h_court_to_image if usable else None,
            quality=quality,
            inlier_landmarks=fit.inlier_landmarks,
            outlier_landmarks=fit.outlier_landmarks,
            source=CalibrationSource.DIRECT,
            provenance={**provenance, "solver": fit.solver},
        )

        return CalibrationAttempt(
            calibration=calibration,
            evidence=evidence,
            dropped_evidence=dropped_evidence,
            dropped_correspondences=correspondence_set.dropped,
        )
