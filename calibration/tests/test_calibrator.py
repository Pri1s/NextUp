"""End-to-end calibration, and the statuses that keep it honest.

The point of this layer is that it says no. With provisional slot semantics, a
calibrator that always returns a transform is worse than useless — it launders
bad evidence into confident-looking geometry. So most of these tests check that a
bad input produces the *right refusal*, distinguishable from every other refusal.
"""

from __future__ import annotations

import unittest

import numpy as np

from contracts.calibration_types import (
    CalibrationStatus,
    RawKeypointFrame,
    RawKeypointObservation,
)
from contracts.court_layout import HalfCourtLandmark, load_registered_layout
from calibration.adapters.evidence import load_registered_map
from calibration.calibrator import FrameCalibrator
from calibration.estimator import project
from calibration.quality import CalibrationPolicy

PLANTED = np.array(
    [[12.0, 1.4, 300.0], [1.1, 6.5, 180.0], [0.0009, 0.004, 1.0]], dtype=np.float64
)

# The shipped provisional map's confident tier.
PAIRS = {
    HalfCourtLandmark.BASELINE_SIDELINE_FAR: (0, 15),
    HalfCourtLandmark.THREE_POINT_BASELINE_FAR: (1, 14),
    HalfCourtLandmark.LANE_BASELINE_FAR: (2, 13),
    HalfCourtLandmark.LANE_FREE_THROW_FAR: (8, 16),
    HalfCourtLandmark.LANE_FREE_THROW_NEAR: (9, 17),
}


def _detection_from_layout(layout, split=(0.5, 0.5), landmarks=None, jitter=0.0, frame_index=0):
    """Synthesise a detection whose slots project the layout through PLANTED.

    Confidence is deliberately *split* across each pair, reproducing the real
    model's behaviour, so these tests exercise pooling rather than bypassing it.
    """
    observations = []
    rng = np.random.default_rng(3)
    for landmark, (slot_a, slot_b) in PAIRS.items():
        if landmarks is not None and landmark not in landmarks:
            continue
        court = np.array([layout.coordinate(landmark)], dtype=np.float64)
        (x, y) = project(PLANTED, court)[0]
        if jitter:
            x += rng.normal(0.0, jitter)
            y += rng.normal(0.0, jitter)
        observations.append(RawKeypointObservation(slot_a, float(x), float(y), split[0]))
        observations.append(RawKeypointObservation(slot_b, float(x) + 4.0, float(y) + 3.0, split[1]))

    return RawKeypointFrame(
        frame_index=frame_index,
        timestamp_s=0.0,
        source_id="synthetic.mp4",
        image_width=1280,
        image_height=720,
        detection_state="detected",
        instance_count=1,
        instance_confidences=(0.9,),
        selected_instance=0,
        observations=tuple(sorted(observations, key=lambda o: o.slot_index)),
    )


def _no_detection(frame_index=0) -> RawKeypointFrame:
    return RawKeypointFrame(
        frame_index=frame_index,
        timestamp_s=0.0,
        source_id="synthetic.mp4",
        image_width=1280,
        image_height=720,
        detection_state="no_detection",
        instance_count=0,
        instance_confidences=(),
        selected_instance=None,
        observations=(),
    )


class CalibratorTestCase(unittest.TestCase):
    def setUp(self):
        self.layout = load_registered_layout("nba_halfcourt")
        self.map = load_registered_map("reloc2_18_provisional")
        self.calibrator = FrameCalibrator(layout=self.layout, evidence_map=self.map)


class SuccessTests(CalibratorTestCase):
    def test_clean_evidence_calibrates(self):
        attempt = self.calibrator.calibrate(_detection_from_layout(self.layout))
        self.assertTrue(attempt.calibration.usable, attempt.calibration.quality.reasons)

    def test_recovers_court_positions(self):
        attempt = self.calibrator.calibrate(_detection_from_layout(self.layout))
        calibration = attempt.calibration
        for landmark in PAIRS:
            court = np.array([self.layout.coordinate(landmark)], dtype=np.float64)
            (x, y) = project(PLANTED, court)[0]
            recovered = calibration.image_to_court(float(x), float(y))
            np.testing.assert_allclose(recovered, court[0], atol=6.0)

    def test_both_transforms_present_and_mutually_inverse(self):
        calibration = self.calibrator.calibrate(_detection_from_layout(self.layout)).calibration
        image_point = (640.0, 400.0)
        court = calibration.image_to_court(*image_point)
        back = calibration.court_to_image(*court)
        np.testing.assert_allclose(back, image_point, atol=1e-6)

    def test_pooling_happens_before_fitting(self):
        """Five landmarks from ten slots, each pair split 0.5/0.5."""
        attempt = self.calibrator.calibrate(_detection_from_layout(self.layout))
        self.assertEqual(len(attempt.evidence), 5)
        for evidence in attempt.evidence:
            self.assertEqual(len(evidence.source_slots), 2)

    def test_even_split_would_fail_per_slot_but_succeeds_pooled(self):
        """Each slot at 0.20 is below the 0.25 gate; pooled they are 0.40."""
        detection = _detection_from_layout(self.layout, split=(0.20, 0.20))
        attempt = self.calibrator.calibrate(detection)
        self.assertEqual(len(attempt.evidence), 5)
        self.assertTrue(attempt.calibration.usable)

    def test_end_identity_stays_unresolved(self):
        calibration = self.calibrator.calibrate(_detection_from_layout(self.layout)).calibration
        self.assertEqual(calibration.end_identity, "unresolved")
        self.assertEqual(calibration.frame_of_reference, "visible_end_half_court")

    def test_provenance_marks_semantics_provisional(self):
        calibration = self.calibrator.calibrate(_detection_from_layout(self.layout)).calibration
        self.assertEqual(calibration.provenance["slot_semantics"], "provisional_unverified")
        self.assertEqual(calibration.provenance["evidence_map_status"], "provisional")

    def test_quality_is_populated(self):
        quality = self.calibrator.calibrate(_detection_from_layout(self.layout)).calibration.quality
        self.assertIsNotNone(quality.reprojection_px_median)
        self.assertIsNotNone(quality.min_triangle_area_px)
        self.assertIsNotNone(quality.round_trip_error_px)
        self.assertEqual(quality.inlier_count, 5)


class RefusalTests(CalibratorTestCase):
    def test_no_detection_is_not_a_failure(self):
        calibration = self.calibrator.calibrate(_no_detection()).calibration
        self.assertEqual(calibration.status, CalibrationStatus.NOT_ATTEMPTED_NO_COURT)
        self.assertFalse(calibration.usable)

    def test_too_few_landmarks_returns_uncalibrated_not_a_failure(self):
        """Three visible landmarks is thin evidence, not a broken fit."""
        detection = _detection_from_layout(
            self.layout,
            landmarks={
                HalfCourtLandmark.BASELINE_SIDELINE_FAR,
                HalfCourtLandmark.LANE_BASELINE_FAR,
                HalfCourtLandmark.LANE_FREE_THROW_FAR,
            },
        )
        calibration = self.calibrator.calibrate(detection).calibration
        self.assertEqual(
            calibration.status, CalibrationStatus.UNCALIBRATED_INSUFFICIENT_EVIDENCE
        )

    def test_low_confidence_everywhere_returns_uncalibrated(self):
        detection = _detection_from_layout(self.layout, split=(0.05, 0.05))
        calibration = self.calibrator.calibrate(detection).calibration
        self.assertEqual(
            calibration.status, CalibrationStatus.UNCALIBRATED_INSUFFICIENT_EVIDENCE
        )
        self.assertFalse(calibration.usable)

    def test_collinear_evidence_is_rejected_as_degenerate(self):
        """Four baseline landmarks fit perfectly and mean nothing."""
        detection = _detection_from_layout(
            self.layout,
            landmarks={
                HalfCourtLandmark.BASELINE_SIDELINE_FAR,
                HalfCourtLandmark.THREE_POINT_BASELINE_FAR,
                HalfCourtLandmark.LANE_BASELINE_FAR,
            },
        )
        calibration = self.calibrator.calibrate(detection).calibration
        self.assertFalse(calibration.usable)

    def test_no_transform_is_attached_to_a_failure(self):
        calibration = self.calibrator.calibrate(_no_detection()).calibration
        self.assertIsNone(calibration.h_image_to_court)
        self.assertIsNone(calibration.h_court_to_image)

    def test_using_a_transform_from_a_failed_calibration_raises(self):
        calibration = self.calibrator.calibrate(_no_detection()).calibration
        with self.assertRaises(ValueError):
            calibration.image_to_court(100.0, 100.0)

    def test_scrambled_evidence_does_not_yield_a_usable_calibration(self):
        """The check that catches a wrong slot map: self-consistent, still wrong."""
        detection = _detection_from_layout(self.layout)
        observations = list(detection.observations)
        # Swap the image positions of two landmarks' slot pairs.
        by_slot = {o.slot_index: o for o in observations}
        for a, b in ((0, 2), (15, 13)):
            oa, ob = by_slot[a], by_slot[b]
            by_slot[a] = RawKeypointObservation(a, ob.x, ob.y, oa.confidence)
            by_slot[b] = RawKeypointObservation(b, oa.x, oa.y, ob.confidence)
        scrambled = RawKeypointFrame(
            frame_index=0, timestamp_s=0.0, source_id="s.mp4",
            image_width=1280, image_height=720, detection_state="detected",
            instance_count=1, instance_confidences=(0.9,), selected_instance=0,
            observations=tuple(sorted(by_slot.values(), key=lambda o: o.slot_index)),
        )
        calibration = self.calibrator.calibrate(scrambled).calibration
        self.assertFalse(
            calibration.usable,
            f"scrambled correspondences produced {calibration.status.value}",
        )

    def test_every_refusal_explains_itself(self):
        for detection in (
            _no_detection(),
            _detection_from_layout(self.layout, split=(0.05, 0.05)),
        ):
            calibration = self.calibrator.calibrate(detection).calibration
            self.assertTrue(calibration.quality.reasons, calibration.status)


class DropReportingTests(CalibratorTestCase):
    def test_unmapped_slots_are_reported(self):
        attempt = self.calibrator.calibrate(_detection_from_layout(self.layout))
        reasons = dict(attempt.dropped_evidence)
        self.assertEqual(reasons.get("slot_3"), "ambiguous")
        self.assertEqual(reasons.get("slot_12"), "ambiguous")
        self.assertEqual(reasons.get("slot_4"), "unusable")

    def test_probable_tier_excluded_by_default(self):
        attempt = self.calibrator.calibrate(_detection_from_layout(self.layout))
        self.assertIn("tier_excluded:probable", dict(attempt.dropped_evidence).values())

    def test_attempt_serialises(self):
        payload = self.calibrator.calibrate(_detection_from_layout(self.layout)).as_dict()
        self.assertIn("calibration", payload)
        self.assertIn("dropped_evidence", payload)
        self.assertEqual(payload["calibration"]["end_identity"], "unresolved")


class PolicyTests(CalibratorTestCase):
    def test_stricter_reprojection_policy_downgrades(self):
        strict = FrameCalibrator(
            layout=self.layout,
            evidence_map=self.map,
            policy=CalibrationPolicy(max_reprojection_px_median=0.001),
        )
        detection = _detection_from_layout(self.layout, jitter=3.0)
        self.assertFalse(strict.calibrate(detection).calibration.usable)

    def test_policy_is_recorded_in_provenance(self):
        calibration = self.calibrator.calibrate(_detection_from_layout(self.layout)).calibration
        self.assertIn("policy", calibration.provenance)
        self.assertIn("ransac_threshold_px", calibration.provenance["policy"])


if __name__ == "__main__":
    unittest.main()
