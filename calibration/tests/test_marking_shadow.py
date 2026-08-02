from __future__ import annotations

import json
import unittest

import numpy as np

from calibration.markings import ShadowOrchestrator, corridor_width_px
from calibration.markings.shadow import ShadowProvenanceError
from contracts.calibration_types import CalibrationQuality, CalibrationStatus, CourtCalibration
from contracts.court_layout import load_registered_layout


class ShadowEligibilityTests(unittest.TestCase):
    def setUp(self):
        self.layout = load_registered_layout("nba_halfcourt")
        self.image = np.zeros((90, 160, 3), dtype=np.uint8)
        self.h = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        self.hi = self.h

    def calibration(self, status=CalibrationStatus.DEGRADED, inliers=5, reasons=("weak_conditioning:x",)):
        quality = CalibrationQuality(inliers, inliers, 1.0, reasons=tuple(reasons))
        return CourtCalibration(
            1, self.layout.layout_id, status,
            h_image_to_court=self.hi if status.usable else None,
            h_court_to_image=self.h if status.usable else None,
            quality=quality, provenance={"weights_sha256": "a" * 64},
        )

    def test_only_weak_conditioned_five_inlier_degraded_seeds_run(self):
        orchestrator = ShadowOrchestrator(self.layout, "a" * 64)
        record = orchestrator.analyze_frame(self.image, "f", self.calibration())
        self.assertTrue(record.eligible)
        self.assertEqual(record.status.value, "OK")
        for calibration in (
            self.calibration(CalibrationStatus.OK),
            self.calibration(inliers=4),
            self.calibration(reasons=("few_points:5",)),
            self.calibration(CalibrationStatus.UNCALIBRATED_INSUFFICIENT_EVIDENCE),
        ):
            record = orchestrator.analyze_frame(self.image, "f", calibration)
            self.assertFalse(record.eligible)
            self.assertEqual(record.status.value, "ABSTAINED")

    def test_checkpoint_mismatch_is_before_analysis(self):
        with self.assertRaises(ShadowProvenanceError):
            ShadowOrchestrator(self.layout, "b" * 64).analyze_frame(
                self.image, "f", self.calibration()
            )

    def test_wrong_sized_mask_is_rejected_even_for_abstention(self):
        with self.assertRaises(ValueError):
            ShadowOrchestrator(self.layout, "a" * 64).analyze_frame(
                self.image, "f", self.calibration(CalibrationStatus.OK),
                exclusion_mask=np.zeros((10, 10), dtype=np.uint8),
            )

    def test_record_round_trip_and_diagonal_scaled_corridor(self):
        orchestrator = ShadowOrchestrator(self.layout, "a" * 64)
        record = orchestrator.analyze_frame(self.image, "f", self.calibration())
        restored = type(record).from_dict(json.loads(json.dumps(record.as_dict())))
        self.assertEqual(record, restored)
        self.assertAlmostEqual(corridor_width_px(0), 12.0)
        self.assertAlmostEqual(corridor_width_px(20), 35.0)


if __name__ == "__main__":
    unittest.main()
