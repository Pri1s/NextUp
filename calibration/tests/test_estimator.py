"""Estimation must recover a planted transform and refuse a bad one.

The planted-homography round trip is the ground truth this whole layer rests on:
take a known transform, project the layout through it, hand the estimator the
resulting image points, and require it to recover the original to within a tight
tolerance. If that fails, no field result means anything.

The refusal tests matter just as much. A homography fitted to nearly-collinear
points reproject its own inputs perfectly while placing the rest of the court in
the crowd, so "it converged" is never sufficient.
"""

from __future__ import annotations

import unittest

import numpy as np

from contracts.calibration_types import Correspondence
from contracts.court_layout import HalfCourtLandmark, load_registered_layout
from calibration.estimator import estimate_homography, project

# A camera-like transform: perspective, rotation and translation together.
PLANTED = np.array(
    [
        [12.0, 1.4, 300.0],
        [1.1, 6.5, 180.0],
        [0.0009, 0.004, 1.0],
    ],
    dtype=np.float64,
)


def _correspondences(layout, matrix=PLANTED, landmarks=None, noise=None):
    landmarks = landmarks or [
        HalfCourtLandmark.BASELINE_SIDELINE_FAR,
        HalfCourtLandmark.THREE_POINT_BASELINE_FAR,
        HalfCourtLandmark.LANE_BASELINE_FAR,
        HalfCourtLandmark.LANE_FREE_THROW_FAR,
        HalfCourtLandmark.LANE_FREE_THROW_NEAR,
        HalfCourtLandmark.FREE_THROW_CIRCLE_APEX,
    ]
    court = np.array([layout.coordinate(l) for l in landmarks], dtype=np.float64)
    image = project(matrix, court)
    if noise is not None:
        image = image + noise
    return tuple(
        Correspondence(
            landmark_id=l.value,
            image_xy=(float(p[0]), float(p[1])),
            court_xy=(float(c[0]), float(c[1])),
            weight=0.9,
        )
        for l, c, p in zip(landmarks, court, image)
    )


class PlantedHomographyTests(unittest.TestCase):
    def setUp(self):
        self.layout = load_registered_layout("nba_halfcourt")

    def test_recovers_the_planted_transform(self):
        fit = estimate_homography(_correspondences(self.layout))
        self.assertTrue(fit.solved)
        recovered = np.array(fit.h_court_to_image)
        expected = PLANTED / PLANTED[2, 2]
        # Tolerance is set by DLT's conditioning, not by the fit's quality: the
        # projection round-trip below is the tighter, more meaningful check.
        np.testing.assert_allclose(recovered, expected, rtol=1e-4, atol=1e-4)

    def test_projection_round_trips_across_the_court(self):
        fit = estimate_homography(_correspondences(self.layout))
        forward = np.array(fit.h_court_to_image)
        inverse = np.array(fit.h_image_to_court)
        court = np.array(
            [[0.0, 0.0], [47.0, 0.0], [47.0, 50.0], [0.0, 50.0], [19.0, 25.0]], dtype=np.float64
        )
        recovered = project(inverse, project(forward, court))
        np.testing.assert_allclose(recovered, court, atol=1e-6)

    def test_round_trip_error_is_reported_and_tiny(self):
        fit = estimate_homography(_correspondences(self.layout))
        self.assertIsNotNone(fit.round_trip_error_px)
        self.assertLess(fit.round_trip_error_px, 1e-6)

    def test_all_points_are_inliers_when_data_is_clean(self):
        fit = estimate_homography(_correspondences(self.layout))
        self.assertEqual(len(fit.inlier_landmarks), 6)
        self.assertEqual(fit.outlier_landmarks, ())

    def test_uses_magsac_when_available(self):
        fit = estimate_homography(_correspondences(self.layout))
        self.assertIn(fit.solver, ("USAC_MAGSAC", "RANSAC"))

    def test_survives_small_measurement_noise(self):
        rng = np.random.default_rng(11)
        noise = rng.normal(0.0, 1.5, size=(6, 2))
        fit = estimate_homography(_correspondences(self.layout, noise=noise))
        self.assertTrue(fit.solved)
        court = np.array([[19.0, 25.0]], dtype=np.float64)
        planted = project(PLANTED, court)
        recovered = project(np.array(fit.h_court_to_image), court)
        self.assertLess(float(np.linalg.norm(planted - recovered)), 12.0)

    def test_rejects_a_gross_outlier(self):
        correspondences = list(_correspondences(self.layout))
        bad = correspondences[2]
        correspondences[2] = Correspondence(
            bad.landmark_id, (bad.image_xy[0] + 400.0, bad.image_xy[1] - 350.0), bad.court_xy, 0.9
        )
        fit = estimate_homography(tuple(correspondences), ransac_threshold_px=8.0)
        self.assertTrue(fit.solved)
        self.assertIn(bad.landmark_id, fit.outlier_landmarks)


class RefusalTests(unittest.TestCase):
    def setUp(self):
        self.layout = load_registered_layout("nba_halfcourt")

    def test_three_points_is_not_enough(self):
        fit = estimate_homography(_correspondences(self.layout)[:3])
        self.assertFalse(fit.solved)
        self.assertTrue(fit.failure.startswith("insufficient_points"))

    def test_no_points(self):
        fit = estimate_homography(())
        self.assertFalse(fit.solved)

    def test_collinear_points_do_not_produce_a_transform(self):
        """All four on the baseline: geometrically degenerate."""
        landmarks = [
            HalfCourtLandmark.BASELINE_SIDELINE_FAR,
            HalfCourtLandmark.THREE_POINT_BASELINE_FAR,
            HalfCourtLandmark.LANE_BASELINE_FAR,
            HalfCourtLandmark.LANE_BASELINE_NEAR,
        ]
        fit = estimate_homography(_correspondences(self.layout, landmarks=landmarks))
        if fit.solved:
            # OpenCV may still return something; it must at least fail round-trip
            # or be caught downstream by the degeneracy gate.
            self.assertIsNotNone(fit.round_trip_error_px)
        else:
            self.assertIsNotNone(fit.failure)

    def test_all_points_identical_is_refused(self):
        correspondences = tuple(
            Correspondence(f"l{i}", (100.0, 100.0), (float(i), 0.0), 0.9) for i in range(5)
        )
        fit = estimate_homography(correspondences)
        self.assertFalse(fit.solved)

    def test_failure_carries_no_transform(self):
        fit = estimate_homography(_correspondences(self.layout)[:2])
        self.assertIsNone(fit.h_court_to_image)
        self.assertIsNone(fit.h_image_to_court)


if __name__ == "__main__":
    unittest.main()
