"""The parameterisation must be exact, bounded, and free of an OpenCV dependency."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from calibration.estimator import project
from calibration.hybrid.control_points import (
    DEFAULT_DEPTH_BANDS,
    band_indices,
    clip_to_box,
    control_court_points,
    control_points_from_homography,
    homography_from_control_points,
    is_convex,
    local_pixel_scale,
    probe_court_points,
)
from calibration.tests.test_estimator import PLANTED
from contracts.court_layout import HalfCourtLandmark, load_registered_layout


def layout():
    return load_registered_layout("nba_halfcourt")


def deterministic_quads(count: int) -> list[np.ndarray]:
    """Plausible broadcast-ish image quads, generated without an RNG."""
    quads = []
    for index in range(count):
        skew = 40.0 + 3.0 * index
        lift = 20.0 + 2.5 * index
        quads.append(
            np.array(
                [
                    [200.0 - index, 620.0 - lift],
                    [420.0 + skew, 180.0 + lift],
                    [900.0 - skew, 180.0 + lift],
                    [1120.0 + index, 620.0 - lift],
                ],
                dtype=np.float64,
            )
        )
    return quads


class DltTests(unittest.TestCase):
    def test_maps_its_correspondences_exactly(self):
        """The property that actually matters, and it is exact in float64.

        A four-point DLT is an exact solve, not a fit: the four control points
        must land on their image positions to machine precision. This is the
        real correctness check, independent of any other implementation.
        """
        court = control_court_points(layout())
        for index, quad in enumerate(deterministic_quads(50)):
            with self.subTest(quad=index):
                matrix = homography_from_control_points(court, quad)
                self.assertIsNotNone(matrix)
                landed = project(matrix, court)
                self.assertLess(float(np.max(np.linalg.norm(landed - quad, axis=1))), 1e-9)

    def test_agrees_with_opencv_within_float32(self):
        """Cross-check against ``cv2.getPerspectiveTransform``.

        The tolerance is set by OpenCV, not by us: that function only accepts
        float32, so the comparison cannot be tighter than float32 epsilon
        (~1e-7 relative) however exact our own solve is. Exactness is pinned by
        ``test_maps_its_correspondences_exactly``; this test exists to catch a
        structurally different matrix, not to measure precision.
        """
        court = control_court_points(layout())
        for index, quad in enumerate(deterministic_quads(50)):
            with self.subTest(quad=index):
                ours = homography_from_control_points(court, quad)
                theirs = cv2.getPerspectiveTransform(
                    court.astype(np.float32), quad.astype(np.float32)
                )
                theirs = theirs / theirs[2, 2]
                self.assertIsNotNone(ours)
                np.testing.assert_allclose(ours, theirs, rtol=1e-5, atol=1e-8)

    def test_control_points_round_trip_exactly(self):
        for name, matrix in (("planted", np.array(PLANTED, dtype=np.float64)),):
            with self.subTest(matrix=name):
                corners = control_points_from_homography(matrix, layout())
                self.assertIsNotNone(corners)
                rebuilt = homography_from_control_points(control_court_points(layout()), corners)
                self.assertIsNotNone(rebuilt)
                probes = probe_court_points(layout())
                before = project(matrix, probes)
                after = project(rebuilt, probes)
                self.assertLess(float(np.max(np.linalg.norm(after - before, axis=1))), 1e-9)

    def test_result_is_w_normalised(self):
        matrix = homography_from_control_points(
            control_court_points(layout()), deterministic_quads(1)[0]
        )
        self.assertAlmostEqual(float(matrix[2, 2]), 1.0, places=12)

    def test_degenerate_input_returns_none(self):
        collapsed = np.zeros((4, 2), dtype=np.float64)
        self.assertIsNone(homography_from_control_points(control_court_points(layout()), collapsed))
        with self.assertRaises(ValueError):
            homography_from_control_points(control_court_points(layout())[:3], collapsed[:3])


class ProbeSetTests(unittest.TestCase):
    def test_probe_points_are_unique(self):
        probes = probe_court_points(layout())
        unique = {(round(x, 9), round(y, 9)) for x, y in probes}
        self.assertEqual(len(unique), len(probes))

    def test_probe_points_span_every_depth_band(self):
        depth = probe_court_points(layout())[:, 0]
        index = band_indices(depth, DEFAULT_DEPTH_BANDS)
        for position, (name, _, _) in enumerate(DEFAULT_DEPTH_BANDS):
            with self.subTest(band=name):
                self.assertTrue(bool(np.any(index == position)))

    def test_every_band_holds_a_point_that_is_not_a_control_point(self):
        """Otherwise a band's displacement just restates the parameter vector."""
        probes = probe_court_points(layout())
        controls = {(round(x, 9), round(y, 9)) for x, y in control_court_points(layout())}
        index = band_indices(probes[:, 0], DEFAULT_DEPTH_BANDS)
        for position, (name, _, _) in enumerate(DEFAULT_DEPTH_BANDS):
            with self.subTest(band=name):
                in_band = probes[index == position]
                free = [p for p in in_band if (round(p[0], 9), round(p[1], 9)) not in controls]
                self.assertTrue(free, f"band {name} contains only control points")


    def test_probe_set_reaches_the_deepest_fittable_paint(self):
        # The three-point arc apex at 29 ft is where a far-court disagreement
        # first shows up; the named landmarks stop at 24.9 ft.
        depth = probe_court_points(layout())[:, 0]
        self.assertAlmostEqual(float(np.max(depth)), layout().half_length, places=6)
        self.assertTrue(bool(np.any(np.isclose(depth, 29.0, atol=1e-6))))

    def test_most_probe_points_are_not_control_points(self):
        probes = {(round(x, 9), round(y, 9)) for x, y in probe_court_points(layout())}
        controls = {(round(x, 9), round(y, 9)) for x, y in control_court_points(layout())}
        self.assertTrue(controls <= probes)
        self.assertGreaterEqual(len(probes - controls), 11)


class BoxTests(unittest.TestCase):
    def test_a_vector_inside_the_box_is_unchanged(self):
        theta0 = deterministic_quads(1)[0].reshape(-1)
        theta = theta0 + 1.0
        clipped, active = clip_to_box(theta, theta0, np.full(4, 50.0))
        np.testing.assert_allclose(clipped, theta)
        self.assertEqual(active, 0)

    def test_an_excursion_lands_exactly_on_the_boundary(self):
        theta0 = deterministic_quads(1)[0].reshape(-1)
        theta = theta0.copy().reshape(4, 2)
        theta[0] += np.array([300.0, 400.0])  # 500 px away
        clipped, active = clip_to_box(theta.reshape(-1), theta0, np.full(4, 20.0))
        offset = clipped.reshape(4, 2) - theta0.reshape(4, 2)
        self.assertAlmostEqual(float(np.linalg.norm(offset[0])), 20.0, places=9)
        self.assertEqual(active, 1)
        np.testing.assert_allclose(offset[1:], 0.0, atol=1e-12)

    def test_a_zero_radius_pins_the_corner(self):
        theta0 = deterministic_quads(1)[0].reshape(-1)
        clipped, active = clip_to_box(theta0 + 5.0, theta0, np.zeros(4))
        np.testing.assert_allclose(clipped, theta0)
        self.assertEqual(active, 4)


class ConvexityTests(unittest.TestCase):
    def test_a_broadcast_quad_is_convex(self):
        self.assertTrue(is_convex(deterministic_quads(1)[0]))

    def test_a_folded_quad_is_rejected(self):
        quad = deterministic_quads(1)[0].copy()
        quad[[1, 2]] = quad[[2, 1]]  # cross two adjacent corners
        self.assertFalse(is_convex(quad))

    def test_non_finite_is_rejected(self):
        quad = deterministic_quads(1)[0].copy()
        quad[0, 0] = np.nan
        self.assertFalse(is_convex(quad))


class PixelScaleTests(unittest.TestCase):
    def test_a_foot_is_larger_near_the_camera_than_at_midcourt(self):
        matrix = np.array(PLANTED, dtype=np.float64)
        scale = local_pixel_scale(matrix, control_court_points(layout()))
        near_baseline = 0.5 * (scale[0] + scale[3])
        at_midcourt = 0.5 * (scale[1] + scale[2])
        self.assertGreater(near_baseline, at_midcourt)

    def test_scale_is_finite_and_positive_on_a_usable_transform(self):
        scale = local_pixel_scale(np.array(PLANTED, dtype=np.float64), probe_court_points(layout()))
        self.assertTrue(bool(np.all(np.isfinite(scale))))
        self.assertTrue(bool(np.all(scale > 0)))


class LandmarkCoverageTests(unittest.TestCase):
    def test_every_named_landmark_is_probed(self):
        probes = {(round(x, 9), round(y, 9)) for x, y in probe_court_points(layout())}
        for landmark in HalfCourtLandmark:
            with self.subTest(landmark=landmark.value):
                x, y = layout().coordinate(landmark)
                self.assertIn((round(x, 9), round(y, 9)), probes)


if __name__ == "__main__":
    unittest.main()
