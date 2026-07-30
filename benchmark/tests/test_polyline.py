"""Point-to-polyline geometry, where a subtle wrong answer is worse than an error.

Two failure modes are specifically guarded.

Measuring to the nearest *vertex* rather than the nearest point on the curve would
make every comparison depend on how finely each side happened to be sampled -- a
reference drawn with fewer points would score worse for no geometric reason.

Losing the sign would hide bias. Unsigned error cannot tell noise from a
systematic offset, and a systematic offset shared by two annotation passes is
invisible to every other check in the benchmark.
"""

import unittest

import numpy as np

from benchmark.polyline import (
    nearest_on_polyline,
    overlapping_extent,
    polyline_length,
    resample,
)

HORIZONTAL = np.array([[0.0, 0.0], [100.0, 0.0]])


class NearestTests(unittest.TestCase):
    def test_distance_is_to_the_segment_not_the_vertices(self):
        """The midpoint of a long segment is on the line, however far from either end."""
        result = nearest_on_polyline([[50.0, 0.0]], HORIZONTAL)
        self.assertAlmostEqual(float(result.distances[0]), 0.0, places=9)

    def test_perpendicular_distance(self):
        result = nearest_on_polyline([[50.0, 7.0]], HORIZONTAL)
        self.assertAlmostEqual(float(result.distances[0]), 7.0, places=9)
        self.assertTrue(np.allclose(result.closest[0], [50.0, 0.0]))

    def test_projection_is_clamped_to_the_polyline(self):
        """A point past the end lands on the endpoint, not an imagined extension.

        This is what stops a reference drawn across the whole court from being
        credited with explaining paint beyond where it actually stops.
        """
        result = nearest_on_polyline([[150.0, 0.0]], HORIZONTAL)
        self.assertAlmostEqual(float(result.distances[0]), 50.0, places=9)
        self.assertTrue(np.allclose(result.closest[0], [100.0, 0.0]))

    def test_sign_flips_across_the_line(self):
        above = nearest_on_polyline([[50.0, 5.0]], HORIZONTAL).signed[0]
        below = nearest_on_polyline([[50.0, -5.0]], HORIZONTAL).signed[0]
        self.assertAlmostEqual(float(above), -float(below), places=9)
        self.assertNotAlmostEqual(float(above), 0.0)

    def test_sign_is_stable_along_a_curve(self):
        """A constant offset from a curve must read as constant, not alternating."""
        angles = np.linspace(0.0, np.pi, 40)
        arc = np.stack([50.0 * np.cos(angles), 50.0 * np.sin(angles)], axis=1)
        outside = arc * 1.04  # uniformly 4% further from the centre
        signed = nearest_on_polyline(outside[5:-5], arc).signed
        self.assertTrue(np.all(signed > 0) or np.all(signed < 0), signed)

    def test_picks_the_nearest_of_several_segments(self):
        elbow = np.array([[0.0, 0.0], [50.0, 0.0], [50.0, 50.0]])
        result = nearest_on_polyline([[52.0, 30.0]], elbow)
        self.assertAlmostEqual(float(result.distances[0]), 2.0, places=9)
        self.assertEqual(int(result.segment[0]), 1)

    def test_arc_length_tracks_position_along_the_curve(self):
        result = nearest_on_polyline([[25.0, 0.0], [75.0, 0.0]], HORIZONTAL)
        self.assertAlmostEqual(float(result.arc_length[0]), 25.0, places=9)
        self.assertAlmostEqual(float(result.arc_length[1]), 75.0, places=9)

    def test_repeated_points_do_not_divide_by_zero(self):
        degenerate = np.array([[0.0, 0.0], [0.0, 0.0], [10.0, 0.0]])
        result = nearest_on_polyline([[5.0, 3.0]], degenerate)
        self.assertTrue(np.all(np.isfinite(result.distances)))
        self.assertAlmostEqual(float(result.distances[0]), 3.0, places=9)

    def test_empty_query_is_empty_not_an_error(self):
        self.assertEqual(len(nearest_on_polyline(np.zeros((0, 2)), HORIZONTAL)), 0)

    def test_polyline_needs_two_points(self):
        with self.assertRaises(ValueError):
            nearest_on_polyline([[0.0, 0.0]], [[0.0, 0.0]])

    def test_rejects_wrong_shape(self):
        with self.assertRaises(ValueError):
            nearest_on_polyline([0.0, 1.0, 2.0], HORIZONTAL)


class ResampleTests(unittest.TestCase):
    def test_resampling_is_evenly_spaced_by_arc_length(self):
        uneven = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [100.0, 0.0]])
        points = resample(uneven, 5)
        steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        self.assertTrue(np.allclose(steps, 25.0))

    def test_endpoints_are_preserved(self):
        points = resample(HORIZONTAL, 7)
        self.assertTrue(np.allclose(points[0], HORIZONTAL[0]))
        self.assertTrue(np.allclose(points[-1], HORIZONTAL[-1]))

    def test_zero_length_polyline_does_not_divide_by_zero(self):
        points = resample(np.array([[3.0, 4.0], [3.0, 4.0]]), 4)
        self.assertTrue(np.allclose(points, [3.0, 4.0]))

    def test_length(self):
        self.assertAlmostEqual(polyline_length(HORIZONTAL), 100.0)
        self.assertEqual(polyline_length([[1.0, 1.0]]), 0.0)


class OverlapTests(unittest.TestCase):
    def test_shared_extent_drops_the_unmatched_tail(self):
        """Two annotators tracing different runs of one line still agree on the overlap."""
        long_run = np.array([[float(x), 0.0] for x in range(0, 101, 10)])
        short_run = np.array([[float(x), 0.0] for x in range(0, 51, 10)])
        a, b = overlapping_extent(long_run, short_run, tolerance_px=2.0)
        self.assertLessEqual(float(np.max(a[:, 0])), 50.0)
        self.assertEqual(len(b), len(short_run))

    def test_coincident_polylines_are_kept_whole(self):
        a, b = overlapping_extent(HORIZONTAL, HORIZONTAL, tolerance_px=1.0)
        self.assertEqual(len(a), 2)
        self.assertEqual(len(b), 2)

    def test_parallel_but_separated_lines_share_nothing(self):
        far = HORIZONTAL + np.array([0.0, 50.0])
        a, b = overlapping_extent(HORIZONTAL, far, tolerance_px=2.0)
        self.assertEqual(len(a), 0)
        self.assertEqual(len(b), 0)


if __name__ == "__main__":
    unittest.main()
