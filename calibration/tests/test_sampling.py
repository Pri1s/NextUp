"""Sampling must be reproducible.

If two reviewers run the same command and inspect different frames, neither can
check the other's verdict. Determinism here is what makes the sign-off auditable.
"""

from __future__ import annotations

import unittest

from calibration.tools import sampling


class EvenSpacingTests(unittest.TestCase):
    def test_deterministic(self):
        first = sampling.even_indices(243, 24)
        second = sampling.even_indices(243, 24)
        self.assertEqual(first, second)

    def test_indices_are_sorted_unique_and_in_range(self):
        indices = sampling.even_indices(117, 24)
        self.assertEqual(indices, sorted(set(indices)))
        self.assertTrue(all(0 <= i < 117 for i in indices))

    def test_requested_count_is_met_when_there_is_room(self):
        self.assertEqual(len(sampling.even_indices(243, 24)), 24)

    def test_spacing_is_actually_even(self):
        indices = sampling.even_indices(240, 24)
        gaps = {b - a for a, b in zip(indices, indices[1:])}
        self.assertEqual(gaps, {10})

    def test_samples_avoid_the_extreme_endpoints(self):
        """Bin centres, so the first and last samples miss fades and cuts."""
        indices = sampling.even_indices(240, 24)
        self.assertGreater(indices[0], 0)
        self.assertLess(indices[-1], 239)

    def test_more_frames_requested_than_exist_returns_all(self):
        self.assertEqual(sampling.even_indices(5, 24), [0, 1, 2, 3, 4])

    def test_empty_source(self):
        self.assertEqual(sampling.even_indices(0, 24), [])

    def test_zero_requested(self):
        self.assertEqual(sampling.even_indices(100, 0), [])


class ExplicitFrameTests(unittest.TestCase):
    def test_parses_sorts_and_dedupes(self):
        self.assertEqual(sampling.parse_frame_list("80, 0,40, 40"), [0, 40, 80])

    def test_out_of_range_dropped_not_clamped(self):
        """Silently moving a requested frame elsewhere would misreport evidence."""
        self.assertEqual(sampling.parse_frame_list("0,40,9999", frame_count=117), [0, 40])

    def test_rejects_non_integer(self):
        with self.assertRaises(ValueError):
            sampling.parse_frame_list("0,middle")

    def test_rejects_negative(self):
        with self.assertRaises(ValueError):
            sampling.parse_frame_list("-3")

    def test_rejects_all_out_of_range(self):
        with self.assertRaises(ValueError):
            sampling.parse_frame_list("500,600", frame_count=117)


class ResolveTests(unittest.TestCase):
    def test_explicit_wins_over_count(self):
        self.assertEqual(sampling.resolve_indices(117, 24, "0,40,80"), [0, 40, 80])

    def test_falls_back_to_even_spacing(self):
        self.assertEqual(sampling.resolve_indices(117, 24), sampling.even_indices(117, 24))


if __name__ == "__main__":
    unittest.main()
