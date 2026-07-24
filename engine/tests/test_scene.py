"""Scene-cut detector behaviour."""

from __future__ import annotations

import unittest

import numpy as np

from engine.pipeline.scene import SceneCutDetector

DARK = np.full((48, 64, 3), 20, dtype=np.uint8)
BRIGHT = np.full((48, 64, 3), 230, dtype=np.uint8)


class TestSceneCutDetector(unittest.TestCase):
    def test_first_frame_primes_without_cut(self):
        det = SceneCutDetector(threshold=0.5)
        self.assertFalse(det.is_cut(DARK))

    def test_identical_frames_are_not_a_cut(self):
        det = SceneCutDetector(threshold=0.5)
        det.is_cut(DARK)
        self.assertFalse(det.is_cut(DARK))

    def test_hard_color_change_is_a_cut(self):
        det = SceneCutDetector(threshold=0.5)
        det.is_cut(DARK)
        self.assertTrue(det.is_cut(BRIGHT))
        # After settling on the new shot, the next identical frame is not a cut.
        self.assertFalse(det.is_cut(BRIGHT))

    def test_reset_clears_reference(self):
        det = SceneCutDetector(threshold=0.5)
        det.is_cut(DARK)
        det.reset()
        self.assertFalse(det.is_cut(BRIGHT))  # primes again, no comparison


if __name__ == "__main__":
    unittest.main()
