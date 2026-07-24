"""Prove the homography recovers a known synthetic court->image mapping."""

import unittest

import cv2
import numpy as np

from contracts.court_schema import canonical_ids
from contracts.court_template import load_template
from engine.geometry.homography import MIN_POINTS, project, solve_homography


# A non-degenerate synthetic camera: court plane (x=across, y=along) -> image px,
# with a mild perspective term so the recovered map is a true homography.
H_COURT_TO_IMAGE = np.array(
    [
        [18.0, 1.5, 120.0],
        [2.0, 9.0, 80.0],
        [0.0, 0.0008, 1.0],
    ],
    dtype=np.float64,
)


def _project(points_xy: np.ndarray, homography: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, homography).reshape(-1, 2)


class HomographyTests(unittest.TestCase):
    def setUp(self):
        self.template = load_template("nba_94x50")
        self.ids = canonical_ids()
        # Court points as (across, along) — the plane convention solve_homography uses.
        self.court_pts = np.array(
            [[self.template.court_xy_ft[cid][1], self.template.court_xy_ft[cid][0]] for cid in self.ids],
            dtype=np.float64,
        )
        self.image_pts = _project(self.court_pts, H_COURT_TO_IMAGE)

    def _canonical(self, visibility) -> np.ndarray:
        kpts = np.zeros((len(self.ids), 3), dtype=float)
        kpts[:, 0:2] = self.image_pts
        kpts[:, 2] = visibility
        return kpts

    def test_recovers_court_coordinates(self):
        kpts = self._canonical(visibility=2)
        homography, used = solve_homography(kpts, self.template)
        self.assertIsNotNone(homography)
        self.assertEqual(len(used), len(self.ids))
        recovered = project(homography, self.image_pts)  # -> (across, along)
        np.testing.assert_allclose(recovered, self.court_pts, atol=1e-3)

    def test_too_few_points_returns_none(self):
        kpts = self._canonical(visibility=0)
        kpts[0:MIN_POINTS - 1, 2] = 2  # one short of the minimum
        homography, used = solve_homography(kpts, self.template)
        self.assertIsNone(homography)
        self.assertEqual(used, [])


if __name__ == "__main__":
    unittest.main()
