"""The filled NBA court-keypoint map is valid and geometrically consistent."""

from __future__ import annotations

import unittest

import numpy as np

from contracts.court_schema import CANONICAL_COUNT, canonical_ids
from contracts.court_template import load_template
from engine.court.adapters import NBA_COURT_KEYPOINT_MAP, NbaCourtAdapter
from engine.geometry.homography import project, solve_homography


class TestNbaCourtMap(unittest.TestCase):
    def test_map_targets_valid_ids_and_is_injective(self):
        cids = set(canonical_ids())
        targets = list(NBA_COURT_KEYPOINT_MAP.values())
        self.assertLessEqual(set(targets), cids)  # every target is a real canonical id
        self.assertEqual(len(targets), len(set(targets)))  # no canonical id reused
        self.assertEqual(len(NBA_COURT_KEYPOINT_MAP), 18)  # all 18 native points mapped

    def test_adapter_builds_without_error(self):
        # NbaCourtAdapter validates the ids against the schema at construction.
        NbaCourtAdapter(NBA_COURT_KEYPOINT_MAP)

    def test_map_recovers_a_planted_homography(self):
        template = load_template("nba_94x50")
        # Plant a known court->image linear map A, synthesize the 18 native kpts,
        # then confirm the adapter + solver recover it and project points correctly.
        A = np.array([[6.0, 0.0, 40.0], [0.0, 6.0, 30.0], [0.0, 0.0, 1.0]])
        source = np.zeros((18, 3), dtype=float)
        for s, cid in NBA_COURT_KEYPOINT_MAP.items():
            along, across = template.court_xy_ft[cid]
            px = A @ np.array([across, along, 1.0])  # homography.py uses (across, along)
            source[s] = [px[0] / px[2], px[1] / px[2], 2]

        canonical = NbaCourtAdapter(NBA_COURT_KEYPOINT_MAP).to_canonical(source)
        self.assertEqual(canonical.shape, (CANONICAL_COUNT, 3))

        homography, used = solve_homography(canonical, template)
        self.assertIsNotNone(homography)
        self.assertGreaterEqual(len(used), 4)

        cid = NBA_COURT_KEYPOINT_MAP[8]  # north_lane_free_throw_east
        along, across = template.court_xy_ft[cid]
        px = A @ np.array([across, along, 1.0])
        got = project(homography, [px[:2] / px[2]])[0]
        self.assertAlmostEqual(got[0], across, places=1)
        self.assertAlmostEqual(got[1], along, places=1)


if __name__ == "__main__":
    unittest.main()
