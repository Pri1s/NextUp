"""The half-court layout must be derived, symmetric, and end-agnostic.

Geometry here is the one thing in the pipeline that is *not* provisional, so it
has to be exactly right — every calibration is measured against it. The tests
also guard the design decision that makes the interim engine safe: no landmark
name encodes north/south, because nothing has established which end is in shot.
"""

import json
import tempfile
import unittest
from pathlib import Path

from contracts.court_layout import (
    LAYOUT_SCHEMA_VERSION,
    HalfCourtLandmark,
    HalfCourtLayout,
    available_layouts,
    build_layout,
    load_layout,
    load_registered_layout,
    validate_layout,
)

NBA = dict(
    layout_id="test_nba",
    rule_set="NBA",
    source="test",
    half_length=47.0,
    width=50.0,
    lane_width=16.0,
    free_throw_distance=19.0,
    basket_from_baseline=5.25,
    three_point_corner_inset=3.0,
    free_throw_circle_radius=6.0,
)


class BuilderTests(unittest.TestCase):
    def setUp(self):
        self.layout = build_layout(**NBA)

    def test_derives_lane_from_scalars(self):
        """Lane edges come from width/2 +- lane_width/2, not a typed-in table."""
        self.assertEqual(self.layout.coordinate(HalfCourtLandmark.LANE_BASELINE_FAR), (0.0, 17.0))
        self.assertEqual(self.layout.coordinate(HalfCourtLandmark.LANE_BASELINE_NEAR), (0.0, 33.0))

    def test_free_throw_line_is_at_the_rule_book_distance(self):
        self.assertEqual(self.layout.coordinate(HalfCourtLandmark.LANE_FREE_THROW_FAR), (19.0, 17.0))
        self.assertEqual(self.layout.coordinate(HalfCourtLandmark.LANE_FREE_THROW_NEAR), (19.0, 33.0))

    def test_free_throw_apex_is_circle_radius_beyond_the_line(self):
        self.assertEqual(
            self.layout.coordinate(HalfCourtLandmark.FREE_THROW_CIRCLE_APEX), (25.0, 25.0)
        )

    def test_corners_and_midcourt(self):
        self.assertEqual(self.layout.coordinate(HalfCourtLandmark.BASELINE_SIDELINE_FAR), (0.0, 0.0))
        self.assertEqual(
            self.layout.coordinate(HalfCourtLandmark.MIDCOURT_SIDELINE_NEAR), (47.0, 50.0)
        )

    def test_basket_centre(self):
        self.assertEqual(self.layout.basket_center, (5.25, 25.0))

    def test_origin_is_the_visible_baseline(self):
        """+X toward midcourt means every landmark has non-negative x."""
        for _, (x, _) in self.layout.landmarks.items():
            self.assertGreaterEqual(x, 0.0)

    def test_all_landmarks_are_in_bounds(self):
        for landmark, (x, y) in self.layout.landmarks.items():
            self.assertLessEqual(x, self.layout.half_length, landmark.value)
            self.assertLessEqual(y, self.layout.width, landmark.value)

    def test_out_of_bounds_landmark_rejected(self):
        with self.assertRaises(ValueError):
            HalfCourtLayout(
                **{k: v for k, v in NBA.items()},
                landmarks={HalfCourtLandmark.BASELINE_SIDELINE_FAR: (999.0, 0.0)},
            )

    def test_non_feet_units_rejected(self):
        with self.assertRaises(ValueError):
            HalfCourtLayout(**NBA, units="meters")


class EndAgnosticismTests(unittest.TestCase):
    """The design decision that makes an unverified detector safe to use."""

    def test_no_landmark_name_encodes_court_end(self):
        for landmark in HalfCourtLandmark:
            self.assertNotIn("north", landmark.value)
            self.assertNotIn("south", landmark.value)

    def test_sidelines_are_camera_relative_not_compass_relative(self):
        for landmark in HalfCourtLandmark:
            self.assertNotIn("east", landmark.value)
            self.assertNotIn("west", landmark.value)

    def test_sideline_mirrors_are_reciprocal(self):
        for landmark in HalfCourtLandmark:
            mirror = landmark.sideline_mirror
            if mirror is not None:
                self.assertEqual(mirror.sideline_mirror, landmark)

    def test_apex_has_no_mirror(self):
        self.assertIsNone(HalfCourtLandmark.FREE_THROW_CIRCLE_APEX.sideline_mirror)


class ValidatorTests(unittest.TestCase):
    def test_healthy_layout_has_no_problems(self):
        self.assertEqual(validate_layout(build_layout(**NBA)), [])

    def test_detects_asymmetric_mirror_pair(self):
        layout = build_layout(**NBA)
        broken = HalfCourtLayout(
            **{k: v for k, v in NBA.items()},
            landmarks={**layout.landmarks, HalfCourtLandmark.LANE_BASELINE_NEAR: (0.0, 40.0)},
        )
        problems = validate_layout(broken)
        self.assertTrue(any("symmetric" in p for p in problems), problems)

    def test_detects_off_centre_apex(self):
        layout = build_layout(**NBA)
        broken = HalfCourtLayout(
            **{k: v for k, v in NBA.items()},
            landmarks={**layout.landmarks, HalfCourtLandmark.FREE_THROW_CIRCLE_APEX: (25.0, 10.0)},
        )
        self.assertTrue(any("off-centre" in p for p in validate_layout(broken)))

    def test_builder_rejects_a_free_throw_line_past_midcourt(self):
        """Impossible scalars fail at construction, before anyone can fit to them."""
        with self.assertRaises(ValueError):
            build_layout(**{**NBA, "free_throw_distance": 50.0})

    def test_builder_rejects_a_lane_wider_than_the_court(self):
        with self.assertRaises(ValueError):
            build_layout(**{**NBA, "lane_width": 60.0})

    def test_validator_also_catches_bad_scalars_on_direct_construction(self):
        """Belt and braces: the builder is not the only way to make a layout."""
        layout = HalfCourtLayout(**{**NBA, "free_throw_distance": 50.0}, landmarks={})
        self.assertTrue(any("midcourt" in p for p in validate_layout(layout)))

        layout = HalfCourtLayout(**{**NBA, "lane_width": 60.0}, landmarks={})
        self.assertTrue(any("wider" in p for p in validate_layout(layout)))

    def test_validator_catches_a_basket_outside_the_half_court(self):
        layout = HalfCourtLayout(**{**NBA, "basket_from_baseline": 90.0}, landmarks={})
        self.assertTrue(any("basket" in p for p in validate_layout(layout)))


class HashTests(unittest.TestCase):
    def test_same_geometry_same_hash(self):
        self.assertEqual(build_layout(**NBA).content_hash(), build_layout(**NBA).content_hash())

    def test_different_geometry_different_hash(self):
        other = build_layout(**{**NBA, "lane_width": 12.0})
        self.assertNotEqual(build_layout(**NBA).content_hash(), other.content_hash())


class ProfileTests(unittest.TestCase):
    def test_registered_nba_profile_loads_and_validates(self):
        layout = load_registered_layout("nba_halfcourt")
        self.assertEqual(layout.rule_set, "NBA")
        self.assertEqual(layout.half_length, 47.0)
        self.assertEqual(validate_layout(layout), [])

    def test_nba_profile_is_registered(self):
        self.assertIn("nba_halfcourt", available_layouts())

    def test_unknown_layout_raises_with_a_helpful_message(self):
        with self.assertRaises(FileNotFoundError) as caught:
            load_registered_layout("nba_full_court")
        self.assertIn("available", str(caught.exception))

    def test_schema_version_is_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps({"schema_version": "half-court-layout-0.0.1"}))
            with self.assertRaises(ValueError):
                load_layout(path)

    def test_profile_declares_the_current_schema(self):
        path = Path(__file__).resolve().parents[1] / "layouts" / "nba_halfcourt.json"
        self.assertEqual(json.loads(path.read_text())["schema_version"], LAYOUT_SCHEMA_VERSION)


if __name__ == "__main__":
    unittest.main()
