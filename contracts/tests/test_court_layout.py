"""The v2 layout must be authoritative painted-centerline geometry."""

import dataclasses
import json
import math
import tempfile
import unittest
from pathlib import Path

from contracts.court_layout import (
    LAYOUT_SCHEMA_VERSION,
    LINE_CONVENTION,
    HalfCourtLandmark,
    MeasurementReference,
    SourceMeasurement,
    available_layouts,
    build_layout,
    load_layout,
    load_registered_layout,
    validate_layout,
)
from contracts.markings import ArcMarking, MarkingFeature, StraightMarking

CENTERLINE = dict(
    layout_id="test_nba",
    rule_set="test",
    source="test",
    half_length=47.0,
    width=50.0,
    lane_width=16.0,
    free_throw_distance=19.0,
    basket_from_baseline=5.25,
    three_point_corner_inset=3.0,
    three_point_radius=23.75,
    free_throw_circle_radius=6.0,
    center_circle_radius=6.0,
    restricted_area_radius=4.0,
    line_width=2.0 / 12.0,
)


class OfficialConversionTests(unittest.TestCase):
    def setUp(self):
        self.layout = load_registered_layout("nba_halfcourt")

    def test_two_inch_stripe_and_centerline_convention(self):
        self.assertAlmostEqual(self.layout.line_width, 2 / 12)
        self.assertEqual(self.layout.line_convention, LINE_CONVENTION)

    def test_inside_court_dimensions_convert_to_boundary_centerlines(self):
        self.assertAlmostEqual(self.layout.half_length, 47 + 1 / 12)
        self.assertAlmostEqual(self.layout.width, 50 + 2 / 12)
        self.assertEqual(
            self.layout.coordinate(HalfCourtLandmark.MIDCOURT_SIDELINE_NEAR),
            (self.layout.half_length, self.layout.width),
        )

    def test_basket_free_throw_and_lane_conversions(self):
        self.assertAlmostEqual(self.layout.basket_from_baseline, 5 + 4 / 12)
        self.assertAlmostEqual(self.layout.free_throw_distance, 19.0)
        self.assertAlmostEqual(self.layout.lane_width, 15 + 10 / 12)
        far = self.layout.coordinate(HalfCourtLandmark.LANE_BASELINE_FAR)
        near = self.layout.coordinate(HalfCourtLandmark.LANE_BASELINE_NEAR)
        self.assertEqual(far[0], 0.0)
        self.assertAlmostEqual(far[1], 17 + 2 / 12)
        self.assertEqual(near, (0.0, 33.0))

    def test_corner_rule_converts_to_three_feet_two_inches(self):
        self.assertAlmostEqual(self.layout.three_point_corner_inset, 3 + 2 / 12)
        basket_to_line = self.layout.width / 2 - self.layout.three_point_corner_inset
        self.assertAlmostEqual(basket_to_line, 21 + 11 / 12)

    def test_published_radii_convert_on_the_correct_side(self):
        self.assertAlmostEqual(self.layout.three_point_radius, 23 + 8 / 12)
        self.assertAlmostEqual(self.layout.free_throw_circle_radius, 5 + 11 / 12)
        self.assertAlmostEqual(self.layout.center_circle_radius, 5 + 11 / 12)
        self.assertAlmostEqual(self.layout.restricted_area_radius, 4 + 1 / 12)

    def test_three_point_corner_and_arc_join_exactly(self):
        far = self.layout.marking(MarkingFeature.THREE_POINT_CORNER_FAR)
        near = self.layout.marking(MarkingFeature.THREE_POINT_CORNER_NEAR)
        arc = self.layout.marking(MarkingFeature.THREE_POINT_ARC)
        self.assertIsInstance(far, StraightMarking)
        self.assertIsInstance(near, StraightMarking)
        self.assertIsInstance(arc, ArcMarking)
        self.assertLess(math.dist(far.end, arc.sample(2)[0]), 1e-12)
        self.assertLess(math.dist(near.end, arc.sample(2)[-1]), 1e-12)
        self.assertAlmostEqual(max(x for x, _ in arc.sample(81)), 29.0, places=12)


class BuilderTests(unittest.TestCase):
    def setUp(self):
        self.layout = build_layout(**CENTERLINE)

    def test_centerline_inputs_remain_exact(self):
        self.assertEqual(self.layout.half_length, 47.0)
        self.assertEqual(self.layout.coordinate(HalfCourtLandmark.LANE_BASELINE_FAR), (0.0, 17.0))
        self.assertEqual(self.layout.coordinate(HalfCourtLandmark.FREE_THROW_CIRCLE_APEX), (25.0, 25.0))

    def test_every_feature_has_one_primitive(self):
        self.assertEqual(set(self.layout.markings), set(MarkingFeature))

    def test_landmarks_are_primitive_junctions(self):
        far = self.layout.marking(MarkingFeature.LANE_EDGE_FAR)
        self.assertEqual(far.start, self.layout.coordinate(HalfCourtLandmark.LANE_BASELINE_FAR))
        self.assertEqual(far.end, self.layout.coordinate(HalfCourtLandmark.LANE_FREE_THROW_FAR))

    def test_center_circle_is_the_only_closed_primitive(self):
        closed = [feature for feature, primitive in self.layout.markings.items() if primitive.closed]
        self.assertEqual(closed, [MarkingFeature.CENTER_CIRCLE])

    def test_sample_marking_is_deterministic(self):
        first = self.layout.sample_marking(MarkingFeature.THREE_POINT_ARC)
        self.assertEqual(first, self.layout.sample_marking(MarkingFeature.THREE_POINT_ARC))
        self.assertEqual(len(first), 81)

    def test_non_feet_and_out_of_bounds_landmark_rejected(self):
        with self.assertRaises(ValueError):
            dataclasses.replace(self.layout, units="meters")
        with self.assertRaises(ValueError):
            dataclasses.replace(
                self.layout,
                landmarks={**self.layout.landmarks, HalfCourtLandmark.BASELINE_SIDELINE_FAR: (999, 0)},
            )


class EndAgnosticismTests(unittest.TestCase):
    def test_landmark_and_feature_names_encode_no_absolute_end(self):
        for item in (*HalfCourtLandmark, *MarkingFeature):
            for forbidden in ("north", "south", "east", "west"):
                self.assertNotIn(forbidden, item.value)

    def test_sideline_mirrors_are_reciprocal(self):
        for landmark in HalfCourtLandmark:
            mirror = landmark.sideline_mirror
            if mirror is not None:
                self.assertEqual(mirror.sideline_mirror, landmark)


class ValidatorTests(unittest.TestCase):
    def setUp(self):
        self.layout = build_layout(**CENTERLINE)

    def test_healthy_layout_has_no_problems(self):
        self.assertEqual(validate_layout(self.layout), [])

    def test_detects_asymmetric_landmark(self):
        broken = dataclasses.replace(
            self.layout,
            landmarks={**self.layout.landmarks, HalfCourtLandmark.LANE_BASELINE_NEAR: (0, 40)},
        )
        self.assertTrue(any("symmetric" in problem for problem in validate_layout(broken)))

    def test_detects_marking_landmark_disagreement(self):
        broken = dataclasses.replace(
            self.layout,
            landmarks={**self.layout.landmarks, HalfCourtLandmark.LANE_BASELINE_NEAR: (0, 32)},
        )
        self.assertTrue(any("topology" in problem for problem in validate_layout(broken)))

    def test_builder_rejects_impossible_geometry(self):
        with self.assertRaises(ValueError):
            build_layout(**{**CENTERLINE, "free_throw_distance": 50.0})
        with self.assertRaises(ValueError):
            build_layout(**{**CENTERLINE, "lane_width": 60.0})
        with self.assertRaises(ValueError):
            build_layout(**{**CENTERLINE, "three_point_radius": 10.0})


class HashTests(unittest.TestCase):
    def setUp(self):
        self.layout = build_layout(**CENTERLINE)

    def test_same_geometry_same_hash(self):
        self.assertEqual(self.layout.content_hash(), build_layout(**CENTERLINE).content_hash())

    def test_every_calibration_geometry_category_changes_hash(self):
        for field, value in (
            ("three_point_radius", 23.7),
            ("line_width", 0.2),
            ("lane_width", 15.9),
        ):
            with self.subTest(field=field):
                other = build_layout(**{**CENTERLINE, field: value})
                self.assertNotEqual(self.layout.content_hash(), other.content_hash())

    def test_derived_primitive_change_changes_hash(self):
        markings = dict(self.layout.markings)
        baseline = markings[MarkingFeature.BASELINE]
        self.assertIsInstance(baseline, StraightMarking)
        markings[MarkingFeature.BASELINE] = StraightMarking(
            MarkingFeature.BASELINE, baseline.start, (0.0, 49.0), baseline.width_ft
        )
        changed = dataclasses.replace(self.layout, markings=markings)
        self.assertNotEqual(self.layout.content_hash(), changed.content_hash())

    def test_source_edge_convention_changes_hash_even_if_offset_is_same(self):
        specs = dict(self.layout.source_measurements)
        original = specs["three_point_radius"]
        specs["three_point_radius"] = SourceMeasurement(
            original.value,
            original.unit,
            MeasurementReference.OUTSIDE_EDGE,
            original.centerline_offset_half_widths,
            original.diagram_label,
        )
        changed = build_layout(
            layout_id="test_nba", rule_set="test", source="test", source_measurements=specs
        )
        self.assertNotEqual(self.layout.content_hash(), changed.content_hash())

    def test_serialization_contains_source_and_derived_geometry(self):
        data = self.layout.as_dict()
        self.assertIn("source_measurements", data)
        self.assertIn("markings", data)
        self.assertEqual(data["line_convention"], LINE_CONVENTION)
        self.assertEqual(data["content_hash"], self.layout.content_hash())


class ProfileTests(unittest.TestCase):
    def test_registered_nba_profile_loads_and_validates(self):
        layout = load_registered_layout("nba_halfcourt")
        self.assertEqual(layout.rule_set, "NBA 2025-26")
        self.assertEqual(validate_layout(layout), [])
        self.assertIn("Official 2025-26", layout.source)

    def test_nba_profile_is_registered(self):
        self.assertIn("nba_halfcourt", available_layouts())

    def test_unknown_layout_raises_helpfully(self):
        with self.assertRaises(FileNotFoundError) as caught:
            load_registered_layout("nba_full_court")
        self.assertIn("available", str(caught.exception))

    def test_v1_profile_is_rejected_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v1.json"
            path.write_text(json.dumps({"schema_version": "half-court-layout-1.0.0"}))
            with self.assertRaises(ValueError) as caught:
                load_layout(path)
        self.assertIn("must be migrated", str(caught.exception))

    def test_profile_declares_current_schema(self):
        path = Path(__file__).resolve().parents[1] / "layouts" / "nba_halfcourt.json"
        self.assertEqual(json.loads(path.read_text())["schema_version"], LAYOUT_SCHEMA_VERSION)


if __name__ == "__main__":
    unittest.main()
