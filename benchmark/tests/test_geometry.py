"""Benchmark geometry must describe the same floor the engine draws.

The benchmark scores a calibration by projecting court markings and measuring the
distance to annotated paint. If its idea of where the three-point line sits differs
from the engine's, every number it produces is an artefact of that disagreement
rather than a measurement of the fit. So the cross-check against
``calibration.viz`` is the load-bearing test here, not a tidiness one.

The depth-span assertions guard the benchmark's whole reason for existing: the
current evidence covers 19 ft, and only markings reaching past that can show
whether a transform describes the floor or merely fits its own points.
"""

import unittest

import numpy as np

from contracts.annotations import HELD_OUT_FAMILIES, MarkingFeature
from contracts.court_layout import build_layout, load_registered_layout

from benchmark.geometry import (
    DEPTH_BANDS,
    depth_band,
    depth_span,
    marking_polylines,
    validate_geometry,
)
from benchmark.glossary import render_markdown, validate_glossary


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.layout = load_registered_layout("nba_halfcourt")
        self.polylines = marking_polylines(self.layout)

    def test_registered_layout_produces_healthy_geometry(self):
        self.assertEqual(validate_geometry(self.layout), [])

    def test_every_annotatable_feature_has_court_geometry(self):
        """A feature an annotator can label but the scorer cannot project is a hole."""
        self.assertEqual(set(self.polylines), set(MarkingFeature))

    def test_three_point_arc_reaches_beyond_the_current_evidence_span(self):
        """§4.1's claim, checked: the arc is why this benchmark can see drift."""
        _, high = depth_span(self.polylines[MarkingFeature.THREE_POINT_ARC])
        self.assertGreater(high, 19.0)
        self.assertAlmostEqual(high, 29.0, places=2)

    def test_free_throw_circle_far_half_uses_centerline_radius(self):
        """The published six-foot outside radius becomes a 5 ft 11 in centerline."""
        _, high = depth_span(self.polylines[MarkingFeature.FREE_THROW_CIRCLE_FAR_HALF])
        self.assertAlmostEqual(high, 24 + 11 / 12, places=6)

    def test_held_out_families_are_all_curved(self):
        """Held-out evidence must be independent of the straight lane geometry."""
        for feature in HELD_OUT_FAMILIES:
            self.assertEqual(feature.kind, "arc", feature.value)

    def test_arc_points_sit_on_their_defining_radius(self):
        basket = np.array(
            [self.layout.basket_from_baseline, self.layout.width / 2.0], dtype=np.float64
        )
        arc = self.polylines[MarkingFeature.THREE_POINT_ARC].points
        radii = np.linalg.norm(arc - basket, axis=1)
        self.assertTrue(np.allclose(radii, self.layout.three_point_radius, atol=1e-9))

    def test_corner_segments_meet_the_arc(self):
        """The corner straight ends exactly where the curve begins."""
        corner = self.polylines[MarkingFeature.THREE_POINT_CORNER_FAR].points
        arc = self.polylines[MarkingFeature.THREE_POINT_ARC].points
        gap = float(np.min(np.linalg.norm(arc - corner[-1], axis=1)))
        self.assertLess(gap, 1e-9)

    def test_lane_edges_are_symmetric_about_the_court_axis(self):
        centre = self.layout.width / 2.0
        far = self.polylines[MarkingFeature.LANE_EDGE_FAR].points
        near = self.polylines[MarkingFeature.LANE_EDGE_NEAR].points
        self.assertTrue(np.allclose(far[:, 1] + near[:, 1], 2 * centre))

    def test_geometry_follows_the_layout_rather_than_hardcoding_nba(self):
        """A narrower lane must move the lane edges, or the scalars are typed in."""
        narrow = build_layout(
            layout_id="narrow",
            rule_set="test",
            source="test",
            half_length=47.0,
            width=50.0,
            lane_width=12.0,
            free_throw_distance=19.0,
            basket_from_baseline=5.25,
            three_point_corner_inset=3.0,
            free_throw_circle_radius=6.0,
            three_point_radius=23.75,
            center_circle_radius=6.0,
            restricted_area_radius=4.0,
            line_width=2.0 / 12.0,
        )
        edge = marking_polylines(narrow)[MarkingFeature.LANE_EDGE_FAR].points
        self.assertTrue(np.allclose(edge[:, 1], 19.0))

    def test_adapter_points_and_width_come_directly_from_layout(self):
        for feature, polyline in self.polylines.items():
            self.assertTrue(np.allclose(polyline.points, self.layout.sample_marking(feature)))
            self.assertEqual(polyline.width_ft, self.layout.marking(feature).width_ft)
            self.assertEqual(polyline.closed, self.layout.marking(feature).closed)


class DepthBandTests(unittest.TestCase):
    def test_bands_partition_the_half_court_without_gaps(self):
        for (_, _, high), (_, low, _) in zip(DEPTH_BANDS, DEPTH_BANDS[1:]):
            self.assertEqual(high, low)

    def test_first_band_matches_the_current_evidence_span(self):
        """0-19 ft is exactly where the confident landmarks live."""
        self.assertEqual(DEPTH_BANDS[0][2], 19.0)

    def test_band_assignment(self):
        self.assertEqual(depth_band(0.0), "0-19ft")
        self.assertEqual(depth_band(18.9), "0-19ft")
        self.assertEqual(depth_band(19.0), "19-30ft")
        self.assertEqual(depth_band(46.0), "30-47ft")

    def test_overshoot_lands_in_the_far_band(self):
        """The centre circle straddles midcourt; it must not fall off the scale."""
        self.assertEqual(depth_band(53.0), "30-47ft")


class GlossaryTests(unittest.TestCase):
    def setUp(self):
        self.layout = load_registered_layout("nba_halfcourt")

    def test_every_feature_is_described(self):
        self.assertEqual(validate_glossary(), [])

    def test_markdown_names_every_feature_and_junction(self):
        rendered = render_markdown(self.layout)
        for feature in MarkingFeature:
            self.assertIn(f"`{feature.value}`", rendered)

    def test_markdown_states_the_line_convention(self):
        """An annotator who misses this produces a consistent half-stripe bias."""
        self.assertIn("painted centerline", render_markdown(self.layout))

    def test_markdown_carries_no_model_derived_content(self):
        """The glossary ships inside the blind frame tree; it must stay blind."""
        rendered = render_markdown(self.layout).lower()
        for leak in ("slot", "keypoint", "homography", "detector", "predict"):
            self.assertNotIn(leak, rendered)

    def test_dimensions_are_derived_from_the_layout(self):
        rendered = render_markdown(self.layout)
        self.assertIn("29.0 ft from the baseline", rendered)


if __name__ == "__main__":
    unittest.main()
