import json
import math
import unittest

from contracts.markings import (
    ArcMarking,
    MarkingFamily,
    MarkingFeature,
    StraightMarking,
    marking_from_dict,
)


class VocabularyTests(unittest.TestCase):
    def test_every_feature_has_a_family_and_kind(self):
        for feature in MarkingFeature:
            self.assertIsInstance(feature.family, MarkingFamily)
            self.assertIn(feature.kind, ("polyline", "arc"))

    def test_vocabulary_remains_end_agnostic(self):
        for feature in MarkingFeature:
            for forbidden in ("north", "south", "east", "west"):
                self.assertNotIn(forbidden, feature.value)


class StraightTests(unittest.TestCase):
    def test_sampling_is_endpoint_inclusive_and_deterministic(self):
        line = StraightMarking(MarkingFeature.BASELINE, (0, 0), (2, 4), 1 / 6)
        self.assertEqual(line.sample(3), ((0.0, 0.0), (1.0, 2.0), (2.0, 4.0)))
        self.assertEqual(line.sample(3), line.sample(3))

    def test_rejects_curved_feature_and_degenerate_line(self):
        with self.assertRaises(ValueError):
            StraightMarking(MarkingFeature.CENTER_CIRCLE, (0, 0), (1, 1), 1 / 6)
        with self.assertRaises(ValueError):
            StraightMarking(MarkingFeature.BASELINE, (0, 0), (0, 0), 1 / 6)

    def test_json_round_trip(self):
        item = StraightMarking(MarkingFeature.FREE_THROW_LINE, (1, 2), (3, 4), 1 / 6)
        restored = marking_from_dict(json.loads(json.dumps(item.as_dict())))
        self.assertEqual(restored, item)


class ArcTests(unittest.TestCase):
    def test_samples_lie_on_radius_and_include_endpoints(self):
        arc = ArcMarking(
            MarkingFeature.THREE_POINT_ARC, (5, 25), 23, -math.pi / 2, math.pi / 2, 1 / 6
        )
        points = arc.sample(9)
        for point in points:
            self.assertAlmostEqual(math.dist(point, arc.center), 23.0, places=12)
        self.assertAlmostEqual(points[0][0], 5.0)
        self.assertAlmostEqual(points[-1][0], 5.0)

    def test_full_circle_must_be_closed_and_partial_arc_open(self):
        with self.assertRaises(ValueError):
            ArcMarking(MarkingFeature.CENTER_CIRCLE, (0, 0), 6, 0, 2 * math.pi, 1 / 6)
        with self.assertRaises(ValueError):
            ArcMarking(
                MarkingFeature.THREE_POINT_ARC, (0, 0), 6, 0, math.pi, 1 / 6, closed=True
            )

    def test_json_round_trip(self):
        item = ArcMarking(
            MarkingFeature.CENTER_CIRCLE, (47, 25), 6, 0, 2 * math.pi, 1 / 6, closed=True
        )
        restored = marking_from_dict(json.loads(json.dumps(item.as_dict())))
        self.assertEqual(restored, item)

    def test_bad_family_in_json_is_rejected(self):
        item = ArcMarking(
            MarkingFeature.CENTER_CIRCLE, (47, 25), 6, 0, 2 * math.pi, 1 / 6, closed=True
        ).as_dict()
        item["family"] = "lane"
        with self.assertRaises(ValueError):
            marking_from_dict(item)


if __name__ == "__main__":
    unittest.main()
