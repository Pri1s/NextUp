import copy
import math
import unittest

from segment_merge import merge_collinear_segments, observed_segment_length


def line(identifier, p1, p2, *, strength=1.0, floor_confidence=0.5):
    return {
        "id": identifier,
        "type": "line_segment",
        "geometry": {
            "p1": list(p1),
            "p2": list(p2),
            "length_px": math.dist(p1, p2),
        },
        "strength": strength,
        "evidence": {"floor_roi_confidence": floor_confidence},
    }


class SegmentMergeTests(unittest.TestCase):
    width = 2478
    height = 1382

    def merge(self, items, **options):
        return merge_collinear_segments(
            items,
            image_width=self.width,
            image_height=self.height,
            **options,
        )

    def test_base_pass_bridges_at_most_72px_and_preserves_fragments(self):
        items = [
            line("left", [0, 20], [100, 20], strength=.9, floor_confidence=.2),
            line("right", [140, 20], [240, 20], strength=.4, floor_confidence=.9),
        ]
        original = copy.deepcopy(items)

        merged, diagnostics = self.merge(items)

        self.assertEqual(items, original)
        self.assertEqual(len(merged), 1)
        self.assertEqual(diagnostics["base_merge_count"], 1)
        self.assertEqual(diagnostics["extended_merge_count"], 0)
        self.assertEqual(merged[0]["geometry"]["length_px"], 240.0)
        self.assertEqual(observed_segment_length(merged[0]), 200.0)
        provenance = merged[0]["evidence"]["merge_provenance"]
        self.assertEqual(provenance["fragment_lengths_px"], [100.0, 100.0])
        self.assertEqual(provenance["gaps_px"], [40.0])
        self.assertAlmostEqual(provenance["observed_coverage"], 5 / 6, places=4)
        self.assertEqual(merged[0]["evidence"]["source_fragment_lengths_px"], [100.0, 100.0])
        self.assertEqual(merged[0]["evidence"]["observed_length_px"], 200.0)
        self.assertAlmostEqual(merged[0]["evidence"]["observed_coverage_ratio"], 5 / 6, places=4)
        self.assertEqual(merged[0]["evidence"]["merge_gaps_px"], [40.0])
        # Appearance fields are copied from one representative, never combined
        # with the old max-floor-confidence behavior.
        self.assertEqual(merged[0]["evidence"]["floor_roi_confidence"], .2)
        self.assertTrue(provenance["appearance_recompute_required"])

    def test_extended_pass_recovers_known_occluded_court_line(self):
        # These are the two post-base fragments from 001_video_4_f000163.
        items = [
            line(
                "left_lane_fragment",
                [512.6586488837344, 903.7959524499855],
                [1032.785717868342, 959.0149221024609],
            ),
            line(
                "right_lane_fragment",
                [1147.0906576308691, 969.1936242306895],
                [1601.4184100690245, 1020.2718922562808],
            ),
        ]

        merged, diagnostics = self.merge(items)

        self.assertEqual(len(merged), 1)
        self.assertEqual(diagnostics["thresholds"]["extended_max_gap_px"], 123.9)
        self.assertEqual(diagnostics["base_merge_count"], 0)
        self.assertEqual(diagnostics["extended_merge_count"], 1)
        self.assertAlmostEqual(merged[0]["geometry"]["length_px"], 1094.97, places=1)
        self.assertAlmostEqual(observed_segment_length(merged[0]), 980.23, places=1)
        provenance = merged[0]["evidence"]["merge_provenance"]
        self.assertAlmostEqual(provenance["largest_gap_px"], 114.74, places=1)
        self.assertAlmostEqual(provenance["observed_coverage"], .8952, places=3)
        self.assertEqual(provenance["extended_merge_count"], 1)

    def test_extended_pass_revisits_clusters_left_by_greedy_base_pass(self):
        # B initially misses A's 8 px lateral tolerance.  A later short
        # fragment legitimately joins A and moves its TLS fit into agreement
        # with B.  The agglomerative pass must revisit that pair even though
        # its longitudinal gap is below the base 72 px cap.
        items = [
            line("a", [0, 0], [200, 0]),
            line("b", [240, 8.5], [390, 8.5]),
            line("a_second_edge", [0, 8], [100, 8]),
        ]

        merged, diagnostics = self.merge(items)

        self.assertEqual(len(merged), 1)
        self.assertEqual(diagnostics["base_merge_count"], 1)
        self.assertEqual(diagnostics["extended_merge_count"], 1)

    def test_extended_pass_rejects_each_geometric_failure(self):
        cases = {
            # 125 exceeds this image's 123.9 px extended cap.
            "absolute_gap": [line("a", [0, 0], [500, 0]), line("b", [625, 0], [1125, 0])],
            # 100 / 300 exceeds the .30 gap-to-shorter-span ratio.
            "relative_gap": [line("a", [0, 0], [300, 0]), line("b", [400, 0], [700, 0])],
            "lateral_offset": [line("a", [0, 0], [500, 0]), line("b", [600, 9], [1100, 9])],
            "angle": [
                line("a", [0, 0], [500, 0]),
                line("b", [600, 0], [1100, 500 * math.tan(math.radians(3))]),
            ],
        }
        for name, items in cases.items():
            with self.subTest(name=name):
                merged, diagnostics = self.merge(items)
                self.assertEqual(len(merged), 2)
                self.assertEqual(diagnostics["extended_merge_count"], 0)

    def test_extended_pass_rejects_a_low_coverage_input_cluster(self):
        # The first two fragments pass the 72 px base merge, but their resulting
        # cluster has only 200 / 272 observed coverage and cannot be extended.
        items = [
            line("a1", [0, 0], [100, 0]),
            line("a2", [172, 0], [272, 0]),
            line("b", [372, 0], [672, 0]),
        ]

        merged, diagnostics = self.merge(items)

        self.assertEqual(len(merged), 2)
        self.assertEqual(diagnostics["base_merge_count"], 1)
        self.assertEqual(diagnostics["extended_merge_count"], 0)
        coverages = sorted(
            item["evidence"]["merge_provenance"]["observed_coverage"]
            for item in merged
        )
        self.assertAlmostEqual(coverages[0], 200 / 272, places=4)

    def test_result_is_order_and_endpoint_direction_invariant(self):
        first = line("a", [0, 10], [500, 10])
        second = line("b", [600, 10], [1100, 10])
        forward, _ = self.merge([first, second])
        reversed_input, _ = self.merge(
            [
                line("b", [1100, 10], [600, 10]),
                line("a", [500, 10], [0, 10]),
            ]
        )

        self.assertEqual(forward, reversed_input)

    def test_output_ranking_uses_observed_length_not_bridged_span(self):
        occluded = [
            line("a", [0, 0], [500, 0]),
            line("b", [615, 0], [1072, 0]),
        ]
        continuous = line("continuous", [2000, 0], [2000, 1000])

        merged, _ = self.merge([*occluded, continuous])

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["id"], "continuous")
        self.assertEqual(observed_segment_length(merged[0]), 1000.0)
        self.assertGreater(merged[1]["geometry"]["length_px"], 1000.0)
        self.assertLess(observed_segment_length(merged[1]), 1000.0)

    def test_extended_gap_cap_is_clamped(self):
        _, small = merge_collinear_segments([], image_width=640, image_height=480)
        _, large = merge_collinear_segments([], image_width=4096, image_height=2160)
        self.assertEqual(small["thresholds"]["extended_max_gap_px"], 72.0)
        self.assertEqual(large["thresholds"]["extended_max_gap_px"], 144.0)

    def test_observed_length_helper_supports_legacy_lines(self):
        self.assertEqual(observed_segment_length(line("legacy", [0, 0], [30, 40])), 50.0)


if __name__ == "__main__":
    unittest.main()
