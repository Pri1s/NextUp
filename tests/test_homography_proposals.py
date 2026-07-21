import math
import unittest

from homography_proposals import (
    derive_template_orientation_families,
    discover_u_motifs,
    generate_guided_proposals,
)


def template_line(identifier, p1, p2):
    return {"id": identifier, "p1": list(p1), "p2": list(p2)}


def detected_line(identifier, p1, p2, strength=1.0):
    length = math.dist(p1, p2)
    return {
        "id": identifier,
        "type": "line_segment",
        "geometry": {
            "p1": list(p1),
            "p2": list(p2),
            "length_px": length,
            "angle_rad": math.atan2(p2[1] - p1[1], p2[0] - p1[0]),
        },
        "strength": strength,
    }


def ray(identifier, start, angle_degrees, length=100.0, strength=1.0):
    angle = math.radians(angle_degrees)
    end = (start[0] + length * math.cos(angle), start[1] + length * math.sin(angle))
    return detected_line(identifier, start, end, strength)


def intersection(identifier, first, second, point, strength=1.0):
    return {
        "id": identifier,
        "type": "intersection",
        "geometry": {"point": list(point), "line_ids": [first, second]},
        "strength": strength,
    }


def rectangle_template():
    return {
        "lines": [
            template_line("t_cap", (0, 0), (10, 0)),
            template_line("t_far_cap", (0, 8), (10, 8)),
            template_line("t_leg_left", (0, 0), (0, 8)),
            template_line("t_leg_right", (10, 0), (10, 8)),
        ]
    }


def detection(primitives):
    return {"image_size": {"width": 300, "height": 180}, "primitives": primitives}


def projective_u(prefix="d", offset=(20.0, 80.0), strength=1.0):
    cap = ray(f"{prefix}_cap", offset, -27, 100, strength)
    cap_p1 = tuple(cap["geometry"]["p1"])
    cap_p2 = tuple(cap["geometry"]["p2"])
    leg_left = ray(f"{prefix}_leg_left", cap_p1, 5, 90, strength)
    leg_right = ray(f"{prefix}_leg_right", cap_p2, 7, 95, strength)
    far_cap = ray(f"{prefix}_far_cap", (offset[0] + 15, offset[1] + 45), -44, 80, strength)
    points = [
        intersection(f"{prefix}_corner_left", cap["id"], leg_left["id"], cap_p1, strength),
        intersection(f"{prefix}_corner_right", cap["id"], leg_right["id"], cap_p2, strength),
    ]
    return [cap, far_cap, leg_left, leg_right, *points]


class TemplateFamilyTests(unittest.TestCase):
    def test_derives_two_families_from_geometry_not_names(self):
        families = derive_template_orientation_families(rectangle_template())
        mapping = families["line_to_family"]
        self.assertEqual(mapping["t_cap"], mapping["t_far_cap"])
        self.assertEqual(mapping["t_leg_left"], mapping["t_leg_right"])
        self.assertNotEqual(mapping["t_cap"], mapping["t_leg_left"])


class MotifProposalTests(unittest.TestCase):
    def test_projective_local_buckets_and_both_leg_assignments(self):
        primitives = projective_u()
        motifs = discover_u_motifs(detection(primitives))
        cap_motifs = [motif for motif in motifs if motif["cap_id"] == "d_cap"]
        self.assertEqual(len(cap_motifs), 1)
        self.assertEqual(set(cap_motifs[0]["leg_ids"]), {"d_leg_left", "d_leg_right"})

        proposals, diagnostics = generate_guided_proposals(
            detection(primitives), rectangle_template(), max_proposals=100, random_seed=17
        )
        motif_proposals = [item for item in proposals if item["proposal_type"] == "u_motif"]
        self.assertTrue(motif_proposals)
        self.assertGreater(diagnostics["anchor_counts"]["detected_u_motifs"], 0)

        # The detected cap family fans from -27 to -44 degrees while the legs
        # sit at +5/+7 degrees. It must remain two whole local buckets.
        detected_cap_family = {"d_cap", "d_far_cap"}
        detected_leg_family = {"d_leg_left", "d_leg_right"}
        template_families = derive_template_orientation_families(rectangle_template())["line_to_family"]
        for proposal in motif_proposals:
            grouped = {0: set(), 1: set()}
            for template_id, detected_id in proposal["seed_pairs"].items():
                grouped[template_families[template_id]].add(detected_id)
            self.assertIn(grouped[0], (detected_cap_family, detected_leg_family))
            self.assertIn(grouped[1], (detected_cap_family, detected_leg_family))
            self.assertNotEqual(grouped[0], grouped[1])

        fixed_caps = [
            proposal["seed_pairs"]
            for proposal in motif_proposals
            if proposal["seed_pairs"].get("t_cap") == "d_cap"
            and proposal["seed_pairs"].get("t_far_cap") == "d_far_cap"
        ]
        leg_assignments = {
            (mapping["t_leg_left"], mapping["t_leg_right"]) for mapping in fixed_caps
        }
        self.assertIn(("d_leg_left", "d_leg_right"), leg_assignments)
        self.assertIn(("d_leg_right", "d_leg_left"), leg_assignments)

    def test_round_robin_gives_each_equal_rank_motif_a_budget_slot(self):
        first = projective_u("a", (20.0, 70.0))
        second = projective_u("b", (170.0, 120.0))
        proposals, diagnostics = generate_guided_proposals(
            detection(first + second), rectangle_template(), max_proposals=8, random_seed=5
        )
        self.assertGreaterEqual(diagnostics["proposal_types"]["u_motif"], 2)
        anchors = {
            proposal["source"]["detected_anchor"]
            for proposal in proposals
            if proposal["proposal_type"] == "u_motif"
        }
        self.assertEqual(len(anchors), 2)

    def test_connected_fourth_line_outranks_an_unrelated_stronger_parallel(self):
        lines = [
            detected_line("d_top", (20, 20), (120, 20), strength=.2),
            detected_line("d_bottom", (20, 100), (120, 100), strength=.2),
            detected_line("d_left", (20, 20), (20, 100), strength=.2),
            detected_line("d_right", (120, 20), (120, 100), strength=.2),
            detected_line("d_distractor", (0, 150), (260, 150), strength=1.0),
        ]
        corners = [
            intersection("c_tl", "d_top", "d_left", (20, 20), strength=.2),
            intersection("c_tr", "d_top", "d_right", (120, 20), strength=.2),
            intersection("c_bl", "d_bottom", "d_left", (20, 100), strength=.2),
            intersection("c_br", "d_bottom", "d_right", (120, 100), strength=.2),
        ]
        proposals, _ = generate_guided_proposals(
            detection(lines + corners), rectangle_template(), max_proposals=4, random_seed=3
        )
        self.assertEqual(len(proposals), 4)
        motif_fourths = {
            item["source"]["detected_fourth"]
            for item in proposals
            if item["proposal_type"] == "u_motif"
        }
        self.assertTrue(motif_fourths)
        self.assertNotIn("d_distractor", motif_fourths)

    def test_duplicate_anchor_cap_does_not_outrank_opposite_closure(self):
        lines = [
            detected_line("d_top", (20, 20), (120, 20), strength=.3),
            detected_line("d_top_duplicate", (20, 21), (120, 21), strength=1.0),
            detected_line("d_bottom", (20, 100), (120, 100), strength=.2),
            detected_line("d_left", (20, 20), (20, 100), strength=.3),
            detected_line("d_right", (120, 20), (120, 100), strength=.3),
        ]
        corners = [
            intersection("c_tl", "d_top", "d_left", (20, 20)),
            intersection("c_tr", "d_top", "d_right", (120, 20)),
            intersection("c_tl_duplicate", "d_top_duplicate", "d_left", (20, 21)),
            intersection("c_tr_duplicate", "d_top_duplicate", "d_right", (120, 21)),
            intersection("c_bl", "d_bottom", "d_left", (20, 100)),
            intersection("c_br", "d_bottom", "d_right", (120, 100)),
        ]

        proposals, _ = generate_guided_proposals(
            detection(lines + corners), rectangle_template(), max_proposals=100, random_seed=3
        )
        anchor = "u:d_top:d_left:d_right"
        first = next(item for item in proposals if item["source"].get("detected_anchor") == anchor)

        self.assertEqual(first["source"]["detected_fourth"], "d_bottom")

    def test_observed_paint_length_does_not_shrink_endpoint_adjacency(self):
        cap = detected_line("cap", (0, 0), (1000, 0))
        cap["geometry"]["observed_length_px"] = 100.0
        left = detected_line("left", (80, 0), (80, 200))
        right = detected_line("right", (920, 0), (920, 200))
        primitives = [
            cap,
            left,
            right,
            intersection("left_corner", "cap", "left", (80, 0)),
            intersection("right_corner", "cap", "right", (920, 0)),
        ]
        motifs = discover_u_motifs(detection(primitives))
        self.assertEqual([motif["cap_id"] for motif in motifs], ["cap"])

    def test_canonical_correspondences_are_deduplicated_across_motif_sources(self):
        lines = [
            detected_line("d_top", (20, 20), (120, 20)),
            detected_line("d_bottom", (20, 100), (120, 100)),
            detected_line("d_left", (20, 20), (20, 100)),
            detected_line("d_right", (120, 20), (120, 100)),
        ]
        corners = [
            intersection("c_tl", "d_top", "d_left", (20, 20)),
            intersection("c_tr", "d_top", "d_right", (120, 20)),
            intersection("c_bl", "d_bottom", "d_left", (20, 100)),
            intersection("c_br", "d_bottom", "d_right", (120, 100)),
        ]
        proposals, diagnostics = generate_guided_proposals(
            detection(lines + corners), rectangle_template(), max_proposals=100, random_seed=3
        )
        keys = [tuple(sorted(proposal["seed_pairs"].items())) for proposal in proposals]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertGreater(diagnostics["duplicates"], 0)


class FallbackProposalTests(unittest.TestCase):
    def test_shared_leg_two_corner_fallback_uses_local_projective_buckets(self):
        # The repeated leg family is tight (+5/+7), while the two caps fan from
        # -27 to -44 degrees.  This is not a strict U motif, but it is a valid
        # connected two-corner fallback under the 25-degree local buckets.
        shared_leg = ray("leg_near", (20, 80), 5)
        parallel_leg = ray("leg_far", (20, 145), 7)
        cap_near = ray("cap_near", (20, 80), -27)
        cap_far = ray("cap_far", tuple(shared_leg["geometry"]["p2"]), -44)
        corners = [
            intersection("corner_near", "leg_near", "cap_near", (20, 80)),
            intersection(
                "corner_far",
                "leg_near",
                "cap_far",
                tuple(shared_leg["geometry"]["p2"]),
            ),
        ]

        proposals, diagnostics = generate_guided_proposals(
            detection([shared_leg, parallel_leg, cap_near, cap_far, *corners]),
            rectangle_template(),
            max_proposals=100,
            random_seed=19,
        )

        connected = [item for item in proposals if item["proposal_type"] == "two_corner"]
        self.assertTrue(connected)
        self.assertGreater(diagnostics["anchor_counts"]["detected_two_corner_quartets"], 0)
        self.assertTrue(
            any(set(item["seed_pairs"].values()) == {"leg_near", "leg_far", "cap_near", "cap_far"}
                for item in connected)
        )

    def test_two_corner_fallback_precedes_weaker_fallbacks(self):
        lines = [
            ray("a0", (10, 10), 5),
            ray("b0", (10, 10), -40),
            ray("a1", (130, 90), 7),
            ray("b1", (130, 90), -42),
        ]
        corners = [
            intersection("corner_0", "a0", "b0", (10, 10)),
            intersection("corner_1", "a1", "b1", (130, 90)),
        ]
        proposals, _ = generate_guided_proposals(
            detection(lines + corners), rectangle_template(), max_proposals=8, random_seed=2
        )
        self.assertEqual(len(proposals), 8)
        types = [item["proposal_type"] for item in proposals]
        self.assertEqual(types[0], "two_corner")
        self.assertGreater(types.count("two_corner"), types.count("orientation_only"))

    def test_one_corner_fallback_precedes_orientation_only(self):
        lines = [
            ray("a0", (10, 10), 5),
            ray("b0", (10, 10), -40),
            ray("a1", (130, 90), 7),
            ray("b1", (130, 90), -42),
        ]
        corner = intersection("corner_0", "a0", "b0", (10, 10))
        proposals, _ = generate_guided_proposals(
            detection(lines + [corner]), rectangle_template(), max_proposals=8, random_seed=2
        )
        self.assertEqual(len(proposals), 8)
        types = [item["proposal_type"] for item in proposals]
        self.assertEqual(types[0], "one_corner")
        self.assertGreater(types.count("one_corner"), types.count("orientation_only"))

    def test_false_u_cannot_starve_a_valid_two_corner_fallback(self):
        false_u = projective_u("hud", (20.0, 70.0))
        court_lines = [
            ray("court_a0", (10, 10), 5),
            ray("court_b0", (10, 10), -40),
            ray("court_a1", (180, 90), 7),
            ray("court_b1", (180, 90), -42),
        ]
        court_corners = [
            intersection("court_corner_0", "court_a0", "court_b0", (10, 10)),
            intersection("court_corner_1", "court_a1", "court_b1", (180, 90)),
        ]
        proposals, diagnostics = generate_guided_proposals(
            detection(false_u + court_lines + court_corners),
            rectangle_template(),
            max_proposals=8,
            random_seed=13,
        )
        self.assertEqual(len(proposals), 8)
        self.assertGreater(diagnostics["proposal_types"]["u_motif"], 0)
        self.assertGreater(diagnostics["proposal_types"]["two_corner"], 0)

    def test_malformed_lines_are_counted_once_and_do_not_crash(self):
        valid = [
            ray("a0", (0, 0), 5),
            ray("a1", (0, 40), 7),
            ray("b0", (0, 80), -40),
            ray("b1", (0, 120), -42),
        ]
        malformed = {
            "type": "line_segment",
            "geometry": {"p1": [0, 0], "p2": [float("nan"), 1]},
            "strength": 1.0,
        }
        bad_intersection = intersection("bad", "a0", "missing", (0, 0))
        proposals, diagnostics = generate_guided_proposals(
            detection(valid + [malformed, bad_intersection]),
            rectangle_template(),
            max_proposals=8,
        )
        self.assertTrue(proposals)
        self.assertEqual(diagnostics["invalid_candidates"], 2)
        self.assertEqual(diagnostics["anchor_counts"]["detected_lines_input"], 5)
        self.assertEqual(diagnostics["anchor_counts"]["detected_lines"], 4)

    def test_no_intersection_fallback_is_orientation_guided(self):
        primitives = [
            ray("d_cap", (20, 80), -27),
            ray("d_far_cap", (45, 145), -44),
            ray("d_leg_left", (15, 15), 5),
            ray("d_leg_right", (145, 30), 7),
        ]
        proposals, diagnostics = generate_guided_proposals(
            detection(primitives), rectangle_template(), max_proposals=100, random_seed=11
        )
        self.assertTrue(proposals)
        self.assertTrue(all(item["proposal_type"] == "orientation_only" for item in proposals))
        self.assertEqual(diagnostics["anchor_counts"]["detected_intersections"], 0)

        detected_families = [
            {"d_cap", "d_far_cap"},
            {"d_leg_left", "d_leg_right"},
        ]
        template_families = derive_template_orientation_families(rectangle_template())["line_to_family"]
        for proposal in proposals:
            mapped = {0: set(), 1: set()}
            for template_id, detected_id in proposal["seed_pairs"].items():
                mapped[template_families[template_id]].add(detected_id)
            self.assertIn(mapped[0], detected_families)
            self.assertIn(mapped[1], detected_families)
            self.assertNotEqual(mapped[0], mapped[1])


if __name__ == "__main__":
    unittest.main()
