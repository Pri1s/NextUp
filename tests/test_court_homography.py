import unittest

import cv2
import numpy as np

from court_homography import (
    _apply_floor_hud_evidence,
    _court_floor_confidence,
    _segment_agreement,
    _unique_line_matches,
    detect_primitives,
    load_template,
    solve_homography,
    verify_solution,
)
from hud_detection import HudDetection


def line(identifier, p1, p2, strength=1.0):
    return {
        "id": identifier,
        "type": "line_segment",
        "geometry": {"p1": list(p1), "p2": list(p2), "length_px": float(np.linalg.norm(np.subtract(p2, p1)))},
        "strength": strength,
    }


class SegmentAgreementTests(unittest.TestCase):
    homography = np.eye(3)
    template_line = {"id": "baseline", "p1": [10, 30], "p2": [110, 30]}

    def test_requires_visible_extent_not_a_midpoint(self):
        short_fragment = line("fragment", [38, 30], [83, 30])
        self.assertIsNone(_segment_agreement(self.template_line, short_fragment, self.homography, 160, 100, 10))

    def test_accepts_a_full_segment_with_small_pixel_error(self):
        observed = line("court_mark", [12, 33], [108, 32])
        agreement = _segment_agreement(self.template_line, observed, self.homography, 160, 100, 10)
        self.assertIsNotNone(agreement)
        self.assertGreater(agreement["overlap_ratio"], .8)

    def test_one_detected_segment_cannot_match_two_template_lines(self):
        template = {"lines": [
            {"id": "a", "p1": [10, 30], "p2": [110, 30]},
            {"id": "b", "p1": [10, 30], "p2": [110, 30]},
        ]}
        matches = _unique_line_matches(self.homography, template, [line("shared", [10, 30], [110, 30])], {}, 160, 100, 10)
        self.assertEqual(len(matches), 1)


class FloorEvidenceTests(unittest.TestCase):
    def test_image_height_does_not_increase_floor_confidence(self):
        bgr = np.full((120, 160, 3), (180, 180, 180), np.uint8)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        top = _court_floor_confidence(hsv, (10, 20), (150, 20))
        bottom = _court_floor_confidence(hsv, (10, 100), (150, 100))

        self.assertEqual(top, bottom)

    def test_hud_overlap_downranks_but_retains_a_primitive(self):
        bgr = np.full((100, 100, 3), (200, 200, 200), np.uint8)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = np.zeros((100, 100), np.uint8)
        labels = np.zeros((100, 100), np.int32)
        mask[:, 40:60] = 255
        labels[:, 40:60] = 1
        hud = HudDetection(mask, labels, {"regions": [{"id": "hud_000"}]})
        primitive = line("hud_line", [40, 50], [59, 50], strength=1.0)
        primitive["evidence"] = {"raw_strength": 1.0}

        result = _apply_floor_hud_evidence(primitive, hsv, hud)

        self.assertEqual(result["evidence"]["hud_region_ids"], ["hud_000"])
        self.assertGreater(result["strength"], 0.0)
        self.assertLess(result["strength"], .2)

    def test_merged_span_resamples_hud_overlap_instead_of_inheriting_fragment_evidence(self):
        bgr = np.full((100, 120, 3), (200, 200, 200), np.uint8)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = np.zeros((100, 120), np.uint8)
        labels = np.zeros((100, 120), np.int32)
        mask[:, 60:120] = 255
        labels[:, 60:120] = 1
        hud = HudDetection(mask, labels, {"regions": [{"id": "hud_000"}]})
        merged = line("merged", [0, 50], [119, 50], strength=1.0)
        merged["evidence"] = {
            "raw_strength": 1.0,
            # A pre-merge fragment outside the graphic must not donate this
            # stale zero to the full fitted span.
            "hud_overlap_ratio": 0.0,
            "merge_provenance": {"appearance_recompute_required": True},
        }

        result = _apply_floor_hud_evidence(merged, hsv, hud)

        self.assertGreaterEqual(result["evidence"]["hud_overlap_ratio"], .49)
        self.assertEqual(result["evidence"]["hud_region_ids"], ["hud_000"])
        self.assertLess(result["strength"], .60)


class SolverDistractorTests(unittest.TestCase):
    def test_saturated_hud_grid_does_not_pass_the_unchanged_verification_gate(self):
        image = np.full((240, 320, 3), (165, 185, 205), np.uint8)
        cv2.rectangle(image, (110, 120), (210, 185), (150, 35, 150), -1)
        cv2.rectangle(image, (80, 216), (240, 239), (100, 20, 100), -1)
        for x in range(84, 238, 12):
            colour = (0, 230, 255) if (x // 12) % 2 else (255, 255, 255)
            cv2.rectangle(image, (x, 219), (x + 5, 235), colour, -1)
        cv2.putText(
            image,
            "12:34",
            (118, 234),
            cv2.FONT_HERSHEY_SIMPLEX,
            .5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        detection = detect_primitives(image, min_line_length=15)
        hud_lines = [
            item for item in detection["primitives"]
            if item["type"] == "line_segment"
            and item["evidence"]["hud_overlap_ratio"] >= .90
        ]
        solution = solve_homography(detection, load_template(), hypotheses=1000)
        verification = verify_solution(solution, detection, load_template())

        self.assertTrue(hud_lines)
        self.assertEqual(verification["status"], "fail")
        self.assertIsNone(solution.get("review_candidate"))


class ReviewCandidateTests(unittest.TestCase):
    """Review-only fallback: never loosens automatic acceptance, but retains a
    strictly-relaxed candidate (see court_homography._review_candidate_evaluation)
    for cases the automatic gate correctly rejects."""

    def test_three_disconnected_lines_produce_no_review_candidate(self):
        # Three lines with no shared endpoint and no consistent parallel/right
        # angle relationship: fails the review-only structural requirements
        # (connected pair, inverse-parallel, right-angle) exactly as it would
        # fail the automatic ones. Fewer than 4 lines short-circuits before
        # any proposal is generated at all.
        detection = {
            "image_size": {"width": 400, "height": 300},
            "primitives": [
                line("a", [10, 10], [10, 90], strength=1.0),
                line("b", [200, 200], [280, 205], strength=1.0),
                line("c", [50, 250], [55, 170], strength=1.0),
                line("d", [300, 20], [305, 100], strength=1.0),
            ],
        }
        solution = solve_homography(detection, load_template(), hypotheses=2000)
        self.assertIsNone(solution.get("review_candidate"))

    def test_fewer_than_four_lines_produce_no_review_candidate(self):
        detection = {
            "image_size": {"width": 400, "height": 300},
            "primitives": [
                line("a", [10, 10], [10, 90], strength=1.0),
                line("b", [200, 200], [280, 205], strength=1.0),
            ],
        }
        solution = solve_homography(detection, load_template(), hypotheses=2000)
        self.assertIsNone(solution.get("review_candidate"))
        self.assertIn("review_candidate", solution)

    def test_solve_homography_is_deterministic_for_a_fixed_seed(self):
        image = np.full((240, 320, 3), (165, 185, 205), np.uint8)
        cv2.rectangle(image, (110, 120), (210, 185), (150, 35, 150), -1)
        detection = detect_primitives(image, min_line_length=15)
        template = load_template()
        first = solve_homography(detection, template, hypotheses=500, random_seed=11)
        second = solve_homography(detection, template, hypotheses=500, random_seed=11)
        self.assertEqual(first.get("review_candidate"), second.get("review_candidate"))
        self.assertEqual(first["status"], second["status"])


if __name__ == "__main__":
    unittest.main()
