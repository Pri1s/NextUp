import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from court_homography import (
    _homography_from_line_matches,
    _project,
    detect_primitives,
    load_template,
    solve_homography,
)
from homography_proposals import generate_guided_proposals
from hud_detection import HudContext, build_hud_context, detect_hud


PROJECT_DIR = Path(__file__).resolve().parents[1]
ANCHOR = PROJECT_DIR / "homography_input" / "001_video_4_f000163.jpg"
CONTEXT_DIR = PROJECT_DIR / "dataset" / "frames" / "001_video_4"
LABEL = PROJECT_DIR / "dataset" / "labels" / "001_video_4" / "001_video_4_f000163.json"


def rectangle_iou(first, second):
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


@unittest.skipUnless(ANCHOR.is_file() and CONTEXT_DIR.is_dir(), "known clip-001 validation frames are unavailable")
class KnownFrameUpstreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = build_hud_context(ANCHOR, [CONTEXT_DIR])
        anchor_image = cv2.imread(str(ANCHOR))
        cls.detection = detect_primitives(anchor_image, hud_context=cls.context)

    def test_all_four_frames_find_the_bottom_hud_without_masking_court_paint(self):
        # Manually bounded persistent score graphic and central painted-court
        # area in the checked-in 2478x1382 clip. These are intentionally broad;
        # the test guards component identity rather than pixel-perfect masks.
        expected_bottom_hud = [800, 1120, 820, 262]
        court_paint = [280, 520, 1550, 560]
        self.assertEqual(self.context.frame_indices, (35, 163, 341, 423))
        for anchor_position, frame_index in enumerate(self.context.frame_indices):
            with self.subTest(frame_index=frame_index):
                context = HudContext(
                    images=self.context.images,
                    paths=self.context.paths,
                    frame_indices=self.context.frame_indices,
                    anchor_position=anchor_position,
                    frame_span=self.context.frame_span,
                    mode=self.context.mode,
                )
                hud = detect_hud(self.context.images[anchor_position], context)
                regions = hud.diagnostics["regions"]
                bottom = max(regions, key=lambda region: region["bbox"][1])
                self.assertGreaterEqual(rectangle_iou(bottom["bbox"], expected_bottom_hud), .55)
                self.assertTrue(
                    all(rectangle_iou(region["bbox"], court_paint) < .01 for region in regions)
                )

    def test_known_frame_pool_and_long_occlusion_regressions(self):
        lines = [
            item for item in self.detection["primitives"]
            if item["type"] == "line_segment"
        ]
        top_24_hud_occupancy = sum(
            item["evidence"]["hud_overlap_ratio"] > 0 for item in lines[:24]
        )
        self.assertLessEqual(top_24_hud_occupancy, 2)

        recovered = [
            item for item in lines
            if 1050 <= item["geometry"]["length_px"] <= 1130
            and .80 <= item["evidence"]["observed_coverage_ratio"] <= .86
            and item["evidence"]["extended_merge_count"] >= 1
        ]
        self.assertTrue(recovered)
        self.assertTrue(
            all(
                item["evidence"]["hud_overlap_ratio"] == 0
                for item in lines
                if item["evidence"]["extended_merge_count"] >= 1
            )
        )

    @unittest.skipUnless(LABEL.is_file(), "known f000163 manual labels are unavailable")
    def test_three_corner_fallback_surfaces_the_accurate_lane_hypothesis(self):
        template = load_template()
        proposals, _ = generate_guided_proposals(
            self.detection, template, max_proposals=5000
        )
        expected = {
            "north_lane_west": "line_000",
            "west_sideline": "line_046",
            "north_baseline": "line_001",
            "north_free_throw": "line_005",
        }
        proposal_index, proposal = next(
            (index, item)
            for index, item in enumerate(proposals)
            if item["seed_pairs"] == expected
        )
        self.assertLess(proposal_index, 5000)
        self.assertEqual(proposal["source"]["topology_guided"], True)

        homography = _homography_from_line_matches(
            proposal["template_lines"], proposal["detected_lines"]
        )
        self.assertIsNotNone(homography)
        keypoint_ids = [
            "north_lane_baseline_east",
            "north_lane_baseline_west",
            "north_lane_free_throw_east",
            "north_lane_free_throw_west",
        ]
        projected = _project(
            np.asarray([template["keypoints"][identifier] for identifier in keypoint_ids], np.float32),
            homography,
        )
        labels = json.loads(LABEL.read_text(encoding="utf-8"))["keypoints"]
        manual = np.asarray(
            [[labels[index]["x"], labels[index]["y"]] for index in (2, 3, 8, 9)],
            dtype=float,
        )
        mean_error = float(np.mean(np.linalg.norm(projected - manual, axis=1)))
        self.assertLessEqual(mean_error, 12.0)

    @unittest.skipUnless(LABEL.is_file(), "known f000163 manual labels are unavailable")
    def test_f000163_remains_an_automatic_rejection_but_exposes_a_review_candidate(self):
        template = load_template()
        solution = solve_homography(self.detection, template)

        self.assertEqual(solution["status"], "rejected")
        self.assertTrue(solution["reason"].startswith("no_court_structural_consensus"))

        review_candidate = solution["review_candidate"]
        self.assertIsNotNone(review_candidate)
        self.assertEqual(review_candidate["status"], "review_required")
        self.assertGreaterEqual(review_candidate["matched_keypoints"], 4)
        self.assertLessEqual(review_candidate["mean_matched_error_px"], 12.0)
        self.assertIn("automatic_gate_failures", review_candidate)
        self.assertTrue(review_candidate["automatic_gate_failures"])

        homography = np.asarray(review_candidate["homography"])
        keypoint_ids = [
            "north_lane_baseline_east",
            "north_lane_baseline_west",
            "north_lane_free_throw_east",
            "north_lane_free_throw_west",
        ]
        projected = _project(
            np.asarray([template["keypoints"][identifier] for identifier in keypoint_ids], np.float32),
            homography,
        )
        labels = json.loads(LABEL.read_text(encoding="utf-8"))["keypoints"]
        manual = np.asarray(
            [[labels[index]["x"], labels[index]["y"]] for index in (2, 3, 8, 9)],
            dtype=float,
        )
        mean_error = float(np.mean(np.linalg.norm(projected - manual, axis=1)))
        self.assertLessEqual(mean_error, 12.0)

    def test_f000163_review_candidate_is_deterministic(self):
        template = load_template()
        first = solve_homography(self.detection, template)
        second = solve_homography(self.detection, template)
        self.assertEqual(first["review_candidate"], second["review_candidate"])


if __name__ == "__main__":
    unittest.main()
