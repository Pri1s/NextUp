"""Reviewer artifacts must work offline and must not overclaim.

Two failure modes worth guarding. First, a review page that silently needs the
network is useless on an air-gapped machine and, worse, may render blank without
saying why. Second — the serious one — an artifact that reads like a verified
index-to-landmark map. Phase 0's whole purpose is that no such map exists yet.
"""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

from calibration.io import records as rec
from calibration.tools import review
from calibration.tools.inspect_model import InspectionConfig, run_inspection

from ._synthetic import ArrayFrameSource, checkerboard, stable_detector


class ReviewPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        source = ArrayFrameSource([checkerboard(640, 360, seed=i) for i in range(8)])
        cls.result = run_inspection(
            source=source,
            detector_factory=lambda size: stable_detector(imgsz=size),
            out_root=Path(cls._tmp.name),
            config=InspectionConfig(imgsz=(640, 960), num_frames=3, run_flip_probe=True),
            run_id="review",
        )
        cls.html = cls.result.review_html_path.read_text(encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_no_external_hosts(self):
        """A CSP-free local page must still never reach the network."""
        for pattern in ("http://", "https://", "//cdn", "@import"):
            self.assertNotIn(pattern, self.html, f"review page references {pattern}")

    def test_no_javascript(self):
        self.assertNotIn("<script", self.html.lower())
        self.assertFalse(re.search(r"\son(click|load|error)\s*=", self.html, re.I))

    def test_every_referenced_image_exists(self):
        sources = re.findall(r"<img[^>]+src='([^']+)'", self.html)
        self.assertGreater(len(sources), 10)
        for src in sources:
            self.assertTrue(
                (self.result.run_dir / src).is_file(), f"review page links missing {src}"
            )

    def test_one_filmstrip_section_per_slot(self):
        # Two inference sizes, eighteen slots each.
        self.assertEqual(self.html.count("<h3>slot "), 36)

    def test_states_that_nothing_is_verified(self):
        self.assertIn("Nothing on this page asserts what a slot means", self.html)

    def test_explains_the_flip_verdicts(self):
        for token in ("self", "partner:j", "incoherent"):
            self.assertIn(token, self.html)

    def test_has_a_section_per_inference_size(self):
        self.assertIn("imgsz 640", self.html)
        self.assertIn("imgsz 960", self.html)

    def test_renders_provenance(self):
        run = json.loads(self.result.run_json_path.read_text())
        self.assertIn(run["model"]["sha256"], self.html)

    def test_page_is_self_contained_html(self):
        self.assertTrue(self.html.lstrip().startswith("<!DOCTYPE html>"))
        self.assertIn("</html>", self.html)


class WorksheetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        source = ArrayFrameSource([checkerboard(640, 360, seed=i) for i in range(6)])
        cls.result = run_inspection(
            source=source,
            detector_factory=lambda _size: stable_detector(),
            out_root=Path(cls._tmp.name),
            config=InspectionConfig(num_frames=3, run_flip_probe=True),
            run_id="worksheet",
        )
        cls.markdown = cls.result.worksheet_md_path.read_text(encoding="utf-8")
        cls.payload = json.loads(cls.result.worksheet_json_path.read_text())

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_markdown_has_a_row_per_slot(self):
        rows = [line for line in self.markdown.splitlines() if re.match(r"^\| \d+ \|", line)]
        self.assertEqual(len(rows), 18)

    def test_markdown_is_marked_unverified(self):
        self.assertIn("UNVERIFIED", self.markdown)
        self.assertIn("sign", self.markdown.lower())

    def test_markdown_carries_the_weights_digest(self):
        run = json.loads(self.result.run_json_path.read_text())
        self.assertIn(run["model"]["sha256"], self.markdown)

    def test_markdown_offers_an_ambiguous_verdict(self):
        """Reviewers need a way to say "I don't know" that is not a guess."""
        self.assertIn("ambiguous", self.markdown.lower())

    def test_json_template_has_no_assignments(self):
        self.assertEqual(self.payload["status"], "unverified")
        self.assertFalse(self.payload["is_adapter_map"])
        self.assertEqual(len(self.payload["slots"]), 18)
        for slot in self.payload["slots"]:
            self.assertIsNone(slot["landmark_id"])
            self.assertIsNone(slot["verdict"])

    def test_json_template_is_unsigned(self):
        self.assertIsNone(self.payload["reviewer"])
        self.assertIsNone(self.payload["signed_off_utc"])

    def test_json_template_records_observed_flip_verdicts_as_observations(self):
        """Flip behaviour is evidence and may be prefilled; identity may not."""
        self.assertIn("observed_flip_verdict", self.payload["slots"][0])


class CandidateVocabularyTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_reads_a_schema_style_file_as_pure_data(self):
        path = self.tmp / "vocab.json"
        path.write_text(
            json.dumps({"keypoints": [{"index": 1, "id": "north_baseline_sideline_east"}]})
        )
        self.assertEqual(review.load_candidate_names(path), ["north_baseline_sideline_east"])

    def test_reads_a_bare_list(self):
        path = self.tmp / "vocab.json"
        path.write_text(json.dumps(["a", "b"]))
        self.assertEqual(review.load_candidate_names(path), ["a", "b"])

    def test_appendix_disclaims_that_the_names_apply_to_this_model(self):
        summary = rec.build_summary([], keypoint_count=2, gate=0.25)
        markdown = review.render_slot_review_markdown(
            {"run_id": "r", "model": {}, "source": {}}, 2, summary, None, ["some_landmark"]
        )
        self.assertIn("not a claim", markdown)
        self.assertIn("some_landmark", markdown)

    def test_absent_by_default(self):
        summary = rec.build_summary([], keypoint_count=2, gate=0.25)
        markdown = review.render_slot_review_markdown(
            {"run_id": "r", "model": {}, "source": {}}, 2, summary, None
        )
        self.assertNotIn("Appendix", markdown)


if __name__ == "__main__":
    unittest.main()
