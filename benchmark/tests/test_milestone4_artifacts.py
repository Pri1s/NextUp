"""Closeout contracts for the Milestone 4 result namespace.

The namespace is reserved and empty until the shadow comparison is actually run,
which needs the model checkpoint and a human decision on Milestone 3's gate. So
most of this file skips for now. What it does enforce today is that the
namespace stays *separate* -- M4 must never overwrite the immutable M1 or M3
reports, and a report that appears here must be complete enough to read without
the session that produced it.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULT_DIR = PROJECT_DIR / "benchmark/results/nba_m4_v1"
M1_DIR = PROJECT_DIR / "benchmark/results/nba_m1_v1_human"
M3_DIR = PROJECT_DIR / "benchmark/results/nba_m3_v1"

REQUIRED_ACCEPTANCE_KEYS = {
    "median_improvement_beyond_19ft_pct",
    "meets_25pct",
    "near_court_p95_regression_px",
    "meets_no_near_regression",
    "held_out_median_px",
    "meets_4px",
    "held_out_p95_px",
    "meets_10px",
    "court_median_ft",
    "meets_0_5ft",
    "court_p95_ft",
    "meets_1ft",
    "gross_failures",
    "meets_zero_gross",
    "coverage_change",
    "meets_no_coverage_loss",
}


def reports():
    return sorted(RESULT_DIR.glob("*.json"))


class NamespaceTests(unittest.TestCase):
    def test_the_namespace_exists_and_explains_itself(self):
        readme = RESULT_DIR / "README.md"
        self.assertTrue(readme.is_file())
        text = readme.read_text(encoding="utf-8").lower()
        self.assertIn("no candidate is promoted", text)

    def test_the_namespace_is_separate_from_the_immutable_reports(self):
        """M1 is a v1-layout historical artifact and M3 has its own gate; an M4
        run must not land in either."""
        self.assertNotEqual(RESULT_DIR.resolve(), M1_DIR.resolve())
        self.assertNotEqual(RESULT_DIR.resolve(), M3_DIR.resolve())
        for stray in RESULT_DIR.glob("*.json"):
            with self.subTest(report=stray.name):
                self.assertFalse((M1_DIR / stray.name).exists() and stray.name == "milestone1.json")


class ReportContentTests(unittest.TestCase):
    def setUp(self):
        if not reports():
            self.skipTest("no Milestone 4 report yet; the shadow comparison has not been run")

    def test_every_report_declares_its_schema(self):
        for path in reports():
            with self.subTest(report=path.name):
                payload = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(
                    payload.get("schema_version"), "benchmark-hybrid-score-1.0.0"
                )

    def test_every_report_carries_the_full_acceptance_block(self):
        """A partial acceptance block invites reading the flags that happen to
        be present and ignoring the ones that are not."""
        for path in reports():
            with self.subTest(report=path.name):
                payload = json.loads(path.read_text(encoding="utf-8"))
                acceptance = payload.get("acceptance", {})
                self.assertTrue(REQUIRED_ACCEPTANCE_KEYS <= set(acceptance))

    def test_both_candidates_are_scored_on_the_identical_frame_set(self):
        for path in reports():
            with self.subTest(report=path.name):
                summaries = json.loads(path.read_text(encoding="utf-8"))["summaries"]
                self.assertEqual(
                    summaries["keypoint"]["frames_scored"],
                    summaries["hybrid"]["frames_scored"],
                )

    def test_no_report_claims_a_promotion(self):
        """Milestone 4 is shadow mode. A report asserting a selected challenger
        would mean the selection policy shipped early."""
        for path in reports():
            with self.subTest(report=path.name):
                text = path.read_text(encoding="utf-8")
                self.assertNotIn("select_challenger", text.lower())

    def test_the_extraction_configuration_is_frozen_to_one_value(self):
        """A refiner config hash is meaningless if the evidence under it drifts."""
        for path in reports():
            payload = json.loads(path.read_text(encoding="utf-8"))
            hashes = payload.get("shadow_config_hashes")
            if hashes is None:
                continue
            with self.subTest(report=path.name):
                self.assertEqual(len(set(hashes)), 1)


if __name__ == "__main__":
    unittest.main()
