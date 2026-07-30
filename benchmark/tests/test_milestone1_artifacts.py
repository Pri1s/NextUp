"""Closeout contracts for the tracked Milestone 1 reference and reports."""

import hashlib
import json
import unittest
from pathlib import Path

from contracts.annotations import FrameAnnotation, MarkingFeature, validate


PROJECT_DIR = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_DIR / "benchmark/manifests/nba_m1_v1.json"
REFERENCE_INDEX_PATH = PROJECT_DIR / "benchmark/references/nba_m1_v1_human.json"
REFERENCE_DIR = PROJECT_DIR / "benchmark/references/nba_m1_v1_human"
RESULT_DIR = PROJECT_DIR / "benchmark/results/nba_m1_v1_human"


class Milestone1ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmark_manifest = json.loads(MANIFEST_PATH.read_text())
        cls.reference_index = json.loads(REFERENCE_INDEX_PATH.read_text())

    def test_reference_set_exactly_matches_the_six_locked_frames(self):
        expected = {item["frame_id"] for item in self.benchmark_manifest["frames"]}
        indexed = {item["frame_id"] for item in self.reference_index["frames"]}
        files = {path.name.removesuffix(".frame.json") for path in REFERENCE_DIR.glob("*.frame.json")}

        self.assertEqual(self.reference_index["manifest_id"], "nba_m1_v1")
        self.assertEqual(self.reference_index["authority"], "single_human_reference")
        self.assertEqual(self.reference_index["frame_count"], 6)
        self.assertEqual(indexed, expected)
        self.assertEqual(files, expected)

    def test_each_reference_is_complete_valid_and_hash_pinned(self):
        locked = {item["frame_id"]: item for item in self.benchmark_manifest["frames"]}
        indexed = {item["frame_id"]: item for item in self.reference_index["frames"]}

        for frame_id in sorted(locked):
            path = REFERENCE_DIR / f"{frame_id}.frame.json"
            raw = path.read_bytes()
            annotation = FrameAnnotation.from_dict(json.loads(raw))

            self.assertEqual(validate(annotation, require_complete=True), [], frame_id)
            self.assertEqual(annotation.frame_id, frame_id)
            self.assertEqual(annotation.clip, locked[frame_id]["clip"])
            self.assertEqual(annotation.frame_index, locked[frame_id]["frame_index"])
            self.assertEqual(annotation.coordinate_space, "frame")
            self.assertEqual(annotation.pass_id, "human_reference")
            self.assertEqual(annotation.prompt_version, "human-labeler-1.0.0")
            self.assertEqual(
                len(annotation.features) + len(annotation.skipped),
                len(MarkingFeature),
            )
            for feature in annotation.features:
                minimum = 6 if feature.kind == "arc" else 4
                self.assertGreaterEqual(len(feature.points), minimum, feature.feature.value)

            record = indexed[frame_id]
            self.assertEqual(record["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(record["image_sha256"], annotation.image_sha256)
            self.assertEqual(record["features_annotated"], len(annotation.features))
            self.assertEqual(record["features_skipped"], len(annotation.skipped))

    def test_closeout_and_both_baseline_reports_cover_all_six_frames(self):
        closeout = json.loads((RESULT_DIR / "milestone1.json").read_text())
        self.assertEqual(closeout["status"], "complete")
        self.assertEqual(closeout["reference_set"], "nba_m1_v1_human")
        self.assertEqual(closeout["agent_annotation"]["decision"], "rejected")
        self.assertEqual(
            closeout["expansion_decision"]["decision"],
            "deferred_not_required_for_milestone_1",
        )
        self.assertTrue(closeout["limitations"])

        summaries = []
        expected = {item["frame_id"] for item in self.benchmark_manifest["frames"]}
        for tier in ("confident", "probable"):
            report = json.loads((RESULT_DIR / f"baseline_{tier}.json").read_text())
            self.assertEqual({frame["frame_id"] for frame in report["frames"]}, expected)
            self.assertEqual(report["summary"]["frames"], 6)
            self.assertEqual(report["summary"]["frames_scored"], 5)
            self.assertEqual(report["summary"]["frames_unscored"], 1)
            self.assertEqual(
                report["summary"]["status_counts"],
                {"DEGRADED": 5, "UNCALIBRATED_INSUFFICIENT_EVIDENCE": 1},
            )
            summaries.append(report["summary"])

        self.assertEqual(summaries[0], summaries[1])

    def test_failed_agent_gate_is_preserved_as_the_methodology_evidence(self):
        gate = json.loads((RESULT_DIR / "agent_synthetic_gate.json").read_text())
        self.assertFalse(gate["decision"]["qualified"])
        self.assertFalse(gate["decision"]["real_annotation_passes_allowed"])
        failed = gate["configurations"][0]
        self.assertEqual(failed["checks"]["median"], "pass")
        self.assertEqual(failed["checks"]["p95"], "fail")
        self.assertEqual(failed["checks"]["absolute_mean_signed_per_family"], "fail")


if __name__ == "__main__":
    unittest.main()
