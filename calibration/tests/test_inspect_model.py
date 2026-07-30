"""End-to-end runs against planted detectors.

Covers the two cases a Phase-0 tool has to survive: a normal clip, and a source
where the model never finds a court at all. The second is not an edge case — it
is the behaviour that has to be characterised before a
``NOT_ATTEMPTED_NO_COURT`` gate can exist, so a crash there would be a crash on
the deliverable.
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from calibration.io import provenance as prov
from calibration.io import records as rec
from calibration.tools.inspect_model import InspectionConfig, run_inspection

from ._synthetic import (
    ArrayFrameSource,
    FakeDetector,
    checkerboard,
    default_slot_positions,
    no_detection_detector,
    stable_detector,
)


class InspectionRunTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.source = ArrayFrameSource([checkerboard(640, 360, seed=i) for i in range(12)])

    def _run(self, detector=None, **config_kwargs):
        detector = detector or stable_detector()
        config = InspectionConfig(**{"num_frames": 4, **config_kwargs})
        return run_inspection(
            source=self.source,
            detector_factory=lambda _size: detector,
            out_root=self.tmp,
            config=config,
            run_id="testrun",
        )

    def test_writes_the_full_artifact_set(self):
        result = self._run()
        size_dir = result.run_dir / "imgsz_0640"
        for relative in (
            "run.json",
            "review.html",
            "slot_review.md",
            "slot_review.template.json",
            "imgsz_0640/predictions.jsonl",
            "imgsz_0640/predictions.csv",
            "imgsz_0640/frames.csv",
            "imgsz_0640/summary.json",
            "imgsz_0640/contact_sheet.jpg",
        ):
            self.assertTrue((result.run_dir / relative).is_file(), f"missing {relative}")
        self.assertEqual(len(list((size_dir / "overlays").glob("*.jpg"))), 4)
        self.assertEqual(len(list((size_dir / "slots").glob("*.jpg"))), 18)

    def test_one_filmstrip_per_slot_including_never_detected_ones(self):
        positions = default_slot_positions()

        def predict(_frame, _index):
            # Slot 17 is always below gate; it still needs a strip.
            return [(x, y, 0.01 if i == 17 else 0.9) for i, (x, y) in enumerate(positions)]

        result = self._run(FakeDetector(predict))
        self.assertTrue((result.run_dir / "imgsz_0640" / "slots" / "slot_17.jpg").is_file())

    def test_records_match_the_frames_inspected(self):
        result = self._run()
        frames = rec.read_jsonl(result.run_dir / "imgsz_0640" / "predictions.jsonl")
        indices = json.loads((result.run_dir / "run.json").read_text())["sampling"]["frame_indices"]
        self.assertEqual([f.frame_index for f in frames], indices)

    def test_run_json_captures_provenance(self):
        result = self._run()
        run = json.loads(result.run_json_path.read_text())
        self.assertEqual(run["schema_version"], "phase0-inspection-1.0.0")
        self.assertEqual(run["model"]["keypoint_count"], 18)
        self.assertEqual(len(run["model"]["sha256"]), 64)
        self.assertEqual(run["model"]["train_fliplr"], 0.5)
        self.assertEqual(run["sampling"]["mode"], "even")
        self.assertIn("libraries", run["environment"])
        self.assertIn("available", run["git"])

    def test_run_json_never_claims_phase0_is_complete(self):
        """A green run is not a sign-off, and the artifact must say so."""
        run = json.loads(self._run().run_json_path.read_text())
        self.assertFalse(run["phase0_status"]["slots_assigned"])
        self.assertFalse(run["phase0_status"]["human_signed_off"])

    def test_output_inventory_digests_match_the_files_on_disk(self):
        result = self._run()
        run = json.loads(result.run_json_path.read_text())
        self.assertGreater(len(run["outputs"]), 10)
        for entry in run["outputs"]:
            path = result.run_dir / entry["path"]
            self.assertTrue(path.is_file(), f"inventory lists missing {entry['path']}")
            self.assertEqual(entry["sha256"], prov.sha256_file(path))

    def test_explicit_frames_are_honoured(self):
        result = self._run(explicit_frames="0,5,11")
        frames = rec.read_jsonl(result.run_dir / "imgsz_0640" / "predictions.jsonl")
        self.assertEqual([f.frame_index for f in frames], [0, 5, 11])

    def test_run_is_reproducible(self):
        first = self._run()
        second = run_inspection(
            source=self.source,
            detector_factory=lambda _size: stable_detector(),
            out_root=self.tmp,
            config=InspectionConfig(num_frames=4),
            run_id="testrun2",
        )
        a = rec.read_jsonl(first.run_dir / "imgsz_0640" / "predictions.jsonl")
        b = rec.read_jsonl(second.run_dir / "imgsz_0640" / "predictions.jsonl")
        self.assertEqual([f.frame_index for f in a], [f.frame_index for f in b])
        self.assertEqual([f.observations for f in a], [f.observations for f in b])

    def test_default_run_id_changes_with_config(self):
        from calibration.tools.inspect_model import default_run_id

        stamp = "2026-07-28T00:00:00+00:00"
        a = default_run_id(InspectionConfig(imgsz=(640,)), stamp)
        b = default_run_id(InspectionConfig(imgsz=(960,)), stamp)
        self.assertNotEqual(a, b)


class SweepTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.source = ArrayFrameSource([checkerboard(640, 360, seed=i) for i in range(8)])

    def test_each_size_gets_its_own_directory_and_a_comparison(self):
        result = run_inspection(
            source=self.source,
            detector_factory=lambda size: stable_detector(imgsz=size),
            out_root=self.tmp,
            config=InspectionConfig(imgsz=(640, 960), num_frames=3),
            run_id="sweep",
        )
        self.assertTrue((result.run_dir / "imgsz_0640" / "predictions.jsonl").is_file())
        self.assertTrue((result.run_dir / "imgsz_0960" / "predictions.jsonl").is_file())
        self.assertIsNotNone(result.sweep_csv_path)

        with open(result.sweep_csv_path, encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        self.assertIn("above_gate_rate_640", rows[0])
        self.assertIn("above_gate_rate_960", rows[0])
        self.assertEqual(len(rows) - 1, 18)

    def test_single_size_writes_no_comparison(self):
        result = run_inspection(
            source=self.source,
            detector_factory=lambda _size: stable_detector(),
            out_root=self.tmp,
            config=InspectionConfig(num_frames=2),
            run_id="single",
        )
        self.assertIsNone(result.sweep_csv_path)


class NoCourtTests(unittest.TestCase):
    """The no-detection path is a deliverable, not an error path."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.source = ArrayFrameSource([checkerboard(640, 360, seed=i) for i in range(6)])
        self.result = run_inspection(
            source=self.source,
            detector_factory=lambda _size: no_detection_detector(),
            out_root=self.tmp,
            config=InspectionConfig(num_frames=4, run_flip_probe=True),
            run_id="nocourt",
        )

    def test_run_completes_and_writes_everything(self):
        for relative in ("run.json", "review.html", "slot_review.md", "slot_review.template.json"):
            self.assertTrue((self.result.run_dir / relative).is_file())

    def test_summary_reports_every_frame_as_no_detection(self):
        summary = json.loads(
            (self.result.run_dir / "imgsz_0640" / "summary.json").read_text()
        )
        self.assertEqual(summary["frames_inspected"], 4)
        self.assertEqual(summary["frames_detected"], 0)
        self.assertEqual(summary["frames_no_detection"], 4)

    def test_no_fabricated_observations(self):
        """Not one (0, 0) row — absence stays absence."""
        path = self.result.run_dir / "imgsz_0640" / "predictions.csv"
        with open(path, encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        self.assertEqual(len(rows), 1)  # header only

    def test_frames_csv_still_lists_the_frames(self):
        path = self.result.run_dir / "imgsz_0640" / "frames.csv"
        with open(path, encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))[1:]
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row[2] == "no_detection" for row in rows))

    def test_overlays_and_filmstrips_are_still_produced(self):
        size_dir = self.result.run_dir / "imgsz_0640"
        self.assertEqual(len(list((size_dir / "overlays").glob("*.jpg"))), 4)
        self.assertEqual(len(list((size_dir / "slots").glob("*.jpg"))), 18)

    def test_flip_probe_reports_nothing_evaluated_rather_than_failing(self):
        report = json.loads((self.result.run_dir / "imgsz_0640" / "flip_probe.json").read_text())
        self.assertTrue(all(s["frames_evaluated"] == 0 for s in report["slots"]))
        self.assertTrue(all(s["dominant_verdict"] is None for s in report["slots"]))


class FlipProbeArtifactTests(unittest.TestCase):
    def test_flip_images_and_report_are_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = ArrayFrameSource([checkerboard(640, 360, seed=i) for i in range(5)])
            result = run_inspection(
                source=source,
                detector_factory=lambda _size: stable_detector(),
                out_root=tmp,
                config=InspectionConfig(num_frames=3, run_flip_probe=True),
                run_id="flip",
            )
            size_dir = result.run_dir / "imgsz_0640"
            self.assertTrue((size_dir / "flip_probe.json").is_file())
            self.assertEqual(len(list((size_dir / "flip").glob("*.jpg"))), 3)

    def test_flip_probe_is_off_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = ArrayFrameSource([checkerboard(640, 360, seed=i) for i in range(4)])
            result = run_inspection(
                source=source,
                detector_factory=lambda _size: stable_detector(),
                out_root=tmp,
                config=InspectionConfig(num_frames=2),
                run_id="noflip",
            )
            self.assertFalse((result.run_dir / "imgsz_0640" / "flip_probe.json").exists())


if __name__ == "__main__":
    unittest.main()
