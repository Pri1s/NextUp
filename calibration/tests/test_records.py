"""Records must carry model output through without loss or invention.

The reference pipeline's two failure modes were writing ``(0, 0)`` for absent
points and discarding confidence. Both are silent — they produce a well-formed
file that lies. These tests exist to make either one loud.
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from contracts.calibration_types import RawKeypointFrame, RawKeypointObservation
from calibration.io import records as rec


def _detected(frame_index: int, points, **overrides) -> RawKeypointFrame:
    defaults = dict(
        frame_index=frame_index,
        timestamp_s=frame_index / 30.0,
        source_id="clip.mp4",
        image_width=1280,
        image_height=720,
        detection_state="detected",
        instance_count=1,
        instance_confidences=(0.94,),
        selected_instance=0,
        observations=tuple(
            RawKeypointObservation(i, x, y, c) for i, (x, y, c) in enumerate(points)
        ),
    )
    defaults.update(overrides)
    return RawKeypointFrame(**defaults)


def _no_detection(frame_index: int) -> RawKeypointFrame:
    return RawKeypointFrame(
        frame_index=frame_index,
        timestamp_s=frame_index / 30.0,
        source_id="clip.mp4",
        image_width=1280,
        image_height=720,
        detection_state="no_detection",
        instance_count=0,
        instance_confidences=(),
        selected_instance=None,
        observations=(),
    )


class JsonlTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_round_trip_is_lossless(self):
        frames = [
            _detected(0, [(440.7123456789, 246.7987654321, 0.8381234567)]),
            _no_detection(5),
        ]
        path = rec.write_jsonl(self.tmp / "p.jsonl", frames)
        self.assertEqual(rec.read_jsonl(path), frames)

    def test_float_precision_survives(self):
        original = 0.024691357913
        frames = [_detected(0, [(1232.60000001, 654.29999999, original)])]
        path = rec.write_jsonl(self.tmp / "p.jsonl", frames)
        self.assertEqual(rec.read_jsonl(path)[0].observations[0].confidence, original)

    def test_one_line_per_frame(self):
        path = rec.write_jsonl(self.tmp / "p.jsonl", [_detected(0, [(1.0, 2.0, 0.5)]), _no_detection(1)])
        self.assertEqual(len(path.read_text().strip().splitlines()), 2)

    def test_no_detection_serializes_with_empty_observations(self):
        path = rec.write_jsonl(self.tmp / "p.jsonl", [_no_detection(3)])
        payload = json.loads(path.read_text().strip())
        self.assertEqual(payload["detection_state"], "no_detection")
        self.assertEqual(payload["observations"], [])


class CsvTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _rows(self, path):
        with open(path, encoding="utf-8", newline="") as handle:
            return list(csv.reader(handle))

    def test_prediction_columns(self):
        path = rec.write_predictions_csv(self.tmp / "p.csv", [_detected(0, [(1.0, 2.0, 0.5)])])
        self.assertEqual(tuple(self._rows(path)[0]), rec.PREDICTION_COLUMNS)

    def test_one_row_per_observation(self):
        frames = [
            _detected(0, [(1.0, 2.0, 0.5), (3.0, 4.0, 0.6)]),
            _detected(1, [(5.0, 6.0, 0.7)]),
        ]
        path = rec.write_predictions_csv(self.tmp / "p.csv", frames)
        self.assertEqual(len(self._rows(path)) - 1, 3)

    def test_no_detection_frame_contributes_no_observation_rows(self):
        """The (0, 0) sentinel is banned: absence is absence, not eighteen fake points."""
        path = rec.write_predictions_csv(
            self.tmp / "p.csv", [_no_detection(0), _detected(1, [(9.0, 8.0, 0.5)])]
        )
        rows = self._rows(path)[1:]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "1")
        self.assertNotIn(["0", "0", "0.0", "0.0", "0.0", "0"], rows)

    def test_low_confidence_point_keeps_its_coordinate(self):
        path = rec.write_predictions_csv(self.tmp / "p.csv", [_detected(0, [(1232.6, 654.3, 0.024)])])
        row = self._rows(path)[1]
        self.assertEqual((float(row[2]), float(row[3])), (1232.6, 654.3))
        self.assertEqual(float(row[4]), 0.024)

    def test_frames_csv_lists_every_frame_including_empty_ones(self):
        path = rec.write_frames_csv(self.tmp / "f.csv", [_detected(0, [(1.0, 2.0, 0.5)]), _no_detection(9)])
        rows = self._rows(path)
        self.assertEqual(tuple(rows[0]), rec.FRAME_COLUMNS)
        self.assertEqual([r[0] for r in rows[1:]], ["0", "9"])
        self.assertEqual(rows[2][2], "no_detection")

    def test_frames_csv_records_selected_instance_confidence(self):
        frame = _detected(
            0, [(1.0, 2.0, 0.5)], instance_count=3, instance_confidences=(0.4, 0.94, 0.7),
            selected_instance=1,
        )
        rows = self._rows(rec.write_frames_csv(self.tmp / "f.csv", [frame]))
        self.assertEqual(rows[1][4], "1")
        self.assertEqual(float(rows[1][5]), 0.94)


class SummaryTests(unittest.TestCase):
    def test_above_gate_rate_counts_all_inspected_frames(self):
        frames = [
            _detected(0, [(10.0, 10.0, 0.9)]),
            _detected(1, [(10.0, 10.0, 0.1)]),
            _no_detection(2),
            _detected(3, [(10.0, 10.0, 0.9)]),
        ]
        summary = rec.build_summary(frames, keypoint_count=1, gate=0.25)
        slot = summary["slots"][0]
        self.assertEqual(slot["frames_above_gate"], 2)
        self.assertEqual(slot["frames_detected"], 3)
        self.assertEqual(slot["above_gate_rate"], 0.5)

    def test_spread_only_uses_above_gate_points(self):
        """A below-gate point that wanders must not inflate a good slot's spread."""
        frames = [
            _detected(0, [(100.0, 100.0, 0.9)]),
            _detected(1, [(100.0, 100.0, 0.9)]),
            _detected(2, [(900.0, 700.0, 0.01)]),
        ]
        slot = rec.build_summary(frames, keypoint_count=1, gate=0.25)["slots"][0]
        self.assertEqual(slot["spread_px"], 0.0)

    def test_slot_never_detected_reports_none_not_zero(self):
        """A slot with no data must not look like a slot with data at the origin."""
        slot = rec.build_summary([_no_detection(0)], keypoint_count=1, gate=0.25)["slots"][0]
        self.assertIsNone(slot["confidence_median"])
        self.assertIsNone(slot["x_mean"])
        self.assertEqual(slot["above_gate_rate"], 0.0)

    def test_counts_no_detection_and_multi_instance_frames(self):
        frames = [
            _no_detection(0),
            _detected(1, [(1.0, 2.0, 0.5)], instance_count=2, instance_confidences=(0.9, 0.5)),
        ]
        summary = rec.build_summary(frames, keypoint_count=1, gate=0.25)
        self.assertEqual(summary["frames_no_detection"], 1)
        self.assertEqual(summary["frames_multi_instance"], 1)

    def test_summary_covers_every_slot_even_when_unseen(self):
        summary = rec.build_summary([_detected(0, [(1.0, 2.0, 0.5)])], keypoint_count=18, gate=0.25)
        self.assertEqual(len(summary["slots"]), 18)
        self.assertEqual([s["slot_index"] for s in summary["slots"]], list(range(18)))


class SweepTests(unittest.TestCase):
    def test_sweep_comparison_has_a_column_pair_per_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames = [_detected(0, [(1.0, 2.0, 0.9), (3.0, 4.0, 0.1)])]
            per_size = {
                640: rec.build_summary(frames, 2, 0.25),
                1280: rec.build_summary(frames, 2, 0.25),
            }
            path = rec.write_sweep_comparison(Path(tmp) / "sweep.csv", per_size)
            with open(path, encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(
                rows[0],
                [
                    "slot_index",
                    "above_gate_rate_640",
                    "confidence_median_640",
                    "above_gate_rate_1280",
                    "confidence_median_1280",
                ],
            )
            self.assertEqual(len(rows) - 1, 2)


if __name__ == "__main__":
    unittest.main()
