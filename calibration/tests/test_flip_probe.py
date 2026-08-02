"""The flip probe must distinguish the three outcomes that matter.

Against the real model nobody knows the right answer — that is the point. So the
probe is tested against detectors with *planted* flip behaviour, where the
expected verdict is known by construction:

- a detector that genuinely tracks landmarks through a mirror  -> ``self``
- a detector with a learned mirror permutation                 -> ``partner:j``
- a detector that predicts nonsense on mirrored input          -> ``incoherent``

If the probe cannot tell these apart on planted data, its verdict on the real
model means nothing.
"""

from __future__ import annotations

import unittest

import numpy as np

from contracts.calibration_types import RawKeypointFrame, RawKeypointObservation
from calibration.tools import flip_probe

from ._synthetic import FakeDetector, checkerboard

WIDTH, HEIGHT = 1280, 720
GATE = 0.25

# Four well-separated slots, with 0<->1 planted as a mirror pair.
POSITIONS = [(200.0, 200.0), (1079.0, 200.0), (400.0, 500.0), (879.0, 500.0)]


def _mirror_x(x: float) -> float:
    return WIDTH - 1 - x


def _record(points, frame_index=0) -> RawKeypointFrame:
    return RawKeypointFrame(
        frame_index=frame_index,
        timestamp_s=0.0,
        source_id="clip.mp4",
        image_width=WIDTH,
        image_height=HEIGHT,
        detection_state="detected",
        instance_count=1,
        instance_confidences=(0.9,),
        selected_instance=0,
        observations=tuple(
            RawKeypointObservation(i, x, y, c) for i, (x, y, c) in enumerate(points)
        ),
    )


class MirrorRecordTests(unittest.TestCase):
    def test_maps_x_back_and_leaves_y(self):
        mapped = flip_probe.mirror_record(_record([(100.0, 250.0, 0.8)]))
        self.assertEqual(mapped.observations[0].x, WIDTH - 1 - 100.0)
        self.assertEqual(mapped.observations[0].y, 250.0)

    def test_is_an_involution(self):
        original = _record([(137.5, 250.0, 0.8)])
        self.assertEqual(flip_probe.mirror_record(flip_probe.mirror_record(original)), original)

    def test_confidence_and_slot_are_untouched(self):
        mapped = flip_probe.mirror_record(_record([(1.0, 2.0, 0.0123)]))
        self.assertEqual(mapped.observations[0].confidence, 0.0123)
        self.assertEqual(mapped.observations[0].slot_index, 0)

    def test_no_detection_passes_through(self):
        record = RawKeypointFrame(
            frame_index=0, timestamp_s=0.0, source_id="c.mp4",
            image_width=WIDTH, image_height=HEIGHT, detection_state="no_detection",
            instance_count=0, instance_confidences=(), selected_instance=None, observations=(),
        )
        self.assertEqual(flip_probe.mirror_record(record), record)


class CompareFrameTests(unittest.TestCase):
    def setUp(self):
        self.original = _record([(x, y, 0.9) for x, y in POSITIONS])
        self.tolerance = flip_probe.tolerance_px(WIDTH, HEIGHT)

    def _verdicts(self, mapped_points):
        mapped = _record([(x, y, 0.9) for x, y in mapped_points])
        results = flip_probe.compare_frame(
            self.original, mapped, len(POSITIONS), GATE, self.tolerance
        )
        return [r.verdict for r in results]

    def test_identity_flip_reads_as_self(self):
        self.assertEqual(self._verdicts(POSITIONS), ["self"] * 4)

    def test_swapped_pairs_read_as_partners(self):
        swapped = [POSITIONS[1], POSITIONS[0], POSITIONS[3], POSITIONS[2]]
        self.assertEqual(
            self._verdicts(swapped), ["partner:1", "partner:0", "partner:3", "partner:2"]
        )

    def test_nonsense_reads_as_incoherent(self):
        far = [(640.0, 30.0)] * 4
        self.assertEqual(self._verdicts(far), ["incoherent"] * 4)

    def test_small_jitter_inside_tolerance_still_reads_as_self(self):
        jittered = [(x + 4.0, y - 3.0) for x, y in POSITIONS]
        self.assertEqual(self._verdicts(jittered), ["self"] * 4)

    def test_below_gate_slot_is_not_evaluated(self):
        mapped = _record([(x, y, 0.01) for x, y in POSITIONS])
        results = flip_probe.compare_frame(
            self.original, mapped, len(POSITIONS), GATE, self.tolerance
        )
        self.assertEqual([r.verdict for r in results], ["not_evaluated"] * 4)

    def test_distance_is_recorded(self):
        mapped = _record([(POSITIONS[0][0] + 10.0, POSITIONS[0][1], 0.9)])
        results = flip_probe.compare_frame(self.original, mapped, 1, GATE, self.tolerance)
        self.assertAlmostEqual(results[0].distance_px, 10.0, places=5)

    def test_confidences_from_both_passes_are_retained(self):
        mapped = _record([(POSITIONS[0][0], POSITIONS[0][1], 0.42)])
        results = flip_probe.compare_frame(self.original, mapped, 1, GATE, self.tolerance)
        self.assertEqual(results[0].original_confidence, 0.9)
        self.assertEqual(results[0].flipped_confidence, 0.42)


class RunProbeTests(unittest.TestCase):
    """End-to-end against detectors with planted flip behaviour."""

    def setUp(self):
        self.frames = [checkerboard(WIDTH, HEIGHT, seed=i) for i in range(3)]
        self.records = [_record([(x, y, 0.9) for x, y in POSITIONS], i) for i in range(3)]

    def _run(self, flipped_positions) -> dict:
        """``self.records`` are the unflipped pass; the fake serves the flipped one.

        ``run_flip_probe`` only calls the detector on mirrored frames — the
        originals are handed in — so the fake always answers in mirrored-image
        coordinates.
        """

        def predict(_frame, _index):
            return [(x, y, 0.9) for x, y in flipped_positions]

        detector = FakeDetector(predict, keypoint_count=len(POSITIONS))
        report, _mapped = flip_probe.run_flip_probe(
            detector, self.frames, self.records, GATE
        )
        return report

    def test_true_landmark_tracking_reads_as_self(self):
        """Mirrored prediction at the mirrored position = the same physical point."""
        mirrored = [(_mirror_x(x), y) for x, y in POSITIONS]
        report = self._run(mirrored)
        self.assertEqual(
            [s["dominant_verdict"] for s in report["slots"]], ["self"] * len(POSITIONS)
        )
        self.assertTrue(all(s["agreement"] == 1.0 for s in report["slots"]))

    def test_learned_permutation_reads_as_partners(self):
        """Slot 0 predicting slot 1's physical point under flip — the failure mode."""
        swapped = [POSITIONS[1], POSITIONS[0], POSITIONS[3], POSITIONS[2]]
        mirrored = [(_mirror_x(x), y) for x, y in swapped]
        report = self._run(mirrored)
        self.assertEqual(
            [s["dominant_verdict"] for s in report["slots"]],
            ["partner:1", "partner:0", "partner:3", "partner:2"],
        )

    def test_nonsense_reads_as_incoherent(self):
        report = self._run([(640.0, 30.0)] * len(POSITIONS))
        self.assertEqual(
            [s["dominant_verdict"] for s in report["slots"]], ["incoherent"] * len(POSITIONS)
        )

    def test_report_covers_every_frame_and_slot(self):
        report = self._run([(_mirror_x(x), y) for x, y in POSITIONS])
        self.assertEqual(len(report["frames"]), 3)
        self.assertEqual(len(report["slots"]), len(POSITIONS))
        for frame in report["frames"]:
            self.assertEqual(len(frame["slots"]), len(POSITIONS))

    def test_report_disclaims_court_geometry(self):
        """The probe must not be readable as a landmark claim."""
        report = self._run([(_mirror_x(x), y) for x, y in POSITIONS])
        self.assertIn("assert", report["note"].lower())

    def test_no_detection_frames_are_survivable(self):
        detector = FakeDetector(lambda _f, _i: None, keypoint_count=len(POSITIONS))
        report, mapped = flip_probe.run_flip_probe(detector, self.frames, self.records, GATE)
        self.assertEqual(len(mapped), 3)
        for slot in report["slots"]:
            self.assertEqual(slot["frames_evaluated"], 0)
            self.assertIsNone(slot["dominant_verdict"])


class ToleranceTests(unittest.TestCase):
    def test_scales_with_the_image_diagonal(self):
        self.assertAlmostEqual(
            flip_probe.tolerance_px(1280, 720, 0.02), float(np.hypot(1280, 720)) * 0.02
        )

    def test_larger_image_gets_a_larger_tolerance(self):
        self.assertGreater(
            flip_probe.tolerance_px(3840, 2160), flip_probe.tolerance_px(1280, 720)
        )


if __name__ == "__main__":
    unittest.main()
