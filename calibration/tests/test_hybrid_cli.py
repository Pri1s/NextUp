"""The shadow paths must be additive, and the two entry points must agree.

Milestone 4's done-condition is that the benchmark can compare candidates
"without affecting existing consumers". That is a claim about bytes, so it is
tested as one: ``calibrations.jsonl`` produced with the hybrid path enabled has
to be identical to the file produced without it. A refiner that quietly
perturbed the baseline would corrupt the very run the evaluation rests on, and
nothing downstream would notice.

The second claim is that refinement can be re-run offline. That is what lets the
refiner be swept across Milestone 3's configuration variants without touching
video again, so the offline path has to reproduce the inline one exactly rather
than approximately.
"""

from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from calibration import cli
from calibration.adapters.evidence import load_registered_map
from calibration.estimator import project
from calibration.hybrid.control_points import (
    control_court_points,
    homography_from_control_points,
)
from calibration.io import provenance as prov
from calibration.tests._synthetic import DIGEST, ArrayFrameSource, FakeDetector
from contracts.court_layout import HalfCourtLandmark, load_registered_layout
from contracts.markings import MarkingFeature

LAYOUT = load_registered_layout("nba_halfcourt")
FRAME_COUNT = 2

#: A plausible elevated broadcast view: the baseline wide across the bottom of
#: frame, midcourt compressed toward the top.
BROADCAST_QUAD = np.array(
    [[210.0, 640.0], [470.0, 190.0], [830.0, 190.0], [1090.0, 640.0]], dtype=np.float64
)
TRUTH = homography_from_control_points(control_court_points(LAYOUT), BROADCAST_QUAD)


def painted_court(seed=0):
    """A frame with the court actually painted on it.

    A blank or checkerboard frame yields no marking evidence, so the refiner
    abstains and the byte-comparisons below would be comparing two empty
    results -- true, but proving nothing about the path that matters. Drawing
    the projected layout gives the extractor real ridges to find, so the
    records being compared contain genuine transforms, gates and residuals.
    """
    import cv2

    rng = np.random.default_rng(seed)
    frame = np.full((720, 1280, 3), 70, dtype=np.int16)
    frame += rng.integers(-6, 6, frame.shape)
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    for feature in MarkingFeature:
        court = np.asarray(LAYOUT.marking(feature).sample(200), dtype=np.float64)
        image = project(TRUTH, court)
        if image is None or not np.all(np.isfinite(image)):
            continue
        cv2.polylines(
            frame, [image.reshape(-1, 1, 2).astype(np.int32)], False,
            (205, 205, 205), 3, cv2.LINE_AA,
        )
    return frame


def slot_positions():
    """Place the mapped slots where the real landmarks project.

    Using the evidence map's own slot/landmark pairing means the calibration
    that comes out is the genuine article -- five confident landmarks spanning
    19 ft, three of them collinear -- rather than a shape that merely parses.
    """
    evidence_map = load_registered_map("reloc2_18_provisional")
    positions = [(-500.0, -500.0)] * evidence_map.keypoint_count
    for group in evidence_map.groups:
        if group.tier != "confident":
            continue
        court = LAYOUT.coordinate(HalfCourtLandmark(group.landmark_id))
        image = project(TRUTH, np.array([court]))[0]
        for slot in group.slots:
            positions[slot] = (float(image[0]), float(image[1]))
    return positions


def detector_factory(*_args, **_kwargs):
    positions = slot_positions()

    def predict(_frame, _index):
        return [
            (x, y, 0.9 if x > 0 else 0.01)  # off-frame slots stay below the gate
            for x, y in positions
        ]

    return FakeDetector(predict, keypoint_count=len(positions))


SHADOW_FLAGS = ("--marking-shadow", "--no-marking-overlays")
HYBRID_FLAGS = ("--hybrid-shadow", "--no-marking-overlays", "--no-hybrid-overlays")


class HybridCliTests(unittest.TestCase):
    """Runs ``calibrate-frames`` end to end against planted detections.

    The three canonical runs happen once for the whole class. Each one drives
    the multiscale ridge extractor over full-size frames, which is far too
    expensive to repeat per assertion, and every test here reads the same
    artifacts anyway.
    """

    @classmethod
    def setUpClass(cls):
        cls._patches = []
        frames = [painted_court(seed=index) for index in range(FRAME_COUNT)]

        def source_factory(*_args, **_kwargs):
            return ArrayFrameSource(frames, source_id="clip.mp4")

        evidence_map = dataclasses.replace(
            load_registered_map("reloc2_18_provisional"), weights_sha256=DIGEST
        )
        cls._patch(cli, "VideoFrameSource", source_factory)
        cls._patch(cli, "YoloPoseDetector", detector_factory)
        cls._patch(cli, "load_registered_map", lambda _name: evidence_map)
        cls._patch(prov, "sha256_file", lambda _path: DIGEST)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)

        cls.plain = cls.run_cli("plain")
        cls.shadow = cls.run_cli("shadow", *SHADOW_FLAGS)
        cls.hybrid = cls.run_cli("hybrid", *HYBRID_FLAGS)
        cls.offline = Path(cls.tmp.name) / "offline"
        assert cli.main([
            "refine-shadow",
            "--shadow-run", str(cls.hybrid / "marking_shadow"),
            "--calibrations", str(cls.hybrid / "calibrations.jsonl"),
            "--out", str(cls.offline),
        ]) == 0

    @classmethod
    def _patch(cls, module, name, value):
        original = getattr(module, name)
        setattr(module, name, value)
        cls._patches.append((module, name, original))
        cls.addClassCleanup(lambda m=module, n=name, o=original: setattr(m, n, o))

    @classmethod
    def run_cli(cls, run_id, *extra):
        root = Path(cls.tmp.name) / run_id
        argv = [
            "calibrate-frames",
            "--weights", "fake.pt",
            "--video", "clip.mp4",
            "--out", str(root),
            "--run-id", run_id,
            "--frames", ",".join(str(i) for i in range(FRAME_COUNT)),
            "--no-reprojection",
            "--quiet",
            *extra,
        ]
        assert cli.main(argv) == 0
        return root / "clip" / run_id

    # -- C2 -------------------------------------------------------------

    def test_the_baseline_is_produced_at_all(self):
        """Guards the rest: a byte-comparison of two empty files proves nothing."""
        lines = (self.plain / "calibrations.jsonl").read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), FRAME_COUNT)
        self.assertEqual(
            {json.loads(line)["calibration"]["status"] for line in lines}, {"DEGRADED"}
        )

    def test_the_seed_class_is_the_one_the_shadow_path_accepts(self):
        """If these frames were ineligible the isolation checks below would be
        measuring a path that never ran."""
        record = json.loads((self.plain / "calibrations.jsonl").read_text().splitlines()[0])
        quality = record["calibration"]["quality"]
        self.assertEqual(quality["inlier_count"], 5)
        self.assertTrue(all(r.startswith("weak_conditioning:") for r in quality["reasons"]))

    def test_the_refiner_actually_produced_a_challenger(self):
        """The stronger guard: the compared records must contain a real fit.

        With a blank frame the extractor finds nothing, the refiner abstains,
        and every comparison below would pass while testing nothing.
        """
        record = json.loads((self.hybrid / "hybrid_shadow" / "records.jsonl").read_text().splitlines()[0])
        self.assertEqual(record["status"], "OK")
        self.assertIsNotNone(record["challenger_candidate_id"])
        self.assertTrue(record["fitted_features"])
        self.assertEqual(len(record["candidates"]), 2)

    def test_marking_shadow_does_not_change_calibrations(self):
        self.assertEqual(
            (self.plain / "calibrations.jsonl").read_bytes(),
            (self.shadow / "calibrations.jsonl").read_bytes(),
        )

    def test_hybrid_shadow_does_not_change_calibrations(self):
        """The milestone's done-condition, as bytes."""
        self.assertEqual(
            (self.plain / "calibrations.jsonl").read_bytes(),
            (self.hybrid / "calibrations.jsonl").read_bytes(),
        )

    def test_hybrid_shadow_writes_its_own_namespace(self):
        self.assertTrue((self.hybrid / "hybrid_shadow" / "records.jsonl").is_file())
        self.assertTrue((self.hybrid / "hybrid_shadow" / "run.json").is_file())

    def test_the_hybrid_namespace_never_contains_a_calibrations_file(self):
        self.assertEqual(list((self.hybrid / "hybrid_shadow").glob("calibrations.jsonl")), [])

    def test_hybrid_shadow_implies_marking_shadow(self):
        """The refiner consumes marking evidence; asking for one without the
        other would silently produce an empty namespace."""
        self.assertTrue((self.hybrid / "marking_shadow" / "records.jsonl").is_file())

    def test_every_frame_gets_a_hybrid_disposition(self):
        """A complete ledger: frames the refiner declined still appear."""
        lines = (self.hybrid / "hybrid_shadow" / "records.jsonl").read_text().strip().splitlines()
        self.assertEqual(len(lines), FRAME_COUNT)

    # -- C3 -------------------------------------------------------------

    def test_refine_shadow_reproduces_the_inline_records_exactly(self):
        """The offline path is how the refiner gets swept across Milestone 3's
        configuration variants without touching video again. Approximate
        agreement would make those sweeps incomparable."""
        self.assertEqual(
            (self.hybrid / "hybrid_shadow" / "records.jsonl").read_bytes(),
            (self.offline / "records.jsonl").read_bytes(),
        )

    def test_refine_shadow_needs_no_video_or_checkpoint(self):
        """It reads serialized records only -- which is what makes a sweep cheap.

        The fakes are lifted for the duration, so a hidden dependency on the
        detector or the frame source would surface as a failure here rather
        than being masked by the harness.
        """
        patched = [(module, name, getattr(module, name)) for module, name, _ in self._patches]
        for module, name, original in self._patches:
            setattr(module, name, original)
        try:
            target = Path(self.tmp.name) / "offline_unpatched"
            self.assertEqual(cli.main([
                "refine-shadow",
                "--shadow-run", str(self.hybrid / "marking_shadow"),
                "--calibrations", str(self.hybrid / "calibrations.jsonl"),
                "--out", str(target),
            ]), 0)
            self.assertEqual(
                (target / "records.jsonl").read_bytes(),
                (self.offline / "records.jsonl").read_bytes(),
            )
        finally:
            for module, name, fake in patched:
                setattr(module, name, fake)

    # -- config ---------------------------------------------------------

    def test_an_unknown_hybrid_config_key_is_rejected(self):
        path = Path(self.tmp.name) / "bad.json"
        path.write_text(json.dumps({"not_a_real_knob": 1}), encoding="utf-8")
        with self.assertRaises((SystemExit, ValueError)):
            self.run_cli("badcfg", *HYBRID_FLAGS, "--hybrid-config", str(path))


if __name__ == "__main__":
    unittest.main()
