"""Scoring the two shadow candidates, from serialized records alone.

The scorer must not be able to see what it is measuring: it reads
``HybridRefinementFrame`` records and blind human annotations, and it reaches the
engine only through the model-free modules the blindness wall permits. It also
must not reimplement the metric -- it composes ``score_calibration_entry`` so
that a keypoint candidate scored here and the same transform scored through the
ordinary path cannot disagree.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from benchmark.geometry import marking_polylines
from benchmark.hybrid import (
    HYBRID_SCORE_SCHEMA_VERSION,
    load_hybrid_transforms,
    score_hybrid,
)
from benchmark.metrics import score_frame
from benchmark.polyline import resample
from benchmark.synthetic import broadcast_homography
from calibration.estimator import project
from contracts.annotations import AnnotatedFeature, FrameAnnotation, SamplePoint
from contracts.court_layout import load_registered_layout
from contracts.hybrid_types import (
    CandidateQuality,
    CandidateSource,
    CandidateTransform,
    GateResult,
    HybridFrameStatus,
    HybridRefinementFrame,
    SelectionDecision,
    SelectionOutcome,
)
from contracts.markings import MarkingFeature

LAYOUT = load_registered_layout("nba_halfcourt")
TRUTH = broadcast_homography(LAYOUT, 1280, 720)

#: Chosen to populate all three depth bands. The lane and free-throw line sit
#: inside 19 ft, the arc reaches 29, and the midcourt line lands in 30-47 -- so a
#: missing band in the report means the banding broke, not that the reference
#: happened to stop short.
TRACED = (
    MarkingFeature.LANE_EDGE_FAR,
    MarkingFeature.FREE_THROW_LINE,
    MarkingFeature.THREE_POINT_ARC,
    MarkingFeature.MIDCOURT_LINE,
)
FRAMES = ("video_1_000012", "video_2_000018")


def shifted(matrix, dx, dy):
    """Compose an image-space translation, so error is controllable and uniform."""
    nudge = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])
    result = nudge @ np.asarray(matrix, dtype=np.float64)
    return result / result[2, 2]


def as_tuple(matrix):
    return tuple(tuple(float(v) for v in row) for row in np.asarray(matrix))


def inverse_of(matrix):
    result = np.linalg.inv(np.asarray(matrix, dtype=np.float64))
    return as_tuple(result / result[2, 2])


def annotation(frame_id, image_sha256="a" * 64):
    """A blind reference tracing exactly where TRUTH puts each marking."""
    polylines = marking_polylines(LAYOUT)
    features = []
    for feature in TRACED:
        court = resample(polylines[feature].points, 40)
        image = project(TRUTH, court)
        features.append(AnnotatedFeature(
            feature=feature,
            points=tuple(SamplePoint(x=float(p[0]), y=float(p[1])) for p in image),
        ))
    return FrameAnnotation(
        frame_id=frame_id, clip=frame_id.rsplit("_", 1)[0], frame_index=int(frame_id[-6:]),
        image_width=1280, image_height=720, image_sha256=image_sha256,
        annotator_id="test", pass_id="silver", prompt_version="v1",
        coordinate_space="frame", features=tuple(features),
    )


def transform(candidate_id, source, matrix, layout_hash=None):
    return CandidateTransform(
        candidate_id=candidate_id, source=source, layout_id=LAYOUT.layout_id,
        layout_hash=layout_hash or LAYOUT.content_hash(),
        h_court_to_image=as_tuple(matrix), h_image_to_court=inverse_of(matrix),
        round_trip_error_px=0.0, solver="test",
    )


def quality(candidate_id, source, held_out=None):
    return CandidateQuality(
        candidate_id=candidate_id, source=source, usable=True,
        model_point_px_median=1.0, model_point_px_p95=2.0, primitives=(),
        supported_families=(), max_depth_ft=29.0 if source is CandidateSource.HYBRID else 19.0,
        held_out_px_median=held_out, leave_one_primitive_shift_px=1.0,
        gates=(GateResult("candidate.round_trip_px", True, 0.0, 1.0, "px"),), reasons=(),
    )


def record(frame_id, keypoint_matrix, hybrid_matrix, image_sha256="a" * 64,
           layout_hash=None, extra=(), held_out=1.0):
    baseline_id = f"{frame_id}:keypoint"
    challenger_id = f"{frame_id}:hybrid"
    transforms = [
        transform(baseline_id, CandidateSource.KEYPOINT, keypoint_matrix, layout_hash),
        transform(challenger_id, CandidateSource.HYBRID, hybrid_matrix, layout_hash),
    ]
    candidates = [
        quality(baseline_id, CandidateSource.KEYPOINT),
        quality(challenger_id, CandidateSource.HYBRID, held_out=held_out),
    ]
    for cid, source, matrix in extra:
        transforms.append(transform(cid, source, matrix, layout_hash))
        candidates.append(quality(cid, source))
    return HybridRefinementFrame(
        frame_id=frame_id, image_sha256=image_sha256, status=HybridFrameStatus.OK,
        eligible=True, eligibility_reasons=(), layout_id=LAYOUT.layout_id,
        layout_hash=layout_hash or LAYOUT.content_hash(),
        shadow_config_hash="c" * 64, refiner_config_hash="d" * 64,
        baseline_candidate_id=baseline_id, challenger_candidate_id=challenger_id,
        transforms=tuple(transforms), candidates=tuple(candidates),
        selection=SelectionDecision(
            frame_id, baseline_id, challenger_id, baseline_id,
            SelectionOutcome.KEEP_BASELINE,
            (GateResult("selection.shadow_mode", True),), ("shadow_mode",),
        ),
        fitted_features=(MarkingFeature.THREE_POINT_ARC,),
        held_out_features=(MarkingFeature.FREE_THROW_CIRCLE_FAR_HALF,),
        converged=True,
    )


class Harness:
    """Writes records and references to disk, as the CLI would."""

    def __init__(self, records, annotations=None):
        self.dir = tempfile.TemporaryDirectory()
        root = Path(self.dir.name)
        (root / "refs").mkdir()
        with open(root / "records.jsonl", "w", encoding="utf-8") as handle:
            for item in records:
                handle.write(json.dumps(item.as_dict(), sort_keys=True) + "\n")
        for item in (annotations if annotations is not None else [annotation(r.frame_id) for r in records]):
            (root / "refs" / f"{item.frame_id}.json").write_text(
                json.dumps(item.as_dict()), encoding="utf-8"
            )
        self.records = root / "records.jsonl"
        self.references = root / "refs"

    def score(self, **kwargs):
        return score_hybrid(self.records, self.references, LAYOUT, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.dir.cleanup()


def improving_records(keypoint_px=8.0, hybrid_px=1.0):
    return [
        record(frame_id, shifted(TRUTH, keypoint_px, 0.0), shifted(TRUTH, hybrid_px, 0.0))
        for frame_id in FRAMES
    ]


class ReportShapeTests(unittest.TestCase):
    def test_it_runs_and_declares_its_schema(self):
        with Harness(improving_records()) as harness:
            report = harness.score()
        self.assertEqual(report["schema_version"], HYBRID_SCORE_SCHEMA_VERSION)
        self.assertEqual(report["records"], len(FRAMES))

    def test_both_candidates_are_summarised(self):
        with Harness(improving_records()) as harness:
            report = harness.score()
        self.assertIn(CandidateSource.KEYPOINT.value, report["summaries"])
        self.assertIn(CandidateSource.HYBRID.value, report["summaries"])

    def test_the_comparison_carries_every_depth_band(self):
        with Harness(improving_records()) as harness:
            report = harness.score()
        for band in ("0-19ft", "19-30ft", "30-47ft"):
            self.assertIn(band, report["comparisons"]["bands"])

    def test_transforms_can_be_loaded_by_source(self):
        with Harness(improving_records()) as harness:
            loaded = load_hybrid_transforms(harness.records)
        self.assertEqual(sorted(loaded), sorted(FRAMES))
        for entry in loaded.values():
            self.assertEqual(sorted(entry), ["hybrid", "keypoint"])


class DelegationTests(unittest.TestCase):
    def test_keypoint_numbers_match_the_ordinary_scoring_path(self):
        """It must compose the metric, not reimplement it.

        Two implementations that could disagree would make a scorer defect
        indistinguishable from a calibration defect.
        """
        matrix = shifted(TRUTH, 8.0, 0.0)
        with Harness(improving_records()) as harness:
            report = harness.score()
        direct = score_frame(
            annotation(FRAMES[0]), as_tuple(matrix), inverse_of(matrix), LAYOUT,
        )
        self.assertAlmostEqual(
            report["summaries"]["keypoint"]["median_px"], direct.median_px, places=9
        )


class FrameSetTests(unittest.TestCase):
    def test_both_summaries_cover_the_identical_frame_set(self):
        """Otherwise the comparison is confounded by which frames abstained."""
        with Harness(improving_records()) as harness:
            report = harness.score()
        self.assertEqual(
            report["summaries"]["keypoint"]["frames_scored"],
            report["summaries"]["hybrid"]["frames_scored"],
        )
        self.assertEqual(report["frame_sets"]["paired"], sorted(FRAMES))


class IntegrityTests(unittest.TestCase):
    def test_a_foreign_layout_hash_is_refused(self):
        records = [record(FRAMES[0], shifted(TRUTH, 8.0, 0.0), shifted(TRUTH, 1.0, 0.0),
                          layout_hash="f" * 64)]
        with Harness(records) as harness:
            with self.assertRaises(ValueError) as caught:
                harness.score()
        self.assertIn("layout_hash", str(caught.exception))

    def test_an_image_hash_disagreement_is_refused(self):
        records = [record(FRAMES[0], shifted(TRUTH, 8.0, 0.0), shifted(TRUTH, 1.0, 0.0),
                          image_sha256="b" * 64)]
        with Harness(records, annotations=[annotation(FRAMES[0], "a" * 64)]) as harness:
            with self.assertRaises(ValueError) as caught:
                harness.score()
        self.assertIn("image_sha256", str(caught.exception))

    def test_a_missing_reference_is_refused(self):
        records = improving_records()
        with Harness(records, annotations=[annotation(FRAMES[0])]) as harness:
            with self.assertRaises(ValueError) as caught:
                harness.score()
        self.assertIn("no human reference", str(caught.exception))


class AcceptanceTests(unittest.TestCase):
    def test_a_large_improvement_is_recognised(self):
        with Harness(improving_records(keypoint_px=8.0, hybrid_px=1.0)) as harness:
            acceptance = harness.score()["acceptance"]
        self.assertGreater(acceptance["median_improvement_beyond_19ft_pct"], 25.0)
        self.assertTrue(acceptance["meets_25pct"])

    def test_a_marginal_improvement_is_not_dressed_up(self):
        with Harness(improving_records(keypoint_px=8.0, hybrid_px=7.5)) as harness:
            acceptance = harness.score()["acceptance"]
        self.assertLess(acceptance["median_improvement_beyond_19ft_pct"], 25.0)
        self.assertFalse(acceptance["meets_25pct"])

    def test_a_gross_regression_is_flagged(self):
        """A hybrid far worse than the baseline must never read as acceptable."""
        with Harness(improving_records(keypoint_px=1.0, hybrid_px=40.0)) as harness:
            acceptance = harness.score()["acceptance"]
        self.assertTrue(acceptance["gross_failures"])
        self.assertFalse(acceptance["meets_zero_gross"])

    def test_a_clean_improvement_reports_no_gross_failure(self):
        with Harness(improving_records()) as harness:
            acceptance = harness.score()["acceptance"]
        self.assertEqual(acceptance["gross_failures"], [])
        self.assertTrue(acceptance["meets_zero_gross"])

    def test_held_out_error_is_read_off_the_records(self):
        with Harness(improving_records()) as harness:
            acceptance = harness.score()["acceptance"]
        self.assertTrue(acceptance["meets_10px"])


class RollupTests(unittest.TestCase):
    def test_gate_outcomes_are_counted_per_source(self):
        with Harness(improving_records()) as harness:
            rollups = harness.score()["rollups"]
        self.assertIn("keypoint:candidate.round_trip_px", rollups["gate_pass_counts"])
        self.assertIn("hybrid:candidate.round_trip_px", rollups["gate_pass_counts"])

    def test_frame_dispositions_are_counted(self):
        with Harness(improving_records()) as harness:
            rollups = harness.score()["rollups"]
        self.assertEqual(rollups["abstention_histogram"].get("OK"), len(FRAMES))

    def test_the_summary_reports_real_dispositions_not_unknown(self):
        """The disposition lives on the record, not the transform.

        Without carrying it across, every scored frame read ``UNKNOWN`` and the
        status counts told a reader nothing they could act on.
        """
        with Harness(improving_records()) as harness:
            summaries = harness.score()["summaries"]
        for source in ("keypoint", "hybrid"):
            with self.subTest(source=source):
                counts = summaries[source]["status_counts"]
                self.assertEqual(counts.get("OK"), len(FRAMES))
                self.assertNotIn("UNKNOWN", counts)

    def test_clips_are_broken_out_for_the_rotate_held_out_protocol(self):
        with Harness(improving_records()) as harness:
            report = harness.score()
        self.assertEqual(sorted(report["per_clip"]), ["video_1", "video_2"])


class ExtensibilityTests(unittest.TestCase):
    def test_a_cv_only_candidate_is_summarised_without_a_code_change(self):
        """Section 8.2's third column must not require touching the scorer."""
        records = [
            record(frame_id, shifted(TRUTH, 8.0, 0.0), shifted(TRUTH, 1.0, 0.0),
                   extra=((f"{frame_id}:cv", CandidateSource.CV_ONLY, shifted(TRUTH, 3.0, 0.0)),))
            for frame_id in FRAMES
        ]
        with Harness(records) as harness:
            report = harness.score()
        self.assertIn(CandidateSource.CV_ONLY.value, report["summaries"])


if __name__ == "__main__":
    unittest.main()
