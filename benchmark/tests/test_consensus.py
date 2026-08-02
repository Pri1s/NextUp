"""Consensus has to be conservative in the right direction.

Its failure modes are asymmetric. Wrongly queueing an agreeing feature costs an
adjudicator call. Wrongly accepting a disagreement puts a bad label into the
silver set, where it becomes the reference every later measurement is scored
against and is very hard to notice. So the tests lean on the accept path.

Three specific ways a plausible implementation goes wrong, each tested here:
averaging two traces that run in opposite directions, scoring a coverage
difference as a disagreement, and resolving an identity conflict arithmetically
instead of escalating it.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from contracts.annotations import (
    AnnotatedFeature,
    AnnotatedJunction,
    FrameAnnotation,
    JunctionPoint,
    MarkingFeature,
    SamplePoint,
    SkipReason,
    SkippedFeature,
)
from contracts.court_layout import load_registered_layout

from benchmark.consensus import (
    AGREE_MEDIAN_PX,
    build_consensus,
    compare_feature,
    find_identity_conflicts,
    merge_feature,
    summarize,
    write_consensus,
)

LAYOUT = load_registered_layout("nba_halfcourt")


def feature(kind: MarkingFeature, points, **kwargs) -> AnnotatedFeature:
    return AnnotatedFeature(
        feature=kind,
        points=tuple(SamplePoint(x=float(x), y=float(y)) for x, y in points),
        **kwargs,
    )


def annotation(pass_id: str, features=(), junctions=(), **overrides) -> FrameAnnotation:
    base = dict(
        frame_id="video_2_000141",
        clip="input_videos/video_2.mp4",
        frame_index=141,
        image_width=1280,
        image_height=720,
        image_sha256="e" * 64,
        annotator_id=f"annot-{pass_id}",
        pass_id=pass_id,
        prompt_version="annotator-1.0.0",
        coordinate_space="frame",
        features=tuple(features),
        junctions=tuple(junctions),
    )
    base.update(overrides)
    return FrameAnnotation(**base)


def line(x0, y0, x1, y1, n=9) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)[:, None]
    return np.array([x0, y0]) + t * np.array([x1 - x0, y1 - y0])


class CompareTests(unittest.TestCase):
    def test_identical_traces_agree(self):
        points = line(100, 500, 900, 480)
        result = compare_feature(
            feature(MarkingFeature.BASELINE, points), feature(MarkingFeature.BASELINE, points)
        )
        self.assertEqual(result.verdict, "agreed")
        self.assertAlmostEqual(result.median_px, 0.0, places=6)

    def test_small_offset_still_agrees(self):
        a = line(100, 500, 900, 480)
        result = compare_feature(
            feature(MarkingFeature.BASELINE, a),
            feature(MarkingFeature.BASELINE, a + np.array([0.0, 1.0])),
        )
        self.assertEqual(result.verdict, "agreed")

    def test_large_offset_disagrees(self):
        a = line(100, 500, 900, 480)
        result = compare_feature(
            feature(MarkingFeature.BASELINE, a),
            feature(MarkingFeature.BASELINE, a + np.array([0.0, 12.0])),
        )
        self.assertEqual(result.verdict, "disagreed")
        self.assertGreater(result.median_px, AGREE_MEDIAN_PX)

    def test_coverage_difference_is_not_a_disagreement(self):
        """One pass traced half the line the other did; they still agree on it."""
        result = compare_feature(
            feature(MarkingFeature.BASELINE, line(100, 500, 900, 480, 17)),
            feature(MarkingFeature.BASELINE, line(100, 500, 500, 490, 9)),
        )
        self.assertEqual(result.verdict, "agreed")

    def test_opposite_direction_traces_still_agree(self):
        points = line(100, 500, 900, 480)
        result = compare_feature(
            feature(MarkingFeature.BASELINE, points),
            feature(MarkingFeature.BASELINE, points[::-1]),
        )
        self.assertEqual(result.verdict, "agreed")

    def test_disjoint_traces_report_no_overlap(self):
        result = compare_feature(
            feature(MarkingFeature.BASELINE, line(100, 100, 200, 100)),
            feature(MarkingFeature.BASELINE, line(900, 600, 1000, 600)),
        )
        self.assertEqual(result.verdict, "no_overlap")


class MergeTests(unittest.TestCase):
    def test_merge_sits_between_the_two_traces(self):
        a = line(100, 500, 900, 500)
        merged = merge_feature(
            feature(MarkingFeature.BASELINE, a),
            feature(MarkingFeature.BASELINE, a + np.array([0.0, 2.0])),
        )
        ys = np.array([p.y for p in merged.points])
        self.assertTrue(np.allclose(ys, 501.0, atol=0.2))

    def test_reversed_trace_is_aligned_before_averaging(self):
        """Averaging a curve with its own reverse must return the curve itself.

        Without direction alignment this pairs each point with its opposite and
        collapses the curve toward its midpoint -- a result that lies on neither
        marking and looks like a perfectly ordinary annotation.
        """
        curve = np.stack(
            [np.linspace(100, 900, 21), 400 + 120 * np.sin(np.linspace(0, np.pi, 21))], axis=1
        )
        merged = merge_feature(
            feature(MarkingFeature.THREE_POINT_ARC, curve),
            feature(MarkingFeature.THREE_POINT_ARC, curve[::-1]),
        )
        points = np.array([[p.x, p.y] for p in merged.points])
        from benchmark.polyline import nearest_on_polyline

        self.assertLess(float(np.max(nearest_on_polyline(points, curve).distances)), 1.0)

    def test_merge_keeps_the_more_doubtful_uncertainty(self):
        a = line(100, 500, 900, 500)
        merged = merge_feature(
            feature(MarkingFeature.BASELINE, a, uncertainty_px=1.0),
            feature(MarkingFeature.BASELINE, a, uncertainty_px=4.0),
        )
        self.assertEqual(merged.uncertainty_px, 4.0)


class ConflictTests(unittest.TestCase):
    def test_same_paint_under_two_names_is_a_conflict(self):
        points = line(100, 500, 900, 480)
        conflicts = find_identity_conflicts(
            annotation("a", [feature(MarkingFeature.THREE_POINT_CORNER_FAR, points)]),
            annotation("b", [feature(MarkingFeature.SIDELINE_FAR, points)]),
        )
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].feature_a, MarkingFeature.THREE_POINT_CORNER_FAR)

    def test_genuinely_separate_markings_are_not_a_conflict(self):
        conflicts = find_identity_conflicts(
            annotation("a", [feature(MarkingFeature.SIDELINE_FAR, line(100, 200, 900, 190))]),
            annotation("b", [feature(MarkingFeature.BASELINE, line(100, 500, 900, 480))]),
        )
        self.assertEqual(conflicts, ())

    def test_one_stray_sample_near_another_marking_is_not_a_conflict(self):
        a = line(100, 500, 900, 480, 20)
        b = np.vstack([a[:1], line(100, 200, 900, 190, 19)])
        self.assertEqual(
            find_identity_conflicts(
                annotation("a", [feature(MarkingFeature.BASELINE, a)]),
                annotation("b", [feature(MarkingFeature.SIDELINE_FAR, b)]),
            ),
            (),
        )

    def test_conflicted_feature_never_reaches_the_silver_set(self):
        """§7.4: two labels that may describe different markings are not averaged."""
        points = line(100, 500, 900, 480)
        result = build_consensus(
            annotation("a", [
                feature(MarkingFeature.BASELINE, points),
                feature(MarkingFeature.SIDELINE_FAR, line(100, 200, 900, 190)),
            ]),
            annotation("b", [
                feature(MarkingFeature.BASELINE, points),
                feature(MarkingFeature.SIDELINE_FAR, points + np.array([0.0, 1.0])),
            ]),
            LAYOUT,
        )
        accepted = {item.feature for item in result.silver.features}
        self.assertNotIn(MarkingFeature.SIDELINE_FAR, accepted)
        self.assertIn(MarkingFeature.SIDELINE_FAR, result.disputed)


class ConsensusTests(unittest.TestCase):
    def test_agreeing_features_become_silver_and_others_are_queued(self):
        shared = line(100, 500, 900, 480)
        result = build_consensus(
            annotation("a", [
                feature(MarkingFeature.BASELINE, shared),
                feature(MarkingFeature.FREE_THROW_LINE, line(400, 300, 400, 450)),
            ]),
            annotation("b", [
                feature(MarkingFeature.BASELINE, shared + np.array([0.0, 0.5])),
                feature(MarkingFeature.FREE_THROW_LINE, line(430, 300, 430, 450)),
            ]),
            LAYOUT,
        )
        self.assertEqual(result.agreed_features, (MarkingFeature.BASELINE,))
        self.assertIn(MarkingFeature.FREE_THROW_LINE, result.disputed)

    def test_feature_found_by_only_one_pass_is_queued(self):
        result = build_consensus(
            annotation("a", [feature(MarkingFeature.BASELINE, line(100, 500, 900, 480))]),
            annotation("b", []),
            LAYOUT,
        )
        self.assertIn(MarkingFeature.BASELINE, result.disputed)
        self.assertEqual(result.agreements[0].verdict, "only_a")

    def test_feature_skipped_by_both_passes_is_not_a_disagreement(self):
        skipped = (SkippedFeature(MarkingFeature.CENTER_CIRCLE, SkipReason.OUT_OF_FRAME),)
        result = build_consensus(
            annotation("a", skipped=skipped),
            annotation("b", skipped=skipped),
            LAYOUT,
        )
        self.assertNotIn(MarkingFeature.CENTER_CIRCLE, result.disputed)
        self.assertEqual(result.silver.skipped[0].feature, MarkingFeature.CENTER_CIRCLE)

    def test_adjudication_queue_contains_only_actual_disagreements(self):
        shared = line(100, 500, 900, 480)
        agreed = build_consensus(
            annotation("a", [feature(MarkingFeature.BASELINE, shared)]),
            annotation("b", [feature(MarkingFeature.BASELINE, shared)]),
            LAYOUT,
        )
        disputed = build_consensus(
            annotation("a", [feature(MarkingFeature.FREE_THROW_LINE, shared)]),
            annotation("b", []),
            LAYOUT,
        )
        with tempfile.TemporaryDirectory() as tmp:
            _, _, queue_path = write_consensus(tmp, [agreed, disputed])
            queue = json.loads(Path(queue_path).read_text(encoding="utf-8"))
        self.assertEqual([item["frame_id"] for item in queue["frames"]], [disputed.frame_id])
        self.assertEqual(queue["frames"][0]["features"], ["free_throw_line"])
        self.assertNotIn("median_px", json.dumps(queue))

    def test_mismatched_prompt_versions_are_reported(self):
        """A prompt change invalidates earlier labels; comparing them hides that."""
        points = line(100, 500, 900, 480)
        result = build_consensus(
            annotation("a", [feature(MarkingFeature.BASELINE, points)]),
            annotation("b", [feature(MarkingFeature.BASELINE, points)],
                       prompt_version="annotator-2.0.0"),
            LAYOUT,
        )
        self.assertTrue(any("prompt version" in p for p in result.problems))

    def test_different_images_are_reported(self):
        points = line(100, 500, 900, 480)
        result = build_consensus(
            annotation("a", [feature(MarkingFeature.BASELINE, points)]),
            annotation("b", [feature(MarkingFeature.BASELINE, points)], image_sha256="f" * 64),
            LAYOUT,
        )
        self.assertTrue(any("different images" in p for p in result.problems))

    def test_crop_space_pass_is_refused(self):
        with self.assertRaises(ValueError):
            build_consensus(
                annotation("a", coordinate_space="crop"),
                annotation("b"),
                LAYOUT,
            )

    def test_junction_without_its_markings_is_dropped_from_silver(self):
        """The silver set must not assert a landmark it has no geometry for."""
        point = SamplePoint(300.0, 490.0)
        result = build_consensus(
            annotation("a", junctions=[AnnotatedJunction(JunctionPoint.LANE_BASELINE_FAR, point)]),
            annotation("b", junctions=[AnnotatedJunction(JunctionPoint.LANE_BASELINE_FAR, point)]),
            LAYOUT,
        )
        self.assertEqual(result.silver.junctions, ())

    def test_geometry_check_skips_without_enough_junctions(self):
        result = build_consensus(
            annotation("a", [feature(MarkingFeature.BASELINE, line(100, 500, 900, 480))]),
            annotation("b", [feature(MarkingFeature.BASELINE, line(100, 500, 900, 480))]),
            LAYOUT,
        )
        self.assertEqual(result.geometry_check["status"], "skipped")

    def test_summary_counts_verdicts(self):
        shared = line(100, 500, 900, 480)
        result = build_consensus(
            annotation("a", [feature(MarkingFeature.BASELINE, shared)]),
            annotation("b", [feature(MarkingFeature.BASELINE, shared)]),
            LAYOUT,
        )
        summary = summarize([result])
        self.assertEqual(summary["agreement_rate"], 1.0)
        self.assertEqual(summary["identity_conflicts"], 0)


if __name__ == "__main__":
    unittest.main()
