"""The depth split has to actually separate interpolation from extrapolation.

A homography fitted to landmarks spanning 0-19 ft is well constrained there and
extrapolating everywhere beyond. The characteristic failure is a transform whose
error is invisible near the baseline and large at midcourt -- which is exactly what
``docs/CALIBRATION.md`` describes, and exactly what a pooled average hides, since
most annotated paint is near the baseline where the fit is best.

So the load-bearing test perturbs a known transform and requires the reported
error to *grow with depth*. If the bands cannot show that, the metric cannot
answer the question the milestone exists to ask, however plausible its numbers
look.
"""

import unittest

import numpy as np

from contracts.annotations import (
    AnnotatedFeature,
    FrameAnnotation,
    MarkingFeature,
    SamplePoint,
)
from contracts.court_layout import load_registered_layout

from calibration.estimator import project

from benchmark.geometry import marking_polylines
from benchmark.polyline import resample
from benchmark.metrics import (
    compare_summaries,
    score_frame,
    summarize_scores,
)
from benchmark.synthetic import broadcast_homography

LAYOUT = load_registered_layout("nba_halfcourt")


def annotation_tracing(matrix, features, samples=40) -> FrameAnnotation:
    """An annotation that traces exactly where ``matrix`` puts each marking.

    Sampled along the marking rather than at its vertices: a straight marking has
    only two vertices, so tracing those alone would put no sample in the middle of
    the court and the depth bands would have nothing to separate.
    """
    polylines = marking_polylines(LAYOUT)
    items = []
    for feature in features:
        court_points = resample(polylines[feature].points, samples)
        projected = project(matrix, court_points)
        items.append(
            AnnotatedFeature(
                feature=feature,
                points=tuple(SamplePoint(x=float(p[0]), y=float(p[1])) for p in projected),
            )
        )
    return FrameAnnotation(
        frame_id="synthetic_court_a",
        clip="synthetic",
        frame_index=0,
        image_width=1280,
        image_height=720,
        image_sha256="a" * 64,
        annotator_id="test",
        pass_id="silver",
        prompt_version="v1",
        coordinate_space="frame",
        features=tuple(items),
    )


TRUTH = broadcast_homography(LAYOUT)
INVERSE = np.linalg.inv(TRUTH)

DEPTH_SPANNING = [
    MarkingFeature.SIDELINE_FAR,  # 0-47 ft, the only marking covering every band
]

#: A court-space shear: ``(x, y) -> (x, y + 0.05x)``. It agrees with the truth at
#: the baseline and diverges linearly with depth -- the signature of a homography
#: constrained only by near-court evidence, which is what the current engine has.
#:
#: A *perspective* skew would not do. Warping along x maps the sideline onto
#: itself, so every sample stays on the marking and the error reads as zero no
#: matter how wrong the transform is elsewhere.
DEPTH_SHEAR = np.array([[1.0, 0.0, 0.0], [0.05, 1.0, 0.0], [0.0, 0.0, 1.0]])


class PerfectTransformTests(unittest.TestCase):
    def test_the_transform_that_drew_the_paint_scores_zero(self):
        annotation = annotation_tracing(TRUTH, DEPTH_SPANNING)
        score = score_frame(annotation, TRUTH, INVERSE, LAYOUT)
        self.assertTrue(score.scored)
        self.assertLess(score.median_px, 1e-6)
        self.assertLess(score.max_px, 1e-6)

    def test_every_band_is_reported_for_a_marking_spanning_the_court(self):
        annotation = annotation_tracing(TRUTH, DEPTH_SPANNING)
        score = score_frame(annotation, TRUTH, INVERSE, LAYOUT)
        self.assertEqual({b.band for b in score.bands}, {"0-19ft", "19-30ft", "30-47ft"})

    def test_court_error_is_reported_in_feet(self):
        annotation = annotation_tracing(TRUTH, DEPTH_SPANNING)
        score = score_frame(annotation, TRUTH, INVERSE, LAYOUT)
        self.assertLess(score.median_ft, 1e-6)


class DepthSensitivityTests(unittest.TestCase):
    """The measurement the whole milestone turns on."""

    def setUp(self):
        self.drifting = TRUTH @ DEPTH_SHEAR
        self.annotation = annotation_tracing(TRUTH, DEPTH_SPANNING)
        self.score = score_frame(
            self.annotation, self.drifting, np.linalg.inv(self.drifting), LAYOUT
        )

    def test_error_grows_with_court_depth(self):
        near = self.score.band("0-19ft")
        mid = self.score.band("19-30ft")
        far = self.score.band("30-47ft")
        self.assertLess(near.median_px, mid.median_px)
        self.assertLess(mid.median_px, far.median_px)

    def test_near_court_error_alone_would_look_healthy(self):
        """Why the split matters: the near band flatters a transform that drifts."""
        near = self.score.band("0-19ft")
        far = self.score.band("30-47ft")
        self.assertLess(near.median_px, 4.0)
        self.assertGreater(far.median_px, 4.0 * 2)

    def test_court_space_error_also_grows_with_depth(self):
        near = self.score.band("0-19ft")
        far = self.score.band("30-47ft")
        self.assertLess(near.median_ft, far.median_ft)

    def test_depth_assignment_is_stable_under_a_bad_transform(self):
        """Depth comes off the layout, so a drifting transform barely moves it.

        Not *exactly* unchanged: which part of a marking a sample corresponds to
        does depend on the transform, so a point sitting on a band boundary can
        cross it. What must not happen is a wholesale shift, which would let a bad
        transform relabel its far-court error as near-court.
        """
        good = score_frame(self.annotation, TRUTH, INVERSE, LAYOUT)
        counts_good = {b.band: b.sample_count for b in good.bands}
        counts_drift = {b.band: b.sample_count for b in self.score.bands}

        self.assertEqual(set(counts_good), set(counts_drift))
        self.assertEqual(sum(counts_good.values()), sum(counts_drift.values()))
        for band, count in counts_good.items():
            self.assertLessEqual(abs(count - counts_drift[band]), 2, band)


class RefusalTests(unittest.TestCase):
    def test_a_frame_with_no_transform_is_unscored_not_zero(self):
        """A refusal must not read as a perfect score."""
        annotation = annotation_tracing(TRUTH, DEPTH_SPANNING)
        score = score_frame(annotation, None, None, LAYOUT, status="UNCALIBRATED")
        self.assertFalse(score.scored)
        self.assertEqual(score.status, "UNCALIBRATED")
        self.assertIn("no transform", score.reason)

    def test_an_annotation_with_no_features_is_unscored(self):
        empty = annotation_tracing(TRUTH, [])
        score = score_frame(empty, TRUTH, INVERSE, LAYOUT)
        self.assertFalse(score.scored)

    def test_crop_space_annotation_is_refused(self):
        annotation = annotation_tracing(TRUTH, DEPTH_SPANNING)
        crop_space = FrameAnnotation.from_dict(
            {**annotation.as_dict(), "coordinate_space": "crop"}
        )
        with self.assertRaises(ValueError):
            score_frame(crop_space, TRUTH, INVERSE, LAYOUT)


class HeldOutTests(unittest.TestCase):
    def test_held_out_families_are_reported_separately(self):
        features = [MarkingFeature.LANE_EDGE_FAR, MarkingFeature.FREE_THROW_CIRCLE_FAR_HALF]
        annotation = annotation_tracing(TRUTH, features)
        score = score_frame(annotation, TRUTH, INVERSE, LAYOUT)
        self.assertTrue(score.per_feature["free_throw_circle_far_half"]["held_out"])
        self.assertFalse(score.per_feature["lane_edge_far"]["held_out"])
        self.assertTrue(np.isfinite(score.held_out_median_px))

    def test_held_out_families_can_be_excluded_from_the_fitted_score(self):
        features = [MarkingFeature.LANE_EDGE_FAR, MarkingFeature.FREE_THROW_CIRCLE_FAR_HALF]
        annotation = annotation_tracing(TRUTH, features)
        score = score_frame(annotation, TRUTH, INVERSE, LAYOUT, include_held_out=False)
        self.assertNotIn("free_throw_circle_far_half", score.per_feature)


class SummaryTests(unittest.TestCase):
    def test_summary_keeps_the_bands_apart(self):
        """Aggregating across frames must not flatten the depth signal."""
        annotation = annotation_tracing(TRUTH, DEPTH_SPANNING)
        drifting = TRUTH @ DEPTH_SHEAR
        inverse = np.linalg.inv(drifting)
        scores = [
            score_frame(annotation, drifting, inverse, LAYOUT, status="DEGRADED"),
            score_frame(annotation, drifting, inverse, LAYOUT, status="DEGRADED"),
        ]
        summary = summarize_scores(scores)
        self.assertEqual(summary["frames_scored"], 2)
        self.assertEqual(summary["status_counts"], {"DEGRADED": 2})
        self.assertLess(
            summary["bands"]["0-19ft"]["median_px"], summary["bands"]["19-30ft"]["median_px"]
        )
        self.assertLess(
            summary["bands"]["19-30ft"]["median_px"], summary["bands"]["30-47ft"]["median_px"]
        )

    def test_unscored_frames_are_counted_not_dropped(self):
        annotation = annotation_tracing(TRUTH, DEPTH_SPANNING)
        scores = [
            score_frame(annotation, TRUTH, INVERSE, LAYOUT, status="OK"),
            score_frame(annotation, None, None, LAYOUT, status="UNCALIBRATED"),
        ]
        summary = summarize_scores(scores)
        self.assertEqual(summary["frames"], 2)
        self.assertEqual(summary["frames_unscored"], 1)

    def test_comparison_reports_per_band_change(self):
        annotation = annotation_tracing(TRUTH, DEPTH_SPANNING)
        drifting = TRUTH @ DEPTH_SHEAR
        worse = summarize_scores(
            [score_frame(annotation, drifting, np.linalg.inv(drifting), LAYOUT)]
        )
        better = summarize_scores([score_frame(annotation, TRUTH, INVERSE, LAYOUT)])
        delta = compare_summaries(worse, better)
        self.assertLess(delta["bands"]["30-47ft"]["median_px_change_pct"], -25.0)


if __name__ == "__main__":
    unittest.main()
