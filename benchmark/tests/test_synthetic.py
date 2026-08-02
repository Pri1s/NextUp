"""The ground-truth probe is only useful if its ground truth is actually true.

Everything the probe concludes rests on the claim that the stored centerlines
describe where the renderer put the paint. If they drift, the probe reports a bias
that does not exist and would send the milestone down the wrong fallback -- or
worse, hide one that does.

So the load-bearing test samples the rendered image *along* each stored centerline
and requires it to be brighter than the floor a few pixels to either side. That
checks the renderer and the truth against each other rather than either against
its own assumptions.

The bias tests matter for the other half of the probe's job. Consensus between two
passes cannot see a shared offset; ``score_against_truth`` must, so a planted
offset has to come back with the right size *and the right sign*.
"""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from contracts.annotations import (
    AnnotatedFeature,
    FrameAnnotation,
    MarkingFeature,
    SamplePoint,
)
from contracts.court_layout import load_registered_layout

from benchmark.polyline import nearest_on_polyline
from benchmark.synthetic import (
    broadcast_homography,
    load_truth,
    render_scene,
    score_against_truth,
    write_scene,
)


def offset_along_normal(points: np.ndarray, distance: float) -> np.ndarray:
    """Displace a curve by ``distance`` perpendicular to itself, left of travel.

    A fixed displacement in x or y is nearly tangential wherever the curve happens
    to run that way, so it would plant a bias of unknown size. Offsetting along
    the local normal plants exactly the bias the test then asks for back.
    """
    tangents = np.gradient(points, axis=0)
    norms = np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-9)
    unit = tangents / norms
    normal = np.stack([-unit[:, 1], unit[:, 0]], axis=1)
    return points + normal * distance


def annotation_from(truth: dict, features, normal_offset=0.0, stride=7) -> FrameAnnotation:
    """Build a frame-space annotation that traces the truth, optionally displaced."""
    items = []
    for feature in features:
        points = truth[feature]
        if normal_offset:
            points = offset_along_normal(points, normal_offset)
        points = points[::stride]
        items.append(
            AnnotatedFeature(
                feature=feature,
                points=tuple(SamplePoint(x=float(p[0]), y=float(p[1])) for p in points),
            )
        )
    return FrameAnnotation(
        frame_id="synthetic_court_a",
        clip="synthetic",
        frame_index=0,
        image_width=1280,
        image_height=720,
        image_sha256="c" * 64,
        annotator_id="test",
        pass_id="a",
        prompt_version="v1",
        coordinate_space="frame",
        features=tuple(items),
    )


class SceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layout = load_registered_layout("nba_halfcourt")
        cls.scene = render_scene(cls.layout)
        cls.gray = cv2.cvtColor(cls.scene.image, cv2.COLOR_BGR2GRAY).astype(np.float64)

    def _sample(self, points: np.ndarray) -> np.ndarray:
        height, width = self.gray.shape
        inside = (
            (points[:, 0] >= 6) & (points[:, 0] < width - 6)
            & (points[:, 1] >= 6) & (points[:, 1] < height - 6)
        )
        kept = points[inside]
        if len(kept) == 0:
            return np.zeros(0)
        return self.gray[np.round(kept[:, 1]).astype(int), np.round(kept[:, 0]).astype(int)]

    def test_stored_centerlines_land_on_painted_pixels(self):
        """The claim the whole probe rests on, checked against the pixels."""
        for feature in self.scene.visible():
            points = self.scene.truth[feature]
            on_line = self._sample(points)
            if len(on_line) < 5:
                continue

            tangents = np.gradient(points, axis=0)
            norms = np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-9)
            normal = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1) / norms
            beside = np.concatenate([self._sample(points + normal * 5), self._sample(points - normal * 5)])

            with self.subTest(feature=feature.value):
                self.assertGreater(
                    float(np.median(on_line)),
                    float(np.median(beside)) + 15.0,
                    f"{feature.value}: centerline is not on the paint",
                )

    def test_paint_narrows_with_distance(self):
        """Constant-pixel-width lines would misrepresent how hard far court is.

        The lane's baseline end is much nearer the camera than its free-throw end,
        so its projected stripe must be measurably wider there.
        """
        matrix = self.scene.h_court_to_image
        from calibration.estimator import project

        lane_y = self.layout.marking(MarkingFeature.LANE_EDGE_FAR).start[1]
        width_ft = self.layout.marking(MarkingFeature.LANE_EDGE_FAR).width_ft
        near = project(matrix, np.array([[0.5, lane_y], [0.5, lane_y + width_ft]]))
        far = project(matrix, np.array([[18.5, lane_y], [18.5, lane_y + width_ft]]))
        near_width = float(np.linalg.norm(near[1] - near[0]))
        far_width = float(np.linalg.norm(far[1] - far[0]))
        self.assertGreater(near_width, far_width)

    def test_visibility_excludes_features_outside_the_frame(self):
        visible = self.scene.visible()
        self.assertIn(MarkingFeature.THREE_POINT_ARC, visible)
        self.assertLess(len(visible), len(self.scene.truth))

    def test_rendering_is_deterministic(self):
        again = render_scene(self.layout)
        self.assertTrue(np.array_equal(again.image, self.scene.image))

    def test_a_different_homography_produces_a_different_scene(self):
        other = broadcast_homography(self.layout, 1280, 720) @ np.diag([1.0, 1.0, 1.08])
        moved = render_scene(self.layout, matrix=other)
        self.assertFalse(np.array_equal(moved.image, self.scene.image))

    def test_truth_round_trips_through_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_scene(self.scene, tmp)
            restored = load_truth(Path(tmp) / "truth.json")
        self.assertEqual(set(restored), set(self.scene.truth))
        for feature, points in restored.items():
            self.assertTrue(np.allclose(points, self.scene.truth[feature]))

    def test_truth_is_written_outside_the_frame_tree(self):
        """An annotator pointed at the frames must not be able to read the answer."""
        with tempfile.TemporaryDirectory() as tmp:
            write_scene(self.scene, tmp)
            inside = list((Path(tmp) / "frames").rglob("*"))
            self.assertFalse([p for p in inside if "truth" in p.name])


class ScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layout = load_registered_layout("nba_halfcourt")
        cls.scene = render_scene(cls.layout)
        cls.features = sorted(
            cls.scene.visible() & {MarkingFeature.THREE_POINT_ARC, MarkingFeature.LANE_EDGE_FAR},
            key=lambda f: f.value,
        )

    def test_tracing_the_truth_scores_near_zero(self):
        annotation = annotation_from(self.scene.truth, self.features)
        _, summary = score_against_truth(annotation, self.scene.truth, self.scene.visible())
        self.assertLess(summary["median_px"], 0.01)

    def test_planted_offset_is_recovered_with_its_size_and_sign(self):
        """The shared-bias measurement, checked: both magnitude and direction.

        A 3 px offset along each curve's own normal must come back as 3 px of
        signed offset, and reversing it must reverse the sign. This is the one
        measurement consensus between two passes can never make.
        """
        planted = 3.0
        left = annotation_from(self.scene.truth, self.features, normal_offset=planted)
        right = annotation_from(self.scene.truth, self.features, normal_offset=-planted)

        per_left, summary_left = score_against_truth(
            left, self.scene.truth, self.scene.visible()
        )
        per_right, _ = score_against_truth(right, self.scene.truth, self.scene.visible())

        self.assertAlmostEqual(summary_left["median_px"], planted, places=1)
        for a, b in zip(per_left, per_right):
            with self.subTest(feature=a.feature.value):
                self.assertAlmostEqual(abs(a.mean_signed_px), planted, places=1)
                self.assertAlmostEqual(a.mean_signed_px, -b.mean_signed_px, places=1)

    def test_a_bias_below_the_gate_is_distinguishable_from_one_above_it(self):
        """The gate is 0.75 px; the measurement has to resolve either side of it."""
        small, _ = score_against_truth(
            annotation_from(self.scene.truth, self.features, normal_offset=0.3),
            self.scene.truth,
            self.scene.visible(),
        )
        large, _ = score_against_truth(
            annotation_from(self.scene.truth, self.features, normal_offset=1.5),
            self.scene.truth,
            self.scene.visible(),
        )
        self.assertTrue(all(abs(f.mean_signed_px) < 0.75 for f in small))
        self.assertTrue(all(abs(f.mean_signed_px) > 0.75 for f in large))

    def test_random_noise_leaves_signed_offset_near_zero(self):
        """Noise must not read as bias, or the gate fires on a healthy annotator."""
        rng = np.random.default_rng(3)
        truth = self.scene.truth
        jittered = {
            f: truth[f] + rng.normal(0.0, 2.0, truth[f].shape) for f in self.features
        }
        annotation = annotation_from(jittered, self.features, stride=1)
        per_feature, _ = score_against_truth(annotation, truth, self.scene.visible())
        for item in per_feature:
            self.assertLess(abs(item.mean_signed_px), 0.75, item.feature.value)

    def test_coverage_counts_only_visible_features(self):
        annotation = annotation_from(self.scene.truth, self.features)
        _, summary = score_against_truth(annotation, self.scene.truth, self.scene.visible())
        self.assertEqual(summary["features_present_in_truth"], len(self.scene.visible()))
        self.assertLess(summary["coverage"], 1.0)

    def test_incomplete_probe_annotation_reports_missing_dispositions(self):
        annotation = annotation_from(self.scene.truth, self.features)
        _, summary = score_against_truth(annotation, self.scene.truth, self.scene.visible())
        self.assertTrue(summary["missing_dispositions"])

    def test_feature_absent_from_truth_is_flagged_not_scored(self):
        """An annotator inventing a marking must be visible, not silently averaged."""
        truth = dict(self.scene.truth)
        invented = MarkingFeature.CENTER_CIRCLE
        annotation = annotation_from(truth, [invented])
        truth.pop(invented)
        _, summary = score_against_truth(annotation, truth, frozenset(truth))
        self.assertIn(invented.value, summary["annotated_features_absent_from_truth"])
        self.assertEqual(summary["features_scored"], 0)

    def test_crop_space_annotation_is_refused(self):
        annotation = annotation_from(self.scene.truth, self.features)
        crop_space = FrameAnnotation.from_dict(
            {**annotation.as_dict(), "coordinate_space": "crop"}
        )
        with self.assertRaises(ValueError):
            score_against_truth(crop_space, self.scene.truth)


if __name__ == "__main__":
    unittest.main()
