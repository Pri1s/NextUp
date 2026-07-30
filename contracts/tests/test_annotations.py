"""The annotation schema has to refuse the mistakes an annotator will actually make.

Three of those mistakes are silent if the schema permits them, and each corrupts
the benchmark in a way no downstream metric can detect:

* submitting where visible paint *stops* as though it were a court landmark;
* mixing crop-space and frame-space coordinates, which differ by a factor of the
  crop scale and so produce a plausible-looking number in the wrong place;
* tracing along two different markings under one identity, which a homography
  cannot represent but a least-squares fit will happily average.

The geometric checks lean on one fact: a homography maps straight lines to
straight lines and circles to conics without inflection. A projected court marking
therefore stays straight (or keeps turning one way) no matter where the camera is,
so a violation is evidence about the *label*, never about perspective.
"""

import unittest

from contracts.annotations import (
    ANNOTATION_SCHEMA_VERSION,
    LINE_CONVENTION,
    AnnotatedFeature,
    AnnotatedJunction,
    FrameAnnotation,
    JunctionPoint,
    MarkingFeature,
    SamplePoint,
    SkipReason,
    SkippedFeature,
    Visibility,
    validate,
)


def frame_points(*pairs) -> tuple[SamplePoint, ...]:
    return tuple(SamplePoint(x=x, y=y) for x, y in pairs)


def crop_points(*pairs) -> tuple[SamplePoint, ...]:
    return tuple(SamplePoint(x=x, y=y, crop_id=f"c{i}") for i, (x, y) in enumerate(pairs))


def annotation(**overrides) -> FrameAnnotation:
    base = dict(
        frame_id="video_2_000119",
        clip="video_2.mp4",
        frame_index=119,
        image_width=1280,
        image_height=720,
        image_sha256="a" * 64,
        annotator_id="test",
        pass_id="a",
        prompt_version="v1",
        coordinate_space="frame",
    )
    base.update(overrides)
    return FrameAnnotation(**base)


class VocabularyTests(unittest.TestCase):
    def test_curved_features_declare_themselves_arcs(self):
        self.assertEqual(MarkingFeature.THREE_POINT_ARC.kind, "arc")
        self.assertEqual(MarkingFeature.BASELINE.kind, "polyline")

    def test_every_junction_maps_to_two_distinct_markings(self):
        """The mapping is what makes "an endpoint is not a landmark" enforceable."""
        for junction in JunctionPoint:
            a, b = junction.crossing
            self.assertNotEqual(a, b, f"{junction.value} crosses itself")

    def test_line_convention_is_frozen_into_every_annotation(self):
        self.assertEqual(annotation().line_convention, LINE_CONVENTION)

    def test_annotation_declaring_a_foreign_convention_is_refused(self):
        """A convention change must invalidate old files, not reinterpret them."""
        with self.assertRaises(ValueError):
            annotation(line_convention="rule_book_edge")


class StructuralTests(unittest.TestCase):
    def test_clean_annotation_has_no_problems(self):
        subject = annotation(
            features=(
                AnnotatedFeature(MarkingFeature.BASELINE, frame_points((100, 500), (900, 480))),
                AnnotatedFeature(
                    MarkingFeature.LANE_EDGE_FAR, frame_points((300, 490), (420, 300))
                ),
            ),
            junctions=(
                AnnotatedJunction(JunctionPoint.LANE_BASELINE_FAR, SamplePoint(300, 490)),
            ),
        )
        self.assertEqual(validate(subject), [])

    def test_junction_without_both_crossing_markings_is_rejected(self):
        """The core §5.4 rule: a landmark needs two traced markings under it."""
        subject = annotation(
            features=(
                AnnotatedFeature(MarkingFeature.BASELINE, frame_points((100, 500), (900, 480))),
            ),
            junctions=(
                AnnotatedJunction(JunctionPoint.LANE_BASELINE_FAR, SamplePoint(300, 490)),
            ),
        )
        problems = validate(subject)
        self.assertTrue(any("lane_edge_far" in p for p in problems), problems)

    def test_arc_needs_more_samples_than_a_line(self):
        subject = annotation(
            features=(
                AnnotatedFeature(
                    MarkingFeature.THREE_POINT_ARC, frame_points((100, 400), (200, 380))
                ),
            )
        )
        self.assertTrue(any("at least 5" in p for p in validate(subject)), validate(subject))

    def test_feature_cannot_be_both_annotated_and_skipped(self):
        subject = annotation(
            features=(
                AnnotatedFeature(MarkingFeature.BASELINE, frame_points((100, 500), (900, 480))),
            ),
            skipped=(SkippedFeature(MarkingFeature.BASELINE, SkipReason.NOT_VISIBLE),),
        )
        self.assertTrue(any("both annotated and skipped" in p for p in validate(subject)))

    def test_strict_validation_rejects_an_incomplete_annotation(self):
        subject = annotation(
            features=(
                AnnotatedFeature(MarkingFeature.BASELINE, frame_points((100, 500), (900, 480))),
            )
        )
        problems = validate(subject, require_complete=True)
        self.assertTrue(any("incomplete annotation" in problem for problem in problems))
        self.assertTrue(any("center_circle" in problem for problem in problems))

    def test_strict_validation_accepts_an_explicit_disposition_for_every_feature(self):
        subject = annotation(
            skipped=tuple(
                SkippedFeature(feature, SkipReason.NOT_VISIBLE) for feature in MarkingFeature
            )
        )
        self.assertEqual(validate(subject, require_complete=True), [])

    def test_occluded_index_must_sit_between_two_samples(self):
        subject = annotation(
            features=(
                AnnotatedFeature(
                    MarkingFeature.BASELINE,
                    frame_points((100, 500), (900, 480)),
                    occluded_after=(5,),
                ),
            )
        )
        self.assertTrue(any("occluded_after" in p for p in validate(subject)))

    def test_points_outside_the_image_are_rejected(self):
        subject = annotation(
            features=(
                AnnotatedFeature(MarkingFeature.BASELINE, frame_points((100, 500), (5000, 480))),
            )
        )
        self.assertTrue(any("outside the image" in p for p in validate(subject)))


class CoordinateSpaceTests(unittest.TestCase):
    def test_crop_space_point_without_a_crop_id_is_rejected(self):
        subject = annotation(
            coordinate_space="crop",
            features=(
                AnnotatedFeature(MarkingFeature.BASELINE, frame_points((100, 500), (900, 480))),
            ),
        )
        self.assertTrue(any("no crop_id" in p for p in validate(subject)))

    def test_frame_space_point_still_carrying_a_crop_id_is_rejected(self):
        """A resolved file with a stray crop_id means resolution was partial."""
        subject = annotation(
            features=(
                AnnotatedFeature(MarkingFeature.BASELINE, crop_points((100, 500), (900, 480))),
            ),
        )
        self.assertTrue(any("still carries a crop_id" in p for p in validate(subject)))

    def test_geometry_is_not_judged_in_crop_space(self):
        """Points from different crops share no frame; straightness is meaningless.

        These coordinates would fail the straightness check outright in frame
        space. In crop space the check must stay silent rather than report
        confident nonsense.
        """
        subject = annotation(
            coordinate_space="crop",
            features=(
                AnnotatedFeature(
                    MarkingFeature.BASELINE, crop_points((0, 0), (500, 900), (1000, 10))
                ),
            ),
        )
        self.assertEqual([p for p in validate(subject) if "straight" in p], [])


class GeometryTests(unittest.TestCase):
    def test_straight_marking_bending_off_line_is_rejected(self):
        """Samples spanning two markings is the failure this catches."""
        subject = annotation(
            features=(
                AnnotatedFeature(
                    MarkingFeature.BASELINE,
                    frame_points((100, 500), (500, 560), (900, 500)),
                ),
            )
        )
        self.assertTrue(any("bends" in p for p in validate(subject)), validate(subject))

    def test_perspective_foreshortening_alone_does_not_trip_straightness(self):
        """A projected straight line stays straight, however unevenly sampled."""
        subject = annotation(
            features=(
                AnnotatedFeature(
                    MarkingFeature.BASELINE,
                    frame_points((100, 500), (300, 490), (800, 465), (900, 460)),
                ),
            )
        )
        self.assertEqual([p for p in validate(subject) if "bends" in p], [])

    def test_arc_that_changes_turn_direction_is_rejected(self):
        """A projected circle is a conic with no inflection point."""
        subject = annotation(
            features=(
                AnnotatedFeature(
                    MarkingFeature.THREE_POINT_ARC,
                    frame_points((100, 400), (200, 350), (300, 340), (400, 350), (500, 300)),
                ),
            )
        )
        self.assertTrue(any("inflection" in p for p in validate(subject)), validate(subject))

    def test_consistently_curving_arc_is_accepted(self):
        subject = annotation(
            features=(
                AnnotatedFeature(
                    MarkingFeature.THREE_POINT_ARC,
                    frame_points((100, 400), (200, 350), (300, 335), (400, 350), (500, 400)),
                ),
            )
        )
        self.assertEqual([p for p in validate(subject) if "inflection" in p], [])


class RoundTripTests(unittest.TestCase):
    def test_round_trip_preserves_everything(self):
        subject = annotation(
            features=(
                AnnotatedFeature(
                    MarkingFeature.LANE_EDGE_FAR,
                    frame_points((300, 490), (420, 300)),
                    visibility=Visibility.PARTIALLY_OCCLUDED,
                    uncertainty_px=2.5,
                    occluded_after=(0,),
                    notes="player crossing",
                ),
                AnnotatedFeature(MarkingFeature.BASELINE, frame_points((100, 500), (900, 480))),
            ),
            junctions=(
                AnnotatedJunction(
                    JunctionPoint.LANE_BASELINE_FAR, SamplePoint(300, 490), uncertainty_px=1.5
                ),
            ),
            skipped=(SkippedFeature(MarkingFeature.CENTER_CIRCLE, SkipReason.OUT_OF_FRAME),),
        )
        restored = FrameAnnotation.from_dict(subject.as_dict())
        self.assertEqual(restored, subject)

    def test_foreign_schema_version_is_refused(self):
        payload = annotation().as_dict()
        payload["schema_version"] = "court-annotation-0.9.0"
        with self.assertRaises(ValueError):
            FrameAnnotation.from_dict(payload)

    def test_schema_version_is_recorded(self):
        self.assertEqual(annotation().as_dict()["schema_version"], ANNOTATION_SCHEMA_VERSION)


if __name__ == "__main__":
    unittest.main()
