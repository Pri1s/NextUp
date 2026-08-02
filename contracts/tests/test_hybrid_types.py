import dataclasses
import json
import math
import unittest

from contracts.hybrid_types import (
    HYBRID_RECORD_SCHEMA_VERSION,
    AssignmentAlternative,
    AssignmentStatus,
    AssociationDiagnostics,
    CandidateQuality,
    CandidateSource,
    CandidateTransform,
    EvidenceKind,
    GateResult,
    HybridFrameStatus,
    HybridRefinementFrame,
    ImageMarkingSample,
    MarkingAssignment,
    PrimitiveQuality,
    PrimitiveStability,
    ProbeDisplacement,
    SelectionDecision,
    SelectionOutcome,
    ShadowFrameStatus,
    ShadowMarkingFrame,
    TransformDifference,
    UnlabeledMarkingEvidence,
)
from contracts.markings import MarkingFamily, MarkingFeature


def sample(x=10.0, y=20.0):
    return ImageMarkingSample(x, y, (1.0, 0.0), 0.8, 1.5, 3.0)


def diagnostics():
    return AssociationDiagnostics(2.0, 4.0, 0.7, 0.2)


def accepted():
    return MarkingAssignment(
        evidence_id="e1", status=AssignmentStatus.ACCEPTED,
        selected_feature=MarkingFeature.THREE_POINT_ARC,
        alternatives=(AssignmentAlternative(MarkingFeature.THREE_POINT_ARC, 0.9),),
        diagnostics=diagnostics(), reason_codes=("unique",),
    )


def primitive():
    return PrimitiveQuality(
        MarkingFeature.THREE_POINT_ARC, True, 25, 1.2, 2.4, 10.0, 29.0
    )


def gate():
    return GateResult("physical", True, 1.0, 2.0, "px")


IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
KEYPOINT_ID = "video_1_000012:keypoint"
HYBRID_ID = "video_1_000012:hybrid:0123456789ab"


def transform(candidate_id=KEYPOINT_ID, source=CandidateSource.KEYPOINT, **kwargs):
    fields = dict(
        candidate_id=candidate_id, source=source, layout_id="nba_halfcourt",
        layout_hash="a" * 64, h_court_to_image=IDENTITY, h_image_to_court=IDENTITY,
        round_trip_error_px=0.0, solver="USAC_MAGSAC",
    )
    fields.update(kwargs)
    return CandidateTransform(**fields)


def candidate(candidate_id=KEYPOINT_ID, source=CandidateSource.KEYPOINT, **kwargs):
    fields = dict(
        candidate_id=candidate_id, source=source, usable=True,
        model_point_px_median=1.0, model_point_px_p95=2.0, primitives=(),
        supported_families=(), max_depth_ft=19.0, held_out_px_median=None,
        leave_one_primitive_shift_px=None, gates=(gate(),), reasons=(),
    )
    fields.update(kwargs)
    return CandidateQuality(**fields)


def difference(**kwargs):
    fields = dict(
        baseline_candidate_id=KEYPOINT_ID, challenger_candidate_id=HYBRID_ID,
        probe_point_count=15, probe_px_median=3.0, probe_px_p95=7.0, probe_px_max=9.0,
        max_control_point_shift_ft=0.4, bounds_active=0, scale_ratio=1.01,
        bands=(ProbeDisplacement("0-19ft", 9, 1.0, 2.0),),
    )
    fields.update(kwargs)
    return TransformDifference(**fields)


def refinement(**kwargs):
    """A frame carrying both candidates, as Milestone 4 emits them."""
    fields = dict(
        frame_id="video_1_000012", image_sha256="b" * 64,
        status=HybridFrameStatus.OK, eligible=True, eligibility_reasons=(),
        layout_id="nba_halfcourt", layout_hash="a" * 64,
        shadow_config_hash="c" * 64, refiner_config_hash="d" * 64,
        baseline_candidate_id=KEYPOINT_ID, challenger_candidate_id=HYBRID_ID,
        transforms=(transform(), transform(HYBRID_ID, CandidateSource.HYBRID)),
        candidates=(candidate(), candidate(HYBRID_ID, CandidateSource.HYBRID)),
        difference=difference(),
        selection=SelectionDecision(
            "video_1_000012", KEYPOINT_ID, HYBRID_ID, KEYPOINT_ID,
            SelectionOutcome.KEEP_BASELINE,
            (GateResult("selection.shadow_mode", True),),
            ("shadow_mode:milestone4_never_selects_challenger",),
        ),
        stability=(PrimitiveStability(MarkingFeature.THREE_POINT_ARC, 2.5, True),),
        fitted_features=(MarkingFeature.THREE_POINT_ARC, MarkingFeature.LANE_EDGE_FAR),
        held_out_features=(MarkingFeature.FREE_THROW_CIRCLE_FAR_HALF,),
        outer_iterations=3, inner_iterations=17, converged=True,
        refine_ms=12.5, diagnostics_ms=4.0,
    )
    fields.update(kwargs)
    return HybridRefinementFrame(**fields)


class EvidenceTests(unittest.TestCase):
    def test_evidence_is_unlabeled_and_round_trips(self):
        item = UnlabeledMarkingEvidence(
            "e1", "frame_1", "ridge-v1", EvidenceKind.CURVED,
            (sample(), sample(11, 21)), 0.85,
        )
        restored = UnlabeledMarkingEvidence.from_dict(json.loads(json.dumps(item.as_dict())))
        self.assertEqual(restored, item)
        self.assertNotIn("feature", item.as_dict())

    def test_bad_samples_are_rejected(self):
        with self.assertRaises(ValueError):
            sample().from_dict({**sample().as_dict(), "confidence": 1.1})
        with self.assertRaises(ValueError):
            ImageMarkingSample(float("nan"), 2, None, 0.5, 1)
        with self.assertRaises(ValueError):
            ImageMarkingSample(1, 2, (2, 0), 0.5, 1)
        with self.assertRaises(ValueError):
            UnlabeledMarkingEvidence(
                "e", "f", "s", EvidenceKind.UNKNOWN, (sample(),), 0.5
            )
        with self.assertRaises(ValueError):
            UnlabeledMarkingEvidence(
                "e", "f", "s", EvidenceKind.UNKNOWN, (sample(), sample()), 0.5
            )

    def test_unknown_serialized_field_is_rejected(self):
        data = sample().as_dict()
        data["slot_index"] = 3
        with self.assertRaises(ValueError):
            ImageMarkingSample.from_dict(data)


class AssignmentTests(unittest.TestCase):
    def test_accepted_assignment_round_trips(self):
        item = accepted()
        self.assertEqual(
            MarkingAssignment.from_dict(json.loads(json.dumps(item.as_dict()))), item
        )

    def test_ambiguous_assignment_cannot_claim_selected_feature(self):
        alternatives = (
            AssignmentAlternative(MarkingFeature.BASELINE, 0.6),
            AssignmentAlternative(MarkingFeature.FREE_THROW_LINE, 0.59),
        )
        with self.assertRaises(ValueError):
            MarkingAssignment(
                "e", AssignmentStatus.AMBIGUOUS, MarkingFeature.BASELINE,
                alternatives, diagnostics(), ("low_margin",),
            )
        valid = MarkingAssignment(
            "e", AssignmentStatus.AMBIGUOUS, None, alternatives,
            diagnostics(), ("low_margin",),
        )
        self.assertIsNone(valid.selected_feature)

    def test_rejected_assignment_needs_reason(self):
        with self.assertRaises(ValueError):
            MarkingAssignment("e", AssignmentStatus.REJECTED, None, (), diagnostics())

    def test_wrong_schema_is_rejected(self):
        data = accepted().as_dict()
        data["schema_version"] = "old"
        with self.assertRaises(ValueError):
            MarkingAssignment.from_dict(data)


class QualityTests(unittest.TestCase):
    def test_nested_quality_and_gate_round_trips(self):
        item = CandidateQuality(
            candidate_id="hybrid", source=CandidateSource.HYBRID, usable=True,
            model_point_px_median=1.0, model_point_px_p95=2.0,
            primitives=(primitive(),), supported_families=(MarkingFamily.THREE_POINT,),
            max_depth_ft=29.0, held_out_px_median=3.0,
            leave_one_primitive_shift_px=2.0, gates=(gate(),), reasons=("passed",),
        )
        restored = CandidateQuality.from_dict(json.loads(json.dumps(item.as_dict())))
        self.assertEqual(restored, item)
        self.assertEqual(
            PrimitiveQuality.from_dict(json.loads(json.dumps(primitive().as_dict()))), primitive()
        )
        self.assertEqual(GateResult.from_dict(json.loads(json.dumps(gate().as_dict()))), gate())

    def test_usable_candidate_cannot_have_failed_gate(self):
        failed = GateResult("held_out", False, 12, 10, "px", "too_large")
        with self.assertRaises(ValueError):
            CandidateQuality(
                "h", CandidateSource.HYBRID, True, 1, 2, (primitive(),),
                (MarkingFamily.THREE_POINT,), 29, 3, 2, (failed,),
            )

    def test_unusable_candidate_needs_reason(self):
        with self.assertRaises(ValueError):
            CandidateQuality(
                "h", CandidateSource.HYBRID, False, None, None, (), (), 0,
                None, None, (), (),
            )

    def test_nonfinite_metrics_are_rejected(self):
        with self.assertRaises(ValueError):
            PrimitiveQuality(MarkingFeature.BASELINE, True, 2, math.inf, 2, 0, 1)


class SelectionTests(unittest.TestCase):
    def test_keep_baseline_round_trips(self):
        item = SelectionDecision(
            "frame", "keypoint", "hybrid", "keypoint",
            SelectionOutcome.KEEP_BASELINE, (gate(),), ("tie_goes_to_baseline",),
        )
        restored = SelectionDecision.from_dict(json.loads(json.dumps(item.as_dict())))
        self.assertEqual(restored, item)

    def test_outcome_must_match_selected_id(self):
        with self.assertRaises(ValueError):
            SelectionDecision(
                "frame", "keypoint", "hybrid", "hybrid",
                SelectionOutcome.KEEP_BASELINE, (), ("bad",),
            )
        with self.assertRaises(ValueError):
            SelectionDecision(
                "frame", "keypoint", "hybrid", "keypoint",
                SelectionOutcome.SELECT_CHALLENGER, (), ("bad",),
            )

    def test_no_usable_candidate_selects_nothing(self):
        item = SelectionDecision(
            "frame", "keypoint", None, None, SelectionOutcome.NO_USABLE_CANDIDATE,
            (), ("both_rejected",),
        )
        self.assertIsNone(item.selected_candidate_id)


class CandidateTransformTests(unittest.TestCase):
    def test_solved_transform_round_trips(self):
        item = transform()
        restored = CandidateTransform.from_dict(json.loads(json.dumps(item.as_dict())))
        self.assertEqual(restored, item)
        self.assertTrue(item.solved)

    def test_unsolved_transform_needs_a_failure_and_keeps_no_matrix(self):
        item = transform(
            HYBRID_ID, CandidateSource.HYBRID, h_court_to_image=None,
            h_image_to_court=None, round_trip_error_px=None, solver="none",
            failure="bounds_active",
        )
        self.assertFalse(item.solved)
        self.assertEqual(CandidateTransform.from_dict(json.loads(json.dumps(item.as_dict()))), item)
        with self.assertRaises(ValueError):
            transform(h_court_to_image=None, h_image_to_court=None, round_trip_error_px=None)

    def test_a_solved_transform_may_not_also_claim_a_failure(self):
        with self.assertRaises(ValueError):
            transform(failure="solver_failed")
        with self.assertRaises(ValueError):
            transform(round_trip_error_px=None)

    def test_half_a_transform_pair_is_rejected(self):
        with self.assertRaises(ValueError):
            transform(h_image_to_court=None)

    def test_matrices_must_be_finite_three_by_three_and_w_normalised(self):
        with self.assertRaises(ValueError):
            transform(h_court_to_image=((1.0, 0.0), (0.0, 1.0)))
        with self.assertRaises(ValueError):
            transform(h_court_to_image=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, float("nan"))))
        # Same transform, unnormalised: it must not be representable twice.
        with self.assertRaises(ValueError):
            transform(h_court_to_image=((2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0)))

    def test_layout_hash_travels_with_the_matrix(self):
        with self.assertRaises(ValueError):
            transform(layout_hash="")


class TransformDifferenceTests(unittest.TestCase):
    def test_difference_round_trips(self):
        item = difference()
        self.assertEqual(TransformDifference.from_dict(json.loads(json.dumps(item.as_dict()))), item)

    def test_percentiles_must_be_ordered(self):
        with self.assertRaises(ValueError):
            difference(probe_px_median=8.0, probe_px_p95=7.0)
        with self.assertRaises(ValueError):
            difference(probe_px_p95=12.0)

    def test_degenerate_difference_is_rejected(self):
        with self.assertRaises(ValueError):
            difference(challenger_candidate_id=KEYPOINT_ID)
        with self.assertRaises(ValueError):
            difference(bounds_active=5)
        with self.assertRaises(ValueError):
            difference(scale_ratio=0.0)
        with self.assertRaises(ValueError):
            difference(bands=(
                ProbeDisplacement("0-19ft", 1, 1.0, 1.0),
                ProbeDisplacement("0-19ft", 1, 1.0, 1.0),
            ))

    def test_empty_band_cannot_report_displacement(self):
        self.assertEqual(ProbeDisplacement("30-47ft", 0, 0.0, 0.0).point_count, 0)
        with self.assertRaises(ValueError):
            ProbeDisplacement("30-47ft", 0, 1.0, 1.0)


class HybridRefinementFrameTests(unittest.TestCase):
    def test_frame_round_trips(self):
        item = refinement()
        restored = HybridRefinementFrame.from_dict(json.loads(json.dumps(item.as_dict())))
        self.assertEqual(restored, item)

    def test_every_candidate_needs_a_transform_and_the_reverse(self):
        with self.assertRaises(ValueError):
            refinement(transforms=(transform(),))
        with self.assertRaises(ValueError):
            refinement(candidates=(candidate(),))
        with self.assertRaises(ValueError):
            refinement(transforms=(transform(), transform()))

    def test_exactly_one_keypoint_candidate_and_it_is_the_baseline(self):
        with self.assertRaises(ValueError):
            refinement(transforms=(transform(), transform(HYBRID_ID, CandidateSource.KEYPOINT)))
        with self.assertRaises(ValueError):
            refinement(
                transforms=(
                    transform(KEYPOINT_ID, CandidateSource.HYBRID),
                    transform(HYBRID_ID, CandidateSource.CV_ONLY),
                ),
            )
        with self.assertRaises(ValueError):
            refinement(baseline_candidate_id=HYBRID_ID, challenger_candidate_id=KEYPOINT_ID)

    def test_unknown_candidate_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            refinement(baseline_candidate_id="nobody")
        with self.assertRaises(ValueError):
            refinement(challenger_candidate_id="nobody")

    def test_a_frame_without_a_challenger_records_no_challenger_state(self):
        baseline_only = dict(
            challenger_candidate_id=None, transforms=(transform(),),
            candidates=(candidate(),), difference=None, selection=None,
            stability=(), fitted_features=(),
        )
        item = refinement(**baseline_only)
        self.assertEqual(HybridRefinementFrame.from_dict(json.loads(json.dumps(item.as_dict()))), item)
        with self.assertRaises(ValueError):
            refinement(**{**baseline_only, "difference": difference()})
        with self.assertRaises(ValueError):
            refinement(**{**baseline_only, "fitted_features": (MarkingFeature.LANE_EDGE_FAR,)})

    def test_difference_and_selection_ids_must_agree_with_the_frame(self):
        with self.assertRaises(ValueError):
            refinement(difference=difference(baseline_candidate_id="other"))
        with self.assertRaises(ValueError):
            refinement(selection=SelectionDecision(
                "another_frame", KEYPOINT_ID, HYBRID_ID, KEYPOINT_ID,
                SelectionOutcome.KEEP_BASELINE, (), ("mismatch",),
            ))

    def test_cannot_select_an_unusable_challenger(self):
        unusable = candidate(
            HYBRID_ID, CandidateSource.HYBRID, usable=False,
            gates=(GateResult("hybrid.converged", False, reason="did_not_converge"),),
            reasons=("hybrid.converged",),
        )
        with self.assertRaises(ValueError):
            refinement(
                candidates=(candidate(), unusable),
                selection=SelectionDecision(
                    "video_1_000012", KEYPOINT_ID, HYBRID_ID, HYBRID_ID,
                    SelectionOutcome.SELECT_CHALLENGER, (), ("promoted",),
                ),
            )

    def test_a_feature_cannot_be_both_fitted_and_held_out(self):
        with self.assertRaises(ValueError):
            refinement(
                fitted_features=(MarkingFeature.FREE_THROW_CIRCLE_FAR_HALF,),
                held_out_features=(MarkingFeature.FREE_THROW_CIRCLE_FAR_HALF,),
                stability=(),
            )
        with self.assertRaises(ValueError):
            refinement(fitted_features=(MarkingFeature.LANE_EDGE_FAR, MarkingFeature.LANE_EDGE_FAR))

    def test_stability_may_only_report_fitted_primitives(self):
        with self.assertRaises(ValueError):
            refinement(stability=(PrimitiveStability(MarkingFeature.BASELINE, 1.0, True),))

    def test_abstained_and_failed_frames_carry_no_challenger(self):
        abstained = refinement(
            status=HybridFrameStatus.ABSTAINED, eligible=False,
            eligibility_reasons=("baseline_status:UNCALIBRATED_INSUFFICIENT_EVIDENCE",),
            challenger_candidate_id=None, transforms=(transform(),),
            candidates=(candidate(),), difference=None, selection=None,
            stability=(), fitted_features=(),
        )
        self.assertEqual(
            HybridRefinementFrame.from_dict(json.loads(json.dumps(abstained.as_dict()))),
            abstained,
        )
        with self.assertRaises(ValueError):
            refinement(status=HybridFrameStatus.ABSTAINED, eligible=False,
                       eligibility_reasons=("ineligible",))
        with self.assertRaises(ValueError):
            refinement(status=HybridFrameStatus.REFINEMENT_FAILED, failure_reasons=("ValueError:x",))
        with self.assertRaises(ValueError):
            refinement(
                status=HybridFrameStatus.REFINEMENT_FAILED, challenger_candidate_id=None,
                transforms=(transform(),), candidates=(candidate(),), difference=None,
                selection=None, stability=(), fitted_features=(), failure_reasons=(),
            )

    def test_empty_reasons_and_hashes_are_rejected(self):
        with self.assertRaises(ValueError):
            refinement(eligibility_reasons=("",))
        with self.assertRaises(ValueError):
            refinement(refiner_config_hash="")
        with self.assertRaises(ValueError):
            refinement(shadow_config_hash="")

    def test_ineligible_frame_needs_a_reason(self):
        with self.assertRaises(ValueError):
            refinement(eligible=False, eligibility_reasons=())


class SchemaStabilityTests(unittest.TestCase):
    """Milestone 4 adds records; it must not invalidate Milestone 3 artifacts."""

    def test_milestone3_shadow_frames_still_parse(self):
        frame = ShadowMarkingFrame(
            frame_id="video_1_000012", image_sha256="b" * 64,
            baseline_status="DEGRADED", status=ShadowFrameStatus.OK, eligible=True,
            eligibility_reasons=(), layout_id="nba_halfcourt", layout_hash="a" * 64,
            config_hash="c" * 64, image_width=1280, image_height=720,
            roi_pixels=100, excluded_pixels=0,
            evidence=(UnlabeledMarkingEvidence(
                "e1", "video_1_000012", "ridge", EvidenceKind.CURVED,
                (sample(), sample(11, 21)), 0.85,
            ),),
            assignments=(accepted(),),
        )
        encoded = json.dumps(frame.as_dict(), sort_keys=True)
        self.assertEqual(ShadowMarkingFrame.from_dict(json.loads(encoded)), frame)
        self.assertEqual(json.dumps(ShadowMarkingFrame.from_dict(json.loads(encoded)).as_dict(),
                                    sort_keys=True), encoded)

    def test_new_records_share_the_unchanged_schema_version(self):
        for item in (transform(), refinement()):
            with self.subTest(record=type(item).__name__):
                self.assertEqual(item.schema_version, HYBRID_RECORD_SCHEMA_VERSION)


class SlotIndependenceTests(unittest.TestCase):
    def test_public_record_fields_and_json_expose_no_slots(self):
        instances = [
            sample(),
            UnlabeledMarkingEvidence(
                "e1", "frame", "ridge", EvidenceKind.STRAIGHT,
                (sample(), sample(11, 21)), 0.8,
            ),
            accepted(), primitive(), gate(),
            CandidateQuality(
                "k", CandidateSource.KEYPOINT, True, 1, 2, (), (), 19, None, None,
                (gate(),), ("baseline",),
            ),
            SelectionDecision(
                "frame", "k", None, "k", SelectionOutcome.KEEP_BASELINE,
                (), ("no_challenger",),
            ),
            transform(),
            ProbeDisplacement("0-19ft", 9, 1.0, 2.0),
            difference(),
            PrimitiveStability(MarkingFeature.THREE_POINT_ARC, 2.5, True),
            refinement(),
        ]
        for item in instances:
            with self.subTest(record=type(item).__name__):
                self.assertNotIn("slot", json.dumps(item.as_dict()).lower())
                for field in dataclasses.fields(item):
                    self.assertNotIn("slot", field.name.lower())

    def test_schema_version_is_explicit(self):
        self.assertEqual(HYBRID_RECORD_SCHEMA_VERSION, "hybrid-marking-records-1.0.0")


if __name__ == "__main__":
    unittest.main()
