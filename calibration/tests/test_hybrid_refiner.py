"""The refiner must be honest about what it did and never promote anything.

Milestone 4 runs in shadow: it fits a second candidate, measures both, and
records the pair. It does not select. Every record says ``KEEP_BASELINE``, and a
source scan here enforces that the promotion path is not merely unused but
absent.

The other theme is that a report must not overstate its own result. A solve that
gave up has to say so, an unusable candidate has to name the gate that failed,
and a residual has to be measured against the transform it claims to describe.
The bug this file exists to prevent shipped once already: the optimizer inherited
its damping across outer iterations, found no descent step, stopped 1.8 px from a
transform it had exact evidence for, and reported success.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import unittest
from pathlib import Path

import numpy as np

from calibration.hybrid.orchestrator import (
    HybridGatePolicy,
    HybridOrchestrator,
    HybridProvenanceError,
    HybridRefinerConfig,
)
from calibration.hybrid.refine import refine
from calibration.hybrid.residuals import collect_observations
from calibration.tests.test_hybrid_residuals import (
    ALL_FIVE,
    IMAGE_SHAPE,
    INLIER_LANDMARKS,
    LAYOUT,
    STRAIGHTS,
    TRUTH,
    correspondences,
    shadow_frame,
)
from contracts.annotations import HELD_OUT_FAMILIES
from contracts.calibration_types import (
    CalibrationQuality,
    CalibrationStatus,
    CourtCalibration,
)
from contracts.hybrid_types import (
    CandidateSource,
    HybridFrameStatus,
    HybridRefinementFrame,
    SelectionOutcome,
)
from contracts.markings import MarkingFeature

HYBRID_DIR = Path(__file__).resolve().parents[1] / "hybrid"


def _matrix_tuple(matrix):
    return tuple(tuple(float(value) for value in row) for row in matrix)


def calibration(matrix=TRUTH, reasons=("weak_conditioning:only_2_points_off_dominant_line",)):
    """A DEGRADED five-inlier seed -- the only class the shadow path accepts."""
    inverse = np.linalg.inv(np.asarray(matrix, dtype=np.float64))
    inverse = inverse / inverse[2, 2]
    quality = CalibrationQuality(
        observation_count=5, inlier_count=5, inlier_ratio=1.0,
        reprojection_px_mean=0.1, reprojection_px_median=0.1,
        reprojection_px_p95=0.2, reprojection_px_max=0.3,
        reprojection_ft_median=0.01, reprojection_ft_max=0.03,
        holdout_px_median=0.5, min_triangle_area_px=5000.0, image_hull_fraction=0.3,
        court_span_x_ft=19.0, court_span_y_ft=33.0, scale_ft_per_px=0.1,
        round_trip_error_px=0.0, mean_evidence_confidence=0.9, reasons=reasons,
    )
    return CourtCalibration(
        frame_index=12, status=CalibrationStatus.DEGRADED,
        h_court_to_image=_matrix_tuple(matrix), h_image_to_court=_matrix_tuple(inverse),
        quality=quality, inlier_landmarks=INLIER_LANDMARKS,
        layout_id=LAYOUT.layout_id, provenance={"weights_sha256": "a" * 64},
    )


def refined(features=ALL_FIVE, config=None, frame=None, cal=None):
    orchestrator = HybridOrchestrator(LAYOUT, config or HybridRefinerConfig())
    return orchestrator.refine_frame(
        frame if frame is not None else shadow_frame(features),
        cal if cal is not None else calibration(),
        correspondences(),
        image_shape=IMAGE_SHAPE,
    )


def encode(record):
    return json.dumps(record.as_dict(), sort_keys=True)


class ShadowModeTests(unittest.TestCase):
    def test_never_selects_a_challenger(self):
        record = refined()
        self.assertIsNotNone(record.selection)
        self.assertIs(record.selection.outcome, SelectionOutcome.KEEP_BASELINE)
        self.assertEqual(record.selection.selected_candidate_id, record.baseline_candidate_id)

    def test_the_promotion_path_is_absent_not_merely_unused(self):
        """A rule enforced only by intention is one broken by the next edit."""
        offenders = []
        for path in sorted(HYBRID_DIR.rglob("*.py")):
            if "SELECT_CHALLENGER" in path.read_text(encoding="utf-8"):
                offenders.append(str(path.name))
        self.assertEqual(offenders, [])

    def test_the_scan_would_catch_a_violation(self):
        """A guard that cannot fail is not a guard."""
        self.assertIn("SELECT_CHALLENGER", "outcome = SelectionOutcome.SELECT_CHALLENGER")

    def test_the_baseline_transform_is_recorded_unchanged(self):
        """The keypoint candidate must survive as an exact fallback."""
        record = refined()
        baseline = [t for t in record.transforms if t.source is CandidateSource.KEYPOINT][0]
        self.assertEqual(baseline.h_court_to_image, _matrix_tuple(TRUTH))


class ConvergenceHonestyTests(unittest.TestCase):
    def test_a_starved_outer_budget_is_not_reported_as_converged(self):
        """Being cut off is not the same as arriving.

        With one outer iteration the solve cannot have reached a fixed point, so
        whatever it produces must not claim convergence -- ``hybrid.converged``
        is a gate, and a transform the optimizer abandoned would otherwise pass
        it.
        """
        config = dataclasses.replace(HybridRefinerConfig(), max_outer=1, max_inner=2)
        theta0 = np.asarray(
            [c for c in _seed_theta()], dtype=np.float64
        )
        observations = collect_observations(
            shadow_frame(ALL_FIVE), LAYOUT,
            _seed_matrix(), config, IMAGE_SHAPE,
        )
        result = refine(theta0, correspondences(), observations, LAYOUT, config, IMAGE_SHAPE)
        self.assertFalse(result.converged)

    def test_a_settled_solve_reports_converged(self):
        config = HybridRefinerConfig()
        observations = collect_observations(
            shadow_frame(ALL_FIVE), LAYOUT, _seed_matrix(), config, IMAGE_SHAPE
        )
        result = refine(
            np.asarray(_seed_theta(), dtype=np.float64), correspondences(),
            observations, LAYOUT, config, IMAGE_SHAPE,
        )
        self.assertTrue(result.converged)

    def test_no_marking_evidence_leaves_the_seed_untouched(self):
        """The exactness invariant is structural, not a consequence of the prior.

        With nothing to fit, the refiner returns the seed itself rather than
        whatever a numerical solve happens to land on.
        """
        config = HybridRefinerConfig()
        theta0 = np.asarray(_seed_theta(), dtype=np.float64)
        result = refine(theta0, correspondences(), (), LAYOUT, config, IMAGE_SHAPE)
        self.assertEqual(result.outer_iterations, 0)
        self.assertEqual(result.inner_iterations, 0)
        self.assertTrue(result.converged)
        np.testing.assert_array_equal(result.theta, theta0)


class DeterminismTests(unittest.TestCase):
    def test_two_runs_are_byte_identical(self):
        self.assertEqual(encode(refined()), encode(refined()))

    def test_evidence_order_does_not_change_the_record(self):
        """The record must describe the geometry, not the order it arrived in."""
        frame = shadow_frame(ALL_FIVE)
        shuffled = dataclasses.replace(
            frame,
            evidence=tuple(reversed(frame.evidence)),
            assignments=tuple(reversed(frame.assignments)),
        )
        self.assertEqual(encode(refined(frame=frame)), encode(refined(frame=shuffled)))

    def test_the_record_round_trips(self):
        record = refined()
        self.assertEqual(HybridRefinementFrame.from_dict(json.loads(encode(record))), record)


class HoldOutTests(unittest.TestCase):
    def test_the_held_out_family_is_never_fitted(self):
        record = refined()
        self.assertIn(MarkingFeature.FREE_THROW_CIRCLE_FAR_HALF, record.held_out_features)
        self.assertNotIn(MarkingFeature.FREE_THROW_CIRCLE_FAR_HALF, record.fitted_features)

    def test_the_held_out_feature_still_reports_a_measurement(self):
        """Refusing to fit it is the point; refusing to measure it would make
        the milestone unfalsifiable."""
        record = refined()
        for candidate in record.candidates:
            rows = [p for p in candidate.primitives
                    if p.feature is MarkingFeature.FREE_THROW_CIRCLE_FAR_HALF]
            with self.subTest(candidate=candidate.source.value):
                self.assertEqual(len(rows), 1)
                self.assertFalse(rows[0].fitted)

    def test_hold_out_cannot_be_something_the_benchmark_scores_as_fitted(self):
        self.assertTrue(set(HybridRefinerConfig().held_out_features) <= set(HELD_OUT_FAMILIES))

    def test_a_hold_out_outside_the_benchmark_families_is_rejected(self):
        with self.assertRaises(ValueError):
            dataclasses.replace(
                HybridRefinerConfig(),
                held_out_features=(MarkingFeature.THREE_POINT_ARC,),
            )


class GateTests(unittest.TestCase):
    #: Every policy threshold, mapped to the gate that applies it. Deliberately
    #: explicit rather than name-matched: a heuristic gives false confidence in
    #: both directions, and the whole purpose of this test is that adding a
    #: threshold forces you to say which gate consumes it.
    THRESHOLD_GATES = {
        "model_point_median_px": "candidate.model_point_median_px",
        "model_point_p95_px": "candidate.model_point_p95_px",
        "model_point_max_shift_px": "candidate.model_point_max_shift_px",
        "round_trip_px": "candidate.round_trip_px",
        "scale_ft_per_px_range": "candidate.scale_plausible",
        "min_court_span_x_ft": "candidate.court_span_ft",
        "min_court_span_y_ft": "candidate.court_span_ft",
        "depth_beyond_19ft": "hybrid.depth_beyond_19ft",
        "independent_families": "hybrid.independent_families",
        "no_ambiguous_fitted": "hybrid.no_ambiguous_fitted",
        "held_out_available": "hybrid.held_out_available",
        "held_out_median_px": "hybrid.held_out_median_px",
        "held_out_p95_px": "hybrid.held_out_p95_px",
        "leave_one_primitive_shift_px": "hybrid.leave_one_primitive_shift_px",
        "control_point_shift_ft": "hybrid.control_point_shift_ft",
        "bounds_inactive": "hybrid.bounds_inactive",
        "probe_shift_px": "hybrid.probe_shift_px",
        "scale_ratio_range": "hybrid.scale_ratio",
        "converged": "hybrid.converged",
        "marking_residual_median_px": "hybrid.marking_residual_median_px",
        "validation_advantage_px": "hybrid.validation_advantage_px",
    }

    def test_the_threshold_map_covers_the_policy(self):
        """A new threshold must declare its gate here before it can ship."""
        self.assertEqual(
            set(HybridGatePolicy().__dataclass_fields__), set(self.THRESHOLD_GATES)
        )

    def test_every_configured_threshold_is_actually_checked(self):
        """Stops a threshold that is configured, documented, and never applied."""
        record = refined()
        emitted = {gate.gate_id for candidate in record.candidates for gate in candidate.gates}
        for threshold, gate_id in self.THRESHOLD_GATES.items():
            with self.subTest(threshold=threshold):
                self.assertIn(gate_id, emitted)

    def test_no_gate_is_emitted_without_a_reason_to_exist(self):
        """The reverse direction: a gate nothing configures is unexplained."""
        record = refined()
        emitted = {gate.gate_id for candidate in record.candidates for gate in candidate.gates}
        # ``court_quad_convex`` is a pure predicate with no numeric threshold.
        allowed = set(self.THRESHOLD_GATES.values()) | {"candidate.court_quad_convex"}
        self.assertEqual(emitted - allowed, set())

    def test_every_gate_declares_its_scope(self):
        record = refined()
        for candidate in record.candidates:
            for gate in candidate.gates:
                with self.subTest(gate=gate.gate_id):
                    self.assertRegex(gate.gate_id, r"^(candidate|hybrid)\.")

    def test_the_baseline_is_judged_only_by_candidate_gates(self):
        """Applying "two marking families" to the keypoint candidate would mark
        it unusable, which would be wrong -- it claims no marking support."""
        record = refined()
        baseline = [c for c in record.candidates
                    if c.candidate_id == record.baseline_candidate_id][0]
        self.assertTrue(all(g.gate_id.startswith("candidate.") for g in baseline.gates))

    def test_usable_means_no_failed_gate(self):
        record = refined()
        for candidate in record.candidates:
            with self.subTest(candidate=candidate.source.value):
                failed = [g for g in candidate.gates if not g.passed]
                self.assertEqual(candidate.usable, not failed)
                if failed:
                    self.assertTrue(candidate.reasons)

    def test_a_failed_gate_always_carries_a_reason(self):
        record = refined()
        for candidate in record.candidates:
            for gate in candidate.gates:
                if not gate.passed:
                    with self.subTest(gate=gate.gate_id):
                        self.assertTrue(gate.reason)


class DepthReportingTests(unittest.TestCase):
    def test_the_keypoint_candidate_reports_its_real_nineteen_foot_limit(self):
        """The whole premise: five landmarks spanning 19 ft and no more."""
        record = refined()
        baseline = [c for c in record.candidates
                    if c.candidate_id == record.baseline_candidate_id][0]
        self.assertAlmostEqual(baseline.max_depth_ft, 19.0, places=6)

    def test_arc_evidence_carries_the_challenger_past_the_break(self):
        record = refined()
        challenger = [c for c in record.candidates
                      if c.candidate_id == record.challenger_candidate_id][0]
        self.assertGreater(challenger.max_depth_ft, 24.0)

    def test_the_keypoint_candidate_claims_no_marking_support(self):
        record = refined()
        baseline = [c for c in record.candidates
                    if c.candidate_id == record.baseline_candidate_id][0]
        self.assertEqual(baseline.supported_families, ())


class ProvenanceTests(unittest.TestCase):
    def test_a_foreign_layout_hash_raises_before_any_fitting(self):
        frame = dataclasses.replace(shadow_frame(ALL_FIVE), layout_hash="f" * 64)
        with self.assertRaises(HybridProvenanceError):
            refined(frame=frame)

    def test_a_mismatched_frame_is_rejected(self):
        """Wiring errors are not runtime image failures and must not be swallowed."""
        frame = dataclasses.replace(shadow_frame(ALL_FIVE), layout_id="other_court")
        with self.assertRaises(HybridProvenanceError):
            refined(frame=frame)


class ContainmentTests(unittest.TestCase):
    def test_a_runtime_failure_becomes_a_record_not_an_exception(self):
        """The refiner runs beside the production keypoint path. An unhandled
        exception here would fail a run whose primary output is unrelated."""
        import calibration.hybrid.orchestrator as module

        original = module.refine

        def explode(*args, **kwargs):
            raise RuntimeError("planted failure with an awkward message")

        module.refine = explode
        try:
            record = refined()
        finally:
            module.refine = original
        self.assertIs(record.status, HybridFrameStatus.REFINEMENT_FAILED)
        self.assertTrue(record.failure_reasons)
        self.assertIsNone(record.challenger_candidate_id)

    def test_swallowed_reasons_are_typed_not_raw_library_text(self):
        """A bare ``str(error)`` puts arbitrary library wording into a
        content-hashed artifact, so a numpy or OpenCV rewording would make
        records irreproducible for reasons unrelated to the algorithm."""
        import calibration.hybrid.orchestrator as module

        original = module.refine
        module.refine = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        try:
            record = refined()
        finally:
            module.refine = original
        self.assertTrue(any(reason.startswith("RuntimeError:") for reason in record.failure_reasons))


def ineligible_frame(reasons=("degradation_not_weak_conditioning_only",)):
    """What Milestone 3 emits for a frame outside the accepted seed class.

    The seed-class rule -- DEGRADED, five inliers, and every degradation reason
    a ``weak_conditioning:*`` -- is owned by ``ShadowOrchestrator`` and arrives
    here as ``eligible=False``. The refiner deliberately does not re-derive it:
    a second copy of that rule would be free to drift from the one that decides
    what the extractor actually ran on.
    """
    return dataclasses.replace(
        shadow_frame(ALL_FIVE), eligible=False, eligibility_reasons=reasons
    )


class EligibilityTests(unittest.TestCase):
    def test_an_ineligible_baseline_abstains_without_a_challenger(self):
        record = refined(frame=ineligible_frame())
        self.assertIs(record.status, HybridFrameStatus.ABSTAINED)
        self.assertFalse(record.eligible)
        self.assertTrue(record.eligibility_reasons)
        self.assertIsNone(record.challenger_candidate_id)

    def test_the_abstention_reason_is_carried_through_verbatim(self):
        """Why a frame was skipped has to survive into the ledger."""
        record = refined(frame=ineligible_frame(("inliers:4<5",)))
        self.assertIn("inliers:4<5", record.eligibility_reasons)

    def test_an_abstention_still_records_the_baseline(self):
        """Every frame gets a disposition; a complete ledger is the point."""
        record = refined(frame=ineligible_frame())
        self.assertEqual(len(record.transforms), 1)
        self.assertEqual(record.transforms[0].source, CandidateSource.KEYPOINT)

    def test_an_unusable_baseline_abstains(self):
        cal = dataclasses.replace(
            calibration(),
            status=CalibrationStatus.UNCALIBRATED_INSUFFICIENT_EVIDENCE,
            h_court_to_image=None, h_image_to_court=None, quality=None,
        )
        record = refined(cal=cal)
        self.assertIs(record.status, HybridFrameStatus.ABSTAINED)
        self.assertIsNone(record.challenger_candidate_id)


class IsolationTests(unittest.TestCase):
    """The package's architectural walls, which nothing else enforces."""

    def _imports(self, path):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        found = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module)
        return found

    def test_the_engine_never_imports_the_benchmark(self):
        """Scoring lives outside the engine so runtime code cannot tune itself
        against the modules that grade it."""
        for path in sorted(HYBRID_DIR.rglob("*.py")):
            with self.subTest(module=path.name):
                bad = {m for m in self._imports(path)
                       if m.split(".")[0] in {"benchmark", "ultralytics", "torch"}}
                self.assertEqual(bad, set())

    def test_the_evidence_layer_does_not_depend_on_the_fitting_layer(self):
        """``calibration/markings`` must stay replaceable; the dependency runs
        one way only.

        ``contracts.hybrid_types`` is deliberately not caught here. It is the
        shared, model-neutral record vocabulary, and the evidence layer is
        supposed to speak it -- that is the whole point of it living in
        ``contracts``. The forbidden import is ``calibration.hybrid``, the
        fitting package.
        """
        markings = HYBRID_DIR.parent / "markings"
        for path in sorted(markings.rglob("*.py")):
            with self.subTest(module=path.name):
                bad = {
                    m for m in self._imports(path)
                    if m == "calibration.hybrid" or m.startswith("calibration.hybrid.")
                }
                self.assertEqual(bad, set())

    def test_only_the_io_module_writes_files(self):
        for path in sorted(HYBRID_DIR.rglob("*.py")):
            if path.name in {"io.py", "overlays.py"}:
                continue
            text = path.read_text(encoding="utf-8")
            with self.subTest(module=path.name):
                for marker in ('open(', "write_text", "mkdir("):
                    self.assertNotIn(marker, text, f"{path.name} appears to write files")

    def test_the_scanner_catches_a_planted_violation(self):
        tree = ast.parse("import benchmark.metrics\nfrom benchmark import geometry\n")
        found = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module)
        self.assertTrue({m for m in found if m.split(".")[0] == "benchmark"})


def _seed_theta():
    from calibration.hybrid.control_points import control_points_from_homography

    return (control_points_from_homography(TRUTH, LAYOUT) + 6.0).reshape(-1)


def _seed_matrix():
    from calibration.hybrid.control_points import (
        control_court_points,
        homography_from_control_points,
    )

    return homography_from_control_points(
        control_court_points(LAYOUT), np.asarray(_seed_theta()).reshape(4, 2)
    )


if __name__ == "__main__":
    unittest.main()
