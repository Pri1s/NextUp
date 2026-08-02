"""The residual model must be perpendicular-only, balanced, and unbiased.

Every test here works from a *planted* transform: the layout is projected through
a known homography to manufacture flawless evidence, the seed is displaced, and
the refiner is asked to find its way back. If it cannot recover a transform from
exact, zero-noise evidence it will not do anything trustworthy with real
broadcast paint, and the gates would describe the result as measured and sound.

The two invariance tests are the load-bearing ones. A sample of paint says only
"the line passes this far from me" -- it says nothing about *where along* the line
it sits, and it does not become more informative because the detector happened to
place more samples nearby. Sliding and density invariance are those two claims
made executable.
"""

from __future__ import annotations

import dataclasses
import unittest

import numpy as np

from calibration.estimator import project
from calibration.hybrid.control_points import (
    DEFAULT_DEPTH_BANDS,
    band_indices,
    control_court_points,
    control_points_from_homography,
    homography_from_control_points,
    probe_court_points,
)
from calibration.hybrid.orchestrator import HybridRefinerConfig
from calibration.hybrid.refine import refine
from calibration.hybrid.residuals import (
    collect_observations,
    marking_residuals,
    measure_feature_residuals,
)
from calibration.tests.test_estimator import PLANTED
from contracts.calibration_types import Correspondence
from contracts.court_layout import HalfCourtLandmark, load_registered_layout
from contracts.hybrid_types import (
    AssignmentAlternative,
    AssignmentStatus,
    AssociationDiagnostics,
    EvidenceKind,
    ImageMarkingSample,
    MarkingAssignment,
    ShadowFrameStatus,
    ShadowMarkingFrame,
    UnlabeledMarkingEvidence,
)
from contracts.markings import MarkingFamily, MarkingFeature

LAYOUT = load_registered_layout("nba_halfcourt")
TRUTH = np.array(PLANTED, dtype=np.float64)
IMAGE_SHAPE = (720, 1280)

STRAIGHTS = (
    MarkingFeature.LANE_EDGE_FAR,
    MarkingFeature.LANE_EDGE_NEAR,
    MarkingFeature.FREE_THROW_LINE,
    MarkingFeature.THREE_POINT_CORNER_FAR,
)
ALL_FIVE = STRAIGHTS + (MarkingFeature.THREE_POINT_ARC,)

#: The five landmarks every eligible frame actually has, spanning 19 ft of depth
#: with three of them collinear on x = 0. This weak conditioning is the reason
#: the milestone exists.
INLIER_LANDMARKS = (
    "baseline_sideline_far",
    "three_point_baseline_far",
    "lane_baseline_far",
    "lane_free_throw_far",
    "lane_free_throw_near",
)


def evidence_for(feature, count=40, offset=0.0, normal_offset_px=0.0):
    """Exact projected samples along a feature, positioned deterministically.

    ``offset`` slides the samples along the primitive without changing which
    line they lie on -- the input to sliding invariance. ``normal_offset_px``
    pushes them off the line, used to manufacture a decoy.
    """
    dense = np.asarray(LAYOUT.marking(feature).sample(801), dtype=np.float64)
    span = np.linspace(0.10 + offset, 0.90 + offset, count)
    index = np.clip((span * (len(dense) - 1)).astype(int), 0, len(dense) - 1)
    image = project(TRUTH, dense[index])
    if normal_offset_px:
        image = image + np.array([0.0, normal_offset_px])
    samples = tuple(
        ImageMarkingSample(float(x), float(y), None, 0.9, 1.0, 2.0) for x, y in image
    )
    return UnlabeledMarkingEvidence(
        f"e:{feature.value}", "planted", "test", EvidenceKind.UNKNOWN, samples, 0.9
    )


def shadow_frame(features=ALL_FIVE, count=40, offset=0.0, decoys=()):
    evidence, assignments = [], []
    for feature in features:
        item = evidence_for(feature, count, offset)
        evidence.append(item)
        assignments.append(_accepted(item.evidence_id, feature))
    for feature, push in decoys:
        item = evidence_for(feature, count, offset, normal_offset_px=push)
        item = dataclasses.replace(item, evidence_id=f"decoy:{feature.value}")
        evidence.append(item)
        assignments.append(_accepted(item.evidence_id, feature))
    return ShadowMarkingFrame(
        frame_id="planted", image_sha256="a" * 64, baseline_status="DEGRADED",
        status=ShadowFrameStatus.OK, eligible=True, eligibility_reasons=(),
        layout_id=LAYOUT.layout_id, layout_hash=LAYOUT.content_hash(),
        config_hash="c" * 64, image_width=1280, image_height=720,
        roi_pixels=100000, excluded_pixels=0,
        evidence=tuple(evidence), assignments=tuple(assignments),
    )


def _accepted(evidence_id, feature):
    return MarkingAssignment(
        evidence_id, AssignmentStatus.ACCEPTED, feature,
        (AssignmentAlternative(feature, 0.95),),
        AssociationDiagnostics(1.0, 2.0, 0.9, 0.5), ("unique",),
    )


def correspondences(nudge=None):
    """The five model landmarks, projected exactly; ``nudge`` displaces one."""
    result = []
    for name in INLIER_LANDMARKS:
        court = LAYOUT.coordinate(HalfCourtLandmark(name))
        image = project(TRUTH, np.array([court]))[0]
        if nudge and nudge[0] == name:
            image = image + np.asarray(nudge[1], dtype=np.float64)
        result.append(
            Correspondence(name, (float(image[0]), float(image[1])), court, 1.0, (0,))
        )
    return tuple(result)


def recover(features=ALL_FIVE, seed_px=6.0, count=40, offset=0.0,
            config=None, decoys=(), nudge=None):
    """Displace the seed, refine, and report the result plus its probe error."""
    config = config or HybridRefinerConfig()
    theta0 = (control_points_from_homography(TRUTH, LAYOUT) + seed_px).reshape(-1)
    seed = homography_from_control_points(control_court_points(LAYOUT), theta0.reshape(4, 2))
    frame = shadow_frame(features, count, offset, decoys)
    observations = collect_observations(frame, LAYOUT, seed, config, IMAGE_SHAPE)
    result = refine(theta0, correspondences(nudge), observations, LAYOUT, config, IMAGE_SHAPE)
    return result, observations


def probe_error(matrix):
    """Max displacement from truth over the canonical probe points, in pixels."""
    probes = probe_court_points(LAYOUT)
    return np.linalg.norm(project(matrix, probes) - project(TRUTH, probes), axis=1)


def band_error(matrix):
    probes = probe_court_points(LAYOUT)
    index = band_indices(probes[:, 0], DEFAULT_DEPTH_BANDS)
    distance = probe_error(matrix)
    return {
        name: float(distance[index == position].max())
        for position, (name, _, _) in enumerate(DEFAULT_DEPTH_BANDS)
    }


class PlantedRecoveryTests(unittest.TestCase):
    def test_recovers_a_planted_transform(self):
        result, _ = recover()
        self.assertIsNotNone(result.h_court_to_image)
        self.assertLess(float(probe_error(result.h_court_to_image).max()), 0.05)

    def test_recovers_across_seed_distance(self):
        for seed_px in (0.5, 2.0, 6.0):
            with self.subTest(seed_px=seed_px):
                result, _ = recover(seed_px=seed_px)
                self.assertLess(float(probe_error(result.h_court_to_image).max()), 0.05)

    def test_arc_alone_recovers(self):
        """Regression for the LM damping bug.

        The arc used to sit ~1.8 px out because ``mu`` was initialised once
        outside the outer loop. Re-freezing the feet defines a new objective, so
        inherited damping describes a function that no longer exists: one hard
        iteration inflated ``mu``, after which no descent step existed, the
        solve exited as a stall, and it reported wherever it was standing as
        converged. Reinstating that bug puts this back above 1 px.

        The arc is the sensitive case because it is the only fitted primitive
        carrying depth past the 14.26 ft break, so it is the one whose evidence
        the far-field solution actually depends on.
        """
        result, _ = recover(features=(MarkingFeature.THREE_POINT_ARC,))
        self.assertLess(float(probe_error(result.h_court_to_image).max()), 0.05)

    def test_near_field_recovers_without_any_deep_evidence(self):
        """Straights reach only 19 ft, so judge them where they have evidence.

        Their far-field residual is not an error: with nothing past 19 ft that
        region is weakly constrained and the prior deliberately holds it near
        the seed. See ``ArcDepthContributionTests`` for the other half.
        """
        result, _ = recover(features=STRAIGHTS)
        self.assertLess(band_error(result.h_court_to_image)["0-19ft"], 0.05)

    def test_a_perfect_seed_is_not_disturbed(self):
        result, _ = recover(seed_px=0.0)
        self.assertLess(float(probe_error(result.h_court_to_image).max()), 0.01)


class ArcDepthContributionTests(unittest.TestCase):
    """The milestone's thesis, as an assertion rather than a claim."""

    def test_arc_evidence_materially_improves_the_far_court(self):
        without, _ = recover(features=STRAIGHTS)
        with_arc, _ = recover(features=ALL_FIVE)
        far_without = band_error(without.h_court_to_image)["30-47ft"]
        far_with = band_error(with_arc.h_court_to_image)["30-47ft"]
        self.assertLess(far_with, far_without / 4.0)

    def test_arc_evidence_reaches_past_the_break(self):
        _, observations = recover(features=ALL_FIVE)
        arc = [o for o in observations if o.feature is MarkingFeature.THREE_POINT_ARC]
        self.assertEqual(len(arc), 1)
        # Corner straights stop at 14.26 ft and the lane at 19.0; only the arc
        # carries depth past the break, which is what the depth gate keys on.
        self.assertGreater(float(arc[0].depth_ft.max()), 24.0)


class InvarianceTests(unittest.TestCase):
    def test_sliding_samples_along_a_line_does_not_move_the_transform(self):
        """A sample constrains distance *to* its line, not position along it.

        Fragment endpoints are not court landmarks. If they leaked into the fit
        this drifts, because every offset presents different endpoints for the
        same geometry.
        """
        baseline = None
        for offset in (0.0, 0.02, 0.04, 0.06):
            with self.subTest(offset=offset):
                result, _ = recover(offset=offset)
                probes = project(result.h_court_to_image, probe_court_points(LAYOUT))
                if baseline is None:
                    baseline = probes
                    continue
                drift = float(np.max(np.linalg.norm(probes - baseline, axis=1)))
                self.assertLess(drift, 1e-3)

    def test_sample_density_does_not_move_the_transform(self):
        """``resample_spacing_px`` is a detector tuning knob.

        Weighting by sample count would make the fitted transform a function of
        it. Arc-length territory weighting is what stops that.
        """
        baseline = None
        for count in (20, 40, 80, 160):
            with self.subTest(count=count):
                result, _ = recover(count=count)
                probes = project(result.h_court_to_image, probe_court_points(LAYOUT))
                if baseline is None:
                    baseline = probes
                    continue
                drift = float(np.max(np.linalg.norm(probes - baseline, axis=1)))
                self.assertLess(drift, 0.05)


class BudgetTests(unittest.TestCase):
    def test_marking_weight_never_reaches_the_landmark_total(self):
        """Five semantic landmarks must keep a strict majority over the paint."""
        config = HybridRefinerConfig()
        _, observations = recover(config=config)
        total = sum(float(np.sum(item.weight)) for item in observations)
        self.assertLessEqual(total, config.global_cap + 1e-9)
        self.assertLess(total, float(len(INLIER_LANDMARKS)))

    def test_no_family_exceeds_its_cap(self):
        """The lane has three near-parallel members; uncapped it outvotes the arc."""
        config = HybridRefinerConfig()
        _, observations = recover(config=config)
        totals: dict[MarkingFamily, float] = {}
        for item in observations:
            totals[item.feature.family] = totals.get(item.feature.family, 0.0) + float(np.sum(item.weight))
        for family, value in totals.items():
            with self.subTest(family=family.value):
                self.assertLessEqual(value, config.family_cap + 1e-9)

    def test_every_fitted_primitive_carries_some_weight(self):
        _, observations = recover()
        for item in observations:
            with self.subTest(feature=item.feature.value):
                self.assertGreater(float(np.sum(item.weight)), 0.0)


class RobustnessTests(unittest.TestCase):
    def test_a_decoy_fragment_barely_moves_the_transform(self):
        """Cauchy redescends, so paint that survived association can still be
        abandoned by the fit -- the last line of defence against a false line."""
        clean, _ = recover()
        with_decoy, _ = recover(decoys=((MarkingFeature.LANE_EDGE_FAR, 40.0),))
        probes = probe_court_points(LAYOUT)
        moved = np.linalg.norm(
            project(with_decoy.h_court_to_image, probes)
            - project(clean.h_court_to_image, probes),
            axis=1,
        )
        self.assertLess(float(moved.max()), 1.0)

    def test_a_displaced_landmark_is_not_quietly_discarded(self):
        """Huber does not redescend, deliberately.

        A redescending kernel would let the optimizer walk away from a model
        landmark, and the gate that checks "model points stayed within
        tolerance" would then be measuring a point the fit had already given up
        on. The residual must stay large and visible.
        """
        name = INLIER_LANDMARKS[0]
        result, _ = recover(nudge=(name, (30.0, 0.0)))
        moved = [c for c in correspondences(nudge=(name, (30.0, 0.0))) if c.landmark_id == name][0]
        landed = project(result.h_court_to_image, np.array([moved.court_xy]))[0]
        residual = float(np.linalg.norm(landed - np.asarray(moved.image_xy)))
        self.assertGreater(residual, 20.0)


def measured(features, matrix=TRUTH, config=None):
    """Per-feature fresh residuals, through the same call the records use."""
    config = config or HybridRefinerConfig()
    return measure_feature_residuals(shadow_frame(features), LAYOUT, matrix, config)


class RecordedResidualTests(unittest.TestCase):
    """These exercise ``measure_feature_residuals`` -- the function that actually
    populates ``PrimitiveQuality`` -- rather than a parallel implementation."""

    def test_a_sample_on_a_straight_has_machine_precision_residual(self):
        """A straight template is exact, so there is nothing to excuse here."""
        rows = measured(STRAIGHTS)
        for feature in STRAIGHTS:
            with self.subTest(feature=feature.value):
                self.assertLess(max(d for d, _ in rows[feature]), 1e-9)

    def test_a_sample_on_the_arc_is_within_the_chord_approximation(self):
        """The arc template is a polyline, so a point on the true circle sits a
        sagitta outside its chord. That floor is discretisation, not error.

        Measured against template resolution it is the textbook 1/n^2:
        201 samples -> 4.26e-3 px, 401 -> 1.07e-3 px (exactly 4.00x), and at 801
        it reaches machine precision because the samples then coincide with
        template vertices. At the default 401 that is ~1.5e-4 ft, roughly 50x
        below the 0.05 px acceptance bar and far beneath the association
        corridor, so tightening it would buy nothing but template cost.
        """
        rows = measured((MarkingFeature.THREE_POINT_ARC,))
        self.assertLess(max(d for d, _ in rows[MarkingFeature.THREE_POINT_ARC]), 5e-3)

    def test_arc_chord_error_falls_with_template_resolution(self):
        """Pins the explanation above: if this stopped being ~1/n^2 the floor
        would be something other than discretisation and would need re-diagnosing."""
        errors = []
        for count in (201, 401):
            config = dataclasses.replace(HybridRefinerConfig(), arc_samples=count)
            rows = measured((MarkingFeature.THREE_POINT_ARC,), config=config)
            errors.append(max(d for d, _ in rows[MarkingFeature.THREE_POINT_ARC]))
        self.assertGreater(errors[0] / errors[1], 3.5)
        self.assertLess(errors[0] / errors[1], 4.5)

    def test_residuals_are_a_function_of_the_transform_passed_in(self):
        """The measurement must re-derive foot points against the candidate.

        If it reused correspondences cached from the fit, a wrong transform
        would still read low -- reporting the quantity the optimizer minimised
        instead of the error, and flattering the candidate exactly where it is
        worst.
        """
        displaced = homography_from_control_points(
            control_court_points(LAYOUT),
            control_points_from_homography(TRUTH, LAYOUT) + 6.0,
        )
        against_truth = max(d for d, _ in measured(ALL_FIVE)[MarkingFeature.THREE_POINT_ARC])
        against_wrong = max(
            d for d, _ in measured(ALL_FIVE, matrix=displaced)[MarkingFeature.THREE_POINT_ARC]
        )
        self.assertLess(against_truth, 5e-3)
        self.assertGreater(against_wrong, 1.0)

    def test_depth_is_read_off_the_court_primitive(self):
        """Depth comes from the court template at the matched segment, never
        from inverting the transform onto an image point.

        That is what keeps depth-band membership meaningful: the transform is
        the thing under test, so a depth derived *through* it could not be used
        to say which band its error falls in. The consequence tested here is
        that depths are always bounded by the primitive's real court extent,
        however wrong the candidate transform is.

        Foot *correspondence* does move with the transform -- a wrong transform
        matches a sample to a slightly different point along the primitive, up
        to ~0.7 ft here. That is ordinary ICP behaviour, not a leak.
        """
        arc = LAYOUT.marking(MarkingFeature.THREE_POINT_ARC)
        low = arc.center[0] - arc.radius_ft
        high = arc.center[0] + arc.radius_ft
        displaced = homography_from_control_points(
            control_court_points(LAYOUT),
            control_points_from_homography(TRUTH, LAYOUT) + 6.0,
        )
        for name, matrix in (("truth", TRUTH), ("displaced", displaced)):
            with self.subTest(matrix=name):
                depth = [d for _, d in measured(ALL_FIVE, matrix=matrix)[MarkingFeature.THREE_POINT_ARC]]
                self.assertGreaterEqual(min(depth), low - 1e-9)
                self.assertLessEqual(max(depth), high + 1e-9)
                self.assertGreater(max(depth), 24.0)


class TangentGeometryTests(unittest.TestCase):

    def test_a_foot_landing_at_the_end_of_a_segment_stays_finite(self):
        """Feet at awkward positions must still produce sound residuals.

        This is a general robustness check, not a regression test: the
        implementation takes the local direction from the whole template
        segment, which is immune to where along it the foot sits.

        Honest note on the alternative. An earlier construction took the
        direction from the interpolated foot to the *next* vertex, and it was
        replaced on principle -- a partial chord is not the polyline's tangent,
        and a near-vanishing baseline tripped a ``1e-12`` guard that silently
        zeroed the sample's constraint. But the two are not measurably
        different here: any two distinct points on a straight segment give the
        same unit vector, so even a 1e-4 baseline stays accurate in float64,
        and the guard needs ``t`` within 1e-12 of 1.0 to fire at all. No test in
        this file distinguishes them, and none should pretend to.
        """
        dense = np.asarray(
            LAYOUT.marking(MarkingFeature.THREE_POINT_ARC).sample(401), dtype=np.float64
        )
        # Sit each sample 99.99% of the way along its segment, so the secant to
        # the next vertex is ~1e-4 of a segment long.
        starts, ends = dense[5:25], dense[6:26]
        near_end = starts + 0.9999 * (ends - starts)
        on_vertices = project(TRUTH, near_end)
        samples = tuple(
            ImageMarkingSample(float(x), float(y), None, 0.9, 1.0, 2.0) for x, y in on_vertices
        )
        item = UnlabeledMarkingEvidence(
            "e:vertex", "planted", "test", EvidenceKind.CURVED, samples, 0.9
        )
        frame = ShadowMarkingFrame(
            frame_id="planted", image_sha256="a" * 64, baseline_status="DEGRADED",
            status=ShadowFrameStatus.OK, eligible=True, eligibility_reasons=(),
            layout_id=LAYOUT.layout_id, layout_hash=LAYOUT.content_hash(),
            config_hash="c" * 64, image_width=1280, image_height=720,
            roi_pixels=100000, excluded_pixels=0,
            evidence=(item,),
            assignments=(_accepted("e:vertex", MarkingFeature.THREE_POINT_ARC),),
        )
        observations = collect_observations(
            frame, LAYOUT, TRUTH, HybridRefinerConfig(), IMAGE_SHAPE
        )
        theta = control_points_from_homography(TRUTH, LAYOUT).reshape(-1)
        residuals = marking_residuals(theta, observations, LAYOUT)
        self.assertEqual(len(residuals), len(samples))
        self.assertTrue(bool(np.all(np.isfinite(residuals))))
        self.assertLess(float(np.max(np.abs(residuals))), 1e-6)



if __name__ == "__main__":
    unittest.main()
