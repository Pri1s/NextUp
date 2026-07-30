"""Turning two independent passes into one benchmark, and a queue of arguments.

The design plan's §7.4 asks for two blind passes, agreement where they match, and
adjudication where they do not. Four details decide whether that produces evidence
or noise.

**Compare over the shared extent only.** Two annotators rarely trace the same run
of a marking -- one starts where a player's leg ends, the other a metre further
along. Scoring the union would count that coverage difference as disagreement,
which it is not: it says nothing about whether they agree on *where the line is*.

**Check the direction before averaging.** Nothing stops one pass tracing an arc
left-to-right and the other right-to-left. Averaging those pointwise pairs the
start of one with the end of the other and produces a curve lying on neither, so
direction is normalised first.

**Average only after agreement is established.** §7.4 forbids averaging two labels
that may describe different physical markings, and it is right: the midpoint of
two different markings is a third thing that is on neither. Once the two are known
to describe the same paint, averaging is just noise reduction and is safe.

**Reject identity conflicts rather than resolving them.** If one pass calls a piece
of paint the three-point arc and the other calls it a logo edge or a sideline, at
most one is right and possibly neither. That is a question for an adjudicator
looking at the image, not for arithmetic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
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
    Visibility,
)

from calibration.estimator import estimate_homography, matrix_to_array, project
from contracts.calibration_types import Correspondence
from contracts.court_layout import HalfCourtLandmark, HalfCourtLayout

from .geometry import marking_polylines
from .polyline import nearest_on_polyline, overlapping_extent, resample

CONSENSUS_SCHEMA_VERSION = "benchmark-consensus-1.0.0"

#: Agreement thresholds, matching the pilot gate. A benchmark meant to certify a
#: 4 px median error needs annotation noise well below that.
AGREE_MEDIAN_PX = 2.0
AGREE_P95_PX = 6.0

#: How close two points must be to count as describing the same paint. Adjacent
#: court markings are metres apart on the floor, so this only fires when two
#: passes genuinely traced the same stripe.
SAME_PAINT_PX = 4.0

#: A conflict needs real overlap, not one stray sample near another marking.
CONFLICT_MIN_FRACTION = 0.4

#: Extent search tolerance -- generous, because it decides what is *comparable*,
#: not what agrees.
OVERLAP_TOLERANCE_PX = 30.0

#: Junction landmarks the annotation vocabulary shares with the court layout, for
#: the internal-consistency fit. Curved evidence is deliberately excluded: fitting
#: a homography to marking curves is the Milestone 3/4 problem, not this one.
_JUNCTION_TO_LANDMARK: dict[JunctionPoint, HalfCourtLandmark] = {
    JunctionPoint.BASELINE_SIDELINE_FAR: HalfCourtLandmark.BASELINE_SIDELINE_FAR,
    JunctionPoint.BASELINE_SIDELINE_NEAR: HalfCourtLandmark.BASELINE_SIDELINE_NEAR,
    JunctionPoint.THREE_POINT_BASELINE_FAR: HalfCourtLandmark.THREE_POINT_BASELINE_FAR,
    JunctionPoint.THREE_POINT_BASELINE_NEAR: HalfCourtLandmark.THREE_POINT_BASELINE_NEAR,
    JunctionPoint.LANE_BASELINE_FAR: HalfCourtLandmark.LANE_BASELINE_FAR,
    JunctionPoint.LANE_BASELINE_NEAR: HalfCourtLandmark.LANE_BASELINE_NEAR,
    JunctionPoint.LANE_FREE_THROW_FAR: HalfCourtLandmark.LANE_FREE_THROW_FAR,
    JunctionPoint.LANE_FREE_THROW_NEAR: HalfCourtLandmark.LANE_FREE_THROW_NEAR,
    JunctionPoint.MIDCOURT_SIDELINE_FAR: HalfCourtLandmark.MIDCOURT_SIDELINE_FAR,
    JunctionPoint.MIDCOURT_SIDELINE_NEAR: HalfCourtLandmark.MIDCOURT_SIDELINE_NEAR,
}


def _points_of(item: AnnotatedFeature) -> np.ndarray:
    return np.array([[p.x, p.y] for p in item.points], dtype=np.float64)


def _percentile(values: np.ndarray, fraction: float) -> float:
    if len(values) == 0:
        return float("nan")
    ordered = np.sort(values)
    rank = max(1, min(len(ordered), int(round(fraction * len(ordered) + 0.5))))
    return float(ordered[rank - 1])


def _align_direction(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Reverse ``b`` if it traces the same curve the other way round.

    Without this, averaging pairs one pass's start with the other's end. The result
    is a curve on neither marking, and it would enter the silver set looking like
    a perfectly ordinary annotation.
    """
    if len(a) < 2 or len(b) < 2:
        return b
    same = np.linalg.norm(a[0] - b[0]) + np.linalg.norm(a[-1] - b[-1])
    flipped = np.linalg.norm(a[0] - b[-1]) + np.linalg.norm(a[-1] - b[0])
    return b[::-1] if flipped < same else b


@dataclass(frozen=True, slots=True)
class FeatureAgreement:
    """How two passes compare on one feature."""

    feature: MarkingFeature
    verdict: str  # agreed | disagreed | only_a | only_b | no_overlap
    shared_samples: int
    median_px: float
    p95_px: float
    max_px: float

    @property
    def agreed(self) -> bool:
        return self.verdict == "agreed"

    def as_dict(self) -> dict:
        return {
            "feature": self.feature.value,
            "verdict": self.verdict,
            "shared_samples": self.shared_samples,
            "median_px": self.median_px,
            "p95_px": self.p95_px,
            "max_px": self.max_px,
        }


@dataclass(frozen=True, slots=True)
class IdentityConflict:
    """Two passes gave the same paint different names.

    The most damaging disagreement there is, and the only one where averaging
    would be actively wrong rather than merely imprecise.
    """

    feature_a: MarkingFeature
    feature_b: MarkingFeature
    overlap_fraction: float
    median_px: float

    def as_dict(self) -> dict:
        return {
            "feature_a": self.feature_a.value,
            "feature_b": self.feature_b.value,
            "overlap_fraction": self.overlap_fraction,
            "median_px": self.median_px,
        }


@dataclass(frozen=True, slots=True)
class FrameConsensus:
    """The verdict on one frame."""

    frame_id: str
    agreements: tuple[FeatureAgreement, ...]
    junction_agreements: tuple[tuple[str, float, bool], ...]
    conflicts: tuple[IdentityConflict, ...]
    silver: FrameAnnotation
    disputed: tuple[MarkingFeature, ...]
    disputed_junctions: tuple[JunctionPoint, ...]
    geometry_check: dict
    problems: tuple[str, ...] = ()

    @property
    def agreed_features(self) -> tuple[MarkingFeature, ...]:
        return tuple(a.feature for a in self.agreements if a.agreed)

    def as_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "agreements": [a.as_dict() for a in self.agreements],
            "junction_agreements": [
                {"junction": name, "distance_px": distance, "agreed": agreed}
                for name, distance, agreed in self.junction_agreements
            ],
            "conflicts": [c.as_dict() for c in self.conflicts],
            "disputed": [f.value for f in self.disputed],
            "disputed_junctions": [j.value for j in self.disputed_junctions],
            "geometry_check": self.geometry_check,
            "problems": list(self.problems),
        }


def compare_feature(a: AnnotatedFeature, b: AnnotatedFeature) -> FeatureAgreement:
    """Symmetric nearest-point disagreement over the extent both passes covered."""
    points_a, points_b = _points_of(a), _points_of(b)
    shared_a, shared_b = overlapping_extent(points_a, points_b, OVERLAP_TOLERANCE_PX)

    if len(shared_a) < 2 or len(shared_b) < 2:
        return FeatureAgreement(
            a.feature, "no_overlap", 0, float("nan"), float("nan"), float("nan")
        )

    shared_b = _align_direction(shared_a, shared_b)

    # Measured both ways: A's samples against B's curve and B's against A's. One
    # direction alone would let a pass that traced only the easy half of a marking
    # score perfectly against one that traced all of it.
    both = np.concatenate(
        [
            nearest_on_polyline(shared_a, shared_b).distances,
            nearest_on_polyline(shared_b, shared_a).distances,
        ]
    )
    median = float(np.median(both))
    p95 = _percentile(both, 0.95)
    verdict = "agreed" if (median <= AGREE_MEDIAN_PX and p95 <= AGREE_P95_PX) else "disagreed"

    return FeatureAgreement(
        feature=a.feature,
        verdict=verdict,
        shared_samples=len(shared_a) + len(shared_b),
        median_px=median,
        p95_px=p95,
        max_px=float(np.max(both)),
    )


def merge_feature(a: AnnotatedFeature, b: AnnotatedFeature) -> AnnotatedFeature:
    """Average two agreeing traces of the same marking.

    Safe only once agreement is established. Both are resampled by arc length over
    their shared extent so the average is not weighted toward whichever pass placed
    more points in a region.
    """
    points_a, points_b = _points_of(a), _points_of(b)
    shared_a, shared_b = overlapping_extent(points_a, points_b, OVERLAP_TOLERANCE_PX)
    shared_b = _align_direction(shared_a, shared_b)

    count = max(2, min(len(shared_a), len(shared_b)))
    averaged = (resample(shared_a, count) + resample(shared_b, count)) / 2.0

    visibility = (
        Visibility.CLEAR
        if a.visibility is Visibility.CLEAR and b.visibility is Visibility.CLEAR
        else Visibility.PARTIALLY_OCCLUDED
    )
    return AnnotatedFeature(
        feature=a.feature,
        points=tuple(SamplePoint(x=float(x), y=float(y)) for x, y in averaged),
        visibility=visibility,
        # The more doubtful pass sets the uncertainty. Averaging the estimates
        # would let a confident mistake dilute an honest doubt.
        uncertainty_px=max(a.uncertainty_px, b.uncertainty_px),
        notes="consensus of two independent passes",
    )


def find_identity_conflicts(
    a: FrameAnnotation, b: FrameAnnotation
) -> tuple[IdentityConflict, ...]:
    """Paint that both passes traced but named differently."""
    conflicts = []
    for item_a in a.features:
        points_a = _points_of(item_a)
        for item_b in b.features:
            if item_a.feature is item_b.feature:
                continue
            points_b = _points_of(item_b)
            if len(points_a) < 2 or len(points_b) < 2:
                continue

            distances = nearest_on_polyline(points_a, points_b).distances
            close = distances <= SAME_PAINT_PX
            fraction = float(np.mean(close))
            if fraction >= CONFLICT_MIN_FRACTION:
                conflicts.append(
                    IdentityConflict(
                        feature_a=item_a.feature,
                        feature_b=item_b.feature,
                        overlap_fraction=fraction,
                        median_px=float(np.median(distances[close])),
                    )
                )
    return tuple(conflicts)


def check_internal_geometry(
    annotation: FrameAnnotation, layout: HalfCourtLayout
) -> dict:
    """Fit a homography from the annotated junctions and test the held-out curves.

    This catches labels that agree with each other and are still impossible: two
    passes can share a misidentification, but they cannot make an inconsistent set
    of junctions describe a real court. Annotations span far more court depth than
    the five confident model landmarks, so this fit is better conditioned than the
    calibration it will be used to judge.

    Curved families are the test set, never the training set -- the same held-out
    discipline §8.3 asks of the hybrid fitter.
    """
    usable = [
        (j, _JUNCTION_TO_LANDMARK[j.junction])
        for j in annotation.junctions
        if j.junction in _JUNCTION_TO_LANDMARK
        and layout.has(_JUNCTION_TO_LANDMARK[j.junction])
    ]
    if len(usable) < 4:
        return {
            "status": "skipped",
            "reason": f"only {len(usable)} usable junctions; a homography needs 4",
        }

    correspondences = tuple(
        Correspondence(
            landmark_id=landmark.value,
            court_xy=layout.coordinate(landmark),
            image_xy=(junction.point.x, junction.point.y),
            confidence=1.0,
        )
        for junction, landmark in usable
    )
    fit = estimate_homography(correspondences)
    if not fit.solved:
        return {"status": "failed", "reason": fit.failure, "junctions": len(usable)}

    matrix = matrix_to_array(fit.h_court_to_image)
    polylines = marking_polylines(layout)
    held_out: dict[str, float] = {}
    for item in annotation.features:
        if item.feature.kind != "arc":
            continue
        projected = project(matrix, polylines[item.feature].points)
        if projected is None or not np.all(np.isfinite(projected)):
            continue
        distances = nearest_on_polyline(_points_of(item), projected).distances
        held_out[item.feature.value] = float(np.median(distances))

    return {
        "status": "ok",
        "junctions": len(usable),
        "solver": fit.solver,
        "inliers": list(fit.inlier_landmarks),
        "outliers": list(fit.outlier_landmarks),
        "held_out_median_px": held_out,
    }


def build_consensus(
    pass_a: FrameAnnotation,
    pass_b: FrameAnnotation,
    layout: HalfCourtLayout,
) -> FrameConsensus:
    """Compare two passes over one frame and assemble the silver annotation."""
    problems: list[str] = []
    if pass_a.frame_id != pass_b.frame_id:
        raise ValueError(f"different frames: {pass_a.frame_id!r} vs {pass_b.frame_id!r}")
    if pass_a.image_sha256 != pass_b.image_sha256:
        problems.append("the two passes annotated different images")
    if pass_a.prompt_version != pass_b.prompt_version:
        problems.append(
            f"different prompt versions ({pass_a.prompt_version} vs "
            f"{pass_b.prompt_version}); these labels are not comparable"
        )
    if pass_a.pass_id == pass_b.pass_id:
        problems.append(
            f"both inputs declare pass_id {pass_a.pass_id!r}; isolated passes need "
            "different pass IDs"
        )
    for annotation, name in ((pass_a, "a"), (pass_b, "b")):
        if annotation.coordinate_space != "frame":
            raise ValueError(f"pass {name} is in crop space; resolve it first")

    conflicts = find_identity_conflicts(pass_a, pass_b)
    conflicted = {f for c in conflicts for f in (c.feature_a, c.feature_b)}

    agreements: list[FeatureAgreement] = []
    merged: list[AnnotatedFeature] = []
    merged_skips: list[SkippedFeature] = []
    disputed: list[MarkingFeature] = []
    skips_a = {entry.feature: entry for entry in pass_a.skipped}
    skips_b = {entry.feature: entry for entry in pass_b.skipped}

    dispositioned = (
        pass_a.annotated_features
        | pass_b.annotated_features
        | frozenset(skips_a)
        | frozenset(skips_b)
    )
    for feature in sorted(dispositioned, key=lambda f: f.value):
        item_a, item_b = pass_a.feature(feature), pass_b.feature(feature)
        skipped_a, skipped_b = skips_a.get(feature), skips_b.get(feature)
        if item_a is None and item_b is None and skipped_a and skipped_b:
            agreements.append(
                FeatureAgreement(
                    feature, "both_skipped", 0,
                    float("nan"), float("nan"), float("nan"),
                )
            )
            same_reason = skipped_a.reason is skipped_b.reason
            merged_skips.append(
                SkippedFeature(
                    feature,
                    skipped_a.reason if same_reason else SkipReason.AMBIGUOUS_IDENTITY,
                    "both passes skipped"
                    + ("" if same_reason else f" ({skipped_a.reason.value}/{skipped_b.reason.value})"),
                )
            )
            continue
        if item_a is None or item_b is None:
            agreements.append(
                FeatureAgreement(
                    feature,
                    "only_a" if item_b is None else "only_b",
                    0, float("nan"), float("nan"), float("nan"),
                )
            )
            disputed.append(feature)
            continue

        agreement = compare_feature(item_a, item_b)
        agreements.append(agreement)
        if agreement.agreed and feature not in conflicted:
            merged.append(merge_feature(item_a, item_b))
        else:
            disputed.append(feature)

    junction_agreements = []
    disputed_junctions: list[JunctionPoint] = []
    merged_junctions: list[AnnotatedJunction] = []
    for junction in sorted(
        {j.junction for j in pass_a.junctions} | {j.junction for j in pass_b.junctions},
        key=lambda j: j.value,
    ):
        found_a = next((j for j in pass_a.junctions if j.junction is junction), None)
        found_b = next((j for j in pass_b.junctions if j.junction is junction), None)
        if found_a is None or found_b is None:
            junction_agreements.append((junction.value, float("nan"), False))
            disputed_junctions.append(junction)
            continue
        distance = float(
            np.hypot(found_a.point.x - found_b.point.x, found_a.point.y - found_b.point.y)
        )
        agreed = distance <= AGREE_MEDIAN_PX * 2
        junction_agreements.append((junction.value, distance, agreed))
        if agreed:
            merged_junctions.append(
                AnnotatedJunction(
                    junction=junction,
                    point=SamplePoint(
                        x=(found_a.point.x + found_b.point.x) / 2.0,
                        y=(found_a.point.y + found_b.point.y) / 2.0,
                    ),
                    uncertainty_px=max(found_a.uncertainty_px, found_b.uncertainty_px),
                    notes="consensus of two independent passes",
                )
            )
        else:
            disputed_junctions.append(junction)

    # A junction survives only if both of its markings did. Otherwise the silver
    # set would assert a landmark whose supporting geometry it does not contain.
    accepted = {item.feature for item in merged}
    merged_junctions = [
        j for j in merged_junctions if all(f in accepted for f in j.junction.crossing)
    ]

    disputed_features = tuple(sorted(set(disputed), key=lambda feature: feature.value))
    silver = FrameAnnotation(
        frame_id=pass_a.frame_id,
        clip=pass_a.clip,
        frame_index=pass_a.frame_index,
        image_width=pass_a.image_width,
        image_height=pass_a.image_height,
        image_sha256=pass_a.image_sha256,
        annotator_id=f"consensus({pass_a.annotator_id}+{pass_b.annotator_id})",
        pass_id="silver",
        prompt_version=pass_a.prompt_version,
        coordinate_space="frame",
        features=tuple(merged),
        junctions=tuple(merged_junctions),
        skipped=tuple(merged_skips) + tuple(
            SkippedFeature(feature, SkipReason.AMBIGUOUS_IDENTITY, "passes disagreed")
            for feature in disputed_features
        ),
        notes="silver: features where two independent passes agreed within tolerance",
    )

    geometry = check_internal_geometry(silver, layout)

    return FrameConsensus(
        frame_id=pass_a.frame_id,
        agreements=tuple(agreements),
        junction_agreements=tuple(junction_agreements),
        conflicts=conflicts,
        silver=silver,
        disputed=disputed_features,
        disputed_junctions=tuple(disputed_junctions),
        geometry_check=geometry,
        problems=tuple(problems),
    )


def load_pass(directory: Path | str, frame_id: str) -> FrameAnnotation | None:
    """Prefer the resolved frame-space file; a crop-space one cannot be compared."""
    directory = Path(directory)
    for name in (f"{frame_id}.frame.json", f"{frame_id}.json"):
        path = directory / name
        if path.is_file():
            return FrameAnnotation.from_dict(json.loads(path.read_text(encoding="utf-8")))
    return None


def write_consensus(
    out_dir: Path | str, results: list[FrameConsensus]
) -> tuple[Path, Path, Path]:
    """Write silver labels, the detailed report, and a sanitized actual-dispute queue."""
    out_dir = Path(out_dir)
    silver_dir = out_dir / "silver"
    silver_dir.mkdir(parents=True, exist_ok=True)

    for result in results:
        path = silver_dir / f"{result.frame_id}.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(result.silver.as_dict(), handle, indent=2)
            handle.write("\n")

    report = out_dir / "consensus_report.json"
    with open(report, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": CONSENSUS_SCHEMA_VERSION,
                "frames": [r.as_dict() for r in results],
            },
            handle,
            indent=2,
        )
        handle.write("\n")

    # The adjudicator gets identities but never prior coordinates or disagreement
    # distances. Omitting frames with no dispute is what prevents a needless third
    # annotation pass.
    queue = out_dir / "adjudication_queue.json"
    tasks = []
    for result in results:
        if not result.disputed and not result.disputed_junctions:
            continue
        tasks.append(
            {
                "frame_id": result.frame_id,
                "features": [feature.value for feature in result.disputed],
                "junctions": [junction.value for junction in result.disputed_junctions],
                "identity_candidates": [
                    sorted((conflict.feature_a.value, conflict.feature_b.value))
                    for conflict in result.conflicts
                ],
            }
        )
    with open(queue, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": CONSENSUS_SCHEMA_VERSION,
                "instruction": (
                    "Adjudicate only these actual disagreements in a fresh isolated "
                    "session. Do not open either annotation pass or consensus_report.json."
                ),
                "frames": tasks,
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    return silver_dir, report, queue


def summarize(results: list[FrameConsensus]) -> dict:
    """Aggregate agreement, which is the number the pilot gate turns on."""
    verdicts: dict[str, int] = {}
    medians: list[float] = []
    for result in results:
        for agreement in result.agreements:
            verdicts[agreement.verdict] = verdicts.get(agreement.verdict, 0) + 1
            if agreement.agreed or agreement.verdict == "disagreed":
                medians.append(agreement.median_px)

    comparable = verdicts.get("agreed", 0) + verdicts.get("disagreed", 0)
    return {
        "frames": len(results),
        "verdicts": verdicts,
        "agreement_rate": (verdicts.get("agreed", 0) / comparable) if comparable else 0.0,
        "median_disagreement_px": float(np.median(medians)) if medians else float("nan"),
        "p95_disagreement_px": _percentile(np.array(medians), 0.95) if medians else float("nan"),
        "identity_conflicts": sum(len(r.conflicts) for r in results),
        "frames_with_problems": sum(1 for r in results if r.problems),
    }
