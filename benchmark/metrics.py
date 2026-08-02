"""Scoring a calibration against annotated paint, split by court depth.

This is the measurement the milestone exists to produce. Everything the engine
currently reports about itself is self-consistency: reprojection error measures a
fit against the same five landmarks that produced it, and with three of them
collinear on the baseline against a homography's eight degrees of freedom, tiny
residuals are cheap while the projection drifts feet away from the floor.
``docs/CALIBRATION.md`` says as much, and says the drift is visible in the
reprojected images -- by eye. This turns it into a number.

**The depth split is the whole point.** The confident evidence spans 0-19 ft.
Inside that band a fit is interpolating between its own observations; beyond it,
extrapolating. Pooling the two hides exactly the effect being measured, because
most annotated paint is near the baseline where the fit is best. Reported
separately, the bands answer the actual question: does this transform describe the
floor, or only the points it was built from?

**Error is reported in both pixels and feet.** Pixels are what the annotator
measured and what the acceptance thresholds are written in. Feet are what anyone
downstream cares about, and the conversion is wildly non-uniform across the frame --
the same two-pixel error is inches near the camera and over a foot at the far
baseline, which is precisely why a pixel-only metric flatters a bad far-court fit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from contracts.annotations import HELD_OUT_FAMILIES, FrameAnnotation, MarkingFeature
from contracts.court_layout import HalfCourtLayout

from calibration.estimator import project

from .geometry import DEPTH_BANDS, depth_band, marking_polylines
from .polyline import nearest_on_polyline

METRICS_SCHEMA_VERSION = "benchmark-metrics-1.0.0"


def _percentile(values: np.ndarray, fraction: float) -> float:
    if len(values) == 0:
        return float("nan")
    ordered = np.sort(values)
    rank = max(1, min(len(ordered), int(round(fraction * len(ordered) + 0.5))))
    return float(ordered[rank - 1])


@dataclass(frozen=True, slots=True)
class BandError:
    """Error within one depth band."""

    band: str
    sample_count: int
    median_px: float
    p95_px: float
    max_px: float
    median_ft: float
    p95_ft: float

    def as_dict(self) -> dict:
        return {
            "band": self.band,
            "sample_count": self.sample_count,
            "median_px": self.median_px,
            "p95_px": self.p95_px,
            "max_px": self.max_px,
            "median_ft": self.median_ft,
            "p95_ft": self.p95_ft,
        }


@dataclass(frozen=True, slots=True)
class FrameScore:
    """How well one calibration explains one frame's annotated paint."""

    frame_id: str
    status: str
    scored: bool
    sample_count: int
    median_px: float
    p95_px: float
    max_px: float
    median_ft: float
    bands: tuple[BandError, ...]
    per_feature: dict
    held_out_median_px: float
    reason: str = ""

    def band(self, name: str) -> BandError | None:
        return next((b for b in self.bands if b.band == name), None)

    def as_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "status": self.status,
            "scored": self.scored,
            "reason": self.reason,
            "sample_count": self.sample_count,
            "median_px": self.median_px,
            "p95_px": self.p95_px,
            "max_px": self.max_px,
            "median_ft": self.median_ft,
            "held_out_median_px": self.held_out_median_px,
            "bands": [b.as_dict() for b in self.bands],
            "per_feature": self.per_feature,
        }


def _court_error_ft(
    image_points: np.ndarray,
    closest_image: np.ndarray,
    h_image_to_court: np.ndarray,
) -> np.ndarray:
    """Convert an image-space gap into court feet at the place it occurred.

    Both the annotated point and its match on the projected marking are mapped to
    the floor and the distance taken there, rather than scaling pixels by some
    average feet-per-pixel. The scale varies by an order of magnitude across a
    broadcast frame, so an average would be wrong nearly everywhere.
    """
    a = project(h_image_to_court, image_points)
    b = project(h_image_to_court, closest_image)
    if a is None or b is None:
        return np.full(len(image_points), np.nan)
    return np.linalg.norm(a - b, axis=1)


def score_frame(
    annotation: FrameAnnotation,
    h_court_to_image: np.ndarray | None,
    h_image_to_court: np.ndarray | None,
    layout: HalfCourtLayout,
    status: str = "",
    include_held_out: bool = True,
) -> FrameScore:
    """Measure a calibration against one frame-space reference annotation.

    For each annotated marking, the layout's version of that marking is projected
    through the transform and every annotated sample is measured against it. The
    comparison is anchored on identity: paint the annotator called the three-point
    arc is measured against the projected three-point arc, never against whatever
    projected line happens to be nearest. Measuring to the nearest *anything* would
    reward a transform that put the wrong marking in the right place.
    """
    if annotation.coordinate_space != "frame":
        raise ValueError("score in frame space; resolve the annotation first")

    empty = FrameScore(
        frame_id=annotation.frame_id,
        status=status,
        scored=False,
        sample_count=0,
        median_px=float("nan"),
        p95_px=float("nan"),
        max_px=float("nan"),
        median_ft=float("nan"),
        bands=(),
        per_feature={},
        held_out_median_px=float("nan"),
    )
    if h_court_to_image is None:
        return replace(empty, reason="no transform to score")
    if not annotation.features:
        return replace(empty, reason="annotation carries no features")

    polylines = marking_polylines(layout)
    distances: list[np.ndarray] = []
    court_distances: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    held_out: list[np.ndarray] = []
    per_feature: dict = {}

    for item in annotation.features:
        if item.feature not in polylines:
            continue
        if not include_held_out and item.feature in HELD_OUT_FAMILIES:
            continue

        court_points = polylines[item.feature].points
        projected = project(h_court_to_image, court_points)
        if projected is None or not np.all(np.isfinite(projected)):
            per_feature[item.feature.value] = {"scored": False, "reason": "projection failed"}
            continue

        query = np.array([[p.x, p.y] for p in item.points], dtype=np.float64)
        result = nearest_on_polyline(query, projected)

        # Court depth of each match, read off the *layout* polyline at the same
        # vertex-and-offset the match landed on. The projected and court polylines
        # share their vertices, so carrying the index across is exact -- and it
        # needs no transform to be trusted, which matters when the transform is
        # the thing under test.
        depth = result.interpolate(court_points)[:, 0]

        feet = (
            _court_error_ft(query, result.closest, h_image_to_court)
            if h_image_to_court is not None
            else np.full(len(query), np.nan)
        )

        distances.append(result.distances)
        court_distances.append(feet)
        depths.append(depth)
        if item.feature in HELD_OUT_FAMILIES:
            held_out.append(result.distances)

        per_feature[item.feature.value] = {
            "scored": True,
            "held_out": item.feature in HELD_OUT_FAMILIES,
            "sample_count": len(query),
            "median_px": float(np.median(result.distances)),
            "p95_px": _percentile(result.distances, 0.95),
            "median_ft": float(np.nanmedian(feet)) if np.any(np.isfinite(feet)) else float("nan"),
            "mean_signed_px": float(np.mean(result.signed)),
        }

    if not distances:
        return replace(empty, reason="no annotated feature could be projected")

    pooled = np.concatenate(distances)
    pooled_ft = np.concatenate(court_distances)
    pooled_depth = np.concatenate(depths)

    bands = []
    for name, _, _ in DEPTH_BANDS:
        mask = np.array([depth_band(float(d)) == name for d in pooled_depth])
        if not np.any(mask):
            continue
        band_px = pooled[mask]
        band_ft = pooled_ft[mask]
        finite_ft = band_ft[np.isfinite(band_ft)]
        bands.append(
            BandError(
                band=name,
                sample_count=int(mask.sum()),
                median_px=float(np.median(band_px)),
                p95_px=_percentile(band_px, 0.95),
                max_px=float(np.max(band_px)),
                median_ft=float(np.median(finite_ft)) if len(finite_ft) else float("nan"),
                p95_ft=_percentile(finite_ft, 0.95) if len(finite_ft) else float("nan"),
            )
        )

    finite_ft = pooled_ft[np.isfinite(pooled_ft)]
    return FrameScore(
        frame_id=annotation.frame_id,
        status=status,
        scored=True,
        sample_count=int(len(pooled)),
        median_px=float(np.median(pooled)),
        p95_px=_percentile(pooled, 0.95),
        max_px=float(np.max(pooled)),
        median_ft=float(np.median(finite_ft)) if len(finite_ft) else float("nan"),
        bands=tuple(bands),
        per_feature=per_feature,
        held_out_median_px=(
            float(np.median(np.concatenate(held_out))) if held_out else float("nan")
        ),
    )


def score_calibration_entry(
    annotation: FrameAnnotation,
    entry: dict,
    layout: HalfCourtLayout,
    *,
    include_held_out: bool = True,
) -> FrameScore:
    """Score a stored transform only under the exact geometry that produced it."""
    stored_id = entry.get("layout_id")
    stored_hash = entry.get("layout_hash")
    expected_hash = layout.content_hash()
    if stored_id != layout.layout_id:
        raise ValueError(
            f"calibration layout_id {stored_id!r} does not match scoring layout "
            f"{layout.layout_id!r}"
        )
    if not stored_hash:
        raise ValueError("stored calibration has no layout_hash; mixed-layout scoring is unsafe")
    if stored_hash != expected_hash:
        raise ValueError(
            f"calibration layout_hash {stored_hash} does not match "
            f"{layout.layout_id} hash {expected_hash}"
        )
    return score_frame(
        annotation,
        entry.get("h_court_to_image"),
        entry.get("h_image_to_court"),
        layout,
        status=entry.get("status", "UNKNOWN"),
        include_held_out=include_held_out,
    )


def load_calibration_transforms(path: Path | str) -> dict[str, dict]:
    """Read ``calibrations.jsonl`` into per-frame transforms, keyed by frame id.

    Frames the engine refused are kept, with a null transform and their status.
    Dropping them would quietly turn a refusal into an absence, and "declined to
    guess" is a materially different result from "was never tried" -- the design
    plan's decision hierarchy rests on that distinction.
    """
    transforms: dict[str, dict] = {}
    clip = Path(path).parent.parent.name
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            calibration = record.get("calibration", record)
            index = int(calibration["frame_index"])
            matrix = calibration.get("h_court_to_image")
            inverse = calibration.get("h_image_to_court")
            provenance = calibration.get("provenance") or {}
            transforms[f"{clip}_{index:06d}"] = {
                "status": calibration.get("status", "UNKNOWN"),
                "layout_id": calibration.get("layout_id") or provenance.get("layout_id"),
                "layout_hash": provenance.get("layout_hash"),
                "h_court_to_image": np.array(matrix, dtype=np.float64) if matrix else None,
                "h_image_to_court": np.array(inverse, dtype=np.float64) if inverse else None,
            }
    return transforms


def summarize_scores(scores: list[FrameScore]) -> dict:
    """Aggregate across frames, keeping the depth split intact.

    Per-band figures are the median of each frame's band median, weighted by how
    many samples that frame contributed. That is deliberately not a pooled median
    over raw samples: a frame where the annotator traced one marking densely would
    otherwise outvote three frames that traced several sparsely, and coverage is
    not supposed to be a vote. Weighting keeps a thinly-annotated frame from
    counting as much as a well-covered one without letting it dominate either.
    """
    scored = [s for s in scores if s.scored]
    by_band: dict[str, list[float]] = {}
    by_band_ft: dict[str, list[float]] = {}
    for score in scored:
        for band in score.bands:
            by_band.setdefault(band.band, []).extend([band.median_px] * band.sample_count)
            if np.isfinite(band.median_ft):
                by_band_ft.setdefault(band.band, []).extend([band.median_ft] * band.sample_count)

    bands = {}
    for name, _, _ in DEPTH_BANDS:
        values = np.array(by_band.get(name, []))
        feet = np.array(by_band_ft.get(name, []))
        if not len(values):
            continue
        bands[name] = {
            "sample_count": int(len(values)),
            "median_px": float(np.median(values)),
            "p95_px": _percentile(values, 0.95),
            "median_ft": float(np.median(feet)) if len(feet) else float("nan"),
            "p95_ft": _percentile(feet, 0.95) if len(feet) else float("nan"),
        }

    medians = np.array([s.median_px for s in scored])
    held_out = np.array([s.held_out_median_px for s in scored if np.isfinite(s.held_out_median_px)])
    statuses: dict[str, int] = {}
    for score in scores:
        statuses[score.status] = statuses.get(score.status, 0) + 1

    return {
        "schema_version": METRICS_SCHEMA_VERSION,
        "frames": len(scores),
        "frames_scored": len(scored),
        "frames_unscored": len(scores) - len(scored),
        "status_counts": statuses,
        "median_px": float(np.median(medians)) if len(medians) else float("nan"),
        "p95_px": _percentile(medians, 0.95) if len(medians) else float("nan"),
        "held_out_median_px": float(np.median(held_out)) if len(held_out) else float("nan"),
        "bands": bands,
    }


def compare_summaries(baseline: dict, challenger: dict) -> dict:
    """Per-band change between two runs, which is the §9 acceptance shape.

    The acceptance criteria ask for a 25% improvement beyond 19 ft *and* no
    material regression near the baseline. One number cannot express that, so the
    comparison stays per band.
    """
    deltas = {}
    for name in set(baseline.get("bands", {})) | set(challenger.get("bands", {})):
        before = baseline.get("bands", {}).get(name)
        after = challenger.get("bands", {}).get(name)
        if not before or not after:
            continue
        deltas[name] = {
            "median_px_before": before["median_px"],
            "median_px_after": after["median_px"],
            "median_px_change_pct": (
                100.0 * (after["median_px"] - before["median_px"]) / before["median_px"]
                if before["median_px"] > 1e-9
                else float("nan")
            ),
            "p95_px_before": before["p95_px"],
            "p95_px_after": after["p95_px"],
        }
    return {
        "frames_scored_before": baseline.get("frames_scored"),
        "frames_scored_after": challenger.get("frames_scored"),
        "bands": deltas,
    }
