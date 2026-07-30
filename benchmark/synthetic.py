"""A court whose true marking positions are known exactly.

Consensus between two annotation passes measures repeatability, and repeatability
is not accuracy. Two annotators who both place their samples two pixels toward the
darker edge of every stripe agree perfectly and are both wrong by two pixels, in
the same direction, everywhere. Nothing in the design plan's §7.4 consensus scheme
can see that -- it compares each pass only against the other -- and the risk is
sharpest when both passes run on the same model, which shares whatever tendency it
has with itself.

Rendering a court through a *planted* homography closes the hole. The centerline
of every stripe is known to floating-point precision, so an annotation can be
scored against truth rather than against a copy of itself, and the mean **signed**
offset per marking is a direct measurement of shared bias.

The rendering deliberately reproduces the things that make real frames hard: paint
whose image width shrinks with distance, a wood floor with grain, a large
translucent centre logo whose curves compete with the real arcs, floor text, and a
broadcast graphic over the court. What it does not reproduce -- players, crowd,
motion blur -- would only make it a worse ground truth, not a better test, because
occlusion is measured on the real frames.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from contracts.annotations import FrameAnnotation, MarkingFeature
from contracts.court_layout import HalfCourtLayout

from calibration.estimator import project

from .geometry import marking_polylines
from .polyline import nearest_on_polyline

SYNTHETIC_SCHEMA_VERSION = "benchmark-synthetic-1.0.0"

#: NBA court lines are two inches wide. The benchmark's whole line-convention
#: decision -- centerline, not edge -- is only meaningful at this scale, so the
#: renderer has to honour it rather than drawing hairlines.
LINE_WIDTH_FT = 2.0 / 12.0

DEFAULT_WIDTH, DEFAULT_HEIGHT = 1280, 720

#: Gaussian blur applied after painting, matching the softness a broadcast frame
#: acquires from optics, scaling and compression. Symmetric, so it widens the
#: intensity ramp across a stripe without moving the stripe's centre.
DEFAULT_SOFTNESS_PX = 0.8

#: Court paint is bright but not blown out; a pure-white line against tan wood is
#: an easier edge than any real floor presents.
DEFAULT_LINE_VALUE = 236


@dataclass(frozen=True, slots=True)
class SyntheticScene:
    """A rendered frame and the exact image-space centerlines behind it."""

    scene_id: str
    image: np.ndarray
    h_court_to_image: np.ndarray
    truth: dict[MarkingFeature, np.ndarray]
    width: int
    height: int

    def visible(self, margin_px: float = 2.0) -> frozenset[MarkingFeature]:
        """Features with a real run of paint inside the frame.

        Coverage is meaningless without this. Several markings project partly or
        entirely off-frame at any plausible camera pose -- the centre circle and
        midcourt line usually do -- and counting those as features the annotator
        failed to find would penalise it for correctly skipping what is not there.
        """
        present = set()
        for feature, points in self.truth.items():
            inside = (
                (points[:, 0] >= -margin_px)
                & (points[:, 0] <= self.width + margin_px)
                & (points[:, 1] >= -margin_px)
                & (points[:, 1] <= self.height + margin_px)
            )
            # Two *consecutive* samples inside, so a curve that merely clips a
            # corner does not count as annotatable.
            if np.any(inside[:-1] & inside[1:]):
                present.add(feature)
        return frozenset(present)


def broadcast_homography(
    layout: HalfCourtLayout,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> np.ndarray:
    """A court-to-image transform resembling an elevated broadcast camera.

    Built from four court corners mapped to a trapezoid rather than from camera
    intrinsics: the benchmark needs a *plausible* perspective with known geometry,
    not a physically derived one, and four corners make the foreshortening
    explicit and easy to reason about.
    """
    court = np.array(
        [
            [0.0, 0.0],
            [layout.half_length, 0.0],
            [layout.half_length, layout.width],
            [0.0, layout.width],
        ],
        dtype=np.float64,
    )
    # Far sideline compressed toward the top, near sideline wide at the bottom:
    # the depth compression that makes far-court error hard to see by eye.
    image = np.array(
        [
            [0.10 * width, 0.34 * height],
            [0.88 * width, 0.26 * height],
            [1.34 * width, 0.96 * height],
            [-0.22 * width, 1.16 * height],
        ],
        dtype=np.float64,
    )
    matrix = cv2.getPerspectiveTransform(court.astype(np.float32), image.astype(np.float32))
    return np.asarray(matrix, dtype=np.float64)


def _wood_floor(width: int, height: int, seed: int) -> np.ndarray:
    """Plank grain and a warm base tone, so crops are not uniform fields."""
    rng = np.random.default_rng(seed)
    base = np.zeros((height, width, 3), dtype=np.float64)
    base[:, :] = (150.0, 190.0, 214.0)  # BGR, pale maple

    grain = rng.normal(0.0, 4.0, size=(height, width))
    grain = cv2.GaussianBlur(grain, (0, 0), sigmaX=9.0, sigmaY=1.2)
    base += grain[:, :, None] * 2.2

    for y in range(0, height, 17):
        base[y : y + 1, :, :] -= rng.uniform(3.0, 9.0)

    return np.clip(base, 0, 255).astype(np.uint8)


def _stripe_polygon(
    centerline: np.ndarray, matrix: np.ndarray, half_width_ft: float
) -> np.ndarray | None:
    """Project both edges of a painted stripe of real width.

    Drawing a constant-pixel-width line would give the far court paint the same
    apparent thickness as the near court, and an annotator's ability to find a
    centre depends directly on how many pixels wide the stripe is. Offsetting in
    *court* space before projecting keeps that relationship honest.
    """
    if len(centerline) < 2:
        return None

    tangents = np.gradient(centerline, axis=0)
    norms = np.linalg.norm(tangents, axis=1)
    norms = np.where(norms > 1e-12, norms, 1.0)
    unit = tangents / norms[:, None]
    normal = np.stack([-unit[:, 1], unit[:, 0]], axis=1)

    left = project(matrix, centerline + normal * half_width_ft)
    right = project(matrix, centerline - normal * half_width_ft)
    if left is None or right is None:
        return None
    if not (np.all(np.isfinite(left)) and np.all(np.isfinite(right))):
        return None

    return np.vstack([left, right[::-1]])


def _draw_distractors(canvas: np.ndarray, matrix: np.ndarray, layout: HalfCourtLayout) -> None:
    """Logo curves, floor text and a score bug -- the real sources of false edges."""
    height, width = canvas.shape[:2]
    centre = np.array([[layout.half_length * 0.52, layout.width / 2.0]], dtype=np.float64)
    logo_centre = project(matrix, centre)

    # A translucent logo with its own arcs, deliberately similar in curvature to
    # the free-throw circle and the three-point arc.
    if logo_centre is not None and np.all(np.isfinite(logo_centre)):
        overlay = canvas.copy()
        cx, cy = int(logo_centre[0, 0]), int(logo_centre[0, 1])
        for radius, colour in ((120, (150, 120, 190)), (78, (170, 140, 205))):
            cv2.ellipse(overlay, (cx, cy), (radius, radius // 2), 0, 0, 360, colour, -1)
        cv2.ellipse(overlay, (cx, cy), (150, 74), 0, 200, 340, (120, 95, 165), 5)
        cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)
        cv2.putText(
            canvas, "ARENA", (cx - 96, cy + 12),
            cv2.FONT_HERSHEY_DUPLEX, 1.5, (135, 110, 175), 3, cv2.LINE_AA,
        )

    # Floor text outside the arc, of the kind painted near a baseline.
    cv2.putText(
        canvas, "SYNTHETIC CENTER", (int(0.05 * width), int(0.90 * height)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (168, 178, 196), 2, cv2.LINE_AA,
    )

    # A broadcast graphic crossing the floor.
    bug = canvas.copy()
    cv2.rectangle(
        bug, (int(0.18 * width), int(0.86 * height)), (int(0.72 * width), int(0.93 * height)),
        (40, 30, 25), -1,
    )
    cv2.addWeighted(bug, 0.85, canvas, 0.15, 0, canvas)
    cv2.putText(
        canvas, "HOME  88   AWAY  84", (int(0.20 * width), int(0.913 * height)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA,
    )


def render_scene(
    layout: HalfCourtLayout,
    scene_id: str = "synthetic_court_a",
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    matrix: np.ndarray | None = None,
    seed: int = 7,
    distractors: bool = True,
    softness_px: float = DEFAULT_SOFTNESS_PX,
    line_value: int = DEFAULT_LINE_VALUE,
) -> SyntheticScene:
    """Render a court and return the exact image-space centerline of every marking."""
    matrix = broadcast_homography(layout, width, height) if matrix is None else matrix
    canvas = _wood_floor(width, height, seed)

    if distractors:
        _draw_distractors(canvas, matrix, layout)

    truth: dict[MarkingFeature, np.ndarray] = {}
    half_width = LINE_WIDTH_FT / 2.0

    for feature, polyline in sorted(marking_polylines(layout).items(), key=lambda kv: kv[0].value):
        court_points = polyline.points
        # Densify so the projected curve is smooth and the stripe edges follow it.
        if len(court_points) < 60:
            steps = np.linspace(0.0, 1.0, 200)
            densified = []
            for start, end in zip(court_points, court_points[1:]):
                densified.append(start + steps[:, None] * (end - start))
            court_points = np.vstack(densified)

        polygon = _stripe_polygon(court_points, matrix, half_width)
        centerline = project(matrix, court_points)
        if polygon is None or centerline is None or not np.all(np.isfinite(centerline)):
            continue

        paint = (line_value, line_value, min(255, line_value + 3))
        cv2.fillPoly(canvas, [np.round(polygon).astype(np.int32)], paint, cv2.LINE_AA)
        truth[feature] = centerline

    # Broadcast video is soft: optics, scaling and compression all blur a stripe
    # edge into a ramp several pixels wide. Rendering perfectly crisp paint would
    # make finding a centre easier here than it can ever be on a real frame, and
    # the probe would then overstate how precise an annotator is. The blur is
    # symmetric, so it moves no centerline and the ground truth stays exact.
    if softness_px > 0:
        canvas = cv2.GaussianBlur(canvas, (0, 0), sigmaX=softness_px, sigmaY=softness_px)

    rng = np.random.default_rng(seed + 1)
    canvas = np.clip(
        canvas.astype(np.float64) + rng.normal(0.0, 2.4, canvas.shape), 0, 255
    ).astype(np.uint8)

    return SyntheticScene(
        scene_id=scene_id,
        image=canvas,
        h_court_to_image=matrix,
        truth=truth,
        width=width,
        height=height,
    )


def write_scene(scene: SyntheticScene, out_dir: Path | str) -> Path:
    """Write the frame as a blind export tree plus a *separate* truth file.

    The truth lives outside the frame tree on purpose. An annotator pointed at the
    frames directory must not be able to read the answer, and the surest way to
    guarantee that is for the answer not to be in there.
    """
    out_dir = Path(out_dir)
    images = out_dir / "frames" / "images"
    images.mkdir(parents=True, exist_ok=True)
    frame_path = images / f"{scene.scene_id}.png"
    if not cv2.imwrite(str(frame_path), scene.image):
        raise OSError(f"failed to write {frame_path}")

    truth_path = out_dir / "truth.json"
    with open(truth_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": SYNTHETIC_SCHEMA_VERSION,
                "scene_id": scene.scene_id,
                "width": scene.width,
                "height": scene.height,
                "line_width_ft": LINE_WIDTH_FT,
                "h_court_to_image": scene.h_court_to_image.tolist(),
                "centerlines": {
                    feature.value: points.tolist() for feature, points in scene.truth.items()
                },
            },
            handle,
        )
        handle.write("\n")
    return frame_path


def load_truth(path: Path | str) -> dict[MarkingFeature, np.ndarray]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    schema = payload.get("schema_version")
    if schema != SYNTHETIC_SCHEMA_VERSION:
        raise ValueError(f"unsupported truth schema {schema!r}")
    return {
        MarkingFeature(name): np.asarray(points, dtype=np.float64)
        for name, points in payload["centerlines"].items()
    }


@dataclass(frozen=True, slots=True)
class FeatureAccuracy:
    """How one annotated feature compares with the truth behind it."""

    feature: MarkingFeature
    sample_count: int
    median_px: float
    p95_px: float
    max_px: float
    mean_signed_px: float

    def as_dict(self) -> dict:
        return {
            "feature": self.feature.value,
            "sample_count": self.sample_count,
            "median_px": self.median_px,
            "p95_px": self.p95_px,
            "max_px": self.max_px,
            "mean_signed_px": self.mean_signed_px,
        }


def _percentile(values: np.ndarray, fraction: float) -> float:
    """Nearest-rank percentile, matching ``calibration.io.records``."""
    if len(values) == 0:
        return float("nan")
    ordered = np.sort(values)
    rank = max(1, min(len(ordered), int(round(fraction * len(ordered) + 0.5))))
    return float(ordered[rank - 1])


def score_against_truth(
    annotation: FrameAnnotation,
    truth: dict[MarkingFeature, np.ndarray],
    present: frozenset[MarkingFeature] | None = None,
) -> tuple[list[FeatureAccuracy], dict]:
    """Measure a frame-space annotation against exact centerlines.

    Returns per-feature accuracy plus an aggregate. ``mean_signed_px`` is the
    number that answers the shared-bias question: near zero means the errors are
    noise, consistently non-zero means every sample of that marking sits to one
    side of the paint and two agreeing passes would both be wrong.
    """
    if annotation.coordinate_space != "frame":
        raise ValueError("score against truth in frame space; resolve the annotation first")

    per_feature: list[FeatureAccuracy] = []
    all_distances: list[np.ndarray] = []
    hallucinated: list[str] = []

    for item in annotation.features:
        reference = truth.get(item.feature)
        if reference is None or len(reference) < 2:
            hallucinated.append(item.feature.value)
            continue
        query = np.array([[p.x, p.y] for p in item.points], dtype=np.float64)
        result = nearest_on_polyline(query, reference)
        per_feature.append(
            FeatureAccuracy(
                feature=item.feature,
                sample_count=len(query),
                median_px=float(np.median(result.distances)),
                p95_px=_percentile(result.distances, 0.95),
                max_px=float(np.max(result.distances)),
                mean_signed_px=float(np.mean(result.signed)),
            )
        )
        all_distances.append(result.distances)

    pooled = np.concatenate(all_distances) if all_distances else np.zeros(0)
    if present is None:
        present = frozenset(f for f, points in truth.items() if len(points) >= 2)
    found = annotation.annotated_features & present
    dispositioned = annotation.annotated_features | frozenset(
        entry.feature for entry in annotation.skipped
    )
    missing_dispositions = sorted(
        (feature.value for feature in present - dispositioned)
    )

    summary = {
        "features_scored": len(per_feature),
        "features_present_in_truth": len(present),
        "coverage": len(found) / len(present) if present else 0.0,
        "missing_dispositions": missing_dispositions,
        "annotated_features_absent_from_truth": sorted(hallucinated),
        "sample_count": int(len(pooled)),
        "median_px": float(np.median(pooled)) if len(pooled) else float("nan"),
        "p95_px": _percentile(pooled, 0.95),
        "max_px": float(np.max(pooled)) if len(pooled) else float("nan"),
        "worst_mean_signed_px": (
            max((abs(f.mean_signed_px) for f in per_feature), default=float("nan"))
        ),
    }
    return per_feature, summary
