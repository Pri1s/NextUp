"""Project authoritative layout-owned court markings over a frame."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from contracts.calibration_types import CourtCalibration
from contracts.court_layout import HalfCourtLayout
from contracts.markings import MarkingFamily, MarkingFeature

from .estimator import project

FAMILY_COLORS = {
    MarkingFamily.BOUNDARY: (200, 200, 200),
    MarkingFamily.LANE: (0, 255, 255),
    MarkingFamily.FREE_THROW_CIRCLE: (0, 140, 255),
    MarkingFamily.THREE_POINT: (255, 120, 0),
    MarkingFamily.RESTRICTED_AREA: (0, 255, 120),
    MarkingFamily.MIDCOURT: (255, 255, 255),
}
FEATURE_COLORS = {MarkingFeature.BASELINE: (255, 0, 255)}


def court_markings(layout: HalfCourtLayout) -> dict[MarkingFeature, tuple[tuple[float, float], ...]]:
    """Deterministically sampled views of the layout's analytic primitives."""
    return {feature: layout.sample_marking(feature) for feature in MarkingFeature}


def draw_reprojection(
    frame_bgr: np.ndarray,
    calibration: CourtCalibration,
    layout: HalfCourtLayout,
    evidence: tuple = (),
    thickness: int = 2,
) -> np.ndarray:
    canvas = frame_bgr.copy()
    if calibration.h_court_to_image is None:
        _banner(canvas, calibration.status.value)
        return canvas

    matrix = np.array(calibration.h_court_to_image, dtype=np.float64)
    for feature, polyline in court_markings(layout).items():
        points = project(matrix, np.asarray(polyline, dtype=np.float64))
        if points is None or not np.all(np.isfinite(points)):
            continue
        points = np.clip(points, -1e4, 1e4)
        primitive = layout.marking(feature)
        cv2.polylines(
            canvas,
            [points.astype(np.int32).reshape(-1, 1, 2)],
            primitive.closed,
            FEATURE_COLORS.get(feature, FAMILY_COLORS[feature.family]),
            thickness,
            cv2.LINE_AA,
        )

    for item in evidence:
        cv2.drawMarker(
            canvas, (int(item.x), int(item.y)), (255, 0, 255), cv2.MARKER_CROSS, 18, 2
        )

    quality = calibration.quality
    lines = [
        f"frame {calibration.frame_index}  {calibration.status.value}  "
        f"inliers={quality.inlier_count if quality else '?'}",
        f"reproj_median={_fmt(quality.reprojection_px_median if quality else None)}px  "
        f"holdout={_fmt(quality.holdout_px_median if quality else None)}px",
        f"layout: {layout.layout_id} {layout.content_hash()[:10]}  centerlines: authoritative",
        "end identity: unresolved   slot semantics: provisional",
    ]
    if quality and quality.reasons:
        lines.append("reasons: " + ", ".join(quality.reasons)[:110])
    _header(canvas, lines)
    return canvas


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _header(canvas: np.ndarray, lines: list[str]) -> None:
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], 8 + 20 * len(lines)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)
    for i, line in enumerate(lines):
        cv2.putText(
            canvas, line, (10, 18 + 20 * i),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )


def _banner(canvas: np.ndarray, text: str) -> None:
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
    origin = ((canvas.shape[1] - size[0]) // 2, (canvas.shape[0] + size[1]) // 2)
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)


def write_reprojection(
    path: Path | str,
    frame_bgr: np.ndarray,
    calibration: CourtCalibration,
    layout: HalfCourtLayout,
    evidence: tuple = (),
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), draw_reprojection(frame_bgr, calibration, layout, evidence))
    return path
