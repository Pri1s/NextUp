"""Reprojected court markings — the check the metrics cannot make.

Every numeric gate in ``quality.py`` measures the fit against *its own evidence*.
That is a test of self-consistency, and with five landmarks against a homography's
eight degrees of freedom — three of them collinear on the baseline — self-consistency
is cheap to achieve while the projection still drifts feet away from the paint.

Projecting the full court markings back onto the frame is the one check whose
reference is the floor itself rather than the detector. It caught a systematic
error that sub-pixel reprojection error and a 2 px held-out score both rated as
healthy, so it is an output, not a debugging aid.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from contracts.calibration_types import CourtCalibration
from contracts.court_layout import HalfCourtLayout

from .estimator import project

#: BGR per marking, so a misaligned feature is identifiable at a glance.
MARKING_COLORS = {
    "baseline": (255, 0, 255),
    "sidelines": (200, 200, 200),
    "lane": (0, 255, 255),
    "free_throw_circle": (0, 140, 255),
    "three_point": (255, 120, 0),
    "restricted_area": (0, 255, 120),
    "midcourt": (255, 255, 255),
}


def court_markings(
    layout: HalfCourtLayout,
    three_point_radius: float = 23.75,
    restricted_radius: float = 4.0,
) -> dict[str, list[tuple[float, float]]]:
    """Court lines as polylines in half-court feet."""
    length, width = layout.half_length, layout.width
    centre_y = width / 2.0
    lane_far = centre_y - layout.lane_width / 2.0
    lane_near = centre_y + layout.lane_width / 2.0
    free_throw = layout.free_throw_distance
    basket_x = layout.basket_from_baseline
    inset = layout.three_point_corner_inset
    circle_r = layout.free_throw_circle_radius

    markings: dict[str, list[tuple[float, float]]] = {
        "baseline": [(0.0, 0.0), (0.0, width)],
        "sidelines": [(0.0, 0.0), (length, 0.0)],
        "midcourt": [(length, 0.0), (length, width)],
        "lane": [
            (0.0, lane_far),
            (free_throw, lane_far),
            (free_throw, lane_near),
            (0.0, lane_near),
        ],
        "free_throw_circle": [
            (free_throw + circle_r * math.cos(t), centre_y + circle_r * math.sin(t))
            for t in np.linspace(0.0, 2 * math.pi, 72)
        ],
        "restricted_area": [
            (basket_x + restricted_radius * math.cos(t), centre_y + restricted_radius * math.sin(t))
            for t in np.linspace(-math.pi / 2, math.pi / 2, 40)
        ],
    }

    # Three-point line: straight corner segments up to where the arc meets them.
    dy = centre_y - inset
    break_x = basket_x + math.sqrt(max(three_point_radius**2 - dy**2, 0.0))
    start = math.atan2(inset - centre_y, break_x - basket_x)
    end = math.atan2(width - inset - centre_y, break_x - basket_x)
    markings["three_point"] = (
        [(0.0, inset), (break_x, inset)]
        + [
            (basket_x + three_point_radius * math.cos(t), centre_y + three_point_radius * math.sin(t))
            for t in np.linspace(start, end, 80)
        ]
        + [(break_x, width - inset), (0.0, width - inset)]
    )
    return markings


def draw_reprojection(
    frame_bgr: np.ndarray,
    calibration: CourtCalibration,
    layout: HalfCourtLayout,
    evidence: tuple = (),
    thickness: int = 2,
) -> np.ndarray:
    """Draw the court through ``H_court_to_image`` over the frame."""
    canvas = frame_bgr.copy()
    if calibration.h_court_to_image is None:
        _banner(canvas, calibration.status.value)
        return canvas

    matrix = np.array(calibration.h_court_to_image, dtype=np.float64)
    for name, polyline in court_markings(layout).items():
        points = project(matrix, np.array(polyline, dtype=np.float64))
        if points is None or not np.all(np.isfinite(points)):
            continue
        # Clip to a generous margin: lines heading off-frame are expected, but
        # unbounded coordinates make cv2 unhappy.
        points = np.clip(points, -1e4, 1e4)
        closed = name == "lane"
        cv2.polylines(
            canvas,
            [points.astype(np.int32).reshape(-1, 1, 2)],
            closed,
            MARKING_COLORS.get(name, (255, 255, 255)),
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
