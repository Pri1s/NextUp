"""Auditable raw-evidence and association overlay renderers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from contracts.court_layout import HalfCourtLayout
from contracts.hybrid_types import AssignmentStatus, ShadowMarkingFrame


def render_shadow_overlay(
    frame: np.ndarray,
    record: ShadowMarkingFrame,
    layout: HalfCourtLayout,
    h_court_to_image,
    *,
    roi_mask: np.ndarray | None = None,
    exclusion_mask: np.ndarray | None = None,
    association: bool = False,
) -> np.ndarray:
    """Render a review image without changing the serialized evidence."""
    import cv2

    canvas = np.asarray(frame).copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    if roi_mask is not None:
        contours, _ = cv2.findContours(np.asarray(roi_mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (255, 180, 0), 2)
    if exclusion_mask is not None and np.any(exclusion_mask):
        red = np.zeros_like(canvas)
        red[:, :, 2] = 255
        canvas[np.asarray(exclusion_mask, dtype=bool)] = (
            0.45 * canvas[np.asarray(exclusion_mask, dtype=bool)] + 0.55 * red[np.asarray(exclusion_mask, dtype=bool)]
        ).astype(np.uint8)

    assignment_by_id = {item.evidence_id: item for item in record.assignments}
    colors = {
        AssignmentStatus.ACCEPTED: (0, 220, 0),
        AssignmentStatus.AMBIGUOUS: (0, 220, 255),
        AssignmentStatus.REJECTED: (0, 0, 255),
    }
    for evidence in record.evidence:
        assignment = assignment_by_id.get(evidence.evidence_id)
        color = colors[assignment.status] if assignment else (255, 0, 255)
        points = np.array([[round(sample.x), round(sample.y)] for sample in evidence.samples], dtype=np.int32)
        if len(points) > 1:
            cv2.polylines(canvas, [points.reshape(-1, 1, 2)], False, color, 2, cv2.LINE_AA)
        for point in points[:: max(1, len(points) // 12)]:
            cv2.circle(canvas, tuple(point), 2, color, -1, cv2.LINE_AA)

    if association and h_court_to_image is not None:
        from calibration.estimator import project

        selected = {item.evidence_id: item.selected_feature for item in record.assignments if item.status is AssignmentStatus.ACCEPTED}
        accepted_features = {feature for feature in selected.values() if feature is not None}
        for feature in sorted(accepted_features, key=lambda value: value.value):
            points = np.asarray(layout.sample_marking(feature), dtype=np.float64)
            projected = project(np.asarray(h_court_to_image, dtype=np.float64), points)
            if projected is None or not np.all(np.isfinite(projected)):
                continue
            color = (255, 120, 0) if feature.value == "three_point_arc" else (0, 255, 255)
            cv2.polylines(canvas, [np.round(projected).astype(np.int32).reshape(-1, 1, 2)], False, color, 2, cv2.LINE_AA)

    lines = [
        f"{record.frame_id}  {record.status.value}  eligible={record.eligible}",
        f"evidence={len(record.evidence)}  accepted={sum(1 for item in record.assignments if item.status is AssignmentStatus.ACCEPTED)}",
        f"ROI={record.roi_pixels} excluded={record.excluded_pixels}  config={record.config_hash[:10]}",
    ]
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], 8 + len(lines) * 20), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)
    for index, line in enumerate(lines):
        cv2.putText(canvas, line, (10, 18 + index * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def write_shadow_overlays(
    root: Path | str,
    frame: np.ndarray,
    record: ShadowMarkingFrame,
    layout: HalfCourtLayout,
    h_court_to_image,
    *,
    roi_mask: np.ndarray | None = None,
    exclusion_mask: np.ndarray | None = None,
) -> tuple[Path, Path]:
    import cv2

    root = Path(root)
    raw = root / "raw-evidence" / f"{record.frame_id}.png"
    association = root / "association" / f"{record.frame_id}.png"
    raw.parent.mkdir(parents=True, exist_ok=True)
    association.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(raw), render_shadow_overlay(frame, record, layout, h_court_to_image, roi_mask=roi_mask, exclusion_mask=exclusion_mask))
    cv2.imwrite(str(association), render_shadow_overlay(frame, record, layout, h_court_to_image, roi_mask=roi_mask, exclusion_mask=exclusion_mask, association=True))
    return raw, association
