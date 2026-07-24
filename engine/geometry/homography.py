"""Homography from canonical court keypoints to real court feet.

Given the canonical keypoints a ``CourtModel`` emitted for a frame (pixels) and
the court template for the footage (feet), fit the plane-to-plane homography and
project any image point onto the court. Court-plane coordinates are
``(x = across from the west sideline, y = along from the north baseline)`` in
feet, so a downstream top-down minimap has north up and the length running
vertically.

This is a pure OpenCV/NumPy utility — the immediate proof that the plug boundary
yields something metric. Temporal smoothing across frames is consequent.
"""

from __future__ import annotations

import cv2
import numpy as np

from contracts.court_schema import canonical_ids
from contracts.court_template import CourtTemplate

MIN_POINTS = 4
RANSAC_REPROJ_THRESHOLD = 5.0


def solve_homography(
    canonical_kpts: np.ndarray,
    template: CourtTemplate,
    min_points: int = MIN_POINTS,
):
    """Fit image-px -> court-ft homography from visible canonical keypoints.

    Returns ``(H, used_ids)`` where ``H`` is a ``3x3`` matrix and ``used_ids``
    the canonical ids that contributed, or ``(None, [])`` when fewer than
    ``min_points`` keypoints are available (a homography needs at least 4).
    """
    kpts = np.asarray(canonical_kpts, dtype=float)
    ids = canonical_ids()
    src, dst, used = [], [], []
    for i, cid in enumerate(ids):
        x, y, v = kpts[i]
        if v > 0 and cid in template.court_xy_ft:
            along, across = template.court_xy_ft[cid]
            src.append([x, y])
            dst.append([across, along])  # court plane: x=across, y=along
            used.append(cid)

    if len(src) < min_points:
        return None, []

    src_arr = np.asarray(src, dtype=np.float64)
    dst_arr = np.asarray(dst, dtype=np.float64)
    homography, _mask = cv2.findHomography(src_arr, dst_arr, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD)
    if homography is None:
        return None, []
    return homography, used


def project(homography: np.ndarray, points_px) -> np.ndarray:
    """Project image points (N,2 px) onto the court plane, returning (N,2) feet."""
    pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(pts, np.asarray(homography, dtype=np.float64))
    return projected.reshape(-1, 2)
