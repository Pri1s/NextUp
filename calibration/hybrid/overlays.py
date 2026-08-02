"""Optional visualisation of a hybrid shadow record."""

from __future__ import annotations

import numpy as np

from ..estimator import project
from .control_points import control_court_points, probe_court_points


def render_hybrid_overlay(image, record, layout, *, thickness: int = 2):
    """Render baseline/hybrid court markings and probe displacement arrows."""
    import cv2

    canvas = np.asarray(image).copy()
    transforms = {item.source.value: item for item in record.transforms if item.solved}
    colours = {"keypoint": (255, 160, 0), "hybrid": (0, 220, 80)}
    for source, transform in transforms.items():
        for feature in layout.markings:
            points = np.asarray(layout.marking(feature).sample(81 if feature.kind == "arc" else 31), dtype=np.float64)
            projected = project(np.asarray(transform.h_court_to_image), points)
            if projected is None:
                continue
            polyline = np.round(projected).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [polyline], False, colours.get(source, (255, 255, 255)), thickness)
    baseline = transforms.get("keypoint"); hybrid = transforms.get("hybrid")
    if baseline and hybrid:
        a = project(np.asarray(baseline.h_court_to_image), probe_court_points(layout))
        b = project(np.asarray(hybrid.h_court_to_image), probe_court_points(layout))
        if a is not None and b is not None:
            for left, right in zip(a, b):
                cv2.arrowedLine(canvas, tuple(np.round(left).astype(int)), tuple(np.round(right).astype(int)), (0, 0, 255), 1, tipLength=.2)
    return canvas


render_shadow_overlay = render_hybrid_overlay

