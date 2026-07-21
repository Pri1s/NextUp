"""Inspectable, classical-CV basketball-court homography pipeline.

This module intentionally contains no learned inference and no training-label export.
It detects image primitives, tries unlabeled line correspondences against a JSON court
layout, and records enough evidence for a reviewer to reject weak solutions.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from homography_proposals import generate_guided_proposals
from hud_detection import HudContext, detect_hud, primitive_hud_overlap
from segment_merge import merge_collinear_segments, observed_segment_length

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_TEMPLATE = PROJECT_DIR / "court_templates" / "nba.json"


def load_template(path: str | Path = DEFAULT_TEMPLATE) -> dict[str, Any]:
    """Load a replaceable court template; only JSON geometry is assumed."""
    with open(path, encoding="utf-8") as handle:
        template = json.load(handle)
    for required in ("dimensions", "keypoints", "lines", "arcs"):
        if required not in template:
            raise ValueError(f"Court template {path} is missing {required!r}")
    return template


def _line_equation(p1: Any, p2: Any) -> np.ndarray:
    line = np.cross(np.array([*p1, 1.0]), np.array([*p2, 1.0]))
    norm = np.hypot(line[0], line[1])
    return line / norm if norm else line


def _intersection(a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    point = np.cross(a, b)
    if abs(point[2]) < 1e-8:
        return None
    return point[:2] / point[2]


def _angle_difference(first: float, second: float) -> float:
    return abs((first - second + math.pi / 2) % math.pi - math.pi / 2)


def _segment_angle(segment: dict[str, Any]) -> float:
    p1, p2 = segment["geometry"]["p1"], segment["geometry"]["p2"]
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0]) % math.pi


def _point_to_segment_distance(point: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    direction = p2 - p1
    length_sq = float(direction @ direction)
    if length_sq == 0:
        return float(np.linalg.norm(point - p1))
    t = np.clip(float((point - p1) @ direction) / length_sq, 0, 1)
    return float(np.linalg.norm(point - (p1 + t * direction)))


def _segment_length(segment: dict[str, Any]) -> float:
    p1, p2 = segment["geometry"]["p1"], segment["geometry"]["p2"]
    return math.dist(p1, p2)


def _smoothstep(value: float, start: float, end: float) -> float:
    """A bounded linear ramp used for deliberately soft image priors."""
    if end <= start:
        return float(value >= end)
    return float(np.clip((value - start) / (end - start), 0.0, 1.0))


def _line_brightness_confidence(
    hsv: np.ndarray, p1: tuple[int, int] | tuple[float, float], p2: tuple[int, int] | tuple[float, float]
) -> float:
    height, width = hsv.shape[:2]
    samples = max(12, int(math.dist(p1, p2) / 12))
    xs = np.clip(np.rint(np.linspace(p1[0], p2[0], samples)).astype(int), 0, width - 1)
    ys = np.clip(np.rint(np.linspace(p1[1], p2[1], samples)).astype(int), 0, height - 1)
    values = hsv[ys, xs, 2].astype(float)
    return _smoothstep(float(np.median(values)), 38, 125)


def _floor_confidence(brightness: float, hud_overlap: float = 0.0) -> float:
    """Soft appearance confidence with no image-height or colour assumption."""
    return round(float(np.clip(brightness * (1.0 - hud_overlap), 0.0, 1.0)), 4)


def _court_floor_confidence(
    hsv: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    hud_overlap: float = 0.0,
) -> float:
    """Return brightness/HUD floor confidence without assuming lower is floor."""
    return _floor_confidence(_line_brightness_confidence(hsv, p1, p2), hud_overlap)


def _arc_floor_confidence(hsv: np.ndarray, center: tuple[float, float], radius: float, hud_overlap: float = 0.0) -> float:
    """Use a diameter sample to apply the same soft floor prior to arcs."""
    height, width = hsv.shape[:2]
    cx, cy = center
    p1 = (int(np.clip(round(cx - radius), 0, width - 1)), int(np.clip(round(cy), 0, height - 1)))
    p2 = (int(np.clip(round(cx + radius), 0, width - 1)), int(np.clip(round(cy), 0, height - 1)))
    return _court_floor_confidence(hsv, p1, p2, hud_overlap)


def _apply_floor_hud_evidence(
    primitive: dict[str, Any], hsv: np.ndarray, hud: Any
) -> dict[str, Any]:
    """Recompute soft appearance evidence on final line/arc geometry."""
    geometry = primitive["geometry"]
    overlap = primitive_hud_overlap(primitive, hud)
    hud_overlap = float(overlap["hud_overlap"])
    if primitive["type"] == "line_segment":
        brightness = _line_brightness_confidence(hsv, geometry["p1"], geometry["p2"])
    else:
        center = geometry["center"]
        if primitive["type"] == "circle":
            radius = float(geometry["radius_px"])
        else:
            radius = float(max(geometry["axes_px"]) / 2)
        height, width = hsv.shape[:2]
        p1 = (int(np.clip(round(center[0] - radius), 0, width - 1)), int(np.clip(round(center[1]), 0, height - 1)))
        p2 = (int(np.clip(round(center[0] + radius), 0, width - 1)), int(np.clip(round(center[1]), 0, height - 1)))
        brightness = _line_brightness_confidence(hsv, p1, p2)
    floor_confidence = _floor_confidence(brightness, hud_overlap)
    primitive.setdefault("evidence", {}).update({
        "floor_brightness_confidence": round(float(brightness), 4),
        "hud_overlap_ratio": round(hud_overlap, 4),
        "hud_region_ids": overlap["hud_region_ids"],
        "floor_roi_confidence": floor_confidence,
    })
    raw_strength = float(primitive["evidence"].get("raw_strength", primitive.get("strength", 0.0)))
    # HUD evidence remains soft: even a fully covered primitive is retained,
    # but it cannot occupy the seed pool merely because a broadcast graphic is
    # high-contrast.  Court brightness and HUD overlap are resampled on the
    # final merged span, so no fragment can donate an inherited confidence.
    appearance_multiplier = .12 + .88 * floor_confidence
    primitive["evidence"]["appearance_strength_multiplier"] = round(appearance_multiplier, 4)
    primitive["strength"] = round(raw_strength * appearance_multiplier, 4)
    return primitive


def _clip_segment_to_image(p1: np.ndarray, p2: np.ndarray, width: int, height: int, margin: float = 2.0) -> tuple[np.ndarray, np.ndarray] | None:
    """Clip a projected finite template marking to the visible image rectangle."""
    direction = p2 - p1
    lower, upper = 0.0, 1.0
    for point, delta, minimum, maximum in (
        (p1[0], direction[0], -margin, width - 1 + margin),
        (p1[1], direction[1], -margin, height - 1 + margin),
    ):
        if abs(delta) < 1e-8:
            if point < minimum or point > maximum:
                return None
            continue
        first, second = (minimum - point) / delta, (maximum - point) / delta
        if first > second:
            first, second = second, first
        lower, upper = max(lower, first), min(upper, second)
        if lower > upper:
            return None
    return p1 + lower * direction, p1 + upper * direction


def _line_strength(gray: np.ndarray, p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """Line-vs-floor contrast measured in a small normal-direction strip."""
    length = max(1.0, math.dist(p1, p2))
    count = max(12, int(length / 8))
    x = np.linspace(p1[0], p2[0], count)
    y = np.linspace(p1[1], p2[1], count)
    normal = np.array([-(p2[1] - p1[1]) / length, (p2[0] - p1[0]) / length])
    center, sides = [], []
    height, width = gray.shape[:2]
    for px, py in zip(x, y):
        ix, iy = int(round(px)), int(round(py))
        if 2 <= ix < width - 2 and 2 <= iy < height - 2:
            center.append(float(gray[iy, ix]))
            for sign in (-2, 2):
                sx, sy = int(round(ix + sign * normal[0])), int(round(iy + sign * normal[1]))
                sides.append(float(gray[sy, sx]))
    if not center or not sides:
        return 0.0
    return min(1.0, abs(float(np.mean(center)) - float(np.mean(sides))) / 45.0)


def _dedupe_segments(items: list[dict[str, Any]], angle_tol: float = math.radians(3), distance_tol: float = 14) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda value: value["strength"], reverse=True):
        midpoint = np.mean(np.array([item["geometry"]["p1"], item["geometry"]["p2"]]), axis=0)
        angle = _segment_angle(item)
        duplicate = False
        for present in kept:
            present_midpoint = np.mean(np.array([present["geometry"]["p1"], present["geometry"]["p2"]]), axis=0)
            if _angle_difference(angle, _segment_angle(present)) < angle_tol and np.linalg.norm(midpoint - present_midpoint) < distance_tol:
                duplicate = True
                break
        if not duplicate:
            kept.append(item)
    return kept


def _merge_collinear_segments(
    items: list[dict[str, Any]],
    angle_tol: float = math.radians(2.5),
    distance_tol: float = 8,
    max_gap: float = 72,
    *,
    image_width: int | None = None,
    image_height: int | None = None,
) -> list[dict[str, Any]]:
    """Compatibility wrapper around provenance-preserving two-stage merging."""
    if image_width is None:
        image_width = max(
            (int(math.ceil(max(point[0] for point in (item["geometry"]["p1"], item["geometry"]["p2"])))) + 1 for item in items),
            default=1,
        )
    if image_height is None:
        image_height = max(
            (int(math.ceil(max(point[1] for point in (item["geometry"]["p1"], item["geometry"]["p2"])))) + 1 for item in items),
            default=1,
        )
    merged, _ = merge_collinear_segments(
        items,
        image_width=max(1, image_width),
        image_height=max(1, image_height),
        base_max_gap_px=max_gap,
        angle_tolerance_deg=math.degrees(angle_tol),
        lateral_offset_px=distance_tol,
    )
    return merged


def detect_primitives(
    image: np.ndarray,
    *,
    min_line_length: int = 45,
    max_lines: int = 80,
    hud_context: HudContext | None = None,
) -> dict[str, Any]:
    """Detect unlabelled line, arc/circle, and line-intersection primitives.

    CLAHE plus a saturation-aware floor/paint contrast image helps white/yellow
    court markings survive on uneven gym floors. Returned strengths are image
    evidence only, not court-identity probabilities.
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot detect primitives in an empty image")
    height, width = image.shape[:2]
    hud = detect_hud(image, hud_context)
    lab_l = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    enhanced = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(lab_l)
    # Paint is commonly less saturated than floor; retain luminance edges too.
    contrast = cv2.addWeighted(enhanced, 0.78, 255 - saturation, 0.22, 0)
    blurred = cv2.GaussianBlur(contrast, (5, 5), 0)
    edges = cv2.Canny(blurred, 45, 130, apertureSize=3, L2gradient=True)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    raw = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=42, minLineLength=min_line_length, maxLineGap=13)
    lines: list[dict[str, Any]] = []
    if raw is not None:
        # OpenCV versions return either (N, 1, 4) or (N, 4).
        for raw_index, row in enumerate(np.asarray(raw).reshape(-1, 4)):
            x1, y1, x2, y2 = row
            length = math.hypot(float(x2 - x1), float(y2 - y1))
            support = _line_strength(enhanced, (int(x1), int(y1)), (int(x2), int(y2)))
            raw_strength = min(1.0, 0.55 * min(1.0, length / max(width, height) * 3.0) + 0.45 * support)
            lines.append({
                "id": f"raw_line_{raw_index:04d}",
                "type": "line_segment",
                "geometry": {"p1": [float(x1), float(y1)], "p2": [float(x2), float(y2)], "length_px": round(length, 2), "angle_rad": round(math.atan2(y2-y1, x2-x1), 5)},
                "strength": round(raw_strength, 4),
                "evidence": {"paint_floor_contrast": round(support, 4), "raw_strength": round(raw_strength, 4)},
            })
    lines, line_merge_diagnostics = merge_collinear_segments(
        lines, image_width=width, image_height=height
    )
    lines = [_apply_floor_hud_evidence(line, hsv, hud) for line in lines]
    lines = _dedupe_segments(lines)
    line_merge_diagnostics["deduplicated_count"] = len(lines)
    lines = lines[:max_lines]
    line_merge_diagnostics["emitted_count"] = len(lines)
    for index, line in enumerate(lines):
        line["id"] = f"line_{index:03d}"

    arcs: list[dict[str, Any]] = []
    # Hough circles catches near-frontal circles; contour ellipses catch their
    # projective appearance in broadcast/handheld views.
    min_radius = max(12, int(min(width, height) * 0.018))
    circles = cv2.HoughCircles(cv2.medianBlur(blurred, 7), cv2.HOUGH_GRADIENT, 1.25, minDist=min_radius * 2, param1=100, param2=24, minRadius=min_radius, maxRadius=max(min_radius + 1, int(min(width, height) * .42)))
    if circles is not None:
        for x, y, radius in circles[0, :16]:
            raw_strength = min(1.0, float(radius) / min(width, height) * 4)
            arc = {"type": "circle", "geometry": {"center": [round(float(x), 2), round(float(y), 2)], "radius_px": round(float(radius), 2)}, "strength": round(raw_strength, 4), "evidence": {"detector": "hough_circle", "raw_strength": round(raw_strength, 4)}}
            arcs.append(_apply_floor_hud_evidence(arc, hsv, hud))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        if len(contour) < 24:
            continue
        (cx, cy), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
        if min(axis_a, axis_b) < min_radius * 1.2 or max(axis_a, axis_b) / min(axis_a, axis_b) > 10:
            continue
        perimeter = cv2.arcLength(contour, False)
        coverage = min(1.0, perimeter / (math.pi * (axis_a + axis_b) / 2))
        if coverage < .18:
            continue
        arc = {"type": "arc", "geometry": {"center": [round(float(cx), 2), round(float(cy), 2)], "axes_px": [round(float(axis_a), 2), round(float(axis_b), 2)], "rotation_deg": round(float(angle), 2), "coverage": round(coverage, 3)}, "strength": round(coverage, 4), "evidence": {"detector": "contour_ellipse", "raw_strength": round(coverage, 4)}}
        arcs.append(_apply_floor_hud_evidence(arc, hsv, hud))
    arcs.sort(key=lambda item: item["strength"], reverse=True)
    arcs = arcs[:30]
    for index, arc in enumerate(arcs):
        arc["id"] = f"{arc['type']}_{index:03d}"

    intersections: list[dict[str, Any]] = []
    for first, second in itertools.combinations(lines, 2):
        line_a = _line_equation(first["geometry"]["p1"], first["geometry"]["p2"])
        line_b = _line_equation(second["geometry"]["p1"], second["geometry"]["p2"])
        point = _intersection(line_a, line_b)
        if point is None or not (-.15 * width <= point[0] <= 1.15 * width and -.15 * height <= point[1] <= 1.15 * height):
            continue
        if _angle_difference(_segment_angle(first), _segment_angle(second)) < math.radians(12):
            continue
        # An extended-line crossing is less trustworthy if far from both segments.
        extension_distances = [
            _point_to_segment_distance(point, np.array(line["geometry"]["p1"]), np.array(line["geometry"]["p2"]))
            for line in (first, second)
        ]
        endpoint_distances = [
            min(
                float(np.linalg.norm(point - np.asarray(line["geometry"]["p1"]))),
                float(np.linalg.norm(point - np.asarray(line["geometry"]["p2"]))),
            )
            for line in (first, second)
        ]
        distance = sum(extension_distances)
        strength = first["strength"] * second["strength"] * math.exp(-distance / 24)
        if strength >= .04:
            intersections.append({
                "type": "intersection",
                "geometry": {
                    "point": [round(float(point[0]), 2), round(float(point[1]), 2)],
                    "line_ids": [first["id"], second["id"]],
                },
                "strength": round(float(strength), 4),
                "evidence": {
                    "segment_extension_distances_px": [round(value, 2) for value in extension_distances],
                    "endpoint_distances_px": [round(value, 2) for value in endpoint_distances],
                },
            })
    intersections.sort(key=lambda item: item["strength"], reverse=True)
    intersections = intersections[:150]
    for index, point in enumerate(intersections):
        point["id"] = f"intersection_{index:03d}"
    return {"image_size": {"width": width, "height": height}, "preprocessing": {"edge_detector": "CLAHE luminance + inverse saturation Canny", "edge_pixels": int(np.count_nonzero(edges)), "hud_detection": hud.diagnostics, "line_merge": line_merge_diagnostics}, "primitives": lines + arcs + intersections}


def _project(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(np.asarray(points, dtype=np.float32).reshape(-1, 1, 2), homography).reshape(-1, 2)


def _homography_from_line_matches(template_lines: list[dict[str, Any]], detected_lines: list[dict[str, Any]]) -> np.ndarray | None:
    source_lines = [_line_equation(line["p1"], line["p2"]) for line in template_lines]
    target_lines = [_line_equation(line["geometry"]["p1"], line["geometry"]["p2"]) for line in detected_lines]
    source, target = [], []
    for first, second in itertools.combinations(range(4), 2):
        a, b = _intersection(source_lines[first], source_lines[second]), _intersection(target_lines[first], target_lines[second])
        if a is not None and b is not None and np.isfinite(a).all() and np.isfinite(b).all():
            source.append(a)
            target.append(b)
    if len(source) < 4:
        return None
    homography, _ = cv2.findHomography(np.asarray(source), np.asarray(target), method=0)
    if homography is None or not np.isfinite(homography).all() or abs(np.linalg.det(homography)) < 1e-10:
        return None
    return homography / homography[2, 2]


def _projected_line(template_line: dict[str, Any], homography: np.ndarray) -> np.ndarray | None:
    try:
        result = np.linalg.inv(homography).T @ _line_equation(template_line["p1"], template_line["p2"])
    except np.linalg.LinAlgError:
        return None
    norm = np.hypot(result[0], result[1])
    return result / norm if norm else None


def _arc_residual(template_arc: dict[str, Any], arc: dict[str, Any], homography: np.ndarray) -> float:
    geometry = arc["geometry"]
    if arc["type"] == "circle":
        center = np.array(geometry["center"])
        axes = np.array([geometry["radius_px"], geometry["radius_px"]])
        rotation = 0.0
    else:
        center = np.array(geometry["center"])
        axes = np.array(geometry["axes_px"]) / 2
        rotation = math.radians(geometry["rotation_deg"])
    theta = np.linspace(0, 2 * math.pi, 20, endpoint=False)
    source = np.column_stack((template_arc["center"][0] + template_arc["radius"] * np.cos(theta), template_arc["center"][1] + template_arc["radius"] * np.sin(theta)))
    projected = _project(source, homography)
    cos_a, sin_a = math.cos(rotation), math.sin(rotation)
    shifted = projected - center
    local_x = cos_a * shifted[:, 0] + sin_a * shifted[:, 1]
    local_y = -sin_a * shifted[:, 0] + cos_a * shifted[:, 1]
    radial = np.sqrt((local_x / max(axes[0], 1)) ** 2 + (local_y / max(axes[1], 1)) ** 2)
    return float(np.median(abs(radial - 1)))


def _visible_template_segment(template_line: dict[str, Any], homography: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray] | None:
    projected = _project(np.asarray([template_line["p1"], template_line["p2"]], dtype=np.float32), homography)
    if not np.isfinite(projected).all():
        return None
    clipped = _clip_segment_to_image(projected[0].astype(float), projected[1].astype(float), width, height)
    if clipped is None or np.linalg.norm(clipped[1] - clipped[0]) < 24:
        return None
    return clipped


def _segment_agreement(template_line: dict[str, Any], detected_line: dict[str, Any], homography: np.ndarray, width: int, height: int, line_distance_px: float) -> dict[str, Any] | None:
    """Measure agreement of two *finite* segments, not their infinite lines.

    The template segment is projected and clipped to the image.  A detected Hough
    segment must cover that visible span with both endpoints reasonably placed;
    a short fragment merely sitting near the projected midpoint is not evidence.
    """
    expected = _visible_template_segment(template_line, homography, width, height)
    if expected is None:
        return None
    start, end = expected
    direction = end - start
    expected_length = float(np.linalg.norm(direction))
    unit = direction / expected_length
    normal = np.array([-unit[1], unit[0]])
    detected = np.asarray([detected_line["geometry"]["p1"], detected_line["geometry"]["p2"]], dtype=float)
    offsets = detected - start
    longitudinal = offsets @ unit
    perpendicular = np.abs(offsets @ normal)
    detected_low, detected_high = sorted(float(value) for value in longitudinal)
    overlap = max(0.0, min(expected_length, detected_high) - max(0.0, detected_low)) / expected_length
    detected_length = float(np.linalg.norm(detected[1] - detected[0]))
    length_ratio = detected_length / expected_length
    start_gap = abs(detected_low - 0.0) / expected_length
    end_gap = abs(detected_high - expected_length) / expected_length
    angle_error = _angle_difference(math.atan2(direction[1], direction[0]) % math.pi, _segment_angle(detected_line))
    max_perpendicular = float(np.max(perpendicular))
    mean_perpendicular = float(np.mean(perpendicular))
    # These values are intentionally relative to the visible projected marking,
    # which works for both close and far court geometry without admitting a
    # small scoreboard/text fragment as a full line.
    if (max_perpendicular > line_distance_px or angle_error > math.radians(8)
            or overlap < .58 or length_ratio < .52 or length_ratio > 1.65
            or start_gap > .38 or end_gap > .38):
        return None
    endpoint_alignment = max(0.0, 1.0 - (start_gap + end_gap) / .76)
    quality = (detected_line["strength"] * overlap * (.55 + .45 * endpoint_alignment)
               * math.exp(-mean_perpendicular / max(1.0, line_distance_px))
               * math.exp(-angle_error / math.radians(8)))
    return {
        "quality": float(quality), "distance_px": mean_perpendicular,
        "max_endpoint_distance_px": max_perpendicular,
        "angle_error_deg": math.degrees(angle_error), "overlap_ratio": overlap,
        "length_ratio": length_ratio, "start_gap_ratio": start_gap,
        "end_gap_ratio": end_gap,
    }


def _unique_line_matches(homography: np.ndarray, template: dict[str, Any], lines: list[dict[str, Any]], seed_pairs: dict[str, str], width: int, height: int, line_distance_px: float) -> list[dict[str, Any]]:
    """Return a one-to-one assignment of template and detected line segments."""
    candidates: list[dict[str, Any]] = []
    for template_line in template["lines"]:
        for line in lines:
            agreement = _segment_agreement(template_line, line, homography, width, height, line_distance_px)
            if agreement is not None:
                candidates.append({
                    "template_feature": template_line["id"], "detected_primitive": line["id"],
                    "seed": seed_pairs.get(template_line["id"]) == line["id"], **agreement,
                })
    # A single physical Hough fragment cannot support multiple court markings.
    # Greedy selection is sufficient here because all matches are independently
    # quality-ranked and the downstream structural check requires connectivity.
    selected, used_template, used_detected = [], set(), set()
    for candidate in sorted(candidates, key=lambda value: value["quality"], reverse=True):
        if candidate["template_feature"] in used_template or candidate["detected_primitive"] in used_detected:
            continue
        selected.append(candidate)
        used_template.add(candidate["template_feature"])
        used_detected.add(candidate["detected_primitive"])
    return selected


def _structural_evidence(homography: np.ndarray, template: dict[str, Any], matches: list[dict[str, Any]], lines_by_id: dict[str, dict[str, Any]], width: int, height: int, line_distance_px: float) -> dict[str, Any]:
    """Check that distinct matches form court geometry rather than loose lines."""
    template_by_id = {line["id"]: line for line in template["lines"]}
    connected_pairs = parallel_pairs = right_angle_pairs = inverse_parallel_pairs = 0
    inverse = np.linalg.inv(homography)
    for first, second in itertools.combinations(matches, 2):
        first_template, second_template = template_by_id[first["template_feature"]], template_by_id[second["template_feature"]]
        first_angle = math.atan2(first_template["p2"][1] - first_template["p1"][1], first_template["p2"][0] - first_template["p1"][0]) % math.pi
        second_angle = math.atan2(second_template["p2"][1] - second_template["p1"][1], second_template["p2"][0] - second_template["p1"][0]) % math.pi
        angle_delta = _angle_difference(first_angle, second_angle)
        if angle_delta < math.radians(2):
            parallel_pairs += 1
            transformed = []
            for match in (first, second):
                image_line = lines_by_id[match["detected_primitive"]]
                court_line = _project(np.asarray([image_line["geometry"]["p1"], image_line["geometry"]["p2"]], dtype=np.float32), inverse)
                transformed.append(math.atan2(court_line[1, 1] - court_line[0, 1], court_line[1, 0] - court_line[0, 0]) % math.pi)
            if _angle_difference(transformed[0], transformed[1]) < math.radians(7):
                inverse_parallel_pairs += 1
        elif abs(angle_delta - math.pi / 2) < math.radians(2):
            right_angle_pairs += 1
        # Connected template markings must meet near the projected common end.
        shared = next((point for point in (first_template["p1"], first_template["p2"])
                       if any(np.linalg.norm(np.asarray(point) - np.asarray(other)) < 1e-5 for other in (second_template["p1"], second_template["p2"]))), None)
        if shared is not None:
            projected = _project(np.asarray([shared], dtype=np.float32), homography)[0]
            if 0 <= projected[0] < width and 0 <= projected[1] < height:
                distances = []
                for match in (first, second):
                    segment = lines_by_id[match["detected_primitive"]]["geometry"]
                    distances.append(_point_to_segment_distance(projected, np.asarray(segment["p1"]), np.asarray(segment["p2"])))
                if max(distances) <= max(2 * line_distance_px, 24):
                    connected_pairs += 1
    circle_matches = []
    # A free-throw circle is optional evidence, but when present it must be a
    # unique, low-residual agreement at the homography's projected position and
    # scale.  It cannot be reused for several template circles.
    return {
        "unique_line_matches": len(matches), "connected_pairs": connected_pairs,
        "parallel_pairs": parallel_pairs, "inverse_parallel_pairs": inverse_parallel_pairs,
        "right_angle_pairs": right_angle_pairs, "free_throw_circle_matches": circle_matches,
    }


def _score_homography(homography: np.ndarray, template: dict[str, Any], lines: list[dict[str, Any]], arcs: list[dict[str, Any]], seed_pairs: dict[str, str], width: int, height: int, line_distance_px: float) -> dict[str, Any]:
    matches = _unique_line_matches(homography, template, lines, seed_pairs, width, height, line_distance_px)
    lines_by_id = {line["id"]: line for line in lines}
    structure = _structural_evidence(homography, template, matches, lines_by_id, width, height, line_distance_px)
    # Circle evidence is one-to-one too.  A detected circle can no longer boost
    # every free-throw/three-point template arc at once.
    arc_candidates = []
    for template_arc in template["arcs"]:
        for arc in arcs:
            residual = _arc_residual(template_arc, arc, homography)
            if residual < .16:
                arc_candidates.append((arc["strength"] * (1 - residual), template_arc, arc, residual))
    arc_matches, used_template, used_detected = [], set(), set()
    for quality, template_arc, arc, residual in sorted(arc_candidates, reverse=True, key=lambda value: value[0]):
        if template_arc["id"] in used_template or arc["id"] in used_detected:
            continue
        used_template.add(template_arc["id"])
        used_detected.add(arc["id"])
        arc_matches.append({"template_feature": template_arc["id"], "detected_primitive": arc["id"], "normalized_ellipse_residual": round(residual, 3)})
    structure["free_throw_circle_matches"] = [match for match in arc_matches if "free_throw_circle" in match["template_feature"]]
    line_score = sum(match["quality"] for match in matches)
    arc_score = sum(1 - match["normalized_ellipse_residual"] for match in arc_matches)
    independent_lines = sum(not match["seed"] for match in matches)
    seed_support = sum(1 for template_id, detected_id in seed_pairs.items() if any(match["template_feature"] == template_id and match["detected_primitive"] == detected_id for match in matches))
    structure_score = .75 * structure["connected_pairs"] + .35 * structure["inverse_parallel_pairs"] + .25 * structure["right_angle_pairs"]
    return {"line_matches": matches, "arc_matches": arc_matches, "structural_evidence": structure, "seed_segment_support": seed_support, "independent_line_inliers": independent_lines, "score": round(line_score + .55 * arc_score + .6 * independent_lines + structure_score, 4)}


def _orientation_sanity(homography: np.ndarray, template: dict[str, Any], width: int, height: int) -> str | None:
    length, court_width = template["dimensions"]["length"], template["dimensions"]["width"]
    corners = _project(np.array([[0, 0], [length, 0], [length, court_width], [0, court_width]], dtype=np.float32), homography)
    if not np.isfinite(corners).all():
        return "non_finite_projection"
    cross = []
    for index in range(4):
        a, b, c = corners[index], corners[(index + 1) % 4], corners[(index + 2) % 4]
        first, second = b - a, c - b
        cross.append(float(first[0] * second[1] - first[1] * second[0]))
    if min(abs(value) for value in cross) < 4 or not (all(value > 0 for value in cross) or all(value < 0 for value in cross)):
        return "self_crossing_or_degenerate_court_outline"
    area = abs(sum(corners[index, 0] * corners[(index + 1) % 4, 1] - corners[(index + 1) % 4, 0] * corners[index, 1] for index in range(4))) / 2
    if area < width * height * .003:
        return "implausibly_small_court_projection"
    return None


REVIEW_HUD_OVERLAP_THRESHOLD = 0.25
REVIEW_MIN_HUD_SUPPORTED_SEED_LINES = 3
REVIEW_MIN_UNIQUE_LINE_MATCHES = 3
REVIEW_MIN_MATCHED_KEYPOINTS = 4
REVIEW_MAX_MEAN_ERROR_PX = 12.0
REVIEW_MAX_OUTSIDE_FRACTION = 0.85


def _review_candidate_evaluation(
    *,
    homography: np.ndarray,
    template: dict[str, Any],
    primitives: list[dict[str, Any]],
    width: int,
    height: int,
    scored: dict[str, Any],
    structure: dict[str, Any],
    lines_by_id: dict[str, dict[str, Any]],
    selected_template: list[dict[str, Any]],
    selected_detected: list[dict[str, Any]],
    proposal: dict[str, Any],
    canonical_key: tuple[tuple[str, str], ...],
) -> tuple[dict[str, Any] | None, str | None]:
    """Evaluate one candidate against the review-only (not automatic) thresholds.

    Uses only detected geometry already computed by the unchanged scorer; never
    manual labels or frame-specific identity. Returns ``(evaluation, None)`` when
    eligible, where ``evaluation`` holds the candidate dict plus its deterministic
    rank tuple, or ``(None, reason)`` for the first failing review-only check.
    """
    if scored["seed_segment_support"] < 3:
        return None, "insufficient_seed_segment_support"
    if structure["unique_line_matches"] < REVIEW_MIN_UNIQUE_LINE_MATCHES:
        return None, "insufficient_unique_line_matches"
    if structure["connected_pairs"] < 1:
        return None, "missing_connected_pair"
    if structure["inverse_parallel_pairs"] < 1:
        return None, "missing_inverse_parallel"
    if structure["right_angle_pairs"] < 1:
        return None, "missing_right_angle"

    reprojection = reproject_keypoints(homography, template, primitives, width, height)
    # `reproject_keypoints` matches each template keypoint independently, so a
    # homography that maps several template keypoints on top of one another can
    # double-count a single detected intersection as several "matches". Distinct
    # detected primitives are what "at least four projected keypoints match
    # detected intersections" means: independent confirmations, not duplicates.
    matched_points = [point for point in reprojection["points"] if point["error_px"] is not None]
    matched_keypoints = len({point["detected_primitive"] for point in matched_points})
    mean_error = reprojection["mean_matched_error_px"]
    outside_fraction = reprojection["outside_keypoints"] / max(1, len(reprojection["points"]))
    if matched_keypoints < REVIEW_MIN_MATCHED_KEYPOINTS:
        return None, "insufficient_matched_keypoints"
    if mean_error is None or mean_error > REVIEW_MAX_MEAN_ERROR_PX:
        return None, "mean_error_exceeds_threshold"
    if outside_fraction > REVIEW_MAX_OUTSIDE_FRACTION:
        return None, "outside_fraction_exceeds_threshold"

    seed_matches = [match for match in scored["line_matches"] if match["seed"]]
    supported_seed_matches = [
        match for match in seed_matches
        if lines_by_id[match["detected_primitive"]]["evidence"].get("hud_overlap_ratio", 1.0)
        < REVIEW_HUD_OVERLAP_THRESHOLD
    ]
    if len(supported_seed_matches) < REVIEW_MIN_HUD_SUPPORTED_SEED_LINES:
        return None, "insufficient_hud_supported_seed_lines"

    # Generic re-check of the unchanged automatic thresholds, purely to explain
    # to a reviewer why this candidate did not clear the automatic gate. This
    # never feeds back into automatic acceptance.
    automatic_gate_failures: list[str] = []
    if structure["unique_line_matches"] < 4:
        automatic_gate_failures.append("unique_line_assignment")
    if structure["connected_pairs"] < 1:
        automatic_gate_failures.append("connected_court_substructure")
    if structure["inverse_parallel_pairs"] < 1 or structure["right_angle_pairs"] < 1:
        automatic_gate_failures.append("court_angle_structure")
    if scored["independent_line_inliers"] < 2:
        automatic_gate_failures.append("insufficient_non_seed_line_agreement")
    if matched_keypoints < 4:
        automatic_gate_failures.append(f"only_{matched_keypoints}_matched_keypoints_below_minimum_4")
    if mean_error > 12:
        automatic_gate_failures.append(f"mean_reprojection_error_{mean_error}_exceeds_12px")
    if outside_fraction > .85:
        automatic_gate_failures.append("too_many_reprojected_keypoints_outside_image")
    orientation_issue = _orientation_sanity(homography, template, width, height)
    if orientation_issue:
        automatic_gate_failures.append(orientation_issue)

    structural_sum = (
        structure["connected_pairs"] + structure["inverse_parallel_pairs"] + structure["right_angle_pairs"]
    )
    # The canonical correspondence identity is compared separately (smaller
    # wins) rather than folded into `rank` itself: it is a final deterministic
    # tie-breaker only, reached when every geometric criterion above is
    # genuinely tied (e.g. a template's mirror-symmetric north/south halves
    # produce identical detected-geometry evidence for either assignment).
    rank = (
        matched_keypoints,
        -mean_error,
        structure["unique_line_matches"],
        structural_sum,
        scored["independent_line_inliers"],
        scored["score"],
    )
    candidate = {
        "status": "review_required",
        "homography": np.asarray(homography).round(9).tolist(),
        "seed_correspondences": [
            {"template_feature": source["id"], "detected_primitive": target["id"]}
            for source, target in zip(selected_template, selected_detected)
        ],
        "proposal": {"type": proposal["proposal_type"], "source": proposal.get("source", {})},
        "inliers": scored["line_matches"] + scored["arc_matches"],
        "seed_segment_support": scored["seed_segment_support"],
        "independent_line_inliers": scored["independent_line_inliers"],
        "structural_evidence": structure,
        "agreement_score": scored["score"],
        "reprojection": reprojection["points"],
        "matched_keypoints": matched_keypoints,
        "mean_matched_error_px": mean_error,
        "outside_keypoints": reprojection["outside_keypoints"],
        "outside_keypoint_fraction": round(outside_fraction, 4),
        "hud_support": {
            "seed_line_count": len(seed_matches),
            "hud_supported_seed_line_count": len(supported_seed_matches),
            "hud_overlap_threshold": REVIEW_HUD_OVERLAP_THRESHOLD,
            "seed_lines": [
                {
                    "template_feature": match["template_feature"],
                    "detected_primitive": match["detected_primitive"],
                    "hud_overlap_ratio": lines_by_id[match["detected_primitive"]]["evidence"].get(
                        "hud_overlap_ratio"
                    ),
                }
                for match in seed_matches
            ],
        },
        "automatic_gate_failures": automatic_gate_failures,
        "explanation": (
            "Review-only candidate: satisfies relaxed structural, keypoint-match, and HUD "
            "thresholds but not every automatic acceptance requirement. This is not an "
            "automatic label; it requires human review and explicit approval before use."
        ),
    }
    return {"candidate": candidate, "rank": rank, "canonical_key": canonical_key}, None


def solve_homography(detection: dict[str, Any], template: dict[str, Any], *, hypotheses: int = 5000, line_distance_px: float = 10.0, random_seed: int = 7) -> dict[str, Any]:
    """Use guided, unlabeled four-line correspondences and independent scoring.

    Candidate construction uses detected intersection adjacency and local
    projective orientation families.  It never applies an arbitrary four-line
    permutation.  Each candidate is still evaluated by the existing finite
    segment, one-to-one, structural, and keypoint checks.
    """
    primitives = detection["primitives"]
    lines = [item for item in primitives if item["type"] == "line_segment"]
    arcs = [item for item in primitives if item["type"] in {"arc", "circle"}]
    width, height = detection["image_size"]["width"], detection["image_size"]["height"]
    if len(lines) < 4:
        _, proposal_diagnostics = generate_guided_proposals(
            detection, template, max_proposals=hypotheses, random_seed=random_seed
        )
        return {
            "status": "rejected", "reason": "fewer_than_four_line_segments",
            "homography": None, "inliers": [], "reprojection": [],
            "hypotheses_tried": 0, "hypotheses_rejected": {},
            "proposal_diagnostics": proposal_diagnostics,
            "review_candidate": None,
        }

    proposals, proposal_diagnostics = generate_guided_proposals(
        detection, template, max_proposals=hypotheses, random_seed=random_seed
    )
    proposal_diagnostics.setdefault("invalid_homographies", 0)
    proposal_diagnostics.setdefault("orientation_sanity_rejections", 0)
    lines_by_id = {line["id"]: line for line in lines}
    best: dict[str, Any] | None = None
    best_review: dict[str, Any] | None = None
    rejected_by_structure: dict[str, int] = {"seed_segment_extent": 0, "unique_line_assignment": 0, "connected_court_substructure": 0, "court_angle_structure": 0}
    review_rejections: dict[str, int] = {
        "insufficient_seed_segment_support": 0, "insufficient_unique_line_matches": 0,
        "missing_connected_pair": 0, "missing_inverse_parallel": 0, "missing_right_angle": 0,
        "insufficient_matched_keypoints": 0, "mean_error_exceeds_threshold": 0,
        "outside_fraction_exceeds_threshold": 0, "insufficient_hud_supported_seed_lines": 0,
    }
    review_candidates_evaluated = 0
    tried: set[tuple[tuple[str, str], ...]] = set()
    for proposal in proposals:
        selected_template = proposal["template_lines"]
        selected_detected = proposal["detected_lines"]
        key = tuple(sorted(
            (str(source["id"]), str(target["id"]))
            for source, target in zip(selected_template, selected_detected)
        ))
        if key in tried:
            proposal_diagnostics["duplicates_removed"] = proposal_diagnostics.get("duplicates_removed", 0) + 1
            continue
        tried.add(key)
        homography = _homography_from_line_matches(selected_template, selected_detected)
        if homography is None:
            proposal_diagnostics["invalid_homographies"] += 1
            continue
        orientation_issue = _orientation_sanity(homography, template, width, height)
        if orientation_issue:
            proposal_diagnostics["orientation_sanity_rejections"] += 1
            continue
        seed_pairs = proposal["seed_pairs"]
        # Finite seed extent is a cheap necessary condition for the unchanged
        # full one-to-one scorer.  Apply it before comparing every retained
        # template/detection pair so a large guided budget stays practical.
        direct_seed_support = sum(
            _segment_agreement(source, target, homography, width, height, line_distance_px) is not None
            for source, target in zip(selected_template, selected_detected)
        )
        if direct_seed_support < 3:
            rejected_by_structure["seed_segment_extent"] += 1
            continue
        proposal_diagnostics["candidates_reaching_scoring"] += 1
        scored = _score_homography(homography, template, lines, arcs, seed_pairs, width, height, line_distance_px)
        structure = scored["structural_evidence"]
        # Reject weak candidates here, before the keypoint-intersection gate.
        # The gate remains a separate final safeguard rather than compensating
        # for correspondence evidence that is already geometrically implausible.
        # This if/elif chain is behavior-identical to the previous cascading
        # if/continue checks (same first-failing reason, same counts, same
        # condition gating `best`); it is only restructured so a candidate that
        # fails here can still be considered below for the independent,
        # strictly-relaxed review-only track.
        automatic_reason: str | None = None
        if scored["seed_segment_support"] < 3:
            automatic_reason = "seed_segment_extent"
        elif structure["unique_line_matches"] < 4:
            automatic_reason = "unique_line_assignment"
        elif structure["connected_pairs"] < 1:
            automatic_reason = "connected_court_substructure"
        elif structure["inverse_parallel_pairs"] < 1 or structure["right_angle_pairs"] < 1:
            automatic_reason = "court_angle_structure"

        if automatic_reason is not None:
            rejected_by_structure[automatic_reason] += 1
        elif best is None or scored["score"] > best["score"]:
            best = {
                **scored,
                "homography": homography,
                "seed_correspondences": [
                    {"template_feature": source["id"], "detected_primitive": target["id"]}
                    for source, target in zip(selected_template, selected_detected)
                ],
                "proposal": {
                    "type": proposal["proposal_type"],
                    "source": proposal.get("source", {}),
                },
            }

        review_candidates_evaluated += 1
        evaluation, review_reason = _review_candidate_evaluation(
            homography=homography, template=template, primitives=primitives, width=width, height=height,
            scored=scored, structure=structure, lines_by_id=lines_by_id,
            selected_template=selected_template, selected_detected=selected_detected,
            proposal=proposal, canonical_key=key,
        )
        if review_reason is not None:
            review_rejections[review_reason] += 1
        elif best_review is None or evaluation["rank"] >= best_review["rank"]:
            # A partial view can be exactly symmetric (e.g. a lane's east and
            # west halves mirrored about a shared, axis-invariant baseline and
            # free-throw line): every detected-geometry criterion above ties
            # bit-for-bit, and no further detected evidence can break the tie.
            # The deterministic guided-proposal stream order is the final,
            # generic tie-breaker (later-generated candidates -- reached after
            # topology-guided multi-corner sources are exhausted -- win exact
            # ties), which is reproducible for a fixed `random_seed` without
            # relying on any manual label or frame-specific identity.
            best_review = evaluation
    proposal_diagnostics["unique_mappings_tried"] = len(tried)
    proposal_diagnostics["review_candidates_evaluated"] = review_candidates_evaluated
    proposal_diagnostics["review_rejection_reasons"] = review_rejections
    proposal_diagnostics["review_candidate_found"] = best_review is not None
    if best is None:
        if sum(rejected_by_structure.values()):
            dominant_reason = max(rejected_by_structure, key=rejected_by_structure.get)
        else:
            dominant_reason = "no_candidate_reached_scoring"
        return {"status": "rejected", "reason": f"no_court_structural_consensus:{dominant_reason}", "homography": None, "inliers": [], "reprojection": [], "hypotheses_tried": len(tried), "hypotheses_rejected": rejected_by_structure, "proposal_diagnostics": proposal_diagnostics, "review_candidate": best_review["candidate"] if best_review is not None else None}
    reprojection = reproject_keypoints(best["homography"], template, primitives, width, height)
    mean_error = reprojection["mean_matched_error_px"]
    return {"status": "solved", "homography": np.asarray(best["homography"]).round(9).tolist(), "inliers": best["line_matches"] + best["arc_matches"], "seed_correspondences": best["seed_correspondences"], "proposal": best["proposal"], "independent_line_inliers": best["independent_line_inliers"], "agreement_score": best["score"], "structural_evidence": best["structural_evidence"], "seed_segment_support": best["seed_segment_support"], "reprojection_error_px": mean_error, "reprojection": reprojection["points"], "hypotheses_tried": len(tried), "hypotheses_rejected": rejected_by_structure, "proposal_diagnostics": proposal_diagnostics, "orientation_issue": _orientation_sanity(best["homography"], template, width, height), "review_candidate": None}


def reproject_keypoints(homography: np.ndarray, template: dict[str, Any], primitives: list[dict[str, Any]], width: int, height: int, match_distance_px: float = 25.0) -> dict[str, Any]:
    """Project every template keypoint and compare it to detected intersections."""
    keypoint_items = list(template["keypoints"].items())
    projected = _project(np.asarray([point for _, point in keypoint_items], np.float32), homography)
    intersections = [item for item in primitives if item["type"] == "intersection"]
    detected = np.asarray([item["geometry"]["point"] for item in intersections], dtype=float) if intersections else np.empty((0, 2))
    points, errors = [], []
    for (keypoint_id, _), pixel in zip(keypoint_items, projected):
        in_bounds = bool(0 <= pixel[0] < width and 0 <= pixel[1] < height)
        output: dict[str, Any] = {"id": keypoint_id, "template_xy": template["keypoints"][keypoint_id], "pixel_xy": [round(float(pixel[0]), 2), round(float(pixel[1]), 2)], "in_image_bounds": in_bounds, "detected_primitive": None, "error_px": None}
        if len(detected):
            distances = np.linalg.norm(detected - pixel, axis=1)
            nearest = int(np.argmin(distances))
            if distances[nearest] <= match_distance_px:
                output["detected_primitive"] = intersections[nearest]["id"]
                output["error_px"] = round(float(distances[nearest]), 2)
                errors.append(float(distances[nearest]))
        points.append(output)
    return {"points": points, "mean_matched_error_px": round(float(np.mean(errors)), 3) if errors else None, "matched_keypoints": len(errors), "outside_keypoints": sum(not item["in_image_bounds"] for item in points)}


def verify_solution(solution: dict[str, Any], detection: dict[str, Any], template: dict[str, Any], *, max_error_px: float = 12, min_matched_keypoints: int = 4, min_independent_line_inliers: int = 2, max_outside_fraction: float = .85) -> dict[str, Any]:
    """Apply transparent error and geometric sanity gates to a solved frame."""
    if solution.get("status") != "solved":
        return {"status": "fail", "reasons": [solution.get("reason", "not_solved")], "metrics": {}}
    points = solution["reprojection"]
    matched = [point for point in points if point["error_px"] is not None]
    outside = sum(not point["in_image_bounds"] for point in points)
    reasons = []
    if len(matched) < min_matched_keypoints:
        reasons.append(f"only_{len(matched)}_matched_keypoints_below_minimum_{min_matched_keypoints}")
    mean_error = solution["reprojection_error_px"]
    if mean_error is None or mean_error > max_error_px:
        reasons.append(f"mean_reprojection_error_{mean_error}_exceeds_{max_error_px}px")
    if solution["independent_line_inliers"] < min_independent_line_inliers:
        reasons.append("insufficient_non_seed_line_agreement")
    if outside / max(1, len(points)) > max_outside_fraction:
        reasons.append("too_many_reprojected_keypoints_outside_image")
    if solution.get("orientation_issue"):
        reasons.append(solution["orientation_issue"])
    return {"status": "pass" if not reasons else "fail", "reasons": reasons, "metrics": {"mean_reprojection_error_px": mean_error, "matched_keypoints": len(matched), "outside_keypoints": outside, "independent_line_inliers": solution["independent_line_inliers"], "agreement_score": solution["agreement_score"]}, "thresholds": {"max_error_px": max_error_px, "min_matched_keypoints": min_matched_keypoints, "min_independent_line_inliers": min_independent_line_inliers, "max_outside_fraction": max_outside_fraction}}


def process_frame(
    image_path: str | Path,
    template_path: str | Path = DEFAULT_TEMPLATE,
    *,
    hud_context: HudContext | None = None,
    **solve_options: Any,
) -> dict[str, Any]:
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    template = load_template(template_path)
    detection = detect_primitives(image, hud_context=hud_context)
    solution = solve_homography(detection, template, **solve_options)
    verification = verify_solution(solution, detection, template)
    return {"frame": str(image_path), "template": {"name": template.get("name"), "units": template.get("units"), "path": str(template_path)}, "detection": detection, "solution": solution, "verification": verification}
