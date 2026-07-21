"""Position-agnostic broadcast HUD detection for the court pipeline.

The detector is deliberately conservative.  It does not remove primitives; it
returns a filled mask whose overlap can be used as soft evidence.  With enough
same-clip frames it looks for stable, persistent graphic detail.  A singleton
falls back to a stricter dense-edge/colour/contrast test near an image border.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FRAME_NAME = re.compile(r"^(?P<clip>.+)_f(?P<index>\d+)$")
MAX_CONTEXT_FRAMES = 7
MIN_TEMPORAL_FRAMES = 3
MIN_TEMPORAL_SPAN = 30
SATURATED_PIXEL_MIN = 80


@dataclass(frozen=True)
class HudContext:
    """Loaded, same-clip/same-resolution frames selected around an anchor."""

    images: tuple[np.ndarray, ...]
    paths: tuple[str, ...]
    frame_indices: tuple[int | None, ...]
    anchor_position: int
    frame_span: int
    mode: str


@dataclass(frozen=True)
class HudDetection:
    """A binary HUD mask, its region labels, and JSON-safe diagnostics."""

    mask: np.ndarray
    region_labels: np.ndarray
    diagnostics: dict[str, Any]


def _frame_identity(path: Path) -> tuple[str | None, int | None]:
    match = FRAME_NAME.match(path.stem)
    if match is None:
        return None, None
    return match.group("clip"), int(match.group("index"))


def _candidate_paths(anchor: Path, explicit_paths: Iterable[str | Path]) -> list[Path]:
    clip, _ = _frame_identity(anchor)
    candidates = [anchor]
    if clip is not None and anchor.parent.is_dir():
        candidates.extend(
            path for path in anchor.parent.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            and _frame_identity(path)[0] == clip
        )
    for supplied in explicit_paths:
        path = Path(supplied)
        if path.is_dir():
            candidates.extend(
                item for item in path.rglob("*")
                if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
            )
        else:
            candidates.append(path)
    unique: dict[str, Path] = {}
    for path in candidates:
        try:
            key = str(path.resolve())
        except OSError:
            key = str(path.absolute())
        unique.setdefault(key, path)
    return list(unique.values())


def _evenly_spaced_records(
    records: Sequence[tuple[Path, np.ndarray, int | None]],
    anchor_key: str,
    limit: int,
) -> list[tuple[Path, np.ndarray, int | None]]:
    """Keep the anchor and choose the remaining frames across the full span."""
    if len(records) <= limit:
        return sorted(records, key=lambda item: (item[2] is None, item[2] or 0, str(item[0])))
    anchor = next(item for item in records if str(item[0].resolve()) == anchor_key)
    selected = [anchor]
    remaining = [item for item in records if item is not anchor]
    numeric = [item[2] for item in records if item[2] is not None]
    if numeric:
        grid = list(np.linspace(min(numeric), max(numeric), limit))
        # Visit the range endpoints before interior targets so forcing the
        # anchor into the sample cannot consume the final slot and drop one
        # end of the clip span.
        targets = [grid[0], grid[-1], *grid[1:-1]]
        for target in targets:
            if len(selected) >= limit or not remaining:
                break
            closest = min(
                remaining,
                key=lambda item: (
                    abs((item[2] if item[2] is not None else float("inf")) - target),
                    item[2] is None,
                    str(item[0]),
                ),
            )
            selected.append(closest)
            remaining.remove(closest)
    while len(selected) < limit and remaining:
        # Fill any duplicate-target holes with the frame farthest from the
        # frames already selected, retaining deterministic path tie-breaking.
        closest_distances = []
        selected_indices = [item[2] for item in selected if item[2] is not None]
        for item in remaining:
            distance = min((abs(item[2] - value) for value in selected_indices), default=0) if item[2] is not None else -1
            closest_distances.append((distance, str(item[0]), item))
        chosen = max(closest_distances, key=lambda value: (value[0], value[1]))[2]
        selected.append(chosen)
        remaining.remove(chosen)
    return sorted(selected, key=lambda item: (item[2] is None, item[2] or 0, str(item[0])))


def build_hud_context(
    anchor_path: str | Path,
    context_paths: Iterable[str | Path] = (),
    *,
    anchor_image: np.ndarray | None = None,
    max_frames: int = MAX_CONTEXT_FRAMES,
) -> HudContext:
    """Discover and load at most seven same-clip, same-resolution frames.

    ``context_paths`` may contain files or directories and is useful when the
    anchor lives alone in an export directory.  Explicit frames still have to
    share the anchor's ``<clip>_f<index>`` identity and image resolution.
    """
    anchor = Path(anchor_path)
    anchor_key = str(anchor.resolve())
    anchor_clip, anchor_index = _frame_identity(anchor)
    if max_frames < 1:
        raise ValueError("max_frames must be at least one")
    if anchor_image is None:
        anchor_image = cv2.imread(str(anchor))
    if anchor_image is None or anchor_image.size == 0:
        raise ValueError(f"Could not read HUD anchor image: {anchor}")
    height, width = anchor_image.shape[:2]

    records: list[tuple[Path, np.ndarray, int | None]] = []
    seen_indices: set[int] = set()
    # Put the actual anchor first so a duplicate explicit copy of the same
    # frame index cannot displace it.
    candidates = [anchor] + [path for path in _candidate_paths(anchor, context_paths) if str(path.resolve()) != anchor_key]
    for path in candidates:
        clip, frame_index = _frame_identity(path)
        if anchor_clip is None or clip != anchor_clip:
            if str(path.resolve()) != anchor_key:
                continue
        if frame_index is not None and frame_index in seen_indices:
            continue
        image = anchor_image if str(path.resolve()) == anchor_key else cv2.imread(str(path))
        if image is None or image.size == 0 or image.shape[:2] != (height, width):
            continue
        records.append((path, image, frame_index))
        if frame_index is not None:
            seen_indices.add(frame_index)

    selected = _evenly_spaced_records(records, anchor_key, min(max_frames, MAX_CONTEXT_FRAMES))
    anchor_position = next(index for index, item in enumerate(selected) if str(item[0].resolve()) == anchor_key)
    indices = [item[2] for item in selected]
    numeric = [value for value in indices if value is not None]
    frame_span = max(numeric) - min(numeric) if numeric else 0
    mode = "temporal" if len(selected) >= MIN_TEMPORAL_FRAMES and frame_span >= MIN_TEMPORAL_SPAN else "static_singleton"
    return HudContext(
        images=tuple(item[1] for item in selected),
        paths=tuple(str(item[0]) for item in selected),
        frame_indices=tuple(indices),
        anchor_position=anchor_position,
        frame_span=frame_span,
        mode=mode,
    )


def _odd_window(dimension: int) -> int:
    value = max(15, int(round(.035 * dimension)))
    return value if value % 2 else value + 1


def _ramp(values: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _local_mean(values: np.ndarray, window: tuple[int, int]) -> np.ndarray:
    return cv2.boxFilter(values.astype(np.float32), -1, window, normalize=True, borderType=cv2.BORDER_REFLECT)


def _edge_map(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, 45, 130, apertureSize=3, L2gradient=True) > 0


def _close_seed(seed: np.ndarray, min_dimension: int) -> np.ndarray:
    radius = max(1, int(round(.001 * min_dimension)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.morphologyEx(seed.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0


def _border_distance(x: int, y: int, width: int, height: int, image_width: int, image_height: int) -> int:
    return min(x, y, image_width - (x + width), image_height - (y + height))


def _component_candidates(
    seed: np.ndarray,
    feature_maps: dict[str, np.ndarray],
    *,
    mode: str,
) -> list[dict[str, Any]]:
    height, width = seed.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(seed.astype(np.uint8), connectivity=8)
    output: list[dict[str, Any]] = []
    for component in range(1, count):
        x, y, box_width, box_height, area = (int(value) for value in stats[component])
        area_fraction = area / float(width * height)
        fill = area / float(max(1, box_width * box_height))
        border_distance = _border_distance(x, y, box_width, box_height, width, height)
        pixels = labels == component
        means = {name: float(np.mean(values[pixels])) for name, values in feature_maps.items()}
        if mode == "temporal":
            accepted = (
                .003 <= area_fraction <= .08
                and box_width >= .04 * width and box_height >= .04 * height
                and fill >= .30
                and border_distance <= .04 * min(width, height)
                and means["local_stability"] >= .90
                and means["persistent_edge_density"] >= .055
                and means["saturated_fraction"] >= .25
            )
        else:
            accepted = (
                .003 <= area_fraction <= .08
                and box_width >= .08 * width and box_height >= .05 * height
                and fill >= .30
                and box_width / max(1, box_height) >= 1.5
                and border_distance <= .02 * min(width, height)
                and means["edge_density"] >= .09
                and means["saturated_fraction"] >= .35
                and means["value_std"] >= 40
            )
        if accepted:
            output.append({
                "bbox": [x, y, box_width, box_height],
                "component_area_px": area,
                "area_fraction": area_fraction,
                "fill_ratio": fill,
                "border_distance_px": border_distance,
                "means": means,
            })
    return output


def _rectangles_overlap(first: Sequence[int], second: Sequence[int]) -> bool:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    return ax1 <= bx2 and bx1 <= ax2 and ay1 <= by2 and by1 <= ay2


def _union_rectangles(rectangles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = [dict(rect=item["rect"], components=[item["component"]]) for item in rectangles]
    changed = True
    while changed:
        changed = False
        for first_index in range(len(groups)):
            for second_index in range(first_index + 1, len(groups)):
                if not _rectangles_overlap(groups[first_index]["rect"], groups[second_index]["rect"]):
                    continue
                first, second = groups[first_index], groups.pop(second_index)
                ax1, ay1, ax2, ay2 = first["rect"]
                bx1, by1, bx2, by2 = second["rect"]
                first["rect"] = [min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2)]
                first["components"].extend(second["components"])
                changed = True
                break
            if changed:
                break
    return sorted(groups, key=lambda item: (item["rect"][1], item["rect"][0]))


def _filled_regions(
    components: list[dict[str, Any]],
    image_size: tuple[int, int],
    window: tuple[int, int],
    mode: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    height, width = image_size
    window_width, window_height = window
    padded = []
    for component in components:
        x, y, box_width, box_height = component["bbox"]
        if mode == "temporal":
            x_pad, y_pad = window_width // 2, window_height // 2
        else:
            x_pad = max(window_width // 2, int(math.ceil(.25 * box_width)))
            y_pad = max(window_height // 2, int(math.ceil(.10 * box_height)))
        x1, y1 = max(0, x - x_pad), max(0, y - y_pad)
        x2, y2 = min(width, x + box_width + x_pad), min(height, y + box_height + y_pad)
        padded.append({"rect": [x1, y1, x2, y2], "component": component})

    mask = np.zeros((height, width), np.uint8)
    region_labels = np.zeros((height, width), np.int32)
    diagnostics = []
    for index, group in enumerate(_union_rectangles(padded), 1):
        x1, y1, x2, y2 = group["rect"]
        mask[y1:y2, x1:x2] = 255
        region_labels[y1:y2, x1:x2] = index
        sources = group["components"]
        diagnostics.append({
            "id": f"hud_{index - 1:03d}",
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "source_components": len(sources),
            "source_bboxes": [item["bbox"] for item in sources],
            "area_fraction": round((x2 - x1) * (y2 - y1) / float(width * height), 6),
            "component_area_fraction": round(sum(item["area_fraction"] for item in sources), 6),
            "fill_ratio": round(float(np.mean([item["fill_ratio"] for item in sources])), 4),
            "border_distance_px": min(item["border_distance_px"] for item in sources),
            "means": {
                name: round(float(np.mean([item["means"][name] for item in sources])), 4)
                for name in sources[0]["means"]
            },
        })
    return mask, region_labels, diagnostics


def _temporal_maps(images: Sequence[np.ndarray], anchor_position: int, window: tuple[int, int]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    # Anchor-relative error avoids declaring two mutually similar context
    # frames stable when the actual frame being labelled differs from both.
    anchor = images[anchor_position]
    error_maps = []
    for candidate in images:
        difference = cv2.absdiff(candidate, anchor)
        channel_sum = (
            difference[:, :, 0].astype(np.uint16)
            + difference[:, :, 1].astype(np.uint16)
            + difference[:, :, 2].astype(np.uint16)
        )
        error_maps.append(channel_sum.astype(np.float32) / 3.0)
    median_bgr_error = np.median(np.stack(error_maps), axis=0)
    stable = median_bgr_error <= 10

    anchor_edge = None
    persistent_count = np.zeros(anchor.shape[:2], np.uint8)
    for index, candidate in enumerate(images):
        edge = _edge_map(candidate)
        if index == anchor_position:
            anchor_edge = edge
        persistent_count += cv2.dilate(edge.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    required = int(math.ceil(.60 * len(images)))
    assert anchor_edge is not None
    persistent_edge = anchor_edge & (persistent_count >= required)
    hsv = cv2.cvtColor(anchor, cv2.COLOR_BGR2HSV)
    saturated = hsv[:, :, 1] >= SATURATED_PIXEL_MIN

    local_stability = _local_mean(stable, window)
    persistent_edge_density = _local_mean(persistent_edge, window)
    saturated_fraction = _local_mean(saturated, window)
    score = (
        _ramp(local_stability, .55, .82)
        * _ramp(persistent_edge_density, .008, .04)
        * (.35 + .65 * _ramp(saturated_fraction, .20, .50))
    )
    maps = {
        "local_stability": local_stability,
        "persistent_edge_density": persistent_edge_density,
        "saturated_fraction": saturated_fraction,
        "hud_score": score,
        "median_bgr_error": median_bgr_error,
    }
    return score >= .55, maps


def _static_maps(image: np.ndarray, window: tuple[int, int]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    edge_density = _local_mean(_edge_map(image), window)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturated_fraction = _local_mean(hsv[:, :, 1] >= SATURATED_PIXEL_MIN, window)
    value = hsv[:, :, 2].astype(np.float32)
    value_mean = _local_mean(value, window)
    value_squared_mean = _local_mean(value * value, window)
    value_std = np.sqrt(np.maximum(0, value_squared_mean - value_mean * value_mean))
    seed = (edge_density >= .075) & (saturated_fraction >= .30) & (value_std >= 30)
    return seed, {
        "edge_density": edge_density,
        "saturated_fraction": saturated_fraction,
        "value_std": value_std,
    }


def detect_hud(
    image: np.ndarray,
    context: HudContext | None = None,
    *,
    context_images: Sequence[np.ndarray] = (),
    frame_indices: Sequence[int | None] = (),
    context_frames: Sequence[str] = (),
    anchor_position: int = 0,
) -> HudDetection:
    """Detect broadcast-graphic regions and return a soft-evidence mask.

    Callers normally pass a :class:`HudContext`.  The array-based arguments are
    provided for tests and in-memory/video integrations.  Temporal mode requires
    at least three frames with a numeric frame-index span of at least 30.
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot detect a HUD in an empty image")
    height, width = image.shape[:2]
    if context is not None:
        images = list(context.images)
        indices = list(context.frame_indices)
        paths = list(context.paths)
        anchor_position = context.anchor_position
        mode = context.mode
        frame_span = context.frame_span
    else:
        images = list(context_images) or [image]
        if not any(candidate is image for candidate in images):
            images.insert(0, image)
            anchor_position = 0
        indices = list(frame_indices)
        if len(indices) != len(images):
            indices = [None] * len(images)
        numeric = [value for value in indices if value is not None]
        frame_span = max(numeric) - min(numeric) if numeric else 0
        mode = "temporal" if len(images) >= MIN_TEMPORAL_FRAMES and frame_span >= MIN_TEMPORAL_SPAN else "static_singleton"
        paths = list(context_frames) if len(context_frames) == len(images) else []
    if not 0 <= anchor_position < len(images):
        raise ValueError("anchor_position is outside the HUD context")
    if any(candidate.shape[:2] != (height, width) for candidate in images):
        raise ValueError("HUD context images must have the anchor resolution")

    window = (_odd_window(width), _odd_window(height))
    if mode == "temporal":
        seed, maps = _temporal_maps(images, anchor_position, window)
    else:
        seed, maps = _static_maps(image, window)
    closed = _close_seed(seed, min(width, height))
    components = _component_candidates(closed, maps, mode=mode)
    mask, region_labels, regions = _filled_regions(components, (height, width), window, mode)

    masked_fraction = round(float(np.count_nonzero(mask)) / mask.size, 6)
    context_group = _frame_identity(Path(paths[anchor_position]))[0] if paths else None
    diagnostics: dict[str, Any] = {
        "mode": mode,
        "context_group": context_group,
        "context_frames": paths,
        "context_frame_indices": indices,
        "context_frame_count": len(images),
        "frame_span": frame_span,
        "windows": {"width_px": window[0], "height_px": window[1]},
        "window_px": [window[0], window[1]],
        "regions": regions,
        "masked_fraction": masked_fraction,
        "masked_pixel_fraction": masked_fraction,
        "thresholds": {
            "saturated_pixel_min": SATURATED_PIXEL_MIN,
            "component_area_fraction": [.003, .08],
            "seed_score_min": .55 if mode == "temporal" else None,
            "temporal_min_frames": MIN_TEMPORAL_FRAMES,
            "temporal_min_span": MIN_TEMPORAL_SPAN,
        },
    }
    return HudDetection(mask=mask, region_labels=region_labels, diagnostics=diagnostics)


def _sampled_overlap(
    hud: HudDetection | np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    region_labels: np.ndarray | None,
) -> dict[str, Any]:
    mask = hud.mask if isinstance(hud, HudDetection) else hud
    labels = hud.region_labels if isinstance(hud, HudDetection) else region_labels
    height, width = mask.shape[:2]
    rounded_x = np.rint(xs).astype(np.int32)
    rounded_y = np.rint(ys).astype(np.int32)
    inside = (rounded_x >= 0) & (rounded_x < width) & (rounded_y >= 0) & (rounded_y < height)
    rounded_x, rounded_y = rounded_x[inside], rounded_y[inside]
    total = len(rounded_x)
    overlap = float(np.count_nonzero(mask[rounded_y, rounded_x])) / total if total else 0.0
    region_ids: list[str] = []
    if labels is not None and total:
        identifiers = sorted(int(value) for value in np.unique(labels[rounded_y, rounded_x]) if value > 0)
        region_ids = [f"hud_{value - 1:03d}" for value in identifiers]
    return {"hud_overlap": round(overlap, 4), "hud_region_ids": region_ids}


def line_hud_overlap(
    hud: HudDetection | np.ndarray,
    p1: Sequence[float],
    p2: Sequence[float],
    *,
    thickness: int = 3,
    region_labels: np.ndarray | None = None,
) -> dict[str, Any]:
    """Sample mask/region overlap along a finite line every four pixels."""
    del thickness  # Kept for call compatibility; overlap is centre-line evidence.
    first, second = np.asarray(p1, dtype=float), np.asarray(p2, dtype=float)
    samples = max(2, int(math.ceil(float(np.linalg.norm(second - first)) / 4.0)) + 1)
    points = np.linspace(first, second, samples)
    return _sampled_overlap(hud, points[:, 0], points[:, 1], region_labels)


def arc_hud_overlap(
    hud: HudDetection | np.ndarray,
    center: Sequence[float],
    axes: Sequence[float] | float,
    *,
    rotation_deg: float = 0.0,
    start_deg: float = 0.0,
    end_deg: float = 360.0,
    thickness: int = 3,
    region_labels: np.ndarray | None = None,
) -> dict[str, Any]:
    """Sample mask/region overlap along a circle or ellipse perimeter.

    ``axes`` contains ellipse radii (not OpenCV's full fitted diameters).  A
    scalar draws a circle.
    """
    del thickness  # Kept for call compatibility; overlap is perimeter evidence.
    if isinstance(axes, (int, float, np.number)):
        radii = (float(axes), float(axes))
    else:
        radii = tuple(float(value) for value in axes)
    # Seventy-two samples is dense enough for HUD rectangles while keeping arc
    # scoring independent of image resolution and primitive count.
    full_circle = abs(float(end_deg) - float(start_deg)) >= 360
    angles = np.deg2rad(np.linspace(start_deg, end_deg, 72, endpoint=not full_circle))
    local_x = max(1.0, radii[0]) * np.cos(angles)
    local_y = max(1.0, radii[1]) * np.sin(angles)
    rotation = math.radians(rotation_deg)
    cos_a, sin_a = math.cos(rotation), math.sin(rotation)
    xs = float(center[0]) + cos_a * local_x - sin_a * local_y
    ys = float(center[1]) + sin_a * local_x + cos_a * local_y
    return _sampled_overlap(hud, xs, ys, region_labels)


def primitive_hud_overlap(primitive: dict[str, Any], hud: HudDetection) -> dict[str, Any]:
    """Return HUD evidence for the pipeline's line/circle/arc dictionaries."""
    geometry = primitive["geometry"]
    if primitive["type"] == "line_segment":
        return line_hud_overlap(hud, geometry["p1"], geometry["p2"])
    if primitive["type"] == "circle":
        return arc_hud_overlap(hud, geometry["center"], geometry["radius_px"])
    if primitive["type"] == "arc":
        axes = [float(value) / 2 for value in geometry["axes_px"]]
        return arc_hud_overlap(hud, geometry["center"], axes, rotation_deg=geometry.get("rotation_deg", 0.0))
    return {"hud_overlap": 0.0, "hud_region_ids": []}
