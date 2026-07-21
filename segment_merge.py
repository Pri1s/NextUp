"""Deterministic, provenance-preserving merging for detected line segments.

The first pass intentionally mirrors the permissive 72 px merge used by the
original detector.  A second pass can bridge a longer occlusion, but only when
the two already-merged clusters and their tentative total-least-squares (TLS)
union satisfy stricter geometric and coverage checks.

This module deliberately has no image dependency.  Appearance-derived fields
such as ``strength`` and HUD/floor evidence are copied from one deterministic
representative only; callers should resample and replace those values over the
returned full-span geometry.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import heapq
import math
from typing import Any, Iterable

import numpy as np


STRATEGY_VERSION = "deterministic_base_extended_tls_v1"


@dataclass(frozen=True)
class _Fragment:
    source_id: str | None
    p1: tuple[float, float]
    p2: tuple[float, float]
    length_px: float

    @property
    def key(self) -> tuple[Any, ...]:
        return (
            round(self.p1[0], 8),
            round(self.p1[1], 8),
            round(self.p2[0], 8),
            round(self.p2[1], 8),
            self.source_id or "",
        )


@dataclass
class _Cluster:
    fragments: tuple[_Fragment, ...]
    source_items: tuple[dict[str, Any], ...]
    base_merge_count: int
    extended_merge_count: int
    center: np.ndarray
    direction: np.ndarray
    p1: np.ndarray
    p2: np.ndarray
    span_length_px: float
    observed_length_px: float
    observed_coverage: float
    gaps_px: tuple[float, ...]

    @property
    def angle_rad(self) -> float:
        return math.atan2(float(self.direction[1]), float(self.direction[0]))

    @property
    def midpoint(self) -> np.ndarray:
        return (self.p1 + self.p2) / 2.0

    @property
    def key(self) -> tuple[Any, ...]:
        return tuple(fragment.key for fragment in self.fragments)


def _canonical_endpoints(p1: Iterable[float], p2: Iterable[float]) -> tuple[tuple[float, float], tuple[float, float]]:
    first = tuple(float(value) for value in p1)
    second = tuple(float(value) for value in p2)
    if len(first) != 2 or len(second) != 2:
        raise ValueError("Line endpoints must contain exactly two coordinates")
    if not all(math.isfinite(value) for value in (*first, *second)):
        raise ValueError("Line endpoints must be finite")
    return (first, second) if first <= second else (second, first)


def _fragment_from_item(item: dict[str, Any]) -> _Fragment:
    try:
        p1, p2 = _canonical_endpoints(item["geometry"]["p1"], item["geometry"]["p2"])
    except (KeyError, TypeError) as error:
        raise ValueError("Each line must have geometry.p1 and geometry.p2 endpoints") from error
    return _Fragment(
        source_id=str(item["id"]) if item.get("id") is not None else None,
        p1=p1,
        p2=p2,
        length_px=math.dist(p1, p2),
    )


def _canonical_direction(direction: np.ndarray) -> np.ndarray:
    result = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(result))
    if norm <= 1e-12:
        return np.array([1.0, 0.0])
    result = result / norm
    if result[0] < -1e-12 or (abs(float(result[0])) <= 1e-12 and result[1] < 0):
        result = -result
    return result


def _fit_tls(fragments: tuple[_Fragment, ...]) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(
        [point for fragment in fragments for point in (fragment.p1, fragment.p2)],
        dtype=float,
    )
    center = np.mean(points, axis=0)
    centered = points - center
    if float(np.max(np.linalg.norm(centered, axis=1))) <= 1e-12:
        return center, np.array([1.0, 0.0])
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    return center, _canonical_direction(axes[0])


def _observed_intervals(
    fragments: tuple[_Fragment, ...], center: np.ndarray, direction: np.ndarray
) -> tuple[list[tuple[float, float]], float, tuple[float, ...]]:
    intervals = []
    for fragment in fragments:
        values = [
            float((np.asarray(point, dtype=float) - center) @ direction)
            for point in (fragment.p1, fragment.p2)
        ]
        intervals.append((min(values), max(values)))
    intervals.sort()

    union: list[list[float]] = []
    for low, high in intervals:
        if not union or low > union[-1][1] + 1e-8:
            union.append([low, high])
        else:
            union[-1][1] = max(union[-1][1], high)
    observed = sum(high - low for low, high in union)
    gaps = tuple(max(0.0, union[index + 1][0] - union[index][1]) for index in range(len(union) - 1))
    return [(low, high) for low, high in union], float(observed), gaps


def _build_cluster(
    fragments: Iterable[_Fragment],
    source_items: Iterable[dict[str, Any]],
    *,
    base_merge_count: int = 0,
    extended_merge_count: int = 0,
) -> _Cluster:
    ordered_fragments = tuple(sorted(fragments, key=lambda fragment: fragment.key))
    center, direction = _fit_tls(ordered_fragments)
    intervals, observed_length, gaps = _observed_intervals(ordered_fragments, center, direction)
    low = min(interval[0] for interval in intervals)
    high = max(interval[1] for interval in intervals)
    span_length = max(0.0, high - low)
    coverage = observed_length / span_length if span_length > 1e-12 else 1.0
    return _Cluster(
        fragments=ordered_fragments,
        source_items=tuple(source_items),
        base_merge_count=base_merge_count,
        extended_merge_count=extended_merge_count,
        center=center,
        direction=direction,
        p1=center + low * direction,
        p2=center + high * direction,
        span_length_px=span_length,
        observed_length_px=observed_length,
        observed_coverage=float(np.clip(coverage, 0.0, 1.0)),
        gaps_px=gaps,
    )


def _angle_difference(first: float, second: float) -> float:
    return abs((first - second + math.pi / 2.0) % math.pi - math.pi / 2.0)


def _point_to_line_distance(point: np.ndarray, cluster: _Cluster) -> float:
    normal = np.array([-cluster.direction[1], cluster.direction[0]])
    return abs(float((point - cluster.center) @ normal))


def _symmetric_lateral_offset(first: _Cluster, second: _Cluster) -> float:
    return max(
        _point_to_line_distance(first.midpoint, second),
        _point_to_line_distance(second.midpoint, first),
    )


def _cluster_interval(cluster: _Cluster, center: np.ndarray, direction: np.ndarray) -> tuple[float, float]:
    projections = [
        float((np.asarray(point, dtype=float) - center) @ direction)
        for fragment in cluster.fragments
        for point in (fragment.p1, fragment.p2)
    ]
    return min(projections), max(projections)


def _interval_gap(first: tuple[float, float], second: tuple[float, float]) -> float:
    if first[1] < second[0]:
        return second[0] - first[1]
    if second[1] < first[0]:
        return first[0] - second[1]
    return 0.0


def _tls_endpoint_residual(cluster: _Cluster) -> float:
    normal = np.array([-cluster.direction[1], cluster.direction[0]])
    return max(
        abs(float((np.asarray(point, dtype=float) - cluster.center) @ normal))
        for fragment in cluster.fragments
        for point in (fragment.p1, fragment.p2)
    )


def _combine(first: _Cluster, second: _Cluster, *, pass_name: str) -> _Cluster:
    return _build_cluster(
        (*first.fragments, *second.fragments),
        (*first.source_items, *second.source_items),
        base_merge_count=first.base_merge_count + second.base_merge_count + (pass_name == "base"),
        extended_merge_count=first.extended_merge_count + second.extended_merge_count + (pass_name == "extended"),
    )


def _greedy_base_merge(
    clusters: list[_Cluster],
    *,
    angle_tolerance: float,
    lateral_offset_px: float,
    max_gap_px: float,
) -> list[_Cluster]:
    """Reproduce the old longest-first merge shape without cubic rescans."""
    result: list[_Cluster] = []
    ordered = sorted(clusters, key=lambda cluster: (-cluster.span_length_px, cluster.key))
    for candidate in ordered:
        for index, present in enumerate(result):
            if _angle_difference(candidate.angle_rad, present.angle_rad) > angle_tolerance + 1e-12:
                continue
            if _symmetric_lateral_offset(candidate, present) > lateral_offset_px + 1e-12:
                continue
            # Match the legacy test by projecting the incoming fragment onto
            # the already-built cluster.  TLS is only evaluated for an accepted
            # merge, not for every possible pair in the raw Hough population.
            present_interval = _cluster_interval(present, present.center, present.direction)
            candidate_interval = _cluster_interval(candidate, present.center, present.direction)
            if _interval_gap(present_interval, candidate_interval) > max_gap_px + 1e-12:
                continue
            result[index] = _combine(present, candidate, pass_name="base")
            break
        else:
            result.append(candidate)
    return sorted(result, key=lambda cluster: cluster.key)


def _extended_candidate(
    first: _Cluster,
    second: _Cluster,
    *,
    angle_tolerance: float,
    lateral_offset_px: float,
    endpoint_residual_px: float,
    max_gap_px: float,
    minimum_coverage: float,
    max_gap_span_ratio: float,
) -> tuple[tuple[Any, ...], _Cluster] | None:
    angle_delta = _angle_difference(first.angle_rad, second.angle_rad)
    if angle_delta > angle_tolerance + 1e-12:
        return None
    if first.observed_coverage < minimum_coverage or second.observed_coverage < minimum_coverage:
        return None
    lateral_offset = _symmetric_lateral_offset(first, second)
    if lateral_offset > lateral_offset_px + 1e-12:
        return None

    combined = _combine(first, second, pass_name="extended")
    residual = _tls_endpoint_residual(combined)
    if residual > endpoint_residual_px + 1e-12:
        return None
    first_interval = _cluster_interval(first, combined.center, combined.direction)
    second_interval = _cluster_interval(second, combined.center, combined.direction)
    gap = _interval_gap(first_interval, second_interval)
    # The base pass is deliberately greedy and can leave compatible clusters
    # separate even when their gap is below its cap (for example, after a
    # different fragment was merged first).  The deterministic agglomerative
    # pass therefore considers the full [0, extended cap] range while applying
    # its stricter TLS, coverage, offset, angle, and relative-gap checks.
    if gap > max_gap_px + 1e-12:
        return None
    shorter_span = min(first_interval[1] - first_interval[0], second_interval[1] - second_interval[0])
    if shorter_span <= 1e-12 or gap / shorter_span > max_gap_span_ratio + 1e-12:
        return None
    if combined.observed_coverage < minimum_coverage:
        return None
    return (
        (
            round(gap, 10),
            round(angle_delta, 10),
            round(lateral_offset, 10),
            round(residual, 10),
            first.key,
            second.key,
        ),
        combined,
    )


def _agglomerate_extended(
    clusters: list[_Cluster],
    **candidate_options: Any,
) -> list[_Cluster]:
    """Agglomerate strict candidates with a lazy heap instead of O(n^3) rescans."""
    active = {
        identifier: cluster
        for identifier, cluster in enumerate(sorted(clusters, key=lambda cluster: cluster.key))
    }
    next_identifier = len(active)
    candidates: list[tuple[tuple[Any, ...], int, int, _Cluster]] = []

    def add_candidate(first_id: int, second_id: int) -> None:
        if first_id > second_id:
            first_id, second_id = second_id, first_id
        candidate = _extended_candidate(active[first_id], active[second_id], **candidate_options)
        if candidate is not None:
            key, combined = candidate
            heapq.heappush(candidates, (key, first_id, second_id, combined))

    identifiers = list(active)
    for first_index, first_id in enumerate(identifiers):
        for second_id in identifiers[first_index + 1 :]:
            add_candidate(first_id, second_id)

    while candidates:
        _, first_id, second_id, combined = heapq.heappop(candidates)
        if first_id not in active or second_id not in active:
            continue
        del active[first_id]
        del active[second_id]
        combined_id = next_identifier
        next_identifier += 1
        active[combined_id] = combined
        for other_id in sorted(active):
            if other_id != combined_id:
                add_candidate(other_id, combined_id)
    return sorted(active.values(), key=lambda cluster: cluster.key)


def _representative_key(item: dict[str, Any]) -> tuple[Any, ...]:
    fragment = _fragment_from_item(item)
    strength = float(item.get("strength", 0.0))
    return (-strength, -fragment.length_px, fragment.key, repr(sorted(item.get("evidence", {}).items())))


def _render_cluster(cluster: _Cluster) -> dict[str, Any]:
    representative = min(cluster.source_items, key=_representative_key)
    item = copy.deepcopy(representative)
    geometry = item.setdefault("geometry", {})
    geometry.update(
        {
            "p1": [float(cluster.p1[0]), float(cluster.p1[1])],
            "p2": [float(cluster.p2[0]), float(cluster.p2[1])],
            # Full span is intentionally used by finite-segment agreement.
            "length_px": round(cluster.span_length_px, 2),
            "observed_length_px": round(cluster.observed_length_px, 2),
            "angle_rad": round(cluster.angle_rad, 5),
        }
    )

    fragments = sorted(
        cluster.fragments,
        key=lambda fragment: min(
            float((np.asarray(point, dtype=float) - cluster.center) @ cluster.direction)
            for point in (fragment.p1, fragment.p2)
        ),
    )
    evidence = item.setdefault("evidence", {})
    evidence["raw_strength"] = round(
        max(float(source.get("evidence", {}).get("raw_strength", source.get("strength", 0.0))) for source in cluster.source_items),
        4,
    )
    if any("paint_floor_contrast" in source.get("evidence", {}) for source in cluster.source_items):
        evidence["paint_floor_contrast"] = round(
            max(float(source.get("evidence", {}).get("paint_floor_contrast", 0.0)) for source in cluster.source_items),
            4,
        )
    evidence["merged_segments"] = len(fragments)
    # Keep frequently consumed merge metrics flat for the detection/core layer;
    # the nested object below retains the complete audit trail.
    evidence["source_fragment_lengths_px"] = [round(fragment.length_px, 2) for fragment in fragments]
    evidence["observed_length_px"] = round(cluster.observed_length_px, 2)
    evidence["observed_coverage_ratio"] = round(cluster.observed_coverage, 4)
    evidence["merge_gaps_px"] = [round(gap, 2) for gap in cluster.gaps_px]
    evidence["extended_merge_count"] = cluster.extended_merge_count
    evidence["merge_provenance"] = {
        "strategy": STRATEGY_VERSION,
        "source_ids": [fragment.source_id for fragment in fragments if fragment.source_id is not None],
        "fragment_lengths_px": [round(fragment.length_px, 2) for fragment in fragments],
        "fragments": [
            {
                **({"source_id": fragment.source_id} if fragment.source_id is not None else {}),
                "p1": list(fragment.p1),
                "p2": list(fragment.p2),
                "length_px": round(fragment.length_px, 2),
            }
            for fragment in fragments
        ],
        "span_length_px": round(cluster.span_length_px, 2),
        "observed_length_px": round(cluster.observed_length_px, 2),
        "observed_coverage": round(cluster.observed_coverage, 4),
        "gaps_px": [round(gap, 2) for gap in cluster.gaps_px],
        "total_gap_px": round(sum(cluster.gaps_px), 2),
        "largest_gap_px": round(max(cluster.gaps_px, default=0.0), 2),
        "base_merge_count": cluster.base_merge_count,
        "extended_merge_count": cluster.extended_merge_count,
        "appearance_recompute_required": len(fragments) > 1,
    }
    return item


def observed_segment_length(item: dict[str, Any]) -> float:
    """Return painted support length, falling back to the legacy full span."""
    geometry = item.get("geometry", {})
    if "observed_length_px" in geometry:
        return float(geometry["observed_length_px"])
    provenance = item.get("evidence", {}).get("merge_provenance", {})
    return float(provenance.get("observed_length_px", geometry.get("length_px", 0.0)))


def merge_collinear_segments(
    items: list[dict[str, Any]],
    *,
    image_width: int,
    image_height: int,
    base_max_gap_px: float = 72.0,
    angle_tolerance_deg: float = 2.5,
    lateral_offset_px: float = 8.0,
    endpoint_residual_px: float = 8.0,
    minimum_coverage: float = 0.80,
    max_gap_span_ratio: float = 0.30,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Merge raw line dictionaries and return ``(lines, diagnostics)``.

    Output ordering is by observed painted length, not the bridged full span.
    The input dictionaries are never mutated.  The extended gap cap is
    ``clamp(0.05 * max(image dimension), 72, 144)``.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")
    if base_max_gap_px < 0 or lateral_offset_px < 0 or endpoint_residual_px < 0:
        raise ValueError("Pixel thresholds must be non-negative")
    if not 0.0 <= minimum_coverage <= 1.0:
        raise ValueError("minimum_coverage must be between zero and one")
    if max_gap_span_ratio < 0:
        raise ValueError("max_gap_span_ratio must be non-negative")

    angle_tolerance = math.radians(angle_tolerance_deg)
    extended_max_gap_px = min(144.0, max(72.0, 0.05 * max(image_width, image_height)))
    clusters = [
        _build_cluster((_fragment_from_item(item),), (item,))
        for item in items
    ]
    base_clusters = _greedy_base_merge(
        clusters,
        angle_tolerance=angle_tolerance,
        lateral_offset_px=lateral_offset_px,
        max_gap_px=base_max_gap_px,
    )
    extended_clusters = _agglomerate_extended(
        base_clusters,
        angle_tolerance=angle_tolerance,
        lateral_offset_px=lateral_offset_px,
        endpoint_residual_px=endpoint_residual_px,
        max_gap_px=extended_max_gap_px,
        minimum_coverage=minimum_coverage,
        max_gap_span_ratio=max_gap_span_ratio,
    )

    rendered = [_render_cluster(cluster) for cluster in extended_clusters]
    rendered.sort(
        key=lambda item: (
            -observed_segment_length(item),
            -float(item["geometry"]["length_px"]),
            _fragment_from_item(item).key,
        )
    )
    base_merges = sum(cluster.base_merge_count for cluster in base_clusters)
    extended_merges = sum(cluster.extended_merge_count for cluster in extended_clusters)
    diagnostics = {
        "strategy": STRATEGY_VERSION,
        "thresholds": {
            "base_max_gap_px": float(base_max_gap_px),
            "extended_max_gap_px": round(extended_max_gap_px, 2),
            "angle_tolerance_deg": float(angle_tolerance_deg),
            "symmetric_lateral_offset_px": float(lateral_offset_px),
            "tls_endpoint_residual_px": float(endpoint_residual_px),
            "minimum_cluster_and_union_coverage": float(minimum_coverage),
            "max_gap_to_min_span_ratio": float(max_gap_span_ratio),
        },
        "raw_count": len(items),
        "base_cluster_count": len(base_clusters),
        "extended_cluster_count": len(extended_clusters),
        "base_merge_count": base_merges,
        "extended_merge_count": extended_merges,
    }
    return rendered, diagnostics


__all__ = ["STRATEGY_VERSION", "merge_collinear_segments", "observed_segment_length"]
