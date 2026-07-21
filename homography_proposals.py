"""Guided, unlabeled four-line proposals for court homography estimation.

The scorer in :mod:`court_homography` deliberately asks finite detected segments
to support a candidate homography.  This module improves the hypotheses handed
to that scorer: intersection adjacency and local orientation families replace
independent random line draws and arbitrary four-line permutations.

The module is intentionally independent of the solver.  A proposal contains the
same template and detected line dictionaries supplied by the caller, aligned in
correspondence order, so it can be passed directly to the existing line-based
homography estimator.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
import hashlib
import itertools
import math
from typing import Any


U_LEG_ANGLE_DEG = 12.0
MIN_CROSS_ANGLE_DEG = 12.0
LOCAL_FAMILY_ANGLE_DEG = 25.0
STRATEGY_VERSION = "guided_orientation_adjacency_v1"


def _geometry(line: dict[str, Any]) -> dict[str, Any]:
    return line.get("geometry", line)


def _points(line: dict[str, Any]) -> tuple[tuple[float, float], tuple[float, float]]:
    geometry = _geometry(line)
    p1, p2 = geometry["p1"], geometry["p2"]
    return (float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))


def _line_angle(line: dict[str, Any]) -> float:
    geometry = _geometry(line)
    if "angle_rad" in geometry:
        return float(geometry["angle_rad"]) % math.pi
    p1, p2 = _points(line)
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0]) % math.pi


def _line_length(line: dict[str, Any]) -> float:
    """Return the full endpoint span used by geometric adjacency checks."""
    geometry = _geometry(line)
    if "length_px" in geometry:
        return float(geometry["length_px"])
    p1, p2 = _points(line)
    return math.dist(p1, p2)


def _observed_line_length(line: dict[str, Any]) -> float:
    """Return painted support length for ranking, falling back to full span."""
    geometry = _geometry(line)
    if "observed_length_px" in geometry:
        return float(geometry["observed_length_px"])
    provenance = line.get("evidence", {}).get("merge_provenance", {})
    if "observed_length_px" in provenance:
        return float(provenance["observed_length_px"])
    return _line_length(line)


def _angle_difference(first: float, second: float) -> float:
    return abs((first - second + math.pi / 2.0) % math.pi - math.pi / 2.0)


def _line_equation(line: dict[str, Any]) -> tuple[float, float, float] | None:
    p1, p2 = _points(line)
    a, b = p1[1] - p2[1], p2[0] - p1[0]
    norm = math.hypot(a, b)
    if norm <= 1e-10:
        return None
    return a / norm, b / norm, (p1[0] * p2[1] - p2[0] * p1[1]) / norm


def _intersection(first: dict[str, Any], second: dict[str, Any]) -> tuple[float, float] | None:
    a = _line_equation(first)
    b = _line_equation(second)
    if a is None or b is None:
        return None
    determinant = a[0] * b[1] - b[0] * a[1]
    if abs(determinant) <= 1e-8:
        return None
    return (
        (a[1] * b[2] - b[1] * a[2]) / determinant,
        (a[2] * b[0] - b[2] * a[0]) / determinant,
    )


def _point_to_segment_distance(
    point: tuple[float, float], line: dict[str, Any]
) -> float:
    p1, p2 = _points(line)
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        return math.dist(point, p1)
    scale = max(0.0, min(1.0, ((point[0] - p1[0]) * dx + (point[1] - p1[1]) * dy) / length_sq))
    return math.dist(point, (p1[0] + scale * dx, p1[1] + scale * dy))


def _nearest_endpoint(
    point: tuple[float, float], line: dict[str, Any]
) -> tuple[int, float]:
    p1, p2 = _points(line)
    distances = (math.dist(point, p1), math.dist(point, p2))
    index = 0 if distances[0] <= distances[1] else 1
    return index, distances[index]


def _circular_mean(angles: Iterable[float]) -> float:
    angles = list(angles)
    x = sum(math.cos(2.0 * angle) for angle in angles)
    y = sum(math.sin(2.0 * angle) for angle in angles)
    if abs(x) + abs(y) <= 1e-12:
        return min(angles, default=0.0) % math.pi
    return (math.atan2(y, x) / 2.0) % math.pi


def _identity(value: Any) -> str:
    if isinstance(value, dict):
        if "key" in value:
            return str(value["key"])
        if "id" in value:
            return str(value["id"])
    return str(value)


def _tie_digest(random_seed: int, identity: Any) -> str:
    """Seeded ordering used only after the geometric/evidence rank ties."""
    return hashlib.sha256(f"{random_seed}|{_identity(identity)}".encode("utf-8")).hexdigest()


def _ranked(items: Iterable[Any], random_seed: int) -> list[Any]:
    return sorted(
        items,
        key=lambda item: (
            -float(item.get("rank", 0.0)),
            _tie_digest(random_seed, item),
            _identity(item),
        ),
    )


def _fair_product(first: list[Any], second: list[Any]) -> Iterator[tuple[Any, Any]]:
    """Traverse ranked pairs in expanding rows/columns.

    A pure nested product can starve every detected fourth line after the first;
    a diagonal product can postpone a low-ranked template identity even when the
    best detected fourth line is correct.  Expanding L-shaped layers first try
    every template identity with the strongest detection, then every detection
    with the strongest template, before moving both ranks down together.
    """
    if not first or not second:
        return
    for layer in range(max(len(first), len(second))):
        if layer < len(first):
            for second_index in range(layer, len(second)):
                yield first[layer], second[second_index]
        if layer < len(second):
            for first_index in range(layer + 1, len(first)):
                yield first[first_index], second[layer]


def derive_template_orientation_families(template: dict[str, Any]) -> dict[str, Any]:
    """Cluster template lines into their two geometric court-axis families.

    Angles are represented on the doubled-angle unit circle, where undirected
    lines at ``theta`` and ``theta + pi`` are identical.  The farthest observed
    direction seeds the second cluster; no line names or schema-specific kinds
    are used.
    """
    lines = list(template.get("lines", []))
    if len(lines) < 2:
        raise ValueError("At least two template lines are required to derive orientation families")
    malformed = [line.get("id", "<unknown>") for line in lines if _line_length(line) <= 1e-10]
    if malformed:
        raise ValueError(f"Degenerate template line(s): {', '.join(map(str, malformed))}")

    ordered = sorted(lines, key=lambda line: str(line["id"]))
    vectors = {
        str(line["id"]): (math.cos(2.0 * _line_angle(line)), math.sin(2.0 * _line_angle(line)))
        for line in ordered
    }
    first = ordered[0]
    first_vector = vectors[str(first["id"])]
    second = max(
        ordered[1:],
        key=lambda line: (
            (vectors[str(line["id"])][0] - first_vector[0]) ** 2
            + (vectors[str(line["id"])][1] - first_vector[1]) ** 2,
            str(line["id"]),
        ),
    )
    centers = [first_vector, vectors[str(second["id"])]]
    assignments: dict[str, int] = {}
    for _ in range(20):
        next_assignments: dict[str, int] = {}
        for line in ordered:
            identifier = str(line["id"])
            vector = vectors[identifier]
            distances = [
                (vector[0] - center[0]) ** 2 + (vector[1] - center[1]) ** 2
                for center in centers
            ]
            next_assignments[identifier] = 0 if distances[0] <= distances[1] else 1
        if len(set(next_assignments.values())) < 2:
            raise ValueError("Template lines do not contain two distinct orientation families")
        next_centers: list[tuple[float, float]] = []
        for family_index in range(2):
            members = [vectors[key] for key, family in next_assignments.items() if family == family_index]
            x, y = sum(value[0] for value in members), sum(value[1] for value in members)
            norm = math.hypot(x, y)
            next_centers.append((x / norm, y / norm) if norm else centers[family_index])
        if next_assignments == assignments:
            centers = next_centers
            break
        assignments, centers = next_assignments, next_centers

    unsorted_families = []
    for family_index in range(2):
        members = [line for line in ordered if assignments[str(line["id"])] == family_index]
        angle = _circular_mean(_line_angle(line) for line in members)
        unsorted_families.append({"angle_rad": angle, "lines": members})
    unsorted_families.sort(key=lambda family: (family["angle_rad"], str(family["lines"][0]["id"])))

    line_to_family: dict[str, int] = {}
    summaries: list[dict[str, Any]] = []
    for family_index, family in enumerate(unsorted_families):
        line_ids = [str(line["id"]) for line in family["lines"]]
        line_to_family.update({identifier: family_index for identifier in line_ids})
        summaries.append(
            {
                "family": family_index,
                "angle_rad": round(float(family["angle_rad"]), 8),
                "line_ids": line_ids,
            }
        )
    separation = _angle_difference(summaries[0]["angle_rad"], summaries[1]["angle_rad"])
    if separation < math.radians(MIN_CROSS_ANGLE_DEG):
        raise ValueError("Template orientation families are not geometrically distinct")
    return {"line_to_family": line_to_family, "families": summaries}


def _detected_lines_and_intersections(
    detection_or_primitives: dict[str, Any] | list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if isinstance(detection_or_primitives, dict):
        primitives = list(detection_or_primitives.get("primitives", []))
    else:
        primitives = list(detection_or_primitives)
    return (
        [item for item in primitives if item.get("type") == "line_segment"],
        [item for item in primitives if item.get("type") == "intersection"],
    )


def _validated_detected_lines(
    lines: list[dict[str, Any]], stats: dict[str, int] | None = None
) -> list[dict[str, Any]]:
    """Filter malformed primitives once so proposal diagnostics remain usable."""
    valid: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for line in lines:
        try:
            identifier_value = line.get("id")
            if identifier_value is None or str(identifier_value) == "":
                raise ValueError("missing line id")
            identifier = str(identifier_value)
            if identifier in identifiers:
                raise ValueError("duplicate line id")
            p1, p2 = _points(line)
            coordinates = (*p1, *p2)
            if not all(math.isfinite(value) for value in coordinates) or math.dist(p1, p2) <= 1e-10:
                raise ValueError("invalid endpoints")
            angle = _line_angle(line)
            span_length = _line_length(line)
            observed_length = _observed_line_length(line)
            strength = float(line.get("strength", 0.0))
            if (
                not all(math.isfinite(value) for value in (angle, span_length, observed_length, strength))
                or span_length <= 0.0
                or observed_length <= 0.0
            ):
                raise ValueError("non-finite line evidence")
        except (KeyError, TypeError, ValueError, OverflowError):
            if stats is not None:
                stats["invalid_candidates"] += 1
            continue
        identifiers.add(identifier)
        valid.append(line)
    return valid


def _valid_corners(
    lines: list[dict[str, Any]],
    intersections: list[dict[str, Any]],
    stats: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    lines_by_id = {str(line.get("id")): line for line in lines if line.get("id") is not None}
    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for intersection in intersections:
        geometry = intersection.get("geometry", {})
        identifiers = geometry.get("line_ids", [])
        point = geometry.get("point")
        if (
            len(identifiers) != 2
            or str(identifiers[0]) == str(identifiers[1])
            or any(str(identifier) not in lines_by_id for identifier in identifiers)
            or not isinstance(point, (list, tuple))
            or len(point) != 2
        ):
            if stats is not None:
                stats["invalid_candidates"] += 1
            continue
        try:
            point_tuple = (float(point[0]), float(point[1]))
        except (TypeError, ValueError):
            if stats is not None:
                stats["invalid_candidates"] += 1
            continue
        if not all(math.isfinite(value) for value in point_tuple):
            if stats is not None:
                stats["invalid_candidates"] += 1
            continue
        pair = tuple(sorted((str(identifiers[0]), str(identifiers[1]))))
        first, second = lines_by_id[pair[0]], lines_by_id[pair[1]]
        if _angle_difference(_line_angle(first), _line_angle(second)) < math.radians(MIN_CROSS_ANGLE_DEG):
            if stats is not None:
                stats["orientation_rejected_candidates"] += 1
            continue
        try:
            intersection_strength = float(intersection.get("strength", 0.0))
        except (TypeError, ValueError, OverflowError):
            if stats is not None:
                stats["invalid_candidates"] += 1
            continue
        if not math.isfinite(intersection_strength):
            if stats is not None:
                stats["invalid_candidates"] += 1
            continue
        candidate = {
            "id": str(intersection.get("id", f"{pair[0]}:{pair[1]}")),
            "key": f"{pair[0]}:{pair[1]}",
            "line_ids": pair,
            "point": point_tuple,
            "rank": intersection_strength,
        }
        present = by_pair.get(pair)
        if present is None or candidate["rank"] > present["rank"]:
            by_pair[pair] = candidate
    return list(by_pair.values())


def discover_u_motifs(
    detection_or_primitives: dict[str, Any] | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return connected detected U motifs using endpoint-local intersections.

    The cap must meet two similarly oriented legs near opposite cap endpoints.
    Each crossing must also lie near an endpoint of its leg.  The thresholds are
    deliberately expressed in image pixels because these are detected segments.
    """
    lines, intersections = _detected_lines_and_intersections(detection_or_primitives)
    lines = _validated_detected_lines(lines)
    return _discover_u_motifs(lines, intersections)


def _discover_u_motifs(
    lines: list[dict[str, Any]],
    intersections: list[dict[str, Any]],
    stats: dict[str, int] | None = None,
    corners: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    lines_by_id = {str(line["id"]): line for line in lines}
    if corners is None:
        corners = _valid_corners(lines, intersections, stats)
    adjacent: dict[str, list[dict[str, Any]]] = {identifier: [] for identifier in lines_by_id}
    for corner in corners:
        for cap_id in corner["line_ids"]:
            other_id = corner["line_ids"][1] if corner["line_ids"][0] == cap_id else corner["line_ids"][0]
            adjacent[cap_id].append({**corner, "other_id": other_id})

    motifs_by_key: dict[tuple[str, tuple[str, str]], dict[str, Any]] = {}
    for cap_id, cap_corners in adjacent.items():
        cap = lines_by_id[cap_id]
        cap_endpoint_limit = max(24.0, 0.10 * _line_length(cap))
        for first, second in itertools.combinations(cap_corners, 2):
            if first["other_id"] == second["other_id"]:
                continue
            leg_a, leg_b = lines_by_id[first["other_id"]], lines_by_id[second["other_id"]]
            if _angle_difference(_line_angle(leg_a), _line_angle(leg_b)) > math.radians(U_LEG_ANGLE_DEG):
                if stats is not None:
                    stats["orientation_rejected_candidates"] += 1
                continue
            if min(
                _angle_difference(_line_angle(cap), _line_angle(leg_a)),
                _angle_difference(_line_angle(cap), _line_angle(leg_b)),
            ) < math.radians(MIN_CROSS_ANGLE_DEG):
                if stats is not None:
                    stats["orientation_rejected_candidates"] += 1
                continue
            cap_endpoint_a, cap_distance_a = _nearest_endpoint(first["point"], cap)
            cap_endpoint_b, cap_distance_b = _nearest_endpoint(second["point"], cap)
            if (
                cap_endpoint_a == cap_endpoint_b
                or cap_distance_a > cap_endpoint_limit
                or cap_distance_b > cap_endpoint_limit
            ):
                continue
            _, leg_distance_a = _nearest_endpoint(first["point"], leg_a)
            _, leg_distance_b = _nearest_endpoint(second["point"], leg_b)
            if leg_distance_a > max(24.0, 0.20 * _line_length(leg_a)):
                continue
            if leg_distance_b > max(24.0, 0.20 * _line_length(leg_b)):
                continue
            ordered_legs = tuple(sorted((str(leg_a["id"]), str(leg_b["id"]))))
            key = (cap_id, ordered_legs)
            rank = (
                float(cap.get("strength", 0.0))
                + float(leg_a.get("strength", 0.0))
                + float(leg_b.get("strength", 0.0))
                + float(first["rank"])
                + float(second["rank"])
            )
            motif = {
                "id": f"u:{cap_id}:{ordered_legs[0]}:{ordered_legs[1]}",
                "key": f"u:{cap_id}:{ordered_legs[0]}:{ordered_legs[1]}",
                "cap_id": cap_id,
                "leg_ids": list(ordered_legs),
                "intersection_ids": [first["id"], second["id"]],
                "cap_angle_rad": _line_angle(cap),
                "leg_angle_rad": _circular_mean((_line_angle(leg_a), _line_angle(leg_b))),
                "rank": rank,
            }
            if key not in motifs_by_key or rank > motifs_by_key[key]["rank"]:
                motifs_by_key[key] = motif
    return list(motifs_by_key.values())


def _discover_template_u_motifs(
    template_lines: list[dict[str, Any]], line_to_family: dict[str, int]
) -> list[dict[str, Any]]:
    by_family = {
        family: [line for line in template_lines if line_to_family[str(line["id"])] == family]
        for family in (0, 1)
    }
    motifs: list[dict[str, Any]] = []
    for cap_family in (0, 1):
        for cap in by_family[cap_family]:
            cap_length = _line_length(cap)
            cap_limit = max(1e-6, cap_length * 1e-4)
            for leg_a, leg_b in itertools.combinations(by_family[1 - cap_family], 2):
                point_a, point_b = _intersection(cap, leg_a), _intersection(cap, leg_b)
                if point_a is None or point_b is None:
                    continue
                endpoint_a, cap_distance_a = _nearest_endpoint(point_a, cap)
                endpoint_b, cap_distance_b = _nearest_endpoint(point_b, cap)
                leg_limit_a = max(1e-6, _line_length(leg_a) * 1e-4)
                leg_limit_b = max(1e-6, _line_length(leg_b) * 1e-4)
                if (
                    endpoint_a == endpoint_b
                    or cap_distance_a > cap_limit
                    or cap_distance_b > cap_limit
                    or _nearest_endpoint(point_a, leg_a)[1] > leg_limit_a
                    or _nearest_endpoint(point_b, leg_b)[1] > leg_limit_b
                ):
                    continue
                leg_ids = sorted((str(leg_a["id"]), str(leg_b["id"])))
                motifs.append(
                    {
                        "id": f"template_u:{cap['id']}:{leg_ids[0]}:{leg_ids[1]}",
                        "key": f"template_u:{cap['id']}:{leg_ids[0]}:{leg_ids[1]}",
                        "cap": cap,
                        "legs": [next(line for line in (leg_a, leg_b) if str(line["id"]) == identifier) for identifier in leg_ids],
                        "cap_family": cap_family,
                        "rank": cap_length + _line_length(leg_a) + _line_length(leg_b),
                    }
                )
    return motifs


def _line_rank(line: dict[str, Any], max_dimension: float) -> float:
    return float(line.get("strength", 0.0)) * (
        1.0 + _observed_line_length(line) / max(1.0, max_dimension)
    )


def _closes_legs_at_opposite_ends(
    cap: dict[str, Any],
    fourth: dict[str, Any],
    legs: list[dict[str, Any]],
    *,
    pixel_tolerances: bool,
) -> bool:
    """Whether a fourth cap closes both legs opposite the anchor cap.

    A duplicate detection of the anchor cap intersects the same leg endpoints
    and must not receive the strong connected-fourth ranking boost.  This
    distinction is important in broadcast footage, where both paint edges (or
    several Hough fits of one edge) are often retained as separate primitives.
    """
    for leg in legs:
        cap_point = _intersection(cap, leg)
        fourth_point = _intersection(fourth, leg)
        if cap_point is None or fourth_point is None:
            return False
        cap_endpoint, cap_distance = _nearest_endpoint(cap_point, leg)
        fourth_endpoint, fourth_distance = _nearest_endpoint(fourth_point, leg)
        if cap_endpoint == fourth_endpoint:
            return False
        if pixel_tolerances:
            leg_limit = max(24.0, 0.20 * _line_length(leg))
            fourth_limit = max(24.0, 0.10 * _line_length(fourth))
        else:
            leg_limit = max(1e-6, 1e-4 * _line_length(leg))
            fourth_limit = max(1e-6, 1e-4 * _line_length(fourth))
        if cap_distance > leg_limit or fourth_distance > leg_limit:
            return False
        if _point_to_segment_distance(fourth_point, fourth) > fourth_limit:
            return False
    return True


def _compatible_lines(
    lines: list[dict[str, Any]], anchor: dict[str, Any], excluded: set[str]
) -> list[dict[str, Any]]:
    threshold = math.radians(LOCAL_FAMILY_ANGLE_DEG)
    return [
        line
        for line in lines
        if str(line["id"]) not in excluded
        and _angle_difference(_line_angle(line), _line_angle(anchor)) <= threshold
    ]


def _template_quartets(
    template_lines: list[dict[str, Any]],
    line_to_family: dict[str, int],
) -> list[dict[str, Any]]:
    by_family = {
        family: [line for line in template_lines if line_to_family[str(line["id"])] == family]
        for family in (0, 1)
    }
    output: list[dict[str, Any]] = []
    for pair_a in itertools.combinations(by_family[0], 2):
        for pair_b in itertools.combinations(by_family[1], 2):
            corner_count = 0
            for first in pair_a:
                for second in pair_b:
                    point = _intersection(first, second)
                    if point is None:
                        continue
                    tolerance = max(1e-5, 1e-4 * max(_line_length(first), _line_length(second)))
                    if (
                        _point_to_segment_distance(point, first) <= tolerance
                        and _point_to_segment_distance(point, second) <= tolerance
                    ):
                        corner_count += 1
            ids = tuple(sorted(str(line["id"]) for line in (*pair_a, *pair_b)))
            output.append(
                {
                    "id": f"template_quad:{':'.join(ids)}",
                    "key": f"template_quad:{':'.join(ids)}",
                    "family_a": list(pair_a),
                    "family_b": list(pair_b),
                    "corner_count": corner_count,
                    "rank": 1000.0 * corner_count + sum(_line_length(line) for line in (*pair_a, *pair_b)),
                }
            )
    return output


def _mapping_variants(
    template_a: list[dict[str, Any]],
    template_b: list[dict[str, Any]],
    detected_a: list[dict[str, Any]],
    detected_b: list[dict[str, Any]],
    *,
    proposal_type: str,
    source: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    """Enumerate all and only within-bucket bijections (2! x 2!)."""
    if not all(len(group) == 2 for group in (template_a, template_b, detected_a, detected_b)):
        return
    for detected_a_order in (detected_a, list(reversed(detected_a))):
        for detected_b_order in (detected_b, list(reversed(detected_b))):
            mappings = list(zip(template_a + template_b, detected_a_order + detected_b_order))
            mappings.sort(key=lambda pair: str(pair[0]["id"]))
            selected_template = [pair[0] for pair in mappings]
            selected_detected = [pair[1] for pair in mappings]
            yield {
                "template_lines": selected_template,
                "detected_lines": selected_detected,
                "seed_pairs": {
                    str(template_line["id"]): str(detected_line["id"])
                    for template_line, detected_line in mappings
                },
                "proposal_type": proposal_type,
                "source": source,
            }


@dataclass
class _Source:
    key: str
    rank: float
    proposal_type: str
    iterator: Iterator[dict[str, Any]]


def _motif_sources(
    detected_motifs: list[dict[str, Any]],
    template_motifs: list[dict[str, Any]],
    lines: list[dict[str, Any]],
    template_lines: list[dict[str, Any]],
    template_line_to_family: dict[str, int],
    max_dimension: float,
    random_seed: int,
) -> list[_Source]:
    lines_by_id = {str(line["id"]): line for line in lines}
    sources: list[_Source] = []
    for motif in detected_motifs:
        cap = lines_by_id[motif["cap_id"]]
        legs = [lines_by_id[identifier] for identifier in motif["leg_ids"]]
        fourth_lines = _compatible_lines(lines, cap, {motif["cap_id"], *motif["leg_ids"]})
        fourth_items = [
            {
                "id": str(line["id"]),
                "key": str(line["id"]),
                "line": line,
                # Closing the same connected leg pair is highly informative,
                # but remains a ranking boost rather than an eligibility gate.
                "rank": _line_rank(line, max_dimension)
                + (
                    1000.0
                    if _closes_legs_at_opposite_ends(
                        cap, line, legs, pixel_tolerances=True
                    )
                    else 0.0
                ),
            }
            for line in fourth_lines
        ]
        fourth_items = _ranked(fourth_items, random_seed)
        if not fourth_items:
            continue

        template_options: list[dict[str, Any]] = []
        for template_motif in template_motifs:
            excluded = {
                str(template_motif["cap"]["id"]),
                *(str(line["id"]) for line in template_motif["legs"]),
            }
            for fourth in template_lines:
                if (
                    str(fourth["id"]) not in excluded
                    and template_line_to_family[str(fourth["id"])] == template_motif["cap_family"]
                ):
                    closes_legs = _closes_legs_at_opposite_ends(
                        template_motif["cap"],
                        fourth,
                        template_motif["legs"],
                        pixel_tolerances=False,
                    )
                    key = f"{template_motif['id']}:{fourth['id']}"
                    template_options.append(
                        {
                            "id": key,
                            "key": key,
                            "motif": template_motif,
                            "fourth": fourth,
                            "rank": float(template_motif["rank"])
                            + _line_length(fourth)
                            + (10_000.0 if closes_legs else 0.0),
                        }
                    )
        template_options = _ranked(template_options, random_seed)
        if not template_options:
            continue

        def iterator(
            motif: dict[str, Any] = motif,
            cap: dict[str, Any] = cap,
            legs: list[dict[str, Any]] = legs,
            fourth_items: list[dict[str, Any]] = fourth_items,
            template_options: list[dict[str, Any]] = template_options,
        ) -> Iterator[dict[str, Any]]:
            for fourth_item, option in _fair_product(fourth_items, template_options):
                template_motif = option["motif"]
                source = {
                    "detected_anchor": motif["id"],
                    "template_anchor": template_motif["id"],
                    "detected_fourth": str(fourth_item["line"]["id"]),
                    "template_fourth": str(option["fourth"]["id"]),
                }
                yield from _mapping_variants(
                    [template_motif["cap"], option["fourth"]],
                    template_motif["legs"],
                    [cap, fourth_item["line"]],
                    legs,
                    proposal_type="u_motif",
                    source=source,
                )

        sources.append(
            _Source(
                key=str(motif["id"]),
                rank=float(motif["rank"]),
                proposal_type="u_motif",
                iterator=iterator(),
            )
        )
    return sources


def _classify_against_corner(
    line: dict[str, Any], first: dict[str, Any], second: dict[str, Any]
) -> tuple[bool, bool]:
    threshold = math.radians(LOCAL_FAMILY_ANGLE_DEG)
    first_difference = _angle_difference(_line_angle(line), _line_angle(first))
    second_difference = _angle_difference(_line_angle(line), _line_angle(second))
    # A perspective family can fan out, but a line is assigned to the nearer
    # local direction. Exact ties are kept in both buckets and deduplicated later.
    return (
        first_difference <= threshold and first_difference <= second_difference,
        second_difference <= threshold and second_difference <= first_difference,
    )


def _detected_two_corner_quartets(
    corners: list[dict[str, Any]],
    lines: list[dict[str, Any]],
    max_dimension: float,
    stats: dict[str, int],
) -> list[dict[str, Any]]:
    lines_by_id = {str(line["id"]): line for line in lines}
    by_key: dict[tuple[tuple[str, str], tuple[str, str]], dict[str, Any]] = {}
    for first_corner, second_corner in itertools.combinations(corners, 2):
        first_ids, second_ids = set(first_corner["line_ids"]), set(second_corner["line_ids"])
        union = first_ids | second_ids
        buckets: list[tuple[list[str], list[str]]] = []
        if len(union) == 4:
            anchor_a_id, anchor_b_id = first_corner["line_ids"]
            remaining = list(second_corner["line_ids"])
            for order in (remaining, list(reversed(remaining))):
                if (
                    _angle_difference(_line_angle(lines_by_id[anchor_a_id]), _line_angle(lines_by_id[order[0]]))
                    <= math.radians(LOCAL_FAMILY_ANGLE_DEG)
                    and _angle_difference(_line_angle(lines_by_id[anchor_b_id]), _line_angle(lines_by_id[order[1]]))
                    <= math.radians(LOCAL_FAMILY_ANGLE_DEG)
                ):
                    buckets.append(([anchor_a_id, order[0]], [anchor_b_id, order[1]]))
                else:
                    stats["orientation_rejected_candidates"] += 1
        elif len(union) == 3:
            shared = next(iter(first_ids & second_ids), None)
            if shared is None:
                continue
            others = sorted(union - {shared})
            # This is an incomplete two-corner chain, not a completed U motif.
            # Its two non-shared lines are samples of one projective family at
            # different image locations and may fan more than the U motif's
            # strict 12-degree leg-consistency limit.  Keep them in the local
            # 25-degree bucket used by the ordered fallbacks.
            if (
                _angle_difference(_line_angle(lines_by_id[others[0]]), _line_angle(lines_by_id[others[1]]))
                > math.radians(LOCAL_FAMILY_ANGLE_DEG)
            ):
                stats["orientation_rejected_candidates"] += 1
                continue
            for fourth in _compatible_lines(lines, lines_by_id[shared], union):
                buckets.append(([shared, str(fourth["id"])], others))
        for bucket_a_ids, bucket_b_ids in buckets:
            if len(set((*bucket_a_ids, *bucket_b_ids))) != 4:
                stats["invalid_candidates"] += 1
                continue
            normalized = tuple(
                sorted(
                    (tuple(sorted(bucket_a_ids)), tuple(sorted(bucket_b_ids)))
                )
            )
            rank = (
                float(first_corner["rank"])
                + float(second_corner["rank"])
                + sum(_line_rank(lines_by_id[identifier], max_dimension) for identifier in union)
            )
            candidate = {
                "id": f"two_corner:{first_corner['id']}:{second_corner['id']}:{normalized}",
                "key": f"two_corner:{first_corner['key']}:{second_corner['key']}:{normalized}",
                "family_a": [lines_by_id[identifier] for identifier in bucket_a_ids],
                "family_b": [lines_by_id[identifier] for identifier in bucket_b_ids],
                "corner_ids": [first_corner["id"], second_corner["id"]],
                "rank": rank,
            }
            present = by_key.get(normalized)
            if present is None or candidate["rank"] > present["rank"]:
                by_key[normalized] = candidate
    # Preserve every corner supported by a quartet. A three-corner chain is
    # materially more informative than the arbitrary corner pair that happened
    # to create it first, and deserves topology-preserving variants near the
    # front of the bounded fallback budget.
    for candidate in by_key.values():
        quartet_ids = {
            str(line["id"])
            for line in (*candidate["family_a"], *candidate["family_b"])
        }
        corner_pairs = [
            {
                "line_ids": tuple(str(value) for value in corner["line_ids"]),
                "point": tuple(float(value) for value in corner["point"]),
                "rank": float(corner["rank"]),
            }
            for corner in corners
            if set(corner["line_ids"]).issubset(quartet_ids)
        ]
        corner_pairs.sort(key=lambda corner: (-corner["rank"], corner["line_ids"]))
        candidate["corner_pairs"] = corner_pairs
        candidate["corner_count"] = len(corner_pairs)
        candidate["rank"] = (
            1000.0 * len(corner_pairs)
            + sum(corner["rank"] for corner in corner_pairs)
            + sum(_line_rank(lines_by_id[identifier], max_dimension) for identifier in quartet_ids)
        )
    return list(by_key.values())


def _quartet_source_iterator(
    detected_a: list[dict[str, Any]],
    detected_b: list[dict[str, Any]],
    template_options: list[dict[str, Any]],
    proposal_type: str,
    detected_anchor: str,
) -> Iterator[dict[str, Any]]:
    for option in template_options:
        for axis_swap, (first_bucket, second_bucket) in enumerate(
            ((detected_a, detected_b), (detected_b, detected_a))
        ):
            source = {
                "detected_anchor": detected_anchor,
                "template_anchor": option["id"],
                "axis_swap": bool(axis_swap),
            }
            yield from _mapping_variants(
                option["family_a"],
                option["family_b"],
                first_bucket,
                second_bucket,
                proposal_type=proposal_type,
                source=source,
            )


def _finite_template_corner_point(
    first: dict[str, Any], second: dict[str, Any]
) -> tuple[float, float] | None:
    point = _intersection(first, second)
    if point is None:
        return None
    tolerance = max(1e-5, 1e-4 * max(_line_length(first), _line_length(second)))
    if (
        _point_to_segment_distance(point, first) > tolerance
        or _point_to_segment_distance(point, second) > tolerance
    ):
        return None
    return point


def _multi_corner_topology_sources(
    detected_quartets: list[dict[str, Any]],
    template_options: list[dict[str, Any]],
    random_seed: int,
    *,
    variants_per_quartet: int = 8,
) -> list[_Source]:
    """Prioritize a bounded set of mappings that preserve 3+ observed corners.

    These are still ordinary two-corner fallbacks and still enumerate 2+2
    orientation families. Splitting the strongest mappings into one-shot
    sources lets round-robin scheduling retain east/west and leg-swap variants
    instead of spending a source's sole budget slot on an unrelated template
    rectangle.
    """
    sources: list[_Source] = []
    template_by_id = {
        str(line["id"]): line
        for option in template_options
        for line in (*option["family_a"], *option["family_b"])
    }
    for quartet in detected_quartets:
        detected_corners = quartet.get("corner_pairs", [])
        if len(detected_corners) < 3:
            continue
        detected_by_id = {
            str(line["id"]): line
            for line in (*quartet["family_a"], *quartet["family_b"])
        }
        ranked: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        seen: set[tuple[tuple[str, str], ...]] = set()
        for option in template_options:
            if int(option["corner_count"]) < len(detected_corners):
                continue
            for axis_swap, (first_bucket, second_bucket) in enumerate(
                (
                    (quartet["family_a"], quartet["family_b"]),
                    (quartet["family_b"], quartet["family_a"]),
                )
            ):
                source = {
                    "detected_anchor": str(quartet["id"]),
                    "template_anchor": str(option["id"]),
                    "axis_swap": bool(axis_swap),
                    "detected_corner_count": len(detected_corners),
                    "topology_guided": True,
                }
                for proposal in _mapping_variants(
                    option["family_a"],
                    option["family_b"],
                    first_bucket,
                    second_bucket,
                    proposal_type="two_corner",
                    source=source,
                ):
                    canonical = _canonical_key(proposal)
                    if canonical is None or canonical in seen:
                        continue
                    template_for_detected = {
                        str(detected_id): template_by_id[str(template_id)]
                        for template_id, detected_id in proposal["seed_pairs"].items()
                    }
                    endpoint_residual = 0.0
                    for corner in detected_corners:
                        first_id, second_id = corner["line_ids"]
                        first_template = template_for_detected[first_id]
                        second_template = template_for_detected[second_id]
                        template_point = _finite_template_corner_point(
                            first_template, second_template
                        )
                        if template_point is None:
                            break
                        for detected_id, template_line in (
                            (first_id, first_template),
                            (second_id, second_template),
                        ):
                            detected_ratio = _nearest_endpoint(
                                corner["point"], detected_by_id[detected_id]
                            )[1] / max(_line_length(detected_by_id[detected_id]), 1e-9)
                            template_ratio = _nearest_endpoint(
                                template_point, template_line
                            )[1] / max(_line_length(template_line), 1e-9)
                            endpoint_residual += abs(detected_ratio - template_ratio)
                    else:
                        seen.add(canonical)
                        rank_key = (
                            abs(int(option["corner_count"]) - len(detected_corners)),
                            round(endpoint_residual, 9),
                            -float(option["rank"]),
                            _tie_digest(random_seed, canonical),
                        )
                        ranked.append((rank_key, proposal))
        for topology_rank, (_, proposal) in enumerate(
            sorted(ranked, key=lambda item: item[0])[:variants_per_quartet]
        ):
            proposal["source"]["topology_rank"] = topology_rank
            sources.append(
                _Source(
                    key=f"{quartet['key']}:topology:{topology_rank}",
                    rank=float(quartet["rank"]) + 10_000.0 - topology_rank * .001,
                    proposal_type="two_corner",
                    iterator=iter((proposal,)),
                )
            )
    return sources


def _two_corner_sources(
    detected_quartets: list[dict[str, Any]],
    template_quartets: list[dict[str, Any]],
    random_seed: int,
) -> list[_Source]:
    options = _ranked(
        [option for option in template_quartets if int(option["corner_count"]) >= 2],
        random_seed,
    )
    if not options:
        return []
    topology_sources = _multi_corner_topology_sources(
        detected_quartets, options, random_seed
    )
    ordinary_sources = [
        _Source(
            key=str(quartet["key"]),
            rank=float(quartet["rank"]),
            proposal_type="two_corner",
            iterator=_quartet_source_iterator(
                quartet["family_a"], quartet["family_b"], options, "two_corner", str(quartet["id"])
            ),
        )
        for quartet in detected_quartets
    ]
    return topology_sources + ordinary_sources


def _one_corner_sources(
    corners: list[dict[str, Any]],
    lines: list[dict[str, Any]],
    template_quartets: list[dict[str, Any]],
    max_dimension: float,
    random_seed: int,
) -> list[_Source]:
    lines_by_id = {str(line["id"]): line for line in lines}
    template_options = _ranked(
        [option for option in template_quartets if int(option["corner_count"]) >= 1],
        random_seed,
    )
    if not template_options:
        return []
    sources: list[_Source] = []
    for corner in corners:
        anchor_a, anchor_b = (lines_by_id[identifier] for identifier in corner["line_ids"])
        excluded = set(corner["line_ids"])
        candidates_a, candidates_b = [], []
        for line in lines:
            if str(line["id"]) in excluded:
                continue
            in_a, in_b = _classify_against_corner(line, anchor_a, anchor_b)
            entry = {
                "id": str(line["id"]),
                "key": str(line["id"]),
                "line": line,
                "rank": _line_rank(line, max_dimension),
            }
            if in_a:
                candidates_a.append(entry)
            if in_b:
                candidates_b.append(entry)
        candidates_a, candidates_b = _ranked(candidates_a, random_seed), _ranked(candidates_b, random_seed)
        if not candidates_a or not candidates_b:
            continue

        def iterator(
            corner: dict[str, Any] = corner,
            anchor_a: dict[str, Any] = anchor_a,
            anchor_b: dict[str, Any] = anchor_b,
            candidates_a: list[dict[str, Any]] = candidates_a,
            candidates_b: list[dict[str, Any]] = candidates_b,
            template_options: list[dict[str, Any]] = template_options,
        ) -> Iterator[dict[str, Any]]:
            for extra_a in candidates_a:
                for extra_b in candidates_b:
                    if str(extra_a["line"]["id"]) == str(extra_b["line"]["id"]):
                        continue
                    yield from _quartet_source_iterator(
                        [anchor_a, extra_a["line"]],
                        [anchor_b, extra_b["line"]],
                        template_options,
                        "one_corner",
                        str(corner["id"]),
                    )

        sources.append(
            _Source(
                key=str(corner["key"]),
                rank=float(corner["rank"]),
                proposal_type="one_corner",
                iterator=iterator(),
            )
        )
    return sources


def _orientation_sources(
    lines: list[dict[str, Any]],
    template_quartets: list[dict[str, Any]],
    max_dimension: float,
    random_seed: int,
) -> list[_Source]:
    template_options = _ranked(template_quartets, random_seed)
    sources: list[_Source] = []
    for anchor_a, anchor_b in itertools.combinations(lines, 2):
        if _angle_difference(_line_angle(anchor_a), _line_angle(anchor_b)) < math.radians(MIN_CROSS_ANGLE_DEG):
            continue
        excluded = {str(anchor_a["id"]), str(anchor_b["id"])}
        candidates_a, candidates_b = [], []
        for line in lines:
            if str(line["id"]) in excluded:
                continue
            in_a, in_b = _classify_against_corner(line, anchor_a, anchor_b)
            entry = {
                "id": str(line["id"]),
                "key": str(line["id"]),
                "line": line,
                "rank": _line_rank(line, max_dimension),
            }
            if in_a:
                candidates_a.append(entry)
            if in_b:
                candidates_b.append(entry)
        candidates_a, candidates_b = _ranked(candidates_a, random_seed), _ranked(candidates_b, random_seed)
        if not candidates_a or not candidates_b:
            continue
        source_key = f"orientation:{anchor_a['id']}:{anchor_b['id']}"

        def iterator(
            source_key: str = source_key,
            anchor_a: dict[str, Any] = anchor_a,
            anchor_b: dict[str, Any] = anchor_b,
            candidates_a: list[dict[str, Any]] = candidates_a,
            candidates_b: list[dict[str, Any]] = candidates_b,
            template_options: list[dict[str, Any]] = template_options,
        ) -> Iterator[dict[str, Any]]:
            for extra_a in candidates_a:
                for extra_b in candidates_b:
                    if str(extra_a["line"]["id"]) == str(extra_b["line"]["id"]):
                        continue
                    yield from _quartet_source_iterator(
                        [anchor_a, extra_a["line"]],
                        [anchor_b, extra_b["line"]],
                        template_options,
                        "orientation_only",
                        source_key,
                    )

        sources.append(
            _Source(
                key=source_key,
                rank=_line_rank(anchor_a, max_dimension) + _line_rank(anchor_b, max_dimension),
                proposal_type="orientation_only",
                iterator=iterator(),
            )
        )
    return sources


def _canonical_key(proposal: dict[str, Any]) -> tuple[tuple[str, str], ...] | None:
    seed_pairs = proposal.get("seed_pairs", {})
    if len(seed_pairs) != 4 or len(set(seed_pairs.values())) != 4:
        return None
    return tuple(sorted((str(template_id), str(detected_id)) for template_id, detected_id in seed_pairs.items()))


def _consume_round_robin(
    sources: list[_Source],
    proposals: list[dict[str, Any]],
    seen: set[tuple[tuple[str, str], ...]],
    diagnostics: dict[str, Any],
    max_proposals: int,
    random_seed: int,
) -> None:
    if len(proposals) >= max_proposals:
        return
    ordered_sources = _ranked(
        [
            {
                "id": source.key,
                "key": source.key,
                "rank": source.rank,
                "source": source,
            }
            for source in sources
        ],
        random_seed,
    )
    active = deque(item["source"] for item in ordered_sources)
    # Duplicate-heavy fallback sources are finite, but this guard keeps malformed
    # external inputs from turning proposal construction into unbounded work.
    attempt_limit = max(10_000, max_proposals * 100)
    attempts = 0
    while active and len(proposals) < max_proposals and attempts < attempt_limit:
        source = active.popleft()
        try:
            proposal = next(source.iterator)
        except StopIteration:
            continue
        attempts += 1
        active.append(source)
        key = _canonical_key(proposal)
        if key is None:
            diagnostics["invalid_candidates"] += 1
            continue
        if key in seen:
            diagnostics["duplicates"] += 1
            diagnostics["duplicates_removed"] += 1
            continue
        seen.add(key)
        proposals.append(proposal)
        proposal_type = str(proposal["proposal_type"])
        diagnostics["proposal_types"][proposal_type] += 1


def _weighted_tier_allocations(
    tiers: list[tuple[str, list[_Source], int]], max_proposals: int
) -> dict[str, int]:
    """Reserve budget for every available fallback, favoring stronger tiers."""
    allocations = {name: 0 for name, _, _ in tiers}
    active = [(name, sources, weight) for name, sources, weight in tiers if sources]
    if max_proposals <= 0 or not active:
        return allocations

    # With a tiny budget, preserve the documented tier order. Otherwise give
    # every available tier one slot before apportioning the remaining budget.
    initially_served = min(max_proposals, len(active))
    for name, _, _ in active[:initially_served]:
        allocations[name] += 1
    remaining = max_proposals - initially_served
    if remaining <= 0:
        return allocations

    weight_sum = sum(weight for _, _, weight in active)
    fractional: list[tuple[float, int, str]] = []
    assigned = 0
    for priority, (name, _, weight) in enumerate(active):
        exact = remaining * weight / weight_sum
        whole = int(math.floor(exact))
        allocations[name] += whole
        assigned += whole
        fractional.append((exact - whole, -priority, name))
    for _, _, name in sorted(fractional, reverse=True)[: remaining - assigned]:
        allocations[name] += 1
    return allocations


def generate_guided_proposals(
    detection: dict[str, Any],
    template: dict[str, Any],
    *,
    max_proposals: int = 5000,
    random_seed: int = 7,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate orientation- and adjacency-guided four-line correspondences.

    Proposal priority is U motif, two connected corners, one connected corner,
    then orientation-only. Sources within each tier are consumed round-robin,
    and weighted tier reservations prevent a prolific false motif from starving
    every fallback. The budget remains a strict maximum. ``random_seed`` only
    orders exact rank ties.
    """
    if max_proposals < 0:
        raise ValueError("max_proposals must be non-negative")
    raw_lines, intersections = _detected_lines_and_intersections(detection)
    template_lines = list(template.get("lines", []))
    diagnostics: dict[str, Any] = {
        "strategy_version": STRATEGY_VERSION,
        "proposal_budget": int(max_proposals),
        "anchor_counts": {},
        "proposal_types": {
            "u_motif": 0,
            "two_corner": 0,
            "one_corner": 0,
            "orientation_only": 0,
        },
        "duplicates": 0,
        "duplicates_removed": 0,
        "invalid_candidates": 0,
        "orientation_rejected_candidates": 0,
        # The solver increments this after homography construction and sanity
        # checks, immediately before calling its independent scorer.
        "candidates_reaching_scoring": 0,
    }
    lines = _validated_detected_lines(raw_lines, diagnostics)
    if max_proposals == 0 or len(lines) < 4 or len(template_lines) < 4:
        diagnostics["anchor_counts"] = {
            "detected_lines_input": len(raw_lines),
            "detected_lines": len(lines),
            "detected_intersections": len(intersections),
            "detected_u_motifs": 0,
            "template_u_motifs": 0,
            "detected_two_corner_quartets": 0,
            "detected_multi_corner_quartets": 0,
            "topology_guided_two_corner_sources": 0,
            "detected_one_corner_sources": 0,
            "detected_orientation_sources": 0,
            "template_quartets": 0,
        }
        diagnostics["proposal_type_budgets"] = {
            "u_motif": 0,
            "two_corner": 0,
            "one_corner": 0,
            "orientation_only": 0,
        }
        diagnostics["proposals_returned"] = 0
        return [], diagnostics

    family_data = derive_template_orientation_families(template)
    line_to_family = family_data["line_to_family"]
    template_motifs = _discover_template_u_motifs(template_lines, line_to_family)
    template_quartets = _template_quartets(template_lines, line_to_family)
    corners = _valid_corners(lines, intersections, diagnostics)
    detected_motifs = _discover_u_motifs(
        lines, intersections, diagnostics, corners=corners
    )
    image_size = detection.get("image_size", {})
    max_dimension = float(max(image_size.get("width", 1), image_size.get("height", 1), 1))

    motif_sources = _motif_sources(
        detected_motifs,
        template_motifs,
        lines,
        template_lines,
        line_to_family,
        max_dimension,
        random_seed,
    )
    two_corner_quartets = _detected_two_corner_quartets(
        corners, lines, max_dimension, diagnostics
    )
    two_corner_sources = _two_corner_sources(two_corner_quartets, template_quartets, random_seed)
    one_corner_sources = _one_corner_sources(
        corners, lines, template_quartets, max_dimension, random_seed
    )
    orientation_sources = _orientation_sources(
        lines, template_quartets, max_dimension, random_seed
    )

    diagnostics["template_orientation_families"] = family_data["families"]
    diagnostics["anchor_counts"] = {
        "detected_lines_input": len(raw_lines),
        "detected_lines": len(lines),
        "detected_intersections": len(corners),
        "detected_u_motifs": len(detected_motifs),
        "template_u_motifs": len(template_motifs),
        "detected_two_corner_quartets": len(two_corner_quartets),
        "detected_multi_corner_quartets": sum(
            int(quartet.get("corner_count", 0)) >= 3 for quartet in two_corner_quartets
        ),
        "topology_guided_two_corner_sources": sum(
            ":topology:" in source.key for source in two_corner_sources
        ),
        "detected_one_corner_sources": len(one_corner_sources),
        "detected_orientation_sources": len(orientation_sources),
        "template_quartets": len(template_quartets),
    }

    proposals: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    tiers = [
        ("u_motif", motif_sources, 12),
        ("two_corner", two_corner_sources, 4),
        ("one_corner", one_corner_sources, 2),
        ("orientation_only", orientation_sources, 1),
    ]
    allocations = _weighted_tier_allocations(tiers, max_proposals)
    diagnostics["proposal_type_budgets"] = allocations
    for name, sources, _ in tiers:
        tier_target = min(max_proposals, len(proposals) + allocations[name])
        _consume_round_robin(
            sources, proposals, seen, diagnostics, tier_target, random_seed
        )
    # If a reserved tier is exhausted or duplicates mappings already supplied by
    # a stronger tier, return its unused slots in priority order.
    if len(proposals) < max_proposals:
        for _, sources, _ in tiers:
            _consume_round_robin(
                sources, proposals, seen, diagnostics, max_proposals, random_seed
            )
            if len(proposals) >= max_proposals:
                break
    diagnostics["proposals_returned"] = len(proposals)
    return proposals, diagnostics


__all__ = [
    "STRATEGY_VERSION",
    "derive_template_orientation_families",
    "discover_u_motifs",
    "generate_guided_proposals",
]
