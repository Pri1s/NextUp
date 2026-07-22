"""Validate the canonical court keypoint schema and its reference diagram.

Checks the schema JSON's internal consistency (indices, enums, mirror-pair
closure, real-court vs diagram coordinate agreement), cross-checks the
reference SVG markers, and derives the fliplr keypoint remap (flip_idx) from
the mirror pairs. export_yolo.py imports flip_idx_from_schema so the exported
data.yaml can never drift from the schema.

Usage: python3 validate_schema.py [--schema dataset/schemas/court_keypoints.v3.json]
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_SCHEMA = PROJECT_DIR / "dataset" / "schemas" / "court_keypoints.v3.json"

ENDS = ("north", "south", "mid")
SIDES = ("east", "west", "center")
SVG_NS = "{http://www.w3.org/2000/svg}"
COORD_TOLERANCE = 1e-6


class SchemaError(ValueError):
    """The schema (or its reference diagram) is internally inconsistent."""


def load_schema(path: Path) -> dict:
    if not path.is_file():
        raise SchemaError(f"Schema not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def flip_idx_from_schema(schema: dict) -> list[int]:
    """Zero-based keypoint remap for horizontal-flip augmentation.

    Built from the pair set named by horizontal_flip.uses; points absent from
    that set lie on the mirror axis and map to themselves.
    """
    keypoints = schema["keypoints"]
    index_of = {point["id"]: point["index"] - 1 for point in keypoints}
    pair_set = schema["horizontal_flip"]["uses"]
    flip = list(range(len(keypoints)))
    for id_a, id_b in schema["mirror_pairs"][pair_set]:
        a, b = index_of[id_a], index_of[id_b]
        flip[a], flip[b] = b, a
    for i, j in enumerate(flip):
        if flip[j] != i:
            raise SchemaError(f"flip_idx is not an involution at index {i}")
    return flip


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise SchemaError(message)


def _validate_keypoints(schema: dict) -> None:
    keypoints = schema["keypoints"]
    _check(bool(keypoints), "schema defines no keypoints")
    indices = [point["index"] for point in keypoints]
    _check(
        indices == list(range(1, len(keypoints) + 1)),
        f"keypoint indices must be contiguous 1..{len(keypoints)}, got {indices}",
    )
    ids = [point["id"] for point in keypoints]
    _check(len(set(ids)) == len(ids), "keypoint ids must be unique")

    length = schema["reference_court"]["length"]
    width = schema["reference_court"]["width"]
    diagram = schema["reference_diagram"]
    origin_x, origin_y = diagram["court_origin_xy"]
    scale = diagram["units_per_foot"]
    view_box = diagram["view_box"]

    for point in keypoints:
        pid = point["id"]
        _check(point["end"] in ENDS, f"{pid}: end must be one of {ENDS}")
        _check(point["side"] in SIDES, f"{pid}: side must be one of {SIDES}")
        _check(bool(point["feature"].strip()), f"{pid}: feature must be non-empty")
        if point["end"] in ("north", "south"):
            _check(pid.startswith(point["end"] + "_"), f"{pid}: id must start with its end")
        if point["side"] in ("east", "west"):
            _check(pid.endswith("_" + point["side"]), f"{pid}: id must end with its side")

        along, across = point["court_xy_ft"]
        _check(0.0 <= along <= length and 0.0 <= across <= width, f"{pid}: court_xy_ft outside the court")
        if point["side"] == "center":
            _check(across == width / 2, f"{pid}: side=center requires court_xy_ft across == {width / 2}")
        if point["end"] == "mid":
            _check(along == length / 2, f"{pid}: end=mid requires court_xy_ft along == {length / 2}")

        expected_xy = [origin_x + scale * across, origin_y + scale * along]
        actual_xy = [float(value) for value in point["diagram_xy"]]
        _check(
            all(abs(a - b) <= COORD_TOLERANCE for a, b in zip(actual_xy, expected_xy)),
            f"{pid}: diagram_xy {actual_xy} does not match court_xy_ft mapping {expected_xy}",
        )
        _check(
            view_box[0] <= actual_xy[0] <= view_box[2] and view_box[1] <= actual_xy[1] <= view_box[3],
            f"{pid}: diagram_xy outside the viewBox",
        )


def _validate_mirror_pairs(schema: dict) -> None:
    by_id = {point["id"]: point for point in schema["keypoints"]}
    length = schema["reference_court"]["length"]
    width = schema["reference_court"]["width"]

    # (pair set name, axis-resident points exempt from it, swapped field,
    #  shared field, coordinate that stays fixed, coords that must sum)
    rules = {
        "east_west": ("side", "center", "end", 0, width),
        "north_south": ("end", "mid", "side", 1, length),
    }
    for set_name, (swap_field, exempt_value, keep_field, keep_coord, coord_sum) in rules.items():
        pairs = schema["mirror_pairs"][set_name]
        seen: set[str] = set()
        for id_a, id_b in pairs:
            _check(id_a in by_id and id_b in by_id, f"{set_name}: unknown id in pair ({id_a}, {id_b})")
            a, b = by_id[id_a], by_id[id_b]
            _check(
                a[keep_field] == b[keep_field],
                f"{set_name}: pair ({id_a}, {id_b}) must share {keep_field}",
            )
            _check(
                {a[swap_field], b[swap_field]} == ({"east", "west"} if swap_field == "side" else {"north", "south"}),
                f"{set_name}: pair ({id_a}, {id_b}) must swap {swap_field}",
            )
            _check(
                a["court_xy_ft"][keep_coord] == b["court_xy_ft"][keep_coord],
                f"{set_name}: pair ({id_a}, {id_b}) must keep coordinate {keep_coord}",
            )
            other = 1 - keep_coord
            _check(
                abs(a["court_xy_ft"][other] + b["court_xy_ft"][other] - coord_sum) <= COORD_TOLERANCE,
                f"{set_name}: pair ({id_a}, {id_b}) coordinates must mirror (sum to {coord_sum})",
            )
            for pid in (id_a, id_b):
                _check(pid not in seen, f"{set_name}: {pid} appears in more than one pair")
                seen.add(pid)
        for point in schema["keypoints"]:
            on_axis = point[swap_field] == exempt_value
            _check(
                (point["id"] in seen) != on_axis,
                f"{set_name}: {point['id']} must be {'exempt (on the mirror axis)' if on_axis else 'in exactly one pair'}",
            )
    _check(
        schema["horizontal_flip"]["uses"] in rules,
        f"horizontal_flip.uses must be one of {sorted(rules)}",
    )


def _validate_svg(schema: dict, svg_path: Path) -> None:
    _check(svg_path.is_file(), f"reference diagram not found: {svg_path}")
    root = ET.parse(svg_path).getroot()
    marker_format = schema["reference_diagram"]["marker_group_id_format"]

    groups = {
        element.get("id"): element
        for element in root.iter(f"{SVG_NS}g")
        if (element.get("id") or "").startswith("kpt-")
    }
    expected_ids = {marker_format.format(index=point["index"]) for point in schema["keypoints"]}
    _check(
        set(groups) == expected_ids,
        f"SVG marker groups {sorted(set(groups) ^ expected_ids)} do not match the schema",
    )
    for point in schema["keypoints"]:
        group = groups[marker_format.format(index=point["index"])]
        circle = group.find(f"{SVG_NS}circle")
        text = group.find(f"{SVG_NS}text")
        _check(circle is not None and text is not None, f"kpt-{point['index']}: marker needs a circle and a label")
        marker_xy = [float(circle.get("cx")), float(circle.get("cy"))]
        _check(
            all(abs(a - b) <= COORD_TOLERANCE for a, b in zip(marker_xy, point["diagram_xy"])),
            f"kpt-{point['index']}: SVG marker at {marker_xy}, schema diagram_xy {point['diagram_xy']}",
        )
        _check(
            (text.text or "").strip() == str(point["index"]),
            f"kpt-{point['index']}: label text {text.text!r} does not match the index",
        )


def validate(schema: dict, svg_path: Path | None = None) -> list[int]:
    """Run every check; return the derived flip_idx. Raises SchemaError."""
    _check(schema.get("schema_name") == "basketball-court-keypoints", "unexpected schema_name")
    _check(str(schema.get("schema_version", "")).startswith("3."), "expected a 3.x schema_version")
    _check(set(schema["visibility_values"]) == {"0", "1", "2"}, "visibility_values must define 0, 1, 2")
    _check(schema["visible_ends_values"] == ["north", "south", "both"], "unexpected visible_ends_values")
    _validate_keypoints(schema)
    _validate_mirror_pairs(schema)
    if svg_path is not None:
        _validate_svg(schema, svg_path)
    return flip_idx_from_schema(schema)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the canonical keypoint schema.")
    parser.add_argument(
        "--schema",
        type=Path,
        default=DEFAULT_SCHEMA,
        help=f"Schema JSON (default: {DEFAULT_SCHEMA})",
    )
    args = parser.parse_args()

    schema = load_schema(args.schema)
    svg_path = PROJECT_DIR / schema["reference_diagram"]["asset"]
    flip_idx = validate(schema, svg_path)

    keypoints = schema["keypoints"]
    print(f"OK: {schema['schema_name']} {schema['schema_version']}")
    print(f"  keypoints: {len(keypoints)}")
    print(f"  kpt_shape: [{len(keypoints)}, 3]")
    print(f"  horizontal_flip uses {schema['horizontal_flip']['uses']} pairs")
    print(f"  flip_idx: {flip_idx}")


if __name__ == "__main__":
    main()
