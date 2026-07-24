"""Per-court metric templates for the canonical landmarks.

The 22 canonical keypoint *identities* are court-type independent; only their
real-world coordinates differ (NBA is 94x50 with a 16 ft lane, NFHS is 84x50
with a 12 ft lane, etc.). Each template maps every canonical id to its
``court_xy_ft`` for one court type, so the homography and (later) the minimap
pick the geometry matching the footage while the keypoint identities stay fixed.

A template is validated against the canonical schema on load: its ids must be
exactly the canonical set, coordinates must sit within the court, and the
schema's mirror pairs must actually mirror (same checks the training-side
``validate_schema.py`` applies to the schema's embedded coordinates).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from contracts.court_schema import canonical_ids, load_canonical_schema

TEMPLATES_DIR = Path(__file__).resolve().parent / "court_templates"
COORD_TOLERANCE = 1e-6


class TemplateError(ValueError):
    """A court template is missing or inconsistent with the canonical schema."""


@dataclass(frozen=True)
class CourtTemplate:
    name: str
    length: float
    width: float
    # canonical id -> (along from north baseline, across from west sideline), in feet
    court_xy_ft: dict[str, tuple[float, float]]


def available_templates() -> list[str]:
    return sorted(path.stem for path in TEMPLATES_DIR.glob("*.json"))


def load_template(name: str) -> CourtTemplate:
    path = TEMPLATES_DIR / f"{name}.json"
    if not path.is_file():
        raise TemplateError(
            f"Court template not found: {name} "
            f"(available: {', '.join(available_templates()) or 'none'})"
        )
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    length = float(raw["length"])
    width = float(raw["width"])
    coords = {cid: (float(xy[0]), float(xy[1])) for cid, xy in raw["court_xy_ft"].items()}
    template = CourtTemplate(name=raw.get("template_name", name), length=length, width=width, court_xy_ft=coords)
    _validate(template)
    return template


def _validate(template: CourtTemplate) -> None:
    schema = load_canonical_schema()
    ids = set(canonical_ids())
    if set(template.court_xy_ft) != ids:
        missing = ids - set(template.court_xy_ft)
        extra = set(template.court_xy_ft) - ids
        raise TemplateError(
            f"template {template.name} ids do not match the canonical set; "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )

    by_id = {point["id"]: point for point in schema["keypoints"]}
    for cid, (along, across) in template.court_xy_ft.items():
        if not (0.0 <= along <= template.length and 0.0 <= across <= template.width):
            raise TemplateError(f"{template.name}: {cid} court_xy_ft outside the court")
        point = by_id[cid]
        if point["side"] == "center" and abs(across - template.width / 2) > COORD_TOLERANCE:
            raise TemplateError(f"{template.name}: {cid} side=center requires across == {template.width / 2}")
        if point["end"] == "mid" and abs(along - template.length / 2) > COORD_TOLERANCE:
            raise TemplateError(f"{template.name}: {cid} end=mid requires along == {template.length / 2}")

    _check_mirror(template, schema, "east_west", axis="across", total=template.width)
    _check_mirror(template, schema, "north_south", axis="along", total=template.length)


def _check_mirror(template: CourtTemplate, schema: dict, pair_set: str, axis: str, total: float) -> None:
    """Each mirror pair must reflect across its axis: shared coord equal, the
    other coord summing to the court dimension."""
    coord = 1 if axis == "across" else 0  # court_xy_ft is [along, across]
    keep = 1 - coord
    for id_a, id_b in schema["mirror_pairs"][pair_set]:
        a = template.court_xy_ft[id_a]
        b = template.court_xy_ft[id_b]
        if abs(a[keep] - b[keep]) > COORD_TOLERANCE:
            raise TemplateError(f"{template.name}: {pair_set} pair ({id_a}, {id_b}) must share coordinate {keep}")
        if abs(a[coord] + b[coord] - total) > COORD_TOLERANCE:
            raise TemplateError(f"{template.name}: {pair_set} pair ({id_a}, {id_b}) must sum to {total}")
