"""Engine-facing accessor over the canonical court keypoint schema.

The schema (``dataset/schemas/court_keypoints.v3.json``) names 22 court
landmarks by absolute identity — north/south end, east/west side — never by
camera viewpoint. That identity set is the plug boundary: any court model that
emits these 22 points in canonical order feeds the same downstream, whether an
adapted NBA model now or the HS pose model later.

Deliberately stdlib-only and self-contained so the ``contracts`` wall holds: it
reads the schema JSON in place but does not import ``validate_schema`` (that
stays the training-side validator). ``flip_idx`` is a training-augmentation
concern and is intentionally absent here — the engine does not need it.

Note: the schema JSON currently lives under ``dataset/schemas/`` (training-side
territory) and is read in place to avoid churning the training pipeline. A
future consolidation could promote it into ``contracts/`` as its canonical home.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "dataset" / "schemas" / "court_keypoints.v3.json"

CANONICAL_COUNT = 22

# Visibility convention, mirrored from the schema's visibility_values.
NOT_LABELED = 0
OCCLUDED = 1
VISIBLE = 2


@lru_cache(maxsize=None)
def load_canonical_schema(path: Path = SCHEMA_PATH) -> dict:
    """Load and lightly sanity-check the canonical schema JSON."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Canonical court schema not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        schema = json.load(handle)
    keypoints = schema.get("keypoints", [])
    if len(keypoints) != CANONICAL_COUNT:
        raise ValueError(
            f"Canonical schema must define {CANONICAL_COUNT} keypoints, "
            f"found {len(keypoints)} in {path}"
        )
    indices = [point["index"] for point in keypoints]
    if indices != list(range(1, CANONICAL_COUNT + 1)):
        raise ValueError(f"keypoint indices must be contiguous 1..{CANONICAL_COUNT}")
    return schema


def canonical_ids(path: Path = SCHEMA_PATH) -> list[str]:
    """The 22 canonical keypoint ids in index order (position 0 == index 1)."""
    return [point["id"] for point in load_canonical_schema(path)["keypoints"]]


def id_to_index(path: Path = SCHEMA_PATH) -> dict[str, int]:
    """id -> 0-based position in the canonical (22, 3) keypoint array."""
    return {cid: i for i, cid in enumerate(canonical_ids(path))}


def index_to_id(path: Path = SCHEMA_PATH) -> dict[int, str]:
    """0-based array position -> canonical id."""
    return dict(enumerate(canonical_ids(path)))
