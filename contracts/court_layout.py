"""End-agnostic half-court geometry.

Phase 0 could not establish which end of the court the detector is looking at,
and a single clip of one half-court gives no basis to decide. Rather than guess,
the engine works in a **half-court frame anchored on the visible end**:

```text
        far sideline  Y = 0
        +---------------------------------+
        |                                 |
  X = 0 |  visible baseline               |  X = 47  midcourt
        |                                 |
        +---------------------------------+
        near sideline Y = 50
```

- `+X` runs from the visible baseline toward midcourt.
- `+Y` runs from the far sideline (away from the camera) to the near sideline.
- Units are feet, always.

Both axes are determinate from the image — which baseline is in shot, and which
sideline is nearer the camera — while *neither* requires knowing whether this is
the north or south end. Distances, lane positions and shot ranges are all correct
in this frame. An orientation owner can later map half-court to full court by
supplying end identity alone, without any of this geometry changing.

Landmark identities here are **our** vocabulary with exact rule-book geometry.
They are not claims about what the detector's slots mean; that mapping lives in
the adapter layer and is provisional.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

LAYOUT_SCHEMA_VERSION = "half-court-layout-1.0.0"


class HalfCourtLandmark(str, Enum):
    """Half-court features, named relative to the visible end and the camera.

    "far" and "near" are camera-relative sidelines, determinate from the image
    and stable within a clip. No name encodes north/south or east/west, because
    nothing has established those.
    """

    BASELINE_SIDELINE_FAR = "baseline_sideline_far"
    BASELINE_SIDELINE_NEAR = "baseline_sideline_near"
    THREE_POINT_BASELINE_FAR = "three_point_baseline_far"
    THREE_POINT_BASELINE_NEAR = "three_point_baseline_near"
    LANE_BASELINE_FAR = "lane_baseline_far"
    LANE_BASELINE_NEAR = "lane_baseline_near"
    LANE_FREE_THROW_FAR = "lane_free_throw_far"
    LANE_FREE_THROW_NEAR = "lane_free_throw_near"
    FREE_THROW_CIRCLE_APEX = "free_throw_circle_apex"
    MIDCOURT_SIDELINE_FAR = "midcourt_sideline_far"
    MIDCOURT_SIDELINE_NEAR = "midcourt_sideline_near"

    @property
    def sideline_mirror(self) -> "HalfCourtLandmark | None":
        """The same feature on the opposite sideline, if it has one."""
        return _SIDELINE_MIRRORS.get(self)


_SIDELINE_MIRRORS: dict[HalfCourtLandmark, HalfCourtLandmark] = {}


def _register_mirrors() -> None:
    pairs = [
        (HalfCourtLandmark.BASELINE_SIDELINE_FAR, HalfCourtLandmark.BASELINE_SIDELINE_NEAR),
        (HalfCourtLandmark.THREE_POINT_BASELINE_FAR, HalfCourtLandmark.THREE_POINT_BASELINE_NEAR),
        (HalfCourtLandmark.LANE_BASELINE_FAR, HalfCourtLandmark.LANE_BASELINE_NEAR),
        (HalfCourtLandmark.LANE_FREE_THROW_FAR, HalfCourtLandmark.LANE_FREE_THROW_NEAR),
        (HalfCourtLandmark.MIDCOURT_SIDELINE_FAR, HalfCourtLandmark.MIDCOURT_SIDELINE_NEAR),
    ]
    for a, b in pairs:
        _SIDELINE_MIRRORS[a] = b
        _SIDELINE_MIRRORS[b] = a


_register_mirrors()


@dataclass(frozen=True, slots=True)
class HalfCourtLayout:
    """Landmark coordinates derived from rule-book scalars.

    Coordinates are derived, never hand-entered, so a profile is a few numbers
    from a rule book rather than a table someone could mistype.
    """

    layout_id: str
    rule_set: str
    source: str
    half_length: float
    width: float
    lane_width: float
    free_throw_distance: float
    basket_from_baseline: float
    three_point_corner_inset: float
    free_throw_circle_radius: float
    landmarks: dict[HalfCourtLandmark, tuple[float, float]] = field(default_factory=dict)
    units: str = "feet"

    def __post_init__(self) -> None:
        if self.units != "feet":
            raise ValueError(f"layouts normalise to feet, got {self.units!r}")
        for dimension in (self.half_length, self.width, self.lane_width):
            if dimension <= 0:
                raise ValueError(f"layout dimensions must be positive, got {dimension}")
        for landmark, (x, y) in self.landmarks.items():
            if not 0.0 <= x <= self.half_length:
                raise ValueError(f"{landmark.value} x={x} outside 0..{self.half_length}")
            if not 0.0 <= y <= self.width:
                raise ValueError(f"{landmark.value} y={y} outside 0..{self.width}")

    @property
    def basket_center(self) -> tuple[float, float]:
        return (self.basket_from_baseline, self.width / 2.0)

    def coordinate(self, landmark: HalfCourtLandmark) -> tuple[float, float]:
        try:
            return self.landmarks[landmark]
        except KeyError as error:
            raise KeyError(f"{landmark.value} is absent from layout {self.layout_id}") from error

    def has(self, landmark: HalfCourtLandmark) -> bool:
        return landmark in self.landmarks

    def content_hash(self) -> str:
        """Stable digest of the geometry, for calibration provenance."""
        payload = json.dumps(
            {
                "layout_id": self.layout_id,
                "schema": LAYOUT_SCHEMA_VERSION,
                "landmarks": {k.value: list(v) for k, v in sorted(
                    self.landmarks.items(), key=lambda kv: kv[0].value
                )},
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict:
        return {
            "layout_id": self.layout_id,
            "schema_version": LAYOUT_SCHEMA_VERSION,
            "rule_set": self.rule_set,
            "source": self.source,
            "units": self.units,
            "half_length": self.half_length,
            "width": self.width,
            "content_hash": self.content_hash(),
            "landmarks": {
                k.value: list(v)
                for k, v in sorted(self.landmarks.items(), key=lambda kv: kv[0].value)
            },
        }


def build_layout(
    layout_id: str,
    rule_set: str,
    source: str,
    half_length: float,
    width: float,
    lane_width: float,
    free_throw_distance: float,
    basket_from_baseline: float,
    three_point_corner_inset: float,
    free_throw_circle_radius: float,
) -> HalfCourtLayout:
    """Derive every landmark coordinate from rule-book scalars.

    The lane is centred on the court's long axis, so its edges sit at
    ``width/2 ± lane_width/2``. The free-throw circle apex used here is the
    midcourt-side one, which is the point a detector can actually see as an arc
    vertex.
    """
    centre_y = width / 2.0
    lane_far_y = centre_y - lane_width / 2.0
    lane_near_y = centre_y + lane_width / 2.0

    landmarks = {
        HalfCourtLandmark.BASELINE_SIDELINE_FAR: (0.0, 0.0),
        HalfCourtLandmark.BASELINE_SIDELINE_NEAR: (0.0, width),
        HalfCourtLandmark.THREE_POINT_BASELINE_FAR: (0.0, three_point_corner_inset),
        HalfCourtLandmark.THREE_POINT_BASELINE_NEAR: (0.0, width - three_point_corner_inset),
        HalfCourtLandmark.LANE_BASELINE_FAR: (0.0, lane_far_y),
        HalfCourtLandmark.LANE_BASELINE_NEAR: (0.0, lane_near_y),
        HalfCourtLandmark.LANE_FREE_THROW_FAR: (free_throw_distance, lane_far_y),
        HalfCourtLandmark.LANE_FREE_THROW_NEAR: (free_throw_distance, lane_near_y),
        HalfCourtLandmark.FREE_THROW_CIRCLE_APEX: (
            free_throw_distance + free_throw_circle_radius,
            centre_y,
        ),
        HalfCourtLandmark.MIDCOURT_SIDELINE_FAR: (half_length, 0.0),
        HalfCourtLandmark.MIDCOURT_SIDELINE_NEAR: (half_length, width),
    }

    return HalfCourtLayout(
        layout_id=layout_id,
        rule_set=rule_set,
        source=source,
        half_length=half_length,
        width=width,
        lane_width=lane_width,
        free_throw_distance=free_throw_distance,
        basket_from_baseline=basket_from_baseline,
        three_point_corner_inset=three_point_corner_inset,
        free_throw_circle_radius=free_throw_circle_radius,
        landmarks=landmarks,
    )


def load_layout(path: Path | str) -> HalfCourtLayout:
    """Load a layout profile. Profiles are data; the builder owns the geometry."""
    with open(path, encoding="utf-8") as handle:
        profile = json.load(handle)

    schema = profile.get("schema_version")
    if schema != LAYOUT_SCHEMA_VERSION:
        raise ValueError(f"unsupported layout schema {schema!r}, expected {LAYOUT_SCHEMA_VERSION!r}")

    dimensions = profile["dimensions"]
    lane = profile["lane"]
    return build_layout(
        layout_id=profile["layout_id"],
        rule_set=profile["rule_set"],
        source=profile["source"],
        half_length=float(dimensions["half_length"]),
        width=float(dimensions["width"]),
        lane_width=float(lane["width"]),
        free_throw_distance=float(lane["free_throw_distance"]),
        basket_from_baseline=float(profile["basket"]["center_from_baseline"]),
        three_point_corner_inset=float(profile["three_point"]["corner_inset"]),
        free_throw_circle_radius=float(profile["free_throw_circle"]["radius"]),
    )


LAYOUT_DIR = Path(__file__).resolve().parent / "layouts"


def available_layouts() -> list[str]:
    if not LAYOUT_DIR.is_dir():
        return []
    return sorted(p.stem for p in LAYOUT_DIR.glob("*.json"))


def load_registered_layout(layout_id: str) -> HalfCourtLayout:
    path = LAYOUT_DIR / f"{layout_id}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"unknown layout {layout_id!r}; available: {', '.join(available_layouts()) or 'none'}"
        )
    return load_layout(path)


def validate_layout(layout: HalfCourtLayout) -> list[str]:
    """Structural checks. Returns a list of problems; empty means healthy."""
    problems: list[str] = []
    centre_y = layout.width / 2.0

    for landmark, mirror in _SIDELINE_MIRRORS.items():
        if not (layout.has(landmark) and layout.has(mirror)):
            continue
        x_a, y_a = layout.coordinate(landmark)
        x_b, y_b = layout.coordinate(mirror)
        if abs(x_a - x_b) > 1e-9:
            problems.append(f"{landmark.value}/{mirror.value} differ in x ({x_a} vs {x_b})")
        if abs((y_a + y_b) / 2.0 - centre_y) > 1e-9:
            problems.append(
                f"{landmark.value}/{mirror.value} are not symmetric about y={centre_y}"
            )

    if layout.has(HalfCourtLandmark.FREE_THROW_CIRCLE_APEX):
        _, apex_y = layout.coordinate(HalfCourtLandmark.FREE_THROW_CIRCLE_APEX)
        if abs(apex_y - centre_y) > 1e-9:
            problems.append(f"free-throw circle apex is off-centre (y={apex_y})")

    if layout.free_throw_distance >= layout.half_length:
        problems.append("free-throw line is at or beyond midcourt")
    if layout.lane_width >= layout.width:
        problems.append("lane is wider than the court")
    if layout.basket_from_baseline <= 0 or layout.basket_from_baseline >= layout.half_length:
        problems.append("basket centre is outside the half court")

    return problems
