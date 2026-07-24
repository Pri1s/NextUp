"""Remap a court model's native keypoints into canonical (22, 3) order.

An adapter is the small, model-specific piece that makes different court models
interchangeable behind ``CourtModel``:

* ``IdentityAdapter`` — the source already emits the canonical 22 in order
  (the HS pose model, trained directly on the canonical schema).
* ``NbaCourtAdapter`` — the NBA court model emits its own keypoint set; a
  ``source_index -> canonical_id`` table places each into canonical order and
  leaves unmatched canonical points at ``v == 0``.
"""

from __future__ import annotations

import abc

import numpy as np

from contracts.court_schema import CANONICAL_COUNT, id_to_index


class KeypointAdapter(abc.ABC):
    """Map a source model's keypoint rows into the canonical (22, 3) array."""

    @abc.abstractmethod
    def to_canonical(self, source_xy_v: np.ndarray) -> np.ndarray:
        ...


class IdentityAdapter(KeypointAdapter):
    """Source already produces the canonical 22 keypoints in canonical order."""

    def to_canonical(self, source_xy_v: np.ndarray) -> np.ndarray:
        arr = np.asarray(source_xy_v, dtype=float)
        if arr.shape != (CANONICAL_COUNT, 3):
            raise ValueError(
                f"IdentityAdapter expects a ({CANONICAL_COUNT}, 3) canonical array, "
                f"got {arr.shape}. Use a mapping adapter for non-canonical models."
            )
        return arr


class NbaCourtAdapter(KeypointAdapter):
    """Place a non-canonical model's keypoints into canonical order via a table.

    ``mapping`` is ``{source_index (0-based, in the model's own keypoint order):
    canonical_id}``. Canonical points with no source row stay ``(0, 0, 0)``.
    """

    def __init__(self, mapping: dict[int, str]):
        canonical_index = id_to_index()
        for src_idx, cid in mapping.items():
            if cid not in canonical_index:
                raise ValueError(f"NbaCourtAdapter mapping targets unknown canonical id: {cid!r}")
        self._mapping = {int(src): canonical_index[cid] for src, cid in mapping.items()}

    def to_canonical(self, source_xy_v: np.ndarray) -> np.ndarray:
        arr = np.asarray(source_xy_v, dtype=float)
        out = np.zeros((CANONICAL_COUNT, 3), dtype=float)
        for src_idx, canon_idx in self._mapping.items():
            if 0 <= src_idx < len(arr):
                out[canon_idx] = arr[src_idx]
        return out


# NBA court-model keypoint layout (kpt_shape == [18, 3]) -> canonical ids.
#
# Derived from the model's documented 18-point layout (the same court model used
# in the basketball_analysis reference, whose tactical_view_converter records what
# each index means). Each source index below was matched to a canonical landmark
# by physical court position under one self-consistent orientation:
#   * the model's length axis  (left baseline .. right baseline) -> north .. south
#   * the model's width axis    (top sideline  .. bottom sideline) -> east  .. west
# All 18 native points map to a canonical landmark; the four canonical points the
# model has no keypoint for (both three-point apexes, both center-circle points)
# stay v == 0, which is fine — a homography needs only 4.
#
# NOTE on orientation: this is one coherent assignment (model-left == north). It
# yields an internally consistent homography, so projected court_ft are correct
# and stable; the absolute north/south *labels* assume the camera matches the
# model's training convention. Footage from the opposite sideline would produce
# mirrored labels (still a valid court frame) — resolving that per clip is a
# calibration follow-up, not a blocker. Verify/adjust on domain footage with
# `python -m engine.cli --profile nba --video <clip> --inspect`.
NBA_COURT_KEYPOINT_MAP: dict[int, str] = {
    0: "north_baseline_sideline_east",
    1: "north_three_point_baseline_east",
    2: "north_lane_baseline_east",
    3: "north_lane_baseline_west",
    4: "north_three_point_baseline_west",
    5: "north_baseline_sideline_west",
    6: "midcourt_sideline_west",
    7: "midcourt_sideline_east",
    8: "north_lane_free_throw_east",
    9: "north_lane_free_throw_west",
    10: "south_baseline_sideline_west",
    11: "south_three_point_baseline_west",
    12: "south_lane_baseline_west",
    13: "south_lane_baseline_east",
    14: "south_three_point_baseline_east",
    15: "south_baseline_sideline_east",
    16: "south_lane_free_throw_east",
    17: "south_lane_free_throw_west",
}
