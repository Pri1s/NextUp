"""Point-to-polyline geometry, shared by everything that compares two curves.

Four consumers need the same primitive and would otherwise each grow their own:
scoring a calibration against annotated paint, measuring how far two annotation
passes disagree, measuring an annotation against synthetic ground truth, and
finding the foot point of a detected marking sample on a projected court
primitive during hybrid refinement.

This module lives under ``calibration/`` rather than ``benchmark/`` because the
refiner needs it at runtime and ``benchmark`` may not import anything that
predicts. It is pure numpy: no layout, no detector, no model, nothing that could
leak a prediction into a measurement. ``benchmark.polyline`` re-exports it, so
the benchmark keeps its own name for the same code rather than a second
implementation that could silently disagree.

Two details are deliberate.

**Distance is to the polyline, not to its vertices.** A sample point sits wherever
the annotator put it along the marking; the reference polyline is sampled
somewhere else entirely. Vertex-to-vertex distance would therefore measure the
sampling difference rather than the geometric one, and would grow when a reference
is drawn more coarsely.

**Offsets are signed.** Unsigned error cannot distinguish noise from bias. Two
annotators who both sit two pixels toward the darker edge of every stripe produce
excellent agreement and a systematically wrong benchmark; the mean *signed* offset
is what makes that visible, and it is the only check that can.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class NearestResult:
    """Where each query point lands on the reference polyline."""

    distances: np.ndarray  # (N,) unsigned distance in pixels
    signed: np.ndarray  # (N,) signed perpendicular offset, left of travel positive
    closest: np.ndarray  # (N, 2) the projected point on the polyline
    segment: np.ndarray  # (N,) index of the segment each point projected onto
    t: np.ndarray  # (N,) position within that segment, 0 at its start, 1 at its end
    arc_length: np.ndarray  # (N,) distance along the polyline to the closest point

    def __len__(self) -> int:
        return len(self.distances)

    def interpolate(self, other) -> np.ndarray:
        """Read the corresponding position on a polyline sharing this one's vertices.

        Two polylines related by a projection have the same vertex count and order
        but different arc lengths -- perspective compresses the far end, so a point
        60% of the way along in the image is not 60% of the way along on the floor.
        Carrying the segment index and its local ``t`` across is exact, where
        rescaling arc length would introduce a depth-dependent bias into the very
        measurement that is split by depth.
        """
        other = _as_points(other)
        starts = other[self.segment]
        ends = other[np.minimum(self.segment + 1, len(other) - 1)]
        return starts + self.t[:, None] * (ends - starts)


def _as_points(array) -> np.ndarray:
    points = np.asarray(array, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"expected an (N, 2) array of points, got shape {points.shape}")
    return points


def nearest_on_polyline(query, polyline) -> NearestResult:
    """Project every query point onto the nearest point of ``polyline``.

    Projection is clamped to each segment, so a query beyond the end of the
    reference lands on its endpoint rather than on an imaginary extension. That
    matters for occluded markings: a reference drawn across the whole court should
    not be credited with explaining a point past where it actually stops.
    """
    query = _as_points(query)
    polyline = _as_points(polyline)
    if len(polyline) < 2:
        raise ValueError(f"a polyline needs at least two points, got {len(polyline)}")
    if len(query) == 0:
        empty = np.zeros(0, dtype=np.float64)
        return NearestResult(
            empty, empty, np.zeros((0, 2)), np.zeros(0, dtype=int), empty, empty
        )

    starts = polyline[:-1]  # (S, 2)
    ends = polyline[1:]
    deltas = ends - starts
    lengths_sq = np.einsum("ij,ij->i", deltas, deltas)  # (S,)
    safe = np.where(lengths_sq > 1e-12, lengths_sq, 1.0)

    # (N, S) projection parameter of each query onto each segment, clamped to it.
    offsets = query[:, None, :] - starts[None, :, :]
    t = np.einsum("nsj,sj->ns", offsets, deltas) / safe[None, :]
    t = np.clip(t, 0.0, 1.0)
    t = np.where(lengths_sq[None, :] > 1e-12, t, 0.0)

    projected = starts[None, :, :] + t[:, :, None] * deltas[None, :, :]  # (N, S, 2)
    gaps = query[:, None, :] - projected
    distances_all = np.linalg.norm(gaps, axis=2)  # (N, S)

    segment = np.argmin(distances_all, axis=1)
    rows = np.arange(len(query))
    distances = distances_all[rows, segment]
    closest = projected[rows, segment]

    # Sign against the segment normal, taking the polyline's own direction as the
    # reference. Left of travel is positive.
    direction = deltas[segment]
    norms = np.linalg.norm(direction, axis=1)
    norms = np.where(norms > 1e-12, norms, 1.0)
    unit = direction / norms[:, None]
    normal = np.stack([-unit[:, 1], unit[:, 0]], axis=1)
    signed = np.einsum("nj,nj->n", query - closest, normal)

    local_t = t[rows, segment]
    cumulative = np.concatenate([[0.0], np.cumsum(np.sqrt(lengths_sq))])
    arc_length = cumulative[segment] + local_t * np.sqrt(lengths_sq[segment])

    return NearestResult(distances, signed, closest, segment, local_t, arc_length)


def polyline_length(polyline) -> float:
    points = _as_points(polyline)
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def resample(polyline, count: int) -> np.ndarray:
    """Evenly spaced points along a polyline, by arc length.

    Comparing two curves sampled at whatever spacing each annotator chose would
    weight whichever one happened to place more points in a region. Resampling
    both puts the comparison on an even footing.
    """
    points = _as_points(polyline)
    if count < 2:
        raise ValueError(f"need at least two samples, got {count}")
    if len(points) < 2:
        raise ValueError("cannot resample a polyline with fewer than two points")

    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(steps)])
    total = cumulative[-1]
    if total <= 1e-12:
        return np.repeat(points[:1], count, axis=0)

    targets = np.linspace(0.0, total, count)
    x = np.interp(targets, cumulative, points[:, 0])
    y = np.interp(targets, cumulative, points[:, 1])
    return np.stack([x, y], axis=1)


def overlapping_extent(a, b, tolerance_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Trim each polyline to the part the other actually covers.

    Two annotators rarely trace the same run of a marking: one starts where a
    player's leg ends, the other a metre further along. Comparing over the union
    would score that coverage difference as disagreement, which it is not. Only
    the shared extent says anything about whether they agree on *where the line
    is*.
    """
    a = _as_points(a)
    b = _as_points(b)
    a_keep = nearest_on_polyline(a, b).distances <= tolerance_px
    b_keep = nearest_on_polyline(b, a).distances <= tolerance_px
    return a[a_keep], b[b_keep]
