"""Transform parameterisation: four projected court corners, not nine coefficients.

Optimising raw homography entries is the obvious approach and the wrong one. The
entries carry incommensurable units -- a translation term runs to hundreds of
pixels while a perspective term is order 1e-3 -- so a step that is tiny in one is
enormous in another, and there is no bound on any of them that means anything
physically. Worse, nothing stops the optimiser wandering into a matrix that folds
the court quad inside out while the cost still improves.

Parameterising by *where the four half-court corners land in the image* fixes
both. The eight numbers are all pixel coordinates, so a bound expressed in court
feet converts to a real distance at each corner. Convexity becomes a predicate on
the quad itself. And the four corners are the same quad
``quality._plausibility_failures`` already projects for its own convexity check,
so a parameter vector this module calls feasible cannot fail that gate later.
"""

from __future__ import annotations

import math

import numpy as np

from contracts.court_layout import HalfCourtLandmark, HalfCourtLayout
from contracts.markings import MarkingFeature

from ..estimator import project
from ..quality import _is_convex

#: Court-frame corners of the visible half, in the winding order the plausibility
#: check uses. Order is load-bearing: the parameter vector is these points'
#: image coordinates, flattened, and every consumer assumes this sequence.
CONTROL_CORNER_ORDER = ("baseline_far", "midcourt_far", "midcourt_near", "baseline_near")

#: Depth bands for reporting, duplicated from ``benchmark.geometry.DEPTH_BANDS``.
#:
#: The duplication is deliberate and one-directional: ``calibration`` must never
#: import ``benchmark``, because Milestone 3 put scoring outside the engine
#: precisely so runtime code cannot tune itself against the modules that grade
#: it. A test under ``benchmark/tests/`` -- which may look both ways -- asserts
#: the two definitions stay identical, so the copy cannot drift silently.
DEFAULT_DEPTH_BANDS: tuple[tuple[str, float, float], ...] = (
    ("0-19ft", 0.0, 19.0),
    ("19-30ft", 19.0, 30.0),
    ("30-47ft", 30.0, 47.0),
)


def is_convex(image_quad) -> bool:
    """Whether four projected corners still form a convex, unfolded quad."""
    quad = np.asarray(image_quad, dtype=np.float64).reshape(4, 2)
    if not np.all(np.isfinite(quad)):
        return False
    return bool(_is_convex(quad))


def control_court_points(layout: HalfCourtLayout) -> np.ndarray:
    """The four half-court corners in court feet, as ``(4, 2)``."""
    return np.array(
        [
            (0.0, 0.0),
            (layout.half_length, 0.0),
            (layout.half_length, layout.width),
            (0.0, layout.width),
        ],
        dtype=np.float64,
    )


def probe_court_points(layout: HalfCourtLayout) -> np.ndarray:
    """Court points at which two candidate transforms are compared.

    Displacement of *projected court points* is the only honest way to say how
    far apart two homographies are. Comparing matrix entries is meaningless -- a
    homography is scale-equivalent and a Frobenius norm of the difference is
    dominated by translation -- and comparing in court feet would require
    inverting through one of the transforms under test.

    The set is the eleven named layout landmarks plus four interior points that
    no landmark covers: the basket centre, the three-point arc apex, and the two
    points where the arc meets its corner straights. The additions matter because
    the landmarks reach 24.9 ft at most through ``free_throw_circle_apex``,
    while the arc apex sits at 29 ft -- the deepest paint the refiner can
    actually fit, and therefore the place where a far-court disagreement shows
    up first.

    Four of the eleven landmarks *are* the half-court corners, so four of these
    fifteen points coincide with the control points by construction. That is not
    hidden: their displacement is also reported on its own as
    ``max_control_point_shift_ft``. The remaining eleven are independent of the
    parameter vector, which is what stops the summary being a restatement of it.
    """
    landmarks = [layout.coordinate(landmark) for landmark in HalfCourtLandmark]
    arc = layout.marking(MarkingFeature.THREE_POINT_ARC)
    corner_far = layout.marking(MarkingFeature.THREE_POINT_CORNER_FAR)
    corner_near = layout.marking(MarkingFeature.THREE_POINT_CORNER_NEAR)
    circle = layout.marking(MarkingFeature.CENTER_CIRCLE)
    interior = [
        layout.basket_center,
        (arc.center[0] + arc.radius_ft, arc.center[1]),  # arc apex, the deepest fittable paint
        corner_far.end,  # where the far corner straight meets the arc
        corner_near.end,
        # The centre circle's near edge. Without it the 30-47 ft band contains
        # only the two midcourt corners, which are also control points -- the
        # band would then report the parameter vector back to itself.
        (circle.center[0] - circle.radius_ft, circle.center[1]),
    ]
    return np.array([tuple(float(v) for v in point) for point in landmarks + interior], dtype=np.float64)


def homography_from_control_points(court, image) -> np.ndarray | None:
    """Exact four-point DLT, court -> image, normalised so ``h[2][2] == 1``.

    Deliberately not ``cv2.getPerspectiveTransform``. This sits on the hot path of
    a content-hashed artifact, and a transform that depends on an OpenCV build is
    a transform whose reproducibility depends on something the config hash does
    not cover. The eight-by-eight solve is exact and needs nothing but numpy;
    a unit test pins it against OpenCV to 1e-9 so the two cannot drift.
    """
    court = np.asarray(court, dtype=np.float64).reshape(-1, 2)
    image = np.asarray(image, dtype=np.float64).reshape(-1, 2)
    if len(court) != 4 or len(image) != 4:
        raise ValueError("a four-point DLT needs exactly four correspondences")
    if not np.all(np.isfinite(court)) or not np.all(np.isfinite(image)):
        return None

    rows = np.zeros((8, 8), dtype=np.float64)
    rhs = np.zeros(8, dtype=np.float64)
    for index in range(4):
        cx, cy = court[index]
        ix, iy = image[index]
        rows[2 * index] = (cx, cy, 1.0, 0.0, 0.0, 0.0, -cx * ix, -cy * ix)
        rows[2 * index + 1] = (0.0, 0.0, 0.0, cx, cy, 1.0, -cx * iy, -cy * iy)
        rhs[2 * index] = ix
        rhs[2 * index + 1] = iy

    try:
        solution = np.linalg.solve(rows, rhs)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(solution)):
        return None

    matrix = np.array(
        [
            [solution[0], solution[1], solution[2]],
            [solution[3], solution[4], solution[5]],
            [solution[6], solution[7], 1.0],
        ],
        dtype=np.float64,
    )
    if abs(float(np.linalg.det(matrix))) < 1e-12:
        return None
    return matrix


def control_points_from_homography(matrix, layout: HalfCourtLayout) -> np.ndarray | None:
    """Where a transform puts the four control corners, as ``(4, 2)`` pixels."""
    projected = project(np.asarray(matrix, dtype=np.float64), control_court_points(layout))
    if projected is None or not np.all(np.isfinite(projected)):
        return None
    return projected


def local_pixel_scale(matrix, court) -> np.ndarray:
    """Image length of a one-foot court step at each given court point.

    A displacement bound has to be expressed in court feet, because the pixel
    size of a foot varies by two orders of magnitude between the near baseline
    and midcourt. A uniform pixel bound would be punishing at one end of the
    court and vacuous at the other.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    court = np.asarray(court, dtype=np.float64).reshape(-1, 2)
    base = project(matrix, court)
    stepped_x = project(matrix, court + np.array([1.0, 0.0]))
    stepped_y = project(matrix, court + np.array([0.0, 1.0]))
    if base is None or stepped_x is None or stepped_y is None:
        return np.full(len(court), np.nan)
    dx = np.linalg.norm(stepped_x - base, axis=1)
    dy = np.linalg.norm(stepped_y - base, axis=1)
    return 0.5 * (dx + dy)


def clip_to_box(theta, theta0, radius_px) -> tuple[np.ndarray, int]:
    """Project a parameter vector back into the per-corner displacement box.

    Each corner is bounded independently by a disc, not the vector as a whole by
    a single norm: the corners are physically separate places and a budget shared
    between them would let one swing freely while the others sat still.

    Returns the clipped vector and how many corners are pinned to their bound. A
    solution resting on its bound has not converged, it has been stopped, and the
    count is what makes that visible rather than silently accepted.
    """
    theta = np.asarray(theta, dtype=np.float64).reshape(4, 2)
    theta0 = np.asarray(theta0, dtype=np.float64).reshape(4, 2)
    radius = np.asarray(radius_px, dtype=np.float64).reshape(4)

    offset = theta - theta0
    distance = np.linalg.norm(offset, axis=1)
    active = 0
    clipped = theta.copy()
    for index in range(4):
        limit = radius[index]
        if not math.isfinite(limit) or limit <= 0:
            clipped[index] = theta0[index]
            active += 1
            continue
        if distance[index] > limit:
            clipped[index] = theta0[index] + offset[index] * (limit / distance[index])
            active += 1
        elif distance[index] >= limit * (1.0 - 1e-9) and distance[index] > 0:
            active += 1
    return clipped.reshape(-1), active


def band_indices(depth, bands) -> np.ndarray:
    """Which band each court depth belongs to, clamped at both ends.

    Mirrors ``benchmark.geometry.depth_band`` exactly, including its clamping:
    the half-court runs to 47.083 ft while the last band stops at 47, so a strict
    upper bound would silently drop the midcourt points from every band. The two
    implementations must agree because a probe binned one way here and another
    way in the benchmark would make the per-band numbers incomparable, and
    ``calibration`` may not import ``benchmark`` to share the function.
    """
    depth = np.asarray(depth, dtype=np.float64)
    index = np.zeros(len(depth), dtype=int)
    # Above the last band's ceiling clamps to the last band; below the first
    # band's floor clamps to the first. Everything else lands in its half-open
    # interval, applied afterwards so the clamps cannot overwrite a real match.
    index[depth >= bands[-1][2]] = len(bands) - 1
    for position, (_, low, high) in enumerate(bands):
        index[(depth >= low) & (depth < high)] = position
    return index


def probe_displacement(baseline, challenger, layout: HalfCourtLayout, bands) -> dict:
    """Per-band displacement of the probe points between two transforms.

    Depth is read from the *court* coordinate of each probe point, never from a
    projection, so band membership cannot shift with the transform under test --
    which matters, because the transform is the thing being measured.
    """
    court = probe_court_points(layout)
    a = project(np.asarray(baseline, dtype=np.float64), court)
    b = project(np.asarray(challenger, dtype=np.float64), court)
    if a is None or b is None or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return {"total": None, "bands": {}}

    distances = np.linalg.norm(b - a, axis=1)
    index = band_indices(court[:, 0], bands)
    return {
        "total": distances,
        "bands": {name: distances[index == position] for position, (name, _, _) in enumerate(bands)},
    }
