"""Re-export of the shared point-to-polyline primitive.

The implementation moved to ``calibration.polyline`` when the hybrid refiner
needed it at runtime. It stays reachable under this name because the alternative
-- a second copy inside ``calibration/`` -- would create two implementations that
could disagree, and a disagreement between them would be indistinguishable from
a calibration error. That is the same argument ``tests/test_blindness.py`` makes
for sharing ``calibration.estimator``.

Nothing here predicts: the module is pure numpy geometry with no layout, no
detector, and no model.
"""

from __future__ import annotations

from calibration.polyline import (
    NearestResult,
    nearest_on_polyline,
    overlapping_extent,
    polyline_length,
    resample,
)

__all__ = [
    "NearestResult",
    "nearest_on_polyline",
    "overlapping_extent",
    "polyline_length",
    "resample",
]
