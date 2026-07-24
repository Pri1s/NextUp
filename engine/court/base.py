"""The single plug boundary for the engine.

A ``CourtModel`` turns a video frame into the 22 canonical court keypoints. Both
an adapted NBA court model (today) and the HS pose model (later) satisfy this
identical contract, so swapping between them is a profile change, never an
engine change.
"""

from __future__ import annotations

import abc

import numpy as np

from contracts.court_schema import CANONICAL_COUNT


class CourtModel(abc.ABC):
    """Emit canonical court keypoints for a single frame.

    ``predict`` returns a ``(CANONICAL_COUNT, 3)`` float array of ``(x_px,
    y_px, v)`` rows in canonical index order (row 0 == schema index 1). ``v`` is
    0 (not labeled / not detected), 1 (occluded), or 2 (visible). Rows the model
    could not produce are all-zero with ``v == 0``.
    """

    @abc.abstractmethod
    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        ...

    @staticmethod
    def empty() -> np.ndarray:
        """A canonical keypoint array with nothing detected."""
        return np.zeros((CANONICAL_COUNT, 3), dtype=float)
