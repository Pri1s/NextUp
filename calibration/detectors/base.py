"""The detector protocol.

A detector turns a BGR frame into a ``RawKeypointFrame``: native model slots,
continuous confidences, original-image pixels. It knows nothing about courts.

This protocol is also the seam the tests inject through — ``run_inspection``
accepts any ``KeypointDetector``, so the whole tool is exercised without loading
a 400 MB checkpoint.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from contracts.calibration_types import ModelProvenance, RawKeypointFrame


class KeypointDetector(Protocol):
    """Per-frame keypoint inference."""

    @property
    def keypoint_count(self) -> int:
        """Number of native slots the model emits."""
        ...

    def provenance(self) -> ModelProvenance:
        """Identity of the weights and the inference settings in use."""
        ...

    def detect(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
        timestamp_s: float,
        source_id: str,
    ) -> RawKeypointFrame:
        """Run the model on one frame. Never raises on "nothing detected"."""
        ...
