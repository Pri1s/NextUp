"""Model-seeded hybrid refinement of a keypoint calibration.

Milestone 4 adds a *second* candidate transform, fitted against painted-marking
evidence that Milestone 3 extracted and identified, and records both candidates
side by side with the diagnostics needed to compare them. Nothing here selects:
the refiner always records ``KEEP_BASELINE``. Promotion is Milestone 5.

The package consumes a serialized ``ShadowMarkingFrame`` rather than live
extractor state, so a refinement run needs no video, no checkpoint, and no GPU.
It sits beside ``calibration/markings/`` rather than inside it because fitting
needs ``calibration.quality`` and ``calibration.correspondence``, and the
evidence layer must not acquire a dependency on the layer that interprets
predictions.
"""

from .control_points import (
    control_court_points,
    control_points_from_homography,
    homography_from_control_points,
    is_convex,
    clip_to_box,
    local_pixel_scale,
    probe_court_points,
    probe_displacement,
)
from .orchestrator import HybridGatePolicy, HybridOrchestrator, HybridProvenanceError, HybridRefinerConfig
from .residuals import MarkingObservation, collect_observations, landmark_residuals, marking_residuals, rebalance
from .refine import RefineResult, central_difference_jacobian, refine

__all__ = [
    "control_court_points",
    "control_points_from_homography",
    "homography_from_control_points",
    "is_convex",
    "clip_to_box",
    "local_pixel_scale",
    "probe_court_points",
    "probe_displacement",
    "HybridGatePolicy", "HybridOrchestrator", "HybridProvenanceError", "HybridRefinerConfig",
    "MarkingObservation", "collect_observations", "landmark_residuals", "marking_residuals", "rebalance",
    "RefineResult", "central_difference_jacobian", "refine",
]
