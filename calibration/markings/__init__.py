"""Static-frame painted-marking evidence and shadow association."""

from .association import MarkingAssociator
from .extractor import (
    MarkingExtractor, MarkingExtractorConfig, build_projected_roi_mask,
    corridor_width_px, diagonal_scale,
)
from .shadow import ShadowConfig, ShadowOrchestrator
from .variants import ExtractionVariant, predefined_variants

__all__ = [
    "MarkingAssociator", "MarkingExtractor", "MarkingExtractorConfig",
    "ShadowConfig", "ShadowOrchestrator", "build_projected_roi_mask",
    "corridor_width_px", "diagonal_scale",
    "ExtractionVariant", "predefined_variants",
]
