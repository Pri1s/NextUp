"""The nine locked M3 extraction variants used by offline evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from .association import AssociationConfig
from .extractor import MarkingExtractorConfig


ASSOCIATION_PRESETS = {
    "strict": AssociationConfig(
        min_corridor_fraction=0.85, straight_tangent_deg=10.0,
        arc_tangent_deg=15.0, min_uniqueness_margin=0.30,
        min_inlier_distance_px=10.0,
    ),
    "balanced": AssociationConfig(),
    "coverage": AssociationConfig(
        min_corridor_fraction=0.75, straight_tangent_deg=14.0,
        arc_tangent_deg=20.0, min_uniqueness_margin=0.20,
        min_inlier_distance_px=14.0,
    ),
}


@dataclass(frozen=True, slots=True)
class ExtractionVariant:
    ridge_percentile: float
    association_preset: str
    extractor: MarkingExtractorConfig
    association: AssociationConfig

    @property
    def name(self) -> str:
        return f"ridge{int(self.ridge_percentile)}-{self.association_preset}"

    def as_dict(self) -> dict:
        return {
            "name": self.name, "ridge_percentile": self.ridge_percentile,
            "association_preset": self.association_preset,
            "extractor": self.extractor.as_dict(),
            "association": self.association.as_dict(),
        }

    @property
    def config_hash(self) -> str:
        from .shadow import ShadowConfig
        return ShadowConfig(self.extractor, self.association).content_hash()


def predefined_variants() -> tuple[ExtractionVariant, ...]:
    return tuple(
        ExtractionVariant(
            ridge_percentile=percentile,
            association_preset=preset,
            extractor=MarkingExtractorConfig(ridge_percentile=percentile),
            association=ASSOCIATION_PRESETS[preset],
        )
        for percentile in (90.0, 94.0, 97.0)
        for preset in ("strict", "balanced", "coverage")
    )
