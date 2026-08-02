"""Model-neutral records for future painted-marking hybrid calibration.

Milestone 2 defines and validates the audit vocabulary only. No extraction,
association, fitting, gating, or selection behavior is implemented here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .calibration_types import Matrix3x3
from .markings import MarkingFamily, MarkingFeature

HYBRID_RECORD_SCHEMA_VERSION = "hybrid-marking-records-1.0.0"


def _finite(value: float, name: str, *, minimum: float | None = None) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _confidence(value: float, name: str = "confidence") -> float:
    value = _finite(value, name)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


def _fields(data: dict, required: set[str], optional: set[str] = set()) -> None:
    missing = required - set(data)
    unknown = set(data) - required - optional
    if missing or unknown:
        raise ValueError(f"record fields mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}")


def _schema(data: dict) -> None:
    if data.get("schema_version") != HYBRID_RECORD_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported hybrid record schema {data.get('schema_version')!r}; "
            f"expected {HYBRID_RECORD_SCHEMA_VERSION!r}"
        )


def _nonempty(value: str, name: str) -> str:
    value = str(value)
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


class EvidenceKind(str, Enum):
    STRAIGHT = "straight"
    CURVED = "curved"
    UNKNOWN = "unknown"


class AssignmentStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    AMBIGUOUS = "ambiguous"


class CandidateSource(str, Enum):
    KEYPOINT = "keypoint"
    HYBRID = "hybrid"
    CV_ONLY = "cv_only"


class SelectionOutcome(str, Enum):
    KEEP_BASELINE = "keep_baseline"
    SELECT_CHALLENGER = "select_challenger"
    NO_USABLE_CANDIDATE = "no_usable_candidate"


class HybridFrameStatus(str, Enum):
    """Outcome of a hybrid refinement attempt on one image.

    Deliberately a separate enum from :class:`ShadowFrameStatus` rather than a
    member added to it. ``REFINEMENT_FAILED`` is nonsense for an extraction
    record, and a shared enum would make it representable there.
    """

    OK = "OK"
    ABSTAINED = "ABSTAINED"
    SKIPPED = "SKIPPED"
    REFINEMENT_FAILED = "REFINEMENT_FAILED"


class ShadowFrameStatus(str, Enum):
    """Outcome of a marking-shadow observation.

    These statuses are deliberately independent from keypoint calibration
    status.  A shadow run records a disposition for every frame, including
    frames on which it was not allowed to run.
    """

    OK = "OK"
    ABSTAINED = "ABSTAINED"
    ABSTENTION = "ABSTAINED"
    SKIPPED = "SKIPPED"
    EXTRACTION_FAILED = "EXTRACTION_FAILED"


@dataclass(frozen=True, slots=True)
class ImageMarkingSample:
    x: float
    y: float
    tangent_xy: tuple[float, float] | None
    confidence: float
    uncertainty_px: float
    observed_width_px: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _finite(self.x, "x"))
        object.__setattr__(self, "y", _finite(self.y, "y"))
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        object.__setattr__(self, "uncertainty_px", _finite(self.uncertainty_px, "uncertainty_px", minimum=0))
        if self.observed_width_px is not None:
            object.__setattr__(
                self, "observed_width_px",
                _finite(self.observed_width_px, "observed_width_px", minimum=0),
            )
            if self.observed_width_px == 0:
                raise ValueError("observed_width_px must be positive when present")
        if self.tangent_xy is not None:
            try:
                tx, ty = self.tangent_xy
            except (TypeError, ValueError) as error:
                raise ValueError("tangent_xy must have two values") from error
            tangent = (_finite(tx, "tangent_x"), _finite(ty, "tangent_y"))
            if not math.isclose(math.hypot(*tangent), 1.0, rel_tol=0, abs_tol=1e-6):
                raise ValueError("tangent_xy must be a unit vector")
            object.__setattr__(self, "tangent_xy", tangent)

    def as_dict(self) -> dict:
        return {
            "x": self.x, "y": self.y,
            "tangent_xy": None if self.tangent_xy is None else list(self.tangent_xy),
            "confidence": self.confidence,
            "uncertainty_px": self.uncertainty_px,
            "observed_width_px": self.observed_width_px,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ImageMarkingSample":
        _fields(data, {"x", "y", "tangent_xy", "confidence", "uncertainty_px", "observed_width_px"})
        tangent = data["tangent_xy"]
        return cls(
            x=float(data["x"]), y=float(data["y"]),
            tangent_xy=None if tangent is None else (float(tangent[0]), float(tangent[1])),
            confidence=float(data["confidence"]), uncertainty_px=float(data["uncertainty_px"]),
            observed_width_px=(None if data["observed_width_px"] is None else float(data["observed_width_px"])),
        )


@dataclass(frozen=True, slots=True)
class UnlabeledMarkingEvidence:
    evidence_id: str
    frame_id: str
    source_id: str
    kind: EvidenceKind
    samples: tuple[ImageMarkingSample, ...]
    confidence: float
    schema_version: str = HYBRID_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema({"schema_version": self.schema_version})
        for name in ("evidence_id", "frame_id", "source_id"):
            _nonempty(getattr(self, name), name)
        if len(self.samples) < 2:
            raise ValueError("marking evidence needs at least two support samples")
        coordinates = [(sample.x, sample.y) for sample in self.samples]
        if len(set(coordinates)) != len(coordinates):
            raise ValueError("marking evidence contains duplicate support coordinates")
        object.__setattr__(self, "confidence", _confidence(self.confidence))

    def as_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "evidence_id": self.evidence_id, "frame_id": self.frame_id,
            "source_id": self.source_id, "kind": self.kind.value,
            "confidence": self.confidence,
            "samples": [sample.as_dict() for sample in self.samples],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UnlabeledMarkingEvidence":
        _schema(data)
        _fields(data, {"schema_version", "evidence_id", "frame_id", "source_id", "kind", "confidence", "samples"})
        return cls(
            evidence_id=str(data["evidence_id"]), frame_id=str(data["frame_id"]),
            source_id=str(data["source_id"]), kind=EvidenceKind(data["kind"]),
            confidence=float(data["confidence"]),
            samples=tuple(ImageMarkingSample.from_dict(item) for item in data["samples"]),
            schema_version=str(data["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class AssignmentAlternative:
    feature: MarkingFeature
    score: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "score", _confidence(self.score, "assignment score"))

    def as_dict(self) -> dict:
        return {"feature": self.feature.value, "score": self.score}

    @classmethod
    def from_dict(cls, data: dict) -> "AssignmentAlternative":
        _fields(data, {"feature", "score"})
        return cls(MarkingFeature(data["feature"]), float(data["score"]))


@dataclass(frozen=True, slots=True)
class AssociationDiagnostics:
    distance_px: float
    tangent_error_deg: float
    coverage_fraction: float
    uniqueness_margin: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "distance_px", _finite(self.distance_px, "distance_px", minimum=0))
        object.__setattr__(self, "tangent_error_deg", _finite(self.tangent_error_deg, "tangent_error_deg", minimum=0))
        if self.tangent_error_deg > 90:
            raise ValueError("unoriented tangent error must be <= 90 degrees")
        object.__setattr__(self, "coverage_fraction", _confidence(self.coverage_fraction, "coverage_fraction"))
        object.__setattr__(self, "uniqueness_margin", _finite(self.uniqueness_margin, "uniqueness_margin", minimum=0))

    def as_dict(self) -> dict:
        return {
            "distance_px": self.distance_px,
            "tangent_error_deg": self.tangent_error_deg,
            "coverage_fraction": self.coverage_fraction,
            "uniqueness_margin": self.uniqueness_margin,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AssociationDiagnostics":
        _fields(data, {"distance_px", "tangent_error_deg", "coverage_fraction", "uniqueness_margin"})
        return cls(**{key: float(data[key]) for key in data})


@dataclass(frozen=True, slots=True)
class MarkingAssignment:
    evidence_id: str
    status: AssignmentStatus
    selected_feature: MarkingFeature | None
    alternatives: tuple[AssignmentAlternative, ...]
    diagnostics: AssociationDiagnostics
    reason_codes: tuple[str, ...] = ()
    schema_version: str = HYBRID_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema({"schema_version": self.schema_version})
        _nonempty(self.evidence_id, "evidence_id")
        features = [item.feature for item in self.alternatives]
        if len(set(features)) != len(features):
            raise ValueError("assignment alternatives must have unique features")
        if self.status is AssignmentStatus.ACCEPTED:
            if self.selected_feature is None:
                raise ValueError("accepted assignment requires selected_feature")
            if self.selected_feature not in features:
                raise ValueError("selected_feature must appear among alternatives")
        elif self.selected_feature is not None:
            raise ValueError(f"{self.status.value} assignment cannot claim selected_feature")
        if self.status is AssignmentStatus.AMBIGUOUS and len(self.alternatives) < 2:
            raise ValueError("ambiguous assignment needs at least two alternatives")
        if self.status is AssignmentStatus.REJECTED and not self.reason_codes:
            raise ValueError("rejected assignment needs a reason code")
        if any(not reason for reason in self.reason_codes):
            raise ValueError("reason codes must not be empty")

    def as_dict(self) -> dict:
        return {
            "schema_version": self.schema_version, "evidence_id": self.evidence_id,
            "status": self.status.value,
            "selected_feature": None if self.selected_feature is None else self.selected_feature.value,
            "alternatives": [item.as_dict() for item in self.alternatives],
            "diagnostics": self.diagnostics.as_dict(), "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MarkingAssignment":
        _schema(data)
        _fields(data, {"schema_version", "evidence_id", "status", "selected_feature", "alternatives", "diagnostics", "reason_codes"})
        selected = data["selected_feature"]
        return cls(
            evidence_id=str(data["evidence_id"]), status=AssignmentStatus(data["status"]),
            selected_feature=None if selected is None else MarkingFeature(selected),
            alternatives=tuple(AssignmentAlternative.from_dict(item) for item in data["alternatives"]),
            diagnostics=AssociationDiagnostics.from_dict(data["diagnostics"]),
            reason_codes=tuple(str(item) for item in data["reason_codes"]),
            schema_version=str(data["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class ShadowFeatureSupport:
    """Per-feature audit summary; no fitted geometry is implied."""

    feature: MarkingFeature
    accepted_fragment_count: int
    rejected_fragment_count: int
    ambiguous_fragment_count: int
    evidence_to_template_coverage: float
    template_to_evidence_coverage: float
    selected_evidence_ids: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "accepted_fragment_count", "rejected_fragment_count",
            "ambiguous_fragment_count",
        ):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        for name in ("evidence_to_template_coverage", "template_to_evidence_coverage"):
            object.__setattr__(self, name, _confidence(getattr(self, name), name))
        if len(set(self.selected_evidence_ids)) != len(self.selected_evidence_ids):
            raise ValueError("selected_evidence_ids must be unique")
        if any(not item for item in self.selected_evidence_ids + self.reasons):
            raise ValueError("support IDs and reasons must not be empty")

    def as_dict(self) -> dict:
        return {
            "feature": self.feature.value,
            "family": self.feature.family.value,
            "accepted_fragment_count": self.accepted_fragment_count,
            "rejected_fragment_count": self.rejected_fragment_count,
            "ambiguous_fragment_count": self.ambiguous_fragment_count,
            "evidence_to_template_coverage": self.evidence_to_template_coverage,
            "template_to_evidence_coverage": self.template_to_evidence_coverage,
            "selected_evidence_ids": list(self.selected_evidence_ids),
            "reasons": list(self.reasons),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ShadowFeatureSupport":
        _fields(data, {
            "feature", "family", "accepted_fragment_count", "rejected_fragment_count",
            "ambiguous_fragment_count", "evidence_to_template_coverage",
            "template_to_evidence_coverage", "selected_evidence_ids", "reasons",
        })
        feature = MarkingFeature(data["feature"])
        if data["family"] != feature.family.value:
            raise ValueError("feature support family disagrees with feature")
        return cls(
            feature=feature,
            accepted_fragment_count=int(data["accepted_fragment_count"]),
            rejected_fragment_count=int(data["rejected_fragment_count"]),
            ambiguous_fragment_count=int(data["ambiguous_fragment_count"]),
            evidence_to_template_coverage=float(data["evidence_to_template_coverage"]),
            template_to_evidence_coverage=float(data["template_to_evidence_coverage"]),
            selected_evidence_ids=tuple(str(item) for item in data["selected_evidence_ids"]),
            reasons=tuple(str(item) for item in data["reasons"]),
        )


@dataclass(frozen=True, slots=True)
class ShadowFamilySupport:
    """Bidirectional support and topology readiness for one marking family."""

    family: MarkingFamily
    ready: bool
    accepted_fragment_count: int
    support_px: float
    visible_template_px: float
    evidence_to_template_coverage: float
    template_to_evidence_coverage: float
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.accepted_fragment_count < 0:
            raise ValueError("accepted_fragment_count must be non-negative")
        for name in ("support_px", "visible_template_px"):
            object.__setattr__(self, name, _finite(getattr(self, name), name, minimum=0))
        for name in ("evidence_to_template_coverage", "template_to_evidence_coverage"):
            object.__setattr__(self, name, _confidence(getattr(self, name), name))
        if not self.ready and not self.reasons:
            raise ValueError("unready family support needs a reason")
        if any(not item for item in self.reasons):
            raise ValueError("family reasons must not be empty")

    def as_dict(self) -> dict:
        return {
            "family": self.family.value, "ready": self.ready,
            "accepted_fragment_count": self.accepted_fragment_count,
            "support_px": self.support_px, "visible_template_px": self.visible_template_px,
            "evidence_to_template_coverage": self.evidence_to_template_coverage,
            "template_to_evidence_coverage": self.template_to_evidence_coverage,
            "reasons": list(self.reasons),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ShadowFamilySupport":
        _fields(data, {
            "family", "ready", "accepted_fragment_count", "support_px",
            "visible_template_px", "evidence_to_template_coverage",
            "template_to_evidence_coverage", "reasons",
        })
        return cls(
            family=MarkingFamily(data["family"]), ready=bool(data["ready"]),
            accepted_fragment_count=int(data["accepted_fragment_count"]),
            support_px=float(data["support_px"]), visible_template_px=float(data["visible_template_px"]),
            evidence_to_template_coverage=float(data["evidence_to_template_coverage"]),
            template_to_evidence_coverage=float(data["template_to_evidence_coverage"]),
            reasons=tuple(str(item) for item in data["reasons"]),
        )


@dataclass(frozen=True, slots=True)
class ShadowMarkingFrame:
    """One immutable, model-neutral shadow result for one image."""

    frame_id: str
    image_sha256: str
    baseline_status: str
    status: ShadowFrameStatus
    eligible: bool
    eligibility_reasons: tuple[str, ...]
    layout_id: str
    layout_hash: str
    config_hash: str
    image_width: int
    image_height: int
    roi_pixels: int
    excluded_pixels: int
    evidence: tuple[UnlabeledMarkingEvidence, ...] = ()
    assignments: tuple[MarkingAssignment, ...] = ()
    feature_support: tuple[ShadowFeatureSupport, ...] = ()
    family_support: tuple[ShadowFamilySupport, ...] = ()
    extraction_ms: float = 0.0
    association_ms: float = 0.0
    failure_reasons: tuple[str, ...] = ()
    schema_version: str = HYBRID_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema({"schema_version": self.schema_version})
        for name in ("frame_id", "image_sha256", "baseline_status", "layout_id", "layout_hash", "config_hash"):
            _nonempty(getattr(self, name), name)
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image dimensions must be positive")
        for name in ("roi_pixels", "excluded_pixels"):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        for name in ("extraction_ms", "association_ms"):
            object.__setattr__(self, name, _finite(getattr(self, name), name, minimum=0))
        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("evidence IDs must be unique per frame")
        assignments = {item.evidence_id for item in self.assignments}
        if assignments != set(evidence_ids) or len(self.assignments) != len(evidence_ids):
            raise ValueError("there must be exactly one assignment for every evidence ID")
        if len({item.feature for item in self.feature_support}) != len(self.feature_support):
            raise ValueError("feature support must have unique features")
        if len({item.family for item in self.family_support}) != len(self.family_support):
            raise ValueError("family support must have unique families")
        if any(not item for item in self.eligibility_reasons + self.failure_reasons):
            raise ValueError("frame reasons must not be empty")
        if not self.eligible and not self.eligibility_reasons:
            raise ValueError("ineligible frame needs eligibility reasons")

    def as_dict(self) -> dict:
        return {
            "schema_version": self.schema_version, "frame_id": self.frame_id,
            "image_sha256": self.image_sha256, "baseline_status": self.baseline_status,
            "status": self.status.value, "eligible": self.eligible,
            "eligibility_reasons": list(self.eligibility_reasons),
            "layout_id": self.layout_id, "layout_hash": self.layout_hash,
            "config_hash": self.config_hash, "image_width": self.image_width,
            "image_height": self.image_height, "roi_pixels": self.roi_pixels,
            "excluded_pixels": self.excluded_pixels,
            "evidence": [item.as_dict() for item in self.evidence],
            "assignments": [item.as_dict() for item in self.assignments],
            "feature_support": [item.as_dict() for item in self.feature_support],
            "family_support": [item.as_dict() for item in self.family_support],
            "extraction_ms": self.extraction_ms, "association_ms": self.association_ms,
            "failure_reasons": list(self.failure_reasons),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ShadowMarkingFrame":
        _schema(data)
        _fields(data, {
            "schema_version", "frame_id", "image_sha256", "baseline_status", "status",
            "eligible", "eligibility_reasons", "layout_id", "layout_hash", "config_hash",
            "image_width", "image_height", "roi_pixels", "excluded_pixels", "evidence",
            "assignments", "feature_support", "family_support", "extraction_ms",
            "association_ms", "failure_reasons",
        })
        return cls(
            frame_id=str(data["frame_id"]), image_sha256=str(data["image_sha256"]),
            baseline_status=str(data["baseline_status"]), status=ShadowFrameStatus(data["status"]),
            eligible=bool(data["eligible"]),
            eligibility_reasons=tuple(str(item) for item in data["eligibility_reasons"]),
            layout_id=str(data["layout_id"]), layout_hash=str(data["layout_hash"]),
            config_hash=str(data["config_hash"]), image_width=int(data["image_width"]),
            image_height=int(data["image_height"]), roi_pixels=int(data["roi_pixels"]),
            excluded_pixels=int(data["excluded_pixels"]),
            evidence=tuple(UnlabeledMarkingEvidence.from_dict(item) for item in data["evidence"]),
            assignments=tuple(MarkingAssignment.from_dict(item) for item in data["assignments"]),
            feature_support=tuple(ShadowFeatureSupport.from_dict(item) for item in data["feature_support"]),
            family_support=tuple(ShadowFamilySupport.from_dict(item) for item in data["family_support"]),
            extraction_ms=float(data["extraction_ms"]), association_ms=float(data["association_ms"]),
            failure_reasons=tuple(str(item) for item in data["failure_reasons"]),
            schema_version=str(data["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class PrimitiveQuality:
    feature: MarkingFeature
    fitted: bool
    sample_count: int
    residual_px_median: float | None
    residual_px_p95: float | None
    depth_min_ft: float
    depth_max_ft: float

    def __post_init__(self) -> None:
        if self.sample_count < 0:
            raise ValueError("sample_count must be non-negative")
        for name in ("residual_px_median", "residual_px_p95"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite(value, name, minimum=0))
        if self.fitted and self.sample_count == 0:
            raise ValueError("a fitted primitive needs support samples")
        if self.sample_count == 0 and (self.residual_px_median is not None or self.residual_px_p95 is not None):
            raise ValueError("a primitive without samples cannot report residuals")
        object.__setattr__(self, "depth_min_ft", _finite(self.depth_min_ft, "depth_min_ft", minimum=0))
        object.__setattr__(self, "depth_max_ft", _finite(self.depth_max_ft, "depth_max_ft", minimum=0))
        if self.depth_max_ft < self.depth_min_ft:
            raise ValueError("depth_max_ft must be >= depth_min_ft")

    @property
    def family(self) -> MarkingFamily:
        return self.feature.family

    def as_dict(self) -> dict:
        return {
            "feature": self.feature.value, "family": self.family.value, "fitted": self.fitted,
            "sample_count": self.sample_count, "residual_px_median": self.residual_px_median,
            "residual_px_p95": self.residual_px_p95,
            "depth_min_ft": self.depth_min_ft, "depth_max_ft": self.depth_max_ft,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PrimitiveQuality":
        _fields(data, {"feature", "family", "fitted", "sample_count", "residual_px_median", "residual_px_p95", "depth_min_ft", "depth_max_ft"})
        feature = MarkingFeature(data["feature"])
        if data["family"] != feature.family.value:
            raise ValueError("primitive quality family disagrees with feature")
        return cls(
            feature=feature, fitted=bool(data["fitted"]), sample_count=int(data["sample_count"]),
            residual_px_median=None if data["residual_px_median"] is None else float(data["residual_px_median"]),
            residual_px_p95=None if data["residual_px_p95"] is None else float(data["residual_px_p95"]),
            depth_min_ft=float(data["depth_min_ft"]), depth_max_ft=float(data["depth_max_ft"]),
        )


@dataclass(frozen=True, slots=True)
class GateResult:
    gate_id: str
    passed: bool
    value: float | None = None
    threshold: float | None = None
    unit: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        _nonempty(self.gate_id, "gate_id")
        for name in ("value", "threshold"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite(value, name))
        if not self.passed and not self.reason:
            raise ValueError("a failed gate needs a reason")

    def as_dict(self) -> dict:
        return {
            "gate_id": self.gate_id, "passed": self.passed, "value": self.value,
            "threshold": self.threshold, "unit": self.unit, "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GateResult":
        _fields(data, {"gate_id", "passed", "value", "threshold", "unit", "reason"})
        return cls(
            gate_id=str(data["gate_id"]), passed=bool(data["passed"]),
            value=None if data["value"] is None else float(data["value"]),
            threshold=None if data["threshold"] is None else float(data["threshold"]),
            unit=str(data["unit"]), reason=str(data["reason"]),
        )


@dataclass(frozen=True, slots=True)
class CandidateQuality:
    candidate_id: str
    source: CandidateSource
    usable: bool
    model_point_px_median: float | None
    model_point_px_p95: float | None
    primitives: tuple[PrimitiveQuality, ...]
    supported_families: tuple[MarkingFamily, ...]
    max_depth_ft: float
    held_out_px_median: float | None
    leave_one_primitive_shift_px: float | None
    gates: tuple[GateResult, ...]
    reasons: tuple[str, ...] = ()
    schema_version: str = HYBRID_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema({"schema_version": self.schema_version})
        _nonempty(self.candidate_id, "candidate_id")
        for name in ("model_point_px_median", "model_point_px_p95", "held_out_px_median", "leave_one_primitive_shift_px"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite(value, name, minimum=0))
        object.__setattr__(self, "max_depth_ft", _finite(self.max_depth_ft, "max_depth_ft", minimum=0))
        if len({item.feature for item in self.primitives}) != len(self.primitives):
            raise ValueError("candidate primitive qualities must have unique features")
        if len(set(self.supported_families)) != len(self.supported_families):
            raise ValueError("supported_families must be unique")
        if len({gate.gate_id for gate in self.gates}) != len(self.gates):
            raise ValueError("candidate gates must have unique IDs")
        if self.usable and any(not gate.passed for gate in self.gates):
            raise ValueError("usable candidate cannot contain a failed gate")
        if not self.usable and not self.reasons:
            raise ValueError("unusable candidate needs a reason")

    def as_dict(self) -> dict:
        return {
            "schema_version": self.schema_version, "candidate_id": self.candidate_id,
            "source": self.source.value, "usable": self.usable,
            "model_point_px_median": self.model_point_px_median,
            "model_point_px_p95": self.model_point_px_p95,
            "primitives": [item.as_dict() for item in self.primitives],
            "supported_families": [family.value for family in self.supported_families],
            "max_depth_ft": self.max_depth_ft, "held_out_px_median": self.held_out_px_median,
            "leave_one_primitive_shift_px": self.leave_one_primitive_shift_px,
            "gates": [gate.as_dict() for gate in self.gates], "reasons": list(self.reasons),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CandidateQuality":
        _schema(data)
        _fields(data, {"schema_version", "candidate_id", "source", "usable", "model_point_px_median", "model_point_px_p95", "primitives", "supported_families", "max_depth_ft", "held_out_px_median", "leave_one_primitive_shift_px", "gates", "reasons"})
        return cls(
            candidate_id=str(data["candidate_id"]), source=CandidateSource(data["source"]),
            usable=bool(data["usable"]),
            model_point_px_median=None if data["model_point_px_median"] is None else float(data["model_point_px_median"]),
            model_point_px_p95=None if data["model_point_px_p95"] is None else float(data["model_point_px_p95"]),
            primitives=tuple(PrimitiveQuality.from_dict(item) for item in data["primitives"]),
            supported_families=tuple(MarkingFamily(item) for item in data["supported_families"]),
            max_depth_ft=float(data["max_depth_ft"]),
            held_out_px_median=None if data["held_out_px_median"] is None else float(data["held_out_px_median"]),
            leave_one_primitive_shift_px=None if data["leave_one_primitive_shift_px"] is None else float(data["leave_one_primitive_shift_px"]),
            gates=tuple(GateResult.from_dict(item) for item in data["gates"]),
            reasons=tuple(str(item) for item in data["reasons"]), schema_version=str(data["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    frame_id: str
    baseline_candidate_id: str
    challenger_candidate_id: str | None
    selected_candidate_id: str | None
    outcome: SelectionOutcome
    gates: tuple[GateResult, ...]
    reasons: tuple[str, ...]
    schema_version: str = HYBRID_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema({"schema_version": self.schema_version})
        _nonempty(self.frame_id, "frame_id")
        _nonempty(self.baseline_candidate_id, "baseline_candidate_id")
        if self.challenger_candidate_id is not None:
            _nonempty(self.challenger_candidate_id, "challenger_candidate_id")
            if self.challenger_candidate_id == self.baseline_candidate_id:
                raise ValueError("baseline and challenger IDs must differ")
        if self.outcome is SelectionOutcome.KEEP_BASELINE:
            if self.selected_candidate_id != self.baseline_candidate_id:
                raise ValueError("keep_baseline must select the baseline candidate")
        elif self.outcome is SelectionOutcome.SELECT_CHALLENGER:
            if self.challenger_candidate_id is None or self.selected_candidate_id != self.challenger_candidate_id:
                raise ValueError("select_challenger must select the declared challenger")
        elif self.selected_candidate_id is not None:
            raise ValueError("no_usable_candidate cannot select a candidate")
        if len({gate.gate_id for gate in self.gates}) != len(self.gates):
            raise ValueError("selection gates must have unique IDs")
        if not self.reasons:
            raise ValueError("selection decision needs at least one reason")

    def as_dict(self) -> dict:
        return {
            "schema_version": self.schema_version, "frame_id": self.frame_id,
            "baseline_candidate_id": self.baseline_candidate_id,
            "challenger_candidate_id": self.challenger_candidate_id,
            "selected_candidate_id": self.selected_candidate_id, "outcome": self.outcome.value,
            "gates": [gate.as_dict() for gate in self.gates], "reasons": list(self.reasons),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SelectionDecision":
        _schema(data)
        _fields(data, {"schema_version", "frame_id", "baseline_candidate_id", "challenger_candidate_id", "selected_candidate_id", "outcome", "gates", "reasons"})
        return cls(
            frame_id=str(data["frame_id"]), baseline_candidate_id=str(data["baseline_candidate_id"]),
            challenger_candidate_id=None if data["challenger_candidate_id"] is None else str(data["challenger_candidate_id"]),
            selected_candidate_id=None if data["selected_candidate_id"] is None else str(data["selected_candidate_id"]),
            outcome=SelectionOutcome(data["outcome"]),
            gates=tuple(GateResult.from_dict(item) for item in data["gates"]),
            reasons=tuple(str(item) for item in data["reasons"]), schema_version=str(data["schema_version"]),
        )


def _matrix(value, name: str) -> Matrix3x3:
    """Validate a 3x3 of finite floats, normalised so ``m[2][2] == 1``.

    Two byte-different records must not be able to denote the same transform, so
    the w-normalisation convention of ``calibration.estimator._as_matrix`` is a
    contract invariant rather than a producer's habit.
    """
    try:
        rows = tuple(tuple(float(cell) for cell in row) for row in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a 3x3 numeric matrix") from error
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError(f"{name} must be a 3x3 matrix")
    for row_index, row in enumerate(rows):
        for column, cell in enumerate(row):
            _finite(cell, f"{name}[{row_index}][{column}]")
    if abs(rows[2][2] - 1.0) > 1e-12:
        raise ValueError(f"{name} must be normalised so that m[2][2] == 1, got {rows[2][2]!r}")
    return rows  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class CandidateTransform:
    """A candidate transform pair recorded outside ``CourtCalibration``.

    ``CourtCalibration`` enforces "usable status implies both transforms
    present", so a *rejected* challenger cannot be expressed as one at all.
    Shadow-mode audit has to keep exactly those transforms. ``CourtCalibration``
    therefore remains the sole home of a transform a consumer may **use**; this
    is the audit record of one that has not been promoted.

    ``layout_hash`` sits here rather than only on the frame because benchmark
    scoring refuses a transform whose layout hash is missing or wrong, and that
    guard must read the hash off the same object that carries the matrix.
    """

    candidate_id: str
    source: CandidateSource
    layout_id: str
    layout_hash: str
    h_court_to_image: Matrix3x3 | None
    h_image_to_court: Matrix3x3 | None
    round_trip_error_px: float | None
    solver: str
    failure: str | None = None
    schema_version: str = HYBRID_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema({"schema_version": self.schema_version})
        for name in ("candidate_id", "layout_id", "layout_hash", "solver"):
            _nonempty(getattr(self, name), name)
        solved = self.h_court_to_image is not None
        if solved != (self.h_image_to_court is not None):
            raise ValueError("both transforms must be present or both absent")
        if solved:
            object.__setattr__(self, "h_court_to_image", _matrix(self.h_court_to_image, "h_court_to_image"))
            object.__setattr__(self, "h_image_to_court", _matrix(self.h_image_to_court, "h_image_to_court"))
            if self.failure:
                raise ValueError("a solved candidate transform must not also declare a failure")
            if self.round_trip_error_px is None:
                raise ValueError("a solved candidate transform needs a round-trip error")
            object.__setattr__(
                self, "round_trip_error_px",
                _finite(self.round_trip_error_px, "round_trip_error_px", minimum=0),
            )
        else:
            if not self.failure:
                raise ValueError("an unsolved candidate transform needs a failure reason")
            if self.round_trip_error_px is not None:
                object.__setattr__(
                    self, "round_trip_error_px",
                    _finite(self.round_trip_error_px, "round_trip_error_px", minimum=0),
                )

    @property
    def solved(self) -> bool:
        return self.h_court_to_image is not None

    def as_dict(self) -> dict:
        return {
            "schema_version": self.schema_version, "candidate_id": self.candidate_id,
            "source": self.source.value, "layout_id": self.layout_id,
            "layout_hash": self.layout_hash,
            "h_court_to_image": None if self.h_court_to_image is None
            else [list(row) for row in self.h_court_to_image],
            "h_image_to_court": None if self.h_image_to_court is None
            else [list(row) for row in self.h_image_to_court],
            "round_trip_error_px": self.round_trip_error_px, "solver": self.solver,
            "failure": self.failure,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CandidateTransform":
        _schema(data)
        _fields(data, {
            "schema_version", "candidate_id", "source", "layout_id", "layout_hash",
            "h_court_to_image", "h_image_to_court", "round_trip_error_px", "solver", "failure",
        })
        return cls(
            candidate_id=str(data["candidate_id"]), source=CandidateSource(data["source"]),
            layout_id=str(data["layout_id"]), layout_hash=str(data["layout_hash"]),
            h_court_to_image=None if data["h_court_to_image"] is None
            else _matrix(data["h_court_to_image"], "h_court_to_image"),
            h_image_to_court=None if data["h_image_to_court"] is None
            else _matrix(data["h_image_to_court"], "h_image_to_court"),
            round_trip_error_px=None if data["round_trip_error_px"] is None
            else float(data["round_trip_error_px"]),
            solver=str(data["solver"]),
            failure=None if data["failure"] is None else str(data["failure"]),
            schema_version=str(data["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class ProbeDisplacement:
    """How far a depth band's canonical court points moved between candidates."""

    band: str
    point_count: int
    median_px: float
    max_px: float

    def __post_init__(self) -> None:
        _nonempty(self.band, "band")
        if self.point_count < 0:
            raise ValueError("point_count must be non-negative")
        object.__setattr__(self, "median_px", _finite(self.median_px, "median_px", minimum=0))
        object.__setattr__(self, "max_px", _finite(self.max_px, "max_px", minimum=0))
        if self.max_px < self.median_px:
            raise ValueError("max_px must be >= median_px")
        if self.point_count == 0 and (self.median_px or self.max_px):
            raise ValueError("a band without points cannot report displacement")

    def as_dict(self) -> dict:
        return {
            "band": self.band, "point_count": self.point_count,
            "median_px": self.median_px, "max_px": self.max_px,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProbeDisplacement":
        _fields(data, {"band", "point_count", "median_px", "max_px"})
        return cls(
            band=str(data["band"]), point_count=int(data["point_count"]),
            median_px=float(data["median_px"]), max_px=float(data["max_px"]),
        )


@dataclass(frozen=True, slots=True)
class TransformDifference:
    """How far apart two candidates are, in projected canonical court points.

    Matrix-element differences would be meaningless: a homography is
    scale-equivalent and its entries carry incommensurable units, so a Frobenius
    norm is dominated by translation. Court-feet displacement of image points
    would require inverting through a transform that is itself under test.
    Projected court points are in pixels, are directly interpretable, and are
    what the plan's selection rule actually asks about.
    """

    baseline_candidate_id: str
    challenger_candidate_id: str
    probe_point_count: int
    probe_px_median: float
    probe_px_p95: float
    probe_px_max: float
    max_control_point_shift_ft: float
    bounds_active: int
    scale_ratio: float
    bands: tuple[ProbeDisplacement, ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.baseline_candidate_id, "baseline_candidate_id")
        _nonempty(self.challenger_candidate_id, "challenger_candidate_id")
        if self.baseline_candidate_id == self.challenger_candidate_id:
            raise ValueError("baseline and challenger IDs must differ")
        if self.probe_point_count < 0:
            raise ValueError("probe_point_count must be non-negative")
        if not 0 <= self.bounds_active <= 4:
            raise ValueError("bounds_active must be in 0..4")
        for name in ("probe_px_median", "probe_px_p95", "probe_px_max", "max_control_point_shift_ft"):
            object.__setattr__(self, name, _finite(getattr(self, name), name, minimum=0))
        if not self.probe_px_median <= self.probe_px_p95 <= self.probe_px_max:
            raise ValueError("probe displacements must satisfy median <= p95 <= max")
        object.__setattr__(self, "scale_ratio", _finite(self.scale_ratio, "scale_ratio", minimum=0))
        if self.scale_ratio == 0:
            raise ValueError("scale_ratio must be positive")
        if len({item.band for item in self.bands}) != len(self.bands):
            raise ValueError("difference bands must be unique")

    def as_dict(self) -> dict:
        return {
            "baseline_candidate_id": self.baseline_candidate_id,
            "challenger_candidate_id": self.challenger_candidate_id,
            "probe_point_count": self.probe_point_count,
            "probe_px_median": self.probe_px_median, "probe_px_p95": self.probe_px_p95,
            "probe_px_max": self.probe_px_max,
            "max_control_point_shift_ft": self.max_control_point_shift_ft,
            "bounds_active": self.bounds_active, "scale_ratio": self.scale_ratio,
            "bands": [item.as_dict() for item in self.bands],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TransformDifference":
        _fields(data, {
            "baseline_candidate_id", "challenger_candidate_id", "probe_point_count",
            "probe_px_median", "probe_px_p95", "probe_px_max", "max_control_point_shift_ft",
            "bounds_active", "scale_ratio", "bands",
        })
        return cls(
            baseline_candidate_id=str(data["baseline_candidate_id"]),
            challenger_candidate_id=str(data["challenger_candidate_id"]),
            probe_point_count=int(data["probe_point_count"]),
            probe_px_median=float(data["probe_px_median"]),
            probe_px_p95=float(data["probe_px_p95"]),
            probe_px_max=float(data["probe_px_max"]),
            max_control_point_shift_ft=float(data["max_control_point_shift_ft"]),
            bounds_active=int(data["bounds_active"]), scale_ratio=float(data["scale_ratio"]),
            bands=tuple(ProbeDisplacement.from_dict(item) for item in data["bands"]),
        )


@dataclass(frozen=True, slots=True)
class PrimitiveStability:
    """How far the transform moves when one fitted primitive is withheld.

    A high raw inlier count is not a confidence measure when the inliers are
    correlated pixels from one painted line. This is the measurement that tells
    the two situations apart.
    """

    feature: MarkingFeature
    shift_px: float
    refit_converged: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "shift_px", _finite(self.shift_px, "shift_px", minimum=0))

    @property
    def family(self) -> MarkingFamily:
        return self.feature.family

    def as_dict(self) -> dict:
        return {
            "feature": self.feature.value, "family": self.family.value,
            "shift_px": self.shift_px, "refit_converged": self.refit_converged,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PrimitiveStability":
        _fields(data, {"feature", "family", "shift_px", "refit_converged"})
        feature = MarkingFeature(data["feature"])
        if data["family"] != feature.family.value:
            raise ValueError("primitive stability family disagrees with feature")
        return cls(
            feature=feature, shift_px=float(data["shift_px"]),
            refit_converged=bool(data["refit_converged"]),
        )


@dataclass(frozen=True, slots=True)
class HybridRefinementFrame:
    """One immutable shadow refinement result for one image.

    Carries both candidates and every diagnostic needed to compare them, so a
    benchmark can judge the pair from serialized records alone without importing
    anything that predicts.
    """

    frame_id: str
    image_sha256: str
    status: HybridFrameStatus
    eligible: bool
    eligibility_reasons: tuple[str, ...]
    layout_id: str
    layout_hash: str
    shadow_config_hash: str
    refiner_config_hash: str
    baseline_candidate_id: str
    challenger_candidate_id: str | None
    transforms: tuple[CandidateTransform, ...] = ()
    candidates: tuple[CandidateQuality, ...] = ()
    difference: TransformDifference | None = None
    selection: SelectionDecision | None = None
    stability: tuple[PrimitiveStability, ...] = ()
    fitted_features: tuple[MarkingFeature, ...] = ()
    held_out_features: tuple[MarkingFeature, ...] = ()
    outer_iterations: int = 0
    inner_iterations: int = 0
    converged: bool = False
    refine_ms: float = 0.0
    diagnostics_ms: float = 0.0
    failure_reasons: tuple[str, ...] = ()
    schema_version: str = HYBRID_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema({"schema_version": self.schema_version})
        for name in (
            "frame_id", "image_sha256", "layout_id", "layout_hash",
            "shadow_config_hash", "refiner_config_hash", "baseline_candidate_id",
        ):
            _nonempty(getattr(self, name), name)
        for name in ("outer_iterations", "inner_iterations"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("refine_ms", "diagnostics_ms"):
            object.__setattr__(self, name, _finite(getattr(self, name), name, minimum=0))
        if any(not item for item in self.eligibility_reasons + self.failure_reasons):
            raise ValueError("frame reasons must not be empty")
        if not self.eligible and not self.eligibility_reasons:
            raise ValueError("ineligible frame needs eligibility reasons")

        transform_ids = [item.candidate_id for item in self.transforms]
        candidate_ids = [item.candidate_id for item in self.candidates]
        if len(set(transform_ids)) != len(transform_ids):
            raise ValueError("candidate transforms must have unique IDs")
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate qualities must have unique IDs")
        if set(transform_ids) != set(candidate_ids):
            raise ValueError("every candidate needs a transform and every transform a candidate")
        known = set(candidate_ids)
        if self.baseline_candidate_id not in known:
            raise ValueError("baseline candidate ID is not among the frame's candidates")
        keypoints = [item for item in self.transforms if item.source is CandidateSource.KEYPOINT]
        if len(keypoints) != 1:
            raise ValueError("a frame must record exactly one keypoint candidate")
        if keypoints[0].candidate_id != self.baseline_candidate_id:
            raise ValueError("the keypoint candidate must be the baseline")
        if self.challenger_candidate_id is not None:
            _nonempty(self.challenger_candidate_id, "challenger_candidate_id")
            if self.challenger_candidate_id == self.baseline_candidate_id:
                raise ValueError("baseline and challenger IDs must differ")
            if self.challenger_candidate_id not in known:
                raise ValueError("challenger candidate ID is not among the frame's candidates")
        else:
            if self.difference is not None:
                raise ValueError("a frame without a challenger cannot record a difference")
            if self.fitted_features or self.stability:
                raise ValueError("a frame without a challenger cannot record fitted primitives")

        if self.difference is not None:
            if (
                self.difference.baseline_candidate_id != self.baseline_candidate_id
                or self.difference.challenger_candidate_id != self.challenger_candidate_id
            ):
                raise ValueError("transform difference IDs disagree with the frame")
        if self.selection is not None:
            if self.selection.frame_id != self.frame_id:
                raise ValueError("selection frame ID disagrees with the frame")
            if self.selection.baseline_candidate_id != self.baseline_candidate_id:
                raise ValueError("selection baseline disagrees with the frame")
            if self.selection.challenger_candidate_id != self.challenger_candidate_id:
                raise ValueError("selection challenger disagrees with the frame")
            if self.selection.outcome is SelectionOutcome.SELECT_CHALLENGER:
                chosen = {item.candidate_id: item for item in self.candidates}
                challenger = chosen.get(self.selection.selected_candidate_id or "")
                if challenger is None or not challenger.usable:
                    raise ValueError("cannot select a challenger that is not usable")

        if len(set(self.fitted_features)) != len(self.fitted_features):
            raise ValueError("fitted_features must be unique")
        if len(set(self.held_out_features)) != len(self.held_out_features):
            raise ValueError("held_out_features must be unique")
        if set(self.fitted_features) & set(self.held_out_features):
            raise ValueError("a feature cannot be both fitted and held out")
        stability_features = [item.feature for item in self.stability]
        if len(set(stability_features)) != len(stability_features):
            raise ValueError("stability entries must have unique features")
        if not set(stability_features) <= set(self.fitted_features):
            raise ValueError("stability may only report fitted primitives")

        if self.status is HybridFrameStatus.OK and not self.candidates:
            raise ValueError("an OK frame needs at least one candidate")
        if self.status is HybridFrameStatus.ABSTAINED:
            if not self.eligibility_reasons:
                raise ValueError("an abstained frame needs eligibility reasons")
            if self.challenger_candidate_id is not None:
                raise ValueError("an abstained frame cannot record a challenger")
        if self.status is HybridFrameStatus.REFINEMENT_FAILED:
            if not self.failure_reasons:
                raise ValueError("a failed frame needs failure reasons")
            if self.challenger_candidate_id is not None:
                raise ValueError("a failed frame cannot record a challenger")

    def as_dict(self) -> dict:
        return {
            "schema_version": self.schema_version, "frame_id": self.frame_id,
            "image_sha256": self.image_sha256, "status": self.status.value,
            "eligible": self.eligible,
            "eligibility_reasons": list(self.eligibility_reasons),
            "layout_id": self.layout_id, "layout_hash": self.layout_hash,
            "shadow_config_hash": self.shadow_config_hash,
            "refiner_config_hash": self.refiner_config_hash,
            "baseline_candidate_id": self.baseline_candidate_id,
            "challenger_candidate_id": self.challenger_candidate_id,
            "transforms": [item.as_dict() for item in self.transforms],
            "candidates": [item.as_dict() for item in self.candidates],
            "difference": None if self.difference is None else self.difference.as_dict(),
            "selection": None if self.selection is None else self.selection.as_dict(),
            "stability": [item.as_dict() for item in self.stability],
            "fitted_features": [feature.value for feature in self.fitted_features],
            "held_out_features": [feature.value for feature in self.held_out_features],
            "outer_iterations": self.outer_iterations,
            "inner_iterations": self.inner_iterations, "converged": self.converged,
            "refine_ms": self.refine_ms, "diagnostics_ms": self.diagnostics_ms,
            "failure_reasons": list(self.failure_reasons),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HybridRefinementFrame":
        _schema(data)
        _fields(data, {
            "schema_version", "frame_id", "image_sha256", "status", "eligible",
            "eligibility_reasons", "layout_id", "layout_hash", "shadow_config_hash",
            "refiner_config_hash", "baseline_candidate_id", "challenger_candidate_id",
            "transforms", "candidates", "difference", "selection", "stability",
            "fitted_features", "held_out_features", "outer_iterations",
            "inner_iterations", "converged", "refine_ms", "diagnostics_ms",
            "failure_reasons",
        })
        return cls(
            frame_id=str(data["frame_id"]), image_sha256=str(data["image_sha256"]),
            status=HybridFrameStatus(data["status"]), eligible=bool(data["eligible"]),
            eligibility_reasons=tuple(str(item) for item in data["eligibility_reasons"]),
            layout_id=str(data["layout_id"]), layout_hash=str(data["layout_hash"]),
            shadow_config_hash=str(data["shadow_config_hash"]),
            refiner_config_hash=str(data["refiner_config_hash"]),
            baseline_candidate_id=str(data["baseline_candidate_id"]),
            challenger_candidate_id=None if data["challenger_candidate_id"] is None
            else str(data["challenger_candidate_id"]),
            transforms=tuple(CandidateTransform.from_dict(item) for item in data["transforms"]),
            candidates=tuple(CandidateQuality.from_dict(item) for item in data["candidates"]),
            difference=None if data["difference"] is None
            else TransformDifference.from_dict(data["difference"]),
            selection=None if data["selection"] is None
            else SelectionDecision.from_dict(data["selection"]),
            stability=tuple(PrimitiveStability.from_dict(item) for item in data["stability"]),
            fitted_features=tuple(MarkingFeature(item) for item in data["fitted_features"]),
            held_out_features=tuple(MarkingFeature(item) for item in data["held_out_features"]),
            outer_iterations=int(data["outer_iterations"]),
            inner_iterations=int(data["inner_iterations"]),
            converged=bool(data["converged"]), refine_ms=float(data["refine_ms"]),
            diagnostics_ms=float(data["diagnostics_ms"]),
            failure_reasons=tuple(str(item) for item in data["failure_reasons"]),
            schema_version=str(data["schema_version"]),
        )
