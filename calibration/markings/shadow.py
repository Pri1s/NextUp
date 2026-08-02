"""Shadow-only orchestration, provenance checks, and artifact writing."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from contracts.calibration_types import CalibrationStatus, CourtCalibration
from contracts.court_layout import HalfCourtLayout
from contracts.hybrid_types import ShadowFrameStatus, ShadowMarkingFrame

from .association import AssociationConfig, MarkingAssociator
from .extractor import MarkingExtractor, MarkingExtractorConfig, build_projected_roi_mask, validate_mask


class ShadowProvenanceError(ValueError):
    """Raised before analysis when checkpoint provenance is inconsistent."""


@dataclass(frozen=True, slots=True)
class ShadowConfig:
    extractor: MarkingExtractorConfig = MarkingExtractorConfig()
    association: AssociationConfig = AssociationConfig()

    def as_dict(self) -> dict:
        return {"extractor": self.extractor.as_dict(), "association": self.association.as_dict()}

    def content_hash(self) -> str:
        encoded = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


def canonical_png_sha256(frame: np.ndarray) -> str:
    """Hash the exact PNG byte representation used by shadow artifacts."""
    import cv2

    ok, encoded = cv2.imencode(".png", np.asarray(frame))
    if not ok:
        raise ValueError("could not encode frame as canonical PNG")
    return hashlib.sha256(encoded.tobytes()).hexdigest()


def verify_checkpoint_hash(calibration: CourtCalibration, expected_sha256: str) -> None:
    """Require the calibrated checkpoint to be the evidence-map checkpoint."""
    actual = calibration.provenance.get("weights_sha256")
    if actual != expected_sha256:
        raise ShadowProvenanceError(
            "checkpoint hash mismatch before marking shadow analysis: "
            f"calibration={actual!r}, evidence_map={expected_sha256!r}"
        )


class ShadowOrchestrator:
    """Run marking analysis without changing a :class:`CourtCalibration`."""

    def __init__(
        self,
        layout: HalfCourtLayout,
        expected_checkpoint_sha256: str,
        config: ShadowConfig | None = None,
        extractor: MarkingExtractor | None = None,
    ):
        if not expected_checkpoint_sha256:
            raise ValueError("expected_checkpoint_sha256 is required")
        self.layout = layout
        self.expected_checkpoint_sha256 = expected_checkpoint_sha256
        self.config = config or ShadowConfig()
        self.extractor = extractor or MarkingExtractor(self.config.extractor)

    @property
    def config_hash(self) -> str:
        return self.config.content_hash()

    def analyze_frame(
        self,
        frame: np.ndarray,
        frame_id: str,
        calibration: CourtCalibration,
        exclusion_mask: np.ndarray | None = None,
        *,
        roi_mask: np.ndarray | None = None,
    ) -> ShadowMarkingFrame:
        """Record every frame, returning an abstention for ineligible seeds."""
        verify_checkpoint_hash(calibration, self.expected_checkpoint_sha256)
        image = np.asarray(frame)
        if image.ndim not in (2, 3):
            raise ValueError("frame must be a grayscale or BGR image")
        height, width = int(image.shape[0]), int(image.shape[1])
        image_hash = canonical_png_sha256(image)
        exclusion = validate_mask(exclusion_mask, (height, width), "exclusion_mask")
        status = calibration.status
        reasons = self._eligibility_reasons(calibration)
        eligible = not reasons
        if not eligible:
            return ShadowMarkingFrame(
                frame_id=str(frame_id), image_sha256=image_hash,
                baseline_status=status.value, status=ShadowFrameStatus.ABSTAINED,
                eligible=False, eligibility_reasons=tuple(reasons),
                layout_id=self.layout.layout_id, layout_hash=self.layout.content_hash(),
                config_hash=self.config_hash, image_width=width, image_height=height,
                roi_pixels=0, excluded_pixels=int(np.count_nonzero(exclusion)),
            )

        if calibration.h_court_to_image is None:
            # Defensive: CourtCalibration already enforces this invariant.
            return self._failed(frame_id, image_hash, calibration, width, height, "missing_court_to_image")
        try:
            roi = (
                build_projected_roi_mask(image.shape, self.layout, calibration.h_court_to_image)
                if roi_mask is None else validate_mask(roi_mask, (height, width), "roi_mask")
            )
            started = time.perf_counter()
            evidence = self.extractor.extract(image, str(frame_id), roi, exclusion)
            extraction_ms = (time.perf_counter() - started) * 1000.0
            started = time.perf_counter()
            associated = MarkingAssociator(
                self.layout, calibration.h_court_to_image, calibration.h_image_to_court,
                self.config.association,
                image_shape=image.shape,
            ).associate(evidence)
            association_ms = (time.perf_counter() - started) * 1000.0
            return ShadowMarkingFrame(
                frame_id=str(frame_id), image_sha256=image_hash,
                baseline_status=status.value, status=ShadowFrameStatus.OK,
                eligible=True, eligibility_reasons=(),
                layout_id=self.layout.layout_id, layout_hash=self.layout.content_hash(),
                config_hash=self.config_hash, image_width=width, image_height=height,
                roi_pixels=int(np.count_nonzero(roi)), excluded_pixels=int(np.count_nonzero(exclusion)),
                evidence=tuple(evidence), assignments=associated.assignments,
                feature_support=associated.feature_support, family_support=associated.family_support,
                extraction_ms=extraction_ms, association_ms=association_ms,
            )
        except Exception as error:  # shadow failures must not fail calibration
            message = str(error).replace("\n", " ")
            return self._failed(
                frame_id, image_hash, calibration, width, height,
                f"{type(error).__name__}:{message[:120]}",
            )

    def _eligibility_reasons(self, calibration: CourtCalibration) -> list[str]:
        if calibration.status is not CalibrationStatus.DEGRADED:
            return [f"baseline_status:{calibration.status.value}"]
        quality = calibration.quality
        if quality is None:
            return ["missing_quality"]
        reasons: list[str] = []
        if quality.inlier_count < 5:
            reasons.append(f"inliers:{quality.inlier_count}<5")
        if not quality.reasons:
            reasons.append("degradation_not_weak_conditioning")
        elif any(not item.startswith("weak_conditioning:") for item in quality.reasons):
            reasons.append("degradation_not_weak_conditioning_only")
        return reasons

    def _failed(self, frame_id, image_hash, calibration, width, height, reason):
        return ShadowMarkingFrame(
            frame_id=str(frame_id), image_sha256=image_hash,
            baseline_status=calibration.status.value, status=ShadowFrameStatus.EXTRACTION_FAILED,
            eligible=True, eligibility_reasons=(), layout_id=self.layout.layout_id,
            layout_hash=self.layout.content_hash(), config_hash=self.config_hash,
            image_width=width, image_height=height, roi_pixels=0, excluded_pixels=0,
            failure_reasons=(str(reason),),
        )


def write_shadow_run(
    out_dir: Path | str,
    records: list[ShadowMarkingFrame] | tuple[ShadowMarkingFrame, ...],
    *,
    metadata: dict,
) -> Path:
    """Write the complete frame ledger, including zero-evidence frames."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "records.jsonl", "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.as_dict(), sort_keys=True))
            handle.write("\n")
    payload = dict(metadata)
    payload.setdefault("schema_version", "marking-shadow-run-1.0.0")
    payload["record_count"] = len(records)
    payload["status_counts"] = {
        status.value: sum(1 for item in records if item.status is status)
        for status in ShadowFrameStatus
    }
    (root / "run.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return root


def read_shadow_records(path: Path | str) -> list[ShadowMarkingFrame]:
    return [
        ShadowMarkingFrame.from_dict(json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()
    ]
