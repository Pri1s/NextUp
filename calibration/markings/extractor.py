"""Polarity-independent static-frame ridge extraction.

The extractor has no court vocabulary.  It returns geometric evidence only;
identity is assigned by :mod:`association` using a calibrated layout.
OpenCV is imported lazily so the contracts and CLI parser remain usable in
environments where the optional image runtime is not installed.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from contracts.calibration_types import Matrix3x3
from contracts.court_layout import HalfCourtLayout
from contracts.hybrid_types import EvidenceKind, ImageMarkingSample, UnlabeledMarkingEvidence

REFERENCE_DIAGONAL = math.hypot(1280.0, 720.0)


def _scale(shape: tuple[int, int] | tuple[int, int, int]) -> float:
    height, width = int(shape[0]), int(shape[1])
    return math.hypot(width, height) / REFERENCE_DIAGONAL


def diagonal_scale(image_shape: tuple[int, int] | tuple[int, int, int]) -> float:
    """Scale a locked 1280x720 pixel constant by the frame diagonal."""
    return _scale(image_shape)


def corridor_width_px(
    outside_hull_distance_ft: float,
    image_shape: tuple[int, int] | tuple[int, int, int] = (720, 1280),
) -> float:
    """The registered corridor policy, scaled by frame diagonal."""
    if outside_hull_distance_ft < 0 or not math.isfinite(outside_hull_distance_ft):
        raise ValueError("outside_hull_distance_ft must be finite and non-negative")
    scale = _scale(image_shape)
    return min(35.0 * scale, (12.0 + 1.5 * outside_hull_distance_ft) * scale)


@dataclass(frozen=True, slots=True)
class MarkingExtractorConfig:
    ridge_percentile: float = 94.0
    scales_px: tuple[float, ...] = (1.0, 2.0, 3.5, 5.0)
    min_support_px: float = 12.0
    resample_spacing_px: float = 2.0
    min_local_contrast: float = 0.035
    min_confidence: float = 0.08

    def __post_init__(self) -> None:
        if not 50.0 < self.ridge_percentile < 100.0:
            raise ValueError("ridge_percentile must be in (50, 100)")
        if not self.scales_px or any(value <= 0 for value in self.scales_px):
            raise ValueError("scales_px must contain positive values")
        for name in ("min_support_px", "resample_spacing_px", "min_local_contrast", "min_confidence"):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")

    def as_dict(self) -> dict[str, Any]:
        return {
            "ridge_percentile": self.ridge_percentile,
            "scales_px": list(self.scales_px),
            "min_support_px": self.min_support_px,
            "resample_spacing_px": self.resample_spacing_px,
            "min_local_contrast": self.min_local_contrast,
            "min_confidence": self.min_confidence,
        }

    def content_hash(self) -> str:
        encoded = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


def validate_mask(mask: np.ndarray | None, shape: tuple[int, int], name: str) -> np.ndarray:
    """Return a boolean mask, rejecting non-exact dimensions."""
    if mask is None:
        return np.zeros(shape, dtype=bool)
    value = np.asarray(mask)
    if value.shape != shape:
        raise ValueError(f"{name} must have exact frame shape {shape}, got {value.shape}")
    return value != 0


def build_projected_roi_mask(
    image_shape: tuple[int, int] | tuple[int, int, int],
    layout: HalfCourtLayout,
    h_court_to_image: Matrix3x3 | np.ndarray,
) -> np.ndarray:
    """Rasterize the projected half-court polygon as the mandatory ROI."""
    import cv2

    height, width = int(image_shape[0]), int(image_shape[1])
    matrix = np.asarray(h_court_to_image, dtype=np.float64)
    court = np.asarray(
        [[0.0, 0.0], [layout.half_length, 0.0],
         [layout.half_length, layout.width], [0.0, layout.width]], dtype=np.float64
    )
    from calibration.estimator import project

    projected = project(matrix, court)
    if projected is None or not np.all(np.isfinite(projected)):
        raise ValueError("projected half-court ROI is not finite")
    polygon = np.round(projected).astype(np.int32).reshape(-1, 1, 2)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return mask.astype(bool)


class MarkingExtractor:
    """Extract unlabeled, deterministic ridge fragments from one BGR frame."""

    def __init__(self, config: MarkingExtractorConfig | None = None):
        self.config = config or MarkingExtractorConfig()

    @property
    def config_hash(self) -> str:
        return self.config.content_hash()

    def extract(
        self,
        frame: np.ndarray,
        frame_id: str,
        roi_mask: np.ndarray,
        exclusion_mask: np.ndarray | None = None,
    ) -> tuple[UnlabeledMarkingEvidence, ...]:
        """Return evidence only; this method never assigns a court identity."""
        import cv2

        image = np.asarray(frame)
        if image.ndim not in (2, 3) or image.shape[0] <= 0 or image.shape[1] <= 0:
            raise ValueError("frame must be a non-empty grayscale or BGR image")
        shape = (int(image.shape[0]), int(image.shape[1]))
        roi = validate_mask(roi_mask, shape, "roi_mask")
        excluded = validate_mask(exclusion_mask, shape, "exclusion_mask")
        valid = roi & ~excluded
        if not np.any(valid):
            return ()

        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        else:
            gray = image.astype(np.float32) / 255.0
        scale = _scale(image.shape)
        response = np.zeros(shape, dtype=np.float32)
        tangent_x = np.zeros(shape, dtype=np.float32)
        tangent_y = np.zeros(shape, dtype=np.float32)
        observed_width = np.zeros(shape, dtype=np.float32)

        for raw_sigma in self.config.scales_px:
            sigma = max(0.6, float(raw_sigma) * scale)
            blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
            dxx = cv2.Sobel(blurred, cv2.CV_32F, 2, 0, ksize=3)
            dyy = cv2.Sobel(blurred, cv2.CV_32F, 0, 2, ksize=3)
            dxy = cv2.Sobel(blurred, cv2.CV_32F, 1, 1, ksize=3)
            trace = dxx + dyy
            discriminant = np.sqrt(np.maximum((dxx - dyy) ** 2 + 4.0 * dxy ** 2, 0.0))
            lambda_normal = 0.5 * (trace + np.sign(trace + 1e-12) * discriminant)
            # The eigenvector of the strongest-curvature direction is normal to
            # the stripe.  The tangent is its perpendicular and is unoriented.
            nx = dxy
            ny = lambda_normal - dxx
            norm = np.hypot(nx, ny)
            bad = norm < 1e-5
            nx = np.divide(nx, norm, out=np.zeros_like(nx), where=~bad)
            ny = np.divide(ny, norm, out=np.zeros_like(ny), where=~bad)
            tx, ty = -ny, nx

            # A three-pixel minimum keeps the paired samples outside a normal
            # anti-aliased stripe even on small unit-test/surveillance frames;
            # all policy thresholds and resampling distances remain diagonal
            # scaled below.
            radius = max(3, int(round(1.5 * sigma)))
            yy, xx = np.indices(shape)
            plus_x = np.clip(np.rint(xx + nx * radius).astype(np.int32), 0, shape[1] - 1)
            plus_y = np.clip(np.rint(yy + ny * radius).astype(np.int32), 0, shape[0] - 1)
            minus_x = np.clip(np.rint(xx - nx * radius).astype(np.int32), 0, shape[1] - 1)
            minus_y = np.clip(np.rint(yy - ny * radius).astype(np.int32), 0, shape[0] - 1)
            center = blurred
            side_mean = (blurred[plus_y, plus_x] + blurred[minus_y, minus_x]) / 2.0
            contrast = np.abs(center - side_mean)
            # A stripe center is bright/dark relative to both sides.  The small
            # Hessian term localizes the centerline without allowing the second
            # derivative to erase the flat middle of a wide painted stripe.
            curvature = np.abs(lambda_normal)
            curvature /= max(float(np.percentile(curvature[valid], 95)), 1e-6)
            ridge = contrast * (0.8 + 0.2 * np.minimum(curvature, 1.0))
            ridge[~valid] = 0.0
            replace = ridge > response
            response[replace] = ridge[replace]
            tangent_x[replace], tangent_y[replace] = tx[replace], ty[replace]
            observed_width[replace] = max(1.0, 2.0 * sigma)

        candidates = response[valid]
        if not len(candidates) or float(np.max(candidates)) <= 0:
            return ()
        threshold = max(float(np.percentile(candidates, self.config.ridge_percentile)), 1e-5)
        contrast_mask = response >= threshold
        contrast_mask &= valid
        # Suppress isolated noise and make connected components deterministic.
        kernel = np.ones((3, 3), dtype=np.uint8)
        contrast_mask = cv2.morphologyEx(contrast_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel) != 0
        count, labels, stats, _ = cv2.connectedComponentsWithStats(contrast_mask.astype(np.uint8), 8)
        fragments: list[UnlabeledMarkingEvidence] = []
        min_support = self.config.min_support_px * scale
        spacing = self.config.resample_spacing_px * scale
        for label in range(1, count):
            ys, xs = np.where(labels == label)
            if len(xs) < 3:
                continue
            points = np.column_stack((xs.astype(float), ys.astype(float)))
            centered = points - points.mean(axis=0)
            _, singular, vh = np.linalg.svd(centered, full_matrices=False)
            support = float(singular[0] * 2.0) if len(singular) else 0.0
            if support < min_support:
                continue
            # Order by the dominant axis; this is deterministic and works for
            # short curved fragments without inventing endpoint landmarks.
            axis = vh[0] if len(vh) else np.array([1.0, 0.0])
            order = np.argsort(points @ axis)
            points = points[order]
            weights = response[ys[order], xs[order]]
            points = _centerline_points(points, weights, axis, spacing)
            if len(points) < 2:
                continue
            samples: list[ImageMarkingSample] = []
            for x, y in points:
                ix, iy = int(round(x)), int(round(y))
                ix = min(max(ix, 0), shape[1] - 1)
                iy = min(max(iy, 0), shape[0] - 1)
                tangent = _unit((float(tangent_x[iy, ix]), float(tangent_y[iy, ix])))
                local = float(response[iy, ix]) / max(float(np.max(candidates)), 1e-6)
                samples.append(ImageMarkingSample(
                    float(x), float(y), tangent, min(1.0, max(self.config.min_confidence, local)),
                    max(0.25, 0.75 * float(observed_width[iy, ix])),
                    max(1.0, float(observed_width[iy, ix])),
                ))
            if len(samples) < 2:
                continue
            digest = hashlib.sha1(
                (str(frame_id) + ":" + ":".join(f"{s.x:.3f},{s.y:.3f}" for s in samples)).encode()
            ).hexdigest()[:16]
            fragments.append(UnlabeledMarkingEvidence(
                evidence_id=f"{frame_id}:ridge:{digest}", frame_id=str(frame_id),
                source_id=f"hessian-steger:{self.config_hash[:12]}",
                kind=EvidenceKind.CURVED if _turning(samples) > 0.08 else EvidenceKind.STRAIGHT,
                samples=tuple(samples),
                confidence=float(np.mean([s.confidence for s in samples])),
            ))
        return tuple(sorted(fragments, key=lambda item: item.evidence_id))


def _unit(value: tuple[float, float]) -> tuple[float, float]:
    length = math.hypot(*value)
    return (1.0, 0.0) if length < 1e-9 else (value[0] / length, value[1] / length)


def _resample(points: np.ndarray, spacing: float) -> np.ndarray:
    if len(points) < 2:
        return points
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(distances)))
    if cumulative[-1] <= spacing:
        return points[[0, -1]]
    positions = np.arange(0.0, cumulative[-1] + spacing * 0.5, spacing)
    if positions[-1] < cumulative[-1]:
        positions = np.append(positions, cumulative[-1])
    x = np.interp(positions, cumulative, points[:, 0])
    y = np.interp(positions, cumulative, points[:, 1])
    return np.column_stack((x, y))


def _centerline_points(
    points: np.ndarray, weights: np.ndarray, axis: np.ndarray, spacing: float
) -> np.ndarray:
    """Collapse a stripe component across its normal into weighted centers."""
    coordinate = points @ axis
    start = float(np.min(coordinate))
    bins = np.floor((coordinate - start) / max(spacing, 1.0)).astype(int)
    centers = []
    for bucket in range(int(np.max(bins)) + 1):
        keep = bins == bucket
        if not np.any(keep):
            continue
        local_weights = np.maximum(weights[keep], 1e-6)
        centers.append(np.average(points[keep], axis=0, weights=local_weights))
    return np.asarray(centers, dtype=np.float64)


def _turning(samples: list[ImageMarkingSample]) -> float:
    if len(samples) < 3:
        return 0.0
    angles = [math.atan2(sample.tangent_xy[1], sample.tangent_xy[0]) for sample in samples if sample.tangent_xy]
    if len(angles) < 3:
        return 0.0
    return float(np.std(np.unwrap(angles)))
