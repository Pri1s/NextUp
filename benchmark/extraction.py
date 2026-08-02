"""Contract-only scoring for Milestone 3 marking extraction."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from contracts.annotations import FrameAnnotation
from contracts.hybrid_types import AssignmentStatus, ShadowMarkingFrame
from contracts.markings import MarkingFeature


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, int(round(fraction * len(ordered) + 0.5)) - 1))]


def _reference_segments(annotation_feature) -> list[np.ndarray]:
    points = np.asarray([[point.x, point.y] for point in annotation_feature.points], dtype=np.float64)
    if len(points) < 2:
        return []
    breaks = sorted({int(index) for index in annotation_feature.occluded_after if 0 <= int(index) < len(points) - 1})
    segments = []
    start = 0
    for stop in breaks:
        if stop + 1 - start >= 2:
            segments.append(points[start:stop + 1])
        start = stop + 1
    if len(points) - start >= 2:
        segments.append(points[start:])
    return segments


def _nearest_signed(points: np.ndarray, segments: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not segments:
        return np.full(len(points), np.inf), np.full(len(points), np.nan)
    result = np.full(len(points), np.inf)
    signed = np.full(len(points), np.nan)
    for segment in segments:
        for left, right in zip(segment[:-1], segment[1:]):
            vector = right - left
            denom = float(np.dot(vector, vector))
            fraction = np.clip(((points - left) @ vector) / max(denom, 1e-12), 0.0, 1.0)
            closest = left + fraction[:, None] * vector
            distance = np.linalg.norm(points - closest, axis=1)
            keep = distance < result
            cross = vector[0] * (points[:, 1] - closest[:, 1]) - vector[1] * (points[:, 0] - closest[:, 0])
            result[keep] = distance[keep]
            signed[keep] = cross[keep] / max(float(np.linalg.norm(vector)), 1e-12)
    return result, signed


def _nearest_distances(points: np.ndarray, segments: list[np.ndarray]) -> np.ndarray:
    return _nearest_signed(points, segments)[0]


def score_extraction(records_path: Path | str, references_dir: Path | str) -> dict:
    """Score only serialized shadow records and independent annotations."""
    records = []
    for line in Path(records_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(ShadowMarkingFrame.from_dict(json.loads(line)))
    references = {}
    for path in sorted(Path(references_dir).glob("*.json")):
        annotation = FrameAnnotation.from_dict(json.loads(path.read_text(encoding="utf-8")))
        references[annotation.frame_id] = annotation

    per_feature: dict[str, list[dict]] = {}
    per_frame: dict[str, dict] = {}
    per_clip: dict[str, list[dict]] = {}
    wrong_accepts = 0
    wrong_identity_accepts = 0
    skipped_accepts = 0
    all_errors: list[float] = []
    for record in records:
        annotation = references.get(record.frame_id)
        if annotation is None:
            raise ValueError(f"no human reference for shadow frame {record.frame_id}")
        if record.image_sha256 != annotation.image_sha256:
            raise ValueError(f"image hash mismatch for shadow frame {record.frame_id}")
        annotated = {item.feature: item for item in annotation.features}
        skipped = {item.feature for item in annotation.skipped}
        frame_rows = []
        for assignment in record.assignments:
            feature = assignment.selected_feature or (
                assignment.alternatives[0].feature if assignment.alternatives else MarkingFeature.BASELINE
            )
            evidence = next(item for item in record.evidence if item.evidence_id == assignment.evidence_id)
            points = np.asarray([[sample.x, sample.y] for sample in evidence.samples], dtype=np.float64)
            row = {
                "feature": feature.value, "status": assignment.status.value,
                "accepted": False, "sample_count": len(points),
            }
            if assignment.status is not AssignmentStatus.ACCEPTED or assignment.selected_feature is None:
                row["reason"] = "ambiguous" if assignment.status is AssignmentStatus.AMBIGUOUS else "rejected"
                frame_rows.append(row)
                per_feature.setdefault(feature.value, []).append(row)
                continue
            if feature in skipped or feature not in annotated:
                wrong_accepts += int(feature.family.value == "three_point")
                wrong_identity_accepts += 1
                skipped_accepts += 1
                row["reason"] = "skipped_or_unannotated"
                frame_rows.append(row)
                per_feature.setdefault(feature.value, []).append(row)
                continue
            distances, signed = _nearest_signed(points, _reference_segments(annotated[feature]))
            errors = [float(value) for value in distances if math.isfinite(float(value))]
            signed_errors = [float(value) for value in signed if math.isfinite(float(value))]
            all_errors.extend(errors)
            correct = sum(distance <= 4.0 for distance in errors) / max(len(errors), 1) >= 0.95
            if not correct and feature.family.value == "three_point":
                wrong_accepts += 1
            row.update({
                "accepted": correct, "sample_count": len(errors),
                "median_px": float(np.median(errors)) if errors else float("nan"),
                "p95_px": _percentile(errors, 0.95), "max_px": max(errors) if errors else float("nan"),
                "median_signed_px": float(np.median(signed_errors)) if signed_errors else float("nan"),
                "mean_signed_px": float(np.mean(signed_errors)) if signed_errors else float("nan"),
            })
            frame_rows.append(row)
            per_feature.setdefault(feature.value, []).append(row)
        frame_summary = {
            "frame_id": record.frame_id,
            "clip": annotation.clip,
            "eligible": record.eligible,
            "status": record.status.value,
            "accepted_fragments": sum(1 for item in record.assignments if item.status is AssignmentStatus.ACCEPTED),
            "rejected_fragments": sum(1 for item in record.assignments if item.status is AssignmentStatus.REJECTED),
            "ambiguous_fragments": sum(1 for item in record.assignments if item.status is AssignmentStatus.AMBIGUOUS),
            "abstentions": int(not record.eligible) + sum(1 for item in record.assignments if item.status is not AssignmentStatus.ACCEPTED),
            "rows": frame_rows,
        }
        per_frame[record.frame_id] = frame_summary
        per_clip.setdefault(annotation.clip, []).append(frame_summary)

    feature_summary = {}
    for feature, rows in sorted(per_feature.items()):
        errors = [row["median_px"] for row in rows if math.isfinite(row.get("median_px", float("nan")))]
        signed = [row["mean_signed_px"] for row in rows if math.isfinite(row.get("mean_signed_px", float("nan")))]
        feature_summary[feature] = {
            "fragment_count": len(rows), "accepted_fragment_count": sum(bool(row["accepted"]) for row in rows),
            "rejected_fragment_count": sum(row.get("status") == AssignmentStatus.REJECTED.value for row in rows),
            "ambiguous_fragment_count": sum(row.get("status") == AssignmentStatus.AMBIGUOUS.value for row in rows),
            "median_px": float(np.median(errors)) if errors else float("nan"),
            "p95_px": _percentile(errors, 0.95), "max_px": max(errors) if errors else float("nan"),
            "mean_signed_px": float(np.mean(signed)) if signed else float("nan"),
        }
    clip_summary = {
        clip: {
            "frame_count": len(rows),
            "fragment_count": sum(len(row["rows"]) for row in rows),
            "accepted_fragment_count": sum(sum(bool(item["accepted"]) for item in row["rows"]) for row in rows),
            "rejected_fragment_count": sum(sum(item.get("status") == AssignmentStatus.REJECTED.value for item in row["rows"]) for row in rows),
            "ambiguous_fragment_count": sum(sum(item.get("status") == AssignmentStatus.AMBIGUOUS.value for item in row["rows"]) for row in rows),
            "abstention_count": sum(row["abstentions"] for row in rows),
            "macro_acceptance": float(np.mean([bool(item["accepted"]) for row in rows for item in row["rows"]])) if any(row["rows"] for row in rows) else 0.0,
        }
        for clip, rows in sorted(per_clip.items())
    }
    family_summary: dict[str, dict] = {}
    for feature_name, summary in feature_summary.items():
        family = MarkingFeature(feature_name).family.value
        item = family_summary.setdefault(family, {
            "fragment_count": 0, "accepted_fragment_count": 0,
            "rejected_fragment_count": 0, "ambiguous_fragment_count": 0,
            "signed_values": [], "median_values": [],
        })
        for key in ("fragment_count", "accepted_fragment_count", "rejected_fragment_count", "ambiguous_fragment_count"):
            item[key] += summary[key]
        if math.isfinite(summary["mean_signed_px"]):
            item["signed_values"].append(summary["mean_signed_px"])
        if math.isfinite(summary["median_px"]):
            item["median_values"].append(summary["median_px"])
    for item in family_summary.values():
        item["mean_signed_px"] = float(np.mean(item.pop("signed_values"))) if item["signed_values"] else float("nan")
        item["median_px"] = float(np.median(item.pop("median_values"))) if item["median_values"] else float("nan")
    return {
        "schema_version": "benchmark-extraction-score-1.0.0",
        "records": len(records), "references": len(references),
        "wrong_three_point_accepts": wrong_accepts,
        "wrong_identity_accepts": wrong_identity_accepts,
        "skipped_or_unannotated_accepts": skipped_accepts,
        "all_sample_median_px": float(np.median(all_errors)) if all_errors else float("nan"),
        "all_sample_p95_px": _percentile(all_errors, 0.95),
        "features": feature_summary, "families": family_summary,
        "frames": per_frame, "clips": clip_summary,
    }
