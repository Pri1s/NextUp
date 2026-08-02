"""Contract-only scorer for serialized Milestone 4 records."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from contracts.annotations import FrameAnnotation
from contracts.hybrid_types import CandidateSource, HybridRefinementFrame

from .metrics import compare_summaries, score_calibration_entry, summarize_scores

HYBRID_SCORE_SCHEMA_VERSION = "benchmark-hybrid-score-1.0.0"


def _records(path):
    source = Path(path)
    if source.is_dir():
        source = source / "records.jsonl"
    return [HybridRefinementFrame.from_dict(json.loads(line)) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_hybrid_transforms(records_path) -> dict[str, dict[str, dict]]:
    result = {}
    for record in _records(records_path):
        result[record.frame_id] = {
            item.source.value: {
                "candidate_id": item.candidate_id,
                "source": item.source.value,
                "layout_id": item.layout_id,
                "layout_hash": item.layout_hash,
                "h_court_to_image": item.h_court_to_image,
                "h_image_to_court": item.h_image_to_court,
                "status": record.status.value,
            }
            for item in record.transforms
        }
    return result


def _source_scores(records, references, layout, source, frame_ids, include_held_out):
    scores = []
    for record in records:
        if record.frame_id not in frame_ids:
            continue
        annotation = references.get(record.frame_id)
        if annotation is None:
            raise ValueError(f"no human reference for hybrid frame {record.frame_id}")
        if record.image_sha256 != annotation.image_sha256:
            raise ValueError(f"image_sha256 mismatch for hybrid frame {record.frame_id}")
        transform = next((item for item in record.transforms if item.source.value == source), None)
        if transform is None:
            continue
        entry = transform.as_dict()
        scores.append(score_calibration_entry(annotation, entry, layout, include_held_out=include_held_out))
    return scores


def _record_rollups(records, frame_ids):
    gate_counts = Counter()
    abstentions = Counter()
    usable = Counter()
    leave_one = []
    probe = []
    held = []
    for record in records:
        if record.frame_id not in frame_ids:
            continue
        abstentions[record.status.value] += 1
        for candidate in record.candidates:
            usable[candidate.source.value] += int(candidate.usable)
            for gate in candidate.gates:
                gate_counts[f"{candidate.source.value}:{gate.gate_id}"] += int(gate.passed)
        if record.difference is not None:
            leave_one.append(max((item.shift_px for item in record.stability), default=0.0))
            probe.append(record.difference.probe_px_max)
        challenger = next((item for item in record.candidates if item.source is CandidateSource.HYBRID), None)
        if challenger and challenger.held_out_px_median is not None:
            held.append(challenger.held_out_px_median)
    return {
        "gate_pass_counts": dict(sorted(gate_counts.items())),
        "abstention_histogram": dict(sorted(abstentions.items())),
        "usable_counts": dict(sorted(usable.items())),
        "leave_one_primitive_shift_px": leave_one,
        "probe_px_max": probe,
        "held_out_px_median": held,
    }


def score_hybrid(records_path, references_dir, layout, *, include_held_out=True) -> dict:
    records = _records(records_path)
    references = {
        annotation.frame_id: annotation
        for path in sorted(Path(references_dir).glob("*.json"))
        for annotation in [FrameAnnotation.from_dict(json.loads(path.read_text(encoding="utf-8")))]
    }
    keypoint_all = {record.frame_id for record in records if any(item.source is CandidateSource.KEYPOINT for item in record.transforms)}
    paired = {record.frame_id for record in records if any(item.source is CandidateSource.HYBRID for item in record.transforms)}
    sources = sorted({item.source.value for record in records for item in record.transforms})
    summaries = {}
    scores_by_source = {}
    for source in sources:
        ids = paired if source in {CandidateSource.KEYPOINT.value, CandidateSource.HYBRID.value} else {record.frame_id for record in records}
        scores_by_source[source] = _source_scores(records, references, layout, source, ids, include_held_out)
        summaries[source] = summarize_scores(scores_by_source[source])
    if CandidateSource.KEYPOINT.value in summaries:
        all_scores = _source_scores(records, references, layout, CandidateSource.KEYPOINT.value, keypoint_all, include_held_out)
        summaries["keypoint_all_frames"] = summarize_scores(all_scores)
        scores_by_source["keypoint_all_frames"] = all_scores
    comparisons = {}
    if CandidateSource.KEYPOINT.value in summaries and CandidateSource.HYBRID.value in summaries:
        comparisons = compare_summaries(summaries[CandidateSource.KEYPOINT.value], summaries[CandidateSource.HYBRID.value])
    key = summaries.get(CandidateSource.KEYPOINT.value, {})
    hybrid = summaries.get(CandidateSource.HYBRID.value, {})
    key_scores = {item.frame_id: item for item in scores_by_source.get(CandidateSource.KEYPOINT.value, ())}
    hybrid_scores = {item.frame_id: item for item in scores_by_source.get(CandidateSource.HYBRID.value, ())}
    key_max = max((item.max_px for item in key_scores.values()), default=float("nan"))
    hybrid_max = max((item.max_px for item in hybrid_scores.values()), default=float("nan"))
    key_near = key.get("bands", {}).get("0-19ft", {}).get("p95_px", float("nan"))
    hybrid_near = hybrid.get("bands", {}).get("0-19ft", {}).get("p95_px", float("nan"))
    far_before = key.get("bands", {}).get("19-30ft", {}).get("p95_px", float("nan"))
    far_after = hybrid.get("bands", {}).get("19-30ft", {}).get("p95_px", float("nan"))
    meets_25 = math.isfinite(far_before) and math.isfinite(far_after) and far_before > 0 and (far_before - far_after) / far_before >= .25
    gross = (
        math.isfinite(key_max) and math.isfinite(hybrid_max) and hybrid_max > key_max + 20.0
    ) or (
        math.isfinite(key_near) and math.isfinite(hybrid_near) and hybrid_near > key_near + 2.0 and hybrid_near > key_near * 1.10
    )
    far_before_median = key.get("bands", {}).get("19-30ft", {}).get("median_px", float("nan"))
    far_after_median = hybrid.get("bands", {}).get("19-30ft", {}).get("median_px", float("nan"))
    improvement = ((far_before_median - far_after_median) / far_before_median * 100.0
                   if math.isfinite(far_before_median) and far_before_median > 0 and math.isfinite(far_after_median) else float("nan"))
    gross_failures = []
    for frame_id in sorted(paired):
        before = key_scores.get(frame_id); after = hybrid_scores.get(frame_id)
        if before is None or after is None:
            continue
        near_before = before.band("0-19ft"); near_after = after.band("0-19ft")
        if after.max_px > before.max_px + 20.0 or (near_before and near_after and near_after.p95_px > near_before.p95_px + 2.0 and near_after.p95_px > near_before.p95_px * 1.10):
            gross_failures.append(frame_id)
    held_values = [item.held_out_px_median for record in records if record.frame_id in paired for item in record.candidates if item.source is CandidateSource.HYBRID and item.held_out_px_median is not None]
    hybrid_held = hybrid.get("held_out_median_px", float("nan"))
    hybrid_far = hybrid.get("bands", {}).get("19-30ft", {})
    report = {
        "schema_version": HYBRID_SCORE_SCHEMA_VERSION,
        "records": len(records),
        "frame_sets": {"paired": sorted(paired), "keypoint_all_frames": sorted(keypoint_all)},
        "summaries": summaries,
        "comparisons": comparisons,
        "rollups": _record_rollups(records, paired),
        "acceptance": {
            "median_improvement_beyond_19ft_pct": improvement,
            "meets_25pct": bool(math.isfinite(improvement) and improvement >= 25.0),
            "near_court_p95_regression_px": (hybrid_near - key_near if math.isfinite(hybrid_near) and math.isfinite(key_near) else float("nan")),
            "meets_no_near_regression": bool(not gross and math.isfinite(hybrid_near) and math.isfinite(key_near)),
            "held_out_median_px": hybrid_held,
            "meets_4px": bool(math.isfinite(hybrid_held) and hybrid_held <= 4.0),
            "held_out_p95_px": max(held_values, default=float("nan")),
            "meets_10px": bool(held_values and max(held_values) <= 10.0),
            "court_median_ft": hybrid_far.get("median_ft", float("nan")),
            "meets_0_5ft": bool(hybrid_far.get("median_ft", float("nan")) <= .5),
            "court_p95_ft": hybrid_far.get("p95_ft", float("nan")),
            "meets_1ft": bool(hybrid_far.get("p95_ft", float("nan")) <= 1.0),
            "gross_failures": gross_failures,
            "meets_zero_gross": not gross_failures,
            "coverage_change": hybrid.get("frames_scored", 0) - key.get("frames_scored", 0),
            "meets_no_coverage_loss": hybrid.get("frames_scored", 0) >= key.get("frames_scored", 0),
            "gross_failure": bool(gross),
            "far_band_p95_before": far_before,
            "far_band_p95_after": far_after,
            "pooled_max_before": key_max,
            "pooled_max_after": hybrid_max,
        },
        "per_clip": {},
    }
    clips = defaultdict(lambda: {"keypoint": [], "hybrid": []})
    for record in records:
        if record.frame_id not in paired or record.frame_id not in references:
            continue
        clip = references[record.frame_id].clip
        clips[clip]["keypoint"].append(record.frame_id)
        clips[clip]["hybrid"].append(record.frame_id)
    report["per_clip"] = {clip: {name: sorted(ids) for name, ids in sorted(values.items())} for clip, values in sorted(clips.items())}
    return report
