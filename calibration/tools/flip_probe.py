"""Horizontal-flip slot-correspondence probe.

The checkpoint trained with ``fliplr=0.5`` against a ``data.yaml`` we do not
have, so its ``flip_idx`` is unknown. If that permutation was wrong or identity,
the model learned mirrored images with unmirrored landmark identities — which
corrupts end/side identity while leaving aggregate pose mAP healthy. Nothing in
the loss curve would show it.

The probe makes it visible. Run the model on a frame and on its mirror, map the
mirrored predictions back with ``x' = W - 1 - x``, and ask where each slot
landed:

``self``          slot *i* tracks the same physical point under the flip.
``partner:j``     under the flip, slot *i* lands where slot *j* sits unflipped —
                  the signature of a learned mirror permutation.
``incoherent``    slot *i* lands near nothing it predicted unflipped.

The probe reports **observed slot correspondence only**. It does not claim slot
*j* is a court mirror partner — that requires knowing what the slots mean, which
is exactly what Phase 0 has not yet established. Naming the geometry is the
reviewer's call.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from contracts.calibration_types import RawKeypointFrame, RawKeypointObservation

from ..detectors.base import KeypointDetector
from . import overlays

#: Match tolerance as a fraction of the image diagonal.
DEFAULT_TOLERANCE_FRACTION = 0.02

FONT_LABEL = cv2.FONT_HERSHEY_SIMPLEX


def mirror_record(record: RawKeypointFrame) -> RawKeypointFrame:
    """Map a record taken on a mirrored image back into original coordinates."""
    width = record.image_width
    return RawKeypointFrame(
        frame_index=record.frame_index,
        timestamp_s=record.timestamp_s,
        source_id=record.source_id,
        image_width=record.image_width,
        image_height=record.image_height,
        detection_state=record.detection_state,
        instance_count=record.instance_count,
        instance_confidences=record.instance_confidences,
        selected_instance=record.selected_instance,
        observations=tuple(
            RawKeypointObservation(
                slot_index=o.slot_index,
                x=width - 1 - o.x,
                y=o.y,
                confidence=o.confidence,
                clamped=o.clamped,
            )
            for o in record.observations
        ),
    )


@dataclass(frozen=True, slots=True)
class SlotFlipResult:
    """One slot's flip behaviour on one frame."""

    slot_index: int
    verdict: str  # "self" | "partner:<j>" | "incoherent" | "not_evaluated"
    matched_slot: int | None
    distance_px: float | None
    original_confidence: float | None
    flipped_confidence: float | None

    def as_dict(self) -> dict:
        return {
            "slot_index": self.slot_index,
            "verdict": self.verdict,
            "matched_slot": self.matched_slot,
            "distance_px": self.distance_px,
            "original_confidence": self.original_confidence,
            "flipped_confidence": self.flipped_confidence,
        }


def tolerance_px(width: int, height: int, fraction: float = DEFAULT_TOLERANCE_FRACTION) -> float:
    return float(np.hypot(width, height) * fraction)


def compare_frame(
    original: RawKeypointFrame,
    mirrored: RawKeypointFrame,
    keypoint_count: int,
    gate: float,
    tolerance: float,
) -> list[SlotFlipResult]:
    """Match each mirrored-and-mapped slot against the original predictions."""
    results: list[SlotFlipResult] = []

    anchors = [
        o
        for o in original.observations
        if o.confidence >= gate
    ] if original.detected else []

    for slot_index in range(keypoint_count):
        flipped = mirrored.observation(slot_index) if mirrored.detected else None
        origin = original.observation(slot_index) if original.detected else None
        origin_conf = origin.confidence if origin else None
        flipped_conf = flipped.confidence if flipped else None

        if flipped is None or flipped.confidence < gate or not anchors:
            results.append(
                SlotFlipResult(slot_index, "not_evaluated", None, None, origin_conf, flipped_conf)
            )
            continue

        best = min(anchors, key=lambda a: np.hypot(a.x - flipped.x, a.y - flipped.y))
        distance = float(np.hypot(best.x - flipped.x, best.y - flipped.y))

        if distance > tolerance:
            verdict = "incoherent"
            matched = None
        elif best.slot_index == slot_index:
            verdict = "self"
            matched = slot_index
        else:
            verdict = f"partner:{best.slot_index}"
            matched = best.slot_index

        results.append(
            SlotFlipResult(slot_index, verdict, matched, distance, origin_conf, flipped_conf)
        )
    return results


def run_flip_probe(
    detector: KeypointDetector,
    frames_bgr: Sequence[np.ndarray],
    records: Sequence[RawKeypointFrame],
    gate: float,
    tolerance_fraction: float = DEFAULT_TOLERANCE_FRACTION,
) -> tuple[dict, list[RawKeypointFrame]]:
    """Probe every inspected frame; return the report and the mapped records."""
    per_frame: list[dict] = []
    mapped_records: list[RawKeypointFrame] = []
    votes: dict[int, Counter] = {i: Counter() for i in range(detector.keypoint_count)}
    tolerances: list[float] = []

    for frame_bgr, record in zip(frames_bgr, records):
        flipped_record = detector.detect(
            cv2.flip(frame_bgr, 1),
            record.frame_index,
            record.timestamp_s,
            record.source_id,
        )
        mapped = mirror_record(flipped_record)
        mapped_records.append(mapped)

        tol = tolerance_px(record.image_width, record.image_height, tolerance_fraction)
        tolerances.append(tol)
        results = compare_frame(record, mapped, detector.keypoint_count, gate, tol)
        for result in results:
            votes[result.slot_index][result.verdict] += 1

        per_frame.append(
            {
                "frame_index": record.frame_index,
                "tolerance_px": tol,
                "slots": [r.as_dict() for r in results],
            }
        )

    return (
        {
            "confidence_gate": gate,
            "tolerance_fraction": tolerance_fraction,
            "tolerance_px_median": float(np.median(tolerances)) if tolerances else None,
            "frames": per_frame,
            "slots": [_aggregate(i, votes[i]) for i in range(detector.keypoint_count)],
            "note": (
                "Verdicts describe observed slot-to-slot correspondence under a "
                "horizontal flip. They assert nothing about court geometry; naming "
                "the landmarks is the reviewer's job."
            ),
        },
        mapped_records,
    )


def _aggregate(slot_index: int, counter: Counter) -> dict:
    evaluated = {k: v for k, v in counter.items() if k != "not_evaluated"}
    total = sum(evaluated.values())
    dominant, dominant_count = (None, 0)
    if evaluated:
        dominant, dominant_count = max(evaluated.items(), key=lambda kv: kv[1])
    return {
        "slot_index": slot_index,
        "verdict_counts": dict(counter),
        "frames_evaluated": total,
        "dominant_verdict": dominant,
        "agreement": (dominant_count / total) if total else None,
    }


def draw_correspondence(
    frame_bgr: np.ndarray,
    original: RawKeypointFrame,
    mapped: RawKeypointFrame,
    results: Sequence[SlotFlipResult],
    gate: float,
) -> np.ndarray:
    """Original predictions (circles) vs mirrored-and-mapped ones (squares).

    An arrow joins slot *i*'s original position to where the flip put it. A short
    arrow means ``self``; a long one landing exactly on another circle is the
    permutation, drawn rather than described.
    """
    canvas = frame_bgr.copy()
    by_slot = {r.slot_index: r for r in results}

    for observation in original.observations:
        if observation.confidence < gate:
            continue
        point = (int(round(observation.x)), int(round(observation.y)))
        color = overlays.confidence_color(observation.confidence)
        cv2.circle(canvas, point, 7, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, str(observation.slot_index), (point[0] + 9, point[1] - 8),
                    FONT_LABEL, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, str(observation.slot_index), (point[0] + 9, point[1] - 8),
                    FONT_LABEL, 0.5, color, 1, cv2.LINE_AA)

    for observation in mapped.observations:
        if observation.confidence < gate:
            continue
        point = (int(round(observation.x)), int(round(observation.y)))
        cv2.rectangle(canvas, (point[0] - 7, point[1] - 7), (point[0] + 7, point[1] + 7),
                      (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"{observation.slot_index}'", (point[0] + 9, point[1] + 18),
                    FONT_LABEL, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, f"{observation.slot_index}'", (point[0] + 9, point[1] + 18),
                    FONT_LABEL, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        source = original.observation(observation.slot_index)
        result = by_slot.get(observation.slot_index)
        if source is None or source.confidence < gate or result is None:
            continue
        arrow_color = {
            "self": (0, 255, 0),
            "incoherent": (0, 0, 255),
        }.get(result.verdict, (255, 200, 0))
        cv2.arrowedLine(
            canvas,
            (int(round(source.x)), int(round(source.y))),
            point,
            arrow_color,
            2,
            cv2.LINE_AA,
            tipLength=0.15,
        )

    overlays.draw_header(
        canvas,
        [
            f"flip probe  frame {original.frame_index}",
            "circle = original   square = mirrored & mapped back (x' = W-1-x)",
            "arrow: green self / amber partner / red incoherent",
        ],
    )
    return canvas
