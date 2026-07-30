"""Machine-readable prediction records.

Three views of the same run, because they answer different questions:

``predictions.jsonl``  one ``RawKeypointFrame`` per line — the lossless record.
``predictions.csv``    one row per observation — the design doc's
                       ``(frame, index, x, y, conf)`` view, for spreadsheets.
``frames.csv``         one row per inspected frame — detection state and
                       instance counts, including the frames where nothing was
                       detected at all.

The (0, 0) sentinel is banned everywhere: a slot the model localised poorly keeps
its real predicted coordinate and its real confidence, and a frame with no
detection emits *no* observation rows rather than eighteen fake origin points.
"""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from contracts.calibration_types import RawKeypointFrame

PREDICTION_COLUMNS = ("frame_index", "slot_index", "x", "y", "confidence", "clamped")
FRAME_COLUMNS = (
    "frame_index",
    "timestamp_s",
    "detection_state",
    "instance_count",
    "selected_instance",
    "selected_instance_conf",
)


def write_jsonl(path: Path | str, frames: Iterable[RawKeypointFrame]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for frame in frames:
            handle.write(json.dumps(frame.as_dict(), sort_keys=True))
            handle.write("\n")
    return path


def read_jsonl(path: Path | str) -> list[RawKeypointFrame]:
    with open(path, encoding="utf-8") as handle:
        return [RawKeypointFrame.from_dict(json.loads(line)) for line in handle if line.strip()]


def write_predictions_csv(path: Path | str, frames: Sequence[RawKeypointFrame]) -> Path:
    """One row per observation. Frames with no detection contribute no rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(PREDICTION_COLUMNS)
        for frame in frames:
            for observation in sorted(frame.observations, key=lambda o: o.slot_index):
                writer.writerow(
                    [
                        frame.frame_index,
                        observation.slot_index,
                        repr(observation.x),
                        repr(observation.y),
                        repr(observation.confidence),
                        int(observation.clamped),
                    ]
                )
    return path


def write_frames_csv(path: Path | str, frames: Sequence[RawKeypointFrame]) -> Path:
    """One row per inspected frame, so no-detection frames stay visible."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FRAME_COLUMNS)
        for frame in frames:
            selected_conf = ""
            if frame.selected_instance is not None and frame.instance_confidences:
                selected_conf = repr(frame.instance_confidences[frame.selected_instance])
            writer.writerow(
                [
                    frame.frame_index,
                    repr(frame.timestamp_s),
                    frame.detection_state,
                    frame.instance_count,
                    "" if frame.selected_instance is None else frame.selected_instance,
                    selected_conf,
                ]
            )
    return path


@dataclass(frozen=True, slots=True)
class SlotSummary:
    """Cross-frame behaviour of one model slot.

    ``above_gate_rate`` is the fraction of *inspected* frames where this slot
    cleared the display gate — the number that tells a reviewer whether a slot is
    usable at all. Spread statistics only count above-gate frames, since
    below-gate coordinates wander.
    """

    slot_index: int
    frames_inspected: int
    frames_detected: int
    frames_above_gate: int
    above_gate_rate: float
    confidence_median: float | None
    confidence_p95: float | None
    confidence_max: float | None
    x_mean: float | None
    y_mean: float | None
    x_std: float | None
    y_std: float | None
    spread_px: float | None
    clamped_count: int

    def as_dict(self) -> dict:
        return {
            "slot_index": self.slot_index,
            "frames_inspected": self.frames_inspected,
            "frames_detected": self.frames_detected,
            "frames_above_gate": self.frames_above_gate,
            "above_gate_rate": self.above_gate_rate,
            "confidence_median": self.confidence_median,
            "confidence_p95": self.confidence_p95,
            "confidence_max": self.confidence_max,
            "x_mean": self.x_mean,
            "y_mean": self.y_mean,
            "x_std": self.x_std,
            "y_std": self.y_std,
            "spread_px": self.spread_px,
            "clamped_count": self.clamped_count,
        }


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile — no interpolation, no scipy."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("no values")
    rank = max(1, min(len(ordered), int(round(fraction * len(ordered) + 0.5))))
    return ordered[rank - 1]


def summarize_slots(
    frames: Sequence[RawKeypointFrame],
    keypoint_count: int,
    gate: float,
) -> list[SlotSummary]:
    summaries = []
    detected_frames = [f for f in frames if f.detected]

    for slot_index in range(keypoint_count):
        observations = [
            o for f in detected_frames if (o := f.observation(slot_index)) is not None
        ]
        above = [o for o in observations if o.confidence >= gate]
        confidences = [o.confidence for o in observations]

        xs = [o.x for o in above]
        ys = [o.y for o in above]
        x_mean = statistics.fmean(xs) if xs else None
        y_mean = statistics.fmean(ys) if ys else None
        x_std = statistics.pstdev(xs) if len(xs) > 1 else (0.0 if xs else None)
        y_std = statistics.pstdev(ys) if len(ys) > 1 else (0.0 if ys else None)
        spread = (x_std**2 + y_std**2) ** 0.5 if x_std is not None and y_std is not None else None

        summaries.append(
            SlotSummary(
                slot_index=slot_index,
                frames_inspected=len(frames),
                frames_detected=len(observations),
                frames_above_gate=len(above),
                above_gate_rate=(len(above) / len(frames)) if frames else 0.0,
                confidence_median=statistics.median(confidences) if confidences else None,
                confidence_p95=_percentile(confidences, 0.95) if confidences else None,
                confidence_max=max(confidences) if confidences else None,
                x_mean=x_mean,
                y_mean=y_mean,
                x_std=x_std,
                y_std=y_std,
                spread_px=spread,
                clamped_count=sum(1 for o in observations if o.clamped),
            )
        )
    return summaries


def build_summary(
    frames: Sequence[RawKeypointFrame],
    keypoint_count: int,
    gate: float,
) -> dict:
    detected = sum(1 for f in frames if f.detected)
    multi_instance = sum(1 for f in frames if f.instance_count > 1)
    return {
        "frames_inspected": len(frames),
        "frames_detected": detected,
        "frames_no_detection": len(frames) - detected,
        "frames_multi_instance": multi_instance,
        "confidence_gate": gate,
        "keypoint_count": keypoint_count,
        "slots": [s.as_dict() for s in summarize_slots(frames, keypoint_count, gate)],
    }


def write_json(path: Path | str, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")
    return path


def write_sweep_comparison(path: Path | str, per_size: dict[int, dict]) -> Path:
    """Per-slot detection rate and median confidence at each inference size.

    This is the table that answers "does a larger imgsz recover the far-court
    landmarks?" with data instead of intuition.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sizes = sorted(per_size)
    columns = ["slot_index"]
    for size in sizes:
        columns += [f"above_gate_rate_{size}", f"confidence_median_{size}"]

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        keypoint_count = max(len(per_size[s]["slots"]) for s in sizes)
        for slot_index in range(keypoint_count):
            row: list = [slot_index]
            for size in sizes:
                slots = per_size[size]["slots"]
                slot = slots[slot_index] if slot_index < len(slots) else {}
                row.append(round(slot.get("above_gate_rate") or 0.0, 4))
                median = slot.get("confidence_median")
                row.append("" if median is None else round(median, 4))
            writer.writerow(row)
    return path
