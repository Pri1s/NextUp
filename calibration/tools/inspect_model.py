"""Phase-0 model inspection run.

Decodes a fixed set of frames once, runs the detector over them at each requested
inference size, and writes every artifact a reviewer needs plus the provenance to
re-derive it.

The detector is injected rather than constructed here, which is what lets the
whole pipeline be tested without a 400 MB checkpoint — and what will let a
different model be inspected later without touching this file.

This module produces evidence. It draws no conclusion about what any slot means.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from contracts.calibration_types import SCHEMA_VERSION, RawKeypointFrame

from .. import __version__
from ..detectors.base import KeypointDetector
from ..io import provenance as prov
from ..io import records as rec
from ..io.video import FrameSource
from . import flip_probe, overlays, review
from .sampling import DEFAULT_NUM_FRAMES, resolve_indices

DEFAULT_DISPLAY_GATE = 0.25
DEFAULT_CONTACT_COLUMNS = 4


@dataclass(frozen=True, slots=True)
class InspectionConfig:
    """Everything that shapes a run, hashed into the default run id."""

    imgsz: tuple[int, ...] = (640,)
    num_frames: int = DEFAULT_NUM_FRAMES
    explicit_frames: str | None = None
    display_gate: float = DEFAULT_DISPLAY_GATE
    run_flip_probe: bool = False
    flip_tolerance_fraction: float = flip_probe.DEFAULT_TOLERANCE_FRACTION
    candidate_vocabulary: str | None = None
    filmstrip_crop: int = overlays.CROP_SIZE

    def fingerprint(self) -> str:
        parts = [
            ",".join(str(s) for s in self.imgsz),
            str(self.num_frames),
            str(self.explicit_frames),
            f"{self.display_gate:g}",
            str(self.run_flip_probe),
            f"{self.flip_tolerance_fraction:g}",
        ]
        return prov.sha256_text("|".join(parts))[:12]


@dataclass
class InspectionResult:
    """Where the artifacts landed and what the run saw."""

    run_dir: Path
    run_json_path: Path
    review_html_path: Path
    worksheet_md_path: Path
    worksheet_json_path: Path
    sweep_csv_path: Path | None
    per_size: dict[int, dict] = field(default_factory=dict)
    written: list[Path] = field(default_factory=list)


def default_run_id(config: InspectionConfig, created_utc: str) -> str:
    stamp = created_utc.replace(":", "").replace("-", "").split(".")[0]
    return f"{stamp}_{config.fingerprint()}"


def run_inspection(
    source: FrameSource,
    detector_factory: Callable[[int], KeypointDetector],
    out_root: Path | str,
    config: InspectionConfig = InspectionConfig(),
    run_id: str | None = None,
    source_file: Path | str | None = None,
    repo_dir: Path | str | None = None,
    progress: Callable[[str], None] | None = None,
) -> InspectionResult:
    """Inspect one source and write the full Phase-0 artifact set.

    ``detector_factory`` is called once per inference size, so a sweep loads the
    model per size rather than holding several copies at once.
    """
    say = progress or (lambda _message: None)
    created_utc = prov.utc_now()
    run_id = run_id or default_run_id(config, created_utc)

    info = source.info
    frame_indices = resolve_indices(info.frame_count, config.num_frames, config.explicit_frames)
    if not frame_indices:
        raise ValueError(f"no frames to inspect in {info.source_id}")

    say(f"decoding {len(frame_indices)} frames from {info.source_id}")
    frames_bgr: list[np.ndarray] = []
    kept_indices: list[int] = []
    for index in frame_indices:
        try:
            frames_bgr.append(source.frame_at(index))
        except IndexError:
            continue  # a container that over-reports its length is not a run failure
        kept_indices.append(index)
    if not frames_bgr:
        raise ValueError(f"could not decode any of the requested frames from {info.source_id}")
    timestamps = [source.timestamp_of(i) for i in kept_indices]

    run_dir = Path(out_root) / _source_slug(info.source_id) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    per_size_blocks: dict[int, dict] = {}
    model_provenance = None

    for size in config.imgsz:
        say(f"imgsz {size}: running inference")
        detector = detector_factory(size)
        model_provenance = detector.provenance()
        size_dir = run_dir / f"imgsz_{size:04d}"

        detections = [
            detector.detect(frame, index, timestamp, info.source_id)
            for frame, index, timestamp in zip(frames_bgr, kept_indices, timestamps)
        ]

        flip_report = None
        flip_paths: list[str] = []
        if config.run_flip_probe:
            say(f"imgsz {size}: flip probe")
            flip_report, mapped = flip_probe.run_flip_probe(
                detector,
                frames_bgr,
                detections,
                config.display_gate,
                config.flip_tolerance_fraction,
            )
            flip_paths = _write_flip_images(
                size_dir, frames_bgr, detections, mapped, flip_report, config, written
            )

        block = _write_size_artifacts(
            size_dir=size_dir,
            run_dir=run_dir,
            size=size,
            info=info,
            frames_bgr=frames_bgr,
            detections=detections,
            keypoint_count=detector.keypoint_count,
            config=config,
            flip_report=flip_report,
            flip_paths=flip_paths,
            written=written,
            say=say,
        )
        per_size_blocks[size] = block

    primary_size = config.imgsz[0]
    keypoint_count = model_provenance.keypoint_count if model_provenance else 0

    sweep_path = None
    if len(config.imgsz) > 1:
        sweep_path = rec.write_sweep_comparison(
            run_dir / "sweep_comparison.csv",
            {size: block["summary"] for size, block in per_size_blocks.items()},
        )
        written.append(sweep_path)

    run_payload = _build_run_json(
        run_id=run_id,
        created_utc=created_utc,
        config=config,
        info=info,
        source_file=source_file,
        frame_indices=kept_indices,
        timestamps=timestamps,
        model_provenance=model_provenance,
        repo_dir=repo_dir,
        per_size_blocks=per_size_blocks,
    )

    # Reviewer artifacts reference the record files, so they are written after them.
    primary = per_size_blocks[primary_size]
    review_html_path = run_dir / "review.html"
    review_html_path.write_text(
        review.render_review_html(run_payload, per_size_blocks, primary_size),
        encoding="utf-8",
    )
    written.append(review_html_path)

    candidate_names = (
        review.load_candidate_names(config.candidate_vocabulary)
        if config.candidate_vocabulary
        else []
    )
    worksheet_md_path = run_dir / "slot_review.md"
    worksheet_md_path.write_text(
        review.render_slot_review_markdown(
            run_payload,
            keypoint_count,
            primary["summary"],
            primary.get("flip_probe"),
            candidate_names,
        ),
        encoding="utf-8",
    )
    written.append(worksheet_md_path)

    worksheet_json_path = rec.write_json(
        run_dir / "slot_review.template.json",
        review.render_slot_review_json(run_payload, keypoint_count, primary.get("flip_probe")),
    )
    written.append(worksheet_json_path)

    run_payload["outputs"] = prov.output_inventory(run_dir, written)
    run_json_path = rec.write_json(run_dir / "run.json", run_payload)
    say(f"wrote {run_dir}")

    return InspectionResult(
        run_dir=run_dir,
        run_json_path=run_json_path,
        review_html_path=review_html_path,
        worksheet_md_path=worksheet_md_path,
        worksheet_json_path=worksheet_json_path,
        sweep_csv_path=sweep_path,
        per_size=per_size_blocks,
        written=written,
    )


def _write_size_artifacts(
    *,
    size_dir: Path,
    run_dir: Path,
    size: int,
    info,
    frames_bgr: Sequence[np.ndarray],
    detections: Sequence[RawKeypointFrame],
    keypoint_count: int,
    config: InspectionConfig,
    flip_report: dict | None,
    flip_paths: Sequence[str],
    written: list[Path],
    say: Callable[[str], None],
) -> dict:
    size_dir.mkdir(parents=True, exist_ok=True)

    written.append(rec.write_jsonl(size_dir / "predictions.jsonl", detections))
    written.append(rec.write_predictions_csv(size_dir / "predictions.csv", detections))
    written.append(rec.write_frames_csv(size_dir / "frames.csv", detections))

    summary = rec.build_summary(detections, keypoint_count, config.display_gate)
    written.append(rec.write_json(size_dir / "summary.json", summary))

    if flip_report is not None:
        written.append(rec.write_json(size_dir / "flip_probe.json", flip_report))

    say(f"imgsz {size}: drawing overlays")
    overlay_images = []
    overlay_paths = []
    for frame_bgr, record in zip(frames_bgr, detections):
        header = [
            f"{info.source_id}  frame {record.frame_index}  t={record.timestamp_s:.2f}s  imgsz={size}",
            f"instances={record.instance_count}  state={record.detection_state}",
        ]
        image = overlays.draw_overlay(frame_bgr, record, config.display_gate, header)
        overlay_images.append(image)
        path = overlays.write_image(
            size_dir / "overlays" / f"frame_{record.frame_index:06d}.jpg", image
        )
        written.append(path)
        overlay_paths.append(_relative(path, run_dir))

    contact_path = overlays.write_image(
        size_dir / "contact_sheet.jpg",
        overlays.build_contact_sheet(overlay_images, DEFAULT_CONTACT_COLUMNS),
    )
    written.append(contact_path)

    say(f"imgsz {size}: building {keypoint_count} slot filmstrips")
    filmstrip_paths = []
    for slot_index in range(keypoint_count):
        strip = overlays.build_slot_filmstrip(
            frames_bgr, detections, slot_index, config.display_gate, config.filmstrip_crop
        )
        path = overlays.write_image(size_dir / "slots" / f"slot_{slot_index:02d}.jpg", strip)
        written.append(path)
        filmstrip_paths.append(_relative(path, run_dir))

    return {
        "imgsz": size,
        "summary": summary,
        "flip_probe": flip_report,
        "frame_indices": [d.frame_index for d in detections],
        "overlay_paths": overlay_paths,
        "filmstrip_paths": filmstrip_paths,
        "flip_paths": list(flip_paths),
        "contact_sheet_path": _relative(contact_path, run_dir),
        "dir": size_dir.name,
    }


def _write_flip_images(
    size_dir: Path,
    frames_bgr: Sequence[np.ndarray],
    detections: Sequence[RawKeypointFrame],
    mapped: Sequence[RawKeypointFrame],
    report: dict,
    config: InspectionConfig,
    written: list[Path],
) -> list[str]:
    run_dir = size_dir.parent
    paths = []
    for frame_bgr, original, mapped_record, frame_report in zip(
        frames_bgr, detections, mapped, report["frames"]
    ):
        results = [
            flip_probe.SlotFlipResult(
                slot_index=entry["slot_index"],
                verdict=entry["verdict"],
                matched_slot=entry["matched_slot"],
                distance_px=entry["distance_px"],
                original_confidence=entry["original_confidence"],
                flipped_confidence=entry["flipped_confidence"],
            )
            for entry in frame_report["slots"]
        ]
        image = flip_probe.draw_correspondence(
            frame_bgr, original, mapped_record, results, config.display_gate
        )
        path = overlays.write_image(
            size_dir / "flip" / f"frame_{original.frame_index:06d}.jpg", image
        )
        written.append(path)
        paths.append(_relative(path, run_dir))
    return paths


def _build_run_json(
    *,
    run_id: str,
    created_utc: str,
    config: InspectionConfig,
    info,
    source_file: Path | str | None,
    frame_indices: Sequence[int],
    timestamps: Sequence[float],
    model_provenance,
    repo_dir: Path | str | None,
    per_size_blocks: dict[int, dict],
) -> dict:
    source_block = {
        "source_id": info.source_id,
        "kind": info.kind,
        "frame_count": info.frame_count,
        "fps": info.fps,
        "width": info.width,
        "height": info.height,
    }
    if source_file is not None and Path(source_file).is_file():
        source_block.update(prov.file_facts(source_file))

    model_block = {}
    if model_provenance is not None:
        raw = model_provenance.as_dict()
        model_block = {
            "path": raw["weights_path"],
            "sha256": raw["weights_sha256"],
            "keypoint_count": raw["keypoint_count"],
            "task": raw["task"],
            "class_names": raw["class_names"],
            "train_imgsz": raw["train_imgsz"],
            "train_fliplr": raw["train_fliplr"],
            "train_data": raw["train_data"],
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "tool_version": __version__,
        "run_id": run_id,
        "created_utc": created_utc,
        "git": prov.git_facts(repo_dir or Path.cwd()),
        "model": model_block,
        "environment": prov.environment_facts(
            model_provenance.device if model_provenance else None
        ),
        "source": source_block,
        "sampling": {
            "mode": "explicit" if config.explicit_frames else "even",
            "requested": config.num_frames,
            "explicit_frames": config.explicit_frames,
            "frame_indices": list(frame_indices),
            "timestamps_s": list(timestamps),
        },
        "inference": {
            "imgsz": list(config.imgsz),
            "primary_imgsz": config.imgsz[0],
            "conf": model_provenance.inference_conf if model_provenance else None,
            "display_gate": config.display_gate,
            "flip_probe": config.run_flip_probe,
            "flip_tolerance_fraction": config.flip_tolerance_fraction,
        },
        "results": {
            str(size): {
                "frames_detected": block["summary"]["frames_detected"],
                "frames_no_detection": block["summary"]["frames_no_detection"],
                "dir": block["dir"],
            }
            for size, block in per_size_blocks.items()
        },
        "phase0_status": {
            "slots_assigned": False,
            "human_signed_off": False,
            "note": (
                "Tooling output only. Phase 0 exits on human sign-off in slot_review.md, "
                "not on a successful run."
            ),
        },
    }


def _relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def _source_slug(source_id: str) -> str:
    stem = Path(source_id).stem or "source"
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in stem)
