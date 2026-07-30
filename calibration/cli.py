"""Command-line entry point.

    python -m calibration.cli inspect-model \\
        --weights models/court_keypoint_detector.pt \\
        --video input_videos/video_1.mp4 \\
        --num-frames 24 --imgsz 640,960,1280 --flip-probe

    python -m calibration.cli calibrate-frames \\
        --weights models/court_keypoint_detector.pt \\
        --video input_videos/video_2.mp4 --num-frames 24

    python -m calibration.cli validate-layout nba_halfcourt

``calibrate-frames`` runs on **provisional** slot semantics: the Phase-0 gate has
not been cleared, so it pools redundant slots as weighted evidence, refuses the
landmarks the review could not settle, and returns an explicit uncalibrated
status rather than a guess whenever the evidence will not carry a fit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from contracts.court_layout import (
    available_layouts,
    load_registered_layout,
    validate_layout,
)

from .adapters.evidence import load_registered_map
from .calibrator import FrameCalibrator
from .detectors.yolo_pose import DEFAULT_CONF, DEFAULT_IOU, DEFAULT_MAX_DET, YoloPoseDetector
from .io.video import ImageFrameSource, VideoFrameSource, resolve_image_paths
from .quality import CalibrationPolicy
from .tools import flip_probe
from .tools.inspect_model import (
    DEFAULT_DISPLAY_GATE,
    InspectionConfig,
    run_inspection,
)
from .tools.sampling import DEFAULT_NUM_FRAMES, resolve_indices

DEFAULT_LAYOUT = "nba_halfcourt"
DEFAULT_EVIDENCE_MAP = "reloc2_18_provisional"

REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUT = "engine_out/phase0"


def parse_imgsz(spec: str) -> tuple[int, ...]:
    """``"640,960"`` -> ``(640, 960)``, order preserved; the first is primary."""
    sizes: list[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f"imgsz must be positive, got {value}")
        if value not in sizes:
            sizes.append(value)
    if not sizes:
        raise ValueError(f"no usable inference sizes in {spec!r}")
    return tuple(sizes)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="calibration.cli",
        description="Court-calibration engine (Phase 0: model inspection only).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser(
        "inspect-model",
        help="Run a pose model over selected frames and emit reviewer artifacts.",
        description=(
            "Produces evidence for a human to judge keypoint-slot semantics. "
            "Asserts no landmark identities and writes no adapter map."
        ),
    )
    inspect.add_argument("--weights", required=True, help="path to the .pt pose checkpoint")

    source = inspect.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="video clip to sample frames from")
    source.add_argument(
        "--images",
        help="directory or glob of still images (the intake path for no-court negatives)",
    )

    inspect.add_argument("--out", default=DEFAULT_OUT, help=f"output root (default {DEFAULT_OUT})")
    inspect.add_argument("--run-id", help="override the generated run id")
    inspect.add_argument(
        "--num-frames",
        type=int,
        default=DEFAULT_NUM_FRAMES,
        help=f"evenly spaced frames to inspect (default {DEFAULT_NUM_FRAMES})",
    )
    inspect.add_argument("--frames", help="explicit frame indices, e.g. 0,40,80")
    inspect.add_argument(
        "--imgsz",
        default="640",
        help="inference size, or a comma list to sweep (default 640, matching training)",
    )
    inspect.add_argument("--conf", type=float, default=DEFAULT_CONF, help="detection confidence")
    inspect.add_argument("--iou", type=float, default=DEFAULT_IOU, help="NMS IoU")
    inspect.add_argument("--max-det", type=int, default=DEFAULT_MAX_DET, help="max instances")
    inspect.add_argument(
        "--gate",
        type=float,
        default=DEFAULT_DISPLAY_GATE,
        help=(
            f"keypoint confidence gate for display and statistics (default "
            f"{DEFAULT_DISPLAY_GATE}); records always keep every slot regardless"
        ),
    )
    inspect.add_argument("--device", help="torch device, e.g. mps, cuda:0, cpu")
    inspect.add_argument(
        "--flip-probe",
        action="store_true",
        help="also predict on the mirrored frame and report slot correspondence",
    )
    inspect.add_argument(
        "--flip-tolerance",
        type=float,
        default=flip_probe.DEFAULT_TOLERANCE_FRACTION,
        help="flip match tolerance as a fraction of the image diagonal",
    )
    inspect.add_argument(
        "--candidate-vocabulary",
        help="optional JSON of reference landmark names for the worksheet appendix",
    )
    inspect.add_argument("--quiet", action="store_true", help="suppress progress output")

    calibrate = subparsers.add_parser(
        "calibrate-frames",
        help="Calibrate sampled frames against a half-court layout.",
        description=(
            "Runs on provisional slot semantics: redundant slots are pooled as "
            "weighted evidence, unsettled slots are refused, and frames whose "
            "evidence will not carry a fit return an explicit uncalibrated status."
        ),
    )
    calibrate.add_argument("--weights", required=True, help="path to the .pt pose checkpoint")
    calibrate.add_argument("--video", required=True, help="video clip to sample frames from")
    calibrate.add_argument("--out", default="engine_out/calibration", help="output root")
    calibrate.add_argument("--run-id", help="override the generated run id")
    calibrate.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    calibrate.add_argument("--frames", help="explicit frame indices, e.g. 0,40,80")
    calibrate.add_argument("--layout", default=DEFAULT_LAYOUT, help=f"default {DEFAULT_LAYOUT}")
    calibrate.add_argument(
        "--evidence-map", default=DEFAULT_EVIDENCE_MAP, help=f"default {DEFAULT_EVIDENCE_MAP}"
    )
    calibrate.add_argument(
        "--tiers",
        default="confident",
        help=(
            "evidence tiers to fit from, comma separated (confident,probable). "
            "Default 'confident' fits only from independently supported landmarks."
        ),
    )
    calibrate.add_argument("--imgsz", type=int, default=640, help="inference size (default 640)")
    calibrate.add_argument("--conf", type=float, default=DEFAULT_CONF)
    calibrate.add_argument("--device", help="torch device, e.g. mps, cuda:0, cpu")
    calibrate.add_argument(
        "--no-reprojection",
        action="store_true",
        help=(
            "skip the reprojected-markings images. They are the only check whose "
            "reference is the floor rather than the detector, so skipping them "
            "removes the sole independent validation of a fit."
        ),
    )
    calibrate.add_argument("--quiet", action="store_true")

    validate = subparsers.add_parser(
        "validate-layout", help="Check a half-court layout profile's geometry."
    )
    validate.add_argument("layout_id", nargs="?", default=DEFAULT_LAYOUT)

    return parser


def command_inspect_model(args: argparse.Namespace) -> int:
    config = InspectionConfig(
        imgsz=parse_imgsz(args.imgsz),
        num_frames=args.num_frames,
        explicit_frames=args.frames,
        display_gate=args.gate,
        run_flip_probe=args.flip_probe,
        flip_tolerance_fraction=args.flip_tolerance,
        candidate_vocabulary=args.candidate_vocabulary,
    )

    source_file = None
    if args.video:
        source = VideoFrameSource(args.video)
        source_file = Path(args.video)
    else:
        paths = resolve_image_paths(args.images)
        source = ImageFrameSource(paths, source_id=Path(args.images).name or "images")

    def detector_factory(imgsz: int):
        return YoloPoseDetector(
            args.weights,
            imgsz=imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
        )

    def say(message: str) -> None:
        if not args.quiet:
            print(f"[inspect-model] {message}", flush=True)

    try:
        result = run_inspection(
            source=source,
            detector_factory=detector_factory,
            out_root=args.out,
            config=config,
            run_id=args.run_id,
            source_file=source_file,
            repo_dir=REPO_DIR,
            progress=say,
        )
    finally:
        source.close()

    say(f"review page: {result.review_html_path}")
    say(f"worksheet:   {result.worksheet_md_path}")
    say("Phase 0 is NOT complete until a human fills in slot_review.md and signs off.")
    return 0


def parse_tiers(spec: str) -> tuple[str, ...]:
    tiers = tuple(t.strip() for t in spec.split(",") if t.strip())
    if not tiers:
        raise ValueError(f"no usable tiers in {spec!r}")
    return tiers


def command_calibrate_frames(args: argparse.Namespace) -> int:
    import json
    from datetime import datetime, timezone

    from contracts.calibration_types import CalibrationStatus
    from .io import provenance as prov
    from .io.records import write_json
    from .viz import write_reprojection

    def say(message: str) -> None:
        if not args.quiet:
            print(f"[calibrate-frames] {message}", flush=True)

    layout = load_registered_layout(args.layout)
    problems = validate_layout(layout)
    if problems:
        raise SystemExit(f"layout {args.layout!r} is invalid: {problems}")

    evidence_map = load_registered_map(args.evidence_map)
    calibrator = FrameCalibrator(
        layout=layout,
        evidence_map=evidence_map,
        policy=CalibrationPolicy(),
        tiers=parse_tiers(args.tiers),
    )

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_dir = Path(args.out) / Path(args.video).stem / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    source = VideoFrameSource(args.video)
    detector = YoloPoseDetector(
        args.weights, imgsz=args.imgsz, conf=args.conf, device=args.device
    )

    try:
        indices = resolve_indices(source.info.frame_count, args.num_frames, args.frames)
        say(f"calibrating {len(indices)} frames from {source.info.source_id}")

        attempts = []
        for index in indices:
            try:
                frame = source.frame_at(index)
            except IndexError:
                continue
            detection = detector.detect(
                frame, index, source.timestamp_of(index), source.info.source_id
            )
            attempt = calibrator.calibrate(detection)
            attempts.append(attempt)
            if not args.no_reprojection:
                write_reprojection(
                    run_dir / "reprojected" / f"frame_{index:06d}.jpg",
                    frame,
                    attempt.calibration,
                    layout,
                    attempt.evidence,
                )
    finally:
        source.close()

    counts: dict[str, int] = {}
    for attempt in attempts:
        key = attempt.calibration.status.value
        counts[key] = counts.get(key, 0) + 1

    with open(run_dir / "calibrations.jsonl", "w", encoding="utf-8") as handle:
        for attempt in attempts:
            handle.write(json.dumps(attempt.as_dict(), sort_keys=True))
            handle.write("\n")

    write_json(
        run_dir / "run.json",
        {
            "schema_version": "calibration-run-1.0.0",
            "created_utc": prov.utc_now(),
            "video": str(Path(args.video).resolve()),
            "layout": layout.as_dict(),
            "evidence_map": {
                "map_id": evidence_map.map_id,
                "status": evidence_map.status,
                "tiers_used": list(parse_tiers(args.tiers)),
            },
            "policy": calibrator.policy.as_dict(),
            "environment": prov.environment_facts(args.device),
            "git": prov.git_facts(REPO_DIR),
            "frame_indices": list(indices),
            "status_counts": counts,
            "gate_status": {
                "slot_semantics": "provisional_unverified",
                "end_identity": "unresolved",
                "note": (
                    "Calibrations here rest on an unsigned first-pass slot review. "
                    "Court coordinates are relative to the visible end; no north/south "
                    "identity is claimed."
                ),
            },
        },
    )

    usable = sum(1 for a in attempts if a.calibration.usable)
    say(f"status counts: {counts}")
    say(f"usable: {usable}/{len(attempts)}")
    say(f"wrote {run_dir}")
    return 0


def command_validate_layout(args: argparse.Namespace) -> int:
    try:
        layout = load_registered_layout(args.layout_id)
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error

    problems = validate_layout(layout)
    print(f"layout: {layout.layout_id} ({layout.rule_set})")
    print(f"source: {layout.source}")
    print(f"frame:  visible-end half court, {layout.half_length} x {layout.width} ft")
    print(f"hash:   {layout.content_hash()[:16]}")
    print("end identity: unresolved by design")
    print(f"landmarks ({len(layout.landmarks)}):")
    for landmark, (x, y) in sorted(layout.landmarks.items(), key=lambda kv: kv[0].value):
        print(f"  {landmark.value:<28} ({x:6.2f}, {y:6.2f}) ft")

    if problems:
        print("\nPROBLEMS:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nvalidation: OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inspect-model":
        return command_inspect_model(args)
    if args.command == "calibrate-frames":
        return command_calibrate_frames(args)
    if args.command == "validate-layout":
        return command_validate_layout(args)
    raise SystemExit(f"unknown command {args.command!r}")


if __name__ == "__main__":
    sys.exit(main())
