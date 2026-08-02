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
    calibrate.add_argument(
        "--marking-shadow", action="store_true",
        help="run opt-in static-frame marking extraction in a separate marking_shadow namespace",
    )
    calibrate.add_argument(
        "--marking-config", default=None,
        help="JSON file overriding the locked marking extractor/association config",
    )
    calibrate.add_argument(
        "--marking-mask-dir", default=None,
        help="directory of exact-size caller-supplied binary exclusion masks",
    )
    calibrate.add_argument(
        "--no-marking-overlays", action="store_true",
        help="write shadow records but omit raw-evidence and association PNG overlays",
    )
    calibrate.add_argument(
        "--hybrid-shadow", action="store_true",
        help="imply --marking-shadow and write a separate hybrid_shadow namespace",
    )
    calibrate.add_argument("--hybrid-config", default=None)
    calibrate.add_argument("--no-hybrid-overlays", action="store_true")
    calibrate.add_argument("--quiet", action="store_true")

    refine_shadow = subparsers.add_parser("refine-shadow", help="Refine serialized marking-shadow records offline.")
    refine_shadow.add_argument("--shadow-run", required=True)
    refine_shadow.add_argument("--calibrations", required=True)
    refine_shadow.add_argument("--layout", default=DEFAULT_LAYOUT)
    refine_shadow.add_argument("--out", required=True)
    refine_shadow.add_argument("--hybrid-config", default=None)

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

    if args.hybrid_shadow:
        args.marking_shadow = True
    shadow = None
    shadow_records = []
    shadow_frames = []

    def say(message: str) -> None:
        if not args.quiet:
            print(f"[calibrate-frames] {message}", flush=True)

    layout = load_registered_layout(args.layout)
    problems = validate_layout(layout)
    if problems:
        raise SystemExit(f"layout {args.layout!r} is invalid: {problems}")

    evidence_map = load_registered_map(args.evidence_map)
    if args.marking_shadow:
        actual_checkpoint_sha256 = prov.sha256_file(args.weights)
        if actual_checkpoint_sha256 != evidence_map.weights_sha256:
            raise SystemExit(
                "marking shadow requires the detector checkpoint hash to match the "
                f"evidence map: checkpoint={actual_checkpoint_sha256}, "
                f"evidence_map={evidence_map.weights_sha256}"
            )
        from .markings import ShadowConfig, ShadowOrchestrator
        shadow = ShadowOrchestrator(
            layout=layout,
            expected_checkpoint_sha256=evidence_map.weights_sha256,
            config=_load_marking_config(args.marking_config),
        )
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
            if shadow is not None:
                frame_id = f"{Path(args.video).stem}_{index:06d}"
                exclusion = (
                    _load_marking_mask(
                        args.marking_mask_dir, frame_id, index,
                        frame.shape[1], frame.shape[0],
                    ) if args.marking_mask_dir else None
                )
                shadow_records.append(
                    shadow.analyze_frame(
                        frame, frame_id, attempt.calibration,
                        exclusion_mask=exclusion,
                    )
                )
                shadow_frames.append((frame, attempt.calibration, exclusion))
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

    if shadow is not None:
        from .markings.overlays import write_shadow_overlays
        from .markings.extractor import build_projected_roi_mask
        from .markings.shadow import write_shadow_run

        shadow_root = run_dir / "marking_shadow"
        if not args.no_marking_overlays:
            for record, (frame, calibration, exclusion) in zip(shadow_records, shadow_frames):
                roi = None
                if calibration.h_court_to_image is not None:
                    roi = build_projected_roi_mask(frame.shape, layout, calibration.h_court_to_image)
                write_shadow_overlays(
                    shadow_root, frame, record, layout,
                    calibration.h_court_to_image, roi_mask=roi, exclusion_mask=exclusion,
                )
        write_shadow_run(
            shadow_root, shadow_records,
            metadata={
                "created_utc": prov.utc_now(),
                "namespace": "M3",
                "video": str(Path(args.video).resolve()),
                "layout_id": layout.layout_id,
                "layout_hash": layout.content_hash(),
                "evidence_map_id": evidence_map.map_id,
                "checkpoint_sha256": evidence_map.weights_sha256,
                "config_hash": shadow.config_hash,
                "config": shadow.config.as_dict(),
                "mask_directory": None if args.marking_mask_dir is None else str(Path(args.marking_mask_dir).resolve()),
                "overlays_written": not args.no_marking_overlays,
                "frame_indices": list(indices),
            },
        )
        if args.hybrid_shadow:
            from .correspondence import build_correspondences
            from .hybrid import HybridOrchestrator
            from .hybrid.io import write_hybrid_run
            hybrid_config = _load_hybrid_config(args.hybrid_config)
            orchestrator = HybridOrchestrator(layout, hybrid_config)
            hybrid_records = []
            for record, attempt in zip(shadow_records, attempts):
                correspondences = build_correspondences(
                    attempt.evidence, layout, calibrator.policy.min_pooled_confidence
                ).correspondences
                inlier_ids = set(attempt.calibration.inlier_landmarks)
                hybrid_records.append(orchestrator.refine_frame(
                    record, attempt.calibration,
                    tuple(item for item in correspondences if item.landmark_id in inlier_ids),
                    image_shape=(record.image_height, record.image_width),
                ))
            write_hybrid_run(
                run_dir / "hybrid_shadow", hybrid_records,
                metadata={"layout_id": layout.layout_id, "layout_hash": layout.content_hash(), "refiner_config_hash": orchestrator.config_hash},
            )

    usable = sum(1 for a in attempts if a.calibration.usable)
    say(f"status counts: {counts}")
    say(f"usable: {usable}/{len(attempts)}")
    say(f"wrote {run_dir}")
    return 0


def _load_marking_config(path: str | None):
    """Load only registered shadow knobs; unknown keys fail loudly."""
    import json

    from .markings import ShadowConfig
    from .markings.association import AssociationConfig
    from .markings.extractor import MarkingExtractorConfig

    if path is None:
        return ShadowConfig()
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if set(payload) - {"extractor", "association"}:
        raise ValueError("marking config may contain only extractor and association")
    extractor_payload = dict(payload.get("extractor", {}))
    association_payload = dict(payload.get("association", {}))
    if set(extractor_payload) - set(MarkingExtractorConfig.__dataclass_fields__):
        raise ValueError("unknown marking extractor config field")
    if set(association_payload) - set(AssociationConfig.__dataclass_fields__):
        raise ValueError("unknown marking association config field")
    if "scales_px" in extractor_payload:
        extractor_payload["scales_px"] = tuple(extractor_payload["scales_px"])
    return ShadowConfig(
        extractor=MarkingExtractorConfig(**extractor_payload),
        association=AssociationConfig(**association_payload),
    )


def _load_hybrid_config(path: str | None):
    import json
    from dataclasses import fields
    from contracts.markings import MarkingFeature
    from .hybrid import HybridGatePolicy, HybridRefinerConfig

    if path is None:
        return HybridRefinerConfig()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed = {item.name for item in fields(HybridRefinerConfig)}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"unknown hybrid config field(s): {sorted(unknown)}")
    payload = dict(payload)
    if "held_out_features" in payload:
        payload["held_out_features"] = tuple(MarkingFeature(item) for item in payload["held_out_features"])
    if "depth_bands" in payload:
        payload["depth_bands"] = tuple(tuple(item) for item in payload["depth_bands"])
    if "gates" in payload:
        gate_payload = dict(payload["gates"])
        gate_allowed = {item.name for item in fields(HybridGatePolicy)}
        if set(gate_payload) - gate_allowed:
            raise ValueError("unknown hybrid gate config field")
        for name in ("scale_ft_per_px_range", "scale_ratio_range"):
            if name in gate_payload:
                gate_payload[name] = tuple(gate_payload[name])
        payload["gates"] = HybridGatePolicy(**gate_payload)
    return HybridRefinerConfig(**payload)


def _calibration_from_dict(raw):
    from contracts.calibration_types import CalibrationQuality, CalibrationStatus, CourtCalibration
    quality_raw = raw.get("quality")
    quality = None
    if quality_raw is not None:
        reprojection = quality_raw.get("reprojection_px") or {}
        feet = quality_raw.get("reprojection_ft") or {}
        span = quality_raw.get("court_span_ft") or {}
        quality = CalibrationQuality(
            observation_count=int(quality_raw["observation_count"]),
            inlier_count=int(quality_raw["inlier_count"]),
            inlier_ratio=float(quality_raw["inlier_ratio"]),
            reprojection_px_mean=reprojection.get("mean"), reprojection_px_median=reprojection.get("median"),
            reprojection_px_p95=reprojection.get("p95"), reprojection_px_max=reprojection.get("max"),
            reprojection_ft_median=feet.get("median"), reprojection_ft_max=feet.get("max"),
            holdout_px_median=quality_raw.get("holdout_px_median"), min_triangle_area_px=quality_raw.get("min_triangle_area_px"),
            image_hull_fraction=quality_raw.get("image_hull_fraction"), court_span_x_ft=span.get("x"), court_span_y_ft=span.get("y"),
            scale_ft_per_px=quality_raw.get("scale_ft_per_px"), round_trip_error_px=quality_raw.get("round_trip_error_px"),
            mean_evidence_confidence=quality_raw.get("mean_evidence_confidence"), reasons=tuple(quality_raw.get("reasons", ())),
        )
    matrix = lambda value: None if value is None else tuple(tuple(float(cell) for cell in row) for row in value)
    return CourtCalibration(
        frame_index=int(raw["frame_index"]), layout_id=str(raw["layout_id"]), status=CalibrationStatus(raw["status"]),
        frame_of_reference=raw.get("frame_of_reference", "visible_end_half_court"), end_identity=raw.get("end_identity", "unresolved"),
        units=raw.get("units", "feet"), h_image_to_court=matrix(raw.get("h_image_to_court")), h_court_to_image=matrix(raw.get("h_court_to_image")),
        quality=quality, inlier_landmarks=tuple(raw.get("inlier_landmarks", ())), outlier_landmarks=tuple(raw.get("outlier_landmarks", ())),
        provenance=raw.get("provenance") or {},
    )


def command_refine_shadow(args: argparse.Namespace) -> int:
    import json
    from .correspondence import build_correspondences
    from .hybrid import HybridOrchestrator
    from .hybrid.io import write_hybrid_run
    from .markings.shadow import read_shadow_records
    from contracts.calibration_types import LandmarkEvidence

    layout = load_registered_layout(args.layout)
    config = _load_hybrid_config(args.hybrid_config)
    shadow_records = read_shadow_records(Path(args.shadow_run) / "records.jsonl")
    calibration_lines = [json.loads(line) for line in Path(args.calibrations).read_text(encoding="utf-8").splitlines() if line.strip()]
    by_index = {}
    for item in calibration_lines:
        raw = item.get("calibration", item)
        by_index[int(raw["frame_index"])] = (_calibration_from_dict(raw), tuple(LandmarkEvidence.from_dict(value) for value in item.get("evidence", ())))
    run_payload = {}
    run_path = Path(args.shadow_run) / "run.json"
    if run_path.exists():
        run_payload = json.loads(run_path.read_text(encoding="utf-8"))
    min_confidence = float((run_payload.get("policy") or {}).get("min_pooled_confidence", 0.25))
    orchestrator = HybridOrchestrator(layout, config)
    records = []
    for shadow in sorted(shadow_records, key=lambda item: item.frame_id):
        try:
            frame_index = int(shadow.frame_id.rsplit("_", 1)[1])
            calibration, evidence = by_index[frame_index]
        except (ValueError, KeyError, IndexError) as error:
            raise ValueError(f"cannot match shadow frame {shadow.frame_id}: {error}") from error
        built = build_correspondences(evidence, layout, min_confidence)
        inlier_ids = set(calibration.inlier_landmarks)
        correspondences = tuple(item for item in built.correspondences if item.landmark_id in inlier_ids)
        records.append(orchestrator.refine_frame(shadow, calibration, correspondences, image_shape=(shadow.image_height, shadow.image_width)))
    write_hybrid_run(args.out, records, metadata={"layout_id": layout.layout_id, "layout_hash": layout.content_hash(), "refiner_config_hash": orchestrator.config_hash})
    print(f"wrote {args.out}")
    return 0


def _load_marking_mask(
    directory: str, frame_id: str, frame_index: int, width: int, height: int,
):
    """Resolve exactly one caller-supplied, exact-size exclusion mask."""
    import cv2

    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"marking mask directory does not exist: {root}")
    candidates = sorted({
        *root.glob(f"{frame_id}.*"),
        *root.glob(f"frame_{frame_index:06d}.*"),
    })
    candidates = [path for path in candidates if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"missing marking exclusion mask for {frame_id} in {root}")
    if len(candidates) > 1:
        raise ValueError(f"duplicate marking exclusion masks for {frame_id}: {candidates}")
    mask = cv2.imread(str(candidates[0]), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"could not read marking exclusion mask {candidates[0]}")
    if mask.ndim == 3:
        raise ValueError(f"marking exclusion mask must be binary/grayscale: {candidates[0]}")
    if mask.shape != (height, width):
        raise ValueError(
            f"wrong-sized marking exclusion mask {candidates[0]}: "
            f"expected {(height, width)}, got {mask.shape}"
        )
    return mask


def command_validate_layout(args: argparse.Namespace) -> int:
    try:
        layout = load_registered_layout(args.layout_id)
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error

    problems = validate_layout(layout)
    from contracts.court_layout import LAYOUT_SCHEMA_VERSION

    print(f"layout: {layout.layout_id} ({layout.rule_set})")
    print(f"schema: {LAYOUT_SCHEMA_VERSION}")
    print(f"source: {layout.source}")
    print(f"url:    {layout.source_url}")
    print(f"frame:  visible-end half court, {layout.half_length} x {layout.width} ft")
    print(
        f"paint:  {layout.line_convention}, width={layout.line_width:.6f} ft "
        f"({layout.line_width * 12:.1f} in)"
    )
    print(f"hash:   {layout.content_hash()[:16]}")
    print("end identity: unresolved by design")
    families = sorted({primitive.family.value for primitive in layout.markings.values()})
    print(f"markings ({len(layout.markings)}): {', '.join(families)}")
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
    if args.command == "refine-shadow":
        return command_refine_shadow(args)
    raise SystemExit(f"unknown command {args.command!r}")


if __name__ == "__main__":
    sys.exit(main())
