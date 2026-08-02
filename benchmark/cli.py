"""Command-line entry point for the annotation benchmark.

    python -m benchmark.cli export-frames --manifest nba_m1_v1 --out engine_out/benchmark/m1_v1/frames

    python -m benchmark.cli crop --frames engine_out/benchmark/m1_v1/frames \\
        --frame video_2_000141 --center 640,420 --size 80 --scale 16

    python -m benchmark.cli resolve --frames engine_out/benchmark/m1_v1/frames \\
        --annotation engine_out/benchmark/m1_v1/pass_a/video_2_000141.json

    python -m benchmark.cli validate --annotation <path>
    python -m benchmark.cli status --frames <dir> --pass-dir <dir>

The commands an annotator may run are deliberately narrow: request a magnified
crop, convert crop coordinates back to the frame, and check its own work. Nothing
here reads a model, a prediction or a calibration, because an annotation that has
seen those stops being independent evidence (design plan §7.2).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from contracts.court_layout import load_registered_layout, validate_layout

REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = "nba_m1_v1"
DEFAULT_BENCHMARK_ROOT = "engine_out/benchmark/m1_v1"
DEFAULT_FRAMES_OUT = f"{DEFAULT_BENCHMARK_ROOT}/frames"
DEFAULT_REFERENCE_DIR = "benchmark/references/nba_m1_v1_human"


def _say(quiet: bool, prefix: str):
    def emit(message: str) -> None:
        if not quiet:
            print(f"[{prefix}] {message}", flush=True)

    return emit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark.cli",
        description="Independent court-marking benchmark: blind frames, crops, consensus, scoring.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser(
        "export-frames",
        help="Write the locked frame set as raw PNGs an annotator can read.",
        description=(
            "Exports raw frames and the feature glossary, and nothing else. The tree "
            "is checked for model-derived content before it is declared usable."
        ),
    )
    export.add_argument("--manifest", default=DEFAULT_MANIFEST, help=f"default {DEFAULT_MANIFEST}")
    export.add_argument("--out", default=DEFAULT_FRAMES_OUT, help=f"default {DEFAULT_FRAMES_OUT}")
    export.add_argument("--layout", default=None, help="override the manifest's layout id")
    export.add_argument("--quiet", action="store_true")

    verify = subparsers.add_parser(
        "verify-frames", help="Re-hash an exported frame tree and re-check its blindness."
    )
    verify.add_argument("--frames", default=DEFAULT_FRAMES_OUT)

    verify_prompts = subparsers.add_parser(
        "verify-prompts",
        help="Fail if a versioned annotation prompt changed after it was frozen.",
    )
    verify_prompts.add_argument("--manifest", default=None)

    init_run = subparsers.add_parser(
        "init-run",
        help="Create a fresh annotation namespace and record runtime OpenAI model IDs.",
        description=(
            "This does not invoke a model. It verifies the frozen prompts, pins the "
            "frame hashes, records exact runtime model IDs, and creates isolated "
            "pass/adjudication directories."
        ),
    )
    init_run.add_argument("--run-id", required=True)
    init_run.add_argument("--frames", default=DEFAULT_FRAMES_OUT)
    init_run.add_argument("--out", required=True)
    init_run.add_argument("--pass-a-model", required=True)
    init_run.add_argument("--pass-b-model", required=True)
    init_run.add_argument("--adjudicator-model", required=True)
    init_run.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high"),
        default=None,
        help="accuracy-affecting runtime setting to record (no model is invoked)",
    )
    init_run.add_argument(
        "--image-detail",
        choices=("low", "high", "original"),
        default=None,
        help="accuracy-affecting image-inspection setting to record (no model is invoked)",
    )

    crop = subparsers.add_parser(
        "crop",
        help="Write a magnified, grid-labelled crop of one frame.",
        description=(
            "The crop is the annotator's only working surface. Coordinates are read "
            "off its grid in crop pixels and converted back by 'resolve' -- never by "
            "hand, because an arithmetic slip is silent and unbounded."
        ),
    )
    crop.add_argument("--frames", default=DEFAULT_FRAMES_OUT)
    crop.add_argument("--frame", required=True, help="frame id, e.g. video_2_000141")
    crop.add_argument("--center", required=True, help="frame-space centre, 'x,y'")
    crop.add_argument("--size", type=int, default=None, help="crop side in frame pixels")
    crop.add_argument("--scale", type=int, default=None, help="magnification factor")
    crop.add_argument("--interp", default=None, choices=("nearest", "lanczos"))
    crop.add_argument("--out", default=None, help="crop directory (default <frames>/../crops)")
    crop.add_argument(
        "--no-grid",
        action="store_true",
        help="omit the coordinate grid. Only for looking; a coordinate read off an "
        "ungridded crop is an estimate, which is what the grid exists to prevent.",
    )

    resolve = subparsers.add_parser(
        "resolve",
        help="Convert a crop-space annotation to frame space and check its geometry.",
    )
    resolve.add_argument("--frames", default=DEFAULT_FRAMES_OUT)
    resolve.add_argument("--annotation", required=True)
    resolve.add_argument("--crops", default=None, help="crop directory (default <frames>/../crops)")
    resolve.add_argument("--out", default=None, help="output path (default alongside, .frame.json)")

    validate = subparsers.add_parser(
        "validate", help="Check one annotation file against the schema and geometry."
    )
    validate.add_argument("--annotation", required=True)

    status = subparsers.add_parser(
        "status", help="Which frames of the locked set a pass has finished and validated."
    )
    status.add_argument("--frames", default=DEFAULT_FRAMES_OUT)
    status.add_argument("--pass-dir", required=True)
    status.add_argument(
        "--expect-prompt-version",
        default=None,
        help="fail unless every annotation in the pass records this prompt_version",
    )
    status.add_argument(
        "--expect-annotator-id",
        default=None,
        help="fail unless every annotation records this exact runtime model ID",
    )

    glossary = subparsers.add_parser("glossary", help="Print the annotator's feature glossary.")
    glossary.add_argument("--layout", default="nba_halfcourt")

    synthetic = subparsers.add_parser(
        "render-synthetic",
        help="Render a court whose true marking positions are known exactly.",
        description=(
            "The ground-truth probe. Consensus between two passes measures "
            "repeatability, not accuracy, and cannot see a bias both passes share. "
            "This can: the centerlines are known to floating point."
        ),
    )
    synthetic.add_argument("--out", default=f"{DEFAULT_BENCHMARK_ROOT}/synthetic")
    synthetic.add_argument("--layout", default="nba_halfcourt")
    synthetic.add_argument("--scene-id", default="synthetic_court_a")
    synthetic.add_argument("--seed", type=int, default=7)
    synthetic.add_argument(
        "--no-distractors",
        action="store_true",
        help="omit the logo, floor text and score bug. Easier than any real frame.",
    )

    score_truth = subparsers.add_parser(
        "score-truth",
        help="Score a frame-space annotation against synthetic ground truth.",
        description=(
            "Reports per-feature error and, decisively, the mean *signed* offset: "
            "consistently non-zero means every sample of that marking sits to one "
            "side of the paint, which two agreeing passes would both do."
        ),
    )
    score_truth.add_argument("--annotation", required=True)
    score_truth.add_argument(
        "--truth", default=f"{DEFAULT_BENCHMARK_ROOT}/synthetic/truth.json"
    )

    consensus = subparsers.add_parser(
        "consensus",
        help="Compare two independent passes; write the silver set and the dispute queue.",
        description=(
            "Features where both passes agree within tolerance become silver. "
            "Everything else is queued for an adjudicator, and identity conflicts "
            "are never resolved by averaging."
        ),
    )
    consensus.add_argument("--frames", default=DEFAULT_FRAMES_OUT)
    consensus.add_argument("--pass-a", required=True)
    consensus.add_argument("--pass-b", required=True)
    consensus.add_argument("--out", default=f"{DEFAULT_BENCHMARK_ROOT}/consensus")
    consensus.add_argument("--layout", default="nba_halfcourt")

    score = subparsers.add_parser(
        "score",
        help="Score calibration against the human reference, split by court depth.",
        description=(
            "The measurement this benchmark exists for. Reprojection error tests a "
            "fit against its own five landmarks; this tests it against the floor, "
            "and reports 0-19 ft separately from everything the fit extrapolates."
        ),
    )
    score.add_argument(
        "--reference",
        "--silver",
        dest="reference",
        default=DEFAULT_REFERENCE_DIR,
        help=(
            "directory of frame-space reference annotations; --silver remains "
            "as a compatibility alias for older consensus runs"
        ),
    )
    score.add_argument(
        "--run",
        required=True,
        action="append",
        help=(
            "a calibration run directory holding calibrations.jsonl; repeat once per "
            "clip to reuse the recorded Milestone 1 baselines"
        ),
    )
    score.add_argument("--layout", default="nba_halfcourt")
    score.add_argument("--out", default=None, help="write the full report as JSON")
    score.add_argument(
        "--baseline", default=None, help="an earlier score report to compare against"
    )

    extraction = subparsers.add_parser(
        "score-extraction",
        help="Score serialized marking-shadow evidence against blind human references.",
    )
    extraction.add_argument("--records", required=True, help="marking_shadow/records.jsonl")
    extraction.add_argument("--references", required=True, help="directory of frame-space human references")
    extraction.add_argument("--out", default=None, help="write the full extraction report as JSON")

    hybrid = subparsers.add_parser(
        "score-hybrid", help="Score serialized keypoint and hybrid shadow transforms against blind references."
    )
    hybrid.add_argument("--records", required=True, help="hybrid_shadow/records.jsonl")
    hybrid.add_argument("--references", required=True, help="directory of frame-space human references")
    hybrid.add_argument("--layout", default="nba_halfcourt")
    hybrid.add_argument("--out", default=None, help="write the full hybrid report as JSON")

    return parser


def command_export_frames(args: argparse.Namespace) -> int:
    from .frames import (
        assert_blind,
        export_frames,
        load_manifest,
        validate_manifest,
        write_export_index,
    )
    from .geometry import validate_geometry
    from .glossary import GLOSSARY_VERSION, render_markdown

    say = _say(args.quiet, "export-frames")

    manifest = load_manifest(args.manifest)
    problems = validate_manifest(manifest, REPO_DIR)
    if problems:
        raise SystemExit("manifest is invalid:\n  " + "\n  ".join(problems))

    layout = load_registered_layout(args.layout or manifest.layout_id)
    for check, label in ((validate_layout(layout), "layout"), (validate_geometry(layout), "geometry")):
        if check:
            raise SystemExit(f"{label} is invalid:\n  " + "\n  ".join(check))

    say(f"manifest {manifest.manifest_id}: {len(manifest.frames)} frames, "
        f"{len(manifest.pilot_frames)} pilot")

    exported = export_frames(manifest, args.out, repo_dir=REPO_DIR, progress=say)
    write_export_index(args.out, manifest, exported, render_markdown(layout), GLOSSARY_VERSION)

    leaks = assert_blind(args.out)
    if leaks:
        raise SystemExit(
            "exported tree is not blind -- an annotator reading it could inherit the "
            "model's error:\n  " + "\n  ".join(leaks)
        )

    strata: dict[str, int] = {}
    for frame in exported:
        strata[frame.stratum] = strata.get(frame.stratum, 0) + 1
    say(f"strata: {strata}")
    say(f"blindness check: clean ({len(exported)} frames)")
    say(f"wrote {Path(args.out).resolve()}")
    return 0


def command_verify_frames(args: argparse.Namespace) -> int:
    from .frames import assert_blind, verify_export

    problems = verify_export(args.frames) + assert_blind(args.frames)
    if problems:
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"{args.frames}: intact and blind")
    return 0


def command_verify_prompts(args: argparse.Namespace) -> int:
    from .workflow import PROMPT_MANIFEST, verify_prompts

    manifest = args.manifest or PROMPT_MANIFEST
    problems = verify_prompts(manifest)
    if problems:
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"{Path(manifest)}: prompt versions and hashes are frozen")
    return 0


def command_init_run(args: argparse.Namespace) -> int:
    from .workflow import initialize_run

    try:
        destination = initialize_run(
            args.out,
            run_id=args.run_id,
            frames_dir=args.frames,
            pass_a_model=args.pass_a_model,
            pass_b_model=args.pass_b_model,
            adjudicator_model=args.adjudicator_model,
            reasoning_effort=args.reasoning_effort,
            image_detail=args.image_detail,
        )
    except (FileExistsError, ValueError) as error:
        print(error)
        return 1
    print(f"initialized {destination}")
    print("no annotation models were invoked")
    return 0


def _crop_dir(args: argparse.Namespace) -> Path:
    return Path(args.out) if args.out else Path(args.frames).parent / "crops"


def command_crop(args: argparse.Namespace) -> int:
    from .crops import DEFAULT_INTERP, DEFAULT_SCALE, DEFAULT_SIZE_PX, make_crop

    try:
        x_text, y_text = args.center.split(",")
        center = (float(x_text), float(y_text))
    except ValueError as error:
        raise SystemExit(f"--center must be 'x,y', got {args.center!r}") from error

    crop = make_crop(
        frames_dir=args.frames,
        frame_id=args.frame,
        center=center,
        size_px=args.size or DEFAULT_SIZE_PX,
        scale=args.scale or DEFAULT_SCALE,
        interp=args.interp or DEFAULT_INTERP,
        out_dir=_crop_dir(args),
        grid=not args.no_grid,
    )
    print(json.dumps(crop.as_dict(), indent=2))
    return 0


def command_resolve(args: argparse.Namespace) -> int:
    from contracts.annotations import FrameAnnotation, validate

    from .crops import load_crops, resolve_annotation

    source = Path(args.annotation)
    with open(source, encoding="utf-8") as handle:
        annotation = FrameAnnotation.from_dict(json.load(handle))

    crops = load_crops(args.crops or (Path(args.frames).parent / "crops"))
    resolved = resolve_annotation(annotation, crops)

    destination = Path(args.out) if args.out else source.with_suffix(".frame.json")
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(resolved.as_dict(), handle, indent=2)
        handle.write("\n")

    problems = validate(resolved, require_complete=True)
    print(f"resolved -> {destination}")
    if problems:
        print(f"\n{len(problems)} problem(s) to fix:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("geometry: OK")
    return 0


def command_validate(args: argparse.Namespace) -> int:
    from contracts.annotations import FrameAnnotation, validate

    with open(args.annotation, encoding="utf-8") as handle:
        annotation = FrameAnnotation.from_dict(json.load(handle))

    problems = validate(annotation, require_complete=True)
    print(f"{annotation.frame_id} ({annotation.coordinate_space} space, pass {annotation.pass_id})")
    print(f"features: {len(annotation.features)}  junctions: {len(annotation.junctions)}  "
          f"skipped: {len(annotation.skipped)}")
    if annotation.coordinate_space == "crop":
        print("note: geometry is checked after 'resolve'; crop coordinates share no frame")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nvalidation: OK")
    return 0


def command_status(args: argparse.Namespace) -> int:
    from contracts.annotations import FrameAnnotation, validate

    from .frames import read_export_index
    from .workflow import load_prompt_manifest

    pass_dir = Path(args.pass_dir)
    expected = read_export_index(args.frames)
    expected_prompt_version = args.expect_prompt_version
    if expected_prompt_version is None:
        prompt_manifest = load_prompt_manifest()
        expected_prompt_version = next(
            entry["version"]
            for entry in prompt_manifest["prompts"]
            if entry["role"] == "annotator"
        )

    done, pending, broken = [], [], []
    versions: dict[str, list[str]] = {}
    annotators: dict[str, list[str]] = {}
    for frame in expected:
        resolved = pass_dir / f"{frame.frame_id}.frame.json"
        raw = pass_dir / f"{frame.frame_id}.json"
        if not resolved.is_file():
            if raw.is_file():
                broken.append(f"{frame.frame_id}: crop-space annotation has not been resolved")
            else:
                pending.append(frame.frame_id)
            continue
        path = resolved
        if not path.is_file():
            pending.append(frame.frame_id)
            continue
        try:
            annotation = FrameAnnotation.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (ValueError, KeyError) as error:
            broken.append(f"{frame.frame_id}: unreadable ({error})")
            continue
        versions.setdefault(annotation.prompt_version, []).append(frame.frame_id)
        annotators.setdefault(annotation.annotator_id, []).append(frame.frame_id)
        metadata = (
            annotation.frame_id == frame.frame_id
            and annotation.clip == frame.clip
            and annotation.frame_index == frame.frame_index
            and annotation.image_width == frame.width
            and annotation.image_height == frame.height
        )
        if not metadata:
            broken.append(f"{frame.frame_id}: annotation metadata does not match the export")
            continue
        if annotation.image_sha256 != frame.sha256:
            broken.append(f"{frame.frame_id}: annotated a different image than the export")
            continue
        problems = validate(annotation, require_complete=True)
        (broken if problems else done).append(
            f"{frame.frame_id}: {len(problems)} problem(s)" if problems else frame.frame_id
        )

    print(f"pass {pass_dir.name}: {len(done)}/{len(expected)} complete")
    for label, items in (("pending", pending), ("needs fixing", broken)):
        if items:
            print(f"\n{label}:")
            for item in items:
                print(f"  - {item}")

    # A pass holding two prompt versions is the quiet way this benchmark goes wrong.
    # 'consensus' compares the two passes' versions against each other, so a stale
    # frame annotated under an older prompt in *both* passes sails through, and so
    # does a pilot annotated before a mid-run prompt change. Neither is comparable
    # with the rest, and nothing later in the pipeline can tell.
    if versions:
        print()
        for version, frames in sorted(versions.items()):
            print(f"prompt {version}: {len(frames)} frame(s)")
    if len(versions) > 1:
        print(
            "\nMIXED PROMPT VERSIONS -- these labels are not comparable and must not "
            "be pooled.\nRe-annotate every frame on an older version, or move it out "
            "of this pass directory:"
        )
        for version, frames in sorted(versions.items()):
            for frame_id in sorted(frames):
                print(f"  - {frame_id}: {version}")
        return 1
    if expected_prompt_version and versions:
        found = next(iter(versions))
        if found != expected_prompt_version:
            print(
                f"\nWRONG PROMPT VERSION -- this pass is on {found}, expected "
                f"{expected_prompt_version}. Every frame here needs re-annotating."
            )
            return 1
    if annotators:
        print()
        for annotator, frames in sorted(annotators.items()):
            print(f"annotator {annotator}: {len(frames)} frame(s)")
    if len(annotators) > 1:
        print(
            "\nMIXED ANNOTATOR IDS -- one isolated pass must use the runtime model "
            "recorded for that pass on every frame."
        )
        return 1
    if args.expect_annotator_id and annotators:
        found = next(iter(annotators))
        if found != args.expect_annotator_id:
            print(
                f"\nWRONG ANNOTATOR ID -- this pass records {found}, expected "
                f"{args.expect_annotator_id}."
            )
            return 1
    return 1 if pending or broken else 0


def command_glossary(args: argparse.Namespace) -> int:
    from .glossary import render_markdown

    print(render_markdown(load_registered_layout(args.layout)))
    return 0


def command_render_synthetic(args: argparse.Namespace) -> int:
    from .synthetic import render_scene, write_scene

    layout = load_registered_layout(args.layout)
    scene = render_scene(
        layout,
        scene_id=args.scene_id,
        seed=args.seed,
        distractors=not args.no_distractors,
    )
    path = write_scene(scene, args.out)
    visible = scene.visible()
    print(f"scene:   {scene.scene_id} ({scene.width}x{scene.height})")
    print(f"frame:   {path}")
    print(f"truth:   {Path(args.out) / 'truth.json'}  (outside the frame tree, by design)")
    print(f"visible: {len(visible)} of {len(scene.truth)} markings")
    for feature in sorted(visible, key=lambda f: f.value):
        print(f"  - {feature.value}")
    return 0


def command_score_truth(args: argparse.Namespace) -> int:
    from contracts.annotations import FrameAnnotation, validate

    from .synthetic import load_truth, score_against_truth

    with open(args.annotation, encoding="utf-8") as handle:
        annotation = FrameAnnotation.from_dict(json.load(handle))
    problems = validate(annotation, require_complete=True)
    if annotation.coordinate_space != "frame":
        problems.append("synthetic annotation must be resolved to frame space")
    if problems:
        print("annotation is invalid or incomplete:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    truth = load_truth(args.truth)

    per_feature, summary = score_against_truth(annotation, truth)

    print(f"{annotation.frame_id}  annotator={annotation.annotator_id}  "
          f"prompt={annotation.prompt_version}")
    print(f"\n{'feature':<32} {'n':>4} {'median':>8} {'p95':>8} {'max':>8} {'signed':>8}")
    for item in sorted(per_feature, key=lambda f: f.feature.value):
        print(
            f"{item.feature.value:<32} {item.sample_count:>4} {item.median_px:>8.2f} "
            f"{item.p95_px:>8.2f} {item.max_px:>8.2f} {item.mean_signed_px:>+8.2f}"
        )

    print(f"\noverall  median {summary['median_px']:.2f} px   p95 {summary['p95_px']:.2f} px   "
          f"coverage {summary['coverage']:.0%}")
    if summary["annotated_features_absent_from_truth"]:
        print(f"invented features: {', '.join(summary['annotated_features_absent_from_truth'])}")

    # The gate from the milestone plan, applied rather than described.
    checks = [
        ("median <= 1.50 px", summary["median_px"] <= 1.5),
        ("p95 <= 4.00 px", summary["p95_px"] <= 4.0),
        ("|mean signed| <= 0.75 px per family", summary["worst_mean_signed_px"] <= 0.75),
        ("no invented features", not summary["annotated_features_absent_from_truth"]),
        ("every visible feature has a disposition", not summary["missing_dispositions"]),
    ]
    print()
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
    return 0 if all(passed for _, passed in checks) else 1


def command_consensus(args: argparse.Namespace) -> int:
    from contracts.annotations import validate

    from .consensus import build_consensus, load_pass, summarize, write_consensus
    from .frames import read_export_index
    from .workflow import load_prompt_manifest

    if Path(args.pass_a).resolve() == Path(args.pass_b).resolve():
        print("pass A and pass B point to the same directory; the passes are not isolated")
        return 1

    layout = load_registered_layout(args.layout)
    expected_prompt_version = next(
        entry["version"]
        for entry in load_prompt_manifest()["prompts"]
        if entry["role"] == "annotator"
    )
    results = []
    missing = []
    invalid = []

    for frame in read_export_index(args.frames):
        annotation_a = load_pass(args.pass_a, frame.frame_id)
        annotation_b = load_pass(args.pass_b, frame.frame_id)
        if annotation_a is None or annotation_b is None:
            missing.append(frame.frame_id)
            continue
        frame_invalid = False
        for label, annotation in (("A", annotation_a), ("B", annotation_b)):
            problems = validate(annotation, require_complete=True)
            if annotation.prompt_version != expected_prompt_version:
                problems.append(
                    f"prompt version is {annotation.prompt_version}, expected frozen "
                    f"{expected_prompt_version}"
                )
            if annotation.coordinate_space != "frame":
                problems.append("annotation has not been resolved to frame space")
            metadata_matches = (
                annotation.frame_id == frame.frame_id
                and annotation.clip == frame.clip
                and annotation.frame_index == frame.frame_index
                and annotation.image_width == frame.width
                and annotation.image_height == frame.height
                and annotation.image_sha256 == frame.sha256
            )
            if not metadata_matches:
                problems.append("annotation does not match the exported frame")
            if problems:
                frame_invalid = True
                invalid.append(
                    f"{frame.frame_id} pass {label}: " + "; ".join(problems)
                )
        if frame_invalid:
            continue
        results.append(build_consensus(annotation_a, annotation_b, layout))

    if missing or invalid:
        if missing:
            print("incomplete passes; missing frames:")
            for frame_id in missing:
                print(f"  - {frame_id}")
        if invalid:
            print("invalid or incomplete annotations:")
            for problem in invalid:
                print(f"  - {problem}")
        return 1
    if not results:
        print("no frame has both passes; nothing to compare")
        return 1

    silver_dir, report, queue = write_consensus(args.out, results)
    summary = summarize(results)

    print(f"frames compared: {summary['frames']}")
    print(f"verdicts: {summary['verdicts']}")
    print(f"agreement rate: {summary['agreement_rate']:.0%}")
    print(f"disagreement: median {summary['median_disagreement_px']:.2f} px  "
          f"p95 {summary['p95_disagreement_px']:.2f} px")
    print(f"identity conflicts: {summary['identity_conflicts']}")

    for result in results:
        for problem in result.problems:
            print(f"  ! {result.frame_id}: {problem}")

    print(f"\nsilver: {silver_dir}")
    print(f"report: {report}")
    print(f"adjudication queue: {queue}")

    # The pilot gate, applied.
    checks = [
        ("median disagreement <= 2.00 px", summary["median_disagreement_px"] <= 2.0),
        ("p95 disagreement <= 6.00 px", summary["p95_disagreement_px"] <= 6.0),
        ("zero identity conflicts", summary["identity_conflicts"] == 0),
        ("no cross-pass problems", summary["frames_with_problems"] == 0),
    ]
    print()
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
    return 0 if all(passed for _, passed in checks) else 1


def command_score(args: argparse.Namespace) -> int:
    from contracts.annotations import FrameAnnotation

    from .metrics import (
        compare_summaries,
        load_calibration_transforms,
        score_calibration_entry,
        summarize_scores,
    )

    layout = load_registered_layout(args.layout)
    run_dirs = [Path(run) for run in args.run]
    transforms = {}
    for run_dir in run_dirs:
        loaded = load_calibration_transforms(run_dir / "calibrations.jsonl")
        overlap = sorted(set(transforms) & set(loaded))
        if overlap:
            print(
                f"duplicate calibration frames across --run inputs: "
                f"{', '.join(overlap[:4])}"
            )
            return 1
        transforms.update(loaded)

    scores = []
    missing = []
    for path in sorted(Path(args.reference).glob("*.json")):
        annotation = FrameAnnotation.from_dict(json.loads(path.read_text(encoding="utf-8")))
        entry = transforms.get(annotation.frame_id)
        if entry is None:
            missing.append(annotation.frame_id)
            continue
        try:
            scores.append(score_calibration_entry(annotation, entry, layout))
        except ValueError as error:
            print(f"refusing to score {annotation.frame_id}: {error}")
            return 1

    if missing:
        print("selected reference frames missing from the supplied calibration runs:")
        for frame_id in missing:
            print(f"  - {frame_id}")
        return 1
    if not scores:
        print("no reference frame matched the supplied calibration runs")
        return 1

    summary = summarize_scores(scores)
    print("runs:")
    for run_dir in run_dirs:
        print(f"  - {run_dir}")
    print(f"frames: {summary['frames_scored']} scored, {summary['frames_unscored']} unscored")
    print(f"statuses: {summary['status_counts']}")

    print(f"\n{'band':<10} {'n':>6} {'median px':>10} {'p95 px':>9} {'median ft':>10} {'p95 ft':>8}")
    for name, band in summary["bands"].items():
        print(
            f"{name:<10} {band['sample_count']:>6} {band['median_px']:>10.2f} "
            f"{band['p95_px']:>9.2f} {band['median_ft']:>10.3f} {band['p95_ft']:>8.3f}"
        )

    held_out = summary["held_out_median_px"]
    if held_out == held_out:  # not NaN
        print(f"\nheld-out families (never fitted): median {held_out:.2f} px")

    if args.out:
        payload = {
            "runs": [str(run_dir) for run_dir in run_dirs],
            "summary": summary,
            "frames": [s.as_dict() for s in scores],
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        print(f"\nwrote {args.out}")

    if args.baseline:
        with open(args.baseline, encoding="utf-8") as handle:
            before = json.load(handle)["summary"]
        delta = compare_summaries(before, summary)
        print("\nversus baseline, per band:")
        for name, change in delta["bands"].items():
            print(
                f"  {name:<10} median {change['median_px_before']:.2f} -> "
                f"{change['median_px_after']:.2f} px "
                f"({change['median_px_change_pct']:+.1f}%)"
            )
    return 0


def command_score_extraction(args: argparse.Namespace) -> int:
    from .extraction import score_extraction

    report = score_extraction(args.records, args.references)
    print(f"frames: {len(report['frames'])}  records: {report['records']}")
    print(f"wrong three-point accepts: {report['wrong_three_point_accepts']}")
    print(f"wrong-identity accepts: {report['wrong_identity_accepts']}")
    print(f"skipped/unannotated accepts: {report['skipped_or_unannotated_accepts']}")
    print(f"sample median: {report['all_sample_median_px']:.2f}px  p95: {report['all_sample_p95_px']:.2f}px")
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
        print(f"wrote {path}")
    return 0


def command_score_hybrid(args: argparse.Namespace) -> int:
    from .hybrid import score_hybrid

    report = score_hybrid(args.records, args.references, load_registered_layout(args.layout))
    print(f"records: {report['records']}  paired frames: {len(report['frame_sets']['paired'])}")
    print(f"acceptance: {report['acceptance']}")
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
        print(f"wrote {path}")
    return 0


COMMANDS = {
    "export-frames": command_export_frames,
    "init-run": command_init_run,
    "render-synthetic": command_render_synthetic,
    "score-truth": command_score_truth,
    "consensus": command_consensus,
    "score": command_score,
    "score-extraction": command_score_extraction,
    "score-hybrid": command_score_hybrid,
    "verify-frames": command_verify_frames,
    "verify-prompts": command_verify_prompts,
    "crop": command_crop,
    "resolve": command_resolve,
    "validate": command_validate,
    "status": command_status,
    "glossary": command_glossary,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
