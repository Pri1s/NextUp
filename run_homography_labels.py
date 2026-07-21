#!/usr/bin/env python3
"""Run classical court-geometry detection, solving, and verification on frames."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from court_homography import DEFAULT_TEMPLATE, process_frame
from hud_detection import build_hud_context

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classical-CV basketball court homography labeler (no learned model).")
    parser.add_argument("frames", type=Path, help="Image file or folder of frames (folders are searched recursively).")
    parser.add_argument("--output", type=Path, default=Path("homography_results"), help="Output folder for per-frame JSON and batch_summary.json.")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE, help="Court template JSON; use court_templates/fiba.json or a compatible custom file.")
    parser.add_argument("--hypotheses", type=int, default=5000, help="Maximum guided orientation/adjacency proposal budget per frame.")
    parser.add_argument(
        "--hud-context",
        type=Path,
        action="append",
        default=[],
        help="Optional same-clip image file/folder for temporal HUD detection; repeat as needed.",
    )
    parser.add_argument("--line-distance-px", type=float, default=10.0, help="Maximum pixel line residual for agreement scoring.")
    parser.add_argument("--max-error-px", type=float, default=12.0, help="Verification gate maximum mean matched-keypoint error.")
    parser.add_argument("--min-matched-keypoints", type=int, default=4)
    parser.add_argument("--min-independent-line-inliers", type=int, default=2)
    parser.add_argument("--max-outside-fraction", type=float, default=.85)
    return parser.parse_args()


def images_at(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported frame type: {path}")
        return [path]
    if not path.is_dir():
        raise ValueError(f"Frame input does not exist: {path}")
    return sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)


def output_path(source: Path, input_root: Path, output_root: Path) -> Path:
    relative = Path(source.name) if input_root.is_file() else source.relative_to(input_root)
    return output_root / relative.with_suffix(".json")


def main() -> int:
    args = parse_args()
    try:
        frames = images_at(args.frames)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if not frames:
        print("error: no supported image frames found", file=sys.stderr)
        return 2
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    options = {"hypotheses": args.hypotheses, "line_distance_px": args.line_distance_px}
    for number, frame in enumerate(frames, 1):
        print(f"[{number}/{len(frames)}] {frame}", flush=True)
        try:
            hud_context = build_hud_context(frame, args.hud_context)
            result = process_frame(frame, args.template, hud_context=hud_context, **options)
            # Keep all gate parameters in every result, rather than hiding the
            # decision behind the CLI defaults.
            from court_homography import verify_solution
            from court_homography import load_template
            result["verification"] = verify_solution(result["solution"], result["detection"], load_template(args.template), max_error_px=args.max_error_px, min_matched_keypoints=args.min_matched_keypoints, min_independent_line_inliers=args.min_independent_line_inliers, max_outside_fraction=args.max_outside_fraction)
        except Exception as error:  # Preserve one failed frame without dropping the batch.
            result = {"frame": str(frame), "error": str(error), "verification": {"status": "fail", "reasons": ["processing_error"]}}
        destination = output_path(frame, args.frames, args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        solution = result.get("solution", {})
        records.append({
            "frame": str(frame), "result": str(destination.relative_to(args.output)),
            "solution_status": solution.get("status", "processing_error"),
            "solution_reason": solution.get("reason"),
            "gate_status": result["verification"]["status"], "reasons": result["verification"].get("reasons", []),
            # Diagnostic-only: never affects gate_status or pass_rate below. A
            # human reviewer decides whether this candidate is usable, and that
            # decision is recorded separately (see review_homography_labels.py).
            "review_candidate_available": bool(solution.get("review_candidate")),
        })
    passed = sum(record["gate_status"] == "pass" for record in records)
    review_available = sum(record["review_candidate_available"] for record in records)
    summary = {"input": str(args.frames), "template": str(args.template), "frames": records, "totals": {"frames": len(records), "passed": passed, "failed": len(records) - passed, "pass_rate": round(passed / len(records), 4), "review_candidates_available": review_available}}
    (args.output / "batch_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Gate pass rate: {passed}/{len(records)} ({passed / len(records):.1%})")
    print(f"Review-required candidates available (still automatic failures): {review_available}/{len(records)}")
    print(f"Results: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
