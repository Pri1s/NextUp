"""Engine entry point / smoke test.

Runs a clip through the plug boundary: for each sampled frame it detects
players/ball (if the profile has a detector), predicts the 22 canonical court
keypoints, fits the court homography, and projects each player's foot point onto
the court in feet. Writes a small JSON report (and a first-frame overlay image)
under the output directory.

This is the isolation proof, not an analyzer — there is deliberately no tracking,
team assignment, minimap, or analytics here (see ENGINE.md for the roadmap).

Examples:
    python -m engine.cli --video clip.mp4 --profile nba
    python -m engine.cli --video clip.mp4 --profile nba --inspect   # fill the NBA adapter map
    python -m engine.cli --video hs_clip.mp4 --profile hs           # plug test with the HS model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from contracts.court_schema import canonical_ids
from contracts.court_template import load_template
from engine.geometry.homography import project, solve_homography
from engine.io.video import iter_frames, probe
from engine.profiles import build_court_model, build_detector, get_profile, profile_names

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "engine_out"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, required=True, help="Input clip")
    parser.add_argument("--profile", default="nba", choices=profile_names(), help="Which model/court profile to run")
    parser.add_argument("--stride", type=int, default=15, help="Process every Nth frame (default: 15)")
    parser.add_argument("--max-frames", type=int, default=40, help="Cap frames processed (default: 40)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"Output directory (default: {DEFAULT_OUT})")
    parser.add_argument("--court-weights", type=Path, default=None, help="Override the profile's court model weights")
    parser.add_argument("--detector-weights", type=Path, default=None, help="Override the profile's detector weights")
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print the court model's raw per-index keypoints on a frame (to fill the NBA adapter map), then exit",
    )
    return parser.parse_args()


def _inspect(court_model, video: Path, max_frames: int) -> None:
    print(f"court model kpt_shape: {court_model.kpt_shape}")
    for frame_index, _ts, frame in iter_frames(video, stride=15, max_frames=max_frames):
        source = court_model.predict_source(frame)
        if source.shape[0] == 0:
            continue
        print(f"frame {frame_index}: {source.shape[0]} native keypoints (source_index: x, y, v)")
        for i, (x, y, v) in enumerate(source):
            print(f"  {i:2d}: ({x:8.1f}, {y:8.1f})  v={int(v)}")
        print(
            "\nMatch each source_index above to a canonical id and fill "
            "NBA_COURT_KEYPOINT_MAP in engine/court/adapters.py."
        )
        return
    print("No frames produced court keypoints; try a clearer clip or a lower confidence.")


def _overlay(frame, canonical_kpts, detections, path: Path) -> None:
    image = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.xyxy)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 165, 255), 2)
    for i, (x, y, v) in enumerate(canonical_kpts):
        if v > 0:
            cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.putText(image, str(i + 1), (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def main() -> None:
    args = parse_args()
    if not args.video.is_file():
        raise SystemExit(f"Video not found: {args.video}")

    profile = get_profile(
        args.profile, court_weights=args.court_weights, detector_weights=args.detector_weights
    )
    court_model = build_court_model(profile)

    if args.inspect:
        _inspect(court_model, args.video, args.max_frames)
        return

    template = load_template(profile.court_template)
    detector = build_detector(profile)
    ids = canonical_ids()
    info = probe(args.video)

    frames_report = []
    overlay_written = False
    homography_frames = 0
    for frame_index, timestamp, frame in iter_frames(args.video, stride=args.stride, max_frames=args.max_frames):
        canonical_kpts = court_model.predict(frame)
        visible = int((canonical_kpts[:, 2] > 0).sum())
        homography, used = solve_homography(canonical_kpts, template)

        detections = detector.detect(frame) if detector is not None else []
        players = []
        foot_pts = [det.foot_point() for det in detections]
        court_pts = project(homography, foot_pts) if (homography is not None and foot_pts) else None
        for i, det in enumerate(detections):
            players.append(
                {
                    "cls": det.cls,
                    "name": det.name,
                    "conf": round(det.conf, 4),
                    "foot_px": [round(v, 1) for v in det.foot_point()],
                    "court_ft": [round(float(v), 2) for v in court_pts[i]] if court_pts is not None else None,
                }
            )

        if homography is not None:
            homography_frames += 1
        frames_report.append(
            {
                "frame_index": frame_index,
                "timestamp_s": timestamp,
                "visible_court_keypoints": visible,
                "homography": bool(homography is not None),
                "used_keypoint_ids": used,
                "num_detections": len(detections),
                "players": players,
            }
        )

        if not overlay_written and (visible > 0 or detections):
            overlay_path = args.out / f"{args.video.stem}.{profile.name}.overlay.jpg"
            _overlay(frame, canonical_kpts, detections, overlay_path)
            overlay_written = True

    report = {
        "video": str(args.video),
        "profile": profile.name,
        "court_template": profile.court_template,
        "court_weights": str(profile.court_weights),
        "detector_weights": str(profile.detector_weights) if profile.detector_weights else None,
        "video_info": {"fps": info.fps, "width": info.width, "height": info.height, "frame_count": info.frame_count},
        "canonical_ids": ids,
        "summary": {
            "frames_processed": len(frames_report),
            "frames_with_homography": homography_frames,
        },
        "frames": frames_report,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / f"{args.video.stem}.{profile.name}.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    print(f"profile={profile.name} template={profile.court_template}")
    print(
        f"processed {len(frames_report)} frame(s); "
        f"{homography_frames} had a court homography; "
        f"detector={'on' if detector is not None else 'off'}"
    )
    print(f"report: {report_path}")
    if overlay_written:
        print(f"overlay: {args.out / f'{args.video.stem}.{profile.name}.overlay.jpg'}")


if __name__ == "__main__":
    main()
