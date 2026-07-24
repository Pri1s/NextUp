"""CLI for the full-game frame-processing pipeline.

    python -m engine.pipeline.cli --video game.mp4 --profile nba --fps 5

Streams the clip into ``engine_out/<stem>.<profile>/{run.json, records/seg_*.jsonl}``.
The run is resumable: re-run the same command to continue an interrupted run
(changing the fps/model/video starts fresh, since the resume key changes).

Smoke test on the bundled clip:

    python -m engine.pipeline.cli --video input_videos/001_video_4.mp4 \
        --profile nba --fps 5 --max-seconds 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

from engine.pipeline.records import run_dir_for
from engine.pipeline.runner import DEFAULT_OUT, PipelineConfig, run_pipeline
from engine.pipeline.scene import DEFAULT_SCENE_THRESHOLD
from engine.profiles import profile_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--video", type=Path, required=True, help="Input clip (a full game is fine)")
    parser.add_argument("--profile", default="nba", choices=profile_names(), help="Model/court profile")
    parser.add_argument("--fps", type=float, default=5.0, dest="target_fps", help="Target processing fps (default: 5)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"Output root (default: {DEFAULT_OUT})")
    parser.add_argument("--segment-frames", type=int, default=1000, help="Frames per shard/checkpoint (default: 1000)")
    parser.add_argument("--batch", type=int, default=8, help="Detector batch size (default: 8; lower on <8GB accelerators)")
    parser.add_argument("--device", default=None, help="auto|cpu|cuda|mps (default: auto)")
    parser.add_argument("--ball-conf", type=float, default=0.3, help="Min confidence for a credible ball (default: 0.3)")
    parser.add_argument("--scene-threshold", type=float, default=DEFAULT_SCENE_THRESHOLD, help="Histogram-correlation cut threshold")
    parser.add_argument("--max-seconds", type=float, default=None, help="Cap processing to the first N seconds (smoke test)")
    parser.add_argument("--max-frames", type=int, default=None, help="Cap processed frames (overrides --max-seconds)")
    parser.add_argument("--no-resume", action="store_true", help="Ignore any existing run and start fresh")
    parser.add_argument("--format", default="jsonl", choices=["jsonl"], dest="fmt", help="Record format (parquet planned)")
    parser.add_argument("--keep-keypoints", action="store_true", help="Store raw 22x3 court keypoints per frame (debug)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.video.is_file():
        raise SystemExit(f"Video not found: {args.video}")

    max_frames = args.max_frames
    if max_frames is None and args.max_seconds is not None:
        max_frames = max(1, round(args.max_seconds * args.target_fps))

    device = None if args.device in (None, "auto") else args.device

    config = PipelineConfig(
        target_fps=args.target_fps,
        segment_frames=args.segment_frames,
        batch=args.batch,
        device=device,
        ball_conf=args.ball_conf,
        scene_threshold=args.scene_threshold,
        max_frames=max_frames,
        keep_keypoints=args.keep_keypoints,
        fmt=args.fmt,
        resume=not args.no_resume,
    )

    manifest = run_pipeline(args.video, args.profile, config, out_dir=args.out)

    run_dir = run_dir_for(args.out, args.video.stem, args.profile)
    summary = manifest.get("summary", {})
    print(f"status: {manifest.get('status')}")
    print(f"profile={args.profile}  stride={manifest['config']['stride']}  device={device or 'auto'}")
    print(
        f"frames processed: {summary.get('frames_processed', 0)}  "
        f"with homography: {summary.get('frames_with_homography', 0)}  "
        f"shots: {summary.get('shots', 0)}  "
        f"unique tracks: {summary.get('unique_tracks', 0)}  "
        f"errors: {summary.get('errors', 0)}"
    )
    print(f"records: {run_dir / 'records'}")
    print(f"manifest: {run_dir / 'run.json'}")


if __name__ == "__main__":
    main()
