"""Extract sparse candidate frames from game clips for triage and labeling.

Samples roughly one frame per --interval seconds from every video found in the
input directory, writing a full-resolution JPEG plus a small thumbnail per
sampled frame and recording provenance (clip, frame index, timestamp) in the
shared manifest. Re-running is safe: clips already extracted with the same
interval are skipped, and existing triage/label decisions are never touched.
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import cv2

from pipeline_manifest import (
    DEFAULT_DATASET_DIR,
    PROJECT_DIR,
    counts,
    format_counts,
    load_manifest,
    save_manifest,
)

DEFAULT_VIDEOS_DIR = PROJECT_DIR / "input_videos"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample candidate frames from clips into the dataset."
    )
    parser.add_argument(
        "--videos",
        type=Path,
        default=DEFAULT_VIDEOS_DIR,
        help=f"Directory of input clips (default: {DEFAULT_VIDEOS_DIR})",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Dataset directory (default: {DEFAULT_DATASET_DIR})",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between sampled frames (default: 1.0)",
    )
    parser.add_argument(
        "--thumb-width",
        type=int,
        default=320,
        help="Thumbnail width in pixels (default: 320)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract clips even if already in the manifest",
    )
    parser.add_argument(
        "--clip",
        default=None,
        help="Only process the clip with this id (filename stem)",
    )
    return parser.parse_args()


def clip_id_for(video_path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", video_path.stem)


def needs_extraction(
    clip_entry: dict | None, video_path: Path, interval: float
) -> bool:
    if clip_entry is None:
        return True
    stat = video_path.stat()
    return not (
        clip_entry.get("file_size") == stat.st_size
        and clip_entry.get("file_mtime") == stat.st_mtime
        and clip_entry.get("interval_s") == interval
    )


def extract_clip(
    video_path: Path,
    clip_id: str,
    dataset_dir: Path,
    interval: float,
    thumb_width: int,
) -> tuple[dict, dict]:
    """Sample frames from one clip. Returns (clip_entry, frame_entries)."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, round(fps * interval))

    frames_dir = dataset_dir / "frames" / clip_id
    thumbs_dir = dataset_dir / "thumbs" / clip_id
    frames_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    frame_entries: dict[str, dict] = {}
    frame_index = 0
    try:
        # Sequential decode: grab() every frame, retrieve() only sampled ones.
        # Seeking with CAP_PROP_POS_FRAMES is unreliable on some encodes.
        while True:
            if not capture.grab():
                break
            if frame_index % step == 0:
                read_success, frame = capture.retrieve()
                if not read_success:
                    break
                frame_id = f"{clip_id}_f{frame_index:06d}"
                image_rel = f"frames/{clip_id}/{frame_id}.jpg"
                thumb_rel = f"thumbs/{clip_id}/{frame_id}.jpg"
                cv2.imwrite(
                    str(dataset_dir / image_rel),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 92],
                )
                thumb_height = max(1, round(frame.shape[0] * thumb_width / frame.shape[1]))
                thumb = cv2.resize(
                    frame, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA
                )
                cv2.imwrite(str(dataset_dir / thumb_rel), thumb)
                frame_entries[frame_id] = {
                    "clip_id": clip_id,
                    "frame_index": frame_index,
                    "timestamp_s": round(frame_index / fps, 3),
                    "image": image_rel,
                    "thumb": thumb_rel,
                    "triage": "pending",
                    "label_status": "unlabeled",
                }
            frame_index += 1
    finally:
        capture.release()

    stat = video_path.stat()
    clip_entry = {
        "filename": video_path.name,
        "path": str(video_path),
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_index,
        "interval_s": interval,
        "file_size": stat.st_size,
        "file_mtime": stat.st_mtime,
        "extracted_at": datetime.now().isoformat(timespec="seconds"),
    }
    return clip_entry, frame_entries


def main() -> None:
    args = parse_args()
    if not args.videos.is_dir():
        raise SystemExit(f"Input video directory not found: {args.videos}")

    videos = sorted(
        path
        for path in args.videos.iterdir()
        if path.suffix.lower() in VIDEO_EXTENSIONS and path.is_file()
    )
    if args.clip is not None:
        videos = [path for path in videos if clip_id_for(path) == args.clip]
        if not videos:
            raise SystemExit(f"No video in {args.videos} matches clip id: {args.clip}")

    manifest = load_manifest(args.dataset)
    new_clips = 0
    skipped_clips = 0
    new_frames = 0

    for video_path in videos:
        clip_id = clip_id_for(video_path)
        existing = manifest["clips"].get(clip_id)
        if not args.force and not needs_extraction(existing, video_path, args.interval):
            skipped_clips += 1
            print(f"{clip_id}: already extracted, skipping")
            continue

        clip_entry, frame_entries = extract_clip(
            video_path, clip_id, args.dataset, args.interval, args.thumb_width
        )
        added = 0
        for frame_id, entry in frame_entries.items():
            previous = manifest["frames"].get(frame_id)
            if previous is not None:
                # Preserve triage and labeling progress across re-extraction.
                entry["triage"] = previous.get("triage", "pending")
                entry["label_status"] = previous.get("label_status", "unlabeled")
            else:
                added += 1
            manifest["frames"][frame_id] = entry
        manifest["clips"][clip_id] = clip_entry
        save_manifest(args.dataset, manifest)
        new_clips += 1
        new_frames += added
        print(
            f"{clip_id}: sampled {len(frame_entries)} frames "
            f"({added} new) from {clip_entry['frame_count']} at "
            f"1 per {args.interval:g}s"
        )

    print(
        f"\nProcessed {new_clips} clip(s), skipped {skipped_clips} already-extracted, "
        f"{new_frames} new candidate frame(s)."
    )
    print(format_counts(counts(manifest)))


if __name__ == "__main__":
    main()
