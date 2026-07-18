"""Shared manifest for the frame extraction / triage / labeling pipeline.

The manifest is a single JSON file (dataset/manifest.json) that tracks every
clip that has been processed and every candidate frame extracted from it,
including triage decisions and labeling progress. All pipeline tools read and
write it through this module. Writes are atomic (temp file + os.replace) so a
crash never leaves a half-written manifest.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = PROJECT_DIR / "dataset"

MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1

TRIAGE_STATES = ("pending", "keep", "skip")
LABEL_STATES = ("unlabeled", "labeled")


def empty_manifest() -> dict:
    return {"version": MANIFEST_VERSION, "clips": {}, "frames": {}}


def load_manifest(dataset_dir: Path = DEFAULT_DATASET_DIR) -> dict:
    manifest_path = Path(dataset_dir) / MANIFEST_NAME
    if not manifest_path.is_file():
        return empty_manifest()
    with open(manifest_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    data.setdefault("version", MANIFEST_VERSION)
    data.setdefault("clips", {})
    data.setdefault("frames", {})
    return data


def save_manifest(dataset_dir: Path, data: dict) -> None:
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dataset_dir / MANIFEST_NAME
    fd, temp_path = tempfile.mkstemp(
        prefix=".manifest-", suffix=".json", dir=dataset_dir
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temp_path, manifest_path)
    except BaseException:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise


def counts(data: dict, clip_id: str | None = None) -> dict:
    """Progress counts, overall or restricted to one clip."""
    frames = data.get("frames", {})
    if clip_id is not None:
        frames = {
            frame_id: frame
            for frame_id, frame in frames.items()
            if frame.get("clip_id") == clip_id
        }
        clip_count = 1 if clip_id in data.get("clips", {}) else 0
    else:
        clip_count = len(data.get("clips", {}))

    tally = {
        "clips": clip_count,
        "candidates": len(frames),
        "pending": 0,
        "keep": 0,
        "skip": 0,
        "labeled": 0,
    }
    for frame in frames.values():
        triage = frame.get("triage", "pending")
        if triage in tally:
            tally[triage] += 1
        if frame.get("label_status") == "labeled":
            tally["labeled"] += 1
    return tally


def format_counts(tally: dict) -> str:
    return (
        f"clips: {tally['clips']} | candidates: {tally['candidates']} | "
        f"pending: {tally['pending']} | keep: {tally['keep']} | "
        f"skip: {tally['skip']} | labeled: {tally['labeled']}"
    )


def print_counts(data: dict) -> None:
    print(format_counts(counts(data)))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Show pipeline progress counts.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Dataset directory (default: {DEFAULT_DATASET_DIR})",
    )
    args = parser.parse_args()

    data = load_manifest(args.dataset)
    print_counts(data)
    for clip_id in sorted(data.get("clips", {})):
        tally = counts(data, clip_id=clip_id)
        print(
            f"  {clip_id}: {tally['candidates']} candidates, "
            f"{tally['pending']} pending, {tally['keep']} keep, "
            f"{tally['skip']} skip, {tally['labeled']} labeled"
        )


if __name__ == "__main__":
    main()
