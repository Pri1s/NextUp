"""One-off migration to the v3 keypoint schema. Idempotent.

Removes artifacts recorded under schema 2.x, which are incompatible with v3:
  - deletes dataset/labels/**/*.json whose schema.schema_version is not 3.x
    and resets those frames' label_status in the manifest (the manifest is
    independent of the label files — deleting files alone would leave stale
    "labeled" counts and exports silently skipping frames);
  - strips clip orientation blocks not recorded under a 3.x schema, so every
    clip re-anchors through the new orientation flow.

Stop serve.py before running this — both write the manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_manifest import DEFAULT_DATASET_DIR, load_manifest, save_manifest
from serve import SCHEMA_VERSION_PREFIX, clip_orientation


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove pre-v3 labels and orientation locks.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Dataset directory (default: {DEFAULT_DATASET_DIR})",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.dataset)
    changed = False

    removed_labels = 0
    for label_file in sorted((args.dataset / "labels").glob("*/*.json")):
        with open(label_file, "r", encoding="utf-8") as handle:
            label = json.load(handle)
        version = str(label.get("schema", {}).get("schema_version", ""))
        if version.startswith(SCHEMA_VERSION_PREFIX):
            continue
        frame_id = label_file.stem
        label_file.unlink()
        removed_labels += 1
        changed = True
        frame = manifest["frames"].get(frame_id)
        if frame is not None and frame.get("label_status") == "labeled":
            frame["label_status"] = "unlabeled"
        print(f"removed {label_file.relative_to(args.dataset)} (schema {version or 'unknown'})")

    reset_status = 0
    for frame_id, frame in manifest["frames"].items():
        if frame.get("label_status") != "labeled":
            continue
        clip_dir = args.dataset / "labels" / frame["clip_id"]
        if not (clip_dir / f"{frame_id}.json").is_file():
            frame["label_status"] = "unlabeled"
            reset_status += 1
            changed = True
            print(f"reset label_status for {frame_id} (no label file)")

    removed_orientations = 0
    for clip_id, clip in manifest["clips"].items():
        if "orientation" in clip and clip_orientation(clip) is None:
            del clip["orientation"]
            removed_orientations += 1
            changed = True
            print(f"removed pre-v3 orientation lock on clip {clip_id}")

    if changed:
        save_manifest(args.dataset, manifest)
    print(
        f"done: {removed_labels} label file(s) removed, "
        f"{reset_status} orphaned label_status reset, "
        f"{removed_orientations} orientation lock(s) removed"
        + ("" if changed else " (nothing to do)")
    )


if __name__ == "__main__":
    main()
