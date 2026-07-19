"""Export corrected keypoint labels to an Ultralytics YOLO pose dataset.

Takes every frame with a saved label under dataset/labels/ and writes an
images/ + labels/ tree with normalized pose annotations plus a data.yaml,
ready for `yolo pose train`. Re-running regenerates the export from scratch;
the train/val split is deterministic per frame so it stays stable as the
dataset grows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from pipeline_manifest import DEFAULT_DATASET_DIR, PROJECT_DIR, load_manifest

DEFAULT_OUT = PROJECT_DIR / "export" / "court_pose"
DEFAULT_SCHEMA = PROJECT_DIR / "dataset" / "schemas" / "court_keypoints.v2.json"
# Reflection across the north/south axis swaps east and west fixed locations.
EAST_WEST_FLIP_IDX = [5, 4, 3, 2, 1, 0, 7, 6, 9, 8, 15, 14, 13, 12, 11, 10, 17, 16]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export labeled frames as a YOLO pose dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Dataset directory (default: {DEFAULT_DATASET_DIR})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output directory (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=DEFAULT_SCHEMA,
        help=f"Canonical keypoint schema (default: {DEFAULT_SCHEMA})",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.2,
        help="Fraction of frames assigned to the val split (default: 0.2)",
    )
    parser.add_argument(
        "--bbox",
        choices=("full", "hull"),
        default="full",
        help="Object bbox: 'full' frame (default) or padded 'hull' of visible points",
    )
    return parser.parse_args()


def split_for(frame_id: str, val_frac: float) -> str:
    digest = int(hashlib.sha1(frame_id.encode("utf-8")).hexdigest(), 16)
    return "val" if (digest % 100) < round(val_frac * 100) else "train"


def bbox_line(keypoints: list[dict], width: int, height: int, mode: str) -> str:
    if mode == "hull":
        xs = [p["x"] for p in keypoints if p["v"] > 0]
        ys = [p["y"] for p in keypoints if p["v"] > 0]
        if xs:
            pad = 0.02
            x_min = max(0.0, min(xs) / width - pad)
            x_max = min(1.0, max(xs) / width + pad)
            y_min = max(0.0, min(ys) / height - pad)
            y_max = min(1.0, max(ys) / height + pad)
            return (
                f"{(x_min + x_max) / 2:.6f} {(y_min + y_max) / 2:.6f} "
                f"{x_max - x_min:.6f} {y_max - y_min:.6f}"
            )
    return "0.500000 0.500000 1.000000 1.000000"


def main() -> None:
    args = parse_args()
    if not args.schema.is_file():
        raise SystemExit(f"Canonical keypoint schema not found: {args.schema}")
    with open(args.schema, "r", encoding="utf-8") as handle:
        schema = json.load(handle)
    expected_keypoints = len(schema.get("keypoints", []))
    if expected_keypoints != 18:
        raise SystemExit(f"Canonical schema must define 18 keypoints, found {expected_keypoints}")
    manifest = load_manifest(args.dataset)

    labeled = [
        (frame_id, frame)
        for frame_id, frame in sorted(manifest["frames"].items())
        if frame.get("label_status") == "labeled"
    ]
    if not labeled:
        raise SystemExit("No labeled frames to export. Label some frames first.")

    if args.out.exists():
        shutil.rmtree(args.out)
    for split in ("train", "val"):
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)

    num_keypoints = None
    split_counts = {"train": 0, "val": 0}
    for frame_id, frame in labeled:
        label_file = args.dataset / "labels" / frame["clip_id"] / f"{frame_id}.json"
        if not label_file.is_file():
            print(f"warning: missing label file for {frame_id}, skipping")
            continue
        with open(label_file, "r", encoding="utf-8") as handle:
            label = json.load(handle)

        label_schema = label.get("schema", {})
        if (
            label_schema.get("schema_name") != schema.get("schema_name")
            or label_schema.get("schema_version") != schema.get("schema_version")
        ):
            raise SystemExit(
                f"{frame_id} does not declare canonical schema {schema['schema_name']} "
                f"{schema['schema_version']}"
            )
        keypoints = label["keypoints"]
        width, height = label["image_w"], label["image_h"]
        if len(keypoints) != expected_keypoints:
            raise SystemExit(
                f"{frame_id} has {len(keypoints)} keypoints, expected {expected_keypoints}"
            )
        if num_keypoints is None:
            num_keypoints = len(keypoints)

        parts = ["0", bbox_line(keypoints, width, height, args.bbox)]
        for point in keypoints:
            if point["v"] == 0:
                parts.append("0.000000 0.000000 0")
            else:
                x = min(1.0, max(0.0, point["x"] / width))
                y = min(1.0, max(0.0, point["y"] / height))
                parts.append(f"{x:.6f} {y:.6f} {point['v']}")

        split = split_for(frame_id, args.val_frac)
        split_counts[split] += 1
        image_src = args.dataset / frame["image"]
        shutil.copy2(image_src, args.out / "images" / split / image_src.name)
        with open(args.out / "labels" / split / f"{frame_id}.txt", "w") as handle:
            handle.write(" ".join(parts) + "\n")

    shutil.copy2(args.schema, args.out / args.schema.name)
    yaml_text = (
        f"# Generated by export_yolo.py\n"
        f"# Canonical schema: {args.schema.name} ({schema['schema_version']})\n"
        f"path: {args.out.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"kpt_shape: [{num_keypoints}, 3]\n"
        f"flip_idx: {EAST_WEST_FLIP_IDX}\n"
        f"names:\n  0: court\n"
    )
    with open(args.out / "data.yaml", "w", encoding="utf-8") as handle:
        handle.write(yaml_text)

    print(
        f"Exported {split_counts['train']} train + {split_counts['val']} val "
        f"frames ({num_keypoints} keypoints each) to {args.out}"
    )


if __name__ == "__main__":
    main()
