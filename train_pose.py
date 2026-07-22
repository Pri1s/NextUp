"""Fine-tune a YOLO pose model on the exported court-keypoint dataset.

Thin wrapper around ultralytics training with defaults tuned for this
project: COCO-pretrained init (backbone/neck transfer, pose head is
reinitialized for kpt_shape since COCO pose uses 17 human keypoints, not our
22 court keypoints), MPS on Apple Silicon, and the run name LABELING.md
already documents for --model prefill.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = PROJECT_DIR / "export" / "court_pose" / "data.yaml"


def frame_splits(data_yaml: Path) -> dict[str, str]:
    """frame_id -> 'train'/'val', read from the exported label filenames.

    Mirrors export_yolo.py's images/<split> <-> labels/<split> layout next to
    data_yaml, so serve.py can later show which frames a given checkpoint
    actually trained on.
    """
    base = data_yaml.parent
    splits: dict[str, str] = {}
    for split in ("train", "val"):
        for label_file in sorted((base / "labels" / split).glob("*.txt")):
            splits[label_file.stem] = split
    return splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a YOLO pose model on dataset/labels via export_yolo.py's output."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--model",
        default="yolo11n-pose.pt",
        help="Starting weights: a COCO-pretrained pose checkpoint, or a "
        "runs/pose/.../weights/*.pt to resume fine-tuning (default: yolo11n-pose.pt)",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--patience", type=int, default=30, help="Early-stop patience")
    parser.add_argument(
        "--device", default=None, help="cpu/mps/0 (default: let ultralytics auto-select)"
    )
    parser.add_argument("--project", default=str(PROJECT_DIR / "runs" / "pose"))
    parser.add_argument("--name", default="court_pose_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.is_file():
        raise SystemExit(f"data.yaml not found: {args.data}. Run export_yolo.py first.")

    splits = frame_splits(args.data)

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project=args.project,
        name=args.name,
    )

    save_dir = Path(model.trainer.save_dir)
    manifest = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "data": str(args.data),
        "model_init": args.model,
        "frames": splits,
    }
    manifest_path = save_dir / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote dataset manifest ({len(splits)} frames) to {manifest_path}")


if __name__ == "__main__":
    main()
