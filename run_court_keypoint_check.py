"""Quick visual sanity check for a basketball court keypoint model.

The script deliberately processes one frame at a time so that it can be used on
long game videos without first loading the whole video into memory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_DIR / "models" / "court_keypoint_detector.pt"
DEFAULT_OUTPUT = PROJECT_DIR / "output_videos" / "court_keypoints_check.mp4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a court keypoint model over a video and write an annotated copy."
    )
    parser.add_argument("input_video", type=Path, help="Path to the game video to inspect")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Path to the Ultralytics pose model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output",
        "--output-video",
        dest="output_video",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Annotated video path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Minimum model detection confidence (default: 0.5)",
    )
    parser.add_argument(
        "--keypoint-conf",
        type=float,
        default=0.25,
        help="Minimum keypoint confidence to draw (default: 0.25)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (default: 640)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Ultralytics device, for example cpu, 0, or 0,1 (default: automatic)",
    )
    return parser.parse_args()


def _confidence_color(confidence: float) -> Tuple[int, int, int]:
    """Return a BGR color that makes low-confidence points easy to spot."""
    # Red at 0 confidence, yellow around .5, and green at 1 confidence.
    confidence = max(0.0, min(1.0, confidence))
    red = int(255 * (1.0 - confidence))
    green = int(255 * confidence)
    return (0, green, red)


def _draw_keypoints(
    frame: Any,
    result: Any,
    keypoint_confidence_threshold: float,
) -> Tuple[int, Optional[float]]:
    """Draw all pose instances in a result and return visible count/mean confidence."""
    keypoints = getattr(result, "keypoints", None)
    xy_tensor = getattr(keypoints, "xy", None)
    if xy_tensor is None or len(xy_tensor) == 0:
        return 0, None

    xy = xy_tensor.cpu().numpy()
    confidence = getattr(keypoints, "conf", None)
    confidence_array = confidence.cpu().numpy() if confidence is not None else None

    # Per-keypoint confidence is available for normal Ultralytics pose output.
    # Fall back to the instance's box confidence for older/custom pose output.
    box_confidence = getattr(getattr(result, "boxes", None), "conf", None)
    box_confidence_array = (
        box_confidence.cpu().numpy() if box_confidence is not None else None
    )

    visible_confidences: List[float] = []
    for instance_index, instance in enumerate(xy):
        for keypoint_index, (x, y) in enumerate(instance):
            if confidence_array is not None:
                point_confidence = float(confidence_array[instance_index][keypoint_index])
            elif box_confidence_array is not None:
                point_confidence = float(box_confidence_array[instance_index])
            else:
                point_confidence = 1.0
            if point_confidence < keypoint_confidence_threshold:
                continue

            visible_confidences.append(point_confidence)
            color = _confidence_color(point_confidence)
            point = (int(round(x)), int(round(y)))
            cv2.circle(frame, point, 6, color, -1, cv2.LINE_AA)
            cv2.circle(frame, point, 8, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(
                frame,
                f"K{keypoint_index + 1} {point_confidence:.2f}",
                (point[0] + 9, point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )

    mean_confidence = (
        sum(visible_confidences) / len(visible_confidences)
        if visible_confidences
        else None
    )
    return len(visible_confidences), mean_confidence


def annotate_video(
    input_video: Path,
    output_video: Path,
    model_path: Path,
    detection_confidence: float = 0.5,
    keypoint_confidence_threshold: float = 0.25,
    image_size: int = 640,
    device: Optional[str] = None,
) -> int:
    """Run inference and write an annotated video. Returns the number of frames."""
    if not input_video.is_file():
        raise FileNotFoundError(f"Input video not found: {input_video}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not 0 <= detection_confidence <= 1:
        raise ValueError("--conf must be between 0 and 1")
    if not 0 <= keypoint_confidence_threshold <= 1:
        raise ValueError("--keypoint-conf must be between 0 and 1")

    # Import Ultralytics lazily so --help and simple module checks do not load the model.
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {input_video}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 30.0
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Could not read video dimensions: {input_video}")

    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"avc1"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output_video}")

    frame_count = 0
    try:
        while True:
            read_success, frame = capture.read()
            if not read_success:
                break

            predict_kwargs = {
                "conf": detection_confidence,
                "imgsz": image_size,
                "verbose": False,
            }
            if device is not None:
                predict_kwargs["device"] = device
            result = model.predict(frame, **predict_kwargs)[0]
            visible_count, mean_confidence = _draw_keypoints(
                frame, result, keypoint_confidence_threshold
            )

            summary = f"court keypoints: {visible_count}"
            if mean_confidence is not None:
                summary += f" | mean conf: {mean_confidence:.2f}"
            cv2.rectangle(frame, (8, 8), (330, 39), (0, 0, 0), -1)
            cv2.putText(
                frame,
                summary,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            writer.write(frame)
            frame_count += 1
    finally:
        capture.release()
        writer.release()

    return frame_count


def main() -> None:
    args = parse_args()
    frame_count = annotate_video(
        input_video=args.input_video,
        output_video=args.output_video,
        model_path=args.model,
        detection_confidence=args.conf,
        keypoint_confidence_threshold=args.keypoint_conf,
        image_size=args.imgsz,
        device=args.device,
    )
    print(f"Wrote {frame_count} frames to {args.output_video}")


if __name__ == "__main__":
    main()
