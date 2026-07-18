"""Local web app for triaging candidate frames and correcting court keypoints.

Serves two views over the shared manifest:
  - /triage/<clip_id>: thumbnail grid, mark frames keep/skip
  - /label/<clip_id>:  canvas editor to correct model-predicted keypoints on
                       kept frames; corrections saved as per-frame JSON under
                       dataset/labels/

The YOLO pose model is loaded lazily on the first prediction request. Do not
run extract_frames.py while this server is up — both write the manifest.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from threading import Lock

from flask import Flask, abort, jsonify, request, send_from_directory

from pipeline_manifest import (
    DEFAULT_DATASET_DIR,
    PROJECT_DIR,
    TRIAGE_STATES,
    counts,
    load_manifest,
    save_manifest,
)

DEFAULT_MODEL = PROJECT_DIR / "models" / "court_keypoint_detector.pt"
WEB_DIR = PROJECT_DIR / "web"

app = Flask(__name__, static_folder=str(WEB_DIR / "static"))

state: dict = {}
state_lock = Lock()
model_cache: dict = {}


def get_model():
    """Load the YOLO pose model once, on first use."""
    with state_lock:
        if "model" not in model_cache:
            from ultralytics import YOLO

            model_path = state["model_path"]
            if not model_path.is_file():
                raise FileNotFoundError(f"Model not found: {model_path}")
            model = YOLO(str(model_path))
            kpt_shape = getattr(model.model, "kpt_shape", None)
            model_cache["model"] = model
            model_cache["num_keypoints"] = int(kpt_shape[0]) if kpt_shape else None
        return model_cache["model"]


def predict_keypoints(image_path: Path) -> list[dict]:
    """Run the model on one image and return keypoints for the best instance."""
    model = get_model()
    predict_kwargs = {
        "conf": state["conf"],
        "imgsz": state["imgsz"],
        "verbose": False,
    }
    if state["device"] is not None:
        predict_kwargs["device"] = state["device"]
    result = model.predict(str(image_path), **predict_kwargs)[0]

    keypoints = getattr(result, "keypoints", None)
    xy_tensor = getattr(keypoints, "xy", None)
    if xy_tensor is None or len(xy_tensor) == 0:
        num = model_cache.get("num_keypoints") or 0
        return [{"x": 0.0, "y": 0.0, "v": 0, "src_conf": 0.0} for _ in range(num)]

    xy = xy_tensor.cpu().numpy()
    conf = getattr(keypoints, "conf", None)
    conf = conf.cpu().numpy() if conf is not None else None
    box_conf = getattr(getattr(result, "boxes", None), "conf", None)

    # Multiple detected instances: keep the one with the highest box confidence.
    instance = 0
    if box_conf is not None and len(box_conf) > 1:
        instance = int(box_conf.cpu().numpy().argmax())

    if model_cache.get("num_keypoints") is None:
        model_cache["num_keypoints"] = int(xy.shape[1])

    points = []
    for i, (x, y) in enumerate(xy[instance]):
        point_conf = float(conf[instance][i]) if conf is not None else 1.0
        # Points the model could not place land at (0,0) with ~0 confidence.
        visible = 2 if point_conf >= state["keypoint_conf"] else 0
        points.append(
            {
                "x": round(float(x), 2),
                "y": round(float(y), 2),
                "v": visible,
                "src_conf": round(point_conf, 4),
            }
        )
    return points


def label_path(frame_id: str) -> Path:
    frame = state["manifest"]["frames"].get(frame_id)
    if frame is None:
        abort(404, f"Unknown frame: {frame_id}")
    return state["dataset_dir"] / "labels" / frame["clip_id"] / f"{frame_id}.json"


def page(name: str):
    return send_from_directory(WEB_DIR, name)


@app.get("/")
def index():
    return page("index.html")


@app.get("/triage/<clip_id>")
def triage_page(clip_id: str):
    if clip_id not in state["manifest"]["clips"]:
        abort(404, f"Unknown clip: {clip_id}")
    return page("triage.html")


@app.get("/label")
@app.get("/label/<clip_id>")
def label_page(clip_id: str | None = None):
    if clip_id is not None and clip_id not in state["manifest"]["clips"]:
        abort(404, f"Unknown clip: {clip_id}")
    return page("label.html")


@app.get("/api/clips")
def api_clips():
    manifest = state["manifest"]
    clips = []
    for clip_id in sorted(manifest["clips"]):
        entry = dict(manifest["clips"][clip_id])
        entry["clip_id"] = clip_id
        entry["counts"] = counts(manifest, clip_id=clip_id)
        clips.append(entry)
    return jsonify({"clips": clips, "counts": counts(manifest)})


@app.get("/api/frames")
def api_frames():
    manifest = state["manifest"]
    clip_id = request.args.get("clip")
    triage = request.args.get("triage")
    frames = []
    for frame_id in sorted(manifest["frames"]):
        frame = manifest["frames"][frame_id]
        if clip_id and frame["clip_id"] != clip_id:
            continue
        if triage and frame.get("triage", "pending") != triage:
            continue
        frames.append({"frame_id": frame_id, **frame})
    return jsonify(
        {"frames": frames, "counts": counts(manifest, clip_id=clip_id)}
    )


@app.post("/api/triage")
def api_triage():
    body = request.get_json(force=True)
    frame_id = body.get("frame_id")
    status = body.get("status")
    if status not in TRIAGE_STATES:
        abort(400, f"status must be one of {TRIAGE_STATES}")
    with state_lock:
        frame = state["manifest"]["frames"].get(frame_id)
        if frame is None:
            abort(404, f"Unknown frame: {frame_id}")
        frame["triage"] = status
        save_manifest(state["dataset_dir"], state["manifest"])
        clip_counts = counts(state["manifest"], clip_id=frame["clip_id"])
        total_counts = counts(state["manifest"])
    return jsonify({"frame_id": frame_id, "status": status,
                    "clip_counts": clip_counts, "counts": total_counts})


@app.get("/api/label/<frame_id>")
def api_get_label(frame_id: str):
    frame = state["manifest"]["frames"].get(frame_id)
    if frame is None:
        abort(404, f"Unknown frame: {frame_id}")
    path = label_path(frame_id)
    force_predict = request.args.get("predict") == "1"
    if path.is_file() and not force_predict:
        with open(path, "r", encoding="utf-8") as handle:
            label = json.load(handle)
        return jsonify({"source": "saved", "label": label,
                        "num_keypoints": len(label.get("keypoints", []))})

    clip = state["manifest"]["clips"][frame["clip_id"]]
    points = predict_keypoints(state["dataset_dir"] / frame["image"])
    label = {
        "frame_id": frame_id,
        "image_w": clip["width"],
        "image_h": clip["height"],
        "num_keypoints": len(points),
        "keypoints": points,
    }
    return jsonify({"source": "predicted", "label": label,
                    "num_keypoints": len(points)})


@app.post("/api/label/<frame_id>")
def api_save_label(frame_id: str):
    body = request.get_json(force=True)
    keypoints = body.get("keypoints")
    if not isinstance(keypoints, list) or not keypoints:
        abort(400, "keypoints must be a non-empty list")
    with state_lock:
        frame = state["manifest"]["frames"].get(frame_id)
        if frame is None:
            abort(404, f"Unknown frame: {frame_id}")
        clip = state["manifest"]["clips"][frame["clip_id"]]
        label = {
            "frame_id": frame_id,
            "image_w": clip["width"],
            "image_h": clip["height"],
            "num_keypoints": len(keypoints),
            "keypoints": [
                {
                    "x": float(point["x"]),
                    "y": float(point["y"]),
                    "v": int(point["v"]),
                    "src_conf": float(point.get("src_conf", 0.0)),
                }
                for point in keypoints
            ],
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        path = label_path(frame_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(label, handle, indent=2)
            handle.write("\n")
        frame["label_status"] = "labeled"
        save_manifest(state["dataset_dir"], state["manifest"])
        total_counts = counts(state["manifest"])
    return jsonify({"frame_id": frame_id, "saved": True, "counts": total_counts})


@app.get("/images/<path:relpath>")
def serve_image(relpath: str):
    return send_from_directory(state["dataset_dir"] / "frames", relpath)


@app.get("/thumbs/<path:relpath>")
def serve_thumb(relpath: str):
    return send_from_directory(state["dataset_dir"] / "thumbs", relpath)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the triage and keypoint labeling web app."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Dataset directory (default: {DEFAULT_DATASET_DIR})",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Path to the Ultralytics pose model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--port", type=int, default=8000)
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
        help="Predicted keypoints below this confidence start hidden (default: 0.25)",
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


def main() -> None:
    args = parse_args()
    state.update(
        dataset_dir=args.dataset,
        model_path=args.model,
        conf=args.conf,
        keypoint_conf=args.keypoint_conf,
        imgsz=args.imgsz,
        device=args.device,
        manifest=load_manifest(args.dataset),
    )
    tally = counts(state["manifest"])
    print(f"Loaded manifest: {tally['clips']} clip(s), {tally['candidates']} frames")
    print(f"Open http://127.0.0.1:{args.port}")
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
