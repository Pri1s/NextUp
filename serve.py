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
KEYPOINT_SCHEMA_PATH = PROJECT_DIR / "dataset" / "schemas" / "court_keypoints.v2.json"
ORIENTATION_MIN_POINTS_PER_END = 2
ORIENTATION_MIN_X_SEPARATION_FRACTION = 0.03
NORTH_CONVENTION = "image_left_basket"


def load_keypoint_schema() -> dict:
    if not KEYPOINT_SCHEMA_PATH.is_file():
        raise FileNotFoundError(f"Keypoint schema not found: {KEYPOINT_SCHEMA_PATH}")
    with open(KEYPOINT_SCHEMA_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def schema_provenance() -> dict:
    schema = load_keypoint_schema()
    return {
        "schema_name": schema["schema_name"],
        "schema_version": schema["schema_version"],
    }


def clip_orientation(clip: dict) -> dict | None:
    """Return only orientation metadata compatible with the current anchor axis."""
    orientation = clip.get("orientation")
    if not isinstance(orientation, dict):
        return None
    # Image-top anchors belonged to the superseded schema and must be re-anchored.
    return orientation if orientation.get("north_convention") == NORTH_CONVENTION else None


def normalize_keypoints(points: list[dict], permutation_name: str) -> list[dict]:
    """Return raw prediction points in the fixed canonical index order."""
    schema = load_keypoint_schema()
    permutation = schema["normalization"].get(permutation_name)
    expected = len(schema["keypoints"])
    if not isinstance(permutation, list) or len(permutation) != expected:
        raise ValueError(f"Unknown or invalid normalization: {permutation_name}")
    if len(points) != expected:
        raise ValueError(
            f"Model returned {len(points)} keypoints; canonical schema requires {expected}"
        )
    return [dict(points[raw_index]) for raw_index in permutation]


def infer_orientation(points: list[dict], image_width: int) -> tuple[str | None, dict]:
    """Infer the raw model end relation from the anchor image's X positions."""
    schema = load_keypoint_schema()
    groups = schema["normalization"]["raw_end_groups"]

    def mean_x(indices: list[int]) -> tuple[float | None, int]:
        located = [
            points[index]["x"]
            for index in indices
            if index < len(points)
            and points[index].get("v", 0) > 0
            and not (points[index].get("x") == 0 and points[index].get("y") == 0)
        ]
        return (sum(located) / len(located), len(located)) if located else (None, 0)

    first_x, first_count = mean_x(groups["first"])
    second_x, second_count = mean_x(groups["second"])
    evidence = {
        "first_end_mean_x": round(first_x, 2) if first_x is not None else None,
        "first_end_points": first_count,
        "second_end_mean_x": round(second_x, 2) if second_x is not None else None,
        "second_end_points": second_count,
    }
    if first_count < ORIENTATION_MIN_POINTS_PER_END or second_count < ORIENTATION_MIN_POINTS_PER_END:
        evidence["reason"] = "both end groups (A and B) need at least two located points"
        return None, evidence
    if abs(first_x - second_x) < max(12.0, image_width * ORIENTATION_MIN_X_SEPARATION_FRACTION):
        evidence["reason"] = "Group A and Group B mean X positions are too close to anchor reliably"
        return None, evidence
    # Footage uses a left/right court axis: north is the image-left basket.
    if first_x < second_x:
        return "identity", evidence
    return "rotate_180", evidence


def orientation_for_response(orientation: dict | None) -> dict | None:
    if orientation is None:
        return None
    return {
        "anchor_frame_id": orientation["anchor_frame_id"],
        "north_convention": orientation["north_convention"],
        "raw_first_end_relation": orientation["raw_first_end_relation"],
        "prefill_normalization": orientation["prefill_normalization"],
        "method": orientation["method"],
    }

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


@app.get("/api/keypoint-schema")
def api_keypoint_schema():
    """Return the versioned feature semantics used by the labeling UI."""
    try:
        return jsonify(load_keypoint_schema())
    except FileNotFoundError as error:
        abort(500, str(error))


@app.get("/api/clip/<clip_id>/orientation")
def api_get_orientation(clip_id: str):
    clip = state["manifest"]["clips"].get(clip_id)
    if clip is None:
        abort(404, f"Unknown clip: {clip_id}")
    return jsonify({"clip_id": clip_id, "orientation": orientation_for_response(clip_orientation(clip))})


@app.post("/api/clip/<clip_id>/orientation")
def api_set_orientation(clip_id: str):
    """Lock the one-time canonical orientation before exposing clip prefill."""
    body = request.get_json(force=True)
    frame_id = body.get("frame_id")
    relation = body.get("raw_first_end_relation", "auto")
    if relation not in {"auto", "left", "right"}:
        abort(400, "raw_first_end_relation must be auto, left, or right")

    with state_lock:
        clip = state["manifest"]["clips"].get(clip_id)
        frame = state["manifest"]["frames"].get(frame_id)
        if clip is None:
            abort(404, f"Unknown clip: {clip_id}")
        if frame is None or frame.get("clip_id") != clip_id:
            abort(400, "anchor frame must belong to this clip")
        existing = clip_orientation(clip)
        if existing is not None:
            return jsonify({
                "clip_id": clip_id,
                "orientation": orientation_for_response(existing),
                "already_locked": True,
            })

    if relation == "auto":
        raw_points = predict_keypoints(state["dataset_dir"] / frame["image"])
        normalization, evidence = infer_orientation(raw_points, clip["width"])
        if normalization is None:
            return jsonify({
                "error": "Could not infer a reliable orientation from this anchor frame.",
                "evidence": evidence,
                "requires_manual_relation": True,
            }), 422
        raw_first_end_relation = "left" if normalization == "identity" else "right"
        method = "raw_end_group_mean_x"
    else:
        normalization = "identity" if relation == "left" else "rotate_180"
        raw_first_end_relation = relation
        evidence = {"reason": "operator-selected raw first-end relation"}
        method = "operator_selected"

    orientation = {
        "anchor_frame_id": frame_id,
        "north_convention": NORTH_CONVENTION,
        "raw_first_end_relation": raw_first_end_relation,
        "prefill_normalization": normalization,
        "method": method,
        "evidence": evidence,
        "locked_at": datetime.now().isoformat(timespec="seconds"),
    }
    with state_lock:
        # Another request may have locked it while model inference was running.
        clip = state["manifest"]["clips"][clip_id]
        existing = clip_orientation(clip)
        if existing is None:
            clip["orientation"] = orientation
            save_manifest(state["dataset_dir"], state["manifest"])
        else:
            orientation = existing
    return jsonify({
        "clip_id": clip_id,
        "orientation": orientation_for_response(orientation),
        "already_locked": existing is not None,
    })


@app.get("/api/frame/<frame_id>/raw-prediction")
def api_raw_prediction(frame_id: str):
    """Raw, un-normalized model points for the pre-orientation group preview.

    Available before the clip orientation is locked, unlike /api/label, so the
    labeler can see the two raw end groups while answering the orientation
    question. Never persisted; canonical labels always go through /api/label.
    """
    frame = state["manifest"]["frames"].get(frame_id)
    if frame is None:
        abort(404, f"Unknown frame: {frame_id}")
    schema = load_keypoint_schema()
    points = predict_keypoints(state["dataset_dir"] / frame["image"])
    return jsonify({
        "frame_id": frame_id,
        "points": points,
        "raw_end_groups": schema["normalization"]["raw_end_groups"],
        "schema": schema_provenance(),
    })


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


@app.post("/api/clip/<clip_id>/done")
def api_clip_done(clip_id: str):
    body = request.get_json(force=True)
    with state_lock:
        clip = state["manifest"]["clips"].get(clip_id)
        if clip is None:
            abort(404, f"Unknown clip: {clip_id}")
        clip["done"] = bool(body.get("done"))
        save_manifest(state["dataset_dir"], state["manifest"])
    return jsonify({"clip_id": clip_id, "done": clip["done"]})


@app.get("/api/label/<frame_id>")
def api_get_label(frame_id: str):
    frame = state["manifest"]["frames"].get(frame_id)
    if frame is None:
        abort(404, f"Unknown frame: {frame_id}")
    clip = state["manifest"]["clips"][frame["clip_id"]]
    orientation = clip_orientation(clip)
    if orientation is None:
        return jsonify({
            "error": "Set the clip orientation anchor before requesting prefill.",
            "orientation_required": True,
            "schema": schema_provenance(),
        }), 409

    path = label_path(frame_id)
    force_predict = request.args.get("predict") == "1"
    if path.is_file() and not force_predict:
        with open(path, "r", encoding="utf-8") as handle:
            label = json.load(handle)
        return jsonify({
            "source": "saved",
            "label": label,
            "num_keypoints": len(label.get("keypoints", [])),
            "orientation": orientation_for_response(orientation),
            "schema": schema_provenance(),
        })

    raw_points = predict_keypoints(state["dataset_dir"] / frame["image"])
    try:
        points = normalize_keypoints(raw_points, orientation["prefill_normalization"])
    except ValueError as error:
        abort(500, str(error))
    label = {
        "frame_id": frame_id,
        "image_w": clip["width"],
        "image_h": clip["height"],
        "num_keypoints": len(points),
        "keypoints": points,
    }
    return jsonify({
        "source": "predicted",
        "label": label,
        "num_keypoints": len(points),
        "orientation": orientation_for_response(orientation),
        "schema": schema_provenance(),
    })


@app.post("/api/label/<frame_id>")
def api_save_label(frame_id: str):
    body = request.get_json(force=True)
    keypoints = body.get("keypoints")
    expected = len(load_keypoint_schema()["keypoints"])
    if not isinstance(keypoints, list) or len(keypoints) != expected:
        abort(400, f"keypoints must contain exactly {expected} canonical points")
    with state_lock:
        frame = state["manifest"]["frames"].get(frame_id)
        if frame is None:
            abort(404, f"Unknown frame: {frame_id}")
        clip = state["manifest"]["clips"][frame["clip_id"]]
        orientation = clip_orientation(clip)
        if orientation is None:
            abort(409, "Set the clip orientation anchor before saving a label")
        label = {
            "frame_id": frame_id,
            "image_w": clip["width"],
            "image_h": clip["height"],
            "num_keypoints": expected,
            "schema": schema_provenance(),
            "orientation": orientation_for_response(orientation),
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
    return jsonify({
        "frame_id": frame_id,
        "saved": True,
        "counts": total_counts,
        "schema": label["schema"],
        "orientation": label["orientation"],
    })


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
