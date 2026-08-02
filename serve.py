"""Local web app for hand-labeling court keypoints and benchmark markings.

Serves two views over the shared manifest:
  - /triage/<clip_id>: thumbnail grid, mark frames keep/skip
  - /label/<clip_id>:  canvas editor to place canonical court keypoints on
                       kept frames; labels saved as per-frame JSON under
                       dataset/labels/
  - /benchmark-label:  blind centerline editor for the locked Milestone 1
                       court-marking benchmark

Labeling is manual by default. --model optionally points at a pose model
trained on THIS schema (a court_pose training run) to prefill points; models
with a different keypoint count are refused at startup. The legacy reloc2
court model is not schema-compatible and cannot be used here.

Do not run extract_frames.py while this server is up — both write the manifest.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from threading import Lock

from flask import Flask, abort, jsonify, request, send_from_directory

from benchmark.glossary import feature_catalog, junction_catalog, reference_court
from contracts.annotations import (
    ANNOTATION_SCHEMA_VERSION,
    LINE_CONVENTION,
    FrameAnnotation,
    validate as validate_annotation,
)
from contracts.court_layout import load_registered_layout
from pipeline_manifest import (
    DEFAULT_DATASET_DIR,
    PROJECT_DIR,
    TRIAGE_STATES,
    counts,
    load_manifest,
    save_manifest,
)

WEB_DIR = PROJECT_DIR / "web"
KEYPOINT_SCHEMA_PATH = PROJECT_DIR / "dataset" / "schemas" / "court_keypoints.v3.json"
DEFAULT_BENCHMARK_DIR = (
    PROJECT_DIR / "engine_out" / "benchmark" / "m1_openai_gpt56_v1"
)
HUMAN_ANNOTATOR_PROMPT_VERSION = "human-labeler-1.0.0"
HUMAN_ANNOTATOR_PASS_ID = "human_reference"
NORTH_CONVENTION = "image_left_basket"
SCHEMA_VERSION_PREFIX = "3."
ORIENTATION_MODES = ("both_ends_visible", "declared")
# Inference settings for optional prefill from a model trained on this schema.
PREDICT_CONF = 0.25
PREDICT_KEYPOINT_CONF = 0.25
PREDICT_IMGSZ = 960


def load_keypoint_schema(path: Path = KEYPOINT_SCHEMA_PATH) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Keypoint schema not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def schema_provenance(schema: dict) -> dict:
    return {
        "schema_name": schema["schema_name"],
        "schema_version": schema["schema_version"],
    }


def clip_orientation(clip: dict) -> dict | None:
    """Return only orientation metadata recorded under the current schema."""
    orientation = clip.get("orientation")
    if not isinstance(orientation, dict):
        return None
    if orientation.get("north_convention") != NORTH_CONVENTION:
        return None
    # v2 anchors share the convention string but carried raw-model metadata;
    # anything not recorded under a 3.x schema must be re-anchored.
    if not str(orientation.get("schema_version", "")).startswith(SCHEMA_VERSION_PREFIX):
        return None
    return orientation


def orientation_for_response(orientation: dict | None) -> dict | None:
    if orientation is None:
        return None
    response = {
        "anchor_frame_id": orientation["anchor_frame_id"],
        "north_convention": orientation["north_convention"],
        "method": orientation["method"],
        "schema_version": orientation["schema_version"],
    }
    if orientation.get("declared_end"):
        response["declared_end"] = orientation["declared_end"]
    return response


def load_trained_frames(model_path: Path) -> dict[str, str]:
    """frame_id -> 'train'/'val' for the run that produced model_path's weights.

    Looks for dataset_manifest.json (written by train_pose.py) next to the
    run directory, e.g. runs/pose/court_pose_v1/weights/best.pt ->
    runs/pose/court_pose_v1/dataset_manifest.json. Missing manifest (older
    run, or a checkpoint not produced by train_pose.py) just means no frames
    are flagged as trained-on.
    """
    manifest_path = Path(model_path).resolve().parent.parent / "dataset_manifest.json"
    if not manifest_path.is_file():
        return {}
    with open(manifest_path, "r", encoding="utf-8") as handle:
        return json.load(handle).get("frames", {})


def empty_points(count: int) -> list[dict]:
    return [{"x": 0.0, "y": 0.0, "v": 0, "src_conf": 0.0} for _ in range(count)]


def ends_conflicts(schema: dict, keypoints: list[dict], visible_ends: str) -> list[str]:
    """Ids of placed points that belong to an end declared not visible."""
    if visible_ends == "both":
        return []
    return [
        definition["id"]
        for definition, point in zip(schema["keypoints"], keypoints)
        if point["v"] > 0
        and definition["end"] in ("north", "south")
        and definition["end"] != visible_ends
    ]


def load_benchmark_export(benchmark_dir: Path) -> dict:
    """Load and minimally validate the blind frame export used by the UI."""
    path = benchmark_dir / "frames" / "frames.json"
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark frame export not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        export = json.load(handle)
    if export.get("frame_count") != len(export.get("frames", ())):
        raise ValueError("Benchmark frame_count does not match frames.json")
    if not export.get("frames"):
        raise ValueError("Benchmark export contains no frames")
    return export


def human_annotation_problems(annotation: FrameAnnotation) -> list[str]:
    """Precision rules specific to the human UI, beyond the base contract."""
    problems = validate_annotation(annotation, require_complete=True)
    for item in annotation.features:
        minimum = 6 if item.kind == "arc" else 4
        if len(item.points) < minimum:
            problems.append(
                f"{item.feature.value}: human reference tracing requires at least "
                f"{minimum} observed samples"
            )
    return problems


def build_app(
    dataset_dir: Path,
    model_path: Path | None = None,
    schema_path: Path = KEYPOINT_SCHEMA_PATH,
    benchmark_dir: Path | None = None,
) -> Flask:
    app = Flask(__name__, static_folder=str(WEB_DIR / "static"))
    schema_expected = len(load_keypoint_schema(schema_path)["keypoints"])

    state = {
        "dataset_dir": Path(dataset_dir),
        "manifest": load_manifest(Path(dataset_dir)),
        "model": None,
        "trained_frames": {},
        "benchmark_dir": Path(benchmark_dir) if benchmark_dir is not None else None,
        "benchmark_export": None,
    }
    state_lock = Lock()

    if state["benchmark_dir"] is not None:
        state["benchmark_export"] = load_benchmark_export(state["benchmark_dir"])

    if model_path is not None:
        from ultralytics import YOLO

        if not Path(model_path).is_file():
            raise SystemExit(f"Prefill model not found: {model_path}")
        model = YOLO(str(model_path))
        kpt_shape = getattr(model.model, "kpt_shape", None)
        if not kpt_shape or int(kpt_shape[0]) != schema_expected:
            raise SystemExit(
                f"Prefill model has kpt_shape {kpt_shape}; the schema requires "
                f"{schema_expected} keypoints. Only models trained on this schema can prefill."
            )
        state["model"] = model
        state["trained_frames"] = load_trained_frames(model_path)

    def get_schema() -> dict:
        return load_keypoint_schema(schema_path)

    def predict_keypoints(image_path: Path) -> list[dict]:
        """Run the prefill model on one image; keypoints arrive canonical."""
        result = state["model"].predict(
            str(image_path), conf=PREDICT_CONF, imgsz=PREDICT_IMGSZ, verbose=False
        )[0]
        keypoints = getattr(result, "keypoints", None)
        xy_tensor = getattr(keypoints, "xy", None)
        if xy_tensor is None or len(xy_tensor) == 0:
            return empty_points(schema_expected)

        xy = xy_tensor.cpu().numpy()
        conf = getattr(keypoints, "conf", None)
        conf = conf.cpu().numpy() if conf is not None else None
        box_conf = getattr(getattr(result, "boxes", None), "conf", None)

        # Multiple detected instances: keep the one with the highest box confidence.
        instance = 0
        if box_conf is not None and len(box_conf) > 1:
            instance = int(box_conf.cpu().numpy().argmax())

        points = []
        for i, (x, y) in enumerate(xy[instance]):
            point_conf = float(conf[instance][i]) if conf is not None else 1.0
            visible = 2 if point_conf >= PREDICT_KEYPOINT_CONF else 0
            points.append(
                {
                    "x": round(float(x), 2) if visible else 0.0,
                    "y": round(float(y), 2) if visible else 0.0,
                    "v": visible,
                    "src_conf": round(point_conf, 4) if visible else 0.0,
                }
            )
        return points

    def label_path(frame_id: str) -> Path:
        frame = state["manifest"]["frames"].get(frame_id)
        if frame is None:
            abort(404, f"Unknown frame: {frame_id}")
        return state["dataset_dir"] / "labels" / frame["clip_id"] / f"{frame_id}.json"

    def previous_labeled_frame(frame_id: str, frame: dict) -> str | None:
        """Closest earlier frame in the same clip that already has a saved label.

        Adjacent kept frames are usually close enough in time that copying
        the prior frame's points is a faster starting point to correct than
        placing every point from scratch or trusting a still-undertrained
        model.
        """
        clip_id = frame["clip_id"]
        frame_index = frame["frame_index"]
        candidates = [
            (other["frame_index"], other_id)
            for other_id, other in state["manifest"]["frames"].items()
            if other["clip_id"] == clip_id
            and other_id != frame_id
            and other.get("label_status") == "labeled"
            and other["frame_index"] < frame_index
        ]
        if not candidates:
            return None
        return max(candidates)[1]

    def page(name: str):
        return send_from_directory(WEB_DIR, name)

    def benchmark_frame(frame_id: str) -> dict:
        export = state["benchmark_export"]
        if export is None:
            abort(404, "No benchmark frame export configured")
        for frame in export["frames"]:
            if frame["frame_id"] == frame_id:
                return frame
        abort(404, f"Unknown benchmark frame: {frame_id}")

    def human_label_dir() -> Path:
        if state["benchmark_dir"] is None:
            abort(404, "No benchmark frame export configured")
        return state["benchmark_dir"] / "run" / "human"

    def human_label_path(frame_id: str, *, draft: bool = False) -> Path:
        benchmark_frame(frame_id)
        if draft:
            return human_label_dir() / "drafts" / f"{frame_id}.json"
        return human_label_dir() / f"{frame_id}.frame.json"

    def empty_human_annotation(frame: dict) -> dict:
        return {
            "schema_version": ANNOTATION_SCHEMA_VERSION,
            "line_convention": LINE_CONVENTION,
            "frame_id": frame["frame_id"],
            "clip": frame["clip"],
            "frame_index": frame["frame_index"],
            "image_width": frame["width"],
            "image_height": frame["height"],
            "image_sha256": frame["sha256"],
            "annotator_id": "human-pri",
            "pass_id": HUMAN_ANNOTATOR_PASS_ID,
            "prompt_version": HUMAN_ANNOTATOR_PROMPT_VERSION,
            "coordinate_space": "frame",
            "features": [],
            "junctions": [],
            "skipped": [],
            "notes": "",
        }

    def validate_human_metadata(annotation: FrameAnnotation, frame: dict) -> list[str]:
        expected = {
            "frame_id": frame["frame_id"],
            "clip": frame["clip"],
            "frame_index": frame["frame_index"],
            "image_width": frame["width"],
            "image_height": frame["height"],
            "image_sha256": frame["sha256"],
            "pass_id": HUMAN_ANNOTATOR_PASS_ID,
            "prompt_version": HUMAN_ANNOTATOR_PROMPT_VERSION,
            "coordinate_space": "frame",
        }
        problems = []
        for field, value in expected.items():
            if getattr(annotation, field) != value:
                problems.append(
                    f"{field} does not match the locked benchmark "
                    f"({getattr(annotation, field)!r} != {value!r})"
                )
        if not annotation.annotator_id.strip():
            problems.append("annotator_id must not be empty")
        return problems

    def benchmark_api_payload() -> dict:
        export = state["benchmark_export"]
        if export is None:
            abort(404, "No benchmark frame export configured")
        layout = load_registered_layout(export["layout_id"])
        labels = human_label_dir()
        frames = []
        for frame in export["frames"]:
            frames.append(
                {
                    **frame,
                    "image_url": f"/benchmark-images/{frame['frame_id']}",
                    "complete": (labels / f"{frame['frame_id']}.frame.json").is_file(),
                    "draft": (
                        labels / "drafts" / f"{frame['frame_id']}.json"
                    ).is_file(),
                }
            )
        return {
            "manifest_id": export["manifest_id"],
            "layout_id": export["layout_id"],
            "glossary_version": export["glossary_version"],
            "frames": frames,
            "features": feature_catalog(layout),
            "junctions": junction_catalog(),
            "reference_court": reference_court(layout),
            "skip_reasons": [
                "not_visible",
                "out_of_frame",
                "fully_occluded",
                "too_faint",
                "ambiguous_identity",
            ],
            "visibilities": ["clear", "faint", "partially_occluded"],
            "prompt_version": HUMAN_ANNOTATOR_PROMPT_VERSION,
            "pass_id": HUMAN_ANNOTATOR_PASS_ID,
        }

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

    @app.get("/benchmark-label")
    def benchmark_label_page():
        if state["benchmark_export"] is None:
            abort(404, "No benchmark frame export configured")
        return page("benchmark-label.html")

    @app.get("/api/benchmark")
    def api_benchmark():
        return jsonify(benchmark_api_payload())

    @app.get("/api/benchmark/annotation/<frame_id>")
    def api_get_benchmark_annotation(frame_id: str):
        frame = benchmark_frame(frame_id)
        final_path = human_label_path(frame_id)
        draft_path = human_label_path(frame_id, draft=True)
        draft_is_newer = (
            draft_path.is_file()
            and (
                not final_path.is_file()
                or draft_path.stat().st_mtime_ns > final_path.stat().st_mtime_ns
            )
        )
        if draft_is_newer:
            path, source = draft_path, "draft"
        elif final_path.is_file():
            path, source = final_path, "final"
        else:
            return jsonify(
                {
                    "source": "empty",
                    "finalized": False,
                    "annotation": empty_human_annotation(frame),
                }
            )
        with open(path, "r", encoding="utf-8") as handle:
            annotation = json.load(handle)
        return jsonify(
            {
                "source": source,
                "finalized": final_path.is_file() and source == "final",
                "annotation": annotation,
            }
        )

    @app.post("/api/benchmark/annotation/<frame_id>")
    def api_save_benchmark_annotation(frame_id: str):
        frame = benchmark_frame(frame_id)
        body = request.get_json(force=True)
        finalize = bool(body.get("finalize"))
        try:
            annotation = FrameAnnotation.from_dict(body.get("annotation", {}))
        except (KeyError, TypeError, ValueError) as error:
            return jsonify({"error": f"Invalid annotation: {error}", "problems": [str(error)]}), 400

        metadata_problems = validate_human_metadata(annotation, frame)
        if metadata_problems:
            return jsonify(
                {
                    "error": "Annotation metadata does not match the locked frame.",
                    "problems": metadata_problems,
                }
            ), 400

        if finalize:
            problems = human_annotation_problems(annotation)
        else:
            # Drafts remain writable while incomplete, but returning the same
            # readiness problems lets the UI explain what still blocks finalizing.
            problems = human_annotation_problems(annotation)

        if problems and finalize:
            return jsonify(
                {
                    "error": "Frame is not ready to finalize.",
                    "problems": problems,
                }
            ), 422

        payload = annotation.as_dict()
        draft_path = human_label_path(frame_id, draft=True)
        draft_path.parent.mkdir(parents=True, exist_ok=True)
        with open(draft_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")

        if finalize:
            final_path = human_label_path(frame_id)
            final_path.parent.mkdir(parents=True, exist_ok=True)
            with open(final_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                handle.write("\n")

        return jsonify(
            {
                "frame_id": frame_id,
                "saved": True,
                "finalized": finalize,
                "problems": problems,
            }
        )

    @app.get("/benchmark-images/<frame_id>")
    def serve_benchmark_image(frame_id: str):
        frame = benchmark_frame(frame_id)
        frames_dir = state["benchmark_dir"] / "frames"
        return send_from_directory(frames_dir, frame["path"])

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
            return jsonify(get_schema())
        except FileNotFoundError as error:
            abort(500, str(error))

    @app.get("/api/clip/<clip_id>/orientation")
    def api_get_orientation(clip_id: str):
        clip = state["manifest"]["clips"].get(clip_id)
        if clip is None:
            abort(404, f"Unknown clip: {clip_id}")
        return jsonify(
            {"clip_id": clip_id, "orientation": orientation_for_response(clip_orientation(clip))}
        )

    @app.post("/api/clip/<clip_id>/orientation")
    def api_set_orientation(clip_id: str):
        """Lock the one-time north-end anchor before exposing labeling."""
        body = request.get_json(force=True)
        frame_id = body.get("frame_id")
        mode = body.get("mode")
        if mode not in ORIENTATION_MODES:
            abort(400, f"mode must be one of {ORIENTATION_MODES}")
        declared_end = body.get("declared_end")
        if mode == "declared":
            if declared_end not in ("north", "south"):
                abort(400, "declared_end must be north or south")
        else:
            declared_end = None

        schema = get_schema()
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
            orientation = {
                "schema_version": schema["schema_version"],
                "north_convention": NORTH_CONVENTION,
                "anchor_frame_id": frame_id,
                "method": (
                    "anchor_both_ends_image_left"
                    if mode == "both_ends_visible"
                    else "operator_declared_single_end"
                ),
                "locked_at": datetime.now().isoformat(timespec="seconds"),
            }
            if declared_end:
                orientation["declared_end"] = declared_end
            clip["orientation"] = orientation
            save_manifest(state["dataset_dir"], state["manifest"])
        return jsonify({
            "clip_id": clip_id,
            "orientation": orientation_for_response(orientation),
            "already_locked": False,
        })

    @app.delete("/api/clip/<clip_id>/orientation")
    def api_delete_orientation(clip_id: str):
        """Unlock a clip's anchor; refused once labels exist unless ?force=1."""
        with state_lock:
            clip = state["manifest"]["clips"].get(clip_id)
            if clip is None:
                abort(404, f"Unknown clip: {clip_id}")
            labeled = [
                frame_id
                for frame_id, frame in state["manifest"]["frames"].items()
                if frame.get("clip_id") == clip_id and frame.get("label_status") == "labeled"
            ]
            if labeled and request.args.get("force") != "1":
                abort(
                    409,
                    f"{len(labeled)} labeled frame(s) depend on this orientation; "
                    "pass ?force=1 to unlock anyway",
                )
            clip.pop("orientation", None)
            save_manifest(state["dataset_dir"], state["manifest"])
        return jsonify({"clip_id": clip_id, "orientation": None})

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
            frames.append({
                "frame_id": frame_id,
                **frame,
                "trained_split": state["trained_frames"].get(frame_id),
            })
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
        schema = get_schema()
        predict_available = state["model"] is not None
        clip = state["manifest"]["clips"][frame["clip_id"]]
        orientation = clip_orientation(clip)
        if orientation is None:
            return jsonify({
                "error": "Set the clip orientation anchor before labeling.",
                "orientation_required": True,
                "schema": schema_provenance(schema),
                "predict_available": predict_available,
            }), 409

        path = label_path(frame_id)
        force_predict = request.args.get("predict") == "1" and predict_available
        prev_frame_id = None if force_predict else previous_labeled_frame(frame_id, frame)
        if path.is_file() and not force_predict:
            with open(path, "r", encoding="utf-8") as handle:
                label = json.load(handle)
            source = "saved"
        elif prev_frame_id is not None:
            with open(label_path(prev_frame_id), "r", encoding="utf-8") as handle:
                prev_label = json.load(handle)
            points = [
                {"x": point["x"], "y": point["y"], "v": point["v"], "src_conf": 0.0}
                for point in prev_label["keypoints"]
            ]
            label = {
                "frame_id": frame_id,
                "image_w": clip["width"],
                "image_h": clip["height"],
                "num_keypoints": len(points),
                "keypoints": points,
            }
            source = "previous_frame"
        else:
            points = (
                predict_keypoints(state["dataset_dir"] / frame["image"])
                if predict_available
                else empty_points(len(schema["keypoints"]))
            )
            label = {
                "frame_id": frame_id,
                "image_w": clip["width"],
                "image_h": clip["height"],
                "num_keypoints": len(points),
                "keypoints": points,
            }
            source = "predicted" if predict_available else "empty"
        return jsonify({
            "source": source,
            "label": label,
            "num_keypoints": len(label.get("keypoints", [])),
            "orientation": orientation_for_response(orientation),
            "schema": schema_provenance(schema),
            "predict_available": predict_available,
        })

    @app.post("/api/label/<frame_id>")
    def api_save_label(frame_id: str):
        body = request.get_json(force=True)
        schema = get_schema()
        expected = len(schema["keypoints"])

        visible_ends = body.get("visible_ends")
        if visible_ends not in schema["visible_ends_values"]:
            abort(400, f"visible_ends must be one of {schema['visible_ends_values']}")
        keypoints = body.get("keypoints")
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

            width, height = clip["width"], clip["height"]
            cleaned = []
            for position, point in enumerate(keypoints, start=1):
                try:
                    x = float(point["x"])
                    y = float(point["y"])
                    v = int(point["v"])
                    src_conf = float(point.get("src_conf", 0.0))
                except (KeyError, TypeError, ValueError):
                    abort(400, f"keypoint {position} must provide numeric x, y, v")
                if v not in (0, 1, 2):
                    abort(400, f"keypoint {position}: v must be 0, 1, or 2")
                if v == 0:
                    x, y, src_conf = 0.0, 0.0, 0.0
                elif not (0.0 <= x <= width and 0.0 <= y <= height):
                    abort(400, f"keypoint {position}: coordinates outside the image")
                cleaned.append({"x": x, "y": y, "v": v, "src_conf": src_conf})

            conflicts = ends_conflicts(schema, cleaned, visible_ends)
            if conflicts:
                return jsonify({
                    "error": (
                        f"visible_ends is '{visible_ends}' but points from the other "
                        "end are placed; fix the declaration or remove the points"
                    ),
                    "conflicting_ids": conflicts,
                }), 422

            label = {
                "frame_id": frame_id,
                "image_w": width,
                "image_h": height,
                "num_keypoints": expected,
                "schema": schema_provenance(schema),
                "orientation": orientation_for_response(orientation),
                "visible_ends": visible_ends,
                "keypoints": cleaned,
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
            "visible_ends": visible_ends,
        })

    @app.get("/images/<path:relpath>")
    def serve_image(relpath: str):
        return send_from_directory(state["dataset_dir"] / "frames", relpath)

    @app.get("/thumbs/<path:relpath>")
    def serve_thumb(relpath: str):
        return send_from_directory(state["dataset_dir"] / "thumbs", relpath)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the keypoint and court-marking labeling web app."
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
        default=None,
        help="Optional pose model trained on this schema, used to prefill points",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=DEFAULT_BENCHMARK_DIR,
        help=(
            "Blind benchmark namespace for /benchmark-label "
            f"(default: {DEFAULT_BENCHMARK_DIR})"
        ),
    )
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = build_app(
        args.dataset,
        model_path=args.model,
        benchmark_dir=args.benchmark,
    )
    tally = counts(load_manifest(args.dataset))
    print(f"Loaded manifest: {tally['clips']} clip(s), {tally['candidates']} frames")
    print(f"Open http://127.0.0.1:{args.port}")
    print(f"Benchmark labeler: http://127.0.0.1:{args.port}/benchmark-label")
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
