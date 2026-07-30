import json
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from pipeline_manifest import save_manifest
from contracts.annotations import MarkingFeature
from serve import (
    HUMAN_ANNOTATOR_PASS_ID,
    HUMAN_ANNOTATOR_PROMPT_VERSION,
    KEYPOINT_SCHEMA_PATH,
    build_app,
    load_keypoint_schema,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]

TOY_SCHEMA = {
    "schema_name": "basketball-court-keypoints",
    "schema_version": "3.9.9-toy",
    "visible_ends_values": ["north", "south", "both"],
    "keypoints": [
        {"index": 1, "id": "north_a", "end": "north", "side": "east"},
        {"index": 2, "id": "north_b", "end": "north", "side": "west"},
        {"index": 3, "id": "mid_a", "end": "mid", "side": "east"},
        {"index": 4, "id": "south_a", "end": "south", "side": "east"},
        {"index": 5, "id": "south_b", "end": "south", "side": "west"},
    ],
}


def make_manifest(dataset_dir: Path) -> dict:
    manifest = {
        "version": 1,
        "clips": {"clipA": {"width": 1920, "height": 1080}},
        "frames": {
            "clipA_f000001": {
                "clip_id": "clipA",
                "frame_index": 1,
                "image": "frames/clipA/clipA_f000001.jpg",
                "thumb": "thumbs/clipA/clipA_f000001.jpg",
                "timestamp_s": 0.0,
                "triage": "keep",
                "label_status": "unlabeled",
            },
            "clipA_f000002": {
                "clip_id": "clipA",
                "frame_index": 2,
                "image": "frames/clipA/clipA_f000002.jpg",
                "thumb": "thumbs/clipA/clipA_f000002.jpg",
                "timestamp_s": 2.0,
                "triage": "keep",
                "label_status": "unlabeled",
            },
        },
    }
    save_manifest(dataset_dir, manifest)
    return manifest


def valid_points(count: int) -> list[dict]:
    return [
        {"x": 100.0 + i, "y": 200.0 + i, "v": 2, "src_conf": 0.0}
        for i in range(count)
    ]


class ServeLabelingTests(unittest.TestCase):
    def setUp(self):
        self.dataset_dir = Path(tempfile.mkdtemp()) / "dataset"
        self.dataset_dir.mkdir()
        self.addCleanup(shutil.rmtree, self.dataset_dir.parent, ignore_errors=True)
        make_manifest(self.dataset_dir)
        self.schema = load_keypoint_schema()
        self.n = len(self.schema["keypoints"])
        app = build_app(self.dataset_dir)
        app.testing = True
        self.client = app.test_client()

    def lock_orientation(self, mode="both_ends_visible", **extra):
        return self.client.post(
            "/api/clip/clipA/orientation",
            json={"frame_id": "clipA_f000001", "mode": mode, **extra},
        )

    def save(self, keypoints=None, visible_ends="both", frame_id="clipA_f000001"):
        body = {"keypoints": keypoints or valid_points(self.n)}
        if visible_ends is not None:
            body["visible_ends"] = visible_ends
        return self.client.post(f"/api/label/{frame_id}", json=body)

    def test_label_blocked_before_orientation(self):
        response = self.client.get("/api/label/clipA_f000001")
        self.assertEqual(response.status_code, 409)
        self.assertTrue(response.get_json()["orientation_required"])
        self.assertEqual(self.save().status_code, 409)

    def test_orientation_lock_and_relock(self):
        response = self.lock_orientation()
        self.assertEqual(response.status_code, 200)
        orientation = response.get_json()["orientation"]
        self.assertEqual(orientation["method"], "anchor_both_ends_image_left")
        self.assertEqual(orientation["north_convention"], "image_left_basket")
        self.assertTrue(orientation["schema_version"].startswith("3."))

        again = self.lock_orientation(mode="declared", declared_end="south")
        self.assertTrue(again.get_json()["already_locked"])
        self.assertEqual(
            again.get_json()["orientation"]["method"], "anchor_both_ends_image_left"
        )

    def test_declared_mode_requires_end(self):
        self.assertEqual(self.lock_orientation(mode="declared").status_code, 400)
        response = self.lock_orientation(mode="declared", declared_end="south")
        self.assertEqual(response.get_json()["orientation"]["declared_end"], "south")

    def test_stale_v2_orientation_is_ignored(self):
        # A v2 block shares the convention string but lacks schema_version.
        manifest = make_manifest(self.dataset_dir)
        manifest["clips"]["clipA"]["orientation"] = {
            "anchor_frame_id": "clipA_f000001",
            "north_convention": "image_left_basket",
            "raw_first_end_relation": "left",
            "prefill_normalization": "identity",
            "method": "operator_selected",
        }
        save_manifest(self.dataset_dir, manifest)
        app = build_app(self.dataset_dir)
        app.testing = True
        client = app.test_client()
        self.assertIsNone(client.get("/api/clip/clipA/orientation").get_json()["orientation"])
        self.assertEqual(client.get("/api/label/clipA_f000001").status_code, 409)

    def test_empty_prefill_without_model(self):
        self.lock_orientation()
        data = self.client.get("/api/label/clipA_f000001").get_json()
        self.assertEqual(data["source"], "empty")
        self.assertFalse(data["predict_available"])
        self.assertEqual(len(data["label"]["keypoints"]), self.n)
        self.assertTrue(
            all(p == {"x": 0.0, "y": 0.0, "v": 0, "src_conf": 0.0} for p in data["label"]["keypoints"])
        )

    def test_prefill_copies_previous_labeled_frame(self):
        self.lock_orientation()
        points = valid_points(self.n)
        points[3] = {"x": 500.0, "y": 600.0, "v": 0, "src_conf": 0.9}
        self.assertEqual(self.save(points).status_code, 200)

        data = self.client.get("/api/label/clipA_f000002").get_json()
        self.assertEqual(data["source"], "previous_frame")
        got = data["label"]["keypoints"]
        # v=0 points are zeroed out on save (test_save_zeroes_unlabeled_points),
        # so the copy should reflect that, not the pre-save x/y we sent.
        expected = [(p["x"], p["y"], p["v"]) for p in points]
        expected[3] = (0.0, 0.0, 0)
        self.assertEqual([(p["x"], p["y"], p["v"]) for p in got], expected)
        # src_conf doesn't carry over — it described the previous frame's
        # provenance, not this one's.
        self.assertTrue(all(p["src_conf"] == 0.0 for p in got))

    def test_prefill_falls_back_to_empty_with_no_previous_frame(self):
        self.lock_orientation()
        data = self.client.get("/api/label/clipA_f000001").get_json()
        self.assertEqual(data["source"], "empty")

    def test_predict_param_is_noop_without_a_model(self):
        self.lock_orientation()
        self.assertEqual(self.save().status_code, 200)
        # No model loaded in this fixture, so ?predict=1 can't force a model
        # prediction (the UI hides the repredict control in this case too) —
        # normal prefill priority still applies underneath it.
        data = self.client.get("/api/label/clipA_f000002?predict=1").get_json()
        self.assertEqual(data["source"], "previous_frame")

    def test_save_happy_path(self):
        self.lock_orientation()
        response = self.save()
        self.assertEqual(response.status_code, 200)
        label_file = self.dataset_dir / "labels" / "clipA" / "clipA_f000001.json"
        label = json.loads(label_file.read_text())
        self.assertEqual(label["visible_ends"], "both")
        self.assertEqual(label["schema"]["schema_version"], self.schema["schema_version"])
        self.assertEqual(len(label["keypoints"]), self.n)
        self.assertIn("declared_end", {**label["orientation"], "declared_end": None})
        loaded = self.client.get("/api/label/clipA_f000001").get_json()
        self.assertEqual(loaded["source"], "saved")

    def test_save_requires_visible_ends(self):
        self.lock_orientation()
        self.assertEqual(self.save(visible_ends=None).status_code, 400)
        self.assertEqual(self.save(visible_ends="east").status_code, 400)

    def test_save_zeroes_unlabeled_points(self):
        self.lock_orientation()
        points = valid_points(self.n)
        points[3] = {"x": 500.0, "y": 600.0, "v": 0, "src_conf": 0.7}
        self.assertEqual(self.save(points).status_code, 200)
        label = json.loads(
            (self.dataset_dir / "labels" / "clipA" / "clipA_f000001.json").read_text()
        )
        self.assertEqual(label["keypoints"][3], {"x": 0.0, "y": 0.0, "v": 0, "src_conf": 0.0})

    def test_save_rejects_out_of_bounds_and_bad_v(self):
        self.lock_orientation()
        points = valid_points(self.n)
        points[0]["x"] = 99999.0
        self.assertEqual(self.save(points).status_code, 400)
        points = valid_points(self.n)
        points[0]["v"] = 3
        self.assertEqual(self.save(points).status_code, 400)

    def test_end_conflict_rejected_with_ids(self):
        self.lock_orientation()
        south_index = next(
            i for i, kp in enumerate(self.schema["keypoints"]) if kp["end"] == "south"
        )
        points = [
            {"x": 0.0, "y": 0.0, "v": 0, "src_conf": 0.0} for _ in range(self.n)
        ]
        points[south_index] = {"x": 100.0, "y": 100.0, "v": 2, "src_conf": 0.0}
        response = self.save(points, visible_ends="north")
        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            response.get_json()["conflicting_ids"],
            [self.schema["keypoints"][south_index]["id"]],
        )

    def test_mid_points_exempt_from_end_conflict(self):
        self.lock_orientation()
        mid_index = next(
            i for i, kp in enumerate(self.schema["keypoints"]) if kp["end"] == "mid"
        )
        points = [
            {"x": 0.0, "y": 0.0, "v": 0, "src_conf": 0.0} for _ in range(self.n)
        ]
        points[mid_index] = {"x": 100.0, "y": 100.0, "v": 2, "src_conf": 0.0}
        self.assertEqual(self.save(points, visible_ends="north").status_code, 200)

    def test_delete_orientation_guarded_by_labels(self):
        self.lock_orientation()
        self.save()
        self.assertEqual(self.client.delete("/api/clip/clipA/orientation").status_code, 409)
        response = self.client.delete("/api/clip/clipA/orientation?force=1")
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(
            self.client.get("/api/clip/clipA/orientation").get_json()["orientation"]
        )


class ToySchemaTests(unittest.TestCase):
    """The server must take its keypoint count from the schema file alone."""

    def setUp(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        self.dataset_dir = root / "dataset"
        self.dataset_dir.mkdir()
        make_manifest(self.dataset_dir)
        self.schema_path = root / "toy_schema.json"
        self.schema_path.write_text(json.dumps(TOY_SCHEMA))
        app = build_app(self.dataset_dir, schema_path=self.schema_path)
        app.testing = True
        self.client = app.test_client()

    def test_toy_point_count_enforced(self):
        self.client.post(
            "/api/clip/clipA/orientation",
            json={"frame_id": "clipA_f000001", "mode": "both_ends_visible"},
        )
        full = self.client.post(
            "/api/label/clipA_f000001",
            json={"keypoints": valid_points(5), "visible_ends": "both"},
        )
        self.assertEqual(full.status_code, 200)
        wrong = self.client.post(
            "/api/label/clipA_f000002",
            json={"keypoints": valid_points(22), "visible_ends": "both"},
        )
        self.assertEqual(wrong.status_code, 400)


class BenchmarkLabelingTests(unittest.TestCase):
    """The six-frame human UI saves strict, model-blind benchmark annotations."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.dataset_dir = self.root / "dataset"
        self.dataset_dir.mkdir()
        make_manifest(self.dataset_dir)

        self.benchmark_dir = self.root / "benchmark"
        images = self.benchmark_dir / "frames" / "images"
        images.mkdir(parents=True)
        (images / "video_1_000012.png").write_bytes(b"\x89PNG\r\n\x1a\nfixture")
        self.frame = {
            "frame_id": "video_1_000012",
            "clip": "input_videos/video_1.mp4",
            "frame_index": 12,
            "stratum": "clean",
            "pilot": True,
            "path": "images/video_1_000012.png",
            "sha256": "a" * 64,
            "width": 1280,
            "height": 720,
        }
        (self.benchmark_dir / "frames" / "frames.json").write_text(
            json.dumps(
                {
                    "schema_version": "benchmark-export-1.0.0",
                    "manifest_id": "nba_m1_v1",
                    "layout_id": "nba_halfcourt",
                    "glossary_version": "glossary-1.0.0",
                    "frame_count": 1,
                    "frames": [self.frame],
                }
            ),
            encoding="utf-8",
        )
        app = build_app(self.dataset_dir, benchmark_dir=self.benchmark_dir)
        app.testing = True
        self.client = app.test_client()

    def empty_annotation(self):
        return self.client.get(
            "/api/benchmark/annotation/video_1_000012"
        ).get_json()["annotation"]

    @staticmethod
    def skip_all(annotation, except_features=()):
        except_features = set(except_features)
        annotation["skipped"] = [
            {"feature": feature.value, "reason": "not_visible", "notes": ""}
            for feature in MarkingFeature
            if feature not in except_features
        ]
        return annotation

    def save(self, annotation, finalize=False):
        return self.client.post(
            "/api/benchmark/annotation/video_1_000012",
            json={"annotation": annotation, "finalize": finalize},
        )

    def test_ui_and_catalog_are_available(self):
        page = self.client.get("/benchmark-label")
        try:
            self.assertEqual(page.status_code, 200)
            self.assertIn(b"Court marking labeler", page.data)
            self.assertIn(b"marking-reference-canvas", page.data)
            self.assertIn(b"guide-basket-left", page.data)
            self.assertIn(b"guide-basket-right", page.data)
            self.assertIn(b"reference-half-rule", page.data)
            self.assertIn(b"Trace only visible paint", page.data)
            self.assertIn(b"order only", page.data)
            self.assertIn(b"continue-incomplete", page.data)
        finally:
            page.close()

        payload = self.client.get("/api/benchmark").get_json()
        self.assertEqual(payload["manifest_id"], "nba_m1_v1")
        self.assertEqual(len(payload["frames"]), 1)
        self.assertEqual(len(payload["features"]), len(MarkingFeature))
        self.assertTrue(all(item["description"] for item in payload["features"]))
        self.assertEqual(payload["pass_id"], HUMAN_ANNOTATOR_PASS_ID)
        reference = payload["reference_court"]
        self.assertEqual(reference["half_length"], 47.0)
        self.assertEqual(reference["width"], 50.0)
        self.assertEqual(set(reference["markings"]), {item.value for item in MarkingFeature})
        self.assertEqual(
            set(reference["label_anchors"]), {item.value for item in MarkingFeature}
        )
        self.assertEqual(reference["markings"]["baseline"], [[0.0, 0.0], [0.0, 50.0]])
        self.assertGreater(len(reference["markings"]["three_point_arc"]), 20)
        anchors = reference["label_anchors"]
        self.assertEqual(anchors["baseline"], [0.0, 25.0])
        self.assertEqual(anchors["free_throw_line"], [19.0, 25.0])
        self.assertAlmostEqual(anchors["free_throw_circle_far_half"][0], 25.0, places=2)
        self.assertAlmostEqual(anchors["free_throw_circle_far_half"][1], 25.0, places=9)
        self.assertAlmostEqual(anchors["free_throw_circle_near_half"][0], 13.0, places=2)
        self.assertAlmostEqual(anchors["free_throw_circle_near_half"][1], 25.0, places=9)
        self.assertAlmostEqual(anchors["restricted_area_arc"][0], 9.25, places=2)
        self.assertAlmostEqual(anchors["restricted_area_arc"][1], 25.0, places=9)
        self.assertLess(
            anchors["restricted_area_arc"][0],
            anchors["free_throw_circle_near_half"][0],
        )
        self.assertLess(
            anchors["free_throw_circle_near_half"][0],
            anchors["free_throw_line"][0],
        )
        self.assertLess(
            anchors["free_throw_line"][0],
            anchors["free_throw_circle_far_half"][0],
        )

        # A numbered badge identifies a whole marking, so its centre must sit on
        # that marking rather than looking like a scattered keypoint.
        for feature, anchor in reference["label_anchors"].items():
            distances = []
            for start, end in zip(
                reference["markings"][feature],
                reference["markings"][feature][1:],
            ):
                dx, dy = end[0] - start[0], end[1] - start[1]
                length_sq = dx * dx + dy * dy
                amount = (
                    0.0
                    if length_sq == 0
                    else max(
                        0.0,
                        min(
                            1.0,
                            (
                                (anchor[0] - start[0]) * dx
                                + (anchor[1] - start[1]) * dy
                            )
                            / length_sq,
                        ),
                    )
                )
                closest = (start[0] + amount * dx, start[1] + amount * dy)
                distances.append(math.dist(anchor, closest))
            self.assertLess(min(distances), 1e-9, feature)

    def test_locked_image_is_served_by_frame_id(self):
        response = self.client.get("/benchmark-images/video_1_000012")
        try:
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.data.startswith(b"\x89PNG"))
        finally:
            response.close()
        self.assertEqual(
            self.client.get("/benchmark-images/not-a-frame").status_code, 404
        )

    def test_empty_annotation_has_locked_human_metadata(self):
        annotation = self.empty_annotation()
        self.assertEqual(annotation["frame_id"], self.frame["frame_id"])
        self.assertEqual(annotation["image_sha256"], self.frame["sha256"])
        self.assertEqual(annotation["coordinate_space"], "frame")
        self.assertEqual(annotation["pass_id"], HUMAN_ANNOTATOR_PASS_ID)
        self.assertEqual(
            annotation["prompt_version"], HUMAN_ANNOTATOR_PROMPT_VERSION
        )
        self.assertEqual(annotation["features"], [])
        self.assertEqual(annotation["skipped"], [])

    def test_incomplete_draft_saves_but_cannot_finalize(self):
        annotation = self.empty_annotation()
        draft = self.save(annotation)
        self.assertEqual(draft.status_code, 200)
        self.assertTrue(draft.get_json()["problems"])
        self.assertTrue(
            (
                self.benchmark_dir
                / "run"
                / "human"
                / "drafts"
                / "video_1_000012.json"
            ).is_file()
        )

        final = self.save(annotation, finalize=True)
        self.assertEqual(final.status_code, 422)
        self.assertFalse(
            (
                self.benchmark_dir
                / "run"
                / "human"
                / "video_1_000012.frame.json"
            ).exists()
        )

    def test_complete_annotation_finalizes_and_updates_progress(self):
        annotation = self.skip_all(self.empty_annotation())
        response = self.save(annotation, finalize=True)
        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertTrue(response.get_json()["finalized"])
        final_path = (
            self.benchmark_dir
            / "run"
            / "human"
            / "video_1_000012.frame.json"
        )
        self.assertTrue(final_path.is_file())
        self.assertEqual(
            json.loads(final_path.read_text())["annotator_id"], "human-pri"
        )
        self.assertTrue(
            self.client.get("/api/benchmark").get_json()["frames"][0]["complete"]
        )
        loaded = self.client.get(
            "/api/benchmark/annotation/video_1_000012"
        ).get_json()
        self.assertEqual(loaded["source"], "final")
        self.assertTrue(loaded["finalized"])

    def test_human_trace_requires_four_straight_samples(self):
        annotation = self.skip_all(
            self.empty_annotation(), except_features={MarkingFeature.BASELINE}
        )
        annotation["features"] = [
            {
                "feature": "baseline",
                "kind": "polyline",
                "points": [{"x": 100, "y": 600}, {"x": 900, "y": 600}],
                "visibility": "clear",
                "uncertainty_px": 1,
                "occluded_after": [],
                "notes": "",
            }
        ]
        response = self.save(annotation, finalize=True)
        self.assertEqual(response.status_code, 422)
        self.assertTrue(
            any(
                "requires at least 4" in problem
                for problem in response.get_json()["problems"]
            )
        )

    def test_frame_metadata_cannot_be_changed_even_in_a_draft(self):
        annotation = self.empty_annotation()
        annotation["image_sha256"] = "b" * 64
        response = self.save(annotation)
        self.assertEqual(response.status_code, 400)
        self.assertTrue(
            any(
                "image_sha256" in problem
                for problem in response.get_json()["problems"]
            )
        )


class MigrationTests(unittest.TestCase):
    def test_migrate_removes_v2_artifacts(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        dataset_dir = root / "dataset"
        dataset_dir.mkdir()
        manifest = make_manifest(dataset_dir)
        manifest["frames"]["clipA_f000001"]["label_status"] = "labeled"
        manifest["clips"]["clipA"]["orientation"] = {
            "anchor_frame_id": "clipA_f000001",
            "north_convention": "image_left_basket",
            "prefill_normalization": "identity",
            "raw_first_end_relation": "left",
            "method": "operator_selected",
        }
        save_manifest(dataset_dir, manifest)
        label_dir = dataset_dir / "labels" / "clipA"
        label_dir.mkdir(parents=True)
        (label_dir / "clipA_f000001.json").write_text(
            json.dumps({
                "frame_id": "clipA_f000001",
                "schema": {"schema_name": "basketball-court-keypoints", "schema_version": "2.1.0"},
                "keypoints": [],
            })
        )

        result = subprocess.run(
            [sys.executable, str(PROJECT_DIR / "migrate_to_v3.py"), "--dataset", str(dataset_dir)],
            capture_output=True,
            text=True,
            cwd=PROJECT_DIR,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((label_dir / "clipA_f000001.json").exists())
        migrated = json.loads((dataset_dir / "manifest.json").read_text())
        self.assertEqual(migrated["frames"]["clipA_f000001"]["label_status"], "unlabeled")
        self.assertNotIn("orientation", migrated["clips"]["clipA"])

        # Idempotent: a second run changes nothing and reports nothing to do.
        rerun = subprocess.run(
            [sys.executable, str(PROJECT_DIR / "migrate_to_v3.py"), "--dataset", str(dataset_dir)],
            capture_output=True,
            text=True,
            cwd=PROJECT_DIR,
        )
        self.assertEqual(rerun.returncode, 0, rerun.stderr)
        self.assertIn("nothing to do", rerun.stdout)


if __name__ == "__main__":
    unittest.main()
