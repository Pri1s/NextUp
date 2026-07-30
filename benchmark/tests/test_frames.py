"""The blind frame tree, and the guards that keep it blind.

Contamination here is not hypothetical. The engine already writes one overlaid
image per frame, named by clip and index, in a directory next to this one. Copying
those in would be the obvious convenience and would quietly destroy the
benchmark's independence: an annotator shown the current projection places the
free-throw circle where the projection puts it, and the benchmark then certifies
the drift it exists to detect.

So ``assert_blind`` is tested the way a guard has to be -- by planting exactly that
mistake and requiring it to be caught.
"""

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from benchmark.frames import (
    REQUIRED_STRATA,
    BenchmarkManifest,
    ExportedFrame,
    ManifestFrame,
    assert_blind,
    frame_id_for,
    load_manifest,
    read_export_index,
    sha256_of,
    validate_manifest,
    verify_export,
    write_export_index,
)

REPO_DIR = Path(__file__).resolve().parents[2]


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.manifest = load_manifest("nba_m1_v1")
        self.expanded = load_manifest("nba_m1_v1_expanded")

    def test_shipped_manifest_is_valid(self):
        self.assertEqual(validate_manifest(self.manifest, REPO_DIR), [])
        self.assertEqual(validate_manifest(self.expanded, REPO_DIR), [])

    def test_shipped_manifest_holds_six_frames_two_per_clip(self):
        self.assertEqual(len(self.manifest.frames), 6)
        by_clip: dict[str, int] = {}
        for frame in self.manifest.frames:
            by_clip[frame.clip] = by_clip.get(frame.clip, 0) + 1
        self.assertEqual(sorted(by_clip.values()), [2, 2, 2])

    def test_pilot_is_two_frames_per_clip(self):
        """One easy and one hard per clip, so the gate measures both ends."""
        by_clip: dict[str, int] = {}
        for frame in self.manifest.pilot_frames:
            by_clip[frame.clip] = by_clip.get(frame.clip, 0) + 1
        self.assertEqual(sorted(by_clip.values()), [2, 2, 2])

    def test_locked_easy_and_hard_frame_ids_are_unchanged(self):
        self.assertEqual(
            {frame.frame_id for frame in self.manifest.frames},
            {
                "video_1_000012",
                "video_1_000090",
                "video_2_000018",
                "video_2_000141",
                "video_3_000015",
                "video_3_000146",
            },
        )

    def test_conditional_expansion_is_capped_at_nine_and_contains_the_core(self):
        self.assertEqual(len(self.expanded.frames), 9)
        self.assertLessEqual(
            {frame.frame_id for frame in self.manifest.frames},
            {frame.frame_id for frame in self.expanded.frames},
        )

    def test_every_frame_comes_from_the_already_locked_seventy_two(self):
        """Reusing the existing indices is what lets baselines score without a re-sample."""
        for clip in ("video_1", "video_2", "video_3"):
            run = json.loads(
                (REPO_DIR / "engine_out" / "calibration" / clip / "interim" / "run.json")
                .read_text(encoding="utf-8")
            )
            locked = set(run["frame_indices"])
            chosen = {
                f.frame_index for f in self.expanded.frames if Path(f.clip).stem == clip
            }
            self.assertTrue(chosen <= locked, f"{clip}: {sorted(chosen - locked)} not in the 72")

    def test_missing_stratum_is_reported(self):
        thinned = BenchmarkManifest(
            manifest_id="thin",
            layout_id="nba_halfcourt",
            description="",
            frames=tuple(f for f in self.manifest.frames if f.stratum == "clean"),
        )
        problems = validate_manifest(thinned, REPO_DIR)
        self.assertTrue(any("stratum" in p for p in problems), problems)

    def test_required_strata_cover_the_design_plan_axes(self):
        self.assertIn("occluded", REQUIRED_STRATA)
        self.assertIn("logo_midcourt", REQUIRED_STRATA)
        self.assertIn("graphics_tight", REQUIRED_STRATA)
        expanded_strata = {frame.stratum for frame in self.expanded.frames}
        self.assertIn("pan_blur", expanded_strata)
        self.assertIn("failing_implausible", expanded_strata)

    def test_frame_id_must_match_clip_and_index(self):
        broken = BenchmarkManifest(
            manifest_id="broken",
            layout_id="nba_halfcourt",
            description="",
            frames=(
                ManifestFrame("wrong_id", "input_videos/video_1.mp4", 12, "clean", True, "n"),
            ),
        )
        self.assertTrue(any("does not match" in p for p in validate_manifest(broken, REPO_DIR)))

    def test_single_clip_manifest_is_reported(self):
        """A benchmark from one clip measures one floor and one camera."""
        one_clip = BenchmarkManifest(
            manifest_id="one",
            layout_id="nba_halfcourt",
            description="",
            frames=tuple(
                f for f in self.manifest.frames if f.clip == "input_videos/video_1.mp4"
            ),
        )
        problems = validate_manifest(one_clip, REPO_DIR)
        self.assertTrue(any("all three" in p for p in problems), problems)

    def test_frame_id_helper(self):
        self.assertEqual(frame_id_for("input_videos/video_2.mp4", 119), "video_2_000119")

    def test_round_trip(self):
        restored = BenchmarkManifest.from_dict(self.manifest.as_dict())
        self.assertEqual(restored, self.manifest)

    def test_foreign_schema_is_refused(self):
        payload = self.manifest.as_dict()
        payload["schema_version"] = "benchmark-manifest-0.1.0"
        with self.assertRaises(ValueError):
            BenchmarkManifest.from_dict(payload)


class ExportTreeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out = Path(self._tmp.name) / "frames"
        (self.out / "images").mkdir(parents=True)
        self.addCleanup(self._tmp.cleanup)

        image = np.full((720, 1280, 3), 120, dtype=np.uint8)
        path = self.out / "images" / "video_1_000012.png"
        cv2.imwrite(str(path), image)

        self.frame = ExportedFrame(
            frame_id="video_1_000012",
            clip="input_videos/video_1.mp4",
            frame_index=12,
            stratum="clean",
            pilot=True,
            path="images/video_1_000012.png",
            sha256=sha256_of(path),
            width=1280,
            height=720,
        )
        manifest = BenchmarkManifest("t", "nba_halfcourt", "", ())
        write_export_index(self.out, manifest, [self.frame], "# glossary\n", "glossary-1.0.0")

    def test_intact_tree_verifies_and_is_blind(self):
        self.assertEqual(verify_export(self.out), [])
        self.assertEqual(assert_blind(self.out), [])

    def test_index_round_trips(self):
        self.assertEqual(read_export_index(self.out), [self.frame])

    def test_modified_image_is_detected(self):
        """Two passes must be provably looking at the same bytes, not assumed to."""
        path = self.out / "images" / "video_1_000012.png"
        image = cv2.imread(str(path))
        image[0, 0] = (255, 255, 255)
        cv2.imwrite(str(path), image)
        self.assertTrue(any("content changed" in p for p in verify_export(self.out)))

    def test_missing_image_is_detected(self):
        (self.out / "images" / "video_1_000012.png").unlink()
        self.assertTrue(any("missing" in p for p in verify_export(self.out)))

    def test_planted_reprojection_overlay_is_caught(self):
        """The exact contamination §7.2 warns about: an engine overlay in the tree."""
        (self.out / "reprojected_frame_000012.jpg").write_bytes(b"not really a jpeg")
        problems = assert_blind(self.out)
        self.assertTrue(any("model-derived" in p for p in problems), problems)

    def test_any_unexpected_file_is_reported(self):
        (self.out / "notes.txt").write_text("scratch", encoding="utf-8")
        self.assertTrue(any("unexpected file" in p for p in assert_blind(self.out)))

    def test_non_png_in_images_is_reported(self):
        (self.out / "images" / "sneaky.jpg").write_bytes(b"x")
        self.assertTrue(any("PNG" in p for p in assert_blind(self.out)))

    def test_index_referencing_a_model_output_tree_is_caught(self):
        index = self.out / "frames.json"
        payload = json.loads(index.read_text(encoding="utf-8"))
        payload["frames"][0]["path"] = "engine_out/calibration/video_1/interim/x.png"
        index.write_text(json.dumps(payload), encoding="utf-8")
        self.assertTrue(any("model-output tree" in p for p in assert_blind(self.out)))

    def test_glossary_ships_beside_the_images(self):
        self.assertTrue((self.out / "GLOSSARY.md").is_file())


class ExportedTreeOnDiskTests(unittest.TestCase):
    """The real exported tree, if it has been built."""

    def setUp(self):
        self.out = REPO_DIR / "engine_out" / "benchmark" / "frames"
        if not (self.out / "frames.json").is_file():
            self.skipTest("frame set not exported yet")
        payload = json.loads((self.out / "frames.json").read_text(encoding="utf-8"))
        if payload.get("manifest_id") != "nba_m1_v1":
            self.skipTest("only a legacy benchmark export is present")

    def test_exported_tree_is_intact_and_blind(self):
        self.assertEqual(verify_export(self.out) + assert_blind(self.out), [])

    def test_exported_tree_holds_the_whole_manifest(self):
        self.assertEqual(len(read_export_index(self.out)), 6)

    def test_exported_frames_are_native_resolution(self):
        for frame in read_export_index(self.out):
            self.assertEqual((frame.width, frame.height), (1280, 720), frame.frame_id)


if __name__ == "__main__":
    unittest.main()
