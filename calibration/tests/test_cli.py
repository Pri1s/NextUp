"""CLI surface.

Two things matter here. Argument parsing has to be right, because a mistyped
sweep list silently inspecting one size would waste a review cycle. And the verb
list has to stay at one: exposing ``calibrate-frame`` before the slots have
meaning would invite exactly the work the Phase-0 gate exists to prevent.
"""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from calibration import cli
from calibration.io.video import ImageFrameSource, VideoFrameSource, resolve_image_paths

import cv2
import numpy as np


class ImgszParsingTests(unittest.TestCase):
    def test_single(self):
        self.assertEqual(cli.parse_imgsz("640"), (640,))

    def test_list_preserves_order_so_the_first_is_primary(self):
        self.assertEqual(cli.parse_imgsz("960,640,1280"), (960, 640, 1280))

    def test_whitespace_and_duplicates(self):
        self.assertEqual(cli.parse_imgsz(" 640 , 960,640 "), (640, 960))

    def test_rejects_non_positive(self):
        with self.assertRaises(ValueError):
            cli.parse_imgsz("0")

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            cli.parse_imgsz(" , ")

    def test_rejects_garbage(self):
        with self.assertRaises(ValueError):
            cli.parse_imgsz("large")


class ParserTests(unittest.TestCase):
    def setUp(self):
        self.parser = cli.build_parser()

    @contextlib.contextmanager
    def assertUsageError(self):
        """argparse prints usage to stderr before exiting; keep it out of the report."""
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            yield

    def test_defaults(self):
        args = self.parser.parse_args(["inspect-model", "--weights", "w.pt", "--video", "v.mp4"])
        self.assertEqual(args.imgsz, "640")
        self.assertEqual(args.num_frames, 24)
        self.assertFalse(args.flip_probe)
        self.assertEqual(args.out, "engine_out/phase0")

    def test_default_imgsz_matches_training(self):
        """640 is the checkpoint's training size; drifting off it by default
        would change results without anyone choosing to."""
        args = self.parser.parse_args(["inspect-model", "--weights", "w.pt", "--video", "v.mp4"])
        self.assertEqual(cli.parse_imgsz(args.imgsz), (640,))

    def test_default_frame_count_meets_the_design_floor(self):
        args = self.parser.parse_args(["inspect-model", "--weights", "w.pt", "--video", "v.mp4"])
        self.assertGreaterEqual(args.num_frames, 20)

    def test_flags(self):
        args = self.parser.parse_args(
            [
                "inspect-model",
                "--weights", "w.pt",
                "--video", "v.mp4",
                "--imgsz", "640,1280",
                "--frames", "0,40",
                "--flip-probe",
                "--gate", "0.4",
                "--device", "mps",
            ]
        )
        self.assertEqual(cli.parse_imgsz(args.imgsz), (640, 1280))
        self.assertEqual(args.frames, "0,40")
        self.assertTrue(args.flip_probe)
        self.assertEqual(args.gate, 0.4)
        self.assertEqual(args.device, "mps")

    def test_video_and_images_are_mutually_exclusive(self):
        with self.assertUsageError():
            self.parser.parse_args(
                ["inspect-model", "--weights", "w.pt", "--video", "v.mp4", "--images", "d/"]
            )

    def test_a_source_is_required(self):
        with self.assertUsageError():
            self.parser.parse_args(["inspect-model", "--weights", "w.pt"])

    def test_weights_are_required(self):
        with self.assertUsageError():
            self.parser.parse_args(["inspect-model", "--video", "v.mp4"])

    def test_a_command_is_required(self):
        with self.assertUsageError():
            self.parser.parse_args([])

    def test_exposed_verbs(self):
        """The interim engine adds calibration, but nothing that assumes verified slots.

        ``calibrate-frames`` runs on the provisional evidence map and can return
        an uncalibrated status; there is deliberately no verb that consumes a
        settled index-to-landmark schema, because none exists.
        """
        actions = [
            a for a in self.parser._subparsers._group_actions if hasattr(a, "choices")
        ]
        self.assertEqual(
            sorted(actions[0].choices),
            ["calibrate-frames", "inspect-model", "validate-layout"],
        )

    def test_calibrate_defaults_to_confident_evidence_only(self):
        """Bullet 1: fit only from independently supported landmarks."""
        args = self.parser.parse_args(
            ["calibrate-frames", "--weights", "w.pt", "--video", "v.mp4"]
        )
        self.assertEqual(cli.parse_tiers(args.tiers), ("confident",))
        self.assertEqual(args.evidence_map, "reloc2_18_provisional")
        self.assertEqual(args.layout, "nba_halfcourt")


class ImageSourceTests(unittest.TestCase):
    """The intake path for no-court negatives."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        for name in ("b.png", "a.png"):
            cv2.imwrite(str(self.tmp / name), np.full((90, 160, 3), 120, dtype=np.uint8))
        (self.tmp / "notes.txt").write_text("ignored")

    def test_directory_expands_to_sorted_images_only(self):
        paths = resolve_image_paths(str(self.tmp))
        self.assertEqual([p.name for p in paths], ["a.png", "b.png"])

    def test_glob_expands(self):
        paths = resolve_image_paths(str(self.tmp / "*.png"))
        self.assertEqual(len(paths), 2)

    def test_no_match_raises(self):
        with self.assertRaises(FileNotFoundError):
            resolve_image_paths(str(self.tmp / "*.tiff"))

    def test_image_source_reads_frames(self):
        source = ImageFrameSource(resolve_image_paths(str(self.tmp)))
        self.assertEqual(source.info.frame_count, 2)
        self.assertEqual(source.info.kind, "images")
        self.assertIsNone(source.info.fps)
        self.assertEqual(source.frame_at(0).shape, (90, 160, 3))
        self.assertEqual(source.timestamp_of(1), 0.0)
        source.close()

    def test_out_of_range_index(self):
        source = ImageFrameSource(resolve_image_paths(str(self.tmp)))
        with self.assertRaises(IndexError):
            source.frame_at(5)
        source.close()

    def test_empty_list_rejected(self):
        with self.assertRaises(ValueError):
            ImageFrameSource([])


class VideoSourceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.path = self.tmp / "clip.mp4"
        writer = cv2.VideoWriter(
            str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (160, 90)
        )
        if not writer.isOpened():
            self.skipTest("no mp4v encoder available")
        for i in range(20):
            frame = np.full((90, 160, 3), 10, dtype=np.uint8)
            frame[:, : 8 * (i + 1)] = 240  # a per-frame marker to seek against
            writer.write(frame)
        writer.release()

    def test_reads_metadata(self):
        source = VideoFrameSource(self.path)
        self.assertEqual(source.info.kind, "video")
        self.assertEqual((source.info.width, source.info.height), (160, 90))
        self.assertAlmostEqual(source.info.fps, 30.0, places=1)
        source.close()

    def test_seeks_to_the_requested_frame(self):
        source = VideoFrameSource(self.path)
        early = source.frame_at(2)
        late = source.frame_at(15)
        self.assertLess(int(early[:, :, 0].mean()), int(late[:, :, 0].mean()))
        source.close()

    def test_timestamps_follow_fps(self):
        source = VideoFrameSource(self.path)
        self.assertAlmostEqual(source.timestamp_of(30), 1.0, places=2)
        source.close()

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            VideoFrameSource(self.tmp / "absent.mp4")


if __name__ == "__main__":
    unittest.main()
