import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from hud_detection import (
    HudDetection,
    arc_hud_overlap,
    build_hud_context,
    detect_hud,
    line_hud_overlap,
    primitive_hud_overlap,
)


def synthetic_broadcast_frames() -> list[np.ndarray]:
    """Three moving scenes with one stable, saturated bottom HUD."""
    height, width = 240, 320
    frames = []
    for index in range(3):
        image = np.full((height, width, 3), (165, 185, 205), np.uint8)
        # Moving, textured scene content should not acquire temporal support.
        cv2.rectangle(image, (25 + index * 75, 70), (100 + index * 75, 165), (35, 65, 95), -1)
        cv2.line(image, (0, 60 + index * 35), (319, 115 + index * 25), (245, 245, 245), 4)
        # Smooth saturated court paint is deliberately not a graphic.
        cv2.rectangle(image, (110, 120), (210, 185), (150, 35, 150), -1)
        # Stable, dense, high-contrast broadcast scoreboard on the bottom border.
        cv2.rectangle(image, (80, 216), (240, 239), (100, 20, 100), -1)
        for x in range(84, 238, 12):
            colour = (0, 230, 255) if (x // 12) % 2 else (255, 255, 255)
            cv2.rectangle(image, (x, 219), (x + 5, 235), colour, -1)
        cv2.putText(image, "12:34", (118, 234), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
        frames.append(image)
    return frames


class HudContextTests(unittest.TestCase):
    def test_groups_same_clip_and_resolution_and_samples_at_most_seven(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for frame_index in range(0, 90, 10):
                cv2.imwrite(str(root / f"clip_a_f{frame_index:06d}.png"), np.full((80, 120, 3), frame_index, np.uint8))
            cv2.imwrite(str(root / "clip_b_f000000.png"), np.zeros((80, 120, 3), np.uint8))
            cv2.imwrite(str(root / "clip_a_f000095.png"), np.zeros((40, 60, 3), np.uint8))

            anchor = root / "clip_a_f000040.png"
            context = build_hud_context(anchor)

            self.assertEqual(len(context.images), 7)
            self.assertIn(40, context.frame_indices)
            self.assertEqual(min(value for value in context.frame_indices if value is not None), 0)
            self.assertEqual(max(value for value in context.frame_indices if value is not None), 80)
            self.assertTrue(all(image.shape[:2] == (80, 120) for image in context.images))
            self.assertTrue(all("clip_a_" in path for path in context.paths))
            self.assertEqual(context.mode, "temporal")

    def test_short_context_uses_static_singleton_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for frame_index in (10, 20, 29):
                cv2.imwrite(str(root / f"clip_a_f{frame_index:06d}.png"), np.zeros((80, 120, 3), np.uint8))
            context = build_hud_context(root / "clip_a_f000020.png")
            self.assertEqual(len(context.images), 3)
            self.assertEqual(context.frame_span, 19)
            self.assertEqual(context.mode, "static_singleton")

    def test_explicit_context_directory_supports_an_exported_singleton(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            export = root / "export"
            source = root / "source"
            export.mkdir()
            source.mkdir()
            image = np.zeros((80, 120, 3), np.uint8)
            anchor = export / "clip_a_f000040.png"
            cv2.imwrite(str(anchor), image)
            cv2.imwrite(str(source / "clip_a_f000000.png"), image)
            cv2.imwrite(str(source / "clip_a_f000080.png"), image)

            context = build_hud_context(anchor, [source])

            self.assertEqual(context.frame_indices, (0, 40, 80))
            self.assertEqual(context.mode, "temporal")


class HudDetectionTests(unittest.TestCase):
    def test_temporal_hud_uses_stability_persistent_edges_and_saturation(self):
        frames = synthetic_broadcast_frames()
        result = detect_hud(
            frames[1],
            context_images=frames,
            frame_indices=[0, 30, 60],
            context_frames=["clip_f000000.png", "clip_f000030.png", "clip_f000060.png"],
            anchor_position=1,
        )

        self.assertEqual(result.diagnostics["mode"], "temporal")
        self.assertEqual(result.diagnostics["context_group"], "clip")
        self.assertGreaterEqual(np.mean(result.mask[216:240, 80:241] > 0), .90)
        self.assertEqual(np.count_nonzero(result.mask[120:186, 110:211]), 0)
        self.assertTrue(result.diagnostics["regions"])
        self.assertEqual(result.diagnostics["masked_fraction"], result.diagnostics["masked_pixel_fraction"])
        self.assertEqual(result.diagnostics["window_px"], [15, 15])

    def test_temporal_stability_is_relative_to_the_anchor(self):
        anchor = synthetic_broadcast_frames()[1]
        contexts = [anchor.copy(), anchor, anchor.copy()]
        # The two context frames agree with each other but not with the anchor
        # in the graphic region.  A temporal-median reference would falsely
        # call that region stable; the anchor-relative definition must not.
        for candidate in (contexts[0], contexts[2]):
            candidate[216:240, 80:241] = 255 - candidate[216:240, 80:241]
        result = detect_hud(
            anchor,
            context_images=contexts,
            frame_indices=[0, 30, 60],
            anchor_position=1,
        )
        self.assertEqual(np.count_nonzero(result.mask[210:240, 70:251]), 0)

    def test_static_fallback_finds_dense_hud_but_not_saturated_court_paint(self):
        frame = synthetic_broadcast_frames()[1]
        result = detect_hud(frame)

        self.assertEqual(result.diagnostics["mode"], "static_singleton")
        self.assertGreaterEqual(np.mean(result.mask[216:240, 80:241] > 0), .90)
        self.assertEqual(np.count_nonzero(result.mask[120:186, 110:211]), 0)

    def test_lower_image_position_alone_never_creates_a_mask(self):
        image = np.full((240, 320, 3), (170, 190, 210), np.uint8)
        cv2.rectangle(image, (0, 180), (319, 239), (185, 205, 225), -1)
        result = detect_hud(image)
        self.assertEqual(np.count_nonzero(result.mask), 0)


class HudOverlapTests(unittest.TestCase):
    def setUp(self):
        mask = np.zeros((100, 100), np.uint8)
        labels = np.zeros((100, 100), np.int32)
        mask[:, 40:60] = 255
        labels[:, 40:60] = 1
        self.hud = HudDetection(mask, labels, {"regions": [{"id": "hud_000"}]})

    def test_line_overlap_samples_finite_extent_and_region_ids(self):
        evidence = line_hud_overlap(self.hud, (0, 50), (99, 50))
        self.assertGreater(evidence["hud_overlap"], .15)
        self.assertLess(evidence["hud_overlap"], .25)
        self.assertEqual(evidence["hud_region_ids"], ["hud_000"])
        self.assertEqual(line_hud_overlap(self.hud, (0, 20), (30, 20))["hud_overlap"], 0)

    def test_arc_overlap_uses_seventy_two_perimeter_samples(self):
        evidence = arc_hud_overlap(self.hud, (50, 50), 20)
        self.assertGreater(evidence["hud_overlap"], .25)
        self.assertLess(evidence["hud_overlap"], .40)
        primitive = {"type": "circle", "geometry": {"center": [50, 50], "radius_px": 20}}
        self.assertEqual(primitive_hud_overlap(primitive, self.hud), evidence)


if __name__ == "__main__":
    unittest.main()
