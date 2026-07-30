"""The pictures must not hide what the records contain.

A reviewer trusts the overlay over the JSONL, so an overlay that quietly omits a
low-confidence point is worse than no overlay at all: it makes an unstable slot
look clean. These tests assert that everything the model emitted is visible.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from contracts.calibration_types import RawKeypointFrame, RawKeypointObservation
from calibration.tools import overlays

from ._synthetic import checkerboard


def _record(points, detection_state="detected", **overrides) -> RawKeypointFrame:
    defaults = dict(
        frame_index=7,
        timestamp_s=0.23,
        source_id="clip.mp4",
        image_width=1280,
        image_height=720,
        detection_state=detection_state,
        instance_count=1 if detection_state == "detected" else 0,
        instance_confidences=(0.9,) if detection_state == "detected" else (),
        selected_instance=0 if detection_state == "detected" else None,
        observations=tuple(
            RawKeypointObservation(i, x, y, c, clamped=bool(flag))
            for i, (x, y, c, flag) in enumerate(points)
        ),
    )
    defaults.update(overrides)
    return RawKeypointFrame(**defaults)


def _differs_near(a: np.ndarray, b: np.ndarray, x: int, y: int, radius: int = 14) -> bool:
    y0, y1 = max(0, y - radius), min(a.shape[0], y + radius)
    x0, x1 = max(0, x - radius), min(a.shape[1], x + radius)
    return bool(np.any(a[y0:y1, x0:x1] != b[y0:y1, x0:x1]))


class ColorTests(unittest.TestCase):
    def test_ramp_is_continuous_not_bucketed(self):
        """Two nearby confidences must not collapse to one colour."""
        self.assertNotEqual(overlays.confidence_color(0.10), overlays.confidence_color(0.40))
        self.assertNotEqual(overlays.confidence_color(0.60), overlays.confidence_color(0.95))

    def test_ramp_clamps_outside_range(self):
        self.assertEqual(overlays.confidence_color(-1.0), overlays.confidence_color(0.0))
        self.assertEqual(overlays.confidence_color(2.0), overlays.confidence_color(1.0))


class OverlayTests(unittest.TestCase):
    def setUp(self):
        self.frame = checkerboard()

    def test_below_gate_point_is_still_drawn(self):
        """The whole point of Phase 0 is seeing where low-confidence slots go."""
        record = _record([(600.0, 300.0, 0.02, False)])
        drawn = overlays.draw_overlay(self.frame, record, gate=0.25)
        self.assertTrue(_differs_near(drawn, self.frame, 600, 300))

    def test_above_gate_point_is_drawn(self):
        record = _record([(400.0, 200.0, 0.95, False)])
        drawn = overlays.draw_overlay(self.frame, record, gate=0.25)
        self.assertTrue(_differs_near(drawn, self.frame, 400, 200))

    def test_every_slot_appears(self):
        points = [(100.0 + 60 * i, 400.0, 0.5, False) for i in range(18)]
        drawn = overlays.draw_overlay(self.frame, _record(points), gate=0.25)
        for i in range(18):
            self.assertTrue(
                _differs_near(drawn, self.frame, int(100 + 60 * i), 400),
                f"slot {i} was not drawn",
            )

    def test_clamped_point_gets_a_marker(self):
        plain = overlays.draw_overlay(self.frame, _record([(300.0, 300.0, 0.9, False)]), 0.25)
        flagged = overlays.draw_overlay(self.frame, _record([(300.0, 300.0, 0.9, True)]), 0.25)
        self.assertTrue(np.any(plain != flagged))

    def test_no_detection_frame_renders_a_banner(self):
        record = _record([], detection_state="no_detection")
        drawn = overlays.draw_overlay(self.frame, record, gate=0.25)
        self.assertEqual(drawn.shape, self.frame.shape)
        self.assertTrue(np.any(drawn != self.frame))

    def test_original_frame_is_not_mutated(self):
        before = self.frame.copy()
        overlays.draw_overlay(self.frame, _record([(500.0, 300.0, 0.9, False)]), 0.25)
        np.testing.assert_array_equal(self.frame, before)

    def test_point_outside_the_frame_does_not_crash(self):
        record = _record([(-40.0, 900.0, 0.5, True)])
        drawn = overlays.draw_overlay(self.frame, record, gate=0.25)
        self.assertEqual(drawn.shape, self.frame.shape)

    def test_write_overlay_produces_a_readable_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = overlays.write_overlay(
                Path(tmp) / "nested" / "frame.jpg", self.frame, _record([(1.0, 2.0, 0.5, True)]), 0.25
            )
            self.assertTrue(path.is_file())
            self.assertEqual(cv2.imread(str(path)).shape, self.frame.shape)


class ContactSheetTests(unittest.TestCase):
    def test_grid_dimensions(self):
        images = [checkerboard(320, 180, seed=i) for i in range(6)]
        sheet = overlays.build_contact_sheet(images, columns=4)
        self.assertEqual(sheet.shape[1], 4 * overlays.CONTACT_TILE_W)
        self.assertEqual(sheet.shape[0] % 2, 0)  # two rows of equal height

    def test_single_image(self):
        sheet = overlays.build_contact_sheet([checkerboard(320, 180)], columns=4)
        self.assertEqual(sheet.shape[1], overlays.CONTACT_TILE_W)

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            overlays.build_contact_sheet([])


class FilmstripTests(unittest.TestCase):
    def setUp(self):
        self.frames = [checkerboard(seed=i) for i in range(4)]
        self.records = [
            _record([(300.0, 200.0, 0.9, False)], frame_index=i) for i in range(4)
        ]

    def test_one_tile_per_frame(self):
        strip = overlays.build_slot_filmstrip(self.frames, self.records, 0, 0.25, crop_size=100)
        self.assertEqual(strip.shape[1], 4 * 100)
        self.assertEqual(strip.shape[0], 100 + overlays.FILMSTRIP_CAPTION_H)

    def test_strip_exists_for_a_slot_that_was_never_detected(self):
        """A dead slot still needs a strip, or its absence looks like a bug."""
        strip = overlays.build_slot_filmstrip(self.frames, self.records, 17, 0.25, crop_size=100)
        self.assertEqual(strip.shape[1], 4 * 100)

    def test_no_detection_frames_become_placeholder_tiles(self):
        records = [_record([], detection_state="no_detection", frame_index=i) for i in range(4)]
        strip = overlays.build_slot_filmstrip(self.frames, records, 0, 0.25, crop_size=100)
        self.assertEqual(strip.shape[1], 4 * 100)

    def test_crop_is_centred_on_the_prediction(self):
        """Padding, not shifting — so tiles across a strip stay comparable."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        frame[50, 60] = (255, 255, 255)
        crop = overlays._centered_crop(frame, 60.0, 50.0, 40)
        self.assertEqual(tuple(crop[20, 20]), (255, 255, 255))

    def test_crop_past_the_border_is_padded_not_shifted(self):
        frame = np.full((200, 200, 3), 128, dtype=np.uint8)
        crop = overlays._centered_crop(frame, 2.0, 2.0, 40)
        self.assertEqual(crop.shape, (40, 40, 3))
        self.assertEqual(tuple(crop[0, 0]), (0, 0, 0))  # padded region
        self.assertEqual(tuple(crop[20, 20]), (128, 128, 128))  # the prediction itself

    def test_crop_entirely_outside_the_frame_returns_padding(self):
        frame = np.full((200, 200, 3), 128, dtype=np.uint8)
        crop = overlays._centered_crop(frame, -500.0, -500.0, 40)
        self.assertEqual(crop.shape, (40, 40, 3))
        self.assertFalse(crop.any())


if __name__ == "__main__":
    unittest.main()
