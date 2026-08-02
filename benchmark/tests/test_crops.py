"""The crop transform is the benchmark's precision floor.

Everything an annotator produces passes through this conversion. A transform that
is off by a pixel biases every label in the same direction, which is invisible to
inter-pass agreement -- both passes go through the same converter -- and shows up
only as a constant error against ground truth. So the round trip is asserted
exactly, not approximately.

The refusal tests matter as much as the arithmetic. A point resolved against the
wrong crop lands somewhere entirely else while remaining perfectly well-formed,
and a silently dropped point leaves a shorter feature that still looks plausible.
Both must raise rather than degrade.
"""

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from contracts.annotations import (
    AnnotatedFeature,
    AnnotatedJunction,
    FrameAnnotation,
    JunctionPoint,
    MarkingFeature,
    SamplePoint,
)

from benchmark.crops import (
    DEFAULT_SCALE,
    DEFAULT_SIZE_PX,
    MIN_MEASURING_SCALE,
    CropTransform,
    UnknownCrop,
    crop_id_for,
    load_crops,
    make_crop,
    resolve_annotation,
)

FRAME_ID = "video_9_000042"
WIDTH, HEIGHT = 1280, 720


class CropFixture(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.frames_dir = self.root / "frames"
        (self.frames_dir / "images").mkdir(parents=True)

        rng = np.random.default_rng(0)
        image = rng.integers(80, 200, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)
        cv2.line(image, (100, 600), (1100, 200), (255, 255, 255), 3)
        cv2.imwrite(str(self.frames_dir / "images" / f"{FRAME_ID}.png"), image)
        self.addCleanup(self._tmp.cleanup)

    def crop(self, center, **kwargs) -> CropTransform:
        return make_crop(self.frames_dir, FRAME_ID, center, **kwargs)


class TransformTests(CropFixture):
    def test_frame_to_crop_to_frame_is_exact(self):
        transform = self.crop((640.0, 400.0))
        for frame_point in ((600.0, 380.0), (640.5, 400.25), (700.0, 440.0)):
            crop_point = transform.to_crop(*frame_point)
            recovered = transform.to_frame(*crop_point)
            self.assertAlmostEqual(recovered[0], frame_point[0], places=9)
            self.assertAlmostEqual(recovered[1], frame_point[1], places=9)

    def test_one_crop_pixel_is_a_known_fraction_of_a_frame_pixel(self):
        """The magnification claim, checked: 16x means 0.0625 frame px per crop px."""
        transform = self.crop((640.0, 400.0), scale=16)
        a = transform.to_frame(0.0, 0.0)
        b = transform.to_frame(1.0, 0.0)
        self.assertAlmostEqual(b[0] - a[0], 0.0625, places=9)

    def test_crop_origin_lands_on_whole_frame_pixels(self):
        """A fractional origin would make the grid label a fraction of a pixel."""
        transform = self.crop((640.3, 400.7))
        self.assertEqual(transform.origin_x, int(transform.origin_x))
        self.assertEqual(transform.origin_y, int(transform.origin_y))

    def test_crop_id_is_deterministic(self):
        first = self.crop((640.0, 400.0))
        second = self.crop((640.0, 400.0))
        self.assertEqual(first.crop_id, second.crop_id)
        half = DEFAULT_SIZE_PX // 2
        self.assertEqual(
            first.crop_id,
            crop_id_for(FRAME_ID, 640 - half, 400 - half, DEFAULT_SIZE_PX, DEFAULT_SCALE),
        )

    def test_transform_round_trips_through_json(self):
        transform = self.crop((640.0, 400.0))
        restored = CropTransform.from_dict(json.loads(json.dumps(transform.as_dict())))
        self.assertEqual(restored, transform)


class WindowTests(CropFixture):
    def test_window_is_clamped_to_the_image_not_padded(self):
        """Padding would place invented pixels beside real ones at equal magnification."""
        transform = self.crop((5.0, 5.0))
        x0, y0, x1, y1 = transform.frame_bounds
        self.assertGreaterEqual(x0, 0)
        self.assertGreaterEqual(y0, 0)
        self.assertLessEqual(x1, WIDTH)
        self.assertLessEqual(y1, HEIGHT)

    def test_clamped_crop_keeps_its_full_requested_size(self):
        transform = self.crop((5.0, 5.0), size_px=128)
        self.assertEqual(transform.size_px, 128)

    def test_written_image_matches_the_declared_magnification(self):
        transform = self.crop((640.0, 400.0), size_px=64, scale=14)
        image = cv2.imread(transform.path, cv2.IMREAD_COLOR)
        self.assertEqual(image.shape[:2], (64 * 14, 64 * 14))

    def test_size_larger_than_the_image_is_reduced(self):
        transform = self.crop((640.0, 400.0), size_px=5000)
        self.assertLessEqual(transform.size_px, min(WIDTH, HEIGHT))

    def test_rejects_nonsense_geometry(self):
        for kwargs in ({"size_px": 0}, {"scale": 0}, {"scale": -2}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                self.crop((640.0, 400.0), **kwargs)

    def test_rejects_unknown_interpolation(self):
        with self.assertRaises(ValueError):
            self.crop((640.0, 400.0), interp="bicubic")

    def test_missing_frame_is_a_clear_error(self):
        with self.assertRaises(FileNotFoundError):
            make_crop(self.frames_dir, "no_such_frame", (10.0, 10.0))

    def test_grid_can_be_omitted_without_changing_the_transform(self):
        gridded = self.crop((640.0, 400.0), scale=12)
        plain = self.crop((640.0, 400.0), scale=12, grid=False)
        self.assertEqual(gridded.origin_x, plain.origin_x)
        self.assertEqual(gridded.origin_y, plain.origin_y)

    def test_grid_does_not_obliterate_the_underlying_image(self):
        """A grid dense enough to hide the paint defeats the point of magnifying it."""
        transform = self.crop((640.0, 400.0), scale=12)
        image = cv2.imread(transform.path, cv2.IMREAD_COLOR)
        magenta = np.all(image == np.array([255, 0, 255], dtype=np.uint8), axis=2)
        self.assertLess(magenta.mean(), 0.15)

    def test_gridded_crop_below_the_measuring_floor_is_refused(self):
        """The probe's dominant error came from reading a whole feature off one wide crop.

        A 5x gridded crop is what produced 9 px of smooth, plausible error on paint
        crossing floor-logo lettering, so the tool refuses to produce one. Looking at
        a wide view is still allowed, because looking is not measuring.
        """
        for scale in (1, 5, MIN_MEASURING_SCALE - 1):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                self.crop((640.0, 400.0), scale=scale)
        self.assertEqual(self.crop((640.0, 400.0), scale=5, grid=False).scale, 5)


class ResolveTests(CropFixture):
    def annotation(self, features=(), junctions=()) -> FrameAnnotation:
        return FrameAnnotation(
            frame_id=FRAME_ID,
            clip="video_9.mp4",
            frame_index=42,
            image_width=WIDTH,
            image_height=HEIGHT,
            image_sha256="b" * 64,
            annotator_id="test",
            pass_id="a",
            prompt_version="v1",
            coordinate_space="crop",
            features=features,
            junctions=junctions,
        )

    def test_resolution_maps_points_into_frame_space(self):
        transform = self.crop((640.0, 400.0))
        crops = load_crops(self.root / "crops")
        annotation = self.annotation(
            features=(
                AnnotatedFeature(
                    MarkingFeature.BASELINE,
                    (
                        SamplePoint(0.0, 0.0, transform.crop_id),
                        SamplePoint(512.0, 512.0, transform.crop_id),
                    ),
                ),
            )
        )
        resolved = resolve_annotation(annotation, crops)
        points = resolved.features[0].points
        self.assertEqual(resolved.coordinate_space, "frame")
        self.assertEqual((points[0].x, points[0].y), (transform.origin_x, transform.origin_y))
        offset = 512.0 / transform.scale
        self.assertEqual(
            (points[1].x, points[1].y),
            (transform.origin_x + offset, transform.origin_y + offset),
        )

    def test_resolution_clears_the_crop_id(self):
        """A resolved point still carrying a crop id would resolve twice."""
        transform = self.crop((640.0, 400.0))
        crops = load_crops(self.root / "crops")
        resolved = resolve_annotation(
            self.annotation(
                features=(
                    AnnotatedFeature(
                        MarkingFeature.BASELINE,
                        (
                            SamplePoint(10.0, 10.0, transform.crop_id),
                            SamplePoint(20.0, 20.0, transform.crop_id),
                        ),
                    ),
                )
            ),
            crops,
        )
        self.assertTrue(all(p.crop_id is None for p in resolved.features[0].points))

    def test_junctions_are_resolved_too(self):
        transform = self.crop((640.0, 400.0))
        crops = load_crops(self.root / "crops")
        resolved = resolve_annotation(
            self.annotation(
                junctions=(
                    AnnotatedJunction(
                        JunctionPoint.LANE_BASELINE_FAR,
                        SamplePoint(80.0, 160.0, transform.crop_id),
                    ),
                )
            ),
            crops,
        )
        point = resolved.junctions[0].point
        self.assertEqual(
            (point.x, point.y),
            (transform.origin_x + 80.0 / transform.scale, transform.origin_y + 160.0 / transform.scale),
        )

    def test_unknown_crop_refuses_rather_than_dropping_the_point(self):
        crops = load_crops(self.root / "crops")
        annotation = self.annotation(
            features=(
                AnnotatedFeature(
                    MarkingFeature.BASELINE,
                    (SamplePoint(0.0, 0.0, "never_written"), SamplePoint(1.0, 1.0, "never_written")),
                ),
            )
        )
        with self.assertRaises(UnknownCrop):
            resolve_annotation(annotation, crops)

    def test_crop_from_another_frame_is_refused(self):
        """Resolving against the wrong frame's crop lands the point somewhere else."""
        transform = self.crop((640.0, 400.0))
        foreign = CropTransform(
            crop_id=transform.crop_id,
            frame_id="video_9_000999",
            origin_x=0.0,
            origin_y=0.0,
            scale=8,
            size_px=128,
            interp="nearest",
            path=transform.path,
        )
        annotation = self.annotation(
            features=(
                AnnotatedFeature(
                    MarkingFeature.BASELINE,
                    (
                        SamplePoint(0.0, 0.0, transform.crop_id),
                        SamplePoint(8.0, 8.0, transform.crop_id),
                    ),
                ),
            )
        )
        with self.assertRaises(ValueError):
            resolve_annotation(annotation, {foreign.crop_id: foreign})

    def test_resolving_a_frame_space_annotation_is_a_no_op(self):
        annotation = self.annotation()
        already = FrameAnnotation.from_dict({**annotation.as_dict(), "coordinate_space": "frame"})
        self.assertIs(resolve_annotation(already, {}), already)

    def test_load_crops_finds_everything_written(self):
        first = self.crop((300.0, 300.0))
        second = self.crop((900.0, 500.0))
        crops = load_crops(self.root / "crops")
        self.assertEqual(set(crops), {first.crop_id, second.crop_id})

    def test_load_crops_on_a_missing_directory_is_empty(self):
        self.assertEqual(load_crops(self.root / "nothing_here"), {})


if __name__ == "__main__":
    unittest.main()
