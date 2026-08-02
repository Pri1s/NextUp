import inspect
import unittest

import numpy as np

from calibration.viz import court_markings, draw_reprojection
from contracts.calibration_types import CalibrationStatus, CourtCalibration
from contracts.court_layout import build_layout
from contracts.markings import MarkingFeature


def layout(radius=23.75):
    return build_layout(
        layout_id=f"test_{radius}", rule_set="test", source="test",
        half_length=47.0, width=50.0, lane_width=16.0,
        free_throw_distance=19.0, basket_from_baseline=5.25,
        three_point_corner_inset=3.0, three_point_radius=radius,
        free_throw_circle_radius=6.0, center_circle_radius=6.0,
        restricted_area_radius=4.0, line_width=2 / 12,
    )


def calibration():
    h = ((10.0, 0.0, 20.0), (0.0, 10.0, 20.0), (0.0, 0.0, 1.0))
    inv = ((0.1, 0.0, -2.0), (0.0, 0.1, -2.0), (0.0, 0.0, 1.0))
    return CourtCalibration(
        frame_index=1, layout_id="test", status=CalibrationStatus.DEGRADED,
        h_court_to_image=h, h_image_to_court=inv,
    )


class GeometryAdapterTests(unittest.TestCase):
    def test_samples_are_layout_owned(self):
        court = layout()
        sampled = court_markings(court)
        self.assertEqual(set(sampled), set(MarkingFeature))
        for feature in MarkingFeature:
            self.assertEqual(sampled[feature], court.sample_marking(feature))

    def test_no_radius_override_parameters_remain(self):
        self.assertEqual(list(inspect.signature(court_markings).parameters), ["layout"])

    def test_closed_curve_behavior_comes_from_layout(self):
        court = layout()
        closed = [feature for feature in MarkingFeature if court.marking(feature).closed]
        self.assertEqual(closed, [MarkingFeature.CENTER_CIRCLE])


class RenderTests(unittest.TestCase):
    def test_custom_radius_moves_rendered_three_point_marking(self):
        frame = np.zeros((600, 600, 3), dtype=np.uint8)
        standard = draw_reprojection(frame, calibration(), layout(23.75))
        custom = draw_reprojection(frame, calibration(), layout(23.5))
        self.assertTrue(np.any(standard != custom))

    def test_input_frame_is_not_mutated(self):
        frame = np.zeros((600, 600, 3), dtype=np.uint8)
        before = frame.copy()
        draw_reprojection(frame, calibration(), layout())
        np.testing.assert_array_equal(frame, before)


if __name__ == "__main__":
    unittest.main()
