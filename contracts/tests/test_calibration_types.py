"""The raw detection contract must not quietly lose or invent information.

Phase 0 exists to judge slot semantics from model output, so the contract's job
is to carry that output through unaltered: continuous confidence, real predicted
coordinates, explicit absence. Every test here guards one of those properties.
"""

import dataclasses
import unittest

from contracts.calibration_types import (
    ModelProvenance,
    RawKeypointFrame,
    RawKeypointObservation,
)

DIGEST = "f6263105e5c2338fafcfd5a6fefd7d1d441e87364635e918dfdbb849f2df1377"


def _frame(**overrides):
    defaults = dict(
        frame_index=0,
        timestamp_s=0.0,
        source_id="clip.mp4",
        image_width=1280,
        image_height=720,
        detection_state="detected",
        instance_count=1,
        instance_confidences=(0.94,),
        selected_instance=0,
        observations=(RawKeypointObservation(0, 440.7, 246.7, 0.838),),
    )
    defaults.update(overrides)
    return RawKeypointFrame(**defaults)


class ObservationTests(unittest.TestCase):
    def test_confidence_stays_continuous(self):
        """No rounding, bucketing, or collapse into a visibility flag."""
        for value in (0.0, 0.0234567, 0.24999, 0.5, 0.999999, 1.0):
            self.assertEqual(RawKeypointObservation(3, 1.0, 2.0, value).confidence, value)

    def test_confidence_out_of_range_rejected(self):
        for value in (-0.001, 1.001, 2.0):
            with self.assertRaises(ValueError):
                RawKeypointObservation(0, 1.0, 2.0, value)

    def test_negative_slot_rejected(self):
        with self.assertRaises(ValueError):
            RawKeypointObservation(-1, 1.0, 2.0, 0.5)

    def test_nan_coordinate_rejected(self):
        with self.assertRaises(ValueError):
            RawKeypointObservation(0, float("nan"), 2.0, 0.5)

    def test_immutable(self):
        observation = RawKeypointObservation(0, 1.0, 2.0, 0.5)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            observation.confidence = 0.9

    def test_round_trip_preserves_full_precision(self):
        original = RawKeypointObservation(17, 671.4321098, 472.0987654, 0.9853217, clamped=True)
        self.assertEqual(RawKeypointObservation.from_dict(original.as_dict()), original)

    def test_low_confidence_point_keeps_its_real_coordinate(self):
        """A below-gate slot is not zeroed — the (0, 0) sentinel is banned."""
        observation = RawKeypointObservation(6, 1232.6, 654.3, 0.024)
        self.assertEqual((observation.x, observation.y), (1232.6, 654.3))


class FrameTests(unittest.TestCase):
    def test_no_detection_must_be_empty(self):
        with self.assertRaises(ValueError):
            _frame(detection_state="no_detection", instance_count=0, selected_instance=None)

    def test_no_detection_frame_is_valid_when_empty(self):
        frame = _frame(
            detection_state="no_detection",
            instance_count=0,
            instance_confidences=(),
            selected_instance=None,
            observations=(),
        )
        self.assertFalse(frame.detected)
        self.assertEqual(frame.observations, ())

    def test_no_detection_cannot_select_an_instance(self):
        with self.assertRaises(ValueError):
            _frame(
                detection_state="no_detection",
                instance_count=0,
                instance_confidences=(),
                selected_instance=0,
                observations=(),
            )

    def test_duplicate_slot_rejected(self):
        with self.assertRaises(ValueError):
            _frame(
                observations=(
                    RawKeypointObservation(4, 1.0, 2.0, 0.5),
                    RawKeypointObservation(4, 3.0, 4.0, 0.6),
                )
            )

    def test_unknown_detection_state_rejected(self):
        with self.assertRaises(ValueError):
            _frame(detection_state="maybe")

    def test_selected_instance_must_be_in_range(self):
        with self.assertRaises(ValueError):
            _frame(instance_count=2, instance_confidences=(0.9, 0.8), selected_instance=5)

    def test_multi_instance_confidences_are_all_retained(self):
        """The reference pipeline assumed instance 0; we record the whole picture."""
        frame = _frame(instance_count=3, instance_confidences=(0.4, 0.94, 0.7), selected_instance=1)
        self.assertEqual(frame.instance_confidences, (0.4, 0.94, 0.7))
        self.assertEqual(frame.selected_instance, 1)

    def test_bad_image_size_rejected(self):
        with self.assertRaises(ValueError):
            _frame(image_width=0)

    def test_lookup_by_slot(self):
        frame = _frame()
        self.assertIsNotNone(frame.observation(0))
        self.assertIsNone(frame.observation(9))

    def test_round_trip(self):
        frame = _frame(
            observations=tuple(
                RawKeypointObservation(i, float(i) * 1.5, float(i) * 2.5, i / 20.0)
                for i in range(18)
            )
        )
        self.assertEqual(RawKeypointFrame.from_dict(frame.as_dict()), frame)


class ProvenanceTests(unittest.TestCase):
    def test_digest_must_be_sha256_shaped(self):
        with self.assertRaises(ValueError):
            ModelProvenance("m.pt", "abc123", 18, "pose")

    def test_keypoint_count_must_be_positive(self):
        with self.assertRaises(ValueError):
            ModelProvenance("m.pt", DIGEST, 0, "pose")

    def test_as_dict_stringifies_class_keys_for_json(self):
        provenance = ModelProvenance("m.pt", DIGEST, 18, "pose", class_names={0: "basketball"})
        self.assertEqual(provenance.as_dict()["class_names"], {"0": "basketball"})


if __name__ == "__main__":
    unittest.main()
