"""Pooling must combine pair confidence rather than pick a winner.

This is the layer that makes an unreliable detector usable. The failure it exists
to prevent is concrete: two slots that fire on the same corner and split their
confidence 0.40/0.58 both fall below a 0.25 gate individually in the worst case,
so the landmark the model is *most* certain about gets discarded. Pooling
recovers it.

The other half of the job is refusal. Slots the review could not settle carry no
landmark id, and no amount of confidence should conjure one.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from contracts.calibration_types import RawKeypointFrame, RawKeypointObservation
from calibration.adapters.evidence import (
    EvidenceGroup,
    EvidenceMap,
    load_evidence_map,
    load_registered_map,
    pool_evidence,
)


def _frame(points: dict[int, tuple[float, float, float]], detected=True) -> RawKeypointFrame:
    return RawKeypointFrame(
        frame_index=0,
        timestamp_s=0.0,
        source_id="clip.mp4",
        image_width=1280,
        image_height=720,
        detection_state="detected" if detected else "no_detection",
        instance_count=1 if detected else 0,
        instance_confidences=(0.9,) if detected else (),
        selected_instance=0 if detected else None,
        observations=tuple(
            RawKeypointObservation(i, x, y, c) for i, (x, y, c) in sorted(points.items())
        ) if detected else (),
    )


def _map(groups=None, rejected=None, keypoint_count=18) -> EvidenceMap:
    return EvidenceMap(
        map_id="test",
        status="provisional",
        keypoint_count=keypoint_count,
        weights_sha256="0" * 64,
        groups=tuple(groups or [EvidenceGroup("lane_baseline_far", (2, 13), "confident")]),
        rejected=tuple(rejected or []),
    )


class PoolingTests(unittest.TestCase):
    def test_split_confidence_is_summed_not_maxed(self):
        """The core fix: a 0.40/0.58 split is one strong landmark, not two weak ones."""
        frame = _frame({2: (100.0, 200.0, 0.40), 13: (104.0, 203.0, 0.58)})
        evidence, _ = pool_evidence(frame, _map(), min_confidence=0.25)
        self.assertEqual(len(evidence), 1)
        self.assertAlmostEqual(evidence[0].confidence, 0.98, places=6)

    def test_split_that_would_fail_per_slot_survives_pooling(self):
        """Neither slot clears 0.5 alone; together they clearly do."""
        frame = _frame({2: (100.0, 200.0, 0.26), 13: (103.0, 202.0, 0.30)})
        evidence, _ = pool_evidence(frame, _map(), min_confidence=0.5)
        self.assertEqual(len(evidence), 1)
        self.assertAlmostEqual(evidence[0].confidence, 0.56, places=6)

    def test_position_is_confidence_weighted(self):
        """Separation stays inside the pair-agreement guard, as it is in real data."""
        frame = _frame({2: (100.0, 200.0, 0.25), 13: (120.0, 200.0, 0.75)})
        evidence, _ = pool_evidence(frame, _map())
        self.assertAlmostEqual(evidence[0].x, 115.0, places=6)

    def test_confidence_is_capped_at_one(self):
        frame = _frame({2: (100.0, 200.0, 0.9), 13: (101.0, 201.0, 0.9)})
        evidence, _ = pool_evidence(frame, _map())
        self.assertEqual(evidence[0].confidence, 1.0)

    def test_both_contributing_slots_are_recorded(self):
        frame = _frame({2: (100.0, 200.0, 0.4), 13: (104.0, 203.0, 0.5)})
        evidence, _ = pool_evidence(frame, _map())
        self.assertEqual(evidence[0].source_slots, (2, 13))

    def test_single_slot_group_works(self):
        frame = _frame({7: (500.0, 300.0, 0.9)})
        evidence, _ = pool_evidence(
            frame, _map([EvidenceGroup("midcourt_sideline_far", (7,), "confident")])
        )
        self.assertEqual(evidence[0].source_slots, (7,))

    def test_negligible_slot_does_not_drag_the_position(self):
        """A near-zero vote must not move a strong landmark."""
        frame = _frame({2: (100.0, 200.0, 0.95), 13: (900.0, 700.0, 0.001)})
        evidence, _ = pool_evidence(frame, _map())
        self.assertAlmostEqual(evidence[0].x, 100.0, places=3)


class RefusalTests(unittest.TestCase):
    def test_unmapped_slots_are_dropped_with_a_reason(self):
        frame = _frame({i: (10.0 * i, 20.0, 0.9) for i in range(18)})
        _, dropped = pool_evidence(frame, _map())
        reasons = dict(dropped)
        self.assertIn("slot_3", reasons)
        self.assertIn("slot_12", reasons)

    def test_rejected_slots_report_their_review_reason(self):
        frame = _frame({i: (10.0 * i, 20.0, 0.9) for i in range(18)})
        _, dropped = pool_evidence(
            frame, _map(rejected=[{"slots": [3, 12], "reason": "ambiguous"}])
        )
        self.assertEqual(dict(dropped)["slot_3"], "ambiguous")

    def test_high_confidence_cannot_conjure_an_unmapped_landmark(self):
        """Slot 3/12 at full confidence still contributes nothing."""
        frame = _frame({3: (400.0, 400.0, 1.0), 12: (402.0, 401.0, 1.0)})
        evidence, _ = pool_evidence(frame, _map())
        self.assertEqual(evidence, ())

    def test_tier_below_policy_is_excluded(self):
        frame = _frame({7: (500.0, 300.0, 0.99)})
        evidence, dropped = pool_evidence(
            frame,
            _map([EvidenceGroup("midcourt_sideline_far", (7,), "probable")]),
            tiers=("confident",),
        )
        self.assertEqual(evidence, ())
        self.assertIn("tier_excluded:probable", dict(dropped).values())

    def test_tier_included_when_policy_allows(self):
        frame = _frame({7: (500.0, 300.0, 0.99)})
        evidence, _ = pool_evidence(
            frame,
            _map([EvidenceGroup("midcourt_sideline_far", (7,), "probable")]),
            tiers=("confident", "probable"),
        )
        self.assertEqual(len(evidence), 1)

    def test_pair_disagreement_drops_rather_than_averages(self):
        """If the pair's premise breaks, averaging invents a point neither predicted."""
        frame = _frame({2: (100.0, 200.0, 0.5), 13: (900.0, 600.0, 0.5)})
        evidence, dropped = pool_evidence(frame, _map(), max_pair_spread_px=40.0)
        self.assertEqual(evidence, ())
        self.assertTrue(any("pair_disagreement" in r for _, r in dropped))

    def test_below_confidence_is_dropped_with_the_value(self):
        frame = _frame({2: (100.0, 200.0, 0.05), 13: (101.0, 201.0, 0.05)})
        evidence, dropped = pool_evidence(frame, _map(), min_confidence=0.25)
        self.assertEqual(evidence, ())
        self.assertTrue(any("below_confidence" in r for _, r in dropped))

    def test_no_detection_frame_yields_nothing(self):
        evidence, dropped = pool_evidence(_frame({}, detected=False), _map())
        self.assertEqual(evidence, ())
        self.assertEqual(dropped, (("frame", "no_detection"),))


class MapLoadingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _write(self, **overrides) -> Path:
        data = {
            "map_id": "m",
            "schema_version": "evidence-map-1.0.0",
            "status": "provisional",
            "is_verified_map": False,
            "keypoint_count": 18,
            "groups": [{"landmark_id": "lane_baseline_far", "slots": [2, 13]}],
        }
        data.update(overrides)
        path = self.tmp / "m.json"
        path.write_text(json.dumps(data))
        return path

    def test_loads_a_provisional_map(self):
        evidence_map = load_evidence_map(self._write())
        self.assertEqual(evidence_map.status, "provisional")
        self.assertEqual(evidence_map.groups[0].slots, (2, 13))

    def test_refuses_a_map_claiming_to_be_verified(self):
        """No map may assert settled semantics before human sign-off."""
        with self.assertRaises(ValueError) as caught:
            load_evidence_map(self._write(is_verified_map=True))
        self.assertIn("verified", str(caught.exception))

    def test_refuses_a_non_provisional_status(self):
        with self.assertRaises(ValueError):
            load_evidence_map(self._write(status="approved"))

    def test_refuses_a_slot_claimed_by_two_landmarks(self):
        with self.assertRaises(ValueError):
            load_evidence_map(
                self._write(
                    groups=[
                        {"landmark_id": "lane_baseline_far", "slots": [2, 13]},
                        {"landmark_id": "baseline_sideline_far", "slots": [13, 0]},
                    ]
                )
            )

    def test_refuses_an_unknown_schema(self):
        with self.assertRaises(ValueError):
            load_evidence_map(self._write(schema_version="evidence-map-9.9.9"))


class ShippedMapTests(unittest.TestCase):
    """The map actually shipped must match the first-pass review's conclusions."""

    def setUp(self):
        self.map = load_registered_map("reloc2_18_provisional")

    def test_is_provisional(self):
        self.assertEqual(self.map.status, "provisional")

    def test_maps_the_five_confident_pairs(self):
        confident = {g.landmark_id: g.slots for g in self.map.groups_for_tiers(("confident",))}
        self.assertEqual(
            confident,
            {
                "baseline_sideline_far": (0, 15),
                "three_point_baseline_far": (1, 14),
                "lane_baseline_far": (2, 13),
                "lane_free_throw_far": (8, 16),
                "lane_free_throw_near": (9, 17),
            },
        )

    def test_ambiguous_and_unusable_slots_are_unmapped(self):
        for slot in (3, 4, 5, 10, 11, 12):
            self.assertNotIn(slot, self.map.mapped_slots, f"slot {slot} must stay unmapped")

    def test_midcourt_slots_are_only_probable(self):
        """They have no support in video_2, so they are not default-fit evidence."""
        probable = {g.landmark_id for g in self.map.groups_for_tiers(("probable",))}
        self.assertEqual(probable, {"midcourt_sideline_far", "midcourt_sideline_near"})

    def test_every_landmark_id_exists_in_the_layout_vocabulary(self):
        from contracts.court_layout import HalfCourtLandmark

        for group in self.map.groups:
            HalfCourtLandmark(group.landmark_id)  # raises if unknown


if __name__ == "__main__":
    unittest.main()
