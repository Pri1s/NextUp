"""Records, sink rolling, and resume-key behaviour."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from engine.detect.players import Detection
from engine.pipeline.records import (
    FrameRecord,
    RecordSink,
    ball_dict,
    court_dict,
    load_run_manifest,
    new_run_manifest,
    player_dict,
    resume_key,
    save_run_manifest,
)


class TestRecordBuilders(unittest.TestCase):
    def test_player_dict_rounding_and_shape(self):
        det = Detection(xyxy=(10.111, 20.222, 30.333, 40.444), cls=0, name="Player", conf=0.83456, track_id=7)
        row = player_dict(det, court_ft=(12.3456, 7.6543))
        self.assertEqual(row["track_id"], 7)
        self.assertEqual(row["name"], "Player")
        self.assertEqual(row["conf"], 0.8346)
        self.assertEqual(row["bbox"], [10.1, 20.2, 30.3, 40.4])
        self.assertEqual(row["foot_px"], [20.2, 40.4])  # bottom-center
        self.assertEqual(row["court_ft"], [12.35, 7.65])

    def test_player_dict_no_homography(self):
        det = Detection(xyxy=(0, 0, 10, 10), cls=0, name="Player", conf=0.5)
        row = player_dict(det, court_ft=None)
        self.assertIsNone(row["court_ft"])
        self.assertIsNone(row["track_id"])

    def test_ball_dict_center(self):
        det = Detection(xyxy=(0, 0, 10, 20), cls=1, name="Ball", conf=0.6)
        row = ball_dict(det, court_ft=None)
        self.assertEqual(row["center_px"], [5.0, 10.0])
        self.assertIsNone(row["court_ft"])

    def test_court_dict_homography_serialises(self):
        H = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]])
        court = court_dict(11, H, ["nw_corner", "ne_corner"], keypoints=None)
        self.assertEqual(court["visible_keypoints"], 11)
        self.assertEqual(court["homography"][0], [1.0, 0.0, 2.0])
        self.assertEqual(court["used_keypoint_ids"], ["nw_corner", "ne_corner"])
        self.assertIsNone(court["keypoints"])

    def test_frame_record_json_round_trip(self):
        rec = FrameRecord(
            frame_index=1830,
            timestamp_s=61.0,
            shot_id=4,
            scene_cut=True,
            court=court_dict(0, None, []),
            players=[player_dict(Detection((0, 0, 1, 1), 0, "Player", 0.5, track_id=1), None)],
            ball=None,
        )
        blob = json.dumps(rec.to_dict())
        back = json.loads(blob)
        self.assertEqual(back["frame_index"], 1830)
        self.assertEqual(back["shot_id"], 4)
        self.assertTrue(back["scene_cut"])
        self.assertEqual(back["players"][0]["track_id"], 1)
        self.assertIsNone(back["ball"])
        self.assertIsNone(back["error"])


class TestRecordSink(unittest.TestCase):
    def _record(self, i):
        return FrameRecord(i, float(i), 0, False, court_dict(0, None, []))

    def test_writes_and_rolls_into_separate_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            sink = RecordSink(Path(tmp))
            for i in range(3):
                sink.write(self._record(i))
            self.assertEqual(sink.n_in_segment, 3)
            first_path = sink.path
            sink.roll()
            for i in range(3, 5):
                sink.write(self._record(i))
            self.assertEqual(sink.n_in_segment, 2)
            sink.close()

            shards = sorted(Path(tmp).glob("seg_*.jsonl"))
            self.assertEqual([p.name for p in shards], ["seg_00000.jsonl", "seg_00001.jsonl"])
            self.assertEqual(first_path.name, "seg_00000.jsonl")
            self.assertEqual(len(first_path.read_text().strip().splitlines()), 3)
            self.assertEqual(len(shards[1].read_text().strip().splitlines()), 2)

    def test_reopen_overwrites_partial_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            sink = RecordSink(Path(tmp))
            sink.write(self._record(0))
            sink.write(self._record(1))
            sink.close()
            # Re-run the same (un-checkpointed) segment: should overwrite, not append.
            sink2 = RecordSink(Path(tmp), start_segment=0)
            sink2.write(self._record(0))
            sink2.close()
            lines = (Path(tmp) / "seg_00000.jsonl").read_text().strip().splitlines()
            self.assertEqual(len(lines), 1)


class TestResumeKeyAndManifest(unittest.TestCase):
    def test_resume_key_changes_with_config(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            f.write(b"not a real video")
            f.flush()
            k1 = resume_key(f.name, {"fps": 5})
            k2 = resume_key(f.name, {"fps": 10})
            self.assertNotEqual(k1, k2)
            self.assertEqual(k1, resume_key(f.name, {"fps": 5}))

    def test_manifest_atomic_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "clip.nba"
            manifest = new_run_manifest("clip.mp4", "nba", {"fps": 30}, {"stride": 6}, key="abc")
            save_run_manifest(run_dir, manifest)
            back = load_run_manifest(run_dir)
            self.assertEqual(back["resume_key"], "abc")
            self.assertEqual(back["status"], "running")
            self.assertEqual(back["last_shot_id"], -1)


if __name__ == "__main__":
    unittest.main()
