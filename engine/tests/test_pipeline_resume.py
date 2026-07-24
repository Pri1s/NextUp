"""Resume math: native-frame anchors and the iter_frames start_index gate."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from engine.io.video import iter_frames
from engine.pipeline.records import resume_point
from engine.tests._synthetic import make_clip


class TestResumePoint(unittest.TestCase):
    def test_no_manifest_starts_at_zero(self):
        rp = resume_point(None)
        self.assertEqual((rp.start_index, rp.start_segment, rp.start_shot), (0, 0, 0))

    def test_no_done_segments_starts_at_zero(self):
        rp = resume_point({"segments": [], "last_shot_id": -1})
        self.assertEqual((rp.start_index, rp.start_segment, rp.start_shot), (0, 0, 0))

    def test_resumes_after_last_done_segment_in_native_units(self):
        manifest = {
            "last_shot_id": 3,
            "segments": [
                {"index": 0, "start_frame": 0, "end_frame": 45, "status": "done"},
                {"index": 1, "start_frame": 60, "end_frame": 105, "status": "done"},
            ],
        }
        rp = resume_point(manifest)
        # Next native frame after 105; next shard index; a fresh shot after the seam.
        self.assertEqual(rp.start_index, 106)
        self.assertEqual(rp.start_segment, 2)
        self.assertEqual(rp.start_shot, 4)

    def test_ignores_unfinalized_segments(self):
        manifest = {
            "last_shot_id": 0,
            "segments": [
                {"index": 0, "start_frame": 0, "end_frame": 45, "status": "done"},
                {"index": 1, "start_frame": 60, "end_frame": 90},  # not done
            ],
        }
        rp = resume_point(manifest)
        self.assertEqual(rp.start_index, 46)
        self.assertEqual(rp.start_segment, 1)


class TestIterFramesStartIndex(unittest.TestCase):
    def test_start_index_is_native_and_stride_aligned(self):
        with tempfile.TemporaryDirectory() as tmp:
            clip = make_clip(Path(tmp) / "c.avi", n_frames=20, fps=30)
            got = [idx for idx, _ts, _f in iter_frames(clip, stride=3, start_index=6)]
            self.assertEqual(got, [6, 9, 12, 15, 18])

    def test_start_index_zero_matches_plain(self):
        with tempfile.TemporaryDirectory() as tmp:
            clip = make_clip(Path(tmp) / "c.avi", n_frames=10, fps=30)
            a = [idx for idx, _t, _f in iter_frames(clip, stride=2)]
            b = [idx for idx, _t, _f in iter_frames(clip, stride=2, start_index=0)]
            self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
