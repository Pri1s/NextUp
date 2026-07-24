"""End-to-end pipeline over a synthetic clip with injected fake stages.

Exercises the full run_pipeline loop — shot segmentation, tracking, projection,
sharding, resume, and per-frame error isolation — without real weights or a
``supervision`` install (the stages are fakes).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from contracts.court_template import load_template
from engine.detect.players import Detection
from engine.pipeline.records import run_dir_for
from engine.pipeline.runner import PipelineConfig, Stages, run_pipeline
from engine.tests._synthetic import make_clip


# --------------------------------------------------------------------------- #
# Fakes.
# --------------------------------------------------------------------------- #

class FakeCourtModel:
    """Emits n_visible canonical points on a non-degenerate grid (homography solvable)."""

    def __init__(self, n_visible: int = 8, fail_on_calls=None):
        self._n = n_visible
        self._fail = set(fail_on_calls or [])
        self.calls = 0

    def predict(self, frame):
        self.calls += 1
        if self.calls in self._fail:
            raise RuntimeError("boom")
        arr = np.zeros((22, 3), dtype=float)
        for i in range(self._n):
            arr[i] = [10 + (i % 4) * 20, 10 + (i // 4) * 20, 2]
        return arr


class FakePlayerDetector:
    def __init__(self, n_players: int = 2):
        self._n = n_players

    def _dets(self):
        return [
            Detection((10 + i * 20, 10, 20 + i * 20, 60), cls=0, name="Player", conf=0.9)
            for i in range(self._n)
        ]

    def detect(self, frame):
        return self._dets()

    def detect_batch(self, frames):
        return [self._dets() for _ in frames]


class FakeBallDetector:
    """Alternates a credible ball with a below-threshold one, so some records are null."""

    def __init__(self):
        self.calls = 0

    def _det(self):
        self.calls += 1
        conf = 0.6 if self.calls % 2 == 1 else 0.1
        return [Detection((5, 5, 15, 15), cls=0, name="Ball", conf=conf)]

    def detect(self, frame):
        return self._det()

    def detect_batch(self, frames):
        return [self._det() for _ in frames]


class FakeTracker:
    """Positional, stable ids within a shot; counts resets."""

    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1

    def update(self, dets):
        return [replace(d, track_id=i) for i, d in enumerate(dets)]


def _stages(**overrides) -> Stages:
    return Stages(
        court_model=overrides.get("court_model", FakeCourtModel()),
        template=load_template("nba_94x50"),
        tracker=overrides.get("tracker", FakeTracker()),
        player_detector=overrides.get("player_detector", FakePlayerDetector()),
        ball_detector=overrides.get("ball_detector", FakeBallDetector()),
    )


def _read_records(run_dir: Path) -> list[dict]:
    recs = []
    for shard in sorted((run_dir / "records").glob("seg_*.jsonl")):
        for line in shard.read_text().strip().splitlines():
            if line:
                recs.append(json.loads(line))
    return recs


# High target_fps forces stride==1 regardless of the clip's exact reported fps.
FAST = dict(target_fps=1000.0)


# --------------------------------------------------------------------------- #
# Tests.
# --------------------------------------------------------------------------- #

class TestPipelineE2E(unittest.TestCase):
    def test_full_run_shots_tracks_and_projection(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            clip = make_clip(tmp / "clip.avi", n_frames=12, cut_at=6)
            config = PipelineConfig(segment_frames=1000, batch=4, **FAST)
            manifest = run_pipeline(clip, "nba", config, out_dir=tmp / "out", stages=_stages())

            self.assertEqual(manifest["status"], "done")
            run_dir = run_dir_for(tmp / "out", "clip", "nba")
            recs = _read_records(run_dir)

            # Every processed frame recorded, native indices 0..11 (stride 1).
            self.assertEqual([r["frame_index"] for r in recs], list(range(12)))
            self.assertEqual(manifest["summary"]["frames_processed"], 12)

            # Shot segmentation: hard cut at frame 6 opens shot 1.
            self.assertTrue(recs[0]["scene_cut"])
            self.assertEqual([r["shot_id"] for r in recs[:6]], [0] * 6)
            self.assertEqual([r["shot_id"] for r in recs[6:]], [1] * 6)
            self.assertTrue(recs[6]["scene_cut"])
            self.assertFalse(recs[5]["scene_cut"])

            # Projection worked: homography + court_ft populated for every player.
            self.assertEqual(manifest["summary"]["frames_with_homography"], 12)
            for r in recs:
                self.assertIsNotNone(r["court"]["homography"])
                self.assertEqual(len(r["players"]), 2)
                for p in r["players"]:
                    self.assertIsNotNone(p["court_ft"])
                    self.assertIsNotNone(p["track_id"])

            # (shot_id, track_id) is the join key: 2 players x 2 shots = 4 unique.
            self.assertEqual(manifest["summary"]["unique_tracks"], 4)

            # Ball confidence floor: some credible, some null.
            balls = [r["ball"] for r in recs]
            self.assertTrue(any(b is not None for b in balls))
            self.assertTrue(any(b is None for b in balls))

    def test_resume_appends_without_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            out = tmp / "out"
            clip = make_clip(tmp / "clip.avi", n_frames=12, cut_at=6)

            # Run A: stop after 6 frames, small segments -> two done shards (0..5).
            cfg_a = PipelineConfig(segment_frames=3, batch=4, max_frames=6, **FAST)
            run_pipeline(clip, "nba", cfg_a, out_dir=out, stages=_stages())

            # Run B: same config key (max_frames isn't part of it) -> resume to the end.
            cfg_b = PipelineConfig(segment_frames=3, batch=4, **FAST)
            manifest = run_pipeline(clip, "nba", cfg_b, out_dir=out, stages=_stages())

            run_dir = run_dir_for(out, "clip", "nba")
            recs = _read_records(run_dir)
            indices = [r["frame_index"] for r in recs]
            self.assertEqual(sorted(indices), list(range(12)))
            self.assertEqual(len(indices), len(set(indices)))  # no duplicates

            # Resume boundary begins a fresh shot; frame 6 is that boundary here.
            by_index = {r["frame_index"]: r for r in recs}
            self.assertTrue(by_index[6]["scene_cut"])
            self.assertEqual(by_index[6]["shot_id"], 1)
            self.assertEqual(manifest["status"], "done")

    def test_bad_frame_is_isolated_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            clip = make_clip(tmp / "clip.avi", n_frames=6, fps=30)
            court = FakeCourtModel(fail_on_calls={3})  # 3rd processed frame throws
            config = PipelineConfig(segment_frames=1000, batch=2, **FAST)
            manifest = run_pipeline(
                clip, "nba", config, out_dir=tmp / "out", stages=_stages(court_model=court)
            )

            self.assertEqual(manifest["status"], "done")
            self.assertEqual(manifest["summary"]["errors"], 1)
            recs = _read_records(run_dir_for(tmp / "out", "clip", "nba"))
            self.assertEqual(len(recs), 6)
            failed = recs[2]
            self.assertIsNotNone(failed["error"])
            self.assertEqual(failed["players"], [])
            self.assertTrue(all(r["error"] is None for i, r in enumerate(recs) if i != 2))

    def test_class_filtering_keeps_only_named_classes(self):
        class MixedDetector:
            def _dets(self):
                return [
                    Detection((0, 0, 10, 60), cls=4, name="Player", conf=0.9),
                    Detection((20, 0, 30, 60), cls=4, name="Player", conf=0.8),
                    Detection((40, 0, 50, 10), cls=6, name="Scoreboard", conf=0.95),
                    Detection((60, 0, 70, 10), cls=2, name="Hoop", conf=0.7),
                ]

            def detect(self, frame):
                return self._dets()

            def detect_batch(self, frames):
                return [self._dets() for _ in frames]

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            clip = make_clip(tmp / "clip.avi", n_frames=4, fps=30)
            stages = Stages(
                court_model=FakeCourtModel(),
                template=load_template("nba_94x50"),
                tracker=FakeTracker(),
                player_detector=MixedDetector(),
                ball_detector=None,
                player_classes=("Player",),
            )
            config = PipelineConfig(segment_frames=1000, batch=2, **FAST)
            run_pipeline(clip, "nba", config, out_dir=tmp / "out", stages=stages)
            recs = _read_records(run_dir_for(tmp / "out", "clip", "nba"))
            for r in recs:
                self.assertEqual(len(r["players"]), 2)  # Scoreboard + Hoop dropped
                self.assertTrue(all(p["name"] == "Player" for p in r["players"]))

    def test_consecutive_errors_abort_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            clip = make_clip(tmp / "clip.avi", n_frames=8, fps=30)
            court = FakeCourtModel(fail_on_calls={1, 2, 3, 4, 5, 6, 7, 8})  # every frame throws
            config = PipelineConfig(segment_frames=1000, batch=4, max_consecutive_errors=2, **FAST)
            manifest = run_pipeline(
                clip, "nba", config, out_dir=tmp / "out", stages=_stages(court_model=court)
            )
            self.assertEqual(manifest["status"], "aborted")
            self.assertEqual(manifest["summary"]["frames_processed"], 2)


if __name__ == "__main__":
    unittest.main()
