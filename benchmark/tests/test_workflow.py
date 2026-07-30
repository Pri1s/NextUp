"""The annotation workflow must start clean and record runtime-only choices."""

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from benchmark.cli import build_parser, command_status
from benchmark.frames import BenchmarkManifest, ExportedFrame, write_export_index
from benchmark.workflow import initialize_run


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.frames_dir = self.root / "frames"
        self.frames_dir.mkdir()
        frames = [
            ExportedFrame(
                frame_id=f"video_{clip}_00000{difficulty}",
                clip=f"input_videos/video_{clip}.mp4",
                frame_index=difficulty,
                stratum="clean" if difficulty == 1 else "occluded",
                pilot=True,
                path=f"images/video_{clip}_00000{difficulty}.png",
                sha256=f"{clip}{difficulty}" * 32,
                width=1280,
                height=720,
            )
            for clip in range(1, 4)
            for difficulty in range(1, 3)
        ]
        manifest = BenchmarkManifest("test_m1", "nba_halfcourt", "", ())
        write_export_index(
            self.frames_dir,
            manifest,
            frames,
            "# glossary\n",
            "glossary-test",
        )

    def test_initialization_records_models_and_separates_passes(self):
        run_dir = self.root / "run"
        destination = initialize_run(
            run_dir,
            run_id="m1-openai-v1",
            frames_dir=self.frames_dir,
            pass_a_model="openai-model-a-exact",
            pass_b_model="openai-model-b-exact",
            adjudicator_model="openai-model-c-exact",
            reasoning_effort="high",
            image_detail="original",
        )
        payload = json.loads(destination.read_text(encoding="utf-8"))
        self.assertEqual(payload["provider"], "openai")
        self.assertEqual(
            payload["runtime_settings"],
            {"reasoning_effort": "high", "image_detail": "original"},
        )
        self.assertEqual(payload["frame_count"], 6)
        self.assertEqual(
            [entry["model_id"] for entry in payload["passes"]],
            ["openai-model-a-exact", "openai-model-b-exact"],
        )
        self.assertNotEqual(
            payload["passes"][0]["output_dir"],
            payload["passes"][1]["output_dir"],
        )
        self.assertEqual(
            payload["adjudication"]["start_condition"],
            "nonempty_consensus/adjudication_queue.json",
        )

    def test_nonempty_run_directory_is_refused(self):
        run_dir = self.root / "run"
        run_dir.mkdir()
        (run_dir / "old-label.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            initialize_run(
                run_dir,
                run_id="m1-openai-v1",
                frames_dir=self.frames_dir,
                pass_a_model="a",
                pass_b_model="b",
                adjudicator_model="c",
            )

    def test_status_fails_when_any_expected_annotation_is_missing(self):
        pass_dir = self.root / "pass_a"
        pass_dir.mkdir()
        args = argparse.Namespace(
            frames=str(self.frames_dir),
            pass_dir=str(pass_dir),
            expect_prompt_version="annotator-2.0.0",
            expect_annotator_id="runtime-model",
        )
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(command_status(args), 1)

    def test_score_accepts_one_recorded_run_per_clip_and_legacy_silver_alias(self):
        args = build_parser().parse_args(
            [
                "score",
                "--silver",
                "silver",
                "--run",
                "video_1/m1_confident",
                "--run",
                "video_2/m1_confident",
                "--run",
                "video_3/m1_confident",
            ]
        )
        self.assertEqual(args.reference, "silver")
        self.assertEqual(
            args.run,
            [
                "video_1/m1_confident",
                "video_2/m1_confident",
                "video_3/m1_confident",
            ],
        )

    def test_score_uses_the_tracked_human_reference_by_default(self):
        args = build_parser().parse_args(["score", "--run", "run"])
        self.assertEqual(args.reference, "benchmark/references/nba_m1_v1_human")

    def test_init_run_cli_accepts_accuracy_affecting_runtime_settings(self):
        args = build_parser().parse_args(
            [
                "init-run",
                "--run-id",
                "m1-openai-v1",
                "--out",
                "run",
                "--pass-a-model",
                "a",
                "--pass-b-model",
                "b",
                "--adjudicator-model",
                "c",
                "--reasoning-effort",
                "high",
                "--image-detail",
                "original",
            ]
        )
        self.assertEqual(args.reasoning_effort, "high")
        self.assertEqual(args.image_detail, "original")

    def test_runtime_settings_are_optional_for_older_callers(self):
        destination = initialize_run(
            self.root / "legacy-run",
            run_id="m1-openai-v1",
            frames_dir=self.frames_dir,
            pass_a_model="a",
            pass_b_model="b",
            adjudicator_model="c",
        )
        payload = json.loads(destination.read_text(encoding="utf-8"))
        self.assertEqual(payload["runtime_settings"], {})


if __name__ == "__main__":
    unittest.main()
