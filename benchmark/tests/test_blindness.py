"""The benchmark must not be able to see what it is measuring.

An annotation that has been influenced by the model's predictions inherits the
model's systematic error, and a benchmark built from such annotations certifies
the drift it exists to detect. That is a stated design constraint (§7.2), and a
constraint enforced only by intention is one violated by the next convenient
import.

The wall is not "benchmark must not import calibration". Plenty of the engine is
pure geometry with no model in it, and duplicating a homography solver to avoid
touching it would create two solvers that could disagree — a worse failure. What
the benchmark must never reach is the part that *predicts*: the detector, the slot
adapters, and the calibrator that turns slots into transforms.

This mirrors ``calibration/tests/test_isolation.py``, with one difference: that
scanner matches on the root package, and here the boundary runs *inside*
``calibration``, so this one resolves full dotted module paths.
"""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPO_DIR / "benchmark"
PROMPT_DIR = BENCHMARK_DIR / "prompts"

#: Model-free engine modules the benchmark may share. Reusing these is better than
#: reimplementing them: two homography solvers that disagree would be a defect the
#: benchmark could not distinguish from a calibration error.
#:
#: ``calibration.polyline`` was added for Milestone 4 by the same argument. It is
#: pure numpy point-to-polyline geometry that both the hybrid refiner and the
#: scorer need; it holds no layout, no detector, and no model, so it cannot carry
#: a prediction across the wall. It lives under ``calibration/`` rather than here
#: because the refiner needs it at runtime and the dependency may only point one
#: way.
ALLOWED_CALIBRATION_MODULES = frozenset(
    {
        "calibration.estimator",
        "calibration.io.video",
        "calibration.polyline",
        "calibration.viz",
    }
)

#: Anything that predicts, or interprets a prediction.
FORBIDDEN_MODULES = frozenset(
    {
        "calibration.detectors",
        "calibration.adapters",
        "calibration.calibrator",
        "calibration.quality",
        "calibration.correspondence",
        "calibration.tools",
    }
)

#: Importing an inference stack at all would mean something here runs a model.
FORBIDDEN_ROOTS = frozenset({"ultralytics", "torch", "torchvision"})


def _imported_modules(tree: ast.AST) -> set[str]:
    """Full dotted paths of every absolute import in a module."""
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                modules.add(node.module)
                for alias in node.names:
                    modules.add(f"{node.module}.{alias.name}")
    return modules


def _scan() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for path in sorted(BENCHMARK_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        found[str(path.relative_to(REPO_DIR))] = _imported_modules(tree)
    return found


class BlindnessTests(unittest.TestCase):
    def setUp(self):
        self.imports = _scan()

    def test_directory_was_actually_scanned(self):
        """Guards against a silent pass from a mistyped path."""
        self.assertGreater(len(self.imports), 5)

    def test_benchmark_never_imports_a_predicting_module(self):
        offenders = {}
        for path, modules in self.imports.items():
            if path.startswith("benchmark/tests/"):
                continue
            bad = {
                m for m in modules
                if any(m == f or m.startswith(f + ".") for f in FORBIDDEN_MODULES)
            }
            if bad:
                offenders[path] = sorted(bad)
        self.assertEqual(offenders, {})

    def test_benchmark_never_imports_an_inference_stack(self):
        offenders = {}
        for path, modules in self.imports.items():
            bad = {m for m in modules if m.split(".")[0] in FORBIDDEN_ROOTS}
            if bad:
                offenders[path] = sorted(bad)
        self.assertEqual(offenders, {})

    def test_engine_imports_are_confined_to_the_agreed_list(self):
        """A new engine import has to be a deliberate decision, not a convenience."""
        offenders = {}
        for path, modules in self.imports.items():
            if path.startswith("benchmark/tests/"):
                continue
            engine = {m for m in modules if m.split(".")[0] == "calibration"}
            unexpected = {
                m for m in engine
                if not any(m == a or m.startswith(a + ".") for a in ALLOWED_CALIBRATION_MODULES)
            }
            if unexpected:
                offenders[path] = sorted(unexpected)
        self.assertEqual(offenders, {})

    def test_scanner_actually_catches_a_violation(self):
        """A guard that cannot fail is not a guard."""
        tree = ast.parse(
            "from calibration.detectors.yolo_pose import YoloPoseDetector\n"
            "import ultralytics\n"
            "from calibration.estimator import project\n"
        )
        modules = _imported_modules(tree)
        self.assertIn("calibration.detectors.yolo_pose", modules)
        self.assertIn("ultralytics", modules)
        self.assertTrue(
            any(m.startswith("calibration.detectors") for m in modules),
            "the forbidden-module match would not fire on a real violation",
        )
        self.assertIn("calibration.estimator", modules)


class AgentDefinitionTests(unittest.TestCase):
    """Prompts are OpenAI-runtime compatible without pinning a model in source."""

    def test_prompts_are_frozen(self):
        from benchmark.workflow import verify_prompts

        self.assertEqual(verify_prompts(), [])

    def test_prompts_are_model_neutral_and_select_exact_ids_at_runtime(self):
        for name in ("court-annotator", "court-adjudicator"):
            with self.subTest(prompt=name):
                text = (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8")
                lowered = text.lower()
                self.assertIn("openai multimodal agent", lowered)
                self.assertIn("exact runtime model id", lowered)
                self.assertNotRegex(
                    lowered, r"(?m)^model\s*:", "model IDs belong in the run record"
                )
                self.assertNotIn("sonnet", lowered)
                self.assertNotIn("opus", lowered)
                self.assertNotIn("claude", lowered)

    def test_prompts_declare_a_version(self):
        """A prompt change silently invalidates earlier labels without this."""
        for name, stem in (
            ("court-annotator", "annotator"),
            ("court-adjudicator", "adjudicator"),
        ):
            with self.subTest(agent=name):
                text = (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8")
                self.assertRegex(text, rf"Prompt version: `{stem}-\d+\.\d+\.\d+`")

    def test_prompt_declares_exactly_one_version(self):
        """The declared version and the version in the worked example must agree.

        These drifted apart once: the header was bumped and the example JSON that an
        annotator copies its output format from was not, so every annotation would
        have recorded the superseded version while the prompt claimed the new one.
        Pinning one literal string per agent could not catch it -- the stale example
        satisfied the assertion on its own.
        """
        for name, stem in (
            ("court-annotator", "annotator"),
            ("court-adjudicator", "adjudicator"),
        ):
            with self.subTest(agent=name):
                text = (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8")
                found = set(re.findall(rf"\b{stem}-\d+\.\d+\.\d+", text))
                self.assertEqual(
                    len(found), 1, f"{name}: conflicting prompt versions {sorted(found)}"
                )

    def test_annotator_is_told_not_to_compute_frame_coordinates(self):
        text = (PROMPT_DIR / "court-annotator.md").read_text(encoding="utf-8").lower()
        self.assertIn("never compute a frame coordinate", text)

    def test_adjudicator_is_told_it_may_refuse(self):
        """A forced guess from the tie-breaker is weighted more than an ordinary one."""
        text = (PROMPT_DIR / "court-adjudicator.md").read_text(encoding="utf-8").lower()
        self.assertIn("ambiguous_identity", text)


class DepthBandAgreementTests(unittest.TestCase):
    """The engine's copy of the depth bands must not drift from the benchmark's.

    ``calibration.hybrid`` reports per-band transform displacement and cannot
    import ``benchmark`` to get the band definition -- the whole point of keeping
    scoring outside the engine is that runtime code cannot see the modules that
    grade it. So the definition is duplicated, and this test is the thing that
    stops the duplicate diverging. It lives here because only the benchmark side
    is allowed to look in both directions.
    """

    def test_band_definitions_are_identical(self):
        from calibration.hybrid.control_points import DEFAULT_DEPTH_BANDS

        from benchmark.geometry import DEPTH_BANDS

        self.assertEqual(DEFAULT_DEPTH_BANDS, DEPTH_BANDS)

    def test_band_assignment_agrees_including_the_clamped_edges(self):
        """The half-court runs to 47.083 ft while the last band stops at 47."""
        import numpy as np

        from calibration.hybrid.control_points import DEFAULT_DEPTH_BANDS, band_indices

        from benchmark.geometry import depth_band

        depths = np.array([-5.0, 0.0, 18.999, 19.0, 29.5, 30.0, 46.9, 47.0, 47.0833, 100.0])
        index = band_indices(depths, DEFAULT_DEPTH_BANDS)
        for value, position in zip(depths, index):
            with self.subTest(depth=float(value)):
                self.assertEqual(DEFAULT_DEPTH_BANDS[position][0], depth_band(float(value)))


if __name__ == "__main__":
    unittest.main()
