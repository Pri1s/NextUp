"""Enforce the one-way wall: nothing under engine/ may import a training script.

The engine may depend only on ``contracts`` and third-party libraries. If this
test fails, an engine module reached back into the training pipeline (or vice
versa) and the isolation the engine exists to provide has been broken.
"""

import ast
import unittest
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parents[1]

# Repo-root training/labeling modules the engine must never import.
TRAINING_MODULES = {
    "extract_frames",
    "serve",
    "export_yolo",
    "train_pose",
    "pipeline_manifest",
    "validate_schema",
    "migrate_to_v3",
}


def _imported_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            # Relative imports (level > 0) resolve inside engine/ — always fine.
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    return roots


class IsolationTests(unittest.TestCase):
    def test_engine_does_not_import_training_modules(self):
        offenders = {}
        for path in ENGINE_DIR.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            bad = _imported_roots(tree) & TRAINING_MODULES
            if bad:
                offenders[str(path.relative_to(ENGINE_DIR.parent))] = sorted(bad)
        self.assertEqual(offenders, {}, f"engine modules import training code: {offenders}")


if __name__ == "__main__":
    unittest.main()
