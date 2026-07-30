"""Enforce the isolation wall.

Two directions, both load-bearing:

The engine must never import the repo-root training/labeling scripts. If it did,
calibration would inherit their schema assumptions and their global state, and
the engine would stop being swappable.

``contracts/`` must stay stdlib-only. It is the vocabulary every other layer
shares; the moment it imports numpy or ultralytics it stops being a stable wall
and becomes another dependency to version-match.
"""

from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
CALIBRATION_DIR = REPO_DIR / "calibration"
CONTRACTS_DIR = REPO_DIR / "contracts"

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

# Third-party roots that must not appear under contracts/.
THIRD_PARTY = {"cv2", "numpy", "np", "torch", "ultralytics", "flask", "PIL", "pandas", "scipy"}


def _imported_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            # Relative imports (level > 0) resolve inside the package — always fine.
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    return roots


def _scan(directory: Path, forbidden: set[str]) -> dict[str, list[str]]:
    offenders: dict[str, list[str]] = {}
    for path in sorted(directory.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        bad = _imported_roots(tree) & forbidden
        if bad:
            offenders[str(path.relative_to(REPO_DIR))] = sorted(bad)
    return offenders


class IsolationTests(unittest.TestCase):
    def test_calibration_does_not_import_training_modules(self):
        self.assertEqual(_scan(CALIBRATION_DIR, TRAINING_MODULES), {})

    def test_contracts_does_not_import_training_modules(self):
        self.assertEqual(_scan(CONTRACTS_DIR, TRAINING_MODULES), {})

    def test_contracts_is_stdlib_only(self):
        self.assertEqual(_scan(CONTRACTS_DIR, THIRD_PARTY), {})

    def test_scanner_actually_catches_a_violation(self):
        """A guard that cannot fail is not a guard."""
        tree = ast.parse("import serve\nfrom ultralytics import YOLO\n")
        self.assertEqual(_imported_roots(tree) & TRAINING_MODULES, {"serve"})
        self.assertEqual(_imported_roots(tree) & THIRD_PARTY, {"ultralytics"})

    def test_directories_were_actually_scanned(self):
        """Guards against a silent pass from a mistyped path."""
        self.assertGreater(len(list(CALIBRATION_DIR.rglob("*.py"))), 5)
        self.assertGreater(len(list(CONTRACTS_DIR.rglob("*.py"))), 1)

    def test_no_verified_adapter_map_exists(self):
        """An adapter map may exist, but never one claiming verified semantics.

        The interim engine deliberately ships a *provisional* evidence map: it
        pools slot votes without asserting fixed slot identity. What must not
        appear is a map presented as settled — the design doc's
        ``adapters/maps/reloc2_18.json`` is reserved for after human sign-off,
        and a verified-looking map would let downstream code trust identities the
        Phase-0 gate has not cleared.
        """
        maps_dir = CALIBRATION_DIR / "adapters" / "maps"
        if not maps_dir.is_dir():
            return

        self.assertFalse(
            (maps_dir / "reloc2_18.json").exists(),
            "reloc2_18.json is the post-sign-off path and must not exist yet",
        )

        for path in sorted(maps_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            with self.subTest(map=path.name):
                self.assertEqual(
                    data.get("status"),
                    "provisional",
                    f"{path.name} must declare status 'provisional'",
                )
                self.assertFalse(
                    data.get("is_verified_map", False),
                    f"{path.name} must not claim to be a verified map",
                )

    def test_provisional_map_leaves_ambiguous_slots_unmapped(self):
        """Slots the review could not settle must carry no landmark id.

        Slot 3/12 reads as two different court features depending on which way
        the court faces. Giving it an id would make every frame of one
        orientation silently wrong, which is exactly the failure mode the gate
        exists to prevent.
        """
        path = CALIBRATION_DIR / "adapters" / "maps" / "reloc2_18_provisional.json"
        if not path.is_file():
            self.skipTest("provisional map not present")

        data = json.loads(path.read_text(encoding="utf-8"))
        mapped = {slot for group in data["groups"] for slot in group["slots"]}
        rejected = {slot for entry in data["rejected"] for slot in entry["slots"]}

        self.assertEqual(mapped & rejected, set(), "a slot cannot be both mapped and rejected")
        for slot in (3, 12):
            self.assertNotIn(slot, mapped, f"ambiguous slot {slot} must stay unmapped")
        self.assertTrue(data.get("requires_human_verification"))


if __name__ == "__main__":
    unittest.main()
