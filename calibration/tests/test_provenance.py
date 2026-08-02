"""Provenance must be exact and must survive a hostile environment.

A slot-semantics sign-off is only auditable if the artifacts say which weights,
which frames, and which code produced them. Equally, a missing git or an absent
optional library must degrade to a recorded "unknown" rather than crash the run —
losing an inspection run to a subprocess failure would be absurd.
"""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from calibration.io import provenance as prov


class HashTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_file_digest_matches_hashlib(self):
        path = self.tmp / "blob.bin"
        payload = b"court keypoints" * 5000
        path.write_bytes(payload)
        self.assertEqual(prov.sha256_file(path), hashlib.sha256(payload).hexdigest())

    def test_digest_streams_across_chunk_boundaries(self):
        path = self.tmp / "big.bin"
        payload = bytes(range(256)) * 9000
        path.write_bytes(payload)
        self.assertEqual(
            prov.sha256_file(path, chunk_size=1024), hashlib.sha256(payload).hexdigest()
        )

    def test_text_digest_matches_hashlib(self):
        self.assertEqual(prov.sha256_text("abc"), hashlib.sha256(b"abc").hexdigest())

    def test_file_facts_shape(self):
        path = self.tmp / "weights.pt"
        path.write_bytes(b"not really a model")
        facts = prov.file_facts(path)
        self.assertEqual(set(facts), {"path", "sha256", "size_bytes", "mtime_utc"})
        self.assertEqual(facts["size_bytes"], 18)
        self.assertTrue(Path(facts["path"]).is_absolute())


class EnvironmentTests(unittest.TestCase):
    def test_records_library_versions(self):
        facts = prov.environment_facts(device="mps")
        self.assertEqual(facts["device"], "mps")
        self.assertIn("opencv", facts["libraries"])
        self.assertIn("numpy", facts["libraries"])
        self.assertTrue(facts["python"])

    def test_missing_library_is_recorded_not_raised(self):
        """An absent optional dependency must not take down a run."""
        facts = prov.environment_facts()
        for value in facts["libraries"].values():
            self.assertIsInstance(value, str)


class GitTests(unittest.TestCase):
    def test_non_repo_directory_is_tolerated(self):
        with tempfile.TemporaryDirectory() as tmp:
            facts = prov.git_facts(tmp)
            self.assertIn("available", facts)
            self.assertIn("commit", facts)

    def test_missing_directory_is_tolerated(self):
        facts = prov.git_facts("/nonexistent/path/for/a/test")
        self.assertFalse(facts["available"])
        self.assertIsNone(facts["commit"])


class InventoryTests(unittest.TestCase):
    def test_inventory_digests_match_written_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sub").mkdir()
            first = root / "a.json"
            second = root / "sub" / "b.csv"
            first.write_text("{}")
            second.write_text("x,y\n1,2\n")

            inventory = prov.output_inventory(root, [first, second])
            self.assertEqual([e["path"] for e in inventory], ["a.json", "sub/b.csv"])
            for entry in inventory:
                actual = prov.sha256_file(root / entry["path"])
                self.assertEqual(entry["sha256"], actual)

    def test_inventory_skips_paths_that_were_not_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real = root / "real.json"
            real.write_text("{}")
            inventory = prov.output_inventory(root, [real, root / "ghost.json"])
            self.assertEqual([e["path"] for e in inventory], ["real.json"])


if __name__ == "__main__":
    unittest.main()
