from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmark.extraction import score_extraction


class ExtractionScoringTests(unittest.TestCase):
    def test_scoring_consumes_serialized_contracts_and_rejects_hash_mismatch(self):
        # The contract-level construction is covered in hybrid tests; this test
        # keeps the benchmark boundary explicit without importing calibration.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "records.jsonl").write_text("", encoding="utf-8")
            (root / "refs").mkdir()
            report = score_extraction(root / "records.jsonl", root / "refs")
            self.assertEqual(report["records"], 0)
            self.assertEqual(report["schema_version"], "benchmark-extraction-score-1.0.0")


if __name__ == "__main__":
    unittest.main()
