"""The sole file-writing boundary for hybrid shadow artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from contracts.hybrid_types import HybridFrameStatus, HybridRefinementFrame


def write_hybrid_run(out_dir: Path | str, records, *, metadata: dict) -> Path:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    ordered = sorted(tuple(records), key=lambda item: item.frame_id)
    with open(root / "records.jsonl", "w", encoding="utf-8") as handle:
        for record in ordered:
            handle.write(json.dumps(record.as_dict(), sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    payload = dict(metadata)
    payload.setdefault("schema_version", "hybrid-shadow-run-1.0.0")
    payload["record_count"] = len(ordered)
    payload["status_counts"] = {
        status.value: sum(1 for item in ordered if item.status is status)
        for status in HybridFrameStatus
    }
    (root / "run.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return root


def read_hybrid_records(path: Path | str) -> list[HybridRefinementFrame]:
    source = Path(path)
    if source.is_dir():
        source = source / "records.jsonl"
    return [
        HybridRefinementFrame.from_dict(json.loads(line))
        for line in source.read_text(encoding="utf-8").splitlines() if line.strip()
    ]

