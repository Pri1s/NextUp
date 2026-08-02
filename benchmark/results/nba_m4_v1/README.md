# Milestone 4 result namespace

This namespace is reserved for the shadow-mode comparison of keypoint and
hybrid candidates under the v2 layout. It is intentionally separate from the
immutable M1 and M3 reports.

`benchmark.cli score-hybrid` reads only serialized `hybrid_shadow/records.jsonl`
and blind frame-space references. It reports identical-frame summaries,
per-band comparison, held-out errors, gate/abstention rollups, and the §9
acceptance block. No candidate is promoted in Milestone 4.
