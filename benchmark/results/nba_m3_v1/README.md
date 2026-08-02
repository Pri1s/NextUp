# Milestone 3 result namespace

This namespace is reserved for the v2-layout, shadow-mode extraction benchmark.
It is intentionally separate from the immutable `nba_m1_v1_human` reports.

The extraction scorer reads `marking_shadow/records.jsonl` and blind frame-space
references through `benchmark.cli score-extraction`; it does not import runtime
calibration code. A completed report must include per-feature, family, frame, and
clip macro metrics, occlusion-aware visible-reference coverage, wrong-identity
accepts, abstentions, and the frozen out-of-fold configuration hash.
