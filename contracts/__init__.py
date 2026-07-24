"""Shared, stable contract layer between the two isolated workflows.

The training grounds (labeling + pose fine-tuning at the repo root) and the
game-analysis ``engine`` package must not import each other. Both may depend on
this package, which exposes only the canonical court schema and the per-court
metric templates — the plug boundary that lets an NBA court model today and the
HS pose model later occupy the same seat.

This package is stdlib-only and reads the canonical schema in place at
``dataset/schemas/court_keypoints.v3.json``. It does not import the training
scripts (``validate_schema`` remains the authoritative training-side validator).
"""
