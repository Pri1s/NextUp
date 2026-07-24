"""NBA game-analysis engine — isolated from the training grounds.

This package is walled off from the repo-root training/labeling pipeline
(``extract_frames``, ``serve``, ``export_yolo``, ``train_pose``,
``pipeline_manifest``, ``validate_schema``, ``migrate_to_v3``). It depends only
on the shared ``contracts`` package and third-party libraries — never on the
training scripts, and they never depend on it. ``engine/tests/test_isolation.py``
enforces that wall.

The single plug boundary is ``engine.court.base.CourtModel``: any model that
emits the 22 canonical court keypoints fits the same seat, so the NBA court
model today and the HS pose model later are interchangeable via profiles
(``engine.profiles``).

Scope today is deliberately the plug boundary only. Tracking, team assignment,
minimap rendering, and analytics are the consequent roadmap (see ENGINE.md).
"""
