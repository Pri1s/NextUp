"""Court-calibration engine.

May import ``contracts`` and third-party libraries only — never the repo-root
training or labeling scripts. ``calibration/tests/test_isolation.py`` enforces
this.

Phase 0 only: model-truth inspection tooling. No court layouts, no landmark
semantics, no homography.
"""

__version__ = "0.1.0"
