"""Court-calibration engine.

May import ``contracts`` and third-party libraries only — never the repo-root
training or labeling scripts. ``calibration/tests/test_isolation.py`` enforces
this.

The keypoint calibration path remains provisional. Marking extraction is an
opt-in shadow-only subsystem and never changes the baseline calibration result.
"""

__version__ = "0.1.0"
