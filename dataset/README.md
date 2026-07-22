# Active dataset workspace

This directory is the **active v3 pipeline workspace**, not a legacy dataset.

- `schemas/` contains the permanent canonical keypoint definitions:
  `court_keypoints.v3.json` (22 hand-labeled points) plus its audit file. The
  reloc2-derived schema v2 files have been removed — they defined K1–K18
  point semantics that were never confirmed.
- `frames/`, `thumbs/`, and `manifest.json` are generated at the two-second
  sampling rate and may be regenerated with `python extract_frames.py --reset`.
- `labels/` is created as v3 labels are saved.

Do not place legacy artifacts in this directory.
