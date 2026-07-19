# Active dataset workspace

This directory is the **active v2 pipeline workspace**, not a legacy dataset.

- `schemas/` contains the permanent canonical keypoint definitions.
- `frames/`, `thumbs/`, and `manifest.json` were freshly generated at the
  two-second sampling rate and may be regenerated with
  `python extract_frames.py --reset`.
- `labels/` is created as new v2 labels are saved.

Legacy labels, manifests, extracted frames, thumbnails, and exports were
intentionally discarded and are not stored in this repository. Do not place
legacy artifacts in this directory.
