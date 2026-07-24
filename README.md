# NextUp court keypoints

Hand-labeling pipeline for basketball court keypoints (schema v3). Labeling
conventions live in [`LABELING.md`](LABELING.md); the machine-readable schema
is
[`dataset/schemas/court_keypoints.v3.json`](dataset/schemas/court_keypoints.v3.json).
Once you have some labels, [`TRAINING.md`](TRAINING.md) covers fine-tuning a
pose model on them and reading the results.

> **Two isolated workflows.** This repo is the court-keypoint *training
> grounds* (everything below). The NBA game-analysis *engine* is walled off in
> [`engine/`](engine/) and shares only the canonical schema via
> [`contracts/`](contracts/) — see [`ENGINE.md`](ENGINE.md). The finished HS
> pose model plugs into the same seat the NBA court model occupies.

> **Superseded:** the reloc2-derived K1–K18 model and schema v2 are no longer
> used — the model's point semantics were never confirmed. The weights and
> schema v2 files have been removed; nothing in the v3 pipeline references
> them. The Group A/B orientation flow from v2 is gone; orientation is now
> declared by the labeler directly.

## Install

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Labeling pipeline

Turns raw game clips into hand-labeled canonical-court keypoints. State lives
in `dataset/manifest.json`; each command prints progress counts. The loop:

```
extract_frames.py → triage (keep/skip) → anchor clip orientation
   → label + confirm visible ends
```

**1. Reset and extract candidate frames.** Drop clips into `input_videos/`,
then start a new labeling run with:

```bash
python3 extract_frames.py --reset     # one deterministic candidate per two seconds
```

`--reset` removes only derived state: `dataset/frames`, `dataset/thumbs`,
`dataset/labels`, `dataset/manifest.json`, and `export/`. It preserves input
videos, schemas, and source code, then extracts every input clip from scratch.
Without `--reset`, existing clips at the same cadence are skipped. Use
`--interval` to choose a different cadence; the default is `2.0` seconds.

**2 & 3. Triage and label in the browser.**

```bash
python3 serve.py                     # then open http://127.0.0.1:8000
```

- *Triage view* (per clip): click a frame to cycle pending → keep → skip, or
  use arrows with `K`/`S`/`U`. Aim for variety, not volume — see
  [`LABELING.md`](LABELING.md).
- *Orientation anchor* (once per clip): north is always the image-left basket.
  Find a frame where both baskets are discernible and lock the anchor; if no
  frame shows both ends, declare which end the anchor frame shows instead.
  The lock is recorded in the manifest and echoed into every saved label.
- *Label view*: click a keypoint in the list, move the mouse, and click the
  frame to drop it; drag to adjust. `V`/right-click toggles visible (2) /
  occluded (1); `⌫` unlabels a point (saved as `(0, 0)` with visibility 0).
  On every frame, confirm *visible ends* — North `1` / Both `2` / South `3` —
  before saving; points from a non-visible end are locked out and the server
  rejects contradictions. Labels land in `dataset/labels/<clip>/<frame>.json`
  with schema, orientation, and visible-ends provenance.
- *Model prefill* (optional, round 2+): `python3 serve.py --model
  runs/pose/court_pose_v1/weights/best.pt` prefills points from your own
  trained model; `R` re-predicts. Models whose keypoint count does not match
  the schema are refused.

Do not run `extract_frames.py` while the server is up because both write the
manifest.

`python3 pipeline_manifest.py` prints current counts.

## Keypoint semantics

[`dataset/schemas/court_keypoints.v3.json`](dataset/schemas/court_keypoints.v3.json)
is the authoritative mapping: 22 fixed court landmarks named by absolute
position (north/south end, east/west side), never by camera viewpoint. It
defines north at the top baseline and east at the right sideline, records the
diagram markers, both mirror-pair sets (east/west and north/south), and which
set feeds the horizontal-flip `flip_idx` — the north/south set, because under
the image-left-basket convention a horizontal flip swaps court ends, not
sidelines. The adjacent audit file records the derivations;
`python3 validate_schema.py` checks the whole artifact and prints the derived
`flip_idx`. Every saved label and export uses this one canonical index
meaning.
