# Court keypoint sanity check

A small video-only test harness for visually checking the fine-tuned basketball
court keypoint model. It processes frames one at a time, draws each detected
keypoint, and prints its confidence next to it. Point colors run from red
(lower confidence) to green (higher confidence).

## Install

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The model is expected at `models/court_keypoint_detector.pt` by default.

## Run on any game video

```bash
python3 run_court_keypoint_check.py "/path/to/gym footage.mp4" \
  --output output_videos/gym_footage_check.mp4
```

Useful options:

```text
--model PATH             use a different .pt file
--output PATH            choose the annotated video path
--conf 0.5              model detection threshold
--keypoint-conf 0.25    hide keypoints below this threshold
--device 0              use GPU 0 (omit for Ultralytics' automatic choice)
```

For example, the model can be run on one of several clips without changing any
code:

```bash
python3 run_court_keypoint_check.py /videos/gym_a.mp4 --output /tmp/gym_a.mp4
python3 run_court_keypoint_check.py /videos/gym_b.mp4 --output /tmp/gym_b.mp4
```

The output is an H.264-encoded MP4 with the original frame rate and dimensions, so
it is compatible with standard browser video players. The top-left
overlay shows the number of visible keypoints and their mean confidence for the
current frame. This is intentionally a quick sanity-check tool, not a tracking
or production pipeline.

## Training-data pipeline

Turns raw game clips into corrected canonical-court labels. State lives in
`dataset/manifest.json`; each command prints progress counts.

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
  use arrows with `K`/`S`/`U`.
- *Orientation anchor*: before the first model prefill for a clip, choose its
  anchor frame. In the supplied broadcast footage, north is recorded as the
  basket on that image’s left side. The app compares raw end-group mean X
  positions, locks either the identity or 180-degree normalization, and records
  it in the clip manifest. If the comparison is inconclusive, choose whether
  the raw first end is left or right explicitly.
- *Label view*: fresh prefill is already normalized to the fixed K1–K18 court
  map. The static north-at-top, east-at-right diagram highlights the selected
  point. Drag points to correct them; remove and replace points as needed;
  `V`/right-click cycles visibility (2 visible / 1 occluded / 0 excluded).
  Labels include schema and orientation provenance under
  `dataset/labels/<clip>/<frame>.json`.
- *Labeling from scratch*: unplaced points are hidden. Click a point in the
  list, move the mouse, and click on the frame to drop it. Never-placed points
  are saved as `(0, 0)` with visibility 0.

Do not run `extract_frames.py` while the server is up because both write the
manifest.

**4. Export for training.**

```bash
python3 export_yolo.py               # writes export/court_pose/
```

The exporter validates every saved label against the canonical schema, writes
a deterministic train/validation split, copies `court_keypoints.v2.json` into
the export, and writes the east/west reflection `flip_idx` in `data.yaml`.

`python3 pipeline_manifest.py` prints current counts.

## Keypoint semantics

[`dataset/schemas/court_keypoints.v2.json`](dataset/schemas/court_keypoints.v2.json)
is the authoritative K1–K18 mapping. It defines north at the top baseline and
east at the right sideline, records the diagram markers and the two allowed
raw-prefill permutations, and contains only trained court landmarks: baseline
features, sideline/midcourt intersections, and free-throw-line lane corners.
The adjacent audit records the reviewed geometry. Every saved label and export
uses this one canonical index meaning.
