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

Turns raw game clips into corrected keypoint labels in four steps. All state
lives in `dataset/manifest.json`; every command prints progress counts (clips,
candidates, keep/skip, labeled) when it finishes.

**1. Extract candidate frames.** Drop clips into `input_videos/` and run:

```bash
python3 extract_frames.py            # ~1 frame per second per clip
```

Already-extracted clips are skipped, so re-run it whenever you add footage.
Use `--interval` to change the sampling rate, `--force --clip ID` to redo one
clip (triage/label decisions are preserved).

**2 & 3. Triage and label in the browser.**

```bash
python3 serve.py                     # then open http://127.0.0.1:8000
```

- *Triage view* (per clip): thumbnail grid; click a frame to cycle
  pending → keep → skip, or use arrows + `K`/`S`/`U`. Saves on every click.
- *Label view*: steps through kept frames with the model's predicted keypoints
  drawn on a zoomable canvas (red→green by confidence). Drag points to correct
  them, `V`/right-click cycles visibility (2 visible / 1 occluded / 0 excluded),
  `←`/`→` moves between frames and autosaves. Labels are written to
  `dataset/labels/<clip>/<frame>.json`.

Don't run `extract_frames.py` while the server is up — both write the manifest.

**4. Export for training.**

```bash
python3 export_yolo.py               # writes export/court_pose/
```

Produces a YOLO pose dataset (`images/`, `labels/`, `data.yaml`) from all
labeled frames with a deterministic train/val split. Note: `flip_idx` is not
defined yet, so train with `fliplr: 0.0`.

`python3 pipeline_manifest.py` prints the current counts at any time.
