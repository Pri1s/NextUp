# NBA game-analysis engine

This is the game-analysis **engine**, walled off from the court-keypoint
**training grounds** that make up the rest of this repo (see
[`LABELING.md`](LABELING.md) / [`TRAINING.md`](TRAINING.md)). The two workflows
share nothing but a stable contract, so you can build game analysis on NBA
footage now and plug the HS court model in later without reworking either side.

## The isolation, in one rule

```
training scripts  ─┐                        ┌─  engine/  (this)
(extract_frames,   ├─▶  contracts/  ◀───────┤
 serve, export_    │   (canonical schema +  │   never imports the
 yolo, train_pose) │    court templates)    │   training scripts
                   └────────────────────────┘
```

`engine/` may import **only** `contracts/` and third-party libraries — never the
repo-root training scripts, and they never import it.
`engine/tests/test_isolation.py` enforces this by scanning every engine module's
imports; if it fails, the wall was breached.

## The plug boundary

`engine/court/base.py::CourtModel` is the single seat. `predict(frame_bgr)`
returns the **22 canonical court keypoints** (`(x, y, v)` in schema-v3 index
order). Everything downstream — homography, and the consequent analytics —
consumes canonical keypoints and never knows which model produced them.

```
                 ┌── CourtModel.predict(frame) -> 22 canonical (x,y,v) ──┐
 video frame ──▶ │  NBA:  YoloPoseCourtModel(nba court.pt, NbaCourtAdapter) │ ──▶ homography ──▶ court feet
                 │  HS:   YoloPoseCourtModel(court_pose/best.pt, Identity)   │      (── consequent: minimap,
                 └──────────────────────────────────────────────────────────┘        possession, speed …)
```

The 22 landmark *identities* are the same on every court; only their real-world
coordinates differ, which lives in per-court templates
(`contracts/court_templates/`): `nba_94x50` and `nfhs_84x50`.

## Layout

```
contracts/                     # SHARED, stable — the only cross-workflow dependency
  court_schema.py              #   canonical 22-keypoint accessor (stdlib only)
  court_template.py            #   per-court metric templates + validation
  court_templates/*.json       #   nba_94x50, nfhs_84x50
engine/
  profiles.py                  # registry: nba / hs -> {detector, ball, court model+adapter, template}
  court/base.py                # CourtModel ABC (the plug boundary)
  court/yolo_pose.py           # CourtModel for any Ultralytics YOLO pose .pt
  court/adapters.py            # IdentityAdapter (HS) + NbaCourtAdapter (18-pt map filled)
  detect/players.py            # player/ball detector wrapper (detection only, + detect_batch)
  geometry/homography.py       # canonical keypoints + template -> H; project to court feet
  io/video.py                  # frame iteration (grab/retrieve, reliable decode, start_index)
  track/bytetrack.py           # PlayerTracker: persistent ids via supervision ByteTrack
  pipeline/                    # full-game streaming pipeline (runner, records, scene, cli)
  cli.py                       # `python -m engine.cli` single-frame smoke test / --inspect
models/                        # (gitignored) player_detector.pt, ball_detector_model.pt, court_keypoint_detector.pt
engine_out/                    # (gitignored) reports, overlays, and pipeline run dirs
```

## Profiles

| Profile | Court model | Adapter | Template | Player / ball detector |
|---|---|---|---|---|
| `nba` | `models/court_keypoint_detector.pt` | `NbaCourtAdapter` (18→canonical) | `nba_94x50` | `models/player_detector.pt` / `models/ball_detector_model.pt` |
| `hs` | `runs/pose/court_pose_v1/weights/best.pt` | `IdentityAdapter` | `nfhs_84x50` | none yet |

The `nba` detectors emit several classes (`Player`, `Ball`, `Hoop`, `Ref`,
`Scoreboard`, …); the profile's `player_classes` / `ball_classes` keep only
`Player` and `Ball` so overlays and scoreboards are never tracked as players.

## Install

The engine reuses what training already installs. If you keep a separate venv:

```bash
pip install -r engine/requirements.txt
```

## Run it (NBA)

Weights live flat in `models/` (`player_detector.pt`, `ball_detector_model.pt`,
`court_keypoint_detector.pt`) and the court adapter map is already filled, so the
single-frame smoke test runs directly:

```bash
python -m engine.cli --video <nba_clip>.mp4 --profile nba
```

It writes `engine_out/<clip>.nba.json` (per-frame detections, canonical
keypoints used, and each player's foot point projected to court feet) plus a
first-frame overlay image. There is deliberately **no** tracking, teams,
minimap, or stats yet — this only proves the boundary produces something metric.

## The NBA court adapter map (filled)

The NBA court model emits its own 18-keypoint set (`kpt_shape == [18, 3]`), so it
needs a `source_index -> canonical_id` table (`NBA_COURT_KEYPOINT_MAP` in
`engine/court/adapters.py`). **This is filled**: all 18 native points map to a
canonical landmark, derived from the model's documented layout under one coherent
orientation (model-left = north, model-top = east). The four canonical points the
model has no keypoint for (both three-point apexes, both center-circle points)
stay `v == 0` — a homography needs only 4.
`engine/tests/test_court_map.py` proves the map recovers a planted homography.

To re-derive or adjust for a different court model, inspect its raw output:

```bash
python -m engine.cli --video <nba_clip>.mp4 --profile nba --inspect
```

This prints `kpt_shape` and the per-index `(x, y, v)` keypoints on a frame. Match
each source index to a canonical id (the 22 names are in
`dataset/schemas/court_keypoints.v3.json`; the reference diagram is
`web/static/court-keypoints-reference.v3.svg`).

> **Orientation caveat.** The map assumes the camera matches the model's training
> convention. Footage shot from the opposite sideline yields *mirrored* absolute
> labels (still an internally consistent, valid court frame — projected `court_ft`
> stay correct); resolving north/south per clip is a calibration follow-up.

## The plug test (proves the HS seat works today)

```bash
python -m engine.cli --video <hs_clip>.mp4 --profile hs --stride 30 --max-frames 3
```

This runs the **existing, undertrained** `court_pose_v1` weights through the same
`CourtModel` interface and solves a court homography. The predictions are rough
until the model finishes training — but it demonstrates that plugging in the
finished HS model is a **config swap, not a code change**: retrain, and the `hs`
profile picks up the new `best.pt`.

## Full-game pipeline

`engine/cli.py` processes a *sampled* handful of frames; the **pipeline** streams
a whole game (a ~2h clip) into durable, sharded per-frame records for downstream
analysis. It never loads the video into memory and checkpoints per segment, so a
multi-hour run is resumable.

```bash
python -m engine.pipeline.cli --video <game>.mp4 --profile nba --fps 5 --device mps
# smoke test on the bundled clip:
python -m engine.pipeline.cli --video input_videos/001_video_4.mp4 --profile nba \
    --fps 3 --max-frames 8 --device mps
```

Per frame it runs court model → homography, player/ball detectors, ByteTrack for
persistent ids, then projects each player's foot point and the ball onto the
court. Output: `engine_out/<stem>.<profile>/{run.json, records/seg_%05d.jsonl}`.

**Record schema (`frames-1.0.0`)** — one JSON object per processed frame:

```jsonc
{ "frame_index": 1830,           // NATIVE (unstrided) index — resume/join anchor
  "timestamp_s": 61.0,
  "shot_id": 4, "scene_cut": false,  // shot bumps on a scene cut / resume boundary
  "court": { "visible_keypoints": 11, "homography": [[…]]|null,
             "used_keypoint_ids": […], "keypoints": null },
  "players": [ { "track_id": 7, "cls": 4, "name": "Player", "conf": 0.83,
                 "bbox": […], "foot_px": [x,y], "court_ft": [x,y]|null } ],
  "ball": { "conf": 0.61, "bbox": […], "center_px": [x,y], "court_ft": [x,y]|null } | null,
  "error": null }                // set when a frame failed; run continues
```

**Downstream join contract:** `track_id` is unique and continuous **only within a
`shot_id`**. Join a player across frames on `(shot_id, track_id)`; never assume
continuity across a `scene_cut` or a `--resume` boundary (ByteTrack cannot carry
ids across a camera cut or a cold restart, so the tracker resets at both).

**Scale.** Default processes at `--fps` (5 ⇒ 2h ≈ 36k frames). Decode still walks
every source frame (seeking is deliberately avoided), so runtime is decode +
3 model forwards per processed frame — order ~1–2 h on GPU/MPS, longer on CPU.
`--batch` bounds detector memory (default 8; drop to 4 on <8 GB accelerators);
`--segment-frames` sets the checkpoint/shard granularity. A crash loses at most
the in-flight segment; a per-frame error is recorded and the run continues.

## Consequent roadmap

Everything below wraps the existing seams rather than replacing them:

1. ~~**Tracking** — persistent track IDs over `detect/`.~~ **Done** —
   `engine/track/bytetrack.py`, consumed by the pipeline (shot-scoped ids).
2. **Team assignment** — jersey-color clustering / embeddings on tracked boxes.
3. **Minimap** — top-down court render (reuse the reference-diagram geometry in
   `court_keypoints.v3.json` and `web/static/court-keypoints-reference.v3.svg`),
   with homography smoothing across frames.
4. **Analytics** — ball possession, player speed/distance, zones/heatmaps.
5. **Events** — shots, passes, rebounds.
6. **Export + viewer** — parquet record shards (`--format parquet`), and a web
   results viewer (growing the existing Flask app's sibling space, still isolated).

## Tests

```bash
python -m unittest discover -s engine/tests -p "test_*.py"
# plus the shared contract tests:
python -m unittest contracts.tests.test_court_template
```
