# Phase 0 — model truth

Phase 0 of [`COURT_CALIBRATION_ARCHITECTURE_DESIGN_PLAN.md`](COURT_CALIBRATION_ARCHITECTURE_DESIGN_PLAN.md)
is a hard gate: **no calibration work begins until a human has established what
`models/court_keypoint_detector.pt`'s 18 keypoint slots actually mean.**

This document covers the tooling that produces the evidence for that judgement.
The tooling does not make the judgement. A green run is not a sign-off.

## Why the gate exists

Verified from the checkpoint itself:

| | |
| --- | --- |
| architecture | `yolov8x-pose` fine-tune, task `pose` |
| `kpt_shape` | `[18, 3]` |
| classes | `{0: 'basketball'}` |
| train `imgsz` | 640 |
| train `fliplr` | **0.5** |
| train data | `/kaggle/working/reloc2-7/data.yaml` (unavailable) |
| SHA-256 | `f6263105e5c2338fafcfd5a6fefd7d1d441e87364635e918dfdbb849f2df1377` |

This is the reloc2-family 18-point model, not NextUp's 22-point v3 schema, and
[`README.md`](../README.md) already records that its K1–K18 semantics were never
confirmed. It trained with horizontal flip against a `data.yaml` we do not have,
so its `flip_idx` is unknown. If that permutation was wrong or identity, the
model learned mirrored images with unmirrored landmark identities — which
corrupts end/side identity while leaving aggregate pose mAP healthy. Nothing in
a training curve would reveal it.

A wrong slot assignment does not fail loudly downstream. It produces a plausible,
silently wrong homography, and every distance and speed derived from it inherits
the error. That asymmetry is the whole argument for the gate.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

No new dependencies — `opencv-python`, `numpy` and `ultralytics` already cover it.

## Run

```bash
python -m calibration.cli inspect-model --weights models/court_keypoint_detector.pt --video input_videos/video_1.mp4 --num-frames 24 --imgsz 640,960,1280 --flip-probe --device mps --out engine_out/phase0
```

Useful flags:

| flag | effect |
| --- | --- |
| `--num-frames N` | evenly spaced frames to inspect (default 24) |
| `--frames 0,40,80` | inspect exactly these indices instead |
| `--images <dir\|glob>` | inspect still images — the intake path for no-court negatives |
| `--imgsz 640,960,1280` | sweep inference sizes; the first is primary |
| `--gate 0.25` | confidence gate for **display and statistics only** — records always keep every slot |
| `--flip-probe` | also predict on the mirrored frame and report slot correspondence |
| `--candidate-vocabulary <json>` | append a reference name list to the worksheet |
| `--device mps\|cuda:0\|cpu` | torch device |

Output lands in `engine_out/phase0/<source>/<run_id>/` (gitignored).

## Artifacts

```text
run.json                    provenance: weights digest, frames, env, git, output inventory
review.html                 the reviewer's entry point — open it first
slot_review.md              the worksheet to fill in
slot_review.template.json   machine-readable worksheet, every assignment null
sweep_comparison.csv        per-slot detection rate and confidence at each imgsz
imgsz_0640/
  predictions.jsonl         one lossless record per frame
  predictions.csv           one row per observation (frame, slot, x, y, conf, clamped)
  frames.csv                one row per frame, including no-detection frames
  summary.json              per-slot detection rate, confidence, positional spread
  flip_probe.json           per-slot flip verdicts and votes
  overlays/frame_*.jpg      indexed keypoint overlays
  slots/slot_NN.jpg         per-slot filmstrips
  flip/frame_*.jpg          flip correspondence views
  contact_sheet.jpg         all frames at a glance
```

### Conventions

**Coordinates** are pixels in the original full-resolution frame, top-left
origin, y-down.

**Confidence** is continuous `[0,1]`, never rounded or collapsed into a
visibility flag. The `--gate` affects what is drawn boldly and what is counted in
statistics; it never changes what is recorded.

**Absence is absence.** The reference pipeline's `(0,0)` sentinel is banned. A
frame with no detection emits *no* observation rows, not eighteen fake origin
points. A poorly localised slot keeps its real predicted coordinate, and
predictions that saturate against the frame border are flagged `clamped` so a
genuine edge landmark can be told apart from a pinned one.

**All 18 slots are recorded**, including below-gate ones. This is a deliberate
departure from the design doc's "omit absent observations" rule, which applies
from the Phase-1 adapter boundary onward. In Phase 0 the below-gate behaviour of
a slot *is* the object of study — a slot that is consistently low-confidence in a
consistent place is evidence, and dropping it would defeat the gate.

## How to review

Open `review.html` from disk. It needs no server and reaches no network.

1. **Skim the frame overlays.** Establish which frames actually show court, and
   which are replays, crowd shots or close-ups.
2. **Work one slot at a time down the per-slot filmstrips.** This is the artifact
   that decides the question. Each strip is one slot cropped at its prediction
   across every inspected frame. Ask only: *is this the same physical court
   feature in every tile?*
3. **Check that slot's row in the per-slot table.** A low above-gate rate or a
   large positional spread means the filmstrip's apparent agreement may be luck.
4. **Read the flip verdict.** `self` means the slot survives mirroring;
   `partner:j` means the flip lands it on slot *j*'s position — the signature of a
   learned mirror permutation; `incoherent` means it goes nowhere sensible.
5. **Write the conclusion into `slot_review.md`**, using the verdict vocabulary
   in that file. Where two readings fit, record `ambiguous` and say which two.

An honest "ambiguous" is worth more than a confident guess, for the reason in the
first section: the failure is silent.

## What this tooling does not do

It writes **no verified adapter map** and asserts **no landmark semantics**.
`slot_review.template.json` carries `status: "unverified"` and
`is_adapter_map: false`, and every `landmark_id` is `null` by construction.
`calibration/adapters/maps/reloc2_18.json` — the post-sign-off path — does not
exist, and a test fails if it appears.

An adapter map *does* now exist at
`calibration/adapters/maps/reloc2_18_provisional.json`. It is deliberately not
the same thing: it declares `status: "provisional"`, describes which slots supply
*evidence* for a landmark rather than asserting slot identity, leaves the
unsettled slots unmapped, and is refused by the loader if it ever claims to be
verified. See [`CALIBRATION.md`](CALIBRATION.md) for the interim engine built on
it, and note that its groupings still rest on an **unsigned** first-pass review.

Also out of scope here: temporal processing, and the candidate-map leave-one-out
reprojection study (Phase-0 step 3) in the form the design doc describes.

## Exit criteria

Phase 0 exits when **all** of the following hold:

- every one of the 18 slots has a human-approved assignment in `slot_review.md`;
- the flip probe is coherent and explained;
- the reviewer has signed the worksheet.

If the slots cannot be assigned, escalate rather than proceed: resolve orientation
per clip, restrict the model to end-agnostic use, or retrain with a correct
`flip_idx`.

## Architecture

```text
contracts/                     stdlib-only isolation wall
  calibration_types.py         RawKeypointObservation, RawKeypointFrame, ModelProvenance
calibration/                   imports contracts + third-party only
  cli.py                       the single `inspect-model` verb
  detectors/base.py            KeypointDetector protocol — also the test seam
  detectors/yolo_pose.py       Ultralytics runner
  io/{video,records,provenance}.py
  tools/{sampling,overlays,flip_probe,review,inspect_model}.py
```

`contracts/` may import nothing but the standard library; `calibration/` may
never import the repo-root training or labeling scripts (`serve`,
`extract_frames`, `export_yolo`, `train_pose`, `validate_schema`,
`migrate_to_v3`, `pipeline_manifest`). Both rules are enforced by
`calibration/tests/test_isolation.py`.

`run_inspection()` takes an injected detector, so the entire tool is tested
without loading the 417 MB checkpoint — and a different model can be inspected
later without touching the orchestration.

## Tests

```bash
python -m unittest discover -s calibration/tests -p "test_*.py" -t .
```

```bash
python -m unittest discover -s contracts/tests -p "test_*.py" -t .
```

169 tests, no model download, no network, a few seconds. They cover the contract
invariants, record round-tripping, the ban on fabricated `(0,0)` rows, provenance
digests, deterministic sampling, overlays drawing below-gate points rather than
hiding them, the flip probe against planted `self` / `partner` / `incoherent`
detectors, the no-court path end to end, and the review page being offline and
non-committal.
