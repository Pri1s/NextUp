# Fine-tuning and evaluating the court pose model

This is the step-by-step runbook for turning hand labels (see
[`LABELING.md`](LABELING.md)) into a fine-tuned YOLO pose model and judging
whether it's actually any good. Two scripts do the work:

```
export_yolo.py   dataset/labels/ → export/court_pose/  (YOLO pose format)
train_pose.py    export/court_pose/ → runs/pose/<name>/ (fine-tuned weights)
```

Everything here assumes you're in the project root with the venv active
(`source .venv/bin/activate`, or prefix commands with `.venv/bin/python`).

## 1. Export labels to YOLO format

```bash
python3 export_yolo.py
```

What it does:
- Reads every frame in `dataset/manifest.json` with `label_status: labeled`,
  pulls its JSON label from `dataset/labels/<clip_id>/<frame_id>.json`.
- Writes `export/court_pose/{images,labels}/{train,val}/` in Ultralytics YOLO
  pose format, plus `export/court_pose/data.yaml`.
- Assigns each frame to `train` or `val` **deterministically** — a SHA1 hash
  of the `frame_id`, not a random shuffle. That means re-running this after
  labeling more frames never reshuffles frames that are already assigned; the
  split only grows. Default split is 80/20 (`--val-frac 0.2`).
- Derives `flip_idx` (which keypoint indices swap under a horizontal flip)
  straight from `dataset/schemas/court_keypoints.v3.json`, so it can never
  drift out of sync with the schema.
- The bounding box for every frame is the full image (`--bbox full`, the
  default) — there's exactly one "court" object per image by construction, so
  box detection is not a meaningful thing to optimize or evaluate here. Ignore
  every `Box(...)` metric you see later; `Pose(...)` is the real signal.

Useful flags:

| Flag | Default | Meaning |
|---|---|---|
| `--val-frac` | `0.2` | Fraction of frames held out for validation |
| `--bbox` | `full` | `full` frame, or `hull` (padded box around visible points) |
| `--out` | `export/court_pose` | Output directory |

Re-run this **every time you label a new batch of frames**, before
retraining — it's cheap (just copies files) and always regenerates from
scratch (`export/` is wiped and rebuilt each run).

If it exits with `No labeled frames to export`, go label some frames first
(`python3 serve.py`, see LABELING.md).

## 2. Fine-tune

```bash
python3 train_pose.py
```

Defaults: starts from `yolo11n-pose.pt` (COCO-pretrained — only the
backbone/neck transfer, since COCO's pose head is sized for 17 human
keypoints and ours needs 22; Ultralytics reinitializes the head automatically
and logs `Overriding model.yaml kpt_shape=[17, 3] with kpt_shape=[22, 3]` when
it does), 100 epochs, `imgsz=960`, `batch=4`, early-stop `patience=30`, run
name `court_pose_v1` under `runs/pose/`.

Useful flags:

| Flag | Default | Meaning |
|---|---|---|
| `--data` | `export/court_pose/data.yaml` | Dataset to train on |
| `--model` | `yolo11n-pose.pt` | Starting weights — a COCO checkpoint, or `runs/pose/.../weights/best.pt` to keep fine-tuning an existing run |
| `--epochs` | `100` | Max epochs (early stop can end it sooner) |
| `--imgsz` | `960` | Input resolution |
| `--batch` | `4` | Batch size — raise this once you have more than a handful of val images |
| `--patience` | `30` | Stop if val metrics don't improve for this many epochs |
| `--device` | auto | Force `mps` (Apple Silicon GPU), `cpu`, or a CUDA index. Ultralytics sometimes auto-selects CPU even when MPS is available — pass `--device mps` explicitly to make sure you're using the GPU. |
| `--name` | `court_pose_v1` | Run name under `runs/pose/`. **Ultralytics auto-increments** (`court_pose_v12`, `v13`, …) if the name already exists rather than overwriting — pass a fresh `--name` on purpose, or note the actual folder it prints if you didn't. |

On an M-series Mac, ~30 images at `imgsz=960` with the nano model takes on
the order of 10–15 minutes for 100 epochs on `mps`.

When it finishes, it also writes `runs/pose/<name>/dataset_manifest.json` —
a record of exactly which frame_ids were `train` vs `val` for that run. This
is what makes the train/val badges show up later in the labeling UI (step 4).

## 3. Read the results

Everything lands in `runs/pose/<name>/`. The two things worth actually
opening:

**`results.csv`** — one row per epoch. The columns that matter:

- `metrics/precision(P)`, `metrics/recall(P)`, `metrics/mAP50(P)`,
  `metrics/mAP50-95(P)` — pose metrics on the val split. This is the number
  to track over time as you label more data. `mAP50-95` is the stricter,
  more informative one (averages over multiple keypoint-distance
  thresholds instead of just one loose one).
- Ignore the `(B)` (box) columns — see the note in step 1.

Quick way to check the final epoch's pose numbers without opening a
spreadsheet:

```bash
awk -F, 'END{printf "P=%s R=%s mAP50=%s mAP50-95=%s\n", $12,$13,$14,$15}' \
  runs/pose/court_pose_v1/results.csv
```

**`val_batch0_pred.jpg` vs `val_batch0_labels.jpg`** — a visual side-by-side
of predicted vs. ground-truth keypoints on a batch of val images. Open both
and eyeball them: are the predicted dots roughly where the hand-labeled ones
are? This catches problems a single aggregate number can hide (e.g. the model
consistently confusing two symmetric points, which the flip_idx derivation is
specifically there to prevent).

Other artifacts if you want them: `PosePR_curve.png` /
`PoseF1_curve.png` (precision/recall tradeoff curves), `results.png`
(all metrics plotted across epochs — useful for spotting overfitting: pose
loss still dropping on train but flat/rising on val), `train_batch*.jpg`
(sanity-check the augmented training batches look reasonable, e.g. keypoints
still land on the right court features after a flip).

Weights: `weights/best.pt` (best val performance during the run — this is
what you want) and `weights/last.pt` (final epoch, for resuming training).

### How to read the numbers

With only a few dozen labeled frames, treat all of this as noisy:

- A handful of val images means one wrong prediction swings `mAP50-95` by a
  lot. Don't over-interpret small changes between runs; look for a clear
  trend across several labeling batches, not a single before/after number.
- `mAP50` (loose threshold) will look better than `mAP50-95` (strict,
  averaged over thresholds) — that gap is expected, not a bug.
- If pose `mAP50-95` is going *down* as you add more epochs on the same
  data, that's overfitting on a tiny train set — lower `--epochs`, or
  prioritize labeling more frames over training longer on what you have.

## 4. Use the model, and see what it actually saw

Prefill new labels with the model you just trained:

```bash
python3 serve.py --model runs/pose/court_pose_v1/weights/best.pt
```

Two things this gets you:

- In the label view, points prefill from the model instead of starting
  blank (`R` re-predicts). Verify every point — treat prefill as a
  starting point, not ground truth.
- In both the triage grid and the label view, frames that were in that run's
  `train` or `val` split get a small colored badge (blue = train, purple =
  val) plus a `N trained-on` count in the header. This comes from the
  `dataset_manifest.json` written in step 2 — it's specific to whichever
  checkpoint you pass to `--model`, so it always reflects the actual run, not
  just "has a label." When judging how good the model really is from the
  prefill quality, trust what you see on **val**-badged frames more than
  **train**-badged ones — the model already memorized the train frames
  during fine-tuning.

## 5. Iterate

The intended loop, repeated as you label more:

```
label more frames (serve.py)
   → export_yolo.py           (re-export; split only grows, never reshuffles)
   → train_pose.py             (fine-tune again — either from scratch, or
                                 --model runs/pose/court_pose_v1/weights/best.pt
                                 to keep going from the last checkpoint)
   → check results.csv + val_batch0_pred.jpg vs the previous run
   → serve.py --model ...      (prefill the next labeling batch; prioritize
                                 clips with few/no labels for diversity)
```

Compare `results.csv`'s final-epoch pose `mAP50-95` across runs to see if
labeling more data is actually moving the needle. If it's flat despite more
labels, look at *what* you're adding — near-duplicate frames from the same
clip help less than frames from a new clip/court/angle (see LABELING.md's
"diversity over volume" guidance).

## Troubleshooting

- **`Prefill model has kpt_shape [...]; the schema requires 22 keypoints`**
  — you pointed `--model` at something not trained on this schema (e.g. a
  stray COCO checkpoint, or an old reloc2 model). Only your own
  `runs/pose/.../weights/*.pt` checkpoints qualify.
- **No train/val badges show up in the UI** — either you started `serve.py`
  without `--model`, or the checkpoint's run directory has no
  `dataset_manifest.json` (only runs produced by the current `train_pose.py`
  have one; older ad-hoc runs won't).
- **Training picked `CPU` instead of your GPU** — pass `--device mps`
  explicitly; Ultralytics' auto-selection doesn't always pick MPS on Apple
  Silicon even when it's available.
- **`No labeled frames to export`** — label some frames first, then rerun
  `export_yolo.py`.
