# Milestone 1 court-marking benchmark

Milestone 1 of
[`COURT_CALIBRATION_HYBRID_MARKINGS_PLAN.md`](COURT_CALIBRATION_HYBRID_MARKINGS_PLAN.md)
is complete as an **independently hand-labeled exploratory benchmark**. Its purpose
is to measure whether later hybrid calibration work has a large, consistent
advantage over the current keypoint-only baseline.

The original plan called for two multimodal-agent annotation passes. The frozen
synthetic precision test showed that approach was not reliable enough: the agent
passed median error but failed p95 and systematic signed-offset limits. Following
the plan's stop condition, no real-frame agent labels were accepted. A dedicated
blind human labeling tool was built and one human labeler completed all six
references instead.

Six selected frames and one labeler do not support production accuracy,
reliability, or promotion claims. Those require a larger, independently audited
gold set.

## Completion record

The machine-readable closeout is
[`benchmark/results/nba_m1_v1_human/milestone1.json`](../benchmark/results/nba_m1_v1_human/milestone1.json).
Milestone 1 is complete because:

- the six-frame scope and three conditional additions are locked;
- the annotation contract, NBA feature glossary, and painted-centerline
  convention are frozen in code;
- all six human references have complete annotated-or-skipped dispositions and
  pass schema, metadata, bounds, sample-count, and geometry validation;
- the references are independent of model predictions, homographies, and
  projected overlays;
- the failed agent gate and methodology decision are retained;
- confident- and probable-tier baseline reports are saved; and
- limitations and the decision not to expand yet are explicit.

## Locked scope

`benchmark/manifests/nba_m1_v1.json` contains one easy and one hard frame from
each supplied clip:

| clip | easy | hard |
| --- | --- | --- |
| `video_1` | 12 | 90 |
| `video_2` | 18 | 141 |
| `video_3` | 15 | 146 |

The tracked human references are in:

```text
benchmark/references/
├── nba_m1_v1_human.json
└── nba_m1_v1_human/
    └── <frame-id>.frame.json  # six files
```

The adjacent manifest records each annotation hash, source-image hash,
annotator, and annotated/skipped counts. Each frame has exactly one disposition
for all 14 glossary features.

`benchmark/manifests/nba_m1_v1_expanded.json` adds frames 60, 3, and 177,
respectively. That nine-frame ceiling remains locked but is **not selected for
Milestone 1**. The six core frames cover clean, occluded, graphics/tight, and
logo-heavy views and suffice to establish the baseline. Use the additions only
if a later hybrid comparison is borderline or exposes a concrete coverage gap;
do not expand merely to improve a result.

## Annotation contract and independence

A reference contains visible painted centerlines, not stripe edges or inferred
landmarks. Straight markings require at least four observed samples and arcs at
least six. Labels record visibility, uncertainty, explicit occlusion gaps,
optional visible intersections, and an explicit reason for every skipped
feature. Coordinates remain in the original 1280×720 frame.

The labeling workspace receives only:

- the six blind raw frames and their hashes;
- image dimensions;
- the NBA feature glossary and top-down reference diagram; and
- the output schema and visibility rules.

It does not expose model keypoints, calibration records, projected overlays,
homographies, synthetic truth, or failed agent annotations.

Enforcement lives in `contracts/annotations.py`, `benchmark/geometry.py`,
`benchmark/frames.py`, and the `/benchmark-label` save API. AST isolation tests
prevent benchmark frame preparation from importing calibration or detector
modules.

## Why agent consensus was replaced

The model-neutral prompts remain frozen for reproducibility:

- `benchmark/prompts/court-annotator.md` — `annotator-2.0.0`
- `benchmark/prompts/court-adjudicator.md` — `adjudicator-2.0.0`
- `benchmark/prompts/prompts.json` — versions and SHA-256 hashes

Verify them with:

```bash
.venv/bin/python -m benchmark.cli verify-prompts
```

The first candidate configuration was scored against planted synthetic truth:

| check | result | gate |
| --- | ---: | ---: |
| median error | 1.05 px, pass | ≤ 1.5 px |
| p95 error | 6.10 px, fail | ≤ 4 px |
| worst absolute family signed offset | 3.93 px, fail | ≤ 0.75 px |
| invented features | 0, pass | 0 |
| complete dispositions | pass | required |

The second annotation configuration produced structurally valid output but was
not scored after the first configuration triggered the stop condition. Real
agent passes and adjudication were prohibited. The complete result is retained
at
[`benchmark/results/nba_m1_v1_human/agent_synthetic_gate.json`](../benchmark/results/nba_m1_v1_human/agent_synthetic_gate.json).
This failure is evidence for the methodology change, not an outstanding
completion gate.

The consensus implementation remains available as experimental infrastructure,
but it did not contribute labels to this reference set.

## Human labeling UI

Prepare/export a local blind frame tree, then run:

```bash
.venv/bin/python serve.py \
  --benchmark engine_out/benchmark/m1_openai_gpt56_v1
```

Open `http://127.0.0.1:8000/benchmark-label`. The page provides a numbered
half-court map, zoom/pan, ordered line and arc samples, occlusion gaps, skip
reasons, optional intersections, autosaved drafts, and strict finalization.
Local work is isolated under:

```text
engine_out/benchmark/m1_openai_gpt56_v1/run/human/
├── drafts/
└── <frame-id>.frame.json
```

`engine_out/` is intentionally ignored. Finalized labels are promoted into the
tracked `benchmark/references/nba_m1_v1_human/` directory only after validation.

Validate a local completed pass against the locked frame export with:

```bash
.venv/bin/python -m benchmark.cli status \
  --frames engine_out/benchmark/m1_openai_gpt56_v1/frames \
  --pass-dir benchmark/references/nba_m1_v1_human \
  --expect-prompt-version human-labeler-1.0.0 \
  --expect-annotator-id human-pri
```

## Recorded baseline results

The six selected frame indices already existed in the per-clip `m1_confident`
and `m1_probable` runs, so calibration was not rerun. The scorer requires all
selected clips and retains calibration refusals rather than silently dropping
them.

Regenerate a report with:

```bash
.venv/bin/python -m benchmark.cli score \
  --reference benchmark/references/nba_m1_v1_human \
  --run engine_out/calibration/video_1/m1_confident \
  --run engine_out/calibration/video_2/m1_confident \
  --run engine_out/calibration/video_3/m1_confident \
  --out benchmark/results/nba_m1_v1_human/baseline_confident.json
```

Repeat with the three `m1_probable` directories. `--silver` remains a CLI alias
for compatibility with the abandoned consensus workflow.

Both tiers select the same transforms on these six frames and therefore have the
same result:

| depth band | samples | median px | p95 px | median ft | p95 ft |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0–19 ft | 220 | 5.96 | 8.70 | 0.875 | 1.643 |
| 19–30 ft | 124 | 5.91 | 11.35 | 0.588 | 1.507 |
| 30–47 ft | 6 | 14.24 | 30.86 | 3.028 | 7.780 |

Five frames are scored as `DEGRADED`; `video_3` frame 146 remains
`UNCALIBRATED_INSUFFICIENT_EVIDENCE` and is counted as unscored coverage. The
held-out-family median is 6.43 px. The worsening far-depth result quantitatively
confirms the extrapolation drift previously visible only in overlays, but the
far band has only six samples and must not be treated as a stable population
estimate.

Saved reports:

- [`baseline_confident.json`](../benchmark/results/nba_m1_v1_human/baseline_confident.json)
- [`baseline_probable.json`](../benchmark/results/nba_m1_v1_human/baseline_probable.json)

## Verification

```bash
.venv/bin/python -m unittest discover
```

The suite verifies contracts, blind frame handling, synthetic scoring, crop
conversion, prompt hashes, strict completeness, human UI export, consensus
isolation, baseline matching, and the tracked Milestone 1 artifacts.

## Explicit limitations

- Six deliberately selected frames from three clips are an exploratory benchmark.
- One human produced the references; no independent second-person audit exists.
- One of six baseline frames has no transform to score.
- The far-depth sample is especially small.
- These reports establish a baseline for future shadow-mode comparisons; they do
  not establish that a future hybrid is safe for production.
