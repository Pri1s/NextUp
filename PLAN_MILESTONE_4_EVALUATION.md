# Milestone 4 — Evaluation and Independent Audit (Agent Brief)

**Repo:** `/Users/pri/Projects/Git/game-analyzers/NextUp`
**Read first:** `PLAN_MILESTONE_4.md` §10b (what broke and how it was fixed) and §10c (what is
and is not established). `docs/COURT_CALIBRATION_HYBRID_MARKINGS_PLAN.md` is the design
authority for the milestone as a whole.

**You are not here to finish Milestone 4.** You are here to produce the evidence that decides
whether it should be finished, killed, or reworked — and to check that the machinery producing
that evidence is trustworthy. The decision itself belongs to the project owner.

---

## Context

The calibration engine fits a homography from five model landmarks that span **19 feet of
court depth and no more**, three of them collinear on `x = 0`. Milestone 4 adds a second
candidate transform fitted against painted court markings, to extend that constraint deeper.

The code is implemented and tested: 647 passing, 8 skipped. It recovers a planted transform to
0.02 px, is invariant to sliding samples along a line and to sample density, is byte-for-byte
deterministic, never promotes a challenger, and leaves `calibrations.jsonl` untouched.

**Every one of those numbers comes from planted or synthetic evidence.** The entire benefit
rests on one marking:

| feature | court-x span (ft) |
| --- | --- |
| `lane_edge_far` / `lane_edge_near` | 0 → 19.0 |
| `free_throw_line` | 19.0 (constant) |
| `three_point_corner_far` / `_near` | 0 → 14.26 |
| `three_point_arc` | 14.26 → **29.0** |
| `free_throw_circle_far_half` | 19.0 → 24.92 (**held out for validation**) |

Once the free-throw circle is reserved as the independent check, **`three_point_arc` is the
only fitted primitive carrying any depth past 19 ft.** If it cannot be extracted from
broadcast footage, the milestone has no mechanism of benefit, however correct its arithmetic.

Nobody has ever run the marking extractor on real video. That is the gap you are closing.

---

# Phase 1 — Viability. Do the markings exist in real footage?

**This phase needs no human annotations, no scorer, and no refiner.** It reads counts off
records the extractor already produces. It is the cheapest question available and it can kill
the milestone outright, so it comes first.

## 1.1 — P0: a fresh v2 calibration run

Every stored run in `engine_out/calibration/` was produced under layout hash `90f53fdc…`;
the current v2 layout hashes to `a57d0b4c…`, and `benchmark.metrics.score_calibration_entry`
refuses the mismatch. So a fresh run is a hard prerequisite — and it is a prerequisite for
Milestone 3's own gate too, not something invented for M4.

```bash
python -m calibration.cli calibrate-frames \
  --weights models/court_keypoint_detector.pt \
  --video input_videos/video_1.mp4 \
  --tiers confident \
  --num-frames 9999 \
  --marking-shadow \
  --run-id m4_eval_v2 \
  --out engine_out/calibration
```

Repeat for `video_2.mp4` and `video_3.mp4`. Run the **full frame sets** — 117 + 174 + 243 =
534 frames — not a sample, so temporal jitter stays measurable later and the availability
counts are not a lucky subset.

**`--num-frames 0` means zero frames, not all.** `even_indices` returns an empty list for 0 and
clamps anything above the frame count, so a large number is the idiom for "everything" — the
run would otherwise complete successfully having calibrated nothing. **Assert the frame count
in each `run.json` before continuing** (117 / 174 / 243); an empty or short run is the easiest
way to produce confident, meaningless availability numbers.

Notes:

- `--marking-shadow` in the same pass gives you the M3 evidence without a second decode.
- The checkpoint hash gate should pass unaided: the real checkpoint and
  `calibration/adapters/maps/reloc2_18_provisional.json` both carry `f6263105…`.
- Expect this to take a while — YOLO inference plus a multiscale ridge extractor over 534
  frames at 1280×720. It is not hung.
- `calibrations.jsonl` is asserted byte-identical with and without `--marking-shadow`
  (`calibration/tests/test_hybrid_cli.py`), so the shadow pass cannot contaminate the baseline.

**Sanity-check before continuing:** how many of the 534 frames are `DEGRADED` with 5 inliers
and only `weak_conditioning:*` reasons? That is the seed class the shadow path accepts, and it
is the ceiling on everything downstream. On the six locked benchmark frames the old run gave
5 of 6; if the full-clip rate is wildly different, say so before going further.

## 1.2 — The availability count

Use **one** extraction variant for this — `ridge94-balanced`, the default. Do not sweep all
nine yet; a sweep is only worth it if the default is marginal.

Read directly off `marking_shadow/records.jsonl`. Everything you need is in
`ShadowFeatureSupport` and `ShadowFamilySupport` on each `ShadowMarkingFrame`. Do not write a
scorer, do not load references. Report, across all eligible frames:

| metric | why it matters |
| --- | --- |
| per feature: % of frames with ≥1 **accepted** fragment | basic detectability |
| **`three_point_arc`**: % of frames accepted, and the distribution of its accepted support depth | this is the whole mechanism of benefit |
| **`free_throw_circle_far_half`**: % of frames accepted, and the distribution of accepted sample counts | the independent check; the gate needs ≥12 samples |
| per family: % of frames where `ready` is true | feeds `supported_families` |
| % of frames with **≥2 families ready** | the `hybrid.independent_families` gate |
| ambiguous and rejected fragment counts, with reason codes | tells you *why* something is missing |

Cross-check the arc's depth against the layout rather than assuming: the corner straights stop
at 14.26 ft and the lane at 19.0, so only genuine arc paint past the break can satisfy
`hybrid.depth_beyond_19ft ≥ 24.0 ft`.

## 1.3 — Report, and stop

Produce a short written finding with the numbers above. Then **stop and hand back.** Do not
proceed to Phase 2 or 3 without an explicit decision.

What the outcomes mean — offered as a reading, not as a rule you may apply yourself:

- **Arc rarely accepted** → the milestone has no mechanism of benefit. That is a kill, and the
  honest close-out is a kill note rather than a completion.
- **Arc found but free-throw circle rarely available** → the milestone becomes *unfalsifiable*:
  no independent check exists, so you could not demonstrate the hybrid works or that it does
  not. This is worse than a clean failure and must be resolved before any comparison is run.
  `PLAN_MILESTONE_4.md` §12 lists contingencies (hold out a different feature, or rotate per
  frame); all of them need this data first.
- **Both adequately available** → the milestone is worth evaluating, and Phases 2 and 3 follow.

**Do not weaken a threshold to obtain a pass, and do not declare the milestone viable.**
Report what you measured and what it implies.

---

# Phase 2 — Independent audit of the defect fixes

**Only if Phase 1 says the milestone is alive.** Assume nothing in `PLAN_MILESTONE_4.md` §10b
is correct; it was written by the agent that made the fixes.

Six defects were found and fixed. Two further items were *reported* as defects and later
retracted. Verify both directions — a retraction can be as wrong as a claim.

## 2.1 — Confirm each defect was real and each fix holds

For each row in §10b, reinstate the original behaviour and confirm the named test fails, then
restore and confirm it passes. Previously observed: inherited `mu` → 2 failures; `prior_weight`
0.25 → 10 failures; planted `benchmark` import → 1; planted `SELECT_CHALLENGER` → 1; removed
gate threshold → 23. If any of those does **not** reproduce, the test is not guarding what it
claims.

## 2.2 — Re-examine the two retractions

- **The tangent change** (secant → full template segment) is claimed to be principled but not
  behaviourally load-bearing, with the original "it fixed the arc" attribution retracted as an
  artifact of measuring with `max_outer=40`. Verify independently. If it *does* matter under
  some condition — a coarser template, a shorter fragment, a foot landing pathologically — that
  condition belongs in a test.
- **`raw_perpendicular_error_px`** is claimed to have been dead code, with production already
  measuring residuals freshly via `_fresh_feature_measurements`. Both were consolidated into
  `residuals.measure_feature_residuals`. Confirm the surviving implementation is the one that
  was always in use, and that the consolidation did not change any recorded number.

## 2.3 — Hunt for green-and-vacuous tests

This codebase produced two of them in one session — CLI isolation tests that compared two
empty records, and residual tests that exercised a function production never called. Both were
green. Audit specifically for:

- tests whose fixture produces no evidence, so the assertion is trivially satisfied;
- tests calling a helper that nothing in `calibration/hybrid/` or `benchmark/` invokes;
- `assertRaises` that fires for a different reason than the one named — construct the invalid
  object directly and read the message;
- thresholds loose enough that the pre-fix behaviour would also have passed.

The strongest single check: for each test, ask what production change would make it fail, and
confirm by making that change.

## 2.4 — Sanity-check the numbers that were never independently reproduced

Planted recovery 0.0207 px, sliding invariance 2.8e-4 px, density invariance 7e-4 px, arc
chord error falling exactly 4.00× from 201→401 samples. Reproduce them from a clean checkout.
A scratch probe exists at `calibration/tests/test_hybrid_residuals.py`; prefer running it to
rewriting it, but confirm it measures what its docstrings say.

---

# Phase 3 — The evaluation itself

**Only if Phases 1 and 2 both pass.** This is where the six human-annotated frames get spent,
and they are the scarce resource — do not burn them on questions Phase 1 already answered.

1. **Freeze exactly one extraction variant** before scoring anything, out of fold. Every M4
   record carries `shadow_config_hash`; the final report must show a single value.
2. **`benchmark.cli score-extraction`** against `benchmark/references/nba_m1_v1_human/` — is the
   paint we found where the human says the line is, and did we ever confidently label the wrong
   marking (`wrong_identity_accepts`)?
3. **`benchmark.cli score-hybrid`** for the keypoint-vs-hybrid comparison and the §9 acceptance
   block. Results go under `benchmark/results/nba_m4_v1/`; never overwrite the M1 or M3 reports.
4. **Run the refiner's diagnostics over all 534 frames** for the distributions that need no
   references — gate pass rates, abstention reasons, leave-one-primitive stability, probe shift,
   and held-out error. Held-out error is the strongest internal signal available and it is
   criterion (b) of the kill condition, so look at it first.
5. **Evaluate the pre-registered kill condition** in `PLAN_MILESTONE_4.md` §10. Report the
   verdict against it. Do not restate the thresholds in your own words and do not adjust them.

**The honest limitation, which must appear in the report:** only five of the six locked frames
are eligible, split 2/2/1 across clips, so a per-clip fold has n = 1. That set can detect a
large consistent effect and a gross failure. It cannot settle a borderline result. If the
outcome lands near the 25% bar, the plan names three more frames to annotate rather than
squinting at five.

---

## Rules

1. **Do not weaken any threshold to obtain a pass.** If a criterion cannot be met, that is the
   finding. Report it.
2. **Do not decide the kill.** Produce numbers and state what they imply. An agent motivated to
   complete a task is the wrong entity to judge whether the task should continue.
3. **Change one thing at a time and re-measure.** The most expensive defect in this milestone
   was invisible until two other fixes cleared the noise in front of it, and an improvement was
   attributed to the wrong change because three were made at once.
4. **A green test is not evidence until you know what would make it fail.**
5. **Never write to `calibrations.jsonl`,** and never let anything under `calibration/hybrid/`
   import `benchmark`. Both are enforced; keep it that way.
6. **`SELECT_CHALLENGER` must not appear in `calibration/hybrid/`.** Milestone 4 is shadow mode;
   selection is Milestone 5's decision and is gated on evidence that does not exist yet.
7. **Report failures with their output.** If a run errors, quote it. If a step is skipped, say
   so and why.

## Deliverables

- A short written finding for Phase 1 with the availability numbers, and an explicit stop.
- An audit note for Phase 2: which defects reproduce, which retractions hold, any vacuous tests
  found, any numbers that did not reproduce.
- For Phase 3, the reports under `benchmark/results/nba_m4_v1/` plus a verdict against the
  pre-registered kill condition and a statement of the evidence base's limits.

## Verification

```bash
python -m unittest discover -s . -p 'test*.py'
```

647 passing / 8 skipped is the baseline. Any change you make must hold it, and the walls must
stay green throughout:

```bash
python -m unittest calibration.tests.test_isolation benchmark.tests.test_blindness \
  calibration.tests.test_hybrid_refiner
```
