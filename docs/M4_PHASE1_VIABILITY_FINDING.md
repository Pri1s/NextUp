# Milestone 4 — Phase 1 Viability Finding

**Date:** 2026-08-02
**Scope:** Phase 1 only (viability). Phases 2 and 3 were **not** started.
**Question:** do the painted markings M4 needs exist in real broadcast footage?
**Answer:** the three-point arc does. The held-out free-throw circle does not.

---

## 1. Provenance of the evidence

Fresh v2 calibration run, three clips, full frame sets, one command per clip:

```
python -m calibration.cli calibrate-frames \
  --weights models/court_keypoint_detector.pt --video input_videos/video_N.mp4 \
  --tiers confident --num-frames 9999 --marking-shadow \
  --run-id m4_eval_v2 --out engine_out/calibration
```

All three exited 0. Frame counts asserted against `run.json` before any analysis:

| clip | frames in `run.json` | expected |
| --- | --- | --- |
| video_1 | 117 | 117 |
| video_2 | 174 | 174 |
| video_3 | 243 | 243 |
| **total** | **534** | **534** |

Single extraction variant (`ridge94-balanced`, the default). One config hash and one layout
hash across all 534 records:

- `config_hash = be3cbaf1c36d3848…`
- `layout_hash = a57d0b4ce91ce26d…` — the current v2 layout, so `score_calibration_entry`
  will accept these records. The stale `90f53fdc…` runs are superseded.

Numbers below are read directly off `marking_shadow/records.jsonl` via `ShadowFeatureSupport`
and `ShadowFamilySupport`. No scorer, no refiner, no human references, no annotations spent.
The one derived quantity is arc depth, obtained by back-projecting accepted arc samples
through the frame's own `h_image_to_court` — the same transform the association step used to
project the template.

## 2. The seed class

| | frames | share |
| --- | --- | --- |
| baseline `DEGRADED` | 457 | 85.6% |
| `UNCALIBRATED_INSUFFICIENT_EVIDENCE` | 59 | 11.0% |
| `FAILED_IMPLAUSIBLE` | 18 | 3.4% |
| **eligible (DEGRADED, 5 inliers, `weak_conditioning:*` only)** | **444** | **83.1%** |

The brief asked whether the full-clip rate resembles the 5-of-6 seen on the locked frames.
It does — 83.1% against 83.3%. The seed class is healthy and is **not** the constraint.
Everything below is measured over these 444 eligible frames.

## 3. Per-feature acceptance

| feature | % frames with ≥1 accepted | accepted frags | ambiguous | rejected |
| --- | --- | --- | --- | --- |
| `three_point_arc` | **46.2%** | 275 | 9639 | 3446 |
| `three_point_corner_near` | 42.3% | 447 | 12615 | 3666 |
| `lane_edge_near` | 20.9% | 112 | 14446 | 4184 |
| `lane_edge_far` | 18.7% | 96 | 9375 | 2994 |
| `free_throw_line` | 2.0% | 10 | 6666 | 3029 |
| `three_point_corner_far` | 0.5% | 2 | 2005 | 492 |
| **`free_throw_circle_far_half`** | **0.2%** | **1** | 10848 | 3444 |
| all seven out-of-scope decoys | 0.0% | 0 | — | — |

Zero decoy features were ever accepted across 534 frames. That is the one clean result here:
the associator's scope filter does not leak.

## 4. `three_point_arc` — the mechanism of benefit

**Accepted on 205 of 444 eligible frames (46.2%).** Where accepted, its depth is real:

| quantity | n | min | p25 | median | p75 | max |
| --- | --- | --- | --- | --- | --- | --- |
| max court-x depth (ft) | 205 | 14.7 | 22.9 | **24.9** | 28.7 | 29.4 |
| accepted samples/frame | 205 | 3 | 4 | 6 | 9 | 40 |
| samples past 19 ft | 205 | 0 | 4 | 5 | 8 | 31 |

Cross-checked against the layout rather than assumed. Measured spans:
`three_point_corner_far/near` 0 → 14.26 ft, `lane_edge_*` 0 → 19.0 ft,
`three_point_arc` 14.26 → 29.0 ft. A max depth of 24.9 ft cannot come from any primitive
except genuine arc paint past the break, so the depth is attributable.

**115 of 444 frames (25.9%) carry accepted arc reaching ≥ 24.0 ft**, the
`hybrid.depth_beyond_19ft` threshold.

So the mechanism of benefit exists in real footage. It is sparse — a median of 6 accepted
samples per frame is thin — but it is present and it reaches where the milestone needs it.

## 5. `free_throw_circle_far_half` — the independent check

**Accepted on 1 of 444 frames (0.2%), with 4 samples.**

`hybrid.held_out_available` requires ≥ 12 samples. **0 of 444 frames reach it (0.0%).**

This is not "rarely available". It is unavailable. Across 534 frames of real footage the
held-out validation feature is accepted once, and that one frame still falls a factor of
three short of the sample floor.

## 6. Family readiness

| family | % frames ready | leading not-ready reasons |
| --- | --- | --- |
| `lane` | **0.5%** | `lane_join` 439, `free_throw_line_missing` 432, `lane_edge_missing` 290 |
| `three_point` | **0.0%** | `arc_support` 441, `corner_straight_missing` 251, `corner_join` 251 |
| `free_throw_circle` | **0.0%** | `insufficient_tangent_sweep` 441, `held_out_half_missing` 440 |
| `boundary` / `midcourt` / `restricted_area` | 0.0% | `family_out_of_scope` (by design) |

**Frames with ≥ 2 families ready: 0 of 444 (0.0%).**
Excluding the held-out family, still 0 of 444.

`supported_families` in `hybrid/diagnostics.py:123` derives straight from
`family_support[].ready`, so `hybrid.independent_families ≥ 2` fails on **every frame in the
corpus**. No frame in 534 can pass the gate block as configured.

Note `three_point` fails `arc_support` on 441 of 444 frames — including the 205 where the arc
*was* accepted. Accepted arc fragments are consistently too short to clear
`max(arc_support_px × scale, 0.15 × arc_visible)`.

## 7. Fragment dispositions — why things are missing

Over 444 eligible frames, 35,108 fragments:

| status | count | share |
| --- | --- | --- |
| `AMBIGUOUS` | 26,064 | **74.2%** |
| `REJECTED` | 8,101 | 23.1% |
| `ACCEPTED` | 943 | **2.7%** |

`low_margin` fires on **all 26,064** ambiguous fragments. The median `uniqueness_margin` is
**0.0** against a 0.20 accept bar.

Drill-down on fragments whose top-scoring alternative is a critical feature:

| target | fragments scoring it top-1 | accepted | median margin |
| --- | --- | --- | --- |
| `three_point_arc` | 3648 | 275 (7.5%) | 0.1 |
| `free_throw_circle_far_half` | 1517 | 1 (0.07%) | 0.0 |
| `free_throw_line` | 469 | 10 (2.1%) | 0.0 |

**The paint is being found. The associator cannot say what it is.** 1,517 fragments scored
the free-throw circle as their best match and one survived. The bottleneck is discrimination
in `MarkingAssociator._score`, not extraction — its weighted sum of seven [0,1] quality terms
separates competing primitives by 0.0 at the median, so `min_uniqueness_margin = 0.20`
rejects nearly everything. This is a characterisation of the failure, not a recommendation
to move the threshold.

## 8. The locked benchmark frames

| frame | shadow status | eligible | arc accepted | FT circle | families ready | distinct fittable features |
| --- | --- | --- | --- | --- | --- | --- |
| `video_1_000012` | OK | yes | 0 | 0 | none | 0 |
| `video_1_000090` | OK | yes | 0 | 0 | none | **2** |
| `video_2_000018` | OK | yes | 0 | 0 | none | 0 |
| `video_2_000141` | OK | yes | 3 | 0 | none | **2** |
| `video_3_000015` | OK | yes | 0 | 0 | none | 1 |
| `video_3_000146` | ABSTAINED | no | 0 | 0 | none | — |

Five of six eligible, as §10 predicted. A challenger requires ≥ 2 distinct fitted features
(`min_fitted_features = 2`, `orchestrator.py:174`), so **2 of the 5 eligible locked frames
would produce a challenger at all.** Corpus-wide that rate is 182 of 444 (41.0%).

## 9. Against the pre-registered kill condition

Not a verdict — that is the owner's call. Reporting mechanically what Phase 1 already
determines about §10's criteria:

- **(a)** "fewer than 4 of the 5 eligible frames produce a challenger" — **2 of 5 produce
  one.** Criterion (a) is met on this evidence, before Phase 3 is run.
- **(b)** held-out median error comparison — **cannot be evaluated.** 0 of 444 frames reach
  the 12-sample held-out floor. The criterion the plan calls "the honest core" has no data
  under it anywhere in the corpus.
- **(c), (d)** require challengers and gate output; not evaluated in Phase 1.

## 10. A defect found while counting

**`calibration/markings/association.py:268–275`** — the `FREE_THROW_CIRCLE` readiness
expression has three conjuncts:

```python
ready = bool(half) and circle_support >= max(...) and sweep >= 0.25
if not half: reasons.append("held_out_half_missing")
if sweep < 0.25: reasons.append("insufficient_tangent_sweep")
```

There is no reason code for the `circle_support` conjunct. When the half is accepted with
adequate sweep but support below threshold, this builds
`ShadowFamilySupport(ready=False, reasons=())`, which the contract rejects
(`hybrid_types.py:389`). The `ValueError` is not caught at family level — it propagates and
fails the whole frame's extraction, discarding every fragment on it.

Reproduced directly:

```
ShadowFamilySupport(family=FREE_THROW_CIRCLE, ready=False, ..., reasons=())
  -> ValueError: unready family support needs a reason
```

**Cost: 3 of 534 frames** came back `EXTRACTION_FAILED` with exactly this reason
(`video_1_000027` among them). The impact is small in count but adverse in selection: the
frames it destroys are precisely those where the held-out circle came *closest* to being
usable — accepted, well-swept, marginally short on support. The one measurement M4 most needs
is the one this bug is biased against recording.

I have not fixed it. It is a one-line addition, but changing extraction changes
`config_hash` and every number above, so it belongs to a deliberate re-run, not to this
audit.

## 11. What this establishes, and what it does not

**Established.** Arc paint past the 14.26 ft break is extractable and associable from real
broadcast footage at a 46.2% frame rate, reaching a median 24.9 ft and clearing the 24 ft
depth bar on 25.9% of frames. M4's premise — that there is something out there to fit — is
not refuted.

**Established negatively.** The held-out free-throw circle is not available at any usable
rate (0.0% of frames reach the sample floor), and no frame in 534 has two ready families.

**Not established.** Nothing here says a hybrid fit would be *better*. Availability is
necessary, not sufficient. No transform was fitted, no residual measured, no annotation spent.

**The consequence, stated plainly.** With the held-out feature unavailable corpus-wide, M4
has no independent check available — the milestone is currently **unfalsifiable** in exactly
the sense §10c and §12.3 anticipated. That is the outcome the brief describes as worse than a
clean failure, and §12.3's contingencies (hold out `THREE_POINT_CORNER_NEAR`, or rotate the
held-out feature per frame) now have the data they were waiting on. Both contingencies are
weaker than the original design and both need a decision before any comparison is run.

## 12. Stop

Phase 1 ends here, as instructed. Phase 2 (defect audit) and Phase 3 (evaluation) are not
started and no annotations have been spent. No threshold was weakened, no code changed, and
`calibrations.jsonl` under the pre-existing `m1_*` runs was not touched — the new run wrote
only to `engine_out/calibration/*/m4_eval_v2/`.

Test suite unchanged at **647 passing, 8 skipped**.
