# Interim Hybrid Court Calibration — Architecture and Implementation Plan

**Status:** Milestone 1 complete; Milestones 2–6 remain planned.
**Scope:** The existing NBA half-court calibration engine and the intended NBA source clips under `input_videos/video_1.mp4`, `video_2.mp4`, and `video_3.mp4`. Unrelated sample-frame datasets are outside the evidence base and validation scope of this plan.
**Priority:** Accuracy and false-accept prevention over calibration coverage or runtime.

## 1. Executive recommendation

Build a **model-seeded, rejectable court-marking refinement** while preserving the existing keypoint-only method unchanged as the baseline and fallback.

For a frame with a plausible keypoint calibration:

1. Detect unlabeled painted-marking evidence.
2. Use the keypoint transform to determine which court markings that evidence could represent.
3. Jointly refine a separate hybrid candidate.
4. Validate it against evidence not used for fitting.
5. Select it only when it demonstrably beats or matches the keypoint candidate.

Do not begin with a production CV-only calibrator. The available NBA footage contains large logos, floor text, players, crowds, stanchions, advertisements, and screen graphics. Without the model's semantic seed, generic line detection has too many plausible but incorrect explanations.

A hybrid approach is likely to improve real accuracy on ordinary half-court frames, but this is a hypothesis to prove in shadow mode—not an assumption. If painted-marking extraction cannot achieve high precision, retaining model-only calibration is safer than relaxing the gates.

## 2. Evidence from the current project

### 2.1 Existing engine strengths

The current engine already provides a strong point-calibration foundation:

- MAGSAC/RANSAC fitting in `calibration/estimator.py`;
- explicit failure states and fallback behavior;
- point reprojection, leave-one-out, geometry-span, scale, and convexity checks in `calibration/quality.py`;
- full projected-marking overlays in `calibration/viz.py`;
- clean separation among the detector, provisional adapter map, court layout, estimator, and quality policy;
- deliberate rejection of ambiguous slots 3/12 and unusable slots 4/5/10/11.

Those behaviors must be preserved.

The current estimator is point-only. Its final refit is unweighted, and sampled line pixels must not be inserted as hundreds of ordinary point correspondences. Hybrid fitting needs a separate primitive-balanced refinement layer.

### 2.2 Current calibration limitation

The stored calibration runs contain 72 attempts:

- 61 usable, all `DEGRADED`;
- 8 uncalibrated for insufficient evidence;
- 3 rejected as implausible.

Every usable result has only 19 feet of court-depth support. The five confident model landmarks consist of:

- three points on the baseline; and
- two free-throw-line/lane corners.

This explains why point reprojection can be below one pixel while projected markings farther from the baseline visibly drift. Representative outputs include:

- `engine_out/calibration/video_1/interim/reprojected/frame_000070.jpg`;
- `engine_out/calibration/video_2/interim/reprojected/frame_000119.jpg`;
- `engine_out/calibration/video_3/interim/reprojected/frame_000207.jpg`.

The model and CV signals are complementary:

- the model supplies approximate identity, visible end, near/far side, and a safe starting transform;
- painted markings supply many geometric observations and extend depth beyond the current 19-foot span.

They are not fully independent because they inspect the same image. Their confidence scores must not simply be multiplied.

### 2.3 Available NBA footage

The intended calibration clips are short 1280×720 NBA broadcast sequences:

- `video_1`: 117 frames, 3.9 seconds;
- `video_2`: 174 frames, 5.8 seconds;
- `video_3`: 243 frames, 8.1 seconds.

Together they contain several difficult sources of false evidence:

- large center-court and baseline logos;
- text painted on and around the floor;
- players and officials obscuring markings;
- stanchions, benches, crowds, and advertisements;
- screen-space score graphics crossing the floor;
- motion blur during camera pans.

They are sufficient for a proof of value but not for a broad production false-accept claim.

### 2.4 Probable midcourt slots

Including probable slots 6/7 can provide excellent depth when they are genuinely visible. Existing predictions produce seven `OK` results on `video_1`, but:

- they provide no improvement on `video_2`, which never shows midcourt;
- some transition frames fail held-out checks;
- they do not help normal zoomed half-court frames.

They should remain optional hypotheses that CV may corroborate, not be promoted globally.

## 3. Approach comparison

| Approach | Assessment |
| --- | --- |
| Current model only | Semantically useful and already well guarded, but weak beyond 19 feet. Preserve as baseline and fallback. |
| CV only | Potentially useful later, but too vulnerable to false associations for the first implementation. |
| Model-seeded hybrid | Strongest practical automatic interim approach, provided it rejects uncertain or conflicting marking evidence. |
| Agent-created reference annotations | Suitable for shadow-mode benchmarking and substantially reduces manual busy work. |

If only the three short clips matter and human intervention is acceptable, manually spread landmarks would be the most accurate immediate option. For a reusable automatic pipeline, model-seeded marking refinement is the stronger practical direction.

## 4. Markings to prioritize

### 4.1 Full three-point marking

This is the highest-value initial signal.

- It is long and generally visible in the supplied NBA clips.
- Its curved shape is more distinctive than a generic straight segment.
- On an NBA court it reaches roughly 29 feet from the baseline, versus the current 19-foot evidence span.
- The model's labeled three-point/baseline point identifies one end of the marking.

Treat the corner straights and arc as one court primitive while allowing only part of it to be visible.

### 4.2 Lane rectangle and free-throw line

These are the safest straight markings to identify because the model already establishes:

- the baseline;
- one lane edge;
- the free-throw line.

Detected lane geometry can:

- sharpen both court vanishing directions;
- add support for the missing near lane edge;
- validate model-derived line identities;
- improve near-court precision.

It does not materially extend depth by itself, so lane-only CV must not replace the keypoint result.

### 4.3 Midcourt-side half of the free-throw circle

The outer semicircle reaches 25 feet from the baseline and provides curved evidence independent of the straight lane geometry.

For the first implementation, reserve it as a held-out validation marking instead of fitting against it. This supplies an entire feature family with which to test the proposed hybrid transform.

### 4.4 Midcourt line plus center circle

These provide the strongest depth coverage at 47 feet, but should be added later because:

- they appear mainly during transition frames;
- center-court logos create strong false edges;
- the current baseline landmarks often disappear in those frames.

Never trust a candidate midcourt line by itself. Require center-circle support, temporal support, or both.

### 4.5 Lower initial priority

- **Restricted-area arc:** small, near the baseline, and frequently occluded. Useful as held-out evidence, not as an initial driver.
- **Baseline and sidelines:** useful when clear, but often cropped or confused with non-court boundaries.
- **Unclassified generic Hough segments:** too ambiguous to drive calibration directly.

## 5. Proposed architecture

### 5.1 Preserve the existing keypoint candidate

Run the existing confident-tier calibration unchanged and retain it as `H_keypoint`.

For the first version, attempt hybrid refinement only when:

- `H_keypoint` is usable and physically plausible;
- at least five confident pooled landmarks are present;
- the primary degradation is weak depth conditioning.

Do not initially use hybrid CV to rescue failed or exactly four-point frames.

The current keypoint path must remain callable independently, with unchanged behavior when CV is disabled, unavailable, or rejected.

### 5.2 Detect unlabeled marking evidence

Extract narrow painted-marking centerlines or ridges using polarity-independent image cues such as:

- paired gradients;
- plausible local marking width;
- continuity;
- tangent direction;
- local color and contrast consistency.

Restrict extraction to a conservative estimated court area. Accept optional exclusion masks for:

- players and officials;
- stanchions;
- score bugs and other broadcast graphics;
- known occluded regions.

The detector should output image support, direction, confidence, and uncertainty—but no court identity.

This keeps the detector replaceable. A classical OpenCV implementation may be used first, and a learned semantic marking detector may replace it later without changing the estimator or layout.

### 5.3 Associate evidence using the model seed

Project the known layout markings through `H_keypoint` and create search corridors.

The current labeled points establish approximate locations for:

- the baseline;
- the far lane edge;
- the free-throw line;
- the far start of the three-point marking.

For each detected fragment, require agreement in:

- distance to the projected primitive;
- tangent or curvature;
- approximate court location after rough rectification;
- expected visible extent;
- relationships to nearby markings;
- uniqueness versus alternative identities.

If two identities remain plausible, reject the fragment.

CV association must never promote or reinterpret the intentionally ambiguous model slots. CV evidence stands on its own; it does not turn slots 3/12 into fixed landmarks.

### 5.4 Fit a separate hybrid candidate

Jointly refine a copy of the keypoint transform using:

- robust point residuals for confident model landmarks;
- perpendicular point-to-line residuals for straight markings;
- nearest point-to-curve residuals for arcs;
- a bounded transform change from `H_keypoint`.

A detected segment's endpoints are **not known court points**. They only delimit the portion of the line seen in the image. Every sampled pixel constrains distance perpendicular to the assigned marking; its position along that marking remains unknown.

Only a visibly observed junction between two already identified markings should be eligible to become a point landmark.

Normalize contributions per primitive and spatial region. Hundreds of correlated pixels from one line must not overwhelm five semantic landmarks or count as hundreds of independent observations.

Use a stable transform parameterization, such as projected canonical control points with convexity constraints, rather than directly optimizing raw homography coefficients without bounds.

### 5.5 Validate and select

Retain `H_keypoint` and `H_hybrid` as separate candidates.

Allow the hybrid to replace the keypoint result only when:

- at least one accepted marking extends materially beyond 19 feet;
- at least two independent marking families support it;
- no accepted fragment has an ambiguous identity;
- confident model points remain within a calibrated tolerance;
- a held-out marking family aligns;
- removing any one fitted primitive does not cause a large transform change;
- existing physical and geometric quality gates pass;
- projected canonical court points move by plausible amounts;
- the hybrid has a clear validation advantage.

If the candidates tie or conflict, choose `H_keypoint`.

If a future model supplies a well-spread `OK` calibration, retain it by default unless the hybrid shows an independently measured improvement.

### 5.6 Modularity for the future N/S-aware model

The hybrid engine should consume generic semantic landmark evidence, not raw model slots. The current provisional adapter and a future N/S-aware adapter should both feed the same correspondence interface.

The marking detector remains unlabeled and model-independent. The marking associator consumes:

- a candidate transform;
- a court layout;
- unlabeled marking evidence.

Absolute north/south identity remains outside marking geometry. Painted lines are symmetric and cannot resolve it reliably.

### 5.7 Normative marking-coordinate convention

An annotated marking coordinate means the **visual centerline of the painted stripe**.

- Straight-line samples lie midway between the two visible stripe edges.
- Arc and circle samples lie along the middle of the painted curve.
- A painted-line intersection means the intersection of the two marking centerlines.
- A segment endpoint remains only the end of visible support; it is not a known court landmark.
- If blur, occlusion, glare, or a filled paint region makes the centerline uncertain, the annotation records uncertainty or omits that sample rather than selecting an arbitrary edge.

Observed stripe width is optional diagnostic metadata, not required benchmark truth. Requiring agents to estimate a two-to-four-pixel stripe width in compressed 720p footage would add annotation noise without improving the primary geometric target.

The court layout owns each marking's physical width and records whether the cited rule-book dimension is measured to an inside edge, outside edge, or centerline. The layout builder converts rule-book dimensions to canonical painted centerlines exactly once. Rendering, fitting, and benchmark evaluation all consume those derived centerline primitives.

## 6. Preventing false lines from creating wrong confidence

Use multiple layers of defense:

1. Limit evidence to a conservative court region.
2. Mask players, officials, stanchions, and screen graphics where possible.
3. Detect marking-like ridges rather than arbitrary edges.
4. Use the model transform to narrow the search area.
5. Require marking topology, not isolated segment agreement.
6. Require support from multiple primitive families.
7. Cap each primitive's total influence.
8. Require both:
   - detected pixels near the projected template; and
   - visible projected template portions supported by detected paint.
9. Reject low-margin identity assignments.
10. Keep competing hypotheses long enough to detect non-unique solutions.
11. Validate against an entire unused marking family.
12. Perform leave-one-primitive stability tests.
13. Later, require temporal agreement across neighboring frames.

Log all accepted, rejected, and ambiguous evidence. A diagnostic overlay should clearly show which image pixels affected the transform.

A high raw inlier count is not a confidence measure when those inliers are correlated pixels from one painted line.

## 7. Independent benchmark annotations

### 7.1 Feasibility outcome

Milestone 1 tested whether an OpenAI multimodal, tool-using agent could create the
benchmark reference. The agent could emit structurally valid annotations, but the
frozen planted-truth probe found unacceptable tail error and systematic
centerline offset: 6.10 px p95 against a 4 px limit and 3.93 px worst absolute
family signed offset against a 0.75 px limit. Following the stop condition, no
real-frame agent pass was accepted.

A purpose-built blind human centerline editor was used instead. One human labeler
completed all six locked frames, and those validated annotations are the
Milestone 1 authority. The frozen prompts, synthetic probe, consensus tooling,
and failed gate result remain reproducibility artifacts rather than completion
requirements.

### 7.2 Keep the annotation process independent

Give any agent or human labeler raw frames only. Do not expose:

- current model keypoints;
- current projected court overlays;
- hybrid candidates;
- homography matrices.

Otherwise the benchmark may inherit the same systematic error it is intended to detect.

Provide each labeler with:

- the raw full-resolution frame;
- image dimensions;
- exact NBA feature definitions;
- the required output schema;
- explicit visibility and uncertainty rules.

### 7.3 Annotation contents

For each visible feature, require the labeler to annotate:

- exact painted-line intersections;
- points sampled along the center of the three-point marking;
- points along the free-throw circle;
- lane and free-throw-line centerlines;
- midcourt line and center-circle points where visible;
- occluded intervals;
- visibility and uncertainty;
- a reason when a feature is skipped.

Coordinates must refer to the original frame, not a resized preview.

### 7.4 Agent-consensus contingency and recorded rejection

The implementation supports frozen prompts, two isolated passes, geometric
agreement checks, a sanitized actual-dispute queue, and fresh-session
adjudication. That path is valid only after the agent configuration passes the
synthetic planted-truth gate. The Milestone 1 configuration did not pass, so the
real-frame pass directories were intentionally left empty and no consensus
silver set was created. Failed synthetic annotations remain diagnostic history
and must never be mixed with the human reference.

Any future attempt must use a new run namespace, preserve complete
annotated-or-skipped dispositions, and pass all frozen precision gates before
viewing real benchmark frames. Agreement alone is not accuracy and cannot
supersede the planted-truth result.

### 7.5 Human reference and future audit

The purpose-built editor preserves original coordinates while zooming and
panning, enforces painted-centerline sampling and minimum samples, records gaps
and uncertainty, and refuses incomplete final exports. The six resulting files
are tracked under `benchmark/references/nba_m1_v1_human/` with hashes and
provenance in `benchmark/references/nba_m1_v1_human.json`.

This single-labeler reference is sufficient to establish the exploratory
shadow-mode baseline and determine whether a later hybrid appears to have a
large, consistent advantage worth further investigation. Before production
promotion, obtain an independent audit covering:

- random accepted labels;
- frames where hybrid and model-only differ most;
- logo-heavy and graphics-heavy frames;
- every apparent hybrid failure; and
- the sparse far-depth annotations.

This small benchmark does not support production-level accuracy, reliability,
or promotion claims. Those require a larger independently audited gold set.

### 7.6 Frozen annotation-agent brief (historical)

The rejected agent workflow used rules equivalent to:

> Annotate only visible painted basketball-court markings in original-image pixels. Work from the raw frame without model predictions or projected overlays. Mark the center of each painted line, not arbitrary stripe edges. A line segment endpoint is only the end of visible support and is not a known court landmark. Skip ambiguous or occluded features, and give every glossary feature exactly one annotated-or-skipped disposition. Return structured JSON with feature identity, sampled points, visibility, uncertainty, and notes.

## 8. Independent evaluation

### 8.1 Locked frame set

Use the **existing six pilot frames**, one easy and one hard frame from each of the
three clips:

- `video_1`: frames 12 and 90;
- `video_2`: frames 18 and 141;
- `video_3`: frames 15 and 146.

This is the default and complete Milestone 1 benchmark. Expand only when the
six-frame outcome is borderline against an exploratory decision boundary or when
the observed labels/evaluation show inadequate failure-mode coverage. The locked
conditional additions are `video_1` frame 60 (pan/blur), `video_2` frame 3
(broadcast graphic), and `video_3` frame 177 (existing implausible result), for an
absolute ceiling of nine.

All selected and conditional frames already exist in the recorded calibration
runs. Reuse those baselines and subset them by selected frame ID; do not rerun
calibration merely because the annotation benchmark became smaller.

Lock frame selection before tuning the marking detector. Split evaluation by clip rather than randomly splitting points from the same frame. With only three clips, rotate which clip is held out so the method is tested against unseen floor appearance.

### 8.2 Three-way comparison

On identical frames, report:

1. current confident keypoint-only calibration;
2. model-seeded hybrid calibration;
3. an experimental CV-only challenger, if later implemented.

CV-only may remain an offline ablation; it need not be a production path.

### 8.3 Metrics

Measure:

- image-space error against the locked human reference annotations;
- court-space error in feet;
- errors by depth band:
  - 0–19 feet;
  - 19–30 feet;
  - 30–47 feet;
- median, p95, and maximum error;
- accepted coverage and rejection rate;
- confident-but-wrong results;
- frames where the selector chose the worse candidate;
- temporal jitter through each continuous shot.

Projected overlays remain mandatory visual evidence, but are not themselves the accuracy metric.

Within the hybrid algorithm, hold out a whole marking family rather than random pixels from the same fitted line. In the initial version, fit using model landmarks, lane geometry, and the three-point marking while reserving the free-throw circle for validation.

## 9. Proposed acceptance criteria

These remain exploratory challenger thresholds until the human reference receives an independent audit. A reasonable initial prototype bar is:

- at least 25% lower median and p95 error beyond 19 feet than model-only;
- no significant near-court regression—approximately no more than 2 pixels or 10% in p95;
- selected hybrid candidates near or below:
  - 4-pixel median held-out error;
  - 10-pixel p95 held-out error;
  - 0.5-foot median court error;
  - 1-foot p95 court error;
- zero gross accepted hybrid failures on the locked benchmark;
- no loss of usable coverage because rejected hybrid candidates fall back;
- unchanged keypoint-only output whenever CV is unavailable or rejected.

Only independently audited annotations should control hard promotion thresholds. The single-labeler Milestone 1 reference supports exploratory comparisons only.

A hybrid selection rate is not itself a success metric. Accuracy-first behavior may reject many CV candidates.

## 10. Ordered implementation milestones

### Milestone 1 — Independently labeled exploratory benchmark — COMPLETE

1. Define the annotation schema, painted-centerline convention, and feature
   glossary.
2. Lock six pilot frames—one easy and one hard per clip—and preselect the three
   conditional expansion frames in Section 8.1.
3. Freeze and hash-version model-neutral annotator and adjudicator prompts, record
   runtime model IDs, and preserve the complete-disposition synthetic truth gate.
4. Reject the agent route if it fails any synthetic precision threshold; do not
   run real-frame agent passes or treat agreement as accuracy after a gate failure.
5. Under the stop condition, use the blind purpose-built human editor to complete
   all six references with strict schema, geometry, and metadata validation.
6. Track the finalized references and their hashes independently of ignored local
   frames, model overlays, predictions, and calibration outputs.
7. Reuse the recorded confident- and probable-tier baselines, subset exactly to
   the six selected frames, and save both score reports.
8. Record the methodology decision, baseline results, exploratory limitations,
   and expansion decision. Expand to nine only if a later comparison is borderline
   or exposes inadequate failure-mode coverage.

**Completion evidence:** `benchmark/references/nba_m1_v1_human.json`, the six
annotations beside it, and
`benchmark/results/nba_m1_v1_human/milestone1.json`. The failed synthetic result
is retained in the same result directory. Both baseline reports count five
`DEGRADED` transforms and one `UNCALIBRATED_INSUFFICIENT_EVIDENCE` refusal; they
quantify sharply worsening far-depth extrapolation while explicitly limiting the
claim to exploratory shadow-mode development.

**Stop condition outcome:** triggered. Agent labels were rejected and replaced by
validated blind human references rather than weakened precision thresholds.

### Milestone 2 — Make marking geometry authoritative

Move all calibration-relevant marking geometry into the court layout layer:

- three-point radius and corner transition;
- free-throw circle;
- center circle;
- restricted area;
- line width and coordinate convention.

Both `calibration/viz.py` and the hybrid fitter must consume the same geometry.

Define generic records for:

- unlabeled marking evidence;
- marking assignments;
- candidate quality;
- selection decisions.

**Done when:** rendering and fitting use one layout source and nothing in the marking engine knows model slot numbers.

### Milestone 3 — High-precision extraction in shadow mode

Implement extraction and association for:

1. lane/free-throw straight geometry;
2. the full three-point marking;
3. the free-throw circle as held-out validation.

Produce evidence overlays and metrics without changing the selected calibration.

**Kill criterion:** if accepted three-point evidence cannot achieve high precision on held-out frames, stop rather than weakening rejection thresholds.

### Milestone 4 — Hybrid refinement in shadow mode

Add bounded, primitive-balanced joint refinement. Store:

- `H_keypoint`;
- `H_hybrid`;
- model residuals;
- marking residuals by primitive;
- held-out errors;
- leave-one-primitive stability;
- transform differences;
- rejection reasons.

**Done when:** the benchmark can compare candidates without affecting existing consumers.

### Milestone 5 — Candidate selection and guarded promotion

Implement the champion/challenger policy.

Enable hybrid selection only after it passes the locked benchmark and audited subset. If uncertain, retain keypoint-only.

**Done when:** every selected or rejected hybrid result is explainable and the baseline remains an exact fallback.

### Milestone 6 — Later improvements

In likely value order:

1. midcourt line plus center-circle support;
2. temporal propagation of a trusted seed;
3. temporal foreground and broadcast-graphics masks;
4. CV corroboration of probable midcourt slots;
5. learned semantic marking segmentation if classical CV is insufficient;
6. lens-distortion estimation if held-out residuals show systematic curvature;
7. CV-only calibration under stricter multi-primitive and temporal requirements.

## 11. Risks, unknowns, and decisions needed

1. **Agent precision (resolved for Milestone 1):** the tested multimodal configuration failed the planted-truth tail-error and systematic-offset gates even with full-resolution crop tooling, so agent labels are not benchmark authority.
2. **Benchmark independence:** annotators must remain blind to current and hybrid projections.
3. **Audit standard:** decide whether a small gold subset is checked by the project owner, another person, or an external service.
4. **Centerline conversion data:** annotations are settled as painted centerlines; implementation must source physical stripe widths and per-marking rule-book edge conventions accurately enough to derive those centerlines.
5. **Layout authority:** calibration-relevant marking dimensions currently used by visualization must become authoritative layout data before fitting relies on them.
6. **Graphics-mask ownership:** decide where player, official, stanchion, and broadcast-overlay masks are produced.
7. **Probable slots:** use only as corroborated hypotheses; ambiguous and unusable slots remain excluded.
8. **Temporal processing:** valuable for the available continuous pans, but it should follow static-frame proof rather than hide a weak marking detector.
9. **Lens distortion:** add distortion parameters only if independent residuals justify them.
10. **Absolute end identity:** painted geometry remains symmetric and cannot establish north/south.
11. **Data volume:** the supplied clips support a proof of value, not a broad production false-accept claim.
12. **Runtime budget:** accuracy-first multi-hypothesis fitting and validation may not be real-time; define the acceptable offline cost before optimization.

## 12. Final decision rule

The project should proceed with hybrid implementation only if evaluation against the locked human-reference benchmark shows that model-guided marking extraction can add reliable depth evidence without creating confident false associations.

The production decision hierarchy is:

1. use a demonstrably superior, fully gated hybrid candidate;
2. otherwise retain the good keypoint-only candidate;
3. otherwise report the existing explicit uncalibrated or failed status;
4. never manufacture a transform from ambiguous CV evidence merely to increase coverage.
