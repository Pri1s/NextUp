# Classical court homography pipeline

This is a separate, **classical-CV-only** inspection pipeline. It does not load
`models/court_keypoint_detector.pt`, change the dataset manifest, create
training labels, split data, or fine-tune anything.

## Run

```bash
source .venv/bin/activate
python run_homography_labels.py dataset/frames \
  --template court_templates/nba.json \
  --output homography_results
```

Frames named `<clip>_f<index>` are grouped automatically when their siblings
are in the same directory. When an exported frame is isolated, pass one or
more same-clip files or directories explicitly:

```bash
python run_homography_labels.py homography_input \
  --hud-context dataset/frames/001_video_4 \
  --output homography_results
```

Use `court_templates/fiba.json` for a FIBA court. A custom dimension set is a
JSON file with the same `dimensions`, `keypoints`, `lines`, and `arcs` shape;
coordinates are real-world coordinates in the template's declared units.

Every input image produces a JSON result under `homography_results/`, with:

- `detection.primitives`: unlabelled Hough line segments, circle/ellipse arc
  candidates, and derived line intersections. Every line retains raw-fragment
  provenance, observed painted length, full fitted span, merge gaps, sampled
  brightness, HUD overlap, and contributing HUD-region IDs.
- `detection.preprocessing.hud_detection`: temporal/static mode, context
  frames, local-window sizes, accepted graphic regions, thresholds, and masked
  fraction. HUD regions are soft evidence; primitives are never hard-cropped.
- `detection.preprocessing.line_merge`: the two-pass strategy, exact merge
  thresholds, and raw/base/extended/deduplicated/emitted counts.
- `solution`: the best homography (or an explicit solve rejection), seed
  correspondence hypothesis, independently agreeing line/arc inliers, every
  reprojected template keypoint, nearest detected-intersection error, and
  `proposal_diagnostics` describing anchor counts, proposal types, duplicates,
  invalid/orientation-rejected candidates, and candidates reaching scoring.
- `verification`: pass/fail, the exact reasons, metrics, and thresholds.

`batch_summary.json` records the upstream solution status/reason as well as one
gate result per frame plus total pass/fail count and pass rate.

## Correspondence safeguards

The proposal stage first derives the template's two court-axis families from
line geometry. In the image it discovers connected U motifs: a cap that meets
two similarly oriented legs near their endpoints. Each motif defines local
projective orientation buckets, so broadcast perspective is not reduced to an
image-horizontal/image-vertical assumption. Template U motifs are matched by
explicitly enumerating both leg assignments and only bucket-compatible 2+2
family bijections.

Incomplete views use ordered two-corner, one-corner, and orientation-only
fallbacks. When a two-corner quartet contains a third observed corner, a
bounded topology-preserving prelude keeps its endpoint-compatible mappings
near the front of the fallback budget. Sources are scheduled round-robin,
canonical correspondence sets are deduplicated, and `random_seed` is used only
for deterministic tie order. The `--hypotheses` value is therefore a maximum
guided-proposal budget, not a count of random four-line permutations.
Arbitrary 4! assignments are never generated.

The solver compares finite, visible projected template segments with finite
Hough segments. A correspondence needs endpoint placement, length, and
longitudinal overlap; sharing an infinite line or having a nearby midpoint is
not enough. One detected segment may support only one template marking.

Before a candidate reaches the verification gate, it must have support from at
least three finite seed segments, four unique line assignments, a connected
court substructure, and both parallel and perpendicular court relationships
after inverse projection. Circle evidence is one-to-one and may reinforce a
solution, but it does not replace those line constraints. The final keypoint
intersection gate below is unchanged and remains intentionally conservative.

The default gate is intentionally conservative: it requires four detected
keypoint-intersection matches, two **non-seed** line matches, mean matched
keypoint error at most 12 pixels, a plausible non-degenerate court outline,
and no more than 85% of reprojected points off-image. Adjust only deliberately:

```bash
python run_homography_labels.py frames --output homography_results \
  --max-error-px 10 --min-matched-keypoints 5 --hypotheses 10000
```

A larger proposal budget can cover more low-ranked guided sources, but should
not be used to weaken the verification gate.

## Three outcomes: automatic pass, review required, hard rejection

Every solved frame ends in exactly one of three states:

- **Automatic pass** (`solution.status == "solved"`, `verification.status ==
  "pass"`). The candidate cleared every threshold above: seed segment support
  ≥3, four unique line assignments, a connected court substructure, both
  parallel and perpendicular relationships, four detected keypoint-intersection
  matches with two of them non-seed, mean error ≤12px, and ≤85% off-image.
  `solution.review_candidate` is `null` — an automatic pass never needs review.
- **Review required** (`solution.status == "rejected"`,
  `solution.review_candidate` is not `null`). The best candidate failed at
  least one automatic threshold — most commonly four unique line assignments
  or two non-seed line inliers — but still independently satisfies a second,
  strictly relaxed set of detected-geometry-only thresholds: ≥3 unique line
  matches (not 4), the same seed-support/connected/parallel/right-angle
  minimums, ≥4 matched keypoints with mean error ≤12px and ≤85% off-image (the
  same numeric bars as automatic acceptance), plus a new requirement that ≥3 of
  its seed lines have HUD overlap below 0.25. `review_candidate` carries its
  own homography, reprojected keypoints, HUD support summary, and an
  `automatic_gate_failures` list explaining exactly which automatic thresholds
  it missed. This candidate is not a label — it requires a human decision via
  `review_homography_labels.py` (see below) before it can be used for
  anything.
- **Hard rejection** (`solution.status == "rejected"`,
  `solution.review_candidate` is `null`). No candidate — automatic or
  review-relaxed — cleared even the relaxed thresholds. `solution.reason` and
  `proposal_diagnostics.review_rejection_reasons` record why.

The review-only thresholds never feed back into automatic acceptance:
`verify_solution` and every automatic structural check in `solve_homography`
are unchanged, and a frame that fails today cannot become an automatic pass
merely because a review candidate exists for it.

When a frame's visible markings are locally mirror-symmetric (e.g. a lane view
whose only seed lines are a baseline and free-throw line shared by both
halves), a west-side and an east-side (or north/south) assignment can produce
*bit-identical* detected-geometry evidence — no such criterion can break that
tie. `solve_homography` resolves it with the deterministic guided-proposal
stream order for the given `random_seed`, which is reproducible but not
guaranteed to prefer one physical side over the mirror twin without additional
context. This is exactly the kind of ambiguity a human reviewer resolves at a
glance.

## HUD evidence

Temporal HUD detection uses the anchor plus at most six evenly distributed,
same-clip, same-resolution frames. It requires at least three frames spanning
30 frame indices (roughly 0.5 seconds at 60 fps); otherwise it uses a strict
single-frame fallback. Temporal evidence combines anchor-relative pixel
stability, edges persistent in at least 60% of frames, and local saturation in
windows sized to 3.5% of the image. Only compact, filled, border-adjacent
graphic components are accepted.

The static fallback requires a dense-edge, saturated, high-value-variance
component near a border. This prevents lower-frame position by itself—or a
saturated patch of court paint—from being treated as HUD. Accepted component
rectangles are filled and padded, then line/arc overlap is sampled after line
merging. Soft floor confidence is brightness multiplied by `(1 - hud_overlap)`;
there is no positive-y term or saturation reward.

## Fragmented-line recovery

Raw Hough fragments first use the legacy 72 px merge. A deterministic
agglomerative pass then revisits the remaining clusters with an extended gap
cap of `clamp(0.05 * max(image_dimension), 72, 144)` pixels. It requires at
most 2.5 degrees of angular disagreement, 8 px symmetric lateral offset, 8 px
TLS endpoint residual, 0.80 coverage for both input clusters and their union,
and `gap / shorter_cluster_span <= 0.30`. The union is refit after every merge
until stable.

Observed painted length—not the bridged span—controls proposal ranking. The
full fitted span remains available to the unchanged finite-segment matcher.
HUD overlap and appearance strength are resampled after merging rather than
inherited from whichever fragment happened to be strongest.

## Review and correction

```bash
python review_homography_labels.py homography_results
# open http://127.0.0.1:8010
```

The review UI overlays detected Hough segments and every active point (a
solved frame's own reprojection, or — for a rejected frame with a candidate —
the review candidate's reprojection, drawn with a dashed amber ring so it is
never confused with an automatic pass). A rejected frame with a candidate
shows an amber "REVIEW REQUIRED" banner alongside the unchanged "Gate: FAIL"
heading and the automatic failure reasons. Points can be dragged to correct
them, marked `excluded`, or added manually; each keeps a per-point status of
`unchanged`, `moved`, `excluded`, or `added`. "Approve candidate" / "Reject
candidate" record a reviewer decision alongside those corrections. Reviewer
output — original automatic status/reason, the candidate's proposal identity
and homography, the decision, per-point corrections, a timestamp, and the
source result path — is written only to `homography_results/reviews/`; it
never overwrites the generated result JSON, `batch_summary.json`, or any
training-format label. Approving a candidate is a human decision recorded only
in that review artifact — it never retroactively changes the automatic gate
result or the batch pass rate.

## Current validation artifact

`homography_results/batch_summary.before.json` preserves the supplied batch's
pre-change result and `batch_summary.json` is the rerun. The accompanying
`batch_summary.comparison.json` records the false-solve transition, HUD/merge
acceptance measurements, proposal counts, the unchanged-gate audit, and the
review-candidate summary for the labeled frame.

The guided search now proposes the intended lane U near the front of the
budget, but this frame still cannot pass *automatically*: seed support of at
least three plus two non-seed line inliers requires five distinct one-to-one
finite line matches. Single-frame Hough/LSD variants, all 577 post-merge
lines, and SIFT-aligned pooling across the four available clip frames recover
at most four compatible markings (baseline, two lane legs, and free-throw
cap). The result therefore remains an automatic rejection — no structural,
segment-support, assignment, or keypoint threshold is lowered to turn missing
geometry into a label. That same four-marking hypothesis, however, now
satisfies the strictly relaxed review-only thresholds (mean detected-
intersection error 3.93px over 5 matched keypoints, all HUD-clear) and is
retained as `solution.review_candidate` for a human to inspect and approve;
its four manually labeled lane corners average 6.43px error against the
dataset's ground truth.

## Important limitations

Unlabelled court lines are geometrically ambiguous, especially in partial,
occluded, heavily painted, or extreme-perspective frames. Guided proposals do
not manufacture missing finite geometry: the solver only accepts a result
after other line/arc and keypoint-intersection evidence agrees. A failure is a
useful result: inspect its primitives and rejection reason rather than treating
it as a label. The output is designed to measure what fraction of real footage
can be labeled confidently before any later dataset or learning work.
