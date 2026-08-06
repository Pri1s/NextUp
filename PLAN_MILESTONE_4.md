# Milestone 4 — Hybrid Refinement in Shadow Mode — Implementation Plan

**Status:** Implemented and tested in shadow mode through steps 5–10. **Not evaluated** —
P0/P1 have not run, so nothing here is yet evidence that hybrid refinement helps on real
footage. See "Corrections during implementation" and "What is and is not established".
**Depends on:** M2 (complete), M3 (implemented, never run, not yet gated)
**Tests:** planning baseline 508 → 553 after the parameterization → **647 passing, 8 skipped**
at close-out (`.venv/bin/python -m unittest discover -s . -p 'test*.py'`)

## Context

Every usable calibration the engine has ever produced is supported by five model
landmarks that span **19 feet of court depth and no more**. Verified on
`engine_out/calibration/video_1/m1_confident/calibrations.jsonl`, the inlier set on
every eligible frame is exactly:

| landmark | court (ft) |
| --- | --- |
| `baseline_sideline_far` | (0, 0) |
| `three_point_baseline_far` | (0, 3.17) |
| `lane_baseline_far` | (0, 17.17) |
| `lane_free_throw_far` | (19.0, 17.17) |
| `lane_free_throw_near` | (19.0, 33.0) |

Three of the five are collinear on `x = 0`. That is literally what the degradation
reason `weak_conditioning:only_2_points_off_dominant_line` (`calibration/quality.py:265`)
is reporting. Reprojection reads under a pixel while projected markings past the
free-throw line visibly drift, because nothing past 19 ft constrains the fit.

M3 built the evidence layer: `ShadowMarkingFrame` records carrying
`UnlabeledMarkingEvidence` fragments plus an `ACCEPTED`/`REJECTED`/`AMBIGUOUS`
`MarkingAssignment` per fragment. **Nothing consumes them.** M4 closes that gap in
shadow mode: fit a second transform against the accepted paint, measure it honestly
against evidence it did not fit, and record both candidates side by side without
changing a single byte any existing consumer reads.

**The load-bearing observation.** Depth span of every primitive in the associator's
`ACCEPTED_FEATURES`:

| feature | court-x span (ft) |
| --- | --- |
| `lane_edge_far` / `lane_edge_near` | 0 → 19.0 |
| `free_throw_line` | 19.0 (constant) |
| `three_point_corner_far` / `_near` | 0 → 14.26 |
| `three_point_arc` | 14.26 → **29.0** |
| `free_throw_circle_far_half` | 19.0 → 24.92 (**held out**) |

Once the free-throw circle is reserved for validation, **`THREE_POINT_ARC` is the only
fittable primitive carrying any depth past 19 ft.** Everything else improves near-field
conditioning. So the M4 hypothesis is exactly: *can arc paint past the 14.26 ft break be
extracted, associated and fitted reliably enough to extend the constraint from 19 ft to
29 ft, without hundreds of correlated lane pixels swamping five semantic landmarks?*
Every threshold below is chosen against that sentence.

**Confirmed decisions** (asked and answered before writing this plan):

1. The plan includes the blocking prerequisites P0/P1 as gated steps.
2. `D_max = 1.5 ft` — M4 tests *refinement of a nearly-right transform*, not correction of a badly-extrapolated one.
3. Hand-rolled numpy Levenberg–Marquardt; **do not add scipy**.
4. Move `polyline.py` into `calibration/` and widen the blindness allow-list with a written justification.

## Approach

Add a new package `calibration/hybrid/` that consumes a **serialized**
`ShadowMarkingFrame` plus a `CourtCalibration`, and emits a new
`HybridRefinementFrame` into a `hybrid_shadow/` namespace beside `marking_shadow/`.
Fitting is a bounded, primitive-balanced, projected Levenberg–Marquardt over four
canonical control points — never raw homography coefficients. The refiner computes
every gate but **never selects**: `SelectionOutcome.KEEP_BASELINE` on every path. A
contract-only `benchmark/hybrid.py` scorer reads the serialized records and produces
the §8.2/§8.3/§9 comparison by delegating to the existing
`score_calibration_entry` / `summarize_scores` / `compare_summaries`.

### Why a new package, not files in `calibration/markings/`

Dependency direction. `markings/` imports only `contracts.*` and
`calibration.estimator`, and `extractor.py` states its own constraint: *"This keeps the
detector replaceable."* Fitting needs `calibration.quality` and
`calibration.correspondence`, both on `benchmark`'s forbidden list. Pulling them into
`markings/` would make the *evidence* layer depend transitively on the *prediction
interpretation* layer and destroy the §5.6 modularity claim. The package split also
mirrors §5.2/§5.3/§5.4/§5.5 and leaves an obvious home for M5.

### Why a new orchestrator, not an extension of `ShadowOrchestrator`

`ShadowOrchestrator.config_hash` is the frozen M3 fold identity; folding refinement into
it would couple frozen extraction config to M4 tuning. More importantly, a
`ShadowMarkingFrame` is a complete serializable input — consuming the *record* rather
than live extractor state means M4 re-runs offline over an existing
`marking_shadow/records.jsonl` with **no video, no checkpoint, no GPU**. Given nine M3
variants may need sweeping, that decoupling is what makes M4 evaluable at all.

## 1. Module layout

```
calibration/hybrid/__init__.py
calibration/hybrid/control_points.py   # parameterization
calibration/hybrid/residuals.py        # residual model + primitive balancing
calibration/hybrid/refine.py           # bounded LM
calibration/hybrid/diagnostics.py      # CandidateQuality/PrimitiveQuality/GateResult builders + gate policy
calibration/hybrid/orchestrator.py     # HybridOrchestrator, HybridRefinerConfig, provenance guards
calibration/hybrid/io.py               # write_hybrid_run / read_hybrid_records  (the ONLY writer)
calibration/hybrid/overlays.py         # keypoint-vs-hybrid difference overlay
calibration/polyline.py                # MOVED from benchmark/polyline.py
```

Key signatures:

```python
# calibration/hybrid/control_points.py
def control_court_points(layout) -> np.ndarray            # (4, 2) the half-court corners
def probe_court_points(layout) -> np.ndarray              # (16, 2) 11 landmarks + 5 interior
def band_indices(depth, bands) -> np.ndarray              # mirrors benchmark depth_band, clamped
def is_convex(image_quad) -> bool
def homography_from_control_points(court, image) -> np.ndarray | None   # 4-point DLT
def control_points_from_homography(matrix, layout) -> np.ndarray | None
def local_pixel_scale(matrix, court) -> np.ndarray        # (4,) px per court ft
def clip_to_box(theta, theta0, radius_px) -> tuple[np.ndarray, int]

# calibration/hybrid/refine.py
@dataclass(frozen=True, slots=True)
class RefineResult:
    theta: np.ndarray
    h_court_to_image: np.ndarray | None
    h_image_to_court: np.ndarray | None
    round_trip_error_px: float | None
    cost: float
    outer_iterations: int
    inner_iterations: int
    converged: bool
    bounds_active: int
    convexity_backtracks: int
    failure: str | None

def refine(theta0, correspondences, observations, layout, config, image_shape) -> RefineResult

# calibration/hybrid/orchestrator.py
class HybridProvenanceError(ValueError): ...

class HybridOrchestrator:
    def __init__(self, layout: HalfCourtLayout, config: HybridRefinerConfig | None = None)
    @property
    def config_hash(self) -> str
    def refine_frame(self, shadow: ShadowMarkingFrame, calibration: CourtCalibration,
                     correspondences: tuple[Correspondence, ...], *,
                     image_shape: tuple[int, int]) -> HybridRefinementFrame
```

**On moving `polyline.py`.** The refiner needs clamped point-to-polyline projection;
`benchmark/polyline.py` already implements exactly that (`nearest_on_polyline` returning
`segment`, `t`, and `.interpolate`). `benchmark/tests/test_blindness.py`'s own docstring
settles the question: *"duplicating a homography solver to avoid touching it would create
two solvers that could disagree — a worse failure."* Move it to `calibration/polyline.py`
(pure numpy — no layout, no detector, no model), re-export from `benchmark/polyline.py`
for existing consumers, and add `"calibration.polyline"` to `ALLOWED_CALIBRATION_MODULES`
with a matching comment.

## 2. The residual model

**Unit: image pixels.** Mirroring `estimator.py`'s reasoning. Noise lives in pixels;
`uncertainty_px`, the association corridor, `ransac_threshold_px` and the §9 thresholds
are all pixels. `CalibrationPolicy.scale_ft_per_px_range` spans `(0.005, 0.5)` — 100× —
so a court-feet cost would silently up-weight far evidence by an order of magnitude and
let the arc alone dictate the solution. That is the failure §5.4 forbids.

### Perpendicular-only marking residual

For each accepted, non-held-out assignment with `selected_feature = f`:

1. Sample the layout primitive densely in **court** coordinates: `layout.marking(f).sample(n)`,
   `n = 401` for arcs, `201` for straights (chord error ~2e-4 ft on the three-point arc).
2. Project through the current H, run `nearest_on_polyline`, and **freeze** the foot point
   as a *court* coordinate `c_s` via `NearestResult.interpolate(court_points)` — the same
   index-carrying trick `metrics.score_frame` uses — plus its next template vertex `c_s'`.
3. The inner-solve residual is the **signed normal component only**:

```
a(θ) = π(H(θ), c_s)        b(θ) = π(H(θ), c_s')
u(θ) = (b − a)/‖b − a‖     n(θ) = (−u_y, u_x)
r_s(θ) = n(θ) · (p_s − a(θ))          # one scalar per sample
```

This is the mathematical statement of §5.4's *"Every sampled pixel constrains distance
perpendicular to the assigned marking; its position along that marking remains
unknown."* The along-track component is discarded by construction, so a fragment
endpoint can never act as a landmark. Foot points are **clamped to the template extent** —
samples running past the end of a primitive are evidence of a bad association and must cost
something.

### Model landmark residuals

`r_L(θ) = π(H(θ), court_xy) − image_xy` over the keypoint fit's **inlier**
correspondences, so both candidates are measured on identical evidence and the number is
directly comparable to `CalibrationQuality.reprojection_px_median`.

**Deliberate kernel asymmetry:**

- **Landmarks: Huber**, `σ_L = 2.0 px`, `δ_L = 2.0` (4 px raw = half the RANSAC threshold).
  Huber does not redescend, so the optimizer *cannot abandon a model landmark*. §5.5
  requires model points stay within a calibrated tolerance; a redescending kernel would
  let a landmark be quietly dropped and the gate would then measure a landmark the fit had
  already given up on.
- **Markings: Cauchy**, `δ_M = 1.5` normalized, `σ_s = clamp(sample.uncertainty_px, 0.5, 4.0)`.
  Redescending is correct here — association is the primary filter, but a decoy that
  survived it must be abandonable. This is the last §6 defense inside the optimizer.

IRLS weights (Huber `w = min(1, δ/|r̂|)`, Cauchy `w = 1/(1 + (r̂/δ)²)`) recompute once per
outer iteration and stay frozen within the inner solve, so each inner problem is a genuine
weighted least-squares.

### Primitive balancing — the exact formula

```
E(θ) = Σ_L  1.0 · ρ_Huber(‖r_L(θ)‖/σ_L)
     + Σ_f Σ_{s∈f}  ω_{f,s} · ρ_Cauchy(r_s(θ)/σ_s)
     + Σ_k  ω_P · ‖q_k(θ) − q_k⁰‖²/λ_k²
```

Marking weights are governed by a **budget** `Σ_{s∈f} ω_{f,s} = B_f`, where

```
B_f_raw = β · min(1, ℓ_f/ℓ_ref)
family cap:  if Σ_{f∈family} B_f > C_fam:  scale that family by C_fam/Σ
global cap:  if Σ_all B_f > C_glob:        scale all by C_glob/Σ
```

Within a primitive, distribute by **arc-length territory on the projected template**, not
by sample count: `ω_{f,s} = B_f · a_s / Σ a_{s'}`, where `a_s` is half the arc-length gap
to each neighbour, computed by bucketing foot points into 1 px arc-length bins and
splitting each bin's territory among the samples that landed in it.

| symbol | value | justification |
| --- | --- | --- |
| `β` | **1.0 landmark-equivalent** | Makes §5.4 literal: one fully-supported painted primitive is worth exactly one semantic landmark. |
| `ℓ_ref` | **120 px × `diagonal_scale(shape)`** | Twice the associator's `arc_support_px = 60` gate. A 30 px scrap of lane edge gets a quarter budget; short evidence genuinely constrains less. |
| `C_fam` | **2.0** | LANE has three near-parallel, highly correlated members; uncapped, the lane alone outweighs THREE_POINT. §6 rules 6 and 7. |
| `C_glob` | **4.0** | Hard ceiling: **total marking weight can never reach the landmark total of 5.0.** The keypoint seed keeps a strict majority on every frame. |
| `ω_P` | **0.25 per control point** (1.0 total) | Tikhonov prior toward `θ⁰`, normalized by `λ_k` (px per court ft at that corner) so it reads in feet. |

**Why arc-length territory is the important rule.** `resample_spacing_px` is a tunable in
`MarkingExtractorConfig`. Weighting by sample *count* would make the fit's answer a
function of a detector tuning knob — a latent bug no unit test would surface. Territory
weighting turns the sum into a discretized line integral, invariant to sampling density,
and handles §5.4's "spatial region" requirement: two fragments of the same feature
overlapping in arc length *share* that territory rather than each claiming full weight.

**Why `C_glob < 5.0` is not too timid.** The five landmarks are consistent with a *family*
of transforms — that is what `weak_conditioning` means. Along the under-determined
direction the landmark cost is nearly flat, so even modest marking weight moves the
solution substantially there while barely perturbing well-determined directions. The
balance is not about who wins; it is about which direction each term constrains.

## 3. The parameterization

**Control points:** the four half-court corners in court coordinates —
`(0,0)`, `(half_length,0)`, `(half_length,width)`, `(0,width)`. Chosen deliberately: these
are the *identical* four corners `quality._plausibility_failures` already projects for its
convexity check, so a feasible θ cannot fail `court_quad_not_convex`.
`θ ∈ R⁸` is their image coordinates; `θ⁰ = project(H_keypoint, C)`.

**θ → H:** a numpy **4-point DLT** (build the 8×8 `A`, `np.linalg.solve`, fix `h₂₂ = 1`),
*not* `cv2.getPerspectiveTransform` — this sits on the hot path of a content-hashed
artifact and must not acquire an OpenCV-version dependency. A unit test asserts agreement
with `cv2.getPerspectiveTransform` to 1e-9. Normalize through the same `_as_matrix`
convention (`h/h₂₂`) the estimator uses, so no two byte-different records denote the same
transform.

**Bounded change — two mechanisms, both required:**

- **(a) Hard box in court feet.** With `λ_k` = the image length of a 1 ft court step at
  `C_k` under `H_keypoint`: `‖q_k − q_k⁰‖₂ ≤ D_max · λ_k`, **`D_max = 1.5 ft`**. Feet, not
  pixels — `λ_k` varies ~100× between near baseline and midcourt. Enforced by projection
  onto the box after every accepted step. Corners are ≥40 ft apart, so a 1.5 ft box cannot
  flip the quad; convexity is implied but still checked.
- **(b) Soft prior toward `θ⁰`** (`ω_P = 0.25`). Without it a nearly-flat direction inside
  the box wanders arbitrarily. With it the Hessian is strictly positive-definite even with
  zero marking evidence, and — critically — **with zero markings the unique minimum is
  exactly `θ⁰`**, so the optimizer terminates at iteration 0 and `H_hybrid ≡ H_keypoint` to
  machine precision. That is a testable invariant.

**Preserving existing gates:** after convergence run the real thing — a new public
`calibration.quality.plausibility_failures(...)` (thin wrapper over the existing private
function, no behaviour change) plus `_is_convex` and `_scale_ft_per_px`. The hybrid is
judged by the same physical gates as the baseline, not a parallel reimplementation.

**Optimizer — hand-rolled projected LM:**

```
θ ← θ⁰;  μ ← 1e-3
for outer in range(max_outer = 5):
    freeze foot points and neighbours from H(θ)          # ICP alternation
    recompute IRLS weights and rebalance budgets
    for inner in range(max_inner = 25):
        r ← [landmark residuals; marking residuals; prior residuals]
        J ← central-difference Jacobian, h_k = 1e-4·max(1, |θ_k|)
        solve (JᵀWJ + μ·diag(JᵀWJ)) δ = −JᵀW r
        θ_try, n_active ← clip_to_box(θ + δ, θ⁰, D_max·λ)
        if not convex(θ_try) or E(θ_try) ≥ E(θ):  μ ← 10μ;  continue
        θ ← θ_try;  μ ← max(μ/10, 1e-9)
        if ‖δ‖∞ ≤ 1e-7·(1+‖θ‖∞) or ΔE ≤ 1e-10·(1+E):  break
    if max probe displacement this outer iteration ≤ 0.01 px:  break
```

Determinism: float64 throughout, **no RNG anywhere**, central (not forward) differences,
observations iterated in a fixed sort order `(feature.value, evidence_id, sample_index)`.
Every constant above lives in `HybridRefinerConfig` and enters `refiner_config_hash`.
*Acknowledged risk:* hand-rolled optimizers are a classic bug source — mitigated by
planted-transform recovery, a finite-difference gradient self-check, and the
sliding/density invariance tests in §7.

## 4. The M4 record schema

**Decision: do NOT bump `HYBRID_RECORD_SCHEMA_VERSION`. Add new record types under
`"hybrid-marking-records-1.0.0"`.** `_schema()` compares against a single module constant
shared by `UnlabeledMarkingEvidence`, `MarkingAssignment`, `ShadowMarkingFrame`,
`CandidateQuality` and `SelectionDecision`. Bumping it would make **every** M3
`records.jsonl` unreadable — including the artifacts of the imminent M3 gate. Adding a new
top-level dataclass changes the serialized shape of nothing, because `_fields()` strictness
is per-record. The literal assertion at `contracts/tests/test_hybrid_types.py:209` stays as
written and gains a companion regression test proving a pre-M4 `ShadowMarkingFrame`
fixture still round-trips.

**Where transforms live.** `PLAN_MILESTONE_2.md:47` says the transform stays in
`CourtCalibration`. That was written when every transform was an accepted one.
`CourtCalibration.__post_init__` enforces "usable status ⟺ both transforms present", so a
**rejected challenger cannot be a `CourtCalibration` at all**. Resolution preserving the M2
intent: `CourtCalibration` remains the sole home of a transform a downstream consumer may
*use*; `CandidateTransform` is an **audit record of a transform that has not been
promoted**. `CandidateQuality` stays matrix-free exactly as M2 designed it. Update the M2
line with a note rather than silently contradicting it.

New types in `contracts/hybrid_types.py`, house style (frozen slots, strict
`as_dict`/`from_dict`):

```python
class HybridFrameStatus(str, Enum):          # new enum, not a member added to ShadowFrameStatus
    OK = "OK"; ABSTAINED = "ABSTAINED"; SKIPPED = "SKIPPED"; REFINEMENT_FAILED = "REFINEMENT_FAILED"

@dataclass(frozen=True, slots=True)
class CandidateTransform:
    candidate_id: str; source: CandidateSource
    layout_id: str; layout_hash: str
    h_court_to_image: Matrix3x3 | None; h_image_to_court: Matrix3x3 | None
    round_trip_error_px: float | None; solver: str
    failure: str | None = None
    schema_version: str = HYBRID_RECORD_SCHEMA_VERSION

@dataclass(frozen=True, slots=True)
class ProbeDisplacement:
    band: str; point_count: int; median_px: float; max_px: float

@dataclass(frozen=True, slots=True)
class TransformDifference:
    baseline_candidate_id: str; challenger_candidate_id: str
    probe_point_count: int; probe_px_median: float; probe_px_p95: float; probe_px_max: float
    max_control_point_shift_ft: float; bounds_active: int; scale_ratio: float
    bands: tuple[ProbeDisplacement, ...] = ()

@dataclass(frozen=True, slots=True)
class PrimitiveStability:
    feature: MarkingFeature; shift_px: float; refit_converged: bool

@dataclass(frozen=True, slots=True)
class HybridRefinementFrame:
    frame_id: str; image_sha256: str
    status: HybridFrameStatus; eligible: bool; eligibility_reasons: tuple[str, ...]
    layout_id: str; layout_hash: str
    shadow_config_hash: str; refiner_config_hash: str
    baseline_candidate_id: str; challenger_candidate_id: str | None
    transforms: tuple[CandidateTransform, ...]
    candidates: tuple[CandidateQuality, ...]
    difference: TransformDifference | None = None
    selection: SelectionDecision | None = None
    stability: tuple[PrimitiveStability, ...] = ()
    fitted_features: tuple[MarkingFeature, ...] = ()
    held_out_features: tuple[MarkingFeature, ...] = ()
    outer_iterations: int = 0; inner_iterations: int = 0; converged: bool = False
    refine_ms: float = 0.0; diagnostics_ms: float = 0.0
    failure_reasons: tuple[str, ...] = ()
    schema_version: str = HYBRID_RECORD_SCHEMA_VERSION
```

`CandidateTransform` invariants: both matrices present or both `None`; absent ⟹ `failure`
non-empty; present ⟹ `failure is None` and `round_trip_error_px` finite ≥ 0; each matrix
3×3 finite with **`h[2][2] == 1.0` within 1e-12**. `layout_hash` sits on the transform, not
only the frame, because `score_calibration_entry` refuses a transform whose hash is missing
or wrong — the guard must read the hash off the same object carrying the matrix.

`HybridRefinementFrame` invariants: `transforms` and `candidates` have unique
`candidate_id` and **equal id sets**; `baseline_candidate_id ∈ ids`; challenger, when
present, `∈ ids` and `≠ baseline`; **exactly one** candidate has
`source is CandidateSource.KEYPOINT` and it is the baseline; no challenger ⟹
`difference is None` and `fitted_features == ()` and `stability == ()`; `selection` ids and
`frame_id` must agree with the frame's, and `SELECT_CHALLENGER` ⟹ the challenger is
`usable`; `ABSTAINED` ⟹ non-empty `eligibility_reasons` and no challenger;
`REFINEMENT_FAILED` ⟹ non-empty `failure_reasons` and no challenger; **`fitted_features`
and `held_out_features` disjoint**; `stability` features ⊆ `fitted_features`. No field name
or serialized key contains "slot". The KEEP_BASELINE-only rule is *not* a record invariant
(the record must survive into M5) — it is an orchestrator rule with a dedicated test.

Candidate ids, deterministic and uuid-free, matching the extractor's style:
`f"{frame_id}:keypoint"` and `f"{frame_id}:hybrid:{refiner_config_hash[:12]}"`.

**The transform-difference metric is projected canonical court point displacement in
pixels.** Matrix-element differences are meaningless (H is scale-equivalent; elements have
incommensurable units). Court-feet displacement of image points requires inverting through
a transform under test — circular. Projected court points are in pixels, directly
interpretable, and are *literally* what §5.5 asks for. Probe set is **16 points** — the 11
`HalfCourtLandmark` coordinates plus 5 interior points no landmark covers: the basket
centre, the three-point arc apex at 29 ft, the two arc/corner-straight joins, and the
centre circle's near edge at 41.2 ft.

> **Correction, made during implementation.** This section originally read "15 points
> deliberately distinct from the 4 control points — the 11 landmarks plus the 4 corners."
> That was wrong twice over. The four half-court corners **are** four of the eleven
> landmarks (`baseline_sideline_far/near`, `midcourt_sideline_far/near`), so the set was 11
> unique points, not 15, and the control points were a *subset* of the probe set rather
> than distinct from it. Worse, the landmarks alone left the 30–47 ft band holding nothing
> but the two midcourt corners — that band's displacement would have restated the parameter
> vector back to itself, which is exactly the failure the "distinct" wording was meant to
> prevent. The five interior additions fix both: every band now holds at least one point no
> control point pins (7, 4 and 1 respectively), and `max_control_point_shift_ft` is
> reported separately so the remaining overlap is visible rather than hidden.

Displacement buckets into the same three depth bands the benchmark uses, which is what
makes "did the far court move while the near court stayed put?" answerable from the record
alone. `DEFAULT_DEPTH_BANDS` lives in `calibration/hybrid/control_points.py` beside
`band_indices` and is hashed into the config (`calibration/` must never import
`benchmark/`); `benchmark/tests/test_blindness.py::DepthBandAgreementTests` asserts both
the tuple and the binning agree with `benchmark.geometry`, including the clamped upper edge
— the half-court runs to 47.083 ft while the last band stops at 47, so a strict `<` would
have silently dropped every midcourt probe from all three bands.

## 5. Diagnostics, held-out family, and gates

All computed for **both** candidates on identical evidence.

**Held-out family.** `FREE_THROW_CIRCLE_FAR_HALF` *is* in `ACCEPTED_FEATURES` today.
**M4 must not change `ACCEPTED_FEATURES`** — that is an M3 concern whose config hash is
about to be frozen. Instead the *refiner* owns `held_out_features`, default
`(FREE_THROW_CIRCLE_FAR_HALF,)`, in `HybridRefinerConfig` and therefore in
`refiner_config_hash`. Held-out assignments get weight zero and never appear in
`fitted_features`, but still get a `PrimitiveQuality` with `fitted=False` and **real
residuals** — that *is* the held-out error. Correct layering: M3 says "this paint is the
free-throw circle"; M4 says "and I refuse to fit to it." A cross-contract test asserts
`HybridRefinerConfig.held_out_features ⊆ contracts.annotations.HELD_OUT_FAMILIES`.

**`PrimitiveQuality`** — one row per feature with ≥1 accepted fragment, plus every held-out
feature. `residual_px_median`/`_p95` are **raw, unsigned, unweighted, un-robustified**
clamped point-to-polyline distances under that candidate's transform, with a *fresh* foot
search (not the optimizer's frozen one) — the record is a measurement, not the objective.
`depth_min_ft`/`depth_max_ft` come from the court-x range of foot points on the layout
polyline via `NearestResult.interpolate` — transform-independent, and *observed* depth
contribution rather than template extent.

**`max_depth_ft`** = `max(max court-x over inlier correspondences, max depth_max_ft over
fitted primitives)`. Reads **19.0** for the keypoint candidate (verified above); a hybrid
with genuine arc support should read toward **29.0**. This directly instruments §5.5's
"extends materially beyond 19 feet."

**`supported_families`** = families `ready` in the M3 `ShadowFamilySupport` **and**
contributing ≥1 *fitted* primitive. The keypoint candidate therefore reports `()` —
honest: it claims no marking support.

**`leave_one_primitive_shift_px`** — for each fitted feature `f`, re-run the **full**
refinement from `θ⁰` with `f`'s samples removed, then
`shift_f = max over the 15 probe points of ‖π(H_hybrid,p) − π(H⁻ᶠ,p)‖`, and take the **max**
over `f` (§5.5 says "removing **any one**"). Same probe set as `TransformDifference`, so the
two are directly comparable: *"the hybrid moved 8 px from baseline, and removing the arc
moves it 11 px"* means the arc is doing all the work and the agreement is an accident.
Per-feature breakdown goes in `stability`. Cost: 2–4 extra refinements per frame,
milliseconds in numpy; `refine_ms` makes the budget visible.

### Gate list — 21 gates, all computed, none enforced as selection

`gate_id` prefixes carry scope. `candidate.*` applies to both candidates; `hybrid.*` only
to a challenger (applying "two marking families" to the baseline would wrongly mark it
unusable). `CandidateQuality.usable == all(gate.passed)` by contract. **In M4 `usable` is a
description, not an action:** it means "would be eligible for M5 to consider."

**`candidate.*` — both candidates**

| gate_id | threshold | source |
| --- | --- | --- |
| `candidate.model_point_median_px` | ≤ 6.0 px | `CalibrationPolicy.max_reprojection_px_median` |
| `candidate.model_point_p95_px` | ≤ 14.0 px | `max_reprojection_px_p95` |
| `candidate.model_point_max_shift_px` | ≤ 4.0 px | §5.5 tolerance; half the RANSAC threshold. 0 for the baseline by construction |
| `candidate.round_trip_px` | ≤ 1.0 px | `round_trip_tolerance_px` |
| `candidate.scale_plausible` | inside 0.005–0.5 ft/px | `quality.plausibility_failures` |
| `candidate.court_quad_convex` | `_is_convex` | `quality._is_convex` |
| `candidate.court_span_ft` | ≥ 8.0 ft each axis | `min_court_span_{x,y}_ft` |

**`hybrid.*` — challenger only**

| gate_id | threshold | §5.5 / §9 bullet |
| --- | --- | --- |
| `hybrid.depth_beyond_19ft` | ≥ **24.0 ft** | "extends materially beyond 19 feet". **Non-gameable:** corner straights stop at 14.26 and lane/FT-line at 19.0, so only genuine arc paint past the break can pass |
| `hybrid.independent_families` | ≥ 2 | "at least two independent marking families" |
| `hybrid.no_ambiguous_fitted` | == 0 | "no accepted fragment has an ambiguous identity" |
| `hybrid.held_out_available` | ≥ 12 samples | stops "a held-out family aligns" passing vacuously |
| `hybrid.held_out_median_px` | ≤ 4.0 px | §9 |
| `hybrid.held_out_p95_px` | ≤ 10.0 px | §9 |
| `hybrid.leave_one_primitive_shift_px` | ≤ 6.0 px | "removing any one fitted primitive…" |
| `hybrid.control_point_shift_ft` | ≤ 1.5 ft | equals the box, so it fails exactly on the boundary |
| `hybrid.bounds_inactive` | == 0 | a solution pinned to its box has not converged, it has been clipped |
| `hybrid.probe_shift_px` | ≤ 60.0 px | sanity ceiling, ~5% of frame width |
| `hybrid.scale_ratio` | inside 0.9–1.1 | implied-zoom sanity |
| `hybrid.converged` | true | |
| `hybrid.marking_residual_median_px` | ≤ 3.0 px | the fit must explain the paint it used |
| `hybrid.validation_advantage_px` | > 0.0 | "a clear validation advantage". **The one gate M5 re-tunes** to §9's 25% form; `> 0` is the weakest useful version, present so the field is populated |

**M4 computes and records all 21 and sets `usable`. M4 never emits `SELECT_CHALLENGER`.**
`SelectionDecision` is always emitted with `outcome=KEEP_BASELINE`,
`gates=(GateResult("selection.shadow_mode", passed=True),)`, and
`reasons=("shadow_mode:milestone4_never_selects_challenger",)` — so M4 records already have
the shape M5 consumes and the absence of selection is explicit rather than inferred.

## 6. Plumbing and CLI

**Primary entry point is offline** — this is the one that matters, because it needs no
video or checkpoint:

```bash
python -m calibration.cli refine-shadow --shadow-run engine_out/calibration/video_1/<run>/marking_shadow --calibrations engine_out/calibration/video_1/<run>/calibrations.jsonl --layout nba_halfcourt --out engine_out/calibration/video_1/<run>/hybrid_shadow
```

Correspondences rebuild offline from `calibrations.jsonl`: each attempt serializes its
`evidence` array, so reconstruct `LandmarkEvidence`, call `build_correspondences(evidence,
layout, min_confidence)` with `min_confidence` read from `run.json`'s recorded
`policy.min_pooled_confidence`, and filter to `calibration.inlier_landmarks`.

**Convenience (inline)** on `calibrate-frames`: `--hybrid-shadow` (implies
`--marking-shadow` and says so), `--hybrid-config PATH` (same unknown-key strictness as
`_load_marking_config`), `--no-hybrid-overlays`. A test asserts both paths produce identical
`hybrid_shadow/records.jsonl`.

Output namespace, written after and independently of `calibrations.jsonl`:

```
run_dir/hybrid_shadow/
    run.json                     # "hybrid-shadow-run-1.0.0": layout id/hash, refiner config + hash,
                                 # source marking_shadow config_hash, status counts, gate tallies
    records.jsonl                # one HybridRefinementFrame per frame (complete ledger)
    difference/<frame_id>.png    # keypoint vs hybrid markings, fitted evidence highlighted,
                                 # probe displacement arrows  (§6 diagnostic overlay requirement)
```

**Exception containment: yes, swallow like M3, with two refinements.** The orchestrator
runs beside the production keypoint path; an unhandled exception would fail a run whose
primary output is `calibrations.jsonl`. But (1) contract and provenance errors —
`shadow.layout_hash` mismatch, `image_sha256` disagreement, `frame_id` mismatch — must
raise `HybridProvenanceError` **before** the try, matching `verify_checkpoint_hash`'s
precedent, because those are wiring errors not runtime image failures. And (2) the swallowed
reason must be **typed, not `str(error)`** — M3 writes an arbitrary library message into a
content-hashed artifact, so numpy/OpenCV wording changes would make records irreproducible
for reasons unrelated to the algorithm. M4 records `f"{type(error).__name__}:{sanitized[:120]}"`
and sends the traceback to a sidecar log. Worth back-porting to `shadow.py:139`.

## 7. Benchmark comparison

New `benchmark/hybrid.py`, contract-only, exactly like `benchmark/extraction.py` — imports
`contracts.*` and `benchmark.*` and nothing from `calibration` beyond the allow-list.
**`calibration.markings` and `calibration.hybrid` are never imported.**

```python
HYBRID_SCORE_SCHEMA_VERSION = "benchmark-hybrid-score-1.0.0"

def load_hybrid_transforms(records_path) -> dict[str, dict[str, dict]]   # frame_id -> {source: entry}
def score_hybrid(records_path, references_dir, layout, *, include_held_out=True) -> dict
```

- Parse each line with `HybridRefinementFrame.from_dict`; verify
  `record.image_sha256 == annotation.image_sha256` (the guard `score_extraction` uses).
- Build per-candidate entry dicts from `CandidateTransform` and **delegate to the existing
  `score_calibration_entry`**, so the layout-hash refusal, depth banding and `_court_error_ft`
  are shared rather than reimplemented.
- Key summaries by `CandidateSource.value`, not a hardcoded pair — a later `CV_ONLY` ablation
  (§8.2's third column) then needs no scorer change.
- **Restrict every summary to the identical frame set** (frames that produced a challenger),
  plus a separate `keypoint_all_frames` summary for context; otherwise the §9 comparison is
  confounded by which frames abstained.
- `compare_summaries(keypoint, hybrid)` gives the per-band §9 shape. Add rollups read
  straight off the records: gate pass/fail counts by `gate_id`, abstention-reason histogram,
  `usable` counts, distributions of `leave_one_primitive_shift_px` / `probe_px_max` /
  `held_out_px_median`, and a per-clip breakdown supporting §8.1's rotate-held-out-clip protocol.
- An explicit `acceptance` block evaluating §9: `median_improvement_beyond_19ft_pct` /
  `meets_25pct`, `near_court_p95_regression_px` / `meets_no_near_regression`,
  `held_out_median_px` / `meets_4px`, `held_out_p95_px` / `meets_10px`, `court_median_ft` /
  `meets_0_5ft`, `court_p95_ft` / `meets_1ft`, `gross_failures` / `meets_zero_gross`,
  `coverage_change` / `meets_no_coverage_loss`. A **gross failure** is: hybrid pooled `max_px`
  exceeds keypoint's by > 20 px, **or** the 0–19 ft band p95 regresses by > 2 px or > 10%.

CLI: `command_score_hybrid` + a `score-hybrid` subparser registered in `COMMANDS`, mirroring
`score-extraction` (`--records`, `--references`, `--layout`, `--out`, `--baseline`).
Results namespace **`benchmark/results/nba_m4_v1/`** with a README mirroring
`nba_m3_v1/README.md`; never overwrite the M1 or M3 reports.

## 8. Files to modify

**New**

- `calibration/hybrid/{__init__,control_points,residuals,refine,diagnostics,orchestrator,io,overlays}.py`
- `calibration/polyline.py` (moved from `benchmark/polyline.py`)
- `calibration/tests/test_hybrid_{parameterization,residuals,refiner,isolation}.py`
- `benchmark/hybrid.py`; `benchmark/tests/test_hybrid_score.py`; `benchmark/tests/test_milestone4_artifacts.py`
- `benchmark/results/nba_m4_v1/README.md`
- `NextUp/PLAN_MILESTONE_4.md` (this plan, in repo)

**Modified**

- `contracts/hybrid_types.py` — the six new types above. **No version bump; no existing record touched.**
- `contracts/tests/test_hybrid_types.py`
- `calibration/quality.py` — public `plausibility_failures(...)` wrapper, no behaviour change
- `calibration/cli.py` — `--hybrid-shadow` / `--hybrid-config` / `--no-hybrid-overlays`; new `refine-shadow` command
- `calibration/tests/test_cli.py`
- `calibration/markings/shadow.py` — typed `failure_reasons` back-port
- `benchmark/polyline.py` — re-export shim
- `benchmark/cli.py` — `score-hybrid` subparser + `COMMANDS` entry
- `benchmark/tests/test_blindness.py` — allow-list widening + depth-band agreement test
- `docs/COURT_CALIBRATION_HYBRID_MARKINGS_PLAN.md`, `docs/CALIBRATION.md`, `README.md`,
  `PLAN_MILESTONE_2.md:47` (note on where non-promoted transforms live)

## 9. Reuse

- `calibration.estimator.project`, `matrix_to_array`, and the `_as_matrix` w-normalization convention — never invert a matrix by hand.
- `calibration.quality`: `_percentile`, `_scale_ft_per_px`, `_is_convex`, `_plausibility_failures` (via the new wrapper), and `CalibrationPolicy` thresholds as the source of every `candidate.*` gate value.
- `calibration.correspondence.build_correspondences` for the offline correspondence rebuild.
- `calibration.markings.extractor.diagonal_scale` for every pixel constant, inheriting M3's resolution-independence.
- `contracts.court_layout`: `marking()`, `sample_marking()`, `landmarks`, `content_hash()`. No independent geometry.
- `nearest_on_polyline` + `NearestResult.interpolate` for foot points and transform-independent depth.
- `benchmark.metrics`: `score_calibration_entry`, `summarize_scores`, `compare_summaries`, `FrameScore` — `score_hybrid` composes, never reimplements.
- `ShadowConfig.content_hash()`'s sorted-key JSON SHA-256 pattern for `HybridRefinerConfig`; `ShadowOrchestrator`'s abstention/never-raise/complete-ledger discipline; `write_shadow_run`'s status-count metadata shape.
- Fixtures: `calibration/tests/_synthetic.py`, `test_estimator.PLANTED` + `_correspondences`, `benchmark/synthetic.py` (`broadcast_homography`, `render_scene`, `load_truth`), `test_marking_shadow.ShadowEligibilityTests.calibration`.

## 10. Steps

- [x] **Record the current test baseline.** Measured 2026-08-01: **508 passing, 3 skipped**, via `.venv/bin/python -m unittest discover -s . -p 'test*.py'` (M2 recorded 468/3; M3 added 40).
- [x] **Extend the contracts.** Add `HybridFrameStatus`, `CandidateTransform`, `ProbeDisplacement`, `TransformDifference`, `PrimitiveStability`, `HybridRefinementFrame` with full invariants and JSON round trips. Prove the schema version is unchanged and a pre-M4 `ShadowMarkingFrame` fixture still parses.
- [x] **Relocate the polyline primitive** to `calibration/polyline.py`, re-export from `benchmark/polyline.py`, widen `ALLOWED_CALIBRATION_MODULES` with a written justification.
- [x] **Build the parameterization.** Numpy 4-point DLT, control/probe point sets, `local_pixel_scale`, box projection, convexity predicate. Prove agreement with `cv2.getPerspectiveTransform` and round-trip exactness.
- [x] **Build the residual model.** Frozen-foot perpendicular residuals, Huber/Cauchy IRLS, arc-length territory weighting, per-primitive/family/global budgets.
- [x] **Build the bounded LM.** Projected steps, convexity backtracks, ICP alternation, deterministic termination, and zero-marking stationarity.
- [x] **Build the diagnostics and gate policy.** Both candidates' `CandidateQuality`, `PrimitiveQuality`, `TransformDifference`, leave-one-primitive stability, all 21 `GateResult`s, `SelectionDecision` fixed at `KEEP_BASELINE`.
- [x] **Build the orchestrator and writers.** Eligibility layering, provenance guards, typed failure containment, `hybrid_shadow/` artifacts, and difference overlays.
- [x] **Wire the CLI.** `refine-shadow` first, then `--hybrid-shadow`; the hybrid namespace never contains `calibrations.jsonl`.
- [x] **Build the benchmark scorer.** `benchmark/hybrid.py` + `score-hybrid` delegate to `score_calibration_entry`, key by `CandidateSource`, and emit the §9 acceptance block.
- [ ] **P0 — fresh v2 calibration run.** Stored runs carry layout hash `90f53fdc…`; current v2 is `a57d0b4c…`, so `score_calibration_entry` refuses every existing transform. Run `calibrate-frames --tiers confident` over all three clips, **full frame sets (117 + 174 + 243 = 534)**, not just six, so temporal jitter stays measurable later. This is a shared prerequisite for M3's own gate, not an M4-specific one.
- [ ] **P1 — the M3 shadow run and its numeric gate.** `--marking-shadow`, then `benchmark.cli score-extraction`, then the out-of-fold variant freeze from `predefined_variants()`. **If M3 fails its precision gates, M4 is killed.** M4 must never rescue a weak extractor — a refiner fitting mis-associated paint is exactly the confident-but-wrong failure the whole plan exists to prevent.
- [ ] **P2 — freeze exactly one extraction variant.** `refiner_config_hash` is meaningless if the evidence beneath it drifts. Every M4 record carries `shadow_config_hash`; the report must show one value.
- [ ] **Run M4 diagnostics over all 534 frames**, then the §9 comparison over the six annotated frames. Evaluate the kill condition below. Record the result and its limitations; **do not weaken a threshold to obtain a pass.**
- [x] **Document and close (code).** Done condition holds: `benchmark.cli score-hybrid` compares both candidates from serialized records alone, and `calibrations.jsonl` is byte-identical with and without the hybrid path. 647 passing / 8 skipped. Selection is not implemented. Corrections recorded in §10b; the limits of what this establishes in §10c. **The milestone is not complete** — it is implemented and unevaluated, and closing it needs the P0/P1 result above.

Steps 2–10 proceed **in parallel with P0/P1** on synthetic data — roughly 90% of the code
is testable against `broadcast_homography`, `PLANTED`, and hand-built
`HybridRefinementFrame` fixtures. This matches the M1 precedent of gating on a planted-truth
probe before touching real frames.

### The evidence base is five frames — say so before running

On the locked benchmark, exactly five frames clear the M3 seed gate: `video_1` (2),
`video_2` (2), `video_3` (1); `video_3_000146` is `UNCALIBRATED_INSUFFICIENT_EVIDENCE` and
abstains. **A per-clip fold with n = 1 cannot support a threshold decision.** Therefore:
run M4 over all 534 frames for the *diagnostic* distributions (gate pass rates, abstention
reasons, stability, probe shift, and held-out error — none of which need a human
reference), and reserve the six annotated frames strictly for the §9 accuracy comparison.
Pre-commit to §8.1's conditional expansion **now**: if the six-frame result is borderline
against the 25% bar, expand to the locked nine (`video_1` f60, `video_2` f3, `video_3` f177),
which requires three more blind human annotations — a real cost to schedule before the
result arrives, not after.

### Kill condition — written down before running

> M4 is killed if, on the locked frames with a frozen M3 configuration, any of:
> **(a)** fewer than 4 of the 5 eligible frames produce a challenger at all — the refiner abstains too often to be evaluable;
> **(b)** the hybrid's held-out (free-throw-circle-far-half) median error is not lower than the keypoint candidate's on a majority of challenger-producing frames;
> **(c)** any frame shows a gross accepted failure (hybrid pooled max worse than keypoint by > 20 px) that the gates did **not** mark `usable=False` — a gate that misses a gross failure is worse than no gate;
> **(d)** `leave_one_primitive_shift_px` exceeds the 19–30 ft band's own median error on a majority of frames — the transform is decided by one primitive, so any agreement is an accident.
>
> Thresholds are not weakened to obtain a pass. If M4 is killed, the keypoint-only path is unchanged and M5 is not started.

Criterion **(b)** is the honest core. Held-out error is the only M4 measurement independent
of what was fitted, and unlike the §9 pixel thresholds it needs **no human reference** — so
it can be evaluated over all 534 frames rather than 5. It is the number to look at first.

## 10b. Corrections during implementation

Recorded here rather than quietly patched, in the style of the probe-set correction in §4.

### Defects found and fixed

The refiner, as first written, **did not recover a planted transform**: given exact,
zero-noise evidence and a seed displaced 8.49 px, it stopped 5.97 px away and reported
`converged=True`. Confidently wrong is the failure mode this milestone exists to prevent, so
it is worth naming what caused it.

| # | defect | effect | how it is now caught |
| --- | --- | --- | --- |
| 1 | **LM damping inherited across outer iterations.** `mu` was initialised once outside the loop. Re-freezing the feet defines a new objective, so inherited damping describes a function that no longer exists; one hard iteration inflated `mu`, after which no descent step existed, the solve exited as a stall, and reported where it stood | arc-only stuck at 1.80 px, and a sliding-invariance blowup to 1.8 px at one sample offset | `test_arc_alone_recovers` (reinstating it fails 2 tests) |
| 2 | **Prior weight 0.25 was a pull toward the seed, not a tie-breaker.** It measures corner offsets in court feet; at ~7 px/ft it outweighed a zero-residual data fit. Now `0.001` — bias is linear in the weight once the outer loop was fixed (0.01→0.178 px, 0.003→0.057, 0.001→0.021) | ~6 px of the total error | `PlantedRecoveryTests` (reinstating 0.25 fails 10 tests) |
| 3 | **The outer loop stopped on "barely moved this round".** ICP progress is not monotonic per iteration, so that rule was chaotic: prior 0.001 stopped at 1.70 px while 0.0003 reached 0.008 px, decided purely by which side of the threshold one iteration landed on. The movement check is now a fixed-point test evaluated at exit, never an early break | non-monotonic, irreproducible results | `test_recovers_across_seed_distance` |
| 4 | **`converged` conflated arriving with being stopped.** A stalled solve reported success, which would have let `hybrid.converged` pass on a transform the optimizer abandoned | a gate that could not fail | `ConvergenceHonestyTests` |
| 5 | **`refine-shadow` had never worked.** It crashed on `LandmarkEvidence.from_dict`, which did not exist. This is the offline path that lets the refiner be swept across M3's variants without re-reading video | the sweep would have been blocked on first use | `test_refine_shadow_reproduces_the_inline_records_exactly` |
| 6 | **Scored frames reported `UNKNOWN` status.** `CandidateTransform` carries no status while the disposition lives on the record, so `score_calibration_entry` fell back to its default and the summary's status counts told a reader nothing | misleading report, not a crash | `test_the_summary_reports_real_dispositions_not_unknown` |

**Result: 5.97 px → 0.0207 px** at default config, against the 0.05 px bar.

### Two things reported as defects that were not

Recorded because the record should not overstate what was wrong.

- **The marking tangent** was changed from a secant (interpolated foot → next vertex) to the
  full template-segment direction. This is defensible on principle — a partial chord is not
  the polyline's tangent, and it removes a `1e-12` guard that could silently zero a sample's
  constraint — but it is **not behaviourally load-bearing**: reverting it changes recovery by
  < 1e-5 px, and no honest test distinguishes the two. Any two distinct points on a straight
  segment give the same unit vector, so even a 1e-4 baseline stays accurate in float64. The
  original claim that this fixed the arc was wrong; that was defect 1, measured with
  `max_outer=40`, which was silently working around it.
- **Recorded residuals were already fresh.** `diagnostics._fresh_feature_measurements`
  re-projected the template and re-ran the foot search correctly all along. The "fix" landed
  on `raw_perpendicular_error_px`, which nothing called. That function has since been deleted:
  it and the private helper were two implementations of one quantity, and *not* equivalent
  (clamped point-to-polyline vs perpendicular-to-segment, which diverge at a clamped
  endpoint). There is now one public `residuals.measure_feature_residuals`.

### Test-design corrections

- **Gate coverage was name-matched**, which gave false confidence in both directions. It is
  now an explicit `THRESHOLD_GATES` map plus a reverse check that no gate exists without a
  threshold behind it, so adding a threshold forces you to declare its gate.
- **The CLI isolation tests were vacuous on first writing.** With blank frames the extractor
  finds nothing, the refiner abstains, and both byte-comparisons passed while comparing empty
  records. The fixture now paints the projected court, so the compared records contain a real
  fit; `test_the_refiner_actually_produced_a_challenger` stops it regressing.
- **The arc's residual floor is discretisation, not error** — the arc template is a polyline,
  so a point on the true circle sits a sagitta outside its chord. Confirmed as the textbook
  1/n²: 201 samples → 4.26e-3 px, 401 → 1.07e-3 px (exactly 4.00×). Pinned by
  `test_arc_chord_error_falls_with_template_resolution` so that if the floor ever stops being
  1/n² it gets re-diagnosed rather than given a wider threshold.

## 10c. What is and is not established

**Established.** The machinery is correct and self-consistent: it recovers a planted transform
to 0.0207 px, is invariant to sliding samples along a line (2.8e-4 px) and to sample density
(7e-4 px), is byte-for-byte deterministic and order-independent, never selects a challenger,
leaves `calibrations.jsonl` untouched, and reproduces itself through the offline path. The
depth instrumentation works: keypoint 19.0 ft → hybrid 29.0 ft on planted arc evidence.

**Not established.** Every one of those numbers comes from *planted, zero-noise* evidence, or
from a synthetic frame with the court drawn on it. Nothing here says painted markings can be
found in real broadcast footage, and the whole benefit rests on one marking — the three-point
arc, the only fitted primitive reaching past 19 ft. Nor is it known whether the held-out
free-throw circle appears often enough to serve as an independent check; if it does not, the
milestone has no falsifiable verdict available and that must be resolved before, not after,
the comparison is run. Both questions are answered by P1, need no human annotations, and can
be read off M3's own records across all 534 frames.

## 11. Verification

- Full suite via `.venv/bin/python -m unittest discover -s . -p 'test*.py'`; no regression against the recorded baseline.
- **`calibration/tests/test_hybrid_parameterization.py`:** DLT agrees with `cv2.getPerspectiveTransform` to 1e-9 on 50 deterministic quads; `H → control points → H'` round-trips to < 1e-9 px probe displacement for `PLANTED` and `broadcast_homography`; a θ outside the box lands exactly on the boundary with correct `bounds_active`; a quad-folding θ is rejected and backtracked; `local_pixel_scale` at the near baseline exceeds midcourt.
- **`calibration/tests/test_hybrid_residuals.py`:**
  - *Planted recovery* — project the layout through `PLANTED`, feed exact projected samples of lane edges + FT line + corner + arc as `ACCEPTED` evidence, seed with each control point displaced 6 px, assert recovery to < 0.05 px probe displacement.
  - *Sliding invariance* (**the single most important test in the milestone**) — replace marking samples with a different set of points along the *same* projected lines; recovered transform changes by < 1e-6 px. Proves the residual is perpendicular-only and fragment endpoints are never landmarks.
  - *Density invariance* — halve one primitive's sample count; transform changes < 0.05 px. Proves arc-length territory weighting.
  - *Budget arithmetic* — `Σ marking weights ≤ C_glob`, `Σ per family ≤ C_fam`, `Σ landmark weights == 5`.
  - *Robustness asymmetry* — one bogus fragment 40 px off moves the transform < 1 px (Cauchy redescends) while that primitive's residual reads large; one landmark moved 30 px still has residual ≥ 20 px after refinement (Huber does **not** discard it) and `candidate.model_point_max_shift_px` fires.
  - Finite-difference self-check: Jacobians at `h` and `h/2` agree to 1e-6 relative.
- **`calibration/tests/test_hybrid_refiner.py`:** zero-marking stationarity (frame `ABSTAIN`s with `no_fittable_marking_evidence`, and via the internal API with an empty fitted set the optimizer terminates at iteration 0 with `probe_px_max == 0.0` exactly and `H_hybrid` element-wise equal to `calibration.h_court_to_image`); byte-identical records across two runs and across shuffled evidence order; hold-out never in `fitted_features` with `held_out_px_median` populated for **both** candidates; `held_out_features ⊆ HELD_OUT_FAMILIES`; gate coverage (every policy field referenced by ≥1 emitted `gate_id`, every emitted id prefixed `candidate.`/`hybrid.`); `usable == all(gate.passed)`; `test_milestone4_never_selects_a_challenger` (every record `KEEP_BASELINE`, **and** an AST scan finds zero `SelectionOutcome.SELECT_CHALLENGER` in `calibration/hybrid/`); exactly F refits for F fitted primitives with `leave_one_primitive_shift_px == max(stability shifts)`; patched-LM exception yields `REFINEMENT_FAILED` with a `Type:message` reason; foreign `layout_hash` or mismatched `image_sha256` raises `HybridProvenanceError` **before** any fitting.
- **`calibration/tests/test_hybrid_isolation.py`:** `calibration/markings/**` does not import `calibration.hybrid` (one-way dependency); `calibration/hybrid/**` does not import `benchmark`/`ultralytics`/`torch`; only `io.py` writes files; the scanner catches a planted violation.
- **`calibration/tests/test_cli.py`:** `calibrations.jsonl` **byte-identical** with and without `--hybrid-shadow` (normalizing `created_utc`); `refine-shadow` reproduces the inline path's `records.jsonl` exactly from on-disk inputs; `--hybrid-config` rejects an unknown key; `hybrid_shadow/` contains no file named `calibrations.jsonl`.
- **`benchmark/tests/test_hybrid_score.py`:** `score_hybrid`'s keypoint numbers equal `score_frame`'s exactly (it must delegate); a wrong `layout_hash` or `image_sha256` is refused; both summaries cover the identical frame set; the `acceptance` block flags a planted 30% improvement as `meets_25pct: True` and 10% as `False`, and a planted 25 px max regression as a gross failure; a record carrying a `CV_ONLY` candidate produces a third summary with no code change.
- **`benchmark/tests/test_blindness.py`:** allow-list widened to include `calibration.polyline` with justification; `test_depth_bands_agree` asserts `calibration.hybrid.diagnostics.DEFAULT_DEPTH_BANDS == benchmark.geometry.DEPTH_BANDS`.
- Every record round-trips through `json.dumps(record.as_dict(), sort_keys=True)` → `from_dict`; every invariant has a rejecting test; no field name or serialized key contains "slot".
- Manual review of `hybrid_shadow/difference/*.png` on the eligible frames: highlighted fitted evidence must be paint, and displacement arrows must grow with depth rather than with proximity to the baseline.
- **Done condition:** `benchmark.cli score-hybrid` produces the three-way §8.2 comparison and the §9 acceptance block from serialized records alone, and no existing consumer's output changed by a single byte.

## 12. Open risks to revisit with real numbers

1. **Weighting constants** (`β=1.0`, `C_fam=2.0`, `C_glob=4.0`, `ℓ_ref=120 px`) determine behaviour more than anything else and are *reasoned guesses* until M3 evidence exists. Ship them as hashed config and add a `predefined_refiner_variants()` sweep (`β ∈ {0.5, 1, 2}`) under the same fold-freeze discipline as M3.
2. **`hybrid.independent_families ≥ 2`** — with the FT circle held out, only LANE and THREE_POINT remain, so both must be ready or the frame abstains. This could drive abstention to 100% on the five frames. **Measure on P1 output before finalizing;** this threshold should be measured, not chosen.
3. **The held-out family may be unavailable.** `FREE_THROW_CIRCLE_FAR_HALF` is small, sits against the FT line, and is frequently occluded by players at the top of the key. If M3 shows it is rarely extracted, `hybrid.held_out_available` never passes and **M4 has no independent validation at all** — making the milestone unfalsifiable. Contingencies: hold out `THREE_POINT_CORNER_NEAR` instead (weaker — same family as the arc), or rotate the held-out feature per frame. Needs M3 numbers, but decide the contingency in advance.
4. **Temporal jitter** is M6 scope, but the 534-frame diagnostic run exposes it for free. Consider pre-registering a jitter ceiling as a fifth kill criterion — a hybrid that is better on average and wildly unstable frame-to-frame is worse than the baseline for any downstream consumer.
5. **Refining `OK`-status frames** — §5.1 says no, and no `OK` frames exist today. The M3 eligibility gate stays unchanged.
