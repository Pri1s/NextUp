# Interim court calibration

Calibration that runs on a detector whose slot semantics are **not verified**.

The Phase-0 gate ([`PHASE0.md`](PHASE0.md)) has not been cleared: no human has
signed off on what `models/court_keypoint_detector.pt`'s 18 slots mean. Rather
than wait, this engine treats slot output as *noisy, partially ambiguous
observations* and supplies the rigour the detector cannot. Nothing here promotes
a slot to a fixed landmark identity.

## The six rules this implements

| Rule | Where |
| --- | --- |
| Fit only from independently supported landmarks | `--tiers confident` (default) over `adapters/maps/reloc2_18_provisional.json` |
| Treat duplicate slots as interchangeable evidence, pooling confidence | `adapters/evidence.py::pool_evidence` |
| Reject ambiguous mappings rather than assigning an id | slots 3/12 carry no `landmark_id`; enforced by a test |
| Robust fitting plus reprojection and plausibility gates | `estimator.py` (MAGSAC++ → RANSAC), `quality.py` |
| Explicit uncalibrated result on thin evidence | `CalibrationStatus.UNCALIBRATED_INSUFFICIENT_EVIDENCE` |
| Court end left unresolved | half-court frame; `end_identity == "unresolved"` |

Temporal propagation (reusing a recent good calibration across nearby frames,
revalidated against current observations) is **not** implemented yet. It is the
next increment and needs the quality gates below to validate against.

## Coordinate frame

End-agnostic half court, anchored on whichever end is in shot:

```text
        far sideline centerline  Y = 0
        +---------------------------------+
        |                                 |
  X = 0 |  visible baseline               |  X = 47 ft 1 in  midcourt
        |                                 |
        +---------------------------------+
        near sideline centerline Y = 50 ft 2 in
```

Both axes are determinate from the image; neither requires knowing north from
south. Distances, lane positions and shot ranges are all correct here. An
orientation owner can later map half court to full court by supplying end
identity alone, with no geometry changing.

### Authoritative v2 painted geometry

Layout schema `half-court-layout-2.0.0` uses the painted stripe centerline for
all marking coordinates and junctions. The NBA profile cites the Official
2025–26 NBA Playing Rules, Rule No. 1 court diagram, retains every published
inside/outside edge convention, and derives centerlines using the official
2-inch stripe width. Examples are the 23 ft 8 in three-point centerline radius,
3 ft 2 in corner centerline coordinate, 5 ft 11 in circle radii, and 4 ft 1 in
restricted-area centerline radius.

This is an intentional coordinate/hash break from v1, which used nominal
rule-book edge values directly as if they were centerlines. Existing Milestone 1
reports remain historical v1 results. A v1 transform must not be applied to v2
markings; benchmark scoring enforces matching `layout_hash` provenance. New
keypoint and later hybrid candidates are generated under v2, and hybrid fallback
must return the exact v2 keypoint candidate.

Analytic primitives and physical stripe widths live in `HalfCourtLayout`.
`calibration/viz.py`, benchmark geometry/scoring, the annotation reference court,
glossary, synthetic truth, and future fitting all consume that one source.

## Evidence pooling

Phase-0 inspection found six slot pairs whose members predict the same physical
point a few pixels apart and split their confidence, summing to ≈1.0. Choosing
the more confident member discards half the signal and makes an even split look
like two weak observations rather than one strong one — so a landmark the model
is *most* certain about can fall below a fixed gate.

Pooled confidence is the sum; pooled position is the confidence-weighted mean.
A `max_pair_spread_px` guard drops a group whose members disagree wildly, because
at that point the "same physical point" premise has broken and averaging would
invent a point neither slot predicted.

## Run it

```bash
python -m calibration.cli calibrate-frames --weights models/court_keypoint_detector.pt --video input_videos/video_2.mp4 --num-frames 24 --device mps
```

```bash
python -m calibration.cli validate-layout nba_halfcourt
```

## Hybrid shadow refinement

Milestone 4 consumes serialized `marking_shadow/` records offline and writes
candidate transforms and diagnostics only under `hybrid_shadow/`:

```bash
python -m calibration.cli refine-shadow \
  --shadow-run <run>/marking_shadow \
  --calibrations <run>/calibrations.jsonl \
  --layout nba_halfcourt --out <run>/hybrid_shadow
```

The keypoint and hybrid candidates are recorded side by side, but M4 always
keeps the keypoint baseline. Refinement never mutates `calibrations.jsonl`.

Outputs land in `engine_out/calibration/<clip>/<run_id>/`: `calibrations.jsonl`
(one attempt per frame with evidence and drop reasons), `run.json` (layout,
policy, provenance, status counts), and `reprojected/frame_*.jpg`.

## Statuses

| Status | Meaning |
| --- | --- |
| `OK` | usable, all gates passed |
| `DEGRADED` | usable, with reasons attached — read them |
| `UNCALIBRATED_INSUFFICIENT_EVIDENCE` | court visible, evidence too thin or uncertain to fit |
| `FAILED_INSUFFICIENT_POINTS` | fewer correspondences than a homography needs |
| `FAILED_DEGENERATE_GEOMETRY` | collinear or clustered evidence |
| `FAILED_HIGH_REPROJECTION_ERROR` | fit rejected on residuals or held-out error |
| `FAILED_IMPLAUSIBLE` | implied scale or court shape is physically impossible |
| `NOT_ATTEMPTED_NO_COURT` | detector found nothing; not a failure |

Branch on `calibration.usable`. Anything that needs to explain itself reads
`quality.reasons`.

## Current results, and the honest caveat

On the three supplied clips, 24 sampled frames each, `--tiers confident`:

| clip | usable | uncalibrated | failed |
| --- | --- | --- | --- |
| video_1 | 18 (all DEGRADED) | 4 | 2 implausible |
| video_2 | 24 (all DEGRADED) | 0 | 0 |
| video_3 | 19 (all DEGRADED) | 4 | 1 implausible |

Reprojection residuals are sub-pixel and held-out error is 1.9–2.3 px. **Neither
number should be read as accuracy.**

The five confident landmarks are `(0,0)`, `(0,3)`, `(0,17)`, `(19,17)`, `(19,33)`
— three of them collinear on the baseline. That leaves only two points carrying
information along the court's length, against a homography's eight degrees of
freedom. The fit is therefore well constrained near the evidence and
*extrapolates* everywhere else, which is exactly the regime where every residual
stays tiny while the projection drifts.

The reprojected-markings images make this visible: the lane traces the paint
closely, while the free-throw circle and restricted-area arc sit tens of pixels
off. At the measured scale (~0.04 ft/px) that is roughly one to three feet.

Every such frame is now reported `DEGRADED` with
`weak_conditioning:only_2_points_off_dominant_line`. Treat current output as
usable for coarse, near-evidence questions and **not** for anything depending on
the far half of the court.

### Why the reprojected images are an output, not a debugging aid

Every numeric gate measures a fit against its own evidence — a test of
self-consistency. With five points against eight degrees of freedom,
self-consistency is cheap. Projecting the court markings back onto the frame is
the only check whose reference is the floor rather than the detector, and during
development it caught a systematic error that sub-pixel reprojection error and a
2 px held-out score both rated healthy. Do not disable it casually.

## The single highest-value improvement

Enabling the midcourt landmarks (slots 6 and 7, `probable` tier) extends the
evidence's court-length span from 19 ft to 47 ft and breaks the collinearity:

```bash
python -m calibration.cli calibrate-frames --weights models/court_keypoint_detector.pt --video input_videos/video_1.mp4 --tiers confident,probable --device mps
```

On video_1 this moves 7 frames from `DEGRADED` to `OK` and drops uncalibrated
frames from 4 to 1. They are held at `probable` because video_2 gives them no
support at all — the clip never shows midcourt. Promoting them is a decision for
whoever signs off the slot review.

## Tests

```bash
python -m unittest discover -s calibration/tests -p "test_*.py" -t .
```

```bash
python -m unittest discover -s contracts/tests -p "test_*.py" -t .
```

256 tests. The load-bearing ones: a planted-homography round trip that must
recover a known transform, degeneracy and scrambled-correspondence refusals, the
pooling arithmetic, and a guard that no adapter map may claim verified status.
