# Court Calibration Module — Architecture & Design Plan

**Status:** Static-frame interim calibration implemented; temporal propagation and broader court-type support remain planned. See [`CALIBRATION.md`](CALIBRATION.md).
**Target repository:** `NextUp/`, behind the `contracts/` isolation wall.
**Starting point:** clean-room; do not use the deleted `6a48932` engine tree as a basis.

## 1. Objective

Convert broadcast/stream basketball footage into **metric court coordinates** so downstream analytics reason in feet on a court plane rather than pixels. Calibration detects court landmarks, matches them to known real-world positions, and solves the image↔court homography.

The engine must support NBA, NCAA, NFHS/high-school, and FIBA courts. Landmark identities are shared; dimensions are not. Adding a court type must be a JSON file and swapping a detector must be configuration only. Calibration, quality gating, and analytics must remain unchanged.

## 2. Reference-repository findings and anti-requirements

The reference pipeline (`basketball_analysis/`) loads a YOLO pose model in batches, passes raw Ultralytics `Keypoints` objects through the pipeline, uses one hard-coded 18-point template, and fits a plain least-squares homography without RANSAC, quality metrics, inverse transform, or explicit failures. It uses `(0, 0)` as a missing-keypoint sentinel, ignores keypoint confidence, assumes instance 0, and has no temporal handling.

The template declares FIBA dimensions (`28 × 15 m`) while its offsets are NBA-derived. This introduces systematic position-dependent geometry error that downstream speed/distance calculations inherit. The landmark set is baseline/lane-heavy, so typical broadcast mid-court framing may expose fewer than four usable points. Index order is treated as semantic identity, and the implementation assumes one roughly level sideline camera and a rectilinear lens.

### Explicit anti-requirements

| Do not do this                                    | Instead                                                                                |
| ------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Use `(0,0)` to represent absent keypoints         | Omit absent observations entirely                                                      |
| Pass Ultralytics types through the engine         | Adapters emit plain semantic dataclasses                                               |
| Expose APIs accepting all video frames in memory  | Calibrate strictly per frame                                                           |
| Cache by frame-list length                        | Use content-addressed cache keyed by video, frame, weights, adapter, and policy hashes |
| Use tactical-image pixels as court coordinates    | Use a metric court plane; keep rendering separate                                      |
| Swallow OpenCV errors                             | Return explicit calibration statuses                                                   |
| Make calibration own player foot-point extraction | Expose transforms only; player perception owns its image point                         |
| Couple detector slots to geometry                 | Keep model-index maps and court layouts as independent data                            |

## 3. Model facts and Phase-0 gate

The supplied `models/court_keypoint_detector.pt` is an Ultralytics `yolov8x-pose` fine-tune with `kpt_shape: [18, 3]`, one `basketball` instance class, trained at `imgsz=640`. It appears to be the same 18-point reloc2-family model used by the reference, rather than NextUp’s 22-point v3 schema. Its point semantics are not confirmed.

Training used `fliplr=0.5`; an unverified or identity `flip_idx` may have trained mirrored images with unmirrored landmark identities. This can corrupt east/west or north/south identity even when aggregate pose mAP is good. The 18-point set lacks center-circle and three-point-apex points, making coverage on broadcast mid-court frames a major risk.

**Nothing beyond Phase 0 begins until the model’s index semantics, flip behavior, and negative-frame behavior are verified.**

## 4. Package architecture

```text
contracts/                          # Stable, stdlib-only isolation wall
  landmarks.py                       # LandmarkId vocabulary and metadata
  court_layout.py                    # CourtLayout, builder, loader, validators
  layouts/nba_94x50.json             # Profiles are data
  calibration_types.py               # Detection, calibration, quality, status contracts
  tests/

calibration/                         # Imports contracts + third-party only
  detectors/base.py                  # KeypointDetector protocol
  detectors/yolo_pose.py             # Ultralytics runner -> RawKeypointFrame
  adapters/base.py                   # KeypointAdapter protocol
  adapters/indexed.py                # Table-driven model-index adapter
  adapters/maps/reloc2_18.json       # Phase-0-verified map
  correspondence.py                  # Only detector/layout coupling point
  estimator.py                       # Robust homography and refinement
  quality.py                         # Quality metrics, plausibility, status
  calibrator.py                      # FrameCalibrator public entry point
  temporal.py                        # Clip state machine (Phase 3)
  viz.py                             # Overlays, projected markings, bird’s-eye, warp
  io/records.py                      # calibration-1.0.0 JSONL serialization
  cli.py
  tests/
```

An isolation test must parse imports under `contracts/` and `calibration/` and fail if either imports root training scripts such as `serve`, `extract_frames`, `export_yolo`, `train_pose`, `validate_schema`, or `migrate_to_v3`.

### Layer boundary

```text
BGR frame
  -> detector (model-specific, court-agnostic)
RawKeypointFrame (native slots/confidences/original pixels)
  -> adapter (model-specific, court-agnostic)
KeypointDetection (semantic landmark IDs)
  + CourtLayout (court-specific, model-agnostic)
  -> correspondence.py
Correspondences
  -> estimator.py + quality.py
CourtCalibration (both transforms, status, quality, provenance)
  -> temporal.py for video
Downstream analytics consume only transforms, status, and quality.
```

Nothing left of `correspondence.py` knows the court; nothing right of it knows the model.

## 5. Core contracts

### Landmark vocabulary

`contracts/landmarks.py` defines a universal `LandmarkId` superset of NextUp’s 22 v3 identifiers plus structural metadata:

- descriptive, non-metric feature name;
- end: `north`, `south`, or `mid`;
- side: `east`, `west`, or `center`;
- kind: line intersection, arc apex, or circle intersection;
- north/south and east/west mirror partners.

A test reads `dataset/schemas/court_keypoints.v3.json` as data and asserts its IDs are a subset of this vocabulary. Training code remains outside the isolation wall.

### Detection contract

`LandmarkObservation` contains semantic landmark ID, original full-resolution pixel `(x, y)`, continuous `[0,1]` confidence, and native source index. `KeypointDetection` contains frame ID, image size, a tuple of observations, instances seen, and model provenance.

Rules:

- Pixel coordinates use top-left origin and y-down, and always refer to the original full-resolution frame.
- Confidence remains continuous.
- Below-gate or undetected landmarks are absent, never zeroed.
- Detection has no court type, dimensions, or units.

### Layout contract

`CourtLayout` contains layout ID, rule set, rule-book citation, normalized units, length/width, semantic-landmark coordinates, absent landmarks, marking primitives, and a content hash.

**Normative coordinate convention:** origin is the north-baseline × west-sideline corner; `+X` runs along court length toward the south baseline; `+Y` runs across court width toward the east sideline. Internally normalize to feet at layout load, while retaining explicit units/provenance.

### Calibration contract

`CalibrationStatus` values:

- `OK`
- `DEGRADED`
- `FAILED_INSUFFICIENT_POINTS`
- `FAILED_DEGENERATE_GEOMETRY`
- `FAILED_HIGH_REPROJECTION_ERROR`
- `FAILED_IMPLAUSIBLE`
- `NOT_ATTEMPTED_NO_COURT`

`CourtCalibration` stores frame/layout IDs, units, status, **both** `H_image_to_court` and `H_court_to_image`, full quality data, inlier/outlier landmark IDs, source (`direct`, `propagated`, `smoothed`), source age, and hashes/version provenance. Construction verifies the transforms are mutually inverse within tolerance. Downstream users call `image_to_court()` and `court_to_image()` rather than invert matrices themselves.

## 6. Layout modularity

Layouts declare rule-book scalars rather than hand-entered point tables. A pure builder derives landmarks and marking primitives.

Example profile shape:

```json
{
  "layout_id": "nba_94x50",
  "schema_version": "court-layout-1.0.0",
  "rule_set": "NBA",
  "source": "NBA Rule Book, Rule 1 (edition/date)",
  "units": "feet",
  "dimensions": { "length": 94.0, "width": 50.0 },
  "lane": { "shape": "rectangle", "width": 16.0, "free_throw_distance": 19.0 },
  "basket": { "center_from_baseline": 5.25 },
  "three_point": {
    "radius": 23.75,
    "corner_inset": 3.0,
    "corner_straight": true
  },
  "center_circle": { "radius": 6.0 },
  "features": { "restricted_area": true, "restricted_area_radius": 4.0 },
  "absent_landmarks": []
}
```

Adding a profile:

1. Write `contracts/layouts/<id>.json` with current rule-book values and citation.
2. Run `python -m calibration.cli validate-layout <id>`.
3. Run calibration with `--layout <id>`.

Validation checks IDs, in-bounds coordinates, center/mid constraints, mirror symmetry, and consistency between markings and intersecting landmarks. A parametrized planted-homography test automatically covers every registered profile.

Keep these mappings separate:

| Mapping                            | Owner                      | Varies by     |
| ---------------------------------- | -------------------------- | ------------- |
| model index → `LandmarkId`         | `adapters/maps/*.json`     | model         |
| `LandmarkId` → descriptive feature | `contracts/landmarks.py`   | never         |
| `LandmarkId` → court coordinate    | `contracts/layouts/*.json` | court profile |

The v3 schema’s embedded NFHS coordinates must not be the engine source of truth; layouts own geometry while the training schema remains unchanged.

## 7. Estimation and reliability

### Correspondences

`build_correspondences(detection, layout, policy)` removes observations below confidence threshold or absent from the layout and records every drop reason. It emits image points, court points, semantic IDs, and confidence-derived refinement weights. This is the only model/geometry coupling point.

### Homography

Solve **court → image** first because measurement noise and RANSAC threshold are meaningful in image pixels. Use:

```python
cv2.findHomography(court_pts, image_pts, cv2.USAC_MAGSAC,
                   threshold_px, confidence=0.999, maxIters=10000)
```

Fall back to `cv2.RANSAC` if unavailable. Refit inliers using confidence-weighted Levenberg–Marquardt, invert once, store both transforms, and verify round-trip identity. Four points is the absolute minimum; the default preferred minimum is six and lower counts are at most `DEGRADED`.

### Required quality signals

Every attempt records:

1. Observation/inlier count and inlier ratio.
2. Spatial coverage: inlier image-hull fraction, along/across court span in feet, and minimum triangle area in pixels.
3. Reprojection errors in both image pixels and court feet (mean, median, p95, max).
4. Held-out/leave-k-out error when enough inliers exist; this is necessary to expose a systematically wrong adapter map.
5. Plausibility: orientation, convexity of frame corners in court space, court-in-frame fraction, projected one-foot scale sanity, and horizon/vanishing-line sanity.
6. Explicit reasons for degraded or failed status.

Thresholds live in `CalibrationPolicy` with per-layout overrides, not as court dimensions or magic constants in estimator/quality code.

### Orientation

A 180° rotation can reproject perfectly. The module provides geometric calibration plus landmark evidence, but absolute basket/end identity needs an explicit clip-level owner (aggregated landmark votes, scoreboard, or possession context). Do not silently inherit a possibly wrong orientation.

## 8. Phased implementation plan

### Phase 0 — Model truth (hard gate)

Build `calibration/tools/inspect_model.py` and:

1. Run the model over at least 20 sampled NBA frames; write index/confidence JPEG overlays and CSV rows `(frame, index, x, y, conf)`.
2. Obtain human sign-off and populate `calibration/adapters/maps/reloc2_18.json` with model ID, weights SHA-256, keypoint count, index-to-landmark map, verification frames/reviewer, and notes.
3. Fit candidate-map homographies against `nba_94x50` and measure leave-one-out reprojection error.
4. Run a horizontal flip probe: compare original predictions with mirrored-image predictions mapped back via `x' = W - 1 - x`; determine whether slots correspond to themselves, mirror partners, or incoherent locations.
5. Run crowd/replay/ad negative frames to define the `NOT_ATTEMPTED_NO_COURT` gate.

**Exit criteria:** all 18 slots have human-approved assignments, median held-out reprojection error meets policy target, and the flip probe is coherent. If not, escalate before calibration implementation: resolve orientation per clip, restrict to end-agnostic use, or retrain with correct `flip_idx`.

### Phase 1 — Static-frame MVP

Implement contracts, NBA layout/builder/validator, YOLO detector, indexed adapter, correspondence construction, robust estimator, quality evaluation, `FrameCalibrator`, visualization, and `calibrate-frame` CLI.

Deliverable: one deliberately selected NBA frame, five proof artifacts, and one calibration JSON.

### Phase 2 — Frame-set robustness

Run N sampled frames from one clip without temporal state. Report status success rate, reprojection histogram, inlier histogram, failure reasons, and missing-landmark frequency. Tune `CalibrationPolicy` from data and add held-out cross-validation.

### Phase 3 — Stable video calibration

Implement `temporal.py` as a per-clip state machine:

- detect cuts with downscaled grayscale histogram correlation and hard-reset at each cut;
- re-estimate each frame and rate-limit suspicious inter-frame transform change using projected canonical court points;
- propagate the last good calibration only for limited age when landmarks are absent, optionally using static-region optical flow;
- never smooth raw homography entries; smooth image positions of four canonical court points and re-solve.

Deliverable: annotated video with full reprojected markings plus per-frame quality timeline. Target high game-camera coverage, zero confident-but-wrong homographies, and no calibration crossing cuts.

### Phase 4 — Second layout profile

Add `contracts/layouts/nfhs_84x50.json` and use existing EYBL footage with the 22-point `runs/pose/court_pose_v2` model behind an identity adapter:

```bash
--layout nfhs_84x50 --detector court_pose_v2
```

The modularity claim is tested by requiring no changes in `calibration/` outside adapter-map data and no changes outside layout data for geometry.

## 9. MVP validation artifacts

For one NBA frame, emit:

| Artifact                   | Proof                                                                                                  |
| -------------------------- | ------------------------------------------------------------------------------------------------------ |
| `keypoints_overlay.jpg`    | Semantic names/confidences with inliers green, RANSAC outliers red, confidence-gated observations grey |
| `reprojected_markings.jpg` | Entire court markings projected through `H_court_to_image`; arcs/lines align with paint                |
| `warped_to_court.png`      | Frame warped with `H_image_to_court` and layout lines overlaid                                         |
| `birdseye.png`             | Fixed-scale top-down court with externally supplied player-foot positions                              |
| `calibration.json`         | Both matrices, status/reasons, quality, inlier/outlier IDs, and all provenance hashes                  |

The projected full-court markings are the primary visual correctness artifact.

## 10. Automated validation

- Planted-homography round-trip, parametrized for every layout, with recovery/projection tolerance `< 1e-6`.
- Layout validator tests for mirror symmetry, bounds, center/mid constraints, and marking consistency.
- Adapter tests: unmapped slots dropped; gated points absent rather than zeroed; source index retained; letterbox coordinates restored to original pixels.
- Failure-state tests: collinear points, three points, clustered points, and deliberately scrambled maps.
- Transform round-trip tests across the image frame.
- Import-isolation test for contracts/calibration.

```bash
python -m calibration.cli inspect-model --weights models/court_keypoint_detector.pt --video <nba_clip>.mp4 --out engine_out/phase0
python -m calibration.cli calibrate-frame --video <nba_clip>.mp4 --frame 40 --layout nba_94x50 --adapter reloc2_18 --out engine_out/mvp
python -m calibration.cli validate-layout nba_94x50
python -m unittest discover -s calibration/tests -p "test_*.py"
```

## 11. Risks and decisions to resolve

1. Confirm whether `models/court_keypoint_detector.pt` is the intended model; otherwise provide the newer model’s keypoint shape, label order, and schema relation.
2. Retrieve `reloc2-7/data.yaml` if possible, especially `flip_idx` and keypoint names.
3. Sweep/choose inference `imgsz` (640 matching training versus higher resolution for far landmarks).
4. Locate any authoritative reloc2 index-to-landmark mapping.
5. Select a longer, stable NBA validation clip; short highlight cuts are poor first targets.
6. Define camera scope: broadcast-only versus fixed/wide/high-school cameras, which affects lens-distortion support.
7. Choose output representation: JSONL, Parquet, or in-process only.
8. Decide whether to add content-addressed calibration caching in Phase 2.
9. Assign ownership of absolute court-end orientation.
10. Record current rule-book sources and variants for NCAA/NFHS/FIBA; do not assume one generic “college” profile.

Domain shift is expected: NBA-trained detection does not become correct on NFHS courts merely by swapping layout geometry. Wide lenses, multi-sport lines, glare, oblique cameras, and different markings may require the NextUp 22-point model or further training.

## 12. Review checklist

Before implementation begins, confirm:

- [ ] The intended model is identified.
- [ ] Phase-0 exit criteria are accepted as a hard gate.
- [ ] The coordinate convention is accepted: north-west origin, +X along, +Y across, feet internally.
- [ ] `contracts/` and `calibration/` are accepted as top-level packages.
- [ ] An NBA validation clip is available, or a supplied reference clip is accepted.
- [ ] `CourtCalibration` serialization target is chosen.
