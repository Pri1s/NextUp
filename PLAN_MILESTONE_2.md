# Milestone 2 — Authoritative Marking Geometry Implementation Plan

**Status:** Complete

## Context

Milestone 1 established a locked human-reference benchmark. Milestone 2 is the foundation for later marking extraction and hybrid fitting: the court layout must become the single authoritative source for calibration-relevant painted centerline geometry, and generic marking-domain records must be defined without coupling them to detector slot numbers.

Current findings:

- `contracts/court_layout.py` already owns end-agnostic half-court dimensions and derived point landmarks.
- `calibration/viz.py` currently owns polyline generation plus hard-coded defaults for the NBA three-point radius (23.75 ft) and restricted-area radius (4 ft).
- `benchmark/geometry.py` is a second geometry implementation; it explicitly notes that Milestone 2 should relocate its three-point and restricted-area constants into `HalfCourtLayout`. Benchmark synthetic generation also has a separate 2-inch line-width constant.
- The NBA JSON profile already declares the three-point radius, but `load_layout()` currently ignores it. The profile does not yet declare restricted-area radius, center-circle radius, line width, or per-dimension edge/centerline conventions.
- The official NBA 2025–26 Rule 1 page/diagram is more specific than the current profile citation: all ordinary lines are 2 inches wide; court length/width are inside dimensions; lane width is outside; free-throw-line depth is outside; the three-point and circle radii are outside; and the restricted-area diagram labels its 4-foot radius as inside. These distinctions require explicit one-time conversion rather than treating every published number as a centerline.
- The layout profile currently supplies point-oriented dimensions but not authoritative typed marking primitives.
- Existing layout tests enforce derivation, symmetry, validation, schema versioning, and stable geometry hashing.

## Approach

Extend the existing layout/profile boundary so it owns typed, derived painted-centerline primitives and all source dimensions/conventions. Make visualization consume those primitives directly. Add model-neutral contracts for future extraction, association, quality assessment, and candidate selection, while keeping Milestone 2 independent of actual CV extraction/refinement logic.

Recommended shape after the inventory:

### Shared geometry contracts

- Promote the existing `MarkingFeature` and feature-kind vocabulary out of `contracts/annotations.py` into a stdlib-only shared marking contract, then re-export it from `annotations.py` so existing annotation files/imports and the frozen annotation schema retain the same string values.
- Add immutable analytic primitives for straight segments and circular arcs. Each carries a semantic feature ID, family ID, exact centerline parameters, and physical stripe width. A layout method returns a primitive by feature and samples it deterministically into tuples only when a polyline consumer needs points.
- Preserve atomic annotation features (the two three-point corner straights and arc remain separately addressable), while recording their shared `three_point` family and exact joins. Do the same for lane, circle, boundary, restricted-area, and midcourt families so later quality code can count independent families without string heuristics.
- Keep the contracts stdlib-only; NumPy conversion remains in benchmark/calibration consumers.

### Rule-book profile and coordinate conversion

- Upgrade to `half-court-layout-2.0.0`. Store each published dimension with value, unit, citation, and measurement reference (`inside_edge`, `outside_edge`, `centerline`, or `stripe_width`). Use the official NBA 2025–26 Rule 1 court diagram/page as the precise profile source rather than the current undated generic citation.
- Keep the visible-end, camera-relative axis directions, with the origin at the baseline/far-sideline painted-centerline junction. Convert published inside/outside measurements by half the applicable stripe width in the builder exactly once. With a two-inch stripe, the audited expectations include: midcourt `x = 47 ft 1 in`, court center `y = 25 ft 1 in`, basket `x = 5 ft 4 in`, free-throw centerline `x = 19 ft`, three-point radius `23 ft 8 in`, corner-line center `21 ft 11 in` from the basket (`3 ft 2 in` from the far sideline centerline), free-throw/center-circle radius `5 ft 11 in`, and restricted-area centerline radius `4 ft 1 in`. Derive the three-point break from the corrected arc radius and corner centerline so they join exactly.
- Treat source measurements and derived centerline geometry as different data: retain the former for provenance and expose the latter to all consumers. Document any small shift from v1 nominal coordinates rather than preserving an inaccurate number for output stability.
- Build exact topology and landmarks from the same derived primitives: straight/arc joins must coincide, landmark junctions must equal primitive intersections, far/near pairs must mirror, and the center circle may legitimately extend across the half-court boundary.
- Hash the schema version, coordinate convention, normalized source measurements, stripe widths, analytic primitives, topology, and landmarks. `HalfCourtLayout.as_dict()` must expose enough information to audit the geometry written into calibration run provenance.

### Model-neutral hybrid records

Create a separate `contracts/hybrid_types.py` so algorithm records do not clutter court geometry:

- `ImageMarkingSample`: image coordinate, unit tangent/direction, confidence, uncertainty, and optional observed stripe width.
- `UnlabeledMarkingEvidence`: evidence/frame/source IDs, ordered samples, and only a geometric kind hint (`straight`, `curved`, or `unknown`)—never a court identity or model slot.
- `MarkingAssignment`: evidence ID, accepted/rejected/ambiguous status, selected feature when unique, scored alternatives and association diagnostics (distance, tangent, coverage, margin), plus reason codes.
- `PrimitiveQuality`/`GateResult` and `CandidateQuality`: candidate/source IDs, model-point and per-feature/family residual summaries, supported depth/families, held-out metrics, leave-one-primitive stability, gate outcomes, and reasons. The transform itself remains in `CourtCalibration` rather than being duplicated.

  > **Amended by Milestone 4.** This holds for a transform a consumer may *use*, and
  > `CandidateQuality` is still matrix-free exactly as designed here. But
  > `CourtCalibration` enforces "usable status implies both transforms present", so a
  > *rejected* challenger cannot be expressed as one at all — and shadow-mode audit exists
  > precisely to keep those. M4 therefore adds `CandidateTransform`, an audit record for a
  > transform that has not been promoted, joined to its `CandidateQuality` by `candidate_id`.
  > `CourtCalibration` remains the sole home of a usable transform.
- `SelectionDecision`: frame ID, baseline/challenger/selected candidate IDs, explicit keep-baseline/select-challenger/no-usable outcome, gate results, and reasons.

Non-promoted transforms belong in Milestone 4 `CandidateTransform` records; the
`CourtCalibration` contract remains the sole home for a transform a consumer may
use.

All top-level records get an explicit records schema version, finite/range/uniqueness and status-dependent validation, and lossless `as_dict()`/`from_dict()` behavior through `json.dumps`/`json.loads`. Unknown schemas/enums and internally inconsistent states must fail loudly. No field or import may refer to detector slots.

Confirmed decisions:

- Make a strict layout-schema major-version upgrade; no v1 compatibility loader is required. Migrate the tracked NBA profile and fail clearly on v1 input.
- Every new generic record must support validated `as_dict()`/`from_dict()` JSON round trips immediately.

## Files to modify

Initial likely scope:

- `contracts/court_layout.py`
- `contracts/markings.py` (new shared semantic vocabulary and analytic primitives)
- `contracts/hybrid_types.py` (new model-neutral hybrid records)
- `contracts/annotations.py` (import/re-export the shared feature vocabulary)
- `contracts/layouts/nba_halfcourt.json`
- `contracts/tests/test_court_layout.py`
- `contracts/tests/test_markings.py` and `contracts/tests/test_hybrid_types.py` (new)
- `calibration/viz.py`
- `benchmark/geometry.py` (retain depth utilities and a thin NumPy adapter; remove independent court construction)
- `benchmark/glossary.py` and `benchmark/synthetic.py` (consume layout radii/widths/primitives)
- `benchmark/metrics.py` and benchmark CLI scoring path (carry/check layout hashes before scoring stored transforms)
- `benchmark/tests/test_geometry.py`, `benchmark/tests/test_synthetic.py`, and `benchmark/tests/test_metrics.py`
- `calibration/tests/test_viz.py` (new; `test_overlays.py` covers a different raw-keypoint overlay tool)
- `docs/COURT_CALIBRATION_HYBRID_MARKINGS_PLAN.md` plus relevant calibration/benchmark documentation after completion

## Reuse

- Reuse `HalfCourtLayout`, `build_layout`, `load_layout`, `validate_layout`, and `content_hash()` from `contracts/court_layout.py` rather than introducing a second geometry owner.
- Reuse the complete `MarkingFeature` vocabulary currently in `contracts/annotations.py`; move ownership rather than creating competing IDs.
- Preserve the existing visible-baseline coordinate frame and camera-relative far/near naming.
- Reuse `benchmark.geometry.MarkingPolyline`, `depth_span()`, and the existing consumers' feature-level grouping where practical, but make `MarkingPolyline` a thin NumPy view of layout-owned primitives.
- Reuse `calibration.estimator.project` and the existing overlay drawing flow in `calibration/viz.py`; replace only its private geometry construction and radius override parameters.
- Reuse calibration provenance's existing `layout_hash`; make benchmark transform loading preserve and validate it instead of discarding it.
- Follow existing frozen dataclass validation and `as_dict()` conventions in `contracts/calibration_types.py` and `contracts/annotations.py`.

## Steps

- [x] **Freeze the v2 geometry contract.** Encode the audited source-to-centerline table above from the cited NBA diagram, including the two-inch stripe width and every inside/outside convention. Store the corner rule as the published three-foot gap from the sideline's inside edge to the three-point line's outside edge (equivalently 22 feet outside from basket center), then test the derived `3 ft 2 in` centerline coordinate and exact arc join.
- [x] **Create shared marking vocabulary and primitives.** Move/re-export `MarkingFeature`, add family/kind enums plus validated straight/arc primitives and deterministic sampling, and prove annotation JSON remains compatible.
- [x] **Upgrade the layout schema and builder.** Migrate `nba_halfcourt.json` to strict v2, parse/normalize source measurements, derive all primitives and landmarks, expand `validate_layout()` to check radii, bounds where applicable, symmetry, joins, topology, widths, and landmark/primitive agreement, and reject v1 profiles clearly.
- [x] **Make provenance authoritative.** Expand layout serialization and content hashing to cover all source and derived calibration geometry; add tests showing a radius, line width, edge convention, or primitive change changes the hash.
- [x] **Collapse duplicate geometry owners.** Turn `benchmark/geometry.py` into an adapter over layout primitives; update glossary dimensions, reference-court output, synthetic truth, and painted-stripe rendering to use each primitive's layout-owned width. Remove `THREE_POINT_RADIUS_FT`, `RESTRICTED_AREA_RADIUS_FT`, `LINE_WIDTH_FT`, and the old cross-check between two independently generated courts.
- [x] **Refactor calibration visualization.** Remove hard-coded radius defaults and draw the sampled layout features directly, preserving feature/family colors and closed-curve behavior. Add numeric tests with an identity/simple homography so a custom layout dimension demonstrably moves the rendered primitive.
- [x] **Define model-neutral hybrid records.** Implement the evidence, assignment, diagnostic, candidate-quality, gate, and selection records described above with full JSON round trips and strict invariants. Add explicit tests that ambiguous assignments cannot claim a selected feature and that no API/serialized key exposes slot indices.
- [x] **Prevent mixed-layout scoring.** Preserve `layout_id`/`layout_hash` when benchmark code loads calibration records and reject scoring when a stored transform was produced under a different geometry hash. Keep Milestone 1 reports immutable historical v1 artifacts; generate future baselines under v2 rather than silently applying v2 markings to v1 transforms.
- [x] **Document and close the milestone.** Explain the v1→v2 coordinate/hash break, official source and centerline conversions, update validation output to summarize markings/convention, and mark Milestone 2 complete only after the done criteria and tests pass. Do not implement extraction, fitting, or candidate selection behavior yet.

## Verification

- Run focused contract, benchmark geometry, calibration visualization, and isolation tests with the repository's `unittest` setup (the current `.venv` does not contain `pytest`). The current focused baseline is 53 passing tests across `contracts.tests.test_court_layout`, `benchmark.tests.test_geometry`, and `calibration.tests.test_isolation`.
- Run the complete suite with `.venv/bin/python -m unittest discover -s . -p 'test*.py'`; planning baseline is **468 passing, 3 skipped**.
- Run `python -m calibration.cli validate-layout nba_halfcourt` and verify it reports v2, painted-centerline convention, source citation, marking count, and no geometry problems.
- Numerically verify all official measurement conversions, exact three-point straight/arc continuity, free-throw halves, center/restricted radii, mirror symmetry, and junction agreement. Verify sampled points lie on their analytic primitive and use deterministic endpoint-inclusive sampling.
- Round-trip every new record through `json.dumps(record.as_dict())` and `from_dict(json.loads(...))`; cover wrong schema, NaN/infinity, invalid confidence, duplicate IDs, empty support, non-unit tangent, inconsistent assignment status, and invalid selection references.
- Verify calibration visualization, benchmark scoring/reference court, glossary, and synthetic renderer all move together when a test layout radius or width changes; no independent NBA radius/line-width constants or radius override arguments may remain.
- Verify a v1 profile is rejected, every geometry-affecting field changes the layout hash, and benchmark scoring refuses a v1-hash transform under v2.
- Regenerate representative reprojection/reference-court images for manual inspection. Expected edge-to-centerline corrections are reviewed as intentional small shifts, not forced to pixel-match v1.
- Run isolation and source scans to prove `contracts/` remains stdlib-only and shared marking/hybrid records neither import adapter code nor contain `slot` fields.
- Confirm the Milestone 2 done condition: visualization and benchmark/future fitter inputs come from one layout source, while disabling future CV can still return the exact v2 keypoint candidate. The algorithm is unchanged, but v1 and v2 transform bytes are intentionally not compared because the canonical geometry changed.
