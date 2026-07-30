# Court Keypoint Labeling Guide (schema v3)

This is the labeler-facing contract for hand-labeling court keypoints. The
machine-readable source of truth is `dataset/schemas/court_keypoints.v3.json`
(22 points; validated by `python3 validate_schema.py`). Read this once before
labeling and again whenever a rule feels ambiguous — consistency matters more
than speed, because inconsistent labels are indistinguishable from noise to
the model.

## Why hand labels

The classical homography solver could not tell a symmetric partial court view
(for example, a lane seen head-on) from its mirror image — the pixel evidence
is identical. You can, because you watched the clip and can see baskets,
benches, and broadcast cues. That judgment is exactly what these labels
capture, so orientation is always **recorded data you enter**, never something
inferred later.

## The canonical court and naming

Every keypoint is a fixed, absolute court location named on the canonical
top-down diagram (sidebar "Court map", `north at top, east at right`). Names
never refer to the camera: there is no "near corner" or "left elbow", only
`north_lane_baseline_west` and friends. The 22 points:

| Area | Points |
|---|---|
| Each end (×2) | baseline×sideline corners (2), three-point×baseline junctions (2), lane×baseline corners (2), lane×free-throw corners (2), three-point arc apex (1) |
| Midcourt | midcourt×sideline (2), center-circle×midcourt-line (2) |

## Orientation: which end is north?

**Rule: north is the image-left basket.** With north at image-left, the far
sideline is east and the near sideline is west.

- **Once per clip** you lock the anchor: on a frame where both baskets are
  discernible, confirm the lock (north = image-left is then true by
  definition). If no frame in the clip shows both ends, declare which end the
  visible one is instead.
- **Once per frame** you confirm `visible ends`: **North (1) / Both (2) /
  South (3)**. The choice is prefilled from your last answer but must be
  explicitly confirmed on every frame before saving. The editor blocks
  placing points that belong to an end you said is not visible.
- Assumption behind the rule: the camera stays on one sideline and the court
  axis runs roughly left-right in frame. **Skip frames at triage** where that
  is not true (behind-the-basket views, extreme tilt).

## Point states

Each of the 22 points is in exactly one of three states:

- **Labeled, visible (v=2)** — you can see the mark and clicked it.
- **Labeled, occluded (v=1)** — the spot is hidden (player, referee, overlay)
  but you can pin where it is from surrounding geometry. Place it, then press
  `V` (or right-click) to toggle occluded.
- **Not labeled (v=0)** — off-frame or too uncertain to place. Never guess:
  a wrong confident point hurts more than a missing one.

Precision notes:
- Line intersections: click the intersection of the line *centers*, not the
  paint edges.
- Three-point apexes: the farthest point of the arc from the baseline, on the
  axis through the rim — use the rim/backboard as the axis cue. If you cannot
  pin it within a ball's width, mark it occluded or leave it unlabeled.

## Frame selection: diversity over volume

Initial batch target: **~150–250 labeled frames spread over every clip**,
soft cap ~25–30 keeps per clip. Deliberately cover: north-only, south-only,
both-ends-wide, and midcourt-pan views; zoomed-in and wide shots; clean and
player-crowded lanes; different courts/venues as footage arrives. A few
hundred varied frames beat thousands of near-duplicates.

## Pipeline

```
extract_frames.py  →  triage (keep/skip)  →  anchor clip orientation
   →  label + confirm visible ends
```

Saved labels live in `dataset/labels/<clip_id>/<frame_id>.json` and carry the
schema version, the clip orientation, your `visible_ends` answer, and all 22
`{x, y, v}` points.
