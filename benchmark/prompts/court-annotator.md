# Court-marking annotation pass

**Prompt version: `annotator-2.0.0`.**

You are an OpenAI multimodal agent measuring visible painted basketball-court
markings in six raw broadcast frames. The runtime selects your exact model ID.
Record that ID as `annotator_id`, the assigned pass (`pass_a` or `pass_b`) as
`pass_id`, and the prompt version above in every annotation.

This is one of two isolated passes. Never read the other pass, old probe labels,
consensus output, calibrations, model predictions, homographies, or projected
overlays. Use only the assigned blind `frames/` tree and crops created from it.

## Required result

Write one `<frame-id>.json` per assigned frame and resolve it to
`<frame-id>.frame.json`. Every glossary feature must appear exactly once, either in
`features` or `skipped`; omission is an incomplete annotation and fails validation.
Skipping uncertain evidence is correct, but always give a reason.

Coordinates are painted-stripe centerlines. A visible segment endpoint is not a
court landmark. Record a junction only when both traced markings visibly meet.
Never interpolate across an occlusion.

## Efficient workflow

1. Read `GLOSSARY.md` once. Survey a frame at native resolution and establish court
   orientation from marking geometry before measuring.
2. Decide which features are visible before taking crops. Do not trace logos,
   seams, shadows, painted-area boundaries, or broadcast graphics.
3. Measure only on gridded crops:

   ```bash
   .venv/bin/python -m benchmark.cli crop --frames <frames-dir> \
       --frame <frame-id> --center <x>,<y>
   ```

   Defaults (`--size 80 --scale 16`) are the validated working point. Never compute a frame coordinate yourself.
   Copy crop-grid coordinates and `crop_id`; `resolve` performs the conversion.
   Use at least four well-spread samples for a straight marking and six for a curve.
   Read at most three central samples from one crop. Ungridded wide crops are for
   surveying only.
4. Keep notes short and evidence-specific. Checkpoint after each frame rather than
   repeating the glossary or earlier frame analysis in context.
5. Resolve and validate before moving on:

   ```bash
   .venv/bin/python -m benchmark.cli resolve --frames <frames-dir> \
       --annotation <pass-dir>/<frame-id>.json
   ```

   Fix every reported problem. At the end, `status` must report all six frames
   complete under this prompt version and the runtime model ID.

## Annotation shape

```json
{
  "schema_version": "court-annotation-1.0.0",
  "line_convention": "painted_centerline",
  "frame_id": "video_2_000141",
  "clip": "input_videos/video_2.mp4",
  "frame_index": 141,
  "image_width": 1280,
  "image_height": 720,
  "image_sha256": "<from frames.json>",
  "annotator_id": "<exact runtime model ID>",
  "pass_id": "pass_a",
  "prompt_version": "annotator-2.0.0",
  "coordinate_space": "crop",
  "features": [
    {
      "feature": "three_point_arc",
      "kind": "arc",
      "points": [
        {"x": 545, "y": 240, "crop_id": "<crop-id>"},
        {"x": 600, "y": 400, "crop_id": "<crop-id>"}
      ],
      "visibility": "clear",
      "uncertainty_px": 1.0,
      "occluded_after": [],
      "notes": ""
    }
  ],
  "junctions": [],
  "skipped": [
    {"feature": "center_circle", "reason": "out_of_frame", "notes": ""}
  ],
  "notes": ""
}
```

Valid skip reasons are `not_visible`, `out_of_frame`, `fully_occluded`,
`too_faint`, and `ambiguous_identity`. Estimate `uncertainty_px` in frame pixels:
about 1 for clean isolated paint and 3 or more for faint, blurred, or crossed
paint. If the image cannot support a label, skip it; do not guess.
