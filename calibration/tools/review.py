"""Reviewer artifacts.

Phase 0 ends in a human judgement, not a metric, so the output has to be built
for a person: one page that opens from disk with no server and no network, the
evidence ordered so the decisive view comes first, and a worksheet to write the
verdict into.

Three files:

``review.html``               the reading surface — zero JavaScript, no external
                              hosts, images by relative path, ``<details>`` for
                              collapsing rather than script.
``slot_review.md``            the worksheet the reviewer fills in by hand.
``slot_review.template.json`` the same worksheet, machine-readable, with every
                              assignment ``null`` and ``status: "unverified"``.

None of these is an adapter map, and the template is deliberately not shaped like
one. A map may only be written after a human signs off, and that step is out of
scope here.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Sequence

REVIEW_SCHEMA_VERSION = "slot-review-1.0.0"

_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { margin: 0; padding: 2rem 1.5rem 6rem; font: 15px/1.55 -apple-system, BlinkMacSystemFont,
  "Segoe UI", Helvetica, Arial, sans-serif; background: #fbfbfa; color: #1a1a1a; }
main { max-width: 1180px; margin: 0 auto; }
h1 { font-size: 1.6rem; margin: 0 0 .3rem; letter-spacing: -.01em; }
h2 { font-size: 1.15rem; margin: 2.4rem 0 .7rem; padding-bottom: .35rem;
  border-bottom: 1px solid #d8d6d1; }
h3 { font-size: .95rem; margin: 1.4rem 0 .4rem; font-weight: 600; }
p { margin: .5rem 0; }
.sub { color: #6b6862; font-size: .85rem; margin-bottom: 1.4rem; }
.warn { border-left: 3px solid #b8791f; background: #fdf6e9; padding: .8rem 1rem;
  margin: 1.2rem 0; font-size: .9rem; }
table { border-collapse: collapse; width: 100%; font-size: .82rem; margin: .5rem 0 1rem; }
th, td { text-align: left; padding: .34rem .55rem; border-bottom: 1px solid #e6e4df;
  white-space: nowrap; }
th { font-weight: 600; background: #f3f1ec; position: sticky; top: 0; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.scroll { overflow-x: auto; max-width: 100%; }
.strip { overflow-x: auto; border: 1px solid #e0ded9; background: #111; border-radius: 4px; }
.strip img { display: block; max-width: none; }
.gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(330px, 1fr));
  gap: .8rem; }
.gallery figure { margin: 0; }
.gallery img { width: 100%; border: 1px solid #ddd; border-radius: 3px; display: block; }
.gallery figcaption { font-size: .74rem; color: #6b6862; padding-top: .2rem; }
details { margin: .8rem 0; border: 1px solid #e0ded9; border-radius: 5px; background: #fff; }
details > summary { cursor: pointer; padding: .6rem .9rem; font-weight: 600; font-size: .9rem; }
details[open] > summary { border-bottom: 1px solid #e6e4df; }
.body { padding: .3rem 1rem 1rem; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .85em;
  background: #efedE8; padding: .1em .35em; border-radius: 3px; }
.ok { color: #1a7f37; } .warnv { color: #a15c00; } .bad { color: #b3261e; }
.muted { color: #8b8880; }
ol.steps li { margin: .4rem 0; }
@media (prefers-color-scheme: dark) {
  body { background: #16161a; color: #e8e6e1; }
  h2 { border-color: #33323a; } th { background: #22222a; }
  th, td { border-color: #2b2a32; }
  .sub, .gallery figcaption, .muted { color: #9b988f; }
  .warn { background: #2a2318; border-color: #b8791f; }
  details { background: #1c1c22; border-color: #33323a; }
  details[open] > summary { border-color: #33323a; }
  .gallery img { border-color: #33323a; }
  code { background: #24242c; }
  .ok { color: #4ac26b; } .warnv { color: #d9a441; } .bad { color: #ff6b5e; }
}
"""


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return '<span class="muted">—</span>'
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return escape(str(value))


def _verdict_html(verdict: str | None, agreement: float | None) -> str:
    if verdict is None:
        return '<span class="muted">not evaluated</span>'
    css = "ok" if verdict == "self" else ("bad" if verdict == "incoherent" else "warnv")
    suffix = "" if agreement is None else f" ({agreement * 100:.0f}%)"
    return f'<span class="{css}">{escape(verdict)}{suffix}</span>'


def _slot_table(summary: dict, flip: dict | None) -> str:
    flip_by_slot = {s["slot_index"]: s for s in (flip or {}).get("slots", [])}
    rows = []
    for slot in summary["slots"]:
        f = flip_by_slot.get(slot["slot_index"], {})
        rows.append(
            "<tr>"
            f"<td class='num'>{slot['slot_index']}</td>"
            f"<td class='num'>{slot['frames_above_gate']}/{slot['frames_inspected']}</td>"
            f"<td class='num'>{slot['above_gate_rate'] * 100:.0f}%</td>"
            f"<td class='num'>{_fmt(slot['confidence_median'])}</td>"
            f"<td class='num'>{_fmt(slot['confidence_p95'])}</td>"
            f"<td class='num'>{_fmt(slot['confidence_max'])}</td>"
            f"<td class='num'>{_fmt(slot['spread_px'], 1)}</td>"
            f"<td class='num'>{slot['clamped_count']}</td>"
            f"<td>{_verdict_html(f.get('dominant_verdict'), f.get('agreement'))}</td>"
            "</tr>"
        )
    return (
        "<div class='scroll'><table><thead><tr>"
        "<th>slot</th><th>above gate</th><th>rate</th><th>conf median</th>"
        "<th>conf p95</th><th>conf max</th><th>spread px</th><th>clamped</th>"
        "<th>flip verdict</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def _provenance_table(run: dict) -> str:
    model = run.get("model", {})
    source = run.get("source", {})
    env = run.get("environment", {})
    git = run.get("git", {})
    rows = [
        ("run id", run.get("run_id")),
        ("created (UTC)", run.get("created_utc")),
        ("weights", model.get("path")),
        ("weights sha256", model.get("sha256")),
        ("keypoint slots", model.get("keypoint_count")),
        ("classes", model.get("class_names")),
        ("train imgsz / fliplr", f"{model.get('train_imgsz')} / {model.get('train_fliplr')}"),
        ("train data", model.get("train_data")),
        ("source", source.get("path") or source.get("source_id")),
        ("source sha256", source.get("sha256")),
        ("source size / fps", f"{source.get('width')}x{source.get('height')} @ {source.get('fps')}"),
        ("device", env.get("device")),
        ("libraries", env.get("libraries")),
        ("python", env.get("python")),
        ("git commit", f"{git.get('commit')} (dirty={git.get('dirty')})"),
        ("tool version", run.get("tool_version")),
        ("schema", run.get("schema_version")),
    ]
    body = "".join(
        f"<tr><th>{escape(k)}</th><td>{escape(str(v))}</td></tr>" for k, v in rows
    )
    return f"<div class='scroll'><table><tbody>{body}</tbody></table></div>"


def _sweep_table(per_size: dict[int, dict]) -> str:
    sizes = sorted(per_size)
    if len(sizes) < 2:
        return "<p class='muted'>Single inference size — nothing to compare.</p>"
    header = "".join(f"<th>{s} rate</th><th>{s} median</th>" for s in sizes)
    keypoint_count = max(len(per_size[s]["slots"]) for s in sizes)
    rows = []
    for slot_index in range(keypoint_count):
        cells = []
        for size in sizes:
            slot = per_size[size]["slots"][slot_index]
            cells.append(f"<td class='num'>{slot['above_gate_rate'] * 100:.0f}%</td>")
            cells.append(f"<td class='num'>{_fmt(slot['confidence_median'])}</td>")
        rows.append(f"<tr><td class='num'>{slot_index}</td>{''.join(cells)}</tr>")
    return (
        f"<div class='scroll'><table><thead><tr><th>slot</th>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _gallery(paths: Sequence[str], captions: Sequence[str]) -> str:
    figures = "".join(
        f"<figure><img src='{escape(p)}' alt='{escape(c)}' loading='lazy'>"
        f"<figcaption>{escape(c)}</figcaption></figure>"
        for p, c in zip(paths, captions)
    )
    return f"<div class='gallery'>{figures}</div>"


def _size_section(size: int, block: dict, is_primary: bool) -> str:
    summary = block["summary"]
    flip = block.get("flip_probe")
    strips = "".join(
        f"<h3>slot {i}</h3><div class='strip'>"
        f"<img src='{escape(path)}' alt='slot {i} filmstrip' loading='lazy'></div>"
        for i, path in enumerate(block["filmstrip_paths"])
    )

    flip_gallery = ""
    if block.get("flip_paths"):
        flip_gallery = (
            "<h3>Flip-probe frames</h3>"
            + _gallery(block["flip_paths"], [f"frame {i}" for i in block["frame_indices"]])
        )

    detection_line = (
        f"{summary['frames_detected']}/{summary['frames_inspected']} frames detected · "
        f"{summary['frames_no_detection']} no-detection · "
        f"{summary['frames_multi_instance']} multi-instance · "
        f"gate {summary['confidence_gate']:g}"
    )

    return f"""
<details {"open" if is_primary else ""}>
  <summary>imgsz {size}{" — primary" if is_primary else ""}</summary>
  <div class="body">
    <p class="sub">{escape(detection_line)}</p>
    <h3>Per-slot behaviour</h3>
    {_slot_table(summary, flip)}
    <h3>Per-slot filmstrips</h3>
    <p class="sub">Same slot, every inspected frame, cropped at the prediction.
    A stable slot shows the same physical feature in every tile.</p>
    {strips}
    <h3>Frame overlays</h3>
    {_gallery(block["overlay_paths"], [f"frame {i}" for i in block["frame_indices"]])}
    <p><a href="{escape(block['contact_sheet_path'])}">Full contact sheet</a></p>
    {flip_gallery}
  </div>
</details>
"""


def render_review_html(
    run: dict,
    per_size_blocks: dict[int, dict],
    primary_size: int,
) -> str:
    """Build the single-file review page."""
    sizes = sorted(per_size_blocks)
    sections = "".join(
        _size_section(size, per_size_blocks[size], size == primary_size) for size in sizes
    )
    summaries = {size: per_size_blocks[size]["summary"] for size in sizes}
    source = run.get("source", {})

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Phase 0 slot review — {escape(str(source.get('source_id', '')))}</title>
<style>{_CSS}</style>
</head>
<body>
<main>
<h1>Phase 0 — keypoint slot review</h1>
<p class="sub">{escape(str(source.get('source_id', '')))} ·
{len(run.get('sampling', {}).get('frame_indices', []))} frames ·
{escape(str(run.get('run_id', '')))}</p>

<div class="warn">
<strong>Nothing on this page asserts what a slot means.</strong> These are raw model
predictions and observed regularities. Slot semantics become a fact only when a human
fills in <code>slot_review.md</code> and signs off. Until then no adapter map exists and
no calibration work may depend on a slot's identity.
</div>

<h2>How to review</h2>
<ol class="steps">
  <li>Skim the <strong>frame overlays</strong> to see which frames actually show court.</li>
  <li>Work one slot at a time down the <strong>per-slot filmstrips</strong>. Ask only:
      is this the same physical court feature in every tile?</li>
  <li>Check that slot's row in the <strong>per-slot table</strong> — a low above-gate rate
      or a large spread means the filmstrip agreement may be luck.</li>
  <li>Read the <strong>flip verdict</strong>. <code>self</code> means the slot survives
      mirroring; <code>partner:j</code> means the flip moves it onto slot <em>j</em>'s
      position; <code>incoherent</code> means it goes nowhere sensible.</li>
  <li>Write the conclusion into <code>slot_review.md</code>. If two readings fit,
      record the ambiguity — an honest "ambiguous" is worth more than a guess, because
      a wrong map fails silently downstream.</li>
</ol>

<h2>Run summary</h2>
<div class="scroll"><table><thead><tr>
<th>imgsz</th><th>frames</th><th>detected</th><th>no detection</th><th>multi-instance</th>
</tr></thead><tbody>
{"".join(
    f"<tr><td class='num'>{size}</td>"
    f"<td class='num'>{s['frames_inspected']}</td>"
    f"<td class='num'>{s['frames_detected']}</td>"
    f"<td class='num'>{s['frames_no_detection']}</td>"
    f"<td class='num'>{s['frames_multi_instance']}</td></tr>"
    for size, s in summaries.items()
)}
</tbody></table></div>

<h2>Inference size sweep</h2>
{_sweep_table(summaries)}

<h2>Evidence</h2>
{sections}

<h2>Provenance</h2>
{_provenance_table(run)}
</main>
</body>
</html>
"""


def render_slot_review_markdown(
    run: dict,
    keypoint_count: int,
    summary: dict,
    flip: dict | None,
    candidate_names: Sequence[str] = (),
) -> str:
    """The blank worksheet a reviewer fills in."""
    model = run.get("model", {})
    flip_by_slot = {s["slot_index"]: s for s in (flip or {}).get("slots", [])}
    summary_by_slot = {s["slot_index"]: s for s in summary["slots"]}

    rows = []
    for i in range(keypoint_count):
        s = summary_by_slot.get(i, {})
        f = flip_by_slot.get(i, {})
        rate = s.get("above_gate_rate")
        verdict = f.get("dominant_verdict") or "—"
        rows.append(
            f"| {i} | {'' if rate is None else f'{rate * 100:.0f}%'} | {verdict} "
            f"|  |  |  |  |"
        )

    vocabulary = ""
    if candidate_names:
        listed = "\n".join(f"- `{name}`" for name in candidate_names)
        vocabulary = f"""
## Appendix — candidate landmark names

Supplied via `--candidate-vocabulary` purely as a naming reference, so reviewers write
the same string for the same feature. **Inclusion here is not a claim that this model
predicts these landmarks, or that the counts match.**

{listed}
"""

    return f"""# Phase 0 slot review worksheet — UNVERIFIED

**Status:** unverified. Nothing in this file is a fact until a human fills it in and
signs off at the bottom.

| | |
| --- | --- |
| run id | `{run.get('run_id')}` |
| weights | `{model.get('path')}` |
| weights sha256 | `{model.get('sha256')}` |
| slots | {keypoint_count} |
| source | `{run.get('source', {}).get('source_id')}` |
| frames inspected | {summary['frames_inspected']} |
| generated (UTC) | {run.get('created_utc')} |

Open `review.html` beside this file and work slot by slot through the filmstrips.

## Verdict vocabulary

- **confident** — the same physical feature in essentially every above-gate frame.
- **probable** — consistent, but on too few frames or with visible drift.
- **ambiguous** — two or more readings fit; say which, in the notes.
- **unusable** — rarely above gate, or wanders with no stable referent.

Record `ambiguous` rather than guessing. A wrong slot assignment does not fail loudly
later — it produces a plausible, silently wrong calibration.

## Slots

| slot | above-gate rate | flip verdict | proposed landmark | verdict | evidence frames | ambiguity notes |
| ---: | ---: | --- | --- | --- | --- | --- |
{chr(10).join(rows)}

## Sign-off

Phase 0 exits only when **every** slot above is filled in, the flip behaviour is coherent
and explained, and the reviewer signs here. Anything less means calibration work does not
start — that is the point of the gate.

- Reviewer:
- Date (UTC):
- Outcome: ☐ all slots assigned ☐ partial ☐ escalate (flip incoherent / retrain needed)
- Notes:
{vocabulary}"""


def render_slot_review_json(
    run: dict,
    keypoint_count: int,
    flip: dict | None,
) -> dict:
    """Machine-readable worksheet. Every assignment null; not an adapter map."""
    flip_by_slot = {s["slot_index"]: s for s in (flip or {}).get("slots", [])}
    model = run.get("model", {})
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "status": "unverified",
        "is_adapter_map": False,
        "note": (
            "Reviewer worksheet template. Every landmark_id is null by design. This file "
            "must not be consumed as an index-to-landmark map; a map may only be written "
            "after human sign-off."
        ),
        "run_id": run.get("run_id"),
        "weights_path": model.get("path"),
        "weights_sha256": model.get("sha256"),
        "keypoint_count": keypoint_count,
        "reviewer": None,
        "signed_off_utc": None,
        "slots": [
            {
                "slot_index": i,
                "landmark_id": None,
                "verdict": None,
                "ambiguity": None,
                "evidence_frames": [],
                "observed_flip_verdict": flip_by_slot.get(i, {}).get("dominant_verdict"),
                "notes": "",
            }
            for i in range(keypoint_count)
        ],
    }


def load_candidate_names(path: Path | str) -> list[str]:
    """Read reference landmark names from a JSON file, as pure data.

    Accepts either a bare list of names or an object with a ``keypoints`` array of
    ``{"id": ...}`` entries. No schema is imported and no code is shared — the
    file is read the same way a CSV would be.
    """
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return [str(item) for item in data]
    entries = data.get("keypoints", [])
    return [str(entry["id"]) for entry in entries if isinstance(entry, dict) and "id" in entry]
