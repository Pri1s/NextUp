#!/usr/bin/env python3
"""Local, review-only UI for JSON produced by run_homography_labels.py.

Edits are reviewer overrides stored in a separate reviews directory. This server
never writes training-format labels or modifies the source frame/results files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review classical homography labels in a local browser.")
    parser.add_argument("results", type=Path, help="Folder containing batch_summary.json")
    parser.add_argument("--reviews", type=Path, default=None, help="Reviewer override folder (default: RESULTS/reviews)")
    parser.add_argument("--port", type=int, default=8010)
    return parser.parse_args()


HTML = """<!doctype html><meta charset=utf-8><title>Homography review</title>
<style>
body{margin:0;background:#101114;color:#eee;font:14px system-ui;display:grid;grid-template-columns:1fr 340px;height:100vh}
header{grid-column:1/-1;padding:10px 16px;border-bottom:1px solid #333}
canvas{max-width:100%;max-height:calc(100vh - 52px);margin:auto;display:block}
.stage{overflow:auto;display:grid;place-items:center;background:#050505}
.side{padding:14px;overflow:auto;border-left:1px solid #333}
.fail{color:#ff7777}.pass{color:#7ee787}.review{color:#f5b942}
button{margin:2px;padding:5px;background:#24262b;color:#eee;border:1px solid #555;border-radius:4px;cursor:pointer}
.item{padding:7px;border-bottom:1px solid #292929}
.item small{color:#aaa}
select{background:#222;color:#fff}
#frame{width:100%}
#approve{background:#173d24;border-color:#2f7a4c}
#reject{background:#3d1717;border-color:#7a2f2f}
#addpoint.active{outline:2px solid #f5b942}
.candidate-banner{border:1px solid #f5b94255;background:#f5b94214;padding:8px;border-radius:4px;margin:6px 0}
.actions{display:none}
.actions.visible{display:block}
</style>
<header><b>Classical homography review</b> · <span id=summary></span> <select id=frame></select></header>
<div class=stage><canvas id=canvas></canvas></div>
<aside class=side>
  <div id=status></div>
  <div class="actions" id=candidate-actions>
    <p>This is a <b>review candidate</b>, not an automatic label. Drag a point to correct it, or mark it excluded/added below, then record your decision.</p>
    <button id=approve>Approve candidate</button>
    <button id=reject>Reject candidate</button>
    <button id=addpoint>Add point</button>
  </div>
  <p>Drag a projected point to record a reviewer correction. Reviewer decisions are saved separately under <code>reviews/</code> and never rewrite the pipeline result or batch summary.</p>
  <button id=save>Save review</button>
  <button id=toggle>toggle detected lines</button>
  <div id=points></div>
</aside>
<script>
let batch, result, review={points:{}}, image=new Image(), showLines=true, drag=null, addMode=false;
const c=document.querySelector('canvas'),x=c.getContext('2d');
const q=s=>document.querySelector(s);
function esc(s){return String(s).replace(/[&<>]/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[m]))}

// A rejected frame with a review candidate draws that candidate's projected
// points instead of the (empty) automatic solution.reprojection. An
// automatically-passed frame always shows its own solution points.
function hasReviewCandidate(){return !!(result && result.solution && result.solution.review_candidate)}
function usingCandidate(){return result.verification.status!=='pass' && hasReviewCandidate()}
function sourcePoints(){return usingCandidate() ? (result.solution.review_candidate.reprojection||[]) : (result.solution.reprojection||[])}
function addedPoints(){
  return Object.entries(review.points)
    .filter(([, v]) => v.status === 'added')
    .map(([id, v]) => ({id, pixel_xy: v.pixel_xy, error_px: null, in_image_bounds: true}));
}
function activePoints(){return [...sourcePoints(), ...addedPoints()]}
function point(p){return review.points[p.id]?.pixel_xy||p.pixel_xy}
function pointStatus(p){return review.points[p.id]?.status || 'unchanged'}

async function boot(){
  batch=await (await fetch('/api/batch')).json();
  q('#summary').textContent=`${batch.totals.passed}/${batch.totals.frames} passed (${(batch.totals.pass_rate*100).toFixed(1)}%)`;
  batch.frames.forEach((f,i)=>q('#frame').insertAdjacentHTML('beforeend',
    `<option value="${i}">${esc(f.frame)} — ${f.gate_status}${f.review_candidate_available?' (review)':''}</option>`));
  q('#frame').onchange=load;
  await load();
}
async function load(){
  let f=batch.frames[q('#frame').value];
  result=await (await fetch('/api/result?path='+encodeURIComponent(f.result))).json();
  review=await (await fetch('/api/review?path='+encodeURIComponent(f.result))).json();
  review.points??={};
  addMode=false;
  image=new Image();image.onload=draw;image.src='/api/image?path='+encodeURIComponent(f.result);
  render();
}
function draw(){
  c.width=image.width;c.height=image.height;x.drawImage(image,0,0);
  if(showLines){
    result.detection.primitives.filter(a=>a.type==='line_segment').forEach(a=>{
      x.strokeStyle='rgba(80,180,255,.35)';x.beginPath();x.moveTo(...a.geometry.p1);x.lineTo(...a.geometry.p2);x.stroke();
    });
  }
  const candidateMode=usingCandidate();
  for(const p of activePoints()){
    let [px,py]=point(p), status=pointStatus(p);
    if(candidateMode){
      // Dashed amber ring: visually distinguishes a review candidate's points
      // from an automatically-accepted solution's solid points.
      x.beginPath();x.setLineDash([3,3]);x.lineWidth=2;x.strokeStyle='#f5b942';x.arc(px,py,8,0,7);x.stroke();x.setLineDash([]);
    }
    x.fillStyle = status==='excluded' ? '#666'
      : status==='moved' ? '#6cf'
      : status==='added' ? '#c6f'
      : candidateMode ? '#f5b942'
      : (p.error_px==null ? '#f5b942' : '#fff');
    x.beginPath();x.arc(px,py,6,0,7);x.fill();
    x.fillStyle='#000';x.fillText((p.id||'').replace(/.*_/,'K '),px+8,py-7);
  }
}
function render(){
  let v=result.verification, cls=v.status==='pass'?'pass':'fail';
  let candidate=result.solution.review_candidate;
  let reviewRequired = v.status!=='pass' && candidate;
  let banner = reviewRequired ? `<h3 class=review>REVIEW REQUIRED</h3>` : '';
  let candidateInfo = candidate ? `<div class=candidate-banner><b>Review candidate available</b> (not an automatic label)<br>
    matched keypoints: ${candidate.matched_keypoints} · mean error: ${candidate.mean_matched_error_px}px<br>
    why automatic verification failed: ${esc((candidate.automatic_gate_failures||[]).join(', ') || (v.reasons||[]).join(', '))}</div>` : '';
  q('#status').innerHTML = `${banner}<h3 class=${cls}>Gate: ${v.status.toUpperCase()}</h3>`
    + `<div>${esc((v.reasons||[]).join(', ')||'all checks passed')}</div>`
    + candidateInfo
    + `<pre>${esc(JSON.stringify(v.metrics||{},null,2))}</pre>`;
  q('#candidate-actions').classList.toggle('visible', !!reviewRequired);
  q('#addpoint').classList.toggle('active', addMode);

  q('#points').innerHTML = activePoints().map(p=>{
    let e=p.error_px==null?'no detected intersection':p.error_px+' px';
    let status=pointStatus(p);
    return `<div class=item><b>${esc(p.id)}</b><br><small>error: ${e} · ${p.in_image_bounds?'in bounds':'outside'}</small><br>
      <select data-id="${p.id}">
        <option value=unchanged ${status==='unchanged'?'selected':''}>unchanged</option>
        <option value=moved ${status==='moved'?'selected':''}>moved</option>
        <option value=excluded ${status==='excluded'?'selected':''}>excluded</option>
      </select></div>`;
  }).join('');
  q('#points').querySelectorAll('select').forEach(s=>s.onchange=()=>{
    review.points[s.dataset.id]??={};
    review.points[s.dataset.id].status=s.value;
    draw();
  });
}
c.onmousedown=e=>{
  let r=c.getBoundingClientRect(), px=(e.clientX-r.left)*c.width/r.width, py=(e.clientY-r.top)*c.height/r.height;
  if(addMode){
    let id='manual_'+Date.now();
    review.points[id]={status:'added', pixel_xy:[Math.round(px),Math.round(py)]};
    addMode=false;
    render();draw();
    return;
  }
  let best=null;
  for(let p of activePoints()){let a=point(p),d=Math.hypot(a[0]-px,a[1]-py);if(d<12&&(!best||d<best.d))best={p,d}}
  drag=best?.p;
};
window.onmouseup=()=>drag=null;
c.onmousemove=e=>{
  if(!drag)return;
  let r=c.getBoundingClientRect();
  review.points[drag.id]??={};
  review.points[drag.id].pixel_xy=[Math.round((e.clientX-r.left)*c.width/r.width),Math.round((e.clientY-r.top)*c.height/r.height)];
  if(review.points[drag.id].status!=='added')review.points[drag.id].status='moved';
  draw();
};
async function saveReview(body){
  let f=batch.frames[q('#frame').value];
  await fetch('/api/review?path='+encodeURIComponent(f.result),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
}
q('#save').onclick=async()=>{await saveReview(review);alert('review saved')};
q('#toggle').onclick=()=>{showLines=!showLines;draw()};
q('#addpoint').onclick=()=>{addMode=!addMode;render()};
async function decide(decision){
  let f=batch.frames[q('#frame').value];
  let candidate=result.solution.review_candidate;
  review.source_result_path=f.result;
  review.automatic_status=result.solution.status;
  review.automatic_reason=result.solution.reason||null;
  review.review_candidate={
    proposal_type: candidate.proposal.type,
    proposal_source: candidate.proposal.source,
    homography: candidate.homography,
  };
  review.decision=decision;
  review.decided_at=new Date().toISOString();
  await saveReview(review);
  alert('Decision saved: '+decision);
  render();
}
q('#approve').onclick=()=>decide('approved');
q('#reject').onclick=()=>decide('rejected');
boot();
</script>"""


def build_app(root: Path, reviews: Path) -> Flask:
    """Construct the review Flask app without binding a port; used by main() and tests."""
    summary_path = root / "batch_summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"No batch_summary.json in {root}; run run_homography_labels.py first.")
    app = Flask(__name__)

    def result_path(relative: str) -> Path:
        candidate = (root / relative).resolve()
        if root not in candidate.parents or candidate.suffix != ".json":
            abort(400, "Invalid result path")
        return candidate

    @app.get("/")
    def index():
        return HTML

    @app.get("/api/batch")
    def batch():
        return jsonify(json.loads(summary_path.read_text(encoding="utf-8")))

    @app.get("/api/result")
    def result():
        path = result_path(request.args["path"])
        return jsonify(json.loads(path.read_text(encoding="utf-8")))

    @app.get("/api/image")
    def image():
        path = result_path(request.args["path"])
        frame = Path(json.loads(path.read_text(encoding="utf-8"))["frame"])
        if not frame.is_file():
            abort(404, f"Frame is unavailable: {frame}")
        return send_file(frame)

    def review_path(relative: str) -> Path:
        return reviews / Path(relative).with_suffix(".review.json")

    @app.get("/api/review")
    def get_review():
        path = review_path(request.args["path"])
        return jsonify(json.loads(path.read_text()) if path.is_file() else {"points": {}})

    @app.post("/api/review")
    def save_review():
        relative = request.args["path"]
        result_path(relative)  # validate against traversal before writing
        path = review_path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(request.get_json(force=True), indent=2) + "\n")
        return jsonify({"saved": True})

    return app


def main() -> None:
    args = parse_args()
    root = args.results.resolve()
    reviews = (args.reviews or root / "reviews").resolve()
    app = build_app(root, reviews)
    print(f"Open http://127.0.0.1:{args.port}")
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
