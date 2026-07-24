// Keypoint labeling editor: shows one kept frame at a time on a canvas where
// the labeler places, drags, and states canonical court keypoints by hand.
// Optional model prefill only appears when the server was started with a
// pose model trained on this same schema.

const pathParts = location.pathname.split('/').filter(Boolean);
const clipId = pathParts.length > 1 ? decodeURIComponent(pathParts[1]) : null;
document.getElementById('clip-name').textContent = clipId || 'all kept frames';

// The authoritative versioned mapping is fetched on startup. Every index is a
// fixed north/south/east/west court feature, independent of the camera view.
let keypointSchema = null;
let keypointFeatures = [];

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const wrap = document.getElementById('canvas-wrap');

let frames = [];        // kept frames
let index = 0;          // current frame index
let image = new Image();
let points = [];        // [{x, y, v, src_conf}] in image pixel coords
let selectedPoint = -1;
let dirty = false;
let labelSource = '';
let referenceSvgDocument = null;
let orientationByClip = {};
let orientationPending = false;
let predictAvailable = false;

// Per-frame "visible ends" declaration: prefilled from the last confirmed
// answer for the clip, but the labeler must confirm it on every frame.
let visibleEnds = null;
let endsConfirmed = false;
let loadedVisibleEnds = null;   // value stored in the loaded saved label
const stickyEndsByClip = {};

const END_GROUP_TITLES = { north: 'North end', mid: 'Midcourt', south: 'South end' };

function updateDiagramHighlight() {
  if (!referenceSvgDocument) return;
  referenceSvgDocument.querySelectorAll('.keypoint.active').forEach((node) => {
    node.classList.remove('active');
  });
  const active = placing >= 0 ? placing : selectedPoint;
  if (active >= 0) {
    referenceSvgDocument.getElementById(`kpt-${active + 1}`)?.classList.add('active');
  }
}

// view transform: screen = image * scale + offset
let scale = 1, offsetX = 0, offsetY = 0;
let dragging = null;    // {type: 'point'|'pan', ...}
let spaceDown = false;
let placing = -1;       // index of the unplaced point being spawned, or -1

// Fixed per-keypoint colors make a landmark easy to follow between frames.
// The surrounding interface stays neutral; color is reserved for annotations.
const KEYPOINT_COLORS = [
  '#ef4444', '#f97316', '#eab308', '#84cc16', '#22c55e', '#14b8a6',
  '#06b6d4', '#0ea5e9', '#3b82f6', '#6366f1', '#8b5cf6', '#a855f7',
  '#d946ef', '#ec4899', '#f43f5e', '#fb7185', '#f59e0b', '#10b981',
  '#a3e635', '#2dd4bf', '#c084fc', '#fdba74',
];
function keypointColor(index) {
  return KEYPOINT_COLORS[index % KEYPOINT_COLORS.length];
}

function setStatus(text, isDirty) {
  const el = document.getElementById('status');
  el.textContent = text;
  el.classList.toggle('dirty', !!isDirty);
}

function markDirty() {
  dirty = true;
  setStatus('unsaved changes', true);
}

function updateRemoveButton() {
  const button = document.getElementById('remove-point');
  const point = points[selectedPoint];
  button.disabled = !point || point.unplaced;
}

function fitView() {
  if (!image.width) return;
  scale = Math.min(wrap.clientWidth / image.width, wrap.clientHeight / image.height);
  offsetX = (wrap.clientWidth - image.width * scale) / 2;
  offsetY = (wrap.clientHeight - image.height * scale) / 2;
}

function toScreen(x, y) { return [x * scale + offsetX, y * scale + offsetY]; }
function toImage(sx, sy) { return [(sx - offsetX) / scale, (sy - offsetY) / scale]; }

function draw() {
  canvas.width = wrap.clientWidth;
  canvas.height = wrap.clientHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!image.width) return;
  ctx.save();
  ctx.translate(offsetX, offsetY);
  ctx.scale(scale, scale);
  ctx.drawImage(image, 0, 0);
  ctx.restore();

  points.forEach((point, i) => {
    if (point.unplaced && i !== placing) return;
    const [sx, sy] = toScreen(point.x, point.y);
    const color = i === placing ? '#fafafa' : keypointColor(i);
    ctx.save();
    ctx.globalAlpha = point.v === 1 ? 0.45 : 1.0;
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(sx, sy, 6, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(sx, sy, 8, 0, Math.PI * 2); ctx.stroke();
    if (i === selectedPoint) {
      ctx.globalAlpha = 1;
      ctx.strokeStyle = '#7fb2ff';
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(sx, sy, 11, 0, Math.PI * 2); ctx.stroke();
    }
    ctx.globalAlpha = 1;
    ctx.font = '11px sans-serif';
    ctx.fillStyle = color;
    const conf = point.src_conf > 0 ? ` ${point.src_conf.toFixed(2)}` : '';
    ctx.fillText(`K${i + 1}${conf}`, sx + 10, sy - 8);
    ctx.restore();
  });
  renderPointList();
  updateRemoveButton();
  updateDiagramHighlight();
}

// A point is locked out when the labeler declared its end not visible.
// Midcourt points are never locked; an unset declaration locks nothing.
function pointDisabled(i) {
  const keypoint = keypointFeatures[i];
  if (!keypoint || !visibleEnds || visibleEnds === 'both') return false;
  return (keypoint.end === 'north' || keypoint.end === 'south') && keypoint.end !== visibleEnds;
}

function renderPointList() {
  const list = document.getElementById('point-list');
  list.innerHTML = '';
  let lastEnd = null;
  points.forEach((point, i) => {
    const keypoint = keypointFeatures[i];
    if (keypoint && keypoint.end !== lastEnd) {
      lastEnd = keypoint.end;
      const header = document.createElement('li');
      header.className = 'group-header';
      header.textContent = END_GROUP_TITLES[keypoint.end] || keypoint.end;
      list.appendChild(header);
    }
    const disabled = pointDisabled(i);
    const li = document.createElement('li');
    li.className = [
      i === selectedPoint ? 'selected' : '',
      point.unplaced ? 'unplaced' : '',
      disabled ? 'disabled' : '',
    ].join(' ').trim();
    const info = point.unplaced
      ? (i === placing ? 'placing…' : disabled ? 'end not visible' : 'click to place')
      : point.v === 1 ? 'occluded' : 'visible';
    const dotColor = point.unplaced ? '#555' : keypointColor(i);
    const desc = keypoint
      ? `${keypoint.feature} · ${keypoint.court_position}`
      : 'schema unavailable';
    li.innerHTML = `<span class="dot" style="background:${dotColor}"></span>` +
      `<span class="pt">K${i + 1}<span class="v">${info}</span>` +
      `<span class="desc">${desc}</span></span>`;
    li.addEventListener('click', () => {
      if (pointDisabled(i)) {
        setStatus(`${keypoint.id} is on the ${keypoint.end} end, which you marked not visible`, true);
        return;
      }
      selectedPoint = i;
      if (point.unplaced) startPlacing(i);
      else { cancelPlacing(); draw(); }
    });
    list.appendChild(li);
  });
}

function startPlacing(i) {
  if (pointDisabled(i)) return;
  placing = i;
  selectedPoint = i;
  // Spawn in the center of the view; it follows the mouse until dropped.
  const point = points[i];
  const [ix, iy] = toImage(wrap.clientWidth / 2, wrap.clientHeight / 2);
  point.x = Math.max(0, Math.min(image.width, ix));
  point.y = Math.max(0, Math.min(image.height, iy));
  setStatus(`placing K${i + 1} — click on the image to drop, Esc cancels`, true);
  draw();
}

function cancelPlacing() {
  if (placing < 0) return;
  const point = points[placing];
  point.x = 0;
  point.y = 0;
  placing = -1;
  setStatus(dirty ? 'unsaved changes' : 'ready', dirty);
  draw();
}

function renderProgress() {
  const labeled = frames.filter(f => f.label_status === 'labeled').length;
  const trainedSplit = frames[index] && frames[index].trained_split;
  document.getElementById('progress').innerHTML =
    frames.length
      ? `frame <b>${index + 1}/${frames.length}</b> · ` +
        `<b class="labeled">${labeled}</b> labeled · source: ${labelSource || '—'}` +
        (trainedSplit ? ` · <b class="trained">in ${trainedSplit} split</b>` : '')
      : 'no kept frames — mark some in triage first';
}

function hitPoint(sx, sy) {
  for (let i = points.length - 1; i >= 0; i--) {
    if (points[i].unplaced) continue;
    const [px, py] = toScreen(points[i].x, points[i].y);
    if (Math.hypot(px - sx, py - sy) <= 12) return i;
  }
  return -1;
}

/* ---- clip orientation anchor ---- */

function setOrientationPanel(visible) {
  orientationPending = visible;
  document.getElementById('orientation-panel').hidden = !visible;
}

function showOrientationStatus(orientation) {
  const status = document.getElementById('orientation-status');
  if (!orientation) { status.textContent = ''; return; }
  const declared = orientation.declared_end
    ? ` (anchor frame declared as the ${orientation.declared_end} end)`
    : '';
  status.textContent =
    `Orientation locked at ${orientation.anchor_frame_id}: north = image-left basket${declared}.`;
}

async function fetchOrientation(clip) {
  if (Object.prototype.hasOwnProperty.call(orientationByClip, clip)) {
    return orientationByClip[clip];
  }
  const response = await fetch(`/api/clip/${encodeURIComponent(clip)}/orientation`);
  const data = await response.json();
  orientationByClip[clip] = data.orientation;
  return data.orientation;
}

async function ensureOrientation(frame) {
  const orientation = await fetchOrientation(frame.clip_id);
  showOrientationStatus(orientation);
  if (orientation) {
    setOrientationPanel(false);
    return true;
  }
  setOrientationPanel(true);
  return false;
}

async function setOrientation(mode, declaredEnd) {
  if (!frames.length) return;
  const frame = frames[index];
  const body = { frame_id: frame.frame_id, mode };
  if (declaredEnd) body.declared_end = declaredEnd;
  const response = await fetch(`/api/clip/${encodeURIComponent(frame.clip_id)}/orientation`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    setStatus('could not lock the orientation — try another frame', true);
    return;
  }
  const data = await response.json();
  orientationByClip[frame.clip_id] = data.orientation;
  showOrientationStatus(data.orientation);
  setOrientationPanel(false);
  await loadFrame(false);
}

/* ---- per-frame visible-ends confirmation ---- */

function renderEndsPanel() {
  const panel = document.getElementById('ends-panel');
  panel.classList.toggle('unconfirmed', !endsConfirmed);
  document.getElementById('ends-state').textContent =
    endsConfirmed ? 'confirmed' : 'confirm';
  panel.querySelectorAll('[data-ends]').forEach((button) => {
    button.classList.toggle('selected', button.dataset.ends === visibleEnds);
  });
}

function setEndsPanel(visible) {
  document.getElementById('ends-panel').hidden = !visible;
  if (visible) renderEndsPanel();
}

function flashEndsPanel() {
  const panel = document.getElementById('ends-panel');
  panel.classList.remove('flash');
  void panel.offsetWidth; // restart the animation
  panel.classList.add('flash');
}

function chooseEnds(value) {
  if (orientationPending || !frames.length) return;
  const frame = frames[index];
  const changedSaved = labelSource === 'saved' && loadedVisibleEnds !== null && value !== loadedVisibleEnds;
  visibleEnds = value;
  endsConfirmed = true;
  stickyEndsByClip[frame.clip_id] = value;
  if (changedSaved) markDirty();
  if (placing >= 0 && pointDisabled(placing)) cancelPlacing();
  renderEndsPanel();
  draw();
  if (!dirty) setStatus(`visible ends confirmed: ${value}`);
}

/* ---- frame loading and saving ---- */

function loadFrameImage(frame) {
  return new Promise((resolve) => {
    image = new Image();
    image.onload = () => { fitView(); draw(); resolve(); };
    image.src = '/images/' + frame.image.replace(/^frames\//, '');
  });
}

function updateRepredictButton() {
  document.getElementById('repredict').hidden = !predictAvailable;
}

async function loadFrame(predict) {
  if (!frames.length) { renderProgress(); draw(); return; }
  const frame = frames[index];
  setStatus('loading…');
  selectedPoint = -1;
  placing = -1;
  dirty = false;
  points = [];
  labelSource = '';
  await loadFrameImage(frame);
  renderProgress();
  if (!(await ensureOrientation(frame))) {
    setEndsPanel(false);
    setStatus('lock the clip orientation to start labeling');
    return;
  }

  const url = `/api/label/${frame.frame_id}` + (predict ? '?predict=1' : '');
  const response = await fetch(url);
  const data = await response.json();
  if (!response.ok) {
    setStatus(data.error || 'could not load label', true);
    return;
  }
  predictAvailable = !!data.predict_available;
  updateRepredictButton();
  points = data.label.keypoints;
  points.forEach((point) => {
    point.unplaced = point.x === 0 && point.y === 0 && point.v === 0;
  });
  labelSource = data.source;
  if (labelSource === 'saved' && data.label.visible_ends) {
    visibleEnds = data.label.visible_ends;
    loadedVisibleEnds = visibleEnds;
    endsConfirmed = true;
    stickyEndsByClip[frame.clip_id] = visibleEnds;
  } else {
    loadedVisibleEnds = null;
    visibleEnds = stickyEndsByClip[frame.clip_id] || null;
    endsConfirmed = false;
  }
  setEndsPanel(true);
  showOrientationStatus(data.orientation);
  renderProgress();
  draw();
  if (labelSource === 'saved') {
    setStatus('saved label loaded');
  } else if (labelSource === 'predicted') {
    setStatus('model prefill — verify every point, then confirm visible ends');
  } else if (labelSource === 'previous_frame') {
    setStatus('copied from previous frame — fix/move points, then confirm visible ends');
  } else {
    setStatus('confirm visible ends (1/2/3), then click keypoints in the list to place them');
  }
}

async function save() {
  if (!frames.length || orientationPending) return;
  if (!endsConfirmed || !visibleEnds) {
    flashEndsPanel();
    setStatus('confirm visible ends first — 1 north · 2 both · 3 south', true);
    return;
  }
  const frame = frames[index];
  const res = await fetch(`/api/label/${frame.frame_id}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      visible_ends: visibleEnds,
      keypoints: points.map(({ x, y, v, src_conf }) => ({ x, y, v, src_conf })),
    }),
  });
  if (res.ok) {
    dirty = false;
    labelSource = 'saved';
    loadedVisibleEnds = visibleEnds;
    frame.label_status = 'labeled';
    setStatus('saved ✓');
    renderProgress();
    return;
  }
  let message = 'save failed';
  try {
    const data = await res.json();
    if (data.conflicting_ids) {
      message = `visible ends is "${visibleEnds}" but these points are placed: ` +
        data.conflicting_ids.join(', ');
    } else if (data.error) {
      message = data.error;
    }
  } catch { /* non-JSON error page */ }
  setStatus(message, true);
}

async function move(delta) {
  if (!frames.length) return;
  if (dirty) {
    // Autosave on navigation, but never silently drop an unsavable frame.
    await save();
    if (dirty) return;
  }
  index = Math.min(frames.length - 1, Math.max(0, index + delta));
  loadFrame(false);
}

/* ---- canvas interaction ---- */

canvas.addEventListener('mousedown', (event) => {
  const sx = event.offsetX, sy = event.offsetY;
  if (spaceDown || event.button === 1) {
    dragging = { type: 'pan', startX: sx, startY: sy, ox: offsetX, oy: offsetY };
    return;
  }
  if (event.button !== 0) return;
  if (placing >= 0) {
    const [ix, iy] = toImage(sx, sy);
    const point = points[placing];
    point.x = Math.max(0, Math.min(image.width, ix));
    point.y = Math.max(0, Math.min(image.height, iy));
    point.unplaced = false;
    point.v = 2;
    selectedPoint = placing;
    placing = -1;
    markDirty();
    draw();
    return;
  }
  const hit = hitPoint(sx, sy);
  if (hit >= 0) {
    selectedPoint = hit;
    dragging = { type: 'point', index: hit };
    draw();
  }
});

canvas.addEventListener('mousemove', (event) => {
  if (!dragging) {
    if (placing >= 0) {
      const [ix, iy] = toImage(event.offsetX, event.offsetY);
      const point = points[placing];
      point.x = Math.max(0, Math.min(image.width, ix));
      point.y = Math.max(0, Math.min(image.height, iy));
      draw();
    }
    return;
  }
  if (dragging.type === 'pan') {
    offsetX = dragging.ox + (event.offsetX - dragging.startX);
    offsetY = dragging.oy + (event.offsetY - dragging.startY);
  } else {
    const [ix, iy] = toImage(event.offsetX, event.offsetY);
    const point = points[dragging.index];
    point.x = Math.max(0, Math.min(image.width, ix));
    point.y = Math.max(0, Math.min(image.height, iy));
    markDirty();
  }
  draw();
});

window.addEventListener('mouseup', () => { dragging = null; });

canvas.addEventListener('wheel', (event) => {
  event.preventDefault();
  const factor = event.deltaY < 0 ? 1.15 : 1 / 1.15;
  const [ix, iy] = toImage(event.offsetX, event.offsetY);
  scale = Math.max(0.05, Math.min(20, scale * factor));
  offsetX = event.offsetX - ix * scale;
  offsetY = event.offsetY - iy * scale;
  draw();
}, { passive: false });

canvas.addEventListener('contextmenu', (event) => {
  event.preventDefault();
  const hit = hitPoint(event.offsetX, event.offsetY);
  if (hit >= 0) {
    selectedPoint = hit;
    cycleVisibility();
  }
});

function cycleVisibility() {
  if (selectedPoint < 0 || points[selectedPoint].unplaced) return;
  const point = points[selectedPoint];
  point.v = point.v === 2 ? 1 : 2;
  markDirty();
  draw();
}

function removeSelectedPoint() {
  if (selectedPoint < 0 || !points[selectedPoint] || points[selectedPoint].unplaced) {
    return;
  }
  cancelPlacing();
  const point = points[selectedPoint];
  point.x = 0;
  point.y = 0;
  point.v = 0;
  point.src_conf = 0;
  point.unplaced = true;
  markDirty();
  setStatus(`K${selectedPoint + 1} unlabeled — click it in the list to place it again`, true);
  draw();
}

document.addEventListener('keydown', (event) => {
  if (event.code === 'Space') { spaceDown = true; event.preventDefault(); return; }
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 's') {
    event.preventDefault(); save(); return;
  }
  const key = event.key.toLowerCase();
  if (key === 'arrowright') move(1);
  else if (key === 'arrowleft') move(-1);
  else if (key === 'v') cycleVisibility();
  else if (key === '1') chooseEnds('north');
  else if (key === '2') chooseEnds('both');
  else if (key === '3') chooseEnds('south');
  else if (key === 'backspace' || key === 'delete') {
    event.preventDefault();
    removeSelectedPoint();
  }
  else if (key === 'r' && predictAvailable) loadFrame(true);
  else if (key === 'escape') cancelPlacing();
});
document.addEventListener('keyup', (event) => {
  if (event.code === 'Space') spaceDown = false;
});

document.getElementById('save').addEventListener('click', save);
document.getElementById('anchor-both').addEventListener('click', () => setOrientation('both_ends_visible'));
document.getElementById('anchor-declare-north').addEventListener('click', () => setOrientation('declared', 'north'));
document.getElementById('anchor-declare-south').addEventListener('click', () => setOrientation('declared', 'south'));
document.getElementById('repredict').addEventListener('click', () => loadFrame(true));
document.getElementById('remove-point').addEventListener('click', removeSelectedPoint);
document.getElementById('prev').addEventListener('click', () => move(-1));
document.getElementById('next').addEventListener('click', () => move(1));
document.querySelectorAll('#ends-panel [data-ends]').forEach((button) => {
  button.addEventListener('click', () => chooseEnds(button.dataset.ends));
});
window.addEventListener('resize', () => { fitView(); draw(); });

async function init() {
  const schemaResponse = await fetch('/api/keypoint-schema');
  if (!schemaResponse.ok) throw new Error('Could not load the keypoint schema');
  keypointSchema = await schemaResponse.json();
  keypointFeatures = keypointSchema.keypoints || [];
  document.getElementById('keypoints-count').textContent = `K1–K${keypointFeatures.length}`;

  const reference = document.getElementById('court-reference');
  const connectReference = () => {
    referenceSvgDocument = reference.contentDocument;
    updateDiagramHighlight();
  };
  reference.addEventListener('load', connectReference);
  reference.data = keypointSchema.reference_diagram.asset_url;

  const params = new URLSearchParams({ triage: 'keep' });
  if (clipId) params.set('clip', clipId);
  const data = await (await fetch('/api/frames?' + params)).json();
  frames = data.frames;
  index = 0;
  loadFrame(false);
}

init();
