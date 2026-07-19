// Keypoint correction editor: shows one kept frame at a time on a canvas with
// the model's predicted (or previously saved) points, editable by dragging.

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
    if (point.v === 0 && i !== placing) {
      // excluded: hollow gray ghost with an X
      ctx.strokeStyle = '#888';
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(sx, sy, 6, 0, Math.PI * 2); ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(sx - 5, sy - 5); ctx.lineTo(sx + 5, sy + 5);
      ctx.moveTo(sx - 5, sy + 5); ctx.lineTo(sx + 5, sy - 5);
      ctx.stroke();
    } else {
      ctx.globalAlpha = point.v === 1 ? 0.45 : 1.0;
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(sx, sy, 6, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(sx, sy, 8, 0, Math.PI * 2); ctx.stroke();
    }
    if (i === selectedPoint) {
      ctx.globalAlpha = 1;
      ctx.strokeStyle = '#7fb2ff';
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(sx, sy, 11, 0, Math.PI * 2); ctx.stroke();
    }
    ctx.globalAlpha = 1;
    ctx.font = '11px sans-serif';
    ctx.fillStyle = point.v === 0 && i !== placing ? '#888' : color;
    ctx.fillText(`K${i + 1} ${point.src_conf.toFixed(2)}`, sx + 10, sy - 8);
    ctx.restore();
  });
  renderPointList();
  updateRemoveButton();
  updateDiagramHighlight();
}

function renderPointList() {
  const list = document.getElementById('point-list');
  list.innerHTML = '';
  points.forEach((point, i) => {
    const li = document.createElement('li');
    li.className = [
      i === selectedPoint ? 'selected' : '',
      point.unplaced ? 'unplaced' : '',
    ].join(' ').trim();
    const info = point.unplaced
      ? (i === placing ? 'placing…' : 'click to place')
      : `v${point.v} · ${point.src_conf.toFixed(2)}`;
    const dotColor = point.unplaced ? '#555' : keypointColor(i);
    const keypoint = keypointFeatures[i];
    const desc = keypoint
      ? `${keypoint.feature} · ${keypoint.court_position}`
      : 'schema unavailable';
    li.innerHTML = `<span class="dot" style="background:${dotColor}"></span>` +
      `<span class="pt">K${i + 1}<span class="v">${info}</span>` +
      `<span class="desc">${desc}</span></span>`;
    li.addEventListener('click', () => {
      selectedPoint = i;
      if (point.unplaced) startPlacing(i);
      else { cancelPlacing(); draw(); }
    });
    list.appendChild(li);
  });
}

function startPlacing(i) {
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
  document.getElementById('progress').innerHTML =
    frames.length
      ? `frame <b>${index + 1}/${frames.length}</b> · ` +
        `<b class="labeled">${labeled}</b> labeled · source: ${labelSource}`
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

function setOrientationPanel(visible, evidence = null) {
  orientationPending = visible;
  document.getElementById('orientation-panel').hidden = !visible;
  document.getElementById('orientation-fallback').hidden = !evidence;
  if (evidence) {
    const first = evidence.first_end_points ?? 0;
    const second = evidence.second_end_points ?? 0;
    document.getElementById('orientation-evidence').textContent =
      `${evidence.reason || 'Automatic comparison was inconclusive.'} ` +
      `(located points: first end ${first}, second end ${second}).`;
  }
}

function showOrientationStatus(orientation) {
  const status = document.getElementById('orientation-status');
  if (!orientation) { status.textContent = ''; return; }
  status.textContent = `Orientation locked at ${orientation.anchor_frame_id}: north is image left; ` +
    `prefill normalization ${orientation.prefill_normalization}.`;
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
  setStatus('set the orientation anchor before model prefill is shown');
  return false;
}

async function setOrientation(rawFirstEndRelation) {
  if (!frames.length) return;
  const frame = frames[index];
  const response = await fetch(`/api/clip/${encodeURIComponent(frame.clip_id)}/orientation`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ frame_id: frame.frame_id, raw_first_end_relation: rawFirstEndRelation }),
  });
  const data = await response.json();
  if (!response.ok) {
    setOrientationPanel(true, data.evidence || { reason: data.error || 'Could not set orientation.' });
    setStatus('automatic orientation needs your confirmation', true);
    return;
  }
  orientationByClip[frame.clip_id] = data.orientation;
  showOrientationStatus(data.orientation);
  setOrientationPanel(false);
  await loadFrame(false);
}

function loadFrameImage(frame) {
  return new Promise((resolve) => {
    image = new Image();
    image.onload = () => { fitView(); draw(); resolve(); };
    image.src = '/images/' + frame.image.replace(/^frames\//, '');
  });
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
  if (!(await ensureOrientation(frame))) return;

  const url = `/api/label/${frame.frame_id}` + (predict ? '?predict=1' : '');
  const response = await fetch(url);
  const data = await response.json();
  if (!response.ok) {
    setStatus(data.error || 'could not load label', true);
    return;
  }
  points = data.label.keypoints;
  points.forEach((point) => {
    point.unplaced = point.x === 0 && point.y === 0 && point.v === 0 && point.src_conf === 0;
  });
  labelSource = data.source;
  showOrientationStatus(data.orientation);
  draw();
  if (points.length && points.every((point) => point.unplaced)) {
    setStatus('no model prediction — click a keypoint in the list to place it');
  } else {
    setStatus(labelSource === 'saved' ? 'saved label loaded' : 'normalized model prediction');
  }
}

async function save() {
  if (!frames.length || orientationPending) return;
  const frame = frames[index];
  const res = await fetch(`/api/label/${frame.frame_id}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      keypoints: points.map(({ x, y, v, src_conf }) => ({ x, y, v, src_conf })),
    }),
  });
  if (res.ok) {
    dirty = false;
    labelSource = 'saved';
    frame.label_status = 'labeled';
    setStatus('saved ✓');
    renderProgress();
  } else {
    setStatus('save failed', true);
  }
}

async function move(delta) {
  if (!frames.length) return;
  if (dirty) await save();
  index = Math.min(frames.length - 1, Math.max(0, index + delta));
  loadFrame(false);
}

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
  point.v = point.v === 2 ? 1 : point.v === 1 ? 0 : 2;
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
  setStatus(`K${selectedPoint + 1} removed — click it in the list to place it again`, true);
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
  else if (key === 'backspace' || key === 'delete') {
    event.preventDefault();
    removeSelectedPoint();
  }
  else if (key === 'r') loadFrame(true);
  else if (key === 'escape') cancelPlacing();
});
document.addEventListener('keyup', (event) => {
  if (event.code === 'Space') spaceDown = false;
});

document.getElementById('save').addEventListener('click', save);
document.getElementById('anchor-auto').addEventListener('click', () => setOrientation('auto'));
document.getElementById('anchor-first-left').addEventListener('click', () => setOrientation('left'));
document.getElementById('anchor-first-right').addEventListener('click', () => setOrientation('right'));
document.getElementById('repredict').addEventListener('click', () => loadFrame(true));
document.getElementById('remove-point').addEventListener('click', removeSelectedPoint);
document.getElementById('prev').addEventListener('click', () => move(-1));
document.getElementById('next').addEventListener('click', () => move(1));
window.addEventListener('resize', () => { fitView(); draw(); });

async function init() {
  const reference = document.getElementById('court-reference');
  const connectReference = () => {
    referenceSvgDocument = reference.contentDocument;
    updateDiagramHighlight();
  };
  reference.addEventListener('load', connectReference);
  if (reference.contentDocument) connectReference();
  const schemaResponse = await fetch('/api/keypoint-schema');
  if (!schemaResponse.ok) throw new Error('Could not load the keypoint schema');
  keypointSchema = await schemaResponse.json();
  keypointFeatures = keypointSchema.keypoints || [];
  const params = new URLSearchParams({ triage: 'keep' });
  if (clipId) params.set('clip', clipId);
  const data = await (await fetch('/api/frames?' + params)).json();
  frames = data.frames;
  index = 0;
  loadFrame(false);
}

init();
