// Keypoint correction editor: shows one kept frame at a time on a canvas with
// the model's predicted (or previously saved) points, editable by dragging.

const pathParts = location.pathname.split('/').filter(Boolean);
const clipId = pathParts.length > 1 ? decodeURIComponent(pathParts[1]) : null;
document.getElementById('clip-name').textContent = clipId || 'all kept frames';

// Where each of the model's 18 court keypoints belongs, derived by overlaying
// its confident predictions on reference footage. "End A" is the basket end it
// detects in that footage, "end B" the opposite end; far/near is the sideline
// farthest from / closest to the camera. K5/K7/K8/K11 (half-court features)
// are inferred from low-confidence predictions — keep ends and sides
// consistent within a clip when placing by hand.
const KEYPOINT_DESCRIPTIONS = [
  '3pt line at baseline · far side · end A',
  'key corner at baseline · far side · end A',
  'baseline center behind hoop · end A',
  'key corner at baseline · near side · end A',
  'center circle at half-court · near side',
  '3pt line at baseline · near side · end A',
  'half-court line at near sideline',
  'half-court line at far sideline',
  'key corner at free-throw line · far side · end A',
  'key corner at free-throw line · near side · end A',
  'center circle at half-court · far side',
  '3pt line at baseline · near side · end B',
  'key corner at baseline · near side · end B',
  'baseline center behind hoop · end B',
  'key corner at baseline · far side · end B',
  '3pt line at baseline · far side · end B',
  'key corner at free-throw line · far side · end B',
  'key corner at free-throw line · near side · end B',
];

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

// view transform: screen = image * scale + offset
let scale = 1, offsetX = 0, offsetY = 0;
let dragging = null;    // {type: 'point'|'pan', ...}
let spaceDown = false;
let placing = -1;       // index of the unplaced point being spawned, or -1

function confColor(c) {
  const clamped = Math.max(0, Math.min(1, c));
  return `rgb(${Math.round(255 * (1 - clamped))}, ${Math.round(255 * clamped)}, 0)`;
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
    const color = i === placing ? '#7fb2ff' : confColor(point.src_conf);
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
    const dotColor = point.unplaced ? '#555' : confColor(point.src_conf);
    li.innerHTML = `<span class="dot" style="background:${dotColor}"></span>` +
      `<span class="pt">K${i + 1}<span class="v">${info}</span>` +
      `<span class="desc">${KEYPOINT_DESCRIPTIONS[i] || ''}</span></span>`;
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

async function loadFrame(predict) {
  if (!frames.length) { renderProgress(); draw(); return; }
  const frame = frames[index];
  setStatus('loading…');
  const url = `/api/label/${frame.frame_id}` + (predict ? '?predict=1' : '');
  const data = await (await fetch(url)).json();
  points = data.label.keypoints;
  // (0,0) with zero confidence is the model's "not found" output — treat those
  // as unplaced: hidden until spawned from the list.
  points.forEach((point) => {
    point.unplaced = point.x === 0 && point.y === 0 && point.v === 0 && point.src_conf === 0;
  });
  labelSource = data.source;
  selectedPoint = -1;
  placing = -1;
  dirty = false;
  image = new Image();
  image.onload = () => {
    fitView();
    draw();
    if (points.length && points.every((point) => point.unplaced)) {
      setStatus('no model prediction — click a keypoint in the list to place it');
    } else {
      setStatus(labelSource === 'saved' ? 'saved label loaded' : 'model prediction');
    }
  };
  image.src = '/images/' + frame.image.replace(/^frames\//, '');
  renderProgress();
}

async function save() {
  if (!frames.length) return;
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

document.addEventListener('keydown', (event) => {
  if (event.code === 'Space') { spaceDown = true; event.preventDefault(); return; }
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 's') {
    event.preventDefault(); save(); return;
  }
  const key = event.key.toLowerCase();
  if (key === 'arrowright') move(1);
  else if (key === 'arrowleft') move(-1);
  else if (key === 'v') cycleVisibility();
  else if (key === 'r') loadFrame(true);
  else if (key === 'escape') cancelPlacing();
});
document.addEventListener('keyup', (event) => {
  if (event.code === 'Space') spaceDown = false;
});

document.getElementById('save').addEventListener('click', save);
document.getElementById('repredict').addEventListener('click', () => loadFrame(true));
document.getElementById('prev').addEventListener('click', () => move(-1));
document.getElementById('next').addEventListener('click', () => move(1));
window.addEventListener('resize', () => { fitView(); draw(); });

async function init() {
  const params = new URLSearchParams({ triage: 'keep' });
  if (clipId) params.set('clip', clipId);
  const data = await (await fetch('/api/frames?' + params)).json();
  frames = data.frames;
  index = 0;
  loadFrame(false);
}

init();
