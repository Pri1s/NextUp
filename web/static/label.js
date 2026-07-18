// Keypoint correction editor: shows one kept frame at a time on a canvas with
// the model's predicted (or previously saved) points, editable by dragging.

const pathParts = location.pathname.split('/').filter(Boolean);
const clipId = pathParts.length > 1 ? decodeURIComponent(pathParts[1]) : null;
document.getElementById('clip-name').textContent = clipId || 'all kept frames';

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
    const [sx, sy] = toScreen(point.x, point.y);
    const color = confColor(point.src_conf);
    ctx.save();
    if (point.v === 0) {
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
    ctx.fillStyle = point.v === 0 ? '#888' : color;
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
    li.className = i === selectedPoint ? 'selected' : '';
    li.innerHTML = `<span class="dot" style="background:${confColor(point.src_conf)}"></span>` +
      `K${i + 1}<span class="v">v${point.v} · ${point.src_conf.toFixed(2)}</span>`;
    li.addEventListener('click', () => { selectedPoint = i; draw(); });
    list.appendChild(li);
  });
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
  labelSource = data.source;
  selectedPoint = -1;
  dirty = false;
  image = new Image();
  image.onload = () => { fitView(); draw(); setStatus(labelSource === 'saved' ? 'saved label loaded' : 'model prediction'); };
  image.src = '/images/' + frame.image.replace(/^frames\//, '');
  renderProgress();
}

async function save() {
  if (!frames.length) return;
  const frame = frames[index];
  const res = await fetch(`/api/label/${frame.frame_id}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ keypoints: points }),
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
  const hit = hitPoint(sx, sy);
  if (hit >= 0) {
    selectedPoint = hit;
    dragging = { type: 'point', index: hit };
    draw();
  }
});

canvas.addEventListener('mousemove', (event) => {
  if (!dragging) return;
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
  if (selectedPoint < 0) return;
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
