const clipId = decodeURIComponent(location.pathname.split('/').pop());
document.getElementById('clip-name').textContent = clipId;

let frames = [];
let filter = 'all';
let selected = -1;

const grid = document.getElementById('grid');
const NEXT_STATUS = { pending: 'keep', keep: 'skip', skip: 'pending' };

function countsHtml(c) {
  const trained = frames.reduce((n, f) => n + (f.trained_split ? 1 : 0), 0);
  return `<b>${c.candidates}</b> candidates · <b>${c.pending}</b> pending · ` +
         `<b class="keep">${c.keep}</b> keep · <b class="skip">${c.skip}</b> skip · ` +
         `<b class="labeled">${c.labeled}</b> labeled` +
         (trained ? ` · <b class="trained">${trained}</b> trained-on` : '');
}

function visibleFrames() {
  return filter === 'all' ? frames : frames.filter(f => f.triage === filter);
}

function render() {
  grid.innerHTML = '';
  const shown = visibleFrames();
  shown.forEach((frame, i) => {
    const cell = document.createElement('div');
    cell.className = 'cell ' + frame.triage + (i === selected ? ' selected' : '');
    cell.dataset.frameId = frame.frame_id;
    const img = document.createElement('img');
    img.loading = 'lazy';
    img.src = '/thumbs/' + frame.thumb.replace(/^thumbs\//, '');
    cell.appendChild(img);
    if (frame.trained_split) {
      const badge = document.createElement('span');
      badge.className = 'trained-badge ' + frame.trained_split;
      badge.title = `In the ${frame.trained_split} split of the last fine-tune`;
      badge.textContent = frame.trained_split;
      cell.appendChild(badge);
    }
    const tag = document.createElement('span');
    tag.className = 'tag';
    tag.textContent = `${frame.timestamp_s.toFixed(1)}s · ${frame.triage}` +
      (frame.label_status === 'labeled' ? ' · labeled' : '');
    cell.appendChild(tag);
    cell.addEventListener('click', () => {
      selected = i;
      setStatus(frame, NEXT_STATUS[frame.triage]);
    });
    grid.appendChild(cell);
  });
}

async function setStatus(frame, status) {
  frame.triage = status;
  render();
  const res = await fetch('/api/triage', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ frame_id: frame.frame_id, status }),
  });
  const data = await res.json();
  document.getElementById('counts').innerHTML = countsHtml(data.clip_counts);
}

async function load() {
  const data = await (await fetch('/api/frames?clip=' + encodeURIComponent(clipId))).json();
  frames = data.frames;
  document.getElementById('counts').innerHTML = countsHtml(data.counts);
  render();
}

document.getElementById('filters').addEventListener('click', (event) => {
  const button = event.target.closest('button');
  if (!button) return;
  filter = button.dataset.filter;
  selected = -1;
  document.querySelectorAll('#filters button').forEach(
    b => b.classList.toggle('active', b === button));
  render();
});

document.addEventListener('keydown', (event) => {
  const shown = visibleFrames();
  if (!shown.length) return;
  const columns = Math.max(1, Math.floor(grid.clientWidth / 250));
  const key = event.key.toLowerCase();
  let handled = true;
  if (key === 'arrowright') selected = Math.min(shown.length - 1, selected + 1);
  else if (key === 'arrowleft') selected = Math.max(0, selected - 1);
  else if (key === 'arrowdown') selected = Math.min(shown.length - 1, selected + columns);
  else if (key === 'arrowup') selected = Math.max(0, selected - columns);
  else if (key === 'k' && selected >= 0) setStatus(shown[selected], 'keep');
  else if ((key === 's' || key === 'j') && selected >= 0) setStatus(shown[selected], 'skip');
  else if (key === 'u' && selected >= 0) setStatus(shown[selected], 'pending');
  else handled = false;
  if (handled) {
    event.preventDefault();
    if (selected < 0) selected = 0;
    render();
    const cell = grid.children[selected];
    if (cell) cell.scrollIntoView({ block: 'nearest' });
  }
});

load();
