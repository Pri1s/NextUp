// Human court-marking reference editor.
//
// The canvas stores original 1280x720 frame coordinates regardless of zoom.
// Drafts autosave to an isolated human directory; only an explicit, validated
// "Finish frame" action writes a benchmark-ready .frame.json file.

const canvas = document.getElementById('benchmark-canvas');
const ctx = canvas.getContext('2d');
const canvasWrap = document.getElementById('benchmark-canvas-wrap');
const referenceCanvas = document.getElementById('marking-reference-canvas');
const referenceCtx = referenceCanvas.getContext('2d');
const saveStatus = document.getElementById('benchmark-save-status');
const toast = document.getElementById('benchmark-toast');

let benchmark = null;
let frames = [];
let frameIndex = 0;
let annotation = null;
let image = new Image();
let activeFeatureId = null;
let activeJunctionId = null;
let pendingGap = false;
let dirty = false;
let finalized = false;
let saveTimer = null;
let saveSequence = Promise.resolve();

// screen = image * scale + offset
let scale = 1;
let offsetX = 0;
let offsetY = 0;
let dragging = null;
let spaceDown = false;
let referenceBadgeHits = [];
let referenceBasketSide = 'left';

const featureById = () => new Map(annotation.features.map((item) => [item.feature, item]));
const skipById = () => new Map(annotation.skipped.map((item) => [item.feature, item]));
const junctionById = () => new Map(annotation.junctions.map((item) => [item.junction, item]));
const catalogFeature = (id) => benchmark.features.find((item) => item.id === id);
const catalogJunction = (id) => benchmark.junctions.find((item) => item.id === id);

function titleCase(value) {
  return value.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function featureColor(id, alpha = 1) {
  const index = benchmark.features.findIndex((item) => item.id === id);
  const hue = (index * 47 + 18) % 360;
  return `hsla(${hue} 84% 64% / ${alpha})`;
}

function tracePointLabel(index) {
  return index < 26 ? String.fromCharCode(65 + index) : String(index + 1);
}

function samplePolyline(points, count) {
  if (!points?.length || count < 1) return [];
  if (points.length === 1 || count === 1) return [points[0]];

  const segmentLengths = [];
  const cumulative = [0];
  for (let index = 1; index < points.length; index += 1) {
    const length = Math.hypot(
      points[index][0] - points[index - 1][0],
      points[index][1] - points[index - 1][1],
    );
    segmentLengths.push(length);
    cumulative.push(cumulative.at(-1) + length);
  }
  const total = cumulative.at(-1);
  if (!total) return Array.from({ length: count }, () => points[0]);

  return Array.from({ length: count }, (_, sampleIndex) => {
    const target = total * sampleIndex / (count - 1);
    let segment = 0;
    while (segment < segmentLengths.length - 1 && cumulative[segment + 1] < target) {
      segment += 1;
    }
    const length = segmentLengths[segment];
    const amount = length ? (target - cumulative[segment]) / length : 0;
    return [
      points[segment][0] + (points[segment + 1][0] - points[segment][0]) * amount,
      points[segment][1] + (points[segment + 1][1] - points[segment][1]) * amount,
    ];
  });
}

function referenceTransform() {
  const width = referenceCanvas.clientWidth;
  const height = referenceCanvas.clientHeight;
  const court = benchmark.reference_court;
  const padding = { left: 25, right: 25, top: 19, bottom: 19 };
  const usableWidth = width - padding.left - padding.right;
  const usableHeight = height - padding.top - padding.bottom;
  const courtScale = Math.min(
    usableWidth / court.half_length,
    usableHeight / court.width,
  );
  const renderedWidth = court.half_length * courtScale;
  const renderedHeight = court.width * courtScale;
  const originX = (width - renderedWidth) / 2;
  const originY = (height - renderedHeight) / 2;
  return {
    point: ([x, y]) => ({
      x: originX + (referenceBasketSide === 'right' ? court.half_length - x : x) * courtScale,
      y: originY + y * courtScale,
    }),
    originX,
    originY,
    renderedWidth,
    renderedHeight,
    courtScale,
  };
}

function drawReferencePolyline(feature, transform, selected) {
  const points = benchmark.reference_court.markings[feature.id];
  if (!points?.length) return;
  referenceCtx.save();
  referenceCtx.beginPath();
  referenceCtx.rect(
    transform.originX,
    transform.originY,
    transform.renderedWidth,
    transform.renderedHeight,
  );
  referenceCtx.clip();
  referenceCtx.beginPath();
  points.forEach((point, index) => {
    const p = transform.point(point);
    if (index === 0) referenceCtx.moveTo(p.x, p.y);
    else referenceCtx.lineTo(p.x, p.y);
  });
  if (feature.id === 'free_throw_circle_near_half') {
    referenceCtx.setLineDash([5, 4]);
  }
  if (selected) {
    referenceCtx.strokeStyle = 'rgb(255 255 255 / 0.9)';
    referenceCtx.lineWidth = 7;
    referenceCtx.stroke();
    referenceCtx.beginPath();
    points.forEach((point, index) => {
      const p = transform.point(point);
      if (index === 0) referenceCtx.moveTo(p.x, p.y);
      else referenceCtx.lineTo(p.x, p.y);
    });
  }
  referenceCtx.strokeStyle = featureColor(feature.id, selected ? 1 : 0.48);
  referenceCtx.lineWidth = selected ? 4 : 2;
  referenceCtx.lineCap = 'round';
  referenceCtx.lineJoin = 'round';
  referenceCtx.stroke();
  referenceCtx.restore();
}

function drawReferenceCourt() {
  if (!benchmark?.reference_court || !referenceCanvas.clientWidth) return;
  const width = referenceCanvas.clientWidth;
  const height = referenceCanvas.clientHeight;
  referenceCtx.clearRect(0, 0, width, height);
  const transform = referenceTransform();

  referenceCtx.fillStyle = '#111113';
  referenceCtx.strokeStyle = '#71717a';
  referenceCtx.lineWidth = 1;
  referenceCtx.fillRect(
    transform.originX,
    transform.originY,
    transform.renderedWidth,
    transform.renderedHeight,
  );
  referenceCtx.strokeRect(
    transform.originX,
    transform.originY,
    transform.renderedWidth,
    transform.renderedHeight,
  );

  // A faint lane field makes the line identities legible without introducing
  // any image-derived or model-derived evidence.
  const laneTopLeft = transform.point([0, 17]);
  const laneBottomRight = transform.point([19, 33]);
  referenceCtx.fillStyle = 'rgb(255 255 255 / 0.025)';
  referenceCtx.fillRect(
    Math.min(laneTopLeft.x, laneBottomRight.x),
    Math.min(laneTopLeft.y, laneBottomRight.y),
    Math.abs(laneBottomRight.x - laneTopLeft.x),
    Math.abs(laneBottomRight.y - laneTopLeft.y),
  );

  const inactive = benchmark.features.filter((feature) => feature.id !== activeFeatureId);
  inactive.forEach((feature) => drawReferencePolyline(feature, transform, false));
  const active = catalogFeature(activeFeatureId);
  if (active) drawReferencePolyline(active, transform, true);

  const basket = transform.point(benchmark.reference_court.basket_center);
  referenceCtx.beginPath();
  referenceCtx.arc(basket.x, basket.y, 4, 0, Math.PI * 2);
  referenceCtx.strokeStyle = '#d4d4d8';
  referenceCtx.lineWidth = 1.5;
  referenceCtx.stroke();
  referenceCtx.beginPath();
  referenceCtx.moveTo(basket.x - 2, basket.y - 11);
  referenceCtx.lineTo(basket.x - 2, basket.y + 11);
  referenceCtx.strokeStyle = '#71717a';
  referenceCtx.stroke();

  // The feature number identifies the marking. A–D demonstrate the minimum
  // ordered samples needed to turn that marking into a trace.
  if (active) {
    const guidePoints = samplePolyline(
      benchmark.reference_court.markings[active.id],
      active.minimum_samples + 2,
    ).slice(1, -1);
    guidePoints.forEach((point, index) => {
      const { x, y } = transform.point(point);
      referenceCtx.beginPath();
      referenceCtx.arc(x, y, 7, 0, Math.PI * 2);
      referenceCtx.fillStyle = '#fafafa';
      referenceCtx.fill();
      referenceCtx.strokeStyle = featureColor(active.id);
      referenceCtx.lineWidth = 2;
      referenceCtx.stroke();
      referenceCtx.fillStyle = '#09090b';
      referenceCtx.font = '750 8px Inter, system-ui, sans-serif';
      referenceCtx.textAlign = 'center';
      referenceCtx.textBaseline = 'middle';
      referenceCtx.fillText(tracePointLabel(index), x, y + 0.5);
    });
  }

  referenceBadgeHits = [];
  benchmark.features.forEach((feature, index) => {
    // Anchors are selected from the same authoritative polylines drawn above,
    // so a number can never drift away from the marking it names.
    const anchor = benchmark.reference_court.label_anchors[feature.id];
    if (!anchor) return;
    const { x, y } = transform.point(anchor);
    const selected = feature.id === activeFeatureId;
    // The selected feature is already identified in the card immediately below
    // the map. Hiding its large badge here keeps it from covering A–F on short arcs.
    if (selected) return;
    const radius = selected ? 12 : 9.5;

    referenceCtx.beginPath();
    referenceCtx.arc(x, y, radius + (selected ? 2 : 0), 0, Math.PI * 2);
    referenceCtx.fillStyle = selected ? '#fafafa' : '#09090b';
    referenceCtx.fill();
    referenceCtx.beginPath();
    referenceCtx.arc(x, y, radius, 0, Math.PI * 2);
    referenceCtx.fillStyle = featureColor(feature.id);
    referenceCtx.fill();
    referenceCtx.strokeStyle = selected ? '#09090b' : 'rgb(255 255 255 / 0.5)';
    referenceCtx.lineWidth = selected ? 2 : 1;
    referenceCtx.stroke();
    referenceCtx.fillStyle = '#09090b';
    referenceCtx.font = `750 ${selected ? 11 : 9}px Inter, system-ui, sans-serif`;
    referenceCtx.textAlign = 'center';
    referenceCtx.textBaseline = 'middle';
    referenceCtx.fillText(String(index + 1), x, y + 0.5);
    referenceBadgeHits.push({ id: feature.id, x, y, radius: radius + 5 });
  });

  const selectedIndex = benchmark.features.findIndex(
    (feature) => feature.id === activeFeatureId,
  );
  document.getElementById('reference-selected-number').textContent =
    selectedIndex >= 0 ? String(selectedIndex + 1).padStart(2, '0') : '—';
  document.getElementById('reference-selected-name').textContent =
    active ? titleCase(active.id) : 'Select a marking';
  document.getElementById('reference-selected-hint').textContent =
    active
      ? `A → ${tracePointLabel(active.minimum_samples - 1)} shows order only—not fixed locations`
      : 'Choose a number or list item';
}

function resizeReferenceCanvas() {
  const dpr = window.devicePixelRatio || 1;
  referenceCanvas.width = Math.round(referenceCanvas.clientWidth * dpr);
  referenceCanvas.height = Math.round(referenceCanvas.clientHeight * dpr);
  referenceCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  drawReferenceCourt();
}

function setReferenceBasketSide(side, persist = true) {
  referenceBasketSide = side === 'right' ? 'right' : 'left';
  document.getElementById('guide-basket-left').classList.toggle(
    'active',
    referenceBasketSide === 'left',
  );
  document.getElementById('guide-basket-right').classList.toggle(
    'active',
    referenceBasketSide === 'right',
  );
  document.getElementById('guide-basket-left').setAttribute(
    'aria-pressed',
    String(referenceBasketSide === 'left'),
  );
  document.getElementById('guide-basket-right').setAttribute(
    'aria-pressed',
    String(referenceBasketSide === 'right'),
  );
  document.getElementById('reference-half-rule').textContent =
    `Label only the half between midcourt and the ${referenceBasketSide} basket. ` +
    `Ignore matching paint on the ${referenceBasketSide === 'right' ? 'left' : 'right'} half.`;
  if (persist && annotation?.frame_id) {
    localStorage.setItem(
      `benchmark-guide-basket-side:${annotation.frame_id}`,
      referenceBasketSide,
    );
  }
  drawReferenceCourt();
}

function setStatus(message, mode = '') {
  saveStatus.textContent = message;
  saveStatus.className = `status-pill ${mode}`.trim();
}

function showToast(message, isError = false) {
  toast.textContent = message;
  toast.className = `toast show${isError ? ' error' : ''}`;
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => { toast.className = 'toast'; }, 3200);
}

function markDirty() {
  dirty = true;
  finalized = false;
  setStatus('Unsaved changes', 'dirty');
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => persist(false), 450);
}

async function persist(finalize) {
  clearTimeout(saveTimer);
  if (!annotation) return false;
  annotation.annotator_id = document.getElementById('annotator-id').value.trim();
  annotation.notes = document.getElementById('frame-notes').value;
  const frameId = annotation.frame_id;
  setStatus(finalize ? 'Checking frame…' : 'Saving…');

  const requestBody = JSON.stringify({ annotation, finalize });
  let result = false;
  saveSequence = saveSequence.then(async () => {
    const response = await fetch(`/api/benchmark/annotation/${encodeURIComponent(frameId)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: requestBody,
    });
    const payload = await response.json();
    if (!response.ok) {
      const detail = (payload.problems || [payload.error]).join('\n');
      setStatus('Needs attention', 'error');
      showToast(detail, true);
      result = false;
      return;
    }
    dirty = false;
    finalized = Boolean(payload.finalized);
    const frame = frames.find((item) => item.frame_id === frameId);
    if (frame && finalized) frame.complete = true;
    setStatus(finalized ? 'Frame complete' : 'Draft saved', finalized ? 'complete' : '');
    updateProgress();
    result = true;
  }).catch((error) => {
    setStatus('Save failed', 'error');
    showToast(error.message, true);
    result = false;
  });
  await saveSequence;
  return result;
}

function fitImage() {
  if (!image.naturalWidth) return;
  const margin = 28;
  scale = Math.min(
    (canvasWrap.clientWidth - margin * 2) / image.naturalWidth,
    (canvasWrap.clientHeight - margin * 2) / image.naturalHeight,
  );
  scale = Math.max(0.05, scale);
  offsetX = (canvasWrap.clientWidth - image.naturalWidth * scale) / 2;
  offsetY = (canvasWrap.clientHeight - image.naturalHeight * scale) / 2;
  draw();
}

function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(canvasWrap.clientWidth * dpr);
  canvas.height = Math.round(canvasWrap.clientHeight * dpr);
  canvas.style.width = `${canvasWrap.clientWidth}px`;
  canvas.style.height = `${canvasWrap.clientHeight}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}

function screenPoint(point) {
  return { x: point.x * scale + offsetX, y: point.y * scale + offsetY };
}

function imagePoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(image.naturalWidth, (event.clientX - rect.left - offsetX) / scale)),
    y: Math.max(0, Math.min(image.naturalHeight, (event.clientY - rect.top - offsetY) / scale)),
  };
}

function insertionIndexForPoint(points, point) {
  if (points.length < 2) return points.length;
  let best = null;
  for (let index = 0; index < points.length - 1; index += 1) {
    const start = points[index];
    const end = points[index + 1];
    const dx = end.x - start.x;
    const dy = end.y - start.y;
    const lengthSquared = dx * dx + dy * dy;
    const rawAmount = lengthSquared
      ? ((point.x - start.x) * dx + (point.y - start.y) * dy) / lengthSquared
      : 0;
    const amount = Math.max(0, Math.min(1, rawAmount));
    const closest = {
      x: start.x + dx * amount,
      y: start.y + dy * amount,
    };
    const distance = Math.hypot(point.x - closest.x, point.y - closest.y);
    if (!best || distance < best.distance) {
      best = { index, rawAmount, distance };
    }
  }
  if (best.index === 0 && best.rawAmount <= 0) return 0;
  if (best.index === points.length - 2 && best.rawAmount >= 1) return points.length;
  return best.index + 1;
}

function arcTraceChangesDirection(points) {
  let direction = 0;
  for (let index = 1; index < points.length - 1; index += 1) {
    const a = points[index - 1];
    const b = points[index];
    const c = points[index + 1];
    const turn = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
    if (Math.abs(turn) < 1e-9) continue;
    const nextDirection = Math.sign(turn);
    if (direction && nextDirection !== direction) return true;
    direction = nextDirection;
  }
  return false;
}

function linePath(points, gaps, color, selected) {
  if (!points.length) return;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = color;
  ctx.lineWidth = selected ? 3 : 2;
  ctx.beginPath();
  points.forEach((point, index) => {
    const p = screenPoint(point);
    if (index === 0 || gaps.includes(index - 1)) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  });
  ctx.stroke();
}

function draw() {
  const width = canvasWrap.clientWidth;
  const height = canvasWrap.clientHeight;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#050505';
  ctx.fillRect(0, 0, width, height);
  if (!image.naturalWidth || !annotation) return;
  ctx.imageSmoothingEnabled = scale < 1;
  ctx.drawImage(
    image,
    offsetX,
    offsetY,
    image.naturalWidth * scale,
    image.naturalHeight * scale,
  );

  for (const feature of annotation.features) {
    const selected = feature.feature === activeFeatureId && !activeJunctionId;
    linePath(
      feature.points,
      feature.occluded_after || [],
      featureColor(feature.feature, selected ? 1 : 0.62),
      selected,
    );
    feature.points.forEach((point, pointIndex) => {
      const p = screenPoint(point);
      ctx.beginPath();
      ctx.arc(p.x, p.y, selected ? 7 : 3.2, 0, Math.PI * 2);
      ctx.fillStyle = featureColor(feature.feature, selected ? 1 : 0.72);
      ctx.fill();
      if (selected) {
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1.25;
        ctx.stroke();
        ctx.fillStyle = '#09090b';
        ctx.font = '750 9px ui-monospace, monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(tracePointLabel(pointIndex), p.x, p.y + 0.5);
      }
    });
  }

  for (const junction of annotation.junctions) {
    const p = screenPoint(junction.point);
    const selected = junction.junction === activeJunctionId;
    ctx.save();
    ctx.translate(p.x, p.y);
    ctx.rotate(Math.PI / 4);
    ctx.fillStyle = selected ? '#fff' : 'rgb(255 255 255 / 0.75)';
    ctx.strokeStyle = '#09090b';
    ctx.lineWidth = 2;
    ctx.fillRect(-5, -5, 10, 10);
    ctx.strokeRect(-5, -5, 10, 10);
    ctx.restore();
  }

  document.getElementById('zoom-reset').textContent = `${Math.round(scale * 100)}%`;
}

function currentFeature() {
  return annotation.features.find((item) => item.feature === activeFeatureId) || null;
}

function ensureFeature() {
  let feature = currentFeature();
  if (!feature) {
    const catalog = catalogFeature(activeFeatureId);
    annotation.skipped = annotation.skipped.filter((item) => item.feature !== activeFeatureId);
    feature = {
      feature: activeFeatureId,
      kind: catalog.kind,
      points: [],
      visibility: document.getElementById('feature-visibility').value,
      uncertainty_px: Number(document.getElementById('feature-uncertainty').value),
      occluded_after: [],
      notes: document.getElementById('feature-notes').value,
    };
    annotation.features.push(feature);
  }
  return feature;
}

function featureIsComplete(id) {
  const catalog = catalogFeature(id);
  const feature = annotation.features.find((item) => item.feature === id);
  if (feature) return feature.points.length >= catalog.minimum_samples;
  return annotation.skipped.some((item) => item.feature === id);
}

function annotationProgress() {
  const features = featureById();
  const skips = skipById();
  const progress = {
    complete: 0,
    skipped: 0,
    started: 0,
    untouched: 0,
    missingSamples: 0,
    firstIncomplete: null,
  };

  benchmark.features.forEach((catalog) => {
    const feature = features.get(catalog.id);
    if (skips.has(catalog.id)) {
      progress.complete += 1;
      progress.skipped += 1;
    } else if (feature?.points.length >= catalog.minimum_samples) {
      progress.complete += 1;
    } else if (feature?.points.length) {
      progress.started += 1;
      progress.missingSamples += catalog.minimum_samples - feature.points.length;
      progress.firstIncomplete ||= catalog.id;
    } else {
      progress.untouched += 1;
      progress.firstIncomplete ||= catalog.id;
    }
  });
  return progress;
}

function dispositionCount() {
  return benchmark.features.filter((item) => featureIsComplete(item.id)).length;
}

function annotationComplete() {
  return dispositionCount() === benchmark.features.length;
}

function renderFeatureList() {
  const container = document.getElementById('feature-list');
  container.innerHTML = '';
  const features = featureById();
  const skips = skipById();
  benchmark.features.forEach((item, itemIndex) => {
    const feature = features.get(item.id);
    const skipped = skips.get(item.id);
    const complete = featureIsComplete(item.id);
    const remainingSamples = feature
      ? Math.max(0, item.minimum_samples - feature.points.length)
      : item.minimum_samples;
    const button = document.createElement('button');
    button.className = `feature-row${item.id === activeFeatureId && !activeJunctionId ? ' active' : ''}`;
    button.dataset.feature = item.id;
    button.innerHTML = `
      <span class="feature-index">${String(itemIndex + 1).padStart(2, '0')}</span>
      <span class="feature-row-swatch" style="--feature-color:${featureColor(item.id)}"></span>
      <span class="feature-row-copy">
        <b>${titleCase(item.id)}</b>
        <small>${item.kind} · ${feature ? complete
          ? `${feature.points.length} points · complete`
          : `${feature.points.length} / ${item.minimum_samples} points · add ${remainingSamples}` :
          skipped ? titleCase(skipped.reason) : `needs ${item.minimum_samples}+ samples`}</small>
      </span>
      <span class="feature-state ${complete ? skipped ? 'skipped' : 'done' : ''}">
        ${complete ? skipped ? '—' : '✓' : '○'}
      </span>`;
    button.addEventListener('click', () => selectFeature(item.id));
    container.appendChild(button);
  });
  document.getElementById('feature-progress').textContent =
    `${dispositionCount()} / ${benchmark.features.length}`;
}

function renderActiveFeature() {
  const catalog = catalogFeature(activeFeatureId);
  if (!catalog) return;
  const feature = currentFeature();
  const skipped = annotation.skipped.find((item) => item.feature === activeFeatureId);
  const skipMode = Boolean(skipped);

  document.getElementById('active-feature-swatch').style.background = featureColor(catalog.id);
  document.getElementById('active-feature-kind').textContent =
    `${catalog.kind} · ${catalog.dimensions}${catalog.held_out ? ' · held-out check' : ''}`;
  document.getElementById('active-feature-name').textContent = titleCase(catalog.id);
  document.getElementById('active-feature-description').textContent = catalog.description;
  document.getElementById('active-feature-warning').textContent =
    `Telling it apart: ${catalog.discrimination}`;
  document.getElementById('sample-count').textContent =
    feature ? feature.points.length >= catalog.minimum_samples
      ? `${feature.points.length} points · complete`
      : `${feature.points.length} / ${catalog.minimum_samples} minimum`
      : skipped ? 'Skipped' : `0 / ${catalog.minimum_samples} minimum`;
  const remainingSamples = Math.max(
    0,
    catalog.minimum_samples - (feature?.points.length || 0),
  );
  const requirement = document.getElementById('trace-requirement');
  const invalidArcOrder = catalog.kind === 'arc' &&
    feature?.points.length >= 4 &&
    arcTraceChangesDirection(feature.points);
  if (invalidArcOrder) {
    requirement.textContent =
      'This trace doubles back across the curve. Clear it and click around the painted arc ' +
      'continuously in one direction—clockwise or counterclockwise.';
  } else if (remainingSamples && catalog.kind === 'arc') {
    requirement.textContent =
      `Add ${remainingSamples} more observed point${remainingSamples === 1 ? '' : 's'} on ` +
      'visible paint. Continue around the curve in one direction; never jump across its interior.';
  } else if (remainingSamples) {
    requirement.textContent =
      `Add ${remainingSamples} more observed point${remainingSamples === 1 ? '' : 's'} anywhere ` +
      'along the visible paint. Click between existing letters to insert a sample; ' +
      'the map letters show order, not fixed court locations.';
  } else {
    requirement.textContent =
      'Minimum reached. Add more points wherever they improve the shape of the trace.';
  }
  requirement.classList.toggle('error', invalidArcOrder);
  requirement.classList.toggle('complete', remainingSamples === 0 && !invalidArcOrder);
  document.getElementById('feature-visibility').value = feature?.visibility || 'clear';
  document.getElementById('feature-uncertainty').value =
    String(feature?.uncertainty_px ?? 1);
  document.getElementById('feature-notes').value = feature?.notes || skipped?.notes || '';
  document.getElementById('skip-reason').value = skipped?.reason || 'not_visible';
  setMode(skipMode ? 'skip' : 'trace', false);
  document.getElementById('mark-gap').classList.toggle('active', pendingGap);
  document.getElementById('undo-point').disabled = !feature?.points.length;
  document.getElementById('clear-trace').disabled = !feature;
}

function renderJunctions() {
  const container = document.getElementById('junction-list');
  const existing = junctionById();
  container.innerHTML = '';
  benchmark.junctions.forEach((item) => {
    const placed = existing.has(item.id);
    const constituentsPresent = item.crossing.every(
      (featureId) => annotation.features.some((feature) => feature.feature === featureId),
    );
    const button = document.createElement('button');
    button.className = `junction-row${item.id === activeJunctionId ? ' active' : ''}`;
    button.disabled = !constituentsPresent;
    button.innerHTML = `
      <span class="junction-marker">${placed ? '◆' : '◇'}</span>
      <span><b>${titleCase(item.id)}</b>
      <small>${item.crossing.map(titleCase).join(' × ')}</small></span>`;
    button.title = constituentsPresent
      ? 'Click, then place the visible centerline intersection'
      : 'Trace both crossing markings first';
    button.addEventListener('click', () => selectJunction(item.id));
    container.appendChild(button);
  });
  document.getElementById('junction-progress').textContent =
    `${annotation.junctions.length} placed`;
  document.getElementById('clear-junction').disabled = !activeJunctionId ||
    !existing.has(activeJunctionId);
}

function updateCompletion() {
  const complete = annotationComplete();
  const progress = annotationProgress();
  const continueButton = document.getElementById('continue-incomplete');
  document.getElementById('finish-frame').disabled = !complete;
  continueButton.hidden = complete;
  continueButton.dataset.feature = progress.firstIncomplete || '';

  let message = 'Every marking is accounted for. Review the overlay, then finish.';
  if (!complete && progress.started && !progress.untouched) {
    message = `${progress.started} trace${progress.started === 1 ? ' is' : 's are'} started ` +
      `but unfinished—add ${progress.missingSamples} more point` +
      `${progress.missingSamples === 1 ? '' : 's'} total. Each trace needs at least four ` +
      'ordered points along the same painted marking.';
  } else if (!complete && progress.started) {
    message = `${progress.started} trace${progress.started === 1 ? ' is' : 's are'} unfinished ` +
      `(${progress.missingSamples} more point${progress.missingSamples === 1 ? '' : 's'} needed), ` +
      `and ${progress.untouched} marking${progress.untouched === 1 ? '' : 's'} still need a trace ` +
      'or skip reason.';
  } else if (!complete) {
    message = `${progress.untouched} marking${progress.untouched === 1 ? '' : 's'} not started. ` +
      'Trace each with at least four ordered points, or skip it with a reason.';
  }
  document.getElementById('completion-message').textContent = message;
}

function updateProgress() {
  const completed = frames.filter((item) => item.complete).length;
  document.getElementById('benchmark-progress').innerHTML =
    `<b>${completed}</b> of <b>${frames.length}</b> frames complete · frame ${frameIndex + 1}`;
  renderFrameDots();
}

function renderFrameDots() {
  const chip = document.getElementById('frame-chip');
  const frame = frames[frameIndex];
  chip.innerHTML = `
    <span>${frame.frame_id}</span>
    <b>${titleCase(frame.stratum)}</b>
    <span class="frame-dots" aria-label="Frame progress">
      ${frames.map((item, index) =>
        `<i class="${index === frameIndex ? 'active' : ''} ${item.complete ? 'complete' : ''}"
          title="${item.frame_id}"></i>`).join('')}
    </span>`;
}

function renderAll() {
  renderFeatureList();
  renderActiveFeature();
  renderJunctions();
  updateCompletion();
  updateProgress();
  drawReferenceCourt();
  draw();
}

function setMode(mode, update = true) {
  const skip = mode === 'skip';
  document.getElementById('mode-trace').classList.toggle('active', !skip);
  document.getElementById('mode-skip').classList.toggle('active', skip);
  document.getElementById('mode-trace').setAttribute('aria-selected', String(!skip));
  document.getElementById('mode-skip').setAttribute('aria-selected', String(skip));
  document.getElementById('trace-controls').hidden = skip;
  document.getElementById('skip-controls').hidden = !skip;
  if (update && !skip) {
    annotation.skipped = annotation.skipped.filter((item) => item.feature !== activeFeatureId);
    markDirty();
    renderAll();
  }
}

function selectFeature(id) {
  activeFeatureId = id;
  activeJunctionId = null;
  pendingGap = false;
  renderAll();
}

function selectJunction(id) {
  activeJunctionId = id;
  pendingGap = false;
  renderJunctions();
  draw();
  showToast(`Click the visible ${titleCase(id)} centerline intersection.`);
}

function nearestEditablePoint(event) {
  const mouse = {
    x: event.clientX - canvas.getBoundingClientRect().left,
    y: event.clientY - canvas.getBoundingClientRect().top,
  };
  if (activeJunctionId) {
    const junction = annotation.junctions.find((item) => item.junction === activeJunctionId);
    if (!junction) return null;
    const p = screenPoint(junction.point);
    return Math.hypot(mouse.x - p.x, mouse.y - p.y) <= 12
      ? { type: 'junction', item: junction }
      : null;
  }
  const feature = currentFeature();
  if (!feature) return null;
  let best = null;
  feature.points.forEach((point, index) => {
    const p = screenPoint(point);
    const distance = Math.hypot(mouse.x - p.x, mouse.y - p.y);
    if (distance <= 11 && (!best || distance < best.distance)) {
      best = { type: 'point', item: point, index, distance };
    }
  });
  return best;
}

canvas.addEventListener('pointerdown', (event) => {
  if (!annotation || !image.naturalWidth) return;
  const editable = nearestEditablePoint(event);
  if (spaceDown || event.button === 1) {
    dragging = {
      type: 'pan',
      startX: event.clientX,
      startY: event.clientY,
      offsetX,
      offsetY,
    };
  } else if (editable) {
    dragging = { ...editable };
  } else {
    const point = imagePoint(event);
    if (activeJunctionId) {
      const catalog = catalogJunction(activeJunctionId);
      const ready = catalog.crossing.every(
        (id) => annotation.features.some((feature) => feature.feature === id),
      );
      if (!ready) {
        showToast('Trace both crossing markings before placing this intersection.', true);
        return;
      }
      annotation.junctions = annotation.junctions.filter(
        (item) => item.junction !== activeJunctionId,
      );
      annotation.junctions.push({
        junction: activeJunctionId,
        point,
        uncertainty_px: 1,
        notes: '',
      });
      markDirty();
      renderAll();
      return;
    }
    if (!activeFeatureId ||
        !document.getElementById('skip-controls').hidden) return;
    const feature = ensureFeature();
    if (pendingGap && feature.points.length) {
      feature.occluded_after.push(feature.points.length - 1);
      pendingGap = false;
      feature.points.push(point);
    } else if (feature.kind === 'arc') {
      // Arc samples must be clicked continuously around the curve. Inferring an
      // insertion position from an incomplete chord can put a point on the wrong
      // side and create a self-crossing trace.
      feature.points.push(point);
    } else {
      const insertionIndex = insertionIndexForPoint(feature.points, point);
      feature.occluded_after = feature.occluded_after.map(
        (index) => index >= insertionIndex ? index + 1 : index,
      );
      feature.points.splice(insertionIndex, 0, point);
      if (insertionIndex < feature.points.length - 1) {
        showToast(
          `Inserted as ${tracePointLabel(insertionIndex)}; later sample letters were updated.`,
        );
      }
    }
    markDirty();
    renderAll();
  }
  canvas.setPointerCapture(event.pointerId);
});

canvas.addEventListener('pointermove', (event) => {
  if (image.naturalWidth) {
    const point = imagePoint(event);
    document.getElementById('canvas-coordinates').textContent =
      `x ${point.x.toFixed(1)} · y ${point.y.toFixed(1)}`;
  }
  if (!dragging) return;
  if (dragging.type === 'pan') {
    offsetX = dragging.offsetX + event.clientX - dragging.startX;
    offsetY = dragging.offsetY + event.clientY - dragging.startY;
  } else {
    const point = imagePoint(event);
    dragging.item.x = point.x;
    dragging.item.y = point.y;
    dirty = true;
  }
  draw();
});

canvas.addEventListener('pointerup', () => {
  if (dragging && dragging.type !== 'pan') {
    markDirty();
    renderAll();
  }
  dragging = null;
});

canvas.addEventListener('wheel', (event) => {
  event.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const mouseX = event.clientX - rect.left;
  const mouseY = event.clientY - rect.top;
  const imageX = (mouseX - offsetX) / scale;
  const imageY = (mouseY - offsetY) / scale;
  const factor = event.deltaY < 0 ? 1.18 : 1 / 1.18;
  scale = Math.max(0.1, Math.min(16, scale * factor));
  offsetX = mouseX - imageX * scale;
  offsetY = mouseY - imageY * scale;
  draw();
}, { passive: false });

referenceCanvas.addEventListener('pointermove', (event) => {
  const rect = referenceCanvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const hit = referenceBadgeHits.find(
    (badge) => Math.hypot(x - badge.x, y - badge.y) <= badge.radius,
  );
  referenceCanvas.style.cursor = hit ? 'pointer' : 'default';
  referenceCanvas.title = hit ? titleCase(hit.id) : 'Numbered court marking map';
});
referenceCanvas.addEventListener('click', (event) => {
  const rect = referenceCanvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const hit = referenceBadgeHits.find(
    (badge) => Math.hypot(x - badge.x, y - badge.y) <= badge.radius,
  );
  if (hit) selectFeature(hit.id);
});

async function loadFrame(index) {
  if (annotation && dirty) await persist(false);
  frameIndex = (index + frames.length) % frames.length;
  const frame = frames[frameIndex];
  setStatus('Loading…');
  document.getElementById('empty-canvas').hidden = false;
  const response = await fetch(
    `/api/benchmark/annotation/${encodeURIComponent(frame.frame_id)}`,
  );
  const payload = await response.json();
  annotation = payload.annotation;
  finalized = payload.finalized;
  dirty = false;
  setReferenceBasketSide(
    localStorage.getItem(`benchmark-guide-basket-side:${frame.frame_id}`) || 'left',
    false,
  );
  document.getElementById('annotator-id').value =
    localStorage.getItem('benchmark-annotator-id') || annotation.annotator_id || 'human-pri';
  document.getElementById('frame-notes').value = annotation.notes || '';
  activeFeatureId = benchmark.features.find((item) => !featureIsComplete(item.id))?.id ||
    benchmark.features[0].id;
  activeJunctionId = null;
  pendingGap = false;

  await new Promise((resolve, reject) => {
    image = new Image();
    image.onload = resolve;
    image.onerror = reject;
    image.src = `${frame.image_url}?sha=${frame.sha256.slice(0, 12)}`;
  });
  document.getElementById('empty-canvas').hidden = true;
  resizeCanvas();
  fitImage();
  setStatus(finalized ? 'Frame complete' : payload.source === 'draft' ? 'Draft loaded' : 'Ready',
    finalized ? 'complete' : '');
  renderAll();
}

async function finishFrame() {
  if (!annotationComplete()) return;
  const ok = await persist(true);
  if (ok) {
    showToast('Frame finalized and validated.');
    const nextIncomplete = frames.findIndex((item, index) => index > frameIndex && !item.complete);
    if (nextIncomplete >= 0) setTimeout(() => loadFrame(nextIncomplete), 450);
  }
}

document.getElementById('mode-trace').addEventListener('click', () => setMode('trace'));
document.getElementById('mode-skip').addEventListener('click', () => setMode('skip'));
document.getElementById('guide-basket-left').addEventListener(
  'click',
  () => setReferenceBasketSide('left'),
);
document.getElementById('guide-basket-right').addEventListener(
  'click',
  () => setReferenceBasketSide('right'),
);
document.getElementById('confirm-skip').addEventListener('click', () => {
  const reason = document.getElementById('skip-reason').value;
  const notes = document.getElementById('feature-notes').value;
  annotation.features = annotation.features.filter((item) => item.feature !== activeFeatureId);
  annotation.junctions = annotation.junctions.filter((junction) => {
    const catalog = catalogJunction(junction.junction);
    return !catalog.crossing.includes(activeFeatureId);
  });
  annotation.skipped = annotation.skipped.filter((item) => item.feature !== activeFeatureId);
  annotation.skipped.push({ feature: activeFeatureId, reason, notes });
  pendingGap = false;
  markDirty();
  const next = benchmark.features.find((item) => !featureIsComplete(item.id));
  if (next) activeFeatureId = next.id;
  renderAll();
});

document.getElementById('feature-visibility').addEventListener('change', (event) => {
  ensureFeature().visibility = event.target.value;
  markDirty();
});
document.getElementById('feature-uncertainty').addEventListener('change', (event) => {
  ensureFeature().uncertainty_px = Number(event.target.value);
  markDirty();
});
document.getElementById('feature-notes').addEventListener('input', (event) => {
  const feature = currentFeature();
  const skipped = annotation.skipped.find((item) => item.feature === activeFeatureId);
  if (feature) feature.notes = event.target.value;
  if (skipped) skipped.notes = event.target.value;
  markDirty();
});
document.getElementById('frame-notes').addEventListener('input', markDirty);
document.getElementById('annotator-id').addEventListener('change', (event) => {
  localStorage.setItem('benchmark-annotator-id', event.target.value.trim());
  annotation.annotator_id = event.target.value.trim();
  markDirty();
});

document.getElementById('undo-point').addEventListener('click', () => {
  const feature = currentFeature();
  if (!feature?.points.length) return;
  const removedIndex = feature.points.length - 1;
  feature.points.pop();
  feature.occluded_after = feature.occluded_after.filter((index) => index < removedIndex - 1);
  pendingGap = false;
  if (!feature.points.length) {
    annotation.features = annotation.features.filter((item) => item.feature !== activeFeatureId);
  }
  markDirty();
  renderAll();
});
document.getElementById('mark-gap').addEventListener('click', () => {
  if (!currentFeature()?.points.length) {
    showToast('Place at least one point before starting a gap.', true);
    return;
  }
  pendingGap = !pendingGap;
  renderActiveFeature();
});
document.getElementById('clear-trace').addEventListener('click', () => {
  annotation.features = annotation.features.filter((item) => item.feature !== activeFeatureId);
  annotation.junctions = annotation.junctions.filter((junction) => {
    const catalog = catalogJunction(junction.junction);
    return !catalog.crossing.includes(activeFeatureId);
  });
  pendingGap = false;
  markDirty();
  renderAll();
});

document.getElementById('continue-incomplete').addEventListener('click', (event) => {
  const featureId = event.currentTarget.dataset.feature;
  if (!featureId) return;
  selectFeature(featureId);
  document.querySelector('.active-feature-card').scrollIntoView({
    behavior: 'smooth',
    block: 'start',
  });
});
document.getElementById('clear-junction').addEventListener('click', () => {
  annotation.junctions = annotation.junctions.filter(
    (item) => item.junction !== activeJunctionId,
  );
  markDirty();
  renderAll();
});

document.getElementById('finish-frame').addEventListener('click', finishFrame);
document.getElementById('previous-frame').addEventListener('click', () => loadFrame(frameIndex - 1));
document.getElementById('next-frame').addEventListener('click', () => loadFrame(frameIndex + 1));
document.getElementById('zoom-reset').addEventListener('click', fitImage);
document.getElementById('zoom-in').addEventListener('click', () => {
  scale = Math.min(16, scale * 1.25);
  draw();
});
document.getElementById('zoom-out').addEventListener('click', () => {
  scale = Math.max(0.1, scale / 1.25);
  draw();
});

window.addEventListener('keydown', (event) => {
  const editingText = ['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement.tagName);
  if (event.code === 'Space' && !editingText) {
    event.preventDefault();
    spaceDown = true;
    canvas.classList.add('panning');
  }
  if (editingText) return;
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 's') {
    event.preventDefault();
    if (annotationComplete()) finishFrame();
    else persist(false);
  } else if (event.key === 'ArrowLeft') {
    event.preventDefault();
    loadFrame(frameIndex - 1);
  } else if (event.key === 'ArrowRight') {
    event.preventDefault();
    loadFrame(frameIndex + 1);
  } else if (event.key === 'Backspace' || event.key === 'Delete') {
    event.preventDefault();
    document.getElementById('undo-point').click();
  } else if (event.key.toLowerCase() === 'g') {
    document.getElementById('mark-gap').click();
  } else if (event.key === 'Escape') {
    activeJunctionId = null;
    renderAll();
  }
});
window.addEventListener('keyup', (event) => {
  if (event.code === 'Space') {
    spaceDown = false;
    canvas.classList.remove('panning');
  }
});
window.addEventListener('resize', () => {
  resizeCanvas();
  resizeReferenceCanvas();
});
window.addEventListener('beforeunload', (event) => {
  if (dirty) {
    event.preventDefault();
    event.returnValue = '';
  }
});

async function start() {
  try {
    const response = await fetch('/api/benchmark');
    if (!response.ok) throw new Error(await response.text());
    benchmark = await response.json();
    frames = benchmark.frames;
    resizeReferenceCanvas();
    await loadFrame(frames.findIndex((frame) => !frame.complete) >= 0
      ? frames.findIndex((frame) => !frame.complete)
      : 0);
  } catch (error) {
    setStatus('Could not load', 'error');
    document.getElementById('empty-canvas').textContent = error.message;
    showToast(error.message, true);
  }
}

start();
