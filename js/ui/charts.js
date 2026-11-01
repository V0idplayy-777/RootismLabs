import { fitCanvas, measureCanvas, showTooltip, hideTooltip } from './dom.js';

export const SERIES_COLORS = ['#e0a33e', '#4fa3a5', '#5b87b5', '#c9564c', '#6aa84f', '#8a7bb0', '#c2a06a', '#7fa8a0', '#b0765f', '#9aa3ad'];

const AXIS = '#8b9298';
const GRID = '#23272b';
const GRID_STRONG = '#2e3338';
const PANEL = '#1c1f22';

function niceStep(range, target) {
  if (!Number.isFinite(range) || range <= 0) return 1;
  const rough = range / Math.max(1, target);
  const mag = Math.pow(10, Math.floor(Math.log10(rough)));
  const norm = rough / mag;
  let step;
  if (norm <= 1) step = 1;
  else if (norm <= 2) step = 2;
  else if (norm <= 2.5) step = 2.5;
  else if (norm <= 5) step = 5;
  else step = 10;
  return step * mag;
}

function formatTick(value, span) {
  if (value === 0) return '0';
  const abs = Math.abs(value);
  if (span !== undefined && span > 0) {
    if (abs >= 1e6) return (value / 1e6).toFixed(1) + 'M';
    if (abs >= 1e4) return (value / 1e3).toFixed(0) + 'k';
    if (span >= 100) return value.toFixed(0);
    if (span >= 10) return value.toFixed(1);
    if (span >= 1) return value.toFixed(2);
    if (span >= 0.01) return value.toFixed(3);
    return value.toExponential(1);
  }
  if (abs >= 1e6) return (value / 1e6).toFixed(1) + 'M';
  if (abs >= 1e4) return (value / 1e3).toFixed(1) + 'k';
  if (abs >= 100) return value.toFixed(0);
  if (abs >= 1) return value.toFixed(2);
  return value.toPrecision(2);
}

export function chartFrame(canvas, opts) {
  const options = opts || {};
  const size = measureCanvas(canvas);
  const ctx = fitCanvas(canvas, options.width || size.width, options.height || size.height);
  const width = options.width || size.width;
  const height = options.height || size.height;
  ctx.clearRect(0, 0, width, height);
  if (options.background !== false) {
    ctx.fillStyle = options.background || PANEL;
    ctx.fillRect(0, 0, width, height);
  }
  const pad = Object.assign({ top: 10, right: 12, bottom: 22, left: 44 }, options.pad);
  const plot = { x: pad.left, y: pad.top, w: Math.max(10, width - pad.left - pad.right), h: Math.max(10, height - pad.top - pad.bottom) };
  return { ctx, width, height, plot, pad, options };
}

function drawAxes(frame, xDomain, yDomain, opts) {
  const { ctx, plot } = frame;
  const options = opts || {};
  ctx.save();
  ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
  ctx.textBaseline = 'middle';

  const yTicks = options.yTicks || 5;
  const yStep = niceStep(yDomain[1] - yDomain[0], yTicks);
  const yStart = Math.ceil(yDomain[0] / yStep) * yStep;
  ctx.strokeStyle = GRID;
  ctx.fillStyle = AXIS;
  ctx.lineWidth = 1;
  ctx.textAlign = 'right';
  for (let v = yStart; v <= yDomain[1] + yStep * 0.001; v += yStep) {
    const y = plot.y + plot.h - ((v - yDomain[0]) / (yDomain[1] - yDomain[0] || 1)) * plot.h;
    if (y < plot.y - 1 || y > plot.y + plot.h + 1) continue;
    ctx.beginPath();
    ctx.moveTo(plot.x, Math.round(y) + 0.5);
    ctx.lineTo(plot.x + plot.w, Math.round(y) + 0.5);
    ctx.stroke();
    ctx.fillText(formatTick(v, yDomain[1] - yDomain[0]), plot.x - 5, y);
  }

  const xCount = options.xTicks || Math.max(2, Math.min(8, Math.floor(plot.w / 62)));
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  if (options.xLabels) {
    const labels = options.xLabels;
    const every = Math.max(1, Math.ceil(labels.length / xCount));
    for (let i = 0; i < labels.length; i += every) {
      const x = plot.x + ((i - xDomain[0]) / (xDomain[1] - xDomain[0] || 1)) * plot.w;
      if (x < plot.x - 1 || x > plot.x + plot.w + 1) continue;
      ctx.strokeStyle = GRID;
      ctx.beginPath();
      ctx.moveTo(Math.round(x) + 0.5, plot.y);
      ctx.lineTo(Math.round(x) + 0.5, plot.y + plot.h);
      ctx.stroke();
      ctx.fillStyle = AXIS;
      ctx.fillText(String(labels[i]), x, plot.y + plot.h + 5);
    }
  } else {
    const xStep = niceStep(xDomain[1] - xDomain[0], xCount);
    const xStart = Math.ceil(xDomain[0] / xStep) * xStep;
    for (let v = xStart; v <= xDomain[1] + xStep * 0.001; v += xStep) {
      const x = plot.x + ((v - xDomain[0]) / (xDomain[1] - xDomain[0] || 1)) * plot.w;
      if (x < plot.x - 1 || x > plot.x + plot.w + 1) continue;
      ctx.strokeStyle = GRID;
      ctx.beginPath();
      ctx.moveTo(Math.round(x) + 0.5, plot.y);
      ctx.lineTo(Math.round(x) + 0.5, plot.y + plot.h);
      ctx.stroke();
      ctx.fillStyle = AXIS;
      ctx.fillText(formatTick(v, xDomain[1] - xDomain[0]), x, plot.y + plot.h + 5);
    }
  }

  ctx.strokeStyle = GRID_STRONG;
  ctx.beginPath();
  ctx.moveTo(plot.x + 0.5, plot.y);
  ctx.lineTo(plot.x + 0.5, plot.y + plot.h + 0.5);
  ctx.lineTo(plot.x + plot.w, plot.y + plot.h + 0.5);
  ctx.stroke();

  if (options.zeroLine && yDomain[0] < 0 && yDomain[1] > 0) {
    const y = plot.y + plot.h - ((0 - yDomain[0]) / (yDomain[1] - yDomain[0])) * plot.h;
    ctx.strokeStyle = '#3b4249';
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(plot.x, y);
    ctx.lineTo(plot.x + plot.w, y);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  if (options.xTitle) {
    ctx.fillStyle = AXIS;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'bottom';
    ctx.fillText(options.xTitle, plot.x + plot.w, frame.height - 1);
  }
  if (options.yTitle) {
    ctx.save();
    ctx.translate(9, plot.y + plot.h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = AXIS;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(options.yTitle, 0, 0);
    ctx.restore();
  }
  ctx.restore();
}

function drawLegend(frame, series, opts) {
  if (!series || series.length < 1) return;
  if (opts && opts.legend === false) return;
  const { ctx, plot } = frame;
  ctx.save();
  ctx.font = '10.5px ui-monospace, SFMono-Regular, Menlo, monospace';
  ctx.textBaseline = 'middle';
  let x = plot.x + 6;
  const y = plot.y + 8;
  for (const s of series) {
    if (!s.label) continue;
    ctx.fillStyle = s.color || SERIES_COLORS[0];
    ctx.fillRect(x, y - 1, 9, 2);
    if (s.dash) {
      ctx.globalAlpha = 0.55;
      ctx.fillRect(x, y - 3, 9, 6);
      ctx.globalAlpha = 1;
    }
    ctx.fillStyle = '#a8aeb4';
    ctx.textAlign = 'left';
    ctx.fillText(s.label, x + 13, y);
    x += 13 + ctx.measureText(s.label).width + 14;
    if (x > plot.x + plot.w - 20) break;
  }
  ctx.restore();
}

function domainOf(series, opts) {
  const options = opts || {};
  let xMin = Infinity;
  let xMax = -Infinity;
  let yMin = Infinity;
  let yMax = -Infinity;
  for (const s of series) {
    const xs = s.x;
    const ys = s.y;
    for (let i = 0; i < ys.length; i++) {
      const xv = xs ? xs[i] : i;
      const yv = ys[i];
      if (!Number.isFinite(yv)) continue;
      if (xv < xMin) xMin = xv;
      if (xv > xMax) xMax = xv;
      if (yv < yMin) yMin = yv;
      if (yv > yMax) yMax = yv;
    }
  }
  if (!Number.isFinite(xMin)) { xMin = 0; xMax = 1; }
  if (!Number.isFinite(yMin)) { yMin = 0; yMax = 1; }
  if (options.yMin !== undefined) yMin = options.yMin;
  if (options.yMax !== undefined) yMax = options.yMax;
  if (options.xMin !== undefined) xMin = options.xMin;
  if (options.xMax !== undefined) xMax = options.xMax;
  if (yMin === yMax) { yMin -= 0.5; yMax += 0.5; }
  if (xMin === xMax) { xMin -= 0.5; xMax += 0.5; }
  if (options.padY !== false) {
    const span = yMax - yMin;
    yMin -= span * 0.06;
    yMax += span * 0.08;
  }
  return [[xMin, xMax], [yMin, yMax]];
}

export function lineChart(canvas, series, opts) {
  const frame = renderLine(canvas, series, opts);
  canvas._lastDraw = () => renderLine(canvas, series, opts);
  return frame;
}

function renderLine(canvas, series, opts) {
  const options = opts || {};
  const list = (series || []).filter((s) => s && s.y && s.y.length);
  const frame = chartFrame(canvas, options);
  if (!list.length) {
    emptyChart(frame, options.emptyText || 'no data yet');
    return frame;
  }
  const domains = domainOf(list, options);
  drawAxes(frame, domains[0], domains[1], options);
  const { ctx, plot } = frame;
  const [xDomain, yDomain] = domains;
  const toX = (v, i, s) => plot.x + (((s.x ? s.x[i] : i) - xDomain[0]) / (xDomain[1] - xDomain[0])) * plot.w;
  const toY = (v) => plot.y + plot.h - ((v - yDomain[0]) / (yDomain[1] - yDomain[0])) * plot.h;

  ctx.save();
  ctx.beginPath();
  ctx.rect(plot.x, plot.y, plot.w, plot.h);
  ctx.clip();

  for (const band of options.bands || []) {
    ctx.fillStyle = band.color || 'rgba(224,163,62,0.07)';
    const x1 = plot.x + ((band.x0 - xDomain[0]) / (xDomain[1] - xDomain[0])) * plot.w;
    const x2 = plot.x + ((band.x1 - xDomain[0]) / (xDomain[1] - xDomain[0])) * plot.w;
    ctx.fillRect(Math.min(x1, x2), plot.y, Math.abs(x2 - x1), plot.h);
  }

  list.forEach((s, index) => {
    const color = s.color || SERIES_COLORS[index % SERIES_COLORS.length];
    ctx.strokeStyle = color;
    ctx.lineWidth = s.width || 1.5;
    ctx.lineJoin = 'round';
    ctx.globalAlpha = s.alpha === undefined ? 1 : s.alpha;
    if (s.dash) ctx.setLineDash(s.dash);
    if (s.fill) {
      ctx.beginPath();
      let started = false;
      let lastX = plot.x;
      for (let i = 0; i < s.y.length; i++) {
        const v = s.y[i];
        if (!Number.isFinite(v)) continue;
        const x = toX(v, i, s);
        const y = toY(v);
        if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
        lastX = x;
      }
      if (started) {
        const baseY = toY(Math.max(yDomain[0], options.fillBase === undefined ? yDomain[0] : options.fillBase));
        ctx.lineTo(lastX, baseY);
        ctx.lineTo(plot.x, baseY);
        ctx.closePath();
        ctx.fillStyle = s.fill;
        ctx.globalAlpha = 0.16;
        ctx.fill();
        ctx.globalAlpha = s.alpha === undefined ? 1 : s.alpha;
      }
    }
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < s.y.length; i++) {
      const v = s.y[i];
      if (!Number.isFinite(v)) { started = false; continue; }
      const x = toX(v, i, s);
      const y = toY(v);
      if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;

    const points = s.y.length;
    if (options.points !== false && points <= 60) {
      ctx.fillStyle = color;
      for (let i = 0; i < points; i++) {
        const v = s.y[i];
        if (!Number.isFinite(v)) continue;
        ctx.beginPath();
        ctx.arc(toX(v, i, s), toY(v), points <= 24 ? 2.1 : 1.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    if (s.marker !== undefined && Number.isFinite(s.marker)) {
      const x = plot.x + ((s.marker - xDomain[0]) / (xDomain[1] - xDomain[0])) * plot.w;
      ctx.strokeStyle = color;
      ctx.setLineDash([2, 3]);
      ctx.globalAlpha = 0.65;
      ctx.beginPath();
      ctx.moveTo(x, plot.y);
      ctx.lineTo(x, plot.y + plot.h);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    }
  });
  ctx.restore();

  if (options.threshold !== undefined) {
    const y = toY(options.threshold);
    if (y >= plot.y && y <= plot.y + plot.h) {
      ctx.strokeStyle = options.thresholdColor || '#6d4a45';
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(plot.x, y);
      ctx.lineTo(plot.x + plot.w, y);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  drawLegend(frame, list.map((s, i) => ({ label: s.label, color: s.color || SERIES_COLORS[i % SERIES_COLORS.length], dash: s.dash })), options);

  if (options.hover !== false) {
    frame.ctx = ctx;
    frame.plot = plot;
    canvas._hoverFrame = frame;
    attachLineHover(canvas, frame, list, xDomain, yDomain, options, toX, toY);
  }
  return frame;
}

function attachLineHover(canvas, frame, series, xDomain, yDomain, options, toX, toY) {
  if (canvas._hoverBound) return;
  canvas._hoverBound = true;
  canvas.addEventListener('mousemove', (event) => {
    const snapshot = canvas._hoverFrame;
    if (!snapshot) return;
    const rect = canvas.getBoundingClientRect();
    const mx = event.clientX - rect.left;
    const plot = snapshot.plot;
    if (mx < plot.x || mx > plot.x + plot.w) { hideTooltip(); return; }
    const xv = xDomain[0] + ((mx - plot.x) / plot.w) * (xDomain[1] - xDomain[0]);
    let best = 0;
    let bestDist = Infinity;
    const primary = series[0];
    for (let i = 0; i < primary.y.length; i++) {
      const px = primary.x ? primary.x[i] : i;
      const d = Math.abs(px - xv);
      if (d < bestDist) { bestDist = d; best = i; }
    }
    const lines = [];
    const xLabel = options.xLabel ? options.xLabel(primary.x ? primary.x[best] : best) : String(primary.x ? primary.x[best] : best);
    lines.push('<b>' + (options.xTitle || 'x') + ' ' + xLabel + '</b>');
    for (const s of series) {
      const v = s.y[best];
      if (!Number.isFinite(v)) continue;
      lines.push('<span style="color:' + (s.color || SERIES_COLORS[0]) + '">' + (s.label || 'series') + '</span> ' + (options.valueFormat ? options.valueFormat(v) : formatTick(v, yDomain[1] - yDomain[0])));
    }
    showTooltip(lines.join('<br>'), event.clientX, event.clientY);
    snapshot();
    const ctx = snapshot.ctx;
    const px = toX(primary.y[best], best, primary);
    ctx.save();
    ctx.strokeStyle = '#5a6169';
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(px, plot.y);
    ctx.lineTo(px, plot.y + plot.h);
    ctx.stroke();
    ctx.setLineDash([]);
    series.forEach((s, index) => {
      const v = s.y[best];
      if (!Number.isFinite(v)) return;
      ctx.fillStyle = s.color || SERIES_COLORS[index % SERIES_COLORS.length];
      ctx.beginPath();
      ctx.arc(toX(v, best, s), toY(v), 2.6, 0, Math.PI * 2);
      ctx.fill();
    });
    ctx.restore();
  });
  canvas.addEventListener('mouseleave', () => {
    hideTooltip();
    if (canvas._lastDraw) canvas._lastDraw();
  });
}

export function emptyChart(frame, message) {
  const { ctx, plot } = frame;
  ctx.save();
  ctx.strokeStyle = GRID;
  ctx.strokeRect(plot.x + 0.5, plot.y + 0.5, plot.w, plot.h);
  ctx.fillStyle = '#5b6268';
  ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(message || 'no data', plot.x + plot.w / 2, plot.y + plot.h / 2);
  ctx.restore();
}

export function barChart(canvas, bars, opts) {
  const options = opts || {};
  const frame = chartFrame(canvas, options);
  const list = bars || [];
  if (!list.length) { emptyChart(frame, options.emptyText || 'no data'); return frame; }
  const horizontal = options.horizontal !== false;
  const { ctx, plot } = frame;
  let max = -Infinity;
  let min = Infinity;
  for (const b of list) {
    if (b.value > max) max = b.value;
    if (b.value < min) min = b.value;
  }
  if (options.max !== undefined) max = options.max;
  if (options.min !== undefined) min = options.min;
  if (min > 0 && !options.keepMin) min = 0;
  if (max === min) max = min + 1;

  ctx.save();
  ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
  const slot = horizontal ? plot.h / list.length : plot.w / list.length;
  const barSize = Math.max(2, Math.min(slot * 0.66, horizontal ? 16 : 42));
  const scale = (v) => ((v - min) / (max - min));

  list.forEach((b, i) => {
    const color = b.color || (b.tone === 'bad' ? '#c9564c' : b.tone === 'warn' ? '#e0a33e' : b.tone === 'good' ? '#6aa84f' : SERIES_COLORS[i % SERIES_COLORS.length]);
    const frac = scale(Math.max(min, Math.min(max, b.value)));
    if (horizontal) {
      const y = plot.y + i * slot + (slot - barSize) / 2;
      const zeroX = plot.x + scale(Math.max(min, 0)) * plot.w;
      const w = Math.max(0, frac * plot.w - (scale(Math.max(min, 0)) * plot.w));
      ctx.fillStyle = '#191c1f';
      ctx.fillRect(plot.x, y, plot.w, barSize);
      ctx.fillStyle = color;
      ctx.fillRect(zeroX, y, Math.max(0, w), barSize);
      ctx.fillStyle = AXIS;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      const label = b.label === undefined ? String(i) : String(b.label);
      const shown = label.length > (options.labelChars || 16) ? label.slice(0, (options.labelChars || 16) - 1) + '…' : label;
      ctx.fillText(shown, plot.x - 5, y + barSize / 2);
      if (options.values !== false) {
        ctx.textAlign = 'left';
        ctx.fillStyle = '#a8aeb4';
        ctx.fillText(b.display === undefined ? formatTick(b.value, max - min) : String(b.display), zeroX + Math.max(0, w) + 5, y + barSize / 2);
      }
    } else {
      const x = plot.x + i * slot + (slot - barSize) / 2;
      const baseY = plot.y + plot.h - scale(Math.max(min, 0)) * plot.h;
      const h = Math.max(0, frac * plot.h - scale(Math.max(min, 0)) * plot.h);
      ctx.fillStyle = '#191c1f';
      ctx.fillRect(x, plot.y, barSize, plot.h);
      ctx.fillStyle = color;
      ctx.fillRect(x, baseY - h, barSize, h);
      ctx.fillStyle = AXIS;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      const label = b.label === undefined ? String(i) : String(b.label);
      const shown = label.length > (options.labelChars || 10) ? label.slice(0, (options.labelChars || 10) - 1) + '…' : label;
      ctx.fillText(shown, x + barSize / 2, plot.y + plot.h + 4);
    }
  });

  if (horizontal) {
    const step = niceStep(max - min, 4);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.strokeStyle = GRID;
    for (let v = Math.ceil(min / step) * step; v <= max + step * 0.001; v += step) {
      const x = plot.x + scale(v) * plot.w;
      ctx.beginPath();
      ctx.moveTo(x, plot.y);
      ctx.lineTo(x, plot.y + plot.h);
      ctx.stroke();
      ctx.fillStyle = AXIS;
      ctx.fillText(formatTick(v, max - min), x, plot.y + plot.h + 4);
    }
  }
  ctx.restore();
  return frame;
}

export function histogram(canvas, values, opts) {
  const options = opts || {};
  const frame = chartFrame(canvas, options);
  const data = (values || []).filter((v) => Number.isFinite(v));
  if (!data.length) { emptyChart(frame, options.emptyText || 'no data'); return frame; }
  let min = data[0];
  let max = data[0];
  for (const v of data) { if (v < min) min = v; if (v > max) max = v; }
  if (min === max) { min -= 0.5; max += 0.5; }
  const bins = options.bins || Math.max(6, Math.min(60, Math.round(Math.sqrt(data.length) * 1.6)));
  const counts = new Array(bins).fill(0);
  const sums = new Array(bins).fill(0);
  for (const v of data) {
    let index = Math.floor(((v - min) / (max - min)) * bins);
    if (index >= bins) index = bins - 1;
    if (index < 0) index = 0;
    counts[index]++;
    sums[index] += v;
  }
  const { ctx, plot } = frame;
  let peak = 1;
  for (const c of counts) if (c > peak) peak = c;
  const slot = plot.w / bins;
  ctx.save();
  ctx.strokeStyle = GRID;
  ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
  const yStep = niceStep(peak, 4);
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let v = 0; v <= peak + yStep * 0.001; v += yStep) {
    const y = plot.y + plot.h - (v / peak) * plot.h;
    ctx.fillStyle = AXIS;
    ctx.fillText(String(Math.round(v)), plot.x - 5, y);
    ctx.beginPath();
    ctx.moveTo(plot.x, y);
    ctx.lineTo(plot.x + plot.w, y);
    ctx.stroke();
  }
  counts.forEach((c, i) => {
    const h = (c / peak) * plot.h;
    const x = plot.x + i * slot;
    ctx.fillStyle = options.color || '#4fa3a5';
    ctx.globalAlpha = 0.85;
    ctx.fillRect(x + 0.5, plot.y + plot.h - h, Math.max(1, slot - 1), h);
    ctx.globalAlpha = 1;
  });
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillStyle = AXIS;
  ctx.fillText(formatTick(min, max - min), plot.x, plot.y + plot.h + 4);
  ctx.fillText(formatTick(max, max - min), plot.x + plot.w, plot.y + plot.h + 4);
  ctx.restore();
  frame.stats = { n: data.length, min, max, bins, counts, means: sums.map((s, i) => (counts[i] ? s / counts[i] : null)) };
  return frame;
}

export function confusionChart(canvas, matrix, labels, opts) {
  const options = opts || {};
  const frame = chartFrame(canvas, options);
  const n = labels.length;
  if (!n || !matrix || !matrix.length) { emptyChart(frame, 'no confusion data'); return frame; }
  const { ctx, plot } = frame;
  const pad = options.labelPad === undefined ? Math.min(96, Math.max(38, Math.ceil(plot.w * 0.16))) : options.labelPad;
  const gridX = plot.x + pad;
  const gridY = plot.y;
  const size = Math.min(plot.w - pad, plot.h - 16);
  const cell = size / n;
  let max = 1;
  const normalized = options.normalize !== false;
  const rows = [];
  for (let i = 0; i < n; i++) {
    const total = matrix[i].reduce((a, b) => a + b, 0);
    rows.push(total);
    for (let j = 0; j < n; j++) {
      const v = normalized && total ? matrix[i][j] / total : matrix[i][j];
      if (v > max) max = v;
    }
  }
  ctx.save();
  ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const raw = matrix[i][j];
      const v = normalized && rows[i] ? raw / rows[i] : raw;
      const t = max ? v / max : 0;
      const x = gridX + j * cell;
      const y = gridY + i * cell;
      if (i === j) {
        ctx.fillStyle = 'rgba(79,163,165,' + (0.12 + 0.72 * t).toFixed(3) + ')';
      } else {
        ctx.fillStyle = 'rgba(201,86,76,' + (0.10 + 0.74 * t).toFixed(3) + ')';
      }
      ctx.fillRect(x, y, cell - 1, cell - 1);
      if (cell >= 20 || (i === j && cell >= 14)) {
        ctx.fillStyle = t > 0.55 ? '#14171a' : '#b7bcc1';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(normalized ? (v * 100).toFixed(0) : String(raw), x + (cell - 1) / 2, y + (cell - 1) / 2);
      }
    }
  }
  ctx.fillStyle = AXIS;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let i = 0; i < n; i++) {
    const label = String(labels[i]);
    const shown = label.length > 15 ? label.slice(0, 14) + '…' : label;
    ctx.fillText(shown, gridX - 5, gridY + i * cell + cell / 2);
  }
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (let j = 0; j < n; j++) {
    const label = String(labels[j]);
    const shown = label.length > 7 ? label.slice(0, 6) + '…' : label;
    ctx.save();
    ctx.translate(gridX + j * cell + cell / 2, gridY + n * cell + 3);
    ctx.rotate(-Math.PI / 5);
    ctx.textAlign = 'right';
    ctx.fillText(shown, 0, 0);
    ctx.restore();
  }
  ctx.restore();

  if (options.hover !== false) {
    const handler = (event) => {
      const rect = canvas.getBoundingClientRect();
      const mx = event.clientX - rect.left;
      const my = event.clientY - rect.top;
      const j = Math.floor((mx - gridX) / cell);
      const i = Math.floor((my - gridY) / cell);
      if (i < 0 || j < 0 || i >= n || j >= n) { hideTooltip(); return; }
      const total = rows[i] || 1;
      showTooltip('<b>true ' + labels[i] + ' → predicted ' + labels[j] + '</b><br>' + matrix[i][j] + ' of ' + total + ' (' + (100 * matrix[i][j] / total).toFixed(1) + '% of the class)', event.clientX, event.clientY);
    };
    canvas.addEventListener('mousemove', handler);
    canvas.addEventListener('mouseleave', hideTooltip);
  }
  return frame;
}

export function curveChart(canvas, points, opts) {
  const options = opts || {};
  const frame = chartFrame(canvas, Object.assign({ xMin: 0, xMax: 1, yMin: 0, yMax: 1, padY: false }, options));
  const { ctx, plot } = frame;
  ctx.save();
  ctx.strokeStyle = GRID;
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(plot.x, plot.y + plot.h);
  ctx.lineTo(plot.x + plot.w, plot.y);
  ctx.stroke();
  ctx.setLineDash([]);
  if (points && points.length) {
    ctx.strokeStyle = options.color || SERIES_COLORS[0];
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    points.forEach((p, i) => {
      const x = plot.x + p.x * plot.w;
      const y = plot.y + plot.h - p.y * plot.h;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    if (options.fill) {
      ctx.lineTo(plot.x + plot.w, plot.y + plot.h);
      ctx.lineTo(plot.x, plot.y + plot.h);
      ctx.closePath();
      ctx.fillStyle = options.fill;
      ctx.globalAlpha = 0.14;
      ctx.fill();
      ctx.globalAlpha = 1;
    }
  }
  ctx.restore();
  drawAxes(frame, [0, 1], [0, 1], Object.assign({ xTicks: 5, yTicks: 5 }, options, { xTitle: options.xTitle, yTitle: options.yTitle }));
  if (options.annotation) {
    ctx.save();
    ctx.font = '10.5px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillStyle = '#a8aeb4';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'bottom';
    ctx.fillText(options.annotation, plot.x + plot.w - 4, plot.y + plot.h - 4);
    ctx.restore();
  }
  return frame;
}

export function calibrationChart(canvas, report, opts) {
  const options = opts || {};
  const frame = chartFrame(canvas, Object.assign({ xMin: 0, xMax: 1, yMin: 0, yMax: 1, padY: false }, options));
  const { ctx, plot } = frame;
  const bins = (report && report.bins) || [];
  ctx.save();
  ctx.strokeStyle = GRID;
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(plot.x, plot.y + plot.h);
  ctx.lineTo(plot.x + plot.w, plot.y);
  ctx.stroke();
  ctx.setLineDash([]);
  let maxCount = 1;
  for (const b of bins) if (b.count > maxCount) maxCount = b.count;
  for (const b of bins) {
    const x0 = plot.x + b.lo * plot.w;
    const x1 = plot.x + b.hi * plot.w;
    const w = Math.max(2, x1 - x0 - 1);
    const h = (b.count / maxCount) * (plot.h * 0.55);
    ctx.fillStyle = 'rgba(91,135,181,0.30)';
    ctx.fillRect(x0, plot.y + plot.h - h, w, h);
    if (b.count) {
      const px = b.confidence === undefined ? (b.lo + b.hi) / 2 : b.confidence;
      const py = b.accuracy === undefined ? px : b.accuracy;
      const cx = plot.x + px * plot.w;
      const cy = plot.y + plot.h - py * plot.h;
      ctx.strokeStyle = options.color || SERIES_COLORS[0];
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(x0 + w / 2, plot.y + plot.h);
      ctx.lineTo(cx, cy);
      ctx.stroke();
      ctx.fillStyle = options.color || SERIES_COLORS[0];
      ctx.beginPath();
      ctx.arc(cx, cy, 3, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#16181a';
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }
  ctx.restore();
  drawAxes(frame, [0, 1], [0, 1], Object.assign({ xTicks: 5, yTicks: 5 }, options, { xTitle: options.xTitle || 'mean predicted confidence', yTitle: options.yTitle || 'observed accuracy' }));
  ctx.save();
  ctx.font = '9.5px ui-monospace, SFMono-Regular, Menlo, monospace';
  ctx.fillStyle = '#5f676e';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillText('bars: samples per bin', plot.x + 4, plot.y + 2);
  ctx.restore();
  return frame;
}

export function sparkline(canvas, values, opts) {
  const options = opts || {};
  const frame = chartFrame(canvas, Object.assign({ background: false, pad: { top: 2, right: 2, bottom: 2, left: 2 } }, options));
  const { ctx, plot } = frame;
  const data = (values || []).filter((v) => Number.isFinite(v));
  if (data.length < 2) return frame;
  let min = data[0];
  let max = data[0];
  for (const v of data) { if (v < min) min = v; if (v > max) max = v; }
  if (min === max) { min -= 1; max += 1; }
  ctx.save();
  ctx.strokeStyle = options.color || SERIES_COLORS[0];
  ctx.lineWidth = options.width || 1.2;
  ctx.beginPath();
  data.forEach((v, i) => {
    const x = plot.x + (i / (data.length - 1)) * plot.w;
    const y = plot.y + plot.h - ((v - min) / (max - min)) * plot.h;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.restore();
  return frame;
}

export function heatmap(canvas, grid, opts) {
  const options = opts || {};
  const frame = chartFrame(canvas, options);
  const rows = grid ? grid.length : 0;
  const cols = rows ? grid[0].length : 0;
  if (!rows || !cols) { emptyChart(frame, options.emptyText || 'no data'); return frame; }
  const { ctx, plot } = frame;
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
    const v = grid[i][j];
    if (!Number.isFinite(v)) continue;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (!Number.isFinite(min)) { min = 0; max = 1; }
  if (min === max) max = min + 1;
  const cw = plot.w / cols;
  const ch = plot.h / rows;
  ctx.save();
  const ramp = options.ramp || 'magma';
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      const v = grid[i][j];
      const t = Number.isFinite(v) ? (v - min) / (max - min) : 0;
      ctx.fillStyle = rampColor(ramp, t);
      ctx.fillRect(plot.x + j * cw, plot.y + i * ch, Math.ceil(cw), Math.ceil(ch));
    }
  }
  ctx.restore();
  return frame;
}

export function rampColor(name, t) {
  const v = Math.max(0, Math.min(1, t));
  const stops = name === 'viridis'
    ? [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]]
    : name === 'cool'
      ? [[18, 22, 26], [35, 74, 86], [64, 132, 140], [146, 197, 194], [232, 244, 240]]
      : [[18, 20, 24], [70, 44, 60], [150, 74, 52], [222, 132, 52], [248, 214, 130]];
  const seg = v * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(seg));
  const f = seg - i;
  const a = stops[i];
  const b = stops[i + 1];
  return 'rgb(' + Math.round(a[0] + (b[0] - a[0]) * f) + ',' + Math.round(a[1] + (b[1] - a[1]) * f) + ',' + Math.round(a[2] + (b[2] - a[2]) * f) + ')';
}

export function waveform(canvas, samples, opts) {
  const options = opts || {};
  const frame = chartFrame(canvas, Object.assign({ background: '#14171a', pad: { top: 4, right: 4, bottom: 4, left: 4 } }, options));
  const { ctx, plot } = frame;
  const data = samples || [];
  if (!data.length) { emptyChart(frame, 'no audio'); return frame; }
  ctx.save();
  ctx.strokeStyle = options.color || '#4fa3a5';
  ctx.lineWidth = 1;
  const step = Math.max(1, Math.floor(data.length / Math.max(1, plot.w)));
  ctx.beginPath();
  for (let x = 0; x < plot.w; x++) {
    const start = Math.floor((x / plot.w) * data.length);
    let lo = 1;
    let hi = -1;
    for (let k = 0; k < step && start + k < data.length; k++) {
      const v = data[start + k];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    const mid = plot.y + plot.h / 2;
    const amp = plot.h / 2 - 1;
    ctx.moveTo(plot.x + x + 0.5, mid - hi * amp);
    ctx.lineTo(plot.x + x + 0.5, mid - lo * amp);
  }
  ctx.stroke();
  ctx.strokeStyle = '#2b3035';
  ctx.beginPath();
  ctx.moveTo(plot.x, plot.y + plot.h / 2);
  ctx.lineTo(plot.x + plot.w, plot.y + plot.h / 2);
  ctx.stroke();
  ctx.restore();
  return frame;
}

export function imageToCanvas(canvas, image, opts) {
  const options = opts || {};
  const channels = options.channels || 3;
  const height = options.height;
  const width = options.width;
  const data = image;
  if (!data || !width || !height) {
    const frame = chartFrame(canvas, options);
    emptyChart(frame, 'no image');
    return;
  }
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  canvas.style.width = (options.displayWidth || width) + 'px';
  canvas.style.height = (options.displayHeight || height) + 'px';
  canvas.style.imageRendering = options.smooth ? 'auto' : 'pixelated';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  const out = ctx.createImageData(width, height);
  let min = Infinity;
  let max = -Infinity;
  if (options.autoScale !== false) {
    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      if (!Number.isFinite(v)) continue;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    if (!Number.isFinite(min)) { min = 0; max = 1; }
    if (min === max) max = min + 1;
  } else { min = options.min === undefined ? 0 : options.min; max = options.max === undefined ? 1 : options.max; }
  const span = max - min || 1;
  const plane = width * height;
  for (let p = 0; p < plane; p++) {
    let r;
    let g;
    let b;
    if (channels >= 3) {
      r = data[p];
      g = data[plane + p];
      b = data[2 * plane + p];
    } else {
      r = data[p];
      g = data[p];
      b = data[p];
    }
    const o = p * 4;
    out.data[o] = clampByte(((r - min) / span) * 255);
    out.data[o + 1] = clampByte(((g - min) / span) * 255);
    out.data[o + 2] = clampByte(((b - min) / span) * 255);
    out.data[o + 3] = 255;
  }
  ctx.putImageData(out, 0, 0);
}

function clampByte(v) {
  if (!Number.isFinite(v)) return 0;
  return Math.max(0, Math.min(255, Math.round(v)));
}

export function drawBudgetBar(node, info) {
  node.textContent = '';
  const bar = document.createElement('div');
  bar.className = 'bar';
  const fill = document.createElement('i');
  const target = document.createElement('u');
  bar.appendChild(fill);
  bar.appendChild(target);
  const lo = info.lo;
  const hi = info.hi;
  const span = Math.max(1, hi - lo);
  const frac = Math.max(0, Math.min(1, (info.value - lo) / span));
  fill.style.width = (frac * 100).toFixed(2) + '%';
  fill.style.background = info.tone === 'over' ? '#c9564c' : info.tone === 'under' ? '#5b87b5' : '#4fa3a5';
  if (info.marker !== undefined && info.marker !== null) {
    const mfrac = Math.max(0, Math.min(1, (info.marker - lo) / span));
    target.style.left = (mfrac * 100).toFixed(2) + '%';
  } else target.style.display = 'none';
  node.appendChild(bar);
  return node;
}
