import { LAYERS, LAYER_GROUPS } from '../core/layers.js';
import { analyze, compactNumber } from '../core/model.js';
import { fitCanvas, measureCanvas } from './dom.js';

const NODE_W = 172;
const NODE_HEAD = 20;
const NODE_ROW = 12;
const PORT_R = 4.2;

const GROUP_COLOR = {
  input: '#8b9298',
  core: '#e0a33e',
  conv: '#4fa3a5',
  pool: '#5b87b5',
  sequence: '#8a7bb0',
  shape: '#c2a06a',
  regularize: '#7d8a6a'
};

function groupOf(type) {
  const layer = LAYERS[type];
  if (!layer) return type === 'input' ? 'input' : 'core';
  return layer.group || 'core';
}

function colorOf(type) {
  return GROUP_COLOR[groupOf(type)] || '#8b9298';
}

export function groupColor(group) {
  return GROUP_COLOR[group] || '#8b9298';
}

export function groupColors() {
  return LAYER_GROUPS.map((g) => ({ key: g.key, label: g.label, color: GROUP_COLOR[g.key] || '#8b9298' }));
}

function nodeHeight(node, info) {
  const message = info && info.errorByNode ? info.errorByNode[node.id] : null;
  return NODE_HEAD + 2 * NODE_ROW + 14 + (message ? 13 : 0);
}

export class GraphEditor {
  constructor(options) {
    this.canvas = options.canvas;
    this.spec = options.spec || null;
    this.info = options.info || null;
    this.onSelect = options.onSelect || null;
    this.onChange = options.onChange || null;
    this.selectedId = options.selectedId || null;
    this.view = { x: 0, y: 0, scale: 1 };
    this.drag = null;
    this.link = null;
    this.hoverPort = null;
    this.hoverNode = null;
    this.contextMenu = options.contextMenu || null;
    this.dirty = false;
    this.boxes = [];
    this._bind();
  }

  setSpec(spec, info) {
    this.spec = spec;
    this.info = info || (spec ? this.analyze() : null);
    if (this.selectedId && spec && !spec.nodes.some((n) => n.id === this.selectedId)) this.selectedId = null;
    this.render();
  }

  analyze() {
    if (!this.spec) return null;
    const result = analyze(this.spec);
    const byNode = {};
    for (const stat of result.nodeStats || []) byNode[stat.id] = stat;
    const errors = {};
    for (const err of result.errors || []) errors[err.id] = err.message;
    result.byNode = byNode;
    result.errorByNode = errors;
    return result;
  }

  _bind() {
    const canvas = this.canvas;
    canvas.addEventListener('pointerdown', (event) => this._down(event));
    canvas.addEventListener('pointermove', (event) => this._move(event));
    canvas.addEventListener('pointerup', (event) => this._up(event));
    canvas.addEventListener('pointerleave', () => { this.hoverPort = null; this.hoverNode = null; });
    canvas.addEventListener('wheel', (event) => {
      event.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const mx = event.clientX - rect.left;
      const my = event.clientY - rect.top;
      const before = this.toWorld(mx, my);
      const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
      this.view.scale = Math.max(0.25, Math.min(2.6, this.view.scale * factor));
      const after = this.toWorld(mx, my);
      this.view.x += (after.x - before.x) * this.view.scale;
      this.view.y += (after.y - before.y) * this.view.scale;
      this.render();
    }, { passive: false });
    canvas.addEventListener('contextmenu', (event) => {
      event.preventDefault();
      const hit = this._hit(event);
      if (hit && hit.kind === 'node') {
        this.select(hit.node.id);
        if (this.contextMenu) this.contextMenu(event.clientX, event.clientY, hit.node);
      } else if (this.contextMenu) {
        this.contextMenu(event.clientX, event.clientY, null);
      }
    });
    canvas.addEventListener('dblclick', (event) => {
      const hit = this._hit(event);
      if (hit && hit.kind === 'node' && this.onInspect) this.onInspect(hit.node);
    });
  }

  toWorld(sx, sy) {
    return { x: (sx - this.view.x) / this.view.scale, y: (sy - this.view.y) / this.view.scale };
  }

  toScreen(wx, wy) {
    return { x: wx * this.view.scale + this.view.x, y: wy * this.view.scale + this.view.y };
  }

  _eventPoint(event) {
    const rect = this.canvas.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  }

  _boxes() {
    if (!this.spec) return [];
    const out = [];
    for (const node of this.spec.nodes) {
      const pos = node.pos || { x: 0, y: 0 };
      const h = nodeHeight(node, this.info);
      const ports = node.type === 'input' ? 0 : (LAYERS[node.type] ? LAYERS[node.type].ports || 1 : 1);
      out.push({ node, x: pos.x, y: pos.y, w: NODE_W, h, inPorts: ports });
    }
    return out;
  }

  _hit(event) {
    const point = this._eventPoint(event);
    const world = this.toWorld(point.x, point.y);
    const boxes = this._boxes();
    for (const box of boxes) {
      if (box.inPorts > 0) {
        for (let i = 0; i < box.inPorts; i++) {
          const py = box.y + NODE_HEAD + 8 + i * 14;
          if (Math.hypot(world.x - box.x, world.y - py) <= PORT_R + 4) return { kind: 'port', node: box.node, port: i, side: 'in', box };
        }
      }
      const oy = box.y + NODE_HEAD + 8;
      if (Math.hypot(world.x - (box.x + box.w), world.y - oy) <= PORT_R + 4) return { kind: 'port', node: box.node, port: 0, side: 'out', box };
    }
    for (let i = boxes.length - 1; i >= 0; i--) {
      const box = boxes[i];
      if (world.x >= box.x && world.x <= box.x + box.w && world.y >= box.y && world.y <= box.y + box.h) return { kind: 'node', node: box.node, box };
    }
    const edge = this._hitEdge(world);
    if (edge) return { kind: 'edge', edge };
    return null;
  }

  _hitEdge(world) {
    if (!this.spec) return null;
    const boxes = {};
    for (const box of this._boxes()) boxes[box.node.id] = box;
    for (const edge of this.spec.edges) {
      const a = boxes[edge.from];
      const b = boxes[edge.to];
      if (!a || !b) continue;
      const p0 = { x: a.x + a.w, y: a.y + NODE_HEAD + 8 };
      const p1 = { x: b.x, y: b.y + NODE_HEAD + 8 + (edge.toPort || 0) * 14 };
      for (let t = 0; t <= 1.0001; t += 0.04) {
        const x = bez(p0.x, p1.x, t);
        const y = bezY(p0.y, p1.y, t);
        if (Math.hypot(world.x - x, world.y - y) < 5) return edge;
      }
    }
    return null;
  }

  _down(event) {
    if (event.button === 2) return;
    this.canvas.setPointerCapture(event.pointerId);
    const point = this._eventPoint(event);
    const hit = this._hit(event);
    if (hit && hit.kind === 'port') {
      if (hit.side === 'out') {
        this.link = { from: hit.node.id, to: null, x: point.x, y: point.y };
      } else {
        const existing = this.spec.edges.find((e) => e.to === hit.node.id && (e.toPort || 0) === hit.port);
        if (existing) {
          this.spec.edges = this.spec.edges.filter((e) => e !== existing);
          this.link = { from: existing.from, to: hit.node.id, toPort: hit.port, x: point.x, y: point.y, detached: true };
          this._changed();
        }
      }
      return;
    }
    if (hit && hit.kind === 'node') {
      this.select(hit.node.id);
      const world = this.toWorld(point.x, point.y);
      this.drag = { kind: 'node', id: hit.node.id, dx: world.x - hit.box.x, dy: world.y - hit.box.y, moved: false };
      return;
    }
    if (hit && hit.kind === 'edge') {
      this.select(null);
      this.drag = { kind: 'edge', edge: hit.edge };
      return;
    }
    this.select(null);
    this.drag = { kind: 'pan', sx: point.x, sy: point.y, vx: this.view.x, vy: this.view.y };
  }

  _move(event) {
    const point = this._eventPoint(event);
    if (this.link) {
      this.link.x = point.x;
      this.link.y = point.y;
      const hit = this._hit(event);
      this.link.target = hit && hit.kind === 'port' && hit.side === 'in' ? { id: hit.node.id, port: hit.port } : null;
      this.render();
      return;
    }
    if (this.drag && this.drag.kind === 'node') {
      const world = this.toWorld(point.x, point.y);
      const node = this.spec.nodes.find((n) => n.id === this.drag.id);
      if (node) {
        node.pos = { x: Math.round(world.x - this.drag.dx), y: Math.round(world.y - this.drag.dy) };
        this.drag.moved = true;
        this.dirty = true;
        this.render();
      }
      return;
    }
    if (this.drag && this.drag.kind === 'pan') {
      this.view.x = this.drag.vx + (point.x - this.drag.sx);
      this.view.y = this.drag.vy + (point.y - this.drag.sy);
      this.render();
      return;
    }
    const hit = this._hit(event);
    this.hoverNode = hit && hit.kind === 'node' ? hit.node.id : null;
    this.hoverPort = hit && hit.kind === 'port' ? hit : null;
    this.canvas.style.cursor = hit ? (hit.kind === 'port' ? 'crosshair' : hit.kind === 'edge' ? 'pointer' : 'grab') : 'default';
    this.render();
  }

  _up(event) {
    if (this.link) {
      const hit = this._hit(event);
      if (hit && hit.kind === 'port' && hit.side === 'in' && hit.node.type !== 'input') {
        const toPort = hit.port;
        const from = this.link.from;
        if (from && from !== hit.node.id) {
          this.spec.edges = this.spec.edges.filter((e) => !(e.to === hit.node.id && (e.toPort || 0) === toPort));
          this.spec.edges.push({ from, to: hit.node.id, toPort });
          this._changed();
        }
      } else if (this.link.detached) {
        this._changed();
      }
      this.link = null;
      this.render();
      return;
    }
    if (this.drag && this.drag.kind === 'node' && this.drag.moved) this._changed(true);
    this.drag = null;
    try { this.canvas.releasePointerCapture(event.pointerId); } catch (e) { this.canvas = this.canvas; }
  }

  _changed(positionsOnly) {
    this.dirty = false;
    this.info = this.analyze();
    if (this.onChange) this.onChange(this.spec, this.info, positionsOnly === true);
    else this.render();
  }

  select(id) {
    if (this.selectedId === id) return;
    this.selectedId = id;
    if (this.onSelect) this.onSelect(id, this.spec ? this.spec.nodes.find((n) => n.id === id) || null : null);
    this.render();
  }

  refresh() {
    this.info = this.analyze();
    this.render();
  }

  fit(margin) {
    const boxes = this._boxes();
    if (!boxes.length) return;
    const size = measureCanvas(this.canvas);
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (const box of boxes) {
      minX = Math.min(minX, box.x);
      minY = Math.min(minY, box.y);
      maxX = Math.max(maxX, box.x + box.w);
      maxY = Math.max(maxY, box.y + box.h);
    }
    const pad = margin === undefined ? 28 : margin;
    const scale = Math.max(0.25, Math.min(1.6, Math.min((size.width - pad * 2) / Math.max(1, maxX - minX), (size.height - pad * 2) / Math.max(1, maxY - minY))));
    this.view.scale = scale;
    this.view.x = pad - minX * scale + Math.max(0, (size.width - pad * 2 - (maxX - minX) * scale) / 2);
    this.view.y = pad - minY * scale + Math.max(0, (size.height - pad * 2 - (maxY - minY) * scale) / 2);
    this.render();
  }

  autoLayout() {
    if (!this.spec) return;
    const levels = {};
    const order = [];
    const indegree = {};
    const outgoing = {};
    for (const node of this.spec.nodes) { indegree[node.id] = 0; outgoing[node.id] = []; }
    for (const edge of this.spec.edges) {
      if (indegree[edge.to] === undefined || outgoing[edge.from] === undefined) continue;
      indegree[edge.to]++;
      outgoing[edge.from].push(edge.to);
    }
    const queue = this.spec.nodes.filter((n) => indegree[n.id] === 0).map((n) => n.id);
    for (const id of queue) levels[id] = 0;
    let guard = 0;
    while (queue.length && guard < 5000) {
      guard++;
      const id = queue.shift();
      order.push(id);
      for (const next of outgoing[id]) {
        levels[next] = Math.max(levels[next] === undefined ? 0 : levels[next], levels[id] + 1);
        indegree[next]--;
        if (indegree[next] === 0) queue.push(next);
      }
    }
    for (const node of this.spec.nodes) if (levels[node.id] === undefined) levels[node.id] = 0;
    const byLevel = {};
    for (const node of this.spec.nodes) {
      const level = levels[node.id];
      if (!byLevel[level]) byLevel[level] = [];
      byLevel[level].push(node);
    }
    for (const level of Object.keys(byLevel)) {
      const list = byLevel[level];
      list.forEach((node, index) => {
        node.pos = { x: 40 + Number(level) * (NODE_W + 46), y: 60 + index * (nodeHeight(node, this.info) + 34) };
      });
    }
    this._changed();
    this.fit();
  }

  centerOn(nodeId) {
    const box = this._boxes().find((b) => b.node.id === nodeId);
    if (!box) return;
    const size = measureCanvas(this.canvas);
    this.view.x = size.width / 2 - (box.x + box.w / 2) * this.view.scale;
    this.view.y = size.height / 2 - (box.y + box.h / 2) * this.view.scale;
    this.render();
  }

  render() {
    if (!this.canvas) return;
    const size = measureCanvas(this.canvas);
    const ctx = fitCanvas(this.canvas, size.width, size.height);
    ctx.fillStyle = '#15181a';
    ctx.fillRect(0, 0, size.width, size.height);
    this._drawGrid(ctx, size);
    if (!this.spec) {
      ctx.fillStyle = '#5b6268';
      ctx.font = '12px ui-monospace, SFMono-Regular, Menlo, monospace';
      ctx.textAlign = 'center';
      ctx.fillText('pick a preset or start from a blank graph', size.width / 2, size.height / 2);
      return;
    }
    const boxes = this._boxes();
    const boxById = {};
    for (const box of boxes) boxById[box.node.id] = box;
    ctx.save();
    ctx.translate(this.view.x, this.view.y);
    ctx.scale(this.view.scale, this.view.scale);
    this._drawEdges(ctx, boxById);
    for (const box of boxes) this._drawNode(ctx, box);
    if (this.link) this._drawLink(ctx, boxById);
    ctx.restore();
    this._drawHud(ctx, size);
  }

  _drawGrid(ctx, size) {
    const step = 26 * this.view.scale;
    if (step < 7) return;
    ctx.save();
    ctx.strokeStyle = '#1c2023';
    ctx.lineWidth = 1;
    const offX = this.view.x % step;
    const offY = this.view.y % step;
    ctx.beginPath();
    for (let x = offX; x < size.width; x += step) { ctx.moveTo(Math.round(x) + 0.5, 0); ctx.lineTo(Math.round(x) + 0.5, size.height); }
    for (let y = offY; y < size.height; y += step) { ctx.moveTo(0, Math.round(y) + 0.5); ctx.lineTo(size.width, Math.round(y) + 0.5); }
    ctx.stroke();
    ctx.restore();
  }

  _drawEdges(ctx, boxById) {
    ctx.save();
    ctx.lineWidth = 1.6;
    for (const edge of this.spec.edges) {
      const a = boxById[edge.from];
      const b = boxById[edge.to];
      if (!a || !b) continue;
      const p0 = { x: a.x + a.w, y: a.y + NODE_HEAD + 8 };
      const p1 = { x: b.x, y: b.y + NODE_HEAD + 8 + (edge.toPort || 0) * 14 };
      const active = this.selectedId === edge.from || this.selectedId === edge.to;
      const isError = this.info && this.info.errorByNode && this.info.errorByNode[edge.to];
      ctx.strokeStyle = isError ? '#7d423c' : active ? '#e0a33e' : this.hoverNode && (this.hoverNode === edge.from || this.hoverNode === edge.to) ? '#5d666e' : '#3b4249';
      ctx.beginPath();
      const mid = Math.max(24, Math.abs(p1.x - p0.x) * 0.45);
      ctx.moveTo(p0.x, p0.y);
      ctx.bezierCurveTo(p0.x + mid, p0.y, p1.x - mid, p1.y, p1.x, p1.y);
      ctx.stroke();
      ctx.fillStyle = ctx.strokeStyle;
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p1.x - 5, p1.y - 3);
      ctx.lineTo(p1.x - 5, p1.y + 3);
      ctx.closePath();
      ctx.fill();
    }
    ctx.restore();
  }

  _drawLink(ctx, boxById) {
    const from = boxById[this.link.from];
    if (!from) return;
    const start = { x: from.x + from.w, y: from.y + NODE_HEAD + 8 };
    const end = this.toWorld(this.link.x, this.link.y);
    ctx.save();
    ctx.strokeStyle = this.link.target ? '#6aa84f' : '#e0a33e';
    ctx.lineWidth = 1.8;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    const mid = Math.max(24, Math.abs(end.x - start.x) * 0.45);
    ctx.moveTo(start.x, start.y);
    ctx.bezierCurveTo(start.x + mid, start.y, end.x - mid, end.y, end.x, end.y);
    ctx.stroke();
    ctx.restore();
  }

  _drawNode(ctx, box) {
    const node = box.node;
    const layer = LAYERS[node.type];
    const stat = this.info && this.info.byNode ? this.info.byNode[node.id] : null;
    const message = this.info && this.info.errorByNode ? this.info.errorByNode[node.id] : null;
    const selected = this.selectedId === node.id;
    const isOutput = this.spec.output === node.id;
    const x = box.x;
    const y = box.y;
    const w = box.w;
    const h = box.h;

    ctx.save();
    ctx.fillStyle = selected ? '#23272b' : '#1e2225';
    ctx.strokeStyle = message ? '#8d4a43' : selected ? '#e0a33e' : isOutput ? '#3f6f70' : '#2f353b';
    ctx.lineWidth = selected || message ? 1.6 : 1;
    roundRect(ctx, x, y, w, h, 3);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = colorOf(node.type);
    ctx.globalAlpha = 0.9;
    roundRectTop(ctx, x, y, w, NODE_HEAD, 3);
    ctx.fill();
    ctx.globalAlpha = 0.16;
    ctx.fillStyle = '#000';
    roundRectTop(ctx, x, y, w, NODE_HEAD, 3);
    ctx.fill();
    ctx.globalAlpha = 1;

    ctx.fillStyle = colorOf(node.type);
    ctx.fillRect(x, y, 3, NODE_HEAD);

    ctx.font = '600 10.5px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#e4e7ea';
    const label = stat && stat.label ? stat.label : (layer ? layer.label : node.type);
    ctx.fillText(truncate(ctx, label, w - 46), x + 9, y + NODE_HEAD / 2 + 0.5);

    if (isOutput) {
      ctx.font = '9.5px ui-monospace, SFMono-Regular, Menlo, monospace';
      ctx.fillStyle = '#4fa3a5';
      ctx.textAlign = 'right';
      ctx.fillText('OUT', x + w - 7, y + NODE_HEAD / 2 + 0.5);
    } else if (stat && stat.params) {
      ctx.font = '9.5px ui-monospace, SFMono-Regular, Menlo, monospace';
      ctx.fillStyle = '#9aa1a8';
      ctx.textAlign = 'right';
      ctx.fillText(compactNumber(stat.params), x + w - 7, y + NODE_HEAD / 2 + 0.5);
    }

    ctx.textAlign = 'left';
    ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillStyle = '#8b9298';
    const summary = stat && stat.text ? stat.text : summarizeCfg(node);
    ctx.fillText(truncate(ctx, summary, w - 16), x + 9, y + NODE_HEAD + 10);

    const shapeText = stat && stat.out ? fmtShape(stat.out) : (node.type === 'input' ? fmtShape([1].concat(this.spec.input.shape)) : '—');
    ctx.fillStyle = '#5f676e';
    ctx.fillText(truncate(ctx, shapeText, w - 16), x + 9, y + NODE_HEAD + 10 + NODE_ROW);

    if (message) {
      ctx.fillStyle = '#d08b83';
      ctx.font = '9.5px ui-monospace, SFMono-Regular, Menlo, monospace';
      ctx.fillText(truncate(ctx, '⚠ ' + message, w - 16), x + 9, y + h - 7);
    }

    if (node.type !== 'input') {
      const ports = (layer ? layer.ports || 1 : 1);
      for (let i = 0; i < ports; i++) {
        const py = y + NODE_HEAD + 8 + i * 14;
        const connected = this.spec.edges.some((e) => e.to === node.id && (e.toPort || 0) === i);
        ctx.beginPath();
        ctx.arc(x, py, PORT_R, 0, Math.PI * 2);
        ctx.fillStyle = connected ? '#3b4249' : '#5d3a35';
        ctx.fill();
        ctx.strokeStyle = this.hoverPort && this.hoverPort.node.id === node.id && this.hoverPort.port === i ? '#e0a33e' : '#15181a';
        ctx.lineWidth = 1.2;
        ctx.stroke();
      }
    }
    const oy = y + NODE_HEAD + 8;
    ctx.beginPath();
    ctx.arc(x + w, oy, PORT_R, 0, Math.PI * 2);
    ctx.fillStyle = this.spec.edges.some((e) => e.from === node.id) ? '#4a5158' : '#5d3a35';
    ctx.fill();
    ctx.strokeStyle = this.hoverPort && this.hoverPort.node.id === node.id && this.hoverPort.side === 'out' ? '#e0a33e' : '#15181a';
    ctx.lineWidth = 1.2;
    ctx.stroke();
    ctx.restore();
  }

  _drawHud(ctx, size) {
    ctx.save();
    ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillStyle = '#5f676e';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'bottom';
    ctx.fillText(Math.round(this.view.scale * 100) + '%', size.width - 8, size.height - 6);
    if (this.info) {
      ctx.textAlign = 'left';
      const valid = this.info.valid;
      ctx.fillStyle = valid ? '#5f676e' : '#d08b83';
      ctx.fillText(valid
        ? this.info.totalParams.toLocaleString() + ' params   ' + compactNumber(this.info.macs) + ' MACs   out ' + fmtShape(this.info.outputShape)
        : (this.info.errors.length ? this.info.errors.length + ' problem' + (this.info.errors.length === 1 ? '' : 's') + ': ' + this.info.errors[0].message : 'invalid graph'), 8, size.height - 6);
    }
    ctx.restore();
  }

  exportImage(scale) {
    const boxes = this._boxes();
    if (!boxes.length) return null;
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (const box of boxes) {
      minX = Math.min(minX, box.x);
      minY = Math.min(minY, box.y);
      maxX = Math.max(maxX, box.x + box.w);
      maxY = Math.max(maxY, box.y + box.h);
    }
    const pad = 18;
    const s = scale || 2;
    const out = document.createElement('canvas');
    out.width = Math.round((maxX - minX + pad * 2) * s);
    out.height = Math.round((maxY - minY + pad * 2) * s);
    const ctx = out.getContext('2d');
    ctx.setTransform(s, 0, 0, s, 0, 0);
    ctx.fillStyle = '#15181a';
    ctx.fillRect(0, 0, out.width / s, out.height / s);
    const saved = { x: this.view.x, y: this.view.y, scale: this.view.scale };
    this.view = { x: pad - minX, y: pad - minY, scale: 1 };
    ctx.save();
    ctx.translate(this.view.x, this.view.y);
    const boxById = {};
    for (const box of boxes) boxById[box.node.id] = box;
    this._drawEdges(ctx, boxById);
    for (const box of boxes) this._drawNode(ctx, box);
    ctx.restore();
    this.view = saved;
    return out;
  }
}

function bez(x0, x1, t) {
  const mid = Math.max(24, Math.abs(x1 - x0) * 0.45);
  const mt = 1 - t;
  return mt * mt * mt * x0 + 3 * mt * mt * t * (x0 + mid) + 3 * mt * t * t * (x1 - mid) + t * t * t * x1;
}

function bezY(y0, y1, t) {
  const mt = 1 - t;
  return mt * mt * mt * y0 + 3 * mt * mt * t * y0 + 3 * mt * t * t * y1 + t * t * t * y1;
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

function roundRectTop(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h);
  ctx.lineTo(x, y + h);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

function truncate(ctx, text, width) {
  if (ctx.measureText(text).width <= width) return text;
  let out = text;
  while (out.length > 1 && ctx.measureText(out + '…').width > width) out = out.slice(0, -1);
  return out + '…';
}

export function fmtShape(shape) {
  if (!shape) return '—';
  const list = Array.isArray(shape) ? shape : [shape];
  return '[' + list.map((v) => (v === -1 ? 'B' : v)).join('×') + ']';
}

export function summarizeCfg(node) {
  const cfg = node.cfg || {};
  const parts = [];
  for (const key of Object.keys(cfg)) {
    const value = cfg[key];
    if (value === null || value === undefined) continue;
    if (typeof value === 'boolean') { if (value) parts.push(key); continue; }
    if (typeof value === 'number') { parts.push(key + ' ' + value); continue; }
    if (typeof value === 'string') { parts.push(value); continue; }
    if (Array.isArray(value)) { parts.push(key + ' ' + value.join(',')); continue; }
  }
  return parts.join('  ') || node.type;
}
