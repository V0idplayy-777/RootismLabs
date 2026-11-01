import { LAYERS, defaultConfig } from './layers.js';
import { Initializers, Rng, shapeSize, Node } from './tensor.js';

function topoSort(spec) {
  const nodes = spec.nodes.slice();
  const byId = new Map();
  for (const n of nodes) byId.set(n.id, n);
  const incoming = new Map();
  const outgoing = new Map();
  for (const n of nodes) {
    incoming.set(n.id, []);
    outgoing.set(n.id, []);
  }
  for (const e of spec.edges || []) {
    if (!byId.has(e.from) || !byId.has(e.to)) continue;
    outgoing.get(e.from).push(e);
    incoming.get(e.to).push(e);
  }
  const indeg = new Map();
  for (const n of nodes) indeg.set(n.id, incoming.get(n.id).length);
  const queue = nodes.filter((n) => indeg.get(n.id) === 0).map((n) => n.id);
  const order = [];
  while (queue.length) {
    queue.sort();
    const id = queue.shift();
    order.push(id);
    for (const e of outgoing.get(id)) {
      indeg.set(e.to, indeg.get(e.to) - 1);
      if (indeg.get(e.to) === 0) queue.push(e.to);
    }
  }
  if (order.length !== nodes.length) {
    for (const n of nodes) if (!order.includes(n.id)) order.push(n.id);
  }
  return { order, incoming, outgoing, byId };
}

export function analyze(spec) {
  const { order, incoming } = topoSort(spec);
  const shapes = {};
  const errors = [];
  const paramList = [];
  const stateList = [];
  const nodeStats = [];
  let macs = 0;
  let activationBytes = 0;
  const inShape = (spec.input && spec.input.shape) ? [1].concat(spec.input.shape) : null;

  for (const id of order) {
    const node = spec.nodes.find((n) => n.id === id);
    const type = node.type;
    const L = LAYERS[type];
    if (!L) {
      errors.push({ id, message: 'unknown layer type ' + type });
      continue;
    }
    const cfg = Object.assign(defaultConfig(type), node.cfg || {});
    const ins = (incoming.get(id) || [])
      .slice()
      .sort((a, b) => (a.toPort || 0) - (b.toPort || 0))
      .map((e) => shapes[e.from] || null);
    let out = null;
    if (type === 'input') {
      out = inShape;
    } else {
      if (L.ports > 0 && ins.some((s) => !s)) {
        errors.push({ id, message: 'missing or unresolved input' });
      } else {
        try {
          out = L.infer(ins, cfg, spec);
        } catch (err) {
          out = null;
        }
      }
    }
    if (!out) {
      if (!errors.some((e) => e.id === id)) errors.push({ id, message: 'shape inference failed' });
      shapes[id] = null;
      nodeStats.push({ id, type, label: L.label, params: 0, macs: 0, out: null, error: true });
      continue;
    }
    shapes[id] = out;
    let ps = [];
    let ss = [];
    try {
      ps = L.params(ins, cfg, spec) || [];
      ss = (L.states ? L.states(ins, cfg, spec) : []) || [];
    } catch (err) {
      ps = [];
    }
    let np = 0;
    for (const p of ps) {
      const size = shapeSize(p.shape);
      np += size;
      paramList.push({ name: id + '.' + p.suffix, shape: p.shape, init: p.init || 'glorot', std: p.std, node: id, trainable: true });
    }
    for (const s of ss) {
      stateList.push({ name: id + '.' + s.suffix, shape: s.shape, init: s.init || 'zeros', node: id, trainable: false });
    }
    let m = 0;
    try {
      m = L.macs(ins, out, cfg) || 0;
    } catch (err) {
      m = 0;
    }
    macs += m;
    activationBytes += shapeSize(out) * 4;
    nodeStats.push({
      id,
      type,
      label: L.label,
      text: L.text ? L.text(cfg, spec) : '',
      params: np,
      macs: m,
      in: ins,
      out
    });
  }

  const outId = spec.output || (order.length ? order[order.length - 1] : null);
  const totalParams = paramList.reduce((s, p) => s + shapeSize(p.shape), 0);
  const totalStates = stateList.reduce((s, p) => s + shapeSize(p.shape), 0);
  return {
    order,
    shapes,
    errors,
    paramList,
    stateList,
    nodeStats,
    macs,
    totalParams,
    totalStates,
    paramBytes: totalParams * 4,
    activationBytes,
    outputId: outId,
    outputShape: outId ? shapes[outId] : null,
    valid: errors.length === 0 && !!outId && !!shapes[outId]
  };
}

export class Model {
  constructor(spec, seed) {
    this.spec = JSON.parse(JSON.stringify(spec));
    this.seed = seed === undefined ? 1234 : seed;
    this.rng = new Rng(this.seed);
    this.info = analyze(this.spec);
    this.params = new Map();
    this.states = new Map();
    this.build();
  }

  build() {
    this.params.clear();
    this.states.clear();
    for (const p of this.info.paramList) {
      const shape = p.shape;
      const init = Initializers[p.init] || Initializers.glorot;
      const node = init(shape, this.rng, { std: p.std });
      node.leaf = true;
      node.pname = p.name;
      node.g = new Float32Array(node.d.length);
      this.params.set(p.name, node);
    }
    for (const s of this.info.stateList) {
      const init = Initializers[s.init] || Initializers.zeros;
      const node = init(s.shape, this.rng, {});
      node.leaf = true;
      node.pname = s.name;
      node.trainable = false;
      this.states.set(s.name, node);
    }
    this.bnStates = new Map();
    for (const n of this.spec.nodes) {
      if (n.type === 'batchnorm') {
        const m = this.states.get(n.id + '.runningMean');
        const v = this.states.get(n.id + '.runningVar');
        this.bnStates.set(n.id, {
          runningMean: m ? Array.from(m.d) : [0],
          runningVar: v ? Array.from(v.d) : [1],
          meanNode: m,
          varNode: v
        });
      }
    }
  }

  reseed(seed) {
    this.seed = seed;
    this.rng = new Rng(seed);
    this.build();
  }

  trainableParams() {
    return Array.from(this.params.values());
  }

  paramCount() {
    let c = 0;
    for (const p of this.params.values()) c += p.d.length;
    return c;
  }

  forward(x, mode, rng) {
    const { order, incoming } = topoSort(this.spec);
    const values = new Map();
    const byId = new Map();
    for (const n of this.spec.nodes) byId.set(n.id, n);
    const runRng = rng || this.rng;
    for (const id of order) {
      const node = byId.get(id);
      const L = LAYERS[node.type];
      if (!L) continue;
      const cfg = Object.assign(defaultConfig(node.type), node.cfg || {});
      const edges = (incoming.get(id) || []).slice().sort((a, b) => (a.toPort || 0) - (b.toPort || 0));
      const ins = edges.map((e) => values.get(e.from));
      if (node.type === 'input') {
        values.set(id, x);
        continue;
      }
      if (ins.some((v) => v === undefined || v === null)) continue;
      const p = {};
      for (const [k, v] of this.params) {
        if (k.startsWith(id + '.')) p[k.slice(id.length + 1)] = v;
      }
      const staticOut = this.info.shapes[id];
      const shapes = {
        in: edges.map((e) => this.info.shapes[e.from]),
        out: staticOut ? [ins[0].shape[0]].concat(staticOut.slice(1)) : null
      };
      const ctx = {
        mode,
        rng: runRng,
        state: this.bnStates.get(id) || { runningMean: [0], runningVar: [1] },
        nodeId: id
      };
      const out = L.run(ctx, ins, cfg, p, shapes);
      values.set(id, out);
    }
    return values.get(this.info.outputId);
  }

  syncStates() {
    for (const [id, st] of this.bnStates) {
      if (st.meanNode) st.meanNode.d.set(Float32Array.from(st.runningMean));
      if (st.varNode) st.varNode.d.set(Float32Array.from(st.runningVar));
    }
  }

  weights() {
    const out = [];
    for (const [name, node] of this.params) out.push({ name, shape: node.shape.slice(), data: new Float32Array(node.d) });
    for (const [name, node] of this.states) out.push({ name, shape: node.shape.slice(), data: new Float32Array(node.d) });
    return out;
  }

  loadWeights(list) {
    for (const w of list) {
      const p = this.params.get(w.name) || this.states.get(w.name);
      if (!p) continue;
      if (p.d.length !== w.data.length) continue;
      p.d.set(w.data);
      if (this.bnStates) {
        for (const [id, st] of this.bnStates) {
          if (nameMatches(id, w.name, 'runningMean') && st.meanNode) st.runningMean = Array.from(w.data);
          if (nameMatches(id, w.name, 'runningVar') && st.varNode) st.runningVar = Array.from(w.data);
        }
      }
    }
  }

  zeroGrad() {
    for (const p of this.params.values()) if (p.g) p.g.fill(0);
  }

  summary() {
    const info = this.info;
    return {
      params: info.totalParams,
      states: info.totalStates,
      macs: info.macs,
      flops: info.macs * 2,
      paramBytes: info.totalParams * 4,
      activationBytes: info.activationBytes,
      outputShape: info.outputShape,
      errors: info.errors,
      nodes: info.nodeStats
    };
  }

  clone(seed) {
    const m = new Model(this.spec, seed === undefined ? this.seed : seed);
    m.loadWeights(this.weights());
    m.syncStates();
    return m;
  }
}

function nameMatches(id, name, suffix) {
  return name === id + '.' + suffix;
}

export function blankGraph(inputShape, inputName) {
  const inId = 'in';
  const outId = 'out';
  return {
    id: null,
    name: 'Untitled model',
    version: 1,
    input: { name: inputName || 'x', shape: inputShape.slice(), dtype: 'float32' },
    nodes: [
      { id: inId, type: 'input', cfg: {}, pos: { x: 40, y: 160 } },
      { id: outId, type: 'dense', cfg: { units: 2, activation: 'linear', useBias: true }, pos: { x: 340, y: 160 } }
    ],
    edges: [{ from: inId, to: outId, toPort: 0 }],
    output: outId,
    head: 'softmax',
    meta: {}
  };
}

export function graphStats(spec) {
  const info = analyze(spec);
  return info;
}

export function specToText(spec) {
  const info = analyze(spec);
  const lines = [];
  lines.push('input: ' + ((spec.input && spec.input.shape) || []).join(' × '));
  for (const s of info.nodeStats) {
    if (s.type === 'input') continue;
    lines.push(
      pad(s.id, 10) + pad(s.label, 20) + pad(s.text || '', 26) +
      pad(s.out ? s.out.join('×') : '?', 18) + pad(fmt(s.params), 10) + fmt(s.macs)
    );
  }
  lines.push('─'.repeat(94));
  lines.push(pad('total', 56) + pad(fmt(info.totalParams), 18) + fmt(info.macs));
  return lines.join('\n');
}

function pad(s, n) {
  s = String(s);
  return s.length >= n ? s.slice(0, n) : s + ' '.repeat(n - s.length);
}

function fmt(n) {
  if (n === undefined || n === null) return '0';
  if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'k';
  return String(n);
}

export { fmt as compactNumber };
