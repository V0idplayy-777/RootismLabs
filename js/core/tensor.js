export function shapeSize(shape) {
  let n = 1;
  for (let i = 0; i < shape.length; i++) n *= shape[i];
  return n;
}

export function strides(shape) {
  const out = new Array(shape.length);
  let acc = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    out[i] = acc;
    acc *= shape[i];
  }
  return out;
}

export function shapeEq(a, b) {
  if (!a || !b) return false;
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

export function nodeSize(n) {
  return shapeSize(n.shape);
}

export class Node {
  constructor(data, shape) {
    this.d = data instanceof Float32Array ? data : Float32Array.from(data);
    this.shape = shape.slice();
    this.g = null;
    this.parents = [];
    this.bwd = null;
    this.op = '';
    this.leaf = false;
  }

  get size() {
    return shapeSize(this.shape);
  }

  at(...idx) {
    const st = strides(this.shape);
    let off = 0;
    for (let i = 0; i < idx.length; i++) off += idx[i] * st[i];
    return this.d[off];
  }

  zeroGrad() {
    if (this.g) this.g.fill(0);
  }

  toArray() {
    return Array.from(this.d);
  }

  clone() {
    const n = new Node(new Float32Array(this.d), this.shape);
    n.op = this.op;
    return n;
  }
}

export function tensor(data, shape) {
  if (data instanceof Node) return data;
  if (typeof data === 'number') return new Node(new Float32Array([data]), [1]);
  if (!shape) {
    shape = inferShape(data);
    data = flattenNested(data);
  }
  return new Node(Float32Array.from(data), shape);
}

function inferShape(arr) {
  const shape = [];
  let cur = arr;
  while (Array.isArray(cur)) {
    shape.push(cur.length);
    cur = cur[0];
  }
  return shape;
}

function flattenNested(arr, out) {
  out = out || [];
  if (Array.isArray(arr)) {
    for (let i = 0; i < arr.length; i++) flattenNested(arr[i], out);
  } else {
    out.push(arr);
  }
  return out;
}

export function zeros(shape) {
  return new Node(new Float32Array(shapeSize(shape)), shape);
}

export function ones(shape) {
  const n = shapeSize(shape);
  const d = new Float32Array(n);
  d.fill(1);
  return new Node(d, shape);
}

export function full(shape, v) {
  const n = shapeSize(shape);
  const d = new Float32Array(n);
  d.fill(v);
  return new Node(d, shape);
}

export function param(name, shape, data) {
  const n = new Node(data || new Float32Array(shapeSize(shape)), shape);
  n.leaf = true;
  n.op = 'param';
  n.pname = name;
  n.g = new Float32Array(n.d.length);
  return n;
}

export class Rng {
  constructor(seed) {
    this.s = (seed >>> 0) || 1;
  }
  next() {
    let t = (this.s += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  range(a, b) {
    return a + (b - a) * this.next();
  }
  int(n) {
    return Math.floor(this.next() * n);
  }
  gauss() {
    let u = 0;
    let v = 0;
    while (u === 0) u = this.next();
    while (v === 0) v = this.next();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  pick(arr) {
    return arr[this.int(arr.length)];
  }
  shuffle(arr) {
    const a = arr.slice();
    for (let i = a.length - 1; i > 0; i--) {
      const j = this.int(i + 1);
      const t = a[i];
      a[i] = a[j];
      a[j] = t;
    }
    return a;
  }
}

export function randomNormal(shape, rng, mean, std) {
  mean = mean || 0;
  std = std === undefined ? 1 : std;
  const n = shapeSize(shape);
  const d = new Float32Array(n);
  for (let i = 0; i < n; i++) d[i] = mean + std * rng.gauss();
  return new Node(d, shape);
}

export function randomUniform(shape, rng, lo, hi) {
  const n = shapeSize(shape);
  const d = new Float32Array(n);
  for (let i = 0; i < n; i++) d[i] = rng.range(lo, hi);
  return new Node(d, shape);
}

export const Initializers = {
  zeros(shape, rng) {
    return zeros(shape);
  },
  ones(shape, rng) {
    return ones(shape);
  },
  constant(shape, rng, opts) {
    return full(shape, (opts && opts.value) || 0);
  },
  normal(shape, rng, opts) {
    return randomNormal(shape, rng, 0, (opts && opts.std) || 0.01);
  },
  uniform(shape, rng, opts) {
    const b = (opts && opts.range) || 0.05;
    return randomUniform(shape, rng, -b, b);
  },
  glorot(shape, rng) {
    const fan = fanAvg(shape);
    const lim = Math.sqrt(6 / fan);
    return randomUniform(shape, rng, -lim, lim);
  },
  he(shape, rng) {
    const fan = fanIn(shape);
    const std = Math.sqrt(2 / fan);
    return randomNormal(shape, rng, 0, std);
  },
  lecun(shape, rng) {
    const fan = fanIn(shape);
    const std = Math.sqrt(1 / fan);
    return randomNormal(shape, rng, 0, std);
  },
  orthogonal(shape, rng) {
    const rows = shape[0];
    const cols = shapeSize(shape.slice(1));
    const a = randomNormal([rows, cols], rng, 0, 1);
    const q = gramSchmidt(a.d, rows, cols);
    const scale = Math.sqrt(2 / (rows + cols));
    const d = new Float32Array(q.length);
    for (let i = 0; i < q.length; i++) d[i] = q[i] * scale;
    return new Node(d, shape);
  },
  identity(shape, rng, opts) {
    const gain = (opts && opts.gain) || 1;
    const d = new Float32Array(shapeSize(shape));
    const n = Math.min(shape[0], shape[1]);
    for (let i = 0; i < n; i++) d[i * shape[1] + i] = gain;
    return new Node(d, shape);
  }
};

function fanIn(shape) {
  if (shape.length === 1) return shape[0];
  if (shape.length === 2) return shape[0];
  let f = shape[shape.length - 1];
  for (let i = 0; i < shape.length - 2; i++) f *= shape[i];
  return Math.max(1, f);
}

function fanAvg(shape) {
  if (shape.length === 1) return shape[0];
  if (shape.length === 2) return (shape[0] + shape[1]) / 2;
  const rec = shapeSize(shape.slice(0, shape.length - 2));
  return (shape[shape.length - 2] + shape[shape.length - 1]) / 2 * Math.max(1, rec);
}

function gramSchmidt(d, rows, cols) {
  const out = new Float32Array(d);
  const n = Math.min(rows, cols);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < i; j++) {
      let dot = 0;
      for (let k = 0; k < cols; k++) dot += out[i * cols + k] * out[j * cols + k];
      for (let k = 0; k < cols; k++) out[i * cols + k] -= dot * out[j * cols + k];
    }
    let norm = 0;
    for (let k = 0; k < cols; k++) norm += out[i * cols + k] * out[i * cols + k];
    norm = Math.sqrt(norm) || 1;
    for (let k = 0; k < cols; k++) out[i * cols + k] /= norm;
  }
  return out;
}

export function gemm(A, B, C, M, N, K, alpha, beta, tA, tB) {
  alpha = alpha === undefined ? 1 : alpha;
  beta = beta === undefined ? 0 : beta;
  if (beta === 0) {
    for (let i = 0; i < M * N; i++) C[i] = 0;
  } else if (beta !== 1) {
    for (let i = 0; i < M * N; i++) C[i] *= beta;
  }
  const aStep = tA ? M : 1;
  const aRow = tA ? 1 : K;
  const bStep = tB ? K : 1;
  const bRow = tB ? 1 : N;
  for (let i = 0; i < M; i++) {
    const aBase = i * aRow;
    const cBase = i * N;
    for (let k = 0; k < K; k++) {
      const a = alpha * A[aBase + k * aStep];
      if (a === 0) continue;
      const bBase = k * bRow;
      for (let j = 0; j < N; j++) {
        C[cBase + j] += a * B[bBase + j * bStep];
      }
    }
  }
  return C;
}

export function addInto(dst, src, scale) {
  scale = scale === undefined ? 1 : scale;
  if (!dst) return src;
  if (scale === 1) {
    for (let i = 0; i < dst.length; i++) dst[i] += src[i];
  } else {
    for (let i = 0; i < dst.length; i++) dst[i] += scale * src[i];
  }
  return dst;
}

export function ensureGrad(n) {
  if (!n.g) n.g = new Float32Array(n.d.length);
  return n.g;
}

export function topo(root) {
  const order = [];
  const seen = new Set();
  const done = new Set();
  const stack = [[root, false]];
  while (stack.length) {
    const frame = stack.pop();
    const n = frame[0];
    if (frame[1]) {
      if (!done.has(n)) {
        done.add(n);
        order.push(n);
      }
      continue;
    }
    if (seen.has(n)) continue;
    seen.add(n);
    stack.push([n, true]);
    for (let i = n.parents.length - 1; i >= 0; i--) stack.push([n.parents[i], false]);
  }
  return order;
}

export function backward(root, seed) {
  const order = topo(root);
  root.g = seed ? Float32Array.from(seed) : new Float32Array(root.d.length).fill(1);
  for (let i = order.length - 1; i >= 0; i--) {
    const n = order[i];
    if (n.bwd && n.g) {
      n.bwd(n.g);
    }
  }
  for (let i = 0; i < order.length; i++) {
    order[i].bwd = null;
  }
}

export function gradNorm(nodes) {
  let s = 0;
  for (const n of nodes) {
    if (!n.g) continue;
    for (let i = 0; i < n.g.length; i++) s += n.g[i] * n.g[i];
  }
  return Math.sqrt(s);
}

export function clipGrad(nodes, maxNorm) {
  const gn = gradNorm(nodes);
  if (!(gn > 0) || !isFinite(gn)) {
    for (const n of nodes) if (n.g) n.g.fill(0);
    return gn;
  }
  if (gn > maxNorm) {
    const scale = maxNorm / gn;
    for (const n of nodes) {
      if (!n.g) continue;
      for (let i = 0; i < n.g.length; i++) n.g[i] *= scale;
    }
  }
  return gn;
}

export function countParams(nodes) {
  let c = 0;
  for (const n of nodes) c += n.d.length;
  return c;
}

export function reduceSumToShape(gy, targetShape, outShape) {
  const tSize = shapeSize(targetShape);
  const oSize = shapeSize(outShape);
  if (tSize === oSize) return gy;
  const out = new Float32Array(tSize);
  if (tSize === 1) {
    let s = 0;
    for (let i = 0; i < gy.length; i++) s += gy[i];
    out[0] = s;
    return out;
  }
  const tr = targetShape.length;
  const or = outShape.length;
  const pad = or - tr;
  const tStr = strides(targetShape);
  const oStr = strides(outShape);
  const idx = new Array(or).fill(0);
  for (let flat = 0; flat < oSize; flat++) {
    let rem = flat;
    for (let a = 0; a < or; a++) {
      idx[a] = Math.floor(rem / oStr[a]);
      rem -= idx[a] * oStr[a];
    }
    let tOff = 0;
    for (let a = 0; a < tr; a++) {
      const dim = pad + a;
      if (targetShape[a] === 1) continue;
      tOff += idx[dim] * tStr[a];
    }
    out[tOff] += gy[flat];
  }
  return out;
}

export function broadcastShape(a, b) {
  const rank = Math.max(a.length, b.length);
  const out = new Array(rank);
  for (let i = 0; i < rank; i++) {
    const da = i < rank - a.length ? 1 : a[i - (rank - a.length)];
    const db = i < rank - b.length ? 1 : b[i - (rank - b.length)];
    if (da !== db && da !== 1 && db !== 1) {
      throw new Error('shape mismatch ' + a.join('x') + ' vs ' + b.join('x'));
    }
    out[i] = Math.max(da, db);
  }
  return out;
}
