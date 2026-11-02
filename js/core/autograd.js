import {
  Node, tensor, shapeSize, strides, gemm, ensureGrad, addInto, broadcastShape
} from './tensor.js';

function mk(d, shape, parents, bwd, op) {
  const n = new Node(d, shape);
  n.parents = parents;
  n.bwd = bwd;
  n.op = op || '';
  return n;
}

function alignedStrides(outShape, inShape) {
  const rank = outShape.length;
  const pad = rank - inShape.length;
  const st = strides(inShape);
  const out = new Array(rank).fill(0);
  for (let i = 0; i < inShape.length; i++) {
    out[pad + i] = inShape[i] === 1 ? 0 : st[i];
  }
  return out;
}

function broadcastLoop(outShape, sa, sb, fn) {
  const size = shapeSize(outShape);
  const st = strides(outShape);
  const rank = outShape.length;
  const d = new Float32Array(size);
  const idx = new Array(rank).fill(0);
  for (let flat = 0; flat < size; flat++) {
    let rem = flat;
    for (let a = 0; a < rank; a++) {
      idx[a] = Math.floor(rem / st[a]);
      rem -= idx[a] * st[a];
    }
    let oa = 0;
    let ob = 0;
    for (let a = 0; a < rank; a++) {
      oa += idx[a] * sa[a];
      ob += idx[a] * sb[a];
    }
    d[flat] = fn(oa, ob, flat);
  }
  return d;
}

function asNode(v) {
  if (v instanceof Node) return v;
  if (typeof v === 'number') return new Node(new Float32Array([v]), [1]);
  return tensor(v);
}

function binary(a, b, f, dfda, dfdb, op) {
  a = asNode(a);
  b = asNode(b);
  const outShape = broadcastShape(a.shape, b.shape);
  const sameShape = a.shape.length === b.shape.length && a.size === b.size &&
    outShape.length === a.shape.length && a.shape.every((v, i) => v === b.shape[i]);
  const scalarB = b.size === 1;
  const scalarA = a.size === 1;
  let d;
  if (sameShape) {
    const n = a.d.length;
    d = new Float32Array(n);
    for (let i = 0; i < n; i++) d[i] = f(a.d[i], b.d[i]);
  } else if (scalarB) {
    const bv = b.d[0];
    const n = a.d.length;
    d = new Float32Array(n);
    for (let i = 0; i < n; i++) d[i] = f(a.d[i], bv);
  } else if (scalarA) {
    const av = a.d[0];
    const n = b.d.length;
    d = new Float32Array(n);
    for (let i = 0; i < n; i++) d[i] = f(av, b.d[i]);
  } else {
    const biasCase = b.shape.length === 1 && a.shape[a.shape.length - 1] === b.shape[0] &&
      outShape.length === a.shape.length;
    if (biasCase) {
      const K = b.shape[0];
      const n = a.d.length;
      d = new Float32Array(n);
      for (let i = 0; i < n; i++) d[i] = f(a.d[i], b.d[i % K]);
    } else {
      const sa = alignedStrides(outShape, a.shape);
      const sb = alignedStrides(outShape, b.shape);
      d = broadcastLoop(outShape, sa, sb, (oa, ob) => f(a.d[oa], b.d[ob]));
    }
  }
  const out = mk(d, outShape, [a, b], null, op);
  out.bwd = (gy) => {
    if (dfda) a.g = addInto(ensureGrad(a), gradOf(a, b, gy, dfda, outShape, sameShape, scalarA, scalarB, false));
    if (dfdb) b.g = addInto(ensureGrad(b), gradOf(b, a, gy, dfdb, outShape, sameShape, scalarB, scalarA, true));
  };
  return out;
}

function gradOf(target, other, gy, df, outShape, sameShape, targetScalar, otherScalar, swap) {
  const apply = targetScalar || otherScalar || !swap
    ? (tv, ov) => (swap ? df(ov, tv) : df(tv, ov))
    : (tv, ov) => df(ov, tv);
  if (sameShape) {
    const n = gy.length;
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = gy[i] * apply(target.d[i], other.d[i]);
    return out;
  }
  if (targetScalar) {
    const tv = target.d[0];
    if (otherScalar) return new Float32Array([gy.length ? gy[0] * apply(tv, other.d[0]) : 0]);
    let s = 0;
    for (let i = 0; i < gy.length; i++) s += gy[i] * apply(tv, other.d[i % other.d.length]);
    return new Float32Array([s]);
  }
  if (otherScalar) {
    const ov = other.d[0];
    const n = gy.length;
    const out = new Float32Array(target.d.length);
    for (let i = 0; i < n; i++) out[i] = gy[i] * apply(target.d[i], ov);
    return out;
  }
  if (target.shape.length === 1 && outShape[outShape.length - 1] === target.shape[0] &&
    shapeSize(outShape) === other.d.length) {
    const K = target.shape[0];
    const out = new Float32Array(K);
    for (let i = 0; i < gy.length; i++) out[i % K] += gy[i] * apply(target.d[i % K], other.d[i]);
    return out;
  }
  const st = alignedStrides(outShape, target.shape);
  const so = alignedStrides(outShape, other.shape);
  const rank = outShape.length;
  const os = strides(outShape);
  const idx = new Array(rank).fill(0);
  const out = new Float32Array(target.d.length);
  for (let flat = 0; flat < gy.length; flat++) {
    let rem = flat;
    for (let r = 0; r < rank; r++) {
      idx[r] = Math.floor(rem / os[r]);
      rem -= idx[r] * os[r];
    }
    let to = 0;
    let oo = 0;
    for (let r = 0; r < rank; r++) {
      to += idx[r] * st[r];
      oo += idx[r] * so[r];
    }
    out[to] += gy[flat] * apply(target.d[to], other.d[oo]);
  }
  return out;
}

export function add(a, b) {
  return binary(a, b, (x, y) => x + y, () => 1, () => 1, 'add');
}

export function sub(a, b) {
  return binary(a, b, (x, y) => x - y, () => 1, () => -1, 'sub');
}

export function mul(a, b) {
  return binary(a, b, (x, y) => x * y, (x, y) => y, (x, y) => x, 'mul');
}

export function div(a, b) {
  return binary(a, b, (x, y) => x / y, (x, y) => 1 / y, (x, y) => -x / (y * y), 'div');
}

export function neg(a) {
  return mul(a, new Node(new Float32Array([-1]), [1]));
}

export function scalar(v) {
  return new Node(new Float32Array([v]), [1]);
}

export function relu(a) {
  const n = a.d.length;
  const d = new Float32Array(n);
  for (let i = 0; i < n; i++) d[i] = a.d[i] > 0 ? a.d[i] : 0;
  const out = mk(d, a.shape, [a], null, 'relu');
  out.bwd = (gy) => {
    const g = new Float32Array(n);
    for (let i = 0; i < n; i++) g[i] = a.d[i] > 0 ? gy[i] : 0;
    a.g = addInto(ensureGrad(a), g);
  };
  return out;
}

export function leakyRelu(a, alpha) {
  alpha = alpha === undefined ? 0.01 : alpha;
  const n = a.d.length;
  const d = new Float32Array(n);
  for (let i = 0; i < n; i++) d[i] = a.d[i] > 0 ? a.d[i] : alpha * a.d[i];
  const out = mk(d, a.shape, [a], null, 'leaky_relu');
  out.bwd = (gy) => {
    const g = new Float32Array(n);
    for (let i = 0; i < n; i++) g[i] = a.d[i] > 0 ? gy[i] : alpha * gy[i];
    a.g = addInto(ensureGrad(a), g);
  };
  return out;
}

export function elu(a, alpha) {
  alpha = alpha === undefined ? 1 : alpha;
  const n = a.d.length;
  const d = new Float32Array(n);
  for (let i = 0; i < n; i++) d[i] = a.d[i] > 0 ? a.d[i] : alpha * (Math.exp(a.d[i]) - 1);
  const out = mk(d, a.shape, [a], null, 'elu');
  out.bwd = (gy) => {
    const g = new Float32Array(n);
    for (let i = 0; i < n; i++) g[i] = a.d[i] > 0 ? gy[i] : gy[i] * (d[i] + alpha);
    a.g = addInto(ensureGrad(a), g);
  };
  return out;
}

export function gelu(a) {
  const n = a.d.length;
  const d = new Float32Array(n);
  const k = Math.sqrt(2 / Math.PI);
  for (let i = 0; i < n; i++) {
    const x = a.d[i];
    d[i] = 0.5 * x * (1 + Math.tanh(k * (x + 0.044715 * x * x * x)));
  }
  const out = mk(d, a.shape, [a], null, 'gelu');
  out.bwd = (gy) => {
    const g = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const x = a.d[i];
      const t = Math.tanh(k * (x + 0.044715 * x * x * x));
      const dt = k * (1 + 3 * 0.044715 * x * x) * (1 - t * t);
      g[i] = gy[i] * (0.5 * (1 + t) + 0.5 * x * dt);
    }
    a.g = addInto(ensureGrad(a), g);
  };
  return out;
}

export function sigmoid(a) {
  const n = a.d.length;
  const d = new Float32Array(n);
  for (let i = 0; i < n; i++) d[i] = 1 / (1 + Math.exp(-a.d[i]));
  const out = mk(d, a.shape, [a], null, 'sigmoid');
  out.bwd = (gy) => {
    const g = new Float32Array(n);
    for (let i = 0; i < n; i++) g[i] = gy[i] * d[i] * (1 - d[i]);
    a.g = addInto(ensureGrad(a), g);
  };
  return out;
}

export function tanh(a) {
  const n = a.d.length;
  const d = new Float32Array(n);
  for (let i = 0; i < n; i++) d[i] = Math.tanh(a.d[i]);
  const out = mk(d, a.shape, [a], null, 'tanh');
  out.bwd = (gy) => {
    const g = new Float32Array(n);
    for (let i = 0; i < n; i++) g[i] = gy[i] * (1 - d[i] * d[i]);
    a.g = addInto(ensureGrad(a), g);
  };
  return out;
}

export function softplus(a) {
  const n = a.d.length;
  const d = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const x = a.d[i];
    d[i] = x > 30 ? x : Math.log1p(Math.exp(x));
  }
  const out = mk(d, a.shape, [a], null, 'softplus');
  out.bwd = (gy) => {
    const g = new Float32Array(n);
    for (let i = 0; i < n; i++) g[i] = gy[i] / (1 + Math.exp(-a.d[i]));
    a.g = addInto(ensureGrad(a), g);
  };
  return out;
}

export function hardSigmoid(a) {
  const n = a.d.length;
  const d = new Float32Array(n);
  for (let i = 0; i < n; i++) d[i] = Math.min(1, Math.max(0, a.d[i] * 0.2 + 0.5));
  const out = mk(d, a.shape, [a], null, 'hard_sigmoid');
  out.bwd = (gy) => {
    const g = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const v = a.d[i] * 0.2 + 0.5;
      g[i] = v > 0 && v < 1 ? gy[i] * 0.2 : 0;
    }
    a.g = addInto(ensureGrad(a), g);
  };
  return out;
}

export function applyActivation(a, name) {
  switch (name) {
    case 'relu': return relu(a);
    case 'leaky_relu': return leakyRelu(a, 0.1);
    case 'elu': return elu(a);
    case 'gelu': return gelu(a);
    case 'sigmoid': return sigmoid(a);
    case 'tanh': return tanh(a);
    case 'softplus': return softplus(a);
    case 'hard_sigmoid': return hardSigmoid(a);
    case 'linear':
    case 'none':
    default: return a;
  }
}

export function exp(a) {
  const n = a.d.length;
  const d = new Float32Array(n);
  for (let i = 0; i < n; i++) d[i] = Math.exp(a.d[i]);
  const out = mk(d, a.shape, [a], null, 'exp');
  out.bwd = (gy) => {
    const g = new Float32Array(n);
    for (let i = 0; i < n; i++) g[i] = gy[i] * d[i];
    a.g = addInto(ensureGrad(a), g);
  };
  return out;
}

export function log(a, eps) {
  eps = eps || 1e-12;
  const n = a.d.length;
  const d = new Float32Array(n);
  for (let i = 0; i < n; i++) d[i] = Math.log(Math.max(a.d[i], eps));
  const out = mk(d, a.shape, [a], null, 'log');
  out.bwd = (gy) => {
    const g = new Float32Array(n);
    for (let i = 0; i < n; i++) g[i] = gy[i] / Math.max(a.d[i], eps);
    a.g = addInto(ensureGrad(a), g);
  };
  return out;
}

export function sqrt(a) {
  const n = a.d.length;
  const d = new Float32Array(n);
  for (let i = 0; i < n; i++) d[i] = Math.sqrt(Math.max(a.d[i], 0));
  const out = mk(d, a.shape, [a], null, 'sqrt');
  out.bwd = (gy) => {
    const g = new Float32Array(n);
    for (let i = 0; i < n; i++) g[i] = d[i] > 0 ? gy[i] / (2 * d[i]) : 0;
    a.g = addInto(ensureGrad(a), g);
  };
  return out;
}

export function abs(a) {
  const n = a.d.length;
  const d = new Float32Array(n);
  for (let i = 0; i < n; i++) d[i] = Math.abs(a.d[i]);
  const out = mk(d, a.shape, [a], null, 'abs');
  out.bwd = (gy) => {
    const g = new Float32Array(n);
    for (let i = 0; i < n; i++) g[i] = gy[i] * (a.d[i] >= 0 ? 1 : -1);
    a.g = addInto(ensureGrad(a), g);
  };
  return out;
}

export function square(a) {
  const n = a.d.length;
  const d = new Float32Array(n);
  for (let i = 0; i < n; i++) d[i] = a.d[i] * a.d[i];
  const out = mk(d, a.shape, [a], null, 'square');
  out.bwd = (gy) => {
    const g = new Float32Array(n);
    for (let i = 0; i < n; i++) g[i] = gy[i] * 2 * a.d[i];
    a.g = addInto(ensureGrad(a), g);
  };
  return out;
}

export function softmax(a, axis) {
  axis = axis === undefined ? a.shape.length - 1 : axis;
  if (axis < 0) axis += a.shape.length;
  const st = strides(a.shape);
  const outer = shapeSize(a.shape.slice(0, axis));
  const dim = a.shape[axis];
  const inner = shapeSize(a.shape.slice(axis + 1));
  const d = new Float32Array(a.d.length);
  for (let o = 0; o < outer; o++) {
    for (let i = 0; i < inner; i++) {
      let max = -Infinity;
      for (let k = 0; k < dim; k++) {
        const idx = (o * dim + k) * inner + i;
        if (a.d[idx] > max) max = a.d[idx];
      }
      let sum = 0;
      for (let k = 0; k < dim; k++) {
        const idx = (o * dim + k) * inner + i;
        const e = Math.exp(a.d[idx] - max);
        d[idx] = e;
        sum += e;
      }
      for (let k = 0; k < dim; k++) {
        const idx = (o * dim + k) * inner + i;
        d[idx] /= sum;
      }
    }
  }
  const out = mk(d, a.shape, [a], null, 'softmax');
  out.bwd = (gy) => {
    const g = new Float32Array(a.d.length);
    for (let o = 0; o < outer; o++) {
      for (let i = 0; i < inner; i++) {
        let dot = 0;
        for (let k = 0; k < dim; k++) {
          const idx = (o * dim + k) * inner + i;
          dot += gy[idx] * d[idx];
        }
        for (let k = 0; k < dim; k++) {
          const idx = (o * dim + k) * inner + i;
          g[idx] = d[idx] * (gy[idx] - dot);
        }
      }
    }
    a.g = addInto(ensureGrad(a), g);
  };
  return out;
}

export function logSoftmax(a, axis) {
  const sm = softmax(a, axis);
  return log(sm, 1e-9);
}

export function sumAll(a) {
  let s = 0;
  for (let i = 0; i < a.d.length; i++) s += a.d[i];
  const out = mk(new Float32Array([s]), [1], [a], null, 'sum');
  out.bwd = (gy) => {
    const g = new Float32Array(a.d.length).fill(gy[0]);
    a.g = addInto(ensureGrad(a), g);
  };
  return out;
}

export function meanAll(a) {
  let s = 0;
  for (let i = 0; i < a.d.length; i++) s += a.d[i];
  const n = a.d.length;
  const out = mk(new Float32Array([s / n]), [1], [a], null, 'mean');
  out.bwd = (gy) => {
    const g = new Float32Array(n).fill(gy[0] / n);
    a.g = addInto(ensureGrad(a), g);
  };
  return out;
}

export function sumAxis(a, axis, keep) {
  axis = axis < 0 ? axis + a.shape.length : axis;
  const st = strides(a.shape);
  const outer = shapeSize(a.shape.slice(0, axis));
  const dim = a.shape[axis];
  const inner = shapeSize(a.shape.slice(axis + 1));
  const outShape = keep
    ? a.shape.map((v, i) => (i === axis ? 1 : v))
    : a.shape.filter((v, i) => i !== axis);
  const d = new Float32Array(shapeSize(outShape.length ? outShape : [1]));
  for (let o = 0; o < outer; o++) {
    for (let i = 0; i < inner; i++) {
      let s = 0;
      for (let k = 0; k < dim; k++) s += a.d[(o * dim + k) * inner + i];
      d[o * inner + i] = s;
    }
  }
  const out = mk(d, outShape.length ? outShape : [1], [a], null, 'sum_axis');
  out.bwd = (gy) => {
    const g = new Float32Array(a.d.length);
    for (let o = 0; o < outer; o++) {
      for (let i = 0; i < inner; i++) {
        const gv = gy[o * inner + i];
        for (let k = 0; k < dim; k++) g[(o * dim + k) * inner + i] = gv;
      }
    }
    a.g = addInto(ensureGrad(a), g);
  };
  return out;
}

export function meanAxis(a, axis, keep) {
  const dim = a.shape[axis < 0 ? axis + a.shape.length : axis];
  return mul(sumAxis(a, axis, keep), scalar(1 / dim));
}

export function maxAxis(a, axis, keep) {
  axis = axis < 0 ? axis + a.shape.length : axis;
  const outer = shapeSize(a.shape.slice(0, axis));
  const dim = a.shape[axis];
  const inner = shapeSize(a.shape.slice(axis + 1));
  const outShape = keep ? a.shape.map((v, i) => (i === axis ? 1 : v)) : a.shape.filter((v, i) => i !== axis);
  const d = new Float32Array(shapeSize(outShape.length ? outShape : [1]));
  const arg = new Int32Array(d.length);
  for (let o = 0; o < outer; o++) {
    for (let i = 0; i < inner; i++) {
      let best = -Infinity;
      let bi = 0;
      for (let k = 0; k < dim; k++) {
        const v = a.d[(o * dim + k) * inner + i];
        if (v > best) {
          best = v;
          bi = k;
        }
      }
      d[o * inner + i] = best;
      arg[o * inner + i] = bi;
    }
  }
  const out = mk(d, outShape.length ? outShape : [1], [a], null, 'max_axis');
  out.bwd = (gy) => {
    const g = new Float32Array(a.d.length);
    for (let o = 0; o < outer; o++) {
      for (let i = 0; i < inner; i++) {
        g[(o * dim + arg[o * inner + i]) * inner + i] = gy[o * inner + i];
      }
    }
    a.g = addInto(ensureGrad(a), g);
  };
  return out;
}

export function reshape(a, shape) {
  const out = mk(a.d, shape, [a], null, 'reshape');
  out.bwd = (gy) => {
    a.g = addInto(ensureGrad(a), gy);
  };
  return out;
}

export function flatten(a) {
  const b = a.shape[0];
  return reshape(a, [b, a.d.length / b]);
}

export function transpose2d(a) {
  const R = a.shape[0];
  const C = a.shape[1];
  const d = new Float32Array(R * C);
  for (let i = 0; i < R; i++) {
    for (let j = 0; j < C; j++) d[j * R + i] = a.d[i * C + j];
  }
  const out = mk(d, [C, R], [a], null, 'transpose');
  out.bwd = (gy) => {
    const g = new Float32Array(R * C);
    for (let j = 0; j < C; j++) {
      for (let i = 0; i < R; i++) g[i * C + j] = gy[j * R + i];
    }
    a.g = addInto(ensureGrad(a), g);
  };
  return out;
}

export function concat(nodes, axis) {
  axis = axis || 0;
  const outShape = nodes[0].shape.slice();
  outShape[axis] = nodes.reduce((s, n) => s + n.shape[axis], 0);
  const size = shapeSize(outShape);
  const d = new Float32Array(size);
  const offsets = [];
  let off = 0;
  for (const n of nodes) {
    offsets.push(off);
    off += n.shape[axis];
  }
  const pre = shapeSize(outShape.slice(0, axis));
  const post = shapeSize(outShape.slice(axis + 1));
  const total = outShape[axis];
  for (let p = 0; p < pre; p++) {
    for (let k = 0; k < nodes.length; k++) {
      const n = nodes[k];
      const len = n.shape[axis] * post;
      d.set(n.d.subarray(p * len, (p + 1) * len), (p * total + offsets[k]) * post);
    }
  }
  const out = mk(d, outShape, nodes, null, 'concat');
  out.bwd = (gy) => {
    for (let k = 0; k < nodes.length; k++) {
      const n = nodes[k];
      const len = n.shape[axis] * post;
      const g = new Float32Array(n.d.length);
      for (let p = 0; p < pre; p++) {
        g.set(gy.subarray((p * total + offsets[k]) * post, (p * total + offsets[k]) * post + len), p * len);
      }
      n.g = addInto(ensureGrad(n), g);
    }
  };
  return out;
}

export function sliceRows(a, start, count) {
  const C = a.d.length / a.shape[0];
  const d = a.d.subarray(start * C, (start + count) * C);
  const outShape = a.shape.slice();
  outShape[0] = count;
  const out = mk(new Float32Array(d), outShape, [a], null, 'slice');
  out.bwd = (gy) => {
    const g = new Float32Array(a.d.length);
    g.set(gy, start * C);
    a.g = addInto(ensureGrad(a), g);
  };
  return out;
}

export function matmul(a, b) {
  const M = a.shape[0];
  const K = a.shape[1];
  const N = b.shape[1];
  const d = new Float32Array(M * N);
  gemm(a.d, b.d, d, M, N, K, 1, 0, false, false);
  const out = mk(d, [M, N], [a, b], null, 'matmul');
  out.bwd = (gy) => {
    const ga = new Float32Array(M * K);
    gemm(gy, b.d, ga, M, K, N, 1, 0, false, true);
    const gb = new Float32Array(K * N);
    gemm(a.d, gy, gb, K, N, M, 1, 0, true, false);
    a.g = addInto(ensureGrad(a), ga);
    b.g = addInto(ensureGrad(b), gb);
  };
  return out;
}

export function linear(x, W, b) {
  const y = matmul(x, W);
  return b ? add(y, b) : y;
}

export function embedding(ids, W) {
  const V = W.shape[0];
  const D = W.shape[1];
  const B = ids.shape[0];
  const T = ids.shape.length > 1 ? ids.shape[1] : 1;
  const d = new Float32Array(B * T * D);
  const flatIds = ids.data || ids.d;
  for (let i = 0; i < B * T; i++) {
    const id = Math.max(0, Math.min(V - 1, flatIds[i] | 0));
    d.set(W.d.subarray(id * D, id * D + D), i * D);
  }
  const out = mk(d, [B, T, D], [W], null, 'embedding');
  out.bwd = (gy) => {
    const g = new Float32Array(V * D);
    for (let i = 0; i < B * T; i++) {
      const id = Math.max(0, Math.min(V - 1, flatIds[i] | 0));
      for (let j = 0; j < D; j++) g[id * D + j] += gy[i * D + j];
    }
    W.g = addInto(ensureGrad(W), g);
  };
  return out;
}

export function dropout(a, rate, training, rng) {
  if (!training || rate <= 0) return a;
  const n = a.d.length;
  const mask = new Float32Array(n);
  const keep = 1 - rate;
  const scale = 1 / keep;
  for (let i = 0; i < n; i++) mask[i] = rng.next() < keep ? scale : 0;
  const d = new Float32Array(n);
  for (let i = 0; i < n; i++) d[i] = a.d[i] * mask[i];
  const out = mk(d, a.shape, [a], null, 'dropout');
  out.bwd = (gy) => {
    const g = new Float32Array(n);
    for (let i = 0; i < n; i++) g[i] = gy[i] * mask[i];
    a.g = addInto(ensureGrad(a), g);
  };
  return out;
}

function pool2d(a, k, s, pad, mode) {
  const N = a.shape[0];
  const C = a.shape[1];
  const H = a.shape[2];
  const W = a.shape[3];
  const kH = Array.isArray(k) ? k[0] : k;
  const kW = Array.isArray(k) ? k[1] : k;
  const sH = Array.isArray(s) ? s[0] : s;
  const sW = Array.isArray(s) ? s[1] : s;
  const pH = Array.isArray(pad) ? pad[0] : pad || 0;
  const pW = Array.isArray(pad) ? pad[1] : pad || 0;
  const oH = Math.floor((H + 2 * pH - kH) / sH) + 1;
  const oW = Math.floor((W + 2 * pW - kW) / sW) + 1;
  const d = new Float32Array(N * C * oH * oW);
  const argMax = mode === 'max' ? new Int32Array(d.length) : null;
  for (let n = 0; n < N; n++) {
    for (let c = 0; c < C; c++) {
      for (let oh = 0; oh < oH; oh++) {
        for (let ow = 0; ow < oW; ow++) {
          let best = mode === 'max' ? -Infinity : 0;
          let cnt = 0;
          let bi = -1;
          for (let kh = 0; kh < kH; kh++) {
            const ih = oh * sH - pH + kh;
            if (ih < 0 || ih >= H) continue;
            for (let kw = 0; kw < kW; kw++) {
              const iw = ow * sW - pW + kw;
              if (iw < 0 || iw >= W) continue;
              const v = a.d[((n * C + c) * H + ih) * W + iw];
              cnt++;
              if (mode === 'max') {
                if (v > best) {
                  best = v;
                  bi = ih * W + iw;
                }
              } else {
                best += v;
              }
            }
          }
          const oi = ((n * C + c) * oH + oh) * oW + ow;
          if (mode === 'max') {
            d[oi] = cnt ? best : 0;
            argMax[oi] = bi;
          } else {
            d[oi] = cnt ? best / (kH * kW) : 0;
          }
        }
      }
    }
  }
  const out = mk(d, [N, C, oH, oW], [a], null, mode + 'pool2d');
  out.bwd = (gy) => {
    const g = new Float32Array(a.d.length);
    for (let n = 0; n < N; n++) {
      for (let c = 0; c < C; c++) {
        for (let oh = 0; oh < oH; oh++) {
          for (let ow = 0; ow < oW; ow++) {
            const oi = ((n * C + c) * oH + oh) * oW + ow;
            const gv = gy[oi];
            if (mode === 'max') {
              if (argMax[oi] >= 0) g[((n * C + c) * H) * W + argMax[oi]] += gv;
            } else {
              const each = gv / (kH * kW);
              for (let kh = 0; kh < kH; kh++) {
                const ih = oh * sH - pH + kh;
                if (ih < 0 || ih >= H) continue;
                for (let kw = 0; kw < kW; kw++) {
                  const iw = ow * sW - pW + kw;
                  if (iw < 0 || iw >= W) continue;
                  g[((n * C + c) * H + ih) * W + iw] += each;
                }
              }
            }
          }
        }
      }
    }
    a.g = addInto(ensureGrad(a), g);
  };
  return out;
}

export function maxPool2d(a, k, s, pad) {
  return pool2d(a, k, s === undefined ? k : s, pad || 0, 'max');
}

export function avgPool2d(a, k, s, pad) {
  return pool2d(a, k, s === undefined ? k : s, pad || 0, 'avg');
}

export function globalAvgPool2d(a) {
  const N = a.shape[0];
  const C = a.shape[1];
  const HW = a.shape[2] * a.shape[3];
  const d = new Float32Array(N * C);
  for (let n = 0; n < N; n++) {
    for (let c = 0; c < C; c++) {
      let s = 0;
      const base = (n * C + c) * HW;
      for (let i = 0; i < HW; i++) s += a.d[base + i];
      d[n * C + c] = s / HW;
    }
  }
  const out = mk(d, [N, C, 1, 1], [a], null, 'global_avg_pool2d');
  out.bwd = (gy) => {
    const g = new Float32Array(a.d.length);
    for (let n = 0; n < N; n++) {
      for (let c = 0; c < C; c++) {
        const gv = gy[n * C + c] / HW;
        const base = (n * C + c) * HW;
        for (let i = 0; i < HW; i++) g[base + i] = gv;
      }
    }
    a.g = addInto(ensureGrad(a), g);
  };
  return out;
}

export function globalMaxPool1d(a) {
  const N = a.shape[0];
  const C = a.shape[1];
  const L = a.shape[2];
  const d = new Float32Array(N * C);
  const arg = new Int32Array(N * C);
  for (let n = 0; n < N; n++) {
    for (let c = 0; c < C; c++) {
      let best = -Infinity;
      let bi = 0;
      const base = (n * C + c) * L;
      for (let i = 0; i < L; i++) {
        if (a.d[base + i] > best) {
          best = a.d[base + i];
          bi = i;
        }
      }
      d[n * C + c] = best;
      arg[n * C + c] = bi;
    }
  }
  const out = mk(d, [N, C], [a], null, 'global_max_pool1d');
  out.bwd = (gy) => {
    const g = new Float32Array(a.d.length);
    for (let n = 0; n < N; n++) {
      for (let c = 0; c < C; c++) g[(n * C + c) * L + arg[n * C + c]] = gy[n * C + c];
    }
    a.g = addInto(ensureGrad(a), g);
  };
  return out;
}

export function globalAvgPool1d(a) {
  const N = a.shape[0];
  const C = a.shape[1];
  const L = a.shape[2];
  const d = new Float32Array(N * C);
  for (let n = 0; n < N; n++) {
    for (let c = 0; c < C; c++) {
      let s = 0;
      const base = (n * C + c) * L;
      for (let i = 0; i < L; i++) s += a.d[base + i];
      d[n * C + c] = s / L;
    }
  }
  const out = mk(d, [N, C], [a], null, 'global_avg_pool1d');
  out.bwd = (gy) => {
    const g = new Float32Array(a.d.length);
    for (let n = 0; n < N; n++) {
      for (let c = 0; c < C; c++) {
        const gv = gy[n * C + c] / L;
        const base = (n * C + c) * L;
        for (let i = 0; i < L; i++) g[base + i] = gv;
      }
    }
    a.g = addInto(ensureGrad(a), g);
  };
  return out;
}

export function maxPool1d(a, k, s, pad) {
  k = k || 2;
  s = s === undefined ? k : s;
  pad = pad || 0;
  const N = a.shape[0];
  const C = a.shape[1];
  const L = a.shape[2];
  const oL = Math.floor((L + 2 * pad - k) / s) + 1;
  const d = new Float32Array(N * C * oL);
  const arg = new Int32Array(d.length);
  for (let n = 0; n < N; n++) {
    for (let c = 0; c < C; c++) {
      for (let o = 0; o < oL; o++) {
        let best = -Infinity;
        let bi = -1;
        for (let j = 0; j < k; j++) {
          const ii = o * s - pad + j;
          if (ii < 0 || ii >= L) continue;
          const v = a.d[(n * C + c) * L + ii];
          if (v > best) {
            best = v;
            bi = ii;
          }
        }
        const oi = (n * C + c) * oL + o;
        d[oi] = bi >= 0 ? best : 0;
        arg[oi] = bi;
      }
    }
  }
  const out = mk(d, [N, C, oL], [a], null, 'maxpool1d');
  out.bwd = (gy) => {
    const g = new Float32Array(a.d.length);
    for (let i = 0; i < gy.length; i++) {
      if (arg[i] >= 0) g[Math.floor(i / oL) * L + arg[i]] += gy[i];
    }
    a.g = addInto(ensureGrad(a), g);
  };
  return out;
}

export function avgPool1d(a, k, s, pad) {
  k = k || 2;
  s = s === undefined ? k : s;
  pad = pad || 0;
  const N = a.shape[0];
  const C = a.shape[1];
  const L = a.shape[2];
  const oL = Math.floor((L + 2 * pad - k) / s) + 1;
  const d = new Float32Array(N * C * oL);
  for (let n = 0; n < N; n++) {
    for (let c = 0; c < C; c++) {
      for (let o = 0; o < oL; o++) {
        let sum = 0;
        for (let j = 0; j < k; j++) {
          const ii = o * s - pad + j;
          if (ii < 0 || ii >= L) continue;
          sum += a.d[(n * C + c) * L + ii];
        }
        d[(n * C + c) * oL + o] = sum / k;
      }
    }
  }
  const out = mk(d, [N, C, oL], [a], null, 'avgpool1d');
  out.bwd = (gy) => {
    const g = new Float32Array(a.d.length);
    for (let n = 0; n < N; n++) {
      for (let c = 0; c < C; c++) {
        for (let o = 0; o < oL; o++) {
          const each = gy[(n * C + c) * oL + o] / k;
          for (let j = 0; j < k; j++) {
            const ii = o * s - pad + j;
            if (ii < 0 || ii >= L) continue;
            g[(n * C + c) * L + ii] += each;
          }
        }
      }
    }
    a.g = addInto(ensureGrad(a), g);
  };
  return out;
}

function im2col(a, kh, kw, sh, sw, ph, pw, oH, oW, dh, dw) {
  dh = dh || 1;
  dw = dw || 1;
  const N = a.shape[0];
  const C = a.shape[1];
  const H = a.shape[2];
  const W = a.shape[3];
  const cols = new Float32Array(N * oH * oW * C * kh * kw);
  let p = 0;
  for (let n = 0; n < N; n++) {
    for (let oh = 0; oh < oH; oh++) {
      for (let ow = 0; ow < oW; ow++) {
        for (let c = 0; c < C; c++) {
          for (let i = 0; i < kh; i++) {
            const ih = oh * sh - ph + i * dh;
            for (let j = 0; j < kw; j++) {
              const iw = ow * sw - pw + j * dw;
              if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                cols[p] = a.d[((n * C + c) * H + ih) * W + iw];
              }
              p++;
            }
          }
        }
      }
    }
  }
  return cols;
}

function col2im(cols, shape, kh, kw, sh, sw, ph, pw, oH, oW, dh, dw) {
  dh = dh || 1;
  dw = dw || 1;
  const N = shape[0];
  const C = shape[1];
  const H = shape[2];
  const W = shape[3];
  const g = new Float32Array(N * C * H * W);
  let p = 0;
  for (let n = 0; n < N; n++) {
    for (let oh = 0; oh < oH; oh++) {
      for (let ow = 0; ow < oW; ow++) {
        for (let c = 0; c < C; c++) {
          for (let i = 0; i < kh; i++) {
            const ih = oh * sh - ph + i * dh;
            for (let j = 0; j < kw; j++) {
              const iw = ow * sw - pw + j * dw;
              const v = cols[p++];
              if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                g[((n * C + c) * H + ih) * W + iw] += v;
              }
            }
          }
        }
      }
    }
  }
  return g;
}

export function conv2d(a, W, b, opts) {
  opts = opts || {};
  const stride = opts.stride || 1;
  const pad = opts.pad || 0;
  const dilationOpt = opts.dilation || 1;
  const dh = Array.isArray(dilationOpt) ? dilationOpt[0] : dilationOpt;
  const dw = Array.isArray(dilationOpt) ? dilationOpt[1] : dilationOpt;
  const groups = opts.groups || 1;
  const sh = Array.isArray(stride) ? stride[0] : stride;
  const sw = Array.isArray(stride) ? stride[1] : stride;
  const ph = Array.isArray(pad) ? pad[0] : pad;
  const pw = Array.isArray(pad) ? pad[1] : pad;
  const N = a.shape[0];
  const Cin = a.shape[1];
  const H = a.shape[2];
  const Wd = a.shape[3];
  const Cout = W.shape[0];
  const kh = W.shape[2];
  const kw = W.shape[3];
  const ekh = (kh - 1) * dh + 1;
  const ekw = (kw - 1) * dw + 1;
  const oH = Math.floor((H + 2 * ph - ekh) / sh) + 1;
  const oW = Math.floor((Wd + 2 * pw - ekw) / sw) + 1;
  const K = (Cin / groups) * kh * kw;
  const d = new Float32Array(N * Cout * oH * oW);
  let colsPerGroup = [];
  {
    for (let gi = 0; gi < groups; gi++) {
      const aSlice = groups === 1 ? a : sliceChannels(a, gi * (Cin / groups), Cin / groups);
      colsPerGroup.push(im2col(aSlice, kh, kw, sh, sw, ph, pw, oH, oW, dh, dw));
    }
    for (let gi = 0; gi < groups; gi++) {
      const cols = colsPerGroup[gi];
      const M = N * oH * oW;
      const outC = Cout / groups;
      const tmp = new Float32Array(M * outC);
      const wSub = groups === 1 ? W.d : W.d.subarray(gi * outC * K, (gi + 1) * outC * K);
      gemm(cols, wSub, tmp, M, outC, K, 1, 0, false, true);
      for (let n = 0; n < N; n++) {
        for (let c = 0; c < outC; c++) {
          const oc = gi * outC + c;
          for (let p = 0; p < oH * oW; p++) {
            d[((n * Cout) + oc) * oH * oW + p] = tmp[(n * oH * oW + p) * outC + c];
          }
        }
      }
    }
  }
  if (b) {
    for (let n = 0; n < N; n++) {
      for (let c = 0; c < Cout; c++) {
        const bv = b.d[c];
        const base = (n * Cout + c) * oH * oW;
        for (let p = 0; p < oH * oW; p++) d[base + p] += bv;
      }
    }
  }
  const parents = b ? [a, W, b] : [a, W];
  const out = mk(d, [N, Cout, oH, oW], parents, null, 'conv2d');
  out.bwd = (gy) => {
    const M = N * oH * oW;
    const outC = Cout / groups;
    const gyNHWC = new Float32Array(M * outC);
    for (let gi = 0; gi < groups; gi++) {
      for (let n = 0; n < N; n++) {
        for (let c = 0; c < outC; c++) {
          const oc = gi * outC + c;
          for (let p = 0; p < oH * oW; p++) {
            gyNHWC[(n * oH * oW + p) * outC + c] = gy[((n * Cout) + oc) * oH * oW + p];
          }
        }
      }
      const gW = new Float32Array(outC * K);
      gemm(gyNHWC, colsPerGroup[gi], gW, outC, K, M, 1, 0, true, false);
      if (groups === 1) {
        W.g = addInto(ensureGrad(W), gW);
      } else {
        const full = new Float32Array(W.d.length);
        full.set(gW, gi * outC * K);
        W.g = addInto(ensureGrad(W), full);
      }
      const dcols = new Float32Array(M * K);
      const wSub = groups === 1 ? W.d : W.d.subarray(gi * outC * K, (gi + 1) * outC * K);
      gemm(gyNHWC, wSub, dcols, M, K, outC, 1, 0, false, false);
      const aSlice = groups === 1 ? a : sliceChannels(a, gi * (Cin / groups), Cin / groups);
      const g = col2im(dcols, aSlice.shape, kh, kw, sh, sw, ph, pw, oH, oW, dh, dw);
      if (groups === 1) {
        a.g = addInto(ensureGrad(a), g);
      } else {
        const full = new Float32Array(a.d.length);
        const sub = aSlice.shape;
        let q = 0;
        for (let n = 0; n < sub[0]; n++) {
          for (let c = 0; c < sub[1]; c++) {
            const ic = gi * sub[1] + c;
            for (let p = 0; p < sub[2] * sub[3]; p++) {
              full[((n * Cin + ic) * sub[2]) * sub[3] + p] = g[q++];
            }
          }
        }
        a.g = addInto(ensureGrad(a), full);
      }
    }
    if (b) {
      const gb = new Float32Array(Cout);
      for (let n = 0; n < N; n++) {
        for (let c = 0; c < Cout; c++) {
          let s = 0;
          const base = (n * Cout + c) * oH * oW;
          for (let p = 0; p < oH * oW; p++) s += gy[base + p];
          gb[c] += s;
        }
      }
      b.g = addInto(ensureGrad(b), gb);
    }
  };
  return out;
}

function sliceChannels(a, start, count) {
  const N = a.shape[0];
  const C = a.shape[1];
  const HW = a.shape[2] * a.shape[3];
  const d = new Float32Array(N * count * HW);
  for (let n = 0; n < N; n++) {
    d.set(a.d.subarray((n * C + start) * HW, (n * C + start + count) * HW), n * count * HW);
  }
  return new Node(d, [N, count, a.shape[2], a.shape[3]]);
}

export function permuteNCT(a) {
  const N = a.shape[0];
  const A = a.shape[1];
  const B = a.shape[2];
  const d = new Float32Array(a.d.length);
  for (let n = 0; n < N; n++) {
    for (let i = 0; i < A; i++) {
      for (let j = 0; j < B; j++) d[(n * B + j) * A + i] = a.d[(n * A + i) * B + j];
    }
  }
  const out = mk(d, [N, B, A], [a], null, 'permute_nct');
  out.bwd = (gy) => {
    const g = new Float32Array(a.d.length);
    for (let n = 0; n < N; n++) {
      for (let i = 0; i < A; i++) {
        for (let j = 0; j < B; j++) g[(n * A + i) * B + j] = gy[(n * B + j) * A + i];
      }
    }
    a.g = addInto(ensureGrad(a), g);
  };
  return out;
}

export function conv1d(a, W, b, opts) {
  opts = opts || {};
  const x4 = reshape(a, [a.shape[0], a.shape[1], 1, a.shape[2]]);
  const w4 = reshape(W, [W.shape[0], W.shape[1], 1, W.shape[3]]);
  const y = conv2d(x4, w4, b, {
    stride: [1, Array.isArray(opts.stride) ? opts.stride[0] : opts.stride || 1],
    pad: [0, Array.isArray(opts.pad) ? opts.pad[0] : opts.pad || 0],
    dilation: [1, opts.dilation || 1],
    groups: opts.groups || 1
  });
  return reshape(y, [y.shape[0], y.shape[1], y.shape[3]]);
}

export function upsample2d(a, factor, mode) {
  mode = mode || 'nearest';
  const N = a.shape[0];
  const C = a.shape[1];
  const H = a.shape[2];
  const W = a.shape[3];
  const oH = H * factor;
  const oW = W * factor;
  const d = new Float32Array(N * C * oH * oW);
  const srcIdx = new Int32Array(oH * oW * 2);
  if (mode === 'nearest') {
    for (let oh = 0; oh < oH; oh++) {
      for (let ow = 0; ow < oW; ow++) {
        const ih = Math.min(H - 1, Math.floor(oh / factor));
        const iw = Math.min(W - 1, Math.floor(ow / factor));
        srcIdx[(oh * oW + ow) * 2] = ih;
        srcIdx[(oh * oW + ow) * 2 + 1] = iw;
      }
    }
    for (let n = 0; n < N; n++) {
      for (let c = 0; c < C; c++) {
        for (let oh = 0; oh < oH; oh++) {
          for (let ow = 0; ow < oW; ow++) {
            const ih = srcIdx[(oh * oW + ow) * 2];
            const iw = srcIdx[(oh * oW + ow) * 2 + 1];
            d[((n * C + c) * oH + oh) * oW + ow] = a.d[((n * C + c) * H + ih) * W + iw];
          }
        }
      }
    }
  } else {
    for (let n = 0; n < N; n++) {
      for (let c = 0; c < C; c++) {
        for (let oh = 0; oh < oH; oh++) {
          const yf = (oh + 0.5) / factor - 0.5;
          const y0 = Math.max(0, Math.min(H - 1, Math.floor(yf)));
          const y1 = Math.max(0, Math.min(H - 1, y0 + 1));
          const wy = Math.max(0, Math.min(1, yf - y0));
          for (let ow = 0; ow < oW; ow++) {
            const xf = (ow + 0.5) / factor - 0.5;
            const x0 = Math.max(0, Math.min(W - 1, Math.floor(xf)));
            const x1 = Math.max(0, Math.min(W - 1, x0 + 1));
            const wx = Math.max(0, Math.min(1, xf - x0));
            const base = (n * C + c) * H * W;
            const v = a.d[base + y0 * W + x0] * (1 - wx) * (1 - wy) +
              a.d[base + y0 * W + x1] * wx * (1 - wy) +
              a.d[base + y1 * W + x0] * (1 - wx) * wy +
              a.d[base + y1 * W + x1] * wx * wy;
            d[((n * C + c) * oH + oh) * oW + ow] = v;
            srcIdx[(oh * oW + ow) * 2] = y0 * W + x0;
            srcIdx[(oh * oW + ow) * 2 + 1] = 0;
          }
        }
      }
    }
  }
  const out = mk(d, [N, C, oH, oW], [a], null, 'upsample2d');
  out.bwd = (gy) => {
    const g = new Float32Array(a.d.length);
    if (mode === 'nearest') {
      for (let n = 0; n < N; n++) {
        for (let c = 0; c < C; c++) {
          for (let oh = 0; oh < oH; oh++) {
            for (let ow = 0; ow < oW; ow++) {
              const ih = srcIdx[(oh * oW + ow) * 2];
              const iw = srcIdx[(oh * oW + ow) * 2 + 1];
              g[((n * C + c) * H + ih) * W + iw] += gy[((n * C + c) * oH + oh) * oW + ow];
            }
          }
        }
      }
    } else {
      for (let n = 0; n < N; n++) {
        for (let c = 0; c < C; c++) {
          for (let oh = 0; oh < oH; oh++) {
            const yf = (oh + 0.5) / factor - 0.5;
            const y0 = Math.max(0, Math.min(H - 1, Math.floor(yf)));
            const y1 = Math.max(0, Math.min(H - 1, y0 + 1));
            const wy = Math.max(0, Math.min(1, yf - y0));
            for (let ow = 0; ow < oW; ow++) {
              const xf = (ow + 0.5) / factor - 0.5;
              const x0 = Math.max(0, Math.min(W - 1, Math.floor(xf)));
              const x1 = Math.max(0, Math.min(W - 1, x0 + 1));
              const wx = Math.max(0, Math.min(1, xf - x0));
              const gv = gy[((n * C + c) * oH + oh) * oW + ow];
              const base = (n * C + c) * H * W;
              g[base + y0 * W + x0] += gv * (1 - wx) * (1 - wy);
              g[base + y0 * W + x1] += gv * wx * (1 - wy);
              g[base + y1 * W + x0] += gv * (1 - wx) * wy;
              g[base + y1 * W + x1] += gv * wx * wy;
            }
          }
        }
      }
    }
    a.g = addInto(ensureGrad(a), g);
  };
  return out;
}

export function layernorm(a, gamma, beta, eps) {
  eps = eps || 1e-5;
  const last = a.shape[a.shape.length - 1];
  const rows = a.d.length / last;
  const d = new Float32Array(a.d.length);
  const mean = new Float32Array(rows);
  const rstd = new Float32Array(rows);
  for (let r = 0; r < rows; r++) {
    let s = 0;
    for (let i = 0; i < last; i++) s += a.d[r * last + i];
    const m = s / last;
    mean[r] = m;
    let v = 0;
    for (let i = 0; i < last; i++) {
      const t = a.d[r * last + i] - m;
      v += t * t;
    }
    const rs = 1 / Math.sqrt(v / last + eps);
    rstd[r] = rs;
    for (let i = 0; i < last; i++) {
      d[r * last + i] = (a.d[r * last + i] - m) * rs * gamma.d[i] + beta.d[i];
    }
  }
  const parents = [a, gamma, beta];
  const out = mk(d, a.shape, parents, null, 'layernorm');
  out.bwd = (gy) => {
    const ga = new Float32Array(a.d.length);
    const gg = new Float32Array(gamma.d.length);
    const gb = new Float32Array(beta.d.length);
    for (let r = 0; r < rows; r++) {
      let s1 = 0;
      let s2 = 0;
      for (let i = 0; i < last; i++) {
        const xh = (a.d[r * last + i] - mean[r]) * rstd[r];
        const g = gy[r * last + i] * gamma.d[i];
        s1 += g;
        s2 += g * xh;
        gg[i] += gy[r * last + i] * xh;
        gb[i] += gy[r * last + i];
      }
      for (let i = 0; i < last; i++) {
        const xh = (a.d[r * last + i] - mean[r]) * rstd[r];
        ga[r * last + i] = rstd[r] * (gy[r * last + i] * gamma.d[i] - s1 / last - xh * s2 / last);
      }
    }
    a.g = addInto(ensureGrad(a), ga);
    gamma.g = addInto(ensureGrad(gamma), gg);
    beta.g = addInto(ensureGrad(beta), gb);
  };
  return out;
}

export function batchnorm2d(a, gamma, beta, state, training, momentum, eps) {
  eps = eps || 1e-5;
  momentum = momentum === undefined ? 0.1 : momentum;
  const N = a.shape[0];
  const C = a.shape[1];
  const HW = a.shape[2] * a.shape[3];
  const count = N * HW;
  const d = new Float32Array(a.d.length);
  const xhat = training ? new Float32Array(a.d.length) : null;
  const mean = new Float32Array(C);
  const varr = new Float32Array(C);
  if (training) {
    for (let c = 0; c < C; c++) {
      let s = 0;
      for (let n = 0; n < N; n++) {
        const base = (n * C + c) * HW;
        for (let i = 0; i < HW; i++) s += a.d[base + i];
      }
      mean[c] = s / count;
    }
    for (let c = 0; c < C; c++) {
      let s = 0;
      for (let n = 0; n < N; n++) {
        const base = (n * C + c) * HW;
        for (let i = 0; i < HW; i++) {
          const t = a.d[base + i] - mean[c];
          s += t * t;
        }
      }
      varr[c] = s / count;
    }
    for (let c = 0; c < C; c++) {
      state.runningMean[c] = (1 - momentum) * state.runningMean[c] + momentum * mean[c];
      state.runningVar[c] = (1 - momentum) * state.runningVar[c] + momentum * varr[c];
    }
  } else {
    for (let c = 0; c < C; c++) {
      mean[c] = state.runningMean[c];
      varr[c] = state.runningVar[c];
    }
  }
  const rstd = new Float32Array(C);
  for (let c = 0; c < C; c++) rstd[c] = 1 / Math.sqrt(varr[c] + eps);
  for (let n = 0; n < N; n++) {
    for (let c = 0; c < C; c++) {
      const base = (n * C + c) * HW;
      const rs = rstd[c];
      const g = gamma.d[c];
      const bt = beta.d[c];
      for (let i = 0; i < HW; i++) {
        const xh = (a.d[base + i] - mean[c]) * rs;
        if (xhat) xhat[base + i] = xh;
        d[base + i] = xh * g + bt;
      }
    }
  }
  const out = mk(d, a.shape, training ? [a, gamma, beta] : [], null, 'batchnorm2d');
  if (training) {
    out.bwd = (gy) => {
      const ga = new Float32Array(a.d.length);
      const gg = new Float32Array(C);
      const gb = new Float32Array(C);
      for (let c = 0; c < C; c++) {
        let s1 = 0;
        let s2 = 0;
        for (let n = 0; n < N; n++) {
          const base = (n * C + c) * HW;
          for (let i = 0; i < HW; i++) {
            gg[c] += gy[base + i] * xhat[base + i];
            gb[c] += gy[base + i];
            s1 += gy[base + i] * gamma.d[c];
            s2 += gy[base + i] * gamma.d[c] * xhat[base + i];
          }
        }
        const rs = rstd[c];
        for (let n = 0; n < N; n++) {
          const base = (n * C + c) * HW;
          for (let i = 0; i < HW; i++) {
            ga[base + i] = rs / count * (count * gy[base + i] * gamma.d[c] - s1 - xhat[base + i] * s2);
          }
        }
      }
      a.g = addInto(ensureGrad(a), ga);
      gamma.g = addInto(ensureGrad(gamma), gg);
      beta.g = addInto(ensureGrad(beta), gb);
    };
  }
  return out;
}

export function batchnorm1d(a, gamma, beta, state, training, momentum, eps) {
  eps = eps || 1e-5;
  momentum = momentum === undefined ? 0.1 : momentum;
  const N = a.shape[0];
  const C = a.shape[a.shape.length - 1];
  const extra = a.d.length / (N * C);
  const d = new Float32Array(a.d.length);
  const xhat = training ? new Float32Array(a.d.length) : null;
  const mean = new Float32Array(C);
  const varr = new Float32Array(C);
  if (training) {
    for (let c = 0; c < C; c++) {
      let s = 0;
      for (let n = 0; n < N; n++) {
        for (let e = 0; e < extra; e++) s += a.d[(n * extra + e) * C + c];
      }
      mean[c] = s / (N * extra);
    }
    for (let c = 0; c < C; c++) {
      let s = 0;
      for (let n = 0; n < N; n++) {
        for (let e = 0; e < extra; e++) {
          const t = a.d[(n * extra + e) * C + c] - mean[c];
          s += t * t;
        }
      }
      varr[c] = s / (N * extra);
    }
    for (let c = 0; c < C; c++) {
      state.runningMean[c] = (1 - momentum) * state.runningMean[c] + momentum * mean[c];
      state.runningVar[c] = (1 - momentum) * state.runningVar[c] + momentum * varr[c];
    }
  } else {
    for (let c = 0; c < C; c++) {
      mean[c] = state.runningMean[c];
      varr[c] = state.runningVar[c];
    }
  }
  const rstd = new Float32Array(C);
  for (let c = 0; c < C; c++) rstd[c] = 1 / Math.sqrt(varr[c] + eps);
  for (let idx = 0; idx < a.d.length; idx++) {
    const c = idx % C;
    const xh = (a.d[idx] - mean[c]) * rstd[c];
    if (xhat) xhat[idx] = xh;
    d[idx] = xh * gamma.d[c] + beta.d[c];
  }
  const count = N * extra;
  const out = mk(d, a.shape, training ? [a, gamma, beta] : [], null, 'batchnorm1d');
  if (training) {
    out.bwd = (gy) => {
      const ga = new Float32Array(a.d.length);
      const gg = new Float32Array(C);
      const gb = new Float32Array(C);
      for (let c = 0; c < C; c++) {
        let s1 = 0;
        let s2 = 0;
        for (let idx = c; idx < a.d.length; idx += C) {
          gg[c] += gy[idx] * xhat[idx];
          gb[c] += gy[idx];
          s1 += gy[idx] * gamma.d[c];
          s2 += gy[idx] * gamma.d[c] * xhat[idx];
        }
        const rs = rstd[c];
        for (let idx = c; idx < a.d.length; idx += C) {
          ga[idx] = rs / count * (count * gy[idx] * gamma.d[c] - s1 - xhat[idx] * s2);
        }
      }
      a.g = addInto(ensureGrad(a), ga);
      gamma.g = addInto(ensureGrad(gamma), gg);
      beta.g = addInto(ensureGrad(beta), gb);
    };
  }
  return out;
}

export function gru(x, params, returnSeq) {
  const Wx = params.Wx;
  const Wh = params.Wh;
  const b = params.b;
  const B = x.shape[0];
  const T = x.shape[1];
  const F = x.shape[2];
  const H = Wh.shape[0];
  const xz = new Float32Array(B * T * 3 * H);
  for (let t = 0; t < T; t++) {
    const xs = new Float32Array(B * F);
    for (let n = 0; n < B; n++) xs.set(x.d.subarray((n * T + t) * F, (n * T + t + 1) * F), n * F);
    const o = new Float32Array(B * 3 * H);
    gemm(xs, Wx.d, o, B, 3 * H, F, 1, 0, false, false);
    xz.set(o, t * B * 3 * H);
  }
  const cache = [];
  const hs = new Float32Array(B * T * H);
  let hPrev = new Float32Array(B * H);
  for (let t = 0; t < T; t++) {
    const hz = new Float32Array(B * 3 * H);
    gemm(hPrev, Wh.d, hz, B, 3 * H, H, 1, 0, false, false);
    const r = new Float32Array(B * H);
    const z = new Float32Array(B * H);
    const n = new Float32Array(B * H);
    const hCur = new Float32Array(B * H);
    for (let i = 0; i < B; i++) {
      for (let j = 0; j < H; j++) {
        const rv = 1 / (1 + Math.exp(-(xz[(t * B + i) * 3 * H + j] + hz[i * 3 * H + j] + b.d[j])));
        const zv = 1 / (1 + Math.exp(-(xz[(t * B + i) * 3 * H + H + j] + hz[i * 3 * H + H + j] + b.d[H + j])));
        const nv = Math.tanh(xz[(t * B + i) * 3 * H + 2 * H + j] + rv * (hz[i * 3 * H + 2 * H + j] + b.d[2 * H + j]));
        const hv = (1 - zv) * nv + zv * hPrev[i * H + j];
        r[i * H + j] = rv;
        z[i * H + j] = zv;
        n[i * H + j] = nv;
        hCur[i * H + j] = hv;
        hs[(i * T + t) * H + j] = hv;
      }
    }
    cache.push({ r, z, n, hPrev, hz, hCur });
    hPrev = hCur;
  }
  const outShape = returnSeq ? [B, T, H] : [B, H];
  const outData = returnSeq ? hs : hPrev;
  const out = mk(new Float32Array(outData), outShape, [x, Wx, Wh, b], null, 'gru');
  out.bwd = (gy) => {
    const gx = new Float32Array(B * T * F);
    const gWx = new Float32Array(F * 3 * H);
    const gWh = new Float32Array(H * 3 * H);
    const gb = new Float32Array(3 * H);
    const gxz = new Float32Array(B * T * 3 * H);
    let dh = new Float32Array(B * H);
    if (!returnSeq) dh.set(gy);
    for (let t = T - 1; t >= 0; t--) {
      const c = cache[t];
      const dOut = new Float32Array(B * H);
      for (let i = 0; i < B * H; i++) {
        dOut[i] = dh[i] + (returnSeq ? gy[(Math.floor(i / H) * T + t) * H + (i % H)] : 0);
      }
      const dn = new Float32Array(B * H);
      const dz = new Float32Array(B * H);
      const dr = new Float32Array(B * H);
      const dhPrev = new Float32Array(B * H);
      const dhz = new Float32Array(B * 3 * H);
      for (let i = 0; i < B; i++) {
        for (let j = 0; j < H; j++) {
          const o = i * H + j;
          dn[o] = dOut[o] * (1 - c.z[o]);
          dz[o] = dOut[o] * (c.hPrev[o] - c.n[o]);
          const dnRaw = dn[o] * (1 - c.n[o] * c.n[o]);
          gxz[(t * B + i) * 3 * H + 2 * H + j] = dnRaw;
          dr[o] = dnRaw * (c.hz[i * 3 * H + 2 * H + j] + b.d[2 * H + j]);
          dhPrev[o] = dOut[o] * c.z[o] + dr[o] * c.r[i * H + j];
          const drRaw = dr[o] * c.r[o] * (1 - c.r[o]);
          const dzRaw = dz[o] * c.z[o] * (1 - c.z[o]);
          gxz[(t * B + i) * 3 * H + j] = drRaw;
          gxz[(t * B + i) * 3 * H + H + j] = dzRaw;
          dhz[i * 3 * H + j] = drRaw;
          dhz[i * 3 * H + H + j] = dzRaw;
          dhz[i * 3 * H + 2 * H + j] = dnRaw * c.r[i * H + j];
          gb[j] += drRaw;
          gb[H + j] += dzRaw;
          gb[2 * H + j] += dnRaw;
        }
      }
      const gWhStep = new Float32Array(H * 3 * H);
      gemm(c.hPrev, dhz, gWhStep, H, 3 * H, B, 1, 0, true, false);
      for (let i = 0; i < gWhStep.length; i++) gWh[i] += gWhStep[i];
      const dhPrev2 = new Float32Array(B * H);
      gemm(dhz, Wh.d, dhPrev2, B, H, 3 * H, 1, 0, false, true);
      for (let i = 0; i < B * H; i++) dhPrev[i] += dhPrev2[i];
      const xs = new Float32Array(B * F);
      for (let i = 0; i < B; i++) xs.set(x.d.subarray((i * T + t) * F, (i * T + t + 1) * F), i * F);
      const gxzStep = new Float32Array(B * 3 * H);
      gxzStep.set(gxz.subarray(t * B * 3 * H, (t + 1) * B * 3 * H));
      const gWxStep = new Float32Array(F * 3 * H);
      gemm(xs, gxzStep, gWxStep, F, 3 * H, B, 1, 0, true, false);
      for (let i = 0; i < gWxStep.length; i++) gWx[i] += gWxStep[i];
      const gxStep = new Float32Array(B * F);
      gemm(gxzStep, Wx.d, gxStep, B, F, 3 * H, 1, 0, false, true);
      for (let i = 0; i < B; i++) gx.set(gxStep.subarray(i * F, (i + 1) * F), (i * T + t) * F);
      dh = dhPrev;
    }
    x.g = addInto(ensureGrad(x), gx);
    Wx.g = addInto(ensureGrad(Wx), gWx);
    Wh.g = addInto(ensureGrad(Wh), gWh);
    b.g = addInto(ensureGrad(b), gb);
  };
  return out;
}

export function lstm(x, params, returnSeq) {
  const Wx = params.Wx;
  const Wh = params.Wh;
  const b = params.b;
  const B = x.shape[0];
  const T = x.shape[1];
  const F = x.shape[2];
  const H = Wh.shape[0];
  const G = 4 * H;
  const cache = [];
  const hs = new Float32Array(B * T * H);
  let hPrev = new Float32Array(B * H);
  let cPrev = new Float32Array(B * H);
  const xzAll = new Float32Array(B * T * G);
  for (let t = 0; t < T; t++) {
    const xs = new Float32Array(B * F);
    for (let n = 0; n < B; n++) xs.set(x.d.subarray((n * T + t) * F, (n * T + t + 1) * F), n * F);
    const o = new Float32Array(B * G);
    gemm(xs, Wx.d, o, B, G, F, 1, 0, false, false);
    xzAll.set(o, t * B * G);
    const hz = new Float32Array(B * G);
    gemm(hPrev, Wh.d, hz, B, G, H, 1, 0, false, false);
    const iG = new Float32Array(B * H);
    const fG = new Float32Array(B * H);
    const oG = new Float32Array(B * H);
    const gG = new Float32Array(B * H);
    const cCur = new Float32Array(B * H);
    const hCur = new Float32Array(B * H);
    for (let n = 0; n < B; n++) {
      for (let j = 0; j < H; j++) {
        const base = (t * B + n) * G;
        const iv = 1 / (1 + Math.exp(-(xzAll[base + j] + hz[n * G + j] + b.d[j])));
        const fv = 1 / (1 + Math.exp(-(xzAll[base + H + j] + hz[n * G + H + j] + b.d[H + j])));
        const ov = 1 / (1 + Math.exp(-(xzAll[base + 2 * H + j] + hz[n * G + 2 * H + j] + b.d[2 * H + j])));
        const gv = Math.tanh(xzAll[base + 3 * H + j] + hz[n * G + 3 * H + j] + b.d[3 * H + j]);
        const cv = fv * cPrev[n * H + j] + iv * gv;
        const hv = ov * Math.tanh(cv);
        iG[n * H + j] = iv;
        fG[n * H + j] = fv;
        oG[n * H + j] = ov;
        gG[n * H + j] = gv;
        cCur[n * H + j] = cv;
        hCur[n * H + j] = hv;
        hs[(n * T + t) * H + j] = hv;
      }
    }
    cache.push({ iG, fG, oG, gG, cPrev, cCur, hPrev, hz, hCur });
    hPrev = hCur;
    cPrev = cCur;
  }
  const outShape = returnSeq ? [B, T, H] : [B, H];
  const out = mk(returnSeq ? hs : new Float32Array(hPrev), outShape, [x, Wx, Wh, b], null, 'lstm');
  out.bwd = (gy) => {
    const gx = new Float32Array(B * T * F);
    const gWx = new Float32Array(F * G);
    const gWh = new Float32Array(H * G);
    const gb = new Float32Array(G);
    let dh = new Float32Array(B * H);
    let dc = new Float32Array(B * H);
    for (let t = T - 1; t >= 0; t--) {
      const c = cache[t];
      const dOut = new Float32Array(B * H);
      for (let n = 0; n < B; n++) {
        for (let j = 0; j < H; j++) {
          dOut[n * H + j] = dh[n * H + j] + (returnSeq ? gy[(n * T + t) * H + j] : 0);
        }
      }
      const dgates = new Float32Array(B * G);
      const dhPrev = new Float32Array(B * H);
      const dcPrev = new Float32Array(B * H);
      for (let n = 0; n < B; n++) {
        for (let j = 0; j < H; j++) {
          const tanhC = Math.tanh(c.cCur[n * H + j]);
          const dC = dc[n * H + j] + dOut[n * H + j] * c.oG[n * H + j] * (1 - tanhC * tanhC);
          const doRaw = dOut[n * H + j] * tanhC * c.oG[n * H + j] * (1 - c.oG[n * H + j]);
          const diRaw = dC * c.gG[n * H + j] * c.iG[n * H + j] * (1 - c.iG[n * H + j]);
          const dfRaw = dC * c.cPrev[n * H + j] * c.fG[n * H + j] * (1 - c.fG[n * H + j]);
          const dgRaw = dC * c.iG[n * H + j] * (1 - c.gG[n * H + j] * c.gG[n * H + j]);
          const base = (t * B + n) * G;
          dgates[base + j] = diRaw;
          dgates[base + H + j] = dfRaw;
          dgates[base + 2 * H + j] = doRaw;
          dgates[base + 3 * H + j] = dgRaw;
          gb[j] += diRaw;
          gb[H + j] += dfRaw;
          gb[2 * H + j] += doRaw;
          gb[3 * H + j] += dgRaw;
          dcPrev[n * H + j] = dC * c.fG[n * H + j];
        }
      }
      const gWhStep = new Float32Array(H * G);
      gemm(c.hPrev, dgates, gWhStep, H, G, B, 1, 0, true, false);
      for (let i = 0; i < gWhStep.length; i++) gWh[i] += gWhStep[i];
      const xs = new Float32Array(B * F);
      for (let n = 0; n < B; n++) xs.set(x.d.subarray((n * T + t) * F, (n * T + t + 1) * F), n * F);
      const gWxStep = new Float32Array(F * G);
      gemm(xs, dgates, gWxStep, F, G, B, 1, 0, true, false);
      for (let i = 0; i < gWxStep.length; i++) gWx[i] += gWxStep[i];
      const gxStep = new Float32Array(B * F);
      gemm(dgates, Wx.d, gxStep, B, F, G, 1, 0, false, true);
      for (let n = 0; n < B; n++) gx.set(gxStep.subarray(n * F, (n + 1) * F), (n * T + t) * F);
      for (let n = 0; n < B; n++) {
        for (let j = 0; j < H; j++) {
          let s = 0;
          for (let k = 0; k < G; k++) s += dgates[(t * B + n) * G + k] * Wh.d[j * G + k];
          dhPrev[n * H + j] = s;
        }
      }
      dh = dhPrev;
      dc = dcPrev;
    }
    x.g = addInto(ensureGrad(x), gx);
    Wx.g = addInto(ensureGrad(Wx), gWx);
    Wh.g = addInto(ensureGrad(Wh), gWh);
    b.g = addInto(ensureGrad(b), gb);
  };
  return out;
}

export function lastTimestep(a) {
  const B = a.shape[0];
  const T = a.shape[1];
  const H = a.shape[2];
  const d = new Float32Array(B * H);
  for (let n = 0; n < B; n++) d.set(a.d.subarray((n * T + T - 1) * H, (n * T + T) * H), n * H);
  const out = mk(d, [B, H], [a], null, 'last_timestep');
  out.bwd = (gy) => {
    const g = new Float32Array(a.d.length);
    for (let n = 0; n < B; n++) g.set(gy.subarray(n * H, (n + 1) * H), (n * T + T - 1) * H);
    a.g = addInto(ensureGrad(a), g);
  };
  return out;
}

export function meanTimesteps(a) {
  const B = a.shape[0];
  const T = a.shape[1];
  const H = a.shape[2];
  const d = new Float32Array(B * H);
  for (let n = 0; n < B; n++) {
    for (let j = 0; j < H; j++) {
      let s = 0;
      for (let t = 0; t < T; t++) s += a.d[(n * T + t) * H + j];
      d[n * H + j] = s / T;
    }
  }
  const out = mk(d, [B, H], [a], null, 'mean_timesteps');
  out.bwd = (gy) => {
    const g = new Float32Array(a.d.length);
    for (let n = 0; n < B; n++) {
      for (let t = 0; t < T; t++) {
        for (let j = 0; j < H; j++) g[(n * T + t) * H + j] = gy[n * H + j] / T;
      }
    }
    a.g = addInto(ensureGrad(a), g);
  };
  return out;
}

export function softmaxCrossEntropy(logits, labels, opts) {
  opts = opts || {};
  const B = logits.shape[0];
  const C = logits.shape[1];
  const ls = logits.d;
  const probs = new Float32Array(B * C);
  const losses = new Float32Array(B);
  const w = opts.weights || null;
  const smoothing = opts.labelSmoothing || 0;
  let total = 0;
  let wsum = 0;
  for (let n = 0; n < B; n++) {
    let max = -Infinity;
    for (let c = 0; c < C; c++) max = Math.max(max, ls[n * C + c]);
    let sum = 0;
    for (let c = 0; c < C; c++) {
      const e = Math.exp(ls[n * C + c] - max);
      probs[n * C + c] = e;
      sum += e;
    }
    for (let c = 0; c < C; c++) probs[n * C + c] /= sum;
    const y = labels[n] | 0;
    const p = Math.max(probs[n * C + y], 1e-12);
    let l = -Math.log(p);
    if (smoothing > 0) {
      let s2 = 0;
      for (let c = 0; c < C; c++) s2 += Math.log(Math.max(probs[n * C + c], 1e-12));
      l = (1 - smoothing) * l - smoothing * s2 / C;
    }
    const cw = w ? w[y] : 1;
    losses[n] = l;
    total += l * cw;
    wsum += cw;
  }
  const denom = opts.reduction === 'sum' ? 1 : Math.max(wsum, 1e-9);
  const out = mk(new Float32Array([total / denom]), [1], [logits], null, 'softmax_cross_entropy');
  out.bwd = (gy) => {
    const g = new Float32Array(B * C);
    for (let n = 0; n < B; n++) {
      const y = labels[n] | 0;
      const cw = w ? w[y] : 1;
      const scale = gy[0] * cw / denom;
      for (let c = 0; c < C; c++) {
        const target = (c === y ? 1 - smoothing : 0) + smoothing / C;
        g[n * C + c] = scale * (probs[n * C + c] - target);
      }
    }
    logits.g = addInto(ensureGrad(logits), g);
  };
  out.perSample = losses;
  out.probs = probs;
  return out;
}

export function sigmoidBinaryCrossEntropy(logits, targets, opts) {
  opts = opts || {};
  const n = logits.d.length, ls = logits.d, ts = targets.d;
  const w = opts.weights || null, posW = opts.posWeight === undefined ? 1 : opts.posWeight;
  let total = 0, wsum = 0;
  const perElem = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const x = ls[i], t = ts[i], common = Math.log1p(Math.exp(-Math.abs(x)));
    const l = posW * t * (Math.max(-x, 0) + common) + (1 - t) * (Math.max(x, 0) + common);
    perElem[i] = l;
    const cw = w ? w[i] : 1;
    total += l * cw; wsum += cw;
  }
  const denom = opts.reduction === 'sum' ? 1 : Math.max(wsum, 1e-9);
  const out = mk(new Float32Array([total / denom]), [1], [logits], null, 'bce_with_logits');
  out.bwd = (gy) => {
    const g = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const p = 1 / (1 + Math.exp(-ls[i])), t = ts[i], cw = w ? w[i] : 1;
      g[i] = gy[0] * cw * (posW * t * (p - 1) + (1 - t) * p) / denom;
    }
    logits.g = addInto(ensureGrad(logits), g);
  };
  out.perSample = perElem;
  return out;
}

// Focal loss with the derivative of the focusing term included. Soft targets
// are supported by weighting the positive and negative terms separately.
export function focalBinaryCrossEntropy(logits, targets, opts) {
  opts = opts || {};
  const gamma = opts.gamma === undefined ? 2 : opts.gamma;
  const posW = opts.posWeight === undefined ? 1 : opts.posWeight;
  const n = logits.d.length, probs = new Float32Array(n), posLoss = new Float32Array(n), negLoss = new Float32Array(n);
  let total = 0;
  for (let i = 0; i < n; i++) {
    const x = logits.d[i], y = targets.d[i];
    if (!Number.isFinite(y) || y < 0 || y > 1) throw new Error('Focal binary targets must be between 0 and 1.');
    const p = 1 / (1 + Math.exp(-x)), common = Math.log1p(Math.exp(-Math.abs(x)));
    const pos = Math.max(-x, 0) + common, neg = Math.max(x, 0) + common;
    probs[i] = p; posLoss[i] = pos; negLoss[i] = neg;
    total += posW * y * Math.pow(1 - p, gamma) * pos + (1 - y) * Math.pow(p, gamma) * neg;
  }
  const denom = Math.max(1, n);
  const out = mk(new Float32Array([total / denom]), [1], [logits], null, 'focal_bce');
  out.bwd = (gy) => {
    const g = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const y = targets.d[i], p = probs[i], dp = p * (1 - p);
      const posFocus = Math.pow(1 - p, gamma), negFocus = Math.pow(p, gamma);
      const posDerivative = gamma === 0 ? 0 : -gamma * Math.pow(Math.max(1e-12, 1 - p), gamma - 1) * dp;
      const negDerivative = gamma === 0 ? 0 : gamma * Math.pow(Math.max(1e-12, p), gamma - 1) * dp;
      g[i] = gy[0] / denom * (posW * y * (posFocus * (p - 1) + posLoss[i] * posDerivative) +
        (1 - y) * (negFocus * p + negLoss[i] * negDerivative));
    }
    logits.g = addInto(ensureGrad(logits), g);
  };
  return out;
}

export function focalSoftmaxCrossEntropy(logits, labels, opts) {
  opts = opts || {};
  const gamma = opts.gamma === undefined ? 2 : opts.gamma;
  const smoothing = opts.labelSmoothing || 0, weights = opts.weights || null;
  const B = logits.shape[0], C = logits.shape[1];
  const probs = new Float32Array(B * C), losses = new Float32Array(B);
  let total = 0, wsum = 0;
  for (let i = 0; i < B; i++) {
    const y = labels[i];
    if (!Number.isInteger(y) || y < 0 || y >= C) throw new Error('Class index out of bounds.');
    let max = -Infinity, sum = 0;
    for (let c = 0; c < C; c++) max = Math.max(max, logits.d[i*C+c]);
    for (let c = 0; c < C; c++) { const p = Math.exp(logits.d[i*C+c] - max); probs[i*C+c] = p; sum += p; }
    for (let c = 0; c < C; c++) probs[i*C+c] /= sum;
    const pt = Math.max(probs[i*C+y], 1e-12);
    let ce = -(1 - smoothing) * Math.log(pt);
    if (smoothing) for (let c = 0; c < C; c++) ce -= smoothing / C * Math.log(Math.max(probs[i*C+c], 1e-12));
    losses[i] = ce;
    const w = weights ? weights[y] : 1;
    total += w * Math.pow(1 - pt, gamma) * ce; wsum += w;
  }
  const denom = Math.max(wsum, 1e-9);
  const out = mk(new Float32Array([total / denom]), [1], [logits], null, 'focal_softmax_cross_entropy');
  out.bwd = (gy) => {
    const g = new Float32Array(B * C);
    for (let i = 0; i < B; i++) {
      const y = labels[i], pt = probs[i*C+y], oneMinus = Math.max(1e-12, 1 - pt), focus = Math.pow(oneMinus, gamma);
      const w = weights ? weights[y] : 1;
      for (let c = 0; c < C; c++) {
        const p = probs[i*C+c], target = (c === y ? 1 - smoothing : 0) + smoothing / C;
        const df = gamma === 0 ? 0 : -gamma * Math.pow(oneMinus, gamma - 1) * pt * ((c === y ? 1 : 0) - p);
        g[i*C+c] = gy[0] * w * (focus * (p - target) + losses[i] * df) / denom;
      }
    }
    logits.g = addInto(ensureGrad(logits), g);
  };
  return out;
}

export function mseLoss(pred, target, reduction) {
  const diff = sub(pred, target);
  const sq = square(diff);
  return reduction === 'sum' ? sumAll(sq) : meanAll(sq);
}

export function maeLoss(pred, target, reduction) {
  const diff = abs(sub(pred, target));
  return reduction === 'sum' ? sumAll(diff) : meanAll(diff);
}

export function huberLoss(pred, target, delta) {
  delta = delta || 1;
  const diff = sub(pred, target);
  const a = abs(diff);
  const n = a.d.length;
  const d = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    d[i] = a.d[i] <= delta ? 0.5 * diff.d[i] * diff.d[i] : delta * (a.d[i] - 0.5 * delta);
  }
  const out = mk(d, a.shape, [diff], null, 'huber');
  out.bwd = (gy) => {
    const g = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      g[i] = gy[i] * (a.d[i] <= delta ? diff.d[i] : delta * (diff.d[i] > 0 ? 1 : -1));
    }
    diff.g = addInto(ensureGrad(diff), g);
  };
  return meanAll(out);
}

export function maskedMse(pred, target, mask) {
  const diff = sub(pred, target);
  const sq = square(diff);
  const m = mul(sq, mask);
  const s = sumAll(m);
  const denom = Math.max(mask.d.reduce((a, v) => a + v, 0), 1e-9);
  return mul(s, scalar(1 / denom));
}

export function cosineSimilarityLoss(a, b) {
  const dot = sumAll(mul(a, b));
  const na = sqrt(sumAll(square(a)));
  const nb = sqrt(sumAll(square(b)));
  return sub(scalar(1), div(dot, add(mul(na, nb), scalar(1e-8))));
}

export function tripletMarginLoss(anchor, positive, negative, margin) {
  margin = margin === undefined ? 0.2 : margin;
  const dp = sqrt(add(sumAll(square(sub(anchor, positive))), scalar(1e-8)));
  const dn = sqrt(add(sumAll(square(sub(anchor, negative))), scalar(1e-8)));
  return relu(add(sub(dp, dn), scalar(margin)));
}

export function contrastiveInfoNce(sim, temperature) {
  const scaled = mul(sim, scalar(1 / (temperature || 0.1)));
  const B = scaled.shape[0];
  const labels = new Int32Array(B);
  for (let i = 0; i < B; i++) labels[i] = i;
  return softmaxCrossEntropy(scaled, labels);
}

export function l2normalize(a, axis) {
  axis = axis === undefined ? a.shape.length - 1 : axis;
  const dim = a.shape[axis];
  const rows = a.d.length / dim;
  const d = new Float32Array(a.d.length);
  const norms = new Float32Array(rows);
  for (let r = 0; r < rows; r++) {
    let s = 0;
    for (let i = 0; i < dim; i++) s += a.d[r * dim + i] * a.d[r * dim + i];
    const n = Math.sqrt(s + 1e-12);
    norms[r] = n;
    for (let i = 0; i < dim; i++) d[r * dim + i] = a.d[r * dim + i] / n;
  }
  const out = mk(d, a.shape, [a], null, 'l2normalize');
  out.bwd = (gy) => {
    const g = new Float32Array(a.d.length);
    for (let r = 0; r < rows; r++) {
      let dot = 0;
      for (let i = 0; i < dim; i++) dot += gy[r * dim + i] * d[r * dim + i];
      for (let i = 0; i < dim; i++) {
        g[r * dim + i] = (gy[r * dim + i] - dot * d[r * dim + i]) / norms[r];
      }
    }
    a.g = addInto(ensureGrad(a), g);
  };
  return out;
}

export function infoNceBatch(zA, zB, temperature) {
  const temp = temperature || 0.07;
  const B = zA.shape[0];
  const D = zA.shape[1];
  const sim = new Float32Array(B * B);
  gemm(zA.d, zB.d, sim, B, B, D, 1 / temp, 0, false, true);
  const logits = new Node(sim, [B, B]);
  const labels = new Int32Array(B);
  for (let i = 0; i < B; i++) labels[i] = i;
  const loss = softmaxCrossEntropy(logits, labels);
  const lossBwd = loss.bwd;
  const outer = mk(new Float32Array([loss.d[0]]), [1], [zA, zB], null, 'infonce');
  outer.bwd = (gy) => {
    logits.g = new Float32Array(B * B);
    lossBwd(new Float32Array([gy[0]]));
    const gA = new Float32Array(B * D);
    const gB = new Float32Array(B * D);
    gemm(logits.g, zB.d, gA, B, D, B, 1 / temp, 0, false, false);
    gemm(logits.g, zA.d, gB, B, D, B, 1 / temp, 0, true, false);
    zA.g = addInto(ensureGrad(zA), gA);
    zB.g = addInto(ensureGrad(zB), gB);
  };
  outer.probs = loss.probs;
  return outer;
}

export function infoNceMasked(zA, zB, temperature, positiveMask) {
  const temp = temperature || 0.07;
  const B = zA.shape[0];
  const D = zA.shape[1];
  const sim = new Float32Array(B * B);
  gemm(zA.d, zB.d, sim, B, B, D, 1 / temp, 0, false, true);
  const rowMax = new Float32Array(B);
  for (let i = 0; i < B; i++) {
    let m = -Infinity;
    for (let j = 0; j < B; j++) if (sim[i * B + j] > m) m = sim[i * B + j];
    rowMax[i] = m;
  }
  const expAll = new Float32Array(B * B);
  const denAll = new Float32Array(B);
  const denPos = new Float32Array(B);
  for (let i = 0; i < B; i++) {
    let sa = 0;
    let sp = 0;
    for (let j = 0; j < B; j++) {
      const e = Math.exp(sim[i * B + j] - rowMax[i]);
      expAll[i * B + j] = e;
      sa += e;
      if (positiveMask[i * B + j]) sp += e;
    }
    denAll[i] = sa;
    denPos[i] = sp;
  }
  let total = 0;
  for (let i = 0; i < B; i++) {
    if (denPos[i] <= 0) continue;
    total += -Math.log(denPos[i]) + Math.log(denAll[i]);
  }
  const out = mk(new Float32Array([total / B]), [1], [zA, zB], null, 'infonce_masked');
  out.bwd = (gy) => {
    const g = new Float32Array(B * B);
    const s = gy[0] / B;
    for (let i = 0; i < B; i++) {
      if (denPos[i] <= 0) continue;
      for (let j = 0; j < B; j++) {
        const b = expAll[i * B + j] / denAll[i];
        const a = positiveMask[i * B + j] ? expAll[i * B + j] / denPos[i] : 0;
        g[i * B + j] = s * (b - a);
      }
    }
    const gA = new Float32Array(B * D);
    const gB = new Float32Array(B * D);
    gemm(g, zB.d, gA, B, D, B, 1 / temp, 0, false, false);
    gemm(g, zA.d, gB, B, D, B, 1 / temp, 0, true, false);
    zA.g = addInto(ensureGrad(zA), gA);
    zB.g = addInto(ensureGrad(zB), gB);
  };
  return out;
}

export function flipTime(a) {
  const B = a.shape[0];
  const T = a.shape[1];
  const rest = a.d.length / (B * T);
  const d = new Float32Array(a.d.length);
  for (let n = 0; n < B; n++) {
    for (let t = 0; t < T; t++) {
      d.set(a.d.subarray((n * T + t) * rest, (n * T + t + 1) * rest), (n * T + (T - 1 - t)) * rest);
    }
  }
  const out = mk(d, a.shape, [a], null, 'flip_time');
  out.bwd = (gy) => {
    const g = new Float32Array(a.d.length);
    for (let n = 0; n < B; n++) {
      for (let t = 0; t < T; t++) {
        g.set(gy.subarray((n * T + (T - 1 - t)) * rest, (n * T + (T - 1 - t) + 1) * rest), (n * T + t) * rest);
      }
    }
    a.g = addInto(ensureGrad(a), g);
  };
  return out;
}

export function squeezeLast(a) {
  const shape = a.shape.slice(0, a.shape.length - 2);
  return reshape(a, shape.length ? shape : [a.shape[0]]);
}

export function channelToSpace(a, factor) {
  const N = a.shape[0];
  const C = a.shape[1];
  const H = a.shape[2];
  const W = a.shape[3];
  const oc = C / (factor * factor);
  const d = new Float32Array(a.d.length);
  for (let n = 0; n < N; n++) {
    for (let c = 0; c < oc; c++) {
      for (let fy = 0; fy < factor; fy++) {
        for (let fx = 0; fx < factor; fx++) {
          const ic = (c * factor + fy) * factor + fx;
          for (let h = 0; h < H; h++) {
            for (let w = 0; w < W; w++) {
              d[((n * oc + c) * (H * factor) + h * factor + fy) * (W * factor) + w * factor + fx] =
                a.d[((n * C + ic) * H + h) * W + w];
            }
          }
        }
      }
    }
  }
  const out = mk(d, [N, oc, H * factor, W * factor], [a], null, 'channel_to_space');
  out.bwd = (gy) => {
    const g = new Float32Array(a.d.length);
    for (let n = 0; n < N; n++) {
      for (let c = 0; c < oc; c++) {
        for (let fy = 0; fy < factor; fy++) {
          for (let fx = 0; fx < factor; fx++) {
            const ic = (c * factor + fy) * factor + fx;
            for (let h = 0; h < H; h++) {
              for (let w = 0; w < W; w++) {
                g[((n * C + ic) * H + h) * W + w] =
                  gy[((n * oc + c) * (H * factor) + h * factor + fy) * (W * factor) + w * factor + fx];
              }
            }
          }
        }
      }
    }
    a.g = addInto(ensureGrad(a), g);
  };
  return out;
}
