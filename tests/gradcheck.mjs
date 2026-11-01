import { Node, Rng, backward, param, tensor } from '../js/core/tensor.js';
import * as A from '../js/core/autograd.js';

const rng = new Rng(1234);

function numeric(fn, node, eps) {
  eps = eps || 1e-3;
  const out = [];
  for (let i = 0; i < node.d.length; i++) {
    const old = node.d[i];
    node.d[i] = old + eps;
    const a = fn();
    node.d[i] = old - eps;
    const b = fn();
    node.d[i] = old;
    out.push((a - b) / (2 * eps));
  }
  return out;
}

function check(name, build, leaves) {
  const loss = build();
  backward(loss);
  const analytic = leaves.map((L) => Float32Array.from(L.g || new Float32Array(L.d.length)));
  let maxErr = 0;
  for (let li = 0; li < leaves.length; li++) {
    const L = leaves[li];
    const num = numeric(() => build().d[0], L);
    for (let i = 0; i < num.length; i++) {
      const absErr = Math.abs(num[i] - analytic[li][i]);
      const e = absErr / Math.max(1e-6, Math.abs(num[i]) + Math.abs(analytic[li][i]));
      if (e > maxErr && absErr > 1e-3) maxErr = e;
    }
  }
  console.log((maxErr < 1e-2 ? 'PASS' : 'FAIL') + '  ' + name + '  relerr=' + maxErr.toExponential(2));
  return maxErr < 1e-2;
}

let ok = true;

{
  const x = param('x', [2, 3], new Float32Array([0.3, -0.7, 1.1, 0.2, 0.9, -0.4]));
  const W = param('W', [3, 2], new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]));
  const b = param('b', [2], new Float32Array([0.1, -0.2]));
  ok &= check('linear+bias+relu+mse', () => {
    x.g = null; W.g = null; b.g = null;
    const y = A.relu(A.add(A.matmul(x, W), b));
    return A.mseLoss(y, tensor([[0.5, 0.5], [1, 1]]));
  }, [x, W, b]);
}

{
  const x = param('x', [2, 4], new Float32Array([0.3, -0.7, 1.1, 0.2, 0.2, 0.9, -0.4, 0.5]));
  const g = param('g', [4], new Float32Array([1, 1, 1, 1]));
  const bt = param('bt', [4], new Float32Array([0, 0, 0, 0]));
  const w = tensor([[0.3, -0.5, 0.2, 0.7], [0.1, 0.4, -0.6, 0.2]]);
  ok &= check('layernorm', () => {
    x.g = null; g.g = null; bt.g = null;
    return A.sumAll(A.mul(A.layernorm(x, g, bt), w));
  }, [x, g, bt]);
}

{
  const x = param('x', [2, 2, 3, 3], new Float32Array(18).map((_, i) => Math.sin(i) * 0.5));
  const W = param('W', [2, 2, 2, 2], new Float32Array(16).map((_, i) => (i % 5) * 0.1 - 0.2));
  const b = param('b', [2], new Float32Array([0.1, -0.1]));
  ok &= check('conv2d', () => {
    x.g = null; W.g = null; b.g = null;
    const y = A.conv2d(x, W, b, { stride: 1, pad: 0 });
    return A.sumAll(A.mul(y, y));
  }, [x, W, b]);
}

{
  const x = param('x', [1, 2, 4, 4], new Float32Array(32).map((_, i) => Math.cos(i) * 0.4));
  const W = param('W', [3, 2, 3, 3], new Float32Array(54).map((_, i) => ((i * 37) % 11) * 0.02 - 0.1));
  ok &= check('conv2d padded stride2', () => {
    x.g = null; W.g = null;
    const y = A.conv2d(x, W, null, { stride: 2, pad: 1 });
    return A.sumAll(A.relu(y));
  }, [x, W]);
}

{
  const x = param('x', [1, 2, 5, 5], new Float32Array(50).map((_, i) => Math.sin(i * 1.3) * 0.7));
  ok &= check('maxpool2d', () => {
    x.g = null;
    return A.sumAll(A.maxPool2d(x, 2, 2, 0));
  }, [x]);
  ok &= check('avgpool2d', () => {
    x.g = null;
    return A.sumAll(A.avgPool2d(x, 2, 2, 0));
  }, [x]);
  ok &= check('globalavgpool2d', () => {
    x.g = null;
    return A.sumAll(A.globalAvgPool2d(x));
  }, [x]);
  ok &= check('upsample2d nearest', () => {
    x.g = null;
    return A.sumAll(A.mul(A.upsample2d(x, 2, 'nearest'), 1.0));
  }, [x]);
}

{
  const x = param('x', [2, 3, 6], new Float32Array(36).map((_, i) => Math.sin(i) * 0.6));
  const W = param('W', [4, 3, 1, 3], new Float32Array(36).map((_, i) => (i % 7) * 0.03 - 0.1));
  ok &= check('conv1d', () => {
    x.g = null; W.g = null;
    const y = A.conv1d(x, W, null, { stride: 1, pad: 1 });
    return A.sumAll(A.tanh(y));
  }, [x, W]);
  ok &= check('maxpool1d', () => {
    x.g = null;
    return A.sumAll(A.maxPool1d(x, 2, 2, 0));
  }, [x]);
  ok &= check('globalavgpool1d', () => {
    x.g = null;
    return A.sumAll(A.globalAvgPool1d(x));
  }, [x]);
  ok &= check('globalmaxpool1d', () => {
    x.g = null;
    return A.sumAll(A.globalMaxPool1d(x));
  }, [x]);
}

{
  const x = param('x', [3, 4], new Float32Array(12).map((_, i) => (i % 5) * 0.4 - 0.8));
  ok &= check('softmax+ce', () => {
    x.g = null;
    return A.softmaxCrossEntropy(x, new Int32Array([0, 2, 1]));
  }, [x]);
  ok &= check('bce_with_logits', () => {
    x.g = null;
    return A.sigmoidBinaryCrossEntropy(x, tensor([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]]));
  }, [x]);
  ok &= check('huber', () => {
    x.g = null;
    return A.huberLoss(x, tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.1, 0.1], [0.2, 0.3, 0.4, 0.5]]), 0.5);
  }, [x]);
}

{
  const x = tensor([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [-0.2, 0.1, 0.5], [0.3, -0.4, 0.2]]], [1, 4, 3]);
  const Wx = param('Wx', [3, 6], new Float32Array(18).map((_, i) => (i % 5) * 0.05 - 0.1));
  const Wh = param('Wh', [2, 6], new Float32Array(12).map((_, i) => (i % 3) * 0.07 - 0.05));
  const b = param('b', [6], new Float32Array(6).fill(0.01));
  ok &= check('gru', () => {
    Wx.g = null; Wh.g = null; b.g = null;
    const h = A.gru(x, { Wx, Wh, b }, true);
    return A.sumAll(A.mul(h, h));
  }, [Wx, Wh, b]);
  ok &= check('gru last', () => {
    Wx.g = null; Wh.g = null; b.g = null;
    const h = A.gru(x, { Wx, Wh, b }, false);
    return A.sumAll(A.mul(h, h));
  }, [Wx, Wh, b]);
}

{
  const x = tensor([[[0.1, 0.2], [0.4, 0.5], [-0.2, 0.1], [0.3, -0.4]]], [1, 4, 2]);
  const Wx = param('Wx', [2, 8], new Float32Array(16).map((_, i) => (i % 5) * 0.05 - 0.1));
  const Wh = param('Wh', [2, 8], new Float32Array(16).map((_, i) => (i % 3) * 0.07 - 0.05));
  const b = param('b', [8], new Float32Array(8).fill(0.01));
  ok &= check('lstm', () => {
    Wx.g = null; Wh.g = null; b.g = null;
    const h = A.lstm(x, { Wx, Wh, b }, true);
    return A.sumAll(A.mul(h, h));
  }, [Wx, Wh, b]);
}

{
  const a = param('a', [1, 2, 4, 4], new Float32Array(32).map((_, i) => Math.sin(i) * 0.5));
  const gm = param('gm', [2], new Float32Array([1, 1]));
  const bt = param('bt', [2], new Float32Array([0, 0]));
  const state = { runningMean: [0, 0], runningVar: [1, 1] };
  const wn = tensor(new Float32Array(32).map((_, i) => ((i * 13) % 7) * 0.2 - 0.6), [1, 2, 4, 4]);
  ok &= check('batchnorm2d train', () => {
    a.g = null; gm.g = null; bt.g = null;
    return A.sumAll(A.mul(A.batchnorm2d(a, gm, bt, state, true, 0.1), wn));
  }, [a, gm, bt]);
}

{
  const a = param('a', [2, 3], new Float32Array([0.2, -0.3, 0.5, 0.9, -0.1, 0.4]));
  const bNode = param('c', [2, 3], new Float32Array([0.1, 0.1, -0.2, 0.3, 0.2, 0.1]));
  const s = param('s', [1], new Float32Array([0.7]));
  ok &= check('binary broadcast same', () => {
    a.g = null; bNode.g = null;
    return A.sumAll(A.mul(A.add(a, bNode), A.sub(a, bNode)));
  }, [a, bNode]);
  ok &= check('binary scalar', () => {
    a.g = null; s.g = null;
    return A.sumAll(A.div(A.mul(a, s), A.add(a, s)));
  }, [a, s]);
}

{
  const e = param('e', [5, 3], new Float32Array(15).map((_, i) => (i % 4) * 0.2 - 0.3));
  const ids = new Node(new Float32Array([0, 3, 1, 4]), [2, 2]);
  ok &= check('embedding', () => {
    e.g = null;
    return A.sumAll(A.mul(A.embedding(ids, e), 1.0));
  }, [e]);
}

{
  const za = param('za', [3, 4], new Float32Array([0.2, -0.4, 0.6, 0.1, 0.5, 0.2, -0.3, 0.7, -0.1, 0.4, 0.2, 0.8]));
  const zb = param('zb', [3, 4], new Float32Array([0.3, -0.2, 0.5, 0.2, 0.4, 0.1, -0.4, 0.6, -0.2, 0.5, 0.1, 0.7]));
  ok &= check('infonce', () => {
    za.g = null; zb.g = null;
    return A.infoNceBatch(A.l2normalize(za), A.l2normalize(zb), 0.5);
  }, [za, zb]);
  ok &= check('l2normalize', () => {
    za.g = null;
    return A.sumAll(A.mul(A.l2normalize(za), 1.0));
  }, [za]);
}

console.log(ok ? 'ALL GRADIENT CHECKS PASSED' : 'SOME CHECKS FAILED');
process.exit(ok ? 0 : 1);
