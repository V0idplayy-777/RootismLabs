import * as A from './autograd.js';
import { Node, zeros, ones, shapeSize } from './tensor.js';

function int(v, d) {
  const n = parseInt(v, 10);
  return isFinite(n) ? n : d;
}

function num(v, d) {
  const n = parseFloat(v);
  return isFinite(n) ? n : d;
}

function padFor(mode, k, stride, inDim) {
  if (mode === 'valid' || mode === 0 || mode === '0') return 0;
  if (mode === 'same') return Math.floor(k / 2);
  const n = parseInt(mode, 10);
  return isFinite(n) ? n : Math.floor(k / 2);
}

function convOut(inDim, k, stride, pad) {
  return Math.floor((inDim + 2 * pad - k) / stride) + 1;
}

function denseRun(ctx, ins, cfg, p) {
  let x = ins[0];
  let squeezed = false;
  if (x.shape.length === 1) {
    x = A.reshape(x, [1, x.shape[0]]);
    squeezed = true;
  }
  if (x.shape.length > 3) x = A.reshape(x, [x.shape[0], shapeSize(x.shape.slice(1))]);
  if (x.shape.length === 3) {
    const B = x.shape[0];
    const T = x.shape[1];
    const F = x.shape[2];
    const flat = A.reshape(x, [B * T, F]);
    let y = A.matmul(flat, p.kernel);
    if (p.bias) y = A.add(y, p.bias);
    y = A.applyActivation(y, cfg.activation || 'relu');
    return A.reshape(y, [B, T, cfg.units]);
  }
  let y = A.matmul(x, p.kernel);
  if (p.bias) y = A.add(y, p.bias);
  y = A.applyActivation(y, cfg.activation || 'relu');
  return squeezed ? A.reshape(y, [y.shape[1]]) : y;
}

function toRank3(x) {
  if (x.shape.length === 3) return x;
  if (x.shape.length === 2) return A.reshape(x, [x.shape[0], 1, x.shape[1]]);
  return x;
}

export const LAYERS = {
  input: {
    key: 'input',
    label: 'Input',
    group: 'io',
    ports: 0,
    fields: [],
    infer(ins, cfg, spec) {
      return (spec && spec.input && spec.input.shape) ? [1].concat(spec.input.shape) : null;
    },
    params() {
      return [];
    },
    run(ctx, ins) {
      return ins[0];
    },
    macs() {
      return 0;
    },
    text(cfg, spec) {
      const s = (spec && spec.input && spec.input.shape) || [];
      return 'input ' + (s.length ? s.join('×') : '—');
    }
  },

  dense: {
    key: 'dense',
    label: 'Dense',
    group: 'core',
    ports: 1,
    fields: [
      { key: 'units', label: 'Units', type: 'int', def: 64, min: 1, max: 4096 },
      { key: 'activation', label: 'Activation', type: 'select', def: 'relu', options: ['relu', 'gelu', 'elu', 'tanh', 'sigmoid', 'leaky_relu', 'softplus', 'linear'] },
      { key: 'useBias', label: 'Bias', type: 'bool', def: true },
      { key: 'kernelInit', label: 'Weight init', type: 'select', def: 'glorot', options: ['glorot', 'he', 'normal', 'uniform', 'zeros'] }
    ],
    infer(ins, cfg) {
      if (!ins[0]) return null;
      const s = ins[0].slice();
      s[s.length - 1] = int(cfg.units, 64);
      return s;
    },
    params(ins, cfg) {
      const fan = ins[0] ? ins[0][ins[0].length - 1] : 1;
      const out = [{ suffix: 'kernel', shape: [fan, int(cfg.units, 64)], init: cfg.kernelInit || 'glorot' }];
      if (cfg.useBias !== false) out.push({ suffix: 'bias', shape: [int(cfg.units, 64)], init: 'zeros' });
      return out;
    },
    run: denseRun,
    macs(ins, outs, cfg) {
      const fan = ins[0] ? ins[0][ins[0].length - 1] : 1;
      const reps = ins[0] ? shapeSize(ins[0].slice(1, -1)) || 1 : 1;
      return fan * int(cfg.units, 64) * reps;
    },
    text(cfg) {
      return int(cfg.units, 64) + ' · ' + (cfg.activation || 'relu');
    }
  },

  activation: {
    key: 'activation',
    label: 'Activation',
    group: 'core',
    ports: 1,
    fields: [
      { key: 'kind', label: 'Function', type: 'select', def: 'relu', options: ['relu', 'gelu', 'elu', 'tanh', 'sigmoid', 'leaky_relu', 'softplus', 'hard_sigmoid'] }
    ],
    infer(ins) {
      return ins[0] ? ins[0].slice() : null;
    },
    params() {
      return [];
    },
    run(ctx, ins, cfg) {
      return A.applyActivation(ins[0], cfg.kind || 'relu');
    },
    macs(ins) {
      return ins[0] ? shapeSize(ins[0]) : 0;
    },
    text(cfg) {
      return cfg.kind || 'relu';
    }
  },

  softmax: {
    key: 'softmax',
    label: 'Softmax',
    group: 'core',
    ports: 1,
    fields: [{ key: 'axis', label: 'Axis', type: 'int', def: -1 }],
    infer(ins) {
      return ins[0] ? ins[0].slice() : null;
    },
    params() {
      return [];
    },
    run(ctx, ins, cfg) {
      return A.softmax(ins[0], int(cfg.axis, -1));
    },
    macs(ins) {
      return ins[0] ? shapeSize(ins[0]) * 2 : 0;
    },
    text() {
      return 'softmax';
    }
  },

  dropout: {
    key: 'dropout',
    label: 'Dropout',
    group: 'regularize',
    ports: 1,
    fields: [{ key: 'rate', label: 'Rate', type: 'float', def: 0.2, min: 0, max: 0.9, step: 0.05 }],
    infer(ins) {
      return ins[0] ? ins[0].slice() : null;
    },
    params() {
      return [];
    },
    run(ctx, ins, cfg) {
      return A.dropout(ins[0], num(cfg.rate, 0.2), ctx.mode === 'train', ctx.rng);
    },
    macs() {
      return 0;
    },
    text(cfg) {
      return 'p=' + num(cfg.rate, 0.2).toFixed(2);
    }
  },

  flatten: {
    key: 'flatten',
    label: 'Flatten',
    group: 'shape',
    ports: 1,
    fields: [],
    infer(ins) {
      if (!ins[0]) return null;
      return [ins[0][0], shapeSize(ins[0].slice(1))];
    },
    params() {
      return [];
    },
    run(ctx, ins) {
      return A.flatten(ins[0]);
    },
    macs() {
      return 0;
    },
    text() {
      return 'flatten';
    }
  },

  reshape: {
    key: 'reshape',
    label: 'Reshape',
    group: 'shape',
    ports: 1,
    fields: [{ key: 'shape', label: 'Target shape', type: 'text', def: '8,4,4' }],
    infer(ins, cfg) {
      if (!ins[0]) return null;
      const parts = String(cfg.shape || '').split(/[,\s×x]+/).map((v) => parseInt(v, 10)).filter((v) => isFinite(v) && v > 0);
      if (!parts.length) return null;
      const want = shapeSize(parts);
      const have = shapeSize(ins[0].slice(1));
      if (want !== have) return null;
      return [ins[0][0]].concat(parts);
    },
    params() {
      return [];
    },
    run(ctx, ins, cfg, p, shapes) {
      return A.reshape(ins[0], shapes.out);
    },
    macs() {
      return 0;
    },
    text(cfg) {
      return String(cfg.shape || '');
    }
  },

  concat: {
    key: 'concat',
    label: 'Concat',
    group: 'shape',
    ports: 2,
    fields: [{ key: 'axis', label: 'Axis', type: 'int', def: -1 }],
    infer(ins, cfg) {
      if (!ins[0] || !ins[1]) return null;
      if (ins[0].length !== ins[1].length) return null;
      const rank = ins[0].length;
      let axis = int(cfg.axis, -1);
      if (axis < 0) axis += rank;
      const out = ins[0].slice();
      for (let i = 0; i < rank; i++) {
        if (i === axis) continue;
        if (ins[0][i] !== ins[1][i]) return null;
      }
      out[axis] = ins[0][axis] + ins[1][axis];
      return out;
    },
    params() {
      return [];
    },
    run(ctx, ins, cfg) {
      let axis = int(cfg.axis, -1);
      if (axis < 0) axis += ins[0].shape.length;
      return A.concat([ins[0], ins[1]], axis);
    },
    macs() {
      return 0;
    },
    text(cfg) {
      return 'axis ' + int(cfg.axis, -1);
    }
  },

  add: {
    key: 'add',
    label: 'Add',
    group: 'shape',
    ports: 2,
    fields: [],
    infer(ins) {
      if (!ins[0] || !ins[1]) return null;
      if (ins[0].join() !== ins[1].join()) return null;
      return ins[0].slice();
    },
    params() {
      return [];
    },
    run(ctx, ins) {
      return A.add(ins[0], ins[1]);
    },
    macs(ins) {
      return ins[0] ? shapeSize(ins[0]) : 0;
    },
    text() {
      return 'residual add';
    }
  },

  multiply: {
    key: 'multiply',
    label: 'Multiply',
    group: 'shape',
    ports: 2,
    fields: [],
    infer(ins) {
      if (!ins[0] || !ins[1]) return null;
      if (ins[0].join() !== ins[1].join()) return null;
      return ins[0].slice();
    },
    params() {
      return [];
    },
    run(ctx, ins) {
      return A.mul(ins[0], ins[1]);
    },
    macs(ins) {
      return ins[0] ? shapeSize(ins[0]) : 0;
    },
    text() {
      return 'gate';
    }
  },

  layernorm: {
    key: 'layernorm',
    label: 'LayerNorm',
    group: 'regularize',
    ports: 1,
    fields: [{ key: 'eps', label: 'Epsilon', type: 'float', def: 0.00001, step: 0.00001 }],
    infer(ins) {
      return ins[0] ? ins[0].slice() : null;
    },
    params(ins, cfg) {
      const c = ins[0] ? ins[0][ins[0].length - 1] : 1;
      return [
        { suffix: 'gamma', shape: [c], init: 'ones' },
        { suffix: 'beta', shape: [c], init: 'zeros' }
      ];
    },
    run(ctx, ins, cfg, p) {
      return A.layernorm(ins[0], p.gamma, p.beta, num(cfg.eps, 1e-5));
    },
    macs(ins) {
      return ins[0] ? shapeSize(ins[0]) * 5 : 0;
    },
    text() {
      return 'layernorm';
    }
  },

  batchnorm: {
    key: 'batchnorm',
    label: 'BatchNorm',
    group: 'regularize',
    ports: 1,
    fields: [
      { key: 'momentum', label: 'Momentum', type: 'float', def: 0.1, step: 0.01 },
      { key: 'eps', label: 'Epsilon', type: 'float', def: 0.00001, step: 0.00001 }
    ],
    infer(ins) {
      return ins[0] ? ins[0].slice() : null;
    },
    params(ins, cfg) {
      const rank = ins[0] ? ins[0].length : 2;
      const c = rank === 4 ? ins[0][1] : ins[0][ins[0].length - 1];
      return [
        { suffix: 'gamma', shape: [c], init: 'ones' },
        { suffix: 'beta', shape: [c], init: 'zeros' }
      ];
    },
    states(ins) {
      const rank = ins[0] ? ins[0].length : 2;
      const c = rank === 4 ? ins[0][1] : ins[0][ins[0].length - 1];
      return [
        { suffix: 'runningMean', shape: [c], init: 'zeros' },
        { suffix: 'runningVar', shape: [c], init: 'ones' }
      ];
    },
    run(ctx, ins, cfg, p) {
      const st = ctx.state;
      if (ins[0].shape.length === 4) {
        return A.batchnorm2d(ins[0], p.gamma, p.beta, st, ctx.mode === 'train', num(cfg.momentum, 0.1), num(cfg.eps, 1e-5));
      }
      return A.batchnorm1d(ins[0], p.gamma, p.beta, st, ctx.mode === 'train', num(cfg.momentum, 0.1), num(cfg.eps, 1e-5));
    },
    macs(ins) {
      return ins[0] ? shapeSize(ins[0]) * 4 : 0;
    },
    text() {
      return 'batchnorm';
    }
  },

  conv2d: {
    key: 'conv2d',
    label: 'Conv 2D',
    group: 'conv',
    ports: 1,
    fields: [
      { key: 'filters', label: 'Filters', type: 'int', def: 16, min: 1, max: 512 },
      { key: 'kernel', label: 'Kernel', type: 'int', def: 3, min: 1, max: 11 },
      { key: 'stride', label: 'Stride', type: 'int', def: 1, min: 1, max: 4 },
      { key: 'padding', label: 'Padding', type: 'select', def: 'same', options: ['same', 'valid', '0', '1', '2', '3'] },
      { key: 'dilation', label: 'Dilation', type: 'int', def: 1, min: 1, max: 4 },
      { key: 'groups', label: 'Groups', type: 'int', def: 1, min: 1, max: 64 },
      { key: 'activation', label: 'Activation', type: 'select', def: 'relu', options: ['relu', 'gelu', 'elu', 'tanh', 'sigmoid', 'leaky_relu', 'linear'] },
      { key: 'useBias', label: 'Bias', type: 'bool', def: true },
      { key: 'kernelInit', label: 'Weight init', type: 'select', def: 'he', options: ['glorot', 'he', 'normal', 'uniform', 'zeros'] }
    ],
    infer(ins, cfg) {
      if (!ins[0] || ins[0].length !== 4) return null;
      const k = int(cfg.kernel, 3);
      const s = int(cfg.stride, 1);
      const dil = int(cfg.dilation, 1);
      const ek = (k - 1) * dil + 1;
      const p = padFor(cfg.padding, ek, s, ins[0][2]);
      return [ins[0][0], int(cfg.filters, 16), convOut(ins[0][2], ek, s, p), convOut(ins[0][3], ek, s, p)];
    },
    params(ins, cfg) {
      if (!ins[0] || ins[0].length !== 4) return [];
      const k = int(cfg.kernel, 3);
      const g = Math.max(1, int(cfg.groups, 1));
      const f = int(cfg.filters, 16);
      const out = [{ suffix: 'kernel', shape: [f, ins[0][1] / g, k, k], init: cfg.kernelInit || 'he' }];
      if (cfg.useBias !== false) out.push({ suffix: 'bias', shape: [f], init: 'zeros' });
      return out;
    },
    run(ctx, ins, cfg, p) {
      const k = int(cfg.kernel, 3);
      const s = int(cfg.stride, 1);
      const dil = int(cfg.dilation, 1);
      const ek = (k - 1) * dil + 1;
      const pad = padFor(cfg.padding, ek, s, ins[0].shape[2]);
      let y = A.conv2d(ins[0], p.kernel, cfg.useBias !== false ? p.bias : null, {
        stride: s,
        pad,
        dilation: dil,
        groups: Math.max(1, int(cfg.groups, 1))
      });
      return A.applyActivation(y, cfg.activation || 'relu');
    },
    macs(ins, outs, cfg) {
      if (!ins[0] || !outs) return 0;
      const k = int(cfg.kernel, 3);
      const g = Math.max(1, int(cfg.groups, 1));
      return outs[1] * outs[2] * outs[3] * (ins[0][1] / g) * k * k;
    },
    text(cfg) {
      return int(cfg.filters, 16) + ' · ' + int(cfg.kernel, 3) + '×' + int(cfg.kernel, 3) +
        ' s' + int(cfg.stride, 1) + ' · ' + (cfg.activation || 'relu');
    }
  },

  depthwiseconv2d: {
    key: 'depthwiseconv2d',
    label: 'Depthwise Conv',
    group: 'conv',
    ports: 1,
    fields: [
      { key: 'kernel', label: 'Kernel', type: 'int', def: 3, min: 1, max: 7 },
      { key: 'stride', label: 'Stride', type: 'int', def: 1, min: 1, max: 4 },
      { key: 'padding', label: 'Padding', type: 'select', def: 'same', options: ['same', 'valid', '0', '1', '2'] },
      { key: 'multiplier', label: 'Depth multiplier', type: 'int', def: 1, min: 1, max: 4 },
      { key: 'activation', label: 'Activation', type: 'select', def: 'relu', options: ['relu', 'gelu', 'elu', 'tanh', 'sigmoid', 'linear'] }
    ],
    infer(ins, cfg) {
      if (!ins[0] || ins[0].length !== 4) return null;
      const k = int(cfg.kernel, 3);
      const s = int(cfg.stride, 1);
      const p = padFor(cfg.padding, k, s, ins[0][2]);
      return [ins[0][0], ins[0][1] * int(cfg.multiplier, 1), convOut(ins[0][2], k, s, p), convOut(ins[0][3], k, s, p)];
    },
    params(ins, cfg) {
      if (!ins[0] || ins[0].length !== 4) return [];
      const k = int(cfg.kernel, 3);
      const m = int(cfg.multiplier, 1);
      return [
        { suffix: 'kernel', shape: [ins[0][1] * m, 1, k, k], init: 'he' },
        { suffix: 'bias', shape: [ins[0][1] * m], init: 'zeros' }
      ];
    },
    run(ctx, ins, cfg, p) {
      const k = int(cfg.kernel, 3);
      const s = int(cfg.stride, 1);
      const pad = padFor(cfg.padding, k, s, ins[0].shape[2]);
      let y = A.conv2d(ins[0], p.kernel, p.bias, { stride: s, pad, groups: ins[0].shape[1] });
      return A.applyActivation(y, cfg.activation || 'relu');
    },
    macs(ins, outs, cfg) {
      if (!outs) return 0;
      return outs[1] * outs[2] * outs[3] * int(cfg.kernel, 3) * int(cfg.kernel, 3);
    },
    text(cfg) {
      return int(cfg.kernel, 3) + '×' + int(cfg.kernel, 3) + ' dw · ' + (cfg.activation || 'relu');
    }
  },

  separableconv2d: {
    key: 'separableconv2d',
    label: 'Separable Conv',
    group: 'conv',
    ports: 1,
    fields: [
      { key: 'filters', label: 'Filters', type: 'int', def: 16, min: 1, max: 512 },
      { key: 'kernel', label: 'Kernel', type: 'int', def: 3, min: 1, max: 7 },
      { key: 'stride', label: 'Stride', type: 'int', def: 1, min: 1, max: 4 },
      { key: 'padding', label: 'Padding', type: 'select', def: 'same', options: ['same', 'valid', '0', '1', '2'] },
      { key: 'activation', label: 'Activation', type: 'select', def: 'relu', options: ['relu', 'gelu', 'elu', 'tanh', 'sigmoid', 'linear'] }
    ],
    infer(ins, cfg) {
      if (!ins[0] || ins[0].length !== 4) return null;
      const k = int(cfg.kernel, 3);
      const s = int(cfg.stride, 1);
      const p = padFor(cfg.padding, k, s, ins[0][2]);
      return [ins[0][0], int(cfg.filters, 16), convOut(ins[0][2], k, s, p), convOut(ins[0][3], k, s, p)];
    },
    params(ins, cfg) {
      if (!ins[0] || ins[0].length !== 4) return [];
      const k = int(cfg.kernel, 3);
      const f = int(cfg.filters, 16);
      return [
        { suffix: 'dw', shape: [ins[0][1], 1, k, k], init: 'he' },
        { suffix: 'dwbias', shape: [ins[0][1]], init: 'zeros' },
        { suffix: 'pw', shape: [f, ins[0][1], 1, 1], init: 'he' },
        { suffix: 'pwbias', shape: [f], init: 'zeros' }
      ];
    },
    run(ctx, ins, cfg, p) {
      const k = int(cfg.kernel, 3);
      const s = int(cfg.stride, 1);
      const pad = padFor(cfg.padding, k, s, ins[0].shape[2]);
      let y = A.conv2d(ins[0], p.dw, p.dwbias, { stride: s, pad, groups: ins[0].shape[1] });
      y = A.applyActivation(y, cfg.activation || 'relu');
      y = A.conv2d(y, p.pw, p.pwbias, { stride: 1, pad: 0 });
      return A.applyActivation(y, cfg.activation || 'relu');
    },
    macs(ins, outs, cfg) {
      if (!ins[0] || !outs) return 0;
      const k = int(cfg.kernel, 3);
      return ins[0][1] * ins[0][2] * ins[0][3] * k * k + outs[1] * outs[2] * outs[3] * ins[0][1];
    },
    text(cfg) {
      return int(cfg.filters, 16) + ' sep · ' + int(cfg.kernel, 3) + '×' + int(cfg.kernel, 3);
    }
  },

  conv1d: {
    key: 'conv1d',
    label: 'Conv 1D',
    group: 'conv',
    ports: 1,
    fields: [
      { key: 'filters', label: 'Filters', type: 'int', def: 16, min: 1, max: 256 },
      { key: 'kernel', label: 'Kernel', type: 'int', def: 3, min: 1, max: 15 },
      { key: 'stride', label: 'Stride', type: 'int', def: 1, min: 1, max: 8 },
      { key: 'padding', label: 'Padding', type: 'select', def: 'same', options: ['same', 'valid', '0', '1', '2', '3'] },
      { key: 'dilation', label: 'Dilation', type: 'int', def: 1, min: 1, max: 8 },
      { key: 'activation', label: 'Activation', type: 'select', def: 'relu', options: ['relu', 'gelu', 'elu', 'tanh', 'sigmoid', 'leaky_relu', 'linear'] },
      { key: 'useBias', label: 'Bias', type: 'bool', def: true },
      { key: 'kernelInit', label: 'Weight init', type: 'select', def: 'he', options: ['glorot', 'he', 'normal', 'uniform', 'zeros'] }
    ],
    infer(ins, cfg) {
      if (!ins[0] || ins[0].length !== 3) return null;
      const k = int(cfg.kernel, 3);
      const s = int(cfg.stride, 1);
      const dil = int(cfg.dilation, 1);
      const ek = (k - 1) * dil + 1;
      const p = padFor(cfg.padding, ek, s, ins[0][1]);
      return [ins[0][0], convOut(ins[0][1], ek, s, p), int(cfg.filters, 16)];
    },
    params(ins, cfg) {
      if (!ins[0] || ins[0].length !== 3) return [];
      const k = int(cfg.kernel, 3);
      const f = int(cfg.filters, 16);
      const out = [{ suffix: 'kernel', shape: [f, ins[0][2], 1, k], init: cfg.kernelInit || 'he' }];
      if (cfg.useBias !== false) out.push({ suffix: 'bias', shape: [f], init: 'zeros' });
      return out;
    },
    run(ctx, ins, cfg, p) {
      const k = int(cfg.kernel, 3);
      const s = int(cfg.stride, 1);
      const dil = int(cfg.dilation, 1);
      const ek = (k - 1) * dil + 1;
      const pad = padFor(cfg.padding, ek, s, ins[0].shape[1]);
      let y = A.conv1d(A.permuteNCT(ins[0]), p.kernel, cfg.useBias !== false ? p.bias : null, { stride: s, pad, dilation: dil });
      return A.applyActivation(A.permuteNCT(y), cfg.activation || 'relu');
    },
    macs(ins, outs, cfg) {
      if (!ins[0] || !outs) return 0;
      return outs[1] * outs[2] * ins[0][2] * int(cfg.kernel, 3);
    },
    text(cfg) {
      return int(cfg.filters, 16) + ' · k' + int(cfg.kernel, 3) + ' s' + int(cfg.stride, 1);
    }
  },

  maxpool2d: {
    key: 'maxpool2d',
    label: 'MaxPool 2D',
    group: 'pool',
    ports: 1,
    fields: [
      { key: 'kernel', label: 'Kernel', type: 'int', def: 2, min: 1, max: 8 },
      { key: 'stride', label: 'Stride', type: 'int', def: 2, min: 1, max: 8 },
      { key: 'padding', label: 'Padding', type: 'select', def: 'valid', options: ['valid', 'same', '0', '1'] }
    ],
    infer(ins, cfg) {
      if (!ins[0] || ins[0].length !== 4) return null;
      const k = int(cfg.kernel, 2);
      const s = int(cfg.stride, k);
      const p = padFor(cfg.padding === 'valid' ? 0 : cfg.padding, k, s, ins[0][2]);
      return [ins[0][0], ins[0][1], convOut(ins[0][2], k, s, p), convOut(ins[0][3], k, s, p)];
    },
    params() {
      return [];
    },
    run(ctx, ins, cfg, p, shapes) {
      const k = int(cfg.kernel, 2);
      const s = int(cfg.stride, k);
      const pad = padFor(cfg.padding === 'valid' ? 0 : cfg.padding, k, s, ins[0].shape[2]);
      return A.maxPool2d(ins[0], k, s, pad);
    },
    macs(ins) {
      return ins[0] ? shapeSize(ins[0]) : 0;
    },
    text(cfg) {
      return int(cfg.kernel, 2) + ' / s' + int(cfg.stride, 2);
    }
  },

  avgpool2d: {
    key: 'avgpool2d',
    label: 'AvgPool 2D',
    group: 'pool',
    ports: 1,
    fields: [
      { key: 'kernel', label: 'Kernel', type: 'int', def: 2, min: 1, max: 8 },
      { key: 'stride', label: 'Stride', type: 'int', def: 2, min: 1, max: 8 }
    ],
    infer(ins, cfg) {
      if (!ins[0] || ins[0].length !== 4) return null;
      const k = int(cfg.kernel, 2);
      const s = int(cfg.stride, k);
      return [ins[0][0], ins[0][1], convOut(ins[0][2], k, s, 0), convOut(ins[0][3], k, s, 0)];
    },
    params() {
      return [];
    },
    run(ctx, ins, cfg) {
      const k = int(cfg.kernel, 2);
      return A.avgPool2d(ins[0], k, int(cfg.stride, k));
    },
    macs(ins) {
      return ins[0] ? shapeSize(ins[0]) : 0;
    },
    text(cfg) {
      return int(cfg.kernel, 2) + ' / s' + int(cfg.stride, 2);
    }
  },

  globalavgpool2d: {
    key: 'globalavgpool2d',
    label: 'Global AvgPool',
    group: 'pool',
    ports: 1,
    fields: [],
    infer(ins) {
      if (!ins[0] || ins[0].length !== 4) return null;
      return [ins[0][0], ins[0][1]];
    },
    params() {
      return [];
    },
    run(ctx, ins) {
      return A.reshape(A.globalAvgPool2d(ins[0]), [ins[0].shape[0], ins[0].shape[1]]);
    },
    macs(ins) {
      return ins[0] ? shapeSize(ins[0]) : 0;
    },
    text() {
      return 'gap';
    }
  },

  maxpool1d: {
    key: 'maxpool1d',
    label: 'MaxPool 1D',
    group: 'pool',
    ports: 1,
    fields: [
      { key: 'kernel', label: 'Kernel', type: 'int', def: 2, min: 1, max: 16 },
      { key: 'stride', label: 'Stride', type: 'int', def: 2, min: 1, max: 16 }
    ],
    infer(ins, cfg) {
      if (!ins[0] || ins[0].length !== 3) return null;
      const k = int(cfg.kernel, 2);
      const s = int(cfg.stride, k);
      return [ins[0][0], convOut(ins[0][1], k, s, 0), ins[0][2]];
    },
    params() {
      return [];
    },
    run(ctx, ins, cfg) {
      const k = int(cfg.kernel, 2);
      return A.permuteNCT(A.maxPool1d(A.permuteNCT(ins[0]), k, int(cfg.stride, k)));
    },
    macs(ins) {
      return ins[0] ? shapeSize(ins[0]) : 0;
    },
    text(cfg) {
      return int(cfg.kernel, 2) + ' / s' + int(cfg.stride, 2);
    }
  },

  avgpool1d: {
    key: 'avgpool1d',
    label: 'AvgPool 1D',
    group: 'pool',
    ports: 1,
    fields: [
      { key: 'kernel', label: 'Kernel', type: 'int', def: 2, min: 1, max: 16 },
      { key: 'stride', label: 'Stride', type: 'int', def: 2, min: 1, max: 16 }
    ],
    infer(ins, cfg) {
      if (!ins[0] || ins[0].length !== 3) return null;
      const k = int(cfg.kernel, 2);
      const s = int(cfg.stride, k);
      return [ins[0][0], convOut(ins[0][1], k, s, 0), ins[0][2]];
    },
    params() {
      return [];
    },
    run(ctx, ins, cfg) {
      const k = int(cfg.kernel, 2);
      return A.permuteNCT(A.avgPool1d(A.permuteNCT(ins[0]), k, int(cfg.stride, k)));
    },
    macs(ins) {
      return ins[0] ? shapeSize(ins[0]) : 0;
    },
    text(cfg) {
      return int(cfg.kernel, 2) + ' / s' + int(cfg.stride, 2);
    }
  },

  globalavgpool1d: {
    key: 'globalavgpool1d',
    label: 'Global AvgPool 1D',
    group: 'pool',
    ports: 1,
    fields: [],
    infer(ins) {
      if (!ins[0] || ins[0].length !== 3) return null;
      return [ins[0][0], ins[0][2]];
    },
    params() {
      return [];
    },
    run(ctx, ins) {
      return A.globalAvgPool1d(A.permuteNCT(ins[0]));
    },
    macs(ins) {
      return ins[0] ? shapeSize(ins[0]) : 0;
    },
    text() {
      return 'gap1d';
    }
  },

  globalmaxpool1d: {
    key: 'globalmaxpool1d',
    label: 'Global MaxPool 1D',
    group: 'pool',
    ports: 1,
    fields: [],
    infer(ins) {
      if (!ins[0] || ins[0].length !== 3) return null;
      return [ins[0][0], ins[0][2]];
    },
    params() {
      return [];
    },
    run(ctx, ins) {
      return A.globalMaxPool1d(A.permuteNCT(ins[0]));
    },
    macs(ins) {
      return ins[0] ? shapeSize(ins[0]) : 0;
    },
    text() {
      return 'gmp1d';
    }
  },

  upsample2d: {
    key: 'upsample2d',
    label: 'Upsample 2D',
    group: 'shape',
    ports: 1,
    fields: [
      { key: 'factor', label: 'Factor', type: 'int', def: 2, min: 2, max: 4 },
      { key: 'mode', label: 'Mode', type: 'select', def: 'nearest', options: ['nearest', 'bilinear'] }
    ],
    infer(ins, cfg) {
      if (!ins[0] || ins[0].length !== 4) return null;
      const f = int(cfg.factor, 2);
      return [ins[0][0], ins[0][1], ins[0][2] * f, ins[0][3] * f];
    },
    params() {
      return [];
    },
    run(ctx, ins, cfg) {
      return A.upsample2d(ins[0], int(cfg.factor, 2), cfg.mode || 'nearest');
    },
    macs(ins, outs) {
      return outs ? shapeSize(outs) : 0;
    },
    text(cfg) {
      return '×' + int(cfg.factor, 2) + ' ' + (cfg.mode || 'nearest');
    }
  },

  subpixelupsample: {
    key: 'subpixelupsample',
    label: 'Sub-pixel Upsample',
    group: 'conv',
    ports: 1,
    fields: [
      { key: 'factor', label: 'Factor', type: 'int', def: 2, min: 2, max: 4 },
      { key: 'filters', label: 'Out channels', type: 'int', def: 1, min: 1, max: 16 },
      { key: 'kernel', label: 'Kernel', type: 'int', def: 3, min: 1, max: 7 },
      { key: 'activation', label: 'Activation', type: 'select', def: 'linear', options: ['linear', 'relu', 'tanh', 'sigmoid'] }
    ],
    infer(ins, cfg) {
      if (!ins[0] || ins[0].length !== 4) return null;
      const f = int(cfg.factor, 2);
      return [ins[0][0], int(cfg.filters, 1), ins[0][2] * f, ins[0][3] * f];
    },
    params(ins, cfg) {
      if (!ins[0] || ins[0].length !== 4) return [];
      const f = int(cfg.factor, 2);
      const k = int(cfg.kernel, 3);
      const oc = int(cfg.filters, 1);
      return [
        { suffix: 'kernel', shape: [oc * f * f, ins[0][1], k, k], init: 'he' },
        { suffix: 'bias', shape: [oc * f * f], init: 'zeros' }
      ];
    },
    run(ctx, ins, cfg, p) {
      const k = int(cfg.kernel, 3);
      const pad = Math.floor(k / 2);
      let y = A.conv2d(ins[0], p.kernel, p.bias, { stride: 1, pad });
      y = A.channelToSpace(y, int(cfg.factor, 2));
      return A.applyActivation(y, cfg.activation || 'linear');
    },
    macs(ins, outs, cfg) {
      if (!ins[0] || !outs) return 0;
      const k = int(cfg.kernel, 3);
      const f = int(cfg.factor, 2);
      return ins[0][2] * ins[0][3] * int(cfg.filters, 1) * f * f * ins[0][1] * k * k;
    },
    text(cfg) {
      return '×' + int(cfg.factor, 2) + ' subpixel → ' + int(cfg.filters, 1);
    }
  },

  pad2d: {
    key: 'pad2d',
    label: 'Pad 2D',
    group: 'shape',
    ports: 1,
    fields: [{ key: 'amount', label: 'Padding', type: 'int', def: 1, min: 0, max: 8 }],
    infer(ins, cfg) {
      if (!ins[0] || ins[0].length !== 4) return null;
      const p = int(cfg.amount, 1);
      return [ins[0][0], ins[0][1], ins[0][2] + 2 * p, ins[0][3] + 2 * p];
    },
    params() {
      return [];
    },
    run(ctx, ins, cfg, p, shapes) {
      const amount = int(cfg.amount, 1);
      const s = ins[0].shape;
      const out = zeros([s[0], s[1], s[2] + 2 * amount, s[3] + 2 * amount]);
      for (let n = 0; n < s[0]; n++) {
        for (let c = 0; c < s[1]; c++) {
          for (let h = 0; h < s[2]; h++) {
            for (let w = 0; w < s[3]; w++) {
              out.d[((n * s[1] + c) * (s[2] + 2 * amount) + h + amount) * (s[3] + 2 * amount) + w + amount] =
                ins[0].d[((n * s[1] + c) * s[2] + h) * s[3] + w];
            }
          }
        }
      }
      const node = new Node(out.d, out.shape);
      node.parents = [ins[0]];
      node.op = 'pad2d';
      node.bwd = (gy) => {
        const g = new Float32Array(ins[0].d.length);
        for (let n = 0; n < s[0]; n++) {
          for (let c = 0; c < s[1]; c++) {
            for (let h = 0; h < s[2]; h++) {
              for (let w = 0; w < s[3]; w++) {
                g[((n * s[1] + c) * s[2] + h) * s[3] + w] =
                  gy[((n * s[1] + c) * (s[2] + 2 * amount) + h + amount) * (s[3] + 2 * amount) + w + amount];
              }
            }
          }
        }
        if (!ins[0].g) ins[0].g = new Float32Array(ins[0].d.length);
        for (let i = 0; i < g.length; i++) ins[0].g[i] += g[i];
      };
      return node;
    },
    macs() {
      return 0;
    },
    text(cfg) {
      return int(cfg.amount, 1) + 'px';
    }
  },

  embedding: {
    key: 'embedding',
    label: 'Embedding',
    group: 'sequence',
    ports: 1,
    fields: [
      { key: 'vocab', label: 'Vocab size', type: 'int', def: 256, min: 2, max: 200000 },
      { key: 'dim', label: 'Embedding dim', type: 'int', def: 16, min: 1, max: 512 }
    ],
    infer(ins, cfg) {
      if (!ins[0]) return null;
      const B = ins[0][0];
      const T = ins[0].length > 1 ? shapeSize(ins[0].slice(1)) : 1;
      return [B, T, int(cfg.dim, 16)];
    },
    params(ins, cfg) {
      return [{ suffix: 'table', shape: [int(cfg.vocab, 256), int(cfg.dim, 16)], init: 'normal', std: 0.1 }];
    },
    run(ctx, ins, cfg, p) {
      const sh = ins[0].shape;
      const x = sh.length === 2 ? ins[0] : A.reshape(ins[0], [sh[0], shapeSize(sh.slice(1))]);
      return A.embedding(x, p.table);
    },
    macs() {
      return 0;
    },
    text(cfg) {
      return int(cfg.vocab, 256) + ' → ' + int(cfg.dim, 16);
    }
  },

  gru: {
    key: 'gru',
    label: 'GRU',
    group: 'sequence',
    ports: 1,
    fields: [
      { key: 'units', label: 'Units', type: 'int', def: 32, min: 1, max: 512 },
      { key: 'returnSeq', label: 'Return sequences', type: 'bool', def: true },
      { key: 'bidirectional', label: 'Bidirectional', type: 'bool', def: false }
    ],
    infer(ins, cfg) {
      if (!ins[0] || ins[0].length !== 3) return null;
      const u = int(cfg.units, 32);
      const h = cfg.bidirectional ? u * 2 : u;
      return cfg.returnSeq === false ? [ins[0][0], h] : [ins[0][0], ins[0][1], h];
    },
    params(ins, cfg) {
      if (!ins[0] || ins[0].length !== 3) return [];
      const u = int(cfg.units, 32);
      const f = ins[0][2];
      const out = [
        { suffix: 'Wx', shape: [f, 3 * u], init: 'glorot' },
        { suffix: 'Wh', shape: [u, 3 * u], init: 'orthogonal' },
        { suffix: 'b', shape: [3 * u], init: 'zeros' }
      ];
      if (cfg.bidirectional) {
        out.push({ suffix: 'bWx', shape: [f, 3 * u], init: 'glorot' });
        out.push({ suffix: 'bWh', shape: [u, 3 * u], init: 'orthogonal' });
        out.push({ suffix: 'bb', shape: [3 * u], init: 'zeros' });
      }
      return out;
    },
    run(ctx, ins, cfg, p) {
      const x = toRank3(ins[0]);
      const fwd = A.gru(x, { Wx: p.Wx, Wh: p.Wh, b: p.b }, true);
      let seq = fwd;
      if (cfg.bidirectional) {
        const bwd = A.flipTime(A.gru(A.flipTime(x), { Wx: p.bWx, Wh: p.bWh, b: p.bb }, true));
        seq = A.concat([fwd, bwd], 2);
      }
      if (cfg.returnSeq === false) return A.lastTimestep(seq);
      return seq;
    },
    macs(ins, outs, cfg) {
      if (!ins[0] || ins[0].length !== 3) return 0;
      const u = int(cfg.units, 32);
      const dir = cfg.bidirectional ? 2 : 1;
      return ins[0][1] * dir * (3 * u * ins[0][2] + 3 * u * u);
    },
    text(cfg) {
      return int(cfg.units, 32) + (cfg.bidirectional ? ' bi' : '') + (cfg.returnSeq === false ? ' last' : ' seq');
    }
  },

  lstm: {
    key: 'lstm',
    label: 'LSTM',
    group: 'sequence',
    ports: 1,
    fields: [
      { key: 'units', label: 'Units', type: 'int', def: 32, min: 1, max: 512 },
      { key: 'returnSeq', label: 'Return sequences', type: 'bool', def: true },
      { key: 'bidirectional', label: 'Bidirectional', type: 'bool', def: false }
    ],
    infer(ins, cfg) {
      if (!ins[0] || ins[0].length !== 3) return null;
      const u = int(cfg.units, 32);
      const h = cfg.bidirectional ? u * 2 : u;
      return cfg.returnSeq === false ? [ins[0][0], h] : [ins[0][0], ins[0][1], h];
    },
    params(ins, cfg) {
      if (!ins[0] || ins[0].length !== 3) return [];
      const u = int(cfg.units, 32);
      const f = ins[0][2];
      const out = [
        { suffix: 'Wx', shape: [f, 4 * u], init: 'glorot' },
        { suffix: 'Wh', shape: [u, 4 * u], init: 'orthogonal' },
        { suffix: 'b', shape: [4 * u], init: 'zeros' }
      ];
      if (cfg.bidirectional) {
        out.push({ suffix: 'bWx', shape: [f, 4 * u], init: 'glorot' });
        out.push({ suffix: 'bWh', shape: [u, 4 * u], init: 'orthogonal' });
        out.push({ suffix: 'bb', shape: [4 * u], init: 'zeros' });
      }
      return out;
    },
    run(ctx, ins, cfg, p) {
      const x = toRank3(ins[0]);
      const fwd = A.lstm(x, { Wx: p.Wx, Wh: p.Wh, b: p.b }, true);
      let seq = fwd;
      if (cfg.bidirectional) {
        const bwd = A.flipTime(A.lstm(A.flipTime(x), { Wx: p.bWx, Wh: p.bWh, b: p.bb }, true));
        seq = A.concat([fwd, bwd], 2);
      }
      if (cfg.returnSeq === false) return A.lastTimestep(seq);
      return seq;
    },
    macs(ins, outs, cfg) {
      if (!ins[0] || ins[0].length !== 3) return 0;
      const u = int(cfg.units, 32);
      const dir = cfg.bidirectional ? 2 : 1;
      return ins[0][1] * dir * (4 * u * ins[0][2] + 4 * u * u);
    },
    text(cfg) {
      return int(cfg.units, 32) + (cfg.bidirectional ? ' bi' : '') + (cfg.returnSeq === false ? ' last' : ' seq');
    }
  },

  temporalpool: {
    key: 'temporalpool',
    label: 'Temporal Pool',
    group: 'sequence',
    ports: 1,
    fields: [{ key: 'mode', label: 'Mode', type: 'select', def: 'mean', options: ['mean', 'last', 'max'] }],
    infer(ins) {
      if (!ins[0] || ins[0].length !== 3) return null;
      return [ins[0][0], ins[0][2]];
    },
    params() {
      return [];
    },
    run(ctx, ins, cfg) {
      const x = toRank3(ins[0]);
      if (cfg.mode === 'last') return A.lastTimestep(x);
      if (cfg.mode === 'max') {
        const B = x.shape[0];
        const T = x.shape[1];
        const H = x.shape[2];
        const d = new Float32Array(B * H);
        const arg = new Int32Array(B * H);
        for (let n = 0; n < B; n++) {
          for (let j = 0; j < H; j++) {
            let best = -Infinity;
            let bi = 0;
            for (let t = 0; t < T; t++) {
              const v = x.d[(n * T + t) * H + j];
              if (v > best) {
                best = v;
                bi = t;
              }
            }
            d[n * H + j] = best;
            arg[n * H + j] = bi;
          }
        }
        const node = new Node(d, [B, H]);
        node.parents = [x];
        node.op = 'temporal_max';
        node.bwd = (gy) => {
          const g = new Float32Array(x.d.length);
          for (let n = 0; n < B; n++) {
            for (let j = 0; j < H; j++) g[(n * T + arg[n * H + j]) * H + j] = gy[n * H + j];
          }
          if (!x.g) x.g = new Float32Array(x.d.length);
          for (let i = 0; i < g.length; i++) x.g[i] += g[i];
        };
        return node;
      }
      return A.meanTimesteps(x);
    },
    macs(ins) {
      return ins[0] ? shapeSize(ins[0]) : 0;
    },
    text(cfg) {
      return cfg.mode || 'mean';
    }
  },

  l2normalize: {
    key: 'l2normalize',
    label: 'L2 Normalize',
    group: 'regularize',
    ports: 1,
    fields: [],
    infer(ins) {
      return ins[0] ? ins[0].slice() : null;
    },
    params() {
      return [];
    },
    run(ctx, ins) {
      return A.l2normalize(ins[0]);
    },
    macs(ins) {
      return ins[0] ? shapeSize(ins[0]) * 2 : 0;
    },
    text() {
      return 'l2';
    }
  },

  gaussiannoise: {
    key: 'gaussiannoise',
    label: 'Gaussian Noise',
    group: 'regularize',
    ports: 1,
    fields: [{ key: 'std', label: 'Std dev', type: 'float', def: 0.05, step: 0.01 }],
    infer(ins) {
      return ins[0] ? ins[0].slice() : null;
    },
    params() {
      return [];
    },
    run(ctx, ins, cfg) {
      if (ctx.mode !== 'train') return ins[0];
      const std = num(cfg.std, 0.05);
      const d = new Float32Array(ins[0].d.length);
      for (let i = 0; i < d.length; i++) d[i] = ins[0].d[i] + std * ctx.rng.gauss();
      const node = new Node(d, ins[0].shape);
      node.parents = [ins[0]];
      node.op = 'gaussian_noise';
      node.bwd = (gy) => {
        if (!ins[0].g) ins[0].g = new Float32Array(ins[0].d.length);
        for (let i = 0; i < gy.length; i++) ins[0].g[i] += gy[i];
      };
      return node;
    },
    macs() {
      return 0;
    },
    text(cfg) {
      return 'σ=' + num(cfg.std, 0.05);
    }
  }
};

export const LAYER_GROUPS = [
  { key: 'core', label: 'Core' },
  { key: 'conv', label: 'Convolution' },
  { key: 'pool', label: 'Pooling' },
  { key: 'sequence', label: 'Sequence' },
  { key: 'shape', label: 'Shape & merge' },
  { key: 'regularize', label: 'Normalise & regularise' }
];

export function layerFields(type) {
  const L = LAYERS[type];
  return L ? L.fields : [];
}

export function defaultConfig(type) {
  const cfg = {};
  const L = LAYERS[type];
  if (!L) return cfg;
  for (const f of L.fields) cfg[f.key] = f.def;
  return cfg;
}

export function isLayerType(t) {
  return Object.prototype.hasOwnProperty.call(LAYERS, t);
}
