import { analyze } from '../core/model.js';
import { defaultConfig } from '../core/layers.js';
import { toFloat16Array, fromFloat16Array, precisionByKey } from '../core/quant.js';

export const TF_PRODUCER = 'Rootism Labs';

const DT_FLOAT = 'DT_FLOAT';
const DT_HALF = 'DT_HALF';
const DT_INT32 = 'DT_INT32';
const DT_BOOL = 'DT_BOOL';

function num(v, d) {
  const n = typeof v === 'string' ? parseFloat(v) : v;
  return Number.isFinite(n) ? n : d;
}

function int(v, d) {
  return Math.round(num(v, d));
}

function b64(bytes) {
  let binary = '';
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  if (typeof btoa === 'function') return btoa(binary);
  return Buffer.from(bytes).toString('base64');
}

function encodeString(text) {
  return b64(new TextEncoder().encode(String(text)));
}

function shapeProto(dims) {
  return { dim: dims.map((d) => ({ size: String(d) })) };
}

export function attrType(value) {
  return { type: value };
}

export function attrStr(value) {
  return { s: encodeString(value) };
}

export function attrInt(value) {
  return { i: String(Math.round(value)) };
}

export function attrFloat(value) {
  return { f: value };
}

export function attrBool(value) {
  return { b: !!value };
}

export function attrShape(dims) {
  return { shape: shapeProto(dims) };
}

export function attrIntList(values) {
  return { list: { i: values.map((v) => String(Math.round(v))) } };
}

export function attrFloatList(values) {
  return { list: { f: values.map((v) => v) } };
}

export function attrTypeList(values) {
  return { list: { type: values.slice() } };
}

export function attrTensorValue(dtype, dims) {
  return { tensor: { dtype, tensorShape: shapeProto(dims) } };
}

export class TfGraph {
  constructor(options) {
    const opts = options || {};
    this.nodes = [];
    this.nodeNames = new Set();
    this.weights = [];
    this.weightBytes = [];
    this.weightTotal = 0;
    this.precision = precisionByKey(opts.precision || 'fp32');
    this.dtype = this.precision.key === 'fp16' ? DT_HALF : DT_FLOAT;
    this.inputs = [];
    this.outputs = [];
    this.counter = 0;
    this.intCache = new Map();
    this.floatCache = new Map();
  }

  tmp(prefix) {
    this.counter += 1;
    return (prefix || 'tmp') + '_' + this.counter;
  }

  unique(name) {
    let candidate = String(name).replace(/[^A-Za-z0-9_.\-/]/g, '_');
    if (!this.nodeNames.has(candidate)) {
      this.nodeNames.add(candidate);
      return candidate;
    }
    let i = 1;
    while (this.nodeNames.has(candidate + '_' + i)) i++;
    candidate = candidate + '_' + i;
    this.nodeNames.add(candidate);
    return candidate;
  }

  add(name, op, inputs, attrs) {
    const finalName = this.unique(name);
    const node = { name: finalName, op };
    const ins = (inputs || []).filter((i) => i !== undefined && i !== null);
    if (ins.length) node.input = ins;
    if (attrs && Object.keys(attrs).length) node.attr = attrs;
    this.nodes.push(node);
    return finalName;
  }

  placeholder(name, dims, dtype) {
    const d = [-1].concat(dims.slice());
    this.inputs.push({ name, dims: d, dtype: dtype || DT_FLOAT });
    return this.add(name, 'Placeholder', [], {
      dtype: attrType(dtype || DT_FLOAT),
      shape: { shape: shapeProto(d) }
    });
  }

  constFloat(name, dims, values) {
    const flat = values instanceof Float32Array ? values : Float32Array.from(values);
    let bytes;
    let dtype = DT_FLOAT;
    if (this.precision.key === 'fp16') {
      const half = toFloat16Array(flat);
      bytes = new Uint8Array(half.buffer, half.byteOffset, half.byteLength);
      dtype = DT_HALF;
    } else {
      bytes = new Uint8Array(flat.buffer, flat.byteOffset, flat.byteLength);
    }
    this.weights.push({ name, shape: dims.slice(), dtype: this.precision.key === 'fp16' ? 'float16' : 'float32', elements: flat.length });
    this.weightBytes.push(bytes);
    this.weightTotal += bytes.length;
    return this.add(name, 'Const', [], {
      dtype: attrType(dtype),
      value: attrTensorValue(dtype, dims)
    });
  }

  constInt32(name, dims, values) {
    const key = 'i:' + dims.join(',') + ':' + Array.from(values).join(',');
    if (this.intCache.has(key)) return this.intCache.get(key);
    const created = this.constInt32Raw(name, dims, values);
    this.intCache.set(key, created);
    return created;
  }

  constInt32Raw(name, dims, values) {
    const arr = values instanceof Int32Array ? values : Int32Array.from(values, (v) => Math.round(v));
    const bytes = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
    this.weights.push({ name, shape: dims.slice(), dtype: 'int32', elements: arr.length });
    this.weightBytes.push(bytes);
    this.weightTotal += bytes.length;
    return this.add(name, 'Const', [], {
      dtype: attrType(DT_INT32),
      value: attrTensorValue(DT_INT32, dims)
    });
  }

  constScalar(name, value, dtype) {
    if ((dtype || DT_INT32) === DT_INT32) return this.constInt32(name, [], [value]);
    const key = 'f:' + value;
    if (this.floatCache.has(key)) return this.floatCache.get(key);
    const created = this.constFloat(name, [], [value]);
    this.floatCache.set(key, created);
    return created;
  }

  identity(value, name) {
    return this.add(name || this.tmp('id'), 'Identity', [value], { T: attrType(DT_FLOAT) });
  }

  transpose(value, perm, name) {
    const permConst = this.constInt32(this.tmp('perm'), [perm.length], perm);
    return this.add(name || this.tmp('transpose'), 'Transpose', [value, permConst], {
      T: attrType(DT_FLOAT),
      Tperm: attrType(DT_INT32)
    });
  }

  reshape(value, dims, name) {
    const shapeConst = this.constInt32(this.tmp('shape'), [dims.length], dims);
    return this.add(name || this.tmp('reshape'), 'Reshape', [value, shapeConst], {
      T: attrType(DT_FLOAT),
      Tshape: attrType(DT_INT32)
    });
  }

  binary(op, a, bValue, name, extraAttrs) {
    return this.add(name || this.tmp(op.toLowerCase()), op, [a, bValue], Object.assign({ T: attrType(DT_FLOAT) }, extraAttrs || {}));
  }

  unary(op, value, name, extraAttrs) {
    return this.add(name || this.tmp(op.toLowerCase()), op, [value], Object.assign({ T: attrType(DT_FLOAT) }, extraAttrs || {}));
  }

  activation(value, kind, namePrefix) {
    const k = String(kind || 'linear');
    const p = namePrefix || 'act';
    switch (k) {
      case 'relu': return this.unary('Relu', value, p + '/Relu');
      case 'sigmoid': return this.unary('Sigmoid', value, p + '/Sigmoid');
      case 'tanh': return this.unary('Tanh', value, p + '/Tanh');
      case 'elu': return this.unary('Elu', value, p + '/Elu');
      case 'softplus': return this.unary('Softplus', value, p + '/Softplus');
      case 'leaky_relu': return this.unary('LeakyRelu', value, p + '/LeakyRelu', { alpha: attrFloat(0.1) });
      case 'hard_sigmoid': {
        const scaled = this.binary('Mul', value, this.constScalar(this.tmp('a'), 0.2, DT_FLOAT), p + '/hs_mul');
        const shifted = this.binary('Add', scaled, this.constScalar(this.tmp('b'), 0.5, DT_FLOAT), p + '/hs_add');
        const clippedMin = this.binary('Maximum', shifted, this.constScalar(this.tmp('z'), 0, DT_FLOAT), p + '/hs_max');
        return this.binary('Minimum', clippedMin, this.constScalar(this.tmp('o'), 1, DT_FLOAT), p + '/hs_min');
      }
      case 'gelu': {
        const c0 = this.constScalar(this.tmp('c0'), 0.7978845608028654, DT_FLOAT);
        const c1 = this.constScalar(this.tmp('c1'), 0.044715, DT_FLOAT);
        const half = this.constScalar(this.tmp('half'), 0.5, DT_FLOAT);
        const one = this.constScalar(this.tmp('one'), 1, DT_FLOAT);
        const x2 = this.binary('Mul', value, value, p + '/gelu_x2');
        const x3 = this.binary('Mul', x2, value, p + '/gelu_x3');
        const t1 = this.binary('Mul', c1, x3, p + '/gelu_t1');
        const t2 = this.binary('Add', value, t1, p + '/gelu_t2');
        const t3 = this.binary('Mul', c0, t2, p + '/gelu_t3');
        const th = this.unary('Tanh', t3, p + '/gelu_tanh');
        const t4 = this.binary('Add', one, th, p + '/gelu_t4');
        const t5 = this.binary('Mul', value, t4, p + '/gelu_t5');
        return this.binary('Mul', half, t5, p + '/gelu_out');
      }
      default: return value;
    }
  }

  reduceMean(value, axes, keepDims, name) {
    const axesConst = this.constInt32(this.tmp('axes'), [axes.length], axes);
    return this.add(name || this.tmp('mean'), 'Mean', [value, axesConst], {
      T: attrType(DT_FLOAT),
      Tidx: attrType(DT_INT32),
      keep_dims: attrBool(keepDims)
    });
  }

  reduceMax(value, axes, keepDims, name) {
    const axesConst = this.constInt32(this.tmp('axes'), [axes.length], axes);
    return this.add(name || this.tmp('max'), 'Max', [value, axesConst], {
      T: attrType(DT_FLOAT),
      Tidx: attrType(DT_INT32),
      keep_dims: attrBool(keepDims)
    });
  }

  reduceSum(value, axes, keepDims, name) {
    const axesConst = this.constInt32(this.tmp('axes'), [axes.length], axes);
    return this.add(name || this.tmp('sum'), 'Sum', [value, axesConst], {
      T: attrType(DT_FLOAT),
      Tidx: attrType(DT_INT32),
      keep_dims: attrBool(keepDims)
    });
  }

  stridedSlice(value, begin, end, strides, name, options) {
    const opts = options || {};
    const rank = begin.length;
    const beginMask = opts.beginMask === undefined ? (rank === 3 ? 5 : rank === 2 ? 1 : 0) : opts.beginMask;
    const endMask = opts.endMask === undefined ? (rank === 3 ? 5 : rank === 2 ? 1 : 0) : opts.endMask;
    const b = this.constInt32(this.tmp('begin'), [rank], begin);
    const e = this.constInt32(this.tmp('end'), [rank], end);
    const s = this.constInt32(this.tmp('strides'), [rank], strides);
    return this.add(name || this.tmp('slice'), 'StridedSlice', [value, b, e, s], {
      T: attrType(DT_FLOAT),
      Index: attrType(DT_INT32),
      begin_mask: attrInt(beginMask),
      end_mask: attrInt(endMask),
      ellipsis_mask: attrInt(0),
      new_axis_mask: attrInt(0),
      shrink_axis_mask: attrInt(opts.shrinkAxis || 0)
    });
  }

  squeeze(value, dims, name) {
    return this.add(name || this.tmp('squeeze'), 'Squeeze', [value], {
      T: attrType(DT_FLOAT),
      squeeze_dims: attrIntList(dims)
    });
  }

  pack(values, axis, name) {
    return this.add(name || this.tmp('pack'), 'Pack', values, {
      T: attrType(DT_FLOAT),
      N: attrInt(values.length),
      axis: attrInt(axis)
    });
  }

  concat(values, axis, name) {
    const axisConst = this.constInt32(this.tmp('axis'), [], [axis]);
    return this.add(name || this.tmp('concat'), 'ConcatV2', values.concat([axisConst]), {
      T: attrType(DT_FLOAT),
      Tidx: attrType(DT_INT32),
      N: attrInt(values.length)
    });
  }

  declareOutput(name, dims) {
    this.outputs.push({ name, dims: dims.slice(), dtype: DT_FLOAT });
    return name;
  }

  weightData() {
    const out = new Uint8Array(this.weightTotal);
    let p = 0;
    for (const chunk of this.weightBytes) {
      out.set(chunk, p);
      p += chunk.length;
    }
    return out;
  }

  modelJson(options) {
    const opts = options || {};
    const signatureInputs = {};
    for (const i of this.inputs) {
      signatureInputs[i.name] = { name: i.name + ':0', dtype: i.dtype, tensorShape: shapeProto(i.dims) };
    }
    const signatureOutputs = {};
    for (const o of this.outputs) {
      signatureOutputs[o.name] = { name: o.name + ':0', dtype: o.dtype, tensorShape: shapeProto(o.dims) };
    }
    return {
      format: 'graph-model',
      generatedBy: TF_PRODUCER,
      convertedBy: TF_PRODUCER,
      modelTopology: {
        node: this.nodes,
        library: {},
        versions: { producer: 1 }
      },
      weightsManifest: [{
        paths: [opts.weightsPath || 'model.weights.bin'],
        weights: this.weights.map((w) => ({ name: w.name, shape: w.shape, dtype: w.dtype }))
      }],
      signature: { inputs: signatureInputs, outputs: signatureOutputs },
      userDefinedMetadata: opts.userDefinedMetadata || undefined
    };
  }
}

function permuteTensor(data, shape, perm) {
  const newShape = perm.map((p) => shape[p]);
  const out = new Float32Array(data.length);
  const strides = new Array(shape.length);
  let s = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = s;
    s *= shape[i];
  }
  const newStrides = new Array(newShape.length);
  let t = 1;
  for (let i = newShape.length - 1; i >= 0; i--) {
    newStrides[i] = t;
    t *= newShape[i];
  }
  const idx = new Array(shape.length).fill(0);
  for (let o = 0; o < out.length; o++) {
    let rem = o;
    for (let i = 0; i < newShape.length; i++) {
      idx[perm[i]] = Math.floor(rem / newStrides[i]);
      rem -= Math.floor(rem / newStrides[i]) * newStrides[i];
    }
    let src = 0;
    for (let i = 0; i < shape.length; i++) src += idx[i] * strides[i];
    out[o] = data[src];
  }
  return { data: out, shape: newShape };
}

function permutePad(mode, kernel) {
  if (mode === 'valid' || mode === undefined || mode === null || mode === 0 || mode === '0') return 0;
  if (mode === 'same') return Math.floor(kernel / 2);
  const n = parseInt(mode, 10);
  return Number.isFinite(n) ? n : Math.floor(kernel / 2);
}

function bilinearMatrix(outLen, inLen, factor, transposed) {
  const m = new Float32Array(outLen * inLen);
  for (let o = 0; o < outLen; o++) {
    const src = (o + 0.5) / factor - 0.5;
    const i0 = Math.max(0, Math.min(inLen - 1, Math.floor(src)));
    const i1 = Math.max(0, Math.min(inLen - 1, i0 + 1));
    const w = Math.max(0, Math.min(1, src - i0));
    if (transposed) {
      m[i0 * outLen + o] += 1 - w;
      m[i1 * outLen + o] += w;
    } else {
      m[o * inLen + i0] += 1 - w;
      m[o * inLen + i1] += w;
    }
  }
  return m;
}

function effectiveKernel(k, dilation) {
  const d = Math.max(1, int(dilation, 1));
  return k + (k - 1) * (d - 1);
}

function pickWeight(W, names) {
  for (const n of names) if (W[n]) return W[n];
  return null;
}

function extractGateColumns(data, rows, totalCols, start, width) {
  const out = new Float32Array(rows * width);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < width; c++) out[r * width + c] = data[r * totalCols + start + c];
  }
  return out;
}

export function buildTfGraph(model, options) {
  const opts = options || {};
  const spec = model.spec;
  const info = analyze(spec);
  if (!info.valid) throw new Error('graph is not valid');

  const g = new TfGraph({ precision: opts.precision || 'fp32' });
  const weightList = model.weights();
  const W = {};
  for (const w of weightList) W[w.name] = w;

  const nodeById = {};
  let inputId = null;
  for (const n of spec.nodes) {
    nodeById[n.id] = n;
    if (n.type === 'input') inputId = n.id;
  }
  const edgesByTarget = {};
  for (const e of spec.edges) {
    if (!edgesByTarget[e.to]) edgesByTarget[e.to] = [];
    edgesByTarget[e.to].push(e);
  }
  for (const key of Object.keys(edgesByTarget)) edgesByTarget[key].sort((p, q) => (p.toPort || 0) - (q.toPort || 0));

  const inputName = opts.inputName || (spec.input && spec.input.name) || 'input';
  const outputName = opts.outputName || 'output';
  const unsupported = [];
  const notes = [];
  const valueName = {};
  const valueShape = {};
  for (const id of info.order) valueShape[id] = info.shapes[id];
  valueShape[inputId] = [null].concat(spec.input.shape.slice());

  const x = g.placeholder(inputName, spec.input.shape.slice(), DT_FLOAT);
  valueName[inputId] = x;

  function inputsFor(id) {
    return (edgesByTarget[id] || []).map((e) => valueName[e.from]);
  }

  function shapeOfSource(id) {
    const list = edgesByTarget[id] || [];
    return list.length ? valueShape[list[0].from] : null;
  }

  function putWeight(name, dims, data) {
    return g.constFloat(name, dims, data);
  }

  function weightConst(prefix, w, perm) {
    if (!w) throw new Error('missing weight ' + prefix);
    let data = w.data;
    let shape = w.shape.slice();
    if (perm) {
      const p = permuteTensor(data, shape, perm);
      data = p.data;
      shape = p.shape;
    }
    return putWeight(prefix, shape, data);
  }

  function nhwc(value, shape) {
    if (shape.length === 4) return g.transpose(value, [0, 2, 3, 1]);
    return value;
  }

  function nchw(value, shape) {
    if (shape.length === 4) return g.transpose(value, [0, 3, 1, 2]);
    return value;
  }

  function padZero(value, padsValue, name) {
    const zero = g.constScalar(name + '/pad_zero', 0, DT_FLOAT);
    return g.add(name + '/pad', 'Pad', [value, padsValue, zero], { T: attrType(DT_FLOAT), Tpaddings: attrType(DT_INT32) });
  }

  function padNhwc(value, padH, padW, name) {
    if (padH <= 0 && padW <= 0) return value;
    const pads = g.constInt32(name + '/pads', [4, 2], [0, 0, padH, padH, padW, padW, 0, 0]);
    return padZero(value, pads, name);
  }

  function padNegative(value, padH, padW, name) {
    if (padH <= 0 && padW <= 0) return value;
    const pads = g.constInt32(name + '/pads', [4, 2], [0, 0, padH, padH, padW, padW, 0, 0]);
    const zeroed = padZero(value, pads, name);
    const ones = g.unary('OnesLike', value, name + '/ones');
    const mask = padZero(ones, pads, name + '/mask');
    const one = g.constScalar(name + '/mask_one', 1, DT_FLOAT);
    const big = g.constScalar(name + '/mask_big', 1e30, DT_FLOAT);
    const outside = g.binary('Sub', one, mask, name + '/outside');
    return g.binary('Sub', zeroed, g.binary('Mul', outside, big, name + '/penalty'), name + '/shifted');
  }

  function convAttrs(strideH, strideW, dilH, dilW) {
    return {
      T: attrType(DT_FLOAT),
      strides: attrIntList([1, strideH, strideW, 1]),
      padding: attrStr('VALID'),
      data_format: attrStr('NHWC'),
      dilations: attrIntList([1, dilH, dilW, 1]),
      use_cudnn_on_gpu: attrBool(true)
    };
  }

  function groupedConv(id, xVal, xShape, kw, bw, cfg, groups, k, s, dil, pad) {
    const cin = xShape[1];
    const perGroupIn = cin / groups;
    const perGroupOut = kw.shape[0] / groups;
    const pieces = [];
    for (let gi = 0; gi < groups; gi++) {
      const channelSlice = g.stridedSlice(xVal, [0, 0, 0, gi * perGroupIn], [0, 0, 0, (gi + 1) * perGroupIn], [1, 1, 1, 1], id + '/gslice_' + gi, { beginMask: 7, endMask: 7 });
      const sub = new Float32Array(perGroupOut * perGroupIn * k * k);
      for (let m = 0; m < perGroupOut; m++) {
        const srcStart = (gi * perGroupOut + m) * perGroupIn * k * k;
        sub.set(kw.data.subarray(srcStart, srcStart + perGroupIn * k * k), m * perGroupIn * k * k);
      }
      const permuted = permuteTensor(sub, [perGroupOut, perGroupIn, k, k], [2, 3, 1, 0]);
      const filter = putWeight(id + '/kernel_g' + gi, permuted.shape, permuted.data);
      const padded = padNhwc(channelSlice, pad, pad, id + '/pad_g' + gi);
      pieces.push(g.add(id + '/conv_g' + gi, 'Conv2D', [padded, filter], convAttrs(s, s, dil, dil)));
    }
    return pieces.length === 1 ? pieces[0] : g.concat(pieces, 3, id + '/group_concat');
  }

  function emitConv2d(id, cfg, xVal, xShape, weightNames, groups) {
    const k = int(cfg.kernel, 3);
    const s = int(cfg.stride, 1);
    const dil = int(cfg.dilation, 1);
    const ek = effectiveKernel(k, dil);
    const pad = permutePad(cfg.padding, ek);
    const kw = pickWeight(W, weightNames.kernel);
    const bw = pickWeight(W, weightNames.bias);
    if (!kw) throw new Error('missing conv kernel for ' + id);
    const cin = xShape[1];
    const inNhwcRaw = nhwc(xVal, xShape);
    const isDepthwise = groups > 1 && kw.shape[1] === 1 && groups === cin;
    let y;
    if (isDepthwise) {
      const mult = kw.shape[0] / cin;
      const asGrouped = new Float32Array(kw.data);
      const permuted = permuteTensor(asGrouped, [cin, mult, k, k], [2, 3, 0, 1]);
      const filter = putWeight(id + '/depthwise_kernel', permuted.shape, permuted.data);
      const padded = padZero(inNhwcRaw, g.constInt32(id + '/pads', [4, 2], [0, 0, pad, pad, pad, pad, 0, 0]), id);
      y = g.add(id + '/conv', 'DepthwiseConv2dNative', [padded, filter], {
        T: attrType(DT_FLOAT),
        strides: attrIntList([1, s, s, 1]),
        padding: attrStr('VALID'),
        data_format: attrStr('NHWC'),
        dilations: attrIntList([1, dil, dil, 1])
      });
    } else if (groups > 1) {
      y = groupedConv(id, inNhwcRaw, xShape, kw, bw, cfg, groups, k, s, dil, pad);
    } else {
      const permuted = permuteTensor(kw.data, kw.shape.slice(), [2, 3, 1, 0]);
      const filter = putWeight(id + '/kernel', permuted.shape, permuted.data);
      const padded = padNhwc(inNhwcRaw, pad, pad, id);
      y = g.add(id + '/conv', 'Conv2D', [padded, filter], convAttrs(s, s, dil, dil));
    }
    if (bw && cfg.useBias !== false) {
      const biasValue = g.reshape(g.constFloat(id + '/bias_flat', [bw.data.length], bw.data), [1, 1, 1, bw.data.length]);
      y = g.binary('Add', y, biasValue, id + '/bias_add');
    }
    y = nchw(y, xShape);
    return g.activation(y, cfg.activation === undefined ? 'relu' : cfg.activation, id);
  }

  function emitConv1d(id, cfg, xVal, xShape) {
    const k = int(cfg.kernel, 3);
    const s = int(cfg.stride, 1);
    const dil = int(cfg.dilation, 1);
    const ek = effectiveKernel(k, dil);
    const pad = permutePad(cfg.padding, ek);
    const kw = pickWeight(W, [id + '.kernel']);
    const bw = pickWeight(W, [id + '.bias']);
    if (!kw) throw new Error('missing conv1d kernel for ' + id);
    const permuted = permuteTensor(kw.data, kw.shape.slice(), [2, 3, 1, 0]);
    const filter = putWeight(id + '/kernel', [1, k, kw.shape[1], kw.shape[0]], permuted.data);
    const asImage = g.reshape(xVal, [-1, 1, xShape[1], xShape[2]]);
    const padded = pad > 0 ? padNhwc(asImage, 0, pad, id) : asImage;
    let y = g.add(id + '/conv', 'Conv2D', [padded, filter], convAttrs(1, s, 1, dil));
    const outFilters = kw.shape[0];
    if (bw && cfg.useBias !== false) {
      const biasValue = g.reshape(g.constFloat(id + '/bias_flat', [bw.data.length], bw.data), [1, 1, 1, outFilters]);
      y = g.binary('Add', y, biasValue, id + '/bias_add');
    }
    const outLen = Math.floor((xShape[1] + 2 * pad - ek) / s) + 1;
    const flat = g.reshape(y, [-1, outLen, outFilters]);
    return g.activation(flat, cfg.activation === undefined ? 'relu' : cfg.activation, id);
  }

  function emitPool2d(id, cfg, xVal, xShape, op) {
    const k = int(cfg.kernel, 2);
    const s = int(cfg.stride, k);
    const pad = op === 'MaxPool' ? permutePad(cfg.padding, k) : 0;
    let inNhwc = nhwc(xVal, xShape);
    if (pad > 0) {
      inNhwc = op === 'MaxPool' ? padNegative(inNhwc, pad, pad, id) : padNhwc(inNhwc, pad, pad, id);
    }
    const y = g.add(id + '/pool', op, [inNhwc], {
      T: attrType(DT_FLOAT),
      ksize: attrIntList([1, k, k, 1]),
      strides: attrIntList([1, s, s, 1]),
      padding: attrStr('VALID'),
      data_format: attrStr('NHWC')
    });
    return nchw(y, xShape);
  }

  function emitPool1d(id, cfg, xVal, xShape, op) {
    const k = int(cfg.kernel, 2);
    const s = int(cfg.stride, k);
    const asImage = g.reshape(xVal, [-1, 1, xShape[1], xShape[2]]);
    const y = g.add(id + '/pool', op, [asImage], {
      T: attrType(DT_FLOAT),
      ksize: attrIntList([1, 1, k, 1]),
      strides: attrIntList([1, 1, s, 1]),
      padding: attrStr('VALID'),
      data_format: attrStr('NHWC')
    });
    const outLen = Math.floor((xShape[1] - k) / s) + 1;
    return g.reshape(y, [-1, outLen, xShape[2]]);
  }

  function gateSlice(value, start, width, rank, name) {
    if (rank === 2) {
      return g.stridedSlice(value, [0, start], [0, start + width], [1, 1], name);
    }
    return g.stridedSlice(value, [0, 0, start], [0, 0, start + width], [1, 1, 1], name);
  }

  function emitRnn(id, cfg, xVal, xShape, kind) {
    const units = int(cfg.units, 32);
    const T = xShape[1];
    const F = xShape[2];
    const bidirectional = !!cfg.bidirectional;
    const gates = kind === 'lstm' ? 4 : 3;
    const gateOrder = kind === 'lstm' ? ['i', 'f', 'o', 'c'] : ['r', 'z', 'n'];
    const suffixes = bidirectional ? [['Wx', 'Wh', 'b'], ['bWx', 'bWh', 'bb']] : [['Wx', 'Wh', 'b']];
    const streams = [];

    for (let d = 0; d < suffixes.length; d++) {
      const [wxName, whName, bName] = suffixes[d];
      const wx = pickWeight(W, [id + '.' + wxName]);
      const wh = pickWeight(W, [id + '.' + whName]);
      const bb = pickWeight(W, [id + '.' + bName]);
      if (!wx || !wh || !bb) throw new Error('missing rnn weights for ' + id);
      const prefix = id + (d === 1 ? '_bw' : '_fw');
      const gateX = {};
      const gateH = {};
      const gateB = {};
      for (let gi = 0; gi < gates; gi++) {
        const name = gateOrder[gi];
        gateX[name] = putWeight(prefix + '/Wx_' + name, [F, units], extractGateColumns(wx.data, F, gates * units, gi * units, units));
        gateH[name] = putWeight(prefix + '/Wh_' + name, [units, units], extractGateColumns(wh.data, units, gates * units, gi * units, units));
        gateB[name] = putWeight(prefix + '/b_' + name, [units], bb.data.subarray(gi * units, (gi + 1) * units));
      }
      const projections = {};
      for (const name of gateOrder) {
        projections[name] = g.add(prefix + '/xw_' + name, 'MatMul', [xVal, gateX[name]], {
          T: attrType(DT_FLOAT), transpose_a: attrBool(false), transpose_b: attrBool(false)
        });
      }
      let h = g.constFloat(prefix + '/zero_h', [1, units], new Float32Array(units));
      let c = kind === 'lstm' ? g.constFloat(prefix + '/zero_c', [1, units], new Float32Array(units)) : null;
      const one = kind === 'lstm' ? null : g.constScalar(prefix + '/one', 1, DT_FLOAT);
      const hidden = [];
      const steps = d === 1 && cfg.returnSeq === false ? 1 : T;

      for (let t = 0; t < steps; t++) {
        const step = d === 1 ? T - 1 - t : t;
        const stepValues = {};
        for (const name of gateOrder) {
          stepValues[name] = g.stridedSlice(projections[name], [0, step, 0], [0, step + 1, 0], [1, 1, 1], prefix + '/' + name + '_' + t, { shrinkAxis: 2 });
        }
        const full = {};
        for (const name of gateOrder) {
          const hw = g.add(prefix + '/hw_' + name + '_' + t, 'MatMul', [h, gateH[name]], {
            T: attrType(DT_FLOAT), transpose_a: attrBool(false), transpose_b: attrBool(false)
          });
          full[name] = g.binary('Add', hw, gateB[name], prefix + '/full_' + name + '_' + t);
        }
        let hNext;
        if (kind === 'lstm') {
          const iGate = g.unary('Sigmoid', g.binary('Add', stepValues.i, full.i, prefix + '/si_' + t), prefix + '/i_' + t);
          const fGate = g.unary('Sigmoid', g.binary('Add', stepValues.f, full.f, prefix + '/sf_' + t), prefix + '/f_' + t);
          const oGate = g.unary('Sigmoid', g.binary('Add', stepValues.o, full.o, prefix + '/so_' + t), prefix + '/o_' + t);
          const cCand = g.unary('Tanh', g.binary('Add', stepValues.c, full.c, prefix + '/sc_' + t), prefix + '/c_' + t);
          const fc = g.binary('Mul', fGate, c, prefix + '/fc_' + t);
          const ig = g.binary('Mul', iGate, cCand, prefix + '/ig_' + t);
          c = g.binary('Add', fc, ig, prefix + '/cnext_' + t);
          hNext = g.binary('Mul', oGate, g.unary('Tanh', c, prefix + '/tanhc_' + t), prefix + '/h_' + t);
        } else {
          const rGate = g.unary('Sigmoid', g.binary('Add', stepValues.r, full.r, prefix + '/sr_' + t), prefix + '/r_' + t);
          const zGate = g.unary('Sigmoid', g.binary('Add', stepValues.z, full.z, prefix + '/sz_' + t), prefix + '/z_' + t);
          const reset = g.binary('Mul', rGate, full.n, prefix + '/reset_' + t);
          const nGate = g.unary('Tanh', g.binary('Add', stepValues.n, reset, prefix + '/npre_' + t), prefix + '/n_' + t);
          const left = g.binary('Mul', g.binary('Sub', one, zGate, prefix + '/omz_' + t), nGate, prefix + '/left_' + t);
          hNext = g.binary('Add', left, g.binary('Mul', zGate, h, prefix + '/right_' + t), prefix + '/h_' + t);
        }
        h = hNext;
        hidden.push(h);
      }
      streams.push({ seq: d === 1 ? hidden.slice().reverse() : hidden, first: hidden[0], last: hidden[hidden.length - 1], backward: d === 1 });
    }

    if (cfg.returnSeq === false) {
      const finals = streams.map((s) => (s.backward ? s.first : s.last));
      return finals.length === 1 ? finals[0] : g.concat(finals, 1, id + '/concat_final');
    }
    const packed = streams.map((s, i) => g.pack(s.seq, 1, id + '/pack_' + i));
    return packed.length === 1 ? packed[0] : g.concat(packed, 2, id + '/concat_seq');
  }

  function emitBatchNorm(id, cfg, xVal, xShape) {
    const gamma = pickWeight(W, [id + '.gamma']);
    const beta = pickWeight(W, [id + '.beta']);
    const mean = pickWeight(W, [id + '.runningMean', id + '.mean']);
    const variance = pickWeight(W, [id + '.runningVar', id + '.var']);
    if (!gamma || !beta || !mean || !variance) throw new Error('missing batchnorm statistics for ' + id + '; train the model before exporting');
    const channels = gamma.data.length;
    const eps = num(cfg.eps, 1e-5);
    const is4d = xShape.length === 4;
    let v = xVal;
    if (is4d) v = nhwc(v, xShape);
    const meanDims = is4d ? [1, 1, 1, channels] : [1, channels];
    const meanConst = putWeight(id + '/mean', meanDims, mean.data);
    const varPlus = new Float32Array(channels);
    for (let i = 0; i < channels; i++) varPlus[i] = 1 / Math.sqrt(Math.max(variance.data[i], 0) + eps);
    const scaleConst = putWeight(id + '/rstd', meanDims, varPlus);
    const combined = new Float32Array(channels);
    for (let i = 0; i < channels; i++) combined[i] = gamma.data[i] * varPlus[i];
    const shift = new Float32Array(channels);
    for (let i = 0; i < channels; i++) shift[i] = beta.data[i] - mean.data[i] * varPlus[i] * gamma.data[i];
    const betaConst = putWeight(id + '/beta', meanDims, shift);
    let y = g.binary('Sub', v, meanConst, id + '/bn_sub');
    y = g.binary('Mul', y, scaleConst, id + '/bn_scale');
    y = g.binary('Mul', y, putWeight(id + '/gamma', meanDims, combined), id + '/bn_gamma');
    y = g.binary('Add', y, betaConst, id + '/bn_beta');
    if (is4d) y = nchw(y, xShape);
    return y;
  }

  function emitNode(id) {
    const node = nodeById[id];
    const cfg = Object.assign({}, defaultConfig(node.type), node.cfg || {});
    const ins = inputsFor(id);
    const xVal = ins[0];
    const xShape = shapeOfSource(id);
    if (!xVal) throw new Error('node ' + id + ' has no input edge');

    switch (node.type) {
      case 'input':
        return xVal;
      case 'dense': {
        const kw = pickWeight(W, [id + '.kernel']);
        if (!kw) throw new Error('missing dense kernel for ' + id);
        const kConst = putWeight(id + '/kernel', kw.shape.slice(), kw.data);
        const mm = g.add(id + '/matmul', 'MatMul', [xVal, kConst], {
          T: attrType(DT_FLOAT), transpose_a: attrBool(false), transpose_b: attrBool(false)
        });
        let y = mm;
        const bw = pickWeight(W, [id + '.bias']);
        if (bw && cfg.useBias !== false) {
          const bConst = putWeight(id + '/bias', [bw.data.length], bw.data);
          y = g.binary('Add', y, bConst, id + '/bias_add');
        }
        return g.activation(y, cfg.activation === undefined ? 'relu' : cfg.activation, id);
      }
      case 'conv2d':
        return emitConv2d(id, cfg, xVal, xShape, { kernel: [id + '.kernel'], bias: [id + '.bias'] }, Math.max(1, int(cfg.groups, 1)));
      case 'depthwiseconv2d':
        return emitConv2d(id, cfg, xVal, xShape, { kernel: [id + '.kernel'], bias: [id + '.bias'] }, xShape[1]);
      case 'separableconv2d': {
        const k = int(cfg.kernel, 3);
        const s = int(cfg.stride, 1);
        const pad = permutePad(cfg.padding, k);
        const dw = pickWeight(W, [id + '.dw']);
        const dwb = pickWeight(W, [id + '.dwbias']);
        const pw = pickWeight(W, [id + '.pw']);
        const pwb = pickWeight(W, [id + '.pwbias']);
        if (!dw || !pw) throw new Error('missing separable weights for ' + id);
        const act = cfg.activation === undefined ? 'relu' : cfg.activation;
        const cin = xShape[1];
        const dwPerm = permuteTensor(dw.data, dw.shape, [2, 3, 0, 1]);
        const dwFilter = putWeight(id + '/depthwise_kernel', dwPerm.shape, dwPerm.data);
        const inNhwc = padNhwc(nhwc(xVal, xShape), pad, pad, id + '/dw');
        let y = g.add(id + '/depthwise', 'DepthwiseConv2dNative', [inNhwc, dwFilter], {
          T: attrType(DT_FLOAT),
          strides: attrIntList([1, s, s, 1]),
          padding: attrStr('VALID'),
          data_format: attrStr('NHWC'),
          dilations: attrIntList([1, 1, 1, 1])
        });
        if (dwb) {
          const biasValue = g.reshape(g.constFloat(id + '/dw_bias_flat', [dwb.data.length], dwb.data), [1, 1, 1, dwb.data.length]);
          y = g.binary('Add', y, biasValue, id + '/dw_bias');
        }
        y = g.activation(y, act, id + '/dw');
        const pwPerm = permuteTensor(pw.data, pw.shape, [2, 3, 1, 0]);
        const pwFilter = putWeight(id + '/pointwise_kernel', [1, 1, cin, pw.shape[0]], pwPerm.data);
        y = g.add(id + '/pointwise', 'Conv2D', [y, pwFilter], convAttrs(1, 1, 1, 1));
        if (pwb) {
          const biasValue = g.reshape(g.constFloat(id + '/pw_bias_flat', [pwb.data.length], pwb.data), [1, 1, 1, pwb.data.length]);
          y = g.binary('Add', y, biasValue, id + '/pw_bias');
        }
        y = g.activation(y, act, id + '/pw');
        return nchw(y, xShape);
      }
      case 'conv1d':
        return emitConv1d(id, cfg, xVal, xShape);
      case 'subpixelupsample': {
        const k = int(cfg.kernel, 3);
        const factor = int(cfg.factor, 2);
        const kw = pickWeight(W, [id + '.kernel']);
        const bw = pickWeight(W, [id + '.bias']);
        if (!kw) throw new Error('missing subpixel kernel for ' + id);
        const convCfg = { kernel: k, stride: 1, padding: String(Math.floor(k / 2)), dilation: 1, activation: 'linear', useBias: cfg.useBias !== false };
        const convOut = emitConv2d(id + '/conv', convCfg, xVal, xShape, { kernel: [id + '.kernel'], bias: [id + '.bias'] }, 1);
        void kw;
        void bw;
        const oc = kw.shape[0] / (factor * factor);
        const H = xShape[2];
        const Wd = xShape[3];
        const nhwcValue = nhwc(convOut, info.shapes[id + '/conv'] || [1, kw.shape[0], H, Wd]);
        const asBlocks = g.reshape(nhwcValue, [-1, H, Wd, oc, factor, factor]);
        const reordered = g.transpose(asBlocks, [0, 1, 4, 2, 5, 3]);
        const merged = g.reshape(reordered, [-1, H * factor, Wd * factor, oc]);
        const back = g.transpose(merged, [0, 3, 1, 2]);
        return g.activation(back, cfg.activation || 'linear', id);
      }
      case 'maxpool2d':
        return emitPool2d(id, cfg, xVal, xShape, 'MaxPool');
      case 'avgpool2d':
        return emitPool2d(id, cfg, xVal, xShape, 'AvgPool');
      case 'globalavgpool2d': {
        const nhwcValue = nhwc(xVal, xShape);
        const m = g.reduceMean(nhwcValue, [1, 2], false, id + '/gap');
        return m;
      }
      case 'maxpool1d':
        return emitPool1d(id, cfg, xVal, xShape, 'MaxPool');
      case 'avgpool1d':
        return emitPool1d(id, cfg, xVal, xShape, 'AvgPool');
      case 'globalavgpool1d':
        return g.reduceMean(xVal, [1], false, id + '/gmp');
      case 'globalmaxpool1d':
        return g.reduceMax(xVal, [1], false, id + '/gmp');
      case 'temporalpool': {
        if (cfg.mode === 'last') {
          const t = xShape[1] - 1;
          const sliced = g.stridedSlice(xVal, [0, t, 0], [0, t + 1, 0], [1, 1, 1], id + '/last');
          return g.squeeze(sliced, [1], id + '/last_sq');
        }
        if (cfg.mode === 'max') return g.reduceMax(xVal, [1], false, id + '/tp');
        return g.reduceMean(xVal, [1], false, id + '/tp');
      }
      case 'upsample2d': {
        const factor = int(cfg.factor, 2);
        const H = xShape[2];
        const Wd = xShape[3];
        const oH = H * factor;
        const oW = Wd * factor;
        if (cfg.mode === 'bilinear') {
          const awValue = putWeight(id + '/resize_w', [Wd, oW], bilinearMatrix(oW, Wd, factor, true));
          const ahValue = putWeight(id + '/resize_h', [H, oH], bilinearMatrix(oH, H, factor, true));
          const wideW = g.add(id + '/resize_w_mul', 'MatMul', [g.reshape(xVal, [-1, Wd]), awValue], { T: attrType(DT_FLOAT), transpose_a: attrBool(false), transpose_b: attrBool(false) });
          const asWide = g.reshape(wideW, [-1, xShape[1], H, oW]);
          const rowsH = g.reshape(g.transpose(asWide, [0, 1, 3, 2], id + '/resize_t1'), [-1, H]);
          const wideH = g.add(id + '/resize_h_mul', 'MatMul', [rowsH, ahValue], { T: attrType(DT_FLOAT), transpose_a: attrBool(false), transpose_b: attrBool(false) });
          return g.transpose(g.reshape(wideH, [-1, xShape[1], oW, oH]), [0, 1, 3, 2], id + '/resize_t2');
        }
        const inNhwc = nhwc(xVal, xShape);
        const sizeConst = g.constInt32(id + '/size', [2], [oH, oW]);
        const y = g.add(id + '/resize', 'ResizeNearestNeighbor', [inNhwc, sizeConst], {
          T: attrType(DT_FLOAT),
          Tdim: attrType(DT_INT32),
          align_corners: attrBool(false),
          half_pixel_centers: attrBool(false)
        });
        return nchw(y, xShape);
      }
      case 'pad2d': {
        const amount = int(cfg.amount, 1);
        const pads = g.constInt32(id + '/pads', [4, 2], [0, 0, 0, 0, amount, amount, amount, amount]);
        const value = g.constScalar(id + '/pad_value', 0, DT_FLOAT);
        return g.add(id + '/pad', 'Pad', [xVal, pads, value], { T: attrType(DT_FLOAT), Tpaddings: attrType(DT_INT32) });
      }
      case 'embedding': {
        const table = pickWeight(W, [id + '.table']);
        if (!table) throw new Error('missing embedding table for ' + id);
        let idxSource = xVal;
        if (xShape.length !== 2) {
          const flat = xShape.slice(1).reduce((a, c) => a * c, 1);
          idxSource = g.reshape(xVal, [-1, flat]);
        }
        const idxInt = g.add(id + '/cast', 'Cast', [idxSource], { SrcT: attrType(DT_FLOAT), DstT: attrType(DT_INT32) });
        const tableConst = putWeight(id + '/table', table.shape.slice(), table.data);
        const axisConst = g.constInt32(id + '/axis', [], [0]);
        return g.add(id + '/gather', 'GatherV2', [tableConst, idxInt, axisConst], {
          Tparams: attrType(DT_FLOAT), Tindices: attrType(DT_INT32), Taxis: attrType(DT_INT32), batch_dims: attrInt(0)
        });
      }
      case 'gru':
        return emitRnn(id, cfg, xVal, xShape, 'gru');
      case 'lstm':
        return emitRnn(id, cfg, xVal, xShape, 'lstm');
      case 'layernorm': {
        const gamma = pickWeight(W, [id + '.gamma']);
        const beta = pickWeight(W, [id + '.beta']);
        if (!gamma || !beta) throw new Error('missing layernorm weights for ' + id);
        const eps = num(cfg.eps, 1e-5);
        const last = xShape[xShape.length - 1];
        const mean = g.reduceMean(xVal, [-1], true, id + '/ln_mean');
        const diff = g.binary('Sub', xVal, mean, id + '/ln_sub');
        const sq = g.binary('Mul', diff, diff, id + '/ln_sq');
        const variance = g.reduceMean(sq, [-1], true, id + '/ln_var');
        const epsConst = g.constScalar(id + '/ln_eps', eps, DT_FLOAT);
        const added = g.binary('Add', variance, epsConst, id + '/ln_add');
        const rstd = g.unary('Rsqrt', added, id + '/ln_rsqrt');
        const normed = g.binary('Mul', diff, rstd, id + '/ln_norm');
        const scaled = g.binary('Mul', normed, putWeight(id + '/gamma', [last], gamma.data), id + '/ln_gamma');
        return g.binary('Add', scaled, putWeight(id + '/beta', [last], beta.data), id + '/ln_beta');
      }
      case 'batchnorm':
        return emitBatchNorm(id, cfg, xVal, xShape);
      case 'l2normalize': {
        const sq = g.binary('Mul', xVal, xVal, id + '/l2_sq');
        const sum = g.reduceSum(sq, [-1], true, id + '/l2_sum');
        const eps = g.constScalar(id + '/l2_eps', 1e-12, DT_FLOAT);
        const added = g.binary('Add', sum, eps, id + '/l2_add');
        const denom = g.unary('Sqrt', added, id + '/l2_sqrt');
        return g.binary('RealDiv', denom === null ? xVal : xVal, denom, id + '/l2_div');
      }
      case 'flatten': {
        const total = xShape.slice(1).reduce((a, c) => a * c, 1);
        return g.reshape(xVal, [-1, total], id + '/flatten');
      }
      case 'reshape': {
        const target = String(cfg.shape || '').split(',').map((v) => parseInt(v.trim(), 10)).filter((v) => Number.isFinite(v));
        if (!target.length) throw new Error('reshape node ' + id + ' has no target shape');
        return g.reshape(xVal, [-1].concat(target), id + '/reshape');
      }
      case 'softmax':
        return g.unary('Softmax', xVal, id + '/softmax');
      case 'activation':
        return g.activation(xVal, cfg.kind || 'relu', id);
      case 'concat':
        return g.concat(ins, int(cfg.axis, -1), id + '/concat');
      case 'add':
        return g.binary('Add', ins[0], ins[1], id + '/add');
      case 'multiply':
        return g.binary('Mul', ins[0], ins[1], id + '/mul');
      case 'dropout':
        notes.push('Dropout node ' + id + ' exported as identity; inference graphs do not apply dropout.');
        return g.identity(xVal, id + '/identity');
      case 'gaussiannoise':
        notes.push('Gaussian noise node ' + id + ' exported as identity; inference graphs do not inject noise.');
        return g.identity(xVal, id + '/identity');
      default:
        unsupported.push(node.type);
        notes.push('Layer type ' + node.type + ' has no TensorFlow op and was exported as identity.');
        return g.identity(xVal, id + '/identity');
    }
  }

  for (const id of info.order) {
    if (id === inputId) continue;
    valueName[id] = emitNode(id);
  }

  let outputValue = valueName[spec.output];
  if (!outputValue) throw new Error('output node ' + spec.output + ' was not emitted');
  const head = opts.head === undefined ? spec.head : opts.head;
  const headApplied = applyTfHead(g, outputValue, head, notes);
  outputValue = headApplied.value;
  g.identity(outputValue, outputName);

  const outShape = info.outputShape ? info.outputShape.slice() : [1];
  g.declareOutput(outputName, outShape);

  return {
    graph: g,
    info,
    unsupported,
    notes,
    head: headApplied.applied,
    headKind: head || 'linear',
    inputName,
    outputName,
    outputShape: outShape,
    precision: g.precision.key,
    fixedSequenceLength: spec.nodes.some((n) => n.type === 'gru' || n.type === 'lstm') ? spec.input.shape[0] : null
  };
}

function applyTfHead(g, value, head, notes) {
  switch (String(head || 'linear')) {
    case 'softmax':
    case 'softmax-seq':
      return { value: g.unary('Softmax', value, 'head/softmax'), applied: 'softmax' };
    case 'sigmoid':
    case 'sigmoid-seq':
    case 'mask':
    case 'heatmap':
      return { value: g.unary('Sigmoid', value, 'head/sigmoid'), applied: 'sigmoid' };
    case 'embedding':
      notes.push('Embedding head exported as raw vectors; apply L2 normalisation in the consuming code for cosine distances.');
      return { value, applied: 'identity' };
    case 'det':
      notes.push('Detection head exported as raw grid outputs: channel 0 is objectness logits, channels 1-4 are box coordinates.');
      return { value, applied: 'identity' };
    default:
      return { value, applied: 'linear' };
  }
}

export function exportTfjs(model, options) {
  const opts = options || {};
  if ((opts.precision || 'fp32') === 'int8' || (opts.precision || 'fp32') === 'uint8') {
    throw new Error('TensorFlow.js export supports float32 and float16. Integer precision is available from the safetensors export.');
  }
  const built = buildTfGraph(model, opts);
  const weightData = built.graph.weightData();
  const userDefinedMetadata = Object.assign({
    producer: TF_PRODUCER,
    precision: built.precision,
    head: built.headKind,
    params: String(model.paramCount ? model.paramCount() : 0)
  }, opts.metadata || {});
  const modelJson = built.graph.modelJson({
    weightsPath: opts.weightsPath || 'model.weights.bin',
    userDefinedMetadata
  });
  return {
    modelJson,
    weightData,
    nodeCount: built.graph.nodes.length,
    weightCount: built.graph.weights.length,
    unsupported: built.unsupported,
    notes: built.notes,
    head: built.head,
    inputName: built.inputName,
    outputName: built.outputName,
    inputShape: model.spec.input.shape.slice(),
    outputShape: built.outputShape,
    precision: built.precision,
    fixedSequenceLength: built.fixedSequenceLength,
    byteLength: weightData.length + JSON.stringify(modelJson).length
  };
}

export function tfjsFloat16Decode(data) {
  return fromFloat16Array(new Uint16Array(data.buffer, data.byteOffset, data.byteLength / 2));
}

export { toFloat16Array };
