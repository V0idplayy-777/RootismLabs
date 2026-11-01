import { ProtoWriter, ONNX_FIELDS as F, ONNX_DTYPE, ONNX_ATTR as AT } from '../core/protobuf.js';
import { analyze } from '../core/model.js';
import { defaultConfig } from '../core/layers.js';
import { toFloat16Array, toBFloat16Array, precisionByKey, quantizeInt8 } from '../core/quant.js';

export const DEFAULT_OPSET = 20;
export const IR_VERSION = 10;
export const PRODUCER = 'Rootism Labs';
export const PRODUCER_VERSION = '1';

function num(v, d) {
  const n = typeof v === 'string' ? parseFloat(v) : v;
  return Number.isFinite(n) ? n : d;
}

function int(v, d) {
  return Math.round(num(v, d));
}

function parsePadMode(mode) {
  if (mode === 'valid' || mode === undefined || mode === null) return 0;
  if (mode === 'same') return -1;
  const n = parseInt(mode, 10);
  return Number.isFinite(n) ? n : -1;
}

function padAmount(mode, effectiveKernel, inDim) {
  const m = parsePadMode(mode);
  if (m === 0) return 0;
  if (m > 0) return m;
  return Math.floor(effectiveKernel / 2);
}

function effectiveKernel(k, dilation) {
  const d = Math.max(1, int(dilation, 1));
  return k + (k - 1) * (d - 1);
}

function outDim(inDim, k, stride, pad) {
  return Math.floor((inDim + 2 * pad - k) / stride) + 1;
}

export class OnnxBuilder {
  constructor(options) {
    const opts = options || {};
    this.opset = int(opts.opset, DEFAULT_OPSET);
    this.nodes = [];
    this.initializers = [];
    this.inputs = [];
    this.outputs = [];
    this.valueInfo = [];
    this.counter = 0;
    this.constCache = new Map();
    this.weightDtype = ONNX_DTYPE.float32;
    this.elemType = ONNX_DTYPE.float32;
  }

  tmp(prefix) {
    this.counter += 1;
    return (prefix || 'tmp') + '_' + this.counter;
  }

  addInitializer(name, dims, dtype, bytes, floatValues) {
    this.initializers.push({ name, dims: dims.slice(), dtype, bytes, floatValues: floatValues || null });
    return name;
  }

  constFloat32(values, dims, prefix) {
    const data = values instanceof Float32Array ? values : Float32Array.from(values);
    const shape = dims && dims.length ? dims.slice() : [data.length];
    const name = this.tmp(prefix || 'c');
    if (this.elemType === ONNX_DTYPE.float16) {
      const h = toFloat16Array(data);
      this.addInitializer(name, shape, ONNX_DTYPE.float16, new Uint8Array(h.buffer, h.byteOffset, h.byteLength), data);
      return name;
    }
    this.addInitializer(name, shape, ONNX_DTYPE.float32, new Uint8Array(data.buffer, data.byteOffset, data.byteLength), data);
    return name;
  }

  constFloat32Strict(values, dims, prefix) {
    const data = values instanceof Float32Array ? values : Float32Array.from(values);
    const shape = dims && dims.length ? dims.slice() : [data.length];
    const name = this.tmp(prefix || 'cf');
    this.addInitializer(name, shape, ONNX_DTYPE.float32, new Uint8Array(data.buffer, data.byteOffset, data.byteLength), null);
    return name;
  }

  constInt64(values, prefix) {
    const key = 'i64:' + values.join(',');
    if (this.constCache.has(key)) return this.constCache.get(key);
    const data = new BigInt64Array(values.length);
    for (let i = 0; i < values.length; i++) data[i] = BigInt(Math.round(values[i]));
    const name = this.tmp(prefix || 'shape');
    this.addInitializer(name, [data.length], ONNX_DTYPE.int64, new Uint8Array(data.buffer), null);
    this.constCache.set(key, name);
    return name;
  }

  constFloatScalar(value, prefix) {
    return this.constFloat32([value], [], prefix);
  }

  weight(name, dims, values, dtype) {
    const data = values instanceof Float32Array ? values : Float32Array.from(values);
    if (dtype === ONNX_DTYPE.float16) {
      const h = toFloat16Array(data);
      return this.addInitializer(name, dims, ONNX_DTYPE.float16, new Uint8Array(h.buffer, h.byteOffset, h.byteLength), data);
    }
    if (dtype === ONNX_DTYPE.bfloat16) {
      const h = toBFloat16Array(data);
      return this.addInitializer(name, dims, ONNX_DTYPE.bfloat16, new Uint8Array(h.buffer, h.byteOffset, h.byteLength), data);
    }
    return this.addInitializer(name, dims, ONNX_DTYPE.float32, new Uint8Array(data.buffer, data.byteOffset, data.byteLength), data);
  }

  node(opType, inputs, outputs, attrs, name, domain) {
    this.nodes.push({ opType, inputs: inputs.filter((i) => i !== undefined && i !== null), outputs, attrs: attrs || [], name: name || this.tmp(opType.toLowerCase()), domain: domain || '' });
    return outputs[0];
  }

  io(name, elemType, shape, dynamicBatch) {
    return { name, elemType, shape: shape.slice(), dynamicBatch: dynamicBatch !== false };
  }

  declareInput(name, sampleShape, elemType) {
    this.inputs.push(this.io(name, elemType === undefined ? this.elemType : elemType, [-1].concat(sampleShape.slice()), true));
    return name;
  }

  declareOutput(name, fullShape, elemType) {
    this.outputs.push(this.io(name, elemType === undefined ? this.elemType : elemType, fullShape.slice(), true));
    return name;
  }

  transpose(value, perm, prefix) {
    return this.node('Transpose', [value], [this.tmp(prefix || 'tr')], [attrInts('perm', perm)]);
  }

  reshape(value, shape, prefix) {
    return this.node('Reshape', [value, this.constInt64(shape)], [this.tmp(prefix || 'rs')]);
  }

  reduce(op, value, axes, keepDims, prefix) {
    if (this.opset >= 18) {
      return this.node(op, [value, this.constInt64(axes)], [this.tmp(prefix || 'red')], [attrInt('keepdims', keepDims ? 1 : 0)]);
    }
    return this.node(op, [value], [this.tmp(prefix || 'red')], [attrInts('axes', axes), attrInt('keepdims', keepDims ? 1 : 0)]);
  }

  activation(value, kind, prefix) {
    const k = String(kind || 'linear');
    switch (k) {
      case 'relu': return this.node('Relu', [value], [this.tmp(prefix || 'relu')]);
      case 'sigmoid': return this.node('Sigmoid', [value], [this.tmp(prefix || 'sig')]);
      case 'tanh': return this.node('Tanh', [value], [this.tmp(prefix || 'tanh')]);
      case 'elu': return this.node('Elu', [value], [this.tmp(prefix || 'elu')], [attrFloat('alpha', 1)]);
      case 'leaky_relu': return this.node('LeakyRelu', [value], [this.tmp(prefix || 'lrelu')], [attrFloat('alpha', 0.1)]);
      case 'softplus': return this.node('Softplus', [value], [this.tmp(prefix || 'sp')]);
      case 'hard_sigmoid': return this.node('HardSigmoid', [value], [this.tmp(prefix || 'hs')], [attrFloat('alpha', 0.2), attrFloat('beta', 0.5)]);
      case 'selu': return this.node('Selu', [value], [this.tmp(prefix || 'selu')]);
      case 'gelu': return this.gelu(value, prefix);
      default: return value;
    }
  }

  gelu(value, prefix) {
    if (this.opset >= 20) {
      return this.node('Gelu', [value], [this.tmp(prefix || 'gelu')], [attrString('approximate', 'tanh')]);
    }
    const c0 = this.constFloatScalar(0.7978845608028654);
    const c1 = this.constFloatScalar(0.044715);
    const half = this.constFloatScalar(0.5);
    const one = this.constFloatScalar(1);
    const x2 = this.node('Mul', [value, value], [this.tmp('x2')]);
    const x3 = this.node('Mul', [x2, value], [this.tmp('x3')]);
    const t = this.node('Mul', [c1, x3], [this.tmp('t1')]);
    const s = this.node('Add', [value, t], [this.tmp('t2')]);
    const u = this.node('Mul', [c0, s], [this.tmp('t3')]);
    const th = this.node('Tanh', [u], [this.tmp('t4')]);
    const add1 = this.node('Add', [one, th], [this.tmp('t5')]);
    const mulx = this.node('Mul', [value, add1], [this.tmp('t6')]);
    return this.node('Mul', [half, mulx], [this.tmp(prefix || 'gelu')]);
  }

  encode() {
    const graph = new ProtoWriter();
    for (const n of this.nodes) {
      const w = new ProtoWriter();
      for (const i of n.inputs) w.stringField(F.node.input, i);
      for (const o of n.outputs) w.stringField(F.node.output, o);
      w.stringField(F.node.name, n.name);
      w.stringField(F.node.opType, n.opType);
      if (n.domain) w.stringField(F.node.domain, n.domain);
      for (const a of n.attrs) w.messageField(F.node.attribute, a);
      graph.messageField(F.graph.node, w);
    }
    graph.stringField(F.graph.name, this.graphName || 'rootism_graph');
    for (const init of this.initializers) {
      const t = new ProtoWriter();
      if (init.dims.length) t.repeatedVarint(F.tensor.dims, init.dims);
      t.varintField(F.tensor.dataType, init.dtype);
      t.stringField(F.tensor.name, init.name);
      t.bytesField(F.tensor.rawData, init.bytes);
      graph.messageField(F.graph.initializer, t);
    }
    for (const v of this.inputs) graph.messageField(F.graph.input, valueInfoWriter(v));
    for (const v of this.outputs) graph.messageField(F.graph.output, valueInfoWriter(v));
    for (const v of this.valueInfo) graph.messageField(F.graph.valueInfo, valueInfoWriter(v));
    return graph;
  }

  toBytes(options) {
    const opts = options || {};
    this.graphName = opts.graphName || this.graphName || 'rootism_graph';
    const model = new ProtoWriter();
    model.varintField(F.model.irVersion, IR_VERSION);
    const opset = new ProtoWriter();
    opset.stringField(F.opset.domain, '');
    opset.varintField(F.opset.version, this.opset);
    model.messageField(F.model.opsetImport, opset);
    model.stringField(F.model.producerName, opts.producerName || PRODUCER);
    model.stringField(F.model.producerVersion, opts.producerVersion || PRODUCER_VERSION);
    model.stringField(F.model.domain, opts.domain || 'labs.rootism');
    model.varintField(F.model.modelVersion, int(opts.modelVersion, 1));
    if (opts.docString) model.stringField(F.model.docString, opts.docString);
    model.messageField(F.model.graph, this.encode());
    if (opts.metadata) {
      for (const key of Object.keys(opts.metadata)) {
        const entry = new ProtoWriter();
        entry.stringField(F.stringEntry.key, key);
        entry.stringField(F.stringEntry.value, String(opts.metadata[key]));
        model.messageField(F.model.metadataProps, entry);
      }
    }
    return model.finish();
  }
}

function valueInfoWriter(v) {
  const shape = new ProtoWriter();
  for (let i = 0; i < v.shape.length; i++) {
    const dim = new ProtoWriter();
    if (i === 0 && v.dynamicBatch) dim.stringField(F.dim.dimParam, 'batch');
    else dim.varintField(F.dim.dimValue, v.shape[i]);
    shape.messageField(F.shape.dim, dim);
  }
  const tensorType = new ProtoWriter();
  tensorType.varintField(F.tensorType.elemType, v.elemType);
  tensorType.messageField(F.tensorType.shape, shape);
  const type = new ProtoWriter();
  type.messageField(F.typeProto.tensorType, tensorType);
  const out = new ProtoWriter();
  out.stringField(F.valueInfo.name, v.name);
  out.messageField(F.valueInfo.type, type);
  return out;
}

export function attrInt(name, value) {
  const w = new ProtoWriter();
  w.stringField(F.attribute.name, name);
  w.varintField(F.attribute.type, AT.int);
  w.varintField(F.attribute.int, value);
  return w;
}

export function attrFloat(name, value) {
  const w = new ProtoWriter();
  w.stringField(F.attribute.name, name);
  w.varintField(F.attribute.type, AT.float);
  w.floatField(F.attribute.float, value);
  return w;
}

export function attrString(name, value) {
  const w = new ProtoWriter();
  w.stringField(F.attribute.name, name);
  w.varintField(F.attribute.type, AT.string);
  w.bytesField(F.attribute.string, new TextEncoder().encode(String(value)));
  return w;
}

export function attrInts(name, values) {
  const w = new ProtoWriter();
  w.stringField(F.attribute.name, name);
  w.varintField(F.attribute.type, AT.ints);
  for (const v of values) w.varintField(F.attribute.ints, v);
  return w;
}

export function attrFloats(name, values) {
  const w = new ProtoWriter();
  w.stringField(F.attribute.name, name);
  w.varintField(F.attribute.type, AT.floats);
  for (const v of values) w.floatField(F.attribute.floats, v);
  return w;
}

export function attrTensor(name, tensorWriter) {
  const w = new ProtoWriter();
  w.stringField(F.attribute.name, name);
  w.varintField(F.attribute.type, AT.tensor);
  w.messageField(F.attribute.tensor, tensorWriter);
  return w;
}

function reshapeWeight(b, dims, values) {
  const data = values instanceof Float32Array ? values : Float32Array.from(values);
  const name = b.tmp('w');
  return b.weight(name, dims, data, b.weightDtype);
}

function biasShape(rank, channels) {
  const dims = new Array(rank).fill(1);
  dims[1] = channels;
  return dims;
}

function gatePermute(units, blocks, order) {
  const out = [];
  for (const src of order) for (let j = 0; j < units; j++) out.push(src * units + j);
  void blocks;
  return out;
}

function pickWeight(W, names) {
  for (const n of names) if (W[n]) return W[n];
  return null;
}

export function buildOnnxGraph(model, options) {
  const opts = options || {};
  const spec = model.spec;
  const info = analyze(spec);
  if (!info.valid) throw new Error('graph is not valid: ' + (info.errors[0] && (info.errors[0].message || JSON.stringify(info.errors[0]))));

  const precision = precisionByKey(opts.precision || 'fp32');
  const integerWeights = precision.key === 'int8' || precision.key === 'uint8';
  const b = new OnnxBuilder({ opset: int(opts.opset, DEFAULT_OPSET) });
  b.weightDtype = integerWeights ? ONNX_DTYPE.float32 : precision.onnxDtype;
  b.elemType = integerWeights ? ONNX_DTYPE.float32 : precision.onnxDtype;

  const weightList = model.weights();
  const W = {};
  for (const w of weightList) W[w.name] = w;

  const nodeById = {};
  let inputId = null;
  for (const n of spec.nodes) {
    nodeById[n.id] = n;
    if (n.type === 'input') inputId = n.id;
  }
  if (!inputId) throw new Error('graph has no input node');

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

  b.declareInput(inputName, spec.input.shape.slice(), b.elemType);
  valueName[inputId] = inputName;
  valueShape[inputName] = [null].concat(spec.input.shape.slice());

  for (const id of info.order) valueShape[id] = info.shapes[id];

  const shapeOf = (valueKey) => valueShape[valueKey] || null;

  function inputsFor(id) {
    const list = edgesByTarget[id] || [];
    return list.map((e) => valueName[e.from]);
  }

  function shapeOfSource(id) {
    const list = edgesByTarget[id] || [];
    if (!list.length) return null;
    return shapeOf(list[0].from);
  }

  function emitActivation(x, kind, cfg) {
    if (kind === 'linear' || kind === 'none' || kind === undefined) return x;
    void cfg;
    return b.activation(x, kind);
  }

  function emitConv2d(id, cfg, x, xShape) {
    const k = int(cfg.kernel, 3);
    const s = int(cfg.stride, 1);
    const dil = int(cfg.dilation, 1);
    const groups = Math.max(1, int(cfg.groups, 1));
    const ek = effectiveKernel(k, dil);
    const p = padAmount(cfg.padding, ek, xShape[2]);
    const kw = pickWeight(W, [id + '.kernel']);
    if (!kw) throw new Error('missing kernel for ' + id);
    const out = b.node('Conv', [x, reshapeWeight(b, kw.shape.slice(), kw.data)], [b.tmp('conv')], [
      attrInts('kernel_shape', [k, k]),
      attrInts('strides', [s, s]),
      attrInts('pads', [p, p, p, p]),
      attrInts('dilations', [dil, dil]),
      attrInt('group', groups)
    ], id + '_conv');
    let y = out;
    const bw = pickWeight(W, [id + '.bias']);
    if (bw && cfg.useBias !== false) {
      y = b.node('Add', [y, reshapeWeight(b, biasShape(4, bw.data.length), bw.data)], [b.tmp('bias')], [], id + '_bias');
    }
    return emitActivation(y, cfg.activation === undefined ? 'relu' : cfg.activation, cfg);
  }

  function emitConv1d(id, cfg, x, xShape) {
    const k = int(cfg.kernel, 3);
    const s = int(cfg.stride, 1);
    const dil = int(cfg.dilation, 1);
    const ek = effectiveKernel(k, dil);
    const p = padAmount(cfg.padding, ek, xShape[1]);
    const kw = pickWeight(W, [id + '.kernel']);
    if (!kw) throw new Error('missing kernel for ' + id);
    const ncl = b.transpose(x, [0, 2, 1], id + '_ncl');
    const dims = kw.shape.slice();
    const kernelData = kw.data;
    const wDims = [dims[0], dims[1], dims[3]];
    const y = b.node('Conv', [ncl, reshapeWeight(b, wDims, kernelData)], [b.tmp('conv1d')], [
      attrInts('kernel_shape', [k]),
      attrInts('strides', [s]),
      attrInts('pads', [p, p]),
      attrInts('dilations', [dil]),
      attrInt('group', 1)
    ], id + '_conv');
    let z = y;
    const bw = pickWeight(W, [id + '.bias']);
    if (bw && cfg.useBias !== false) {
      z = b.node('Add', [z, reshapeWeight(b, [1, bw.data.length, 1], bw.data)], [b.tmp('bias1d')], [], id + '_bias');
    }
    const nlc = b.transpose(z, [0, 2, 1], id + '_nlc');
    return emitActivation(nlc, cfg.activation === undefined ? 'relu' : cfg.activation, cfg);
  }

  function emitPool1d(id, cfg, x, op) {
    const k = int(cfg.kernel, 2);
    const s = int(cfg.stride, k);
    const ncl = b.transpose(x, [0, 2, 1], id + '_ncl');
    const pooled = b.node(op, [ncl], [b.tmp('pool1d')], [
      attrInts('kernel_shape', [k]),
      attrInts('strides', [s]),
      attrInts('pads', [0, 0])
    ], id + '_pool');
    return b.transpose(pooled, [0, 2, 1], id + '_nlc');
  }

  function emitPool2d(id, cfg, x, xShape, op) {
    const k = int(cfg.kernel, 2);
    const s = int(cfg.stride, k);
    const p = op === 'MaxPool' ? padAmount(cfg.padding, k, xShape[2]) : 0;
    return b.node(op, [x], [b.tmp('pool2d')], [
      attrInts('kernel_shape', [k, k]),
      attrInts('strides', [s, s]),
      attrInts('pads', [p, p, p, p])
    ], id + '_pool');
  }

  function emitRnn(id, cfg, x, xShape, kind) {
    const units = int(cfg.units, 32);
    const bidirectional = !!cfg.bidirectional;
    const dirs = bidirectional ? 2 : 1;
    const gates = kind === 'lstm' ? 4 : 3;
    const blocksIn = kind === 'lstm' ? [0, 2, 1, 3] : [1, 0, 2];
    const rowsIn = gates;
    const suffixes = bidirectional ? [['Wx', 'Wh', 'b'], ['bWx', 'bWh', 'bb']] : [['Wx', 'Wh', 'b']];
    const inputSize = xShape[xShape.length - 1];

    const wStack = new Float32Array(dirs * gates * units * inputSize);
    const rStack = new Float32Array(dirs * gates * units * units);
    const bStack = new Float32Array(dirs * 2 * gates * units);

    for (let d = 0; d < dirs; d++) {
      const [wxName, whName, bName] = suffixes[d];
      const wx = pickWeight(W, [id + '.' + wxName]);
      const wh = pickWeight(W, [id + '.' + whName]);
      const bb = pickWeight(W, [id + '.' + bName]);
      if (!wx || !wh || !bb) throw new Error('missing rnn weights for ' + id);
      const srcCols = rowsIn * units;
      const perm = gatePermute(units, rowsIn, blocksIn);
      for (let r = 0; r < gates * units; r++) {
        const srcRow = r < gates * units ? perm[r] : -1;
        for (let c = 0; c < inputSize; c++) {
          wStack[((d * gates * units) + r) * inputSize + c] = srcRow >= 0 ? wx.data[c * srcCols + srcRow] : 0;
        }
        for (let c = 0; c < units; c++) {
          rStack[((d * gates * units) + r) * units + c] = srcRow >= 0 ? wh.data[c * srcCols + srcRow] : 0;
        }
        bStack[d * 2 * gates * units + gates * units + r] = srcRow >= 0 ? bb.data[srcRow] : 0;
      }
    }

    const seq = b.transpose(x, [1, 0, 2], id + '_seq');
    const wInit = reshapeWeight(b, [dirs, gates * units, inputSize], wStack);
    const rInit = reshapeWeight(b, [dirs, gates * units, units], rStack);
    const bInit = reshapeWeight(b, [dirs, 2 * gates * units], bStack);
    const attrs = [
      attrString('direction', bidirectional ? 'bidirectional' : 'forward'),
      attrInt('hidden_size', units),
      attrInt('layout', 0)
    ];
    if (kind === 'gru') attrs.push(attrInt('linear_before_reset', 1));
    const yName = b.tmp('Y');
    const yhName = b.tmp('Yh');
    b.node(kind === 'lstm' ? 'LSTM' : 'GRU', [seq, wInit, rInit, bInit], [yName, yhName], attrs, id + '_' + kind);

    if (cfg.returnSeq === false) {
      const flat = b.transpose(yhName, [1, 0, 2], id + '_yh');
      return b.reshape(flat, [-1, dirs * units], id + '_out');
    }
    const reordered = b.transpose(yName, [2, 0, 1, 3], id + '_y');
    return b.reshape(reordered, [-1, xShape[1], dirs * units], id + '_out');
  }

  function emitBatchNorm(id, cfg, x, xShape) {
    const gamma = pickWeight(W, [id + '.gamma']);
    const beta = pickWeight(W, [id + '.beta']);
    const mean = pickWeight(W, [id + '.runningMean', id + '.mean']);
    const variance = pickWeight(W, [id + '.runningVar', id + '.var']);
    if (!gamma || !beta || !mean || !variance) throw new Error('missing batchnorm stats for ' + id + ' (train the model before exporting)');
    const channels = gamma.data.length;
    const eps = num(cfg.eps, 1e-5);
    const rank = xShape.length;
    let input = x;
    if (rank === 3) input = b.transpose(x, [0, 2, 1], id + '_ncl');
    const y = b.node('BatchNormalization', [
      input,
      reshapeWeight(b, [channels], gamma.data),
      reshapeWeight(b, [channels], beta.data),
      reshapeWeight(b, [channels], mean.data),
      reshapeWeight(b, [channels], variance.data)
    ], [b.tmp('bn')], [attrFloat('epsilon', eps), attrInt('training_mode', 0)], id + '_bn');
    if (rank === 3) return b.transpose(y, [0, 2, 1], id + '_nlc');
    return y;
  }

  function emitNode(id) {
    const node = nodeById[id];
    const cfg = Object.assign({}, defaultConfig(node.type), node.cfg || {});
    const ins = inputsFor(id);
    const x = ins[0];
    const xShape = shapeOfSource(id);
    if (!x) throw new Error('node ' + id + ' has no input edge');

    switch (node.type) {
      case 'input':
        return x;
      case 'dense': {
        const kw = pickWeight(W, [id + '.kernel']);
        if (!kw) throw new Error('missing kernel for ' + id);
        const mm = b.node('MatMul', [x, reshapeWeight(b, kw.shape.slice(), kw.data)], [b.tmp('mm')], [], id + '_matmul');
        let y = mm;
        const bw = pickWeight(W, [id + '.bias']);
        if (bw && cfg.useBias !== false) y = b.node('Add', [y, reshapeWeight(b, [bw.data.length], bw.data)], [b.tmp('bias')], [], id + '_bias');
        return emitActivation(y, cfg.activation === undefined ? 'relu' : cfg.activation, cfg);
      }
      case 'conv2d':
        return emitConv2d(id, cfg, x, xShape);
      case 'depthwiseconv2d': {
        const groups = xShape[1];
        const localCfg = Object.assign({}, cfg, { groups });
        return emitConv2d(id, localCfg, x, xShape);
      }
      case 'separableconv2d': {
        const k = int(cfg.kernel, 3);
        const s = int(cfg.stride, 1);
        const p = padAmount(cfg.padding, k, xShape[2]);
        const dw = pickWeight(W, [id + '.dw']);
        const dwb = pickWeight(W, [id + '.dwbias']);
        const pw = pickWeight(W, [id + '.pw']);
        const pwb = pickWeight(W, [id + '.pwbias']);
        if (!dw || !pw) throw new Error('missing separable weights for ' + id);
        const act = cfg.activation === undefined ? 'relu' : cfg.activation;
        let y = b.node('Conv', [x, reshapeWeight(b, dw.shape.slice(), dw.data)], [b.tmp('dw')], [
          attrInts('kernel_shape', [k, k]), attrInts('strides', [s, s]), attrInts('pads', [p, p, p, p]),
          attrInts('dilations', [1, 1]), attrInt('group', xShape[1])
        ], id + '_dw');
        if (dwb) y = b.node('Add', [y, reshapeWeight(b, biasShape(4, dwb.data.length), dwb.data)], [b.tmp('dwbias')], [], id + '_dwbias');
        y = b.activation(y, act);
        y = b.node('Conv', [y, reshapeWeight(b, pw.shape.slice(), pw.data)], [b.tmp('pw')], [
          attrInts('kernel_shape', [1, 1]), attrInts('strides', [1, 1]), attrInts('pads', [0, 0, 0, 0]),
          attrInts('dilations', [1, 1]), attrInt('group', 1)
        ], id + '_pw');
        if (pwb) y = b.node('Add', [y, reshapeWeight(b, biasShape(4, pwb.data.length), pwb.data)], [b.tmp('pwbias')], [], id + '_pwbias');
        return b.activation(y, act);
      }
      case 'conv1d':
        return emitConv1d(id, cfg, x, xShape);
      case 'subpixelupsample': {
        const k = int(cfg.kernel, 3);
        const factor = int(cfg.factor, 2);
        const p = Math.floor(k / 2);
        const kw = pickWeight(W, [id + '.kernel']);
        const bw = pickWeight(W, [id + '.bias']);
        if (!kw) throw new Error('missing kernel for ' + id);
        let y = b.node('Conv', [x, reshapeWeight(b, kw.shape.slice(), kw.data)], [b.tmp('subpix')], [
          attrInts('kernel_shape', [k, k]), attrInts('strides', [1, 1]), attrInts('pads', [p, p, p, p]),
          attrInts('dilations', [1, 1]), attrInt('group', 1)
        ], id + '_conv');
        if (bw) y = b.node('Add', [y, reshapeWeight(b, biasShape(4, bw.data.length), bw.data)], [b.tmp('bias')], [], id + '_bias');
        y = b.node('DepthToSpace', [y], [b.tmp('d2s')], [attrInt('blocksize', factor), attrString('mode', 'CRD')], id + '_d2s');
        return b.activation(y, cfg.activation || 'linear');
      }
      case 'maxpool2d':
        return emitPool2d(id, cfg, x, xShape, 'MaxPool');
      case 'avgpool2d':
        return emitPool2d(id, cfg, x, xShape, 'AveragePool');
      case 'globalavgpool2d': {
        const g = b.node('GlobalAveragePool', [x], [b.tmp('gap')], [], id + '_gap');
        return b.reshape(g, [-1, xShape[1]], id + '_gap_r');
      }
      case 'maxpool1d':
        return emitPool1d(id, cfg, x, 'MaxPool');
      case 'avgpool1d':
        return emitPool1d(id, cfg, x, 'AveragePool');
      case 'globalavgpool1d':
        return b.reduce('ReduceMean', x, [1], false, id + '_gmp');
      case 'globalmaxpool1d':
        return b.reduce('ReduceMax', x, [1], false, id + '_gmp');
      case 'temporalpool': {
        if (cfg.mode === 'last') {
          const idx = b.constInt64([xShape[1] - 1], id + '_last');
          const g = b.node('Gather', [x, idx], [b.tmp('last')], [attrInt('axis', 1)], id + '_last_g');
          return b.reshape(g, [-1, xShape[2]], id + '_last_r');
        }
        return b.reduce(cfg.mode === 'max' ? 'ReduceMax' : 'ReduceMean', x, [1], false, id + '_tp');
      }
      case 'upsample2d': {
        const factor = int(cfg.factor, 2);
        const mode = cfg.mode === 'bilinear' ? 'linear' : 'nearest';
        const roi = b.constFloat32Strict([], [0], id + '_roi');
        const scales = b.constFloat32Strict([1, 1, factor, factor], [4], id + '_scales');
        const attrs = [attrString('mode', mode)];
        if (mode === 'nearest') {
          attrs.push(attrString('coordinate_transformation_mode', 'asymmetric'));
          attrs.push(attrString('nearest_mode', 'floor'));
        } else {
          attrs.push(attrString('coordinate_transformation_mode', 'pytorch_half_pixel'));
          attrs.push(attrInt('exclude_outside', 0));
        }
        return b.node('Resize', [x, roi, scales], [b.tmp('resize')], attrs, id + '_resize');
      }
      case 'pad2d': {
        const amount = int(cfg.amount, 1);
        const pads = b.constInt64([0, 0, amount, amount, 0, 0, amount, amount], id + '_pads');
        const value = b.constFloat32([0], [1], id + '_padv');
        return b.node('Pad', [x, pads, value], [b.tmp('pad')], [attrString('mode', 'constant')], id + '_pad');
      }
      case 'embedding': {
        const table = pickWeight(W, [id + '.table']);
        if (!table) throw new Error('missing embedding table for ' + id);
        let idxSource = x;
        if (xShape.length !== 2) {
          const flat = xShape.slice(1).reduce((a, c) => a * c, 1);
          idxSource = b.reshape(x, [-1, flat], id + '_flat');
        }
        const idx = b.node('Cast', [idxSource], [b.tmp('idx')], [attrInt('to', ONNX_DTYPE.int64)], id + '_cast');
        return b.node('Gather', [reshapeWeight(b, table.shape.slice(), table.data), idx], [b.tmp('emb')], [attrInt('axis', 0)], id + '_gather');
      }
      case 'gru':
        return emitRnn(id, cfg, x, xShape, 'gru');
      case 'lstm':
        return emitRnn(id, cfg, x, xShape, 'lstm');
      case 'layernorm': {
        const gamma = pickWeight(W, [id + '.gamma']);
        const beta = pickWeight(W, [id + '.beta']);
        if (!gamma || !beta) throw new Error('missing layernorm weights for ' + id);
        if (b.opset >= 17) {
          return b.node('LayerNormalization', [x, reshapeWeight(b, [gamma.data.length], gamma.data), reshapeWeight(b, [beta.data.length], beta.data)], [b.tmp('ln')], [
            attrInt('axis', -1), attrFloat('epsilon', num(cfg.eps, 1e-5))
          ], id + '_ln');
        }
        const last = xShape[xShape.length - 1];
        const mean = b.reduce('ReduceMean', x, [-1], true, id + '_mean');
        const centered = b.node('Sub', [x, mean], [b.tmp('sub')], [], id + '_sub');
        const sq = b.node('Mul', [centered, centered], [b.tmp('sq')], [], id + '_sq');
        const varName = b.reduce('ReduceMean', sq, [-1], true, id + '_var');
        const epsC = b.constFloatScalar(num(cfg.eps, 1e-5), id + '_eps');
        const added = b.node('Add', [varName, epsC], [b.tmp('adde')], [], id + '_adde');
        const denom = b.node('Sqrt', [added], [b.tmp('sqrt')], [], id + '_sqrt');
        const normed = b.node('Div', [centered, denom], [b.tmp('div')], [], id + '_div');
        const scaled = b.node('Mul', [normed, reshapeWeight(b, [last], gamma.data)], [b.tmp('mul')], [], id + '_mul');
        return b.node('Add', [scaled, reshapeWeight(b, [last], beta.data)], [b.tmp('ln')], [], id + '_ln');
      }
      case 'batchnorm':
        return emitBatchNorm(id, cfg, x, xShape);
      case 'l2normalize': {
        const sq = b.node('Mul', [x, x], [b.tmp('sq')], [], id + '_sq');
        const sum = b.reduce('ReduceSum', sq, [-1], true, id + '_sum');
        const eps = b.constFloatScalar(1e-12, id + '_eps');
        const added = b.node('Add', [sum, eps], [b.tmp('add')], [], id + '_add');
        const denom = b.node('Sqrt', [added], [b.tmp('sqrt')], [], id + '_sqrt');
        return b.node('Div', [x, denom], [b.tmp('l2')], [], id + '_div');
      }
      case 'flatten': {
        const total = xShape.slice(1).reduce((a, c) => a * c, 1);
        return b.reshape(x, [-1, total], id + '_flat');
      }
      case 'reshape': {
        const target = String(cfg.shape || '').split(',').map((v) => parseInt(v.trim(), 10)).filter((v) => Number.isFinite(v));
        if (!target.length) throw new Error('reshape node ' + id + ' has no target shape');
        return b.reshape(x, [-1].concat(target), id + '_rs');
      }
      case 'softmax':
        return b.node('Softmax', [x], [b.tmp('sm')], [attrInt('axis', int(cfg.axis, -1))], id + '_softmax');
      case 'activation':
        return b.activation(x, cfg.kind || 'relu');
      case 'concat':
        return b.node('Concat', ins, [b.tmp('cc')], [attrInt('axis', int(cfg.axis, -1))], id + '_concat');
      case 'add':
        return b.node('Add', [ins[0], ins[1]], [b.tmp('add')], [], id + '_add');
      case 'multiply':
        return b.node('Mul', [ins[0], ins[1]], [b.tmp('mul')], [], id + '_mul');
      case 'dropout':
        notes.push('Dropout node ' + id + ' exported as identity; inference graphs do not apply dropout.');
        return b.node('Identity', [x], [b.tmp('id')], [], id + '_identity');
      case 'gaussiannoise':
        notes.push('Gaussian noise node ' + id + ' exported as identity; inference graphs do not inject noise.');
        return b.node('Identity', [x], [b.tmp('id')], [], id + '_identity');
      default:
        unsupported.push(node.type);
        notes.push('Layer type ' + node.type + ' has no ONNX mapping and was exported as identity.');
        return b.node('Identity', [x], [b.tmp('id')], [], id + '_identity');
    }
  }

  for (const id of info.order) {
    if (id === inputId) continue;
    valueName[id] = emitNode(id);
  }

  let outputValue = valueName[spec.output];
  if (!outputValue) throw new Error('graph output node ' + spec.output + ' was not emitted');
  const head = opts.head === undefined ? spec.head : opts.head;
  const headApplied = applyHead(b, outputValue, head, notes);
  outputValue = headApplied.value;

  const outShape = info.outputShape ? info.outputShape.slice() : [];
  b.declareOutput(outputName, outShape.length ? outShape : [1], b.elemType);
  if (outputValue !== outputName) b.node('Identity', [outputValue], [outputName], [], 'graph_output');

  let quantization = null;
  if (integerWeights) {
    quantization = quantizeBuilderWeights(b, precision.key, notes);
  }

  return {
    builder: b,
    info,
    unsupported,
    notes,
    quantization,
    head: headApplied.applied,
    headKind: head || 'linear',
    inputName,
    outputName,
    outputShape: outShape,
    opset: b.opset,
    precision: precision.key
  };
}

const QDQ_MIN_ELEMENTS = 64;

export function quantizeBuilderWeights(builder, precisionKey, notes) {
  const signed = precisionKey !== 'uint8';
  const kept = [];
  const dequantNodes = [];
  const report = {
    precision: precisionKey,
    scheme: signed ? 'int8 symmetric per output channel' : 'uint8 asymmetric per output channel',
    tensorsQuantized: 0,
    tensorsKept: 0,
    originalBytes: 0,
    quantizedBytes: 0,
    worstRelativeL2: 0,
    worstTensor: null,
    perTensor: []
  };
  for (const init of builder.initializers) {
    const isFloat = init.dtype === ONNX_DTYPE.float32 && init.floatValues && init.floatValues.length;
    if (!isFloat || init.floatValues.length < QDQ_MIN_ELEMENTS) {
      kept.push(init);
      if (isFloat) report.tensorsKept++;
      continue;
    }
    const values = init.floatValues;
    const dims = init.dims && init.dims.length ? init.dims.slice() : [values.length];
    const perChannel = dims.length >= 2;
    const quant = quantizeInt8(values, { shape: dims, axis: 0, signed, perChannel });
    const dataBytes = new Uint8Array(quant.data.buffer, quant.data.byteOffset, quant.data.byteLength);
    const scaleValues = Float32Array.from(quant.scales);
    const scaleBytes = new Uint8Array(scaleValues.buffer, scaleValues.byteOffset, scaleValues.byteLength);
    const zpValues = signed ? Int8Array.from(quant.zeroPoints) : Uint8Array.from(quant.zeroPoints);
    const zpBytes = new Uint8Array(zpValues.buffer, zpValues.byteOffset, zpValues.byteLength);
    const qName = init.name + '_q';
    const scaleName = init.name + '_scale';
    const zpName = init.name + '_zero_point';
    const scaleDims = perChannel ? [scaleValues.length] : [];
    kept.push({ name: qName, dims: dims.slice(), dtype: signed ? ONNX_DTYPE.int8 : ONNX_DTYPE.uint8, bytes: dataBytes, floatValues: null });
    kept.push({ name: scaleName, dims: scaleDims, dtype: ONNX_DTYPE.float32, bytes: scaleBytes, floatValues: null });
    kept.push({ name: zpName, dims: scaleDims, dtype: signed ? ONNX_DTYPE.int8 : ONNX_DTYPE.uint8, bytes: zpBytes, floatValues: null });
    const attrs = perChannel ? [attrInt('axis', 0)] : [];
    builder.nodes.unshift({
      opType: 'DequantizeLinear',
      inputs: [qName, scaleName, zpName],
      outputs: [init.name],
      attrs,
      name: init.name + '_dequant',
      domain: ''
    });
    dequantNodes.push(init.name + '_dequant');
    const originalBytes = values.length * 4;
    const quantizedBytes = dataBytes.length + scaleBytes.length + zpBytes.length;
    let sumSq = 0;
    let refSq = 0;
    for (let i = 0; i < values.length; i++) {
      const d = quant.restored ? quant.restored[i] - values[i] : 0;
      sumSq += d * d;
      refSq += values[i] * values[i];
    }
    const relativeL2 = refSq > 0 ? Math.sqrt(sumSq) / Math.sqrt(refSq) : 0;
    report.tensorsQuantized++;
    report.originalBytes += originalBytes;
    report.quantizedBytes += quantizedBytes;
    if (relativeL2 > report.worstRelativeL2) {
      report.worstRelativeL2 = relativeL2;
      report.worstTensor = init.name;
    }
    report.perTensor.push({ name: init.name, shape: dims, elements: values.length, perChannel, relativeL2, originalBytes, quantizedBytes });
  }
  builder.initializers = kept;
  report.compressionRatio = report.quantizedBytes ? report.originalBytes / report.quantizedBytes : 1;
  report.sizeSaved = report.originalBytes - report.quantizedBytes;
  report.dequantNodes = dequantNodes.length;
  if (report.tensorsQuantized) {
    notes.push('Integer export stores ' + report.tensorsQuantized + ' weight tensor' + (report.tensorsQuantized === 1 ? '' : 's') + ' as ' + precisionKey + ' with a DequantizeLinear per tensor (' + report.scheme + '). Tensors under ' + QDQ_MIN_ELEMENTS + ' elements stay float32 because their scales would cost more than they save. The graph still executes in float32, so activations are unchanged.');
  }
  return report;
}

function applyHead(b, value, head, notes) {
  switch (String(head || 'linear')) {
    case 'softmax':
    case 'softmax-seq':
      return { value: b.node('Softmax', [value], [b.tmp('head_softmax')], [attrInt('axis', -1)], 'head_softmax'), applied: 'softmax' };
    case 'sigmoid':
    case 'sigmoid-seq':
    case 'mask':
    case 'heatmap':
      return { value: b.node('Sigmoid', [value], [b.tmp('head_sigmoid')], [], 'head_sigmoid'), applied: 'sigmoid' };
    case 'embedding':
      notes.push('Embedding head exported as raw vectors; apply L2 normalisation in the consuming code if you need cosine distances.');
      return { value, applied: 'identity' };
    case 'det':
      notes.push('Detection head exported as raw grid outputs: channel 0 is objectness logits, channels 1-4 are box coordinates. Apply the sigmoid to channel 0 only.');
      return { value, applied: 'identity' };
    default:
      return { value, applied: 'linear' };
  }
}

export function exportOnnx(model, options) {
  const opts = options || {};
  const built = buildOnnxGraph(model, opts);
  const metadata = Object.assign({
    producer: PRODUCER,
    precision: built.precision,
    opset: String(built.opset),
    head: built.headKind,
    params: String(model.paramCount ? model.paramCount() : 0)
  }, opts.metadata || {});
  const bytes = built.builder.toBytes({
    graphName: opts.graphName || 'rootism_graph',
    docString: opts.docString || '',
    metadata,
    producerName: opts.producerName || PRODUCER
  });
  return {
    bytes,
    nodeCount: built.builder.nodes.length,
    initializerCount: built.builder.initializers.length,
    unsupported: built.unsupported,
    notes: built.notes,
    head: built.head,
    opset: built.builder.opset,
    inputName: built.inputName,
    outputName: built.outputName,
    inputShape: model.spec.input.shape.slice(),
    outputShape: built.outputShape,
    byteLength: bytes.length,
    precision: built.precision
  };
}
