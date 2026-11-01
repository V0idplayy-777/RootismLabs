import { PRESET_MODEL_LIST, buildPresetGraph } from '../js/presets/models.js';
import { Model, analyze, blankGraph } from '../js/core/model.js';
import { LAYERS, defaultConfig } from '../js/core/layers.js';
import { exportOnnx } from '../js/export/onnx.js';
import { parseMessage, first, asString, asInt64s, ONNX_FIELDS as F, ONNX_DTYPE } from '../js/core/protobuf.js';

const DTYPE_SIZE = {};
DTYPE_SIZE[ONNX_DTYPE.float32] = 4;
DTYPE_SIZE[ONNX_DTYPE.float16] = 2;
DTYPE_SIZE[ONNX_DTYPE.bfloat16] = 2;
DTYPE_SIZE[ONNX_DTYPE.int64] = 8;
DTYPE_SIZE[ONNX_DTYPE.int32] = 4;
DTYPE_SIZE[ONNX_DTYPE.int8] = 1;
DTYPE_SIZE[ONNX_DTYPE.uint8] = 1;

const problems = [];
let checked = 0;
let totalNodes = 0;
const opCounts = {};

function readValueInfo(bytes) {
  const vi = parseMessage(bytes);
  const name = asString(first(vi, F.valueInfo.name));
  const type = parseMessage(first(vi, F.valueInfo.type));
  const tt = parseMessage(first(type, F.typeProto.tensorType));
  const elemType = first(tt, F.tensorType.elemType);
  const shapeMsg = parseMessage(first(tt, F.tensorType.shape));
  const dims = (shapeMsg[F.shape.dim] || []).map((d) => {
    const dim = parseMessage(d.value);
    const param = first(dim, F.dim.dimParam);
    if (param) return asString(param);
    return first(dim, F.dim.dimValue);
  });
  return { name, elemType, dims };
}

function readNode(bytes) {
  const n = parseMessage(bytes);
  const inputs = (n[F.node.input] || []).map((e) => asString(e.value));
  const outputs = (n[F.node.output] || []).map((e) => asString(e.value));
  const attrs = {};
  for (const a of n[F.node.attribute] || []) {
    const at = parseMessage(a.value);
    const name = asString(first(at, F.attribute.name));
    const type = first(at, F.attribute.type);
    if (type === 2) attrs[name] = first(at, F.attribute.int);
    else if (type === 1) attrs[name] = first(at, F.attribute.float);
    else if (type === 3) attrs[name] = asString(first(at, F.attribute.string));
    else if (type === 7) attrs[name] = (at[F.attribute.ints] || []).map((e) => e.value);
    else if (type === 6) attrs[name] = (at[F.attribute.floats] || []).map((e) => e.value);
    else attrs[name] = 'type' + type;
  }
  return { op: asString(first(n, F.node.opType)), name: asString(first(n, F.node.name)), inputs, outputs, attrs };
}

function readInitializer(bytes) {
  const t = parseMessage(bytes);
  const name = asString(first(t, F.tensor.name));
  const dimsField = t[F.tensor.dims];
  const dims = dimsField ? dimsField.map((e) => e.value) : [];
  const dtype = first(t, F.tensor.dataType);
  const raw = first(t, F.tensor.rawData) || new Uint8Array(0);
  return { name, dims, dtype, raw };
}

function checkModel(key, bytes, expectedOut) {
  const model = parseMessage(bytes);
  if (!model[F.model.graph]) throw new Error('no graph');
  const irVersion = first(model, F.model.irVersion);
  if (!(irVersion > 0)) throw new Error('bad ir_version ' + irVersion);
  const opsets = (model[F.model.opsetImport] || []).map((e) => {
    const o = parseMessage(e.value);
    return { domain: asString(first(o, F.opset.domain) || ''), version: first(o, F.opset.version) };
  });
  if (!opsets.length || !(opsets[0].version > 0)) throw new Error('no opset import');

  const graph = parseMessage(first(model, F.model.graph));
  const nodes = (graph[F.graph.node] || []).map((e) => readNode(e.value));
  const inits = (graph[F.graph.initializer] || []).map((e) => readInitializer(e.value));
  const inputs = (graph[F.graph.input] || []).map((e) => readValueInfo(e.value));
  const outputs = (graph[F.graph.output] || []).map((e) => readValueInfo(e.value));

  if (!inputs.length) throw new Error('no graph inputs');
  if (!outputs.length) throw new Error('no graph outputs');
  if (inputs[0].dims[0] !== 'batch') throw new Error('input batch dim is not symbolic: ' + JSON.stringify(inputs[0].dims));
  if (outputs[0].dims[0] !== 'batch') throw new Error('output batch dim is not symbolic: ' + JSON.stringify(outputs[0].dims));

  const declaredOut = [null].concat(outputs[0].dims.slice(1));
  if (declaredOut.length !== expectedOut.length) throw new Error('declared output rank ' + declaredOut.length + ' vs analysed ' + expectedOut.length);
  for (let i = 1; i < expectedOut.length; i++) {
    if (declaredOut[i] !== expectedOut[i]) throw new Error('declared output dim ' + i + ' = ' + declaredOut[i] + ' vs analysed ' + expectedOut[i]);
  }

  const available = new Set(inputs.map((i) => i.name));
  for (const init of inits) {
    if (available.has(init.name)) throw new Error('duplicate tensor name ' + init.name);
    available.add(init.name);
    let elements = 1;
    for (const d of init.dims) elements *= d;
    const size = DTYPE_SIZE[init.dtype] || 0;
    if (!size) throw new Error('initializer ' + init.name + ' has unknown dtype ' + init.dtype);
    if (init.raw.length !== elements * size) {
      throw new Error('initializer ' + init.name + ' dims ' + JSON.stringify(init.dims) + ' imply ' + (elements * size) + ' bytes but raw_data has ' + init.raw.length);
    }
    if (init.dtype === ONNX_DTYPE.float32) {
      const view = new DataView(init.raw.buffer, init.raw.byteOffset, init.raw.length);
      for (let i = 0; i < init.raw.length / 4; i++) {
        const v = view.getFloat32(i * 4, true);
        if (!Number.isFinite(v)) throw new Error('initializer ' + init.name + ' contains a non-finite weight at index ' + i);
      }
    }
  }

  for (const n of nodes) {
    opCounts[n.op] = (opCounts[n.op] || 0) + 1;
    for (const inp of n.inputs) {
      if (!available.has(inp)) throw new Error('node ' + n.name + ' (' + n.op + ') reads undefined tensor ' + inp);
    }
    if (!n.outputs.length) throw new Error('node ' + n.name + ' has no outputs');
    for (const out of n.outputs) {
      if (available.has(out)) throw new Error('tensor ' + out + ' produced twice');
      available.add(out);
    }
    if (n.op === 'Transpose') {
      const perm = n.attrs.perm;
      if (!Array.isArray(perm) || !perm.length) throw new Error('Transpose ' + n.name + ' has no perm');
      const seen = new Set(perm);
      if (seen.size !== perm.length) throw new Error('Transpose ' + n.name + ' perm has repeated values: ' + JSON.stringify(perm));
      for (let i = 0; i < perm.length; i++) if (perm.indexOf(i) < 0) throw new Error('Transpose ' + n.name + ' perm is not a permutation of 0..' + (perm.length - 1));
    }
    if (n.op === 'Conv') {
      const ks = n.attrs.kernel_shape;
      const pads = n.attrs.pads;
      const strides = n.attrs.strides;
      if (!Array.isArray(ks) || !ks.length) throw new Error('Conv ' + n.name + ' has no kernel_shape');
      if (Array.isArray(pads) && pads.length !== ks.length * 2) throw new Error('Conv ' + n.name + ' pads length ' + pads.length + ' does not match kernel rank');
      if (Array.isArray(strides) && strides.length !== ks.length) throw new Error('Conv ' + n.name + ' strides length does not match kernel rank');
      if (n.attrs.group !== undefined && !(n.attrs.group >= 1)) throw new Error('Conv ' + n.name + ' has bad group');
    }
  }

  if (!available.has(outputs[0].name)) throw new Error('graph output ' + outputs[0].name + ' is never produced');
  totalNodes += nodes.length;
  return { nodes: nodes.length, inits: inits.length, opset: opsets[0].version };
}

for (const preset of PRESET_MODEL_LIST) {
  const spec = buildPresetGraph(preset);
  const model = new Model(spec);
  model.build(20260);
  const info = analyze(spec);
  let result;
  try {
    result = exportOnnx(model, { head: 'linear', docString: 'conformance ' + preset.key });
  } catch (e) {
    problems.push([preset.key, 'export threw: ' + e.message]);
    continue;
  }
  try {
    const summary = checkModel(preset.key, result.bytes, info.outputShape);
    checked++;
    const line = preset.key.padEnd(14) + ' nodes ' + String(summary.nodes).padStart(3) + '  initializers ' + String(summary.inits).padStart(2) + '  opset ' + summary.opset + '  bytes ' + String(result.bytes.length).padStart(7);
    console.log(line);
  } catch (e) {
    problems.push([preset.key, e.message]);
  }
}

const LAYER_CASES = [
  ['dense', [12], { units: 5, activation: 'gelu' }],
  ['dense', [12, 6], { units: 5, activation: 'elu' }],
  ['dense', [12, 6], { units: 5, activation: 'leaky_relu' }],
  ['dense', [12, 6], { units: 5, activation: 'softplus' }],
  ['activation', [12, 6], { kind: 'hard_sigmoid' }],
  ['activation', [12, 6], { kind: 'tanh' }],
  ['activation', [12, 6], { kind: 'sigmoid' }],
  ['softmax', [12, 6], { axis: -1 }],
  ['dropout', [12, 6], { rate: 0.2 }],
  ['gaussiannoise', [12, 6], { std: 0.05 }],
  ['flatten', [3, 4, 4], {}],
  ['reshape', [24], { shape: '4,6' }],
  ['concat', [12, 6], { axis: -1 }],
  ['add', [12, 6], {}],
  ['multiply', [12, 6], {}],
  ['layernorm', [12, 6], {}],
  ['batchnorm', [12, 6], {}],
  ['batchnorm', [3, 8, 8], {}],
  ['l2normalize', [12, 6], {}],
  ['conv2d', [3, 16, 16], { filters: 5, kernel: 3, padding: 'same', activation: 'relu' }],
  ['conv2d', [3, 16, 16], { filters: 5, kernel: 3, padding: 'valid', stride: 2, dilation: 2, activation: 'linear' }],
  ['conv2d', [6, 16, 16], { filters: 6, kernel: 3, padding: 'same', groups: 3, activation: 'linear' }],
  ['depthwiseconv2d', [8, 16, 16], { kernel: 3, multiplier: 2, activation: 'relu' }],
  ['separableconv2d', [3, 16, 16], { filters: 7, kernel: 3, activation: 'relu' }],
  ['conv1d', [30, 12], { filters: 7, kernel: 3, activation: 'relu' }],
  ['conv1d', [30, 12], { filters: 7, kernel: 5, stride: 2, padding: 'valid', dilation: 2, activation: 'linear' }],
  ['maxpool2d', [4, 16, 16], { kernel: 2, stride: 2, padding: 'same' }],
  ['avgpool2d', [4, 16, 16], { kernel: 2, stride: 2 }],
  ['globalavgpool2d', [4, 16, 16], {}],
  ['maxpool1d', [30, 12], { kernel: 2, stride: 2 }],
  ['avgpool1d', [30, 12], { kernel: 3, stride: 1 }],
  ['globalavgpool1d', [30, 12], {}],
  ['globalmaxpool1d', [30, 12], {}],
  ['temporalpool', [30, 12], { mode: 'mean' }],
  ['temporalpool', [30, 12], { mode: 'max' }],
  ['temporalpool', [30, 12], { mode: 'last' }],
  ['upsample2d', [4, 8, 8], { factor: 2, mode: 'nearest' }],
  ['upsample2d', [4, 8, 8], { factor: 3, mode: 'bilinear' }],
  ['subpixelupsample', [16, 8, 8], { factor: 2, filters: 4, kernel: 3, activation: 'linear' }],
  ['pad2d', [4, 8, 8], { amount: 2 }],
  ['embedding', [16], { vocab: 32, dim: 8 }],
  ['embedding', [4, 16], { vocab: 32, dim: 8 }],
  ['gru', [12, 6], { units: 5, returnSeq: true }],
  ['gru', [12, 6], { units: 5, returnSeq: false }],
  ['gru', [12, 6], { units: 5, returnSeq: true, bidirectional: true }],
  ['lstm', [12, 6], { units: 5, returnSeq: true }],
  ['lstm', [12, 6], { units: 5, returnSeq: false }],
  ['lstm', [12, 6], { units: 5, returnSeq: true, bidirectional: true }]
];

let layerChecked = 0;
const coveredTypes = new Set();
for (const [type, inShape, cfg] of LAYER_CASES) {
  const graph = blankGraph(inShape.slice(), 'x');
  graph.nodes.length = 1;
  graph.edges.length = 0;
  graph.nodes.push({ id: 't', type, cfg, pos: { x: 0, y: 0 } });
  graph.edges.push({ from: 'in', to: 't', toPort: 0 });
  if ((LAYERS[type] && LAYERS[type].ports) === 2) graph.edges.push({ from: 'in', to: 't', toPort: 1 });
  graph.output = 't';
  graph.head = 'linear';
  coveredTypes.add(type);
  const label = type + ' ' + JSON.stringify(inShape) + ' ' + JSON.stringify(cfg);
  const info = analyze(graph);
  if (!info.valid) {
    problems.push([label, 'reference graph invalid']);
    continue;
  }
  const model = new Model(graph);
  model.build(4242);
  if (type === 'batchnorm') model.syncStates();
  let result;
  try {
    result = exportOnnx(model, { head: 'linear' });
  } catch (e) {
    problems.push([label, 'export threw: ' + e.message]);
    continue;
  }
  try {
    checkModel(label, result.bytes, info.outputShape);
    layerChecked++;
  } catch (e) {
    problems.push([label, e.message]);
  }
}

const uncovered = Object.keys(LAYERS).filter((k) => k !== 'input' && !coveredTypes.has(k));

console.log('');
console.log('layer cases checked ' + layerChecked + ' / ' + LAYER_CASES.length + '   layer types with no case: ' + (uncovered.length ? uncovered.join(', ') : 'none'));
console.log('presets checked ' + checked + ' / ' + PRESET_MODEL_LIST.length + '   onnx nodes emitted ' + totalNodes);
console.log('op histogram ' + Object.keys(opCounts).sort().map((k) => k + ':' + opCounts[k]).join(' '));
if (problems.length) {
  console.log('');
  console.log('FAILURES ' + problems.length);
  for (const [k, msg] of problems) console.log('  ' + k + ': ' + msg);
  process.exit(1);
}
console.log('ONNX CONFORMANCE PASS');
