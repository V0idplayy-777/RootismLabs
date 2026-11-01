import { PRESET_MODEL_LIST, buildPresetGraph } from '../js/presets/models.js';
import { Model, analyze, blankGraph } from '../js/core/model.js';
import { LAYERS } from '../js/core/layers.js';
import { exportTfjs } from '../js/export/tfjs.js';

const DTYPE_BYTES = {
  float32: 4,
  float16: 2,
  int32: 4,
  int64: 8,
  uint8: 1,
  int8: 1,
  bool: 1
};

const problems = [];
let checked = 0;
let totalNodes = 0;
const opCounts = {};

function fail(label, msg) {
  problems.push([label, msg]);
}

function dimsOf(node) {
  const shape = node.attr && node.attr.shape && node.attr.shape.shape;
  if (!shape || !shape.dim) return null;
  return shape.dim.map((d) => parseInt(d.size, 10));
}

function tensorDims(node) {
  const t = node.attr && node.attr.value && node.attr.value.tensor;
  if (!t || !t.tensorShape || !t.tensorShape.dim) return [];
  return t.tensorShape.dim.map((d) => parseInt(d.size, 10));
}

function decodeBase64(value) {
  const table = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
  const clean = String(value).replace(/=+$/, '');
  const bytes = [];
  let buffer = 0;
  let bits = 0;
  for (const ch of clean) {
    const v = table.indexOf(ch);
    if (v < 0) continue;
    buffer = (buffer << 6) | v;
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      bytes.push((buffer >> bits) & 0xff);
    }
  }
  return String.fromCharCode.apply(null, bytes);
}

function attrStr(node, key) {
  const a = node.attr && node.attr[key];
  if (!a) return undefined;
  if (a.s !== undefined) return decodeBase64(a.s);
  if (a.type !== undefined) return a.type;
  return undefined;
}

function attrInts(node, key) {
  const a = node.attr && node.attr[key];
  if (!a) return undefined;
  if (a.list && a.list.i) return a.list.i.map((v) => parseInt(v, 10));
  if (a.i !== undefined) return parseInt(a.i, 10);
  return undefined;
}

function readConstBytes(weightData, weights, name) {
  let offset = 0;
  for (const w of weights) {
    const count = w.shape.reduce((a, b) => a * b, 1);
    const size = count * (DTYPE_BYTES[w.dtype] || 4);
    if (w.name === name) return weightData.subarray(offset, offset + size);
    offset += size;
  }
  return null;
}

function checkGraph(label, modelJson, weightData, inputShape, outputShape) {
  if (modelJson.format !== 'graph-model') throw new Error('format is ' + modelJson.format);
  const topology = modelJson.modelTopology;
  if (!topology || !Array.isArray(topology.node) || !topology.node.length) throw new Error('modelTopology.node missing');
  if (!topology.versions || topology.versions.producer === undefined) throw new Error('versions.producer missing');
  const manifest = modelJson.weightsManifest;
  if (!Array.isArray(manifest) || manifest.length !== 1) throw new Error('weightsManifest must have exactly one group');
  if (!Array.isArray(manifest[0].paths) || manifest[0].paths.length !== 1) throw new Error('weightsManifest paths must have one entry');
  const weights = manifest[0].weights;

  const seenNames = new Set();
  for (const w of weights) {
    if (seenNames.has(w.name)) throw new Error('duplicate weight ' + w.name);
    seenNames.add(w.name);
    if (!Array.isArray(w.shape)) throw new Error('weight ' + w.name + ' has no shape');
    if (!DTYPE_BYTES[w.dtype]) throw new Error('weight ' + w.name + ' has unknown dtype ' + w.dtype);
  }
  let accounted = 0;
  for (const w of weights) accounted += w.shape.reduce((a, b) => a * b, 1) * DTYPE_BYTES[w.dtype];
  if (accounted !== weightData.length) throw new Error('weight bytes ' + accounted + ' != payload ' + weightData.length);

  const nodes = topology.node;
  const byName = new Map();
  for (const n of nodes) {
    if (byName.has(n.name)) throw new Error('duplicate node name ' + n.name);
    byName.set(n.name, n);
    if (!n.op) throw new Error('node ' + n.name + ' has no op');
    opCounts[n.op] = (opCounts[n.op] || 0) + 1;
  }

  const emitted = new Set();
  let placeholders = 0;
  let consts = 0;
  for (let i = 0; i < nodes.length; i++) {
    const n = nodes[i];
    if (n.op === 'Placeholder') {
      placeholders++;
      const dims = dimsOf(n);
      if (!dims || dims.length !== inputShape.length + 1) throw new Error('placeholder ' + n.name + ' rank ' + JSON.stringify(dims));
      if (dims[0] !== -1) throw new Error('placeholder ' + n.name + ' must have a dynamic batch dim, got ' + JSON.stringify(dims));
      for (let d = 0; d < inputShape.length; d++) {
        if (dims[d + 1] !== inputShape[d]) throw new Error('placeholder ' + n.name + ' dim ' + d + ' is ' + dims[d + 1] + ' expected ' + inputShape[d]);
      }
    }
    if (n.op === 'Const') {
      consts++;
      const w = seenNames.has(n.name) ? weights.find((x) => x.name === n.name) : null;
      if (!w) throw new Error('Const ' + n.name + ' has no weight manifest entry');
      const td = tensorDims(n);
      if (td.length !== w.shape.length || td.some((v, k) => v !== w.shape[k])) {
        throw new Error('Const ' + n.name + ' tensor shape ' + JSON.stringify(td) + ' != manifest ' + JSON.stringify(w.shape));
      }
      const want = n.attr.value.tensor.dtype === 'DT_HALF' ? 'float16' : n.attr.value.tensor.dtype === 'DT_INT32' ? 'int32' : 'float32';
      if (want !== w.dtype) throw new Error('Const ' + n.name + ' dtype ' + w.dtype + ' != tensor ' + want);
    }
    for (const raw of n.input || []) {
      const name = raw.startsWith('^') ? raw.slice(1) : raw;
      const outIndex = name.indexOf(':');
      const base = outIndex >= 0 ? name.slice(0, outIndex) : name;
      if (!byName.has(base)) throw new Error('node ' + n.name + ' references unknown input ' + base);
      if (!emitted.has(base)) throw new Error('node ' + n.name + ' uses ' + base + ' before it is defined');
    }
    if (n.op === 'Pad') {
      const ins = n.input || [];
      if (ins.length < 2) throw new Error('Pad ' + n.name + ' needs paddings input');
      const padConst = byName.get(ins[1]);
      const pd = padConst ? weights.find((w) => w.name === ins[1]) : null;
      if (!pd) throw new Error('Pad ' + n.name + ' paddings input ' + ins[1] + ' is not a constant');
      if (pd.shape.length !== 2 || pd.shape[1] !== 2) throw new Error('Pad ' + n.name + ' paddings shape ' + JSON.stringify(pd.shape));
      if (ins.length > 2) {
        const cv = weights.find((w) => w.name === ins[2]);
        if (cv && cv.dtype === 'float32') {
          const bytes = readConstBytes(weightData, weights, ins[2]);
          const view = new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
          if (view.length !== 1 || view[0] !== 0) {
            throw new Error('Pad ' + n.name + ' uses a non-zero constant ' + (view.length ? view[0] : 'empty') + '; the tfjs pad kernel drops tensor constants');
          }
        }
      }
    }
    if (n.op === 'Conv2D' || n.op === 'DepthwiseConv2dNative') {
      const pad = attrStr(n, 'padding');
      if (pad !== 'VALID' && pad !== 'SAME') throw new Error(n.op + ' ' + n.name + ' padding ' + pad);
      const strides = attrInts(n, 'strides');
      if (!Array.isArray(strides) || strides.length !== 4) throw new Error(n.op + ' ' + n.name + ' strides ' + JSON.stringify(strides));
      if (attrStr(n, 'data_format') !== 'NHWC') throw new Error(n.op + ' ' + n.name + ' data_format must be NHWC');
      const filterName = (n.input || [])[1];
      const fw = weights.find((w) => w.name === filterName);
      if (!fw) throw new Error(n.op + ' ' + n.name + ' filter ' + filterName + ' is not a constant');
      if (fw.shape.length !== 4) throw new Error(n.op + ' ' + n.name + ' filter rank ' + fw.shape.length);
      if (n.op === 'DepthwiseConv2dNative' && fw.shape[0] !== fw.shape[1]) {
        throw new Error('depthwise filter must be square in space, got ' + JSON.stringify(fw.shape));
      }
      if (pad === 'SAME') throw new Error(n.op + ' ' + n.name + ' must not rely on TensorFlow SAME padding; it is not symmetric for stride > 1');
    }
    if (n.op === 'MaxPool' || n.op === 'AvgPool') {
      const ksize = attrInts(n, 'ksize');
      if (!Array.isArray(ksize) || ksize.length !== 4) throw new Error(n.op + ' ' + n.name + ' ksize ' + JSON.stringify(ksize));
      if (attrStr(n, 'padding') !== 'VALID') throw new Error(n.op + ' ' + n.name + ' must use VALID with explicit padding');
      if (attrStr(n, 'data_format') !== 'NHWC') throw new Error(n.op + ' ' + n.name + ' data_format must be NHWC');
      if (n.op === 'MaxPool') {
        const producer = byName.get((n.input || [])[0]);
        if (producer && producer.op === 'Sub') {
          const padSource = byName.get((producer.input || [])[0]);
          if (!padSource || padSource.op !== 'Pad') throw new Error('MaxPool ' + n.name + ' mask arithmetic is malformed');
        }
      }
    }
    if (n.op === 'MatMul') {
      if (attrStr(n, 'T') !== 'DT_FLOAT' && attrStr(n, 'T') !== 'DT_HALF') throw new Error('MatMul ' + n.name + ' dtype ' + attrStr(n, 'T'));
    }
    if (n.op === 'GatherV2') {
      if (attrInts(n, 'batch_dims') !== 0) throw new Error('GatherV2 ' + n.name + ' batch_dims must be 0');
    }
    emitted.add(n.name);
  }

  if (placeholders !== 1) throw new Error('expected exactly one placeholder, found ' + placeholders);
  void consts;

  const signature = modelJson.signature;
  if (!signature || !signature.inputs || !signature.outputs) throw new Error('signature missing');
  const sigInputs = Object.keys(signature.inputs);
  const sigOutputs = Object.keys(signature.outputs);
  if (sigInputs.length !== 1) throw new Error('signature must declare one input, got ' + sigInputs.length);
  if (sigOutputs.length !== 1) throw new Error('signature must declare one output, got ' + sigOutputs.length);
  const inSigShape = signature.inputs[sigInputs[0]].tensorShape;
  const inDims = inSigShape && inSigShape.dim ? inSigShape.dim.map((d) => parseInt(d.size, 10)) : null;
  if (!inDims || inDims[0] !== -1) throw new Error('signature input must declare a dynamic batch dim');
  for (let i = 0; i < inputShape.length; i++) {
    if (inDims[i + 1] !== inputShape[i]) throw new Error('signature input dim ' + i + ' is ' + inDims[i + 1] + ' expected ' + inputShape[i]);
  }
  const inTensor = signature.inputs[sigInputs[0]].name;
  const outTensor = signature.outputs[sigOutputs[0]].name;
  if (!byName.has(inTensor.replace(':0', ''))) throw new Error('signature input ' + inTensor + ' is not a node');
  const outNode = byName.get(outTensor.replace(':0', ''));
  if (!outNode) throw new Error('signature output ' + outTensor + ' is not a node');
  if (outNode.op !== 'Identity') throw new Error('signature output node must be Identity, got ' + outNode.op);

  const used = new Set();
  const stack = [outNode.name];
  while (stack.length) {
    const name = stack.pop();
    if (used.has(name)) continue;
    used.add(name);
    const node = byName.get(name);
    for (const raw of node.input || []) stack.push(raw.replace('^', '').split(':')[0]);
  }
  if (!used.has(inTensor.replace(':0', ''))) throw new Error('output does not depend on the model input');
  const orphan = nodes.filter((n) => !used.has(n.name));
  if (orphan.length) throw new Error(orphan.length + ' unreachable nodes, e.g. ' + orphan[0].name);

  if (outputShape) {
    const declared = signature.outputs[sigOutputs[0]].tensorShape;
    const flat = declared && declared.dim ? declared.dim.map((d) => parseInt(d.size, 10)) : null;
    if (!flat) throw new Error('signature output has no shape');
    if (flat.length !== outputShape.length) throw new Error('output rank ' + flat.length + ' expected ' + outputShape.length);
    for (let i = 1; i < outputShape.length; i++) {
      if (flat[i] !== outputShape[i]) throw new Error('output dim ' + i + ' is ' + flat[i] + ' expected ' + outputShape[i]);
    }
    if (flat[0] !== -1 && flat[0] !== outputShape[0]) throw new Error('output batch dim ' + flat[0]);
  }

  totalNodes += nodes.length;
  return { nodes: nodes.length, weights: weights.length, bytes: weightData.length };
}

function exportAndCheck(label, model, inputShape, outputShape, options) {
  let result;
  try {
    result = exportTfjs(model, options || {});
  } catch (e) {
    fail(label, 'export threw: ' + e.message);
    return null;
  }
  try {
    const summary = checkGraph(label, result.modelJson, result.weightData, inputShape, outputShape);
    checked++;
    return { summary, result };
  } catch (e) {
    fail(label, e.message);
    return null;
  }
}

console.log('tensorflow.js graph-model conformance');
console.log('');

for (const preset of PRESET_MODEL_LIST) {
  const spec = buildPresetGraph(preset);
  const info = analyze(spec);
  const model = new Model(spec);
  model.build(20260);
  model.syncStates();
  const out = exportAndCheck(preset.key, model, spec.input.shape, info.outputShape, { head: spec.head });
  if (!out) continue;
  const hasRnn = spec.nodes.some((n) => n.type === 'gru' || n.type === 'lstm');
  if (hasRnn && out.result.fixedSequenceLength !== spec.input.shape[0]) {
    fail(preset.key, 'fixedSequenceLength should be ' + spec.input.shape[0] + ', got ' + out.result.fixedSequenceLength);
  }
  if (!hasRnn && out.result.fixedSequenceLength !== null) {
    fail(preset.key, 'fixedSequenceLength should be null for a non-recurrent model');
  }
  if (out.result.unsupported.length) fail(preset.key, 'unsupported layer types: ' + out.result.unsupported.join(', '));
  const line = preset.key.padEnd(14) + ' nodes ' + String(out.summary.nodes).padStart(5) + '  consts ' + String(out.summary.weights).padStart(4) + '  weights ' + String(out.result.weightData.length).padStart(7) + '  json ' + String(JSON.stringify(out.result.modelJson).length).padStart(7);
  console.log(line);
}

const LAYER_CASES = [
  ['dense', [12], { units: 5, activation: 'gelu' }],
  ['dense', [12, 6], { units: 5, activation: 'elu' }],
  ['dense', [12, 6], { units: 5, activation: 'leaky_relu' }],
  ['dense', [12, 6], { units: 5, activation: 'softplus' }],
  ['dense', [12, 6], { units: 5, activation: 'hard_sigmoid' }],
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
  ['conv2d', [3, 15, 15], { filters: 5, kernel: 3, padding: 'same', stride: 2, activation: 'linear' }],
  ['conv2d', [6, 16, 16], { filters: 6, kernel: 3, padding: 'same', groups: 3, activation: 'linear' }],
  ['depthwiseconv2d', [8, 16, 16], { kernel: 3, multiplier: 2, activation: 'relu' }],
  ['separableconv2d', [3, 16, 16], { filters: 7, kernel: 3, activation: 'relu' }],
  ['conv1d', [30, 12], { filters: 7, kernel: 3, activation: 'relu' }],
  ['conv1d', [30, 12], { filters: 7, kernel: 5, stride: 2, padding: 'same', activation: 'linear' }],
  ['maxpool2d', [4, 16, 16], { kernel: 2, stride: 2, padding: 'same' }],
  ['maxpool2d', [4, 16, 16], { kernel: 3, stride: 1, padding: 'same' }],
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
  ['gru', [12, 6], { units: 5, returnSeq: false, bidirectional: true }],
  ['lstm', [12, 6], { units: 5, returnSeq: true }],
  ['lstm', [12, 6], { units: 5, returnSeq: false }],
  ['lstm', [12, 6], { units: 5, returnSeq: true, bidirectional: true }],
  ['lstm', [12, 6], { units: 5, returnSeq: false, bidirectional: true }]
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
    fail(label, 'reference graph invalid: ' + info.errors.join('; '));
    continue;
  }
  const model = new Model(graph);
  model.build(4242);
  model.syncStates();
  const out = exportAndCheck(label, model, graph.input.shape, info.outputShape, { head: 'linear' });
  if (out) layerChecked++;
}

const uncovered = Object.keys(LAYERS).filter((k) => k !== 'input' && !coveredTypes.has(k));

const fp16Spec = buildPresetGraph(PRESET_MODEL_LIST[0]);
const fp16Model = new Model(fp16Spec);
fp16Model.build(99);
fp16Model.syncStates();
let fp16;
try {
  fp16 = exportTfjs(fp16Model, { precision: 'fp16', head: fp16Spec.head });
} catch (e) {
  fail('fp16', 'export threw: ' + e.message);
}
if (fp16) {
  const halfWeights = fp16.modelJson.weightsManifest[0].weights.filter((w) => w.dtype === 'float16');
  if (!halfWeights.length) fail('fp16', 'no float16 weights in the manifest');
  const intWeights = fp16.modelJson.weightsManifest[0].weights.filter((w) => w.dtype === 'int32');
  let bytes = 0;
  for (const w of fp16.modelJson.weightsManifest[0].weights) bytes += w.shape.reduce((a, b) => a * b, 1) * DTYPE_BYTES[w.dtype];
  if (bytes !== fp16.weightData.length) fail('fp16', 'weight bytes ' + bytes + ' != payload ' + fp16.weightData.length);
  const baseline = exportTfjs(fp16Model, { head: fp16Spec.head });
  const ratio = baseline.weightData.length / fp16.weightData.length;
  if (ratio < 1.7 || ratio > 2.1) fail('fp16', 'compression ratio ' + ratio.toFixed(3) + ' is not close to 2');
  try {
    checkGraph('fp16', fp16.modelJson, fp16.weightData, fp16Spec.input.shape, analyze(fp16Spec).outputShape);
    checked++;
  } catch (e) {
    fail('fp16', e.message);
  }
  console.log('');
  console.log('fp16 export    float16 tensors ' + halfWeights.length + '  int32 tensors ' + intWeights.length + '  ratio ' + ratio.toFixed(3) + '  bytes ' + fp16.weightData.length);
}

for (const bad of ['int8', 'uint8']) {
  let threw = false;
  try {
    exportTfjs(fp16Model, { precision: bad });
  } catch (e) {
    threw = true;
  }
  if (!threw) fail(bad, 'integer precision must be rejected by the tfjs exporter');
}

console.log('');
console.log('layer cases checked ' + layerChecked + ' / ' + LAYER_CASES.length + '   layer types with no case: ' + (uncovered.length ? uncovered.join(', ') : 'none'));
console.log('graphs checked ' + checked + '   tfjs nodes emitted ' + totalNodes);
console.log('op histogram ' + Object.keys(opCounts).sort().map((k) => k + ':' + opCounts[k]).join(' '));
if (problems.length) {
  console.log('');
  console.log('FAILURES ' + problems.length);
  for (const [k, msg] of problems) console.log('  ' + k + ': ' + msg);
  process.exit(1);
}
console.log('TENSORFLOW.JS CONFORMANCE PASS');
