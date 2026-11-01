import { analyze } from '../js/core/model.js';
import { Trainer, normalizeConfig } from '../js/core/train.js';
import { taskFor } from '../js/core/tasks.js';
import * as G from '../js/data/generators.js';

let counter = 0;
const nid = (p) => p + (counter++);

function build(inputShape, layers, head) {
  const nodes = [];
  const edges = [];
  const inId = nid('in');
  nodes.push({ id: inId, type: 'input', cfg: {}, pos: { x: 0, y: 0 } });
  let prev = inId;
  layers.forEach((spec, i) => {
    const id = nid('n');
    nodes.push({ id, type: spec.type, cfg: spec.cfg || {}, pos: { x: 180 * (i + 1), y: 0 } });
    edges.push({ from: prev, to: id });
    prev = id;
  });
  return {
    id: nid('graph'),
    name: 'smoke',
    version: 1,
    input: { name: 'x', shape: inputShape, dtype: 'float32' },
    nodes,
    edges,
    output: prev,
    head: head || 'linear',
    meta: {}
  };
}

const dense = (units, activation) => ({ type: 'dense', cfg: { units, activation: activation || 'relu', useBias: true } });
const conv1d = (filters, cfg) => Object.assign({ type: 'conv1d', cfg: Object.assign({ filters, kernel: 3, activation: 'relu' }, cfg || {}) });
const conv2d = (filters, cfg) => Object.assign({ type: 'conv2d', cfg: Object.assign({ filters, kernel: 3, activation: 'relu' }, cfg || {}) });
const head = (units, activation) => {
  const out = [dense(units, 'linear')];
  if (activation) out.push({ type: 'activation', cfg: { kind: activation } });
  return out;
};
const softmaxHead = (units) => [dense(units, 'linear'), { type: 'softmax', cfg: { axis: -1 } }];

const CASES = [
  {
    label: 'multiclass/mlp', task: 'multiclass', data: () => G.genLangId({ size: 'small' }), input: [640], head: 'softmax',
    layers: [dense(96), { type: 'dropout', cfg: { rate: 0.2 } }, dense(64)].concat(softmaxHead(12))
  },
  {
    label: 'multiclass/lstm', task: 'multiclass', data: () => G.genImuActivity({ size: 'small' }), input: [64, 6], head: 'softmax',
    layers: [conv1d(12, { kernel: 5, stride: 2 }), { type: 'lstm', cfg: { units: 20, returnSeq: false } }].concat(softmaxHead(8))
  },
  {
    label: 'multiclass/conv2d', task: 'multiclass', data: () => G.genNsfwStandIn({ size: 'small' }), input: [3, 32, 32], head: 'softmax',
    layers: [conv2d(8, { stride: 2 }), { type: 'depthwiseconv2d', cfg: { kernel: 3, activation: 'relu' } }, { type: 'globalavgpool2d', cfg: {} }].concat(softmaxHead(3))
  },
  {
    label: 'binary/kws', task: 'binary', data: () => G.genWakeWord({ size: 'small' }), input: [27, 32], head: 'sigmoid',
    layers: [conv1d(16), { type: 'gru', cfg: { units: 24, returnSeq: false } }].concat(head(1, 'sigmoid'))
  },
  {
    label: 'binary/bot-mlp', task: 'binary', data: () => G.genBotDetection({ size: 'small' }), input: [96], head: 'sigmoid',
    layers: [dense(48), { type: 'batchnorm', cfg: {} }, dense(24)].concat(head(1, 'sigmoid'))
  },
  {
    label: 'frame_binary/vad', task: 'frame_binary', data: () => G.genVad({ size: 'small' }), input: [31, 32], head: 'sigmoid-seq',
    layers: [conv1d(16), { type: 'gru', cfg: { units: 24, returnSeq: true } }].concat(head(1, 'sigmoid'))
  },
  {
    label: 'onset/beats', task: 'onset', data: () => G.genBeatOnset({ size: 'small' }), input: [98, 32], head: 'sigmoid-seq',
    layers: [conv1d(16, { kernel: 5 }), conv1d(16)].concat(head(1, 'sigmoid'))
  },
  {
    label: 'seq_tag/pii-bio', task: 'seq_tag', data: () => G.genPii({ size: 'small' }), input: [96], head: 'softmax-seq',
    layers: [
      { type: 'reshape', cfg: { shape: '96,1' } },
      { type: 'embedding', cfg: { vocab: 128, dim: 12 } },
      conv1d(16)
    ].concat(head(15))
  },
  {
    label: 'lm/char-gru', task: 'lm', data: () => G.genAutocomplete({ size: 'small' }), input: [64], head: 'softmax-seq',
    layers: [{ type: 'embedding', cfg: { vocab: 128, dim: 16 } }, { type: 'gru', cfg: { units: 32, returnSeq: true } }].concat(head(128))
  },
  {
    label: 'lm/byte-compress', task: 'lm', data: () => G.genByteCompression({ size: 'small' }), input: [96], head: 'softmax-seq',
    layers: [{ type: 'embedding', cfg: { vocab: 259, dim: 12 } }, { type: 'gru', cfg: { units: 24, returnSeq: true } }].concat(head(259))
  },
  {
    label: 'embedding/infonce', task: 'embedding', data: () => G.genSentenceEmbeddings({ size: 'small' }), input: [512], head: 'embedding',
    layers: [dense(96), dense(32, 'linear'), { type: 'l2normalize', cfg: {} }]
  },
  {
    label: 'regression/password', task: 'regression', data: () => G.genPasswordStrength({ size: 'small' }), input: [320], head: 'linear',
    layers: [dense(64)].concat(head(1))
  },
  {
    label: 'forecast/gru', task: 'forecast', data: () => G.genForecasting({ size: 'small' }), input: [48, 1], head: 'linear',
    layers: [{ type: 'gru', cfg: { units: 16, returnSeq: false } }].concat(head(8))
  },
  {
    label: 'anomaly/autoencoder', task: 'anomaly', data: () => G.genAnomalyTelemetry({ size: 'small' }), input: [32, 6], head: 'linear',
    layers: [
      { type: 'flatten', cfg: {} }, dense(24), dense(12), dense(24), dense(192, 'linear'),
      { type: 'reshape', cfg: { shape: '32,6' } }
    ]
  },
  {
    label: 'image_map/superres', task: 'image_map', data: () => G.genSuperResolution({ size: 'small' }), input: [3, 32, 32], head: 'mask',
    layers: [conv2d(16), conv2d(16), conv2d(3, { activation: 'sigmoid' })]
  },
  {
    label: 'image_map/style', task: 'image_map', data: () => G.genStyleTransfer({ size: 'small' }), input: [3, 32, 32], head: 'mask',
    layers: [{ type: 'separableconv2d', cfg: { filters: 12, kernel: 3, activation: 'relu' } }, conv2d(3, { activation: 'sigmoid' })]
  },
  {
    label: 'mask/segmentation', task: 'mask', data: () => G.genSegmentation({ size: 'small' }), input: [3, 48, 48], head: 'heatmap',
    layers: [conv2d(8, { stride: 2 }), { type: 'upsample2d', cfg: { factor: 2, mode: 'nearest' } }, conv2d(1, { activation: 'sigmoid' })]
  },
  {
    label: 'mask/saliency', task: 'mask', data: () => G.genSaliency({ size: 'small' }), input: [3, 48, 48], head: 'heatmap',
    layers: [{ type: 'depthwiseconv2d', cfg: { kernel: 3, activation: 'relu' } }, conv2d(1, { kernel: 1, activation: 'sigmoid' })]
  },
  {
    label: 'spectral_mask/denoise', task: 'spectral_mask', data: () => G.genNoiseSuppression({ size: 'small' }), input: [30, 129], head: 'mask',
    layers: [conv1d(24), { type: 'gru', cfg: { units: 24, returnSeq: true } }].concat(head(129, 'sigmoid'))
  },
  {
    label: 'pitch_class/mel', task: 'pitch_class', data: () => G.genPitch({ size: 'small' }), input: [17, 48], head: 'softmax',
    layers: [conv1d(16), { type: 'temporalpool', cfg: { mode: 'max' } }].concat(softmaxHead(36))
  },
  {
    label: 'detection/face', task: 'detection', data: () => G.genDetection({ size: 'small' }), input: [3, 48, 48], head: 'det',
    layers: [conv2d(8, { stride: 2 }), conv2d(8, { stride: 2 }), conv2d(5, { activation: 'linear' })]
  },
  {
    label: 'coord_field/texture', task: 'coordinate_field', data: () => G.genNeuralTexture({ size: 'small' }), input: [3], head: 'linear',
    layers: [dense(64), dense(64)].concat(head(3, 'sigmoid'))
  }
];

const results = [];
for (const c of CASES) {
  const graph = build(c.input, c.layers, c.head);
  const cfg = normalizeConfig({
    epochs: 1,
    batchSize: 16,
    optimizer: { kind: 'adam', lr: 0.006 },
    seed: 7,
    evalBatchSize: 32,
    earlyStopping: { enabled: false },
    checkpointBest: false
  }, taskFor(c.task));
  const info = analyze(graph);
  if (!info.valid) {
    console.log('FAIL ' + c.label.padEnd(22) + ' invalid graph ' + JSON.stringify(info.errors));
    results.push(false);
    continue;
  }
  const started = Date.now();
  try {
    const trainer = new Trainer({ spec: graph, dataset: c.data(), taskKey: c.task, config: cfg });
    const res = await trainer.run();
    const row = res.history[res.history.length - 1];
    const vm = res.metrics.val || res.metrics.test || res.metrics.train;
    const scalars = (vm && vm.scalars) || {};
    const shown = Object.entries(scalars).slice(0, 5).map((e) => e[0] + '=' + (typeof e[1] === 'number' ? e[1].toFixed(3) : e[1])).join(' ');
    const finite = Number.isFinite(row.trainLoss) && Number.isFinite(row.valLoss);
    console.log((finite ? 'ok   ' : 'FAIL ') + c.label.padEnd(22) +
      ' p=' + String(info.totalParams).padStart(7) +
      ' macs=' + String(Math.round(info.macs)).padStart(9) +
      ' out=' + JSON.stringify(info.outputShape).padEnd(15) +
      ' train=' + row.trainLoss.toFixed(4) + ' val=' + row.valLoss.toFixed(4) +
      ' ' + (Date.now() - started) + 'ms\n     ' + shown);
    results.push(finite);
  } catch (e) {
    console.log('FAIL ' + c.label.padEnd(22) + ' ' + e.message);
    console.log('     ' + e.stack.split('\n').slice(1, 4).join('\n     '));
    results.push(false);
  }
}

const passed = results.filter(Boolean).length;
console.log('\n' + passed + '/' + results.length + ' task+graph combinations trained without error');
if (passed !== results.length) process.exitCode = 1;
