import { analyze, specToText } from '../core/model.js';
import { LAYERS } from '../core/layers.js';
import { taskFor } from '../core/tasks.js';
import { sha256Text, shortHash } from '../core/hash.js';
import { metricMeta, formatMetric, formatBytes, formatDuration, headlineFor, orderScalars } from '../presets/metrics.js';

export const CARD_VERSION = '1.0';
export const TOOL_NAME = 'Rootism Labs';
export const TOOL_VERSION = '1.0.0';

function stableStringify(value) {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return '[' + value.map(stableStringify).join(',') + ']';
  const keys = Object.keys(value).sort();
  return '{' + keys.map((k) => JSON.stringify(k) + ':' + stableStringify(value[k])).join(',') + '}';
}

function configHash(value) {
  return shortHash(sha256Text(stableStringify(value === undefined ? null : value)), 16);
}

function round(value, digits) {
  if (value === null || value === undefined || !Number.isFinite(value)) return null;
  const f = Math.pow(10, digits === undefined ? 6 : digits);
  return Math.round(value * f) / f;
}

function scalarBlock(scalars, keys) {
  const out = {};
  for (const k of keys || Object.keys(scalars || {})) {
    if (scalars && scalars[k] !== undefined) out[k] = round(scalars[k], 6);
  }
  return out;
}

export function architectureRows(spec, info) {
  const analysis = info || analyze(spec);
  const rows = [];
  const paramTotals = {};
  for (const p of analysis.paramList) {
    const nodeId = p.name.split('.')[0];
    paramTotals[nodeId] = (paramTotals[nodeId] || 0) + p.shape.reduce((a, b) => a * b, 1);
  }
  const statById = {};
  for (const st of analysis.nodeStats || []) statById[st.id] = st;
  const byId = {};
  for (const n of spec.nodes) byId[n.id] = n;
  for (const id of analysis.order) {
    const node = byId[id];
    if (!node || node.type === 'input') continue;
    const def = LAYERS[node.type] || {};
    const st = statById[id] || {};
    rows.push({
      id,
      type: node.type,
      label: def.label || node.type,
      group: def.group || 'other',
      config: JSON.parse(JSON.stringify(node.cfg || {})),
      inputs: (spec.edges || []).filter((e) => e.to === id).map((e) => e.from),
      outputShape: (analysis.shapes[id] || []).slice(),
      params: paramTotals[id] || st.params || 0,
      macs: st.macs || 0,
      text: st.text || ''
    });
  }
  return rows;
}

function splitCounts(dataset) {
  if (!dataset || !dataset.splits) return null;
  return {
    train: dataset.splits.train ? dataset.splits.train.length : 0,
    val: dataset.splits.val ? dataset.splits.val.length : 0,
    test: dataset.splits.test ? dataset.splits.test.length : 0
  };
}

function datasetBlock(dataset) {
  if (!dataset) return null;
  const block = {
    id: dataset.id,
    name: dataset.name,
    modality: dataset.modality,
    task: dataset.task,
    source: dataset.source,
    presetKey: dataset.presetKey || null,
    version: dataset.version,
    samples: dataset.count,
    xShape: (dataset.xShape || []).slice(),
    yShape: (dataset.yShape || []).slice(),
    labels: (dataset.labels || []).slice(),
    classes: (dataset.labels || []).length || null,
    splits: splitCounts(dataset),
    splitMethod: dataset.splitMethod,
    splitRatios: (dataset.splitRatios || []).slice(),
    splitSeed: dataset.splitSeed,
    license: dataset.license ? JSON.parse(JSON.stringify(dataset.license)) : null,
    provenance: dataset.provenance ? JSON.parse(JSON.stringify(dataset.provenance)) : null,
    featureSpec: dataset.featureSpec ? JSON.parse(JSON.stringify(dataset.featureSpec)) : null,
    sampleRate: dataset.sampleRate || null,
    units: dataset.units || null,
    columns: dataset.columns ? dataset.columns.slice() : null,
    digest: dataset.digest ? JSON.parse(JSON.stringify(dataset.digest)) : null,
    notes: dataset.notes || ''
  };
  if (dataset.stats) {
    block.stats = {
      xMin: round(dataset.stats.xMin, 4),
      xMax: round(dataset.stats.xMax, 4),
      xMean: round(dataset.stats.xMean, 6),
      xStd: round(dataset.stats.xStd, 6),
      yMin: round(dataset.stats.yMin, 4),
      yMax: round(dataset.stats.yMax, 4),
      labelCounts: dataset.stats.labelCounts ? dataset.stats.labelCounts.slice() : null,
      imbalanceRatio: round(dataset.stats.imbalanceRatio, 3),
      missing: dataset.stats.missing || 0,
      duplicatesRemoved: dataset.stats.duplicatesRemoved || dataset.removedDuplicates || 0
    };
  }
  if (dataset.quality) block.quality = JSON.parse(JSON.stringify(dataset.quality));
  return block;
}

function metricsBlock(splitMetrics, taskKey) {
  if (!splitMetrics) return null;
  const out = {};
  for (const split of Object.keys(splitMetrics)) {
    const m = splitMetrics[split];
    if (!m) continue;
    const entry = { scalars: scalarBlock(m.scalars || m) };
    if (m.perClass) {
      entry.perClass = m.perClass.map((c) => ({
        label: c.label,
        index: c.index,
        support: c.support,
        precision: round(c.precision, 4),
        recall: round(c.recall, 4),
        f1: round(c.f1, 4)
      }));
    }
    if (m.confusion) entry.confusion = { labels: m.confusion.labels.slice(), matrix: m.confusion.matrix.map((r) => r.slice()) };
    if (m.calibration) {
      entry.calibration = {
        ece: round(m.calibration.ece, 4),
        maxError: round(m.calibration.maxError, 4),
        brier: round(m.calibration.brier, 4),
        bins: (m.calibration.bins || []).map((b) => ({
          confidence: round(b.confidence, 4),
          accuracy: round(b.accuracy, 4),
          count: b.count
        }))
      };
    }
    if (m.thresholds) entry.thresholds = m.thresholds.map((t) => ({ threshold: round(t.threshold, 4), far: round(t.far, 4), frr: round(t.frr, 4), f1: round(t.f1, 4) }));
    if (m.worst) entry.worst = m.worst.slice(0, 12).map((w) => ({ index: w.index, label: w.label, predicted: w.predicted, error: round(w.error, 4), text: w.text === undefined ? null : String(w.text).slice(0, 160) }));
    out[split] = entry;
  }
  if (taskKey) {
    for (const split of Object.keys(out)) {
      out[split].orderedKeys = orderScalars(taskKey, out[split].scalars);
    }
  }
  return out;
}

function trainingBlock(run, dataset) {
  if (!run) return null;
  const cfg = run.config || {};
  return {
    seed: run.seed === undefined ? cfg.seed : run.seed,
    taskKey: run.taskKey,
    monitor: run.monitor || null,
    primaryMetric: run.primaryMetric || null,
    epochsRequested: cfg.epochs,
    epochsRun: run.epochsRun === undefined ? (run.history || []).length : run.epochsRun,
    steps: run.steps,
    samplesSeen: run.samplesSeen,
    durationMs: run.durationMs,
    startedAt: run.startedAt,
    endedAt: run.endedAt,
    best: run.best ? { epoch: run.best.epoch, value: round(run.best.value, 6), metric: run.best.metric || null } : null,
    optimizer: cfg.optimizer ? JSON.parse(JSON.stringify(cfg.optimizer)) : null,
    schedule: cfg.schedule ? JSON.parse(JSON.stringify(cfg.schedule)) : null,
    loss: cfg.loss || null,
    batchSize: cfg.batchSize,
    evalBatchSize: cfg.evalBatchSize,
    gradClip: cfg.gradClip,
    labelSmoothing: cfg.labelSmoothing,
    classWeighting: cfg.classWeighting,
    posWeight: cfg.posWeight,
    focalGamma: cfg.focalGamma,
    temperature: cfg.temperature,
    huberDelta: cfg.huberDelta,
    confThreshold: cfg.confThreshold,
    decisionThreshold: cfg.decisionThreshold,
    ema: cfg.ema,
    emaDecay: cfg.emaDecay,
    balancedSampler: cfg.balancedSampler,
    accumulationSteps: cfg.accumulationSteps,
    inputNoise: cfg.inputNoise,
    earlyStopping: cfg.earlyStopping ? JSON.parse(JSON.stringify(cfg.earlyStopping)) : null,
    augmentation: cfg.augmentation ? JSON.parse(JSON.stringify(cfg.augmentation)) : null,
    timeBudgetSeconds: cfg.timeBudgetSeconds,
    configHash: configHash(cfg),
    datasetDigest: dataset && dataset.digest ? dataset.digest.contentHash : null,
    history: (run.history || []).map((h) => {
      const row = { epoch: h.epoch };
      const scalars = h.scalars || h;
      for (const k of Object.keys(scalars)) {
        if (typeof scalars[k] === 'number') row[k] = round(scalars[k], 6);
      }
      return row;
    })
  };
}

function artifactBlock(artifacts) {
  if (!artifacts || !artifacts.length) return [];
  return artifacts.map((a) => ({
    format: a.format,
    precision: a.precision || 'fp32',
    files: (a.files || []).map((f) => ({ name: f.name, bytes: f.bytes === undefined ? (f.data ? f.data.length : null) : f.bytes, sha256: f.sha256 || null })),
    nodes: a.nodes === undefined ? null : a.nodes,
    weights: a.weights === undefined ? null : a.weights,
    opset: a.opset === undefined ? null : a.opset,
    bytes: a.bytes === undefined ? null : a.bytes,
    inputName: a.inputName || null,
    outputName: a.outputName || null,
    inputShape: a.inputShape ? a.inputShape.slice() : null,
    outputShape: a.outputShape ? a.outputShape.slice() : null,
    fixedSequenceLength: a.fixedSequenceLength === undefined ? null : a.fixedSequenceLength,
    head: a.head || null,
    notes: (a.notes || []).slice(),
    unsupported: (a.unsupported || []).slice()
  }));
}

function quantBlock(quantization) {
  if (!quantization) return null;
  return {
    baseline: quantization.baseline ? { precision: quantization.baseline.precision, metric: quantization.baseline.metric === undefined ? null : round(quantization.baseline.metric, 6), metricKey: quantization.baseline.metricKey || null } : null,
    variants: (quantization.variants || []).map((v) => ({
      precision: v.precision,
      bytes: v.bytes === undefined ? null : v.bytes,
      compressionRatio: v.report && v.report.compressionRatio ? round(v.report.compressionRatio, 3) : null,
      sizeSaved: v.report && v.report.sizeSaved !== undefined ? v.report.sizeSaved : null,
      maxRelativeL2: v.report && v.report.maxRelativeL2 !== undefined ? round(v.report.maxRelativeL2, 8) : null,
      worstTensor: v.report ? v.report.worstTensor : null,
      worstSnrDb: v.report && v.report.worstSnrDb !== undefined ? round(v.report.worstSnrDb, 3) : null,
      metric: v.metric === undefined ? null : round(v.metric, 6),
      metricKey: v.metricKey || null,
      metricDelta: v.metric === undefined || !quantization.baseline || quantization.baseline.metric === undefined ? null : round(v.metric - quantization.baseline.metric, 6),
      verdict: v.verdict || null,
      perTensor: v.report && v.report.perTensor ? v.report.perTensor.slice(0, 24).map((t) => ({ name: t.name, snrDb: round(t.snrDb, 3), maxRelativeL2: round(t.maxRelativeL2, 8) })) : []
    }))
  };
}

export function usageSnippets(card) {
  const inputName = (card.model.inputName || 'input').replace(/[^A-Za-z0-9_]/g, '_');
  const shape = card.model.inputShape || [];
  const shapeLiteral = '[1' + shape.map((d) => ', ' + d).join('') + ']';
  return {
    onnx: [
      'import onnxruntime as ort',
      'import numpy as np',
      '',
      'session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])',
      'x = np.zeros(' + shapeLiteral + ', dtype=np.float32)',
      'out = session.run(None, {"' + inputName + '": x})[0]',
      'print(out.shape)'
    ].join('\n'),
    tfjs: [
      'import * as tf from "@tensorflow/tfjs";',
      '',
      'const model = await tf.loadGraphModel("tfjs/model.json");',
      'const x = tf.zeros(' + shapeLiteral + ');',
      'const out = model.predict(x);',
      '(Array.isArray(out) ? out[0] : out).print();'
    ].join('\n'),
    safetensors: [
      'from safetensors.numpy import load_file',
      '',
      'tensors = load_file("model.safetensors")',
      'for name, array in tensors.items():',
      '    print(name, array.shape, array.dtype)'
    ].join('\n'),
    browser: [
      'const spec = await (await fetch("config/spec.json")).json();',
      'const tensors = await (await fetch("weights/model.safetensors")).arrayBuffer();',
      'const model = new Model(spec);',
      'model.build(' + (card.reproducibility ? card.reproducibility.seed : 0) + ');',
      'model.loadWeights(readSafetensors(new Uint8Array(tensors)).tensors);',
      'const out = model.forward(new Node(input, [1' + shape.map((d) => ', ' + d).join('') + ']), "eval");'
    ].join('\n')
  };
}

export function limitationLines(card) {
  const lines = [];
  const info = card.model;
  if (info.params <= 60000) lines.push('At ' + info.params.toLocaleString('en-US') + ' parameters this model has very little capacity. It learns the pattern in the training distribution and generalises poorly to inputs that look different.');
  else if (info.params <= 400000) lines.push('At ' + info.params.toLocaleString('en-US') + ' parameters the model is deliberately small. Expect it to fail on out-of-distribution inputs rather than degrade gracefully.');
  const ds = card.dataset;
  if (ds) {
    if (ds.samples <= 400) lines.push('Trained on ' + ds.samples + ' examples. Every metric in this card has a wide confidence interval; treat differences below a few points as noise.');
    if (ds.splits && ds.splits.test && ds.splits.test < 30) lines.push('The held-out test split has only ' + ds.splits.test + ' examples, so test-set numbers are indicative rather than conclusive.');
    if (ds.stats && ds.stats.imbalanceRatio && ds.stats.imbalanceRatio > 5) lines.push('Class imbalance of ' + ds.stats.imbalanceRatio.toFixed(1) + 'x between the largest and smallest class. Per-class recall on the rare classes is the number to watch, not overall accuracy.');
    if (ds.source === 'generated' || ds.source === 'preset' || ds.presetKey) lines.push('The training data was synthesised locally for development. Re-train on your own labelled data before relying on this model in production.');
    if (ds.license && ds.license.allowsCommercial === false) lines.push('Commercial rights for dataset licence "' + ds.license.name + '" are unconfirmed or restricted; review the actual terms and applicable obligations before use.');
    if (ds.provenance && ds.provenance.reference && ds.provenance.reference.license && ds.provenance.reference.license.allowsCommercial === false) {
      lines.push('The reference corpus behind this preset (' + ds.provenance.reference.name + ') is not cleared for commercial use. The generated stand-in data is CC0, but a production model needs real data with real rights.');
    }
  }
  const t = card.model.task;
  if (t === 'binary' || t === 'frame_binary') lines.push('The operating threshold is a product decision, not a training artefact. Choose it on validation data, then assess it once on independent test data.');
  if (t === 'detection') lines.push('Box regression is the weak part of small detectors. Objectness scores are usually well calibrated long before the coordinates are accurate.');
  if (t === 'spectral_mask') lines.push('Mask-based audio enhancement removes stationary noise well and musical noise badly. Listen to the output before shipping it.');
  if (t === 'lm') lines.push('A character or byte level model of this size produces fluent local structure and unreliable long-range facts. Do not use it as a knowledge source.');
  if (t === 'anomaly') lines.push('Anomaly scores are relative to the training distribution. Re-fit the threshold whenever the deployment data drifts.');
  if (card.model.fixedSequenceLength) lines.push('The TensorFlow.js graph is statically unrolled for sequence length ' + card.model.fixedSequenceLength + '. Inputs must be padded or truncated to exactly that length. The ONNX graph accepts variable length.');
  const quant = card.quantization;
  if (quant && quant.variants && quant.variants.length) {
    const worst = quant.variants.reduce((a, b) => ((a.maxRelativeL2 || 0) > (b.maxRelativeL2 || 0) ? a : b));
    if (worst && worst.maxRelativeL2 > 0.01) lines.push('The ' + worst.precision + ' export moves weights by up to ' + (worst.maxRelativeL2 * 100).toFixed(2) + '% relative L2. Re-measure the task metric after quantising, do not assume it holds.');
  }
  lines.push('No adversarial robustness testing was performed. Small models are typically more brittle under adversarial perturbation, not less.');
  return lines;
}

export function buildModelCard(input) {
  const opts = input || {};
  const spec = opts.spec;
  const info = spec ? analyze(spec) : null;
  const dataset = opts.dataset || null;
  const run = opts.run || null;
  const taskKey = opts.taskKey || (run && run.taskKey) || (dataset && dataset.task) || (spec && spec.meta && spec.meta.task) || 'multiclass';
  const task = taskFor(taskKey) || { key: taskKey, label: taskKey };
  const headlineKey = opts.primaryMetric || headlineFor(taskKey);
  const testMetrics = run && run.metrics ? run.metrics.test || run.metrics.val : null;
  const valMetrics = run && run.metrics ? run.metrics.val : null;
  const headlineSource = testMetrics || valMetrics;
  const headlineValue = headlineSource && headlineSource.scalars ? headlineSource.scalars[headlineKey] : undefined;

  const model = {
    name: opts.modelName || (spec && spec.name) || 'Untitled model',
    key: opts.modelKey || (spec && spec.meta && spec.meta.key) || null,
    family: opts.family || (spec && spec.meta && spec.meta.family) || null,
    version: (spec && spec.version) || 1,
    task: taskKey,
    taskLabel: task.label || taskKey,
    head: (spec && spec.head) || 'linear',
    inputName: spec && spec.input ? spec.input.name : null,
    inputShape: spec && spec.input ? spec.input.shape.slice() : [],
    inputDtype: spec && spec.input ? spec.input.dtype || 'float32' : 'float32',
    outputShape: info ? info.outputShape.slice() : [],
    params: info ? info.totalParams : 0,
    trainableParams: opts.trainableParams === undefined ? (info ? info.totalParams : 0) : opts.trainableParams,
    states: info ? info.totalStates : 0,
    macs: info ? info.macs : 0,
    paramBytes: info ? info.paramBytes : 0,
    activationBytes: info ? info.activationBytes : 0,
    peakMemoryBytes: info ? info.paramBytes + info.activationBytes : 0,
    layers: info ? architectureRows(spec, info) : [],
    nodeStats: info ? JSON.parse(JSON.stringify(info.nodeStats)) : {},
    specHash: spec ? shortHash(sha256Text(stableStringify(spec)), 16) : null,
    specText: spec ? specToText(spec) : '',
    fixedSequenceLength: opts.fixedSequenceLength === undefined ? null : opts.fixedSequenceLength,
    description: opts.description || (spec && spec.meta && spec.meta.description) || ''
  };

  const card = {
    cardVersion: CARD_VERSION,
    generatedAt: opts.generatedAt || Date.now(),
    tool: { name: TOOL_NAME, version: TOOL_VERSION, runtime: typeof window !== 'undefined' && typeof document !== 'undefined' ? 'browser' : 'node' },
    model,
    dataset: datasetBlock(dataset),
    training: trainingBlock(run, dataset),
    results: {
      headline: { key: headlineKey, label: metricMeta(headlineKey).label, value: headlineValue === undefined ? null : round(headlineValue, 6), formatted: formatMetric(headlineKey, headlineValue), direction: metricMeta(headlineKey).direction },
      splits: run && run.metrics ? metricsBlock(run.metrics, taskKey) : null,
      comparedAgainst: opts.comparedAgainst || null
    },
    exports: artifactBlock(opts.artifacts),
    quantization: quantBlock(opts.quantization),
    reproducibility: {
      seed: run ? (run.seed === undefined ? (run.config || {}).seed : run.seed) : (opts.seed === undefined ? null : opts.seed),
      datasetContentHash: dataset && dataset.digest ? dataset.digest.contentHash : null,
      datasetSpecHash: dataset && dataset.digest ? dataset.digest.specHash : null,
      specHash: model.specHash,
      configHash: run && run.config ? configHash(run.config) : null,
      splitSeed: dataset ? dataset.splitSeed : null,
      splitMethod: dataset ? dataset.splitMethod : null,
      splitRatios: dataset ? (dataset.splitRatios || []).slice() : null,
      durationMs: run ? run.durationMs : null,
      startedAt: run ? run.startedAt : null,
      endedAt: run ? run.endedAt : null,
      steps: run ? run.steps : null,
      samplesSeen: run ? run.samplesSeen : null,
      userAgent: typeof navigator === 'undefined' ? null : String(navigator.userAgent || '')
    },
    notes: (opts.notes || []).slice()
  };
  card.limitations = limitationLines(card);
  card.usage = usageSnippets(card);
  return card;
}

function table(headers, rows) {
  const out = [];
  out.push('| ' + headers.join(' | ') + ' |');
  out.push('|' + headers.map(() => ' --- ').join('|') + '|');
  for (const r of rows) out.push('| ' + r.map((c) => (c === null || c === undefined ? '-' : String(c))).join(' | ') + ' |');
  return out.join('\n');
}

function bullets(list) {
  return list.map((l) => '- ' + l).join('\n');
}

export function cardToMarkdown(card) {
  const L = [];
  L.push('# ' + card.model.name);
  L.push('');
  L.push('Model card generated by ' + card.tool.name + ' ' + card.tool.version + ' on ' + new Date(card.generatedAt).toISOString().slice(0, 19).replace('T', ' ') + ' UTC.');
  L.push('');
  if (card.model.description) {
    L.push(card.model.description);
    L.push('');
  }

  L.push('## Summary');
  L.push('');
  L.push(table(['Field', 'Value'], [
    ['Task', card.model.taskLabel + ' (' + card.model.task + ')'],
    ['Parameters', card.model.params.toLocaleString('en-US')],
    ['Multiply-accumulate ops', card.model.macs.toLocaleString('en-US')],
    ['Weight memory', formatBytes(card.model.paramBytes)],
    ['Peak activation memory (batch 1)', formatBytes(card.model.activationBytes)],
    ['Input', card.model.inputName + ' ' + JSON.stringify(card.model.inputShape) + ' ' + card.model.inputDtype],
    ['Output', JSON.stringify(card.model.outputShape) + ', head ' + card.model.head],
    ['Dataset', card.dataset ? card.dataset.name + ' (' + card.dataset.samples + ' samples, ' + card.dataset.modality + ')' : 'not attached'],
    ['Headline metric', card.results.headline.label + ' = ' + card.results.headline.formatted],
    ['Seed', card.reproducibility.seed === null ? '-' : card.reproducibility.seed],
    ['Dataset hash', card.reproducibility.datasetContentHash || '-'],
    ['Architecture hash', card.reproducibility.specHash || '-']
  ]));
  L.push('');

  L.push('## Intended use');
  L.push('');
  L.push('This model performs ' + card.model.taskLabel.toLowerCase() + ' on inputs shaped ' + JSON.stringify(card.model.inputShape) + ' per sample. It is sized for on-device and edge deployment: ' + card.model.params.toLocaleString('en-US') + ' parameters, ' + formatBytes(card.model.paramBytes) + ' of weights.');
  if (card.dataset) {
    L.push('');
    L.push('In-distribution inputs look like the "' + card.dataset.name + '" data described below. Anything outside that description is out of scope.');
  }
  L.push('');

  L.push('## Architecture');
  L.push('');
  L.push(table(['#', 'Layer', 'Type', 'Output shape', 'Params', 'MACs'], card.model.layers.map((r, i) => [
    i + 1, r.id, r.type, JSON.stringify(r.outputShape), r.params ? r.params.toLocaleString('en-US') : '-', r.macs ? r.macs.toLocaleString('en-US') : '-'
  ])));
  L.push('');
  const groups = {};
  for (const r of card.model.layers) {
    groups[r.type] = groups[r.type] || { count: 0, params: 0 };
    groups[r.type].count++;
    groups[r.type].params += r.params;
  }
  L.push(table(['Layer type', 'Count', 'Params'], Object.keys(groups).sort().map((k) => [k, groups[k].count, groups[k].params.toLocaleString('en-US')])));
  L.push('');

  if (card.dataset) {
    const d = card.dataset;
    L.push('## Dataset');
    L.push('');
    L.push(table(['Field', 'Value'], [
      ['Name', d.name],
      ['Modality', d.modality],
      ['Task', d.task],
      ['Samples', d.samples],
      ['Feature shape', JSON.stringify(d.xShape)],
      ['Target shape', JSON.stringify(d.yShape)],
      ['Classes', d.classes === null ? '-' : d.classes + (d.labels && d.labels.length ? ' (' + d.labels.slice(0, 12).join(', ') + (d.labels.length > 12 ? ', ...' : '') + ')' : '')],
      ['Split method', d.splitMethod + ' ' + JSON.stringify(d.splitRatios) + ' seed ' + d.splitSeed],
      ['Train / val / test', d.splits ? d.splits.train + ' / ' + d.splits.val + ' / ' + d.splits.test : '-'],
      ['Source', d.source + (d.presetKey ? ' (preset ' + d.presetKey + ')' : '')],
      ['Feature hash', d.digest ? d.digest.featureHash : '-'],
      ['Content hash', d.digest ? d.digest.contentHash : '-'],
      ['Size on disk', d.digest ? formatBytes(d.digest.bytes) : '-']
    ]));
    L.push('');
    L.push('### Licence');
    L.push('');
    if (d.license) {
      L.push('- Attached data: **' + d.license.name + '** (' + d.license.id + ')');
      L.push('- Commercial use: ' + (d.license.allowsCommercial ? 'permitted' : '**not permitted**'));
      if (d.license.notes) L.push('- ' + d.license.notes);
    } else {
      L.push('- No licence recorded. Treat the data as all-rights-reserved until you set one.');
    }
    if (d.provenance && d.provenance.reference) {
      const r = d.provenance.reference;
      L.push('- Reference corpus: **' + r.name + '**');
      if (r.url) L.push('- Reference URL: ' + r.url);
      if (r.license) L.push('- Reference licence: ' + r.license.name + ' (' + r.license.id + '), commercial use ' + (r.license.allowsCommercial ? 'permitted' : '**not permitted**'));
      if (r.note) L.push('- ' + r.note);
    }
    L.push('');
    L.push('The weights in this bundle are yours. The licence above describes the data they were trained on, which is what usually decides whether you can ship them.');
    L.push('');
    if (d.stats) {
      L.push('### Data quality');
      L.push('');
      L.push(table(['Check', 'Value'], [
        ['Feature range', (d.stats.xMin === null ? '-' : d.stats.xMin) + ' .. ' + (d.stats.xMax === null ? '-' : d.stats.xMax)],
        ['Feature mean / std', (d.stats.xMean === null ? '-' : d.stats.xMean) + ' / ' + (d.stats.xStd === null ? '-' : d.stats.xStd)],
        ['Imbalance ratio', d.stats.imbalanceRatio === null ? '-' : d.stats.imbalanceRatio + 'x'],
        ['Missing values', d.stats.missing],
        ['Duplicate samples removed', d.stats.duplicatesRemoved]
      ]));
      if (d.stats.labelCounts && d.labels && d.labels.length) {
        L.push('');
        L.push(table(['Class', 'Samples', 'Share'], d.stats.labelCounts.map((c, i) => [d.labels[i] === undefined ? i : d.labels[i], c, d.samples ? (100 * c / d.samples).toFixed(1) + '%' : '-'])));
      }
      L.push('');
    }
  }

  if (card.training) {
    const t = card.training;
    L.push('## Training');
    L.push('');
    L.push(table(['Field', 'Value'], [
      ['Loss', t.loss],
      ['Optimizer', t.optimizer ? t.optimizer.kind + ', lr ' + t.optimizer.lr + (t.optimizer.weightDecay ? ', wd ' + t.optimizer.weightDecay : '') : '-'],
      ['Schedule', t.schedule ? t.schedule.kind + (t.schedule.warmupEpochs ? ', warmup ' + t.schedule.warmupEpochs : '') + (t.schedule.minLrFraction !== undefined ? ', min lr ' + t.schedule.minLrFraction + 'x' : '') : '-'],
      ['Epochs', t.epochsRun + ' of ' + t.epochsRequested + ' requested'],
      ['Batch size', t.batchSize + ' (eval ' + t.evalBatchSize + ')'],
      ['Gradient clipping', t.gradClip],
      ['Label smoothing', t.labelSmoothing],
      ['Class weighting', t.classWeighting],
      ['Balanced sampler', t.balancedSampler ? 'on' : 'off'],
      ['EMA', t.ema ? 'on, decay ' + t.emaDecay : 'off'],
      ['Early stopping', t.earlyStopping && t.earlyStopping.enabled ? t.earlyStopping.monitor + ', patience ' + t.earlyStopping.patience : 'off'],
      ['Augmentation', t.augmentation && t.augmentation.enabled ? (t.augmentation.ops || []).join(', ') || 'enabled' : 'off'],
      ['Seed', t.seed],
      ['Wall clock', formatDuration(t.durationMs)],
      ['Steps', t.steps],
      ['Samples seen', t.samplesSeen === undefined ? '-' : t.samplesSeen.toLocaleString('en-US')],
      ['Best checkpoint', t.best ? 'epoch ' + t.best.epoch + ' at ' + t.best.value : '-'],
      ['Config hash', t.configHash]
    ]));
    L.push('');
    if (t.history && t.history.length) {
      const keys = [];
      for (const row of t.history) for (const k of Object.keys(row)) if (k !== 'epoch' && keys.indexOf(k) < 0 && keys.length < 6) keys.push(k);
      L.push('### Epoch history');
      L.push('');
      L.push(table(['Epoch'].concat(keys.map((k) => metricMeta(k).label)), t.history.map((row) => [row.epoch].concat(keys.map((k) => formatMetric(k, row[k]))))));
      L.push('');
    }
  }

  if (card.results.splits) {
    L.push('## Results');
    L.push('');
    const splits = Object.keys(card.results.splits);
    const keys = [];
    for (const s of splits) for (const k of card.results.splits[s].orderedKeys || Object.keys(card.results.splits[s].scalars)) if (keys.indexOf(k) < 0 && keys.length < 14) keys.push(k);
    L.push(table(['Metric'].concat(splits.map((s) => s + ' (n=' + (card.dataset && card.dataset.splits ? card.dataset.splits[s] || 0 : 0) + ')')), keys.map((k) => [metricMeta(k).label].concat(splits.map((s) => formatMetric(k, card.results.splits[s].scalars[k]))))));
    L.push('');
    for (const s of splits) {
      const block = card.results.splits[s];
      if (block.perClass && block.perClass.length) {
        L.push('### Per-class on ' + s);
        L.push('');
        L.push(table(['Class', 'Support', 'Precision', 'Recall', 'F1'], block.perClass.map((c) => [c.label, c.support, c.precision === null ? '-' : (c.precision * 100).toFixed(1) + '%', c.recall === null ? '-' : (c.recall * 100).toFixed(1) + '%', c.f1 === null ? '-' : (c.f1 * 100).toFixed(1) + '%'])));
        L.push('');
      }
      if (block.confusion && block.confusion.labels.length) {
        L.push('### Confusion matrix on ' + s + ' (rows = true, columns = predicted)');
        L.push('');
        L.push(table(['true \\ pred'].concat(block.confusion.labels), block.confusion.matrix.map((row, i) => [block.confusion.labels[i]].concat(row))));
        L.push('');
      }
      if (block.calibration) {
        L.push('### Calibration on ' + s);
        L.push('');
        L.push('Expected calibration error ' + (block.calibration.ece * 100).toFixed(2) + '%, max bin error ' + (block.calibration.maxError * 100).toFixed(2) + '%' + (block.calibration.brier === null ? '' : ', Brier ' + block.calibration.brier.toFixed(4)) + '.');
        if (block.calibration.bins && block.calibration.bins.length) {
          L.push('');
          L.push(table(['Confidence bin', 'Mean confidence', 'Observed accuracy', 'Count'], block.calibration.bins.map((b) => [b.bin === undefined ? '-' : b.bin, (b.confidence * 100).toFixed(1) + '%', (b.accuracy * 100).toFixed(1) + '%', b.count])));
        }
        L.push('');
      }
      if (block.thresholds && block.thresholds.length) {
        L.push('### Threshold sweep on ' + s);
        L.push('');
        L.push(table(['Threshold', 'FAR', 'FRR', 'F1'], block.thresholds.map((t) => [t.threshold.toFixed(3), (t.far * 100).toFixed(2) + '%', (t.frr * 100).toFixed(2) + '%', t.f1 === null ? '-' : (t.f1 * 100).toFixed(1) + '%'])));
        L.push('');
      }
      if (block.worst && block.worst.length) {
        L.push('### Worst examples on ' + s);
        L.push('');
        L.push(table(['Index', 'True', 'Predicted', 'Error', 'Sample'], block.worst.map((w) => [w.index, w.label === null || w.label === undefined ? '-' : w.label, w.predicted === null || w.predicted === undefined ? '-' : w.predicted, w.error === null ? '-' : w.error, w.text || '-'])));
        L.push('');
      }
    }
  }

  if (card.exports.length) {
    L.push('## Exported artifacts');
    L.push('');
    const rows = [];
    for (const a of card.exports) {
      for (const f of a.files) rows.push([f.name, a.format, a.precision, formatBytes(f.bytes || 0), f.sha256 ? f.sha256.slice(0, 16) : '-']);
    }
    L.push(table(['File', 'Format', 'Precision', 'Size', 'SHA-256 (first 16)'], rows));
    L.push('');
    const notes = [];
    for (const a of card.exports) for (const n of a.notes || []) notes.push('[' + a.format + '/' + a.precision + '] ' + n);
    if (notes.length) {
      L.push('Exporter notes:');
      L.push('');
      L.push(bullets(notes));
      L.push('');
    }
  }

  if (card.quantization && card.quantization.variants.length) {
    L.push('## Quantization');
    L.push('');
    const b = card.quantization.baseline;
    if (b) L.push('Baseline ' + b.precision + (b.metricKey ? ' ' + b.metricKey + ' = ' + formatMetric(b.metricKey, b.metric) : '') + '.');
    L.push('');
    L.push(table(['Precision', 'Size', 'Compression', 'Worst weight error (rel L2)', 'Worst tensor SNR', b && b.metricKey ? metricMeta(b.metricKey).label : 'Metric', 'Delta', 'Verdict'], card.quantization.variants.map((v) => [
      v.precision,
      formatBytes(v.bytes || 0),
      v.compressionRatio === null ? '-' : v.compressionRatio.toFixed(2) + 'x',
      v.maxRelativeL2 === null ? '-' : v.maxRelativeL2.toExponential(2),
      v.worstSnrDb === null ? '-' : v.worstSnrDb.toFixed(1) + ' dB',
      v.metricKey && v.metric !== null ? formatMetric(v.metricKey, v.metric) : '-',
      v.metricDelta === null ? '-' : (v.metricDelta >= 0 ? '+' : '') + formatMetric(v.metricKey || (b ? b.metricKey : 'loss'), v.metricDelta),
      v.verdict || '-'
    ])));
    L.push('');
    L.push('Weight-error columns measure the export only. The metric column re-runs evaluation with the quantised weights loaded back into the model, which is the number that matters.');
    L.push('');
  }

  L.push('## Reproducibility');
  L.push('');
  const r = card.reproducibility;
  L.push(table(['Field', 'Value'], [
    ['Random seed', r.seed === null ? '-' : r.seed],
    ['Dataset split seed', r.splitSeed === null ? '-' : r.splitSeed + ' (' + r.splitMethod + ' ' + JSON.stringify(r.splitRatios) + ')'],
    ['Dataset content hash', r.datasetContentHash || '-'],
    ['Dataset spec hash', r.datasetSpecHash || '-'],
    ['Architecture hash', r.specHash || '-'],
    ['Training config hash', r.configHash || '-'],
    ['Steps', r.steps === null ? '-' : r.steps],
    ['Samples seen', r.samplesSeen === null ? '-' : r.samplesSeen],
    ['Wall clock', formatDuration(r.durationMs)],
    ['Started', r.startedAt ? new Date(r.startedAt).toISOString() : '-'],
    ['Finished', r.endedAt ? new Date(r.endedAt).toISOString() : '-'],
    ['User agent', r.userAgent || '-']
  ]));
  L.push('');
  L.push('Re-running training with the same seed, the same architecture hash and the same dataset content hash reproduces these numbers on the same JavaScript engine. Different engines reorder floating point additions and will drift in the last few digits.');
  L.push('');

  L.push('## Loading the model');
  L.push('');
  L.push('### ONNX Runtime (Python)');
  L.push('');
  L.push('```python');
  L.push(card.usage.onnx);
  L.push('```');
  L.push('');
  L.push('### TensorFlow.js');
  L.push('');
  L.push('```javascript');
  L.push(card.usage.tfjs);
  L.push('```');
  L.push('');
  L.push('### safetensors (raw weights)');
  L.push('');
  L.push('```python');
  L.push(card.usage.safetensors);
  L.push('```');
  L.push('');

  L.push('## Limitations');
  L.push('');
  L.push(bullets(card.limitations));
  L.push('');

  if (card.notes.length) {
    L.push('## Notes from the author');
    L.push('');
    L.push(bullets(card.notes));
    L.push('');
  }

  L.push('## Card provenance');
  L.push('');
  L.push('Card format version ' + card.cardVersion + '. Generated in the ' + card.tool.runtime + ' by ' + card.tool.name + ' ' + card.tool.version + '. This file is part of the export bundle and is meant to be read by whoever has to decide whether the model is safe to deploy.');
  L.push('');
  return L.join('\n');
}

export function cardFiles(card) {
  const markdown = cardToMarkdown(card);
  return [
    { name: 'README.md', data: markdown },
    { name: 'model-card.json', data: JSON.stringify(card, null, 2) },
    { name: 'config/architecture.json', data: JSON.stringify({ name: card.model.name, version: card.model.version, task: card.model.task, head: card.model.head, input: { name: card.model.inputName, shape: card.model.inputShape, dtype: card.model.inputDtype }, output: card.model.outputShape, params: card.model.params, macs: card.model.macs, layers: card.model.layers }, null, 2) },
    { name: 'config/reproducibility.json', data: JSON.stringify(card.reproducibility, null, 2) },
    { name: 'metrics/results.json', data: JSON.stringify(card.results, null, 2) }
  ];
}

export function cardPlainText(card) {
  return cardToMarkdown(card).replace(/[#*`|]/g, '');
}
