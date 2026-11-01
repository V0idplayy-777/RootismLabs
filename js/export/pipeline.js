import { analyze } from '../core/model.js';
import { exportOnnx } from './onnx.js';
import { exportTfjs } from './tfjs.js';
import { safetensorsExport } from './safetensors.js';
import { quantizeNamedTensors, statesToNamed, applyQuantizedToModel, precisionByKey, precisionRecommendation, bytesFor } from '../core/quant.js';
import { Trainer, defaultPrimary, monitorMode, normalizeConfig } from '../core/train.js';
import { taskFor } from '../core/tasks.js';
import { buildModelCard } from './card.js';
import { buildBundle } from './bundle.js';
import { metricMeta, headlineFor } from '../presets/metrics.js';

export const EXPORT_FORMATS = [
  {
    key: 'onnx',
    label: 'ONNX',
    extension: '.onnx',
    mime: 'application/octet-stream',
    directory: 'onnx',
    precisions: ['fp32', 'fp16', 'int8', 'uint8'],
    runtime: 'onnxruntime, ONNX Runtime Web, Core ML, OpenVINO, TensorRT',
    blurb: 'A single self-contained protobuf with the weights inside. Variable sequence lengths work for recurrent models. This is the format to hand to a backend team.'
  },
  {
    key: 'tfjs',
    label: 'TensorFlow.js',
    extension: '.json',
    mime: 'application/json',
    directory: 'tfjs',
    precisions: ['fp32', 'fp16'],
    runtime: 'tf.loadGraphModel in the browser, Node.js, React Native',
    blurb: 'A graph-model JSON plus a flat weight binary. Runs straight in a browser with no build step. Recurrent models are unrolled to a fixed sequence length.'
  },
  {
    key: 'safetensors',
    label: 'safetensors',
    extension: '.safetensors',
    mime: 'application/octet-stream',
    directory: 'weights',
    precisions: ['fp32', 'fp16', 'bf16', 'int8', 'uint8'],
    runtime: 'safetensors in Python or Rust, any framework that reads the format',
    blurb: 'Raw weights with a JSON header and no code execution on load. Integer variants store the quantised tensor plus its per-channel scale.'
  }
];

export function formatByKey(key) {
  for (const f of EXPORT_FORMATS) if (f.key === key) return f;
  return null;
}

export function precisionsFor(formatKey) {
  const f = formatByKey(formatKey);
  if (!f) return [];
  return f.precisions.map((k) => precisionByKey(k));
}

function headFor(spec) {
  return spec && spec.head ? spec.head : 'linear';
}

function metaFor(model, spec, precision, extra) {
  const base = {
    format: 'rootism-labs',
    model: (spec && spec.name) || 'model',
    task: (spec && spec.meta && spec.meta.task) || '',
    params: String(model.paramCount ? model.paramCount() : 0),
    precision
  };
  return Object.assign(base, extra || {});
}

export function exportFormat(model, formatKey, precision, options) {
  const opts = options || {};
  const spec = model.spec;
  const head = opts.head === undefined ? headFor(spec) : opts.head;
  const precisionKey = precision || 'fp32';
  const info = analyze(spec);

  if (formatKey === 'onnx') {
    const out = exportOnnx(model, { precision: precisionKey, head, docString: opts.docString || '', graphName: opts.graphName || 'rootism_graph', metadata: opts.metadata });
    return {
      format: 'onnx',
      precision: precisionKey,
      files: [{ name: 'model.onnx', data: out.bytes }],
      bytes: out.bytes.length,
      nodes: out.nodeCount,
      weights: out.initializerCount,
      opset: out.opset,
      inputName: out.inputName,
      outputName: out.outputName,
      inputShape: out.inputShape,
      outputShape: out.outputShape,
      head: out.head,
      notes: out.notes,
      unsupported: out.unsupported,
      fixedSequenceLength: null,
      extension: '.onnx'
    };
  }

  if (formatKey === 'tfjs') {
    const out = exportTfjs(model, { precision: precisionKey, head, metadata: opts.metadata });
    return {
      format: 'tfjs',
      precision: precisionKey,
      modelJson: out.modelJson,
      weightData: out.weightData,
      files: [
        { name: 'model.json', data: JSON.stringify(out.modelJson) },
        { name: 'model.weights.bin', data: out.weightData }
      ],
      bytes: out.byteLength,
      nodes: out.nodeCount,
      weights: out.weightCount,
      opset: null,
      inputName: out.inputName,
      outputName: out.outputName,
      inputShape: out.inputShape,
      outputShape: out.outputShape,
      head: out.head,
      notes: out.notes,
      unsupported: out.unsupported,
      fixedSequenceLength: out.fixedSequenceLength,
      extension: '.json'
    };
  }

  if (formatKey === 'safetensors') {
    const out = safetensorsExport(model.weights(), {
      precision: precisionKey,
      metadata: metaFor(model, spec, precisionKey, opts.metadata)
    });
    return {
      format: 'safetensors',
      precision: precisionKey,
      files: [{ name: 'model.safetensors', data: out.bytes }],
      bytes: out.bytes.length,
      nodes: null,
      weights: out.tensorCount,
      opset: null,
      inputName: spec.input ? spec.input.name : null,
      outputName: 'output',
      inputShape: spec.input ? spec.input.shape.slice() : [],
      outputShape: info.outputShape.slice(),
      head: head,
      notes: out.quantization ? [out.quantization.scheme + '; scales are stored as <tensor>__scale and zero points as <tensor>__zero_point.'] : [],
      unsupported: [],
      fixedSequenceLength: null,
      extension: '.safetensors',
      quantization: out.quantization || null,
      report: out.report || null
    };
  }

  throw new Error('unknown export format ' + formatKey);
}

export function sizeEstimate(model, formatKey, precisionKey) {
  const info = model.info || analyze(model.spec);
  if (formatKey === 'safetensors') return bytesFor(info.totalParams, precisionKey) + 256 + info.paramList.length * 96;
  if (formatKey === 'tfjs') return bytesFor(info.totalParams, precisionKey) + 2048 + estimateTfjsJson(model);
  return bytesFor(info.totalParams, precisionKey) + 4096 + estimateOnnxGraph(model);
}

function estimateTfjsJson(model) {
  const info = model.info || analyze(model.spec);
  const rnn = info.order.filter((id) => {
    const node = model.spec.nodes.find((n) => n.id === id);
    return node && (node.type === 'gru' || node.type === 'lstm');
  }).length;
  if (!rnn) return 6000 + info.order.length * 900;
  let steps = 0;
  for (const node of model.spec.nodes) {
    if (node.type === 'gru' || node.type === 'lstm') steps += model.spec.input.shape[0] * (node.cfg && node.cfg.bidirectional ? 2 : 1);
  }
  return 6000 + info.order.length * 900 + steps * 4200;
}

function estimateOnnxGraph(model) {
  const info = model.info || analyze(model.spec);
  return 2048 + info.order.length * 620;
}

export function buildEvaluator(input) {
  const spec = input.spec;
  const dataset = input.dataset;
  const taskKey = input.taskKey || dataset.task || 'multiclass';
  const task = taskFor(taskKey);
  const config = normalizeConfig(input.config || {}, task);
  const trainer = new Trainer({ spec, dataset, taskKey, config });
  const prepared = trainer.prepare();
  const splitName = input.split || (dataset.splits.test && dataset.splits.test.length ? 'test' : 'val');
  const idx = dataset.splits[splitName];
  const primary = input.primaryMetric || defaultPrimary(task);
  const direction = monitorMode(task, primary);
  return {
    trainer,
    task,
    taskKey,
    config,
    split: splitName,
    samples: idx.length,
    primaryMetric: primary,
    direction,
    k: prepared.k,
    load(weights) {
      trainer.model.loadWeights(weights);
    },
    measure() {
      const result = trainer.evaluate(idx, prepared.k, trainer.config);
      const scalars = result.scalars || {};
      return {
        scalars,
        primary: scalars[primary] === undefined ? null : scalars[primary],
        loss: result.loss === undefined || !Number.isFinite(result.loss) ? null : result.loss,
        error: result.error || null,
        full: result
      };
    }
  };
}

export function measurePrecisionImpact(input) {
  const model = input.model;
  const dataset = input.dataset;
  const spec = model.spec;
  const precisions = (input.precisions && input.precisions.length ? input.precisions : ['fp16', 'int8']).slice();
  const evaluator = buildEvaluator({
    spec,
    dataset,
    taskKey: input.taskKey,
    config: input.config,
    split: input.split,
    primaryMetric: input.primaryMetric
  });
  const original = model.weights().map((w) => ({ name: w.name, shape: w.shape.slice(), data: Float32Array.from(w.data) }));
  evaluator.load(original.map((w) => ({ name: w.name, shape: w.shape, data: w.data })));
  const baseline = evaluator.measure();
  const out = {
    split: evaluator.split,
    samples: evaluator.samples,
    metricKey: evaluator.primaryMetric,
    metricLabel: metricMeta(evaluator.primaryMetric).label,
    direction: evaluator.direction,
    baseline: {
      precision: 'fp32',
      metric: baseline.primary,
      metricKey: evaluator.primaryMetric,
      loss: baseline.loss,
      scalars: baseline.scalars
    },
    variants: []
  };
  const named = statesToNamed(model);
  for (const key of precisions) {
    if (key === 'fp32') continue;
    const quantized = quantizeNamedTensors(named, key, input.quantOptions || {});
    evaluator.load(original.map((w) => ({ name: w.name, shape: w.shape, data: w.data })));
    applyQuantizedToModel(evaluator.trainer.model, quantized.tensors);
    const measured = evaluator.measure();
    const verdict = measured.primary === null || baseline.primary === null
      ? null
      : precisionRecommendation(quantized.report, baseline.primary, measured.primary, evaluator.direction);
    out.variants.push({
      precision: key,
      bytes: quantized.report.quantizedBytes,
      report: quantized.report,
      metric: measured.primary,
      metricKey: evaluator.primaryMetric,
      loss: measured.loss,
      scalars: measured.scalars,
      verdict: verdict ? verdict.note : null,
      acceptable: verdict ? verdict.acceptable : null,
      marginal: verdict ? verdict.marginal : null,
      rejected: verdict ? verdict.rejected : null,
      relativeDelta: verdict ? verdict.relative : null,
      error: measured.error
    });
    if (input.onProgress) input.onProgress({ precision: key, done: out.variants.length, total: precisions.length });
  }
  evaluator.load(original.map((w) => ({ name: w.name, shape: w.shape, data: w.data })));
  return out;
}

export function quantizationTable(impact) {
  if (!impact) return [];
  const rows = [{
    precision: 'fp32',
    label: precisionByKey('fp32').label,
    bytes: impact.baseline.bytes || null,
    metric: impact.baseline.metric,
    delta: null,
    maxRelativeL2: 0,
    worstSnrDb: null,
    verdict: 'Reference weights as trained.'
  }];
  for (const v of impact.variants) {
    rows.push({
      precision: v.precision,
      label: precisionByKey(v.precision).label,
      bytes: v.bytes,
      metric: v.metric,
      delta: v.metric === null || impact.baseline.metric === null ? null : v.metric - impact.baseline.metric,
      maxRelativeL2: v.report.maxRelativeL2,
      worstSnrDb: v.report.worstSnrDb,
      worstTensor: v.report.worstTensor,
      compressionRatio: v.report.compressionRatio,
      verdict: v.verdict
    });
  }
  return rows;
}

export function runExport(input) {
  const model = input.model;
  const spec = model.spec;
  const dataset = input.dataset || null;
  const run = input.run || null;
  const info = model.info || analyze(spec);
  const selections = input.formats || [{ format: 'onnx', precision: 'fp32' }];
  const artifacts = [];
  const problems = [];

  for (const sel of selections) {
    if (input.onProgress) input.onProgress({ stage: 'export', format: sel.format, precision: sel.precision });
    try {
      artifacts.push(exportFormat(model, sel.format, sel.precision, { metadata: input.metadata, docString: input.docString }));
    } catch (e) {
      problems.push({ format: sel.format, precision: sel.precision, message: e.message });
    }
  }

  let impact = input.quantization || null;
  if (!impact && input.measure && dataset && selections.some((s) => s.precision !== 'fp32')) {
    const wanted = [];
    for (const s of selections) if (s.precision !== 'fp32' && wanted.indexOf(s.precision) < 0) wanted.push(s.precision);
    try {
      impact = measurePrecisionImpact({
        model,
        dataset,
        taskKey: input.taskKey,
        config: input.config,
        precisions: wanted,
        onProgress: input.onProgress
      });
    } catch (e) {
      problems.push({ format: 'quantization', precision: wanted.join(','), message: e.message });
    }
  }

  const cardArtifacts = artifacts.map((a) => ({
    format: a.format,
    precision: a.precision,
    files: a.files.map((f) => ({ name: f.name, bytes: f.data ? f.data.length : null })),
    nodes: a.nodes,
    weights: a.weights,
    opset: a.opset,
    bytes: a.bytes,
    inputName: a.inputName,
    outputName: a.outputName,
    inputShape: a.inputShape,
    outputShape: a.outputShape,
    fixedSequenceLength: a.fixedSequenceLength,
    head: a.head,
    notes: a.notes,
    unsupported: a.unsupported
  }));

  const quantForCard = impact ? {
    baseline: {
      precision: 'fp32',
      metric: impact.baseline.metric,
      metricKey: impact.metricKey,
      bytes: bytesFor(info.totalParams, 'fp32')
    },
    variants: impact.variants.map((v) => ({
      precision: v.precision,
      bytes: v.bytes,
      report: v.report,
      metric: v.metric,
      metricKey: v.metricKey,
      verdict: v.verdict
    }))
  } : null;

  const fixedSequence = artifacts.find((a) => a.fixedSequenceLength);
  const card = buildModelCard({
    spec,
    dataset,
    run,
    taskKey: input.taskKey || (run && run.taskKey) || (dataset && dataset.task) || spec.meta && spec.meta.task,
    modelName: input.modelName || spec.name,
    modelKey: input.modelKey || (spec.meta && spec.meta.key) || null,
    family: input.family || (spec.meta && spec.meta.family) || null,
    description: input.description || (spec.meta && spec.meta.description) || '',
    artifacts: cardArtifacts,
    quantization: quantForCard,
    fixedSequenceLength: fixedSequence ? fixedSequence.fixedSequenceLength : null,
    primaryMetric: input.primaryMetric || (run && run.primaryMetric) || headlineFor(input.taskKey || (dataset && dataset.task) || 'multiclass'),
    notes: input.notes || [],
    seed: input.seed
  });

  let bundle = null;
  if (input.bundle !== false) {
    if (input.onProgress) input.onProgress({ stage: 'bundle' });
    bundle = buildBundle({
      card,
      spec,
      run,
      dataset,
      artifacts,
      includeDataset: input.includeDataset !== false,
      sampleLimit: input.sampleLimit === undefined ? 60 : input.sampleLimit,
      includeFullDataset: !!input.includeFullDataset
    });
  }

  return { card, artifacts, quantization: impact, bundle, problems, info };
}

export function downloadArtifact(artifact, filename) {
  if (typeof document === 'undefined') return false;
  const file = artifact.files && artifact.files[0];
  const data = file ? file.data : artifact.weightData;
  if (!data) return false;
  const blob = new Blob([data], { type: artifact.mime || 'application/octet-stream' });
  triggerDownload(blob, filename || defaultFileName(artifact));
  return true;
}

export function defaultFileName(artifact) {
  const f = formatByKey(artifact.format);
  const suffix = artifact.precision === 'fp32' ? '' : '-' + artifact.precision;
  if (artifact.format === 'tfjs') return 'model' + suffix + '.json';
  return 'model' + suffix + (f ? f.extension : '.bin');
}

function triggerDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.style.display = 'none';
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    URL.revokeObjectURL(url);
    if (a.parentNode) a.parentNode.removeChild(a);
  }, 1500);
}

export function downloadBundle(bundle, filename) {
  if (typeof document === 'undefined') return false;
  triggerDownload(new Blob([bundle.bytes], { type: 'application/zip' }), filename || 'model-bundle.zip');
  return true;
}

export function downloadText(text, filename, mime) {
  if (typeof document === 'undefined') return false;
  triggerDownload(new Blob([text], { type: mime || 'text/plain' }), filename);
  return true;
}
