// Preset coverage: every dataset preset builds, every model preset exports a
// valid graph whose input and output match the dataset it points at, and every
// preset trains for a couple of epochs without producing a non-finite loss.
//
// Run with: node tests/preset-domains.mjs

import { analyze } from '../js/core/model.js';
import { Trainer, normalizeConfig } from '../js/core/train.js';
import { taskFor } from '../js/core/tasks.js';
import { PRESET_MODEL_LIST, buildPresetGraph, validatePreset } from '../js/presets/models.js';
import { PRESET_DATASET_LIST, PRESET_DATASETS, buildPresetDataset } from '../js/presets/datasets.js';
import { GENERATORS } from '../js/data/generators.js';
import { buildModelCard, cardToMarkdown } from '../js/export/card.js';

const problems = [];
const check = (condition, message) => {
  if (!condition) problems.push(message);
};

function expectedOutput(task, ds, outShape) {
  const labels = (ds.labels || []).length;
  if (task === 'anomaly') return ds.xShape;
  if (task === 'embedding') return [outShape[0]];
  if (task === 'multiclass' || task === 'pitch_class') return [labels];
  if (task === 'multilabel') return [labels];
  if (task === 'binary') return [1];
  if (task === 'frame_binary' || task === 'onset') return [ds.xShape[0], 1];
  if (task === 'seq_tag') return [ds.xShape[0], labels];
  if (task === 'detection') return [5, outShape[1], outShape[2]];
  if (task === 'lm') return [ds.xShape[0], outShape[1]];
  return ds.yShape;
}

// 1. datasets -----------------------------------------------------------------
console.log('dataset presets');
const datasets = {};
for (const preset of PRESET_DATASET_LIST) {
  let ds = null;
  try {
    ds = buildPresetDataset(preset.key, { size: 'small' });
  } catch (e) {
    problems.push('dataset ' + preset.key + ' failed to build: ' + e.message);
    continue;
  }
  datasets[preset.key] = ds;
  check(ds.count > 0, 'dataset ' + preset.key + ' is empty');
  check(GENERATORS[preset.generator], 'dataset ' + preset.key + ' points at a missing generator');
  check(ds.licenseInfo && ds.licenseInfo.local, 'dataset ' + preset.key + ' has no licence block');
  check(!!ds.digest, 'dataset ' + preset.key + ' has no digest');
  const seen = new Set();
  let total = 0;
  for (const split of ['train', 'val', 'test']) {
    const idx = ds.splits[split];
    check(!!idx && idx.length > 0, 'dataset ' + preset.key + ' has an empty ' + split + ' split');
    for (const i of idx || []) {
      check(!seen.has(i), 'dataset ' + preset.key + ' puts sample ' + i + ' in two splits');
      seen.add(i);
      total++;
    }
  }
  check(total === ds.count, 'dataset ' + preset.key + ' splits cover ' + total + ' of ' + ds.count + ' samples');
  for (let i = 0; i < ds.x.length; i += Math.max(1, Math.floor(ds.x.length / 97))) {
    check(Number.isFinite(ds.x[i]), 'dataset ' + preset.key + ' has a non-finite feature');
  }
  console.log('  ok  ' + preset.key.padEnd(14) + ' n=' + String(ds.count).padStart(5) + ' x=' + JSON.stringify(ds.xShape) + ' y=' + JSON.stringify(ds.yShape) + ' ' + ds.task);
}

// 2. models -------------------------------------------------------------------
console.log('\nmodel presets');
for (const preset of PRESET_MODEL_LIST) {
  const result = validatePreset(preset);
  check(result.ok, 'model ' + preset.key + ' is invalid: ' + result.problems.join('; '));
  check(!!PRESET_DATASETS[preset.dataset], 'model ' + preset.key + ' points at a missing dataset preset');
  const ds = datasets[preset.dataset];
  if (!ds) continue;
  const outShape = result.info.outputShape.slice(1);
  const expected = expectedOutput(preset.task, ds, outShape);
  check(JSON.stringify(preset.input) === JSON.stringify(ds.xShape),
    'model ' + preset.key + ' input ' + JSON.stringify(preset.input) + ' does not match dataset shape ' + JSON.stringify(ds.xShape));
  check(JSON.stringify(outShape) === JSON.stringify(expected),
    'model ' + preset.key + ' output ' + JSON.stringify(outShape) + ' does not match ' + JSON.stringify(expected));
  check(ds.task === preset.task, 'model ' + preset.key + ' task ' + preset.task + ' does not match dataset task ' + ds.task);
  check(!!preset.budget && !!preset.reference, 'model ' + preset.key + ' is missing budget or reference metadata');
  const ratio = result.info.totalParams / preset.budget.target;
  check(ratio > 0.3 && ratio < 2.5, 'model ' + preset.key + ' is ' + ratio.toFixed(2) + 'x its budget target');
  if (preset.compliance) {
    check(preset.compliance.status === 'regulated', 'model ' + preset.key + ' compliance status is not regulated');
    check((preset.compliance.notice || '').length > 80, 'model ' + preset.key + ' compliance notice is too short to be useful');
  }
}
console.log('  ' + PRESET_MODEL_LIST.length + ' presets checked, ' + problems.length + ' structural problems');

// 3. train every preset for two epochs ---------------------------------------
console.log('\ntwo-epoch training');
for (const preset of PRESET_MODEL_LIST) {
  const ds = datasets[preset.dataset];
  if (!ds) continue;
  const spec = buildPresetGraph(preset);
  const presetTrain = preset.train || {};
  const cfg = normalizeConfig(Object.assign({
    epochs: 2,
    batchSize: 32,
    seed: 17,
    evalBatchSize: 64,
    earlyStopping: { enabled: false },
    checkpointBest: false
  }, {
    optimizer: presetTrain.optimizer || { name: 'adam', lr: 0.006 },
    loss: presetTrain.loss || taskFor(preset.task).defaultLoss,
    huberDelta: presetTrain.huberDelta === undefined ? 1 : presetTrain.huberDelta,
    labelSmoothing: presetTrain.labelSmoothing || 0,
    decisionThreshold: presetTrain.decisionThreshold === undefined ? 0.5 : presetTrain.decisionThreshold,
    confThreshold: presetTrain.confThreshold === undefined ? 0.3 : presetTrain.confThreshold,
    coordScale: presetTrain.coordScale === undefined ? 1 : presetTrain.coordScale,
    temperature: presetTrain.temperature || 0.07
  }), taskFor(preset.task));
  const started = Date.now();
  try {
    const trainer = new Trainer({ spec, dataset: ds, taskKey: preset.task, config: cfg });
    const res = await trainer.run();
    const last = res.history[res.history.length - 1];
    const ok = Number.isFinite(last.trainLoss) && Number.isFinite(last.valLoss);
    check(ok, 'model ' + preset.key + ' produced a non-finite loss');
    const vm = res.metrics.test || res.metrics.val || {};
    const scalars = vm.scalars || {};
    const keys = Object.keys(scalars);
    check(keys.length > 0, 'model ' + preset.key + ' produced no validation metrics');
    let nonFinite = 0;
    for (const k of keys) if (!Number.isFinite(scalars[k])) nonFinite++;
    check(nonFinite === 0, 'model ' + preset.key + ' produced ' + nonFinite + ' non-finite metrics');
    console.log('  ' + (ok ? 'ok  ' : 'FAIL') + ' ' + preset.key.padEnd(14) + ' train=' + last.trainLoss.toFixed(3) + ' val=' + last.valLoss.toFixed(3) + ' ' + (Date.now() - started) + 'ms ' + keys.slice(0, 3).join(' '));
  } catch (e) {
    problems.push('model ' + preset.key + ' failed to train: ' + e.message);
    console.log('  FAIL ' + preset.key + ' ' + e.message);
  }
}

// 4. regulated presets reach the model card --------------------------------
console.log('\nmodel card');
const regulated = PRESET_MODEL_LIST.filter((p) => p.compliance);
check(regulated.length >= 2, 'expected at least two regulated presets, found ' + regulated.length);
for (const preset of PRESET_MODEL_LIST) {
  const spec = buildPresetGraph(preset);
  if (preset.compliance) {
    check(JSON.stringify(spec.meta.compliance) === JSON.stringify(preset.compliance),
      'model ' + preset.key + ' does not carry its compliance notice into the graph metadata');
    check(spec.meta.notes.length === (preset.notes || []).length, 'model ' + preset.key + ' drops its preset notes');
  } else {
    check(spec.meta.compliance === null, 'model ' + preset.key + ' invents a compliance notice');
  }
}
{
  const preset = PRESET_MODEL_LIST.find((p) => p.key === 'ecg');
  const ds = datasets[preset.dataset];
  const spec = buildPresetGraph(preset);
  const cfg = normalizeConfig(Object.assign({ epochs: 1, batchSize: 32, seed: 5, evalBatchSize: 64, earlyStopping: { enabled: false }, checkpointBest: false }, {
    optimizer: preset.train.optimizer, loss: taskFor(preset.task).defaultLoss
  }), taskFor(preset.task));
  const trainer = new Trainer({ spec, dataset: ds, taskKey: preset.task, config: cfg });
  const run = await trainer.run();
  const card = buildModelCard({ spec, dataset: ds, run, taskKey: preset.task });
  const md = cardToMarkdown(card);
  check(!!card.compliance, 'ecg model card has no compliance block');
  check(/software as a medical device/i.test((card.compliance || {}).label || ''), 'ecg compliance label does not name the device regime');
  const regAt = md.indexOf('## Regulatory status');
  const sumAt = md.indexOf('## Summary');
  check(regAt > 0, 'ecg model card markdown has no Regulatory status section');
  check(regAt > 0 && sumAt > regAt, 'Regulatory status must appear before the Summary section');
  console.log('  ok  ecg card carries ' + card.compliance.label + ' before the summary');
}

if (problems.length) {
  console.log('\nFAILURES');
  for (const p of problems) console.log('  - ' + p);
  process.exitCode = 1;
} else {
  console.log('\nPRESET COVERAGE PASS (' + PRESET_DATASET_LIST.length + ' datasets, ' + PRESET_MODEL_LIST.length + ' models)');
}
