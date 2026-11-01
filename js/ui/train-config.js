import { el, clear, attachTooltip } from './dom.js';
import { lineChart } from './charts.js';
import { OPTIMIZERS, SCHEDULES, DEFAULT_OPT, DEFAULT_SCHEDULE, lrAt } from '../core/optimize.js';
import { DEFAULT_TRAIN_CONFIG } from '../core/train.js';
import { TASKS, taskFor } from '../core/tasks.js';
import { metricsForTask, metricMeta } from '../presets/metrics.js';
import { AUG_OPS, opsForModality } from '../data/augment.js';

export function defaultConfig(taskKey) {
  const task = taskFor(taskKey || 'multiclass');
  const cfg = JSON.parse(JSON.stringify(DEFAULT_TRAIN_CONFIG));
  cfg.loss = task.defaultLoss;
  return cfg;
}

export function mergeConfig(existing) {
  const base = JSON.parse(JSON.stringify(DEFAULT_TRAIN_CONFIG));
  if (!existing) return base;
  for (const key of Object.keys(existing)) {
    const value = existing[key];
    if (value === undefined) continue;
    if (value && typeof value === 'object' && !Array.isArray(value) && base[key] && typeof base[key] === 'object') Object.assign(base[key], value);
    else base[key] = value;
  }
  return base;
}

function numberRow(label, value, opts, onChange) {
  const options = opts || {};
  return el('label', { class: 'field' }, [
    el('span', { class: 'lbl', text: label }),
    el('input', {
      type: 'number', value: value, min: options.min, max: options.max, step: options.step === undefined ? 'any' : options.step,
      onchange: (event) => {
        const parsed = parseFloat(event.target.value);
        if (!Number.isFinite(parsed)) return;
        onChange(options.min !== undefined ? Math.max(options.min, parsed) : parsed);
      }
    }),
    options.hint ? el('span', { class: 'tiny faint', text: options.hint }) : null
  ]);
}

function selectRow(label, value, choices, onChange, hint) {
  return el('label', { class: 'field' }, [
    el('span', { class: 'lbl', text: label }),
    el('select', { onchange: (event) => onChange(event.target.value) }, choices.map((c) => {
      const item = typeof c === 'string' ? { value: c, label: c } : c;
      return el('option', { value: String(item.value), text: item.label, selected: String(item.value) === String(value) });
    })),
    hint ? el('span', { class: 'tiny faint', text: hint }) : null
  ]);
}

function boolRow(label, value, onChange, hint) {
  return el('label', { class: 'inline', style: { width: '100%' } }, [
    el('input', { type: 'checkbox', checked: !!value, onchange: (event) => onChange(event.target.checked) }),
    el('span', { text: label }),
    hint ? el('span', { class: 'tiny faint', text: ' — ' + hint }) : null
  ]);
}

function block(title, children, note) {
  return el('div', { class: 'opt-block' }, [el('h4', { text: title })].concat(children).concat(note ? [el('div', { class: 'tiny faint mt4', text: note })] : []));
}

export function configView(ctx, config, options) {
  const opts = options || {};
  const cfg = config;
  const task = taskFor(opts.taskKey || 'multiclass');
  const dataset = opts.dataset;
  const host = el('div', { class: 'col gap8' });
  const onChange = opts.onChange || (() => {});

  const trainN = dataset && dataset.splits && dataset.splits.train ? dataset.splits.train.length : 0;
  const stepsPerEpoch = Math.max(1, Math.ceil(trainN / Math.max(1, cfg.batchSize)));
  const totalSteps = stepsPerEpoch * Math.max(1, cfg.epochs);

  const lossChoices = (task.lossFns || []).map((name) => ({ value: name, label: name }));
  const metricChoices = metricOptions(task);

  host.appendChild(block('Run', [
    el('label', { class: 'field' }, [
      el('span', { class: 'lbl', text: 'run name' }),
      el('input', { type: 'text', value: opts.name || '', oninput: (event) => { opts.name = event.target.value; cfg.runName = event.target.value; onChange(); } })
    ]),
    el('div', { class: 'config-grid' }, [
      numberRow('epochs', cfg.epochs, { min: 1, max: 400, step: 1 }, (v) => { cfg.epochs = v; onChange(); renderLrPreview(); }),
      numberRow('batch size', cfg.batchSize, { min: 1, max: 1024, step: 1 }, (v) => { cfg.batchSize = v; onChange(); renderLrPreview(); }),
      numberRow('seed', cfg.seed, { min: 0, step: 1 }, (v) => { cfg.seed = Math.round(v); onChange(); }),
      numberRow('eval batch', cfg.evalBatchSize, { min: 1, max: 1024, step: 1 }, (v) => { cfg.evalBatchSize = Math.round(v); onChange(); }),
      numberRow('time budget (s)', cfg.timeBudgetSeconds, { min: 0, step: 5 }, (v) => { cfg.timeBudgetSeconds = v; onChange(); }),
      numberRow('grad accumulation', cfg.accumulationSteps, { min: 1, max: 64, step: 1 }, (v) => { cfg.accumulationSteps = Math.round(v); onChange(); })
    ]),
    el('div', { class: 'tiny faint', text: dataset ? 'Training split has ' + trainN + ' samples → ' + stepsPerEpoch + ' steps per epoch, ' + totalSteps.toLocaleString() + ' total.' : 'Load a dataset to see the step count.' })
  ]));

  host.appendChild(block('Objective', [
    selectRow('task', task.key, Object.keys(TASKS).map((k) => ({ value: k, label: TASKS[k].label + ' (' + k + ')' })), (v) => opts.onTask && opts.onTask(v)),
    selectRow('loss', cfg.loss, lossChoices, (v) => { cfg.loss = v; onChange(); renderLossParams(); }),
    selectRow('primary metric', opts.primaryMetric || metricChoices[0].value, metricChoices, (v) => { opts.primaryMetric = v; onChange(); }, 'used for early stopping, checkpointing and the headline number'),
    el('div', { class: 'config-grid', id: 'lossParams' })
  ]));

  const lossParamsHost = host.querySelector('#lossParams');
  function renderLossParams() {
    const host2 = lossParamsHost;
    if (!host2) return;
    clear(host2);
    const loss = cfg.loss;
    if (loss === 'cross_entropy') {
      host2.appendChild(numberRow('label smoothing', cfg.labelSmoothing, { min: 0, max: 0.4, step: 0.01 }, (v) => { cfg.labelSmoothing = v; onChange(); }));
    }
    if (loss === 'focal' || loss === 'focal_bce') {
      host2.appendChild(numberRow('focal gamma', cfg.focalGamma, { min: 0, max: 6, step: 0.1 }, (v) => { cfg.focalGamma = v; onChange(); }));
    }
    if (loss === 'bce' || loss === 'focal_bce' || loss === 'bce_dice') {
      host2.appendChild(numberRow('pos weight', cfg.posWeight, { min: 0.05, max: 50, step: 0.05 }, (v) => { cfg.posWeight = v; onChange(); }));
    }
    if (loss === 'huber') {
      host2.appendChild(numberRow('huber delta', cfg.huberDelta, { min: 0.01, max: 10, step: 0.01 }, (v) => { cfg.huberDelta = v; onChange(); }));
    }
    if (loss === 'infonce' || loss === 'supcon' || loss === 'triplet' || loss === 'cosine') {
      host2.appendChild(numberRow('temperature', cfg.temperature, { min: 0.01, max: 1, step: 0.01 }, (v) => { cfg.temperature = v; onChange(); }));
    }
    if (loss === 'detection') {
      host2.appendChild(numberRow('conf threshold', cfg.confThreshold, { min: 0.01, max: 0.95, step: 0.01 }, (v) => { cfg.confThreshold = v; onChange(); }));
    }
    if (task.key === 'image_map' || task.key === 'coordinate_field' || task.key === 'spectral_mask') {
      host2.appendChild(selectRow('baseline', cfg.baseline, [{ value: 'input', label: 'input (identity)' }, { value: 'mean', label: 'dataset mean' }, { value: 'zero', label: 'zeros' }], (v) => { cfg.baseline = v; onChange(); }, 'the trivial predictor every gain is measured against'));
    }
    if (task.key === 'spectral_mask') {
      host2.appendChild(numberRow('peak norm', cfg.peak, { min: 0.1, max: 10, step: 0.1 }, (v) => { cfg.peak = v; onChange(); }));
      host2.appendChild(numberRow('frame rate', cfg.frameRate, { min: 1, max: 1000, step: 1 }, (v) => { cfg.frameRate = v; onChange(); }));
    }
    if (!host2.children.length) host2.appendChild(el('div', { class: 'tiny faint', text: 'No extra parameters for this loss.' }));
  }
  renderLossParams();

  host.appendChild(block('Optimizer', [
    selectRow('kind', cfg.optimizer.name, Object.keys(OPTIMIZERS).map((k) => ({ value: k, label: OPTIMIZERS[k].label + ' (' + k + ')' })), (v) => { cfg.optimizer.name = v; onChange(); renderOptFields(); }),
    el('div', { class: 'config-grid', id: 'optFields' })
  ]));
  const optFieldsHost = host.querySelector('#optFields');
  function renderOptFields() {
    clear(optFieldsHost);
    const spec = OPTIMIZERS[cfg.optimizer.name] || OPTIMIZERS.adam;
    for (const field of spec.fields) {
      const isInt = field === 'stepSize';
      optFieldsHost.appendChild(numberRow(field, cfg.optimizer[field] === undefined ? DEFAULT_OPT[field] : cfg.optimizer[field], {
        min: field === 'lr' ? 0 : undefined,
        step: field === 'lr' ? 0.0001 : field === 'eps' ? 1e-9 : isInt ? 1 : 0.01
      }, (v) => { cfg.optimizer[field] = v; onChange(); renderLrPreview(); }));
    }
    if (spec.fields.indexOf('nesterov') < 0 && (cfg.optimizer.name === 'sgd' || cfg.optimizer.name === 'momentum')) {
      optFieldsHost.appendChild(boolRow('nesterov', cfg.optimizer.nesterov, (v) => { cfg.optimizer.nesterov = v; onChange(); }));
    }
  }
  renderOptFields();

  host.appendChild(block('Learning rate schedule', [
    selectRow('schedule', cfg.schedule.name, Object.keys(SCHEDULES).map((k) => ({ value: k, label: SCHEDULES[k].label + ' (' + k + ')' })), (v) => { cfg.schedule.name = v; onChange(); renderLrPreview(); }),
    el('div', { class: 'config-grid', id: 'schedFields' }),
    el('canvas', { id: 'lrPreview', style: { width: '100%', height: '74px' } })
  ], 'Preview uses the current epoch and batch size to estimate total steps.'));
  const schedHost = host.querySelector('#schedFields');
  const SCHED_FIELDS = ['warmupSteps', 'minLrRatio', 'decayRate', 'stepSize', 'restartPeriod', 'maxLrMultiplier', 'plateauFactor', 'plateauPatience'];
  for (const field of SCHED_FIELDS) {
    schedHost.appendChild(numberRow(field, cfg.schedule[field], { min: 0, step: field === 'minLrRatio' || field === 'decayRate' || field === 'plateauFactor' ? 0.01 : 1 }, (v) => { cfg.schedule[field] = v; onChange(); renderLrPreview(); }));
  }
  function renderLrPreview() {
    const canvas = host.querySelector('#lrPreview');
    if (!canvas) return;
    const trainCount = dataset && dataset.splits && dataset.splits.train ? dataset.splits.train.length : 256;
    const spe = Math.max(1, Math.ceil(trainCount / Math.max(1, cfg.batchSize)));
    const total = Math.max(2, spe * Math.max(1, cfg.epochs));
    const values = [];
    const stride = Math.max(1, Math.floor(total / 160));
    for (let step = 0; step <= total; step += stride) values.push(lrAt(cfg.optimizer, cfg.schedule, step, total));
    lineChart(canvas, [{ label: 'lr', y: values, color: '#e0a33e' }], {
      height: 74, legend: false, hover: false, background: '#14171a',
      pad: { top: 6, right: 6, bottom: 14, left: 42 }, yTitle: '', xTitle: 'steps (' + total.toLocaleString() + ')'
    });
  }
  renderLrPreview();

  host.appendChild(block('Regularisation', [
    el('div', { class: 'config-grid' }, [
      numberRow('grad clip norm', cfg.gradClip, { min: 0, max: 100, step: 0.1 }, (v) => { cfg.gradClip = v; onChange(); }),
      numberRow('input noise σ', cfg.inputNoise, { min: 0, max: 1, step: 0.005 }, (v) => { cfg.inputNoise = v; onChange(); }),
      numberRow('ema decay', cfg.emaDecay, { min: 0.9, max: 0.9999, step: 0.0001 }, (v) => { cfg.emaDecay = v; onChange(); })
    ]),
    boolRow('weight EMA', cfg.ema, (v) => { cfg.ema = v; onChange(); }, 'evaluate and export the averaged weights'),
    boolRow('balanced sampler', cfg.balancedSampler, (v) => { cfg.balancedSampler = v; onChange(); }, 'oversample rare classes each epoch'),
    selectRow('class weighting', cfg.classWeighting, [
      { value: 'none', label: 'none' },
      { value: 'inverse', label: 'inverse frequency' },
      { value: 'sqrt', label: 'inverse square root (softer)' }
    ], (v) => { cfg.classWeighting = v; onChange(); renderCustomWeights(); }, 'applied inside the loss'),
    el('div', { class: 'config-grid', id: 'customWeights' })
  ]));
  const weightsHost = host.querySelector('#customWeights');
  function renderCustomWeights() {
    clear(weightsHost);
    if (cfg.classWeighting !== 'custom') return;
    const labels = dataset && dataset.labels && dataset.labels.length ? dataset.labels : [];
    if (!labels.length) { weightsHost.appendChild(el('div', { class: 'tiny faint', text: 'No class labels on this dataset.' })); return; }
    if (!cfg.customClassWeights || cfg.customClassWeights.length !== labels.length) cfg.customClassWeights = labels.map(() => 1);
    labels.forEach((label, i) => {
      weightsHost.appendChild(numberRow(label, cfg.customClassWeights[i], { min: 0, max: 100, step: 0.05 }, (v) => { cfg.customClassWeights[i] = v; onChange(); }));
    });
  }
  renderCustomWeights();

  const augOps = dataset ? opsForModality(dataset.modality) : [];
  const augCfg = dataset && dataset.augmentation ? dataset.augmentation : { enabled: false, applyProb: 0.8, ops: [] };
  const augBlock = block('Augmentation', [
    boolRow('augment training batches', augCfg.enabled, (v) => {
      if (!dataset) { ctx.warn('Load a dataset first.'); return; }
      augCfg.enabled = v;
      cfg.augmentation = { enabled: v, ops: augCfg.ops.map((o) => Object.assign({}, o)) };
      dataset.augmentation = augCfg;
      ctx.touchDataset();
      onChange();
    }, 'validation and test splits are never augmented'),
    el('div', { class: 'tiny faint', text: augCfg.ops && augCfg.ops.length ? 'Operators: ' + augCfg.ops.map((o) => { const meta = AUG_OPS.find((a) => a.id === o.id); return (meta ? meta.label : o.id) + ' (' + o.strength + ', p' + o.prob + ')'; }).join(', ') : 'No operators selected.' }),
    el('button', { class: 'sm ghost', text: 'Configure on the Data panel', onclick: () => ctx.go('data') })
  ]);
  host.appendChild(augBlock);

  host.appendChild(block('Stopping and checkpoints', [
    boolRow('early stopping', cfg.earlyStopping.enabled, (v) => { cfg.earlyStopping.enabled = v; onChange(); renderEsFields(); }),
    el('div', { class: 'config-grid', id: 'esFields' }),
    boolRow('restore best checkpoint', cfg.checkpointBest, (v) => { cfg.checkpointBest = v; onChange(); }, 'reload the weights from the best monitored epoch before evaluating'),
    boolRow('evaluate every epoch', cfg.monitorEveryEpoch, (v) => { cfg.monitorEveryEpoch = v; onChange(); })
  ]));
  const esHost = host.querySelector('#esFields');
  function renderEsFields() {
    clear(esHost);
    if (!cfg.earlyStopping.enabled) return;
    esHost.appendChild(numberRow('patience', cfg.earlyStopping.patience, { min: 1, max: 200, step: 1 }, (v) => { cfg.earlyStopping.patience = Math.round(v); onChange(); }));
    esHost.appendChild(numberRow('min delta', cfg.earlyStopping.minDelta, { min: 0, max: 1, step: 0.0001 }, (v) => { cfg.earlyStopping.minDelta = v; onChange(); }));
    esHost.appendChild(selectRow('direction', cfg.earlyStopping.mode, [
      { value: 'auto', label: 'auto (from the metric)' },
      { value: 'max', label: 'maximise' },
      { value: 'min', label: 'minimise' }
    ], (v) => { cfg.earlyStopping.mode = v; onChange(); }));
  }
  renderEsFields();

  return host;
}

function metricOptions(task) {
  const groups = metricsForTask(task.key);
  const out = [];
  for (const group of groups) {
    for (const key of group.metrics) {
      const meta = metricMeta(key);
      out.push({ value: key, label: (meta ? meta.label : key) + ' · ' + group.title });
    }
  }
  return out.length ? out : [{ value: 'loss', label: 'loss' }];
}

export function configSummary(cfg, taskKey, primaryMetric) {
  const task = taskFor(taskKey);
  return {
    task: task.label,
    loss: cfg.loss,
    primary: primaryMetric || metricOptions(task)[0].value,
    optimizer: cfg.optimizer.name + ' lr ' + cfg.optimizer.lr,
    schedule: cfg.schedule.name,
    epochs: cfg.epochs,
    batch: cfg.batchSize,
    seed: cfg.seed,
    gradClip: cfg.gradClip,
    ema: cfg.ema ? cfg.emaDecay : 'off',
    classWeighting: cfg.classWeighting,
    augmentation: cfg.augmentation && cfg.augmentation.enabled ? (cfg.augmentation.ops || []).length + ' ops' : 'off',
    earlyStopping: cfg.earlyStopping.enabled ? cfg.earlyStopping.patience + ' epochs' : 'off'
  };
}

export { DEFAULT_SCHEDULE };
