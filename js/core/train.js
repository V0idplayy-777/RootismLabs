import { Model, analyze } from './model.js';
import { Optimizer, DEFAULT_OPT, DEFAULT_SCHEDULE, lrAt, Ema } from './optimize.js';
import { Rng, backward, gradNorm, clipGrad, Node, shapeSize } from './tensor.js';
import { TASKS, taskFor, makeBatchNode, subsetView } from './tasks.js';
import { augmentBatch } from '../data/augment.js';

export const DEFAULT_TRAIN_CONFIG = {
  seed: 20260101,
  epochs: 15,
  batchSize: 32,
  loss: null,
  optimizer: Object.assign({}, DEFAULT_OPT),
  schedule: Object.assign({}, DEFAULT_SCHEDULE),
  gradClip: 1,
  labelSmoothing: 0,
  classWeighting: 'none',
  customClassWeights: [],
  posWeight: 1,
  focalGamma: 2,
  temperature: 0.07,
  huberDelta: 1,
  confThreshold: 0.3,
  peak: 1,
  baseline: 'input',
  frameRate: 100,
  ema: false,
  emaDecay: 0.999,
  earlyStopping: { enabled: true, monitor: 'primary', mode: 'auto', patience: 8, minDelta: 0.0005 },
  augmentation: { enabled: false, ops: [] },
  balancedSampler: false,
  accumulationSteps: 1,
  timeBudgetSeconds: 0,
  checkpointBest: true,
  inputNoise: 0,
  evalBatchSize: 64,
  monitorEveryEpoch: true
};

export function normalizeConfig(cfg, task) {
  const base = JSON.parse(JSON.stringify(DEFAULT_TRAIN_CONFIG));
  const merged = deepMerge(base, cfg || {});
  if (!merged.loss) merged.loss = task.defaultLoss;
  merged.optimizer = Object.assign({}, DEFAULT_OPT, merged.optimizer || {});
  merged.schedule = Object.assign({}, DEFAULT_SCHEDULE, merged.schedule || {});
  return merged;
}

function deepMerge(a, b) {
  for (const k of Object.keys(b)) {
    if (b[k] && typeof b[k] === 'object' && !Array.isArray(b[k]) && a[k] && typeof a[k] === 'object') {
      deepMerge(a[k], b[k]);
    } else if (b[k] !== undefined) {
      a[k] = b[k];
    }
  }
  return a;
}

export function computeClassWeights(ds, idx, k) {
  const counts = new Array(k).fill(0);
  const per = shapeSize(ds.yShape || [1]);
  for (const i of idx) {
    const c = Math.round(ds.y[i * per]);
    if (c >= 0 && c < k) counts[c]++;
  }
  const total = counts.reduce((a, b) => a + b, 0);
  const present = counts.filter((c) => c > 0).length || 1;
  return counts.map((c) => (c > 0 ? total / (present * c) : 1));
}

function yieldToUi() {
  return new Promise((r) => setTimeout(r, 0));
}

export class Trainer {
  constructor(opts) {
    this.spec = opts.spec;
    this.dataset = opts.dataset;
    this.task = taskFor(opts.taskKey || (opts.spec && opts.spec.task) || 'multiclass');
    this.config = normalizeConfig(opts.config, this.task);
    this.hooks = opts.hooks || {};
    this.state = 'idle';
    this.epoch = 0;
    this.history = [];
    this.logs = [];
    this.best = null;
    this.cancelled = false;
    this.pausePromise = null;
    this.model = new Model(this.spec, this.config.seed);
    this.rng = new Rng(this.config.seed ^ 0x9e3779b9);
    this.splits = this.dataset.splits;
    this.primaryMetric = opts.primaryMetric || null;
    const monitored = this.primaryMetric || defaultPrimary(this.task);
    if (!this.config.earlyStopping.mode || this.config.earlyStopping.mode === 'auto') {
      this.config.earlyStopping.mode = monitorMode(this.task, monitored);
    }
    this.monitorName = monitored;
  }

  log(level, message) {
    const entry = { t: Date.now(), level, message, epoch: this.epoch };
    this.logs.push(entry);
    if (this.hooks.onLog) this.hooks.onLog(entry);
  }

  pause() {
    if (this.state !== 'running') return;
    this.state = 'paused';
    this.pausePromise = new Promise((r) => {
      this.resumeFn = r;
    });
    this.log('info', 'training paused');
  }

  resume() {
    if (this.state !== 'paused') return;
    this.state = 'running';
    if (this.resumeFn) this.resumeFn();
    this.resumeFn = null;
    this.log('info', 'training resumed');
  }

  stop() {
    this.cancelled = true;
    this.state = 'stopping';
    if (this.resumeFn) this.resumeFn();
    this.log('warn', 'stop requested');
  }

  async maybePause() {
    while (this.state === 'paused' && !this.cancelled) {
      await this.pausePromise;
      await yieldToUi();
    }
  }

  prepare() {
    const cfg = this.config;
    const train = this.splits.train;
    const info = this.model.info;
    const k = info.outputShape ? info.outputShape[info.outputShape.length - 1] : 2;
    if (cfg.classWeighting === 'balanced' && (this.task.key === 'multiclass' || this.task.key === 'seq_tag')) {
      cfg.classWeights = computeClassWeights(this.dataset, train, k);
    } else if (cfg.classWeighting === 'custom' && cfg.customClassWeights && cfg.customClassWeights.length) {
      cfg.classWeights = cfg.customClassWeights.slice();
    } else {
      cfg.classWeights = null;
    }
    return { k };
  }

  batchesPerEpoch() {
    const n = this.splits.train.length;
    const acc = Math.max(1, this.config.accumulationSteps | 0);
    return Math.max(1, Math.ceil(Math.ceil(n / this.config.batchSize) / acc));
  }

  totalSteps() {
    return this.batchesPerEpoch() * this.config.epochs;
  }

  async run() {
    const startedAt = Date.now();
    this.state = 'running';
    const cfg = this.config;
    const { k } = this.prepare();
    this.model.reseed(cfg.seed);
    const params = this.model.trainableParams();
    const opt = new Optimizer(params, cfg.optimizer, cfg.schedule);
    opt.setTotalSteps(this.totalSteps());
    const ema = cfg.ema ? new Ema(params, cfg.emaDecay) : null;
    const trainIdx = this.splits.train;
    const valIdx = this.splits.val;
    const totalSteps = this.totalSteps();
    const bpe = this.batchesPerEpoch();
    let stepsDone = 0;
    let samplesSeen = 0;
    const monitor = resolveMonitor(cfg, this.task, this.primaryMetric, k);

    this.log('info', 'run ' + (this.runId || '') + ' · seed ' + cfg.seed + ' · ' + this.model.paramCount().toLocaleString() + ' params · ' +
      totalSteps + ' optimizer steps');
    this.log('info', 'train ' + trainIdx.length + ' · val ' + valIdx.length + ' · test ' + this.splits.test.length);

    let bestValue = cfg.earlyStopping.mode === 'min' ? Infinity : -Infinity;
    let bestWeights = null;
    let patienceLeft = cfg.earlyStopping.patience;
    let lastTrainTime = 0;

    for (let epoch = 0; epoch < cfg.epochs; epoch++) {
      if (this.cancelled) break;
      this.epoch = epoch;
      const epochStart = performance.now();
      const order = cfg.balancedSampler
        ? balancedOrder(this.dataset, trainIdx, k, new Rng(cfg.seed + epoch))
        : shuffleIdx(trainIdx, new Rng(cfg.seed * 31 + epoch * 7919));
      let trainLoss = 0;
      let trainBatches = 0;
      let gradSum = 0;
      let gradCount = 0;
      let lrLast = cfg.optimizer.lr;
      const acc = Math.max(1, cfg.accumulationSteps | 0);
      this.model.zeroGrad();
      let accSteps = 0;

      for (let s = 0; s < order.length; s += cfg.batchSize) {
        if (this.cancelled) break;
        await this.maybePause();
        const idx = order.subarray(s, Math.min(order.length, s + cfg.batchSize));
        if (!idx.length) continue;
        const loss = this.stepBatch(idx, cfg, k);
        if (!loss || !isFinite(loss.d[0])) {
          this.log('warn', 'non-finite loss at epoch ' + (epoch + 1) + ' step ' + s + ' — skipping batch');
          this.model.zeroGrad();
          continue;
        }
        backward(loss, new Float32Array([1 / acc]));
        trainLoss += loss.d[0];
        trainBatches++;
        accSteps++;
        if (accSteps >= acc) {
          const gn = cfg.gradClip > 0 ? clipGrad(params, cfg.gradClip) : gradNorm(params);
          gradSum += gn;
          gradCount++;
          lrLast = opt.apply();
          this.model.zeroGrad();
          accSteps = 0;
          stepsDone++;
          if (ema) ema.update();
        }
        samplesSeen += idx.length;
        if ((s / cfg.batchSize) % 8 === 0) {
          await yieldToUi();
          if (this.hooks.onProgress) {
            const done = epoch * bpe + Math.floor(s / cfg.batchSize);
            const elapsed = (performance.now() - startedAt) / 1000;
            const pct = totalSteps ? done / totalSteps : 0;
            this.hooks.onProgress({
              phase: 'train',
              epoch: epoch + 1,
              epochs: cfg.epochs,
              step: done,
              totalSteps,
              percent: pct,
              elapsed,
              eta: pct > 0 ? elapsed / pct - elapsed : 0,
              trainLoss: trainLoss / Math.max(1, trainBatches),
              lr: lrLast,
              samplesSeen
            });
          }
        }
      }

      lastTrainTime = (performance.now() - epochStart) / 1000;
      if (ema) ema.apply();
      this.model.syncStates();
      const valMetrics = valIdx.length ? this.evaluate(valIdx, k, cfg) : null;
      const valLoss = valMetrics ? valMetrics.loss : NaN;
      if (ema) ema.restore();

      const monitored = monitor.extract(valMetrics, trainLoss / Math.max(1, trainBatches));
      const row = {
        epoch: epoch + 1,
        trainLoss: trainLoss / Math.max(1, trainBatches),
        valLoss,
        lr: lrLast,
        gradNorm: gradCount ? gradSum / gradCount : 0,
        seconds: lastTrainTime,
        samplesPerSec: lastTrainTime > 0 ? samplesSeen / Math.max(1e-9, ((performance.now() - startedAt) / 1000)) : 0,
        monitored: monitored.name,
        monitoredValue: monitored.value,
        scalars: valMetrics ? valMetrics.scalars : {},
        steps: stepsDone
      };
      this.history.push(row);
      opt.reportPlateau(isFinite(valLoss) ? valLoss : row.trainLoss);

      if (cfg.checkpointBest) {
        const improved = cfg.earlyStopping.mode === 'min'
          ? monitored.value < bestValue - (cfg.earlyStopping.minDelta || 0)
          : monitored.value > bestValue + (cfg.earlyStopping.minDelta || 0);
        if (improved && isFinite(monitored.value)) {
          bestValue = monitored.value;
          bestWeights = this.snapshot(ema);
          patienceLeft = cfg.earlyStopping.patience;
          this.best = { epoch: epoch + 1, value: bestValue, metric: monitored.name };
        } else {
          patienceLeft--;
        }
      }

      this.log('info', 'epoch ' + (epoch + 1) + '/' + cfg.epochs +
        ' · train ' + fmt(row.trainLoss, 4) +
        (isFinite(valLoss) ? ' · val ' + fmt(valLoss, 4) : '') +
        ' · ' + monitored.name + ' ' + fmt(monitored.value, 4) +
        ' · lr ' + row.lr.toExponential(2) +
        ' · ' + row.seconds.toFixed(1) + 's');

      if (this.hooks.onEpoch) {
        await this.hooks.onEpoch(row, valMetrics);
      }

      if (cfg.earlyStopping.enabled && cfg.checkpointBest && patienceLeft <= 0) {
        this.log('warn', 'early stopping at epoch ' + (epoch + 1) + ' — no improvement in ' + cfg.earlyStopping.patience + ' epochs');
        break;
      }
      if (cfg.timeBudgetSeconds > 0 && (performance.now() - startedAt) / 1000 > cfg.timeBudgetSeconds) {
        this.log('warn', 'time budget of ' + cfg.timeBudgetSeconds + 's reached at epoch ' + (epoch + 1));
        break;
      }
    }

    if (bestWeights) {
      this.model.loadWeights(bestWeights.weights);
      this.restoreStates(bestWeights.states);
      this.model.syncStates();
    }

    const finalVal = this.splits.val.length ? this.evaluate(this.splits.val, k, cfg) : null;
    const finalTest = this.splits.test.length ? this.evaluate(this.splits.test, k, cfg) : null;
    const finalTrain = this.evaluate(this.splits.train, k, cfg);
    const endedAt = Date.now();

    const result = {
      state: this.cancelled ? 'stopped' : 'completed',
      history: this.history,
      logs: this.logs,
      best: this.best,
      metrics: { train: finalTrain, val: finalVal, test: finalTest },
      modelSummary: this.model.summary(),
      spec: this.spec,
      config: cfg,
      seed: cfg.seed,
      taskKey: this.task.key,
      monitor: monitor.name,
      epochsRun: this.history.length,
      startedAt,
      endedAt,
      durationMs: endedAt - startedAt,
      samplesSeen,
      steps: stepsDone
    };
    this.state = 'done';
    this.log('info', 'run finished in ' + ((endedAt - startedAt) / 1000).toFixed(1) + 's');
    if (this.hooks.onDone) await this.hooks.onDone(result);
    return result;
  }

  snapshot(ema) {
    if (ema) ema.apply();
    const weights = this.model.weights().filter((w) => this.model.params.has(w.name));
    const states = {};
    for (const [id, st] of this.model.bnStates) {
      states[id] = { runningMean: st.runningMean.slice(), runningVar: st.runningVar.slice() };
    }
    if (ema) ema.restore();
    return { weights, states };
  }

  restoreStates(states) {
    for (const [id, st] of this.model.bnStates) {
      const s = states[id];
      if (!s) continue;
      st.runningMean = s.runningMean.slice();
      st.runningVar = s.runningVar.slice();
      if (st.meanNode) st.meanNode.d.set(Float32Array.from(s.runningMean));
      if (st.varNode) st.varNode.d.set(Float32Array.from(s.runningVar));
    }
  }

  stepBatch(idx, cfg, k) {
    const view = subsetView(this.dataset, idx);
    if (cfg.augmentation && cfg.augmentation.enabled && cfg.augmentation.ops.length) {
      augmentBatch(view, view.localIdx, cfg.augmentation, this.rng);
    }
    if (cfg.inputNoise > 0) {
      const x = view.ds.x;
      for (let i = 0; i < x.length; i++) x[i] += cfg.inputNoise * this.rng.gauss();
    }
    const taskCfg = {
      loss: cfg.loss,
      classWeights: cfg.classWeights ? Float32Array.from(cfg.classWeights) : null,
      labelSmoothing: cfg.labelSmoothing,
      posWeight: cfg.posWeight,
      focalGamma: cfg.focalGamma,
      temperature: cfg.temperature,
      huberDelta: cfg.huberDelta,
      confThreshold: cfg.confThreshold,
      peak: cfg.peak,
      baseline: cfg.baseline,
      frameRate: cfg.frameRate
    };
    return this.task.lossFn(this.model, view.ds, view.localIdx, taskCfg, 'train', this.rng);
  }

  evaluate(idx, k, cfg) {
    const taskCfg = {
      loss: cfg ? cfg.loss : this.config.loss,
      classWeights: null,
      labelSmoothing: 0,
      posWeight: 1,
      focalGamma: 2,
      temperature: this.config.temperature,
      huberDelta: 1,
      confThreshold: this.config.confThreshold,
      peak: this.config.peak,
      baseline: this.config.baseline,
      frameRate: this.config.frameRate
    };
    let lossSum = 0;
    let lossCount = 0;
    const B = Math.max(1, (cfg && cfg.evalBatchSize) || this.config.evalBatchSize || 64);
    for (let s = 0; s < idx.length; s += B) {
      const sub = idx.subarray(s, Math.min(idx.length, s + B));
      try {
        const l = this.task.lossFn(this.model, this.dataset, sub, taskCfg, 'eval', this.rng);
        if (l && isFinite(l.d[0])) {
          lossSum += l.d[0] * sub.length;
          lossCount += sub.length;
        }
      } catch (err) {
        break;
      }
    }
    let m;
    try {
      m = this.task.metrics(this.model, this.dataset, idx, taskCfg);
    } catch (err) {
      m = { scalars: {}, error: String(err && err.message) };
    }
    m.loss = lossCount ? lossSum / lossCount : NaN;
    return m;
  }
}

function resolveMonitor(cfg, task, primaryMetric, k) {
  const preferred = primaryMetric || defaultPrimary(task);
  const name = 'val_' + preferred;
  return {
    name: preferred,
    extract(valMetrics, trainLoss) {
      if (!valMetrics || !valMetrics.scalars) return { name: 'train_loss', value: trainLoss };
      if (valMetrics.scalars[preferred] !== undefined) return { name: preferred, value: valMetrics.scalars[preferred] };
      const keys = Object.keys(valMetrics.scalars);
      for (const key of keys) {
        if (isFinite(valMetrics.scalars[key])) return { name: key, value: valMetrics.scalars[key] };
      }
      return { name: 'val_loss', value: valMetrics.loss };
    }
  };
}

export function defaultPrimary(task) {
  switch (task.key) {
    case 'multiclass': return 'accuracy';
    case 'binary': return 'auc_roc';
    case 'multilabel': return 'macro_f1';
    case 'regression': return 'mae';
    case 'seq_tag': return 'entity_f1';
    case 'lm': return 'perplexity';
    case 'image_map': return 'psnr';
    case 'mask': return 'iou';
    case 'frame_binary': return 'auc_roc';
    case 'detection': return 'mAP50';
    case 'embedding': return 'pair_auc';
    case 'anomaly': return 'auroc';
    case 'spectral_mask': return 'si_sdr_improvement';
    case 'pitch_class': return 'raw_pitch_accuracy';
    case 'onset': return 'onset_f1';
    case 'forecast': return 'mae';
    case 'coordinate_field': return 'psnr';
    default: return 'accuracy';
  }
}

export function monitorMode(task, metric) {
  const lower = ['val_loss', 'train_loss', 'mae', 'rmse', 'mse', 'perplexity', 'nll', 'log_loss', 'ece', 'brier',
    'error_rate', 'frr', 'far', 'max_error', 'mape', 'mean_abs_cent_error', 'median_abs_cent_error', 'octave_error_rate',
    'bits_per_symbol', 'bytes_per_symbol', 'alignment', 'uniformity', 'recon_mse_normal',
    'tempo_mae_bpm', 'naive_mae', 'mask_mae', 'si_sdr_input'];
  if (lower.includes(metric)) return 'min';
  return 'max';
}

function shuffleIdx(idx, rng) {
  const a = Int32Array.from(idx);
  for (let i = a.length - 1; i > 0; i--) {
    const j = rng.int(i + 1);
    const t = a[i];
    a[i] = a[j];
    a[j] = t;
  }
  return a;
}

function balancedOrder(ds, idx, k, rng) {
  const buckets = [];
  for (let c = 0; c < k; c++) buckets.push([]);
  const per = shapeSize(ds.yShape || [1]);
  for (const i of idx) {
    const c = Math.round(ds.y[i * per]);
    if (c >= 0 && c < k) buckets[c].push(i);
  }
  const nonEmpty = buckets.filter((b) => b.length);
  if (!nonEmpty.length) return shuffleIdx(idx, rng);
  const target = idx.length;
  const out = [];
  let guard = 0;
  while (out.length < target && guard < target * 4) {
    for (const b of nonEmpty) {
      if (out.length >= target) break;
      out.push(b[rng.int(b.length)]);
      guard++;
    }
  }
  return Int32Array.from(out);
}

function fmt(v, d) {
  if (v === undefined || v === null || !isFinite(v)) return '—';
  return Number(v).toFixed(d === undefined ? 4 : d);
}

export function summarizeRun(run) {
  const info = analyze(run.spec);
  return {
    params: info.totalParams,
    macs: info.macs,
    epochs: run.history.length,
    seconds: run.durationMs / 1000
  };
}

export { Node };
