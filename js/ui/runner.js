import { auditDataset, auditModelDataset } from '../data/preflight.js';
import { Trainer, normalizeConfig, defaultPrimary } from '../core/train.js';
import { analyze } from '../core/model.js';
import { headlineFor } from '../presets/metrics.js';
import { sha256Text, shortHash } from '../core/hash.js';

export function architectureHash(spec) {
  return shortHash(sha256Text(JSON.stringify(spec)), 12);
}
import { newId } from './state.js';

export class RunSession {
  constructor(ctx, options) {
    this.ctx = ctx;
    this.spec = options.spec;
    this.dataset = options.dataset;
    this.taskKey = options.taskKey;
    this.config = normalizeConfig(options.config, { key: options.taskKey, defaultLoss: (options.config && options.config.loss) || null });
    this.name = options.name || 'run';
    this.primaryMetric = options.primaryMetric || null;
    this.trainer = null;
    this.live = {
      phase: 'idle',
      epoch: 0,
      epochs: this.config.epochs,
      percent: 0,
      eta: 0,
      elapsed: 0,
      lr: this.config.optimizer.lr,
      trainLoss: NaN,
      throughput: 0,
      samplesSeen: 0,
      steps: 0,
      history: [],
      lastVal: null,
      weightNorms: null,
      gradNorms: null,
      predictions: []
    };
    this.paused = false;
    this.stopped = false;
    this.result = null;
  }

  async start() {
    const ctx = this.ctx;
    const live = this.live;
    live.phase = 'starting';
    ctx.state.training = this;
    ctx.notify('train');
    const info = analyze(this.spec);
    ctx.info('Starting "' + this.name + '" · ' + info.totalParams.toLocaleString() + ' params · ' + this.taskKey + ' · seed ' + this.config.seed + ' · ' + this.config.epochs + ' epochs · batch ' + this.config.batchSize);
    if (!this.dataset.splits || !this.dataset.splits.train || !this.dataset.splits.train.length) {
      ctx.error('The dataset has no training split. Fix the splits on the Data panel first.');
      live.phase = 'failed';
      ctx.state.training = null;
      ctx.notify('train');
      return null;
    }
    if (!this.dataset.splits.val || !this.dataset.splits.val.length) {
      ctx.warn('No validation split — early stopping and model selection will be blind. Consider a 70/15/15 split.');
    }
    if (!this.dataset.splits.test || !this.dataset.splits.test.length) {
      ctx.warn('No held-out test split — reported numbers will not be an honest generalisation estimate.');
    }
    const audit = auditDataset(this.dataset);
    const compatibility = auditModelDataset(this.spec, this.dataset, this.taskKey, info);
    for (const warning of audit.warnings.concat(compatibility.warnings)) ctx.warn(warning);
    const errors = audit.errors.concat(compatibility.errors);
    if (errors.length) {
      for (const error of errors) ctx.error(error);
      live.phase = 'failed';
      ctx.state.training = null;
      ctx.notify('train');
      return null;
    }
    this.trainer = new Trainer({
      spec: this.spec,
      dataset: this.dataset,
      taskKey: this.taskKey,
      config: this.config,
      primaryMetric: this.primaryMetric,
      hooks: {
        onLog: (entry) => ctx.log(entry.level, entry.message),
        onProgress: (p) => this._progress(p),
        onEpoch: async (row, valMetrics) => this._epoch(row, valMetrics),
        onDone: async () => {}
      }
    });
    live.phase = 'running';
    ctx.notify('train');
    try {
      const result = await this.trainer.run();
      this.result = result;
      live.phase = this.stopped ? 'stopped' : 'done';
      ctx.state.training = null;
      await ctx.finishRun(this, result);
      ctx.notify('train');
      return result;
    } catch (err) {
      live.phase = 'failed';
      ctx.state.training = null;
      ctx.error('Training crashed: ' + (err && err.message ? err.message : String(err)));
      ctx.notify('train');
      return null;
    }
  }

  _progress(p) {
    const live = this.live;
    live.epoch = p.epoch;
    live.epochs = p.epochs;
    live.percent = p.percent;
    live.eta = p.eta;
    live.elapsed = p.elapsed;
    live.lr = p.lr;
    live.trainLoss = p.trainLoss;
    live.samplesSeen = p.samplesSeen;
    live.steps = p.step;
    live.totalSteps = p.totalSteps;
    live.throughput = p.elapsed > 0 ? p.samplesSeen / p.elapsed : 0;
    this.ctx.notifyThrottled('progress');
  }

  _epoch(row, valMetrics) {
    const live = this.live;
    live.history = this.trainer.history.slice();
    live.lastVal = valMetrics || null;
    live.epoch = row.epoch;
    live.trainLoss = row.trainLoss;
    live.lr = row.lr;
    live.gradNorm = row.gradNorm;
    live.seconds = row.seconds;
    live.throughput = row.samplesPerSec;
    live.predictions = extractPredictions(valMetrics, this.dataset);
    live.weightNorms = this._weightNorms();
    live.gradNorms = this._gradNorms();
    this.ctx.notify('epoch');
    return Promise.resolve();
  }

  _weightNorms() {
    try {
      const params = this.trainer.model.params;
      const out = [];
      for (const entry of params) {
        const name = entry[0];
        const node = entry[1];
        if (!node || !node.d || !node.d.length) continue;
        let sum = 0;
        let max = 0;
        for (let i = 0; i < node.d.length; i++) {
          const v = node.d[i];
          sum += v * v;
          const a = Math.abs(v);
          if (a > max) max = a;
        }
        out.push({ name, l2: Math.sqrt(sum), max, n: node.d.length, rms: Math.sqrt(sum / node.d.length) });
      }
      out.sort((a, b) => b.l2 - a.l2);
      return out;
    } catch (e) {
      return null;
    }
  }

  _gradNorms() {
    try {
      const params = this.trainer.model.params;
      const byNode = new Map();
      for (const entry of params) {
        const node = entry[1];
        if (!node || !node.g) continue;
        const id = entry[0].split('.')[0];
        let acc = byNode.get(id);
        if (!acc) { acc = { name: id, sum: 0, n: 0 }; byNode.set(id, acc); }
        for (let i = 0; i < node.g.length; i++) { acc.sum += node.g[i] * node.g[i]; acc.n++; }
      }
      const out = [];
      for (const acc of byNode.values()) out.push({ name: acc.name, l2: Math.sqrt(acc.sum), n: acc.n });
      out.sort((a, b) => b.l2 - a.l2);
      return out.length ? out : null;
    } catch (e) {
      return null;
    }
  }

  pause() {
    if (!this.trainer || this.paused) return;
    this.paused = true;
    this.live.phase = 'paused';
    this.trainer.pause();
    this.ctx.notify('train');
  }

  resume() {
    if (!this.trainer || !this.paused) return;
    this.paused = false;
    this.live.phase = 'running';
    this.trainer.resume();
    this.ctx.notify('train');
  }

  stop() {
    if (!this.trainer) return;
    this.stopped = true;
    if (this.paused) this.resume();
    this.trainer.stop();
    this.live.phase = 'stopping';
    this.ctx.notify('train');
  }

  buildRunRecord(result) {
    const task = this.taskKey;
    const headlineMetric = headlineFor(task);
    const testScalars = result.metrics && result.metrics.test ? result.metrics.test.scalars : null;
    const valScalars = result.metrics && result.metrics.val ? result.metrics.val.scalars : null;
    const sourceScalars = testScalars || valScalars || {};
    const headline = {
      metric: headlineMetric,
      split: testScalars ? 'test' : valScalars ? 'val' : null,
      value: sourceScalars[headlineMetric] === undefined ? null : sourceScalars[headlineMetric]
    };
    const info = analyze(this.spec);
    return {
      id: newId('run'),
      name: this.name,
      taskKey: task,
      modelName: this.spec.name || 'Untitled model',
      datasetName: this.dataset.name || 'Untitled dataset',
      datasetId: this.dataset.id,
      graphId: this.ctx.state.graphId,
      spec: JSON.parse(JSON.stringify(this.spec)),
      config: JSON.parse(JSON.stringify(this.config)),
      seed: this.config.seed,
      history: result.history,
      logs: result.logs,
      best: result.best,
      metrics: result.metrics,
      modelSummary: result.modelSummary,
      monitor: result.monitor,
      headline,
      epochsRun: result.epochsRun,
      durationMs: result.durationMs,
      startedAt: result.startedAt,
      endedAt: result.endedAt,
      samplesSeen: result.samplesSeen,
      steps: result.steps,
      state: result.state,
      totalParams: info.totalParams,
      macs: info.macs,
      weights: this.trainer && this.trainer.model ? this.trainer.model.weights() : [],
      datasetHash: this.dataset.digest ? this.dataset.digest.contentHash : null,
      architectureHash: architectureHash(this.spec),
      createdAt: Date.now()
    };
  }
}

function extractPredictions(valMetrics, dataset) {
  if (!valMetrics || !Array.isArray(valMetrics.errors)) return [];
  return valMetrics.errors.slice(0, 24).map((e) => {
    const index = e.index;
    const meta = dataset.meta && dataset.meta[index] ? dataset.meta[index] : null;
    const text = meta ? (meta.text || meta.preview || null) : null;
    return {
      index,
      text,
      meta,
      predicted: e.predicted !== undefined ? e.predicted : e.pred,
      actual: e.actual !== undefined ? e.actual : e.true,
      confidence: e.confidence !== undefined ? e.confidence : e.score,
      psnr: e.psnr,
      ssim: e.ssim,
      mae: e.mae
    };
  });
}

export function runDurationLabel(ms) {
  const seconds = ms / 1000;
  if (seconds < 60) return seconds.toFixed(1) + 's';
  const minutes = Math.floor(seconds / 60);
  return minutes + 'm ' + Math.round(seconds % 60) + 's';
}
