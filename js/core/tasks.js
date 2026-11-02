import { Node, shapeSize, backward, Rng, addInto, ensureGrad } from './tensor.js';
import * as A from './autograd.js';
import * as M from './metrics.js';

export function sliceRows(ds, field, idx, shape) {
  const per = shapeSize(shape);
  const src = ds[field];
  const out = new Float32Array(idx.length * per);
  for (let i = 0; i < idx.length; i++) {
    const s = idx[i] * per;
    out.set(src.subarray(s, s + per), i * per);
  }
  return out;
}

export function makeBatchNode(ds, idx, field, shape) {
  return new Node(sliceRows(ds, field || 'x', idx, shape || ds.xShape), [idx.length].concat(shape || ds.xShape));
}

export function subsetView(ds, idx) {
  const xPer = shapeSize(ds.xShape);
  const x = new Float32Array(idx.length * xPer);
  for (let i = 0; i < idx.length; i++) {
    x.set(ds.x.subarray(idx[i] * xPer, (idx[i] + 1) * xPer), i * xPer);
  }
  const view = Object.assign({}, ds);
  view.x = x;
  if (ds.y) {
    const yPer = shapeSize(ds.yShape || [1]);
    const y = new Float32Array(idx.length * yPer);
    for (let i = 0; i < idx.length; i++) {
      y.set(ds.y.subarray(idx[i] * yPer, (idx[i] + 1) * yPer), i * yPer);
    }
    view.y = y;
  }
  if (ds.x2) {
    const per2 = shapeSize(ds.xShape);
    const x2 = new Float32Array(idx.length * per2);
    for (let i = 0; i < idx.length; i++) {
      x2.set(ds.x2.subarray(idx[i] * per2, (idx[i] + 1) * per2), i * per2);
    }
    view.x2 = x2;
  }
  if (ds.lengths) {
    const lengths = new Int32Array(idx.length);
    for (let i = 0; i < idx.length; i++) lengths[i] = ds.lengths[idx[i]];
    view.lengths = lengths;
  }
  if (ds.groups) {
    const groups = new Int32Array(idx.length);
    for (let i = 0; i < idx.length; i++) groups[i] = ds.groups[idx[i]];
    view.groups = groups;
  }
  if (ds.boxes) {
    view.boxes = Array.from(idx, (i) => ds.boxes[i]);
  }
  if (ds.meta) {
    view.meta = Array.from(idx, (i) => ds.meta[i]);
  }
  if (ds.wave) {
    view.wave = Array.from(idx, (i) => ds.wave[i]);
  }
  if (ds.images) {
    view.images = Array.from(idx, (i) => ds.images[i]);
  }
  view.localIdx = Int32Array.from(idx.keys());
  view.sourceIndex = idx;
  return { ds: view, localIdx: view.localIdx };
}

function flatTargets(ds, idx) {
  const per = shapeSize(ds.yShape || [1]);
  const out = new Float32Array(idx.length * per);
  for (let i = 0; i < idx.length; i++) {
    const s = idx[i] * per;
    out.set(ds.y.subarray(s, s + per), i * per);
  }
  return out;
}

function labelInts(ds, idx) {
  const per = shapeSize(ds.yShape || [1]);
  const out = new Int32Array(idx.length * per);
  for (let i = 0; i < idx.length; i++) {
    for (let j = 0; j < per; j++) out[i * per + j] = Math.round(ds.y[idx[i] * per + j]);
  }
  return out;
}

export function softmaxRows(flat, rows, k) {
  const out = new Float32Array(flat.length);
  for (let i = 0; i < rows; i++) {
    let max = -Infinity;
    for (let c = 0; c < k; c++) max = Math.max(max, flat[i * k + c]);
    let sum = 0;
    for (let c = 0; c < k; c++) {
      const e = Math.exp(flat[i * k + c] - max);
      out[i * k + c] = e;
      sum += e;
    }
    for (let c = 0; c < k; c++) out[i * k + c] /= sum;
  }
  return out;
}

function softmaxProbs(logits, k) {
  const n = logits.d.length / k;
  const out = new Float32Array(logits.d.length);
  for (let i = 0; i < n; i++) {
    let max = -Infinity;
    for (let c = 0; c < k; c++) max = Math.max(max, logits.d[i * k + c]);
    let sum = 0;
    for (let c = 0; c < k; c++) {
      const e = Math.exp(logits.d[i * k + c] - max);
      out[i * k + c] = e;
      sum += e;
    }
    for (let c = 0; c < k; c++) out[i * k + c] /= sum;
  }
  return out;
}

function sigmoidArr(logits) {
  const out = new Float32Array(logits.d.length);
  for (let i = 0; i < out.length; i++) out[i] = 1 / (1 + Math.exp(-logits.d[i]));
  return out;
}

function ovRocAuc(probs, labels, k) {
  let sum = 0;
  let used = 0;
  for (let c = 0; c < k; c++) {
    const scores = [];
    const ys = [];
    let present = false;
    for (let i = 0; i < labels.length; i++) {
      scores.push(probs[i * k + c]);
      ys.push(labels[i] === c ? 1 : 0);
      if (labels[i] === c) present = true;
    }
    if (!present) continue;
    if (ys.every((v) => v === 1)) continue;
    sum += M.rocCurve(Float32Array.from(scores), Float32Array.from(ys)).auc;
    used++;
  }
  return used ? sum / used : 0;
}

export const TASKS = {
  multiclass: {
    key: 'multiclass',
    label: 'Multi-class classification',
    family: 'classification',
    lossFns: ['cross_entropy', 'focal'],
    defaultLoss: 'cross_entropy',
    lossFn(model, ds, idx, cfg, mode, rng) {
      const x = makeBatchNode(ds, idx);
      const logits = model.forward(x, mode, rng);
      const y = labelInts(ds, idx);
      const k = logits.shape[logits.shape.length - 1];
      if (cfg.loss === 'focal') return A.focalSoftmaxCrossEntropy(logits, y, { gamma: cfg.focalGamma, weights: cfg.classWeights, labelSmoothing: cfg.labelSmoothing || 0 });
      return A.softmaxCrossEntropy(logits, y, { weights: cfg.classWeights, labelSmoothing: cfg.labelSmoothing || 0 });
    },
    predict(model, ds, idx) {
      const chunks = [];
      const B = 64;
      for (let s = 0; s < idx.length; s += B) {
        const sub = idx.subarray(s, Math.min(idx.length, s + B));
        const x = makeBatchNode(ds, sub);
        const out = model.forward(x, 'eval');
        chunks.push({ out, n: sub.length });
      }
      const k = chunks.length ? chunks[0].out.shape[chunks[0].out.shape.length - 1] : 2;
      const logits = new Float32Array(idx.length * k);
      const probs = new Float32Array(idx.length * k);
      let off = 0;
      for (const c of chunks) {
        logits.set(c.out.d, off);
        off += c.out.d.length;
      }
      for (let i = 0; i < idx.length; i++) {
        const p = softmaxProbs(new Node(logits.subarray(i * k, (i + 1) * k), [1, k]), k);
        probs.set(p, i * k);
      }
      const pred = new Int32Array(idx.length);
      for (let i = 0; i < idx.length; i++) pred[i] = M.argmaxRow(probs, i * k, k);
      return { probs, pred, k };
    },
    metrics(model, ds, idx, cfg, primary) {
      const { probs, pred, k } = TASKS.multiclass.predict(model, ds, idx);
      const y = labelInts(ds, idx);
      const cm = M.confusionMatrix(y, pred, k);
      const rep = M.perClassReport(cm, ds.labels || null);
      const acc = M.accuracyFromMatrix(cm);
      let logLoss = 0;
      for (let i = 0; i < y.length; i++) logLoss += -Math.log(Math.max(1e-12, probs[i * k + y[i]]));
      logLoss /= Math.max(1, y.length);
      const confProbs = [];
      const confLabels = [];
      for (let i = 0; i < y.length; i++) {
        confProbs.push(probs[i * k + pred[i]]);
        confLabels.push(pred[i] === y[i] ? 1 : 0);
      }
      const cal = M.calibration(Float32Array.from(confProbs), Float32Array.from(confLabels), 10);
      const auroc = ovRocAuc(probs, y, k);
      const errors = [];
      for (let i = 0; i < y.length; i++) {
        if (pred[i] !== y[i]) errors.push({ index: idx[i], predicted: pred[i], actual: y[i], confidence: probs[i * k + pred[i]] });
      }
      errors.sort((a, b) => b.confidence - a.confidence);
      return {
        scalars: {
          accuracy: acc,
          accuracy_ci_lower: M.wilsonInterval(acc * idx.length, idx.length).lower,
          accuracy_ci_upper: M.wilsonInterval(acc * idx.length, idx.length).upper,
          macro_f1: rep.macroF1,
          weighted_f1: rep.weightedF1,
          macro_precision: rep.macroPrecision,
          macro_recall: rep.macroRecall,
          log_loss: logLoss,
          ece: cal.ece,
          brier: cal.brier,
          macro_auroc: auroc,
          error_rate: 1 - acc
        },
        confusion: { matrix: cm, labels: ds.labels || cm.map((_, i) => String(i)) },
        perClass: rep.rows,
        calibration: cal,
        errors: errors.slice(0, 60),
        samples: idx.length
      };
    }
  },

  binary: {
    key: 'binary',
    label: 'Binary detection',
    family: 'classification',
    lossFns: ['bce', 'focal_bce'],
    defaultLoss: 'bce',
    lossFn(model, ds, idx, cfg, mode, rng) {
      const x = makeBatchNode(ds, idx);
      const out = model.forward(x, mode, rng);
      const y = flatTargets(ds, idx);
      const t = new Node(y, out.shape.length > 1 ? out.shape : [y.length]);
      if (cfg.loss === 'focal_bce') return A.focalBinaryCrossEntropy(out, t, { gamma: cfg.focalGamma, posWeight: cfg.posWeight });
      return A.sigmoidBinaryCrossEntropy(out, t, { posWeight: cfg.posWeight || 1 });
    },
    predict(model, ds, idx) {
      const scores = [];
      const B = 64;
      for (let s = 0; s < idx.length; s += B) {
        const sub = idx.subarray(s, Math.min(idx.length, s + B));
        const out = model.forward(makeBatchNode(ds, sub), 'eval');
        const p = sigmoidArr(out);
        for (let i = 0; i < sub.length; i++) scores.push(p[i]);
      }
      return { scores: Float32Array.from(scores) };
    },
    metrics(model, ds, idx, cfg) {
      const { scores } = TASKS.binary.predict(model, ds, idx);
      const y = new Float32Array(idx.length);
      for (let i = 0; i < idx.length; i++) y[i] = ds.y[idx[i]];
      const roc = M.rocCurve(scores, y);
      const pr = M.prCurve(scores, y);
      const best = { threshold: cfg && Number.isFinite(cfg.decisionThreshold) ? cfg.decisionThreshold : 0.5 };
      const rep = M.binaryReport(scores, y, best.threshold);
      const cal = M.calibration(scores, y, 10);
      const sweep = M.thresholdSweep(scores, y, 21);
      const errors = [];
      for (let i = 0; i < idx.length; i++) {
        const p = scores[i] >= best.threshold ? 1 : 0;
        if (p !== (y[i] > 0.5 ? 1 : 0)) errors.push({ index: idx[i], score: scores[i], actual: y[i], predicted: p });
      }
      return {
        scalars: {
          auc_roc: roc.auc,
          average_precision: pr.ap,
          f1: rep.f1,
          precision: rep.precision,
          recall: rep.recall,
          far: rep.far,
          frr: rep.frr,
          accuracy: rep.accuracy,
          accuracy_ci_lower: M.wilsonInterval(rep.tp + rep.tn, idx.length).lower,
          accuracy_ci_upper: M.wilsonInterval(rep.tp + rep.tn, idx.length).upper,
          mcc: rep.mcc,
          ece: cal.ece,
          brier: cal.brier,
          best_threshold: best.threshold
        },
        roc,
        pr,
        calibration: cal,
        sweep,
        confusion: { matrix: [[rep.tn, rep.fp], [rep.fn, rep.tp]], labels: ['negative', 'positive'] },
        errors: errors.slice(0, 60),
        samples: idx.length
      };
    }
  },

  multilabel: {
    key: 'multilabel',
    label: 'Multi-label classification',
    family: 'classification',
    lossFns: ['bce'],
    defaultLoss: 'bce',
    lossFn(model, ds, idx, cfg, mode, rng) {
      const x = makeBatchNode(ds, idx);
      const out = model.forward(x, mode, rng);
      const y = flatTargets(ds, idx);
      const t = new Node(y, out.shape);
      return A.sigmoidBinaryCrossEntropy(out, t);
    },
    predict(model, ds, idx) {
      const B = 64;
      const first = model.forward(makeBatchNode(ds, idx.subarray(0, 1)), 'eval');
      const perSample = first.d.length;
      const all = new Float32Array(idx.length * perSample);
      let off = 0;
      for (let s = 0; s < idx.length; s += B) {
        const sub = idx.subarray(s, Math.min(idx.length, s + B));
        const out = model.forward(makeBatchNode(ds, sub), 'eval');
        const p = sigmoidArr(out);
        all.set(p.subarray(0, sub.length * perSample), off);
        off += sub.length * perSample;
      }
      return { probs: all, k: perSample };
    },
    metrics(model, ds, idx, cfg) {
      const { probs, k } = TASKS.multilabel.predict(model, ds, idx);
      const y = new Float32Array(idx.length * k);
      for (let i = 0; i < idx.length; i++) {
        for (let c = 0; c < k; c++) y[i * k + c] = ds.y[idx[i] * k + c];
      }
      const rows = [];
      let macroF1 = 0;
      let macroAuc = 0;
      for (let c = 0; c < k; c++) {
        const sc = new Float32Array(idx.length);
        const lc = new Float32Array(idx.length);
        for (let i = 0; i < idx.length; i++) {
          sc[i] = probs[i * k + c];
          lc[i] = y[i * k + c];
        }
        const best = { threshold: cfg && Number.isFinite(cfg.decisionThreshold) ? cfg.decisionThreshold : 0.5 };
        const rep = M.binaryReport(sc, lc, best.threshold);
        const auc = lc.some((v) => v > 0.5) && lc.some((v) => v < 0.5) ? M.rocCurve(sc, lc).auc : 0;
        rows.push({ label: ds.labels ? ds.labels[c] : String(c), index: c, precision: rep.precision, recall: rep.recall, f1: rep.f1, support: Array.from(lc).filter((v) => v > 0.5).length, threshold: best.threshold, auc });
        macroF1 += rep.f1;
        macroAuc += auc;
      }
      let microTp = 0;
      let microFp = 0;
      let microFn = 0;
      for (let i = 0; i < idx.length * k; i++) {
        const p = probs[i] >= 0.5 ? 1 : 0;
        const t = y[i] > 0.5 ? 1 : 0;
        if (p && t) microTp++;
        else if (p && !t) microFp++;
        else if (!p && t) microFn++;
      }
      const microP = microTp + microFp ? microTp / (microTp + microFp) : 0;
      const microR = microTp + microFn ? microTp / (microTp + microFn) : 0;
      return {
        scalars: {
          macro_f1: macroF1 / Math.max(1, k),
          mean_auc: macroAuc / Math.max(1, k),
          micro_precision: microP,
          micro_recall: microR,
          micro_f1: microP + microR ? 2 * microP * microR / (microP + microR) : 0,
          exact_match: exactMatch(probs, y, k, idx.length)
        },
        perClass: rows,
        samples: idx.length
      };
    }
  },

  regression: {
    key: 'regression',
    label: 'Regression',
    family: 'regression',
    lossFns: ['mse', 'mae', 'huber'],
    defaultLoss: 'mse',
    lossFn(model, ds, idx, cfg, mode, rng) {
      const x = makeBatchNode(ds, idx);
      const out = model.forward(x, mode, rng);
      const y = flatTargets(ds, idx);
      const t = new Node(y, out.shape);
      if (cfg.loss === 'mae') return A.maeLoss(out, t);
      if (cfg.loss === 'huber') return A.huberLoss(out, t, cfg.huberDelta || 1);
      return A.mseLoss(out, t);
    },
    predict(model, ds, idx) {
      const per = shapeSize(ds.yShape || [1]);
      const out = new Float32Array(idx.length * per);
      const B = 64;
      for (let s = 0; s < idx.length; s += B) {
        const sub = idx.subarray(s, Math.min(idx.length, s + B));
        const r = model.forward(makeBatchNode(ds, sub), 'eval');
        out.set(r.d.subarray(0, sub.length * per), s * per);
      }
      return { pred: out, per };
    },
    metrics(model, ds, idx, cfg) {
      const { pred, per } = TASKS.regression.predict(model, ds, idx);
      const target = flatTargets(ds, idx);
      const rep = M.regressionReport(pred, target);
      const pearson = M.correlation(pred, target);
      const scatter = [];
      const stride = Math.max(1, Math.floor(idx.length / 400));
      for (let i = 0; i < idx.length; i += stride) scatter.push([target[i], pred[i]]);
      return {
        scalars: { mae: rep.mae, rmse: rep.rmse, r2: rep.r2, mape: rep.mape, max_error: rep.maxError, pearson: pearson },
        scatter,
        errors: worstRegression(pred, target, idx, per),
        samples: idx.length
      };
    }
  },

  seq_tag: {
    key: 'seq_tag',
    label: 'Sequence labelling',
    family: 'sequence',
    lossFns: ['cross_entropy'],
    defaultLoss: 'cross_entropy',
    lossFn(model, ds, idx, cfg, mode, rng) {
      const x = makeBatchNode(ds, idx);
      const out = model.forward(x, mode, rng);
      const T = out.shape[1];
      const k = out.shape[2];
      const y = labelInts(ds, idx);
      const flat = new Int32Array(idx.length * T);
      for (let i = 0; i < idx.length; i++) {
        for (let t = 0; t < T; t++) flat[i * T + t] = y[i * T + t];
      }
      const logits = A.reshape(out, [idx.length * T, k]);
      const w = cfg.classWeights ? Float32Array.from(cfg.classWeights) : null;
      return maskedCrossEntropy(logits, flat, maskFromLengths(ds, idx, T), k, cfg);
    },
    predict(model, ds, idx) {
      const T = ds.yShape ? ds.yShape[0] : ds.xShape[0];
      const chunks = [];
      const B = 32;
      for (let s = 0; s < idx.length; s += B) {
        const sub = idx.subarray(s, Math.min(idx.length, s + B));
        const out = model.forward(makeBatchNode(ds, sub), 'eval');
        chunks.push({ d: out.d, n: sub.length, k: out.shape[out.shape.length - 1], t: out.shape.length === 3 ? out.shape[1] : T });
      }
      const k = chunks[0].k;
      const logits = new Float32Array(idx.length * T * k);
      let off = 0;
      for (const c of chunks) {
        logits.set(c.d.subarray(0, c.n * c.t * k), off);
        off += c.n * T * k;
      }
      return { probs: softmaxRows(logits, idx.length * T, k), k, T };
    },
    metrics(model, ds, idx, cfg) {
      const { probs, k, T } = TASKS.seq_tag.predict(model, ds, idx);
      const y = labelInts(ds, idx);
      const pred = new Int32Array(idx.length * T);
      for (let i = 0; i < idx.length * T; i++) pred[i] = M.argmaxRow(probs, i * k, k);
      const mask = ds.lengths ? maskFromLengths(ds, idx, T) : null;
      const cm = [];
      for (let a = 0; a < k; a++) cm.push(new Array(k).fill(0));
      let correct = 0;
      let total = 0;
      for (let i = 0; i < idx.length; i++) {
        for (let t = 0; t < T; t++) {
          if (mask && !mask[i * T + t]) continue;
          const yt = y[i * T + t];
          const pt = pred[i * T + t];
          if (yt >= 0 && yt < k) cm[yt][pt]++;
          if (yt === pt) correct++;
          total++;
        }
      }
      const rep = M.perClassReport(cm, ds.labels || null);
      const ent = entityF1(pred, y, idx.length, T, mask, k);
      const cal = M.multiclassCalibration(probs, Array.from(y).slice(0, idx.length * T), k, 10);
      return {
        scalars: {
          token_accuracy: total ? correct / total : 0,
          macro_f1: rep.macroF1,
          weighted_f1: rep.weightedF1,
          entity_f1: ent.f1,
          entity_precision: ent.precision,
          entity_recall: ent.recall,
          ece: cal.ece
        },
        confusion: { matrix: cm, labels: ds.labels || cm.map((_, i) => String(i)) },
        perClass: rep.rows,
        calibration: cal,
        samples: idx.length
      };
    }
  },

  lm: {
    key: 'lm',
    label: 'Next-token language model',
    family: 'sequence',
    lossFns: ['cross_entropy'],
    defaultLoss: 'cross_entropy',
    lossFn(model, ds, idx, cfg, mode, rng) {
      const x = makeBatchNode(ds, idx);
      const out = model.forward(x, mode, rng);
      const T = x.shape[1];
      const k = out.shape[out.shape.length - 1];
      const targets = new Int32Array(idx.length * T);
      const maskArr = new Float32Array(idx.length * T);
      for (let i = 0; i < idx.length; i++) {
        const s = idx[i] * T;
        for (let t = 0; t < T; t++) {
          const next = t + 1 < T ? ds.x[s + t + 1] : (ds.y ? ds.y[idx[i]] : -1);
          targets[i * T + t] = Math.round(next);
          maskArr[i * T + t] = t + 1 < T ? 1 : 0;
        }
      }
      const logits = A.reshape(out, [idx.length * T, k]);
      const sm = softmaxProbs(logits, k);
      let loss = 0;
      let msum = 0;
      for (let i = 0; i < targets.length; i++) {
        if (!maskArr[i]) continue;
        loss += -Math.log(Math.max(1e-12, sm[i * k + targets[i]]));
        msum += 1;
      }
      const node = new Node(new Float32Array([loss / Math.max(1, msum)]), [1]);
      node.parents = [logits];
      node.bwd = (gy) => {
        const g = new Float32Array(logits.d.length);
        const scale = gy[0] / Math.max(1, msum);
        for (let i = 0; i < targets.length; i++) {
          if (!maskArr[i]) continue;
          for (let c = 0; c < k; c++) g[i * k + c] = scale * (sm[i * k + c] - (c === targets[i] ? 1 : 0));
        }
        if (!logits.g) logits.g = new Float32Array(logits.d.length);
        for (let i = 0; i < g.length; i++) logits.g[i] += g[i];
      };
      return node;
    },
    predict(model, ds, idx) {
      const T = ds.xShape[0];
      const chunks = [];
      const B = 32;
      for (let s = 0; s < idx.length; s += B) {
        const sub = idx.subarray(s, Math.min(idx.length, s + B));
        const out = model.forward(makeBatchNode(ds, sub), 'eval');
        chunks.push({ d: out.d, n: sub.length, k: out.shape[out.shape.length - 1] });
      }
      const k = chunks[0].k;
      const logits = new Float32Array(idx.length * T * k);
      let off = 0;
      for (const c of chunks) {
        logits.set(c.d.subarray(0, c.n * T * k), off);
        off += c.n * T * k;
      }
      return { probs: softmaxRows(logits, idx.length * T, k), k, T };
    },
    metrics(model, ds, idx, cfg) {
      const { probs, k, T } = TASKS.lm.predict(model, ds, idx);
      let loss = 0;
      let count = 0;
      let correct = 0;
      let bits = 0;
      const errors = [];
      for (let i = 0; i < idx.length; i++) {
        const s = idx[i] * T;
        for (let t = 0; t < T - 1; t++) {
          const row = (i * T + t) * k;
          const target = Math.round(ds.x[s + t + 1]);
          const p = Math.max(1e-12, probs[row + target]);
          loss += -Math.log(p);
          bits += -Math.log2(p);
          count++;
          const pred = M.argmaxRow(probs, row, k);
          if (pred === target) correct++;
          else if (errors.length < 60) errors.push({ index: idx[i], position: t, predicted: pred, actual: target });
        }
      }
      const nll = loss / Math.max(1, count);
      return {
        scalars: {
          perplexity: Math.exp(Math.min(30, nll)),
          nll: nll,
          top1_accuracy: count ? correct / count : 0,
          bits_per_symbol: bits / Math.max(1, count),
          bytes_per_symbol: bits / Math.max(1, count) / 8
        },
        errors,
        samples: idx.length,
        tokens: count
      };
    }
  },

  image_map: {
    key: 'image_map',
    label: 'Image-to-image',
    family: 'image',
    lossFns: ['mse', 'mae', 'l1_l2'],
    defaultLoss: 'mse',
    lossFn(model, ds, idx, cfg, mode, rng) {
      const x = makeBatchNode(ds, idx);
      const out = model.forward(x, mode, rng);
      const y = sliceRows(ds, 'y', idx, ds.yShape);
      const t = new Node(y, [idx.length].concat(ds.yShape));
      const base = cfg.loss === 'mae' ? A.maeLoss(out, t) : A.mseLoss(out, t);
      if (cfg.loss === 'l1_l2') return A.add(A.mul(base, 0.5), A.mul(A.maeLoss(out, t), 0.5));
      return base;
    },
    predict(model, ds, idx) {
      const per = shapeSize(ds.yShape);
      const out = new Float32Array(idx.length * per);
      const B = 8;
      for (let s = 0; s < idx.length; s += B) {
        const sub = idx.subarray(s, Math.min(idx.length, s + B));
        const r = model.forward(makeBatchNode(ds, sub), 'eval');
        out.set(r.d.subarray(0, sub.length * per), s * per);
      }
      return { pred: out, per };
    },
    metrics(model, ds, idx, cfg) {
      const { pred } = TASKS.image_map.predict(model, ds, idx);
      const per = shapeSize(ds.yShape);
      const C = ds.yShape[0];
      const H = ds.yShape[1];
      const W = ds.yShape[2];
      const peak = cfg.peak === undefined ? 1 : cfg.peak;
      let psnrSum = 0;
      let ssimSum = 0;
      let maeSum = 0;
      const worst = [];
      for (let i = 0; i < idx.length; i++) {
        const a = pred.subarray(i * per, (i + 1) * per);
        const b = ds.y.subarray(idx[i] * per, (idx[i] + 1) * per);
        const p = M.psnr(a, b, peak);
        const s = M.ssim(a, b, W, H, C);
        let m = 0;
        for (let j = 0; j < per; j++) m += Math.abs(a[j] - b[j]);
        m /= per;
        psnrSum += p;
        ssimSum += s;
        maeSum += m;
        worst.push({ index: idx[i], psnr: p, ssim: s, mae: m });
      }
      worst.sort((a, b) => a.psnr - b.psnr);
      const baseline = cfg.baseline ? baselineImageMetrics(ds, idx, cfg.baseline) : null;
      return {
        scalars: {
          psnr: psnrSum / idx.length,
          ssim: ssimSum / idx.length,
          mae: maeSum / idx.length,
          baseline_psnr: baseline ? baseline.psnr : 0,
          baseline_ssim: baseline ? baseline.ssim : 0,
          psnr_gain: baseline ? psnrSum / idx.length - baseline.psnr : 0
        },
        errors: worst.slice(0, 20),
        samples: idx.length
      };
    }
  },

  mask: {
    key: 'mask',
    label: 'Segmentation',
    family: 'image',
    lossFns: ['bce', 'dice', 'bce_dice'],
    defaultLoss: 'bce_dice',
    lossFn(model, ds, idx, cfg, mode, rng) {
      const x = makeBatchNode(ds, idx);
      const out = model.forward(x, mode, rng);
      const y = sliceRows(ds, 'y', idx, ds.yShape);
      const t = new Node(y, [idx.length].concat(ds.yShape));
      const bce = A.sigmoidBinaryCrossEntropy(out, t);
      if (cfg.loss === 'bce') return bce;
      const p = A.sigmoid(out);
      const num = A.sumAll(A.mul(p, t));
      const den = A.add(A.sumAll(p), A.sumAll(t));
      const dice = A.sub(1, A.div(A.mul(num, 2), A.add(den, 1)));
      if (cfg.loss === 'dice') return dice;
      return A.add(A.mul(bce, 0.5), A.mul(dice, 0.5));
    },
    predict(model, ds, idx) {
      const per = shapeSize(ds.yShape);
      const out = new Float32Array(idx.length * per);
      const B = 8;
      for (let s = 0; s < idx.length; s += B) {
        const sub = idx.subarray(s, Math.min(idx.length, s + B));
        const r = model.forward(makeBatchNode(ds, sub), 'eval');
        const p = sigmoidArr(r);
        out.set(p.subarray(0, sub.length * per), s * per);
      }
      return { pred: out, per };
    },
    metrics(model, ds, idx, cfg) {
      const { pred, per } = TASKS.mask.predict(model, ds, idx);
      let iouSum = 0;
      let diceSum = 0;
      let accSum = 0;
      const worst = [];
      const allP = [];
      const allT = [];
      for (let i = 0; i < idx.length; i++) {
        const a = pred.subarray(i * per, (i + 1) * per);
        const b = ds.y.subarray(idx[i] * per, (idx[i] + 1) * per);
        const r = M.segmentationReport(a, b, 0.5);
        iouSum += r.iou;
        diceSum += r.dice;
        accSum += r.pixelAccuracy;
        worst.push({ index: idx[i], iou: r.iou, dice: r.dice });
        const stride = Math.max(1, Math.floor(per / 200));
        for (let j = 0; j < per; j += stride) {
          allP.push(a[j]);
          allT.push(b[j]);
        }
      }
      const roc = M.rocCurve(Float32Array.from(allP), Float32Array.from(allT));
      worst.sort((a, b) => a.iou - b.iou);
      return {
        scalars: {
          iou: iouSum / idx.length,
          dice: diceSum / idx.length,
          pixel_accuracy: accSum / idx.length,
          boundary_auc: roc.auc
        },
        roc,
        errors: worst.slice(0, 20),
        samples: idx.length
      };
    }
  },

  frame_binary: {
    key: 'frame_binary',
    label: 'Per-frame detection',
    family: 'audio',
    lossFns: ['bce', 'focal_bce'],
    defaultLoss: 'bce',
    lossFn(model, ds, idx, cfg, mode, rng) {
      const x = makeBatchNode(ds, idx);
      const out = model.forward(x, mode, rng);
      const T = out.shape.length === 3 ? out.shape[1] * out.shape[2] : out.d.length / idx.length;
      const y = new Float32Array(idx.length * T);
      const yT = ds.yShape[0];
      for (let i = 0; i < idx.length; i++) {
        for (let t = 0; t < T; t++) {
          const src = Math.min(yT - 1, Math.round(t * yT / T));
          y[i * T + t] = ds.y[idx[i] * yT + src];
        }
      }
      const t = new Node(y, [idx.length * T]);
      const flat = A.reshape(out, [idx.length * T, 1]);
      if (cfg.loss === 'focal_bce') return A.focalBinaryCrossEntropy(flat, t, { gamma: cfg.focalGamma, posWeight: cfg.posWeight });
      return A.sigmoidBinaryCrossEntropy(flat, t, { posWeight: cfg.posWeight || 1 });
    },
    predict(model, ds, idx) {
      const chunks = [];
      const B = 16;
      for (let s = 0; s < idx.length; s += B) {
        const sub = idx.subarray(s, Math.min(idx.length, s + B));
        const out = model.forward(makeBatchNode(ds, sub), 'eval');
        chunks.push({ p: sigmoidArr(out), n: sub.length, per: out.d.length / sub.length });
      }
      const per = chunks[0].per;
      const scores = new Float32Array(idx.length * per);
      let off = 0;
      for (const c of chunks) {
        scores.set(c.p, off);
        off += c.p.length;
      }
      return { scores, per };
    },
    metrics(model, ds, idx, cfg) {
      const { scores, per } = TASKS.frame_binary.predict(model, ds, idx);
      const yT = ds.yShape[0];
      const y = new Float32Array(idx.length * per);
      for (let i = 0; i < idx.length; i++) {
        for (let t = 0; t < per; t++) {
          const src = Math.min(yT - 1, Math.round(t * yT / per));
          y[i * per + t] = ds.y[idx[i] * yT + src];
        }
      }
      const roc = M.rocCurve(scores, y);
      const pr = M.prCurve(scores, y);
      const best = { threshold: cfg && Number.isFinite(cfg.decisionThreshold) ? cfg.decisionThreshold : 0.5 };
      const rep = M.binaryReport(scores, y, best.threshold);
      const cal = M.calibration(scores, y, 10);
      const sweep = M.thresholdSweep(scores, y, 25);
      let events = 0;
      let detected = 0;
      let i = 0;
      while (i < y.length) {
        if (y[i] > 0.5) {
          let j = i;
          while (j < y.length && y[j] > 0.5) j++;
          events++;
          let hit = false;
          for (let t = i; t < j; t++) if (scores[t] >= best.threshold) hit = true;
          if (hit) detected++;
          i = j;
        } else i++;
      }
      const frameRate = cfg.frameRate || 100;
      const hours = idx.length * per / frameRate / 3600;
      const fa = rep.fp / Math.max(1, idx.length * per) * frameRate * 3600;
      return {
        scalars: {
          auc_roc: roc.auc,
          average_precision: pr.ap,
          frame_accuracy: rep.accuracy,
          frame_f1: rep.f1,
          far: rep.far,
          frr: rep.frr,
          event_recall: events ? detected / events : 1,
          event_precision: detected ? detected / Math.max(1, detected) : 0,
          detections_per_hour: hours > 0 ? fa * hours / Math.max(hours, 1e-9) / Math.max(1, 1) : fa,
          false_accepts_per_hour: fa,
          best_threshold: best.threshold,
          ece: cal.ece,
          precision: rep.precision,
          recall: rep.recall
        },
        roc,
        pr,
        sweep,
        calibration: cal,
        samples: idx.length,
        frames: idx.length * per
      };
    }
  },

  detection: {
    key: 'detection',
    label: 'Object detection',
    family: 'image',
    lossFns: ['detection'],
    defaultLoss: 'detection',
    lossFn(model, ds, idx, cfg, mode, rng) {
      const x = makeBatchNode(ds, idx);
      const out = model.forward(x, mode, rng);
      const cells = out.shape[1];
      const grid = out.shape[2];
      const t = detectionTargets(ds, idx, grid, cells);
      return detectionLoss(out, t, grid, cells, cfg);
    },
    predict(model, ds, idx) {
      const chunks = [];
      const B = 16;
      for (let s = 0; s < idx.length; s += B) {
        const sub = idx.subarray(s, Math.min(idx.length, s + B));
        const out = model.forward(makeBatchNode(ds, sub), 'eval');
        chunks.push({ d: out.d, n: sub.length, shape: out.shape });
      }
      return { chunks };
    },
    metrics(model, ds, idx, cfg) {
      const { chunks } = TASKS.detection.predict(model, ds, idx);
      const predBoxes = [];
      const trueBoxes = [];
      let off = 0;
      const confThreshold = cfg.confThreshold || 0.3;
      const collectFloor = Math.min(0.05, confThreshold);
      for (const c of chunks) {
        const cells = c.shape[1];
        const grid = c.shape[2];
        for (let i = 0; i < c.n; i++) {
          const sampleIdx = idx[off + i];
          for (let gy = 0; gy < grid; gy++) {
            for (let gx = 0; gx < grid; gx++) {
              const base = (i * cells + 0) * grid * grid + gy * grid + gx;
              const conf = 1 / (1 + Math.exp(-c.d[base]));
              if (conf < collectFloor) continue;
              const cx = c.d[base + grid * grid];
              const cy = c.d[base + 2 * grid * grid];
              const w = Math.abs(c.d[base + 3 * grid * grid]);
              const h = Math.abs(c.d[base + 4 * grid * grid]);
              predBoxes.push({
                image: sampleIdx,
                score: conf,
                box: { x: cx - w / 2, y: cy - h / 2, w, h }
              });
            }
          }
          const boxes = ds.boxes ? ds.boxes[sampleIdx] : [];
          for (const b of boxes) trueBoxes.push({ image: sampleIdx, box: b });
        }
        off += c.n;
      }
      const at50 = M.detectionReport(predBoxes, trueBoxes, 0.5);
      let apSum = 0;
      let apCount = 0;
      for (let t = 0.5; t <= 0.95; t += 0.05) {
        const r = M.detectionReport(predBoxes, trueBoxes, t);
        apSum += r.ap;
        apCount++;
      }
      const sweep = [];
      let bestF1 = 0;
      let bestPoint = null;
      for (let th = 0.05; th <= 0.95; th += 0.05) {
        const kept = predBoxes.filter((b) => b.score >= th);
        const r = M.detectionReport(kept, trueBoxes, 0.5);
        const f1 = r.precision + r.recall > 0 ? 2 * r.precision * r.recall / (r.precision + r.recall) : 0;
        sweep.push({ threshold: th, precision: r.precision, recall: r.recall, f1, tp: r.tp, fp: r.fp, fn: r.fn });
        if (f1 > bestF1) {
          bestF1 = f1;
          bestPoint = sweep[sweep.length - 1];
        }
      }
      const fixed = M.detectionReport(predBoxes.filter(b => b.score >= confThreshold), trueBoxes, 0.5);
      return {
        scalars: {
          mAP50: at50.mAP,
          mAP: apCount ? apSum / apCount : 0,
          f1_at_best_conf: bestPoint ? bestPoint.f1 : 0,
          best_conf_threshold: bestPoint ? bestPoint.threshold : confThreshold,
          precision: fixed.precision,
          recall: fixed.recall,
          tp: fixed.tp,
          fp: fixed.fp,
          fn: fixed.fn
        },
        pr: { points: at50.curve.map((c) => ({ recall: c.recall, precision: c.precision, threshold: c.score })), ap: at50.ap },
        sweep,
        confThreshold,
        samples: idx.length,
        boxes: { pred: predBoxes.filter((b) => b.score >= confThreshold).slice(0, 200), true: trueBoxes.slice(0, 200) }
      };
    }
  },

  embedding: {
    key: 'embedding',
    label: 'Embedding / retrieval',
    family: 'representation',
    lossFns: ['infonce', 'supcon', 'triplet', 'cosine'],
    defaultLoss: 'supcon',
    lossFn(model, ds, idx, cfg, mode, rng) {
      const xa = makeBatchNode(ds, idx, 'x', ds.xShape);
      const xb = makeBatchNode(ds, idx, 'x2', ds.xShape);
      const za = A.l2normalize(model.forward(xa, mode, rng));
      const zb = A.l2normalize(model.forward(xb, mode, rng));
      if (cfg.loss === 'cosine') {
        const prod = A.mul(za, zb);
        const s = A.sumAxis(prod, -1);
        return A.sub(1, A.meanAll(s));
      }
      const B = idx.length;
      const mask = new Uint8Array(B * B);
      const grouped = cfg.loss !== 'infonce' && ds.groups;
      for (let i = 0; i < B; i++) {
        if (!grouped) {
          mask[i * B + i] = 1;
          continue;
        }
        for (let j = 0; j < B; j++) mask[i * B + j] = ds.groups[idx[i]] === ds.groups[idx[j]] ? 1 : 0;
      }
      return A.infoNceMasked(za, zb, cfg.temperature || 0.07, mask);
    },
    predict(model, ds, idx) {
      const chunks = [];
      const B = 64;
      for (let s = 0; s < idx.length; s += B) {
        const sub = idx.subarray(s, Math.min(idx.length, s + B));
        const out = model.forward(makeBatchNode(ds, sub, 'x', ds.xShape), 'eval');
        const n = A.l2normalize(out);
        chunks.push(n.d);
      }
      const dim = chunks[0].length / Math.min(B, idx.length);
      const emb = new Float32Array(idx.length * dim);
      let off = 0;
      for (const c of chunks) {
        emb.set(c, off);
        off += c.length;
      }
      return { emb, dim };
    },
    metrics(model, ds, idx, cfg) {
      const { emb, dim } = TASKS.embedding.predict(model, ds, idx);
      const vectors = [];
      for (let i = 0; i < idx.length; i++) vectors.push(Array.from(emb.subarray(i * dim, (i + 1) * dim)));
      const labels = [];
      const groupOf = ds.groups ? ds.groups : null;
      if (groupOf) {
        for (let i = 0; i < idx.length; i++) {
          const rel = [];
          for (let j = 0; j < idx.length; j++) {
            if (i === j) continue;
            if (groupOf[idx[i]] === groupOf[idx[j]]) rel.push(j);
          }
          labels.push(rel);
        }
      } else {
        for (let i = 0; i < idx.length; i++) labels.push(i % 2 === 0 ? [] : []);
      }
      const scores = [];
      const ys = [];
      for (let i = 0; i < idx.length; i++) {
        for (let j = i + 1; j < idx.length; j++) {
          let d = 0;
          for (let k = 0; k < dim; k++) {
            const diff = vectors[i][k] - vectors[j][k];
            d += diff * diff;
          }
          scores.push(1 - Math.sqrt(d) / 2);
          ys.push(groupOf && groupOf[idx[i]] === groupOf[idx[j]] ? 1 : 0);
        }
      }
      const roc = ys.some((v) => v > 0.5) && ys.some((v) => v < 0.5)
        ? M.rocCurve(Float32Array.from(scores), Float32Array.from(ys))
        : { auc: 0, points: [] };
      const pr = ys.some((v) => v > 0.5) ? M.prCurve(Float32Array.from(scores), Float32Array.from(ys)) : { ap: 0, points: [] };
      const retr = labels.some((l) => l.length) ? M.retrievalReport(vectors, vectors, labels, [1, 5, 10], true) : null;
      const altChunks = [];
      for (let s = 0; s < idx.length; s += 64) {
        const sub = idx.subarray(s, Math.min(idx.length, s + 64));
        altChunks.push(A.l2normalize(model.forward(makeBatchNode(ds, sub, 'x2', ds.xShape), 'eval')).d);
      }
      const altDim = altChunks[0].length / Math.min(64, idx.length);
      const alt = new Float32Array(idx.length * altDim);
      let altOff = 0;
      for (const a of altChunks) {
        alt.set(a, altOff);
        altOff += a.length;
      }
      const altVectors = [];
      for (let i = 0; i < idx.length; i++) altVectors.push(Array.from(alt.subarray(i * altDim, (i + 1) * altDim)));
      const pairLabels = [];
      for (let i = 0; i < idx.length; i++) {
        if (groupOf) {
          const rel = [];
          for (let j = 0; j < idx.length; j++) if (groupOf[idx[i]] === groupOf[idx[j]]) rel.push(j);
          pairLabels.push(rel);
        } else {
          pairLabels.push([i]);
        }
      }
      const retrAlt = ds.x2 ? M.retrievalReport(vectors, altVectors, pairLabels, [1, 5]) : null;
      const threshold = cfg && Number.isFinite(cfg.decisionThreshold) ? cfg.decisionThreshold : 0.5;
      const best = M.binaryReport(scores, ys, threshold);
      let align = 0;
      let uniform = 0;
      let pairs = 0;
      for (let i = 0; i < Math.min(200, idx.length); i++) {
        for (let j = i + 1; j < Math.min(200, idx.length); j++) {
          let dot = 0;
          for (let k = 0; k < dim; k++) dot += vectors[i][k] * vectors[j][k];
          if (groupOf && groupOf[idx[i]] === groupOf[idx[j]]) {
            align += Math.pow(1 - dot, 2);
            pairs++;
          } else {
            uniform += Math.exp(-2 * Math.pow(1 - dot, 2));
          }
        }
      }
      return {
        scalars: {
          pair_auc: roc.auc,
          pair_ap: pr.ap,
          duplicate_f1: best.f1,
          duplicate_threshold: best.threshold,
          recall_at_1: retr ? retr['recall@1'] : 0,
          recall_at_5: retr ? retr['recall@5'] : 0,
          recall_at_10: retr ? retr['recall@10'] : 0,
          paraphrase_recall_at_1: retrAlt ? retrAlt['recall@1'] : 0,
          paraphrase_recall_at_5: retrAlt ? retrAlt['recall@5'] : 0,
          map: retr ? retr.map : 0,
          mrr: retr ? retr.mrr : 0,
          alignment: pairs ? align / pairs : 0,
          uniformity: Math.log(Math.max(1e-9, uniform / Math.max(1, (Math.min(200, idx.length) * (Math.min(200, idx.length) - 1)) / 2))),
          dimensions: dim
        },
        roc,
        pr,
        samples: idx.length
      };
    }
  },

  anomaly: {
    key: 'anomaly',
    label: 'Anomaly detection (reconstruction)',
    family: 'unsupervised',
    lossFns: ['mse', 'mae'],
    defaultLoss: 'mse',
    lossFn(model, ds, idx, cfg, mode, rng) {
      const x = makeBatchNode(ds, idx);
      const out = model.forward(x, mode, rng);
      const t = new Node(new Float32Array(x.d), x.shape);
      return cfg.loss === 'mae' ? A.maeLoss(out, t) : A.mseLoss(out, t);
    },
    predict(model, ds, idx) {
      const errors = new Float32Array(idx.length);
      const B = 64;
      const per = shapeSize(ds.xShape);
      for (let s = 0; s < idx.length; s += B) {
        const sub = idx.subarray(s, Math.min(idx.length, s + B));
        const x = makeBatchNode(ds, sub);
        const out = model.forward(x, 'eval');
        for (let i = 0; i < sub.length; i++) {
          let se = 0;
          for (let j = 0; j < per; j++) {
            const d = out.d[i * per + j] - x.d[i * per + j];
            se += d * d;
          }
          errors[s + i] = Math.sqrt(se / per);
        }
      }
      return { errors };
    },
    metrics(model, ds, idx, cfg) {
      const { errors } = TASKS.anomaly.predict(model, ds, idx);
      const y = new Float32Array(idx.length);
      for (let i = 0; i < idx.length; i++) y[i] = ds.y[idx[i]];
      const rep = M.anomalyReport(errors, y);
      const normal = [];
      const anom = [];
      for (let i = 0; i < idx.length; i++) (y[i] > 0.5 ? anom : normal).push(errors[i]);
      return {
        scalars: {
          auroc: rep.auroc,
          average_precision: rep.ap,
          p_at_5: rep.precisionAt['p@5'],
          p_at_10: rep.precisionAt['p@10'],
          recall_at_5fpr: rep.recallAt5Fpr,
          threshold: rep.thresholdAt5Fpr,
          recon_mse_normal: meanSq(normal),
          recon_mse_anomaly: meanSq(anom),
          separation: meanSq(anom) - meanSq(normal),
          contamination: rep.anomalies / Math.max(1, rep.total)
        },
        roc: rep.roc,
        histograms: {
          normal: histogram(normal, 30),
          anomaly: histogram(anom, 30)
        },
        samples: idx.length
      };
    }
  }
};

function histogram(values, bins) {
  if (!values.length) return { bins: [], counts: [], min: 0, max: 1 };
  let min = Infinity;
  let max = -Infinity;
  for (const v of values) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (max - min < 1e-9) max = min + 1;
  const counts = new Array(bins).fill(0);
  const labels = [];
  for (let i = 0; i < bins; i++) labels.push(min + (max - min) * (i + 0.5) / bins);
  for (const v of values) {
    let b = Math.floor((v - min) / (max - min) * bins);
    if (b >= bins) b = bins - 1;
    counts[b]++;
  }
  return { bins: labels, counts, min, max };
}

function meanSq(arr) {
  if (!arr.length) return 0;
  let s = 0;
  for (const v of arr) s += v * v;
  return s / arr.length;
}

function exactMatch(probs, y, k, n) {
  let hits = 0;
  for (let i = 0; i < n; i++) {
    let all = true;
    for (let c = 0; c < k; c++) {
      if ((probs[i * k + c] >= 0.5 ? 1 : 0) !== (y[i * k + c] > 0.5 ? 1 : 0)) {
        all = false;
        break;
      }
    }
    if (all) hits++;
  }
  return n ? hits / n : 0;
}

function worstRegression(pred, target, idx, per) {
  const rows = [];
  for (let i = 0; i < idx.length; i++) {
    let se = 0;
    for (let j = 0; j < per; j++) {
      const d = pred[i * per + j] - target[i * per + j];
      se += d * d;
    }
    rows.push({ index: idx[i], error: Math.sqrt(se / per), predicted: pred[i * per], actual: target[i * per] });
  }
  rows.sort((a, b) => b.error - a.error);
  return rows.slice(0, 40);
}

function maskedCrossEntropy(logits, targets, maskArr, k, cfg) {
  const sm = softmaxProbs(logits, k);
  const smoothing = cfg.labelSmoothing || 0;
  let loss = 0;
  let msum = 0;
  for (let i = 0; i < targets.length; i++) {
    if (!maskArr[i]) continue;
    msum += maskArr[i];
    let l = -Math.log(Math.max(1e-12, sm[i * k + targets[i]]));
    if (smoothing > 0) {
      let s2 = 0;
      for (let c = 0; c < k; c++) s2 += Math.log(Math.max(1e-12, sm[i * k + c]));
      l = (1 - smoothing) * l - smoothing * s2 / k;
    }
    loss += l;
  }
  const node = new Node(new Float32Array([loss / Math.max(1, msum)]), [1]);
  node.parents = [logits];
  node.bwd = (gy) => {
    const g = new Float32Array(logits.d.length);
    const scale = gy[0] / Math.max(1, msum);
    for (let i = 0; i < targets.length; i++) {
      if (!maskArr[i]) continue;
      for (let c = 0; c < k; c++) {
        let t = sm[i * k + c];
        if (smoothing > 0) t = (1 - smoothing) * t + smoothing / k;
        g[i * k + c] = scale * maskArr[i] * (t - (c === targets[i] ? 1 : 0));
      }
    }
    if (!logits.g) logits.g = new Float32Array(logits.d.length);
    for (let i = 0; i < g.length; i++) logits.g[i] += g[i];
  };
  return node;
}

function maskFromLengths(ds, idx, T) {
  const mask = new Float32Array(idx.length * T);
  for (let i = 0; i < idx.length; i++) {
    const L = ds.lengths ? Math.min(T, ds.lengths[idx[i]] || T) : T;
    for (let t = 0; t < L; t++) mask[i * T + t] = 1;
  }
  return mask;
}

function entityF1(pred, truth, n, T, mask, k) {
  const extract = (arr, i) => {
    const ents = [];
    let cur = null;
    for (let t = 0; t < T; t++) {
      if (mask && !mask[i * T + t]) continue;
      const tag = arr[i * T + t];
      if (tag > 0) {
        if (cur && cur.label === tag) {
          cur.end = t;
        } else {
          if (cur) ents.push(cur);
          cur = { label: tag, start: t, end: t };
        }
      } else if (cur) {
        ents.push(cur);
        cur = null;
      }
    }
    if (cur) ents.push(cur);
    return ents;
  };
  let tp = 0;
  let fp = 0;
  let fn = 0;
  for (let i = 0; i < n; i++) {
    const p = extract(pred, i);
    const g = extract(truth, i);
    const used = new Set();
    for (const e of p) {
      let hit = -1;
      for (let j = 0; j < g.length; j++) {
        if (used.has(j)) continue;
        if (g[j].label === e.label && g[j].start === e.start && g[j].end === e.end) {
          hit = j;
          break;
        }
      }
      if (hit >= 0) {
        used.add(hit);
        tp++;
      } else fp++;
    }
    fn += g.length - used.size;
  }
  const precision = tp + fp ? tp / (tp + fp) : 0;
  const recall = tp + fn ? tp / (tp + fn) : 0;
  return { precision, recall, f1: precision + recall ? 2 * precision * recall / (precision + recall) : 0 };
}

function detectionLoss(out, t, grid, cells, cfg) {
  const N = out.shape[0];
  const plane = grid * grid;
  const total = N * cells * plane;
  const coordScale = cfg.coordScale === undefined ? 8 : cfg.coordScale;
  const o = out.d;
  let objSum = 0;
  let coordSum = 0;
  let coordCount = 0;
  for (let n = 0; n < N; n++) {
    let hasObject = false;
    for (let p = 0; p < plane; p++) {
      if (t[n * cells * plane + p] > 0.5) {
        hasObject = true;
        break;
      }
    }
    for (let p = 0; p < plane; p++) {
      const logit = o[n * cells * plane + p];
      const isObj = t[n * cells * plane + p] > 0.5;
      const m = Math.max(0, logit);
      const softplus = m + Math.log1p(Math.exp(-Math.abs(logit)));
      objSum += softplus - (isObj ? logit : 0);
      if (hasObject) {
        for (let c = 1; c < cells; c++) {
          const at = (n * cells + c) * plane + p;
          const d = o[at] - t[at];
          coordSum += d * d;
        }
        coordCount += cells - 1;
      }
    }
  }
  const objDenom = Math.max(1, N * plane);
  const coordDenom = Math.max(1, coordCount);
  const value = objSum / objDenom + coordScale * coordSum / coordDenom;
  const node = new Node(new Float32Array([value]), [1]);
  node.parents = [out];
  node.bwd = (gy) => {
    const g = new Float32Array(total);
    const s = gy[0];
    for (let n = 0; n < N; n++) {
      let hasObject = false;
      for (let p = 0; p < plane; p++) {
        if (t[n * cells * plane + p] > 0.5) {
          hasObject = true;
          break;
        }
      }
      for (let p = 0; p < plane; p++) {
        const at = n * cells * plane + p;
        const logit = o[at];
        const sig = 1 / (1 + Math.exp(-logit));
        const isObj = t[at] > 0.5;
        g[at] = s * (sig - (isObj ? 1 : 0)) / objDenom;
        if (hasObject) {
          for (let c = 1; c < cells; c++) {
            const idx = (n * cells + c) * plane + p;
            g[idx] = s * coordScale * 2 * (o[idx] - t[idx]) / coordDenom;
          }
        }
      }
    }
    out.g = addInto(ensureGrad(out), g);
  };
  return node;
}

function detectionTargets(ds, idx, grid, cells) {
  const plane = grid * grid;
  const out = new Float32Array(idx.length * cells * plane);
  for (let i = 0; i < idx.length; i++) {
    const boxes = ds.boxes ? ds.boxes[idx[i]] : [];
    for (const b of boxes) {
      const gx = Math.min(grid - 1, Math.max(0, Math.floor((b.x + b.w / 2) * grid)));
      const gy = Math.min(grid - 1, Math.max(0, Math.floor((b.y + b.h / 2) * grid)));
      out[i * cells * plane + gy * grid + gx] = 1;
    }
    if (!boxes.length) continue;
    const b = boxes[0];
    for (let p = 0; p < plane; p++) {
      out[(i * cells + 1) * plane + p] = b.x + b.w / 2;
      out[(i * cells + 2) * plane + p] = b.y + b.h / 2;
      out[(i * cells + 3) * plane + p] = b.w;
      out[(i * cells + 4) * plane + p] = b.h;
    }
  }
  return out;
}

function baselineImageMetrics(ds, idx, kind) {
  const per = shapeSize(ds.yShape);
  const C = ds.yShape[0];
  const H = ds.yShape[1];
  const W = ds.yShape[2];
  let psnrSum = 0;
  let ssimSum = 0;
  for (const i of idx) {
    const target = ds.y.subarray(i * per, (i + 1) * per);
    const src = ds.x.subarray(i * shapeSize(ds.xShape), (i + 1) * shapeSize(ds.xShape));
    let ref;
    if (kind === 'input') {
      ref = new Float32Array(per);
      const sH = ds.xShape[1];
      const sW = ds.xShape[2];
      for (let c = 0; c < C; c++) {
        for (let y = 0; y < H; y++) {
          for (let x = 0; x < W; x++) {
            const sy = Math.min(sH - 1, Math.floor(y * sH / H));
            const sx = Math.min(sW - 1, Math.floor(x * sW / W));
            ref[(c * H + y) * W + x] = src[(c * sH + sy) * sW + sx];
          }
        }
      }
    } else {
      let mean = 0;
      for (let j = 0; j < per; j++) mean += target[j];
      mean /= per;
      ref = new Float32Array(per).fill(mean);
    }
    psnrSum += M.psnr(ref, target, 1);
    ssimSum += M.ssim(ref, target, W, H, C);
  }
  return { psnr: psnrSum / Math.max(1, idx.length), ssim: ssimSum / Math.max(1, idx.length) };
}

export function taskFor(key) {
  return TASKS[key] || TASKS.multiclass;
}

export function lossValue(node) {
  return node && node.d && node.d.length ? node.d[0] : 0;
}

export { backward, Rng, flatTargets, labelInts, softmaxProbs, sigmoidArr, maskFromLengths, entityF1, detectionTargets, baselineImageMetrics, histogram, exactMatch, worstRegression, ovRocAuc };

import { istft } from '../data/fft.js';
import { siSdr, segmentalSnr, pesqApprox, onsetF1, peaksFromEnvelope } from './metrics.js';

const EXTRA_TASKS = {
  spectral_mask: {
    key: 'spectral_mask',
    label: 'Spectral mask estimation',
    family: 'audio',
    lossFns: ['mse', 'mae', 'masked_mse'],
    defaultLoss: 'mse',
    lossFn(model, ds, idx, cfg, mode, rng) {
      const x = makeBatchNode(ds, idx);
      const out = model.forward(x, mode, rng);
      const y = sliceRows(ds, 'y', idx, ds.yShape);
      const t = new Node(y, [idx.length].concat(ds.yShape));
      const p = A.sigmoid(out);
      if (cfg.loss === 'mae') return A.maeLoss(p, t);
      return A.mseLoss(p, t);
    },
    predict(model, ds, idx) {
      const per = shapeSize(ds.yShape);
      const out = new Float32Array(idx.length * per);
      const B = 16;
      for (let s = 0; s < idx.length; s += B) {
        const sub = idx.subarray(s, Math.min(idx.length, s + B));
        const r = model.forward(makeBatchNode(ds, sub), 'eval');
        const p = sigmoidArr(r);
        out.set(p.subarray(0, sub.length * per), s * per);
      }
      return { pred: out, per };
    },
    metrics(model, ds, idx, cfg) {
      const { pred, per } = EXTRA_TASKS.spectral_mask.predict(model, ds, idx);
      const T = ds.yShape[0];
      const bins = ds.yShape[1];
      let maskMae = 0;
      let sisdrSum = 0;
      let sisdrBase = 0;
      let segSum = 0;
      let mosSum = 0;
      let count = 0;
      const worst = [];
      for (let i = 0; i < idx.length; i++) {
        const w = ds.wave ? ds.wave[idx[i]] : null;
        const target = ds.y.subarray(idx[i] * per, (idx[i] + 1) * per);
        const est = pred.subarray(i * per, (i + 1) * per);
        let mae = 0;
        for (let k = 0; k < per; k++) mae += Math.abs(est[k] - target[k]);
        mae /= per;
        maskMae += mae;
        if (!w) continue;
        const frames = w.frames;
        const wbins = w.bins;
        const mag = new Float32Array(frames * wbins);
        for (let f = 0; f < frames; f++) {
          const srcT = Math.min(T - 1, Math.round(f * T / frames));
          for (let b = 0; b < wbins; b++) {
            const srcB = Math.min(bins - 1, Math.round(b * bins / wbins));
            mag[f * wbins + b] = w.mag[f * wbins + b] * est[srcT * bins + srcB];
          }
        }
        const enhanced = istft(mag, w.phase, frames, wbins, w.frameSize, w.hop);
        const clean = w.clean;
        const n = Math.min(enhanced.length, clean.length);
        const e = enhanced.subarray(0, n);
        const c = clean.subarray(0, n);
        const noisy = w.noisy.subarray(0, n);
        const s = siSdr(e, c);
        const base = siSdr(noisy, c);
        const seg = segmentalSnr(e, c, Math.round(w.sampleRate * 0.02));
        const pq = pesqApprox(e, c, noisy, w.sampleRate);
        sisdrSum += s;
        sisdrBase += base;
        segSum += seg;
        mosSum += pq.mos;
        count++;
        worst.push({ index: idx[i], siSdr: s, baseline: base, mos: pq.mos, mae, snr: w.snr });
      }
      worst.sort((a, b) => a.siSdr - b.siSdr);
      return {
        scalars: {
          si_sdr: count ? sisdrSum / count : 0,
          si_sdr_input: count ? sisdrBase / count : 0,
          si_sdr_improvement: count ? (sisdrSum - sisdrBase) / count : 0,
          segmental_snr: count ? segSum / count : 0,
          pesq_approx: count ? mosSum / count : 0,
          mask_mae: maskMae / Math.max(1, idx.length)
        },
        errors: worst.slice(0, 20),
        samples: idx.length,
        waveformSamples: count
      };
    }
  },

  pitch_class: {
    key: 'pitch_class',
    label: 'Pitch class estimation',
    family: 'audio',
    lossFns: ['cross_entropy'],
    defaultLoss: 'cross_entropy',
    lossFn: null,
    predict: null,
    metrics(model, ds, idx, cfg) {
      const base = TASKS.multiclass.metrics(model, ds, idx, cfg);
      const { pred } = TASKS.multiclass.predict(model, ds, idx);
      const table = ds.featureSpec && ds.featureSpec.noteTable;
      if (table && ds.meta) {
        const predHz = [];
        const trueHz = [];
        for (let i = 0; i < idx.length; i++) {
          predHz.push(table[Math.min(table.length - 1, pred[i])].hz);
          trueHz.push(ds.meta[idx[i]].hz);
        }
        const pr = pitchReportFrom(predHz, trueHz);
        Object.assign(base.scalars, {
          raw_pitch_accuracy: pr.rpa,
          strict_pitch_accuracy: pr.strictRpa,
          mean_abs_cent_error: pr.meanAbsCentError,
          median_abs_cent_error: pr.medianAbsCentError,
          octave_error_rate: pr.octaveErrorRate
        });
        base.pitch = pr;
      }
      return base;
    }
  },

  onset: {
    key: 'onset',
    label: 'Onset and beat detection',
    family: 'audio',
    lossFns: ['bce', 'focal_bce'],
    defaultLoss: 'focal_bce',
    lossFn: null,
    predict: null,
    metrics(model, ds, idx, cfg) {
      const base = TASKS.frame_binary.metrics(model, ds, idx, cfg);
      const { scores, per } = TASKS.frame_binary.predict(model, ds, idx);
      const frameRate = (ds.config && ds.config.frameRate) || 50;
      const thr = base.scalars.best_threshold || 0.5;
      let f1Sum = 0;
      let tempoErr = 0;
      let n = 0;
      for (let i = 0; i < idx.length; i++) {
        const env = scores.subarray(i * per, (i + 1) * per);
        const peaks = peaksFromEnvelope(env, frameRate, thr, Math.max(2, Math.round(frameRate * 0.06)));
        const extras = ds.config && ds.config.extras ? ds.config.extras[idx[i]] : null;
        if (!extras) continue;
        const r = onsetF1(peaks.map((p) => p.time), extras.onsets, 1.5 / frameRate + 0.03);
        f1Sum += r.f1;
        if (peaks.length > 1) {
          const intervals = [];
          for (let k = 1; k < peaks.length; k++) intervals.push(peaks[k].time - peaks[k - 1].time);
          intervals.sort((a, b) => a - b);
          const median = intervals[Math.floor(intervals.length / 2)] || 1;
          tempoErr += Math.abs(60 / median - extras.bpm);
        }
        n++;
      }
      Object.assign(base.scalars, {
        onset_f1: n ? f1Sum / n : 0,
        tempo_mae_bpm: n ? tempoErr / n : 0
      });
      base.onsetFrames = per;
      return base;
    }
  },

  forecast: {
    key: 'forecast',
    label: 'Sequence forecasting',
    family: 'regression',
    lossFns: ['mse', 'mae', 'huber'],
    defaultLoss: 'mse',
    lossFn: null,
    predict: null,
    metrics(model, ds, idx, cfg) {
      const base = TASKS.regression.metrics(model, ds, idx, cfg);
      const { pred, per } = TASKS.regression.predict(model, ds, idx);
      const target = new Float32Array(idx.length * per);
      for (let i = 0; i < idx.length; i++) {
        target.set(ds.y.subarray(idx[i] * per, (idx[i] + 1) * per), i * per);
      }
      const predSeries = [];
      const trueSeries = [];
      const naiveSeries = [];
      for (let i = 0; i < idx.length; i++) {
        predSeries.push(Array.from(pred.subarray(i * per, (i + 1) * per)));
        trueSeries.push(Array.from(target.subarray(i * per, (i + 1) * per)));
        const last = ds.x[idx[i] * shapeSize(ds.xShape) + shapeSize(ds.xShape) - 1];
        naiveSeries.push(new Array(per).fill(last));
      }
      const perHorizon = [];
      for (let h = 0; h < per; h++) {
        let sae = 0;
        let sse = 0;
        let naiveMae = 0;
        for (let i = 0; i < idx.length; i++) {
          sae += Math.abs(predSeries[i][h] - trueSeries[i][h]);
          sse += Math.pow(predSeries[i][h] - trueSeries[i][h], 2);
          naiveMae += Math.abs(naiveSeries[i][h] - trueSeries[i][h]);
        }
        perHorizon.push({
          horizon: h + 1,
          mae: sae / idx.length,
          rmse: Math.sqrt(sse / idx.length),
          naive_mae: naiveMae / idx.length
        });
      }
      const naiveFlat = [];
      const trueFlat = [];
      naiveSeries.forEach((s, i) => s.forEach((v, h) => { naiveFlat.push(v); trueFlat.push(trueSeries[i][h]); }));
      const naive = regressionSummary(naiveFlat, trueFlat);
      Object.assign(base.scalars, {
        naive_mae: naive.mae,
        skill_score: naive.mae > 0 ? 1 - base.scalars.mae / naive.mae : 0,
        mape: base.scalars.mape
      });
      base.perHorizon = perHorizon;
      return base;
    }
  },

  coordinate_field: {
    key: 'coordinate_field',
    label: 'Neural texture / coordinate field',
    family: 'regression',
    lossFns: ['mse', 'mae'],
    defaultLoss: 'mse',
    lossFn: null,
    predict: null,
    metrics(model, ds, idx, cfg) {
      const base = TASKS.regression.metrics(model, ds, idx, cfg);
      const { pred } = TASKS.regression.predict(model, ds, idx);
      const H = ds.config.imageSize[0];
      const W = ds.config.imageSize[1];
      const C = ds.config.channels;
      const dim = shapeSize(ds.xShape);
      const freqs = (ds.featureSpec && ds.featureSpec.freqs) || [];
      const grid = new Float32Array(H * W * dim);
      for (let iy = 0; iy < H; iy++) {
        for (let ix = 0; ix < W; ix++) {
          const at = (iy * W + ix) * dim;
          const u = W > 1 ? (ix / (W - 1)) * 2 - 1 : 0;
          const v = H > 1 ? (iy / (H - 1)) * 2 - 1 : 0;
          grid[at] = u;
          grid[at + 1] = v;
          grid[at + 2] = Math.sqrt(u * u + v * v);
          for (let k = 0; k < freqs.length && 6 + k * 4 < dim; k++) {
            const f = freqs[k] * Math.PI;
            grid[at + 3 + k * 4] = Math.sin(f * u);
            grid[at + 4 + k * 4] = Math.cos(f * u);
            grid[at + 5 + k * 4] = Math.sin(f * v);
            grid[at + 6 + k * 4] = Math.cos(f * v);
          }
        }
      }
      const render = new Float32Array(H * W * C);
      const B = 512;
      const plane = H * W;
      for (let s = 0; s < plane; s += B) {
        const rows = Math.min(B, plane - s);
        const node = new Node(grid.subarray(s * dim, (s + rows) * dim), [rows, dim]);
        const o = model.forward(node, 'eval');
        for (let i = 0; i < rows; i++) {
          for (let ch = 0; ch < C; ch++) render[ch * plane + s + i] = o.d[i * C + ch];
        }
      }
      const target = new Float32Array(H * W * C);
      const filled = new Uint8Array(H * W);
      for (let i = 0; i < ds.count; i++) {
        const px = (ds.x[i * dim] + 1) / 2 * (W - 1);
        const py = (ds.x[i * dim + 1] + 1) / 2 * (H - 1);
        const ix = Math.round(px);
        const iy = Math.round(py);
        if (ix < 0 || ix >= W || iy < 0 || iy >= H) continue;
        for (let c = 0; c < C; c++) target[(c * H + iy) * W + ix] = ds.y[i * C + c];
        filled[iy * W + ix] = 1;
      }
      let coverage = 0;
      for (let i = 0; i < filled.length; i++) coverage += filled[i];
      base.scalars.psnr = psnrOf(pred, sliceRows(ds, 'y', idx, ds.yShape), 1);
      base.scalars.psnr_render = psnrOf(render, target, 1);
      base.scalars.ssim = ssimOf(render, target, W, H, C);
      base.scalars.pixel_coverage = coverage / (H * W);
      base.scalars.weight_bytes = model.paramCount() * 4;
      base.scalars.raw_image_bytes = H * W * C * 3;
      base.scalars.compression_ratio = (H * W * C * 3) / Math.max(1, model.paramCount() * 2);
      base.scalars.compression_ratio_fp32 = (H * W * C * 3) / Math.max(1, model.paramCount() * 4);
      base.image = { prediction: Array.from(render), target: Array.from(target), H, W, C };
      return base;
    }
  }
};

EXTRA_TASKS.pitch_class.lossFn = TASKS.multiclass.lossFn;
EXTRA_TASKS.pitch_class.predict = TASKS.multiclass.predict;
EXTRA_TASKS.onset.lossFn = TASKS.frame_binary.lossFn;
EXTRA_TASKS.onset.predict = TASKS.frame_binary.predict;
EXTRA_TASKS.forecast.lossFn = TASKS.regression.lossFn;
EXTRA_TASKS.forecast.predict = TASKS.regression.predict;
EXTRA_TASKS.coordinate_field.lossFn = TASKS.regression.lossFn;
EXTRA_TASKS.coordinate_field.predict = TASKS.regression.predict;

function pitchReportFrom(predHz, trueHz) {
  let within50 = 0;
  let within20 = 0;
  let centAbs = 0;
  let octaveErr = 0;
  const errs = [];
  for (let i = 0; i < trueHz.length; i++) {
    const cents = 1200 * Math.log2(Math.max(1e-6, predHz[i]) / trueHz[i]);
    const ac = Math.abs(cents);
    errs.push(ac);
    centAbs += ac;
    if (ac <= 50) within50++;
    if (ac <= 20) within20++;
    if (Math.abs(ac - 1200) < 60 || Math.abs(ac - 2400) < 60) octaveErr++;
  }
  errs.sort((a, b) => a - b);
  const n = Math.max(1, trueHz.length);
  return {
    rpa: within50 / n,
    strictRpa: within20 / n,
    meanAbsCentError: centAbs / n,
    medianAbsCentError: errs[Math.floor(errs.length / 2)] || 0,
    octaveErrorRate: octaveErr / n
  };
}

function psnrOf(a, b, peak) {
  let se = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    se += d * d;
  }
  const mse = se / Math.max(1, a.length);
  return mse <= 1e-12 ? 100 : 10 * Math.log10(peak * peak / mse);
}

function ssimOf(a, b, w, h, channels) {
  const C1 = 0.01 * 0.01;
  const C2 = 0.03 * 0.03;
  let total = 0;
  let count = 0;
  for (let c = 0; c < channels; c++) {
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        let ma = 0;
        let mb = 0;
        let n = 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const i = (c * h + y + dy) * w + x + dx;
            ma += a[i];
            mb += b[i];
            n++;
          }
        }
        ma /= n;
        mb /= n;
        let va = 0;
        let vb = 0;
        let cov = 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const i = (c * h + y + dy) * w + x + dx;
            va += (a[i] - ma) * (a[i] - ma);
            vb += (b[i] - mb) * (b[i] - mb);
            cov += (a[i] - ma) * (b[i] - mb);
          }
        }
        va /= n - 1;
        vb /= n - 1;
        cov /= n - 1;
        total += ((2 * ma * mb + C1) * (2 * cov + C2)) / ((ma * ma + mb * mb + C1) * (va + vb + C2));
        count++;
      }
    }
  }
  return count ? total / count : 1;
}

function regressionSummary(pred, target) {
  let sae = 0;
  let sse = 0;
  for (let i = 0; i < pred.length; i++) {
    const e = pred[i] - target[i];
    sae += Math.abs(e);
    sse += e * e;
  }
  const n = Math.max(1, pred.length);
  return { mae: sae / n, rmse: Math.sqrt(sse / n) };
}

for (const key of Object.keys(EXTRA_TASKS)) TASKS[key] = EXTRA_TASKS[key];
