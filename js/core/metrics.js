export function argmaxRow(arr, offset, n) {
  let bi = 0;
  let bv = -Infinity;
  for (let i = 0; i < n; i++) {
    const v = arr[offset + i];
    if (v > bv) {
      bv = v;
      bi = i;
    }
  }
  return bi;
}

export function confusionMatrix(yTrue, yPred, k) {
  const m = [];
  for (let i = 0; i < k; i++) m.push(new Array(k).fill(0));
  for (let i = 0; i < yTrue.length; i++) {
    const t = yTrue[i] | 0;
    const p = yPred[i] | 0;
    if (t >= 0 && t < k && p >= 0 && p < k) m[t][p]++;
  }
  return m;
}

export function perClassReport(cm, labels) {
  const k = cm.length;
  const rows = [];
  let macroP = 0;
  let macroR = 0;
  let macroF = 0;
  let weightedP = 0;
  let weightedR = 0;
  let weightedF = 0;
  let total = 0;
  const support = [];
  for (let i = 0; i < k; i++) support.push(cm[i].reduce((a, b) => a + b, 0));
  for (let i = 0; i < support.length; i++) total += support[i];
  for (let i = 0; i < k; i++) {
    const tp = cm[i][i];
    let fp = 0;
    let fn = 0;
    for (let j = 0; j < k; j++) {
      if (j !== i) {
        fp += cm[j][i];
        fn += cm[i][j];
      }
    }
    const p = tp + fp > 0 ? tp / (tp + fp) : 0;
    const r = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f = p + r > 0 ? (2 * p * r) / (p + r) : 0;
    rows.push({
      label: labels ? labels[i] : String(i),
      index: i,
      precision: p,
      recall: r,
      f1: f,
      support: support[i],
      tp,
      fp,
      fn
    });
    if (support[i] > 0) {
      macroP += p;
      macroR += r;
      macroF += f;
      weightedP += p * support[i];
      weightedR += r * support[i];
      weightedF += f * support[i];
    }
  }
  const present = support.filter((s) => s > 0).length || 1;
  return {
    rows,
    macroPrecision: macroP / present,
    macroRecall: macroR / present,
    macroF1: macroF / present,
    weightedPrecision: total ? weightedP / total : 0,
    weightedRecall: total ? weightedR / total : 0,
    weightedF1: total ? weightedF / total : 0,
    support: total
  };
}

export function accuracyFromMatrix(cm) {
  let correct = 0;
  let total = 0;
  for (let i = 0; i < cm.length; i++) {
    for (let j = 0; j < cm.length; j++) {
      total += cm[i][j];
      if (i === j) correct += cm[i][j];
    }
  }
  return total ? correct / total : 0;
}

export function predictFromProbs(probs, k) {
  const out = new Int32Array(probs.length / k);
  for (let i = 0; i < out.length; i++) out[i] = argmaxRow(probs, i * k, k);
  return out;
}

export function rocCurve(scores, labels, steps) {
  steps = steps || 200;
  const pairs = [];
  for (let i = 0; i < scores.length; i++) pairs.push([scores[i], labels[i] > 0.5 ? 1 : 0]);
  pairs.sort((a, b) => b[0] - a[0]);
  let P = 0;
  let N = 0;
  for (const p of pairs) {
    if (p[1]) P++;
    else N++;
  }
  const points = [{ fpr: 0, tpr: 0, threshold: 1.0001 }];
  let tp = 0;
  let fp = 0;
  let auc = 0;
  let prevFpr = 0;
  for (let i = 0; i < pairs.length; i++) {
    if (pairs[i][1]) tp++;
    else fp++;
    const next = pairs[i + 1];
    if (!next || next[0] !== pairs[i][0]) {
      const fpr = N ? fp / N : 0;
      const tpr = P ? tp / P : 0;
      auc += (fpr - prevFpr) * (tpr + (P ? (tp - (pairs[i][1] ? 1 : 0)) / P : 0)) / 2;
      prevFpr = fpr;
      points.push({ fpr, tpr, threshold: pairs[i][0] });
    }
  }
  points.push({ fpr: 1, tpr: 1, threshold: -1 });
  const step = Math.max(1, Math.floor(points.length / steps));
  const sampled = [];
  for (let i = 0; i < points.length; i += step) sampled.push(points[i]);
  if (sampled[sampled.length - 1] !== points[points.length - 1]) sampled.push(points[points.length - 1]);
  return { points: sampled, auc: Math.max(0, Math.min(1, auc)), positives: P, negatives: N };
}

export function prCurve(scores, labels, steps) {
  const pairs = [];
  for (let i = 0; i < scores.length; i++) pairs.push([scores[i], labels[i] > 0.5 ? 1 : 0]);
  pairs.sort((a, b) => b[0] - a[0]);
  let P = 0;
  for (const p of pairs) if (p[1]) P++;
  let tp = 0;
  let fp = 0;
  let ap = 0;
  let prevR = 0;
  const points = [{ recall: 0, precision: 1, threshold: 1.0001 }];
  for (let i = 0; i < pairs.length; i++) {
    if (pairs[i][1]) tp++;
    else fp++;
    const next = pairs[i + 1];
    if (!next || next[0] !== pairs[i][0]) {
      const r = P ? tp / P : 0;
      const pr = tp + fp ? tp / (tp + fp) : 0;
      ap += (r - prevR) * pr;
      prevR = r;
      points.push({ recall: r, precision: pr, threshold: pairs[i][0] });
    }
  }
  const step = Math.max(1, Math.floor(points.length / (steps || 200)));
  const sampled = [];
  for (let i = 0; i < points.length; i += step) sampled.push(points[i]);
  if (sampled[sampled.length - 1] !== points[points.length - 1]) sampled.push(points[points.length - 1]);
  return { points: sampled, ap, averagePrecision: ap };
}

export function bestF1Threshold(scores, labels) {
  const cands = Array.from(new Set(Array.from(scores))).sort((a, b) => a - b);
  let best = { threshold: 0.5, f1: 0, precision: 0, recall: 0 };
  const stride = Math.max(1, Math.floor(cands.length / 200));
  for (let i = 0; i < cands.length; i += stride) {
    const t = cands[i];
    let tp = 0;
    let fp = 0;
    let fn = 0;
    for (let j = 0; j < scores.length; j++) {
      const pred = scores[j] >= t ? 1 : 0;
      const truth = labels[j] > 0.5 ? 1 : 0;
      if (pred && truth) tp++;
      else if (pred && !truth) fp++;
      else if (!pred && truth) fn++;
    }
    const p = tp + fp ? tp / (tp + fp) : 0;
    const r = tp + fn ? tp / (tp + fn) : 0;
    const f = p + r ? 2 * p * r / (p + r) : 0;
    if (f > best.f1) best = { threshold: t, f1: f, precision: p, recall: r };
  }
  return best;
}

export function binaryReport(scores, labels, threshold) {
  const t = threshold === undefined ? 0.5 : threshold;
  let tp = 0;
  let fp = 0;
  let fn = 0;
  let tn = 0;
  for (let i = 0; i < scores.length; i++) {
    const pred = scores[i] >= t ? 1 : 0;
    const truth = labels[i] > 0.5 ? 1 : 0;
    if (pred && truth) tp++;
    else if (pred && !truth) fp++;
    else if (!pred && truth) fn++;
    else tn++;
  }
  const precision = tp + fp ? tp / (tp + fp) : 0;
  const recall = tp + fn ? tp / (tp + fn) : 0;
  const specificity = tn + fp ? tn / (tn + fp) : 0;
  return {
    threshold: t,
    tp,
    fp,
    fn,
    tn,
    precision,
    recall,
    f1: precision + recall ? 2 * precision * recall / (precision + recall) : 0,
    accuracy: (tp + tn) / Math.max(1, scores.length),
    far: fp + tn ? fp / (fp + tn) : 0,
    frr: tp + fn ? fn / (tp + fn) : 0,
    specificity,
    balancedAccuracy: (recall + specificity) / 2,
    mcc: matthews(tp, fp, fn, tn)
  };
}

export function matthews(tp, fp, fn, tn) {
  const denom = Math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
  return denom > 0 ? (tp * tn - fp * fn) / denom : 0;
}

export function thresholdSweep(scores, labels, points) {
  points = points || 21;
  const out = [];
  for (let i = 0; i < points; i++) {
    const t = i / (points - 1);
    const r = binaryReport(scores, labels, t);
    out.push({ threshold: t, far: r.far, frr: r.frr, precision: r.precision, recall: r.recall, f1: r.f1, accuracy: r.accuracy });
  }
  return out;
}

export function calibration(probs, labels, bins) {
  bins = bins || 10;
  const bucket = [];
  for (let i = 0; i < bins; i++) bucket.push({ count: 0, confSum: 0, accSum: 0 });
  for (let i = 0; i < probs.length; i++) {
    let b = Math.floor(probs[i] * bins);
    if (b >= bins) b = bins - 1;
    if (b < 0) b = 0;
    bucket[b].count++;
    bucket[b].confSum += probs[i];
    bucket[b].accSum += labels[i] > 0.5 ? 1 : 0;
  }
  let ece = 0;
  let mce = 0;
  const n = probs.length || 1;
  const curve = [];
  for (let i = 0; i < bins; i++) {
    const b = bucket[i];
    const conf = b.count ? b.confSum / b.count : 0;
    const acc = b.count ? b.accSum / b.count : 0;
    ece += (b.count / n) * Math.abs(acc - conf);
    mce = Math.max(mce, Math.abs(acc - conf));
    curve.push({ bin: (i + 0.5) / bins, lo: i / bins, hi: (i + 1) / bins, confidence: conf, accuracy: acc, count: b.count });
  }
  let brier = 0;
  for (let i = 0; i < probs.length; i++) {
    const t = labels[i] > 0.5 ? 1 : 0;
    brier += (probs[i] - t) * (probs[i] - t);
  }
  brier /= n;
  let logLoss = 0;
  for (let i = 0; i < probs.length; i++) {
    const t = labels[i] > 0.5 ? 1 : 0;
    const p = Math.min(1 - 1e-9, Math.max(1e-9, probs[i]));
    logLoss += -(t * Math.log(p) + (1 - t) * Math.log(1 - p));
  }
  logLoss /= n;
  return { bins: curve, ece, mce, brier, logLoss };
}

export function multiclassCalibration(probMatrix, labels, k, bins) {
  bins = bins || 10;
  const flatP = [];
  const flatY = [];
  for (let i = 0; i < labels.length; i++) {
    const pred = argmaxRow(probMatrix, i * k, k);
    const conf = probMatrix[i * k + pred];
    flatP.push(conf);
    flatY.push(pred === (labels[i] | 0) ? 1 : 0);
  }
  return calibration(flatP, flatY, bins);
}

export function regressionReport(pred, target) {
  const n = pred.length;
  if (!n) return { mae: 0, rmse: 0, mse: 0, r2: 0, mape: 0, mapeCoverage: 0, wape: 0, maxError: 0, medae: 0 };
  let sae = 0;
  let sse = 0;
  let sape = 0;
  let apeCount = 0;
  let absTargetSum = 0;
  let maxE = 0;
  const absErr = [];
  let mean = 0;
  for (let i = 0; i < n; i++) mean += target[i];
  mean /= n;
  let absMean = 0;
  for (let i = 0; i < n; i++) absMean += Math.abs(target[i]);
  absMean /= n;
  const apeFloor = Math.max(1e-9, absMean * 0.05);
  let ssTot = 0;
  for (let i = 0; i < n; i++) {
    const e = pred[i] - target[i];
    const ae = Math.abs(e);
    sae += ae;
    sse += e * e;
    absErr.push(ae);
    if (ae > maxE) maxE = ae;
    absTargetSum += Math.abs(target[i]);
    if (Math.abs(target[i]) > apeFloor) {
      sape += ae / Math.abs(target[i]);
      apeCount++;
    }
    const d = target[i] - mean;
    ssTot += d * d;
  }
  absErr.sort((a, b) => a - b);
  const mae = sae / n;
  const rmse = Math.sqrt(sse / n);
  return {
    mae,
    rmse,
    mse: sse / n,
    r2: ssTot > 0 ? 1 - sse / ssTot : 0,
    mape: apeCount ? 100 * sape / apeCount : 0,
    mapeCoverage: apeCount / n,
    wape: absTargetSum > 0 ? 100 * sae / absTargetSum : 0,
    maxError: maxE,
    medae: absErr[Math.floor(n / 2)] || 0
  };
}

export function psnr(a, b, peak) {
  peak = peak === undefined ? 1 : peak;
  let se = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    se += d * d;
  }
  const mse = se / Math.max(1, a.length);
  if (mse <= 1e-12) return 100;
  return 10 * Math.log10((peak * peak) / mse);
}

export function ssim(a, b, w, h, channels) {
  channels = channels || 1;
  const C1 = 0.01 * 0.01;
  const C2 = 0.03 * 0.03;
  const win = 3;
  const half = 1;
  let total = 0;
  let count = 0;
  for (let c = 0; c < channels; c++) {
    for (let y = half; y < h - half; y++) {
      for (let x = half; x < w - half; x++) {
        let ma = 0;
        let mb = 0;
        let n = 0;
        for (let dy = -half; dy <= half; dy++) {
          for (let dx = -half; dx <= half; dx++) {
            const idx = (c * h + y + dy) * w + x + dx;
            ma += a[idx];
            mb += b[idx];
            n++;
          }
        }
        ma /= n;
        mb /= n;
        let va = 0;
        let vb = 0;
        let cov = 0;
        for (let dy = -half; dy <= half; dy++) {
          for (let dx = -half; dx <= half; dx++) {
            const idx = (c * h + y + dy) * w + x + dx;
            va += (a[idx] - ma) * (a[idx] - ma);
            vb += (b[idx] - mb) * (b[idx] - mb);
            cov += (a[idx] - ma) * (b[idx] - mb);
          }
        }
        va /= n - 1 || 1;
        vb /= n - 1 || 1;
        cov /= n - 1 || 1;
        const s = ((2 * ma * mb + C1) * (2 * cov + C2)) / ((ma * ma + mb * mb + C1) * (va + vb + C2));
        total += s;
        count++;
      }
    }
  }
  return count ? total / count : 1;
}

export function siSdr(est, ref) {
  let dot = 0;
  let rr = 0;
  for (let i = 0; i < ref.length; i++) {
    dot += est[i] * ref[i];
    rr += ref[i] * ref[i];
  }
  if (rr < 1e-12) return -100;
  const alpha = dot / rr;
  let num = 0;
  let den = 0;
  for (let i = 0; i < ref.length; i++) {
    const target = alpha * ref[i];
    const noise = est[i] - target;
    num += target * target;
    den += noise * noise;
  }
  if (den < 1e-12) return 60;
  return 10 * Math.log10(num / den);
}

export function segmentalSnr(est, ref, frameLen) {
  frameLen = frameLen || 160;
  let total = 0;
  let count = 0;
  for (let s = 0; s + frameLen <= ref.length; s += frameLen) {
    let sig = 0;
    let noi = 0;
    for (let i = 0; i < frameLen; i++) {
      sig += ref[s + i] * ref[s + i];
      const e = est[s + i] - ref[s + i];
      noi += e * e;
    }
    if (sig < 1e-8) continue;
    total += 10 * Math.log10(sig / Math.max(noi, 1e-10));
    count++;
  }
  return count ? total / count : -100;
}

export function logSpectralDistance(est, ref, spectra) {
  let total = 0;
  for (let i = 0; i < est.length; i++) {
    const d = Math.log10(Math.max(est[i], 1e-10)) - Math.log10(Math.max(ref[i], 1e-10));
    total += d * d;
  }
  return Math.sqrt(total / Math.max(1, est.length));
}

export function pesqApprox(enhanced, clean, noisy, sampleRate) {
  const sisdr = siSdr(enhanced, clean);
  const seg = segmentalSnr(enhanced, clean, Math.round((sampleRate || 16000) * 0.01));
  const inputSeg = segmentalSnr(noisy, clean, Math.round((sampleRate || 16000) * 0.01));
  const improvement = Math.max(0, seg - inputSeg);
  let mos = 1 + 0.13 * sisdr + 0.06 * improvement;
  if (!isFinite(mos)) mos = 1;
  mos = Math.max(1, Math.min(4.5, mos));
  return { mos, sisdr, segSnr: seg, inputSegSnr: inputSeg, pesqLike: (mos - 1) / 3.5 };
}

export function iou(a, b) {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w);
  const y2 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = a.w * a.h + b.w * b.h - inter;
  return union > 0 ? inter / union : 0;
}

export function detectionReport(predBoxes, trueBoxes, iouThreshold) {
  iouThreshold = iouThreshold || 0.5;
  const pairs = [];
  for (const p of predBoxes) {
    for (const t of trueBoxes) {
      if (p.image !== t.image) continue;
      pairs.push({ p, t, iou: iou(p.box, t.box) });
    }
  }
  pairs.sort((a, b) => b.p.score - a.p.score || b.iou - a.iou);
  const matched = new Set();
  let tp = 0;
  let fp = 0;
  const curve = [];
  const total = trueBoxes.length;
  for (const pr of pairs) {
    if (pr.iou >= iouThreshold && !matched.has(pr.t)) {
      matched.add(pr.t);
      tp++;
    } else if (pr.iou >= iouThreshold) {
      fp++;
    } else {
      fp++;
    }
    curve.push({ recall: total ? tp / total : 0, precision: tp + fp ? tp / (tp + fp) : 0, score: pr.p.score });
  }
  let ap = 0;
  let prevR = 0;
  for (const c of curve) {
    ap += (c.recall - prevR) * c.precision;
    prevR = c.recall;
  }
  return {
    ap,
    mAP: ap,
    precision: tp + fp ? tp / (tp + fp) : 0,
    recall: total ? tp / total : 0,
    tp,
    fp,
    fn: total - tp,
    curve
  };
}

export function segmentationReport(predMask, trueMask, threshold) {
  threshold = threshold === undefined ? 0.5 : threshold;
  let tp = 0;
  let fp = 0;
  let fn = 0;
  let tn = 0;
  for (let i = 0; i < trueMask.length; i++) {
    const p = predMask[i] >= threshold ? 1 : 0;
    const t = trueMask[i] > 0.5 ? 1 : 0;
    if (p && t) tp++;
    else if (p && !t) fp++;
    else if (!p && t) fn++;
    else tn++;
  }
  const inter = tp;
  const union = tp + fp + fn;
  return {
    iou: union ? inter / union : 1,
    dice: (2 * tp + fp + fn) ? (2 * tp) / (2 * tp + fp + fn) : 1,
    pixelAccuracy: (tp + tn) / Math.max(1, trueMask.length),
    precision: tp + fp ? tp / (tp + fp) : 0,
    recall: tp + fn ? tp / (tp + fn) : 0,
    tp,
    fp,
    fn,
    tn
  };
}

export function onsetF1(predTimes, trueTimes, tolerance) {
  tolerance = tolerance || 0.05;
  if (!trueTimes.length) return { f1: predTimes.length ? 0 : 1, precision: 0, recall: 0 };
  const used = new Array(trueTimes.length).fill(false);
  let tp = 0;
  for (const p of predTimes) {
    let best = -1;
    let bestD = Infinity;
    for (let i = 0; i < trueTimes.length; i++) {
      if (used[i]) continue;
      const d = Math.abs(p - trueTimes[i]);
      if (d <= tolerance && d < bestD) {
        bestD = d;
        best = i;
      }
    }
    if (best >= 0) {
      used[best] = true;
      tp++;
    }
  }
  const precision = predTimes.length ? tp / predTimes.length : 0;
  const recall = tp / trueTimes.length;
  return { f1: precision + recall ? 2 * precision * recall / (precision + recall) : 0, precision, recall, tp };
}

export function peaksFromEnvelope(env, frameRate, threshold, minGap) {
  const peaks = [];
  const t = threshold === undefined ? 0.3 : threshold;
  const gap = minGap || 3;
  for (let i = 1; i < env.length - 1; i++) {
    if (env[i] > env[i - 1] && env[i] >= env[i + 1] && env[i] > t) {
      if (!peaks.length || i - peaks[peaks.length - 1].index >= gap) peaks.push({ index: i, time: i / frameRate, value: env[i] });
    }
  }
  return peaks;
}

export function anomalyReport(scores, labels) {
  const roc = rocCurve(scores, labels);
  const sorted = Array.from(scores, (s, i) => ({ s, l: labels[i] })).sort((a, b) => b.s - a.s);
  let anomalies = 0;
  for (const l of labels) if (l > 0.5) anomalies++;
  const ks = [1, 5, 10, 20];
  const precisionAt = {};
  for (const k of ks) {
    let tp = 0;
    for (let i = 0; i < Math.min(k, sorted.length); i++) if (sorted[i].l > 0.5) tp++;
    precisionAt['p@' + k] = Math.min(k, sorted.length) ? tp / Math.min(k, sorted.length) : 0;
  }
  const targetFpr = 0.05;
  let point = roc.points[0];
  for (const p of roc.points) {
    if (p.fpr <= targetFpr) point = p;
    else break;
  }
  return {
    auroc: roc.auc,
    ap: prCurve(scores, labels).ap,
    precisionAt,
    thresholdAt5Fpr: point.threshold,
    recallAt5Fpr: point.tpr,
    anomalies,
    total: labels.length,
    roc
  };
}

export function compressionReport(bitsPerSymbol, baselineBitsPerSymbol, symbolCount, bytes) {
  const modelBytes = (bitsPerSymbol * symbolCount) / 8;
  const baselineBytes = (baselineBitsPerSymbol * symbolCount) / 8;
  return {
    bitsPerSymbol,
    baselineBitsPerSymbol,
    bytesPerSymbol: bitsPerSymbol / 8,
    estimatedSize: modelBytes,
    baselineSize: baselineBytes,
    ratioVsBaseline: baselineBytes > 0 ? baselineBytes / Math.max(1, modelBytes) : 0,
    ratioVsRaw: bytes > 0 ? bytes / Math.max(1, modelBytes) : 0,
    savingPercent: bytes > 0 ? 100 * (1 - modelBytes / bytes) : 0
  };
}

export function perplexity(totalLoss, tokenCount) {
  return Math.exp(Math.min(20, totalLoss / Math.max(1, tokenCount)));
}

export function retrievalReport(queryEmb, candEmb, labels, ks, skipDiagonal) {
  ks = ks || [1, 5, 10];
  const out = {};
  for (const k of ks) out['recall@' + k] = 0;
  let mapSum = 0;
  let mrrSum = 0;
  const n = queryEmb.length;
  for (let q = 0; q < n; q++) {
    const scored = [];
    for (let c = 0; c < candEmb.length; c++) {
      if (skipDiagonal && c === q && candEmb === queryEmb) continue;
      let d = 0;
      for (let i = 0; i < queryEmb[q].length; i++) {
        const diff = queryEmb[q][i] - candEmb[c][i];
        d += diff * diff;
      }
      scored.push({ c, d });
    }
    scored.sort((a, b) => a.d - b.d);
    const rel = new Set(labels[q]);
    let hits = 0;
    let firstRank = 0;
    let apSum = 0;
    for (let r = 0; r < scored.length; r++) {
      if (rel.has(scored[r].c)) {
        hits++;
        if (!firstRank) firstRank = r + 1;
        apSum += hits / (r + 1);
      }
      for (const k of ks) {
        if (r + 1 === k && hits > 0) out['recall@' + k] += 1;
      }
    }
    for (const k of ks) {
      if (scored.length < k && hits > 0) out['recall@' + k] += 1;
    }
    mapSum += rel.size ? apSum / rel.size : 0;
    mrrSum += firstRank ? 1 / firstRank : 0;
  }
  for (const k of ks) out['recall@' + k] /= Math.max(1, n);
  out.map = mapSum / Math.max(1, n);
  out.mrr = mrrSum / Math.max(1, n);
  return out;
}

export function pitchReport(predHz, trueHz) {
  let within50 = 0;
  let within20 = 0;
  let centErrSum = 0;
  let centErrAbs = [];
  let octaveErr = 0;
  let n = 0;
  for (let i = 0; i < trueHz.length; i++) {
    if (trueHz[i] <= 0) continue;
    n++;
    const cents = 1200 * Math.log2(Math.max(1e-6, predHz[i]) / trueHz[i]);
    const ac = Math.abs(cents);
    centErrSum += cents;
    centErrAbs.push(ac);
    if (ac <= 50) within50++;
    if (ac <= 20) within20++;
    if (Math.abs(Math.abs(cents) - 1200) < 60 || Math.abs(Math.abs(cents) - 2400) < 60) octaveErr++;
  }
  centErrAbs.sort((a, b) => a - b);
  return {
    rpa: n ? within50 / n : 0,
    strictRpa: n ? within20 / n : 0,
    meanCentError: n ? centErrSum / n : 0,
    medianAbsCentError: centErrAbs.length ? centErrAbs[Math.floor(centErrAbs.length / 2)] : 0,
    meanAbsCentError: n ? centErrAbs.reduce((a, b) => a + b, 0) / n : 0,
    octaveErrorRate: n ? octaveErr / n : 0,
    voicedFrames: n
  };
}

export function sequenceTaggingReport(predLabels, trueLabels, labels, seqLens) {
  const k = labels.length;
  const cm = confusionMatrix(
    flattenSeq(trueLabels, seqLens),
    flattenSeq(predLabels, seqLens),
    k
  );
  const rep = perClassReport(cm, labels);
  const total = flattenSeq(trueLabels, seqLens).length;
  let correct = 0;
  const tp = flattenSeq(trueLabels, seqLens);
  const pp = flattenSeq(predLabels, seqLens);
  for (let i = 0; i < total; i++) if (tp[i] === pp[i]) correct++;
  return {
    tokenAccuracy: total ? correct / total : 0,
    macroF1: rep.macroF1,
    weightedF1: rep.weightedF1,
    rows: rep.rows,
    confusion: cm
  };
}

function flattenSeq(arr, lens) {
  const out = [];
  for (let i = 0; i < arr.length; i++) {
    const L = lens ? lens[i] : arr[i].length;
    for (let j = 0; j < L; j++) out.push(arr[i][j]);
  }
  return out;
}

export function forecastReport(predSeries, trueSeries, horizons) {
  const perHorizon = [];
  for (let h = 0; h < horizons; h++) {
    let sae = 0;
    let sse = 0;
    let n = 0;
    for (let i = 0; i < predSeries.length; i++) {
      if (predSeries[i].length <= h) continue;
      const e = predSeries[i][h] - trueSeries[i][h];
      sae += Math.abs(e);
      sse += e * e;
      n++;
    }
    perHorizon.push({
      horizon: h + 1,
      mae: n ? sae / n : 0,
      rmse: n ? Math.sqrt(sse / n) : 0,
      n
    });
  }
  const flatP = [];
  const flatT = [];
  for (let i = 0; i < predSeries.length; i++) {
    for (let h = 0; h < predSeries[i].length; h++) {
      flatP.push(predSeries[i][h]);
      flatT.push(trueSeries[i][h]);
    }
  }
  return Object.assign(regressionReport(flatP, flatT), { perHorizon });
}

export function correlation(a, b) {
  const n = a.length;
  if (!n) return 0;
  let ma = 0;
  let mb = 0;
  for (let i = 0; i < n; i++) {
    ma += a[i];
    mb += b[i];
  }
  ma /= n;
  mb /= n;
  let num = 0;
  let da = 0;
  let db = 0;
  for (let i = 0; i < n; i++) {
    num += (a[i] - ma) * (b[i] - mb);
    da += (a[i] - ma) * (a[i] - ma);
    db += (b[i] - mb) * (b[i] - mb);
  }
  return da > 0 && db > 0 ? num / Math.sqrt(da * db) : 0;
}

export function entropy(counts) {
  let total = 0;
  for (const c of counts) total += c;
  if (!total) return 0;
  let h = 0;
  for (const c of counts) {
    if (!c) continue;
    const p = c / total;
    h -= p * Math.log2(p);
  }
  return h;
}

export function classBalance(counts) {
  const total = counts.reduce((a, b) => a + b, 0) || 1;
  const h = entropy(counts);
  const maxH = Math.log2(Math.max(2, counts.length));
  const imbalance = counts.length ? Math.max(...counts) / Math.max(1, Math.min(...counts.filter((c) => c >= 0))) : 1;
  return {
    counts,
    total,
    shares: counts.map((c) => c / total),
    entropy: h,
    normalizedEntropy: maxH ? h / maxH : 0,
    imbalanceRatio: isFinite(imbalance) ? imbalance : Infinity,
    emptyClasses: counts.map((c, i) => (c === 0 ? i : -1)).filter((i) => i >= 0)
  };
}
