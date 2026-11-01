import { Rng } from '../core/tensor.js';

export function shuffleArray(arr, rng) {
  const a = Array.from(arr);
  for (let i = a.length - 1; i > 0; i--) {
    const j = rng.int(i + 1);
    const t = a[i];
    a[i] = a[j];
    a[j] = t;
  }
  return a;
}

export function randomSplit(n, ratios, seed) {
  const rng = new Rng(seed === undefined ? 12345 : seed);
  const idx = shuffleArray(Array.from({ length: n }, (_, i) => i), rng);
  return cutRatios(idx, ratios);
}

function cutRatios(idx, ratios) {
  const r = normalizeRatios(ratios);
  const n = idx.length;
  const nTrain = Math.max(1, Math.round(n * r[0]));
  const nVal = r[1] > 0 ? Math.max(1, Math.round(n * r[1])) : 0;
  const train = Int32Array.from(idx.slice(0, nTrain));
  const val = Int32Array.from(idx.slice(nTrain, nTrain + nVal));
  const test = Int32Array.from(idx.slice(nTrain + nVal));
  return { train, val, test };
}

export function normalizeRatios(ratios) {
  const r = (ratios || [0.7, 0.15, 0.15]).slice();
  while (r.length < 3) r.push(0);
  const sum = r.reduce((a, b) => a + b, 0) || 1;
  return r.map((v) => v / sum);
}

export function stratifiedSplit(labels, ratios, seed) {
  const rng = new Rng(seed === undefined ? 12345 : seed);
  const r = normalizeRatios(ratios);
  const byClass = new Map();
  for (let i = 0; i < labels.length; i++) {
    const key = String(labels[i]);
    if (!byClass.has(key)) byClass.set(key, []);
    byClass.get(key).push(i);
  }
  const train = [];
  const val = [];
  const test = [];
  for (const [, members] of byClass) {
    const shuffled = shuffleArray(members, rng);
    const n = shuffled.length;
    let nTrain = Math.round(n * r[0]);
    let nVal = Math.round(n * r[1]);
    if (n >= 3) {
      nTrain = Math.max(1, Math.min(n - 2, nTrain));
      nVal = Math.max(1, Math.min(n - nTrain - 1, nVal));
    } else if (n === 2) {
      nTrain = 1;
      nVal = 1;
    } else {
      nTrain = 1;
      nVal = 0;
    }
    for (let i = 0; i < n; i++) {
      if (i < nTrain) train.push(shuffled[i]);
      else if (i < nTrain + nVal) val.push(shuffled[i]);
      else test.push(shuffled[i]);
    }
  }
  return {
    train: Int32Array.from(train),
    val: Int32Array.from(val),
    test: Int32Array.from(test)
  };
}

export function groupSplit(groups, ratios, seed) {
  const rng = new Rng(seed === undefined ? 12345 : seed);
  const r = normalizeRatios(ratios);
  const unique = Array.from(new Set(Array.from(groups).map((g) => String(g))));
  const shuffled = shuffleArray(unique, rng);
  const nTrain = Math.max(1, Math.round(shuffled.length * r[0]));
  const nVal = r[1] > 0 ? Math.max(1, Math.round(shuffled.length * r[1])) : 0;
  const assign = new Map();
  shuffled.forEach((g, i) => {
    if (i < nTrain) assign.set(g, 0);
    else if (i < nTrain + nVal) assign.set(g, 1);
    else assign.set(g, 2);
  });
  const train = [];
  const val = [];
  const test = [];
  for (let i = 0; i < groups.length; i++) {
    const bucket = assign.get(String(groups[i]));
    if (bucket === 0) train.push(i);
    else if (bucket === 1) val.push(i);
    else test.push(i);
  }
  return {
    train: Int32Array.from(train),
    val: Int32Array.from(val),
    test: Int32Array.from(test)
  };
}

export function temporalSplit(n, ratios) {
  const r = normalizeRatios(ratios);
  const nTrain = Math.max(1, Math.round(n * r[0]));
  const nVal = Math.max(0, Math.round(n * r[1]));
  const train = [];
  const val = [];
  const test = [];
  for (let i = 0; i < n; i++) {
    if (i < nTrain) train.push(i);
    else if (i < nTrain + nVal) val.push(i);
    else test.push(i);
  }
  return { train: Int32Array.from(train), val: Int32Array.from(val), test: Int32Array.from(test) };
}

export function kFold(n, k, seed, stratifyLabels) {
  k = Math.max(2, Math.min(20, k || 5));
  const rng = new Rng(seed === undefined ? 999 : seed);
  const assign = new Int32Array(n).fill(-1);
  if (stratifyLabels) {
    const byClass = new Map();
    for (let i = 0; i < n; i++) {
      const key = String(stratifyLabels[i]);
      if (!byClass.has(key)) byClass.set(key, []);
      byClass.get(key).push(i);
    }
    for (const [, members] of byClass) {
      const shuffled = shuffleArray(members, rng);
      shuffled.forEach((m, j) => {
        assign[m] = j % k;
      });
    }
  }
  const order = shuffleArray(Array.from({ length: n }, (_, i) => i), rng);
  let cursor = 0;
  for (const i of order) {
    if (assign[i] < 0) {
      assign[i] = cursor % k;
      cursor++;
    }
  }
  const folds = [];
  for (let f = 0; f < k; f++) {
    const val = [];
    const train = [];
    for (let i = 0; i < n; i++) (assign[i] === f ? val : train).push(i);
    folds.push({ val: Int32Array.from(val), train: Int32Array.from(train) });
  }
  return folds;
}

export function splitSummary(splits, labels) {
  const out = {};
  for (const key of ['train', 'val', 'test']) {
    const idx = splits[key] || [];
    const counts = {};
    for (const i of idx) {
      const l = labels ? String(labels[i]) : '0';
      counts[l] = (counts[l] || 0) + 1;
    }
    out[key] = { n: idx.length, counts };
  }
  return out;
}
