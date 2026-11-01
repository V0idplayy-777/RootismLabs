import { sha256, sha256Floats, sha256Text, shortHash } from '../core/hash.js';
import { Rng, shapeSize } from '../core/tensor.js';
import { stratifiedSplit, randomSplit, groupSplit, temporalSplit, normalizeRatios } from './split.js';
import { classBalance } from '../core/metrics.js';

export const MODALITIES = {
  text: { label: 'Text', unit: 'sample' },
  audio: { label: 'Audio', unit: 'clip' },
  image: { label: 'Image', unit: 'image' },
  sensor: { label: 'Sensor', unit: 'window' },
  series: { label: 'Time series', unit: 'window' },
  tabular: { label: 'Tabular', unit: 'row' },
  bytes: { label: 'Byte stream', unit: 'sequence' }
};

let seq = 0;

export function newId(prefix) {
  seq += 1;
  return (prefix || 'id') + '_' + Date.now().toString(36) + '_' + seq.toString(36) + '_' + Math.floor(Math.random() * 1e6).toString(36);
}

export function createDataset(opts) {
  const now = Date.now();
  const ds = {
    id: opts.id || newId('ds'),
    name: opts.name || 'Untitled dataset',
    slug: opts.slug || slugify(opts.name || 'untitled'),
    description: opts.description || '',
    modality: opts.modality || 'tabular',
    task: opts.task || 'multiclass',
    version: opts.version || 1,
    source: opts.source || 'custom',
    presetKey: opts.presetKey || null,
    createdAt: now,
    updatedAt: now,
    license: opts.license || { id: 'unset', name: 'Not specified', allowsCommercial: false, notes: '' },
    x: opts.x,
    xShape: opts.xShape,
    x2: opts.x2 || null,
    y: opts.y || new Float32Array(opts.x ? opts.x.length / shapeSize(opts.xShape) : 0),
    yShape: opts.yShape || [1],
    lengths: opts.lengths || null,
    groups: opts.groups || null,
    boxes: opts.boxes || null,
    meta: opts.meta || null,
    wave: opts.wave || null,
    images: opts.images || null,
    labels: opts.labels || [],
    featureSpec: opts.featureSpec || null,
    splits: opts.splits || null,
    splitMethod: opts.splitMethod || 'stratified',
    splitRatios: opts.splitRatios || [0.7, 0.15, 0.15],
    splitSeed: opts.splitSeed === undefined ? 20260101 : opts.splitSeed,
    provenance: opts.provenance || {},
    config: opts.config || {},
    notes: opts.notes || '',
    columns: opts.columns || null,
    sampleRate: opts.sampleRate || null,
    units: opts.units || null
  };
  ds.count = ds.x ? ds.x.length / shapeSize(ds.xShape) : 0;
  if (opts.dedupe !== false) ds.removedDuplicates = dedupeSamples(ds);
  if (!ds.splits) ds.splits = buildSplits(ds);
  ds.digest = computeDigest(ds);
  ds.stats = computeStats(ds);
  return ds;
}

const CLASS_LABELLED_TASKS = { multiclass: 1, binary: 1, pitch_class: 1, anomaly: 1, multilabel: 1 };

export function hasDiscreteLabels(ds) {
  if (!ds.y || !CLASS_LABELLED_TASKS[ds.task]) return false;
  const yPer = shapeSize(ds.yShape || [1]);
  if (yPer === 1) return true;
  return ds.task === 'multilabel';
}

function rowFingerprint(arr, i, per) {
  let h = 2166136261;
  const stride = Math.max(1, Math.floor(per / 192));
  for (let f = 0; f < per; f += stride) {
    const v = Math.round(arr[i * per + f] * 1e6) | 0;
    h = Math.imul(h ^ (v & 0xffff), 16777619) >>> 0;
    h = Math.imul(h ^ ((v >>> 16) & 0xffff), 16777619) >>> 0;
  }
  return h;
}

function rowsIdentical(arr, a, b, per) {
  for (let f = 0; f < per; f++) {
    if (arr[a * per + f] !== arr[b * per + f]) return false;
  }
  return true;
}

function filterTyped(arr, keep, per, Ctor) {
  if (!arr) return null;
  const out = new Ctor(keep.length * per);
  keep.forEach((src, i) => out.set(arr.subarray(src * per, (src + 1) * per), i * per));
  return out;
}

export function dedupeSamples(ds) {
  const per = shapeSize(ds.xShape || [1]);
  if (!ds.x || !per || ds.count < 2) return 0;
  const buckets = new Map();
  const keep = [];
  for (let i = 0; i < ds.count; i++) {
    const fp = rowFingerprint(ds.x, i, per);
    const list = buckets.get(fp);
    if (!list) {
      buckets.set(fp, [i]);
      keep.push(i);
      continue;
    }
    let dupe = false;
    for (const j of list) {
      if (rowsIdentical(ds.x, i, j, per)) {
        dupe = true;
        break;
      }
    }
    if (dupe) continue;
    list.push(i);
    keep.push(i);
  }
  const removed = ds.count - keep.length;
  if (!removed) return 0;
  const yPer = shapeSize(ds.yShape || [1]);
  ds.x = filterTyped(ds.x, keep, per, Float32Array);
  if (ds.x2) ds.x2 = filterTyped(ds.x2, keep, per, Float32Array);
  if (ds.y) ds.y = filterTyped(ds.y, keep, yPer, Float32Array);
  if (ds.lengths) ds.lengths = filterTyped(ds.lengths, keep, 1, Int32Array);
  if (ds.groups) ds.groups = filterTyped(ds.groups, keep, 1, Int32Array);
  for (const field of ['meta', 'boxes', 'wave', 'images']) {
    if (Array.isArray(ds[field])) ds[field] = keep.map((i) => ds[field][i]);
  }
  ds.count = keep.length;
  return removed;
}

export function slugify(s) {
  return String(s).toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 48) || 'dataset';
}

export function buildSplits(ds, method, ratios, seed) {
  const m = method || ds.splitMethod;
  const r = ratios || ds.splitRatios;
  const s = seed === undefined ? ds.splitSeed : seed;
  const n = ds.count;
  ds.splitMethod = m;
  ds.splitRatios = normalizeRatios(r);
  ds.splitSeed = s;
  if (n === 0) return { train: new Int32Array(0), val: new Int32Array(0), test: new Int32Array(0) };
  let splits;
  if (m === 'group' && ds.groups) {
    splits = groupSplit(ds.groups, r, s);
  } else if (m === 'temporal') {
    splits = temporalSplit(n, r);
  } else if (m === 'random') {
    splits = randomSplit(n, r, s);
  } else {
    const labels = labelVector(ds);
    const unique = new Set(Array.from(labels)).size;
    splits = unique > 1 && n >= 9 ? stratifiedSplit(labels, r, s) : randomSplit(n, r, s);
  }
  if (!splits.val.length && n > 4) {
    const moved = splits.train.subarray(splits.train.length - Math.max(1, Math.floor(n * 0.1)));
    splits.val = Int32Array.from(moved);
    splits.train = Int32Array.from(splits.train.subarray(0, splits.train.length - moved.length));
  }
  if (!splits.test.length && splits.val.length > 2) {
    const moved = splits.val.subarray(splits.val.length - Math.max(1, Math.floor(splits.val.length / 2)));
    splits.test = Int32Array.from(moved);
    splits.val = Int32Array.from(splits.val.subarray(0, splits.val.length - moved.length));
  }
  ds.splits = splits;
  ds.digest = computeDigest(ds);
  ds.stats = computeStats(ds);
  return splits;
}

export function labelVector(ds) {
  const n = ds.count;
  const per = shapeSize(ds.yShape || [1]);
  const out = new Int32Array(n);
  for (let i = 0; i < n; i++) out[i] = Math.round(ds.y[i * per]);
  return out;
}

export function computeDigest(ds) {
  const parts = [];
  parts.push('v1');
  parts.push(String(ds.count));
  parts.push(JSON.stringify(ds.xShape));
  parts.push(JSON.stringify(ds.yShape));
  parts.push(JSON.stringify(ds.labels));
  parts.push(ds.splitMethod + '|' + ds.splitSeed + '|' + JSON.stringify(ds.splitRatios));
  parts.push(JSON.stringify(ds.featureSpec || null));
  if (ds.splits) {
    parts.push(splitSignature(ds.splits.train));
    parts.push(splitSignature(ds.splits.val));
    parts.push(splitSignature(ds.splits.test));
  }
  const specHash = sha256Text(parts.join('\n'));
  const xHash = ds.x ? sha256Floats(ds.x, Math.max(1, Math.floor(ds.x.length / 200000))) : sha256Text('empty');
  const yHash = ds.y ? sha256Floats(ds.y, 1) : sha256Text('empty');
  const combined = sha256Text(specHash + '|' + xHash + '|' + yHash);
  return {
    specHash,
    contentHash: combined,
    featureHash: xHash,
    labelHash: yHash,
    short: shortHash(combined, 12),
    bytes: byteSize(ds),
    samples: ds.count
  };
}

function splitSignature(arr) {
  if (!arr || !arr.length) return 'none';
  let h = 2166136261;
  for (let i = 0; i < arr.length; i++) {
    h ^= arr[i];
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h.toString(16) + ':' + arr.length;
}

export function byteSize(ds) {
  let b = 0;
  if (ds.x) b += ds.x.byteLength;
  if (ds.x2) b += ds.x2.byteLength;
  if (ds.y) b += ds.y.byteLength;
  if (ds.lengths) b += ds.lengths.byteLength;
  if (ds.groups) b += ds.groups.byteLength;
  if (ds.wave) {
    for (const w of ds.wave) {
      if (!w) continue;
      if (w.clean) b += w.clean.byteLength;
      if (w.noisy) b += w.noisy.byteLength;
    }
  }
  if (ds.images) {
    for (const im of ds.images) if (im && im.image) b += im.image.byteLength;
  }
  return b;
}

export function computeStats(ds) {
  const n = ds.count;
  const per = shapeSize(ds.xShape);
  const stats = {
    samples: n,
    featureDim: per,
    xShape: ds.xShape.slice(),
    yShape: (ds.yShape || [1]).slice(),
    splits: ds.splits ? { train: ds.splits.train.length, val: ds.splits.val.length, test: ds.splits.test.length } : null,
    bytes: byteSize(ds),
    labelNames: ds.labels ? ds.labels.slice() : []
  };
  if (hasDiscreteLabels(ds)) {
    const yPer = shapeSize(ds.yShape || [1]);
    if (yPer === 1) {
      const counts = new Map();
      for (let i = 0; i < n; i++) {
        const v = Math.round(ds.y[i]);
        counts.set(v, (counts.get(v) || 0) + 1);
      }
      const keys = Array.from(counts.keys()).sort((a, b) => a - b);
      stats.classCounts = keys.map((k) => counts.get(k));
      stats.classKeys = keys;
      stats.balance = classBalance(stats.classCounts);
      if (ds.labels && ds.labels.length) {
        stats.classNames = ds.labels.map((l, i) => ({ name: l, count: counts.get(i) || 0 }));
      }
    } else {
      const sums = new Array(yPer).fill(0);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < yPer; j++) sums[j] += ds.y[i * yPer + j] > 0.5 ? 1 : 0;
      }
      stats.classCounts = sums;
      stats.balance = classBalance(sums);
    }
  }
  const stride = Math.max(1, Math.floor(per / 64));
  const dims = [];
  for (let f = 0; f < per; f += stride) {
    let sum = 0;
    let sumSq = 0;
    let min = Infinity;
    let max = -Infinity;
    let zeros = 0;
    for (let i = 0; i < n; i++) {
      const v = ds.x[i * per + f];
      sum += v;
      sumSq += v * v;
      if (v < min) min = v;
      if (v > max) max = v;
      if (v === 0) zeros++;
    }
    const mean = n ? sum / n : 0;
    dims.push({
      index: f,
      mean,
      std: Math.sqrt(Math.max(0, (n ? sumSq / n : 0) - mean * mean)),
      min: n ? min : 0,
      max: n ? max : 0,
      zeroFraction: n ? zeros / n : 0
    });
  }
  stats.featureDims = dims;
  let nonFinite = 0;
  const checkStride = Math.max(1, Math.floor(ds.x.length / 200000));
  for (let i = 0; i < ds.x.length; i += checkStride) if (!isFinite(ds.x[i])) nonFinite++;
  stats.nonFinite = nonFinite;
  let sparsityZeros = 0;
  let sparsityTotal = 0;
  for (let i = 0; i < Math.min(ds.x.length, 400000); i++) {
    sparsityTotal++;
    if (ds.x[i] === 0) sparsityZeros++;
  }
  stats.sparsity = sparsityTotal ? sparsityZeros / sparsityTotal : 0;
  return stats;
}

export function qualityReport(ds) {
  const issues = [];
  const stats = ds.stats || computeStats(ds);
  const n = ds.count;
  const add = (severity, code, message, detail) => issues.push({ severity, code, message, detail });

  if (n === 0) add('error', 'empty', 'Dataset has no samples', 'Add samples before training.');

  if (stats.nonFinite > 0) {
    add('error', 'non_finite', stats.nonFinite + ' non-finite feature values detected', 'NaN or Infinity values will make training diverge.');
  }

  const bal = hasDiscreteLabels(ds) ? stats.balance : null;
  if (bal) {
    if (bal.emptyClasses.length) {
      add('error', 'empty_class', 'Empty classes: ' + bal.emptyClasses.map((i) => (ds.labels && ds.labels[i]) || i).join(', '),
        'A class with zero samples cannot be learned or evaluated. Remove it or add samples.');
    }
    if (isFinite(bal.imbalanceRatio) && bal.imbalanceRatio > 20) {
      add('warn', 'imbalance', 'Class imbalance ratio is ' + bal.imbalanceRatio.toFixed(1) + ':1',
        'Enable balanced class weights or a balanced sampler, and report macro-F1 rather than accuracy.');
    } else if (isFinite(bal.imbalanceRatio) && bal.imbalanceRatio > 5) {
      add('info', 'imbalance_mild', 'Class imbalance ratio is ' + bal.imbalanceRatio.toFixed(1) + ':1',
        'Watch per-class recall in the evaluation panel.');
    }
    if (bal.normalizedEntropy < 0.55 && bal.counts.length > 2) {
      add('info', 'low_entropy', 'Label distribution is concentrated (normalised entropy ' + bal.normalizedEntropy.toFixed(2) + ')',
        'A constant predictor would already score well on accuracy. Prefer macro-F1 or AUC as the monitored metric.');
    }
  }

  if (ds.splits) {
    const s = ds.splits;
    if (!s.val.length) add('error', 'no_val', 'No validation split', 'Validation is required for early stopping and honest reporting.');
    if (!s.test.length) add('error', 'no_test', 'No held-out test split', 'Test data must never be used for model selection.');
    if (s.val.length && s.val.length < 10) add('warn', 'small_val', 'Validation split has only ' + s.val.length + ' samples', 'Metrics will be noisy; widen the split.');
    if (s.test.length && s.test.length < 20) add('warn', 'small_test', 'Test split has only ' + s.test.length + ' samples', 'Confidence intervals on the reported metrics will be wide.');
    if (n >= 30 && s.train.length && s.train.length < n * 0.4) {
      add('info', 'small_train', 'Training split is ' + (100 * s.train.length / n).toFixed(0) + '% of the data', 'Consider raising the train ratio.');
    }
    for (const key of ['train', 'val', 'test']) {
      const overlap = new Set(s[key]);
      for (const other of ['train', 'val', 'test']) {
        if (other <= key) continue;
        let dup = 0;
        for (const i of s[other]) if (overlap.has(i)) dup++;
        if (dup > 0) add('error', 'split_overlap', dup + ' sample indices appear in both ' + key + ' and ' + other, 'This is train/test leakage.');
      }
    }
  }

  const per = shapeSize(ds.xShape);
  const yPer = shapeSize(ds.yShape || [1]);
  if (n > 0 && per > 0 && per > n * yPer * 10) {
    add('warn', 'curse_dim', per + ' input features supervised by ' + yPer + ' target values across ' + n + ' samples',
      'Add samples or shrink the input; with this ratio the model can memorise the training split.');
  }
  const constant = stats.featureDims ? stats.featureDims.filter((d) => d.std < 1e-9).length : 0;
  if (constant > 0) {
    add('info', 'constant_features', constant + ' sampled feature dimensions have zero variance', 'They carry no signal and can be dropped.');
  }
  if (stats.sparsity > 0.98) {
    add('info', 'sparse_features', 'Feature matrix is ' + (100 * stats.sparsity).toFixed(1) + '% zeros', 'Expected for hashed text features; a dense model still trains fine at this scale.');
  }
  if (n < 100) add('warn', 'tiny_dataset', 'Only ' + n + ' samples', 'Results will have high variance. Report confidence intervals.');

  const duplicates = countDuplicates(ds, 4000);
  if (duplicates.exact > 0) {
    add('warn', 'duplicates', duplicates.exact + ' exact duplicate samples', 'Duplicates inflate the score of whichever split they land in.');
  }
  if (duplicates.crossSplit > 0) {
    add('error', 'leakage', duplicates.crossSplit + ' duplicate samples span two different splits', 'Direct train/test leakage. De-duplicate before splitting.');
  }
  if (duplicates.conflicting > 0) {
    add('warn', 'label_noise', duplicates.conflicting + ' duplicate samples carry conflicting labels', 'Identical inputs with different labels put a hard ceiling on achievable accuracy. Review the labelling, or accept the ceiling deliberately.');
  }
  if (ds.license && ds.license.id === 'unset') {
    add('warn', 'no_license', 'No license recorded for this dataset', 'Commercial use needs a clear license. Set one in the dataset header.');
  } else if (ds.license && ds.license.allowsCommercial === false) {
    add('warn', 'non_commercial_license', 'License "' + ds.license.name + '" does not permit commercial use', 'Do not ship models trained on this data in a commercial product.');
  }
  const order = { error: 0, warn: 1, info: 2 };
  issues.sort((a, b) => order[a.severity] - order[b.severity]);
  return { issues, duplicates, stats };
}

function rowSignature(ds, i, per) {
  let h = 2166136261;
  const stride = Math.max(1, Math.floor(per / 512));
  for (let f = 0; f < per; f += stride) {
    const v = Math.round(ds.x[i * per + f] * 1e5);
    h ^= v & 0xff;
    h = Math.imul(h, 16777619) >>> 0;
    h ^= (v >>> 8) & 0xff;
    h = Math.imul(h, 16777619) >>> 0;
    h ^= (v >>> 16) & 0xff;
    h = Math.imul(h, 16777619) >>> 0;
    h ^= (v >>> 24) & 0xff;
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h >>> 0;
}

export function countDuplicates(ds, limit) {
  const per = shapeSize(ds.xShape);
  const n = Math.min(ds.count, limit || ds.count);
  const yPer = shapeSize(ds.yShape || [1]);
  const seen = new Map();
  let exact = 0;
  let conflicting = 0;
  const splitOf = new Int32Array(ds.count).fill(-1);
  if (ds.splits) {
    ds.splits.train.forEach((i) => { splitOf[i] = 0; });
    ds.splits.val.forEach((i) => { splitOf[i] = 1; });
    ds.splits.test.forEach((i) => { splitOf[i] = 2; });
  }
  let crossSplit = 0;
  for (let i = 0; i < n; i++) {
    const sig = rowSignature(ds, i, per);
    const label = yPer === 1 ? Math.round(ds.y[i]) : 0;
    const prev = seen.get(sig);
    if (prev) {
      exact++;
      if (prev.label !== label) conflicting++;
      if (splitOf[i] >= 0 && prev.split >= 0 && splitOf[i] !== prev.split) crossSplit++;
    } else {
      seen.set(sig, { label, split: splitOf[i] });
    }
  }
  return { exact, conflicting, crossSplit, scanned: n };
}

export function serializeDataset(ds) {
  const out = {};
  for (const key of Object.keys(ds)) {
    const v = ds[key];
    if (v instanceof Float32Array || v instanceof Int32Array || v instanceof Uint8Array) {
      out[key] = { __typed: v.constructor.name, data: Array.from(v) };
    } else if (key === 'digest' || key === 'stats') {
      out[key] = null;
    } else {
      out[key] = v;
    }
  }
  return out;
}

export function cloneDataset(ds) {
  const copy = Object.assign({}, ds);
  copy.x = ds.x ? new Float32Array(ds.x) : null;
  copy.y = ds.y ? new Float32Array(ds.y) : null;
  if (ds.x2) copy.x2 = new Float32Array(ds.x2);
  if (ds.lengths) copy.lengths = Int32Array.from(ds.lengths);
  if (ds.groups) copy.groups = Int32Array.from(ds.groups);
  if (ds.splits) {
    copy.splits = {
      train: Int32Array.from(ds.splits.train),
      val: Int32Array.from(ds.splits.val),
      test: Int32Array.from(ds.splits.test)
    };
  }
  return copy;
}

export function subsetDataset(ds, indices) {
  const per = shapeSize(ds.xShape);
  const yPer = shapeSize(ds.yShape || [1]);
  const x = new Float32Array(indices.length * per);
  const y = new Float32Array(indices.length * yPer);
  const meta = ds.meta ? [] : null;
  for (let i = 0; i < indices.length; i++) {
    const s = indices[i];
    x.set(ds.x.subarray(s * per, (s + 1) * per), i * per);
    y.set(ds.y.subarray(s * yPer, (s + 1) * yPer), i * yPer);
    if (meta) meta.push(ds.meta[s]);
  }
  return createDataset({
    name: ds.name + ' (subset)',
    description: ds.description,
    modality: ds.modality,
    task: ds.task,
    source: 'derived',
    license: ds.license,
    x,
    xShape: ds.xShape,
    y,
    yShape: ds.yShape,
    labels: ds.labels,
    featureSpec: ds.featureSpec,
    meta,
    splitRatios: ds.splitRatios,
    splitSeed: ds.splitSeed,
    splitMethod: ds.splitMethod,
    provenance: Object.assign({}, ds.provenance, { derivedFrom: ds.id })
  });
}

export function mergeDatasets(list, name) {
  if (!list.length) return null;
  const base = list[0];
  const per = shapeSize(base.xShape);
  const yPer = shapeSize(base.yShape || [1]);
  let total = 0;
  for (const ds of list) total += ds.count;
  const x = new Float32Array(total * per);
  const y = new Float32Array(total * yPer);
  const meta = base.meta ? [] : null;
  const groups = base.groups ? new Int32Array(total) : null;
  let off = 0;
  list.forEach((ds, di) => {
    for (let i = 0; i < ds.count; i++) {
      x.set(ds.x.subarray(i * per, (i + 1) * per), (off + i) * per);
      y.set(ds.y.subarray(i * yPer, (i + 1) * yPer), (off + i) * yPer);
      if (meta) meta.push(Object.assign({ sourceDataset: ds.name }, ds.meta ? ds.meta[i] : {}));
      if (groups) groups[off + i] = off + di * 1000000;
    }
    off += ds.count;
  });
  return createDataset({
    name: name || list.map((d) => d.name).join(' + '),
    description: 'Merged from ' + list.length + ' datasets',
    modality: base.modality,
    task: base.task,
    source: 'derived',
    license: list.length === 1 ? list[0].license : { id: 'mixed', name: 'Mixed — see components', allowsCommercial: list.every((d) => d.license.allowsCommercial), notes: list.map((d) => d.name + ': ' + d.license.name).join('; ') },
    x,
    xShape: base.xShape,
    y,
    yShape: base.yShape,
    labels: base.labels,
    featureSpec: base.featureSpec,
    meta,
    groups,
    splitMethod: base.splitMethod,
    splitRatios: base.splitRatios,
    splitSeed: base.splitSeed
  });
}

export function datasetToRows(ds, limit) {
  const per = shapeSize(ds.xShape);
  const yPer = shapeSize(ds.yShape || [1]);
  const n = Math.min(ds.count, limit || 50);
  const rows = [];
  for (let i = 0; i < n; i++) {
    const split = ds.splits.train.includes(i) ? 'train' : ds.splits.val.includes(i) ? 'val' : ds.splits.test.includes(i) ? 'test' : 'unused';
    rows.push({
      index: i,
      split,
      label: yPer === 1 ? Math.round(ds.y[i]) : Array.from(ds.y.subarray(i * yPer, (i + 1) * yPer)),
      labelName: yPer === 1 && ds.labels ? ds.labels[Math.round(ds.y[i])] || String(Math.round(ds.y[i])) : null,
      preview: ds.meta && ds.meta[i] ? ds.meta[i].preview || ds.meta[i].text || JSON.stringify(ds.meta[i]).slice(0, 160) : null,
      features: Array.from(ds.x.subarray(i * per, (i + 1) * per).slice(0, 12))
    });
  }
  return rows;
}

export function recompute(ds) {
  ds.count = ds.x ? ds.x.length / shapeSize(ds.xShape) : 0;
  ds.digest = computeDigest(ds);
  ds.stats = computeStats(ds);
  return ds;
}
