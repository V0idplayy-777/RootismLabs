import { zipWrite, zipRead, zipGet, zipVerify } from '../core/zip.js';
import { sha256, sha256Text } from '../core/hash.js';
import { cardFiles, cardToMarkdown, CARD_VERSION, TOOL_NAME, TOOL_VERSION } from './card.js';
import { datasetToRows, qualityReport } from '../data/dataset.js';
import { formatBytes } from '../presets/metrics.js';

export const BUNDLE_VERSION = 2;

const TEXT_ENCODER = typeof TextEncoder !== 'undefined' ? new TextEncoder() : null;

function toBytes(value) {
  if (value instanceof Uint8Array) return value;
  if (ArrayBuffer.isView(value)) return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
  if (value instanceof ArrayBuffer) return new Uint8Array(value);
  if (typeof value === 'string') {
    if (TEXT_ENCODER) return TEXT_ENCODER.encode(value);
    const out = new Uint8Array(value.length);
    for (let i = 0; i < value.length; i++) out[i] = value.charCodeAt(i) & 0xff;
    return out;
  }
  return new Uint8Array(0);
}

function json(value) {
  return JSON.stringify(value, null, 2);
}

function hashBytes(bytes) {
  return sha256(bytes);
}

function safePath(name) {
  return String(name).replace(/^\/+/, '').replace(/\\/g, '/');
}

export function artifactPaths(artifact) {
  const format = String(artifact.format || 'raw').toLowerCase();
  const precision = String(artifact.precision || 'fp32');
  const suffix = precision === 'fp32' ? '' : '-' + precision;
  if (format === 'onnx') return { dir: 'onnx', stem: 'model' + suffix, ext: '.onnx' };
  if (format === 'tfjs') return { dir: 'tfjs', stem: 'model' + suffix, ext: '.json' };
  if (format === 'safetensors') return { dir: 'weights', stem: 'model' + suffix, ext: '.safetensors' };
  if (format === 'rootism') return { dir: 'native', stem: 'model' + suffix, ext: '.json' };
  return { dir: format, stem: 'model' + suffix, ext: '.bin' };
}

export function artifactEntries(artifact) {
  const layout = artifactPaths(artifact);
  const out = [];
  const files = artifact.files || [];
  if (artifact.format === 'tfjs') {
    const modelJson = artifact.modelJson || (files.find((f) => /\.json$/.test(f.name)) || {}).data;
    const weights = artifact.weightData || (files.find((f) => /\.bin$/.test(f.name)) || {}).data;
    const patched = JSON.parse(JSON.stringify(modelJson));
    patched.weightsManifest[0].paths = [layout.stem + '.weights.bin'];
    out.push({ name: layout.dir + '/' + layout.stem + '.json', data: json(patched) });
    if (weights) out.push({ name: layout.dir + '/' + layout.stem + '.weights.bin', data: toBytes(weights) });
    return out;
  }
  if (files.length === 1) {
    out.push({ name: layout.dir + '/' + layout.stem + layout.ext, data: toBytes(files[0].data) });
    return out;
  }
  for (let i = 0; i < files.length; i++) {
    const f = files[i];
    out.push({ name: layout.dir + '/' + (f.name || layout.stem + '_' + i + layout.ext), data: toBytes(f.data) });
  }
  return out;
}

export function datasetEntries(dataset, options) {
  const opts = options || {};
  if (!dataset) return [];
  const out = [];
  const limit = opts.sampleLimit === undefined ? 60 : opts.sampleLimit;
  out.push({ name: 'dataset/summary.json', data: json(datasetSummary(dataset)) });
  if (limit > 0) out.push({ name: 'dataset/samples.jsonl', data: datasetToRows(dataset, limit).map((r) => JSON.stringify(r)).join('\n') + '\n' });
  const qr = typeof qualityReport === 'function' ? qualityReport(dataset) : null;
  if (qr) out.push({ name: 'dataset/quality.json', data: json(qr) });
  if (opts.includeFull && typeof opts.serialize === 'function') {
    out.push({ name: 'dataset/dataset.json', data: json(opts.serialize(dataset)) });
  }
  return out;
}

export function datasetSummary(dataset) {
  return {
    id: dataset.id,
    name: dataset.name,
    modality: dataset.modality,
    task: dataset.task,
    samples: dataset.count,
    xShape: dataset.xShape,
    yShape: dataset.yShape,
    labels: dataset.labels || [],
    splits: dataset.splits ? { train: dataset.splits.train.length, val: dataset.splits.val.length, test: dataset.splits.test.length } : null,
    splitMethod: dataset.splitMethod,
    splitRatios: dataset.splitRatios,
    splitSeed: dataset.splitSeed,
    license: dataset.license || null,
    provenance: dataset.provenance || null,
    digest: dataset.digest || null,
    stats: dataset.stats || null
  };
}

export function manifestFor(entries, card) {
  const files = entries.map((e) => {
    const bytes = toBytes(e.data);
    return { path: safePath(e.name), bytes: bytes.length, sha256: hashBytes(bytes) };
  });
  files.sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0));
  const totalBytes = files.reduce((a, f) => a + f.bytes, 0);
  return {
    bundleVersion: BUNDLE_VERSION,
    cardVersion: CARD_VERSION,
    tool: { name: TOOL_NAME, version: TOOL_VERSION },
    generatedAt: Date.now(),
    model: card ? { name: card.model.name, key: card.model.key, task: card.model.task, params: card.model.params, head: card.model.head } : null,
    dataset: card && card.dataset ? { name: card.dataset.name, samples: card.dataset.samples, contentHash: card.dataset.digest ? card.dataset.digest.contentHash : null, license: card.dataset.license ? card.dataset.license.name : null } : null,
    formats: card ? (card.exports || []).map((a) => a.format + ':' + a.precision) : [],
    fileCount: files.length,
    totalBytes,
    totalBytesHuman: formatBytes(totalBytes),
    treeSha256: sha256Text(files.map((f) => f.path + ' ' + f.sha256).join('\n')),
    files
  };
}

export function buildBundle(input) {
  const opts = input || {};
  const card = opts.card || null;
  const entries = [];
  const added = new Set();
  const push = (entry) => {
    const name = safePath(entry.name);
    if (added.has(name)) return;
    added.add(name);
    entries.push({ name, data: entry.data });
  };

  if (card) for (const f of cardFiles(card)) push(f);
  for (const f of opts.extraFiles || []) push(f);

  if (opts.spec) push({ name: 'config/spec.json', data: json(opts.spec) });
  if (opts.run && opts.run.config) push({ name: 'config/train.json', data: json(opts.run.config) });
  if (opts.run) {
    push({
      name: 'metrics/run.json',
      data: json({
        seed: opts.run.seed,
        taskKey: opts.run.taskKey,
        monitor: opts.run.monitor,
        epochsRun: opts.run.epochsRun,
        steps: opts.run.steps,
        samplesSeen: opts.run.samplesSeen,
        durationMs: opts.run.durationMs,
        startedAt: opts.run.startedAt,
        endedAt: opts.run.endedAt,
        best: opts.run.best,
        metrics: opts.run.metrics,
        history: opts.run.history
      })
    });
  }

  if (card && card.model && card.model.layers) {
    const labels = card.dataset && card.dataset.labels ? card.dataset.labels : [];
    if (labels.length) push({ name: 'labels.txt', data: labels.join('\n') + '\n' });
  }

  for (const artifact of opts.artifacts || []) {
    for (const f of artifactEntries(artifact)) push(f);
  }

  if (opts.dataset && opts.includeDataset !== false) {
    for (const f of datasetEntries(opts.dataset, { sampleLimit: opts.sampleLimit, includeFull: opts.includeFullDataset, serialize: opts.serializeDataset })) push(f);
  }

  const withManifest = entries.slice();
  const manifest = manifestFor(withManifest, card);
  withManifest.push({ name: 'MANIFEST.json', data: json(manifest) });

  const bytes = zipWrite(withManifest, { date: opts.date || new Date() });
  return { bytes, entries: withManifest, manifest, markdown: card ? cardToMarkdown(card) : '' };
}

export function verifyBundle(bytes) {
  const result = { ok: true, problems: [], files: [] };
  let index;
  try {
    index = zipReadNames(bytes);
  } catch (e) {
    return { ok: false, problems: ['zip could not be parsed: ' + e.message], files: [] };
  }
  result.files = index;
  const manifestEntry = index.find((f) => f.name === 'MANIFEST.json');
  if (!manifestEntry) {
    result.ok = false;
    result.problems.push('MANIFEST.json missing');
    return result;
  }
  let manifest;
  try {
    manifest = JSON.parse(manifestEntry.text);
  } catch (e) {
    result.ok = false;
    result.problems.push('MANIFEST.json is not valid JSON');
    return result;
  }
  const byPath = {};
  for (const f of index) byPath[f.name] = f;
  for (const declared of manifest.files || []) {
    const found = byPath[declared.path];
    if (!found) {
      result.ok = false;
      result.problems.push('declared file missing from the archive: ' + declared.path);
      continue;
    }
    if (found.bytes !== declared.bytes) {
      result.ok = false;
      result.problems.push('size mismatch for ' + declared.path + ': ' + found.bytes + ' vs ' + declared.bytes);
    }
  }
  if (!zipVerify(bytes).ok) {
    result.ok = false;
    result.problems.push('zip central directory failed verification');
  }
  result.manifest = manifest;
  return result;
}

function zipReadNames(bytes) {
  const index = zipRead(bytes);
  return index.map((entry) => {
    const data = zipGet(bytes, entry.name);
    const isText = /\.(json|md|txt|jsonl|csv|tsv|ya?ml)$/i.test(entry.name);
    return {
      name: entry.name,
      bytes: data ? data.length : 0,
      sha256: data ? hashBytes(data) : null,
      text: isText && data ? decodeText(data) : null
    };
  });
}

function decodeText(bytes) {
  if (typeof TextDecoder !== 'undefined') return new TextDecoder().decode(bytes);
  let s = '';
  for (let i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i]);
  return s;
}

export function bundleSummary(bundle) {
  const byDir = {};
  for (const e of bundle.entries) {
    const dir = e.name.indexOf('/') >= 0 ? e.name.slice(0, e.name.indexOf('/')) : '(root)';
    const bytes = toBytes(e.data).length;
    byDir[dir] = byDir[dir] || { files: 0, bytes: 0 };
    byDir[dir].files++;
    byDir[dir].bytes += bytes;
  }
  return {
    files: bundle.entries.length,
    bytes: bundle.bytes.length,
    bytesHuman: formatBytes(bundle.bytes.length),
    byDirectory: Object.keys(byDir).sort().map((k) => ({ directory: k, files: byDir[k].files, bytes: byDir[k].bytes, bytesHuman: formatBytes(byDir[k].bytes) }))
  };
}

export { toBytes as bundleBytes };
