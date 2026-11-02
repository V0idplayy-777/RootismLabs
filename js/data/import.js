import { createDataset, recompute } from './dataset.js';
import { hashFeatures, DEFAULT_TEXT_SPEC, tokenizeWords } from './text.js';
import { logMelFrames, normalizeFeatures, wavDecode, wavEncode, resample, peakNormalize } from './audio.js';
import { zipRead, zipWrite } from '../core/zip.js';
import { utf8Decode, base64Encode, base64Decode, float32ToBase64, base64ToFloat32 } from '../core/base64.js';
import { MODALITIES } from './dataset.js';
import { sha256Text } from '../core/hash.js';

export const NATIVE_VERSION = 3;

export const IMPORT_FORMATS = [
  { key: 'csv', label: 'CSV', extensions: ['.csv'], mime: 'text/csv', kind: 'table' },
  { key: 'tsv', label: 'TSV', extensions: ['.tsv', '.tab'], mime: 'text/tab-separated-values', kind: 'table' },
  { key: 'jsonl', label: 'JSON lines', extensions: ['.jsonl', '.ndjson'], mime: 'application/x-ndjson', kind: 'table' },
  { key: 'json', label: 'JSON', extensions: ['.json'], mime: 'application/json', kind: 'table' },
  { key: 'zip', label: 'Zip archive', extensions: ['.zip'], mime: 'application/zip', kind: 'archive' },
  { key: 'wav', label: 'WAV audio', extensions: ['.wav', '.wave'], mime: 'audio/wav', kind: 'audio' },
  { key: 'image', label: 'Image', extensions: ['.png', '.jpg', '.jpeg', '.webp', '.bmp'], mime: 'image', kind: 'image' },
  { key: 'text', label: 'Plain text', extensions: ['.txt'], mime: 'text/plain', kind: 'text' },
  { key: 'rootism', label: 'Rootism dataset', extensions: ['.rootism.json'], mime: 'application/json', kind: 'native' }
];

export const EXPORT_FORMATS = [
  { key: 'csv', label: 'CSV', extension: '.csv', mime: 'text/csv' },
  { key: 'jsonl', label: 'JSON lines', extension: '.jsonl', mime: 'application/x-ndjson' },
  { key: 'native', label: 'Rootism dataset', extension: '.rootism.json', mime: 'application/json' },
  { key: 'labels', label: 'Labels only', extension: '.txt', mime: 'text/plain' },
  { key: 'coco', label: 'COCO style JSON', extension: '.coco.json', mime: 'application/json' },
  { key: 'wav', label: 'WAV archive', extension: '.zip', mime: 'application/zip' }
];

const LABEL_NAMES = ['label', 'labels', 'class', 'classes', 'category', 'target', 'y', 'intent', 'tag', 'tags', 'sentiment', 'spam', 'toxic', 'language', 'lang', 'activity', 'genre', 'is_positive', 'positive', 'value'];
const IGNORE_NAMES = ['id', 'index', 'idx', 'row', 'row_id', 'uuid', 'key', 'filename', 'file', 'path', 'name', 'comment', 'notes', 'timestamp', 'date'];

function lower(value) {
  return String(value === undefined || value === null ? '' : value).trim().toLowerCase();
}

function extensionOf(name) {
  const clean = String(name || '').toLowerCase();
  const slash = Math.max(clean.lastIndexOf('/'), clean.lastIndexOf('\\'));
  const base = slash >= 0 ? clean.slice(slash + 1) : clean;
  const dot = base.lastIndexOf('.');
  return dot < 0 ? '' : base.slice(dot);
}

function baseName(name) {
  const clean = String(name || '');
  const slash = Math.max(clean.lastIndexOf('/'), clean.lastIndexOf('\\'));
  const base = slash >= 0 ? clean.slice(slash + 1) : clean;
  const dot = base.lastIndexOf('.');
  return dot > 0 ? base.slice(0, dot) : base;
}

function parentDir(name) {
  const clean = String(name || '').replace(/\\/g, '/');
  const slash = clean.lastIndexOf('/');
  if (slash <= 0) return '';
  const dir = clean.slice(0, slash);
  const inner = dir.lastIndexOf('/');
  return inner >= 0 ? dir.slice(inner + 1) : dir;
}

export function detectFormat(name, bytes) {
  const ext = extensionOf(name);
  const base = String(name || '').toLowerCase();
  if (base.endsWith('.rootism.json')) return 'rootism';
  for (const f of IMPORT_FORMATS) {
    if (f.extensions.indexOf(ext) >= 0) return f.key;
  }
  if (bytes && bytes.length > 3) {
    if (bytes[0] === 0x50 && bytes[1] === 0x4b) return 'zip';
    if (bytes[0] === 0x52 && bytes[1] === 0x49 && bytes[2] === 0x46 && bytes[3] === 0x46) return 'wav';
    if (bytes[0] === 0x89 && bytes[1] === 0x50) return 'image';
    if (bytes[0] === 0xff && bytes[1] === 0xd8) return 'image';
  }
  return 'text';
}

export function looksBinary(bytes) {
  const n = Math.min(bytes.length, 4096);
  let suspicious = 0;
  for (let i = 0; i < n; i++) {
    const b = bytes[i];
    if (b === 0) return true;
    if (b < 9 || (b > 13 && b < 32)) suspicious++;
  }
  return suspicious / Math.max(1, n) > 0.1;
}

function splitQuotedLine(line, delimiter) {
  const out = [];
  let current = '';
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (inQuotes) {
      if (ch === '"') {
        if (line[i + 1] === '"') {
          current += '"';
          i++;
        } else inQuotes = false;
      } else current += ch;
    } else if (ch === '"') {
      inQuotes = true;
    } else if (ch === delimiter) {
      out.push(current);
      current = '';
    } else current += ch;
  }
  out.push(current);
  return out;
}

export function detectDelimiter(text) {
  const candidates = [',', '\t', ';', '|'];
  const lines = text.split('\n').filter((l) => l.trim().length).slice(0, 20);
  let best = ',';
  let bestScore = -1;
  for (const d of candidates) {
    let total = 0;
    let consistent = 0;
    let first = -1;
    for (const line of lines) {
      const parts = splitQuotedLine(line, d);
      total += parts.length;
      if (first < 0) first = parts.length;
      else if (parts.length === first) consistent++;
    }
    if (!lines.length) continue;
    const avg = total / lines.length;
    if (avg < 2) continue;
    const score = avg * (1 + consistent / lines.length);
    if (score > bestScore) {
      bestScore = score;
      best = d;
    }
  }
  return best;
}

export function parseDelimited(text, options) {
  const opts = options || {};
  const cleaned = String(text).replace(/^\uFEFF/, '');
  const delimiter = opts.delimiter || detectDelimiter(cleaned);
  const rows = [];
  let buffer = '';
  let inQuotes = false;
  const lines = [];
  for (let i = 0; i < cleaned.length; i++) {
    const ch = cleaned[i];
    if (ch === '"') inQuotes = !inQuotes;
    if (ch === '\n' && !inQuotes) {
      lines.push(buffer);
      buffer = '';
    } else if (ch !== '\r' || inQuotes) buffer += ch;
  }
  if (buffer.length) lines.push(buffer);
  const meaningful = lines.filter((l) => l.trim().length);
  if (!meaningful.length) return { columns: [], rows: [], delimiter, header: false };
  const first = splitQuotedLine(meaningful[0], delimiter);
  const second = meaningful.length > 1 ? splitQuotedLine(meaningful[1], delimiter) : null;
  let header = opts.header;
  if (header === undefined) {
    header = second ? first.length === second.length && first.some((c) => !isNumericToken(c)) : first.some((c) => !isNumericToken(c));
  }
  const columns = header ? first.map((c, i) => (String(c).trim() || 'column_' + i)) : first.map((c, i) => 'column_' + i);
  const body = header ? meaningful.slice(1) : meaningful;
  for (const line of body) {
    const parts = splitQuotedLine(line, delimiter);
    const row = {};
    for (let i = 0; i < columns.length; i++) row[columns[i]] = i < parts.length ? parts[i].trim() : '';
    rows.push(row);
  }
  return { columns, rows, delimiter, header };
}

function isNumericToken(value) {
  const t = String(value).trim();
  if (!t.length) return false;
  return Number.isFinite(Number(t));
}

export function parseJsonl(text) {
  const lines = String(text).split('\n').filter((l) => l.trim().length);
  const rows = [];
  const columns = [];
  let broken = 0;
  for (const line of lines) {
    let obj;
    try {
      obj = JSON.parse(line);
    } catch (e) {
      broken++;
      continue;
    }
    if (obj === null || typeof obj !== 'object' || Array.isArray(obj)) {
      broken++;
      continue;
    }
    for (const k of Object.keys(obj)) if (columns.indexOf(k) < 0) columns.push(k);
    rows.push(obj);
  }
  return { columns, rows, broken };
}

export function parseJsonTable(text) {
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch (e) {
    return null;
  }
  if (parsed && parsed.rootismDataset) return { native: parsed };
  let records = null;
  if (Array.isArray(parsed)) records = parsed;
  else if (parsed && typeof parsed === 'object') {
    for (const key of ['rows', 'data', 'records', 'items', 'samples', 'examples', 'annotations']) {
      if (Array.isArray(parsed[key])) {
        records = parsed[key];
        break;
      }
    }
    if (!records) {
      const values = Object.keys(parsed).map((k) => parsed[k]);
      if (values.length && values.every((v) => Array.isArray(v))) {
        const n = values[0].length;
        if (values.every((v) => v.length === n)) {
          records = [];
          const keys = Object.keys(parsed);
          for (let i = 0; i < n; i++) {
            const row = {};
            for (const k of keys) row[k] = parsed[k][i];
            records.push(row);
          }
        }
      }
    }
  }
  if (!records || !records.length) return null;
  const columns = [];
  const rows = [];
  for (const rec of records) {
    if (rec === null || typeof rec !== 'object' || Array.isArray(rec)) {
      rows.push({ value: rec });
      if (columns.indexOf('value') < 0) columns.push('value');
      continue;
    }
    for (const k of Object.keys(rec)) if (columns.indexOf(k) < 0) columns.push(k);
    rows.push(rec);
  }
  return { columns, rows };
}

export function parseTextLines(text, options) {
  const opts = options || {};
  const sep = opts.separator || '\t';
  const lines = String(text).split('\n').map((l) => l.replace(/\r$/, '')).filter((l) => l.trim().length);
  const columns = ['text'];
  let hasLabel = false;
  for (const l of lines) if (l.indexOf(sep) >= 0) { hasLabel = true; break; }
  if (hasLabel) columns.push('label');
  const rows = lines.map((l) => {
    const idx = l.indexOf(sep);
    if (idx < 0) return { text: l };
    return { text: l.slice(0, idx).trim(), label: l.slice(idx + sep.length).trim() };
  });
  return { columns, rows };
}

export function inferColumns(columns, rows, options) {
  const opts = options || {};
  const n = rows.length;
  const out = [];
  for (const col of columns) {
    let numeric = 0;
    let empty = 0;
    let tokens = 0;
    let chars = 0;
    const distinct = new Map();
    let sum = 0;
    let sumSq = 0;
    let min = Infinity;
    let max = -Infinity;
    let integers = true;
    let multiValue = 0;
    for (const row of rows) {
      const raw = row[col];
      const value = raw === undefined || raw === null ? '' : String(raw).trim();
      if (!value.length) {
        empty++;
        continue;
      }
      if (/[;,|]/.test(value) && !isNumericToken(value)) multiValue++;
      const asNumber = Number(value);
      if (value.length && Number.isFinite(asNumber)) {
        numeric++;
        sum += asNumber;
        sumSq += asNumber * asNumber;
        if (asNumber < min) min = asNumber;
        if (asNumber > max) max = asNumber;
        if (!Number.isInteger(asNumber)) integers = false;
      } else {
        const words = tokenizeWords(value);
        tokens += words.length;
        chars += value.length;
      }
      const key = value.length > 64 ? value.slice(0, 64) : value;
      distinct.set(key, (distinct.get(key) || 0) + 1);
    }
    const filled = n - empty;
    const numericFraction = filled ? numeric / filled : 0;
    const textFraction = filled ? (filled - numeric) / filled : 0;
    const avgTokens = filled ? tokens / filled : 0;
    let kind;
    if (!filled) kind = 'empty';
    else if (numericFraction >= 0.9) kind = 'numeric';
    else if (textFraction >= 0.5 && (avgTokens >= 2.5 || chars / Math.max(1, filled) >= 24 || /^(text|message|sentence|utterance|content|body)$/i.test(col))) kind = 'text';
    else if (distinct.size <= Math.max(2, Math.min(64, Math.ceil(filled * 0.5)))) kind = 'categorical';
    else kind = 'text';
    const counts = Array.from(distinct.entries()).sort((a, b) => b[1] - a[1]);
    out.push({
      name: col,
      kind,
      filled,
      missing: empty,
      missingFraction: n ? empty / n : 0,
      distinct: distinct.size,
      numericFraction,
      integers: integers && numeric > 0,
      avgTokens,
      mean: numeric ? sum / numeric : null,
      std: numeric ? Math.sqrt(Math.max(0, sumSq / numeric - Math.pow(sum / numeric, 2))) : null,
      min: numeric ? min : null,
      max: numeric ? max : null,
      multiValueFraction: filled ? multiValue / filled : 0,
      topValues: counts.slice(0, 12).map(([value, count]) => ({ value, count })),
      ignored: IGNORE_NAMES.indexOf(lower(col)) >= 0 && kind !== 'text',
      role: opts.roles && opts.roles[col] ? opts.roles[col] : null
    });
  }
  return out;
}

export function pickLabelColumn(inferred, rows, options) {
  const opts = options || {};
  if (opts.labelColumn) return opts.labelColumn;
  const usable = inferred.filter((c) => c.kind !== 'empty');
  for (const name of LABEL_NAMES) {
    const found = usable.find((c) => lower(c.name) === name);
    if (found) return found.name;
  }
  for (const name of LABEL_NAMES) {
    const found = usable.find((c) => lower(c.name).indexOf(name) >= 0);
    if (found) return found.name;
  }
  const candidates = usable.filter((c) => !c.ignored && (c.kind === 'categorical' || (c.kind === 'numeric' && c.integers)));
  if (candidates.length) {
    candidates.sort((a, b) => (a.distinct - b.distinct) || (b.filled - a.filled));
    const best = candidates[candidates.length - 1] === candidates[0] ? candidates[0] : candidates.find((c) => c.distinct >= 2 && c.distinct <= 512) || candidates[0];
    if (best && best.distinct >= 2) return best.name;
  }
  return null;
}

export function inferTask(labelInfo, options) {
  const opts = options || {};
  if (opts.task) return opts.task;
  if (!labelInfo) return 'regression';
  if (labelInfo.multiValueFraction > 0.2) return 'multilabel';
  if (labelInfo.kind === 'numeric' && !labelInfo.integers) return 'regression';
  if (labelInfo.kind === 'numeric' && labelInfo.integers && labelInfo.distinct > 64) return 'regression';
  if (labelInfo.distinct === 2) return 'binary';
  return 'multiclass';
}

function buildVocab(column, rows, limit) {
  const counts = new Map();
  for (const row of rows) {
    const value = row[column.name];
    const key = value === undefined || value === null ? '' : String(value).trim();
    if (!key.length) continue;
    counts.set(key, (counts.get(key) || 0) + 1);
  }
  const sorted = Array.from(counts.entries()).sort((a, b) => (b[1] - a[1]) || (a[0] < b[0] ? -1 : 1));
  const vocab = sorted.slice(0, limit).map((e) => e[0]);
  return { vocab, dropped: Math.max(0, sorted.length - vocab.length) };
}

export function planFeatures(inferred, rows, options) {
  const opts = options || {};
  const labelColumn = opts.labelColumn === undefined ? pickLabelColumn(inferred, rows, opts) : opts.labelColumn;
  const plan = [];
  let dim = 0;
  for (const col of inferred) {
    if (col.kind === 'empty') continue;
    if (col.name === labelColumn) continue;
    if (col.ignored && !opts.forceColumns) continue;
    if (opts.featureColumns && opts.featureColumns.indexOf(col.name) < 0) continue;
    if (col.kind === 'text') {
      const textDim = opts.textDim || DEFAULT_TEXT_SPEC.dim;
      plan.push({ column: col.name, kind: 'text', offset: dim, width: textDim, textSpec: Object.assign({}, DEFAULT_TEXT_SPEC, opts.textSpec || {}, { dim: textDim }) });
      dim += textDim;
    } else if (col.kind === 'categorical') {
      const limit = opts.maxLevels === undefined ? 64 : opts.maxLevels;
      const vocab = buildVocab(col, rows, limit);
      plan.push({ column: col.name, kind: 'onehot', offset: dim, width: vocab.vocab.length + (vocab.dropped ? 1 : 0), vocab: vocab.vocab, dropped: vocab.dropped });
      dim += vocab.vocab.length + (vocab.dropped ? 1 : 0);
    } else if (col.kind === 'numeric') {
      plan.push({ column: col.name, kind: 'numeric', offset: dim, width: 1, mean: col.mean, std: col.std, missingFill: opts.fillMissing === undefined ? 0 : opts.fillMissing });
      dim += 1;
    }
  }
  return { plan, dim, labelColumn };
}

export function tableToDataset(input) {
  const opts = input || {};
  const rows = opts.rows || [];
  const columns = opts.columns || (rows.length ? Object.keys(rows[0]) : []);
  if (!rows.length) throw new Error('no rows to import');
  const inferred = inferColumns(columns, rows, opts);
  const selectedLabel = opts.labelColumn || pickLabelColumn(inferred, rows, opts);
  const planning = planFeatures(inferred, rows, Object.assign({}, opts, { labelColumn: selectedLabel }));
  const labelColumn = planning.labelColumn;
  const labelInfo = inferred.find((c) => c.name === labelColumn) || null;
  const task = inferTask(labelInfo, opts);
  const modality = opts.modality || (planning.plan.some((p) => p.kind === 'text') ? 'text' : 'tabular');
  const labelValues = rows.map((r) => (labelColumn ? r[labelColumn] : undefined));
  if (labelColumn && labelValues.some(v => v === undefined || v === null || String(v).trim() === '')) throw new Error('Label column contains missing values; fix or remove those rows before import.');
  if (labelColumn && (task === 'regression' || task === 'forecast') && labelValues.some(v => !Number.isFinite(Number(v)))) throw new Error('Regression targets must be finite numbers.');

  let labels = [];
  let yPer = 1;
  const y = new Float32Array(rows.length * yPer);

  if (!labelColumn) {
    labels = [];
  } else if (task === 'multilabel') {
    const set = new Set();
    const parsed = labelValues.map((v) => {
      const parts = String(v === undefined || v === null ? '' : v).split(/[;|]/).map((p) => p.trim()).filter((p) => p.length);
      for (const p of parts) set.add(p);
      return parts;
    });
    labels = Array.from(set).sort();
    yPer = Math.max(1, labels.length);
    const yMulti = new Float32Array(rows.length * yPer);
    parsed.forEach((parts, i) => {
      for (const p of parts) {
        const idx = labels.indexOf(p);
        if (idx >= 0) yMulti[i * yPer + idx] = 1;
      }
    });
    return finishTableDataset(opts, rows, planning, inferred, task, modality, labels, yMulti, yPer, labelColumn, labelValues);
  } else if (task === 'regression' || task === 'forecast') {
    const numeric = labelValues.map((v) => Number(v));
    for (let i = 0; i < rows.length; i++) y[i] = Number.isFinite(numeric[i]) ? numeric[i] : 0;
    labels = [];
  } else {
    const counts = new Map();
    for (const v of labelValues) {
      const key = String(v === undefined || v === null ? '' : v).trim();
      counts.set(key, (counts.get(key) || 0) + 1);
    }
    labels = Array.from(counts.entries()).sort((a, b) => (b[1] - a[1]) || (a[0] < b[0] ? -1 : 1)).map((e) => e[0]);
    if (task === 'binary' && labels.length === 2) {
      const negative = labels.find((l) => ['0', 'no', 'false', 'negative', 'ham', 'benign', 'normal', 'clean', 'not_spam'].indexOf(lower(l)) >= 0);
      if (negative) labels = [negative, labels.find((l) => l !== negative)];
    }
    for (let i = 0; i < rows.length; i++) {
      const key = String(labelValues[i] === undefined || labelValues[i] === null ? '' : labelValues[i]).trim();
      y[i] = Math.max(0, labels.indexOf(key));
    }
  }
  return finishTableDataset(opts, rows, planning, inferred, task, modality, labels, y, yPer, labelColumn, labelValues);
}

// Applies the persisted table contract to a new row. A model consumes this
// vector, not the original raw row. Unknown categories use the other bucket
// when present, otherwise an all-zero one-hot vector.
export function transformTableRow(row, spec) {
  if (!spec || spec.kind !== 'table') throw new Error('A table feature specification is required.');
  const out = new Float32Array(spec.dim);
  for (const p of spec.columns) {
    const raw = row[p.column];
    const value = raw === undefined || raw === null ? '' : String(raw).trim();
    if (p.kind === 'text') out.set(hashFeatures(value, p.textSpec).subarray(0, p.width), p.offset);
    else if (p.kind === 'onehot') {
      let idx = p.vocab.indexOf(value);
      if (idx < 0 && p.unknownBucket !== null && p.unknownBucket !== undefined) idx = p.unknownBucket;
      if (idx >= 0 && idx < p.width) out[p.offset + idx] = 1;
    } else {
      const num = value === '' ? NaN : Number(value);
      let v = Number.isFinite(num) ? num : (p.missingFill ?? 0);
      if (spec.standardize !== false && p.std !== null && p.std > 1e-9) v = (v - (p.mean || 0)) / p.std;
      out[p.offset] = v;
    }
  }
  return out;
}

function finishTableDataset(opts, rows, planning, inferred, task, modality, labels, y, yPer, labelColumn, labelValues) {
  // Assign rows before learning *any* feature transformations.  The temporary
  // dataset carries only row IDs; duplicate rows are retained and audited.
  let groups = null;
  if (opts.groupColumn) {
    if (opts.groupColumn === labelColumn) throw new Error('The group column cannot also be the label.');
    if (!inferred.some(c => c.name === opts.groupColumn)) throw new Error('Group column not found: ' + opts.groupColumn);
    const ids = new Map();
    groups = Int32Array.from(rows.map(row => {
      const key = String(row[opts.groupColumn] ?? '').trim();
      if (!key) throw new Error('Group column contains missing values.');
      if (!ids.has(key)) ids.set(key, ids.size);
      return ids.get(key);
    }));
  }
  const splitMethod = opts.groupColumn ? 'group' : (opts.splitMethod || 'stratified');
  const temp = createDataset({ x: Float32Array.from(rows, (_, i) => i), xShape: [1], y, yShape: [yPer], task, labels, groups, dedupe: false,
    splitMethod, splitRatios: opts.splitRatios || [0.7, 0.15, 0.15], splitSeed: opts.splitSeed });
  const trainRows = Array.from(temp.splits.train, i => rows[i]);
  if (!trainRows.length) throw new Error('The training split is empty.');
  const featureColumns = inferred.map(c => c.name).filter(c => c !== opts.groupColumn);
  const fittedInference = inferColumns(featureColumns, trainRows, opts);
  planning = planFeatures(fittedInference, trainRows, Object.assign({}, opts, { labelColumn }));
  modality = opts.modality || (planning.plan.some(p => p.kind === 'text') ? 'text' : 'tabular');
  const dim = planning.dim || 1;
  const x = new Float32Array(rows.length * dim);
  const meta = [];
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    for (const p of planning.plan) {
      const raw = row[p.column];
      const value = raw === undefined || raw === null ? '' : String(raw).trim();
      if (p.kind === 'text') {
        const features = hashFeatures(value, p.textSpec);
        x.set(features.subarray(0, p.width), i * dim + p.offset);
      } else if (p.kind === 'onehot') {
        let idx = p.vocab.indexOf(value);
        if (idx < 0) idx = p.dropped ? p.vocab.length : -1;
        if (idx >= 0) x[i * dim + p.offset + idx] = 1;
      } else {
        const num = value === '' ? NaN : Number(value);
        let out = Number.isFinite(num) ? num : p.missingFill;
        if (opts.standardize !== false && p.std !== null && p.std > 1e-9) out = (out - (p.mean || 0)) / p.std;
        x[i * dim + p.offset] = out;
      }
    }
    const preview = planning.plan.find((p) => p.kind === 'text');
    meta.push({
      source: 'imported',
      labelRaw: labelColumn ? String(labelValues[i] === undefined ? '' : labelValues[i]) : null,
      preview: preview ? String(row[preview.column] || '').slice(0, 200) : null,
      row: opts.keepRows === false ? null : JSON.parse(JSON.stringify(row))
    });
  }
  const featureSpec = {
    kind: 'table',
    fittedOn: 'train',
    fitSplitHash: sha256Text(Array.from(temp.splits.train).sort((a, b) => a - b).join(',')),
    standardize: opts.standardize !== false,
    dim,
    columns: planning.plan.map((p) => ({
      column: p.column,
      kind: p.kind,
      offset: p.offset,
      width: p.width,
      vocab: p.vocab || null,
      unknownBucket: p.kind === 'onehot' && p.dropped ? p.vocab.length : null,
      missingFill: p.missingFill === undefined ? null : p.missingFill,
      mean: p.mean === undefined ? null : p.mean,
      std: p.std === undefined ? null : p.std,
      textSpec: p.textSpec || null
    })),
    inferred: inferred.map((c) => ({ name: c.name, kind: c.kind, distinct: c.distinct, missing: c.missing, ignored: c.ignored }))
  };
  const ds = createDataset({
    name: opts.name || 'Imported dataset',
    source: 'imported',
    groups,
    splits: temp.splits,
    description: opts.description || '',
    modality,
    task,
    x,
    xShape: [dim],
    y,
    yShape: [yPer],
    labels,
    meta,
    columns: planning.plan.map((p) => p.column),
    featureSpec,
    license: opts.license || { id: 'user-supplied', name: 'Supplied by the importer', allowsCommercial: opts.allowsCommercial === true, notes: 'You imported this data. Confirm you have the rights to use it, including commercially.' },
    provenance: Object.assign({ origin: 'import', format: opts.format || 'table', rows: rows.length, labelColumn: labelColumn || null }, opts.provenance || {}),
    config: Object.assign({ standardize: opts.standardize !== false, textDim: (planning.plan.find((p) => p.kind === 'text') || {}).width || null }, opts.config || {}),
    splitMethod: temp.splitMethod,
    splitRatios: opts.splitRatios || [0.7, 0.15, 0.15],
    splitSeed: temp.splitSeed
  });
  ds.importReport = {
    rows: rows.length,
    columns: inferred.map((c) => ({ name: c.name, kind: c.kind, distinct: c.distinct, missing: c.missing, ignored: c.ignored })),
    labelColumn: labelColumn || null,
    task,
    modality,
    featureDim: dim,
    classes: labels.length,
    droppedColumns: inferred.filter((c) => c.ignored || c.kind === 'empty').map((c) => c.name)
  };
  return ds;
}



export const AUDIO_FEATURE_PRESETS = {
  kws: { sampleRate: 8000, frameSize: 400, hop: 160, numMel: 32, duration: 0.75, frames: 36 },
  speech16: { sampleRate: 16000, frameSize: 512, hop: 160, numMel: 40, duration: 1, frames: 97 },
  events: { sampleRate: 8000, frameSize: 512, hop: 128, numMel: 32, duration: 0.6, frames: 38 },
  onset: { sampleRate: 11025, frameSize: 1024, hop: 512, numMel: 0, duration: 2.3, frames: 98 }
};

export function audioFeatureConfig(options) {
  const opts = options || {};
  const preset = AUDIO_FEATURE_PRESETS[opts.preset] || AUDIO_FEATURE_PRESETS.kws;
  return Object.assign({}, preset, opts.audio || {});
}

export function signalToFeature(signal, cfg) {
  const target = Math.max(64, Math.round(cfg.sampleRate * cfg.duration));
  let s = signal.length > target ? signal.subarray(0, target) : signal;
  if (s.length < target) {
    const padded = new Float32Array(target);
    padded.set(s);
    s = padded;
  }
  const mel = logMelFrames(Float32Array.from(s), {
    sampleRate: cfg.sampleRate,
    frameSize: cfg.frameSize,
    hop: cfg.hop,
    numMel: cfg.numMel
  });
  const norm = normalizeFeatures(mel.data, mel.frames, mel.bins);
  const frames = cfg.frames || mel.frames;
  const per = frames * mel.bins;
  const out = new Float32Array(per);
  const available = Math.min(per, norm.data.length);
  out.set(norm.data.subarray(0, available));
  return { data: out, shape: [frames, mel.bins], frames: mel.frames, bins: mel.bins };
}

export function decodeAudioFile(bytes, cfg) {
  const decoded = wavDecode(bytes);
  let signal = decoded.samples;
  if (decoded.channels > 1) {
    const mono = new Float32Array(signal.length / decoded.channels);
    for (let i = 0; i < mono.length; i++) {
      let acc = 0;
      for (let c = 0; c < decoded.channels; c++) acc += signal[i * decoded.channels + c];
      mono[i] = acc / decoded.channels;
    }
    signal = mono;
  }
  if (decoded.sampleRate !== cfg.sampleRate) signal = resample(signal, decoded.sampleRate, cfg.sampleRate);
  return peakNormalize(Float32Array.from(signal), 0.95);
}

function findLabelTable(files) {
  const candidates = files.filter((f) => /\.(csv|tsv|jsonl|json|txt)$/i.test(f.name) && !/manifest\.json$/i.test(f.name));
  for (const c of candidates) {
    const text = c.text === undefined ? utf8Decode(c.bytes) : c.text;
    const table = /\.jsonl$/i.test(c.name) ? parseJsonl(text) : /\.json$/i.test(c.name) ? parseJsonTable(text) : /\.txt$/i.test(c.name) ? parseTextLines(text) : parseDelimited(text);
    if (!table || !table.rows || !table.rows.length) continue;
    const columns = table.columns || [];
    const hasName = columns.some((c2) => ['file', 'filename', 'name', 'path', 'audio', 'wav', 'id'].indexOf(lower(c2)) >= 0);
    const hasLabel = columns.some((c2) => LABEL_NAMES.indexOf(lower(c2)) >= 0);
    if (hasName && hasLabel) {
      const nameCol = columns.find((c2) => ['file', 'filename', 'name', 'path', 'audio', 'wav', 'id'].indexOf(lower(c2)) >= 0);
      const labelCol = columns.find((c2) => LABEL_NAMES.indexOf(lower(c2)) >= 0);
      const map = new Map();
      for (const row of table.rows) {
        const key = baseName(row[nameCol]);
        map.set(key, String(row[labelCol] === undefined ? '' : row[labelCol]).trim());
      }
      return { map, nameCol, labelCol, source: c.name };
    }
  }
  return null;
}

export function importAudioFiles(files, options) {
  const opts = options || {};
  const cfg = audioFeatureConfig(opts);
  const audioFiles = files.filter((f) => /\.(wav|wave)$/i.test(f.name));
  const problems = [];
  if (!audioFiles.length) throw new Error('no WAV files found in the selection');
  const table = findLabelTable(files);
  const byFolder = new Map();
  for (const f of audioFiles) {
    const dir = parentDir(f.name);
    if (dir) byFolder.set(dir, (byFolder.get(dir) || 0) + 1);
  }
  const folderLabels = byFolder.size >= 2 && Array.from(byFolder.values()).every((v) => v > 0);
  const rows = [];
  const waves = [];
  for (const f of audioFiles) {
    let signal;
    try {
      signal = decodeAudioFile(f.bytes, cfg);
    } catch (e) {
      problems.push({ file: f.name, message: 'could not decode: ' + e.message });
      continue;
    }
    const key = baseName(f.name);
    let label = null;
    if (table && table.map.has(key)) label = table.map.get(key);
    else if (folderLabels) label = parentDir(f.name);
    else if (opts.defaultLabel !== undefined) label = String(opts.defaultLabel);
    if (label === null) {
      problems.push({ file: f.name, message: 'no label found; add a CSV with a file and label column, or put each class in its own folder' });
      continue;
    }
    let feature;
    try {
      feature = signalToFeature(signal, cfg);
    } catch (e) {
      problems.push({ file: f.name, message: 'feature extraction failed: ' + e.message });
      continue;
    }
    rows.push({ file: key, label, seconds: signal.length / cfg.sampleRate });
    waves.push({ clean: Float32Array.from(signal), noisy: null, sampleRate: cfg.sampleRate });
    feature.data = feature.data;
    rows[rows.length - 1].feature = feature.data;
    rows[rows.length - 1].shape = feature.shape;
  }
  if (!rows.length) throw new Error('no audio samples could be imported' + (problems.length ? ': ' + problems[0].message : ''));
  const shape = rows[0].shape;
  const per = shape[0] * shape[1];
  const mismatched = rows.filter((r) => r.shape[0] !== shape[0] || r.shape[1] !== shape[1]).length;
  if (mismatched) problems.push({ file: mismatched + ' files', message: 'produced a different feature size and were padded or truncated to ' + JSON.stringify(shape) });
  const x = new Float32Array(rows.length * per);
  rows.forEach((r, i) => x.set(r.feature.subarray(0, per), i * per));
  const counts = new Map();
  for (const r of rows) counts.set(r.label, (counts.get(r.label) || 0) + 1);
  const labels = Array.from(counts.entries()).sort((a, b) => (b[1] - a[1]) || (a[0] < b[0] ? -1 : 1)).map((e) => e[0]);
  const task = opts.task || (labels.length === 2 ? 'binary' : labels.length > 2 ? 'multiclass' : 'frame_binary');
  const y = new Float32Array(rows.length);
  rows.forEach((r, i) => { y[i] = Math.max(0, labels.indexOf(r.label)); });
  const ds = createDataset({
    name: opts.name || 'Imported audio',
    description: opts.description || '',
    modality: 'audio',
    task,
    x,
    xShape: shape.slice(),
    y,
    yShape: [1],
    labels,
    wave: waves,
    sampleRate: cfg.sampleRate,
    meta: rows.map((r) => ({ source: 'import', preview: r.file + ' (' + r.seconds.toFixed(2) + ' s)', text: r.file, file: r.file })),
    featureSpec: { kind: 'logmel', sampleRate: cfg.sampleRate, frameSize: cfg.frameSize, hop: cfg.hop, numMel: cfg.numMel, duration: cfg.duration, frames: shape[0], bins: shape[1] },
    license: opts.license || { id: 'user-supplied', name: 'Supplied by the importer', allowsCommercial: opts.allowsCommercial === true, notes: 'Audio you imported. Confirm you hold the rights to the recordings and to any speech in them.' },
    provenance: { origin: 'import', format: 'wav', files: rows.length, labelSource: table ? table.source : folderLabels ? 'folder names' : 'default', audioConfig: cfg },
    config: Object.assign({}, cfg),
    splitMethod: opts.splitMethod || 'stratified',
    splitRatios: opts.splitRatios || [0.7, 0.15, 0.15]
  });
  ds.importReport = { rows: rows.length, problems, labelSource: table ? table.source : folderLabels ? 'folder names' : 'default', task, classes: labels.length };
  return ds;
}

export function importImageFiles(files, options) {
  const opts = options || {};
  const decoder = opts.decodeImage;
  const images = files.filter((f) => /\.(png|jpe?g|webp|bmp)$/i.test(f.name));
  const problems = [];
  if (!images.length) throw new Error('no image files found in the selection');
  if (typeof decoder !== 'function') {
    throw new Error('image decoding needs a browser canvas; import a CSV of features instead or run this in the browser');
  }
  const table = findLabelTable(files);
  const byFolder = new Map();
  for (const f of images) {
    const dir = parentDir(f.name);
    if (dir) byFolder.set(dir, (byFolder.get(dir) || 0) + 1);
  }
  const folderLabels = byFolder.size >= 2;
  const decoded = [];
  for (const f of images) {
    const key = baseName(f.name);
    let label = null;
    if (table && table.map.has(key)) label = table.map.get(key);
    else if (folderLabels) label = parentDir(f.name);
    else if (opts.defaultLabel !== undefined) label = String(opts.defaultLabel);
    if (label === null) {
      problems.push({ file: f.name, message: 'no label found; add a CSV with file and label columns, or use one folder per class' });
      continue;
    }
    try {
      const img = decoder(f.bytes, f.name);
      decoded.push({ file: key, label, image: img.data, channels: img.channels, height: img.height, width: img.width });
    } catch (e) {
      problems.push({ file: f.name, message: 'decode failed: ' + e.message });
    }
  }
  if (!decoded.length) throw new Error('no images could be imported' + (problems.length ? ': ' + problems[0].message : ''));
  const size = opts.imageSize || Math.min(decoded[0].height, decoded[0].width);
  const channels = decoded[0].channels;
  const resizer = opts.resizeImage;
  const per = channels * size * size;
  const x = new Float32Array(decoded.length * per);
  decoded.forEach((d, i) => {
    let data = d.image;
    if (d.height !== size || d.width !== size || d.channels !== channels) {
      if (typeof resizer !== 'function') throw new Error('image ' + d.file + ' is ' + d.channels + 'x' + d.height + 'x' + d.width + ' and no resizer was supplied');
      data = resizer(d.image, d.channels, d.height, d.width, channels, size, size);
    }
    x.set(data.subarray(0, per), i * per);
  });
  const counts = new Map();
  for (const d of decoded) counts.set(d.label, (counts.get(d.label) || 0) + 1);
  const labels = Array.from(counts.entries()).sort((a, b) => (b[1] - a[1]) || (a[0] < b[0] ? -1 : 1)).map((e) => e[0]);
  const task = opts.task || (labels.length === 2 ? 'binary' : 'multiclass');
  const y = new Float32Array(decoded.length);
  decoded.forEach((d, i) => { y[i] = Math.max(0, labels.indexOf(d.label)); });
  const ds = createDataset({
    name: opts.name || 'Imported images',
    modality: 'image',
    task,
    x,
    xShape: [channels, size, size],
    y,
    yShape: [1],
    labels,
    meta: decoded.map((d) => ({ source: 'import', preview: d.file, text: d.file, file: d.file })),
    featureSpec: { kind: 'rgb_unit', channels, size },
    license: opts.license || { id: 'user-supplied', name: 'Supplied by the importer', allowsCommercial: opts.allowsCommercial === true, notes: 'Images you imported. Check the rights before shipping a model trained on them.' },
    provenance: { origin: 'import', format: 'images', files: decoded.length, labelSource: table ? table.source : folderLabels ? 'folder names' : 'default' },
    splitMethod: opts.splitMethod || 'stratified',
    splitRatios: opts.splitRatios || [0.7, 0.15, 0.15]
  });
  ds.importReport = { rows: decoded.length, problems, task, classes: labels.length };
  return ds;
}

export function expandArchive(bytes, options) {
  const opts = options || {};
  const entries = zipRead(bytes);
  const files = [];
  const problems = [];
  for (const e of entries) {
    if (e.name.endsWith('/')) continue;
    if (e.name.indexOf('__MACOSX') === 0 || baseName(e.name).startsWith('.')) continue;
    if (!e.crcOk) problems.push({ file: e.name, message: 'crc mismatch, the archive may be corrupt' });
    const textLike = /\.(csv|tsv|jsonl|json|txt)$/i.test(e.name);
    files.push({ name: e.name, bytes: e.data, text: textLike ? utf8Decode(e.data) : undefined });
    if (files.length > (opts.maxFiles || 4000)) break;
  }
  return { files, problems };
}

export function importFiles(files, options) {
  const opts = options || {};
  const all = [];
  const problems = [];
  for (const f of files || []) {
    const bytes = f.bytes instanceof Uint8Array ? f.bytes : f.bytes ? new Uint8Array(f.bytes) : null;
    const format = f.format || detectFormat(f.name, bytes);
    if (format === 'zip' && bytes) {
      try {
        const expanded = expandArchive(bytes, opts);
        for (const e of expanded.files) all.push(e);
        for (const p of expanded.problems) problems.push(p);
      } catch (e) {
        problems.push({ file: f.name, message: 'could not read the archive: ' + e.message });
      }
      continue;
    }
    all.push({ name: f.name, bytes, text: f.text });
  }
  if (!all.length) return { datasets: [], problems: [{ file: '', message: 'nothing to import' }], report: null };
  const audio = all.filter((f) => /\.(wav|wave)$/i.test(f.name));
  const images = all.filter((f) => /\.(png|jpe?g|webp|bmp)$/i.test(f.name));
  if (audio.length) {
    const ds = importAudioFiles(all, opts);
    return { datasets: [ds], problems: problems.concat((ds.importReport && ds.importReport.problems) || []), report: { kind: 'audio', files: all.length, audio: audio.length } };
  }
  if (images.length) {
    const ds = importImageFiles(all, opts);
    return { datasets: [ds], problems: problems.concat((ds.importReport && ds.importReport.problems) || []), report: { kind: 'image', files: all.length, images: images.length } };
  }
  const tables = all.filter((f) => /\.(csv|tsv|jsonl|json|txt)$/i.test(f.name));
  if (!tables.length) return { datasets: [], problems: problems.concat([{ file: all[0].name, message: 'unrecognised file type' }]), report: null };
  const datasets = [];
  for (const t of tables) {
    const text = t.text === undefined ? utf8Decode(t.bytes) : t.text;
    if (looksBinary(t.bytes || new Uint8Array(0))) {
      problems.push({ file: t.name, message: 'file does not look like text' });
      continue;
    }
    try {
      if (/\.rootism\.json$/i.test(t.name)) {
        const payload = JSON.parse(text);
        datasets.push(nativeToDataset(payload));
        continue;
      }
      let table = null;
      if (/\.jsonl$/i.test(t.name)) table = parseJsonl(text);
      else if (/\.json$/i.test(t.name)) {
        const parsed = parseJsonTable(text);
        if (parsed && parsed.native) {
          datasets.push(nativeToDataset(parsed.native));
          continue;
        }
        table = parsed;
      } else if (/\.txt$/i.test(t.name)) table = parseTextLines(text);
      else table = parseDelimited(text);
      if (!table || !table.rows || !table.rows.length) {
        problems.push({ file: t.name, message: 'no rows found' });
        continue;
      }
      datasets.push(tableToDataset(Object.assign({}, opts, {
        rows: table.rows,
        columns: table.columns,
        name: opts.name || baseName(t.name),
        format: detectFormat(t.name, null)
      })));
    } catch (e) {
      problems.push({ file: t.name, message: e.message });
    }
  }
  return { datasets, problems, report: { kind: 'table', files: all.length, tables: tables.length } };
}

export function datasetToNative(ds) {
  const payload = {
    rootismDataset: true,
    version: NATIVE_VERSION,
    exportedAt: Date.now(),
    name: ds.name,
    source: ds.source,
    presetKey: ds.presetKey,
    slug: ds.slug,
    description: ds.description,
    modality: ds.modality,
    task: ds.task,
    license: ds.license,
    provenance: ds.provenance,
    featureSpec: ds.featureSpec,
    config: ds.config,
    labels: ds.labels,
    xShape: ds.xShape,
    yShape: ds.yShape,
    splitMethod: ds.splitMethod,
    splitRatios: ds.splitRatios,
    splitSeed: ds.splitSeed,
    columns: ds.columns,
    sampleRate: ds.sampleRate,
    digest: ds.digest,
    count: ds.count,
    x: ds.x ? float32ToBase64(ds.x) : null,
    y: ds.y ? float32ToBase64(ds.y) : null,
    x2: ds.x2 ? float32ToBase64(ds.x2) : null,
    lengths: ds.lengths ? base64Encode(new Uint8Array(ds.lengths.buffer, ds.lengths.byteOffset, ds.lengths.byteLength)) : null,
    groups: ds.groups ? base64Encode(new Uint8Array(ds.groups.buffer, ds.groups.byteOffset, ds.groups.byteLength)) : null,
    splits: ds.splits ? {
      train: base64Encode(new Uint8Array(ds.splits.train.buffer, ds.splits.train.byteOffset, ds.splits.train.byteLength)),
      val: base64Encode(new Uint8Array(ds.splits.val.buffer, ds.splits.val.byteOffset, ds.splits.val.byteLength)),
      test: base64Encode(new Uint8Array(ds.splits.test.buffer, ds.splits.test.byteOffset, ds.splits.test.byteLength))
    } : null,
    boxes: ds.boxes || null,
    meta: opts_meta(ds)
  };
  return payload;
}

function opts_meta(ds) {
  if (!ds.meta) return null;
  const limit = Math.min(ds.meta.length, 5000);
  return ds.meta.slice(0, limit).map((m) => {
    if (!m) return null;
    const out = {};
    for (const k of Object.keys(m)) {
      const v = m[k];
      if (v === null || typeof v !== 'object') out[k] = v;
    }
    return out;
  });
}

function int32FromBase64(text) {
  const bytes = base64Decode(text);
  const aligned = bytes.byteOffset % 4 === 0 ? bytes : bytes.slice();
  return new Int32Array(aligned.buffer, aligned.byteOffset, aligned.byteLength / 4);
}

export function nativeToDataset(payload) {
  if (!payload || !payload.rootismDataset) throw new Error('not a Rootism dataset payload');
  if (payload.version > NATIVE_VERSION) throw new Error('dataset was written by a newer version of the tool');
  const splits = payload.splits ? {
    train: int32FromBase64(payload.splits.train),
    val: int32FromBase64(payload.splits.val),
    test: int32FromBase64(payload.splits.test)
  } : null;
  const ds = createDataset({
    name: payload.name,
    source: payload.source || (payload.presetKey ? 'preset' : 'custom'),
    presetKey: payload.presetKey,
    slug: payload.slug,
    description: payload.description,
    modality: payload.modality,
    task: payload.task,
    license: payload.license,
    provenance: payload.provenance,
    featureSpec: payload.featureSpec,
    config: payload.config,
    labels: payload.labels || [],
    xShape: payload.xShape,
    yShape: payload.yShape || [1],
    x: payload.x ? base64ToFloat32(payload.x) : null,
    y: payload.y ? base64ToFloat32(payload.y) : null,
    x2: payload.x2 ? base64ToFloat32(payload.x2) : null,
    lengths: payload.lengths ? int32FromBase64(payload.lengths) : null,
    groups: payload.groups ? int32FromBase64(payload.groups) : null,
    splits,
    splitMethod: payload.splitMethod,
    splitRatios: payload.splitRatios,
    splitSeed: payload.splitSeed,
    columns: payload.columns,
    sampleRate: payload.sampleRate,
    boxes: payload.boxes,
    meta: payload.meta,
    dedupe: false
  });
  return ds;
}

function sampleText(ds, i) {
  if (!ds.meta || !ds.meta[i]) return null;
  const m = ds.meta[i];
  return m.text || m.preview || (m.file ? m.file : null);
}

function splitOf(ds, i) {
  if (!ds.splits) return 'unused';
  if (ds.splits.train.indexOf(i) >= 0) return 'train';
  if (ds.splits.val.indexOf(i) >= 0) return 'val';
  if (ds.splits.test.indexOf(i) >= 0) return 'test';
  return 'unused';
}

export function datasetToCsv(ds, options) {
  const opts = options || {};
  const limit = opts.limit === undefined ? ds.count : Math.min(ds.count, opts.limit);
  const per = ds.xShape.reduce((a, b) => a * b, 1);
  const yPer = (ds.yShape || [1]).reduce((a, b) => a * b, 1);
  const columns = ['index', 'split'].concat(opts.includeFeatures ? ds.xShape.length === 1 ? Array.from({ length: per }, (v, i) => 'x' + i) : ['features'] : []);
  if (ds.meta && ds.meta.some((m) => m && (m.text || m.preview))) columns.push('text');
  for (let k = 0; k < yPer; k++) columns.push(yPer === 1 ? 'label' : 'label_' + k);
  columns.push('label_name');
  const lines = [columns.join(',')];
  for (let i = 0; i < limit; i++) {
    const cells = [String(i), splitOf(ds, i)];
    if (opts.includeFeatures) {
      if (ds.xShape.length === 1) {
        for (let k = 0; k < per; k++) cells.push(csvCell(ds.x[i * per + k]));
      } else {
        cells.push(csvCell('[' + Array.from(ds.x.subarray(i * per, (i + 1) * per)).join(',') + ']'));
      }
    }
    const text = sampleText(ds, i);
    if (columns.indexOf('text') >= 0) cells.push(csvCell(text === null ? '' : text));
    for (let k = 0; k < yPer; k++) cells.push(csvCell(ds.y[i * yPer + k]));
    cells.push(csvCell(yPer === 1 && ds.labels && ds.labels.length ? ds.labels[Math.round(ds.y[i])] || '' : ''));
    lines.push(cells.join(','));
  }
  return lines.join('\n') + '\n';
}

function csvCell(value) {
  const s = value === null || value === undefined ? '' : String(value);
  if (/[",\n]/.test(s)) return '"' + s.replace(/"/g, '""') + '"';
  return s;
}

export function datasetToJsonl(ds, options) {
  const opts = options || {};
  const limit = opts.limit === undefined ? ds.count : Math.min(ds.count, opts.limit);
  const per = ds.xShape.reduce((a, b) => a * b, 1);
  const yPer = (ds.yShape || [1]).reduce((a, b) => a * b, 1);
  const lines = [];
  for (let i = 0; i < limit; i++) {
    const row = { index: i, split: splitOf(ds, i) };
    const text = sampleText(ds, i);
    if (text !== null) row.text = text;
    if (opts.includeFeatures) row.features = Array.from(ds.x.subarray(i * per, (i + 1) * per));
    if (yPer === 1) {
      row.label = ds.labels && ds.labels.length ? ds.labels[Math.round(ds.y[i])] || ds.y[i] : ds.y[i];
      row.label_index = Math.round(ds.y[i]);
    } else {
      row.label = Array.from(ds.y.subarray(i * yPer, (i + 1) * yPer));
    }
    lines.push(JSON.stringify(row));
  }
  return lines.join('\n') + '\n';
}

export function datasetToCoco(ds) {
  const boxes = ds.boxes || [];
  const H = ds.xShape.length >= 2 ? ds.xShape[ds.xShape.length - 2] : 0;
  const W = ds.xShape.length >= 1 ? ds.xShape[ds.xShape.length - 1] : 0;
  const categories = (ds.labels && ds.labels.length ? ds.labels : ['object']).map((name, i) => ({ id: i + 1, name, supercategory: 'none' }));
  const images = [];
  const annotations = [];
  let annId = 1;
  for (let i = 0; i < ds.count; i++) {
    images.push({ id: i + 1, file_name: (ds.meta && ds.meta[i] && ds.meta[i].file) || 'sample_' + i + '.png', width: W, height: H, split: splitOf(ds, i) });
    const list = boxes[i] || [];
    for (const b of list) {
      const x = (b.x - b.w / 2) * W;
      const y = (b.y - b.h / 2) * H;
      annotations.push({
        id: annId++,
        image_id: i + 1,
        category_id: (b.category === undefined ? 0 : b.category) + 1,
        bbox: [Number(x.toFixed(3)), Number(y.toFixed(3)), Number((b.w * W).toFixed(3)), Number((b.h * H).toFixed(3))],
        area: Number((b.w * W * b.h * H).toFixed(3)),
        iscrowd: 0,
        normalised: { cx: b.x, cy: b.y, w: b.w, h: b.h }
      });
    }
  }
  return { info: { description: ds.name, version: String(ds.version || 1), contributor: 'Rootism Labs', license: ds.license ? ds.license.name : '' }, images, annotations, categories };
}

export function datasetToWavArchive(ds, options) {
  const opts = options || {};
  if (!ds.wave || !ds.wave.length) throw new Error('this dataset holds no raw waveforms, so there is nothing to write as WAV');
  const entries = [];
  const limit = opts.limit === undefined ? ds.count : Math.min(ds.count, opts.limit);
  const lines = ['file,label,seconds'];
  for (let i = 0; i < limit; i++) {
    const w = ds.wave[i];
    if (!w || !w.clean) continue;
    const rate = w.sampleRate || ds.sampleRate || 16000;
    const name = 'audio/' + String(i).padStart(5, '0') + '.wav';
    entries.push({ name, data: wavEncodeBytes(w.clean, rate) });
    const label = ds.labels && ds.labels.length ? ds.labels[Math.round(ds.y[i])] || String(ds.y[i]) : String(ds.y[i]);
    lines.push([name.replace('audio/', ''), csvCell(label), (w.clean.length / rate).toFixed(3)].join(','));
  }
  entries.push({ name: 'labels.csv', data: lines.join('\n') + '\n' });
  return entries;
}

function wavEncodeBytes(samples, sampleRate) {
  return wavEncode(samples, sampleRate, 1);
}

export function exportDataset(ds, format, options) {
  const opts = options || {};
  const slug = (ds.slug || ds.name || 'dataset').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'dataset';
  if (format === 'csv') return [{ name: slug + '.csv', data: datasetToCsv(ds, opts), mime: 'text/csv' }];
  if (format === 'jsonl') return [{ name: slug + '.jsonl', data: datasetToJsonl(ds, opts), mime: 'application/x-ndjson' }];
  if (format === 'native') return [{ name: slug + '.rootism.json', data: JSON.stringify(datasetToNative(ds)), mime: 'application/json' }];
  if (format === 'labels') {
    const lines = (ds.labels || []).map((l, i) => i + '\t' + l);
    return [{ name: slug + '-labels.txt', data: lines.join('\n') + '\n', mime: 'text/plain' }];
  }
  if (format === 'coco') return [{ name: slug + '.coco.json', data: JSON.stringify(datasetToCoco(ds), null, 1), mime: 'application/json' }];
  if (format === 'wav') {
    const entries = datasetToWavArchive(ds, opts);
    return [{ name: slug + '-audio.zip', data: zipWrite(entries), mime: 'application/zip' }];
  }
  throw new Error('unknown export format ' + format);
}

export { MODALITIES, recompute, baseName, parentDir, extensionOf, utf8Decode, zipRead, wavDecode, resample, peakNormalize, logMelFrames, normalizeFeatures, base64Encode, base64Decode, float32ToBase64, base64ToFloat32 };
