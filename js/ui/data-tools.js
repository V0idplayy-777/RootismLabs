import { el, clear, attachTooltip, download, pickFiles, readFileBytes, modal } from './dom.js';
import { barChart, heatmap, imageToCanvas, waveform, lineChart, SERIES_COLORS } from './charts.js';
import { AUG_OPS, AUG_GROUPS, opsForModality, opById, augmentPreview } from '../data/augment.js';
import { buildSplits, qualityReport, computeStats } from '../data/dataset.js';
import { kFold, splitSummary } from '../data/split.js';
import { IMPORT_FORMATS, EXPORT_FORMATS, detectFormat, importFiles, exportDataset, datasetToNative } from '../data/import.js';
import { formatBytes } from '../presets/metrics.js';
import { shapeSize } from '../core/tensor.js';

export function sampleText(ds, index) {
  const meta = ds.meta && ds.meta[index] ? ds.meta[index] : null;
  if (!meta) return null;
  return meta.text || meta.preview || meta.label || null;
}

export function labelName(ds, index) {
  if (!ds.labels || !ds.labels.length) return null;
  const per = shapeSize(ds.yShape || [1]);
  if (!ds.y) return null;
  const raw = ds.y[index * per];
  if (per === 1) return ds.labels[Math.round(raw)] !== undefined ? ds.labels[Math.round(raw)] : String(raw);
  const active = [];
  for (let i = 0; i < per; i++) if (ds.y[index * per + i] > 0.5) active.push(ds.labels[i] !== undefined ? ds.labels[i] : String(i));
  return active.length ? active.join(', ') : '(none)';
}

export function samplePreviewCell(ds, index, options) {
  const opts = options || {};
  const size = opts.size || 96;
  const per = shapeSize(ds.xShape);
  const slice = ds.x.subarray(index * per, (index + 1) * per);
  const modality = ds.modality;
  if (modality === 'image' && ds.xShape.length === 3) {
    const canvas = el('canvas');
    imageToCanvas(canvas, slice, { channels: ds.xShape[0], width: ds.xShape[2], height: ds.xShape[1], displayWidth: size, displayHeight: Math.round(size * ds.xShape[1] / ds.xShape[2]) });
    return canvas;
  }
  if (modality === 'audio') {
    if (ds.wave && ds.wave[index] && ds.wave[index].clean) {
      const canvas = el('canvas', { style: { width: size + 'px', height: '34px' } });
      waveform(canvas, ds.wave[index].clean, { width: size, height: 34 });
      return canvas;
    }
    if (ds.xShape.length === 2) {
      const canvas = el('canvas');
      const frames = ds.xShape[0];
      const bins = ds.xShape[1];
      const grid = [];
      for (let f = 0; f < frames; f++) grid.push(Array.from(slice.subarray(f * bins, (f + 1) * bins)));
      heatmap(canvas, grid, { width: size, height: Math.round(size * 0.62), ramp: 'magma', pad: { top: 2, right: 2, bottom: 2, left: 2 } });
      return canvas;
    }
  }
  if ((modality === 'series' || modality === 'sensor') && ds.xShape.length === 2) {
    const canvas = el('canvas');
    const steps = ds.xShape[0];
    const channels = ds.xShape[1];
    const series = [];
    for (let c = 0; c < Math.min(channels, 4); c++) {
      const values = [];
      for (let t = 0; t < steps; t++) values.push(slice[t * channels + c]);
      series.push({ label: 'ch' + c, y: values, color: SERIES_COLORS[c] });
    }
    lineChart(canvas, series, { width: size, height: 40, legend: false, hover: false, pad: { top: 3, right: 3, bottom: 3, left: 3 }, background: '#14171a' });
    return canvas;
  }
  const canvas = el('canvas');
  const values = [];
  const stride = Math.max(1, Math.floor(slice.length / 64));
  for (let i = 0; i < slice.length; i += stride) values.push({ value: slice[i] });
  barChart(canvas, values, { width: size, height: 34, horizontal: false, values: false, labelChars: 0, pad: { top: 3, right: 3, bottom: 3, left: 3 }, background: '#14171a', color: '#4fa3a5' });
  return canvas;
}

export function classBalanceView(ds) {
  const host = el('div', { class: 'col gap6' });
  const stats = ds.stats || computeStats(ds);
  if (!stats.classCounts || !stats.classCounts.length) {
    host.appendChild(el('div', { class: 'tiny faint', text: 'This dataset has no discrete class labels, so there is no balance to report.' }));
    return host;
  }
  const counts = stats.classCounts;
  const names = stats.classNames || stats.labelNames || counts.map((c, i) => 'class ' + i);
  const total = counts.reduce((a, b) => a + b, 0) || 1;
  const bars = counts.map((c, i) => ({
    label: names[i],
    value: c,
    display: c + '  (' + (100 * c / total).toFixed(1) + '%)',
    tone: c === 0 ? 'bad' : c < total / counts.length * 0.35 ? 'warn' : 'good'
  }));
  const canvas = el('canvas', { style: { width: '100%', height: Math.max(60, counts.length * 17 + 22) + 'px' } });
  barChart(canvas, bars, { horizontal: true, labelChars: 20, background: '#191c1f' });
  host.appendChild(canvas);
  const imbalance = Math.max(...counts) / Math.max(1, Math.min(...counts.filter((c) => c > 0)));
  const tone = imbalance > 20 ? 'bad' : imbalance > 6 ? 'warn' : 'good';
  host.appendChild(el('div', { class: 'callout ' + tone }, [
    el('div', { text: 'Imbalance ratio ' + imbalance.toFixed(1) + ':1 · ' + counts.length + ' classes · ' + total + ' labelled samples.' }),
    el('div', { class: 'tiny faint mt4', text: counts.some((c) => c === 0) ? 'At least one class has no samples at all; it can never be learned or measured.' : 'Class-weighted loss or a balanced sampler on the Train panel is the usual fix for a ratio above about 6:1.' })
  ]));
  return host;
}

export function augmentView(ctx, ds, onChange) {
  const host = el('div', { class: 'col gap8' });
  const cfg = ds.augmentation || { enabled: false, applyProb: 0.8, ops: [] };
  ds.augmentation = cfg;
  const ops = opsForModality(ds.modality);
  if (!ops.length) {
    host.appendChild(el('div', { class: 'callout warn', text: 'No augmentation operators are defined for the "' + ds.modality + '" modality.' }));
    return host;
  }
  host.appendChild(el('div', { class: 'row between' }, [
    el('label', { class: 'inline' }, [
      el('input', {
        type: 'checkbox', checked: !!cfg.enabled, onchange: (event) => {
          cfg.enabled = event.target.checked;
          onChange();
          ctx.info('Augmentation ' + (cfg.enabled ? 'enabled' : 'disabled') + ' for training batches.');
        }
      }),
      el('span', { class: 'bold', text: 'Apply augmentation during training' })
    ]),
    el('span', { class: 'tiny faint', text: 'validation and test splits are never augmented' })
  ]));
  host.appendChild(el('div', { class: 'inspector-field' }, [
    el('span', { text: 'apply prob' }),
    el('input', {
      type: 'range', min: 0, max: 1, step: 0.05, value: cfg.applyProb, style: { width: '100%' },
      oninput: (event) => { cfg.applyProb = parseFloat(event.target.value); const out = host.querySelector('#applyProbOut'); if (out) out.textContent = cfg.applyProb.toFixed(2); },
      onchange: () => onChange()
    })
  ]));
  host.appendChild(el('div', { class: 'tiny faint', id: 'applyProbOut', text: 'probability a given sample is augmented at all: ' + Number(cfg.applyProb).toFixed(2) }));

  const list = el('div', { class: 'col gap6' });
  for (const group of AUG_GROUPS) {
    const groupOps = ops.filter((o) => o.group === group);
    if (!groupOps.length) continue;
    const block = el('div', { class: 'opt-block' }, [el('h4', { text: group })]);
    for (const op of groupOps) {
      const active = cfg.ops.find((o) => o.id === op.id);
      const row = el('div', { class: 'row gap6', style: { marginBottom: '5px', flexWrap: 'wrap' } }, [
        el('label', { class: 'inline', style: { width: '148px' } }, [
          el('input', {
            type: 'checkbox', checked: !!active, onchange: (event) => {
              if (event.target.checked) cfg.ops.push({ id: op.id, strength: op.strength.def, prob: 0.5 });
              else cfg.ops = cfg.ops.filter((o) => o.id !== op.id);
              onChange();
              renderAugmentPreview();
            }
          }),
          el('span', { text: op.label })
        ])
      ]);
      attachTooltip(row.querySelector('label'), () => '<b>' + op.label + '</b><br>' + op.describe + '<br><span style="color:#8b9298">strength unit</span> ' + (op.strength.unit || '—'));
      const entry = cfg.ops.find((o) => o.id === op.id);
      const strength = el('input', {
        type: 'range', min: op.strength.min, max: op.strength.max, step: op.strength.step,
        value: entry ? entry.strength : op.strength.def, style: { width: '104px' }, disabled: !entry,
        oninput: (event) => {
          const found = cfg.ops.find((o) => o.id === op.id);
          if (!found) return;
          found.strength = parseFloat(event.target.value);
          strengthOut.textContent = found.strength + ' ' + (op.strength.unit || '');
        },
        onchange: () => { onChange(); renderAugmentPreview(); }
      });
      const prob = el('input', {
        type: 'range', min: 0, max: 1, step: 0.05, value: entry ? entry.prob : 0.5, style: { width: '76px' }, disabled: !entry,
        oninput: (event) => {
          const found = cfg.ops.find((o) => o.id === op.id);
          if (!found) return;
          found.prob = parseFloat(event.target.value);
          probOut.textContent = 'p ' + found.prob.toFixed(2);
        },
        onchange: () => { onChange(); renderAugmentPreview(); }
      });
      const strengthOut = el('span', { class: 'tiny mono faint', style: { width: '86px' }, text: (entry ? entry.strength : op.strength.def) + ' ' + (op.strength.unit || '') });
      const probOut = el('span', { class: 'tiny mono faint', style: { width: '42px' }, text: 'p ' + (entry ? entry.prob.toFixed(2) : '0.50') });
      row.appendChild(strength);
      row.appendChild(strengthOut);
      row.appendChild(prob);
      row.appendChild(probOut);
      block.appendChild(row);
    }
    list.appendChild(block);
  }
  host.appendChild(list);

  const previewHost = el('div', { class: 'panel' }, [
    el('div', { class: 'panel-head' }, [el('h3', { text: 'Augmentation preview' }), el('div', { class: 'spacer' }), el('button', { class: 'xs ghost', text: 'Resample', onclick: () => renderAugmentPreview() })]),
    el('div', { class: 'panel-body tight', id: 'augPreviewBody' })
  ]);
  host.appendChild(previewHost);

  function renderAugmentPreview() {
    const body = previewHost.querySelector('#augPreviewBody');
    if (!body) return;
    clear(body);
    if (!cfg.enabled || !cfg.ops.length) {
      body.appendChild(el('div', { class: 'tiny faint', text: 'Enable augmentation and pick at least one operator to see a before/after preview.' }));
      return;
    }
    const trainIdx = ds.splits && ds.splits.train && ds.splits.train.length ? ds.splits.train : null;
    const index = trainIdx ? trainIdx[Math.floor(Math.random() * trainIdx.length)] : Math.floor(Math.random() * ds.count);
    let preview;
    try {
      preview = augmentPreview(ds, index, Object.assign({}, cfg, { enabled: true, applyProb: 1 }), Math.floor(Math.random() * 1e6));
    } catch (e) {
      body.appendChild(el('div', { class: 'callout bad', text: 'Preview failed: ' + e.message }));
      return;
    }
    const text = sampleText(ds, index);
    body.appendChild(el('div', { class: 'tiny faint mb4', text: 'sample #' + index + (text ? ' · ' + text.slice(0, 90) : '') + ' · label ' + (labelName(ds, index) || '—') }));
    const pair = el('div', { class: 'row gap8 top' });
    pair.appendChild(previewBox('original', preview.original, ds));
    pair.appendChild(previewBox('augmented', preview.augmented, ds));
    body.appendChild(pair);
    const opsActive = cfg.ops.map((o) => { const meta = opById(o.id); return (meta ? meta.label : o.id) + ' ' + o.strength + ' @ p' + o.prob.toFixed(2); }).join(' · ');
    body.appendChild(el('div', { class: 'tiny faint mt4', text: opsActive }));
  }

  renderAugmentPreview();
  return host;
}

function previewBox(title, values, ds) {
  const per = values.length;
  const box = el('div', { class: 'col gap4', style: { flex: '1 1 0', minWidth: '0' } }, [el('div', { class: 'tiny upper faint', text: title })]);
  if (ds.modality === 'image' && ds.xShape.length === 3 && per === shapeSize(ds.xShape)) {
    const canvas = el('canvas');
    imageToCanvas(canvas, values, { channels: ds.xShape[0], width: ds.xShape[2], height: ds.xShape[1], displayWidth: 132, displayHeight: Math.round(132 * ds.xShape[1] / ds.xShape[2]) });
    box.appendChild(canvas);
  } else if (ds.xShape.length === 2 && per === shapeSize(ds.xShape)) {
    const canvas = el('canvas');
    const frames = ds.xShape[0];
    const bins = ds.xShape[1];
    const grid = [];
    for (let f = 0; f < frames; f++) grid.push(Array.from(values.subarray(f * bins, (f + 1) * bins)));
    heatmap(canvas, grid, { width: 150, height: 92, ramp: 'magma', pad: { top: 2, right: 2, bottom: 2, left: 2 } });
    box.appendChild(canvas);
  } else {
    const canvas = el('canvas', { style: { width: '100%', height: '52px' } });
    const stride = Math.max(1, Math.floor(per / 72));
    const bars = [];
    for (let i = 0; i < per; i += stride) bars.push({ value: values[i] });
    barChart(canvas, bars, { horizontal: false, values: false, labelChars: 0, background: '#14171a', pad: { top: 3, right: 3, bottom: 3, left: 3 }, color: '#4fa3a5' });
    box.appendChild(canvas);
  }
  return box;
}

export function qualityView(ctx, ds) {
  const host = el('div', { class: 'col gap8' });
  const report = ds.quality || qualityReport(ds);
  ds.quality = report;
  const issues = report.issues || [];
  const counts = { error: 0, warn: 0, info: 0 };
  for (const issue of issues) counts[issue.severity] = (counts[issue.severity] || 0) + 1;
  host.appendChild(el('div', { class: 'kpi-row' }, [
    kpi('errors', String(counts.error || 0), counts.error ? 'must fix before shipping' : 'none'),
    kpi('warnings', String(counts.warn || 0), counts.warn ? 'review before training' : 'none'),
    kpi('notes', String(counts.info || 0), 'informational'),
    kpi('duplicates', String((report.duplicates && report.duplicates.exact) || 0), (report.duplicates && report.duplicates.crossSplit) ? (report.duplicates.crossSplit + ' leak across splits') : 'exact matches'),
    kpi('samples', String(ds.count), ds.splits ? (ds.splits.train.length + '/' + ds.splits.val.length + '/' + ds.splits.test.length + ' split') : 'unsplit')
  ]));
  const list = el('div', { class: 'wq' });
  if (!issues.length) list.appendChild(el('div', { class: 'item good' }, [el('span', { class: 'ic green', text: '✓' }), el('span', { text: 'No data problems detected.' }), el('span', { class: 'n', text: '' })]));
  for (const issue of issues) {
    const tone = issue.severity === 'error' ? 'bad' : issue.severity === 'warn' ? 'warn' : '';
    const icon = issue.severity === 'error' ? '✕' : issue.severity === 'warn' ? '!' : 'i';
    const color = issue.severity === 'error' ? 'red' : issue.severity === 'warn' ? 'amber' : 'faint';
    list.appendChild(el('div', { class: 'item ' + tone }, [
      el('span', { class: 'ic ' + color, text: icon }),
      el('span', {}, [
        el('div', { text: issue.message }),
        issue.detail ? el('div', { class: 'tiny faint', text: issue.detail }) : null
      ]),
      el('span', { class: 'n', text: issue.code })
    ]));
  }
  host.appendChild(list);
  host.appendChild(el('div', { class: 'row gap4' }, [
    el('button', { class: 'sm ghost', text: 'Re-run checks', onclick: () => { ds.quality = qualityReport(ds); ctx.touchDataset(); ctx.info('Data quality checks re-run: ' + ds.quality.issues.length + ' findings.'); } }),
    el('button', { class: 'sm ghost', text: 'Download report', onclick: () => download((ds.slug || 'dataset') + '-quality.json', JSON.stringify(ds.quality, null, 2), 'application/json') })
  ]));
  return host;
}

function kpi(key, value, sub) {
  return el('div', { class: 'stat' }, [
    el('div', { class: 'k', text: key }),
    el('div', { class: 'v', text: value }),
    el('div', { class: 's', text: sub || '' })
  ]);
}

export function exportDataView(ctx, ds) {
  const host = el('div', { class: 'col gap8' });
  host.appendChild(el('div', { class: 'tiny dim', text: 'Every export carries the label list and the split assignment so the receiving tool can reproduce the same experiment.' }));
  const grid = el('div', { class: 'col gap6' });
  for (const format of EXPORT_FORMATS) {
    const available = formatAvailable(format, ds);
    const row = el('div', { class: 'artifact-row' }, [
      el('div', {}, [
        el('div', { class: 'bold', text: format.label }),
        el('div', { class: 'tiny faint', text: FORMAT_NOTES[format.key] || '' })
      ]),
      el('span', { class: 'badge', text: format.extension }),
      el('span', { class: 'mono tiny faint', text: available ? '' : (format.key === 'wav' ? 'no waveform stored' : 'no boxes stored') }),
      el('button', {
        class: 'sm', text: 'Download', disabled: !available,
        onclick: () => {
          try {
            const files = exportDataset(ds, format.key);
            let total = 0;
            for (const file of files) {
              download(file.name, file.data, file.mime);
              total += file.data.length || file.data.byteLength || 0;
            }
            ctx.ok('Exported ' + files.map((f) => f.name).join(', ') + ' (' + formatBytes(total) + ').');
          } catch (e) {
            ctx.error('Export failed: ' + e.message);
          }
        }
      })
    ]);
    grid.appendChild(row);
  }
  host.appendChild(grid);
  host.appendChild(el('div', { class: 'sep' }));
  host.appendChild(el('div', { class: 'row gap6 wrap' }, [
    el('button', { class: 'sm ghost', text: 'Copy digest', onclick: () => { navigator.clipboard.writeText(ds.digest ? ds.digest.contentHash : '').then(() => ctx.toast('Dataset hash copied.')); } }),
    el('button', { class: 'sm ghost', text: 'Download card fragment', onclick: () => download((ds.slug || 'dataset') + '-provenance.json', JSON.stringify({ name: ds.name, count: ds.count, digest: ds.digest, license: ds.license, provenance: ds.provenance, splits: splitCounts(ds), featureSpec: ds.featureSpec, stats: ds.stats }, null, 2), 'application/json') })
  ]));
  return host;
}

const FORMAT_NOTES = {
  csv: 'One row per sample: id, features, label, split. Re-importable.',
  jsonl: 'One JSON object per line. The usual hand-off format for data tooling.',
  native: 'Full fidelity: features, targets, waveforms, splits, digest and provenance.',
  labels: 'Index and class name per line, for annotation tools.',
  coco: 'COCO-style images/annotations/categories JSON for detection datasets.',
  wav: 'PCM16 WAV files in a ZIP plus the label table, for audio review.'
};

function formatAvailable(format, ds) {
  if (format.key === 'wav') return !!(ds.wave && ds.wave.length && ds.wave.some((w) => w && w.clean));
  if (format.key === 'coco') return !!(ds.boxes && ds.boxes.length);
  return true;
}

export function splitCounts(ds) {
  if (!ds.splits) return null;
  return { train: ds.splits.train.length, val: ds.splits.val.length, test: ds.splits.test.length };
}

export function importView(ctx, onDone) {
  const host = el('div', { class: 'col gap8' });
  const options = {
    audioFeatures: 'kws',
    labelColumn: '',
    delimiter: 'auto',
    task: 'auto',
    maxSizeMb: 200
  };
  host.appendChild(el('div', { class: 'tiny dim' }, [
    'Accepted: ',
    el('span', { class: 'mono', text: IMPORT_FORMATS.map((f) => f.label).join(' · ') })
  ]));
  const dz = el('div', { class: 'dropzone' }, [
    el('div', { class: 'bold', text: 'Drop files here, or click to browse' }),
    el('div', { class: 'tiny faint mt4', text: 'CSV / TSV / JSON / JSONL tables, WAV audio with a label table, images with a COCO or CSV label table, ZIP archives, folders, and .rootism.json exports.' })
  ]);
  const fileLabel = el('div', { class: 'tiny faint', text: 'no files selected' });
  let chosen = [];
  const choose = async () => {
    const files = await pickFiles('', true);
    if (!files.length) return;
    chosen = files;
    fileLabel.textContent = files.length + ' file(s) · ' + formatBytes(files.reduce((a, f) => a + f.size, 0));
    fileList(chosen);
  };
  dz.addEventListener('click', choose);
  dz.addEventListener('dragover', (event) => { event.preventDefault(); dz.classList.add('over'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('over'));
  dz.addEventListener('drop', async (event) => {
    event.preventDefault();
    dz.classList.remove('over');
    const files = await collectDrop(event.dataTransfer);
    if (!files.length) return;
    chosen = files;
    fileLabel.textContent = files.length + ' file(s) · ' + formatBytes(files.reduce((a, f) => a + f.size, 0));
    fileList(chosen);
  });
  host.appendChild(dz);
  host.appendChild(fileLabel);

  const listHost = el('div', { class: 'file-list hidden' });
  host.appendChild(listHost);
  function fileList(files) {
    clear(listHost);
    listHost.classList.remove('hidden');
    for (const file of files.slice(0, 200)) {
      listHost.appendChild(el('div', { class: 'fr' }, [
        el('span', { class: 'nm', text: file.webkitRelativePath || file.name }),
        el('span', { class: 'mono tiny faint', text: formatBytes(file.size) }),
        el('span', { class: 'badge', text: detectFormat(file.name, null) || 'unknown' })
      ]));
    }
    if (files.length > 200) listHost.appendChild(el('div', { class: 'fr tiny faint' }, [el('span', { text: '…and ' + (files.length - 200) + ' more' })]));
  }

  const opts = el('div', { class: 'config-grid' });
  opts.appendChild(selectRow('Audio features', options.audioFeatures, [
    { value: 'kws', label: 'log-mel 36×32 @ 8 kHz (keyword spotting)' },
    { value: 'speech16', label: 'log-mel 98×40 @ 16 kHz (speech)' },
    { value: 'events', label: 'log-mel 38×32 (event detection)' },
    { value: 'onset', label: 'onset envelope 98 bins' }
  ], (value) => { options.audioFeatures = value; }));
  opts.appendChild(selectRow('Task', options.task, [{ value: 'auto', label: 'infer from data' }].concat(Object.keys(TASK_LABELS).map((k) => ({ value: k, label: TASK_LABELS[k] }))), (value) => { options.task = value; }));
  opts.appendChild(textRow('Label column', options.labelColumn, 'blank = infer', (value) => { options.labelColumn = value; }));
  opts.appendChild(selectRow('Delimiter', options.delimiter, [{ value: 'auto', label: 'auto-detect' }, { value: ',', label: 'comma' }, { value: '\t', label: 'tab' }, { value: ';', label: 'semicolon' }, { value: '|', label: 'pipe' }], (value) => { options.delimiter = value; }));
  host.appendChild(opts);

  const status = el('div', { class: 'tiny faint', text: '' });
  host.appendChild(status);
  host.appendChild(el('div', { class: 'row gap6' }, [
    el('button', {
      class: 'primary', text: 'Import', onclick: async () => {
        if (!chosen.length) { status.textContent = 'choose some files first'; return; }
        status.textContent = 'reading ' + chosen.length + ' file(s)…';
        try {
          const files = [];
          for (const file of chosen) files.push({ name: file.webkitRelativePath || file.name, bytes: await readFileBytes(file) });
          const result = await importFiles(files, {
            audioFeatures: options.audioFeatures,
            task: options.task === 'auto' ? null : options.task,
            labelColumn: options.labelColumn || null,
            delimiter: options.delimiter === 'auto' ? null : options.delimiter
          });
          for (const problem of result.problems || []) ctx.warn('Import: ' + problem);
          if (!result.datasets || !result.datasets.length) {
            status.textContent = 'nothing could be imported';
            ctx.error('Import produced no dataset. See the log for the reasons.');
            return;
          }
          const ds = result.datasets[0];
          status.textContent = 'imported ' + ds.count + ' samples as "' + ds.name + '"';
          ctx.ok('Imported ' + ds.count + ' samples from ' + files.length + ' file(s) into "' + ds.name + '".');
          if (result.datasets.length > 1) ctx.info('The import produced ' + result.datasets.length + ' datasets; loaded the first one.');
          onDone(ds);
        } catch (e) {
          status.textContent = 'failed: ' + e.message;
          ctx.error('Import failed: ' + e.message);
        }
      }
    }),
    el('button', { class: 'ghost', text: 'Cancel', onclick: () => { const back = document.querySelector('.modal-back'); if (back) back.remove(); } })
  ]));
  return host;
}

const TASK_LABELS = {
  multiclass: 'multi-class', binary: 'binary', multilabel: 'multi-label', regression: 'regression',
  seq_tag: 'sequence tagging', lm: 'language model', image_map: 'image mapping', mask: 'mask',
  frame_binary: 'frame binary', detection: 'detection', embedding: 'embedding', anomaly: 'anomaly',
  spectral_mask: 'spectral mask', pitch_class: 'pitch class', onset: 'onset', forecast: 'forecast',
  coordinate_field: 'coordinate field'
};

function selectRow(label, value, choices, onChange) {
  return el('label', { class: 'field' }, [
    el('span', { class: 'lbl', text: label }),
    el('select', { onchange: (event) => onChange(event.target.value) }, choices.map((c) => el('option', { value: c.value, text: c.label, selected: String(c.value) === String(value) })))
  ]);
}

function textRow(label, value, hint, onChange) {
  return el('label', { class: 'field' }, [
    el('span', { class: 'lbl', text: label }),
    el('input', { type: 'text', value: value || '', placeholder: hint, onchange: (event) => onChange(event.target.value) })
  ]);
}

async function collectDrop(dataTransfer) {
  const out = [];
  if (!dataTransfer) return out;
  if (dataTransfer.items && dataTransfer.items.length && dataTransfer.items[0].webkitGetAsEntry) {
    const entries = [];
    for (const item of dataTransfer.items) {
      const entry = item.webkitGetAsEntry();
      if (entry) entries.push(entry);
    }
    for (const entry of entries) await walkEntry(entry, '', out);
    if (out.length) return out;
  }
  for (const file of dataTransfer.files) out.push(file);
  return out;
}

function walkEntry(entry, path, out) {
  return new Promise((resolve) => {
    if (entry.isFile) {
      entry.file((file) => {
        Object.defineProperty(file, 'webkitRelativePath', { value: path + file.name, configurable: true });
        out.push(file);
        resolve();
      }, () => resolve());
    } else if (entry.isDirectory) {
      const reader = entry.createReader();
      const readBatch = () => {
        reader.readEntries(async (entries) => {
          if (!entries.length) { resolve(); return; }
          for (const child of entries) await walkEntry(child, path + entry.name + '/', out);
          readBatch();
        }, () => resolve());
      };
      readBatch();
    } else resolve();
  });
}

export function openImportDialog(ctx, onDone) {
  const handle = modal('Import a dataset', [importView(ctx, (ds) => { handle.close(); onDone(ds); })], null, { width: '720px' });
  return handle;
}

export { buildSplits, kFold, splitSummary, datasetToNative };
