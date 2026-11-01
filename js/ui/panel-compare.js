import { el, clear, download } from './dom.js';
import { lineChart, SERIES_COLORS } from './charts.js';
import * as store from './store.js';
import { metricMeta, formatMetric, orderScalars, humanize } from '../presets/metrics.js';

const IGNORED_CONFIG_KEYS = ['customClassWeights'];

export function createComparePanel(ctx) {
  const root = el('div', { class: 'panel-wrap' });
  const toolbar = el('div', { class: 'toolbar' });
  const body = el('div');
  root.appendChild(toolbar);
  root.appendChild(body);
  let records = [];
  let loading = false;

  function selectedIds() {
    const ids = ctx.state.compareIds.filter((id) => ctx.state.runs.some((r) => r.id === id));
    if (!ids.length && ctx.state.runs.length) {
      const fallback = ctx.state.runs.slice(0, Math.min(4, ctx.state.runs.length)).map((r) => r.id);
      ctx.state.compareIds = fallback;
      return fallback;
    }
    return ids;
  }

  async function loadRecords() {
    const ids = selectedIds();
    loading = true;
    const out = [];
    for (const id of ids) {
      const rec = await store.loadRun(id);
      if (rec) out.push(rec);
    }
    records = out;
    loading = false;
    return out;
  }

  function renderToolbar() {
    clear(toolbar);
    toolbar.appendChild(el('span', { class: 'small dim', text: records.length + ' run(s) selected' }));
    toolbar.appendChild(el('div', { class: 'spacer' }));
    toolbar.appendChild(el('button', { class: 'sm ghost', text: 'Select all', onclick: () => { ctx.state.compareIds = ctx.state.runs.map((r) => r.id); render(); } }));
    toolbar.appendChild(el('button', { class: 'sm ghost', text: 'Clear', onclick: () => { ctx.state.compareIds = []; render(); } }));
    toolbar.appendChild(el('button', { class: 'sm ghost', text: 'Export JSON', onclick: () => exportJson() }));
    toolbar.appendChild(el('button', { class: 'sm ghost', text: 'Export CSV', onclick: () => exportCsv() }));
    toolbar.appendChild(el('button', { class: 'sm ghost', text: 'Export MD', onclick: () => exportMd() }));
  }

  function pickerList() {
    const host = el('div', { class: 'panel' }, [
      el('div', { class: 'panel-head' }, [el('h3', { text: 'Runs' }), el('div', { class: 'spacer' }), el('span', { class: 'tiny faint', text: ctx.state.runs.length + ' stored' })]),
      el('div', { class: 'panel-body flush' })
    ]);
    const bodyHost = host.querySelector('.panel-body');
    if (!ctx.state.runs.length) {
      bodyHost.appendChild(el('div', { class: 'tiny faint', style: { padding: '10px' }, text: 'Nothing to compare yet — train two runs with different settings.' }));
      return host;
    }
    const list = el('div', { class: 'scroll', style: { maxHeight: '340px' } });
    for (const run of ctx.state.runs) {
      const checked = ctx.state.compareIds.indexOf(run.id) >= 0;
      list.appendChild(el('div', { class: 'run-row' + (checked ? ' on' : '') }, [
        el('input', {
          type: 'checkbox', checked,
          onchange: (event) => {
            if (event.target.checked) ctx.state.compareIds.push(run.id);
            else ctx.state.compareIds = ctx.state.compareIds.filter((id) => id !== run.id);
            render();
          }
        }),
        el('div', { class: 'nm' }, [
          el('div', { class: 'ell', text: run.name || run.id }),
          el('div', { class: 'tiny faint ell', text: (run.modelName || '') + ' · ' + (run.datasetName || '') + ' · seed ' + run.seed + ' · ' + (run.epochsRun || 0) + ' ep' })
        ]),
        el('span', { class: 'mt', text: run.headline && run.headline.value !== null && run.headline.value !== undefined ? formatMetric(run.headline.metric, run.headline.value) : '—' }),
        el('span', { class: 'badge plain', text: (run.totalParams || 0).toLocaleString() + 'p' })
      ]));
    }
    bodyHost.appendChild(list);
    return host;
  }

  function curveOverlay() {
    if (records.length < 1) return el('div', {});
    const host = el('div', { class: 'chart-grid' });
    host.appendChild(overlayBox('Training loss', (rec) => rec.history.map((h) => h.trainLoss), (rec) => rec.history.map((h) => h.epoch), 'lower is better'));
    host.appendChild(overlayBox('Validation loss', (rec) => rec.history.map((h) => h.valLoss), (rec) => rec.history.map((h) => h.epoch), 'lower is better'));
    const key = monitoredKey();
    host.appendChild(overlayBox('Monitored · ' + humanize(key), (rec) => rec.history.map((h) => (h.scalars && h.scalars[key] !== undefined ? h.scalars[key] : h.monitoredValue)), (rec) => rec.history.map((h) => h.epoch), metricMeta(key).direction === 'up' ? 'higher is better' : 'lower is better'));
    host.appendChild(overlayBox('Learning rate', (rec) => rec.history.map((h) => h.lr), (rec) => rec.history.map((h) => h.epoch), 'schedule shape'));
    host.appendChild(overlayBox('Gradient norm', (rec) => rec.history.map((h) => h.gradNorm), (rec) => rec.history.map((h) => h.epoch), 'stability'));
    host.appendChild(overlayBox('Epoch seconds', (rec) => rec.history.map((h) => h.seconds), (rec) => rec.history.map((h) => h.epoch), 'throughput'));
    return host;
  }

  function monitoredKey() {
    if (!records.length) return 'loss';
    const scalars = {};
    for (const rec of records) for (const row of rec.history || []) Object.assign(scalars, row.scalars || {});
    const ordered = orderScalars(records[0].taskKey, scalars);
    return ordered.length ? ordered[0] : 'loss';
  }

  function overlayBox(title, getY, getX, note) {
    const series = records.map((rec, i) => ({
      label: shortName(rec, i),
      y: getY(rec),
      x: getX(rec),
      color: SERIES_COLORS[i % SERIES_COLORS.length],
      width: 1.6
    }));
    const box = el('div', { class: 'chart-box' }, [
      el('div', { class: 'hd' }, [el('h4', { text: title }), el('span', { class: 'val', text: note })]),
      el('canvas', { style: { width: '100%', height: '176px' } })
    ]);
    lineChart(box.querySelector('canvas'), series, { height: 176, xTitle: 'epoch', pad: { top: 16, right: 12, bottom: 24, left: 48 }, points: false });
    return box;
  }

  function shortName(rec, index) {
    const base = (rec.name || rec.id || ('run ' + index)).replace(/\s*·\s*\d\d-\d\d\s*\d\d:\d\d$/, '');
    return base.length > 26 ? base.slice(0, 25) + '…' : base;
  }

  function metricTable() {
    if (!records.length) return el('div', {});
    const taskKey = records[0].taskKey;
    const scalarsUnion = {};
    for (const rec of records) {
      for (const split of ['train', 'val', 'test']) {
        const m = rec.metrics && rec.metrics[split];
        if (m && m.scalars) Object.assign(scalarsUnion, m.scalars);
      }
    }
    const keys = orderScalars(taskKey, scalarsUnion);
    const splits = ['val', 'test'];
    const host = el('div', { class: 'panel' }, [
      el('div', { class: 'panel-head' }, [el('h3', { text: 'Metric comparison' }), el('div', { class: 'spacer' }), el('span', { class: 'tiny faint', text: 'green = best in row, red = worst' })]),
      el('div', { class: 'panel-body flush' })
    ]);
    const bodyHost = host.querySelector('.panel-body');
    const table = el('table', { class: 'data compact cmp-table' });
    const thead = el('thead');
    const headRow = el('tr', {}, [el('th', { text: 'metric' }), el('th', { text: 'split' })]);
    records.forEach((rec, i) => headRow.appendChild(el('th', { class: 'n', style: { color: SERIES_COLORS[i % SERIES_COLORS.length] }, text: shortName(rec, i) })));
    thead.appendChild(headRow);
    table.appendChild(thead);
    const tbody = el('tbody');
    for (const key of keys) {
      for (const split of splits) {
        const values = records.map((rec) => (rec.metrics && rec.metrics[split] && rec.metrics[split].scalars ? rec.metrics[split].scalars[key] : undefined));
        if (values.every((v) => v === undefined || !isFinite(v))) continue;
        const numeric = values.filter((v) => v !== undefined && isFinite(v));
        const meta = metricMeta(key);
        const best = meta.direction === 'down' ? Math.min(...numeric) : Math.max(...numeric);
        const worst = meta.direction === 'down' ? Math.max(...numeric) : Math.min(...numeric);
        const tr = el('tr', {}, [
          el('td', { text: meta.label }),
          el('td', { class: 'tiny faint', text: split })
        ]);
        values.forEach((v) => {
          const isBest = v !== undefined && isFinite(v) && numeric.length > 1 && Math.abs(v - best) < 1e-12;
          const isWorst = v !== undefined && isFinite(v) && numeric.length > 1 && Math.abs(v - worst) < 1e-12;
          tr.appendChild(el('td', { class: 'n' + (isBest ? ' best' : isWorst ? ' worst' : ''), text: v === undefined || !isFinite(v) ? '—' : formatMetric(key, v) }));
        });
        tbody.appendChild(tr);
      }
    }
    const extraRows = [
      ['parameters', (rec) => (rec.totalParams || 0).toLocaleString(), 'flat'],
      ['macs', (rec) => (rec.macs || 0).toLocaleString(), 'flat'],
      ['epochs run', (rec) => String(rec.epochsRun || 0), 'flat'],
      ['wall time (s)', (rec) => ((rec.durationMs || 0) / 1000).toFixed(1), 'down'],
      ['samples seen', (rec) => (rec.samplesSeen || 0).toLocaleString(), 'flat'],
      ['best epoch', (rec) => rec.best ? String(rec.best.epoch) : '—', 'flat'],
      ['seed', (rec) => String(rec.seed), 'flat'],
      ['dataset hash', (rec) => String(rec.datasetHash || '').slice(0, 10), 'flat'],
      ['architecture hash', (rec) => String(rec.architectureHash || '').slice(0, 10), 'flat']
    ];
    for (const row of extraRows) {
      const tr = el('tr', {}, [el('td', { class: 'faint', text: row[0] }), el('td', { class: 'tiny faint', text: 'run' })]);
      records.forEach((rec) => tr.appendChild(el('td', { class: 'n faint', text: row[1](rec) })));
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    bodyHost.appendChild(el('div', { class: 'scroll', style: { maxHeight: '58vh' } }, [table]));
    return host;
  }

  function configDiff() {
    if (records.length < 2) return el('div', { class: 'callout', text: 'Select at least two runs to diff their configuration.' });
    const flat = records.map((rec) => flattenConfig(rec.config || {}));
    const keys = [];
    for (const f of flat) for (const k of Object.keys(f)) if (keys.indexOf(k) < 0) keys.push(k);
    const differing = keys.filter((k) => {
      if (IGNORED_CONFIG_KEYS.indexOf(k) >= 0) return false;
      const first = JSON.stringify(flat[0][k]);
      return flat.some((f) => JSON.stringify(f[k]) !== first);
    });
    const host = el('div', { class: 'panel' }, [
      el('div', { class: 'panel-head' }, [el('h3', { text: 'Configuration diff' }), el('div', { class: 'spacer' }), el('span', { class: 'tiny faint', text: differing.length + ' of ' + keys.length + ' settings differ' })]),
      el('div', { class: 'panel-body flush' })
    ]);
    const bodyHost = host.querySelector('.panel-body');
    if (!differing.length) {
      bodyHost.appendChild(el('div', { class: 'callout good', style: { margin: '10px' }, text: 'Every setting is identical across these runs. Any metric difference comes from the seed or from non-determinism, not from the configuration.' }));
      return host;
    }
    const table = el('table', { class: 'data compact cmp-table' });
    const thead = el('tr', {}, [el('th', { text: 'setting' })]);
    records.forEach((rec, i) => thead.appendChild(el('th', { class: 'n', style: { color: SERIES_COLORS[i % SERIES_COLORS.length] }, text: shortName(rec, i) })));
    table.appendChild(el('thead', {}, thead));
    const tbody = el('tbody');
    for (const key of differing) {
      const tr = el('tr', {}, [el('td', { class: 'mono tiny', text: key })]);
      records.forEach((rec, i) => {
        const value = flat[i][key];
        tr.appendChild(el('td', { class: 'n', style: { color: value === undefined ? 'var(--faint)' : '' }, text: value === undefined ? '—' : String(value) }));
      });
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    bodyHost.appendChild(el('div', { class: 'scroll', style: { maxHeight: '46vh' } }, [table]));
    return host;
  }

  function flattenConfig(cfg, prefix) {
    const out = {};
    for (const key of Object.keys(cfg || {})) {
      const value = cfg[key];
      const path = prefix ? prefix + '.' + key : key;
      if (value && typeof value === 'object' && !Array.isArray(value)) Object.assign(out, flattenConfig(value, path));
      else if (Array.isArray(value)) out[path] = value.length > 6 ? '[' + value.length + ' values]' : JSON.stringify(value);
      else out[path] = value;
    }
    return out;
  }

  function reproducibilityView() {
    if (!records.length) return el('div', {});
    const host = el('div', { class: 'col gap6' });
    const seeds = unique(records.map((r) => String(r.seed)));
    const data = unique(records.map((r) => String(r.datasetHash)));
    const arch = unique(records.map((r) => String(r.architectureHash)));
    const rows = [
      { label: 'seed', values: seeds, note: seeds.length === 1 ? 'identical — differences are configuration-driven, not luck' : 'different — part of the gap may be seed variance' },
      { label: 'dataset content hash', values: data, note: data.length === 1 ? 'same data behind every run' : 'DIFFERENT data — the comparison is not apples to apples' },
      { label: 'architecture hash', values: arch, note: arch.length === 1 ? 'same graph' : 'different graphs' }
    ];
    for (const row of rows) {
      host.appendChild(el('div', { class: 'callout ' + (row.values.length === 1 ? 'good' : row.label === 'dataset content hash' ? 'bad' : ''), text: row.label + ': ' + row.values.length + ' distinct value(s) — ' + row.note }));
    }
    host.appendChild(el('div', { class: 'tiny faint', text: 'Every stored run carries its seed, its full training configuration, the dataset content hash and the architecture hash. Re-running with the same three reproduces the same numbers on the same machine.' }));
    return host;
  }

  function unique(values) {
    const out = [];
    for (const v of values) if (out.indexOf(v) < 0) out.push(v);
    return out;
  }

  function exportJson() {
    if (!records.length) { ctx.warn('Nothing selected.'); return; }
    download('run-comparison.json', JSON.stringify({
      generatedAt: new Date().toISOString(),
      runs: records.map((rec) => ({
        id: rec.id, name: rec.name, taskKey: rec.taskKey, seed: rec.seed,
        modelName: rec.modelName, datasetName: rec.datasetName,
        datasetHash: rec.datasetHash, architectureHash: rec.architectureHash,
        totalParams: rec.totalParams, macs: rec.macs,
        epochsRun: rec.epochsRun, durationMs: rec.durationMs,
        headline: rec.headline, best: rec.best,
        config: rec.config, metrics: rec.metrics, history: rec.history
      }))
    }, null, 2), 'application/json');
    ctx.ok('Comparison exported as JSON (' + records.length + ' runs).');
  }

  function exportCsv() {
    if (!records.length) { ctx.warn('Nothing selected.'); return; }
    const scalarsUnion = {};
    for (const rec of records) for (const split of ['train', 'val', 'test']) {
      const m = rec.metrics && rec.metrics[split];
      if (m && m.scalars) Object.assign(scalarsUnion, m.scalars);
    }
    const keys = orderScalars(records[0].taskKey, scalarsUnion);
    const columns = ['run', 'split', 'metric', 'value'];
    const lines = [columns.join(',')];
    for (const rec of records) {
      for (const split of ['train', 'val', 'test']) {
        const m = rec.metrics && rec.metrics[split];
        if (!m) continue;
        for (const key of keys) {
          const v = m.scalars ? m.scalars[key] : undefined;
          if (v === undefined) continue;
          lines.push([csvCell(rec.name), split, key, v].join(','));
        }
      }
    }
    download('run-comparison.csv', lines.join('\n'), 'text/csv');
    ctx.ok('Comparison exported as long-form CSV.');
  }

  function csvCell(value) {
    const text = String(value === undefined || value === null ? '' : value);
    return /[",\n]/.test(text) ? '"' + text.replace(/"/g, '""') + '"' : text;
  }

  function exportMd() {
    if (!records.length) { ctx.warn('Nothing selected.'); return; }
    const lines = [];
    lines.push('# Run comparison');
    lines.push('');
    lines.push('| | ' + records.map((r, i) => shortName(r, i)).join(' | ') + ' |');
    lines.push('|' + '---|'.repeat(records.length + 1));
    const meta = [['seed', (r) => r.seed], ['params', (r) => (r.totalParams || 0).toLocaleString()], ['epochs', (r) => r.epochsRun], ['seconds', (r) => ((r.durationMs || 0) / 1000).toFixed(1)], ['dataset hash', (r) => String(r.datasetHash || '').slice(0, 12)], ['arch hash', (r) => String(r.architectureHash || '').slice(0, 12)]];
    for (const row of meta) lines.push('| ' + row[0] + ' | ' + records.map((r) => row[1](r)).join(' | ') + ' |');
    const scalarsUnion = {};
    for (const rec of records) for (const split of ['val', 'test']) {
      const m = rec.metrics && rec.metrics[split];
      if (m && m.scalars) Object.assign(scalarsUnion, m.scalars);
    }
    for (const split of ['val', 'test']) {
      lines.push('');
      lines.push('## ' + split + ' split');
      lines.push('');
      lines.push('| metric | ' + records.map((r, i) => shortName(r, i)).join(' | ') + ' |');
      lines.push('|' + '---|'.repeat(records.length + 1));
      for (const key of orderScalars(records[0].taskKey, scalarsUnion)) {
        const values = records.map((r) => (r.metrics && r.metrics[split] && r.metrics[split].scalars ? r.metrics[split].scalars[key] : undefined));
        if (values.every((v) => v === undefined)) continue;
        lines.push('| ' + metricMeta(key).label + ' | ' + values.map((v) => (v === undefined || !isFinite(v) ? '—' : formatMetric(key, v))).join(' | ') + ' |');
      }
    }
    lines.push('');
    lines.push('_Generated by Rootism Labs._');
    download('run-comparison.md', lines.join('\n'), 'text/markdown');
    ctx.ok('Comparison exported as Markdown.');
  }

  async function renderBody() {
    clear(body);
    await loadRecords();
    const host = el('div', { class: 'col gap8' });
    host.appendChild(pickerList());
    if (!records.length) {
      host.appendChild(el('div', { class: 'callout', text: 'Pick two or more runs above. Comparison overlays their curves, diffs their configuration and checks whether the seed, dataset hash and architecture hash line up.' }));
      body.appendChild(host);
      return;
    }
    host.appendChild(reproducibilityView());
    host.appendChild(curveOverlay());
    host.appendChild(metricTable());
    host.appendChild(configDiff());
    body.appendChild(host);
  }

  function render() {
    renderToolbar();
    renderBody().catch((e) => {
      clear(body);
      body.appendChild(el('div', { class: 'callout bad', text: 'Comparison failed: ' + e.message }));
    });
  }

  function refresh(reason) {
    if (reason === 'progress' || reason === 'log') return;
    render();
  }

  render();
  return { root, refresh, render };
}
