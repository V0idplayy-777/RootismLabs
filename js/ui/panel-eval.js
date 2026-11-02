import { el, clear, download } from './dom.js';
import { lineChart, barChart, confusionChart, calibrationChart, curveChart, histogram, imageToCanvas, heatmap, waveform, SERIES_COLORS } from './charts.js';
import { Model } from '../core/model.js';
import { taskFor } from '../core/tasks.js';
import { makeBatchNode } from '../core/tasks.js';
import { shapeSize } from '../core/tensor.js';
import { metricsForTask, metricMeta, formatMetric, orderScalars, humanize } from '../presets/metrics.js';
import * as store from './store.js';
import { sampleText, samplePreviewCell } from './data-tools.js';

const SPLITS = ['train', 'val', 'test'];

export function createEvalPanel(ctx) {
  const root = el('div', { class: 'panel-wrap' });
  const toolbar = el('div', { class: 'toolbar' });
  const body = el('div');
  root.appendChild(toolbar);
  root.appendChild(body);

  let split = 'test';
  let threshold = null;
  let cache = {};
  let record = null;
  let busy = false;

  function selectedId() {
    return ctx.state.selectedRunId || (ctx.state.runs.length ? ctx.state.runs[0].id : null);
  }

  async function loadRecord() {
    const id = selectedId();
    if (!id) { record = null; return null; }
    if (record && record.id === id) return record;
    record = await store.loadRun(id);
    cache = {};
    return record;
  }

  function renderToolbar() {
    clear(toolbar);
    const runs = ctx.state.runs;
    if (!runs.length) {
      toolbar.appendChild(el('span', { class: 'small dim', text: 'No runs yet. Train a model first — evaluation reads the stored weights, splits and metrics.' }));
      toolbar.appendChild(el('div', { class: 'spacer' }));
      toolbar.appendChild(el('button', { class: 'sm', text: 'Go to Train', onclick: () => ctx.go('train') }));
      return;
    }
    toolbar.appendChild(el('select', {
      style: { maxWidth: '300px' },
      onchange: async (event) => { ctx.state.selectedRunId = event.target.value; record = null; cache = {}; await renderBody(); }
    }, runs.map((r) => el('option', { value: r.id, text: (r.name || r.id) + ' · ' + (r.headline && r.headline.value !== null ? r.headline.metric + ' ' + Number(r.headline.value).toFixed(4) : r.epochsRun + ' ep'), selected: r.id === selectedId() }))));
    toolbar.appendChild(el('div', { class: 'vsep' }));
    toolbar.appendChild(el('label', { class: 'inline' }, [
      el('span', { class: 'tiny faint', text: 'split' }),
      el('select', { onchange: async (event) => { split = event.target.value; await renderBody(); } }, SPLITS.map((s) => el('option', { value: s, text: s, selected: s === split })))
    ]));
    toolbar.appendChild(el('button', { class: 'sm teal', text: busy ? 'Working…' : 'Re-evaluate', disabled: busy, onclick: () => reevaluate() }));
    toolbar.appendChild(el('div', { class: 'spacer' }));
    toolbar.appendChild(el('button', { class: 'sm ghost', text: 'Report JSON', onclick: () => downloadReport('json') }));
    toolbar.appendChild(el('button', { class: 'sm ghost', text: 'Report MD', onclick: () => downloadReport('md') }));
    toolbar.appendChild(el('button', { class: 'sm ghost', text: 'Add to compare', onclick: () => { const id = selectedId(); if (id) { ctx.state.compareIds.push(id); ctx.toast('Added to the comparison set.'); ctx.notify('compare'); } } }));
    toolbar.appendChild(el('button', { class: 'sm primary', text: 'Export model', onclick: () => ctx.go('export') }));
  }

  function datasetFor(record2) {
    const ds = ctx.state.dataset;
    if (!ds) return { ds: null, warning: 'Load the dataset this run was trained on to re-evaluate or inspect samples.' };
    if (record2 && record2.datasetHash && ds.digest && ds.digest.contentHash !== record2.datasetHash) {
      return { ds, warning: 'The loaded dataset does not match the one this run was trained on (hash ' + (ds.digest.short || '') + ' vs ' + String(record2.datasetHash).slice(0, 12) + '…). Stored metrics are still valid; re-evaluation would be meaningless.' };
    }
    return { ds, warning: null };
  }

  async function reevaluate() {
    const rec = await loadRecord();
    if (!rec) { ctx.warn('No run selected.'); return; }
    const info = datasetFor(rec);
    if (!info.ds) { ctx.error(info.warning); return; }
    if (info.warning) { ctx.warn(info.warning); }
    if (!rec.weights || !rec.weights.length) { ctx.error('This run has no stored weights, so it cannot be re-evaluated.'); return; }
    busy = true;
    renderToolbar();
    await new Promise((r) => setTimeout(r, 20));
    try {
      const idx = info.ds.splits ? info.ds.splits[split] : null;
      if (!idx || !idx.length) { ctx.error('The ' + split + ' split is empty.'); return; }
      const model = new Model(rec.spec, rec.seed || rec.config.seed);
      model.loadWeights(rec.weights);
      if (model.syncStates) model.syncStates();
      const task = taskFor(rec.taskKey);
      const taskCfg = Object.assign({
        loss: rec.config.loss,
        classWeights: null,
        labelSmoothing: 0,
        posWeight: 1,
        focalGamma: 2,
        temperature: rec.config.temperature || 0.07,
        huberDelta: 1,
        confThreshold: rec.config.confThreshold ?? 0.3,
        decisionThreshold: rec.config.decisionThreshold ?? 0.5,
        peak: rec.config.peak || 1,
        baseline: rec.config.baseline || 'input',
        frameRate: rec.config.frameRate || 100
      }, threshold !== null ? { confThreshold: threshold, decisionThreshold: threshold } : {});
      const metrics = task.metrics(model, info.ds, idx, taskCfg);
      let loss = NaN;
      try {
        const l = task.lossFn(model, info.ds, idx, taskCfg, 'eval', null);
        if (l && isFinite(l.d[0])) loss = l.d[0];
      } catch (e) { loss = NaN; }
      metrics.loss = loss;
      cache[split + (threshold === null ? '' : '@' + threshold)] = { metrics, model, idx };
      ctx.ok('Re-evaluated "' + rec.name + '" on the ' + split + ' split (' + idx.length + ' samples)' + (threshold !== null ? ' at threshold ' + threshold : '') + '.');
    } catch (e) {
      ctx.error('Re-evaluation failed: ' + e.message);
    } finally {
      busy = false;
      renderToolbar();
      await renderBody();
    }
  }

  function metricsFor(rec) {
    const key = split + (threshold === null ? '' : '@' + threshold);
    if (cache[key]) return cache[key].metrics;
    const stored = rec.metrics ? rec.metrics[split] : null;
    return stored || (rec.metrics ? rec.metrics.val || rec.metrics.train : null);
  }

  async function renderBody() {
    clear(body);
    try {
    const rec = await loadRecord();
    if (!rec) {
      body.appendChild(el('div', { class: 'callout', text: 'Select a run to evaluate.' }));
      return;
    }
    const metrics = metricsFor(rec);
    if (!metrics) {
      body.appendChild(el('div', { class: 'callout bad', text: 'This run has no metrics for the "' + split + '" split.' }));
      return;
    }
    const info = datasetFor(rec);
    const host = el('div', { class: 'col gap8' });
    host.appendChild(runHeader(rec, metrics));
    if (info.warning) host.appendChild(el('div', { class: 'callout warn', text: info.warning }));
    host.appendChild(generalisationView(rec));
    host.appendChild(scalarView(rec, metrics));
    if (metrics.confusion) host.appendChild(confusionView(metrics));
    if (metrics.perClass) host.appendChild(perClassView(metrics));
    if (metrics.calibration) host.appendChild(calibrationView(metrics));
    if (metrics.roc || metrics.pr) host.appendChild(curveView(metrics));
    if (metrics.sweep) host.appendChild(sweepView(rec, metrics));
    if (metrics.histograms) host.appendChild(anomalyView(metrics));
    if (metrics.errors) host.appendChild(worstView(rec, metrics, info.ds));
    host.appendChild(inspectionView(rec, info.ds, metrics));
    body.appendChild(host);
    } catch (e) {
      body.appendChild(el('div', { class: 'callout bad', text: 'Evaluation view failed: ' + e.message }));
    }
  }

  function runHeader(rec, metrics) {
    const task = taskFor(rec.taskKey);
    const scalars = metrics.scalars || {};
    const groups = metricsForTask(rec.taskKey);
    const headline = groups.length && groups[0].metrics.length ? groups[0].metrics[0] : Object.keys(scalars)[0];
    const cards = [];
    const keys = [];
    for (const group of groups.slice(0, 2)) for (const k of group.metrics) if (keys.indexOf(k) < 0 && scalars[k] !== undefined) keys.push(k);
    for (const key of keys.slice(0, 8)) {
      const meta = metricMeta(key);
      cards.push(el('div', { class: 'stat' }, [
        el('div', { class: 'k', text: meta.label + (key === headline ? ' · headline' : '') }),
        el('div', { class: 'v' + (key === headline ? ' amber' : ''), text: formatMetric(key, scalars[key]) }),
        el('div', { class: 's', text: meta.direction === 'up' ? 'higher is better' : meta.direction === 'down' ? 'lower is better' : 'reference' })
      ]));
    }
    return el('div', { class: 'col gap8' }, [
      el('div', { class: 'row between wrap' }, [
        el('div', {}, [
          el('h2', { text: rec.name || rec.id }),
          el('div', { class: 'small dim', text: task.label + ' · ' + rec.modelName + ' · ' + rec.datasetName + ' · seed ' + rec.seed + ' · ' + rec.epochsRun + ' epochs in ' + (rec.durationMs / 1000).toFixed(1) + 's' })
        ]),
        el('div', { class: 'mono tiny faint right' }, [
          el('div', { text: 'data ' + String(rec.datasetHash || '').slice(0, 16) }),
          el('div', { text: 'arch ' + String(rec.architectureHash || '').slice(0, 16) }),
          el('div', { text: (rec.totalParams || 0).toLocaleString() + ' params' })
        ])
      ]),
      el('div', { class: 'kpi-row' }, cards),
      el('div', { class: 'tiny faint', text: 'Showing the ' + split + ' split' + (threshold !== null ? ' at decision threshold ' + threshold : '') + '. Re-evaluate recomputes everything from the stored weights on the currently loaded dataset.' })
    ]);
  }

  function generalisationView(rec) {
    const keys = [];
    for (const s of SPLITS) {
      const m = rec.metrics ? rec.metrics[s] : null;
      if (!m) continue;
      for (const k of Object.keys(m.scalars || {})) if (keys.indexOf(k) < 0) keys.push(k);
    }
    if (keys.length === 0) return el('div', {});
    const ordered = orderScalars(rec.taskKey, Object.fromEntries(keys.map((k) => [k, 0]))).slice(0, 12);
    const table = el('table', { class: 'data compact cmp-table' }, [
      el('thead', {}, el('tr', {}, [el('th', { text: 'metric' })].concat(SPLITS.filter((s) => rec.metrics && rec.metrics[s]).map((s) => el('th', { class: 'n', text: s })))))
    ]);
    const tbody = el('tbody');
    for (const key of ordered) {
      const meta = metricMeta(key);
      const values = SPLITS.filter((s) => rec.metrics && rec.metrics[s]).map((s) => (rec.metrics[s].scalars || {})[key]);
      const tr = el('tr', {}, [el('td', { text: meta.label })]);
      for (const v of values) {
        tr.appendChild(el('td', { class: 'n', text: v === undefined || v === null || !isFinite(v) ? '—' : formatMetric(key, v) }));
      }
      tbody.appendChild(tr);
    }
    const lossRow = el('tr', {}, [el('td', { text: 'loss' })]);
    for (const s of SPLITS.filter((x) => rec.metrics && rec.metrics[x])) {
      const v = rec.metrics[s].loss;
      lossRow.appendChild(el('td', { class: 'n', text: isFinite(v) ? v.toFixed(4) : '—' }));
    }
    tbody.appendChild(lossRow);
    table.appendChild(tbody);

    const trainAcc = rec.metrics && rec.metrics.train && rec.metrics.train.scalars ? rec.metrics.train.scalars : null;
    const testAcc = rec.metrics && rec.metrics.test && rec.metrics.test.scalars ? rec.metrics.test.scalars : null;
    const gapKeys = ordered.filter((k) => metricMeta(k).direction !== 'flat');
    const gaps = [];
    if (trainAcc && testAcc) {
      for (const k of gapKeys) {
        if (trainAcc[k] === undefined || testAcc[k] === undefined) continue;
        const dir = metricMeta(k).direction;
        const gap = dir === 'up' ? trainAcc[k] - testAcc[k] : testAcc[k] - trainAcc[k];
        gaps.push({ key: k, gap });
      }
    }
    const worstGap = gaps.sort((a, b) => b.gap - a.gap)[0];
    const note = worstGap
      ? (worstGap.gap > 0.15 ? 'Large generalisation gap on ' + humanize(worstGap.key) + ' (train beats test by ' + worstGap.gap.toFixed(3) + '). More data, more regularisation, or a smaller model.'
        : worstGap.gap > 0.05 ? 'Moderate gap on ' + humanize(worstGap.key) + ' (' + worstGap.gap.toFixed(3) + '). Acceptable for a small model, worth watching.'
          : 'Train and test agree closely (largest gap ' + worstGap.gap.toFixed(3) + ' on ' + humanize(worstGap.key) + ').')
      : 'Train and test metrics are not both available, so the generalisation gap cannot be computed.';
    return el('div', { class: 'panel' }, [
      el('div', { class: 'panel-head' }, [el('h3', { text: 'Split comparison' }), el('div', { class: 'spacer' }), el('span', { class: 'tiny faint', text: 'train vs val vs test on the same weights' })]),
      el('div', { class: 'panel-body tight' }, [
        el('div', { class: 'scroll', style: { maxHeight: '280px' } }, [table]),
        el('div', { class: 'callout mt8 ' + (worstGap && worstGap.gap > 0.15 ? 'bad' : worstGap && worstGap.gap > 0.05 ? 'warn' : 'good'), text: note })
      ])
    ]);
  }

  function scalarView(rec, metrics) {
    const scalars = metrics.scalars || {};
    const groups = metricsForTask(rec.taskKey);
    const host = el('div', { class: 'col gap8' });
    const shown = [];
    for (const group of groups) {
      const keys = group.metrics.filter((k) => scalars[k] !== undefined);
      if (!keys.length) continue;
      shown.push.apply(shown, keys);
      const table = el('table', { class: 'data compact' }, [
        el('thead', {}, el('tr', {}, [el('th', { text: 'metric' }), el('th', { class: 'n', text: 'value' }), el('th', { text: 'direction' }), el('th', { text: 'what it means' })]))
      ]);
      const tbody = el('tbody');
      for (const key of keys) {
        const meta = metricMeta(key);
        tbody.appendChild(el('tr', {}, [
          el('td', {}, [el('span', { class: 'mono', text: key }), el('span', { class: 'faint small', text: ' — ' + meta.label })]),
          el('td', { class: 'n amber', text: formatMetric(key, scalars[key]) }),
          el('td', { class: 'tiny faint', text: meta.direction === 'up' ? 'higher better' : meta.direction === 'down' ? 'lower better' : 'reference' }),
          el('td', { class: 'tiny faint', text: meta.explain || '' })
        ]));
      }
      table.appendChild(tbody);
      host.appendChild(el('div', { class: 'panel' }, [
        el('div', { class: 'panel-head' }, [el('h3', { text: group.title })]),
        el('div', { class: 'panel-body flush' }, [table])
      ]));
    }
    const rest = Object.keys(scalars).filter((k) => shown.indexOf(k) < 0);
    if (rest.length) {
      host.appendChild(el('div', { class: 'panel' }, [
        el('div', { class: 'panel-head' }, [el('h3', { text: 'Other measurements' })]),
        el('div', { class: 'panel-body tight' }, [
          el('div', { class: 'kv' }, rest.map((k) => [el('dt', { text: humanize(k) }), el('dd', { text: formatMetric(k, scalars[k]) })]).flat())
        ])
      ]));
    }
    return host;
  }

  function confusionView(metrics) {
    const n = metrics.confusion.labels.length;
    const height = Math.max(180, Math.min(460, n * 24 + 46));
    const canvas = el('canvas', { style: { width: '100%', height: height + 'px' } });
    confusionChart(canvas, metrics.confusion.matrix, metrics.confusion.labels, { height });
    const totals = metrics.confusion.matrix.map((row) => row.reduce((a, b) => a + b, 0));
    let offDiagonal = 0;
    let total = 0;
    metrics.confusion.matrix.forEach((row, i) => row.forEach((v, j) => { total += v; if (i !== j) offDiagonal += v; }));
    return el('div', { class: 'panel' }, [
      el('div', { class: 'panel-head' }, [el('h3', { text: 'Confusion matrix' }), el('div', { class: 'spacer' }), el('span', { class: 'tiny faint', text: n + ' classes · ' + total + ' samples · ' + (100 * offDiagonal / Math.max(1, total)).toFixed(1) + '% misclassified' })]),
      el('div', { class: 'panel-body tight' }, [
        split === 'test' ? el('div', { class: 'callout warn', text: 'Do not choose a threshold from the test split. Use validation data for threshold selection, then evaluate once on an untouched test set.' }) : null,
        canvas,
        el('div', { class: 'tiny faint mt4', text: 'Rows are the true class, columns the prediction. Cells are normalised by row so an unbalanced dataset still shows where each class goes.' }),
        el('div', { class: 'row gap4 mt8' }, [
          el('button', { class: 'xs ghost', text: 'Download matrix CSV', onclick: () => download('confusion-' + split + '.csv', [metrics.confusion.labels.join(',')].concat(metrics.confusion.matrix.map((r) => r.join(','))).join('\n'), 'text/csv') })
        ])
      ])
    ]);
  }

  function perClassView(metrics) {
    const rows = metrics.perClass;
    const table = el('table', { class: 'data compact' }, [
      el('thead', {}, el('tr', {}, [el('th', { text: 'class' }), el('th', { class: 'n', text: 'support' }), el('th', { class: 'n', text: 'precision' }), el('th', { class: 'n', text: 'recall' }), el('th', { class: 'n', text: 'f1' }), el('th', { class: 'n', text: 'tp' }), el('th', { class: 'n', text: 'fp' }), el('th', { class: 'n', text: 'fn' }), el('th', { text: 'read' })])),
      el('tbody')
    ]);
    const tbody = table.querySelector('tbody');
    for (const row of rows) {
      const problem = row.support === 0 ? 'no samples in this split' : row.recall < 0.3 ? 'misses most of its own samples' : row.precision < 0.3 ? 'attracts other classes' : row.f1 < 0.4 ? 'weak overall' : '';
      tbody.appendChild(el('tr', {}, [
        el('td', { text: String(row.label) }),
        el('td', { class: 'n', text: String(row.support) }),
        el('td', { class: 'n', text: row.precision.toFixed(3) }),
        el('td', { class: 'n', text: row.recall.toFixed(3) }),
        el('td', { class: 'n', style: { color: row.f1 > 0.7 ? 'var(--green)' : row.f1 < 0.35 ? '#d09a93' : '' }, text: row.f1.toFixed(3) }),
        el('td', { class: 'n faint', text: String(row.tp) }),
        el('td', { class: 'n faint', text: String(row.fp) }),
        el('td', { class: 'n faint', text: String(row.fn) }),
        el('td', { class: 'tiny ' + (problem ? 'amber' : 'faint'), text: problem })
      ]));
    }
    return el('div', { class: 'panel' }, [
      el('div', { class: 'panel-head' }, [el('h3', { text: 'Per-class report' })]),
      el('div', { class: 'panel-body flush' }, [el('div', { class: 'scroll', style: { maxHeight: '320px' } }, [table])])
    ]);
  }

  function calibrationView(metrics) {
    const cal = metrics.calibration;
    const canvas = el('canvas', { style: { width: '100%', height: '190px' } });
    calibrationChart(canvas, cal, { height: 190 });
    const confCanvas = el('canvas', { style: { width: '100%', height: '96px' } });
    histogram(confCanvas, cal.bins.map((b) => b.confidence).filter((v) => isFinite(v)), { bins: 20, height: 96, color: '#5b87b5', background: '#191c1f' });
    const verdict = cal.ece < 0.03 ? 'Well calibrated.' : cal.ece < 0.08 ? 'Slightly over- or under-confident; usable but quote the ECE.' : 'Poorly calibrated. Temperature scaling or more data before trusting the probabilities.';
    return el('div', { class: 'panel' }, [
      el('div', { class: 'panel-head' }, [el('h3', { text: 'Calibration' }), el('div', { class: 'spacer' }), el('span', { class: 'mono tiny', text: 'ECE ' + cal.ece.toFixed(4) + ' · MCE ' + cal.mce.toFixed(4) + ' · Brier ' + cal.brier.toFixed(4) + ' · log loss ' + cal.logLoss.toFixed(4) })]),
      el('div', { class: 'panel-body tight' }, [
        split === 'test' ? el('div', { class: 'callout warn', text: 'Do not choose a threshold from the test split. Use validation data for threshold selection, then evaluate once on an untouched test set.' }) : null,
        canvas,
        el('div', { class: 'callout mt8 ' + (cal.ece < 0.08 ? 'good' : 'warn'), text: verdict }),
        el('div', { class: 'scroll', style: { maxHeight: '180px' } }, [
          el('table', { class: 'data compact mt8' }, [
            el('thead', {}, el('tr', {}, [el('th', { text: 'bin' }), el('th', { class: 'n', text: 'count' }), el('th', { class: 'n', text: 'confidence' }), el('th', { class: 'n', text: 'accuracy' }), el('th', { class: 'n', text: 'gap' })])),
            el('tbody', {}, cal.bins.map((b) => el('tr', {}, [
              el('td', { class: 'mono tiny', text: b.lo.toFixed(1) + '–' + b.hi.toFixed(1) }),
              el('td', { class: 'n', text: String(b.count) }),
              el('td', { class: 'n', text: b.confidence.toFixed(3) }),
              el('td', { class: 'n', text: b.accuracy.toFixed(3) }),
              el('td', { class: 'n', style: { color: Math.abs(b.accuracy - b.confidence) > 0.15 ? '#d09a93' : '' }, text: (b.accuracy - b.confidence).toFixed(3) })
            ])))
          ])
        ])
      ])
    ]);
  }

  function curveView(metrics) {
    const host = el('div', { class: 'chart-grid' });
    if (metrics.roc && metrics.roc.points) {
      const canvas = el('canvas', { style: { width: '100%', height: '200px' } });
      curveChart(canvas, metrics.roc.points, { height: 200, xTitle: 'false positive rate', yTitle: 'true positive rate', annotation: 'AUC ' + metrics.roc.auc.toFixed(4), fill: '#4fa3a5' });
      host.appendChild(el('div', { class: 'chart-box' }, [
        el('div', { class: 'hd' }, [el('h4', { text: 'ROC' }), el('span', { class: 'val', text: 'AUC ' + metrics.roc.auc.toFixed(4) })]),
        canvas,
        el('div', { class: 'tiny faint', text: (metrics.roc.positives || 0) + ' positives, ' + (metrics.roc.negatives || 0) + ' negatives' })
      ]));
    }
    if (metrics.pr && metrics.pr.points) {
      const canvas = el('canvas', { style: { width: '100%', height: '200px' } });
      curveChart(canvas, metrics.pr.points, { height: 200, xTitle: 'recall', yTitle: 'precision', annotation: 'AP ' + (metrics.pr.ap === undefined ? metrics.pr.averagePrecision : metrics.pr.ap).toFixed(4), color: '#e0a33e', fill: '#e0a33e' });
      host.appendChild(el('div', { class: 'chart-box' }, [
        el('div', { class: 'hd' }, [el('h4', { text: 'Precision-recall' }), el('span', { class: 'val', text: 'AP ' + (metrics.pr.ap === undefined ? metrics.pr.averagePrecision : metrics.pr.ap).toFixed(4) })]),
        canvas,
        el('div', { class: 'tiny faint', text: 'The curve that matters when the positive class is rare.' })
      ]));
    }
    return host;
  }

  function sweepView(rec, metrics) {
    const sweep = metrics.sweep;
    const isDetection = sweep.length && sweep[0].tp !== undefined;
    const table = el('table', { class: 'data compact' }, [
      el('thead', {}, el('tr', {}, isDetection
        ? [el('th', { class: 'n', text: 'conf' }), el('th', { class: 'n', text: 'precision' }), el('th', { class: 'n', text: 'recall' }), el('th', { class: 'n', text: 'f1' }), el('th', { class: 'n', text: 'tp' }), el('th', { class: 'n', text: 'fp' }), el('th', { class: 'n', text: 'fn' })]
        : [el('th', { class: 'n', text: 'threshold' }), el('th', { class: 'n', text: 'FAR' }), el('th', { class: 'n', text: 'FRR' }), el('th', { class: 'n', text: 'precision' }), el('th', { class: 'n', text: 'recall' }), el('th', { class: 'n', text: 'f1' }), el('th', { class: 'n', text: 'accuracy' })])),
      el('tbody')
    ]);
    const tbody = table.querySelector('tbody');
    let bestIndex = 0;
    let bestF1 = -1;
    sweep.forEach((row, i) => { if (row.f1 > bestF1) { bestF1 = row.f1; bestIndex = i; } });
    sweep.forEach((row, i) => {
      const tr = el('tr', { class: i === bestIndex ? 'selected' : '' }, isDetection
        ? [el('td', { class: 'n', text: row.threshold.toFixed(2) }), el('td', { class: 'n', text: row.precision.toFixed(3) }), el('td', { class: 'n', text: row.recall.toFixed(3) }), el('td', { class: 'n', text: row.f1.toFixed(3) }), el('td', { class: 'n faint', text: String(row.tp) }), el('td', { class: 'n faint', text: String(row.fp) }), el('td', { class: 'n faint', text: String(row.fn) })]
        : [el('td', { class: 'n', text: row.threshold.toFixed(2) }), el('td', { class: 'n', style: { color: row.far > 0.05 ? '#d09a93' : '' }, text: row.far.toFixed(3) }), el('td', { class: 'n', style: { color: row.frr > 0.2 ? '#d09a93' : '' }, text: row.frr.toFixed(3) }), el('td', { class: 'n', text: row.precision.toFixed(3) }), el('td', { class: 'n', text: row.recall.toFixed(3) }), el('td', { class: 'n', text: row.f1.toFixed(3) }), el('td', { class: 'n faint', text: row.accuracy.toFixed(3) })]);
      tbody.appendChild(tr);
    });
    const canvas = el('canvas', { style: { width: '100%', height: '150px' } });
    lineChart(canvas, isDetection
      ? [
        { label: 'precision', y: sweep.map((r) => r.precision), x: sweep.map((r) => r.threshold), color: SERIES_COLORS[1] },
        { label: 'recall', y: sweep.map((r) => r.recall), x: sweep.map((r) => r.threshold), color: SERIES_COLORS[0] },
        { label: 'f1', y: sweep.map((r) => r.f1), x: sweep.map((r) => r.threshold), color: SERIES_COLORS[2], dash: [3, 3] }
      ]
      : [
        { label: 'FAR', y: sweep.map((r) => r.far), x: sweep.map((r) => r.threshold), color: SERIES_COLORS[3] },
        { label: 'FRR', y: sweep.map((r) => r.frr), x: sweep.map((r) => r.threshold), color: SERIES_COLORS[0] },
        { label: 'f1', y: sweep.map((r) => r.f1), x: sweep.map((r) => r.threshold), color: SERIES_COLORS[1], dash: [3, 3] }
      ], { height: 150, xTitle: isDetection ? 'confidence threshold' : 'decision threshold', pad: { top: 14, right: 12, bottom: 24, left: 42 } });
    const controls = isDetection ? null : el('div', { class: 'row gap6 mt8' }, [
      el('span', { class: 'tiny faint', text: 'decision threshold' }),
      el('input', {
        type: 'range', min: 0.01, max: 0.99, step: 0.01, value: threshold === null ? (metrics.scalars.best_threshold || 0.5) : threshold, style: { width: '180px' },
        oninput: (event) => { const out = root.querySelector('#thrOut'); if (out) out.textContent = parseFloat(event.target.value).toFixed(2); }
      }),
      el('span', { class: 'mono tiny amber', id: 'thrOut', text: (threshold === null ? (metrics.scalars.best_threshold || 0.5) : threshold).toFixed(2) }),
      el('button', { class: 'xs', text: 'Re-evaluate at this threshold', onclick: (event) => { const input = event.target.parentNode.querySelector('input'); threshold = parseFloat(input.value); reevaluate(); } }),
      threshold !== null ? el('button', { class: 'xs ghost', text: 'Clear', onclick: () => { threshold = null; renderBody(); } }) : null
    ]);
    return el('div', { class: 'panel' }, [
      el('div', { class: 'panel-head' }, [el('h3', { text: isDetection ? 'Confidence sweep' : 'Threshold sweep' }), el('div', { class: 'spacer' }), el('span', { class: 'tiny faint', text: 'best F1 at ' + sweep[bestIndex].threshold.toFixed(2) })]),
      el('div', { class: 'panel-body tight' }, [
        split === 'test' ? el('div', { class: 'callout warn', text: 'Do not choose a threshold from the test split. Use validation data for threshold selection, then evaluate once on an untouched test set.' }) : null,
        canvas,
        controls,
        el('div', { class: 'scroll mt8', style: { maxHeight: '230px' } }, [table]),
        el('div', { class: 'tiny faint mt4', text: isDetection ? 'Pick the confidence your product can live with; precision/recall trade directly against detections per hour.' : 'FAR is the false-accept rate, FRR the false-reject rate. Shipping usually means choosing a FAR budget first and reading the FRR off this table.' })
      ])
    ]);
  }

  function anomalyView(metrics) {
    const hist = metrics.histograms;
    const canvas = el('canvas', { style: { width: '100%', height: '150px' } });
    const normal = hist.normal || [];
    const anomaly = hist.anomaly || [];
    lineChart(canvas, [
      { label: 'normal', y: normal.map((b) => b.count === undefined ? b : b.count), x: normal.map((b, i) => (b.bin === undefined ? i : b.bin)), color: SERIES_COLORS[1] },
      { label: 'anomaly', y: anomaly.map((b) => b.count === undefined ? b : b.count), x: anomaly.map((b, i) => (b.bin === undefined ? i : b.bin)), color: SERIES_COLORS[3] }
    ], { height: 150, xTitle: 'reconstruction error', pad: { top: 14, right: 12, bottom: 24, left: 42 } });
    return el('div', { class: 'chart-box' }, [
      el('div', { class: 'hd' }, [el('h4', { text: 'Score separation' }), el('span', { class: 'val', text: metrics.scalars.separation !== undefined ? 'sep ' + metrics.scalars.separation.toFixed(3) : '' })]),
      canvas,
      el('div', { class: 'tiny faint', text: 'Where the two distributions stop overlapping is where your threshold has to sit.' })
    ]);
  }

  function worstView(rec, metrics, ds) {
    const errors = metrics.errors || [];
    const rows = errors.slice(0, 40).map((e) => {
      const index = e.index;
      const preview = ds ? (() => { try { return samplePreviewCell(ds, index, { size: 76 }); } catch (err) { return el('span', { class: 'tiny faint', text: '—' }); } })() : null;
      const text = ds ? sampleText(ds, index) : null;
      const label = ds && ds.labels && ds.labels.length ? ds.labels[Math.round(ds.y[index * shapeSize(ds.yShape || [1])])] : null;
      const bits = Object.keys(e).filter((k) => k !== 'index').map((k) => k + ' ' + (typeof e[k] === 'number' ? e[k].toFixed(4) : String(e[k]))).join(' · ');
      return el('div', { class: 'sample-row', style: { gridTemplateColumns: '84px 1fr auto' } }, [
        preview || el('span', {}),
        el('div', {}, [
          el('div', { class: 'ell', style: { maxWidth: '460px' }, text: '#' + index + (text ? ' · ' + text.slice(0, 120) : '') + (label ? ' · true: ' + label : '') }),
          el('div', { class: 'tiny faint mono ell', style: { maxWidth: '460px' }, text: bits })
        ]),
        ds ? el('button', { class: 'xs ghost', text: 'inspect', onclick: () => inspectSample(rec, ds, index) }) : null
      ]);
    });
    return el('div', { class: 'panel' }, [
      el('div', { class: 'panel-head' }, [el('h3', { text: 'Worst examples' }), el('div', { class: 'spacer' }), el('span', { class: 'tiny faint', text: rows.length + ' of ' + errors.length + ' on the ' + split + ' split' })]),
      el('div', { class: 'panel-body flush' }, [el('div', { class: 'scroll', style: { maxHeight: '420px' } }, rows)])
    ]);
  }

  function inspectionView(rec, ds, metrics) {
    const host = el('div', { class: 'panel' }, [
      el('div', { class: 'panel-head' }, [el('h3', { text: 'Sample inspection' }), el('div', { class: 'spacer' }), el('span', { class: 'tiny faint', text: 'runs the stored weights on individual samples' })]),
      el('div', { class: 'panel-body tight', id: 'inspectBody' })
    ]);
    const body2 = host.querySelector('#inspectBody');
    if (!ds) { body2.appendChild(el('div', { class: 'callout warn', text: 'Load the dataset to inspect individual samples.' })); return host; }
    if (!rec.weights || !rec.weights.length) { body2.appendChild(el('div', { class: 'callout warn', text: 'This run has no stored weights.' })); return host; }
    const idx = ds.splits ? ds.splits[split] : null;
    if (!idx || !idx.length) { body2.appendChild(el('div', { class: 'callout warn', text: 'The ' + split + ' split is empty.' })); return host; }
    const picks = [];
    const stride = Math.max(1, Math.floor(idx.length / 8));
    for (let i = 0; i < idx.length && picks.length < 8; i += stride) picks.push(idx[i]);
    const strip = el('div', { class: 'preview-strip' });
    for (const index of picks) {
      const cell = el('div', { class: 'preview-cell', style: { cursor: 'pointer' }, onclick: () => renderSampleDetail(index) });
      try { cell.appendChild(samplePreviewCell(ds, index, { size: 108 })); } catch (e) { cell.appendChild(el('div', { class: 'tiny faint', text: '—' })); }
      cell.appendChild(el('div', { class: 'cap', text: '#' + index + ' · ' + (sampleText(ds, index) || labelOf(ds, index) || '') }));
      strip.appendChild(cell);
    }
    body2.appendChild(strip);
    const detail = el('div', { class: 'mt8', id: 'sampleDetail' });
    body2.appendChild(detail);
    let model = null;
    try {
      model = new Model(rec.spec, rec.seed || rec.config.seed);
      model.loadWeights(rec.weights);
      if (model.syncStates) model.syncStates();
    } catch (e) {
      body2.appendChild(el('div', { class: 'callout bad', text: 'Could not rebuild the model from the stored weights: ' + e.message }));
      return host;
    }
    renderSampleDetail(picks[0]);

    function renderSampleDetail(index) {
      clear(detail);
      try {
        detail.appendChild(sampleDetail(rec, ds, model, index, metrics));
      } catch (e) {
        detail.appendChild(el('div', { class: 'callout bad', text: 'Inspection failed: ' + e.message }));
      }
    }
    return host;
  }

  function labelOf(ds, index) {
    if (!ds.labels || !ds.labels.length || !ds.y) return null;
    const per = shapeSize(ds.yShape || [1]);
    return ds.labels[Math.round(ds.y[index * per])] || null;
  }

  function inspectSample(rec, ds, index) {
    if (!rec.weights || !rec.weights.length) { ctx.warn('No stored weights for this run.'); return; }
    let model;
    try {
      model = new Model(rec.spec, rec.seed || rec.config.seed);
      model.loadWeights(rec.weights);
      if (model.syncStates) model.syncStates();
    } catch (e) { ctx.error('Could not rebuild the model: ' + e.message); return; }
    const content = el('div', {});
    try { content.appendChild(sampleDetail(rec, ds, model, index, null)); } catch (e) { content.appendChild(el('div', { class: 'callout bad', text: e.message })); }
    const m = el('div', { class: 'modal-back' });
    const box = el('div', { class: 'modal', style: { maxWidth: '860px' } }, [
      el('header', {}, [el('h2', { text: 'Sample #' + index }), el('div', { class: 'spacer' }), el('button', { class: 'ghost sm', text: 'Close', onclick: () => m.remove() })]),
      el('div', { class: 'body' }, [content])
    ]);
    m.appendChild(box);
    m.addEventListener('mousedown', (event) => { if (event.target === m) m.remove(); });
    document.body.appendChild(m);
  }

  function sampleDetail(rec, ds, model, index, metrics) {
    const task = taskFor(rec.taskKey);
    const per = shapeSize(ds.xShape);
    const yPer = shapeSize(ds.yShape || [1]);
    const x = ds.x.subarray(index * per, (index + 1) * per);
    const y = ds.y ? ds.y.subarray(index * yPer, (index + 1) * yPer) : null;
    const host = el('div', { class: 'col gap8' });
    let output = null;
    try {
      const node = model.forward(makeBatchNode(ds, Int32Array.from([index])), 'eval');
      output = { shape: node.shape.slice(), data: Float32Array.from(node.d) };
    } catch (e) {
      host.appendChild(el('div', { class: 'callout bad', text: 'Forward pass failed: ' + e.message }));
      return host;
    }
    const head = rec.spec.head || 'linear';
    const activated = applyHeadLocal(output.data, head);
    const text = sampleText(ds, index);
    const meta = ds.meta && ds.meta[index] ? ds.meta[index] : null;
    host.appendChild(el('div', { class: 'row between wrap' }, [
      el('div', {}, [
        el('div', { class: 'bold', text: 'Sample #' + index + (text ? ' — ' + text.slice(0, 160) : '') }),
        el('div', { class: 'tiny faint', text: (meta ? Object.keys(meta).filter((k) => k !== 'text' && k !== 'preview').map((k) => k + ': ' + meta[k]).join(' · ') : '') })
      ]),
      el('div', { class: 'mono tiny faint', text: 'head ' + head + ' · output ' + output.shape.join('×') })
    ]));

    if (ds.modality === 'image' && ds.xShape.length === 3) {
      const strip = el('div', { class: 'preview-strip' });
      strip.appendChild(imageCell('input', x, ds.xShape, 150));
      if (y && yPer === per) strip.appendChild(imageCell('target', y, ds.xShape, 150));
      else if (y && ds.yShape && ds.yShape.length === 3) strip.appendChild(imageCell('target', y, ds.yShape, 150));
      if (activated.length === per) strip.appendChild(imageCell('prediction', activated, ds.xShape, 150));
      else if (y && activated.length === yPer) strip.appendChild(imageCell('prediction', activated, ds.yShape, 150));
      host.appendChild(strip);
      if (y && activated.length === y.length) {
        const psnr = psnrLocal(activated, y);
        host.appendChild(el('div', { class: 'tiny faint', text: 'PSNR against the target for this sample: ' + psnr.toFixed(2) + ' dB' }));
      }
    } else if (ds.xShape.length === 2 && (ds.modality === 'audio' || ds.task === 'spectral_mask' || task.key === 'spectral_mask')) {
      const frames = ds.xShape[0];
      const bins = ds.xShape[1];
      const strip = el('div', { class: 'preview-strip' });
      strip.appendChild(heatCell('input', x, frames, bins, 168, 96));
      if (y && y.length === frames * bins) strip.appendChild(heatCell('target', y, frames, bins, 168, 96));
      if (activated.length === frames * bins) strip.appendChild(heatCell('prediction', activated, frames, bins, 168, 96));
      host.appendChild(strip);
      if (ds.wave && ds.wave[index]) {
        const w = ds.wave[index];
        const waveStrip = el('div', { class: 'preview-strip' });
        if (w.noisy) waveStrip.appendChild(waveCell('noisy input', w.noisy, 210));
        if (w.clean) waveStrip.appendChild(waveCell('clean target', w.clean, 210));
        host.appendChild(waveStrip);
        host.appendChild(el('div', { class: 'tiny faint', text: 'stored SNR ' + (w.snr === undefined ? '—' : Number(w.snr).toFixed(1) + ' dB') + ' · ' + w.sampleRate + ' Hz · ' + w.frames + ' frames × ' + w.bins + ' bins' }));
      }
    } else if (task.key === 'forecast' || (ds.xShape.length === 2 && ds.modality === 'series')) {
      const canvas = el('canvas', { style: { width: '100%', height: '160px' } });
      const steps = ds.xShape[0];
      const channels = ds.xShape[1];
      const hist = [];
      for (let t = 0; t < steps; t++) hist.push(x[t * channels]);
      const horizon = y ? Array.from(y) : [];
      const predHorizon = Array.from(activated).slice(0, horizon.length);
      const xs = [];
      for (let t = 0; t < steps + horizon.length; t++) xs.push(t - steps);
      lineChart(canvas, [
        { label: 'history', y: hist.concat(new Array(horizon.length).fill(NaN)), x: xs.slice(0, steps + horizon.length), color: SERIES_COLORS[5] },
        { label: 'target', y: new Array(steps - 1).fill(NaN).concat([hist[hist.length - 1]]).concat(horizon), x: xs.slice(0, steps + horizon.length), color: SERIES_COLORS[1] },
        { label: 'prediction', y: new Array(steps - 1).fill(NaN).concat([hist[hist.length - 1]]).concat(predHorizon), x: xs.slice(0, steps + horizon.length), color: SERIES_COLORS[0], dash: [4, 3] }
      ], { height: 160, xTitle: 'step relative to now', pad: { top: 14, right: 12, bottom: 24, left: 44 } });
      host.appendChild(el('div', { class: 'chart-box' }, [el('div', { class: 'hd' }, [el('h4', { text: 'Forecast' })]), canvas]));
    } else if (task.key === 'frame_binary' || task.key === 'onset' || task.key === 'vad') {
      const canvas = el('canvas', { style: { width: '100%', height: '140px' } });
      const n = Math.max(activated.length, y ? y.length : 0);
      lineChart(canvas, [
        { label: 'target', y: y ? Array.from(y) : new Array(n).fill(NaN), color: SERIES_COLORS[1] },
        { label: 'prediction', y: Array.from(activated), color: SERIES_COLORS[0] }
      ], { height: 140, xTitle: 'frame', yMin: -0.05, yMax: 1.05, pad: { top: 14, right: 12, bottom: 24, left: 38 } });
      host.appendChild(el('div', { class: 'chart-box' }, [el('div', { class: 'hd' }, [el('h4', { text: 'Frame probabilities' })]), canvas]));
    } else if (task.key === 'multiclass' || task.key === 'multilabel') {
      const k = activated.length;
      const labels = ds.labels && ds.labels.length ? ds.labels : Array.from({ length: k }, (v, i) => 'class ' + i);
      const canvas = el('canvas', { style: { width: '100%', height: Math.max(70, k * 15 + 26) + 'px' } });
      barChart(canvas, Array.from({ length: k }, (v, i) => ({ label: labels[i], value: activated[i], display: activated[i].toFixed(3) })), { horizontal: true, height: Math.max(70, k * 15 + 26), labelChars: 20, max: 1, min: 0, background: '#191c1f' });
      host.appendChild(el('div', { class: 'chart-box' }, [
        el('div', { class: 'hd' }, [el('h4', { text: 'Predicted probabilities' }), el('span', { class: 'val', text: 'true: ' + (labelOf(ds, index) || '—') })]),
        canvas
      ]));
    } else if (task.key === 'binary') {
      host.appendChild(el('div', { class: 'callout', text: 'Score ' + activated[0].toFixed(4) + ' · true label ' + (labelOf(ds, index) || (y ? y[0].toFixed(2) : '—')) + (metrics && metrics.scalars && metrics.scalars.best_threshold !== undefined ? ' · decision threshold ' + metrics.scalars.best_threshold.toFixed(3) : '') }));
    } else {
      const limit = Math.min(24, activated.length);
      host.appendChild(el('div', { class: 'kv' }, [
        el('dt', { text: 'output[0…' + (limit - 1) + ']' }), el('dd', { text: Array.from(activated.slice(0, limit)).map((v) => v.toFixed(3)).join(', ') }),
        y ? el('dt', { text: 'target[0…' + (limit - 1) + ']' }) : null, y ? el('dd', { text: Array.from(y.slice(0, limit)).map((v) => v.toFixed(3)).join(', ') }) : null
      ]));
    }

    if (task.key === 'detection' && ds.boxes && ds.boxes[index]) {
      host.appendChild(el('div', { class: 'tiny faint', text: 'True boxes: ' + JSON.stringify(ds.boxes[index]) }));
    }
    return host;
  }

  function imageCell(title, data, shape, size) {
    const canvas = el('canvas');
    imageToCanvas(canvas, data, { channels: shape[0], width: shape[2], height: shape[1], displayWidth: size, displayHeight: Math.round(size * shape[1] / shape[2]) });
    return el('div', { class: 'preview-cell' }, [canvas, el('div', { class: 'cap', text: title })]);
  }

  function heatCell(title, data, frames, bins, width, height) {
    const canvas = el('canvas');
    const grid = [];
    for (let f = 0; f < frames; f++) grid.push(Array.from(data.subarray(f * bins, (f + 1) * bins)));
    heatmap(canvas, grid, { width, height, ramp: 'magma', pad: { top: 2, right: 2, bottom: 2, left: 2 } });
    return el('div', { class: 'preview-cell' }, [canvas, el('div', { class: 'cap', text: title })]);
  }

  function waveCell(title, samples, width) {
    const canvas = el('canvas', { style: { width: width + 'px', height: '52px' } });
    waveform(canvas, samples, { width, height: 52 });
    return el('div', { class: 'preview-cell' }, [canvas, el('div', { class: 'cap', text: title })]);
  }

  function applyHeadLocal(data, head) {
    const out = Float32Array.from(data);
    if (head === 'sigmoid' || head === 'mask' || head === 'heatmap') {
      for (let i = 0; i < out.length; i++) out[i] = 1 / (1 + Math.exp(-out[i]));
      return out;
    }
    if (head === 'softmax') {
      let max = -Infinity;
      for (let i = 0; i < out.length; i++) if (out[i] > max) max = out[i];
      let sum = 0;
      for (let i = 0; i < out.length; i++) { out[i] = Math.exp(out[i] - max); sum += out[i]; }
      for (let i = 0; i < out.length; i++) out[i] /= sum || 1;
      return out;
    }
    if (head === 'embedding') {
      let sum = 0;
      for (let i = 0; i < out.length; i++) sum += out[i] * out[i];
      const norm = Math.sqrt(sum) || 1;
      for (let i = 0; i < out.length; i++) out[i] /= norm;
      return out;
    }
    return out;
  }

  function psnrLocal(pred, target) {
    let mse = 0;
    for (let i = 0; i < target.length; i++) { const d = pred[i] - target[i]; mse += d * d; }
    mse /= Math.max(1, target.length);
    if (mse <= 1e-12) return 99;
    return 10 * Math.log10(1 / mse);
  }

  function downloadReport(format) {
    if (!record) { ctx.warn('No run selected.'); return; }
    const metrics = metricsFor(record);
    if (!metrics) { ctx.warn('No metrics to export.'); return; }
    if (format === 'json') {
      download((record.name || 'run').toLowerCase().replace(/[^a-z0-9]+/g, '-') + '-eval-' + split + '.json', JSON.stringify({
        run: { id: record.id, name: record.name, taskKey: record.taskKey, seed: record.seed, epochsRun: record.epochsRun, durationMs: record.durationMs },
        model: { name: record.modelName, params: record.totalParams, macs: record.macs, architectureHash: record.architectureHash },
        dataset: { name: record.datasetName, hash: record.datasetHash },
        split,
        threshold,
        config: record.config,
        metrics
      }, null, 2), 'application/json');
      ctx.ok('Evaluation report written for the ' + split + ' split.');
      return;
    }
    const scalars = metrics.scalars || {};
    const lines = [];
    lines.push('# Evaluation — ' + (record.name || record.id));
    lines.push('');
    lines.push('| | |');
    lines.push('|---|---|');
    lines.push('| Task | ' + record.taskKey + ' |');
    lines.push('| Model | ' + record.modelName + ' (' + (record.totalParams || 0).toLocaleString() + ' params) |');
    lines.push('| Dataset | ' + record.datasetName + ' (`' + String(record.datasetHash || '').slice(0, 16) + '`) |');
    lines.push('| Split | ' + split + ' |');
    lines.push('| Seed | ' + record.seed + ' |');
    lines.push('| Epochs | ' + record.epochsRun + ' |');
    lines.push('| Wall time | ' + (record.durationMs / 1000).toFixed(1) + ' s |');
    lines.push('');
    lines.push('## Metrics');
    lines.push('');
    lines.push('| Metric | Value | Direction |');
    lines.push('|---|---:|---|');
    for (const key of orderScalars(record.taskKey, scalars)) {
      const meta = metricMeta(key);
      lines.push('| ' + meta.label + ' (`' + key + '`) | ' + formatMetric(key, scalars[key]) + ' | ' + (meta.direction === 'up' ? 'higher is better' : meta.direction === 'down' ? 'lower is better' : 'reference') + ' |');
    }
    if (metrics.perClass) {
      lines.push('');
      lines.push('## Per class');
      lines.push('');
      lines.push('| Class | Support | Precision | Recall | F1 |');
      lines.push('|---|---:|---:|---:|---:|');
      for (const row of metrics.perClass) lines.push('| ' + row.label + ' | ' + row.support + ' | ' + row.precision.toFixed(3) + ' | ' + row.recall.toFixed(3) + ' | ' + row.f1.toFixed(3) + ' |');
    }
    if (metrics.confusion) {
      lines.push('');
      lines.push('## Confusion matrix');
      lines.push('');
      lines.push('| true \\ pred | ' + metrics.confusion.labels.join(' | ') + ' |');
      lines.push('|' + '---|'.repeat(metrics.confusion.labels.length + 1));
      metrics.confusion.matrix.forEach((row, i) => lines.push('| ' + metrics.confusion.labels[i] + ' | ' + row.join(' | ') + ' |'));
    }
    if (metrics.calibration) {
      lines.push('');
      lines.push('## Calibration');
      lines.push('');
      lines.push('ECE ' + metrics.calibration.ece.toFixed(4) + ', MCE ' + metrics.calibration.mce.toFixed(4) + ', Brier ' + metrics.calibration.brier.toFixed(4) + ', log loss ' + metrics.calibration.logLoss.toFixed(4) + '.');
    }
    if (metrics.sweep) {
      lines.push('');
      lines.push('## Threshold sweep');
      lines.push('');
      const keys = Object.keys(metrics.sweep[0]);
      lines.push('| ' + keys.join(' | ') + ' |');
      lines.push('|' + keys.map(() => '---:').join('|') + '|');
      for (const row of metrics.sweep) lines.push('| ' + keys.map((k) => (typeof row[k] === 'number' ? row[k].toFixed(4) : row[k])).join(' | ') + ' |');
    }
    lines.push('');
    lines.push('_Generated by Rootism Labs. Reproduce with seed ' + record.seed + ', dataset hash `' + record.datasetHash + '`, architecture hash `' + record.architectureHash + '`._');
    download((record.name || 'run').toLowerCase().replace(/[^a-z0-9]+/g, '-') + '-eval-' + split + '.md', lines.join('\n'), 'text/markdown');
    ctx.ok('Markdown evaluation report written.');
  }

  function render() {
    renderToolbar();
    renderBody().catch((e) => {
      clear(body);
      body.appendChild(el('div', { class: 'callout bad', text: 'Evaluation view failed: ' + e.message }));
    });
  }

  function refresh(reason) {
    if (reason === 'progress') return;
    if (reason === 'log') return;
    if (reason === 'selectRun' || reason === 'runs') { record = null; cache = {}; }
    render();
  }

  render();
  return { root, refresh, render };
}

