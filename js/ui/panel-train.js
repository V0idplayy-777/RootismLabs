import { el, clear, download } from './dom.js';
import { lineChart, barChart, confusionChart, calibrationChart, curveChart, histogram, SERIES_COLORS } from './charts.js';
import { configView, defaultConfig, mergeConfig, configSummary } from './train-config.js';
import { taskFor } from '../core/tasks.js';
import { metricMeta, formatMetric, headlineFor, orderScalars } from '../presets/metrics.js';
import { modelPresetByKey } from '../presets/models.js';

const SUBTABS = [
  { key: 'live', label: 'Live' },
  { key: 'epochs', label: 'Epochs' },
  { key: 'weights', label: 'Weights & gradients' },
  { key: 'samples', label: 'Predictions' },
  { key: 'log', label: 'Log' }
];

export function createTrainPanel(ctx) {
  const root = el('div', { class: 'panel-wrap' });
  const toolbar = el('div', { class: 'toolbar' });
  const body = el('div');
  root.appendChild(toolbar);
  root.appendChild(body);
  let subtab = 'live';
  let metricCursor = 'loss';
  let refs = {};

  function currentConfig() {
    if (!ctx.state.trainConfig) {
      ctx.state.trainConfig = defaultConfig(ctx.state.taskKey || (ctx.state.dataset ? ctx.state.dataset.task : 'multiclass'));
      ctx.state.trainConfig.epochs = 18;
      ctx.state.trainConfig.batchSize = 32;
    }
    return ctx.state.trainConfig;
  }

  function presetConfig() {
    const spec = ctx.state.graph;
    const key = spec && spec.meta ? spec.meta.presetKey : null;
    const preset = key ? modelPresetByKey(key) : null;
    return preset && preset.train ? JSON.parse(JSON.stringify(preset.train)) : null;
  }

  function renderToolbar() {
    clear(toolbar);
    const session = ctx.state.training;
    const cfg = currentConfig();
    const ready = !!(ctx.state.graph && ctx.state.graphStats && ctx.state.graphStats.valid && ctx.state.dataset && ctx.state.dataset.count);
    toolbar.appendChild(el('button', {
      class: 'primary', text: session ? 'Running…' : 'Start run', disabled: !!session || !ready,
      onclick: () => startRun()
    }));
    toolbar.appendChild(el('button', { class: 'sm', text: 'Pause', disabled: !session || session.paused, onclick: () => session && session.pause() }));
    toolbar.appendChild(el('button', { class: 'sm', text: 'Resume', disabled: !session || !session.paused, onclick: () => session && session.resume() }));
    toolbar.appendChild(el('button', { class: 'sm danger', text: 'Stop', disabled: !session, onclick: () => session && session.stop() }));
    toolbar.appendChild(el('div', { class: 'vsep' }));
    const preset = presetConfig();
    toolbar.appendChild(el('button', {
      class: 'sm ghost', text: 'Preset config', disabled: !preset,
      onclick: () => {
        if (!preset) return;
        Object.assign(cfg, mergeConfig(Object.assign({}, cfg, preset)));
        ctx.state.trainConfig = mergeConfig(cfg);
        ctx.info('Applied the preset training config: ' + JSON.stringify(preset));
        render();
      }
    }));
    toolbar.appendChild(el('button', { class: 'sm ghost', text: 'Reset config', onclick: () => { ctx.state.trainConfig = defaultConfig(ctx.state.taskKey || 'multiclass'); render(); ctx.info('Training config reset to defaults.'); } }));
    toolbar.appendChild(el('button', { class: 'sm ghost', text: 'Copy config', onclick: () => { navigator.clipboard.writeText(JSON.stringify(ctx.state.trainConfig, null, 2)).then(() => ctx.toast('Config JSON copied.')); } }));
    toolbar.appendChild(el('button', { class: 'sm ghost', text: 'Download config', onclick: () => download('train-config.json', JSON.stringify(ctx.state.trainConfig, null, 2), 'application/json') }));
    toolbar.appendChild(el('div', { class: 'spacer' }));
    if (!ready) {
      toolbar.appendChild(el('span', { class: 'tiny faint', text: !ctx.state.graph ? 'needs an architecture' : !ctx.state.graphStats || !ctx.state.graphStats.valid ? 'architecture has errors' : 'needs a dataset' }));
    } else {
      toolbar.appendChild(el('span', { class: 'tiny faint', text: ctx.state.graphStats.totalParams.toLocaleString() + ' params · ' + ctx.state.dataset.count + ' samples · ' + (ctx.state.taskKey || ctx.state.dataset.task) }));
    }
  }

  async function startRun() {
    const cfg = currentConfig();
    const spec = ctx.state.graph;
    const preset = presetConfig();
    const name = cfg.runName || ((spec ? spec.name : 'model') + ' · ' + new Date().toISOString().slice(5, 16).replace('T', ' '));
    cfg.runName = name;
    const result = await ctx.startRun({
      config: Object.assign({}, cfg, { augmentation: ctx.state.dataset && ctx.state.dataset.augmentation ? ctx.state.dataset.augmentation : cfg.augmentation }),
      taskKey: ctx.state.taskKey || ctx.state.dataset.task,
      name,
      primaryMetric: cfg.primaryMetric || null
    });
    if (result) {
      subtab = 'live';
      render();
    }
  }

  function renderHead() {
    const session = ctx.state.training;
    const live = session ? session.live : (ctx.state.lastLive || null);
    const host = el('div', { class: 'train-head' });
    if (!live) {
      host.appendChild(el('div', {}, [
        el('div', { class: 'lbl', text: 'status' }),
        el('div', { class: 'big', text: 'idle' })
      ]));
      host.appendChild(el('div', { class: 'small dim', text: 'Configure the run on the left, then start it. Progress, loss, learning rate, gradient norms and per-epoch metrics stream in here while it trains.' }));
      return host;
    }
    const phase = live.phase || 'idle';
    host.appendChild(el('div', {}, [el('div', { class: 'lbl', text: 'phase' }), el('div', { class: 'big', style: { color: phaseColor(phase), fontSize: '15px' }, text: phase })]));
    host.appendChild(headStat('epoch', (live.epoch || 0) + ' / ' + (live.epochs || 0)));
    host.appendChild(headStat('train loss', fmtNum(live.trainLoss)));
    host.appendChild(headStat('lr', live.lr !== undefined && isFinite(live.lr) ? Number(live.lr).toExponential(2) : '—'));
    host.appendChild(headStat('eta', (live.eta || 0).toFixed(0) + 's'));
    host.appendChild(headStat('elapsed', (live.elapsed || 0).toFixed(0) + 's'));
    host.appendChild(headStat('throughput', (live.throughput || 0).toFixed(0) + ' /s'));
    host.appendChild(headStat('samples seen', (live.samplesSeen || 0).toLocaleString()));
    host.appendChild(headStat('grad norm', fmtNum(live.gradNorm)));
    if (live.lastVal && live.lastVal.scalars) {
      const key = monitoredKey(live);
      host.appendChild(headStat(key.replace(/_/g, ' '), fmtNum(live.lastVal.scalars[key]), 'amber'));
    }
    const progress = el('div', { class: 'progress', style: { flex: '1 1 140px', minWidth: '120px' } }, [el('i', { style: { width: ((live.percent || 0) * 100).toFixed(1) + '%' } })]);
    host.appendChild(progress);
    return host;
  }

  function monitoredKey(live) {
    const scalars = live.lastVal && live.lastVal.scalars ? live.lastVal.scalars : {};
    const cfg = ctx.state.trainConfig || {};
    const preferred = cfg.primaryMetric || headlineFor(ctx.state.taskKey || 'multiclass');
    if (scalars[preferred] !== undefined) return preferred;
    const keys = Object.keys(scalars);
    return keys.length ? keys[0] : 'loss';
  }

  function headStat(label, value, tone) {
    return el('div', {}, [
      el('div', { class: 'lbl', text: label }),
      el('div', { class: 'num', style: { fontSize: '14px', color: tone === 'amber' ? 'var(--amber)' : 'var(--text)' }, text: value })
    ]);
  }

  function phaseColor(phase) {
    if (phase === 'running') return 'var(--amber)';
    if (phase === 'done') return 'var(--green)';
    if (phase === 'failed' || phase === 'stopped') return 'var(--red)';
    if (phase === 'paused') return 'var(--blue)';
    return 'var(--text)';
  }

  function fmtNum(v) {
    if (v === undefined || v === null || !isFinite(v)) return '—';
    const abs = Math.abs(v);
    if (abs >= 1000) return v.toFixed(0);
    if (abs >= 10) return v.toFixed(2);
    if (abs >= 0.01) return v.toFixed(4);
    return v.toExponential(2);
  }

  function history() {
    const session = ctx.state.training;
    if (session && session.live.history.length) return session.live.history;
    if (ctx.state.lastLive && ctx.state.lastLive.history) return ctx.state.lastLive.history;
    return [];
  }

  function renderCharts() {
    const rows = history();
    const grid = el('div', { class: 'chart-grid' });
    grid.appendChild(chartBox('Loss', rows.length ? fmtNum(rows[rows.length - 1].valLoss) : '—', (canvas) => {
      const epochs = rows.map((r) => r.epoch);
      lineChart(canvas, [
        { label: 'train', y: rows.map((r) => r.trainLoss), x: epochs, color: SERIES_COLORS[0] },
        { label: 'val', y: rows.map((r) => r.valLoss), x: epochs, color: SERIES_COLORS[1] }
      ], { height: 148, xTitle: 'epoch', pad: { top: 14, right: 12, bottom: 24, left: 46 }, threshold: rows.length && rows[0].trainLoss !== undefined ? null : undefined });
    }, rows.length ? '' : 'waiting for the first epoch'));

    const key = rows.length ? monitoredKey(ctx.state.training ? ctx.state.training.live : ctx.state.lastLive) : 'loss';
    grid.appendChild(chartBox('Monitored · ' + key.replace(/_/g, ' '), rows.length ? fmtNum(rows[rows.length - 1].monitoredValue) : '—', (canvas) => {
      lineChart(canvas, [{ label: key, y: rows.map((r) => r.monitoredValue), x: rows.map((r) => r.epoch), color: SERIES_COLORS[1], marker: bestEpoch(rows) }], { height: 148, xTitle: 'epoch', pad: { top: 14, right: 12, bottom: 24, left: 46 } });
    }, rows.length ? 'dashed line marks the best checkpoint' : ''));

    grid.appendChild(chartBox('Learning rate', rows.length ? Number(rows[rows.length - 1].lr).toExponential(2) : '—', (canvas) => {
      lineChart(canvas, [{ label: 'lr', y: rows.map((r) => r.lr), x: rows.map((r) => r.epoch), color: SERIES_COLORS[2] }], { height: 148, xTitle: 'epoch', pad: { top: 14, right: 12, bottom: 24, left: 52 } });
    }, ''));

    grid.appendChild(chartBox('Gradient norm', rows.length ? fmtNum(rows[rows.length - 1].gradNorm) : '—', (canvas) => {
      lineChart(canvas, [{ label: '‖g‖', y: rows.map((r) => r.gradNorm), x: rows.map((r) => r.epoch), color: SERIES_COLORS[3] }], { height: 148, xTitle: 'epoch', pad: { top: 14, right: 12, bottom: 24, left: 46 } });
    }, 'mean over the epoch, after clipping'));

    grid.appendChild(chartBox('Epoch time / throughput', rows.length ? rows[rows.length - 1].seconds.toFixed(1) + 's' : '—', (canvas) => {
      lineChart(canvas, [
        { label: 'seconds', y: rows.map((r) => r.seconds), x: rows.map((r) => r.epoch), color: SERIES_COLORS[4] },
        { label: 'samples/s ÷10', y: rows.map((r) => (r.samplesPerSec || 0) / 10), x: rows.map((r) => r.epoch), color: SERIES_COLORS[5], dash: [3, 3] }
      ], { height: 148, xTitle: 'epoch', pad: { top: 14, right: 12, bottom: 24, left: 46 } });
    }, ''));

    const scalarKeys = scalarUnion(rows);
    const choices = scalarKeys.length ? scalarKeys : ['loss'];
    if (choices.indexOf(metricCursor) < 0) metricCursor = choices[0];
    const metricBox = el('div', { class: 'chart-box' }, [
      el('div', { class: 'hd' }, [
        el('h4', { text: 'Validation metric' }),
        el('select', {
          style: { marginLeft: '8px', fontSize: '10.5px', padding: '1px 4px' },
          onchange: (event) => { metricCursor = event.target.value; renderBody(); }
        }, choices.map((k) => el('option', { value: k, text: k, selected: k === metricCursor }))),
        el('span', { class: 'val', text: rows.length && rows[rows.length - 1].scalars[metricCursor] !== undefined ? fmtNum(rows[rows.length - 1].scalars[metricCursor]) : '—' })
      ]),
      el('canvas', { style: { width: '100%', height: '148px' } }),
      el('div', { class: 'tiny faint', text: metricMeta(metricCursor).explain || '' })
    ]);
    const metricCanvas = metricBox.querySelector('canvas');
    lineChart(metricCanvas, [{ label: metricCursor, y: rows.map((r) => (r.scalars ? r.scalars[metricCursor] : NaN)), x: rows.map((r) => r.epoch), color: SERIES_COLORS[0] }], { height: 148, xTitle: 'epoch', pad: { top: 14, right: 12, bottom: 24, left: 46 }, emptyText: 'metric not produced yet' });
    grid.appendChild(metricBox);
    return grid;
  }

  function bestEpoch(rows) {
    const cfg = ctx.state.trainConfig || {};
    const mode = cfg.earlyStopping && cfg.earlyStopping.mode === 'min' ? 'min' : 'max';
    let best = null;
    let bestValue = mode === 'min' ? Infinity : -Infinity;
    for (const r of rows) {
      if (!isFinite(r.monitoredValue)) continue;
      const better = mode === 'min' ? r.monitoredValue < bestValue : r.monitoredValue > bestValue;
      if (better) { bestValue = r.monitoredValue; best = r.epoch; }
    }
    return best;
  }

  function scalarUnion(rows) {
    const out = [];
    for (const r of rows) for (const k of Object.keys(r.scalars || {})) if (out.indexOf(k) < 0) out.push(k);
    const task = taskFor(ctx.state.taskKey || 'multiclass');
    return orderScalars(task.key, Object.fromEntries(out.map((k) => [k, 0])));
  }

  function chartBox(title, value, draw, note) {
    const box = el('div', { class: 'chart-box' }, [
      el('div', { class: 'hd' }, [el('h4', { text: title }), el('span', { class: 'val', text: value })]),
      el('canvas', { style: { width: '100%', height: '148px' } }),
      note ? el('div', { class: 'tiny faint', text: note }) : null
    ]);
    draw(box.querySelector('canvas'));
    return box;
  }

  function renderEpochTable() {
    const rows = history();
    const host = el('div', { class: 'panel' }, [
      el('div', { class: 'panel-head' }, [el('h3', { text: 'Epoch table' }), el('div', { class: 'spacer' }), el('button', { class: 'xs ghost', text: 'Download CSV', onclick: () => downloadEpochs(rows) })]),
      el('div', { class: 'panel-body flush' })
    ]);
    const bodyHost = host.querySelector('.panel-body');
    if (!rows.length) { bodyHost.appendChild(el('div', { class: 'tiny faint', style: { padding: '10px' }, text: 'No epochs recorded yet.' })); return host; }
    const scalarKeys = scalarUnion(rows).slice(0, 8);
    const table = el('table', { class: 'data compact' }, [
      el('thead', {}, el('tr', {}, [
        el('th', { class: 'n', text: 'ep' }), el('th', { class: 'n', text: 'train' }), el('th', { class: 'n', text: 'val' }),
        el('th', { class: 'n', text: 'lr' }), el('th', { class: 'n', text: '‖g‖' }), el('th', { class: 'n', text: 'sec' })
      ].concat(scalarKeys.map((k) => el('th', { class: 'n', text: k })))))
    ]);
    const tbody = el('tbody');
    const best = bestEpoch(rows);
    for (const r of rows) {
      const tr = el('tr', { class: r.epoch === best ? 'selected' : '' }, [
        el('td', { class: 'n', text: String(r.epoch) }),
        el('td', { class: 'n', text: fmtNum(r.trainLoss) }),
        el('td', { class: 'n', text: fmtNum(r.valLoss) }),
        el('td', { class: 'n faint', text: Number(r.lr).toExponential(1) }),
        el('td', { class: 'n faint', text: fmtNum(r.gradNorm) }),
        el('td', { class: 'n faint', text: r.seconds.toFixed(1) })
      ]);
      for (const k of scalarKeys) tr.appendChild(el('td', { class: 'n', text: r.scalars && r.scalars[k] !== undefined ? fmtNum(r.scalars[k]) : '—' }));
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    bodyHost.appendChild(el('div', { class: 'epoch-table-wrap' }, [table]));
    return host;
  }

  function downloadEpochs(rows) {
    const keys = scalarUnion(rows);
    const header = ['epoch', 'trainLoss', 'valLoss', 'lr', 'gradNorm', 'seconds', 'samplesPerSec', 'monitored', 'monitoredValue'].concat(keys);
    const lines = [header.join(',')];
    for (const r of rows) {
      lines.push([r.epoch, r.trainLoss, r.valLoss, r.lr, r.gradNorm, r.seconds, r.samplesPerSec, r.monitored, r.monitoredValue].concat(keys.map((k) => (r.scalars && r.scalars[k] !== undefined ? r.scalars[k] : ''))).join(','));
    }
    download('epochs.csv', lines.join('\n'), 'text/csv');
  }

  function renderLiveMetrics() {
    const session = ctx.state.training;
    const live = session ? session.live : ctx.state.lastLive;
    const val = live ? live.lastVal : null;
    const host = el('div', { class: 'col gap8' });
    if (!val) {
      host.appendChild(el('div', { class: 'callout', text: 'Validation metrics appear here after the first epoch.' }));
      return host;
    }
    const dataset = ctx.state.dataset;
    const scalars = val.scalars || {};
    const task = taskFor(ctx.state.taskKey || (dataset ? dataset.task : 'multiclass'));
    const ordered = orderScalars(task.key, scalars);
    const grid = el('div', { class: 'grid-3' });
    for (const key of ordered) {
      const meta = metricMeta(key);
      grid.appendChild(el('div', { class: 'stat' }, [
        el('div', { class: 'k', text: meta.label }),
        el('div', { class: 'v', text: formatMetric(key, scalars[key]) }),
        el('div', { class: 's', text: meta.direction === 'up' ? 'higher is better' : meta.direction === 'down' ? 'lower is better' : '' })
      ]));
    }
    host.appendChild(el('div', {}, [el('div', { class: 'section-title', text: 'Validation scalars', count: ordered.length + ' metrics' }), grid]));

    if (val.confusion && val.confusion.matrix) {
      const box = el('div', { class: 'panel' }, [
        el('div', { class: 'panel-head' }, [el('h3', { text: 'Confusion matrix (validation)' }), el('div', { class: 'spacer' }), el('span', { class: 'tiny faint', text: 'row = true, column = predicted, normalised per row' })]),
        el('div', { class: 'panel-body tight' }, [
          (() => {
            const n = val.confusion.labels.length;
            const canvas = el('canvas', { style: { width: '100%', height: Math.max(160, Math.min(420, n * 22 + 40)) + 'px' } });
            confusionChart(canvas, val.confusion.matrix, val.confusion.labels, { height: Math.max(160, Math.min(420, n * 22 + 40)) });
            return canvas;
          })()
        ])
      ]);
      host.appendChild(box);
    }

    if (Array.isArray(val.perClass)) {
      const table = el('table', { class: 'data compact' }, [
        el('thead', {}, el('tr', {}, [el('th', { text: 'class' }), el('th', { class: 'n', text: 'support' }), el('th', { class: 'n', text: 'precision' }), el('th', { class: 'n', text: 'recall' }), el('th', { class: 'n', text: 'f1' }), el('th', { class: 'n', text: 'tp' }), el('th', { class: 'n', text: 'fp' }), el('th', { class: 'n', text: 'fn' })])),
        el('tbody', {}, val.perClass.map((row) => el('tr', {}, [
          el('td', { text: String(row.label) }),
          el('td', { class: 'n', text: String(row.support) }),
          el('td', { class: 'n', text: fmtNum(row.precision) }),
          el('td', { class: 'n', text: fmtNum(row.recall) }),
          el('td', { class: 'n', text: fmtNum(row.f1) }),
          el('td', { class: 'n faint', text: String(row.tp) }),
          el('td', { class: 'n faint', text: String(row.fp) }),
          el('td', { class: 'n faint', text: String(row.fn) })
        ])))
      ]);
      host.appendChild(el('div', { class: 'panel' }, [
        el('div', { class: 'panel-head' }, [el('h3', { text: 'Per class' })]),
        el('div', { class: 'panel-body flush' }, [el('div', { class: 'scroll', style: { maxHeight: '230px' } }, [table])])
      ]));
    }

    if (val.calibration && val.calibration.bins) {
      const canvas = el('canvas', { style: { width: '100%', height: '168px' } });
      calibrationChart(canvas, val.calibration, { height: 168 });
      host.appendChild(el('div', { class: 'chart-box' }, [
        el('div', { class: 'hd' }, [el('h4', { text: 'Reliability' }), el('span', { class: 'val', text: 'ECE ' + fmtNum(val.calibration.ece) + ' · Brier ' + fmtNum(val.calibration.brier) })]),
        canvas,
        el('div', { class: 'tiny faint', text: 'Bar height is the sample count in the bin; the dot is the mean predicted probability against the observed rate.' })
      ]));
    }
    if (val.roc && val.roc.points) {
      const canvas = el('canvas', { style: { width: '100%', height: '168px' } });
      curveChart(canvas, val.roc.points, { height: 168, xTitle: 'false positive rate', yTitle: 'true positive rate', annotation: 'AUC ' + fmtNum(val.roc.auc) });
      host.appendChild(el('div', { class: 'chart-box' }, [el('div', { class: 'hd' }, [el('h4', { text: 'ROC' })]), canvas]));
    }
    if (val.pr && val.pr.points) {
      const canvas = el('canvas', { style: { width: '100%', height: '168px' } });
      curveChart(canvas, val.pr.points, { height: 168, xTitle: 'recall', yTitle: 'precision', annotation: 'AP ' + fmtNum(val.pr.ap) });
      host.appendChild(el('div', { class: 'chart-box' }, [el('div', { class: 'hd' }, [el('h4', { text: 'Precision-recall' })]), canvas]));
    }
    return host;
  }

  function renderWeights() {
    const session = ctx.state.training;
    const live = session ? session.live : ctx.state.lastLive;
    const host = el('div', { class: 'col gap8' });
    if (!live || (!live.weightNorms && !live.gradNorms)) {
      host.appendChild(el('div', { class: 'callout', text: 'Weight and gradient statistics are collected at the end of each epoch.' }));
      return host;
    }
    if (live.weightNorms) {
      const top = live.weightNorms.slice(0, 18);
      const canvas = el('canvas', { style: { width: '100%', height: (top.length * 16 + 26) + 'px' } });
      barChart(canvas, top.map((w) => ({ label: w.name, value: w.l2, display: w.l2.toFixed(2) + '  (rms ' + w.rms.toFixed(3) + ')' })), { horizontal: true, labelChars: 26, height: top.length * 16 + 26, background: '#191c1f' });
      host.appendChild(el('div', { class: 'panel' }, [
        el('div', { class: 'panel-head' }, [el('h3', { text: 'Weight L2 norms' }), el('div', { class: 'spacer' }), el('span', { class: 'tiny faint', text: top.length + ' of ' + live.weightNorms.length + ' tensors' })]),
        el('div', { class: 'panel-body tight' }, [canvas])
      ]));
      const all = live.weightNorms.map((w) => w.rms);
      const hist = el('canvas', { style: { width: '100%', height: '96px' } });
      histogram(hist, all, { bins: 26, color: '#e0a33e', height: 96, background: '#191c1f' });
      host.appendChild(el('div', { class: 'chart-box' }, [el('div', { class: 'hd' }, [el('h4', { text: 'Distribution of tensor RMS' })]), hist]));
    }
    if (live.gradNorms) {
      const top = live.gradNorms.slice(0, 18);
      const canvas = el('canvas', { style: { width: '100%', height: (top.length * 16 + 26) + 'px' } });
      barChart(canvas, top.map((g) => ({ label: g.name, value: g.l2, display: g.l2.toExponential(2) })), { horizontal: true, labelChars: 22, height: top.length * 16 + 26, background: '#191c1f', color: SERIES_COLORS[3] });
      host.appendChild(el('div', { class: 'panel' }, [
        el('div', { class: 'panel-head' }, [el('h3', { text: 'Per-layer gradient norms' }), el('div', { class: 'spacer' }), el('span', { class: 'tiny faint', text: 'last batch of the epoch' })]),
        el('div', { class: 'panel-body tight' }, [canvas]),
        el('div', { class: 'panel-body tight' }, [el('div', { class: 'tiny faint', text: 'Layers stuck near zero are not learning; layers orders of magnitude above the rest are where clipping is doing the work.' })])
      ]));
    }
    return host;
  }

  function renderPredictions() {
    const session = ctx.state.training;
    const live = session ? session.live : ctx.state.lastLive;
    const host = el('div', { class: 'col gap8' });
    const preds = live && live.predictions ? live.predictions : [];
    if (!preds.length) {
      host.appendChild(el('div', { class: 'callout', text: 'The hardest validation samples appear here after the first epoch.' }));
      return host;
    }
    const dataset = ctx.state.dataset;
    const rows = preds.map((p) => {
      const predicted = p.predicted !== undefined && p.predicted !== null ? (dataset && dataset.labels && dataset.labels.length && Number.isInteger(p.predicted) ? dataset.labels[p.predicted] : p.predicted) : '—';
      const actual = p.actual !== undefined && p.actual !== null ? (dataset && dataset.labels && dataset.labels.length && Number.isInteger(p.actual) ? dataset.labels[p.actual] : p.actual) : '—';
      const wrong = String(predicted) !== String(actual);
      return el('div', { class: 'sample-row' }, [
        el('div', { class: 'ell', text: (p.text || '(no text)') + (p.meta ? '  ·  ' + Object.keys(p.meta).filter((k) => k !== 'text' && k !== 'preview').map((k) => k + ':' + p.meta[k]).join(' ') : '') }),
        el('span', { class: 'mono tiny', style: { color: wrong ? '#e0938c' : 'var(--muted)' }, text: 'pred ' + predicted }),
        el('span', { class: 'mono tiny teal', text: 'true ' + actual }),
        el('span', { class: 'mono tiny faint', text: p.confidence !== undefined && p.confidence !== null ? Number(p.confidence).toFixed(3) : p.psnr !== undefined ? 'psnr ' + Number(p.psnr).toFixed(2) : '' })
      ]);
    });
    host.appendChild(el('div', { class: 'panel' }, [
      el('div', { class: 'panel-head' }, [el('h3', { text: 'Hardest validation samples' }), el('div', { class: 'spacer' }), el('span', { class: 'tiny faint', text: rows.length + ' shown' })]),
      el('div', { class: 'panel-body flush' }, [el('div', { class: 'scroll', style: { maxHeight: '46vh' } }, rows)])
    ]));
    return host;
  }

  function renderRunLog() {
    const host = el('div', { class: 'panel' }, [
      el('div', { class: 'panel-head' }, [el('h3', { text: 'Training log' }), el('div', { class: 'spacer' }), el('button', { class: 'xs ghost', text: 'Open session log', onclick: () => ctx.openLog() })]),
      el('div', { class: 'panel-body flush' })
    ]);
    const body = host.querySelector('.panel-body');
    const logBox = el('div', { class: 'log', style: { maxHeight: '46vh', border: '0', borderRadius: '0' } });
    const entries = ctx.state.logs.slice(-160);
    for (const entry of entries) {
      logBox.appendChild(el('div', { class: 'line lvl-' + entry.level }, [
        el('span', { class: 't', text: new Date(entry.t).toTimeString().slice(0, 8) + '  ' }),
        entry.message
      ]));
    }
    body.appendChild(logBox);
    logBox.scrollTop = logBox.scrollHeight;
    return host;
  }

  function renderRight() {
    const host = el('div', { class: 'col gap8' });
    host.appendChild(renderHead());
    host.appendChild(el('div', { class: 'tab-strip' }, SUBTABS.map((t) => el('button', { class: subtab === t.key ? 'on' : '', text: t.label, onclick: () => { subtab = t.key; renderBody(); } }))));
    if (subtab === 'live') {
      host.appendChild(renderCharts());
      host.appendChild(renderLiveMetrics());
    } else if (subtab === 'epochs') {
      host.appendChild(renderEpochTable());
      host.appendChild(renderCharts());
    } else if (subtab === 'weights') {
      host.appendChild(renderWeights());
    } else if (subtab === 'samples') {
      host.appendChild(renderPredictions());
    } else if (subtab === 'log') {
      host.appendChild(renderRunLog());
    }
    return host;
  }

  function renderLeft() {
    const cfg = currentConfig();
    const host = el('div', { class: 'col gap8' });
    const summary = configSummary(cfg, ctx.state.taskKey || (ctx.state.dataset ? ctx.state.dataset.task : 'multiclass'), cfg.primaryMetric);
    host.appendChild(el('div', { class: 'panel' }, [
      el('div', { class: 'panel-head' }, [el('h3', { text: 'Run plan' })]),
      el('div', { class: 'panel-body tight' }, [
        el('div', { class: 'kv' }, Object.keys(summary).map((k) => [el('dt', { text: k }), el('dd', { text: String(summary[k]) })]).flat())
      ])
    ]));
    host.appendChild(configView(ctx, cfg, {
      taskKey: ctx.state.taskKey || (ctx.state.dataset ? ctx.state.dataset.task : 'multiclass'),
      dataset: ctx.state.dataset,
      name: cfg.runName || '',
      primaryMetric: cfg.primaryMetric || null,
      onTask: (taskKey) => { ctx.state.taskKey = taskKey; ctx.state.trainConfig = defaultConfig(taskKey); render(); },
      onChange: () => { renderToolbar(); }
    }));
    return host;
  }

  function renderBody() {
    clear(body);
    body.appendChild(el('div', { class: 'split train' }, [renderLeft(), renderRight()]));
  }

  function render() {
    renderToolbar();
    renderBody();
  }

  function refresh(reason) {
    if (reason === 'progress') {
      const session = ctx.state.training;
      if (!session) return;
      const head = body.querySelector('.train-head');
      if (head) {
        const fresh = renderHead();
        head.replaceWith(fresh);
      }
      renderToolbar();
      return;
    }
    if (reason === 'log') {
      if (subtab === 'log') renderBody();
      return;
    }
    if (reason === 'epoch') {
      const session = ctx.state.training;
      if (session) ctx.state.lastLive = session.live;
      renderBody();
      return;
    }
    render();
  }

  render();
  return { root, refresh, render };
}
