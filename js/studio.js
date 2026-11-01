import { el, clear, download, modal, setFavicon, svgEl, markSvg } from './ui/dom.js';
import * as S from './ui/state.js';
import { analyze, compactNumber } from './core/model.js';
import { recompute, qualityReport } from './data/dataset.js';
import { taskFor } from './core/tasks.js';
import { formatBytes } from './presets/metrics.js';
import { createBuildPanel } from './ui/panel-build.js';
import { createDataPanel } from './ui/panel-data.js';
import { createTrainPanel } from './ui/panel-train.js';
import { createEvalPanel } from './ui/panel-eval.js';
import { createComparePanel } from './ui/panel-compare.js';
import { createExportPanel } from './ui/panel-export.js';
import { createLibraryPanel } from './ui/panel-library.js';
import { RunSession } from './ui/runner.js';
import { TOOL_VERSION } from './export/card.js';

export const PANELS = [
  { key: 'build', label: 'Build', idx: '01', hint: 'architecture' },
  { key: 'data', label: 'Data', idx: '02', hint: 'dataset' },
  { key: 'train', label: 'Train', idx: '03', hint: 'fit' },
  { key: 'evaluate', label: 'Evaluate', idx: '04', hint: 'measure' },
  { key: 'compare', label: 'Compare', idx: '05', hint: 'runs' },
  { key: 'export', label: 'Export', idx: '06', hint: 'ship' },
  { key: 'library', label: 'Library', idx: '07', hint: 'saved work' }
];

const state = S.state;
let panelHost = null;
let railNav = null;
let topStats = null;
let statusBar = null;
let logDrawer = null;
let logOpen = false;
let toastHost = null;
const panelInstances = {};
let pendingFrame = null;

function analyzeSpec(spec) {
  const info = analyze(spec);
  const byNode = {};
  for (const stat of info.nodeStats || []) byNode[stat.id] = stat;
  const errorByNode = {};
  for (const err of info.errors || []) errorByNode[err.id] = err.message;
  info.byNode = byNode;
  info.errorByNode = errorByNode;
  return info;
}

export const ctx = {
  state,
  panels: panelInstances,
  go,
  notify: (reason) => S.notify(reason),
  notifyThrottled,
  log: S.log,
  info: S.info,
  warn: S.warn,
  error: S.error,
  ok: S.ok,
  toast,
  setSpec,
  touchGraph,
  saveGraph,
  setDataset,
  touchDataset,
  saveDataset,
  task: () => taskFor(state.taskKey || 'multiclass'),
  requireModel,
  requireDataset,
  startRun,
  finishRun,
  analyzeSpec,
  openLog: () => setLogOpen(true),
  closeLog: () => setLogOpen(false)
};

function notifyThrottled(reason) {
  if (pendingFrame) return;
  pendingFrame = requestAnimationFrame(() => {
    pendingFrame = null;
    S.notify(reason);
  });
}

function toast(message, tone) {
  if (!toastHost) return;
  const node = el('div', { class: 'toast ' + (tone || 'info'), text: message });
  toastHost.appendChild(node);
  requestAnimationFrame(() => node.classList.add('in'));
  setTimeout(() => {
    node.classList.remove('in');
    setTimeout(() => node.remove(), 260);
  }, tone === 'error' ? 6200 : 3600);
}

function setSpec(spec, options) {
  const opts = options || {};
  state.graph = spec;
  state.graphStats = spec ? analyzeSpec(spec) : null;
  state.graphDirty = true;
  if (!opts.keepId) {
    state.graphId = null;
    state.graphSavedAt = null;
  }
  if (!state.taskKey && spec && spec.meta && spec.meta.task) state.taskKey = spec.meta.task;
  S.notify('graph');
}

function touchGraph() {
  if (!state.graph) return;
  state.graphStats = analyzeSpec(state.graph);
  state.graphDirty = true;
  S.notify('graph');
}

async function saveGraph(name) {
  if (!state.graph) { toast('Nothing to save — load a preset first.', 'error'); return null; }
  if (!state.graphStats || !state.graphStats.valid) { toast('The graph is not valid yet; fix the reported problems first.', 'error'); return null; }
  const id = await S.saveCurrentGraph(name);
  toast('Saved "' + (state.graph.name || 'model') + '" to the library.');
  S.ok('Saved model ' + (state.graph.name || id) + ' (' + state.graphStats.totalParams.toLocaleString() + ' params).');
  return id;
}

function setDataset(ds, options) {
  const opts = options || {};
  state.dataset = ds;
  state.datasetDirty = true;
  if (!opts.keepId) state.datasetId = ds ? ds.id : null;
  if (ds) {
    touchDataset(true);
    if (ds.task) state.taskKey = ds.task;
  } else {
    S.notify('dataset');
  }
}

function touchDataset(silent) {
  if (!state.dataset) return;
  try {
    recompute(state.dataset);
    state.dataset.quality = qualityReport(state.dataset);
  } catch (e) {
    S.warn('Dataset recompute failed: ' + e.message);
  }
  state.datasetDirty = true;
  if (!silent) S.notify('dataset');
}

async function saveDataset(name) {
  if (!state.dataset) { toast('No dataset loaded.', 'error'); return null; }
  const id = await S.saveCurrentDataset(name);
  toast('Saved "' + state.dataset.name + '" (' + state.dataset.count + ' samples).');
  S.ok('Saved dataset ' + state.dataset.name + ' · ' + (state.dataset.digest ? state.dataset.digest.short : '') + ' · ' + state.dataset.count + ' samples.');
  return id;
}

function requireModel() {
  if (!state.graph) { toast('Load or build an architecture first (Build panel).', 'error'); go('build'); return null; }
  if (!state.graphStats || !state.graphStats.valid) { toast('The architecture has errors: ' + (state.graphStats ? state.graphStats.errors.map((e) => e.message).join('; ') : 'unknown'), 'error'); go('build'); return null; }
  return { spec: state.graph, info: state.graphStats };
}

function requireDataset() {
  if (!state.dataset) { toast('Load or generate a dataset first (Data panel).', 'error'); go('data'); return null; }
  if (!state.dataset.count) { toast('The dataset is empty.', 'error'); go('data'); return null; }
  return state.dataset;
}

async function startRun(options) {
  if (state.training) { toast('A run is already in progress.', 'error'); return null; }
  const model = requireModel();
  if (!model) return null;
  const dataset = requireDataset();
  if (!dataset) return null;
  const taskKey = options.taskKey || state.taskKey || dataset.task || 'multiclass';
  const specShape = model.spec.input.shape;
  const dsShape = dataset.xShape;
  if (specShape.length !== dsShape.length || specShape.some((v, i) => v !== dsShape[i])) {
    const answer = await confirmShapeMismatch(specShape, dsShape);
    if (!answer) return null;
  }
  const session = new RunSession(ctx, {
    spec: JSON.parse(JSON.stringify(model.spec)),
    dataset,
    taskKey,
    config: options.config,
    name: options.name || (model.spec.name || 'model') + ' · ' + new Date().toISOString().slice(5, 16).replace('T', ' '),
    primaryMetric: options.primaryMetric || null
  });
  state.sessions = state.sessions || [];
  state.sessions.push(session);
  return session.start();
}

function confirmShapeMismatch(specShape, dsShape) {
  return new Promise((resolve) => {
    const handle = modal('Input shape mismatch', [
      el('p', { text: 'The architecture expects a per-sample input of [' + specShape.join(' × ') + '] but the dataset provides [' + dsShape.join(' × ') + '].' }),
      el('p', { class: 'dim small', text: 'Training will fail unless one of them changes. Continue anyway only if you know what you are doing.' })
    ], [
      el('button', { class: 'ghost', text: 'Cancel', onclick: () => { handle.close(); resolve(false); } }),
      el('button', { class: 'danger', text: 'Continue anyway', onclick: () => { handle.close(); resolve(true); } })
    ], { width: '480px' });
  });
}

async function finishRun(session, result) {
  const record = session.buildRunRecord(result);
  record.name = session.name;
  const id = await S.recordRun(record);
  state.selectedRunId = id;
  state.runIds.push(id);
  const headline = record.headline && record.headline.value !== null ? record.headline.metric + ' ' + Number(record.headline.value).toFixed(4) : 'no headline metric';
  S.ok('Run "' + record.name + '" finished: ' + record.epochsRun + ' epochs in ' + (record.durationMs / 1000).toFixed(1) + 's · ' + headline + ' · seed ' + record.seed + ' · data ' + (record.datasetHash || '').slice(0, 12));
  toast('Run finished — ' + headline, 'ok');
  return record;
}

function go(panelKey) {
  state.panel = panelKey;
  renderRail();
  renderPanel();
  S.notify('panel');
}

function panelInstance(key) {
  if (panelInstances[key]) return panelInstances[key];
  const factory = {
    build: createBuildPanel,
    data: createDataPanel,
    train: createTrainPanel,
    evaluate: createEvalPanel,
    compare: createComparePanel,
    export: createExportPanel,
    library: createLibraryPanel
  }[key];
  if (!factory) return null;
  const instance = factory(ctx);
  panelInstances[key] = instance;
  return instance;
}

function renderPanel() {
  if (!panelHost) return;
  const instance = panelInstance(state.panel);
  clear(panelHost);
  if (!instance) {
    panelHost.appendChild(el('div', { class: 'panel-wrap' }, el('p', { class: 'dim', text: 'Unknown panel.' })));
    return;
  }
  panelHost.appendChild(instance.root);
  if (instance.refresh) {
    try { instance.refresh(); } catch (e) { S.error('Panel ' + state.panel + ' failed to refresh: ' + e.message); }
  }
}

function renderRail() {
  if (!railNav) return;
  clear(railNav);
  for (const panel of PANELS) {
    const button = el('button', {
      class: state.panel === panel.key ? 'on' : '',
      onclick: () => go(panel.key)
    }, [
      el('span', { class: 'idx', text: panel.idx }),
      el('span', { text: panel.label }),
      el('span', { class: 'tag' + (railTag(panel.key).set ? ' set' : ''), text: railTag(panel.key).text })
    ]);
    railNav.appendChild(button);
  }
  const side = document.getElementById('railSide');
  if (side) renderRailSide(side);
}

function railTag(key) {
  if (key === 'build') return state.graphStats ? { text: compactNumber(state.graphStats.totalParams) + 'p', set: true } : { text: '—', set: false };
  if (key === 'data') return state.dataset ? { text: String(state.dataset.count), set: true } : { text: '—', set: false };
  if (key === 'train') return state.training ? { text: state.training.live.phase, set: true } : { text: String(state.runIds.length || state.runs.length || 0), set: state.runs.length > 0 };
  if (key === 'evaluate') return state.selectedRunId ? { text: 'ready', set: true } : { text: '—', set: false };
  if (key === 'compare') return state.compareIds.length ? { text: String(state.compareIds.length), set: true } : { text: '—', set: false };
  if (key === 'export') return state.selectedRunId ? { text: 'ready', set: true } : { text: '—', set: false };
  if (key === 'library') return { text: String(state.graphs.length + state.datasets.length + state.runs.length), set: false };
  return { text: '', set: false };
}

function renderRailSide(side) {
  clear(side);
  side.appendChild(railBlock('Current model', state.graph
    ? [miniRow(state.graph.name || 'Untitled model', state.graphStats ? state.graphStats.totalParams.toLocaleString() + ' p' : '', true)]
    : [el('div', { class: 'tiny faint', text: 'nothing loaded' })]));
  side.appendChild(railBlock('Current dataset', state.dataset
    ? [miniRow(state.dataset.name || 'Untitled dataset', state.dataset.count + ' n', true),
       el('div', { class: 'tiny faint', text: (state.dataset.modality || '') + ' · ' + (state.dataset.task || '') + ' · ' + (state.dataset.digest ? state.dataset.digest.short : '') })]
    : [el('div', { class: 'tiny faint', text: 'nothing loaded' })]));
  const runs = state.runs.slice(0, 6);
  side.appendChild(railBlock('Recent runs (' + state.runs.length + ')', runs.length
    ? runs.map((r) => miniRow(r.name || r.id, r.headline && r.headline.value !== null && r.headline.value !== undefined ? fmtHeadline(r.headline) : (r.epochsRun || 0) + ' ep', state.selectedRunId === r.id, () => { state.selectedRunId = r.id; S.notify('selectRun'); go('evaluate'); }))
    : [el('div', { class: 'tiny faint', text: 'no runs yet' })]));
  side.appendChild(railBlock('Storage', [
    el('div', { class: 'tiny faint', text: formatBytes(state.storage.usage || 0) + ' used' + (state.storage.quota ? ' of ' + formatBytes(state.storage.quota) : '') }),
    el('div', { class: 'row gap4 mt4' }, [
      el('button', { class: 'xs ghost', text: 'Log', onclick: () => setLogOpen(!logOpen) }),
      el('button', { class: 'xs ghost', text: 'Reset', onclick: () => confirmReset() })
    ])
  ]));
}

function fmtHeadline(headline) {
  const value = headline.value;
  if (value === null || value === undefined || !isFinite(value)) return '—';
  const abs = Math.abs(value);
  const text = abs >= 1000 ? value.toFixed(0) : abs >= 10 ? value.toFixed(2) : value.toFixed(4);
  return headline.metric.replace(/_/g, ' ') + ' ' + text;
}

function railBlock(title, children) {
  return el('div', { class: 'block' }, [el('h4', { text: title })].concat(children));
}

function miniRow(name, meta, active, onclick) {
  return el('div', { class: 'mini' + (active ? ' on' : ''), onclick: onclick || null }, [
    el('span', { class: 'nm', text: name }),
    el('span', { class: 'mt', text: meta })
  ]);
}

function confirmReset() {
  const handle = modal('Reset the workspace', [
    el('p', { text: 'This clears the current model and dataset from the editor. Saved library entries are kept.' }),
    el('p', { class: 'dim small', text: 'Use Library then Purge storage if you want the saved work removed as well.' })
  ], [
    el('button', { class: 'ghost', text: 'Cancel', onclick: () => handle.close() }),
    el('button', {
      class: 'danger', text: 'Clear editor', onclick: () => {
        handle.close();
        S.resetWorkspace();
        renderShellOnly();
        renderPanel();
      }
    })
  ], { width: '460px' });
}

function renderTopStats() {
  if (!topStats) return;
  clear(topStats);
  const stats = state.graphStats;
  topStats.appendChild(statCell('params', stats ? stats.totalParams.toLocaleString() : '—', stats && stats.totalParams > (state.budgetTarget || 1e6) ? 'warn' : ''));
  topStats.appendChild(statCell('macs', stats ? compactNumber(stats.macs) : '—'));
  topStats.appendChild(statCell('weights', stats ? formatBytes(stats.paramBytes) : '—'));
  topStats.appendChild(statCell('dataset', state.dataset ? state.dataset.count + ' × ' + state.dataset.xShape.join('×') : '—'));
  topStats.appendChild(statCell('task', state.taskKey || '—'));
  topStats.appendChild(statCell('graph', stats ? (stats.valid ? 'valid' : stats.errors.length + ' errors') : '—', stats && !stats.valid ? 'bad' : ''));
}

function statCell(label, value, tone) {
  return el('span', { class: 'cell' + (tone ? ' ' + tone : '') }, [label + ' ', el('b', { text: value })]);
}

function renderStatus() {
  if (!statusBar) return;
  clear(statusBar);
  const training = state.training;
  const dot = el('span', { class: 'dot' + (training ? (training.live.phase === 'running' ? ' run' : '') : '') });
  statusBar.appendChild(dot);
  let statusText = 'idle';
  if (training) {
    const live = training.live;
    if (live.phase === 'running' || live.phase === 'paused') {
      statusText = live.phase + ' · epoch ' + live.epoch + '/' + live.epochs + ' · ' + (live.percent * 100).toFixed(0) + '% · eta ' + live.eta.toFixed(0) + 's · lr ' + (live.lr || 0).toExponential(2) + ' · ' + (live.throughput || 0).toFixed(0) + ' samples/s';
    } else statusText = live.phase;
  }
  statusBar.appendChild(el('span', { class: 'seg', text: statusText }));
  const last = state.logs.length ? state.logs[state.logs.length - 1] : null;
  statusBar.appendChild(el('span', { class: 'seg grow ell', style: { color: last ? levelColor(last.level) : '' }, text: last ? last.message : 'Ready. Pick a preset to begin.' }));
  statusBar.appendChild(el('span', { class: 'seg', text: state.graphs.length + ' models · ' + state.datasets.length + ' datasets · ' + state.runs.length + ' runs' }));
  statusBar.appendChild(el('span', { class: 'seg', text: 'v' + TOOL_VERSION }));
}

function levelColor(level) {
  return level === 'error' ? '#e0938c' : level === 'warn' ? '#e0a33e' : level === 'ok' ? '#96c47f' : 'var(--muted)';
}

function setLogOpen(open) {
  logOpen = open;
  if (logDrawer) logDrawer.classList.toggle('open', open);
  renderLog();
}

function renderLog() {
  if (!logDrawer) return;
  const host = logDrawer.querySelector('.log');
  if (!host) return;
  clear(host);
  const entries = state.logs.slice(-400);
  for (const entry of entries) {
    host.appendChild(el('div', { class: 'line lvl-' + entry.level }, [
      el('span', { class: 't', text: new Date(entry.t).toTimeString().slice(0, 8) + '  ' }),
      entry.message
    ]));
  }
  host.scrollTop = host.scrollHeight;
}

function buildShell() {
  const app = el('div', { id: 'app' });

  const topbar = el('header', { id: 'topbar' }, [
    el('div', { class: 'mark' }, [
      markSvg(),
      el('span', { class: 'word', html: 'Rootism<em>Labs</em>' })
    ]),
    el('div', { id: 'projectBox' }, [
      el('input', {
        class: 'name', type: 'text', value: state.projectName || 'Untitled project',
        onchange: (event) => { state.projectName = event.target.value; }
      }),
      el('span', { class: 'tiny faint', id: 'dirtyFlag', text: '' })
    ]),
    topStats = el('div', { id: 'topStats' }),
    el('div', { class: 'spacer' }),
    el('button', { class: 'sm ghost', text: 'Save model', onclick: () => saveGraph() }),
    el('button', { class: 'sm ghost', text: 'Save dataset', onclick: () => saveDataset() }),
    el('button', { class: 'sm ghost', text: 'Log', onclick: () => setLogOpen(!logOpen) }),
    el('a', { class: 'btn sm ghost', href: 'index.html', text: 'Overview' })
  ]);

  railNav = el('nav');
  const rail = el('aside', { id: 'rail' }, [railNav, el('div', { id: 'railSide' })]);

  panelHost = el('div', { id: 'panelHost' });
  const main = el('main', { id: 'main' }, [panelHost]);

  statusBar = el('footer', { id: 'statusbar' });

  logDrawer = el('div', { id: 'logDrawer' }, [
    el('div', { class: 'row between', style: { padding: '5px 10px', borderBottom: '1px solid var(--border)' } }, [
      el('h4', { text: 'Session log' }),
      el('div', { class: 'row gap4' }, [
        el('button', { class: 'xs ghost', text: 'Copy', onclick: () => copyLog() }),
        el('button', { class: 'xs ghost', text: 'Download', onclick: () => downloadLog() }),
        el('button', { class: 'xs ghost', text: 'Clear', onclick: () => { state.logs = []; renderLog(); renderStatus(); } }),
        el('button', { class: 'xs ghost', text: 'Close', onclick: () => setLogOpen(false) })
      ])
    ]),
    el('div', { class: 'log', style: { flex: '1 1 auto', border: '0', borderRadius: '0' } })
  ]);

  toastHost = el('div', { id: 'toasts' });

  app.appendChild(topbar);
  app.appendChild(rail);
  app.appendChild(main);
  app.appendChild(statusBar);
  document.body.appendChild(app);
  document.body.appendChild(logDrawer);
  document.body.appendChild(toastHost);
}

function copyLog() {
  const text = state.logs.map((entry) => new Date(entry.t).toISOString() + ' [' + entry.level + '] ' + entry.message).join('\n');
  navigator.clipboard.writeText(text).then(() => toast('Log copied.'), () => toast('Copy failed.', 'error'));
}

function downloadLog() {
  const payload = {
    generatedAt: new Date().toISOString(),
    tool: 'Rootism Labs',
    version: TOOL_VERSION,
    model: state.graph ? { name: state.graph.name, params: state.graphStats ? state.graphStats.totalParams : null, spec: state.graph } : null,
    dataset: state.dataset ? { name: state.dataset.name, count: state.dataset.count, digest: state.dataset.digest } : null,
    runs: state.runIds,
    log: state.logs
  };
  download('rootism-session-' + Date.now().toString(36) + '.json', JSON.stringify(payload, null, 2), 'application/json');
}

function renderShellOnly() {
  renderRail();
  renderTopStats();
  renderStatus();
  renderLog();
  const flag = document.getElementById('dirtyFlag');
  if (flag) flag.textContent = (state.graphDirty || state.datasetDirty) ? 'unsaved' : '';
}

function bindKeys() {
  document.addEventListener('keydown', (event) => {
    const target = event.target;
    const typing = target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT' || target.isContentEditable);
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 's') {
      event.preventDefault();
      if (state.dataset && state.datasetDirty) saveDataset();
      else if (state.graph) saveGraph();
      return;
    }
    if (event.key === 'Escape') {
      if (logOpen) { setLogOpen(false); return; }
      const back = document.querySelector('.modal-back');
      if (back) back.remove();
      return;
    }
    if (typing || event.metaKey || event.ctrlKey || event.altKey) return;
    const index = parseInt(event.key, 10);
    if (index >= 1 && index <= PANELS.length) { go(PANELS[index - 1].key); return; }
    if (event.key === '`') { setLogOpen(!logOpen); return; }
    if ((event.key === 'Delete' || event.key === 'Backspace') && state.panel === 'build') {
      const instance = panelInstances.build;
      if (instance && instance.deleteSelected) { event.preventDefault(); instance.deleteSelected(); }
    }
  });
}

async function boot() {
  state.sessions = [];
  const bootMsg = document.getElementById('bootmsg');
  if (bootMsg) bootMsg.remove();
  setFavicon();
  buildShell();
  bindKeys();
  state.projectName = 'Untitled project';
  state.budgetTarget = 1000000;
  S.subscribe((reason) => {
    if (reason === 'log') { renderStatus(); renderLog(); return; }
    if (reason === 'progress') { renderStatus(); refreshActive(reason); return; }
    renderShellOnly();
    refreshActive(reason);
  });
  await S.initWorkspace();
  S.info('Workspace ready. Nothing is loaded — pick a model preset on the Build panel or a dataset preset on the Data panel.');
  go('build');
  renderShellOnly();
}

function refreshActive(reason) {
  const instance = panelInstances[state.panel];
  if (instance && instance.refresh) {
    try { instance.refresh(reason); } catch (e) { S.error('Panel "' + state.panel + '" failed: ' + e.message); }
  }
}

if (typeof document !== 'undefined') {
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
}
