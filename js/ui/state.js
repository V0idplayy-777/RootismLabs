import * as store from './store.js';

const listeners = new Set();

export const state = {
  ready: false,
  panel: 'build',
  graphs: [],
  datasets: [],
  runs: [],
  graph: null,
  graphId: null,
  graphDirty: false,
  dataset: null,
  datasetId: null,
  datasetDirty: false,
  taskKey: null,
  trainConfig: null,
  runIds: [],
  selectedRunId: null,
  compareIds: [],
  training: null,
  logs: [],
  storage: { usage: 0, quota: 0 }
};

export function subscribe(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

export function notify(reason) {
  for (const fn of listeners) {
    try {
      fn(reason || 'change', state);
    } catch (e) {
      console.error(e);
    }
  }
}

export function patch(values, reason) {
  Object.assign(state, values);
  notify(reason);
}

export function log(level, message, detail) {
  const entry = { t: Date.now(), level, message, detail: detail === undefined ? null : detail };
  state.logs.push(entry);
  if (state.logs.length > 2000) state.logs.splice(0, state.logs.length - 2000);
  notify('log');
  return entry;
}

export const info = (message, detail) => log('info', message, detail);
export const warn = (message, detail) => log('warn', message, detail);
export const error = (message, detail) => log('error', message, detail);
export const ok = (message, detail) => log('ok', message, detail);

export async function initWorkspace() {
  state.graphs = (await store.listGraphs()).map(normalizeGraphRecord).sort(byNewest);
  const datasetSummaries = await store.listDatasetSummaries();
  state.datasets = datasetSummaries.sort(byNewest);
  state.runs = (await store.listRunSummaries()).sort(byNewest);
  state.storage = await store.storageEstimate();
  state.ready = true;
  notify('init');
}

function byNewest(a, b) {
  return (b.createdAt || b.endedAt || 0) - (a.createdAt || a.endedAt || 0);
}

function normalizeGraphRecord(record) {
  return {
    id: record.id,
    name: record.name || 'Untitled model',
    createdAt: record.createdAt || Date.now(),
    taskKey: record.taskKey || null,
    preset: record.preset || null,
    forkedFrom: record.forkedFrom || null,
    budget: record.budget || null,
    totalParams: record.totalParams || 0,
    macs: record.macs || 0,
    inputShape: record.inputShape || null,
    spec: record.spec
  };
}

export function newId(prefix) {
  return (prefix || 'id') + '_' + Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
}

export async function saveCurrentGraph(nameOverride) {
  if (!state.graph) return null;
  const id = state.graphId || newId('graph');
  const record = {
    id,
    name: nameOverride || state.graph.name || 'Untitled model',
    createdAt: state.graphSavedAt || Date.now(),
    taskKey: state.taskKey,
    preset: state.graph.meta && state.graph.meta.preset ? state.graph.meta.preset : null,
    forkedFrom: state.graph.meta && state.graph.meta.forkedFrom ? state.graph.meta.forkedFrom : null,
    budget: state.graph.meta && state.graph.meta.budget ? state.graph.meta.budget : null,
    totalParams: state.graphStats ? state.graphStats.totalParams : 0,
    macs: state.graphStats ? state.graphStats.macs : 0,
    inputShape: state.graph.input ? state.graph.input.shape : null,
    spec: JSON.parse(JSON.stringify(state.graph))
  };
  await store.saveGraph(id, record);
  state.graphId = id;
  state.graphDirty = false;
  state.graphSavedAt = record.createdAt;
  state.graphs = (await store.listGraphs()).map(normalizeGraphRecord).sort(byNewest);
  notify('graphs');
  return id;
}

export async function loadGraphRecord(id) {
  const record = await store.loadGraph(id);
  if (!record) return null;
  state.graph = JSON.parse(JSON.stringify(record.spec));
  state.graphId = record.id;
  state.graphDirty = false;
  state.graphSavedAt = record.createdAt;
  if (record.taskKey) state.taskKey = record.taskKey;
  notify('graph');
  return record;
}

export async function deleteGraphRecord(id) {
  await store.deleteGraph(id);
  state.graphs = state.graphs.filter((g) => g.id !== id);
  if (state.graphId === id) { state.graphId = null; state.graphDirty = true; }
  notify('graphs');
}

export async function saveCurrentDataset(nameOverride) {
  if (!state.dataset) return null;
  if (nameOverride) state.dataset.name = nameOverride;
  await store.saveDataset(state.dataset);
  state.datasetId = state.dataset.id;
  state.datasetDirty = false;
  state.datasets = (await store.listDatasetSummaries()).sort(byNewest);
  notify('datasets');
  return state.dataset.id;
}

export async function loadDatasetRecord(id) {
  const ds = await store.loadDataset(id);
  if (!ds) return null;
  state.dataset = ds;
  state.datasetId = ds.id;
  state.datasetDirty = false;
  if (ds.task) state.taskKey = ds.task;
  notify('dataset');
  return ds;
}

export async function deleteDatasetRecord(id) {
  await store.deleteDataset(id);
  state.datasets = state.datasets.filter((d) => d.id !== id);
  if (state.datasetId === id) { state.datasetId = null; state.dataset = null; }
  notify('datasets');
}

export async function recordRun(run) {
  if (!run.id) run.id = newId('run');
  run.createdAt = run.createdAt || Date.now();
  await store.saveRun(run);
  state.runs = (await store.listRunSummaries()).sort(byNewest);
  if (!state.runIds.includes(run.id)) state.runIds.push(run.id);
  notify('runs');
  return run.id;
}

export async function deleteRunRecord(id) {
  await store.deleteRun(id);
  state.runs = state.runs.filter((r) => r.id !== id);
  state.runIds = state.runIds.filter((r) => r !== id);
  state.compareIds = state.compareIds.filter((r) => r !== id);
  if (state.selectedRunId === id) state.selectedRunId = state.runs.length ? state.runs[0].id : null;
  notify('runs');
}

export function selectRun(id) {
  state.selectedRunId = id;
  notify('selectRun');
}

export function toggleCompare(id) {
  const index = state.compareIds.indexOf(id);
  if (index >= 0) state.compareIds.splice(index, 1);
  else state.compareIds.push(id);
  notify('compare');
}

export function setPanel(panel) {
  state.panel = panel;
  notify('panel');
}

export function setGraph(spec, options) {
  const opts = options || {};
  state.graph = spec;
  state.graphDirty = opts.keepId ? state.graphDirty : true;
  if (!opts.keepId) { state.graphId = opts.id || null; state.graphSavedAt = null; }
  notify('graph');
}

export function setDataset(ds, options) {
  const opts = options || {};
  state.dataset = ds;
  state.datasetDirty = opts.keepId ? state.datasetDirty : true;
  if (!opts.keepId) { state.datasetId = ds ? ds.id : null; }
  if (ds && ds.task) state.taskKey = ds.task;
  notify('dataset');
}

export function resetWorkspace() {
  state.graph = null;
  state.graphId = null;
  state.graphDirty = false;
  state.dataset = null;
  state.datasetId = null;
  state.datasetDirty = false;
  state.taskKey = null;
  state.trainConfig = null;
  state.runIds = [];
  state.selectedRunId = null;
  state.compareIds = [];
  state.logs = [];
  notify('reset');
}
