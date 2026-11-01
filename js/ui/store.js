export const DB_NAME = 'rootism-labs';
export const DB_VERSION = 1;
export const STORES = ['settings', 'graphs', 'datasets', 'runs', 'artifacts'];

let dbPromise = null;
const memory = new Map(STORES.map((name) => [name, new Map()]));

export function storageAvailable() {
  return typeof indexedDB !== 'undefined' && indexedDB !== null;
}

export function openDb() {
  if (dbPromise) return dbPromise;
  if (!storageAvailable()) {
    dbPromise = Promise.resolve(null);
    return dbPromise;
  }
  dbPromise = new Promise((resolve) => {
    let request;
    try {
      request = indexedDB.open(DB_NAME, DB_VERSION);
    } catch (e) {
      resolve(null);
      return;
    }
    request.onupgradeneeded = () => {
      const db = request.result;
      for (const name of STORES) if (!db.objectStoreNames.contains(name)) db.createObjectStore(name);
    };
    request.onsuccess = () => {
      const db = request.result;
      db.onversionchange = () => { try { db.close(); } catch (e) { dbPromise = null; } };
      resolve(db);
    };
    request.onerror = () => resolve(null);
    request.onblocked = () => resolve(null);
  });
  return dbPromise;
}

function wrap(request) {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error('storage failure'));
  });
}

function txStore(store, mode) {
  return openDb().then((db) => {
    if (!db) return null;
    const transaction = db.transaction(store, mode);
    return transaction.objectStore(store);
  });
}

export function put(storeName, key, value) {
  return txStore(storeName, 'readwrite').then((store) => {
    if (!store) { memory.get(storeName).set(key, value); return key; }
    return wrap(store.put(value, key));
  });
}

export function get(storeName, key) {
  return txStore(storeName, 'readonly').then((store) => {
    if (!store) return memory.get(storeName).get(key);
    return wrap(store.get(key));
  });
}

export function del(storeName, key) {
  return txStore(storeName, 'readwrite').then((store) => {
    if (!store) { memory.get(storeName).delete(key); return undefined; }
    return wrap(store.delete(key));
  });
}

export function keys(storeName) {
  return txStore(storeName, 'readonly').then((store) => {
    if (!store) return Array.from(memory.get(storeName).keys());
    return wrap(store.getAllKeys());
  });
}

export function all(storeName) {
  return txStore(storeName, 'readonly').then((store) => {
    if (!store) return Array.from(memory.get(storeName).values());
    return wrap(store.getAll());
  });
}

export function clearStore(storeName) {
  return txStore(storeName, 'readwrite').then((store) => {
    if (!store) { memory.get(storeName).clear(); return undefined; }
    return wrap(store.clear());
  });
}

export function count(storeName) {
  return txStore(storeName, 'readonly').then((store) => {
    if (!store) return memory.get(storeName).size;
    return wrap(store.count());
  });
}

export async function settingsGet(key, fallback) {
  const value = await get('settings', key);
  return value === undefined || value === null ? fallback : value;
}

export async function settingsSet(key, value) {
  await put('settings', key, value);
  return value;
}

export async function storageEstimate() {
  if (navigator.storage && navigator.storage.estimate) {
    try {
      const estimate = await navigator.storage.estimate();
      return { usage: estimate.usage || 0, quota: estimate.quota || 0 };
    } catch (e) {
      return { usage: 0, quota: 0 };
    }
  }
  return { usage: 0, quota: 0 };
}

export async function requestPersistence() {
  if (navigator.storage && navigator.storage.persist) {
    try { return await navigator.storage.persist(); } catch (e) { return false; }
  }
  return false;
}

function cloneValue(value) {
  if (value === null || value === undefined) return value;
  if (ArrayBuffer.isView(value)) return value;
  if (Array.isArray(value)) return value.map(cloneValue);
  if (typeof value === 'object') {
    const out = {};
    for (const key of Object.keys(value)) out[key] = cloneValue(value[key]);
    return out;
  }
  return value;
}

export function serializableDataset(ds) {
  const out = {};
  for (const key of Object.keys(ds)) out[key] = cloneValue(ds[key]);
  return out;
}

export function reviveDataset(stored) {
  return stored ? cloneValue(stored) : null;
}

export async function saveDataset(ds) {
  await put('datasets', ds.id, serializableDataset(ds));
  return ds.id;
}

export async function loadDataset(id) {
  return reviveDataset(await get('datasets', id));
}

export async function listDatasetSummaries() {
  const records = await all('datasets');
  return records.map((record) => ({
    id: record.id,
    name: record.name,
    modality: record.modality,
    task: record.task,
    count: record.count,
    labels: record.labels ? record.labels.length : 0,
    xShape: record.xShape,
    digest: record.digest,
    license: record.license,
    presetKey: record.presetKey,
    provenance: record.provenance,
    createdAt: record.createdAt,
    splits: record.splits ? { train: record.splits.train ? record.splits.train.length : 0, val: record.splits.val ? record.splits.val.length : 0, test: record.splits.test ? record.splits.test.length : 0 } : null,
    stats: record.stats ? { samples: record.stats.samples, classCounts: record.stats.classCounts, balance: record.stats.balance, bytes: record.stats.bytes } : null
  }));
}

export async function deleteDataset(id) {
  await del('datasets', id);
}

export function trimRunForStorage(run) {
  const out = Object.assign({}, run);
  const weights = out.weights || null;
  delete out.weights;
  if (out.logs && out.logs.length > 4000) out.logs = out.logs.slice(-4000);
  if (out.metrics) {
    out.metrics = {};
    for (const split of Object.keys(run.metrics || {})) {
      const value = run.metrics[split];
      if (!value) { out.metrics[split] = value; continue; }
      const trimmed = Object.assign({}, value);
      for (const key of Object.keys(trimmed)) {
        const item = trimmed[key];
        if (Array.isArray(item)) {
          trimmed[key] = item.length > 4096 ? item.slice(0, 4096) : item;
        } else if (item && typeof item === 'object' && !(item instanceof Float32Array)) {
          const shallow = {};
          for (const inner of Object.keys(item)) {
            const v = item[inner];
            if (Array.isArray(v) && v.length > 4096) shallow[inner] = v.slice(0, 4096);
            else if (v && typeof v === 'object' && !(v instanceof Float32Array)) shallow[inner] = JSON.parse(JSON.stringify(v));
            else shallow[inner] = v;
          }
          trimmed[key] = shallow;
        }
      }
      out.metrics[split] = trimmed;
    }
  }
  if (out.samples) out.samples = out.samples.slice(0, 200);
  delete out.trainer;
  delete out.model;
  const cloned = JSON.parse(JSON.stringify(out));
  if (weights) cloned.weights = weights.map((w) => ({ name: w.name, shape: w.shape, data: w.data instanceof Float32Array ? w.data : Float32Array.from(w.data || []) }));
  return cloned;
}

export async function saveRun(run) {
  const trimmed = trimRunForStorage(run);
  await put('runs', trimmed.id || (run.id = 'run_' + Date.now().toString(36)), trimmed);
  return run.id;
}

export async function listRunSummaries() {
  const records = await all('runs');
  return records.map((record) => ({
    id: record.id,
    name: record.name,
    taskKey: record.taskKey,
    monitor: record.monitor,
    seed: record.seed,
    epochsRun: record.epochsRun,
    durationMs: record.durationMs,
    createdAt: record.createdAt || record.endedAt,
    modelName: record.modelName,
    datasetName: record.datasetName,
    params: record.modelSummary ? record.modelSummary.totalParams : null,
    macs: record.modelSummary ? record.modelSummary.macs : null,
    headline: record.headline,
    best: record.best,
    splits: record.metrics ? Object.keys(record.metrics) : [],
    datasetHash: record.datasetHash,
    architectureHash: record.architectureHash,
    hasWeights: !!(record.weights && record.weights.length),
    weightBytes: record.weights ? record.weights.reduce((a, w) => a + (w.data ? w.data.length * 4 : 0), 0) : 0
  }));
}

export async function loadRun(id) {
  return get('runs', id);
}

export async function deleteRun(id) {
  await del('runs', id);
  await del('artifacts', 'art_' + id);
}

export async function saveGraph(id, value) {
  await put('graphs', id, value);
  return id;
}

export async function listGraphs() {
  return all('graphs');
}

export async function loadGraph(id) {
  return get('graphs', id);
}

export async function deleteGraph(id) {
  await del('graphs', id);
}

export async function saveArtifacts(runId, payload) {
  await put('artifacts', 'art_' + runId, payload);
}

export async function loadArtifacts(runId) {
  return get('artifacts', 'art_' + runId);
}
