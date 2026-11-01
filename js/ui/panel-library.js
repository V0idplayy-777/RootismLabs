import { el, clear, download, modal, copyText } from './dom.js';
import * as store from './store.js';
import * as S from './state.js';
import { formatBytes } from '../presets/metrics.js';
import { analyze, compactNumber } from '../core/model.js';
import { architectureHash } from './runner.js';
import { TOOL_VERSION } from '../export/card.js';

export function createLibraryPanel(ctx) {
  const root = el('div', { class: 'panel-wrap' });
  const toolbar = el('div', { class: 'toolbar' });
  const body = el('div');
  root.appendChild(toolbar);
  root.appendChild(body);

  function renderToolbar() {
    clear(toolbar);
    toolbar.appendChild(el('span', { class: 'small dim', text: 'Everything you save stays in this browser. Nothing is uploaded.' }));
    toolbar.appendChild(el('div', { class: 'spacer' }));
    toolbar.appendChild(el('button', { class: 'sm ghost', text: 'Export workspace index', onclick: () => exportIndex() }));
    toolbar.appendChild(el('button', { class: 'sm ghost', text: 'Refresh storage', onclick: async () => { ctx.state.storage = await store.storageEstimate(); await S.initWorkspace(); render(); } }));
    toolbar.appendChild(el('button', { class: 'sm danger', text: 'Purge everything', onclick: () => purgeAll() }));
  }

  function exportIndex() {
    const payload = {
      tool: 'Rootism Labs',
      version: TOOL_VERSION,
      generatedAt: new Date().toISOString(),
      models: ctx.state.graphs.map((g) => ({ id: g.id, name: g.name, taskKey: g.taskKey, preset: g.preset, totalParams: g.totalParams, macs: g.macs, inputShape: g.inputShape, createdAt: g.createdAt, spec: g.spec })),
      datasets: ctx.state.datasets.map((d) => ({ id: d.id, name: d.name, modality: d.modality, task: d.task, count: d.count, digest: d.digest, license: d.license, splits: d.splits, createdAt: d.createdAt })),
      runs: ctx.state.runs.map((r) => ({ id: r.id, name: r.name, taskKey: r.taskKey, seed: r.seed, modelName: r.modelName, datasetName: r.datasetName, datasetHash: r.datasetHash, architectureHash: r.architectureHash, totalParams: r.totalParams, macs: r.macs, epochsRun: r.epochsRun, durationMs: r.durationMs, headline: r.headline, best: r.best, config: r.config }))
    };
    download('rootism-workspace-index.json', JSON.stringify(payload, null, 2), 'application/json');
    ctx.ok('Workspace index exported (' + payload.models.length + ' models, ' + payload.datasets.length + ' datasets, ' + payload.runs.length + ' runs).');
  }

  function purgeAll() {
    const handle = modal('Purge all stored work', [
      el('p', { text: 'This deletes every saved model, dataset and run from this browser, and clears the editor.' }),
      el('p', { class: 'dim small', text: ctx.state.graphs.length + ' models · ' + ctx.state.datasets.length + ' datasets · ' + ctx.state.runs.length + ' runs · ' + formatBytes(ctx.state.storage.usage || 0) })
    ], [
      el('button', { class: 'ghost', text: 'Cancel', onclick: () => handle.close() }),
      el('button', {
        class: 'danger', text: 'Delete everything', onclick: async () => {
          handle.close();
          for (const name of ['graphs', 'datasets', 'runs', 'artifacts']) await store.clearStore(name);
          S.resetWorkspace();
          await S.initWorkspace();
          ctx.ok('Storage purged.');
          render();
        }
      })
    ], { width: '480px' });
  }

  function modelSection() {
    const rows = ctx.state.graphs.map((g) => el('div', { class: 'run-row', style: { gridTemplateColumns: '1fr auto auto auto auto' } }, [
      el('div', { class: 'nm' }, [
        el('div', { class: 'ell', text: g.name }),
        el('div', { class: 'tiny faint ell', text: (g.taskKey || 'no task') + (g.preset ? ' · preset ' + g.preset : ' · custom') + ' · ' + (g.inputShape ? '[' + g.inputShape.join('×') + ']' : '') + ' · ' + new Date(g.createdAt).toLocaleString() })
      ]),
      el('span', { class: 'mt', text: (g.totalParams || 0).toLocaleString() + ' p · ' + compactNumber(g.macs || 0) }),
      el('button', { class: 'xs', text: 'Open', onclick: async () => { await S.loadGraphRecord(g.id); ctx.state.graphStats = ctx.analyzeSpec(ctx.state.graph); ctx.toast('Opened "' + g.name + '".'); ctx.go('build'); } }),
      el('button', { class: 'xs ghost', text: 'JSON', onclick: () => download((g.name || 'model').toLowerCase().replace(/[^a-z0-9]+/g, '-') + '.json', JSON.stringify(g.spec, null, 2), 'application/json') }),
      el('button', { class: 'xs danger', text: 'Delete', onclick: () => confirmDelete('model', g.name, async () => { await S.deleteGraphRecord(g.id); render(); }) })
    ]));
    return section('Saved models', ctx.state.graphs.length, rows, [
      el('button', { class: 'xs ghost', text: 'Save the editor model', onclick: () => ctx.saveGraph() })
    ]);
  }

  function datasetSection() {
    const rows = ctx.state.datasets.map((d) => el('div', { class: 'run-row', style: { gridTemplateColumns: '1fr auto auto auto auto' } }, [
      el('div', { class: 'nm' }, [
        el('div', { class: 'ell', text: d.name }),
        el('div', { class: 'tiny faint ell', text: (d.modality || '') + ' · ' + (d.task || '') + ' · ' + (d.digest ? d.digest.short : '') + ' · ' + new Date(d.createdAt).toLocaleString() })
      ]),
      el('span', { class: 'mt', text: d.count + ' n' + (d.splits ? ' · ' + d.splits.train + '/' + d.splits.val + '/' + d.splits.test : '') }),
      el('span', { class: 'badge ' + (d.license && d.license.allowsCommercial ? 'green' : 'red'), text: d.license ? d.license.name : 'no license' }),
      el('button', { class: 'xs', text: 'Open', onclick: async () => { await S.loadDatasetRecord(d.id); ctx.touchDataset(); ctx.toast('Opened "' + d.name + '".'); ctx.go('data'); } }),
      el('button', { class: 'xs danger', text: 'Delete', onclick: () => confirmDelete('dataset', d.name, async () => { await S.deleteDatasetRecord(d.id); render(); }) })
    ]));
    return section('Saved datasets', ctx.state.datasets.length, rows, [
      el('button', { class: 'xs ghost', text: 'Save the editor dataset', onclick: () => ctx.saveDataset() })
    ]);
  }

  function runSection() {
    const rows = ctx.state.runs.map((r) => el('div', { class: 'run-row', style: { gridTemplateColumns: '1fr auto auto auto auto' } }, [
      el('div', { class: 'nm' }, [
        el('div', { class: 'ell', text: r.name || r.id }),
        el('div', { class: 'tiny faint ell', text: (r.modelName || '') + ' · ' + (r.datasetName || '') + ' · seed ' + r.seed + ' · ' + (r.epochsRun || 0) + ' epochs · ' + ((r.durationMs || 0) / 1000).toFixed(1) + 's · ' + new Date(r.createdAt || r.endedAt || Date.now()).toLocaleString() })
      ]),
      el('span', { class: 'mt', text: r.headline && r.headline.value !== null && r.headline.value !== undefined ? r.headline.metric + ' ' + Number(r.headline.value).toFixed(4) : '—' }),
      el('span', { class: 'badge plain', text: (r.totalParams || 0).toLocaleString() + ' p' }),
      el('button', { class: 'xs', text: 'Evaluate', onclick: () => { ctx.state.selectedRunId = r.id; ctx.notify('selectRun'); ctx.go('evaluate'); } }),
      el('button', { class: 'xs danger', text: 'Delete', onclick: () => confirmDelete('run', r.name || r.id, async () => { await S.deleteRunRecord(r.id); render(); }) })
    ]));
    return section('Runs', ctx.state.runs.length, rows, [
      el('button', { class: 'xs ghost', text: 'Compare', onclick: () => ctx.go('compare') })
    ]);
  }

  function section(title, count, rows, actions) {
    return el('div', { class: 'panel' }, [
      el('div', { class: 'panel-head' }, [
        el('h3', { text: title }),
        el('span', { class: 'tiny faint', text: String(count) }),
        el('div', { class: 'spacer' })
      ].concat(actions || [])),
      el('div', { class: 'panel-body flush' }, rows.length ? rows : [el('div', { class: 'tiny faint', style: { padding: '10px' }, text: 'Nothing saved yet.' })])
    ]);
  }

  function confirmDelete(kind, name, fn) {
    const handle = modal('Delete ' + kind, [
      el('p', { text: 'Delete the ' + kind + ' "' + name + '"? This cannot be undone.' })
    ], [
      el('button', { class: 'ghost', text: 'Cancel', onclick: () => handle.close() }),
      el('button', { class: 'danger', text: 'Delete', onclick: async () => { handle.close(); await fn(); ctx.ok('Deleted ' + kind + ' "' + name + '".'); } })
    ], { width: '440px' });
  }

  function storageSection() {
    const usage = ctx.state.storage.usage || 0;
    const quota = ctx.state.storage.quota || 0;
    const frac = quota ? Math.min(1, usage / quota) : 0;
    return el('div', { class: 'panel' }, [
      el('div', { class: 'panel-head' }, [el('h3', { text: 'Storage' }), el('div', { class: 'spacer' }), el('span', { class: 'mono tiny faint', text: formatBytes(usage) + (quota ? ' of ' + formatBytes(quota) : '') })]),
      el('div', { class: 'panel-body tight' }, [
        el('div', { class: 'progress' + (frac > 0.8 ? '' : ' teal') }, [el('i', { style: { width: (frac * 100).toFixed(2) + '%', background: frac > 0.8 ? 'var(--red)' : '' } })]),
        el('div', { class: 'tiny faint mt8', text: 'Stored with IndexedDB in this browser. Runs keep their weights, so any saved run can be re-evaluated or exported later without retraining.' }),
        el('div', { class: 'row gap4 mt8' }, [
          el('button', { class: 'xs ghost', text: 'Ask for persistent storage', onclick: async () => { const granted = await store.requestPersistence(); ctx.toast(granted ? 'Persistent storage granted.' : 'The browser declined persistent storage.', granted ? 'ok' : 'warn'); } }),
          el('button', { class: 'xs ghost', text: 'Copy storage summary', onclick: () => copyText(JSON.stringify({ usage, quota, models: ctx.state.graphs.length, datasets: ctx.state.datasets.length, runs: ctx.state.runs.length })).then(() => ctx.toast('Copied.')) })
        ])
      ])
    ]);
  }

  function aboutSection() {
    const spec = ctx.state.graph;
    const hash = spec ? architectureHash(spec) : null;
    return el('div', { class: 'panel' }, [
      el('div', { class: 'panel-head' }, [el('h3', { text: 'Current editor state' })]),
      el('div', { class: 'panel-body tight' }, [
        el('div', { class: 'kv' }, [
          el('dt', { text: 'model' }), el('dd', { text: spec ? spec.name + ' (' + (ctx.state.graphStats ? ctx.state.graphStats.totalParams.toLocaleString() : '?') + ' params, hash ' + hash + ')' : 'none' }),
          el('dt', { text: 'dataset' }), el('dd', { text: ctx.state.dataset ? ctx.state.dataset.name + ' (' + ctx.state.dataset.count + ' samples, hash ' + (ctx.state.dataset.digest ? ctx.state.dataset.digest.short : '?') + ')' : 'none' }),
          el('dt', { text: 'task' }), el('dd', { text: ctx.state.taskKey || '—' }),
          el('dt', { text: 'tool version' }), el('dd', { text: TOOL_VERSION })
        ]),
        el('div', { class: 'row gap4 mt8' }, [
          el('button', { class: 'xs ghost', text: 'Clear editor', onclick: () => { S.resetWorkspace(); render(); ctx.go('build'); } })
        ])
      ])
    ]);
  }

  function render() {
    renderToolbar();
    clear(body);
    body.appendChild(el('div', { class: 'col gap8' }, [
      aboutSection(),
      storageSection(),
      modelSection(),
      datasetSection(),
      runSection()
    ]));
  }

  function refresh(reason) {
    if (reason === 'progress' || reason === 'log') return;
    render();
  }

  render();
  return { root, refresh, render };
}
