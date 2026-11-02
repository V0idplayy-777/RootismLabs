import { el, clear, attachTooltip, download, pickFiles, readFileText, modal } from './dom.js';
import { GraphEditor, fmtShape, groupColors, summarizeCfg } from './graph-editor.js';
import { LAYERS, LAYER_GROUPS, layerFields, defaultConfig, isLayerType } from '../core/layers.js';
import { analyze, blankGraph, compactNumber } from '../core/model.js';
import { PRESET_MODEL_LIST, PRESET_FAMILIES, modelPresetByKey, buildPresetGraph, scaleToFit, budgetPosition } from '../presets/models.js';
import { TASKS, taskFor } from '../core/tasks.js';
import { defaultConfig as defaultTrainConfig } from './train-config.js';
import { graphForDataset } from '../presets/from-data.js';
import { formatBytes } from '../presets/metrics.js';

export const HEADS = [
  { value: 'linear', label: 'linear', note: 'Raw network output. Use for regression, masks and coordinate fields.' },
  { value: 'softmax', label: 'softmax', note: 'Mutually exclusive classes. Exported graphs end in linear; the head is applied by the runtime.' },
  { value: 'sigmoid', label: 'sigmoid', note: 'Independent probabilities. Binary and multi-label outputs.' },
  { value: 'mask', label: 'mask', note: 'Sigmoid mask over a spectral or spatial grid.' },
  { value: 'heatmap', label: 'heatmap', note: 'Sigmoid heatmap, one channel, spatial output.' },
  { value: 'det', label: 'detection', note: 'Objectness sigmoid plus box coordinates on the remaining channels.' },
  { value: 'embedding', label: 'embedding', note: 'L2-normalised vector for retrieval and similarity.' }
];

let galleryFamily = 'all';

export function createBuildPanel(ctx) {
  const root = el('div', { class: 'panel-wrap' });
  let editor = null;
  let galleryOpen = false;

  const toolbar = el('div', { class: 'toolbar' });
  const body = el('div', {});
  root.appendChild(toolbar);
  root.appendChild(body);

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

  function renderToolbar() {
    const spec = ctx.state.graph;
    clear(toolbar);
    if (!spec) {
      if (ctx.state.dataset) toolbar.appendChild(el('button', { text: 'Model for loaded data', onclick: () => buildForData() }));
      toolbar.appendChild(el('span', { class: 'small dim', text: 'No architecture loaded. Start from a preset, a blank graph, or a saved model.' }));
      toolbar.appendChild(el('div', { class: 'spacer' }));
      toolbar.appendChild(el('button', { text: 'Blank graph', onclick: () => startBlank() }));
      toolbar.appendChild(el('button', { text: 'Load .json spec', onclick: () => loadSpecFile() }));
      return;
    }
    const nameInput = el('input', {
      type: 'text', value: spec.name || 'Untitled model', style: { width: '190px', fontWeight: '600' },
      onchange: (event) => { spec.name = event.target.value; ctx.touchGraph(); }
    });
    toolbar.appendChild(nameInput);
    toolbar.appendChild(el('div', { class: 'vsep' }));
    if (ctx.state.dataset) toolbar.appendChild(el('button', { class: 'sm', text: 'Model for loaded data', onclick: () => buildForData() }));
    toolbar.appendChild(el('button', { class: 'sm', text: 'Preset library', onclick: () => openGallery() }));
    toolbar.appendChild(el('button', { class: 'sm', text: 'Add layer', onclick: () => openPalette() }));
    toolbar.appendChild(el('button', { class: 'sm ghost', text: 'Auto layout', onclick: () => { if (editor) { editor.autoLayout(); ctx.touchGraph(); } } }));
    toolbar.appendChild(el('button', { class: 'sm ghost', text: 'Fit', onclick: () => editor && editor.fit() }));
    toolbar.appendChild(el('div', { class: 'vsep' }));
    toolbar.appendChild(el('button', { class: 'sm ghost', text: 'Duplicate', onclick: () => duplicateSelected() }));
    toolbar.appendChild(el('button', { class: 'sm ghost', text: 'Delete', onclick: () => deleteSelected() }));
    toolbar.appendChild(el('div', { class: 'spacer' }));
    const stats = ctx.state.graphStats;
    if (stats) {
      toolbar.appendChild(el('span', { class: 'mono small', style: { color: stats.valid ? 'var(--text-dim)' : '#e0938c' }, text: stats.valid ? stats.totalParams.toLocaleString() + ' params · ' + compactNumber(stats.macs) + ' MACs' : stats.errors.length + ' problem(s)' }));
    }
    toolbar.appendChild(el('button', { class: 'sm ghost', text: 'Export spec', onclick: () => download((spec.name || 'model').toLowerCase().replace(/[^a-z0-9]+/g, '-') + '.json', JSON.stringify(spec, null, 2), 'application/json') }));
    toolbar.appendChild(el('button', { class: 'sm primary', text: 'Save model', onclick: () => ctx.saveGraph() }));
  }

  function buildForData() {
    try {
      const spec = graphForDataset(ctx.state.dataset);
      if (ctx.state.graph && !confirm('Replace the current architecture with a new untrained data-shaped baseline? Save it first if needed.')) return;
      ctx.state.taskKey = ctx.state.dataset.task;
      const config = defaultTrainConfig(ctx.state.dataset.task);
      config.epochs = 60; config.batchSize = 16; config.seed = 7;
      config.optimizer.name = 'adam'; config.optimizer.lr = 0.01;
      config.schedule.name = 'constant'; config.earlyStopping.patience = 12;
      ctx.state.trainConfig = config;
      ctx.setSpec(spec);
      galleryOpen = false;
      ctx.ok('Created an untrained baseline matching the dataset. Validate it on representative held-out data.');
      render();
    } catch (err) { ctx.error(err.message); }
  }

  function startBlank() {
    const spec = blankGraph([640], 'x');
    spec.name = 'Untitled model';
    spec.meta = { task: 'multiclass' };
    ctx.setSpec(spec);
    ctx.state.taskKey = 'multiclass';
    ctx.notify();
  }

  async function loadSpecFile() {
    const files = await pickFiles('.json,application/json', false);
    if (!files.length) return;
    try {
      const text = await readFileText(files[0]);
      const spec = JSON.parse(text);
      if (!spec || !Array.isArray(spec.nodes) || !Array.isArray(spec.edges)) throw new Error('not a Rootism Labs model spec');
      if (!spec.input || !Array.isArray(spec.input.shape)) spec.input = { name: 'x', shape: [640], dtype: 'float32' };
      for (const node of spec.nodes) if (!node.pos) node.pos = { x: 60, y: 160 };
      spec.name = spec.name || files[0].name.replace(/\.json$/i, '');
      ctx.setSpec(spec);
      if (spec.meta && spec.meta.task) ctx.state.taskKey = spec.meta.task;
      ctx.ok('Loaded ' + spec.name + ' from ' + files[0].name);
      ctx.notify();
    } catch (e) {
      ctx.error('Could not read the spec: ' + e.message);
    }
  }

  function openGallery() {
    galleryOpen = true;
    render();
  }

  function closeGallery() {
    galleryOpen = false;
    render();
  }

  function openPalette() {
    const list = el('div', { class: 'layer-palette' });
    for (const group of LAYER_GROUPS) {
      const types = Object.keys(LAYERS).filter((k) => LAYERS[k].group === group.key);
      if (!types.length) continue;
      const box = el('div', { class: 'palette-group' }, [el('h4', { text: group.label })]);
      for (const type of types) {
        const color = groupColors().find((g) => g.key === group.key);
        box.appendChild(el('div', {
          class: 'palette-item',
          onclick: () => { addNode(type); handle.close(); }
        }, [
          el('span', { class: 'sw', style: { background: color ? color.color : '#8b9298' } }),
          el('span', { text: LAYERS[type].label }),
          el('span', { class: 'k', text: type })
        ]));
      }
      list.appendChild(box);
    }
    const handle = modal('Add a layer', [list, el('p', { class: 'small dim mt8', text: 'The new layer is appended after the current output. Drag from its right port to rewire.' })], null, { width: '520px' });
  }

  function addNode(type) {
    const spec = ctx.state.graph;
    if (!spec || !isLayerType(type)) return;
    let n = spec.nodes.filter((x) => x.type === type).length;
    let id = type + '_' + n;
    while (spec.nodes.some((x) => x.id === id)) { n++; id = type + '_' + n; }
    const last = spec.nodes.length ? spec.nodes[spec.nodes.length - 1] : null;
    const pos = last ? { x: (last.pos ? last.pos.x : 40) + 60, y: (last.pos ? last.pos.y : 160) + 90 } : { x: 60, y: 160 };
    const node = { id, type, cfg: defaultConfig(type), pos };
    spec.nodes.push(node);
    const source = spec.output || (last ? last.id : 'in');
    spec.edges.push({ from: source, to: id, toPort: 0 });
    spec.output = id;
    if (editor) editor.selectedId = id;
    ctx.setSpec(spec, { keepId: true });
    ctx.info('Added ' + LAYERS[type].label + ' (' + id + ')');
  }

  function deleteSelected() {
    const spec = ctx.state.graph;
    const id = editor ? editor.selectedId : null;
    if (!spec || !id) return;
    const node = spec.nodes.find((n) => n.id === id);
    if (!node) return;
    if (node.type === 'input') { ctx.warn('The input node cannot be deleted.'); return; }
    const incoming = spec.edges.filter((e) => e.to === id);
    const outgoing = spec.edges.filter((e) => e.from === id);
    spec.nodes = spec.nodes.filter((n) => n.id !== id);
    spec.edges = spec.edges.filter((e) => e.from !== id && e.to !== id);
    for (const out of outgoing) {
      const bridge = incoming[0];
      if (bridge) spec.edges.push({ from: bridge.from, to: out.to, toPort: out.toPort || 0 });
    }
    if (spec.output === id) spec.output = incoming.length ? incoming[0].from : (spec.nodes.length ? spec.nodes[spec.nodes.length - 1].id : 'in');
    if (editor) editor.selectedId = null;
    ctx.setSpec(spec, { keepId: true });
    ctx.info('Removed ' + id + ' and rewired around it.');
  }

  function duplicateSelected() {
    const spec = ctx.state.graph;
    const id = editor ? editor.selectedId : null;
    if (!spec || !id) return;
    const node = spec.nodes.find((n) => n.id === id);
    if (!node || node.type === 'input') return;
    let n = 0;
    let newId = node.type + '_copy' + (n || '');
    while (spec.nodes.some((x) => x.id === newId)) { n++; newId = node.type + '_copy' + n; }
    const copy = { id: newId, type: node.type, cfg: JSON.parse(JSON.stringify(node.cfg)), pos: { x: (node.pos ? node.pos.x : 40) + 24, y: (node.pos ? node.pos.y : 160) + 96 } };
    const index = spec.nodes.indexOf(node);
    spec.nodes.splice(index + 1, 0, copy);
    const sources = spec.edges.filter((e) => e.to === node.id).map((e) => ({ from: e.from, toPort: e.toPort || 0 }));
    for (const s of sources) spec.edges.push({ from: s.from, to: newId, toPort: s.toPort });
    for (const e of spec.edges) if (e.from === node.id) e.from = newId;
    if (spec.output === node.id) spec.output = newId;
    if (editor) editor.selectedId = newId;
    ctx.setSpec(spec, { keepId: true });
    ctx.info('Duplicated ' + id + ' as ' + newId);
  }

  function renderGallery() {
    let family = galleryFamily;
    const chips = el('div', { class: 'pill-row' });
    const families = [{ key: 'all', label: 'All' }].concat(PRESET_FAMILIES);
    for (const f of families) {
      chips.appendChild(el('button', {
        class: 'sm' + (family === f.key ? ' primary' : ' ghost'),
        text: f.label,
        onclick: () => { galleryFamily = f.key; render(); }
      }));
    }
    const list = PRESET_MODEL_LIST.filter((p) => family === 'all' || p.family === family);
    const grid = el('div', { class: 'preset-grid' });
    for (const preset of list) grid.appendChild(presetCard(preset));
    return el('div', { class: 'col gap8' }, [
      el('div', { class: 'row between wrap' }, [
        el('div', {}, [
          el('h2', { text: 'Model presets' }),
          el('div', { class: 'small dim', text: list.length + ' of ' + PRESET_MODEL_LIST.length + ' architectures. Each one is a graph you can fork and edit.' })
        ]),
        el('div', { class: 'row' }, [
          ctx.state.dataset ? el('button', { text: 'Model for loaded data', onclick: () => buildForData() }) : null,
          el('button', { text: 'Blank graph', onclick: () => startBlank() }),
          el('button', { text: 'Load .json spec', onclick: () => loadSpecFile() }),
          ctx.state.graph ? el('button', { class: 'ghost', text: 'Back to editor', onclick: () => closeGallery() }) : null
        ])
      ]),
      chips,
      grid
    ]);
  }

  function presetCard(preset) {
    const spec = buildPresetGraph(preset);
    const info = analyzeSpec(spec);
    const active = ctx.state.graph && ctx.state.graph.meta && ctx.state.graph.meta.presetKey === preset.key;
    const card = el('div', { class: 'preset-card' + (active ? ' on' : ''), onclick: () => choosePreset(preset) }, [
      el('span', { class: 'strip', style: { background: familyColor(preset.family) } }),
      el('div', { class: 'ttl' }, [
        el('h3', { text: preset.name }),
        el('span', { class: 'fam', text: preset.family })
      ]),
      el('div', { class: 'tag', text: preset.tagline }),
      el('div', { class: 'foot' }, [
        el('span', { class: 'p', text: info.totalParams.toLocaleString() + ' p' }),
        el('span', { text: compactNumber(info.macs) + ' MACs' }),
        el('span', { text: 'in ' + fmtShape(spec.input.shape) }),
        el('span', { text: 'out ' + fmtShape(info.outputShape) }),
        el('span', { class: 'badge teal', text: preset.task })
      ])
    ]);
    attachTooltip(card, () => '<b>' + preset.name + '</b><br>' + preset.description + '<br><br><span style="color:#8b9298">head</span> ' + preset.head + ' · <span style="color:#8b9298">task</span> ' + preset.task + '<br><span style="color:#8b9298">dataset</span> ' + preset.dataset + '<br><span style="color:#8b9298">budget</span> ' + preset.budget.target.toLocaleString() + ' params (range ' + preset.budget.min.toLocaleString() + '–' + preset.budget.max.toLocaleString() + ')<br><span style="color:#8b9298">reference</span> ' + preset.reference.params.map((v) => v.toLocaleString()).join('–') + ' params');
    return card;
  }

  function familyColor(family) {
    const map = { text: '#e0a33e', audio: '#4fa3a5', vision: '#5b87b5', sensor: '#8a7bb0', series: '#6aa84f', wildcard: '#b0765f' };
    return map[family] || '#8b9298';
  }

  function choosePreset(preset) {
    const spec = buildPresetGraph(preset);
    spec.name = preset.name;
    spec.meta = Object.assign({}, spec.meta, { presetKey: preset.key, budget: preset.budget, reference: preset.reference, datasetKey: preset.dataset, description: preset.description, tagline: preset.tagline });
    ctx.state.taskKey = preset.task;
    ctx.state.presetTrain = JSON.parse(JSON.stringify(preset.train || {}));
    ctx.setSpec(spec);
    galleryOpen = false;
    ctx.ok('Loaded preset ' + preset.name + ' — ' + analyze(spec).totalParams.toLocaleString() + ' parameters.');
    if (editor) editor.fit();
  }

  function renderEditor() {
    const spec = ctx.state.graph;
    const info = ctx.state.graphStats;
    const canvasHost = el('div', { style: { position: 'relative', border: '1px solid var(--border)', borderRadius: '3px', background: '#15181a', overflow: 'hidden' } });
    const canvas = el('canvas', { style: { width: '100%', height: '52vh', minHeight: '360px', display: 'block' } });
    canvasHost.appendChild(canvas);
    canvasHost.appendChild(el('div', { class: 'tiny faint', style: { position: 'absolute', left: '8px', top: '6px', pointerEvents: 'none' }, text: 'drag nodes · drag from a port to rewire · wheel to zoom · right-click for node actions' }));

    editor = new GraphEditor({
      canvas,
      spec,
      info,
      selectedId: editor ? editor.selectedId : null,
      onSelect: () => renderInspector(),
      onChange: (next, nextInfo, positionsOnly) => {
        ctx.state.graph = next;
        ctx.state.graphStats = nextInfo;
        ctx.state.graphDirty = true;
        if (!positionsOnly) ctx.notify('graph');
        else { renderSummary(); renderLayerTable(); }
      },
      contextMenu: (x, y, node) => showNodeMenu(x, y, node)
    });
    editor.onInspect = (node) => renderInspector();
    requestAnimationFrame(() => { editor.fit(); renderSummary(); renderLayerTable(); renderInspector(); });

    const right = el('div', { class: 'col gap8' }, [
      inspectorPanel(),
      summaryPanel(),
      budgetPanel()
    ]);

    return el('div', { class: 'split editor' }, [
      el('div', { class: 'col gap8' }, [canvasHost, layerTable()]),
      right
    ]);
  }

  function showNodeMenu(x, y, node) {
    const existing = document.getElementById('nodeMenu');
    if (existing) existing.remove();
    const items = [];
    if (node) {
      items.push({ label: 'Set as output', fn: () => { ctx.state.graph.output = node.id; ctx.touchGraph(); } });
      items.push({ label: 'Duplicate', fn: () => duplicateSelected() });
      items.push({ label: 'Disconnect inputs', fn: () => { ctx.state.graph.edges = ctx.state.graph.edges.filter((e) => e.to !== node.id); ctx.touchGraph(); } });
      if (node.type !== 'input') items.push({ label: 'Delete node', fn: () => deleteSelected(), tone: 'danger' });
    } else {
      items.push({ label: 'Add layer…', fn: () => openPalette() });
      items.push({ label: 'Auto layout', fn: () => { editor.autoLayout(); ctx.touchGraph(); } });
      items.push({ label: 'Fit to view', fn: () => editor.fit() });
    }
    const menu = el('div', {
      id: 'nodeMenu',
      style: {
        position: 'fixed', left: x + 'px', top: y + 'px', zIndex: 700,
        background: '#101315', border: '1px solid var(--border-2)', borderRadius: '3px',
        boxShadow: '0 8px 26px rgba(0,0,0,0.6)', padding: '3px', minWidth: '168px'
      }
    });
    for (const item of items) {
      menu.appendChild(el('button', {
        class: 'ghost sm',
        style: { display: 'block', width: '100%', textAlign: 'left', color: item.tone === 'danger' ? '#e0938c' : '' },
        text: item.label,
        onclick: () => { menu.remove(); item.fn(); }
      }));
    }
    document.body.appendChild(menu);
    const remove = (event) => { if (!menu.contains(event.target)) { menu.remove(); document.removeEventListener('pointerdown', remove); } };
    setTimeout(() => document.addEventListener('pointerdown', remove), 0);
  }

  function inspectorPanel() {
    const panel = el('div', { class: 'panel' }, [
      el('div', { class: 'panel-head' }, [el('h3', { text: 'Inspector' }), el('div', { class: 'spacer' }), el('span', { class: 'tiny faint', id: 'inspHint', text: '' })]),
      el('div', { class: 'panel-body tight', id: 'inspBody' })
    ]);
    return panel;
  }

  function renderInspector() {
    const host = root.querySelector('#inspBody');
    const hint = root.querySelector('#inspHint');
    if (!host) return;
    clear(host);
    const spec = ctx.state.graph;
    if (!spec) return;
    const id = editor ? editor.selectedId : null;
    const node = spec.nodes.find((n) => n.id === id);
    if (!node) {
      if (hint) hint.textContent = 'model';
      host.appendChild(modelFields(spec));
      return;
    }
    if (hint) hint.textContent = node.id;
    if (node.type === 'input') {
      host.appendChild(el('div', { class: 'inspector-field' }, [
        el('span', { text: 'name' }),
        el('input', { type: 'text', value: spec.input.name || 'x', onchange: (event) => { spec.input.name = event.target.value; ctx.touchGraph(); } })
      ]));
      host.appendChild(el('div', { class: 'inspector-field' }, [
        el('span', { text: 'shape' }),
        el('input', {
          type: 'text', value: spec.input.shape.join(', '),
          onchange: (event) => {
            const parts = event.target.value.split(',').map((v) => parseInt(v.trim(), 10)).filter((v) => Number.isFinite(v) && v > 0);
            if (!parts.length) { ctx.warn('Shape needs at least one positive integer.'); return; }
            spec.input.shape = parts;
            ctx.setSpec(spec, { keepId: true });
          }
        })
      ]));
      host.appendChild(el('div', { class: 'tiny faint', text: 'Per-sample shape; the batch dimension is added at runtime.' }));
      host.appendChild(el('div', { class: 'inspector-field' }, [
        el('span', { text: 'dtype' }),
        el('select', { onchange: (event) => { spec.input.dtype = event.target.value; ctx.touchGraph(); } }, ['float32', 'int8', 'uint8'].map((d) => el('option', { value: d, text: d, selected: (spec.input.dtype || 'float32') === d })))
      ]));
      return;
    }
    const fields = layerFields(node.type);
    for (const f of fields) {
      const row = el('div', { class: 'inspector-field' }, [el('span', { text: f.label.toLowerCase() })]);
      const current = node.cfg[f.key] === undefined ? f.def : node.cfg[f.key];
      if (f.type === 'select') {
        row.appendChild(el('select', {
          onchange: (event) => { node.cfg[f.key] = event.target.value; ctx.touchGraph(); }
        }, f.options.map((o) => el('option', { value: o, text: o, selected: String(o) === String(current) }))));
      } else if (f.type === 'bool') {
        row.appendChild(el('label', { class: 'inline' }, [
          el('input', { type: 'checkbox', checked: !!current, onchange: (event) => { node.cfg[f.key] = event.target.checked; ctx.touchGraph(); } }),
          el('span', { class: 'tiny dim', text: current ? 'on' : 'off' })
        ]));
      } else if (f.type === 'int' || f.type === 'number') {
        row.appendChild(el('input', {
          type: 'number', value: current, min: f.min, max: f.max, step: f.step || 1,
          onchange: (event) => {
            const parsed = parseInt(event.target.value, 10);
            if (!Number.isFinite(parsed)) return;
            node.cfg[f.key] = f.min !== undefined ? Math.max(f.min, parsed) : parsed;
            ctx.touchGraph();
          }
        }));
      } else {
        row.appendChild(el('input', {
          type: 'text', value: current === null || current === undefined ? '' : (Array.isArray(current) ? current.join(',') : String(current)),
          onchange: (event) => {
            let value = event.target.value;
            if (Array.isArray(current) || /^-?[\d.,\s-]+$/.test(value)) {
              value = value.split(',').map((v) => parseFloat(v.trim())).filter((v) => Number.isFinite(v));
            }
            node.cfg[f.key] = value;
            ctx.touchGraph();
          }
        }));
      }
      host.appendChild(row);
    }
    const stat = ctx.state.graphStats && ctx.state.graphStats.byNode ? ctx.state.graphStats.byNode[node.id] : null;
    host.appendChild(el('div', { class: 'sep' }));
    host.appendChild(el('div', { class: 'kv' }, [
      el('dt', { text: 'in' }), el('dd', { text: stat && stat.in ? stat.in.map(fmtShape).join(' + ') : '—' }),
      el('dt', { text: 'out' }), el('dd', { text: stat ? fmtShape(stat.out) : '—' }),
      el('dt', { text: 'params' }), el('dd', { text: stat ? stat.params.toLocaleString() : '0' }),
      el('dt', { text: 'macs' }), el('dd', { text: stat ? stat.macs.toLocaleString() : '0' })
    ]));
    host.appendChild(el('div', { class: 'row gap4 mt8 wrap' }, [
      el('button', { class: 'sm ghost', text: 'Set as output', onclick: () => { spec.output = node.id; ctx.touchGraph(); } }),
      el('button', { class: 'sm ghost', text: 'Duplicate', onclick: () => duplicateSelected() }),
      el('button', { class: 'sm danger', text: 'Delete', onclick: () => deleteSelected() })
    ]));
    const err = ctx.state.graphStats && ctx.state.graphStats.errorByNode ? ctx.state.graphStats.errorByNode[node.id] : null;
    if (err) host.appendChild(el('div', { class: 'callout bad mt8', text: err }));
  }

  function modelFields(spec) {
    const wrap = el('div', {});
    wrap.appendChild(el('div', { class: 'inspector-field' }, [
      el('span', { text: 'output node' }),
      el('select', {
        onchange: (event) => { spec.output = event.target.value; ctx.touchGraph(); }
      }, spec.nodes.filter((n) => n.type !== 'input').map((n) => el('option', { value: n.id, text: n.id + ' (' + (LAYERS[n.type] ? LAYERS[n.type].label : n.type) + ')', selected: spec.output === n.id })))
    ]));
    wrap.appendChild(el('div', { class: 'inspector-field' }, [
      el('span', { text: 'head' }),
      el('select', {
        onchange: (event) => { spec.head = event.target.value; ctx.touchGraph(); renderInspector(); }
      }, HEADS.map((h) => el('option', { value: h.value, text: h.label, selected: (spec.head || 'linear') === h.value })))
    ]));
    const headNote = HEADS.find((h) => h.value === (spec.head || 'linear'));
    wrap.appendChild(el('div', { class: 'tiny faint', text: headNote ? headNote.note : '' }));
    const taskKey = ctx.state.taskKey || (spec.meta && spec.meta.task) || 'multiclass';
    wrap.appendChild(el('div', { class: 'inspector-field' }, [
      el('span', { text: 'task' }),
      el('select', {
        onchange: (event) => { ctx.state.taskKey = event.target.value; spec.meta = Object.assign({}, spec.meta, { task: event.target.value }); ctx.touchGraph(); renderInspector(); }
      }, Object.keys(TASKS).map((k) => el('option', { value: k, text: TASKS[k].label, selected: k === taskKey })))
    ]));
    const task = taskFor(taskKey);
    if (task) {
      wrap.appendChild(el('div', { class: 'tiny faint', text: 'loss options: ' + task.lossFns.join(', ') + ' · default ' + task.defaultLoss }));
    }
    wrap.appendChild(el('div', { class: 'sep' }));
    wrap.appendChild(el('div', { class: 'tiny dim', text: 'Select a node in the graph to edit its configuration.' }));
    return wrap;
  }

  function summaryPanel() {
    return el('div', { class: 'panel' }, [
      el('div', { class: 'panel-head' }, [el('h3', { text: 'Model summary' })]),
      el('div', { class: 'panel-body tight', id: 'summaryBody' })
    ]);
  }

  function renderSummary() {
    const host = root.querySelector('#summaryBody');
    if (!host) return;
    clear(host);
    const info = ctx.state.graphStats;
    if (!info) { host.appendChild(el('div', { class: 'tiny dim', text: '—' })); return; }
    host.appendChild(el('div', { class: 'kv' }, [
      el('dt', { text: 'parameters' }), el('dd', { text: info.totalParams.toLocaleString() }),
      el('dt', { text: 'macs' }), el('dd', { text: info.macs.toLocaleString() }),
      el('dt', { text: 'flops' }), el('dd', { text: (info.macs * 2).toLocaleString() }),
      el('dt', { text: 'weights' }), el('dd', { text: formatBytes(info.paramBytes) }),
      el('dt', { text: 'activations' }), el('dd', { text: formatBytes(info.activationBytes) }),
      el('dt', { text: 'input' }), el('dd', { text: fmtShape(ctx.state.graph.input.shape) }),
      el('dt', { text: 'output' }), el('dd', { text: fmtShape(info.outputShape) }),
      el('dt', { text: 'layers' }), el('dd', { text: String(info.nodeStats.length) }),
      el('dt', { text: 'states' }), el('dd', { text: String(info.totalStates) }),
      el('dt', { text: 'status' }), el('dd', { text: info.valid ? 'graph is valid' : info.errors.length + ' problem(s)' })
    ]));
    if (!info.valid) {
      host.appendChild(el('div', { class: 'callout bad mt8' }, info.errors.slice(0, 5).map((e) => el('div', { text: e.id + ': ' + e.message }))));
    }
  }

  function budgetPanel() {
    return el('div', { class: 'panel' }, [
      el('div', { class: 'panel-head' }, [
        el('h3', { text: 'Parameter budget' }),
        el('div', { class: 'spacer' }),
        el('input', {
          type: 'number', id: 'budgetTarget', min: 100, step: 1000, style: { width: '96px' },
          value: ctx.state.budgetTarget || 1000000,
          onchange: (event) => {
            const parsed = parseInt(event.target.value, 10);
            if (Number.isFinite(parsed) && parsed > 0) { ctx.state.budgetTarget = parsed; renderSummary(); renderBudget(); }
          }
        })
      ]),
      el('div', { class: 'panel-body tight', id: 'budgetBody' })
    ]);
  }

  function renderBudget() {
    const host = root.querySelector('#budgetBody');
    if (!host) return;
    clear(host);
    const info = ctx.state.graphStats;
    const spec = ctx.state.graph;
    if (!info || !spec) return;
    const target = ctx.state.budgetTarget || 1000000;
    const presetBudget = spec.meta && spec.meta.budget ? spec.meta.budget : null;
    const reference = spec.meta && spec.meta.reference ? spec.meta.reference : null;
    const value = info.totalParams;
    const position = presetBudget ? budgetPosition({ budget: presetBudget, reference }, value) : null;
    const hi = Math.max(target * 1.6, value * 1.25, presetBudget ? presetBudget.max : 0);
    const frac = Math.max(0, Math.min(1, value / hi));
    const tone = value > target ? 'var(--red)' : value > target * 0.85 ? 'var(--amber)' : 'var(--teal)';
    const bar = el('div', { class: 'bar' }, [
      el('i', { style: { width: (frac * 100).toFixed(2) + '%', background: tone } }),
      el('u', { style: { left: ((target / hi) * 100).toFixed(2) + '%' } }),
      presetBudget ? el('s', { style: { left: ((Math.min(hi, presetBudget.max) / hi) * 100).toFixed(2) + '%' } }) : null
    ]);
    host.appendChild(el('div', { class: 'budget-row' }, [
      el('span', { class: 'mono small', style: { color: tone }, text: value.toLocaleString() }),
      bar,
      el('span', { class: 'mono small dim', text: target >= 1000000 ? (target / 1000000).toFixed(target % 1000000 === 0 ? 0 : 2) + 'M' : target >= 1000 ? Math.round(target / 1000) + 'k' : String(target) })
    ]));
    const pct = target ? (value / target) * 100 : 0;
    host.appendChild(el('div', { class: 'tiny mt4', style: { color: tone }, text: pct.toFixed(1) + '% of target · ' + (value > target ? (value - target).toLocaleString() + ' over' : (target - value).toLocaleString() + ' spare') }));
    if (presetBudget && position) {
      host.appendChild(el('div', { class: 'tiny faint mt8', text: 'Preset window ' + presetBudget.min.toLocaleString() + '–' + presetBudget.max.toLocaleString() + ' (target ' + presetBudget.target.toLocaleString() + '). Currently ' + position.label + ' at ' + (position.ratio * 100).toFixed(0) + '% of that target.' }));
    }
    if (reference) {
      const [lo, hiRef] = reference.params;
      const scale = (value / lo).toFixed(3);
      host.appendChild(el('div', { class: 'callout mt8' }, [
        el('div', { class: 'tiny dim', text: 'Production reference: ' + lo.toLocaleString() + '–' + hiRef.toLocaleString() + ' params — this graph is ' + scale + '× the low end.' }),
        el('div', { class: 'tiny faint mt4', text: reference.note })
      ]));
    }
    host.appendChild(el('div', { class: 'row gap4 mt8 wrap' }, [
      el('button', { class: 'sm ghost', text: 'Fit to target', onclick: () => fitToTarget() }),
      el('button', { class: 'sm ghost', text: 'Halve', onclick: () => scaleBy(0.5) }),
      el('button', { class: 'sm ghost', text: 'Double', onclick: () => scaleBy(2) })
    ]));
  }

  function fitToTarget() {
    const spec = ctx.state.graph;
    const target = ctx.state.budgetTarget || 1000000;
    if (!spec) return;
    const result = scaleToFit(spec, target);
    if (result.spec === spec) { ctx.warn('Could not scale this graph toward ' + target.toLocaleString() + ' parameters.'); return; }
    ctx.setSpec(result.spec, { keepId: true });
    ctx.ok('Scaled to ' + result.params.toLocaleString() + ' parameters (factor ' + result.factor.toFixed(3) + ').');
  }

  function scaleBy(factor) {
    const spec = ctx.state.graph;
    if (!spec) return;
    const result = scaleToFit(spec, Math.max(64, Math.round(spec ? analyze(spec).totalParams * factor : 1000)));
    ctx.setSpec(result.spec, { keepId: true });
    ctx.info('Scaled by ' + factor + '× → ' + result.params.toLocaleString() + ' parameters.');
  }

  function layerTable() {
    return el('div', { class: 'panel' }, [
      el('div', { class: 'panel-head' }, [el('h3', { text: 'Layer table' }), el('div', { class: 'spacer' }), el('span', { class: 'tiny faint', id: 'layerCount', text: '' })]),
      el('div', { class: 'panel-body flush', id: 'layerTableBody' })
    ]);
  }

  function renderLayerTable() {
    const host = root.querySelector('#layerTableBody');
    const count = root.querySelector('#layerCount');
    if (!host) return;
    clear(host);
    const info = ctx.state.graphStats;
    const spec = ctx.state.graph;
    if (!info || !spec) return;
    if (count) count.textContent = info.nodeStats.length + ' nodes · ' + spec.edges.length + ' edges';
    const table = el('table', { class: 'data compact' }, [
      el('thead', {}, el('tr', {}, [
        el('th', { text: '#' }), el('th', { text: 'node' }), el('th', { text: 'type' }), el('th', { text: 'config' }),
        el('th', { text: 'in' }), el('th', { text: 'out' }), el('th', { class: 'n', text: 'params' }), el('th', { class: 'n', text: 'macs' }), el('th', { class: 'n', text: 'share' })
      ]))
    ]);
    const tbody = el('tbody');
    const total = Math.max(1, info.totalParams);
    info.nodeStats.forEach((stat, index) => {
      const node = spec.nodes.find((n) => n.id === stat.id);
      const share = stat.params / total;
      const tr = el('tr', {
        onclick: () => { if (editor) { editor.select(stat.id); editor.centerOn(stat.id); } renderInspector(); },
        style: { cursor: 'pointer' }
      }, [
        el('td', { class: 'n faint', text: String(index) }),
        el('td', { class: 'mono', text: stat.id }),
        el('td', {}, [el('span', { class: 'badge', style: { borderColor: groupColorFor(node ? node.type : 'dense') }, text: stat.label })]),
        el('td', { class: 'faint small ell', style: { maxWidth: '230px' }, text: node ? summarizeCfg(node) : '' }),
        el('td', { class: 'mono faint', text: stat.in ? stat.in.map(fmtShape).join(' + ') : '—' }),
        el('td', { class: 'mono', text: fmtShape(stat.out) }),
        el('td', { class: 'n', text: stat.params.toLocaleString() }),
        el('td', { class: 'n', text: compactNumber(stat.macs) }),
        el('td', { class: 'n', style: { color: share > 0.25 ? 'var(--amber)' : 'var(--muted)' }, text: (share * 100).toFixed(1) + '%' })
      ]);
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    host.appendChild(el('div', { class: 'scroll', style: { maxHeight: '210px' } }, [table]));
  }

  function groupColorFor(type) {
    const group = LAYERS[type] ? LAYERS[type].group : null;
    const found = groupColors().find((g) => g.key === group);
    return found ? found.color : '#8b9298';
  }

  function render() {
    renderToolbar();
    clear(body);
    if (galleryOpen || !ctx.state.graph) body.appendChild(renderGallery());
    else body.appendChild(renderEditor());
  }

  function refresh() {
    if (galleryOpen || !ctx.state.graph) { render(); return; }
    if (!editor || editor.spec !== ctx.state.graph) { render(); return; }
    editor.setSpec(ctx.state.graph, ctx.state.graphStats);
    renderSummary();
    renderBudget();
    renderLayerTable();
    renderInspector();
  }

  const resizeObserver = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(() => { if (editor) editor.render(); }) : null;
  if (resizeObserver) resizeObserver.observe(root);

  render();
  return { root, refresh, render, deleteSelected, duplicateSelected };
}

