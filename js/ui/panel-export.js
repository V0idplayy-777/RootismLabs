import { el, clear, download, copyText, attachTooltip } from './dom.js';
import { Model } from '../core/model.js';
import { EXPORT_FORMATS, precisionsFor, sizeEstimate, runExport, downloadArtifact, downloadBundle, downloadText, quantizationTable, measurePrecisionImpact } from '../export/pipeline.js';
import { bundleSummary, verifyBundle } from '../export/bundle.js';
import { cardToMarkdown, cardFiles } from '../export/card.js';
import { formatBytes, formatMetric } from '../presets/metrics.js';
import * as store from './store.js';

export function createExportPanel(ctx) {
  const root = el('div', { class: 'panel-wrap' });
  const toolbar = el('div', { class: 'toolbar' });
  const body = el('div');
  root.appendChild(toolbar);
  root.appendChild(body);

  const selections = [{ format: 'onnx', precision: 'fp32' }, { format: 'tfjs', precision: 'fp32' }, { format: 'safetensors', precision: 'fp32' }];
  const options = {
    source: 'run',
    runId: null,
    modelName: '',
    description: '',
    docString: '',
    notes: '',
    measure: true,
    split: 'val',
    bundle: true,
    includeDataset: false,
    includeFullDataset: false,
    sampleLimit: 60
  };
  let result = null;
  let impact = null;
  let busy = false;
  let progressText = '';

  function isSelected(format, precision) {
    return selections.some((s) => s.format === format && s.precision === precision);
  }

  function toggle(format, precision) {
    const index = selections.findIndex((s) => s.format === format && s.precision === precision);
    if (index >= 0) selections.splice(index, 1);
    else selections.push({ format, precision });
  }

  function availableRuns() {
    return ctx.state.runs.filter((r) => r.hasWeights !== false);
  }

  async function buildModel() {
    const runs = availableRuns();
    const useRun = options.source === 'run' && runs.length;
    if (useRun) {
      const id = options.runId || ctx.state.selectedRunId || runs[0].id;
      options.runId = id;
      const record = await store.loadRun(id);
      if (!record) throw new Error('could not load run ' + id);
      const model = new Model(record.spec, record.seed || (record.config ? record.config.seed : 1));
      if (record.weights && record.weights.length) model.loadWeights(record.weights);
      if (model.syncStates) model.syncStates();
      return { model, record };
    }
    if (!ctx.state.graph) throw new Error('no architecture loaded');
    const model = new Model(ctx.state.graph, ctx.state.trainConfig ? ctx.state.trainConfig.seed : 20260101);
    return { model, record: null };
  }

  function renderToolbar() {
    clear(toolbar);
    const runs = availableRuns();
    toolbar.appendChild(el('label', { class: 'inline' }, [
      el('span', { class: 'tiny faint', text: 'weights from' }),
      el('select', {
        onchange: (event) => { options.source = event.target.value; render(); }
      }, [
        el('option', { value: 'run', text: runs.length ? 'a trained run' : 'a trained run (none available)', selected: options.source === 'run', disabled: !runs.length }),
        el('option', { value: 'editor', text: 'the editor (randomly initialised)', selected: options.source === 'editor' })
      ])
    ]));
    if (options.source === 'run' && runs.length) {
      toolbar.appendChild(el('select', {
        style: { maxWidth: '280px' },
        onchange: (event) => { options.runId = event.target.value; result = null; impact = null; render(); }
      }, runs.map((r) => el('option', { value: r.id, text: (r.name || r.id) + ' · ' + (r.headline && r.headline.value !== null ? r.headline.metric + ' ' + Number(r.headline.value).toFixed(4) : ''), selected: (options.runId || ctx.state.selectedRunId || runs[0].id) === r.id }))));
    }
    toolbar.appendChild(el('div', { class: 'vsep' }));
    toolbar.appendChild(el('button', { class: 'sm ghost', text: 'Estimate sizes', onclick: () => estimate() }));
    toolbar.appendChild(el('button', { class: 'sm teal', text: busy ? 'Working…' : 'Measure quantization', disabled: busy || !ctx.state.dataset, onclick: () => measure() }));
    toolbar.appendChild(el('button', { class: 'sm primary', text: busy ? 'Working…' : 'Build artifacts', disabled: busy || !selections.length, onclick: () => build() }));
    toolbar.appendChild(el('div', { class: 'spacer' }));
    if (progressText) toolbar.appendChild(el('span', { class: 'tiny faint', text: progressText }));
    toolbar.appendChild(el('span', { class: 'mono tiny faint', text: selections.length + ' artifact(s) selected' }));
  }

  async function estimate() {
    busy = true;
    progressText = 'estimating…';
    renderToolbar();
    try {
      const built = await buildModel();
      estimates = selections.map((s) => ({ sel: s, bytes: sizeEstimate(built.model, s.format, s.precision) }));
      progressText = 'sizes estimated';
      ctx.info('Estimated artifact sizes: ' + estimates.map((e) => e.sel.format + ' ' + e.sel.precision + ' ' + formatBytes(e.bytes)).join(', '));
    } catch (e) {
      ctx.error('Size estimate failed: ' + e.message);
      progressText = 'estimate failed';
    } finally {
      busy = false;
      render();
    }
  }
  let estimates = [];

  async function measure() {
    const built = await buildModel().catch((e) => { ctx.error('Could not build the model: ' + e.message); return null; });
    if (!built) return;
    const ds = ctx.state.dataset;
    if (!ds) { ctx.error('Load the dataset this model was trained on to measure quantization impact.'); return; }
    if (built.record && built.record.datasetHash && (!ds.digest || ds.digest.contentHash !== built.record.datasetHash)) { ctx.error('Load the exact dataset used by this run before measuring quantization.'); return; }
    if (options.split === 'test') ctx.warn('Selecting precision on test data contaminates the test set. Use validation for selection.');
    const wanted = [];
    for (const key of ['fp16', 'bf16', 'int8', 'uint8']) {
      const inSelection = selections.some((s) => s.precision === key);
      if (inSelection || options.measure) wanted.push(key);
    }
    busy = true;
    progressText = 'measuring quantization impact…';
    renderToolbar();
    await new Promise((r) => setTimeout(r, 20));
    try {
      impact = measurePrecisionImpact({
        model: built.model,
        dataset: ds,
        taskKey: built.record ? built.record.taskKey : (ctx.state.taskKey || ds.task),
        config: built.record ? built.record.config : ctx.state.trainConfig,
        split: options.split,
        primaryMetric: built.record && built.record.headline ? built.record.headline.metric : null,
        precisions: wanted
      });
      progressText = 'impact measured on ' + impact.samples + ' ' + impact.split + ' samples';
      ctx.ok('Measured quantization impact against ' + (impact.metricLabel || impact.metricKey) + ' on the ' + impact.split + ' split (' + impact.samples + ' samples).');
    } catch (e) {
      ctx.error('Quantization measurement failed: ' + e.message);
      impact = null;
      progressText = 'measurement failed';
    } finally {
      busy = false;
      render();
    }
  }

  async function build() {
    busy = true;
    progressText = 'building…';
    renderToolbar();
    await new Promise((r) => setTimeout(r, 20));
    try {
      const built = await buildModel();
      const record = built.record;
      const ds = ctx.state.dataset;
      if (record && ds && record.datasetHash && (!ds.digest || ds.digest.contentHash !== record.datasetHash)) throw new Error('The loaded dataset does not match this run. Load the original dataset before exporting, to avoid bundling the wrong preprocessing and labels.');
      if (record && !ds) ctx.warn('No dataset loaded: this export cannot include its preprocessing, label mapping, or data provenance. Load the original dataset to include them.');
      result = runExport({
        model: built.model,
        dataset: ds,
        run: record,
        formats: selections.slice(),
        taskKey: record ? record.taskKey : (ctx.state.taskKey || (ds ? ds.task : 'multiclass')),
        config: record ? record.config : ctx.state.trainConfig,
        modelName: options.modelName || (record ? record.modelName : ctx.state.graph.name),
        description: options.description || (built.model.spec.meta ? built.model.spec.meta.description : ''),
        docString: options.docString || ((options.modelName || built.model.spec.name || 'Model') + ' exported from Rootism Labs'),
        notes: options.notes ? options.notes.split('\n').filter((l) => l.trim()) : [],
        seed: record ? record.seed : (ctx.state.trainConfig ? ctx.state.trainConfig.seed : null),
        quantization: impact,
        measure: options.measure && !!ds,
        bundle: options.bundle,
        includeDataset: options.includeDataset,
        includeFullDataset: options.includeFullDataset,
        sampleLimit: options.sampleLimit,
        primaryMetric: record && record.headline ? record.headline.metric : null,
        onProgress: (p) => { progressText = p.stage === 'bundle' ? 'packing bundle…' : 'exporting ' + p.format + ' ' + p.precision + '…'; renderToolbar(); }
      });
      for (const problem of result.problems || []) ctx.error('Export problem (' + problem.format + ' ' + problem.precision + '): ' + problem.message);
      if (result.artifacts.length) {
        ctx.ok('Built ' + result.artifacts.length + ' artifact(s): ' + result.artifacts.map((a) => a.format + ' ' + a.precision + ' ' + formatBytes(a.bytes)).join(', ') + (result.bundle ? ' · bundle ' + formatBytes(result.bundle.bytes.length) + ' / ' + result.bundle.entries.length + ' files' : '') + '.');
        progressText = 'built ' + result.artifacts.length + ' artifact(s)';
      } else progressText = 'nothing built';
    } catch (e) {
      ctx.error('Export failed: ' + e.message);
      progressText = 'export failed';
      result = null;
    } finally {
      busy = false;
      render();
    }
  }

  function formatMatrix() {
    const host = el('div', { class: 'panel' }, [
      el('div', { class: 'panel-head' }, [el('h3', { text: 'Formats and precision' }), el('div', { class: 'spacer' }), el('span', { class: 'tiny faint', text: 'click a cell to include it' })]),
      el('div', { class: 'panel-body tight' })
    ]);
    const bodyHost = host.querySelector('.panel-body');
    const table = el('table', { class: 'data compact' });
    const head = el('tr', {}, [el('th', { text: 'format' })]);
    const allPrecisions = ['fp32', 'fp16', 'bf16', 'int8', 'uint8'];
    for (const p of allPrecisions) head.appendChild(el('th', { class: 'n', text: p }));
    head.appendChild(el('th', { text: 'runtime' }));
    table.appendChild(el('thead', {}, head));
    const tbody = el('tbody');
    for (const format of EXPORT_FORMATS) {
      const supported = precisionsFor(format.key).map((p) => p.key);
      const tr = el('tr', {}, [
        el('td', {}, [el('div', { class: 'bold', text: format.label }), el('div', { class: 'tiny faint', text: format.extension })])
      ]);
      for (const p of allPrecisions) {
        const ok = supported.indexOf(p) >= 0;
        const on = isSelected(format.key, p);
        const estimateRow = estimates.find((e) => e.sel.format === format.key && e.sel.precision === p);
        const cell = el('td', { class: 'n' });
        if (!ok) {
          cell.appendChild(el('span', { class: 'tiny faint', text: '—' }));
        } else {
          const box = el('button', {
            class: 'xs ' + (on ? 'primary' : 'ghost'),
            onclick: () => { toggle(format.key, p); render(); }
          }, [el('span', { text: on ? '● ' : '○ ' }), el('span', { text: estimateRow ? formatBytes(estimateRow.bytes) : p })]);
          attachTooltip(box, () => {
            const meta = precisionsFor(format.key).find((x) => x.key === p);
            return '<b>' + format.label + ' · ' + meta.label + '</b><br>' + meta.summary + '<br><span style="color:#8b9298">' + meta.bits + ' bits per weight' + (meta.lossless ? ' · lossless' : '') + '</span>';
          });
          cell.appendChild(box);
        }
        tr.appendChild(cell);
      }
      tr.appendChild(el('td', { class: 'tiny faint', style: { maxWidth: '260px' }, text: format.runtime }));
      tbody.appendChild(tr);
      const blurbRow = el('tr', {}, [el('td', { colspan: String(allPrecisions.length + 2), class: 'tiny faint', text: format.blurb })]);
      tbody.appendChild(blurbRow);
    }
    table.appendChild(tbody);
    bodyHost.appendChild(table);
    return host;
  }

  function optionsPanel() {
    const host = el('div', { class: 'panel' }, [
      el('div', { class: 'panel-head' }, [el('h3', { text: 'Export options' })]),
      el('div', { class: 'panel-body' })
    ]);
    const bodyHost = host.querySelector('.panel-body');
    bodyHost.appendChild(el('div', { class: 'config-grid' }, [
      textRow('model name', options.modelName || (ctx.state.graph ? ctx.state.graph.name : ''), (v) => { options.modelName = v; }),
      textRow('evaluation split', options.split, (v) => { options.split = v; }, 'split'),
      areaRow('description', options.description, (v) => { options.description = v; }, 'goes into the model card summary'),
      areaRow('doc string', options.docString, (v) => { options.docString = v; }, 'embedded in the ONNX file'),
      areaRow('notes', options.notes, (v) => { options.notes = v; }, 'one per line, appended to the card')
    ]));
    bodyHost.appendChild(el('div', { class: 'sep' }));
    bodyHost.appendChild(el('div', { class: 'col gap4' }, [
      boolRow('measure quantization impact during build', options.measure, (v) => { options.measure = v; }, 're-evaluates every non-fp32 artifact on the chosen split'),
      boolRow('build a single ZIP bundle', options.bundle, (v) => { options.bundle = v; }),
      boolRow('include dataset samples in the bundle', options.includeDataset, (v) => { options.includeDataset = v; }, 'May disclose private data; disabled by default.'),
      boolRow('include the full dataset in the bundle', options.includeFullDataset, (v) => { options.includeFullDataset = v; }, 'large; only do this if the recipient needs to reproduce training'),
      numberRow('sample limit', options.sampleLimit, (v) => { options.sampleLimit = v; })
    ]));
    return host;
  }

  function textRow(label, value, onChange, kind) {
    return el('label', { class: 'field' }, [
      el('span', { class: 'lbl', text: label }),
      kind === 'split'
        ? el('select', { onchange: (event) => onChange(event.target.value) }, ['train', 'val', 'test'].map((s) => el('option', { value: s, text: s, selected: s === value })))
        : el('input', { type: 'text', value: value || '', oninput: (event) => onChange(event.target.value) })
    ]);
  }

  function areaRow(label, value, onChange, hint) {
    return el('label', { class: 'field full' }, [
      el('span', { class: 'lbl', text: label }),
      el('textarea', { rows: '2', value: value || '', oninput: (event) => onChange(event.target.value), style: { width: '100%' } }),
      hint ? el('span', { class: 'tiny faint', text: hint }) : null
    ]);
  }

  function boolRow(label, value, onChange, hint) {
    return el('label', { class: 'inline', style: { width: '100%' } }, [
      el('input', { type: 'checkbox', checked: !!value, onchange: (event) => onChange(event.target.checked) }),
      el('span', { text: label }),
      hint ? el('span', { class: 'tiny faint', text: ' — ' + hint }) : null
    ]);
  }

  function numberRow(label, value, onChange) {
    return el('label', { class: 'field' }, [
      el('span', { class: 'lbl', text: label }),
      el('input', { type: 'number', min: 0, max: 5000, step: 10, value: value, onchange: (event) => onChange(parseInt(event.target.value, 10) || 0) })
    ]);
  }

  function quantPanel() {
    if (!impact) {
      return el('div', { class: 'panel' }, [
        el('div', { class: 'panel-head' }, [el('h3', { text: 'Quantization impact' })]),
        el('div', { class: 'panel-body' }, [
          el('div', { class: 'callout', text: ctx.state.dataset ? 'Press "Measure quantization" to evaluate the same weights at every precision on the ' + options.split + ' split and see the accuracy you lose.' : 'Load the dataset this model was trained on to measure accuracy before and after quantization.' })
        ])
      ]);
    }
    const rows = quantizationTable(impact);
    const table = el('table', { class: 'data compact' }, [
      el('thead', {}, el('tr', {}, [
        el('th', { text: 'precision' }), el('th', { class: 'n', text: 'size' }), el('th', { class: 'n', text: 'ratio' }),
        el('th', { class: 'n', text: (impact.metricLabel || impact.metricKey) }), el('th', { class: 'n', text: 'delta' }),
        el('th', { class: 'n', text: 'max rel L2' }), el('th', { class: 'n', text: 'worst SNR' }), el('th', { text: 'verdict' })
      ])),
      el('tbody')
    ]);
    const tbody = table.querySelector('tbody');
    const baseline = rows[0];
    for (const row of rows) {
      const delta = row.delta;
      const tone = delta === null ? '' : Math.abs(delta) < 1e-9 ? 'green' : (row.precision === 'fp16' || row.precision === 'bf16') ? (Math.abs(delta) < 0.01 ? 'green' : 'amber') : Math.abs(delta) < 0.02 ? 'green' : Math.abs(delta) < 0.05 ? 'amber' : 'red';
      tbody.appendChild(el('tr', {}, [
        el('td', {}, [el('span', { class: 'mono', text: row.precision }), el('span', { class: 'tiny faint', text: ' ' + row.label })]),
        el('td', { class: 'n', text: row.bytes ? formatBytes(row.bytes) : '—' }),
        el('td', { class: 'n faint', text: row.compressionRatio ? row.compressionRatio.toFixed(2) + '×' : (baseline.bytes && row.bytes ? (baseline.bytes / row.bytes).toFixed(2) + '×' : '—') }),
        el('td', { class: 'n', text: row.metric === null || row.metric === undefined ? '—' : formatMetric((impact.metricLabel || impact.metricKey), row.metric) }),
        el('td', { class: 'n ' + tone, text: delta === null ? 'reference' : (delta > 0 ? '+' : '') + delta.toFixed(4) }),
        el('td', { class: 'n faint', text: row.maxRelativeL2 === undefined || row.maxRelativeL2 === null ? '—' : Number(row.maxRelativeL2).toExponential(2) }),
        el('td', { class: 'n faint', text: row.worstSnrDb === null || row.worstSnrDb === undefined ? '—' : Number(row.worstSnrDb).toFixed(1) + ' dB' }),
        el('td', { class: 'tiny', text: row.verdict || '' })
      ]));
    }
    return el('div', { class: 'panel' }, [
      el('div', { class: 'panel-head' }, [
        el('h3', { text: 'Quantization impact' }),
        el('div', { class: 'spacer' }),
        el('span', { class: 'tiny faint', text: (impact.metricLabel || impact.metricKey) + ' on ' + impact.samples + ' ' + impact.split + ' samples' }),
        el('button', { class: 'xs ghost', text: 'Download CSV', onclick: () => download('quantization-impact.csv', ['precision,label,bytes,metric,delta,maxRelativeL2,worstSnrDb,verdict'].concat(rows.map((r) => [r.precision, '"' + r.label + '"', r.bytes === null ? '' : r.bytes, r.metric === null ? '' : r.metric, r.delta === null ? '' : r.delta, r.maxRelativeL2 === undefined ? '' : r.maxRelativeL2, r.worstSnrDb === null || r.worstSnrDb === undefined ? '' : r.worstSnrDb, '"' + String(r.verdict || '').replace(/"/g, '') + '"'].join(','))).join('\n'), 'text/csv') })
      ]),
      el('div', { class: 'panel-body flush' }, [el('div', { class: 'scroll' }, [table])])
    ]);
  }

  function artifactsPanel() {
    if (!result) return el('div', {});
    const host = el('div', { class: 'col gap8' });
    const rows = result.artifacts.map((artifact) => {
      const buttons = el('div', { class: 'row gap4' });
      for (const file of artifact.files || []) {
        buttons.appendChild(el('button', {
          class: 'xs', text: 'Download ' + file.name,
          onclick: () => {
            download(file.name, file.data, artifact.mime || 'application/octet-stream');
            ctx.ok('Downloaded ' + file.name + ' (' + formatBytes((file.data.length || file.data.byteLength || 0)) + ').');
          }
        }));
      }
      if (!artifact.files || !artifact.files.length) {
        buttons.appendChild(el('button', { class: 'xs', text: 'Download', onclick: () => downloadArtifact(artifact) }));
      }
      return el('div', { class: 'artifact-row' }, [
        el('div', {}, [
          el('div', { class: 'bold', text: artifact.format + ' · ' + artifact.precision }),
          el('div', { class: 'tiny faint', text: (artifact.nodes ? artifact.nodes + ' graph nodes · ' : '') + (artifact.weights || 0) + ' tensors · in "' + artifact.inputName + '" ' + JSON.stringify(artifact.inputShape) + ' → out "' + artifact.outputName + '" ' + JSON.stringify(artifact.outputShape) + (artifact.opset ? ' · opset ' + artifact.opset : '') + (artifact.head ? ' · head ' + artifact.head : '') + (artifact.fixedSequenceLength ? ' · unrolled to ' + artifact.fixedSequenceLength + ' steps' : '') })
        ]),
        el('span', { class: 'mono small amber', text: formatBytes(artifact.bytes) }),
        el('span', { class: 'badge', text: artifact.extension }),
        buttons
      ]);
    });
    const list = rows.concat((result.artifacts.length && result.artifacts[0].notes && result.artifacts[0].notes.length) ? [el('div', { class: 'tiny faint', text: 'Export notes: ' + result.artifacts.map((a) => (a.notes || []).join(' ')).filter((t) => t).join(' | ') })] : []);
    host.appendChild(el('div', { class: 'panel' }, [
      el('div', { class: 'panel-head' }, [el('h3', { text: 'Artifacts' }), el('div', { class: 'spacer' }), el('span', { class: 'tiny faint', text: result.artifacts.length + ' built · ' + formatBytes(result.artifacts.reduce((a, b) => a + b.bytes, 0)) })]),
      el('div', { class: 'panel-body tight' }, [el('div', { class: 'col gap6' }, list)])
    ]));
    if (result.problems && result.problems.length) {
      host.appendChild(el('div', { class: 'callout bad' }, result.problems.map((p) => el('div', { text: p.format + ' ' + p.precision + ': ' + p.message }))));
    }
    return host;
  }

  function bundlePanel() {
    if (!result || !result.bundle) return el('div', {});
    const bundle = result.bundle;
    const summary = bundleSummary(bundle);
    const verification = verifyBundle(bundle.bytes);
    const tree = el('div', { class: 'scroll', style: { maxHeight: '300px' } });
    const table = el('table', { class: 'data compact' }, [
      el('thead', {}, el('tr', {}, [el('th', { text: 'path' }), el('th', { class: 'n', text: 'bytes' }), el('th', { text: 'sha256' })])),
      el('tbody', {}, bundle.entries.map((e) => {
        const manifestRow = bundle.manifest && bundle.manifest.files ? bundle.manifest.files.find((f) => f.path === e.name) : null;
        return el('tr', {}, [
          el('td', { class: 'mono tiny', text: e.name }),
          el('td', { class: 'n', text: formatBytes((e.data.length || e.data.byteLength || 0)) }),
          el('td', { class: 'mono tiny faint', text: manifestRow ? manifestRow.sha256.slice(0, 24) + '…' : '' })
        ]);
      }))
    ]);
    tree.appendChild(table);
    return el('div', { class: 'panel' }, [
      el('div', { class: 'panel-head' }, [
        el('h3', { text: 'Bundle' }),
        el('div', { class: 'spacer' }),
        el('span', { class: 'mono tiny', text: summary.files + ' files · ' + summary.bytesHuman }),
        el('button', { class: 'xs primary', text: 'Download ZIP', onclick: () => { downloadBundle(bundle, bundleName()); ctx.ok('Downloaded ' + bundleName() + ' (' + summary.bytesHuman + ').'); } })
      ]),
      el('div', { class: 'panel-body tight' }, [
        el('div', { class: 'row gap6 wrap mb8' }, summary.byDirectory.map((d) => el('span', { class: 'chip', text: d.directory + ' · ' + d.files + ' · ' + d.bytesHuman }))),
        el('div', { class: 'callout ' + (verification.ok ? 'good' : 'bad'), text: verification.ok ? 'Bundle verified: every declared file is present at the declared size and the ZIP checksums pass.' : 'Bundle verification failed: ' + verification.problems.join('; ') }),
        el('div', { class: 'kv mt8' }, [
          el('dt', { text: 'tree sha256' }), el('dd', { text: bundle.manifest ? bundle.manifest.treeSha256 : '—' }),
          el('dt', { text: 'bundle version' }), el('dd', { text: bundle.manifest ? String(bundle.manifest.bundleVersion) : '—' }),
          el('dt', { text: 'generated' }), el('dd', { text: bundle.manifest ? new Date(bundle.manifest.generatedAt).toISOString() : '—' })
        ]),
        tree
      ])
    ]);
  }

  function bundleName() {
    const base = (options.modelName || (ctx.state.graph ? ctx.state.graph.name : 'model')).toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'model';
    return base + '-bundle.zip';
  }

  function cardPanel() {
    if (!result || !result.card) return el('div', {});
    const markdown = (result.bundle && result.bundle.markdown) || cardToMarkdown(result.card);
    const preview = el('div', { class: 'card-preview', text: markdown });
    const files = cardFiles(result.card);
    return el('div', { class: 'panel' }, [
      el('div', { class: 'panel-head' }, [
        el('h3', { text: 'Model card' }),
        el('div', { class: 'spacer' }),
        el('span', { class: 'tiny faint', text: markdown.length.toLocaleString() + ' characters' }),
        el('button', { class: 'xs ghost', text: 'Copy markdown', onclick: () => copyText(markdown).then(() => ctx.toast('Model card copied.')) }),
        el('button', { class: 'xs ghost', text: 'Download README.md', onclick: () => downloadText(markdown, 'README.md', 'text/markdown') }),
        el('button', { class: 'xs ghost', text: 'Download card JSON', onclick: () => download('model-card.json', JSON.stringify(result.card, null, 2), 'application/json') })
      ]),
      el('div', { class: 'panel-body tight' }, [
        el('div', { class: 'row gap4 wrap mb8' }, files.map((f) => el('button', { class: 'xs ghost', text: f.name, onclick: () => download(f.name.split('/').pop(), f.data, 'application/octet-stream') }))),
        preview
      ])
    ]);
  }

  function render() {
    renderToolbar();
    clear(body);
    const host = el('div', { class: 'col gap8' });
    host.appendChild(formatMatrix());
    host.appendChild(optionsPanel());
    host.appendChild(quantPanel());
    host.appendChild(artifactsPanel());
    host.appendChild(bundlePanel());
    host.appendChild(cardPanel());
    body.appendChild(host);
  }

  function refresh(reason) {
    if (reason === 'progress' || reason === 'log' || reason === 'epoch') return;
    if (reason === 'runs' && !options.runId) result = null;
    render();
  }

  render();
  return { root, refresh, render };
}
