import { el, clear, attachTooltip, download, copyText, modal } from './dom.js';
import { barChart, histogram, lineChart, SERIES_COLORS } from './charts.js';
import { PRESET_DATASET_LIST, PRESET_DATASETS, datasetPresetByKey, datasetPresetsForModality, datasetLicense, datasetSizeEstimate, buildPresetDataset } from '../presets/datasets.js';
import { GENERATOR_LIST, generatorByKey, buildDataset, generatorsForModality, referenceFor } from '../data/generators.js';
import { MODALITIES, subsetDataset, buildSplits, labelVector, computeStats, recompute, qualityReport } from '../data/dataset.js';
import { kFold, splitSummary } from '../data/split.js';
import { augmentView, qualityView, exportDataView, samplePreviewCell, sampleText, labelName, classBalanceView, openImportDialog, splitCounts } from './data-tools.js';
import { formatBytes } from '../presets/metrics.js';
import { TASKS } from '../core/tasks.js';
import { shapeSize } from '../core/tensor.js';

const MODALITY_COLOR = { text: '#e0a33e', audio: '#4fa3a5', image: '#5b87b5', sensor: '#8a7bb0', series: '#6aa84f', tabular: '#b0765f' };
const PAGE_SIZE = 40;

export function createDataPanel(ctx) {
  const root = el('div', { class: 'panel-wrap' });
  const toolbar = el('div', { class: 'toolbar' });
  const body = el('div');
  root.appendChild(toolbar);
  root.appendChild(body);
  let tab = 'overview';
  let galleryModality = 'all';
  let galleryKind = 'preset';
  let page = 0;
  let filterSplit = 'all';
  let filterLabel = 'all';
  let filterText = '';
  let selected = new Set();

  function ds() { return ctx.state.dataset; }

  function renderToolbar() {
    clear(toolbar);
    const data = ds();
    if (!data) {
      toolbar.appendChild(el('span', { class: 'small dim', text: 'No dataset loaded. Presets are generated locally and deterministically; imports keep your own license.' }));
      toolbar.appendChild(el('div', { class: 'spacer' }));
      toolbar.appendChild(el('button', { text: 'Import files', onclick: () => openImportDialog(ctx, onImported) }));
      return;
    }
    toolbar.appendChild(el('input', {
      type: 'text', value: data.name, style: { width: '210px', fontWeight: '600' },
      onchange: (event) => { data.name = event.target.value; data.slug = data.name.toLowerCase().replace(/[^a-z0-9]+/g, '-'); ctx.touchDataset(); }
    }));
    toolbar.appendChild(el('span', { class: 'badge', style: { borderColor: MODALITY_COLOR[data.modality] || '#2a2e33' }, text: data.modality }));
    toolbar.appendChild(el('span', { class: 'badge teal', text: data.task }));
    toolbar.appendChild(el('span', { class: 'mono tiny faint', text: data.count + ' samples · ' + formatBytes(data.stats ? data.stats.bytes : 0) }));
    toolbar.appendChild(el('div', { class: 'vsep' }));
    toolbar.appendChild(el('button', { class: 'sm ghost', text: 'Presets', onclick: () => { galleryKind = 'preset'; openGallery(); } }));
    toolbar.appendChild(el('button', { class: 'sm ghost', text: 'Generate', onclick: () => { galleryKind = 'generator'; openGallery(); } }));
    toolbar.appendChild(el('button', { class: 'sm ghost', text: 'Import', onclick: () => openImportDialog(ctx, onImported) }));
    toolbar.appendChild(el('div', { class: 'spacer' }));
    toolbar.appendChild(el('span', { class: 'mono tiny faint', text: data.digest ? data.digest.short : '' }));
    toolbar.appendChild(el('button', { class: 'sm primary', text: 'Save dataset', onclick: () => ctx.saveDataset() }));
  }

  function onImported(imported) {
    ctx.setDataset(imported);
    ctx.ok('Dataset "' + imported.name + '" loaded with ' + imported.count + ' samples.');
    tab = 'overview';
    render();
  }

  let galleryOpen = false;
  function openGallery() { galleryOpen = true; render(); }
  function closeGallery() { galleryOpen = false; render(); }

  function renderGallery() {
    const isPreset = galleryKind === 'preset';
    const items = isPreset ? PRESET_DATASET_LIST : GENERATOR_LIST;
    const list = items.filter((item) => galleryModality === 'all' || item.modality === galleryModality);
    const modalityChoices = [{ key: 'all', label: 'All' }].concat(Object.keys(MODALITIES).map((k) => ({ key: k, label: MODALITIES[k].label })));
    const chips = el('div', { class: 'pill-row' }, modalityChoices.map((m) => el('button', {
      class: 'sm' + (galleryModality === m.key ? ' primary' : ' ghost'), text: m.label,
      onclick: () => { galleryModality = m.key; render(); }
    })));
    const grid = el('div', { class: 'preset-grid' });
    for (const item of list) grid.appendChild(isPreset ? presetCard(item) : generatorCard(item));
    return el('div', { class: 'col gap8' }, [
      el('div', { class: 'row between wrap' }, [
        el('div', {}, [
          el('h2', { text: isPreset ? 'Dataset presets' : 'Synthetic generators' }),
          el('div', { class: 'small dim', text: isPreset
            ? list.length + ' ready-made datasets. Each one is generated in your browser from a fixed seed, so a run is reproducible without shipping the data. The license shown is for the generated samples and for the real-world corpus they stand in for.'
            : list.length + ' generators with their own knobs. Presets are thin wrappers over these.' })
        ]),
        el('div', { class: 'row' }, [
          el('button', { class: galleryKind === 'preset' ? 'primary' : 'ghost', text: 'Presets', onclick: () => { galleryKind = 'preset'; render(); } }),
          el('button', { class: galleryKind === 'generator' ? 'primary' : 'ghost', text: 'Generators', onclick: () => { galleryKind = 'generator'; render(); } }),
          el('button', { text: 'Import files', onclick: () => openImportDialog(ctx, onImported) }),
          ds() ? el('button', { class: 'ghost', text: 'Back to dataset', onclick: () => closeGallery() }) : null
        ])
      ]),
      chips,
      grid
    ]);
  }

  function licenseBadge(licenseInfo, license) {
    const local = licenseInfo && licenseInfo.local ? licenseInfo.local : license;
    const reference = licenseInfo ? licenseInfo.reference : null;
    const commercial = local ? local.allowsCommercial : false;
    const badge = el('span', { class: 'badge ' + (commercial ? 'green' : 'red'), text: local ? local.name : 'unknown license' });
    const parts = [];
    if (local) parts.push('<b>Generated samples</b><br>' + local.name + (local.notes ? '<br><span style="color:#8b9298">' + local.notes + '</span>' : ''));
    if (reference) {
      parts.push('<b>Stands in for</b><br>' + reference.name + '<br>' + (reference.license ? reference.license.name + ' — commercial use ' + (reference.license.allowsCommercial ? 'allowed' : 'NOT allowed') : 'license varies') + (reference.url ? '<br><span style="color:#8b9298">' + reference.url + '</span>' : '') + (reference.note ? '<br><span style="color:#8b9298">' + reference.note + '</span>' : ''));
    }
    attachTooltip(badge, () => parts.join('<br><br>'));
    return badge;
  }

  function presetCard(preset) {
    const licenseInfo = datasetLicense(preset.key);
    const sizeCounts = ['small', 'medium', 'large'].map((s) => datasetSizeEstimate(preset.key, s) || 0);
    const card = el('div', { class: 'preset-card' + (ds() && ds().presetKey === preset.key ? ' on' : ''), onclick: () => openPresetDialog(preset) }, [
      el('span', { class: 'strip', style: { background: MODALITY_COLOR[preset.modality] || '#8b9298' } }),
      el('div', { class: 'ttl' }, [
        el('h3', { text: preset.name }),
        el('span', { class: 'fam', text: preset.modality })
      ]),
      el('div', { class: 'tag', text: preset.tagline }),
      el('div', { class: 'foot' }, [
        licenseBadge(licenseInfo, null),
        el('span', { class: 'badge teal', text: preset.task }),
        el('span', { text: sizeCounts.some((n) => n) ? sizeCounts.map((n) => n.toLocaleString()).join(' / ') + ' samples' : '—' })
      ])
    ]);
    attachTooltip(card, () => '<b>' + preset.name + '</b><br>' + (preset.notes || '') + '<br><br><span style="color:#8b9298">generator</span> ' + preset.generator + '<br><span style="color:#8b9298">splits</span> ' + preset.splits.method + ' ' + preset.splits.ratios.join('/') + '<br><span style="color:#8b9298">augmentation</span> ' + (preset.augment || []).map((a) => a.id).join(', '));
    return card;
  }

  function generatorCard(gen) {
    const reference = referenceFor(gen.key);
    const card = el('div', { class: 'preset-card', onclick: () => openGeneratorDialog(gen) }, [
      el('span', { class: 'strip', style: { background: MODALITY_COLOR[gen.modality] || '#8b9298' } }),
      el('div', { class: 'ttl' }, [
        el('h3', { text: gen.label || gen.name || gen.key }),
        el('span', { class: 'fam', text: gen.modality })
      ]),
      el('div', { class: 'tag', text: gen.headline || gen.blurb || 'Synthetic generator.' }),
      el('div', { class: 'foot' }, [
        el('span', { class: 'badge green', text: 'generated locally' }),
        reference ? el('span', { class: 'badge ' + (reference.license && reference.license.allowsCommercial ? 'green' : 'red'), text: reference.name.slice(0, 26) }) : null
      ])
    ]);
    attachTooltip(card, () => '<b>' + (gen.label || gen.key) + '</b><br>' + (gen.blurb || '') + '<br><br><span style="color:#8b9298">task</span> ' + gen.task + ' · <span style="color:#8b9298">modality</span> ' + gen.modality + (gen.metrics ? '<br><span style="color:#8b9298">reports</span> ' + gen.metrics.join(', ') : '') + (reference ? '<br><br><b>Stands in for</b><br>' + reference.name + '<br>' + (reference.license ? reference.license.name : '') + '<br>' + (reference.note || '') : ''));
    return card;
  }

  function openPresetDialog(preset) {
    const sizes = ['small', 'medium', 'large'].map((s) => datasetSizeEstimate(preset.key, s) || 0);
    const options = { size: preset.defaults.size || 'medium', seed: preset.defaults.seed === undefined ? 11 : preset.defaults.seed };
    const licenseInfo = datasetLicense(preset.key);
    const bodyNodes = [
      el('p', { text: preset.tagline }),
      preset.notes ? el('div', { class: 'callout', text: preset.notes }) : null,
      el('div', { class: 'config-grid mt8' }, [
        el('label', { class: 'field' }, [
          el('span', { class: 'lbl', text: 'size' }),
          el('select', { onchange: (event) => { options.size = event.target.value; updateEstimate(); } }, ['small', 'medium', 'large'].map((s, i) => el('option', { value: s, text: s + ' · about ' + (sizes[i] || 0).toLocaleString() + ' samples', selected: options.size === s })))
        ]),
        el('label', { class: 'field' }, [
          el('span', { class: 'lbl', text: 'seed' }),
          el('input', { type: 'number', value: options.seed, oninput: (event) => { options.seed = parseInt(event.target.value, 10) || 0; } })
        ])
      ]),
      el('div', { class: 'kv mt8' }, [
        el('dt', { text: 'modality' }), el('dd', { text: preset.modality }),
        el('dt', { text: 'task' }), el('dd', { text: preset.task }),
        el('dt', { text: 'generator' }), el('dd', { text: preset.generator }),
        el('dt', { text: 'splits' }), el('dd', { text: preset.splits.method + ' ' + preset.splits.ratios.join(' / ') }),
        el('dt', { text: 'augmentation' }), el('dd', { text: (preset.augment || []).map((a) => a.id + ' ' + a.strength).join(', ') || 'none preset' })
      ]),
      el('div', { class: 'mt8' }, [
        el('div', { class: 'section-title', text: 'License' }),
        el('div', { class: 'kv' }, [
          el('dt', { text: 'generated' }), el('dd', { text: licenseInfo.local.name + (licenseInfo.local.allowsCommercial ? ' · commercial use allowed' : '') }),
          el('dt', { text: 'reference' }), el('dd', { text: licenseInfo.reference ? licenseInfo.reference.name + ' · ' + (licenseInfo.reference.license ? licenseInfo.reference.license.name : 'license varies') : 'none' }),
          el('dt', { text: 'note' }), el('dd', { text: licenseInfo.reference ? licenseInfo.reference.note : '' })
        ])
      ]),
      el('div', { class: 'tiny faint mt8', id: 'estimateLine', text: '' })
    ];
    function updateEstimate() {
      const line = handle.box.querySelector('#estimateLine');
      if (!line) return;
      const estimate = datasetSizeEstimate(preset.key, options.size);
      line.textContent = estimate ? 'Will generate about ' + estimate.toLocaleString() + ' samples at this size.' : 'Sample count depends on the generator settings.';
    }
    const handle = modal('Build "' + preset.name + '"', bodyNodes, [
      el('button', { class: 'ghost', text: 'Cancel', onclick: () => handle.close() }),
      el('button', {
        class: 'primary', text: 'Generate dataset', onclick: () => {
          handle.close();
          try {
            const built = buildPresetDataset(preset.key, { size: options.size, seed: options.seed });
            built.name = preset.name;
            ctx.setDataset(built);
            ctx.state.taskKey = built.task;
            ctx.ok('Generated "' + built.name + '": ' + built.count + ' samples, ' + built.xShape.join('×') + ' features, digest ' + built.digest.short + '.');
            tab = 'overview';
            galleryOpen = false;
            render();
          } catch (e) {
            ctx.error('Could not build the preset: ' + e.message);
          }
        }
      })
    ], { width: '620px' });
    updateEstimate();
  }

  function openGeneratorDialog(gen) {
    const options = {};
    const fields = [];
    for (const opt of gen.options || []) {
      options[opt.key] = opt.def;
      let control;
      if (opt.type === 'select') {
        control = el('select', { onchange: (event) => { options[opt.key] = opt.options.map(String).indexOf(event.target.value) >= 0 && typeof opt.def === 'number' ? Number(event.target.value) : event.target.value; } },
          opt.options.map((o) => el('option', { value: String(o), text: String(o), selected: String(o) === String(opt.def) })));
      } else if (opt.type === 'bool') {
        control = el('label', { class: 'inline' }, [el('input', { type: 'checkbox', checked: !!opt.def, onchange: (event) => { options[opt.key] = event.target.checked; } })]);
      } else if (opt.type === 'text') {
        control = el('input', { type: 'text', value: opt.def === undefined ? '' : String(opt.def), oninput: (event) => { options[opt.key] = event.target.value; } });
      } else {
        control = el('input', { type: 'number', value: opt.def, min: opt.min, max: opt.max, step: opt.step || 1, oninput: (event) => { options[opt.key] = parseFloat(event.target.value); } });
      }
      fields.push(el('label', { class: 'field' }, [
        el('span', { class: 'lbl', text: opt.label }),
        control,
        opt.min !== undefined && opt.max !== undefined ? el('span', { class: 'tiny faint', text: 'range ' + opt.min + ' … ' + opt.max }) : null
      ]));
    }
    const reference = referenceFor(gen.key);
    const handle = modal('Generate "' + (gen.label || gen.key) + '"', [
      el('p', { text: gen.blurb || '' }),
      el('div', { class: 'config-grid mt8' }, fields.concat([
        el('label', { class: 'field' }, [
          el('span', { class: 'lbl', text: 'dataset name' }),
          el('input', { type: 'text', value: gen.label || gen.key, oninput: (event) => { options.datasetName = event.target.value; } })
        ])
      ])),
      el('div', { class: 'kv mt8' }, [
        el('dt', { text: 'generator' }), el('dd', { text: gen.key }),
        el('dt', { text: 'task' }), el('dd', { text: gen.task }),
        el('dt', { text: 'modality' }), el('dd', { text: gen.modality }),
        el('dt', { text: 'reports' }), el('dd', { text: (gen.metrics || []).join(', ') || '—' })
      ]),
      reference ? el('div', { class: 'callout warn mt8' }, [
        el('div', { class: 'bold', text: 'Stands in for: ' + reference.name }),
        el('div', { class: 'tiny mt4', text: reference.license ? reference.license.name + ' — commercial use ' + (reference.license.allowsCommercial ? 'allowed' : 'NOT allowed') : 'license varies' }),
        reference.note ? el('div', { class: 'tiny faint mt4', text: reference.note }) : null
      ]) : null,
      el('div', { class: 'tiny faint mt8', text: 'Generated samples carry a CC0-style local license: nothing third-party is embedded, so the samples and models trained on them are yours.' })
    ], [
      el('button', { class: 'ghost', text: 'Cancel', onclick: () => handle.close() }),
      el('button', {
        class: 'primary', text: 'Generate', onclick: () => {
          handle.close();
          try {
            const built = buildDataset(gen.key, options);
            built.name = options.datasetName || gen.label || gen.key;
            built.source = 'generated';
            ctx.setDataset(built);
            ctx.state.taskKey = built.task;
            ctx.ok('Generated "' + built.name + '": ' + built.count + ' samples, features ' + built.xShape.join('×') + '.');
            galleryOpen = false;
            tab = 'overview';
            render();
          } catch (e) {
            ctx.error('Generator failed: ' + e.message);
          }
        }
      })
    ], { width: '640px' });
  }

  function tabs() {
    const list = [
      { key: 'overview', label: 'Overview' },
      { key: 'samples', label: 'Samples & labels' },
      { key: 'splits', label: 'Splits' },
      { key: 'augment', label: 'Augmentation' },
      { key: 'quality', label: 'Quality' },
      { key: 'export', label: 'Export' }
    ];
    return el('div', { class: 'tab-strip' }, list.map((t) => el('button', {
      class: tab === t.key ? 'on' : '', text: t.label, onclick: () => { tab = t.key; render(); }
    })));
  }

  function renderOverview() {
    const data = ds();
    const stats = data.stats || computeStats(data);
    const licenseInfo = data.licenseInfo || (data.presetKey ? datasetLicense(data.presetKey) : null);
    const reference = data.provenance && data.provenance.reference ? data.provenance.reference : (licenseInfo ? licenseInfo.reference : null);
    const left = el('div', { class: 'col gap8' }, [
      el('div', { class: 'panel' }, [
        el('div', { class: 'panel-head' }, [el('h3', { text: 'Contents' })]),
        el('div', { class: 'panel-body tight' }, [
          el('div', { class: 'kv' }, [
            el('dt', { text: 'samples' }), el('dd', { text: String(data.count) }),
            el('dt', { text: 'modality' }), el('dd', { text: data.modality }),
            el('dt', { text: 'task' }), el('dd', { text: data.task + ' — ' + (TASKS[data.task] ? TASKS[data.task].label : '') }),
            el('dt', { text: 'features' }), el('dd', { text: data.xShape.join(' × ') + ' (' + shapeSize(data.xShape) + ' values)' }),
            el('dt', { text: 'target' }), el('dd', { text: (data.yShape || [1]).join(' × ') }),
            el('dt', { text: 'classes' }), el('dd', { text: data.labels && data.labels.length ? data.labels.length + ' — ' + data.labels.slice(0, 8).join(', ') + (data.labels.length > 8 ? ', …' : '') : 'continuous / none' }),
            el('dt', { text: 'bytes' }), el('dd', { text: formatBytes(stats.bytes || 0) }),
            el('dt', { text: 'sparsity' }), el('dd', { text: stats.sparsity !== undefined ? (100 * stats.sparsity).toFixed(1) + '%' : '—' }),
            el('dt', { text: 'non-finite' }), el('dd', { text: String(stats.nonFinite || 0) }),
            el('dt', { text: 'source' }), el('dd', { text: data.source + (data.presetKey ? ' · preset ' + data.presetKey : '') })
          ])
        ])
      ]),
      el('div', { class: 'panel' }, [
        el('div', { class: 'panel-head' }, [el('h3', { text: 'Reproducibility' })]),
        el('div', { class: 'panel-body tight' }, [
          el('div', { class: 'kv' }, [
            el('dt', { text: 'content hash' }), el('dd', { text: data.digest ? data.digest.contentHash : '—' }),
            el('dt', { text: 'feature hash' }), el('dd', { text: data.digest ? data.digest.featureHash : '—' }),
            el('dt', { text: 'label hash' }), el('dd', { text: data.digest ? data.digest.labelHash : '—' }),
            el('dt', { text: 'split method' }), el('dd', { text: data.splitMethod + ' · seed ' + data.splitSeed + ' · ' + (data.splitRatios || []).map((r) => r.toFixed(2)).join('/') }),
            el('dt', { text: 'feature spec' }), el('dd', { text: data.featureSpec ? JSON.stringify(data.featureSpec) : '—' })
          ]),
          el('div', { class: 'row gap4 mt8' }, [
            el('button', { class: 'xs ghost', text: 'Copy content hash', onclick: () => copyText(data.digest ? data.digest.contentHash : '').then(() => ctx.toast('Hash copied.')) }),
            el('button', { class: 'xs ghost', text: 'Copy provenance', onclick: () => copyText(JSON.stringify({ name: data.name, digest: data.digest, license: data.license, provenance: data.provenance, splits: splitCounts(data), featureSpec: data.featureSpec }, null, 2)).then(() => ctx.toast('Provenance copied.')) })
          ])
        ])
      ]),
      el('div', { class: 'panel' }, [
        el('div', { class: 'panel-head' }, [el('h3', { text: 'License' })]),
        el('div', { class: 'panel-body tight' }, [
          el('div', { class: 'row gap6 wrap mb8' }, [
            el('span', { class: 'badge ' + (data.license && data.license.allowsCommercial ? 'green' : 'red'), text: data.license ? data.license.name : 'unset' }),
            el('span', { class: 'tiny faint', text: data.license && data.license.allowsCommercial ? 'commercial use allowed' : 'commercial use NOT cleared' })
          ]),
          data.license && data.license.notes ? el('div', { class: 'tiny dim', text: data.license.notes }) : null,
          reference ? el('div', { class: 'callout warn mt8' }, [
            el('div', { class: 'bold', text: 'Stands in for: ' + reference.name }),
            el('div', { class: 'tiny mt4', text: (reference.license ? reference.license.name + ' — commercial use ' + (reference.license.allowsCommercial ? 'allowed' : 'not allowed') : 'license varies') }),
            reference.url ? el('div', { class: 'tiny', html: '<a href="' + reference.url + '" target="_blank" rel="noreferrer">' + reference.url + '</a>' }) : null,
            reference.note ? el('div', { class: 'tiny faint mt4', text: reference.note }) : null
          ]) : null,
          el('div', { class: 'row gap4 mt8' }, [
            el('button', { class: 'xs ghost', text: 'Set license', onclick: () => openLicenseDialog(data) })
          ])
        ])
      ])
    ]);

    const right = el('div', { class: 'col gap8' }, [
      el('div', { class: 'panel' }, [
        el('div', { class: 'panel-head' }, [el('h3', { text: 'Class balance' })]),
        el('div', { class: 'panel-body tight' }, [classBalanceView(data)])
      ]),
      el('div', { class: 'panel' }, [
        el('div', { class: 'panel-head' }, [el('h3', { text: 'Feature distribution' }), el('div', { class: 'spacer' }), el('span', { class: 'tiny faint', text: (stats.featureDims ? stats.featureDims.length : 0) + ' sampled dims' })]),
        el('div', { class: 'panel-body tight' }, [featureStatsView(data, stats)])
      ]),
      el('div', { class: 'panel' }, [
        el('div', { class: 'panel-head' }, [el('h3', { text: 'Splits' })]),
        el('div', { class: 'panel-body tight' }, [splitsSummaryView(data)])
      ]),
      data.presetNotes ? el('div', { class: 'callout', text: data.presetNotes }) : null
    ]);
    return el('div', { class: 'split data' }, [left, right]);
  }

  function featureStatsView(data, stats) {
    const host = el('div', { class: 'col gap6' });
    const dims = stats.featureDims || [];
    if (!dims.length) { host.appendChild(el('div', { class: 'tiny faint', text: 'No feature statistics available.' })); return host; }
    const means = dims.map((d) => d.mean);
    const stds = dims.map((d) => d.std);
    const canvas = el('canvas', { style: { width: '100%', height: '86px' } });
    histogram(canvas, stds, { bins: 28, color: '#4fa3a5', background: '#191c1f' });
    host.appendChild(el('div', { class: 'tiny upper faint', text: 'per-dimension standard deviation' }));
    host.appendChild(canvas);
    const canvas2 = el('canvas', { style: { width: '100%', height: '86px' } });
    histogram(canvas2, means, { bins: 28, color: '#e0a33e', background: '#191c1f' });
    host.appendChild(el('div', { class: 'tiny upper faint', text: 'per-dimension mean' }));
    host.appendChild(canvas2);
    const dead = dims.filter((d) => d.std < 1e-9).length;
    host.appendChild(el('div', { class: 'tiny ' + (dead ? 'amber' : 'faint'), text: dead + ' of ' + dims.length + ' sampled dimensions are constant (zero variance).' }));
    return host;
  }

  function splitsSummaryView(data) {
    const host = el('div', { class: 'col gap6' });
    const counts = splitCounts(data);
    if (!counts) { host.appendChild(el('div', { class: 'tiny faint', text: 'No splits.' })); return host; }
    const total = Math.max(1, counts.train + counts.val + counts.test);
    host.appendChild(el('div', { class: 'split-viz' }, [
      el('i', { style: { width: (counts.train / total * 100) + '%', background: '#4fa3a5' } }),
      el('i', { style: { width: (counts.val / total * 100) + '%', background: '#e0a33e' } }),
      el('i', { style: { width: (counts.test / total * 100) + '%', background: '#5b87b5' } })
    ]));
    host.appendChild(el('div', { class: 'row between tiny' }, [
      el('span', { class: 'teal', text: 'train ' + counts.train }),
      el('span', { class: 'amber', text: 'val ' + counts.val }),
      el('span', { class: 'blue', text: 'test ' + counts.test })
    ]));
    const labels = data.labels && data.labels.length ? labelVectorFor(data) : null;
    const summary = splitSummary(data.splits, labels);
    const table = el('table', { class: 'data compact' }, [
      el('thead', {}, el('tr', {}, [el('th', { text: 'split' }), el('th', { class: 'n', text: 'n' })].concat((labels ? uniqueLabels(data) : []).slice(0, 8).map((l) => el('th', { class: 'n', text: String(l).slice(0, 7) })))))
    ]);
    const tbody = el('tbody');
    for (const key of ['train', 'val', 'test']) {
      const row = summary[key];
      const tr = el('tr', {}, [el('td', { text: key }), el('td', { class: 'n', text: String(row.n) })]);
      if (labels) {
        for (const name of uniqueLabels(data).slice(0, 8)) tr.appendChild(el('td', { class: 'n faint', text: String(row.counts[name] || 0) }));
      }
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    host.appendChild(el('div', { class: 'scroll', style: { maxHeight: '150px' } }, [table]));
    if (!counts.val) host.appendChild(el('div', { class: 'callout bad', text: 'There is no validation split. Early stopping and model selection would be blind.' }));
    if (!counts.test) host.appendChild(el('div', { class: 'callout warn', text: 'There is no held-out test split. Reported metrics would not be an honest generalisation estimate.' }));
    return host;
  }

  function uniqueLabels(data) {
    const out = [];
    for (const l of data.labels) if (out.indexOf(l) < 0) out.push(l);
    return out;
  }

  function labelVectorFor(data) {
    try {
      const v = labelVector(data);
      return data.labels && data.labels.length ? Array.from(v).map((i) => data.labels[i] !== undefined ? data.labels[i] : String(i)) : v;
    } catch (e) {
      return null;
    }
  }

  function openLicenseDialog(data) {
    const presets = [
      { id: 'cc0', name: 'CC0 1.0', allowsCommercial: true, notes: 'Public domain dedication.' },
      { id: 'cc-by-4.0', name: 'CC BY 4.0', allowsCommercial: true, notes: 'Attribution required.' },
      { id: 'cc-by-sa-4.0', name: 'CC BY-SA 4.0', allowsCommercial: true, notes: 'Attribution and share-alike.' },
      { id: 'cc-by-nc-4.0', name: 'CC BY-NC 4.0', allowsCommercial: false, notes: 'Non-commercial only.' },
      { id: 'apache-2.0', name: 'Apache 2.0', allowsCommercial: true, notes: 'Permissive with patent grant.' },
      { id: 'cdla-permissive', name: 'CDLA-Permissive 2.0', allowsCommercial: true, notes: 'Community data license, permissive.' },
      { id: 'own-data', name: 'Our own data', allowsCommercial: true, notes: 'Collected in-house; rights held by us.' },
      { id: 'user-supplied', name: 'Supplied by the importer', allowsCommercial: true, notes: 'You confirmed you hold the rights to this data.' }
    ];
    let chosen = data.license || presets[0];
    const nameInput = el('input', { type: 'text', value: chosen.name, style: { width: '100%' } });
    const notesInput = el('input', { type: 'text', value: chosen.notes || '', style: { width: '100%' } });
    const commercial = el('input', { type: 'checkbox', checked: !!chosen.allowsCommercial });
    const list = el('div', { class: 'col gap4' }, presets.map((p) => el('button', {
      class: 'sm ghost', style: { textAlign: 'left' }, text: p.name + ' — ' + (p.allowsCommercial ? 'commercial ok' : 'non-commercial'),
      onclick: () => { chosen = p; nameInput.value = p.name; notesInput.value = p.notes; commercial.checked = p.allowsCommercial; }
    })));
    const handle = modal('Dataset license', [
      el('div', { class: 'split', style: { gridTemplateColumns: '1fr 1fr' } }, [
        el('div', { class: 'col gap6' }, [
          el('label', { class: 'field' }, [el('span', { class: 'lbl', text: 'license name' }), nameInput]),
          el('label', { class: 'field' }, [el('span', { class: 'lbl', text: 'notes' }), notesInput]),
          el('label', { class: 'inline' }, [commercial, el('span', { text: 'permits commercial use' })]),
          el('div', { class: 'tiny faint', text: 'The license travels with the dataset digest and appears in every model card you export.' })
        ]),
        list
      ])
    ], [
      el('button', { class: 'ghost', text: 'Cancel', onclick: () => handle.close() }),
      el('button', {
        class: 'primary', text: 'Save license', onclick: () => {
          data.license = { id: chosen.id || 'custom', name: nameInput.value, allowsCommercial: commercial.checked, notes: notesInput.value };
          handle.close();
          ctx.touchDataset();
          ctx.ok('License set to "' + data.license.name + '".');
          render();
        }
      })
    ], { width: '620px' });
  }

  function renderSamples() {
    const data = ds();
    const per = shapeSize(data.xShape);
    const yPer = shapeSize(data.yShape || [1]);
    const indices = filteredIndices(data);
    const pageCount = Math.max(1, Math.ceil(indices.length / PAGE_SIZE));
    if (page >= pageCount) page = pageCount - 1;
    const slice = indices.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
    const host = el('div', { class: 'col gap8' });
    const controls = el('div', { class: 'toolbar' }, [
      el('span', { class: 'tiny faint', text: indices.length + ' of ' + data.count + ' samples' }),
      selectInline('split', filterSplit, [{ value: 'all', label: 'all splits' }, { value: 'train', label: 'train' }, { value: 'val', label: 'val' }, { value: 'test', label: 'test' }], (v) => { filterSplit = v; page = 0; render(); }),
      selectInline('label', filterLabel, [{ value: 'all', label: 'all labels' }].concat(uniqueLabels(data).map((l) => ({ value: String(l), label: String(l) }))), (v) => { filterLabel = v; page = 0; render(); }),
      el('input', { type: 'search', placeholder: 'search text or metadata', value: filterText, style: { width: '190px' }, oninput: (event) => { filterText = event.target.value.toLowerCase(); page = 0; renderBody(); } }),
      el('div', { class: 'spacer' }),
      el('button', { class: 'sm ghost', text: 'Select none', onclick: () => { selected = new Set(); renderBody(); } }),
      el('button', { class: 'sm ghost', text: 'Remove selected', disabled: !selected.size, onclick: () => removeSelected() }),
      el('span', { class: 'tiny faint', text: 'page ' + (page + 1) + '/' + pageCount }),
      el('button', { class: 'sm ghost', text: '‹', onclick: () => { page = Math.max(0, page - 1); renderBody(); } }),
      el('button', { class: 'sm ghost', text: '›', onclick: () => { page = Math.min(pageCount - 1, page + 1); renderBody(); } })
    ]);
    host.appendChild(controls);
    const tableHost = el('div', { id: 'samplesTable' });
    host.appendChild(tableHost);
    renderTable(tableHost, data, slice, per, yPer);
    return host;
  }

  function renderBody() {
    const host = root.querySelector('#samplesTable');
    if (!host) { render(); return; }
    const data = ds();
    const indices = filteredIndices(data);
    const pageCount = Math.max(1, Math.ceil(indices.length / PAGE_SIZE));
    if (page >= pageCount) page = pageCount - 1;
    const slice = indices.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
    renderTable(host, data, slice, shapeSize(data.xShape), shapeSize(data.yShape || [1]));
  }

  function renderTable(host, data, slice, per, yPer) {
    clear(host);
    const splitOf = splitLookup(data);
    const table = el('table', { class: 'data compact' }, [
      el('thead', {}, el('tr', {}, [
        el('th', { text: '' }), el('th', { class: 'n', text: '#' }), el('th', { text: 'preview' }),
        el('th', { text: 'text / metadata' }), el('th', { text: 'label' }), el('th', { text: 'split' }), el('th', { class: 'n', text: '‖x‖' })
      ]))
    ]);
    const tbody = el('tbody');
    for (const index of slice) {
      const text = sampleText(data, index);
      const meta = data.meta && data.meta[index] ? data.meta[index] : null;
      const metaBits = meta ? Object.keys(meta).filter((k) => k !== 'text' && k !== 'preview').slice(0, 3).map((k) => k + ': ' + String(meta[k]).slice(0, 26)).join(' · ') : '';
      let norm = 0;
      const stride = Math.max(1, Math.floor(per / 256));
      for (let f = 0; f < per; f += stride) { const v = data.x[index * per + f]; norm += v * v; }
      norm = Math.sqrt(norm);
      const checkbox = el('input', {
        type: 'checkbox', checked: selected.has(index),
        onchange: (event) => { if (event.target.checked) selected.add(index); else selected.delete(index); }
      });
      const preview = el('td');
      try { preview.appendChild(samplePreviewCell(data, index, { size: 84 })); } catch (e) { preview.appendChild(el('span', { class: 'tiny faint', text: '—' })); }
      const labelCell = el('td');
      if (data.labels && data.labels.length && yPer === 1) {
        labelCell.appendChild(el('select', {
          style: { maxWidth: '130px' },
          onchange: (event) => {
            data.y[index * yPer] = parseInt(event.target.value, 10);
            ctx.touchDataset();
            ctx.info('Relabelled sample #' + index + ' → ' + data.labels[data.y[index * yPer]] + '.');
          }
        }, data.labels.map((l, i) => el('option', { value: String(i), text: String(l), selected: Math.round(data.y[index * yPer]) === i }))));
      } else if (data.labels && data.labels.length && yPer > 1) {
        labelCell.appendChild(el('div', { class: 'tiny', text: labelName(data, index) || '—' }));
      } else {
        const values = [];
        for (let k = 0; k < yPer; k++) values.push(Number(data.y[index * yPer + k]).toFixed(3));
        labelCell.appendChild(el('span', { class: 'mono tiny', text: values.join(', ') }));
      }
      tbody.appendChild(el('tr', {}, [
        el('td', {}, [checkbox]),
        el('td', { class: 'n faint', text: String(index) }),
        preview,
        el('td', {}, [
          el('div', { class: 'ell', style: { maxWidth: '320px' }, text: text || '—' }),
          metaBits ? el('div', { class: 'tiny faint ell', style: { maxWidth: '320px' }, text: metaBits }) : null
        ]),
        labelCell,
        el('td', {}, [el('span', { class: 'badge ' + splitTone(splitOf[index]), text: splitOf[index] || '—' })]),
        el('td', { class: 'n', text: norm.toFixed(3) })
      ]));
    }
    table.appendChild(tbody);
    host.appendChild(el('div', { class: 'scroll', style: { maxHeight: '58vh' } }, [table]));
  }

  function splitTone(name) {
    return name === 'train' ? 'teal' : name === 'val' ? 'amber' : name === 'test' ? 'blue' : 'plain';
  }

  function splitLookup(data) {
    const out = {};
    if (!data.splits) return out;
    for (const i of data.splits.train || []) out[i] = 'train';
    for (const i of data.splits.val || []) out[i] = 'val';
    for (const i of data.splits.test || []) out[i] = 'test';
    return out;
  }

  function filteredIndices(data) {
    const splitOf = splitLookup(data);
    const out = [];
    const needle = filterText.trim();
    for (let i = 0; i < data.count; i++) {
      if (filterSplit !== 'all' && splitOf[i] !== filterSplit) continue;
      if (filterLabel !== 'all') {
        const name = labelName(data, i);
        if (String(name) !== filterLabel) continue;
      }
      if (needle) {
        const text = sampleText(data, i);
        const meta = data.meta && data.meta[i] ? JSON.stringify(data.meta[i]).toLowerCase() : '';
        if (!(text && text.toLowerCase().includes(needle)) && !meta.includes(needle)) continue;
      }
      out.push(i);
    }
    return out;
  }

  function removeSelected() {
    const data = ds();
    const keep = [];
    for (let i = 0; i < data.count; i++) if (!selected.has(i)) keep.push(i);
    if (keep.length === data.count) return;
    const removed = data.count - keep.length;
    const handle = modal('Remove samples', [
      el('p', { text: 'Remove ' + removed + ' selected sample(s) from "' + data.name + '"? Splits are rebuilt afterwards.' })
    ], [
      el('button', { class: 'ghost', text: 'Cancel', onclick: () => handle.close() }),
      el('button', {
        class: 'danger', text: 'Remove', onclick: () => {
          handle.close();
          try {
            const next = subsetDataset(data, Int32Array.from(keep));
            next.name = data.name;
            next.license = data.license;
            next.augmentation = data.augmentation;
            next.splitMethod = data.splitMethod;
            next.splitRatios = data.splitRatios;
            next.splitSeed = data.splitSeed;
            buildSplits(next, data.splitMethod, data.splitRatios, data.splitSeed);
            next.provenance = Object.assign({}, data.provenance, { edited: (data.provenance && data.provenance.edited || 0) + 1, removedSamples: removed });
            selected = new Set();
            ctx.setDataset(next, { keepId: true });
            ctx.ok('Removed ' + removed + ' samples; ' + next.count + ' remain.');
            render();
          } catch (e) {
            ctx.error('Could not remove samples: ' + e.message);
          }
        }
      })
    ], { width: '460px' });
  }

  function renderSplits() {
    const data = ds();
    const host = el('div', { class: 'col gap8' });
    const options = {
      method: data.splitMethod || 'stratified',
      ratios: (data.splitRatios || [0.7, 0.15, 0.15]).slice(),
      seed: data.splitSeed === undefined ? 20260101 : data.splitSeed,
      folds: 5,
      valFold: 0,
      testFold: 1
    };
    const methods = [
      { value: 'stratified', label: 'stratified — keeps the class ratio in every split' },
      { value: 'random', label: 'random — plain shuffle' },
      { value: 'group', label: 'group — no group appears in two splits (needs groups)', disabled: !data.groups },
      { value: 'temporal', label: 'temporal — first N% train, then val, then test (time-ordered data)' }
    ];
    const left = el('div', { class: 'col gap8' }, [
      el('div', { class: 'panel' }, [
        el('div', { class: 'panel-head' }, [el('h3', { text: 'Split configuration' })]),
        el('div', { class: 'panel-body' }, [
          el('div', { class: 'config-grid' }, [
            el('label', { class: 'field full' }, [
              el('span', { class: 'lbl', text: 'method' }),
              el('select', { onchange: (event) => { options.method = event.target.value; render(); } }, methods.map((m) => el('option', { value: m.value, text: m.label, selected: options.method === m.value, disabled: !!m.disabled })))
            ]),
            ratioField('train', 0), ratioField('val', 1), ratioField('test', 2),
            el('label', { class: 'field' }, [
              el('span', { class: 'lbl', text: 'seed' }),
              el('input', { type: 'number', value: options.seed, onchange: (event) => { options.seed = parseInt(event.target.value, 10) || 0; } })
            ])
          ]),
          el('div', { class: 'row gap6 mt8 wrap' }, [
            el('button', { class: 'primary', text: 'Rebuild splits', onclick: () => applySplits() }),
            el('button', { class: 'ghost', text: '70 / 15 / 15', onclick: () => { options.ratios = [0.7, 0.15, 0.15]; render(); } }),
            el('button', { class: 'ghost', text: '80 / 10 / 10', onclick: () => { options.ratios = [0.8, 0.1, 0.1]; render(); } }),
            el('button', { class: 'ghost', text: '60 / 20 / 20', onclick: () => { options.ratios = [0.6, 0.2, 0.2]; render(); } })
          ]),
          el('div', { class: 'tiny faint mt8', text: 'A held-out test split is mandatory for an honest number. Validation drives early stopping and checkpoint selection; test is measured once.' })
        ])
      ]),
      el('div', { class: 'panel' }, [
        el('div', { class: 'panel-head' }, [el('h3', { text: 'K-fold cross validation' })]),
        el('div', { class: 'panel-body' }, [
          el('div', { class: 'config-grid' }, [
            el('label', { class: 'field' }, [
              el('span', { class: 'lbl', text: 'folds' }),
              el('input', { type: 'number', min: 2, max: 20, value: options.folds, onchange: (event) => { options.folds = Math.max(2, Math.min(20, parseInt(event.target.value, 10) || 5)); renderFolds(); } })
            ]),
            el('label', { class: 'field' }, [
              el('span', { class: 'lbl', text: 'validation fold' }),
              el('select', { onchange: (event) => { options.valFold = parseInt(event.target.value, 10); renderFolds(); } }, Array.from({ length: options.folds }, (v, i) => el('option', { value: String(i), text: 'fold ' + (i + 1), selected: options.valFold === i })))
            ]),
            el('label', { class: 'field' }, [
              el('span', { class: 'lbl', text: 'test fold' }),
              el('select', { onchange: (event) => { options.testFold = parseInt(event.target.value, 10); renderFolds(); } }, Array.from({ length: options.folds }, (v, i) => el('option', { value: String(i), text: 'fold ' + (i + 1), selected: options.testFold === i })))
            ])
          ]),
          el('div', { class: 'row gap6 mt8' }, [
            el('button', { class: 'teal', text: 'Apply folds as splits', onclick: () => applyFolds() }),
            el('span', { class: 'tiny faint', id: 'foldInfo', text: '' })
          ])
        ])
      ])
    ]);

    function ratioField(label, index) {
      return el('label', { class: 'field' }, [
        el('span', { class: 'lbl', text: label + ' ratio' }),
        el('input', {
          type: 'number', min: 0, max: 1, step: 0.01, value: options.ratios[index],
          onchange: (event) => { options.ratios[index] = Math.max(0, Math.min(1, parseFloat(event.target.value) || 0)); }
        })
      ]);
    }

    function renderFolds() {
      const info = host.querySelector('#foldInfo');
      if (info) info.textContent = options.valFold === options.testFold ? 'validation and test must be different folds' : 'val fold ' + (options.valFold + 1) + ', test fold ' + (options.testFold + 1) + ', train = the rest';
    }

    function applySplits() {
      try {
        buildSplits(data, options.method, options.ratios, options.seed);
        ctx.touchDataset();
        ctx.ok('Rebuilt splits with ' + options.method + ' ' + options.ratios.map((r) => r.toFixed(2)).join('/') + ' seed ' + options.seed + ' → ' + data.splits.train.length + '/' + data.splits.val.length + '/' + data.splits.test.length + '.');
        render();
      } catch (e) {
        ctx.error('Could not rebuild splits: ' + e.message);
      }
    }

    function applyFolds() {
      if (options.valFold === options.testFold) { ctx.warn('Pick different folds for validation and test.'); return; }
      try {
        const labels = data.labels && data.labels.length ? labelVector(data) : null;
        const assign = kFold(data.count, options.folds, options.seed, labels);
        const train = [];
        const val = [];
        const test = [];
        for (let i = 0; i < data.count; i++) {
          if (assign[i] === options.valFold) val.push(i);
          else if (assign[i] === options.testFold) test.push(i);
          else train.push(i);
        }
        data.splits = { train: Int32Array.from(train), val: Int32Array.from(val), test: Int32Array.from(test) };
        data.splitMethod = 'kfold-' + options.folds;
        data.splitSeed = options.seed;
        data.splitRatios = [train.length / data.count, val.length / data.count, test.length / data.count];
        ctx.touchDataset();
        ctx.ok('Applied ' + options.folds + '-fold assignment: ' + train.length + '/' + val.length + '/' + test.length + '.');
        render();
      } catch (e) {
        ctx.error('Could not apply folds: ' + e.message);
      }
    }

    const right = el('div', { class: 'col gap8' }, [
      el('div', { class: 'panel' }, [
        el('div', { class: 'panel-head' }, [el('h3', { text: 'Current assignment' })]),
        el('div', { class: 'panel-body tight' }, [splitsSummaryView(data)])
      ]),
      el('div', { class: 'panel' }, [
        el('div', { class: 'panel-head' }, [el('h3', { text: 'Leakage checks' })]),
        el('div', { class: 'panel-body tight' }, [leakageView(data)])
      ])
    ]);

    host.appendChild(el('div', { class: 'split data' }, [left, right]));
    renderFolds();
    return host;
  }

  function leakageView(data) {
    const host = el('div', { class: 'col gap6' });
    const report = data.quality || qualityReport(data);
    const dup = report.duplicates || {};
    const overlap = splitOverlap(data);
    host.appendChild(el('div', { class: 'kv' }, [
      el('dt', { text: 'exact duplicates' }), el('dd', { text: String(dup.exact || 0) }),
      el('dt', { text: 'cross-split duplicates' }), el('dd', { text: String(dup.crossSplit || 0) }),
      el('dt', { text: 'conflicting labels' }), el('dd', { text: String(dup.conflicting || 0) }),
      el('dt', { text: 'index overlap' }), el('dd', { text: overlap + ' shared indices' })
    ]));
    const bad = (dup.crossSplit || 0) > 0 || overlap > 0;
    host.appendChild(el('div', { class: 'callout ' + (bad ? 'bad' : 'good'), text: bad
      ? 'Train and test share content. Any metric you report is optimistic and not defensible.'
      : 'No leakage detected between the splits.' }));
    host.appendChild(el('div', { class: 'row gap4' }, [
      el('button', { class: 'xs ghost', text: 'Re-check', onclick: () => { data.quality = qualityReport(data); ctx.touchDataset(); render(); } })
    ]));
    return host;
  }

  function splitOverlap(data) {
    if (!data.splits) return 0;
    const seen = new Set(data.splits.train || []);
    let overlap = 0;
    for (const i of data.splits.val || []) if (seen.has(i)) overlap++;
    for (const i of data.splits.test || []) if (seen.has(i)) overlap++;
    return overlap;
  }

  function selectInline(label, value, choices, onChange) {
    return el('label', { class: 'inline' }, [
      el('span', { class: 'tiny faint', text: label }),
      el('select', { onchange: (event) => onChange(event.target.value) }, choices.map((c) => el('option', { value: c.value, text: c.label, selected: String(c.value) === String(value) })))
    ]);
  }

  function renderDatasetView() {
    const data = ds();
    const host = el('div', { class: 'col gap8' });
    host.appendChild(tabs());
    if (tab === 'overview') host.appendChild(renderOverview());
    else if (tab === 'samples') host.appendChild(renderSamples());
    else if (tab === 'splits') host.appendChild(renderSplits());
    else if (tab === 'augment') host.appendChild(augmentView(ctx, data, () => ctx.touchDataset()));
    else if (tab === 'quality') host.appendChild(qualityView(ctx, data));
    else if (tab === 'export') host.appendChild(exportDataView(ctx, data));
    return host;
  }

  function render() {
    renderToolbar();
    clear(body);
    if (galleryOpen || !ds()) body.appendChild(renderGallery());
    else body.appendChild(renderDatasetView());
  }

  function refresh(reason) {
    if (reason === 'progress' || reason === 'log') return;
    render();
  }

  render();
  return { root, refresh, render };
}
