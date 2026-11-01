import { el, markSvg, setFavicon } from './ui/dom.js';
import { PRESET_MODEL_LIST, PRESET_FAMILIES, buildPresetGraph } from './presets/models.js';
import { PRESET_DATASET_LIST, datasetLicense, datasetSizeEstimate } from './presets/datasets.js';
import { analyze } from './core/model.js';
import { TASKS } from './core/tasks.js';
import { HEADLINE_METRIC, METRIC_META, metricMeta, metricsForTask } from './presets/metrics.js';
import { EXPORT_FORMATS, precisionsFor } from './export/pipeline.js';
import { LAYERS, LAYER_GROUPS } from './core/layers.js';
import { OPTIMIZERS, SCHEDULES } from './core/optimize.js';
import { AUG_OPS } from './data/augment.js';
import { IMPORT_FORMATS, EXPORT_FORMATS as DATA_FORMATS } from './data/import.js';
import { PRECISIONS } from './core/quant.js';
import { TOOL_NAME, TOOL_VERSION } from './export/card.js';
import { BUNDLE_VERSION } from './export/bundle.js';

const FAMILY_LABEL = {};
for (const family of PRESET_FAMILIES) FAMILY_LABEL[family.key] = family.label;

const MODALITY_LABEL = {
  text: 'Text',
  audio: 'Audio',
  image: 'Image',
  sensor: 'Sensor',
  series: 'Series',
  tabular: 'Tabular',
  bytes: 'Bytes'
};

function taskLabel(key) {
  const task = TASKS[key];
  return task && task.label ? task.label : key;
}

function metricLabel(key) {
  const meta = metricMeta(key);
  return meta && meta.label ? meta.label : key;
}

function shortUrl(url) {
  try {
    const parsed = new URL(url);
    const path = parsed.pathname === '/' ? '' : parsed.pathname;
    return parsed.host + path;
  } catch (e) {
    return url;
  }
}

function shapeText(shape) {
  if (!shape || !shape.length) return '—';
  return shape.join(' × ');
}

function whole(n) {
  return n === null || n === undefined || Number.isNaN(n) ? '—' : Math.round(n).toLocaleString('en-US');
}

function dataTable(columns, rows) {
  return el('div', { class: 'tablewrap' }, [
    el('table', { class: 'data' }, [
      el('thead', {}, [
        el('tr', {}, columns.map((column) => el('th', {
          class: column.numeric ? 'n' : (column.wrap ? 'wrapcell' : ''),
          text: column.label
        })))
      ]),
      el('tbody', {}, rows)
    ])
  ]);
}

function cell(content, className) {
  return el('td', { class: className || '' }, Array.isArray(content) ? content : [content]);
}

function badge(text, tone) {
  return el('span', { class: 'badge' + (tone ? ' ' + tone : ''), text });
}

function sub(text) {
  return text ? el('span', { class: 'sub', text }) : null;
}

function sectionHead(index, title, count) {
  return el('h2', {}, [
    el('span', { class: 'idx', text: index }),
    el('span', { text: title }),
    count ? el('span', { class: 'count', text: count }) : null
  ]);
}

function block(index, title, count, children) {
  return el('section', { class: 'block', id: title.toLowerCase().replace(/[^a-z]+/g, '-') }, [sectionHead(index, title, count)].concat(children));
}

const ANALYZED = {};

function analyzedPreset(preset) {
  if (Object.prototype.hasOwnProperty.call(ANALYZED, preset.key)) return ANALYZED[preset.key];
  let info = null;
  try {
    info = analyze(buildPresetGraph(preset));
  } catch (e) {
    info = null;
  }
  ANALYZED[preset.key] = info;
  return info;
}

function modelRows() {
  const rows = [];
  const ordered = PRESET_MODEL_LIST.slice().sort((a, b) => {
    const fa = PRESET_FAMILIES.findIndex((f) => f.key === a.family);
    const fb = PRESET_FAMILIES.findIndex((f) => f.key === b.family);
    if (fa !== fb) return fa - fb;
    return a.name.localeCompare(b.name);
  });
  for (const preset of ordered) {
    const info = analyzedPreset(preset);
    const reference = preset.reference || null;
    rows.push(el('tr', {}, [
      cell([el('span', { class: 'bold', text: preset.name }), sub(preset.tagline)], 'wrapcell'),
      cell(FAMILY_LABEL[preset.family] || preset.family),
      cell([taskLabel(preset.task), sub(preset.head + ' head')]),
      cell([shapeText(preset.input), sub(preset.inputName)], 'n'),
      cell(whole(info ? info.totalParams : null), 'n'),
      cell(whole(info ? info.macs : null), 'n'),
      cell(whole(preset.budget ? preset.budget.target : null), 'n'),
      cell(reference ? [
        reference.params ? whole(reference.params[0]) + ' – ' + whole(reference.params[1]) + ' params in production' : 'reference implementation',
        sub(reference.note || '')
      ] : '—', 'wrapcell')
    ]));
    if (preset.description) rows[rows.length - 1].firstChild.title = preset.description;
  }
  return rows;
}

function datasetRows() {
  const rows = [];
  const ordered = PRESET_DATASET_LIST.slice().sort((a, b) => a.name.localeCompare(b.name));
  for (const preset of ordered) {
    const licence = datasetLicense(preset.key) || {};
    const local = licence.local || null;
    const reference = licence.reference || null;
    const sizes = ['small', 'medium', 'large']
      .map((size) => ({ size, count: datasetSizeEstimate(preset.key, size) }))
      .filter((item) => item.count);
    const knobs = Object.keys(preset.defaults || {})
      .filter((key) => key !== 'size' && key !== 'seed')
      .map((key) => key + ' ' + preset.defaults[key]);
    rows.push(el('tr', {}, [
      cell([el('span', { class: 'bold', text: preset.name }), sub(preset.tagline)], 'wrapcell'),
      cell(MODALITY_LABEL[preset.modality] || preset.modality),
      cell(taskLabel(preset.task)),
      cell([
        sizes.length ? sizes.map((item) => item.size.slice(0, 1) + ' ' + item.count.toLocaleString('en-US')).join('  ·  ') : '—',
        sub('split ' + (preset.splits && preset.splits.method ? preset.splits.method : 'random') + (preset.splits && preset.splits.ratios ? ' ' + preset.splits.ratios.map((r) => Math.round(r * 100)).join('/') : ''))
      ], 'n'),
      cell(knobs.length ? knobs.join(', ') : '—', 'wrapcell'),
      cell(local ? [
        badge(local.allowsCommercial ? 'commercial use ok' : 'check licence', local.allowsCommercial ? 'green' : 'red'),
        sub(local.name)
      ] : '—'),
      cell(reference ? [
        reference.license && reference.license.allowsCommercial === false
          ? badge('non-commercial · ' + reference.license.name, 'red')
          : el('span', { class: 'tiny faint', text: reference.license ? reference.license.name : 'licence not stated' }),
        el('div', { class: 'sub', text: reference.name }),
        reference.url ? el('span', { class: 'sub' }, [el('a', { href: reference.url, target: '_blank', rel: 'noreferrer', text: shortUrl(reference.url) })]) : null,
        sub(reference.note || '')
      ] : '—', 'wrapcell')
    ]));
    if (preset.notes) rows[rows.length - 1].firstChild.title = preset.notes;
  }
  return rows;
}

function evaluationRows() {
  const rows = [];
  const keys = Object.keys(TASKS);
  for (const key of keys) {
    const task = TASKS[key];
    const headline = HEADLINE_METRIC[key] || null;
    const groups = metricsForTask(key) || [];
    rows.push(el('tr', {}, [
      cell([el('span', { class: 'bold', text: task.label }), sub(key)], 'wrapcell'),
      cell(headline ? [metricLabel(headline), sub(headline)] : '—'),
      cell(groups.map((group) => el('span', { class: 'sub', style: { display: 'block', marginTop: '1px' } }, [
        el('b', { text: group.title + ': ' }),
        document.createTextNode(group.metrics.map((m) => metricLabel(m)).join(', '))
      ])), 'wrapcell')
    ]));
  }
  return rows;
}

function exportRows() {
  return EXPORT_FORMATS.map((format) => el('tr', {}, [
    cell([el('span', { class: 'bold', text: format.label }), ' ', badge(format.extension, 'plain')]),
    cell(precisionsFor(format.key).map((precision) => badge(precision.key || precision, 'teal')), 'wrapcell'),
    cell(format.runtime, 'wrapcell'),
    cell([format.blurb, sub('written to ' + format.directory + '/')], 'wrapcell')
  ]));
}

const STAGES = [
  {
    title: 'Build',
    body: 'A node and edge graph editor with live parameter counts, MACs per sample, activation memory and a budget bar. Presets are graphs: fork one and edit it, or start from an empty canvas with 32 layer types.'
  },
  {
    title: 'Data',
    body: 'Generate a labelled dataset from a preset, or import CSV, TSV, JSONL, JSON, WAV, images, text or a zip of them. Splits are stratified, grouped, temporal or k-fold. The quality tab reports leakage, duplicates, constant and sparse features, label noise, class imbalance and missing licence.'
  },
  {
    title: 'Train',
    body: 'Eight optimizers, ten learning-rate schedules, class weights, EMA, gradient clipping, augmentation, early stopping and a monitored metric you choose. Every epoch records train and validation loss, the monitored metric, learning rate, gradient norm, weight norm and throughput.'
  },
  {
    title: 'Evaluate',
    body: 'Confusion matrix, per-class precision, recall and F1, calibration with ECE, MCE, Brier and log loss, ROC and PR curves, a threshold sweep with FAR and FRR, and the worst individual examples — computed separately on train, validation and test.'
  },
  {
    title: 'Compare',
    body: 'Select any saved runs and line them up: overlaid loss and metric curves, a metric-by-metric table with the best and worst value in each row marked, a config diff, and the reproducibility block — seed, dataset content hash, architecture hash, tool version.'
  },
  {
    title: 'Export',
    body: 'ONNX, TensorFlow.js and safetensors at fp32, fp16, bf16, int8 or uint8. Before you download anything, the accuracy of each quantized variant is measured against fp32 on your own test split, and the delta decides the verdict.'
  },
  {
    title: 'Library',
    body: 'Graphs, datasets, runs and artifacts are stored in IndexedDB with their hashes. The workspace starts empty and nothing is pre-seeded, so what is in there is what you made.'
  }
];

const BUNDLE_FILES = [
  'README.md — the model card, rendered as markdown',
  'model-card.json — the same card as structured data',
  'config/architecture.json — layer table, parameter counts, shapes',
  'config/spec.json — the graph itself, re-importable',
  'config/train.json — the full training configuration',
  'config/reproducibility.json — seed, hashes, tool version, timings',
  'metrics/results.json — every metric on every split',
  'metrics/run.json — epoch history and the monitored curve',
  'labels.txt — the class list in index order',
  'dataset/summary.json — counts, splits, digest, licence',
  'dataset/samples.jsonl — a sample of the training data',
  'dataset/quality.json — the findings from the quality report',
  'onnx/model[-precision].onnx',
  'tfjs/model[-precision].json and .weights.bin',
  'weights/model[-precision].safetensors',
  'MANIFEST.json — sha256 per file plus a tree hash of the bundle'
];

const CARD_SECTIONS = [
  'Summary',
  'Intended use',
  'Architecture',
  'Dataset',
  'Training',
  'Results',
  'Exported artifacts',
  'Quantization',
  'Reproducibility',
  'Loading the model — ONNX Runtime, TensorFlow.js and safetensors snippets that run as written',
  'Limitations',
  'Notes from the author',
  'Card provenance'
];

const METHOD = [
  'No build step. The repository is static files; any static host serves it, GitHub Pages included.',
  'No dependencies and no CDN. The zip writer, the DEFLATE decoder, the ONNX protobuf encoder, SHA-256, the FFT and every chart are written out in this repository.',
  'No uploads. There is no server to send anything to. Datasets are generated locally or read from files you pick.',
  'Deterministic. A run records its seed, the dataset content hash, the architecture hash and the whole config, so a number can be reproduced or refuted later.',
  'Blank on first load. The workspace opens empty; presets are one click away but nothing is waiting for you.',
  'JavaScript only, ES modules, no TypeScript and no transpiler anywhere in the tree.'
];

const LIMITS = [
  'Small is a constraint, not a style. A browser tab trains 2k to 400k parameters in seconds, so that is the range on offer; each preset names what the same job usually runs on in production, and it is bigger.',
  'Generated datasets are stand-ins. They have the right shape, label distribution and difficulty knobs, which makes them good for wiring up a pipeline and for regression tests. Ship on real data.',
  'PESQ is approximated in-browser and labelled "PESQ (approx.)" everywhere it appears. The reference implementation is not something a static page can carry.',
  'Quantization deltas are only as good as the split they are measured on. On sixty test samples a two percent delta is inside the noise; the table prints the sample count next to the verdict for that reason.',
  'Several reference datasets named beside the presets are non-commercial. The licence column exists so that gets checked before shipping, not after.',
  'Browser memory is the ceiling. Datasets in the hundreds of megabytes will make the tab unhappy, and training stops when the tab is backgrounded on some browsers.'
];

function facts() {
  const items = [
    { v: PRESET_MODEL_LIST.length, k: 'model presets' },
    { v: PRESET_DATASET_LIST.length, k: 'dataset presets' },
    { v: Object.keys(LAYERS).length, k: 'layer types' },
    { v: Object.keys(TASKS).length, k: 'task types' },
    { v: Object.keys(METRIC_META).length, k: 'metric definitions' },
    { v: EXPORT_FORMATS.length, k: 'export formats' },
    { v: PRECISIONS.length, k: 'precisions' },
    { v: 0, k: 'dependencies' },
    { v: 0, k: 'uploads' }
  ];
  return el('div', { class: 'facts' }, items.map((item) => el('div', { class: 'fact' }, [
    el('div', { class: 'v', text: String(item.v) }),
    el('div', { class: 'k', text: item.k })
  ])));
}

function counts() {
  const items = [
    { v: Object.keys(LAYERS).length, k: 'layer types in the palette, in ' + LAYER_GROUPS.length + ' groups' },
    { v: Object.keys(OPTIMIZERS).length, k: 'optimizers' },
    { v: Object.keys(SCHEDULES).length, k: 'learning-rate schedules' },
    { v: AUG_OPS.length, k: 'augmentation operations' },
    { v: IMPORT_FORMATS.length, k: 'import formats' },
    { v: DATA_FORMATS.length, k: 'dataset export formats' },
    { v: PRECISIONS.length, k: 'weight precisions' },
    { v: Object.keys(METRIC_META).length, k: 'named metrics with direction and explanation' },
    { v: Object.keys(TASKS).length, k: 'task types with their own loss, predict and metric set' },
    { v: BUNDLE_VERSION, k: 'export bundle format version' }
  ];
  return el('div', { class: 'counts' }, items.map((item) => el('div', { class: 'c' }, [
    el('span', { class: 'v', text: String(item.v) }),
    el('span', { class: 'k', text: item.k })
  ])));
}

function cardList(title, items) {
  return el('div', { class: 'card' }, [
    el('h4', { text: title }),
    el('ul', {}, items.map((item) => el('li', { text: item })))
  ]);
}

function topbar() {
  const links = [
    { href: '#workflow', label: 'Workflow' },
    { href: '#model-presets', label: 'Model presets' },
    { href: '#dataset-presets', label: 'Dataset presets' },
    { href: '#evaluation', label: 'Evaluation' },
    { href: '#export', label: 'Export' },
    { href: '#what-is-in-the-box', label: 'Contents' },
    { href: '#limits', label: 'Limits' }
  ];
  return el('header', { class: 'topbar' }, [
    markSvg(20),
    el('div', { class: 'wordmark' }, [document.createTextNode('Rootism Labs '), el('span', { text: '· model workbench' })]),
    el('div', { class: 'spacer' }),
    el('nav', {}, links.map((link) => el('a', { href: link.href, text: link.label }))),
    el('a', { class: 'btn primary', href: 'studio.html', text: 'Open the workspace' })
  ]);
}

function hero() {
  return el('div', { class: 'hero' }, [
    el('div', { class: 'kicker', text: 'runs in the browser · nothing installed · nothing uploaded' }),
    el('h1', { text: 'Build, train, measure and export small models that you can put in a product.' }),
    el('p', {}, [
      document.createTextNode('Rootism Labs is a workbench for neural networks in the range where a browser tab is enough: a few thousand to a few hundred thousand parameters. You draw or pick an architecture, generate or import a dataset, train it with a configuration you control, measure it with the metric that belongs to the task, and export it as '),
      el('strong', { text: 'ONNX, TensorFlow.js or safetensors' }),
      document.createTextNode(' with a model card and a manifest of hashes attached.')
    ]),
    el('p', { text: 'There is no account, no server and no build step. The page is static JavaScript modules; training, evaluation, quantization and packaging all happen on your machine, and the files land in your download folder. Every number the tool reports comes with the seed, the dataset hash and the configuration that produced it, because a metric you cannot reproduce is an anecdote.' }),
    el('div', { class: 'cta-row' }, [
      el('a', { class: 'btn primary', href: 'studio.html', text: 'Open the workspace' }),
      el('a', { class: 'btn', href: '#model-presets', text: 'See the 27 presets' }),
      el('span', { class: 'hint', text: 'The workspace opens empty. Presets, generators and imports are inside.' })
    ]),
    facts()
  ]);
}

function workflow() {
  return block('01', 'Workflow', STAGES.length + ' stages', [
    el('p', { class: 'intro', text: 'The panels are laid out in the order the work happens. Each one is reachable from the rail on the left of the workspace, and each keeps its own state so you can move between them mid-run.' }),
    el('div', { class: 'steps' }, STAGES.map((stage, index) => el('div', { class: 'step' }, [
      el('div', { class: 'n', text: String(index + 1).padStart(2, '0') }),
      el('div', { class: 't', text: stage.title }),
      el('div', { class: 'd', text: stage.body })
    ])))
  ]);
}

function models() {
  const usable = PRESET_MODEL_LIST
    .map((preset) => analyzedPreset(preset))
    .filter((info) => info && info.valid)
    .map((info) => info.totalParams);
  const lo = usable.length ? Math.min.apply(null, usable) : 0;
  const hi = usable.length ? Math.max.apply(null, usable) : 0;
  return block('02', 'Model presets', PRESET_MODEL_LIST.length + ' graphs', [
    el('p', { class: 'intro', text: 'A preset is a saved graph, not a black box: opening one puts it in the editor where every layer, its configuration and its edges are editable. Parameter counts and MACs below are computed from the graphs at page load, so this table cannot drift from the tool. The last column says what the same job usually runs on in production, which is larger — that gap is the point of the exercise.' }),
    el('p', { class: 'intro', text: 'Range across the ' + usable.length + ' presets: ' + whole(lo) + ' to ' + whole(hi) + ' parameters. Hover a preset name for the long description.' }),
    dataTable([
      { label: 'preset' },
      { label: 'family' },
      { label: 'task' },
      { label: 'input per sample', numeric: true },
      { label: 'params', numeric: true },
      { label: 'MACs', numeric: true },
      { label: 'budget', numeric: true },
      { label: 'production reference', wrap: true }
    ], modelRows())
  ]);
}

function datasets() {
  return block('03', 'Dataset presets', PRESET_DATASET_LIST.length + ' generators', [
    el('p', { class: 'intro', text: 'Each preset dataset is generated in the browser from a deterministic seed, so the same seed gives the same samples, the same content hash and the same trained model. Nothing third-party is embedded, which is why the local licence is yours. The right-hand column names the public dataset the task is modelled on and that dataset’s licence, because the generated stand-in is not a substitute for the real thing — and the real thing is sometimes not commercial. References marked in red are the ones to check with a lawyer before you train on the real data and ship the result.' }),
    el('p', { class: 'intro', text: 'Sizes: s = small, m = medium (the default), l = large. Import is available for CSV, TSV, JSONL, JSON, WAV, images, plain text and zipped archives, and audio imports are converted to the feature layout the task needs.' }),
    dataTable([
      { label: 'preset' },
      { label: 'modality' },
      { label: 'task' },
      { label: 'samples', numeric: true },
      { label: 'generation options', wrap: true },
      { label: 'licence of what you generate' },
      { label: 'modelled on', wrap: true }
    ], datasetRows())
  ]);
}

function evaluation() {
  return block('04', 'Evaluation', Object.keys(TASKS).length + ' task types', [
    el('p', { class: 'intro', text: 'A loss curve says a model optimised; it does not say the model is any good. Each task type carries its own loss, its own prediction decoding and its own metric set, and the evaluation panel draws all of them on the train, validation and test splits separately. Per-class precision and recall, confusion matrices and calibration are always there for classification tasks; the headline metric changes with the task.' }),
    dataTable([
      { label: 'task' },
      { label: 'headline metric' },
      { label: 'also reported', wrap: true }
    ], evaluationRows())
  ]);
}

function exportBlock() {
  return block('05', 'Export', EXPORT_FORMATS.length + ' formats · ' + PRECISIONS.length + ' precisions', [
    el('p', { class: 'intro', text: 'Export is the product. The graph is converted op by op — recurrent layers are unrolled with their real weight layouts, padding is emitted explicitly, batch norm is folded, and int8 and uint8 are written as proper quantize-dequantize pairs with per-output-channel scales rather than as a lossy blob. Each artifact is verified against the in-browser model before it is offered for download, and the quantized variants are measured on your test split so the accuracy cost is a number, not a guess.' }),
    dataTable([
      { label: 'format' },
      { label: 'precisions', wrap: true },
      { label: 'runs on', wrap: true },
      { label: 'what you get', wrap: true }
    ], exportRows()),
    el('h3', { text: 'What lands in the download folder' }),
    el('div', { class: 'cols' }, [
      cardList('Bundle contents (zip, format v' + BUNDLE_VERSION + ')', BUNDLE_FILES),
      cardList('Model card sections', CARD_SECTIONS)
    ])
  ]);
}

function contents() {
  return block('06', 'What is in the box', '', [
    el('p', { class: 'intro', text: 'Counts read from the modules the workspace actually imports.' }),
    counts()
  ]);
}

function method() {
  return block('07', 'How it is built', '', [
    el('div', { class: 'cols' }, [
      cardList('Constraints kept', METHOD),
      cardList('Storage and files', [
        'IndexedDB holds graphs, datasets, runs and artifacts, with typed arrays stored natively rather than base64.',
        'A run stores the trained weights, so a model can be re-evaluated, compared or exported after a page reload.',
        'Storage estimate and persistent-storage request are on the Library panel, next to a purge that asks first.',
        'Datasets round-trip through a native JSON format that preserves features, labels, splits and the content hash bit for bit.'
      ])
    ])
  ]);
}

function limits() {
  return block('08', 'Limits', '', [
    el('p', { class: 'intro', text: 'Stated plainly, because the alternative is finding out at the worst moment.' }),
    el('div', { class: 'cols' }, [
      cardList('Read this before shipping', LIMITS)
    ])
  ]);
}

function footer() {
  return el('footer', { class: 'foot' }, [
    el('span', { class: 'mono', text: TOOL_NAME + ' ' + TOOL_VERSION }),
    el('span', { text: 'bundle format v' + BUNDLE_VERSION }),
    el('span', { text: 'every table on this page is rendered from the modules the workspace runs' }),
    el('span', { text: 'tool code Apache-2.0 · generated datasets are yours · reference datasets keep their own licences' }),
    el('span', { class: 'spacer' }),
    el('a', { href: 'studio.html', text: 'Open the workspace' })
  ]);
}

function build() {
  setFavicon();
  const host = document.getElementById('page');
  if (!host) return;
  host.appendChild(hero());
  host.appendChild(workflow());
  host.appendChild(models());
  host.appendChild(datasets());
  host.appendChild(evaluation());
  host.appendChild(exportBlock());
  host.appendChild(contents());
  host.appendChild(method());
  host.appendChild(limits());
  host.appendChild(footer());
}

function start() {
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
}

function boot() {
  const bar = topbar();
  document.body.insertBefore(bar, document.body.firstChild);
  const bootMsg = document.getElementById('bootmsg');
  if (bootMsg) bootMsg.remove();
  try {
    build();
  } catch (e) {
    const host = document.getElementById('page');
    if (host) host.appendChild(el('div', { class: 'panel' }, [
      el('div', { class: 'panel-body' }, [
        el('div', { class: 'bold red', text: 'This page failed to render: ' + e.message }),
        el('div', { class: 'small dim', text: 'The workspace itself is a separate page and is not affected.' })
      ])
    ]));
  }
}

start();
