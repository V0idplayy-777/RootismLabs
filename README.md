# Rootism Labs

A workbench for small neural networks that runs entirely in the browser. Draw or pick an
architecture, generate or import a dataset, train it with a configuration you control, measure
it with the metric that belongs to the task, compare runs side by side, and export ONNX,
TensorFlow.js or safetensors artifacts with int8/fp16 quantization and a model card attached.

No account, no server, no uploads, no build step, no dependencies.

## Running it

The repository is static files. Serve the root over HTTP and open `index.html`:

```
python3 -m http.server 8000
```

GitHub Pages: Settings → Pages → source `main` branch, `/` root. The landing page is `index.html`,
the workspace is `studio.html`.

ES modules are loaded directly, so the site has to be served over `http(s)`. Opening `studio.html`
from the filesystem with `file://` will be blocked by module CORS in every browser.

There is nothing to install and nothing to compile. The zip writer, the DEFLATE decoder, the ONNX
protobuf encoder, SHA-256, the FFT and every chart are written out in this repository.

## Layout

| path | what is in it |
| --- | --- |
| `index.html`, `js/landing.js`, `assets/css/landing.css` | landing page; every table on it is rendered from the same modules the workspace imports |
| `studio.html`, `js/studio.js`, `assets/css/studio.css` | the workspace shell: topbar, rail, status bar, panels |
| `js/core/` | tensors and autograd, 32 layer types, model analysis, 8 optimizers, 10 schedules, metrics, 17 task types, trainer, hashing, zip, inflate, base64, protobuf, quantization |
| `js/data/` | FFT, audio features, text features, image ops, signal ops, splitting, dataset object, 29 augmentation ops, synthetic generators, import/export of 9 formats |
| `js/presets/` | 27 model presets, 27 dataset presets, 127 metric definitions with direction and explanation |
| `js/export/` | ONNX, TensorFlow.js, safetensors, model card, bundle, export pipeline |
| `js/ui/` | DOM helpers, IndexedDB store, workspace state, charts, graph editor, and one module per panel |
| `tests/` | Node-runnable suites; no browser, no DOM, no network |

## Verification

```
node tests/gradcheck.mjs     numeric gradients against finite differences
node tests/smoke.mjs         22 task and graph combinations trained end to end
node tests/onnx.mjs          27 presets, emitted op histogram, structural conformance
node tests/tfjs.mjs          80 graphs, emitted op histogram, structural conformance
node tests/inflate.mjs       108 inflate checks against stored, fixed and dynamic Huffman blocks
node tests/import.mjs        dataset round trips, table inference, audio feature layouts, zip decode
node tests/preflight.mjs     data leakage checks and split integrity regressions
node tests/training-integrity.mjs  optimizer, accumulation and timing regressions
node tests/production-path.mjs  import and export pipeline regressions
```

During development the exporters were also checked against external oracles, which are not needed
to run or test the site: `onnx` plus `onnxruntime` in Python validated all 27 presets at fp32,
fp16, int8 and uint8 (108 of 108 graphs, numeric comparison against the in-browser model),
`@tensorflow/tfjs` 4.22 loaded and ran every exported graph model, and `safetensors` 0.8.0 read
back the exported weights.

An optional browser test uses a separately installed `playwright-core`, Chromium and an Iris
CSV with headers (the application itself has no such dependencies):

```
python3 -m http.server 8000
# In another terminal:
PLAYWRIGHT_MODULE=/path/to/playwright-core/index.mjs CHROMIUM_PATH=/path/to/chromium \
  node tests/browser-workflow.mjs http://127.0.0.1:8000 /path/to/iris-browser.csv
```

It imports through the UI, trains, reloads the saved run and data, and downloads/checks the
exported ZIP. The CSV should have four feature columns and a `label` column.

## Using your own vector data

1. In **Data**, import a labeled CSV/TSV/JSON table. Pick a group column when rows share
   a person, session or source, or a temporal split when appropriate. The group ID is excluded
   from features. Confirm the inferred task, label mapping, splits and data quality.
2. In **Build**, choose **Model for loaded data** for a compact vector classification or
   regression baseline. It starts with random weights. Other modalities need a compatible
   architecture and preprocessing; the automatic builder does not support them.
3. In **Train**, choose the metric and training settings and start a run. The training dataset
   and run are stored in the browser. Review validation metrics and error cases, then evaluate
   the selected model once on an independent test set.
4. In **Export**, load the same saved dataset and build a ZIP. For a table model, use the
   bundled `preprocessing/table.mjs` before passing feature vectors to the ONNX or TF.js model.
   Run parity and performance checks in the actual deployment runtime.

## Deployment readiness

The architecture presets are **untrained starting points**, not validated production models.
The dataset presets generate synthetic examples; they do not download the named reference
corpora. Synthetic scores are useful for testing the pipeline, not evidence of performance on
customers or live data. Small models can be commercially useful, but this repository does not
currently provide the representative training data, pretrained weights, or independent benchmarks
needed to claim that any preset is ready to deploy.

For a real application, import data you have rights and consent to use, define the input and
label contract, split by independent source/person/session or time where appropriate, and
measure on a representative untouched test set against task-specific acceptance criteria.
Validate exported preprocessing and predictions in the target runtime, and monitor drift and
failure modes after deployment. Check applicable dataset, model, privacy and regulatory terms.
The training UI audits non-finite data, invalid split indices, exact cross-split input duplicates,
group overlap, and missing training classes. It warns rather than claiming these checks catch
near-duplicates, overlapping time windows, distribution shift, or other forms of leakage.
Binary and multiclass accuracy reports include a 95% Wilson interval. It assumes independent,
representative samples and does not account for clustering, model selection or distribution shift.
Table import assigns the split before fitting feature types, numerical normalization and category
vocabularies. Use the optional group column to keep people/sessions/sources together, or select a
temporal split for ordered rows. If you change the split later, reimport the table: training is
blocked when the fitted training indices change. Older imported tables may use the previous,
leaky preprocessing; reimport them. Exact duplicates are reported, not automatically removed.
A group split with too few groups may leave validation or test empty rather than leak groups.
For vector classification or regression data, **Build → Model for loaded data** creates a small,
untrained, shape-compatible baseline. Binary labels are ordered negative then positive when a
recognizable negative label exists; otherwise inspect the label mapping before training.
Bundle exports for table data include `preprocessing/table.mjs` and its feature specification;
apply that transform to each raw row before calling the exported model. The feature specification can contain category values, so inspect it before sharing. Unknown categories map
to the other bucket when present, or an all-zero vector. Decision thresholds are fixed in the
training configuration; choose them using validation data, not the test set. Detection's
`f1_at_best_conf` is a split-optimized diagnostic and must not
be presented as untouched-test operating-point results.

An optional integration benchmark can be run on scikit-learn's `sklearn/datasets/data/iris.csv`:
`node tests/iris-benchmark.mjs /path/to/iris.csv`. With file SHA-256
`f13ffa8fdd56fd8e6c8d16d4081a3fbd3114bcd0aae4256c43205169cd9d1449`, the fixed-seed run
produced 19/21 test accuracy (90.5%), with 105 training and 24 validation rows. The fp32 ONNX
export matched ONNX Runtime probabilities within 7.3e-8 on those 21 inputs. This is a small
integration check of one task, not a commercial benchmark or a guarantee for other presets.
A separate Chromium UI run imported the Iris CSV, trained the model, reloaded the saved data
and run, and downloaded the bundle; its ONNX probabilities matched the browser model within
6e-8 on the 21 test rows. The browser test used a different fixed split from the Node test. `tests/sklearn-benchmarks.mjs` also runs fixed-seed checks on external
scikit-learn breast-cancer, wine and diabetes data: one run produced binary accuracy 81/85,
wine accuracy 25/26 and diabetes regression MAE 43.0 on 66 held-out rows. These historical,
small datasets and single splits do not establish clinical or commercial suitability.

## Storage

Graphs, datasets, runs and artifacts live in IndexedDB on the machine running the browser. A run
stores its trained weights and saves the training dataset locally before starting, so a model can
be re-evaluated, compared or exported after a reload. Training is blocked if the browser cannot
save the dataset. The data remains in IndexedDB until deleted; dataset contents may be sensitive.
The workspace starts empty; nothing is pre-seeded. The Library panel reports the storage estimate,
can request persistent storage, and can purge everything after asking.

## Licensing

The tool is Apache-2.0. Preset datasets are generated from a deterministic seed; no third-party
dataset is intentionally bundled. This is not legal advice or a guarantee of
commercial suitability; verify the rights and obligations applicable to your data, models and use.

The presets name a public dataset that each task is modelled on, and that dataset keeps its own
licence. Twelve references are non-commercial, restricted or mixed, and both the landing page table
and the Data panel mark them in red:

| preset | reference | status |
| --- | --- | --- |
| `intent` | CLINC150 / SNIPS / MASSIVE | mixed, check per corpus |
| `spam` | Enron-Spam / SMS Spam Collection | UCI, research use |
| `code` | The Stack v2 / per-repo sources | per-file licences |
| `password` | leaked password corpora | prohibited |
| `embeddings` | Quora Question Pairs / PAWS | mixed |
| `pitch` | MedleyDB / MDB-stem-synth | CC BY-NC 4.0 |
| `events` | AudioSet / UrbanSound8K / FSD50K | mixed |
| `onset` | GTZAN / Ballroom | mixed |
| `denoiseimg` | SIDD / PolyU / Open Images | mixed |
| `detection` | WIDER FACE / HaGRID / EgoHands | mixed |
| `saliency` | SALICON / MSRA10K | mixed |
| `nsfw` | NudeNet dataset | restricted |

Generated samples are a stand-in for the real thing, not a replacement for checking it.
