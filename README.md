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
```

During development the exporters were also checked against external oracles, which are not needed
to run or test the site: `onnx` plus `onnxruntime` in Python validated all 27 presets at fp32,
fp16, int8 and uint8 (108 of 108 graphs, numeric comparison against the in-browser model),
`@tensorflow/tfjs` 4.22 loaded and ran every exported graph model, and `safetensors` 0.8.0 read
back the exported weights.

## Storage

Graphs, datasets, runs and artifacts live in IndexedDB on the machine running the browser. A run
stores its trained weights, so a model can be re-evaluated, compared or exported after a reload.
The workspace starts empty; nothing is pre-seeded. The Library panel reports the storage estimate,
can request persistent storage, and can purge everything after asking.

## Licensing

The tool is Apache-2.0. Datasets generated in the browser come from a deterministic seed and embed
no third-party content, so they and the models trained on them are yours to use commercially.

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
