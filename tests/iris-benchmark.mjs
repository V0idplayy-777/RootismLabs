// Optional real-data integration benchmark. Supply scikit-learn's iris.csv as an
// argument; the dataset is not bundled and its terms must be checked separately.
import fs from 'node:fs';
import { tableToDataset } from '../js/data/import.js';
import { graphForDataset } from '../js/presets/from-data.js';
import { Trainer } from '../js/core/train.js';
import { exportOnnx } from '../js/export/onnx.js';
import { Node } from '../js/core/tensor.js';
if (!process.argv[2]) throw new Error('Usage: node tests/iris-benchmark.mjs path/to/iris.csv [onnx-output-prefix]');
const lines = fs.readFileSync(process.argv[2], 'utf8').trim().split(/\r?\n/);
if (!lines[0].startsWith('150,4,')) throw new Error('Expected scikit-learn iris.csv with metadata header.');
const rows = lines.slice(1).map(line => {
  const [a, b, c, d, label] = line.split(',');
  return { a, b, c, d, label };
});
const ds = tableToDataset({ rows, labelColumn: 'label', splitSeed: 42 });
const graph = graphForDataset(ds);
const trainer = new Trainer({ spec: graph, dataset: ds, taskKey: ds.task, config: {
  epochs: 60, batchSize: 16, seed: 7, optimizer: { name: 'adam', lr: 0.01 }, schedule: { name: 'constant' },
  earlyStopping: { enabled: true, patience: 12 }, checkpointBest: true
} });
const result = await trainer.run();
console.log(JSON.stringify({ split: ['train','val','test'].map(k => ds.splits[k].length), epochs: result.epochsRun,
  best: result.best, test: result.metrics.test.scalars }, null, 2));
if (process.argv[3]) {
  const n = ds.splits.test.length, width = ds.xShape[0], classes = ds.labels.length;
  const features = new Float32Array(n * width);
  ds.splits.test.forEach((index, i) => features.set(ds.x.subarray(index * width, (index + 1) * width), i * width));
  const raw = trainer.model.forward(new Node(features, [n, width]), 'eval').d;
  const probabilities = [];
  for (let i = 0; i < n; i++) {
    const logits = Array.from(raw.subarray(i * classes, (i + 1) * classes));
    const max = Math.max(...logits), exps = logits.map(v => Math.exp(v - max)), sum = exps.reduce((a, b) => a + b, 0);
    probabilities.push(...exps.map(v => v / sum));
  }
  fs.writeFileSync(process.argv[3] + '.onnx', exportOnnx(trainer.model).bytes);
  fs.writeFileSync(process.argv[3] + '.json', JSON.stringify({ features: Array.from(features), probabilities, n, width, classes }));
}
