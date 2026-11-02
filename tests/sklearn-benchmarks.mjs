// Optional integration benchmarks; datasets must be downloaded separately.
// Usage: node tests/sklearn-benchmarks.mjs breast_cancer.csv wine_data.csv diabetes_data_raw.txt diabetes_target.txt
import fs from 'node:fs';
import { tableToDataset } from '../js/data/import.js';
import { graphForDataset } from '../js/presets/from-data.js';
import { Trainer } from '../js/core/train.js';
const [cancer, wine, dx, dy] = process.argv.slice(2);
if (!cancer || !wine || !dx || !dy) throw new Error('Provide the four scikit-learn dataset files (decompress diabetes gzip files first).');
function classification(file) {
  const lines = fs.readFileSync(file, 'utf8').trim().split(/\r?\n/);
  const width = Number(lines.shift().split(',')[1]);
  return lines.map(line => { const a = line.split(','), row = { label: a[width] }; for (let j = 0; j < width; j++) row['feature_' + j] = a[j]; return row; });
}
const x = fs.readFileSync(dx, 'utf8').trim().split(/\r?\n/), y = fs.readFileSync(dy, 'utf8').trim().split(/\r?\n/);
if (x.length !== y.length) throw new Error('Diabetes feature/target length mismatch.');
const regression = x.map((line, i) => { const row = { label: y[i] }; line.trim().split(/\s+/).forEach((v,j) => { row['feature_' + j] = v; }); return row; });
for (const [name, rows, task] of [['breast cancer', classification(cancer), 'binary'], ['wine', classification(wine), 'multiclass'], ['diabetes', regression, 'regression']]) {
  const ds = tableToDataset({ rows, labelColumn: 'label', task, splitMethod: task === 'regression' ? 'random' : 'stratified', splitSeed: 42 });
  const graph = graphForDataset(ds);
  const trainer = new Trainer({ spec: graph, dataset: ds, taskKey: task, config: {
    epochs: 60, batchSize: 16, seed: 7, optimizer: { name: 'adam', lr: 0.01 }, schedule: { name: 'constant' },
    earlyStopping: { enabled: true, patience: 12 }, checkpointBest: true
  } });
  const result = await trainer.run();
  console.log(JSON.stringify({ name, task, params: trainer.model.paramCount(), split: ['train', 'val', 'test'].map(k => ds.splits[k].length),
    epochs: result.epochsRun, labelMapping: ds.labels, test: result.metrics.test.scalars }));
}
