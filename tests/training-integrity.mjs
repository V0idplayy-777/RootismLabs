import assert from 'node:assert/strict';
import { Trainer } from '../js/core/train.js';
import { createDataset } from '../js/data/dataset.js';
import { PRESET_MODEL_LIST } from '../js/presets/models.js';

const graph = { input: { shape: [1] }, nodes: [
  { id: 'in', type: 'input', cfg: {} },
  { id: 'out', type: 'dense', cfg: { units: 1, activation: 'linear', useBias: true } }
], edges: [{ from: 'in', to: 'out' }], output: 'out', head: 'linear' };
const ds = createDataset({ x: Float32Array.from([1,2,3,4,5,6,7]), xShape: [1], y: Float32Array.from([2,4,6,8,10,12,14]), yShape: [1], task: 'regression', splits: { train: Int32Array.from([0,1,2,3,4]), val: Int32Array.from([5]), test: Int32Array.from([6]) } });
async function run(batchSize, accumulationSteps) {
  const trainer = new Trainer({ spec: graph, dataset: ds, taskKey: 'regression', config: { epochs: 1, batchSize, accumulationSteps, seed: 42, gradClip: 0, checkpointBest: false, optimizer: { name: 'sgd', lr: 0.01 }, schedule: { name: 'constant' } } });
  const result = await trainer.run();
  return { weights: trainer.model.weights(), result };
}
const large = await run(4, 1), accumulated = await run(2, 2);
assert.equal(accumulated.result.history[0].steps, 2, 'partial accumulation window must be applied');
for (let i = 0; i < large.weights.length; i++) for (let j = 0; j < large.weights[i].data.length; j++) assert(Math.abs(large.weights[i].data[j] - accumulated.weights[i].data[j]) < 1e-5, 'accumulation should match larger batches for this model');
assert(accumulated.result.history[0].samplesPerSec > 0 && accumulated.result.history[0].samplesPerSec < 1e7);
assert(PRESET_MODEL_LIST.every(p => !p.train || !p.train.optimizer || p.train.optimizer.name), 'preset optimizer names must be used by optimizer');
console.log('training integrity checks passed');
