// Compact baselines for vector data; performance must be validated for the task.
export function graphForDataset(ds) {
  if (!ds || !ds.xShape || ds.xShape.length !== 1 || ds.xShape[0] < 1) throw new Error('This builder supports one-dimensional feature vectors only.');
  const task = ds.task;
  let units, head;
  if (task === 'binary') { units = 1; head = 'sigmoid'; }
  else if (task === 'multiclass') { units = (ds.labels || []).length; head = 'softmax'; }
  else if (task === 'multilabel') { units = (ds.yShape || [])[0]; head = 'sigmoid'; }
  else if (task === 'regression') { units = (ds.yShape || [1])[0]; head = 'linear'; }
  else throw new Error('Automatic vector models support binary, multiclass, multilabel, and regression tasks.');
  if (!Number.isInteger(units) || units < 1 || (task === 'multiclass' && units < 2)) throw new Error('Dataset must have valid output labels.');
  const dims = Math.max(16, Math.min(128, Math.round(Math.sqrt(ds.xShape[0]) * 4)));
  const nodes = [
    { id: 'input', type: 'input', cfg: {}, pos: { x: 40, y: 180 } },
    { id: 'hidden', type: 'dense', cfg: { units: dims, activation: 'relu', useBias: true }, pos: { x: 230, y: 180 } },
    { id: 'dropout', type: 'dropout', cfg: { rate: 0.15 }, pos: { x: 420, y: 180 } },
    { id: 'output', type: 'dense', cfg: { units, activation: 'linear', useBias: true }, pos: { x: 610, y: 180 } }
  ];
  return { id: 'dataset-baseline', name: ds.name + ' baseline', version: 1,
    input: { name: 'features', shape: ds.xShape.slice(), dtype: 'float32' }, nodes,
    edges: [{ from: 'input', to: 'hidden' }, { from: 'hidden', to: 'dropout' }, { from: 'dropout', to: 'output' }],
    output: 'output', head, meta: { task, dataset: ds.name, description: 'Untrained compact vector baseline; validate on representative held-out data.' } };
}
