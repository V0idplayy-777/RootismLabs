import { shapeSize } from '../core/tensor.js';
import { sha256Text } from '../core/hash.js';

// A conservative audit, not a guarantee that a dataset is representative or independent.
export function auditDataset(ds) {
  const errors = [], warnings = [];
  const n = ds.count, xPer = shapeSize(ds.xShape || [1]), yPer = shapeSize(ds.yShape || [1]);
  if (!Number.isInteger(n) || n < 1 || !ds.x || ds.x.length !== n * xPer || !ds.y || ds.y.length !== n * yPer || (ds.x2 && ds.x2.length !== n * xPer)) {
    errors.push('Dataset dimensions do not match the sample count.');
    return { errors, warnings };
  }
  for (const [name, data] of [['features', ds.x], ['second input', ds.x2], ['targets', ds.y]]) {
    if (data) for (const value of data) if (!Number.isFinite(value)) { errors.push('Non-finite values found in ' + name + '.'); break; }
  }
  const owners = new Int8Array(n).fill(-1);
  const names = ['train', 'val', 'test'];
  const groups = new Map();
  const fingerprints = new Map();
  let duplicateInputs = false, groupLeakage = false, invalid = false;
  for (let split = 0; split < names.length; split++) {
    const indices = ds.splits && ds.splits[names[split]];
    if (!indices) { errors.push('Missing ' + names[split] + ' split.'); continue; }
    for (const i of indices) {
      if (!Number.isInteger(i) || i < 0 || i >= n || owners[i] !== -1) { invalid = true; continue; }
      owners[i] = split;
      if (ds.groups && ds.groups.length === n) {
        const group = ds.groups[i];
        if (groups.has(group) && groups.get(group) !== split) groupLeakage = true;
        else groups.set(group, split);
      }
      // Hash the complete input, then compare on collisions; include paired inputs.
      let hash = 2166136261;
      for (const a of [ds.x, ds.x2]) if (a) for (let j = 0; j < xPer; j++) {
        // String conversion preserves small differences which rounded hashes miss.
        const value = String(a[i * xPer + j]);
        for (let c = 0; c < value.length; c++) hash = Math.imul(hash ^ value.charCodeAt(c), 16777619);
        hash = Math.imul(hash ^ 255, 16777619);
      }
      const list = fingerprints.get(hash) || [];
      let sameSplitDuplicate = false;
      for (const prior of list) {
        let same = true;
        for (const a of [ds.x, ds.x2]) if (a) for (let j = 0; j < xPer; j++) {
          if (a[i * xPer + j] !== a[prior.i * xPer + j]) { same = false; break; }
        }
        if (same && prior.split !== split) duplicateInputs = true;
        if (same && prior.split === split) sameSplitDuplicate = true;
      }
      if (!sameSplitDuplicate) list.push({ i, split });
      fingerprints.set(hash, list);
    }
  }
  if (invalid) errors.push('Split indices are invalid or repeated; rebuild the splits before training.');
  if (owners.some(v => v === -1)) warnings.push('Some samples are not assigned to a split.');
  if (duplicateInputs) warnings.push('Identical inputs occur across splits; validation or test scores may be inflated.');
  if (groupLeakage) warnings.push('Groups occur in multiple splits; use a group split when samples share a person, source or session.');
  if (ds.splitMethod === 'group' && (!ds.groups || ds.groups.length !== n)) warnings.push('Group splitting was requested but valid group IDs are missing.');
  if (ds.featureSpec && ds.featureSpec.kind === 'table') {
    if (ds.featureSpec.fittedOn !== 'train' || !ds.featureSpec.fitSplitHash) warnings.push('This table uses legacy preprocessing fitted before splitting; reimport it to avoid leakage.');
    else if (ds.splits && ds.splits.train && sha256Text(Array.from(ds.splits.train).sort((a, b) => a - b).join(',')) !== ds.featureSpec.fitSplitHash) errors.push('The training split changed after table preprocessing was fitted. Reimport the dataset using the desired split.');
  }
  if (ds.source === 'preset') warnings.push('This dataset is synthetic. Its scores do not establish real-world performance; evaluate on representative, independently collected data.');
  if (!ds.license || ds.license.id === 'unset' || ds.license.allowsCommercial === false) warnings.push('Dataset commercial rights are unconfirmed or restricted; verify license, rights and consent before commercial use.');
  if (['binary', 'multiclass', 'pitch_class', 'anomaly'].includes(ds.task) && yPer === 1 && ds.splits) {
    const train = new Set(Array.from(ds.splits.train || [], i => ds.y[i]));
    for (const name of ['val', 'test']) {
      if (Array.from(ds.splits[name] || []).some(i => !train.has(ds.y[i]))) warnings.push(name + ' contains labels absent from training.');
    }
  }
  return { errors, warnings };
}

export function auditModelDataset(spec, ds, task, info) {
  const errors = [], warnings = [];
  const input = spec && spec.input && spec.input.shape;
  if (!input || !ds.xShape || input.length !== ds.xShape.length || input.some((v, i) => v !== ds.xShape[i])) {
    errors.push('Model input shape ' + JSON.stringify(input) + ' does not match dataset shape ' + JSON.stringify(ds.xShape) + '.');
  }
  if (ds.task && task !== ds.task) errors.push('Selected task ' + task + ' does not match dataset task ' + ds.task + '.');
  const out = info && info.outputShape;
  const width = out && out[out.length - 1];
  if (task === 'binary' && width !== 1) errors.push('Binary classification needs one output logit.');
  if (task === 'multiclass' && ds.labels && ds.labels.length && width !== ds.labels.length) errors.push('Model output width must match the number of dataset classes.');
  if (task === 'multilabel' && width !== shapeSize(ds.yShape || [1])) errors.push('Model output width must match the number of target labels.');
  if (task === 'regression' && width !== shapeSize(ds.yShape || [1])) errors.push('Model output width must match the number of regression targets.');
  if (task === 'binary' && ds.y) for (const v of ds.y) if (v !== 0 && v !== 1) { errors.push('Binary targets must be 0 or 1.'); break; }
  if (ds.provenance && ds.provenance.origin === 'import' && ds.featureSpec && ds.featureSpec.kind === 'table' && !ds.provenance.labelColumn) warnings.push('No label column was selected; targets were not supplied.');
  return { errors, warnings };
}
