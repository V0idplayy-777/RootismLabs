const F32_1 = new Float32Array(1);
const U32_1 = new Uint32Array(F32_1.buffer);

export function float32ToFloat16(value) {
  F32_1[0] = value;
  const bits = U32_1[0];
  const sign = (bits >>> 16) & 0x8000;
  const exp = (bits >>> 23) & 0xff;
  const frac = bits & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (frac ? 0x0200 : 0);
  if (exp === 0) return sign;
  const newExp = exp - 127 + 15;
  if (newExp >= 31) return sign | 0x7c00;
  if (newExp <= 0) {
    if (newExp < -10) return sign;
    const m = frac | 0x800000;
    const shift = 14 - newExp;
    return sign | ((m + (1 << (shift - 1))) >>> shift);
  }
  const rounded = frac + 0x1000;
  if (rounded & 0x800000) return sign | ((newExp + 1) << 10);
  return sign | (newExp << 10) | (rounded >>> 13);
}

export function float16ToFloat32(half) {
  const sign = half & 0x8000 ? -1 : 1;
  const exp = (half >>> 10) & 0x1f;
  const frac = half & 0x3ff;
  if (exp === 0) return sign * Math.pow(2, -14) * (frac / 1024);
  if (exp === 31) return frac ? NaN : sign * Infinity;
  return sign * Math.pow(2, exp - 15) * (1 + frac / 1024);
}

export function toFloat16Array(source) {
  const out = new Uint16Array(source.length);
  for (let i = 0; i < source.length; i++) out[i] = float32ToFloat16(source[i]);
  return out;
}

export function fromFloat16Array(source) {
  const out = new Float32Array(source.length);
  for (let i = 0; i < source.length; i++) out[i] = float16ToFloat32(source[i]);
  return out;
}

export function toBFloat16Array(source) {
  const out = new Uint16Array(source.length);
  for (let i = 0; i < source.length; i++) {
    F32_1[0] = source[i];
    const bits = U32_1[0];
    const lsb = (bits >>> 16) & 1;
    out[i] = ((bits + 0x7fff + lsb) >>> 16) & 0xffff;
  }
  return out;
}

export function fromBFloat16Array(source) {
  const out = new Float32Array(source.length);
  for (let i = 0; i < source.length; i++) {
    U32_1[0] = source[i] << 16;
    out[i] = F32_1[0];
  }
  return out;
}

function statsOf(values) {
  let min = Infinity;
  let max = -Infinity;
  let maxAbs = 0;
  let sumAbs = 0;
  let sumSq = 0;
  let bad = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!Number.isFinite(v)) {
      bad++;
      continue;
    }
    const a = v < 0 ? -v : v;
    if (v < min) min = v;
    if (v > max) max = v;
    if (a > maxAbs) maxAbs = a;
    sumAbs += a;
    sumSq += v * v;
  }
  if (min === Infinity) {
    min = 0;
    max = 0;
  }
  return { min, max, maxAbs, sumAbs, sumSq, count: values.length, bad };
}

function sliceChannel(values, shape, axis) {
  const total = values.length;
  if (!shape || shape.length < 2) return [values];
  const a = axis < 0 ? shape.length + axis : axis;
  if (a < 0 || a >= shape.length) return [values];
  const channels = shape[a];
  if (!(channels > 1) || total % channels !== 0) return [values];
  let inner = 1;
  for (let i = a + 1; i < shape.length; i++) inner *= shape[i];
  let outer = 1;
  for (let i = 0; i < a; i++) outer *= shape[i];
  if (outer !== 1) return [values];
  const per = total / channels;
  const out = [];
  for (let c = 0; c < channels; c++) out.push(values.subarray(c * per, c * per + per));
  void inner;
  return out;
}

export function quantizeInt8(values, options) {
  const opts = options || {};
  const signed = opts.signed !== false;
  const shape = opts.shape || [values.length];
  const axis = opts.axis === undefined ? 0 : opts.axis;
  const lo = signed ? -127 : 0;
  const hi = signed ? 127 : 255;
  const groups = opts.perChannel === false ? [values] : sliceChannel(values, shape, axis);
  const data = signed ? new Int8Array(values.length) : new Uint8Array(values.length);
  const scales = [];
  const zeroPoints = [];
  const groupStats = [];
  let offset = 0;

  for (const group of groups) {
    const st = statsOf(group);
    let scale;
    let zero;
    if (signed) {
      scale = Math.max(st.maxAbs, 1e-12) / 127;
      zero = 0;
    } else {
      let min = st.min;
      let max = st.max;
      if (min > 0) min = 0;
      if (max < 0) max = 0;
      if (max - min < 1e-12) {
        min -= 0.5;
        max += 0.5;
      }
      scale = (max - min) / 255;
      zero = Math.round(-min / scale);
      if (zero < 0) zero = 0;
      if (zero > 255) zero = 255;
    }
    for (let i = 0; i < group.length; i++) {
      const v = Number.isFinite(group[i]) ? group[i] : 0;
      let q = Math.round(v / scale) + zero;
      if (q < lo) q = lo;
      if (q > hi) q = hi;
      data[offset + i] = q;
    }
    scales.push(scale);
    zeroPoints.push(zero);
    groupStats.push({ min: st.min, max: st.max, maxAbs: st.maxAbs, rms: Math.sqrt(st.sumSq / Math.max(1, st.count)) });
    offset += group.length;
  }

  return {
    dtype: signed ? 'int8' : 'uint8',
    data,
    scale: scales.length === 1 ? scales[0] : scales,
    scales,
    zeroPoint: zeroPoints.length === 1 ? zeroPoints[0] : zeroPoints,
    zeroPoints,
    perChannel: groups.length > 1,
    channelAxis: groups.length > 1 ? axis : -1,
    groups: groupStats,
    shape
  };
}

export function dequantizeInt8(quant) {
  const out = new Float32Array(quant.data.length);
  const scales = quant.scales || [quant.scale];
  const zeros = quant.zeroPoints || [quant.zeroPoint];
  const per = quant.data.length / scales.length;
  for (let g = 0; g < scales.length; g++) {
    const scale = scales[g] || 1e-12;
    const zero = zeros[g] || 0;
    for (let i = 0; i < per; i++) {
      const idx = g * per + i;
      out[idx] = (quant.data[idx] - zero) * scale;
    }
  }
  return out;
}

export function quantError(original, restored) {
  const n = Math.min(original.length, restored.length);
  let sumSqOrig = 0;
  let sumSqErr = 0;
  let maxAbs = 0;
  let sumAbs = 0;
  let changed = 0;
  let bad = 0;
  for (let i = 0; i < n; i++) {
    const a = original[i];
    const b = restored[i];
    if (!Number.isFinite(a) || !Number.isFinite(b)) {
      bad++;
      continue;
    }
    const e = a - b;
    sumSqOrig += a * a;
    sumSqErr += e * e;
    const ae = e < 0 ? -e : e;
    if (ae > maxAbs) maxAbs = ae;
    sumAbs += ae;
    if (ae > 0) changed++;
  }
  const signal = Math.sqrt(sumSqOrig);
  const noise = Math.sqrt(sumSqErr);
  return {
    maxAbsError: maxAbs,
    meanAbsError: n ? sumAbs / n : 0,
    relativeL2: signal > 0 ? noise / signal : 0,
    snrDb: noise > 0 ? 20 * Math.log10(signal / noise) : Infinity,
    changedValues: changed,
    changedFraction: n ? changed / n : 0,
    nonFinite: bad,
    elements: n
  };
}

export const PRECISIONS = [
  {
    key: 'fp32', label: 'float32', bits: 32, onnxDtype: 1, safeTensorDtype: 'F32',
    summary: 'Unchanged weights. Largest file, no accuracy loss, the right default when the target has a float pipeline.',
    lossless: true
  },
  {
    key: 'fp16', label: 'float16', bits: 16, onnxDtype: 10, safeTensorDtype: 'F16',
    summary: 'IEEE half precision. Halves the file and keeps the dynamic range, so gradients and activations stay representable.',
    lossless: false
  },
  {
    key: 'bf16', label: 'bfloat16', bits: 16, onnxDtype: 16, safeTensorDtype: 'BF16',
    summary: 'Brain float. Same exponent range as float32 with ten mantissa bits, which survives very small and very large weights better than float16.',
    lossless: false
  },
  {
    key: 'int8', label: 'int8 symmetric', bits: 8, onnxDtype: 3, safeTensorDtype: 'I8',
    summary: 'Eight-bit signed integers with a scale per output channel. A quarter of the float32 size and the usual target for integer-only inference.',
    lossless: false
  },
  {
    key: 'uint8', label: 'uint8 asymmetric', bits: 8, onnxDtype: 2, safeTensorDtype: 'U8',
    summary: 'Eight-bit unsigned with scale and zero point per output channel. Matches the TFLite convention, including its asymmetric handling of one-sided distributions.',
    lossless: false
  }
];

export function precisionByKey(key) {
  for (const p of PRECISIONS) if (p.key === key) return p;
  return PRECISIONS[0];
}

export function bytesFor(params, key) {
  const p = precisionByKey(key);
  return Math.ceil((params * p.bits) / 8);
}

function encodeValues(values, precision, shape, opts) {
  if (precision.key === 'fp32') {
    return { tensor: { dtype: 'float32', data: Float32Array.from(values), restored: Float32Array.from(values) }, bits: 32 };
  }
  if (precision.key === 'fp16') {
    const q = toFloat16Array(values);
    return { tensor: { dtype: 'float16', data: q, restored: fromFloat16Array(q) }, bits: 16 };
  }
  if (precision.key === 'bf16') {
    const q = toBFloat16Array(values);
    return { tensor: { dtype: 'bfloat16', data: q, restored: fromBFloat16Array(q) }, bits: 16 };
  }
  const q = quantizeInt8(values, {
    signed: precision.key === 'int8',
    shape: shape || [values.length],
    axis: opts && opts.axis !== undefined ? opts.axis : 0,
    perChannel: opts ? opts.perChannel !== false : true
  });
  return { tensor: { dtype: q.dtype, data: q.data, scale: q.scale, scales: q.scales, zeroPoint: q.zeroPoint, zeroPoints: q.zeroPoints, perChannel: q.perChannel, channelAxis: q.channelAxis, groups: q.groups, restored: dequantizeInt8(q) }, bits: 8 };
}

export function quantizeNamedTensors(named, precisionKey, options) {
  const precision = precisionByKey(precisionKey);
  const opts = options || {};
  const tensors = [];
  const report = {
    precision: precision.key,
    label: precision.label,
    bits: precision.bits,
    tensors: 0,
    elements: 0,
    originalBytes: 0,
    quantizedBytes: 0,
    maxRelativeL2: 0,
    meanRelativeL2: 0,
    worstTensor: null,
    worstSnrDb: Infinity,
    perTensor: []
  };
  let sumRel = 0;

  for (const entry of named) {
    const values = entry.values instanceof Float32Array ? entry.values : Float32Array.from(entry.values);
    const shape = entry.shape || [values.length];
    const enc = encodeValues(values, precision, shape, opts);
    const t = enc.tensor;
    const err = quantError(values, t.restored);
    t.name = entry.name;
    t.shape = shape;
    t.elementCount = values.length;
    t.error = err;
    t.originalBytes = values.length * 4;
    t.quantizedBytes = Math.ceil((values.length * enc.bits) / 8) + (t.scales ? t.scales.length * 4 * 2 : 0);
    tensors.push(t);
    report.tensors++;
    report.elements += values.length;
    report.originalBytes += t.originalBytes;
    report.quantizedBytes += t.quantizedBytes;
    sumRel += err.relativeL2;
    if (err.relativeL2 > report.maxRelativeL2) {
      report.maxRelativeL2 = err.relativeL2;
      report.worstTensor = entry.name;
    }
    if (err.snrDb < report.worstSnrDb) report.worstSnrDb = err.snrDb;
    report.perTensor.push({
      name: entry.name,
      shape,
      elements: values.length,
      dtype: t.dtype,
      relativeL2: err.relativeL2,
      maxAbsError: err.maxAbsError,
      snrDb: err.snrDb === Infinity ? null : err.snrDb,
      originalBytes: t.originalBytes,
      quantizedBytes: t.quantizedBytes,
      perChannel: !!t.perChannel
    });
  }

  report.meanRelativeL2 = report.tensors ? sumRel / report.tensors : 0;
  report.compressionRatio = report.quantizedBytes ? report.originalBytes / report.quantizedBytes : 1;
  report.sizeSaved = report.originalBytes - report.quantizedBytes;
  if (report.worstSnrDb === Infinity) report.worstSnrDb = null;
  return { precision, tensors, report };
}

export function statesToNamed(model) {
  const list = typeof model.weights === 'function' ? model.weights() : [];
  const out = [];
  for (const w of list) {
    if (!w || !w.data) continue;
    const values = w.data instanceof Float32Array ? w.data : Float32Array.from(w.data);
    const shape = w.shape && w.shape.length ? w.shape.slice() : [values.length];
    out.push({ name: w.name, values, shape });
  }
  return out;
}

export function applyQuantizedToModel(model, tensors) {
  if (typeof model.loadWeights !== 'function') return 0;
  const list = tensors.map((t) => ({ name: t.name, shape: t.shape, data: t.restored }));
  model.loadWeights(list);
  return list.length;
}

export function activationRanges(model, batches) {
  const ranges = {};
  const visit = (values, key) => {
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      if (!Number.isFinite(v)) continue;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    if (min === Infinity) return;
    if (!ranges[key]) ranges[key] = { min, max, seen: values.length };
    else {
      ranges[key].min = Math.min(ranges[key].min, min);
      ranges[key].max = Math.max(ranges[key].max, max);
      ranges[key].seen += values.length;
    }
  };
  for (const batch of batches) {
    for (const key of Object.keys(batch)) {
      const entry = batch[key];
      if (entry && entry.v) visit(entry.v, key);
      else if (entry && entry.length !== undefined) visit(entry, key);
    }
  }
  return ranges;
}

export function precisionRecommendation(report, baselineMetric, quantMetric, direction) {
  const lower = direction === 'down';
  const delta = quantMetric - baselineMetric;
  const rel = baselineMetric !== 0 ? Math.abs(delta / baselineMetric) : Math.abs(delta);
  const verdict = {
    precision: report.precision,
    delta,
    relative: rel,
    acceptable: rel < 0.02,
    marginal: rel >= 0.02 && rel < 0.05,
    rejected: rel >= 0.05,
    note: ''
  };
  if (verdict.acceptable) verdict.note = 'The metric moved by less than two percent of its value. This precision is safe to ship.';
  else if (verdict.marginal) verdict.note = 'The metric moved by between two and five percent. Check the per-class breakdown before shipping.';
  else verdict.note = 'The metric moved by more than five percent. Do not ship at this precision without retraining or calibration.';
  verdict.better = lower ? quantMetric < baselineMetric : quantMetric > baselineMetric;
  return verdict;
}
