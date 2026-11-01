import { Rng, shapeSize } from '../core/tensor.js';
import { hashFeatures } from './text.js';
import {
  flipHorizontal, flipVertical, rotate90, randomCrop, bilinearResize,
  adjustBrightness, adjustContrast, addGaussianNoise, gaussianBlur, cutout, jpegQuantize
} from './image.js';

export const AUG_OPS = [
  { id: 'typos', label: 'Typos', modality: ['text'], group: 'noise', strength: { def: 0.06, min: 0.01, max: 0.3, step: 0.01, unit: 'char rate' }, describe: 'Insert, drop, swap and duplicate characters.' },
  { id: 'word_drop', label: 'Word dropout', modality: ['text'], group: 'noise', strength: { def: 0.1, min: 0, max: 0.5, step: 0.01, unit: 'word rate' }, describe: 'Randomly delete words from the input.' },
  { id: 'word_shuffle', label: 'Word order jitter', modality: ['text'], group: 'noise', strength: { def: 2, min: 1, max: 6, step: 1, unit: 'window' }, describe: 'Swap neighbouring words inside a small window.' },
  { id: 'case_flip', label: 'Case flip', modality: ['text'], group: 'format', strength: { def: 0.3, min: 0, max: 1, step: 0.05, unit: 'rate' }, describe: 'Randomly change letter case.' },
  { id: 'truncate', label: 'Truncate', modality: ['text'], group: 'format', strength: { def: 0.15, min: 0, max: 0.5, step: 0.05, unit: 'fraction' }, describe: 'Cut a fraction of the text away.' },
  { id: 'spec_noise', label: 'Spectral noise', modality: ['audio'], group: 'noise', strength: { def: 0.15, min: 0, max: 1, step: 0.01, unit: 'std' }, describe: 'Additive Gaussian noise on log-mel bins.' },
  { id: 'time_mask', label: 'Time masking', modality: ['audio'], group: 'mask', strength: { def: 2, min: 0, max: 8, step: 1, unit: 'bands' }, describe: 'SpecAugment: zero out random time spans.' },
  { id: 'freq_mask', label: 'Frequency masking', modality: ['audio'], group: 'mask', strength: { def: 2, min: 0, max: 8, step: 1, unit: 'bands' }, describe: 'SpecAugment: zero out random frequency bands.' },
  { id: 'gain', label: 'Gain jitter', modality: ['audio'], group: 'level', strength: { def: 0.15, min: 0, max: 1, step: 0.01, unit: '±factor' }, describe: 'Random amplitude scaling.' },
  { id: 'pitch_shift', label: 'Pitch shift', modality: ['audio'], group: 'spectral', strength: { def: 2, min: 0, max: 8, step: 1, unit: '±bins' }, describe: 'Roll the frequency axis to simulate pitch change.' },
  { id: 'time_shift', label: 'Time shift', modality: ['audio'], group: 'spectral', strength: { def: 3, min: 0, max: 12, step: 1, unit: '±frames' }, describe: 'Slide the clip forward or back in time.' },
  { id: 'reverb', label: 'Reverb tail', modality: ['audio'], group: 'noise', strength: { def: 0.25, min: 0, max: 0.8, step: 0.05, unit: 'mix' }, describe: 'Add exponentially decaying echoes.' },
  { id: 'flip_h', label: 'Flip horizontal', modality: ['image'], group: 'geometry', strength: { def: 1, min: 0, max: 1, step: 1, unit: '' }, describe: 'Mirror left-right.' },
  { id: 'flip_v', label: 'Flip vertical', modality: ['image'], group: 'geometry', strength: { def: 1, min: 0, max: 1, step: 1, unit: '' }, describe: 'Mirror top-bottom.' },
  { id: 'rotate90', label: 'Rotate 90°', modality: ['image'], group: 'geometry', strength: { def: 1, min: 0, max: 3, step: 1, unit: 'steps' }, describe: 'Rotate by a multiple of 90 degrees.' },
  { id: 'crop_scale', label: 'Random crop + resize', modality: ['image'], group: 'geometry', strength: { def: 0.15, min: 0, max: 0.5, step: 0.05, unit: 'max crop' }, describe: 'Crop a random region and scale it back.' },
  { id: 'translate', label: 'Translate', modality: ['image'], group: 'geometry', strength: { def: 0.1, min: 0, max: 0.4, step: 0.05, unit: 'fraction' }, describe: 'Shift the image inside its frame.' },
  { id: 'brightness', label: 'Brightness jitter', modality: ['image'], group: 'photometric', strength: { def: 0.15, min: 0, max: 0.6, step: 0.05, unit: '±' }, describe: 'Additive brightness change.' },
  { id: 'contrast', label: 'Contrast jitter', modality: ['image'], group: 'photometric', strength: { def: 0.2, min: 0, max: 1, step: 0.05, unit: '±' }, describe: 'Multiplicative contrast change.' },
  { id: 'img_noise', label: 'Sensor noise', modality: ['image'], group: 'noise', strength: { def: 0.05, min: 0, max: 0.3, step: 0.01, unit: 'std' }, describe: 'Additive Gaussian pixel noise.' },
  { id: 'blur', label: 'Defocus blur', modality: ['image'], group: 'noise', strength: { def: 1, min: 0, max: 3, step: 1, unit: 'radius' }, describe: 'Gaussian blur.' },
  { id: 'cutout', label: 'Cutout', modality: ['image'], group: 'mask', strength: { def: 0.25, min: 0, max: 0.6, step: 0.05, unit: 'size fraction' }, describe: 'Blank out a random square.' },
  { id: 'jpeg', label: 'Compression artefacts', modality: ['image'], group: 'noise', strength: { def: 0.4, min: 0.1, max: 1, step: 0.1, unit: 'quality' }, describe: 'Posterise to simulate low-quality JPEG.' },
  { id: 'jitter', label: 'Sensor jitter', modality: ['sensor', 'series', 'tabular'], group: 'noise', strength: { def: 0.05, min: 0, max: 0.4, step: 0.01, unit: 'std' }, describe: 'Additive Gaussian noise per channel.' },
  { id: 'scale_warp', label: 'Magnitude warp', modality: ['sensor', 'series', 'tabular'], group: 'level', strength: { def: 0.1, min: 0, max: 0.6, step: 0.05, unit: '±' }, describe: 'Smooth per-channel gain drift.' },
  { id: 'channel_mask', label: 'Channel dropout', modality: ['sensor', 'series', 'tabular'], group: 'mask', strength: { def: 0.1, min: 0, max: 0.5, step: 0.05, unit: 'rate' }, describe: 'Zero out whole channels to simulate a dead sensor.' },
  { id: 'window_shift', label: 'Window shift', modality: ['sensor', 'series', 'tabular'], group: 'geometry', strength: { def: 2, min: 0, max: 10, step: 1, unit: '±samples' }, describe: 'Slide the analysis window in time.' },
  { id: 'time_warp', label: 'Time warp', modality: ['sensor', 'series', 'tabular'], group: 'geometry', strength: { def: 0.08, min: 0, max: 0.3, step: 0.02, unit: '±rate' }, describe: 'Resample the window slightly faster or slower.' },
  { id: 'point_dropout', label: 'Sample dropout', modality: ['sensor', 'series', 'tabular'], group: 'mask', strength: { def: 0.05, min: 0, max: 0.4, step: 0.01, unit: 'rate' }, describe: 'Drop individual readings to simulate packet loss.' }
];

export const AUG_GROUPS = ['noise', 'mask', 'geometry', 'photometric', 'spectral', 'level', 'format'];

export function opsForModality(modality) {
  return AUG_OPS.filter((o) => o.modality.includes(modality));
}

export function opById(id) {
  return AUG_OPS.find((o) => o.id === id) || null;
}

export function defaultAugConfig(modality) {
  return {
    enabled: false,
    applyProb: 0.8,
    ops: opsForModality(modality).slice(0, 3).map((o) => ({ id: o.id, strength: o.strength.def, prob: 0.5 }))
  };
}

export function augmentBatch(view, idx, cfg, rng) {
  if (!cfg || !cfg.enabled || !cfg.ops || !cfg.ops.length) return 0;
  const ds = view.ds;
  let applied = 0;
  for (const i of idx) {
    if (rng.next() > (cfg.applyProb === undefined ? 0.8 : cfg.applyProb)) continue;
    for (const op of cfg.ops) {
      if (rng.next() > (op.prob === undefined ? 0.5 : op.prob)) continue;
      applyOp(ds, i, op, rng);
      applied++;
    }
  }
  return applied;
}

function applyOp(ds, i, op, rng) {
  const spec = ds.featureSpec || {};
  const shape = ds.xShape;
  const per = shapeSize(shape);
  const start = i * per;
  switch (ds.modality) {
    case 'text':
      applyTextOp(ds, i, op, rng, spec);
      break;
    case 'audio':
      applyAudioOp(ds, i, op, rng, shape);
      break;
    case 'image':
      applyImageOp(ds, i, op, rng, shape);
      break;
    default:
      applySignalOp(ds, i, op, rng, shape);
  }
}

function applyTextOp(ds, i, op, rng, spec) {
  if (!ds.meta || !ds.meta[i] || ds.meta[i].text === undefined) return;
  let text = ds.meta[i].text;
  const s = op.strength;
  switch (op.id) {
    case 'typos': {
      const chars = text.split('');
      const n = Math.max(1, Math.round(chars.length * s));
      for (let k = 0; k < n; k++) {
        const p = rng.int(chars.length);
        const which = rng.int(4);
        if (which === 0 && p + 1 < chars.length) {
          const t = chars[p];
          chars[p] = chars[p + 1];
          chars[p + 1] = t;
        } else if (which === 1) {
          chars.splice(p, 1);
        } else if (which === 2) {
          chars.splice(p, 0, chars[p] || 'a');
        } else {
          chars[p] = 'abcdefghijklmnopqrstuvwxyz'[rng.int(26)];
        }
      }
      text = chars.join('');
      break;
    }
    case 'word_drop': {
      const words = text.split(/\s+/);
      const kept = words.filter(() => rng.next() > s);
      text = kept.length ? kept.join(' ') : words[0];
      break;
    }
    case 'word_shuffle': {
      const words = text.split(/\s+/);
      const w = Math.max(1, Math.round(s));
      for (let k = 0; k < words.length - 1; k++) {
        if (rng.next() < 0.4) {
          const j = Math.min(words.length - 1, k + 1 + rng.int(w));
          const t = words[k];
          words[k] = words[j];
          words[j] = t;
        }
      }
      text = words.join(' ');
      break;
    }
    case 'case_flip': {
      text = text.split('').map((c) => (rng.next() < s ? (c === c.toLowerCase() ? c.toUpperCase() : c.toLowerCase()) : c)).join('');
      break;
    }
    case 'truncate': {
      const keep = Math.max(1, Math.round(text.length * (1 - s)));
      text = rng.next() < 0.5 ? text.slice(0, keep) : text.slice(text.length - keep);
      break;
    }
    default:
      return;
  }
  const features = hashFeatures(text, spec);
  const per = shapeSize(ds.xShape);
  for (let k = 0; k < per; k++) ds.x[i * per + k] = features[k] !== undefined ? features[k] : 0;
}

function applyAudioOp(ds, i, op, rng, shape) {
  const T = shape[0];
  const F = shape[1];
  const per = T * F;
  const base = i * per;
  switch (op.id) {
    case 'spec_noise': {
      for (let k = 0; k < per; k++) ds.x[base + k] += op.strength * rng.gauss();
      break;
    }
    case 'time_mask': {
      const count = Math.round(op.strength);
      for (let c = 0; c < count; c++) {
        const len = 1 + rng.int(Math.max(1, Math.floor(T / 8)));
        const t0 = rng.int(Math.max(1, T - len));
        for (let t = t0; t < t0 + len; t++) {
          for (let f = 0; f < F; f++) ds.x[base + t * F + f] = 0;
        }
      }
      break;
    }
    case 'freq_mask': {
      const count = Math.round(op.strength);
      for (let c = 0; c < count; c++) {
        const len = 1 + rng.int(Math.max(1, Math.floor(F / 6)));
        const f0 = rng.int(Math.max(1, F - len));
        for (let t = 0; t < T; t++) {
          for (let f = f0; f < f0 + len; f++) ds.x[base + t * F + f] = 0;
        }
      }
      break;
    }
    case 'gain': {
      const g = 1 + rng.range(-op.strength, op.strength);
      for (let k = 0; k < per; k++) ds.x[base + k] *= g;
      break;
    }
    case 'pitch_shift': {
      const shift = rng.int(Math.round(op.strength) * 2 + 1) - Math.round(op.strength);
      if (shift === 0) break;
      const buf = new Float32Array(per);
      for (let t = 0; t < T; t++) {
        for (let f = 0; f < F; f++) {
          const src = f - shift;
          buf[t * F + f] = src >= 0 && src < F ? ds.x[base + t * F + src] : 0;
        }
      }
      ds.x.set(buf, base);
      break;
    }
    case 'time_shift': {
      const shift = rng.int(Math.round(op.strength) * 2 + 1) - Math.round(op.strength);
      if (shift === 0) break;
      const buf = new Float32Array(per);
      for (let t = 0; t < T; t++) {
        const src = t - shift;
        if (src >= 0 && src < T) buf.set(ds.x.subarray(base + src * F, base + src * F + F), t * F);
      }
      ds.x.set(buf, base);
      break;
    }
    case 'reverb': {
      const buf = new Float32Array(ds.x.subarray(base, base + per));
      const mix = op.strength;
      for (let d = 1; d <= 3; d++) {
        const delay = d * Math.max(1, Math.floor(T / 12));
        const amp = mix / d;
        for (let t = delay; t < T; t++) {
          for (let f = 0; f < F; f++) ds.x[base + t * F + f] += amp * buf[(t - delay) * F + f];
        }
      }
      break;
    }
    default:
      break;
  }
}

function applyImageOp(ds, i, op, rng, shape) {
  const C = shape[0];
  const H = shape[1];
  const W = shape[2];
  const per = C * H * W;
  const base = i * per;
  const buf = new Float32Array(ds.x.subarray(base, base + per));
  let out = buf;
  switch (op.id) {
    case 'flip_h':
      if (rng.next() < op.strength) out = flipHorizontal(buf, C, H, W);
      break;
    case 'flip_v':
      if (rng.next() < op.strength) out = flipVertical(buf, C, H, W);
      break;
    case 'rotate90': {
      const times = rng.int(Math.max(1, Math.round(op.strength)) + 1);
      if (times > 0) {
        const r = rotate90(buf, C, H, W, times);
        out = r.data;
      }
      break;
    }
    case 'crop_scale': {
      const size = Math.max(4, Math.round(Math.min(H, W) * (1 - rng.range(0, op.strength))));
      const cropped = randomCrop(buf, C, H, W, size, rng);
      out = bilinearResize(cropped, C, size, size, H, W);
      break;
    }
    case 'translate': {
      const dx = Math.round(rng.range(-op.strength, op.strength) * W);
      const dy = Math.round(rng.range(-op.strength, op.strength) * H);
      out = new Float32Array(per);
      for (let c = 0; c < C; c++) {
        for (let y = 0; y < H; y++) {
          for (let x = 0; x < W; x++) {
            const sy = y - dy;
            const sx = x - dx;
            if (sy >= 0 && sy < H && sx >= 0 && sx < W) out[(c * H + y) * W + x] = buf[(c * H + sy) * W + sx];
          }
        }
      }
      break;
    }
    case 'brightness':
      out = adjustBrightness(buf, rng.range(-op.strength, op.strength));
      break;
    case 'contrast':
      out = adjustContrast(buf, 1 + rng.range(-op.strength, op.strength));
      break;
    case 'img_noise':
      out = addGaussianNoise(buf, rng, op.strength);
      break;
    case 'blur':
      if (rng.next() < 0.6) out = gaussianBlur(buf, C, H, W, Math.max(1, Math.round(op.strength)));
      break;
    case 'cutout':
      if (rng.next() < 0.5) out = cutout(buf, C, H, W, rng, Math.max(2, Math.round(Math.min(H, W) * op.strength)));
      break;
    case 'jpeg':
      if (rng.next() < 0.6) out = jpegQuantize(buf, C, H, W, Math.max(0.1, 1 - op.strength));
      break;
    default:
      return;
  }
  if (out.length === per) ds.x.set(out, base);
}

function applySignalOp(ds, i, op, rng, shape) {
  const T = shape.length > 1 ? shape[0] : 1;
  const C = shape.length > 1 ? shape[1] : shape[0];
  const per = T * C;
  const base = i * per;
  switch (op.id) {
    case 'jitter': {
      for (let k = 0; k < per; k++) ds.x[base + k] += op.strength * rng.gauss();
      break;
    }
    case 'scale_warp': {
      for (let c = 0; c < C; c++) {
        const g = 1 + rng.range(-op.strength, op.strength);
        const drift = rng.range(-op.strength, op.strength) / Math.max(1, T);
        for (let t = 0; t < T; t++) ds.x[base + t * C + c] *= g + drift * t;
      }
      break;
    }
    case 'channel_mask': {
      for (let c = 0; c < C; c++) {
        if (rng.next() < op.strength) {
          for (let t = 0; t < T; t++) ds.x[base + t * C + c] = 0;
        }
      }
      break;
    }
    case 'window_shift': {
      const shift = rng.int(Math.round(op.strength) * 2 + 1) - Math.round(op.strength);
      if (!shift) break;
      const buf = new Float32Array(per);
      for (let t = 0; t < T; t++) {
        const src = t - shift;
        if (src >= 0 && src < T) buf.set(ds.x.subarray(base + src * C, base + src * C + C), t * C);
        else buf.set(ds.x.subarray(base, base + C), t * C);
      }
      ds.x.set(buf, base);
      break;
    }
    case 'time_warp': {
      const factor = 1 + rng.range(-op.strength, op.strength);
      const buf = new Float32Array(per);
      for (let t = 0; t < T; t++) {
        const src = t * factor;
        const t0 = Math.max(0, Math.min(T - 1, Math.floor(src)));
        const t1 = Math.max(0, Math.min(T - 1, t0 + 1));
        const f = src - t0;
        for (let c = 0; c < C; c++) buf[t * C + c] = ds.x[base + t0 * C + c] * (1 - f) + ds.x[base + t1 * C + c] * f;
      }
      ds.x.set(buf, base);
      break;
    }
    case 'point_dropout': {
      for (let k = 0; k < per; k++) {
        if (rng.next() < op.strength) ds.x[base + k] = 0;
      }
      break;
    }
    default:
      break;
  }
}

export function augmentPreview(ds, index, cfg, seed) {
  const per = shapeSize(ds.xShape);
  const view = { ds: { x: new Float32Array(ds.x.subarray(index * per, (index + 1) * per)), xShape: ds.xShape, modality: ds.modality, featureSpec: ds.featureSpec, meta: ds.meta ? [ds.meta[index]] : null }, localIdx: Int32Array.from([0]) };
  const rng = new Rng(seed === undefined ? 99 : seed);
  augmentBatch(view, view.localIdx, cfg, rng);
  return {
    original: Array.from(ds.x.subarray(index * per, (index + 1) * per)),
    augmented: Array.from(view.ds.x)
  };
}
