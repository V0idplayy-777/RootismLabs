import { Fft, nextPow2, stft, istft, hannWindow } from './fft.js';
import { Rng } from '../core/tensor.js';

export function hzToMel(hz) {
  return 2595 * Math.log10(1 + hz / 700);
}

export function melToHz(mel) {
  return 700 * (Math.pow(10, mel / 2595) - 1);
}

const FB_CACHE = new Map();

export function melFilterbank(numFilters, fftBins, sampleRate, fmin, fmax) {
  fmin = fmin === undefined ? 20 : fmin;
  fmax = fmax === undefined ? sampleRate / 2 : fmax;
  const cacheKey = numFilters + '|' + fftBins + '|' + sampleRate + '|' + fmin + '|' + fmax;
  const cached = FB_CACHE.get(cacheKey);
  if (cached) return cached;
  const nfft = (fftBins - 1) * 2;
  const melMin = hzToMel(fmin);
  const melMax = hzToMel(fmax);
  const points = new Int32Array(numFilters + 2);
  for (let i = 0; i < numFilters + 2; i++) {
    const mel = melMin + (melMax - melMin) * i / (numFilters + 1);
    points[i] = Math.floor((nfft + 1) * melToHz(mel) / sampleRate);
  }
  const fb = [];
  for (let m = 0; m < numFilters; m++) {
    const row = new Float32Array(fftBins);
    for (let k = points[m]; k < points[m + 1]; k++) {
      if (k >= 0 && k < fftBins && points[m + 1] > points[m]) row[k] = (k - points[m]) / (points[m + 1] - points[m]);
    }
    for (let k = points[m + 1]; k < points[m + 2]; k++) {
      if (k >= 0 && k < fftBins && points[m + 2] > points[m + 1]) row[k] = (points[m + 2] - k) / (points[m + 2] - points[m + 1]);
    }
    fb.push(row);
  }
  FB_CACHE.set(cacheKey, fb);
  return fb;
}

export function logMelFrames(signal, opts) {
  const sampleRate = opts.sampleRate || 16000;
  const frameSize = opts.frameSize || 512;
  const hop = opts.hop || 160;
  const numMel = opts.numMel || 40;
  const n = nextPow2(frameSize);
  const s = stft(signal, frameSize, hop);
  const fb = melFilterbank(numMel, s.bins, sampleRate, opts.fmin, opts.fmax);
  const frames = s.frames;
  const out = new Float32Array(frames * numMel);
  for (let f = 0; f < frames; f++) {
    for (let m = 0; m < numMel; m++) {
      let acc = 0;
      const row = fb[m];
      for (let b = 0; b < s.bins; b++) {
        if (row[b] === 0) continue;
        acc += row[b] * s.mag[f * s.bins + b] * s.mag[f * s.bins + b];
      }
      out[f * numMel + m] = Math.log(Math.max(1e-8, acc));
    }
  }
  return { data: out, frames, bins: numMel, hop, frameSize: n, sampleRate, stftFrames: s.frames, stftBins: s.bins };
}

export function linearPowerFrames(signal, opts) {
  const frameSize = opts.frameSize || 512;
  const hop = opts.hop || 128;
  const s = stft(signal, frameSize, hop);
  const out = new Float32Array(s.frames * s.bins);
  for (let i = 0; i < out.length; i++) {
    out[i] = Math.log(Math.max(1e-8, s.mag[i] * s.mag[i]));
  }
  return { data: out, frames: s.frames, bins: s.bins, mag: s.mag, phase: s.phase, hop, frameSize: s.frameSize };
}

export function mfccLite(signal, opts) {
  const mel = logMelFrames(signal, opts);
  const numCoeff = opts.numCoeff || 13;
  const frames = mel.frames;
  const numMel = mel.bins;
  const out = new Float32Array(frames * numCoeff);
  for (let f = 0; f < frames; f++) {
    for (let c = 0; c < numCoeff; c++) {
      let sum = 0;
      for (let m = 0; m < numMel; m++) {
        sum += mel.data[f * numMel + m] * Math.cos(Math.PI * c * (m + 0.5) / numMel);
      }
      out[f * numCoeff + c] = sum * Math.sqrt(2 / numMel);
    }
  }
  return { data: out, frames, bins: numCoeff };
}

export function normalizeFeatures(data, frames, bins, stats) {
  if (!stats) {
    const mean = new Float32Array(bins);
    const std = new Float32Array(bins);
    for (let b = 0; b < bins; b++) {
      let s = 0;
      for (let f = 0; f < frames; f++) s += data[f * bins + b];
      mean[b] = s / Math.max(1, frames);
      let v = 0;
      for (let f = 0; f < frames; f++) {
        const d = data[f * bins + b] - mean[b];
        v += d * d;
      }
      std[b] = Math.sqrt(v / Math.max(1, frames)) + 1e-6;
    }
    stats = { mean: Array.from(mean), std: Array.from(std) };
  }
  const out = new Float32Array(data.length);
  for (let f = 0; f < frames; f++) {
    for (let b = 0; b < bins; b++) {
      out[f * bins + b] = (data[f * bins + b] - stats.mean[b]) / stats.std[b];
    }
  }
  return { data: out, stats };
}

export function deltaFeatures(data, frames, bins, order) {
  order = order || 1;
  let cur = data;
  for (let o = 0; o < order; o++) {
    const next = new Float32Array(cur.length);
    for (let f = 0; f < frames; f++) {
      for (let b = 0; b < bins; b++) {
        const f1 = Math.max(0, f - 2);
        const f2 = Math.min(frames - 1, f + 2);
        let num = 0;
        let den = 0;
        for (let k = f1; k <= f2; k++) {
          const w = k - f;
          num += w * cur[k * bins + b];
          den += w * w;
        }
        next[f * bins + b] = den > 0 ? num / den : 0;
      }
    }
    cur = next;
  }
  return cur;
}

export function onsetEnvelope(signal, opts) {
  const frameSize = opts.frameSize || 512;
  const hop = opts.hop || 128;
  const s = stft(signal, frameSize, hop);
  const env = new Float32Array(s.frames);
  for (let f = 1; f < s.frames; f++) {
    let sum = 0;
    for (let b = 0; b < s.bins; b++) {
      const d = s.mag[f * s.bins + b] - s.mag[(f - 1) * s.bins + b];
      if (d > 0) sum += d;
    }
    env[f] = sum;
  }
  let max = 0;
  for (let i = 0; i < env.length; i++) max = Math.max(max, env[i]);
  if (max > 0) for (let i = 0; i < env.length; i++) env[i] /= max;
  return { data: env, frames: s.frames, hop, sampleRate: opts.sampleRate || 16000, frameRate: opts.sampleRate ? opts.sampleRate / hop : 125 };
}

export function yinPitch(signal, sampleRate, frameSize, hop, threshold) {
  frameSize = frameSize || 1024;
  hop = hop || 256;
  threshold = threshold === undefined ? 0.12 : threshold;
  const frames = Math.max(1, Math.floor((signal.length - frameSize) / hop) + 1);
  const maxLag = Math.min(Math.floor(frameSize / 2), Math.floor(sampleRate / 55));
  const minLag = Math.max(2, Math.floor(sampleRate / 1400));
  const out = new Float32Array(frames);
  const conf = new Float32Array(frames);
  const diff = new Float32Array(maxLag + 2);
  const cmnd = new Float32Array(maxLag + 2);
  const half = Math.floor(frameSize / 2);
  for (let f = 0; f < frames; f++) {
    const start = f * hop;
    diff.fill(0);
    for (let lag = 0; lag <= maxLag; lag++) {
      let sum = 0;
      for (let i = 0; i < half; i++) {
        const a = signal[start + i];
        const b = signal[start + i + lag];
        const d = a - b;
        sum += d * d;
      }
      diff[lag] = sum;
    }
    cmnd[0] = 1;
    let running = 0;
    for (let lag = 1; lag <= maxLag; lag++) {
      running += diff[lag];
      cmnd[lag] = running > 1e-12 ? diff[lag] * lag / running : 1;
    }
    let chosen = -1;
    let globalLag = minLag;
    let globalVal = Infinity;
    for (let lag = minLag; lag < maxLag; lag++) {
      if (cmnd[lag] < globalVal) {
        globalVal = cmnd[lag];
        globalLag = lag;
      }
      if (chosen < 0 && cmnd[lag] < threshold &&
        cmnd[lag] <= cmnd[lag - 1] && cmnd[lag] <= cmnd[lag + 1]) {
        chosen = lag;
      }
    }
    const lag0 = chosen >= 0 ? chosen : globalLag;
    const val = cmnd[lag0];
    if (lag0 > minLag && val < 0.6) {
      let lag = lag0;
      const a = cmnd[lag0 - 1];
      const b = cmnd[lag0];
      const c = cmnd[lag0 + 1];
      const denom = a - 2 * b + c;
      if (Math.abs(denom) > 1e-12) {
        const delta = 0.5 * (a - c) / denom;
        if (Math.abs(delta) < 1) lag = lag0 + delta;
      }
      out[f] = sampleRate / Math.max(1e-6, lag);
      conf[f] = Math.max(0, 1 - val);
    } else {
      out[f] = 0;
      conf[f] = 0;
    }
  }
  return { data: out, frames, hop, frameRate: sampleRate / hop, confidence: conf };
}

export function wavEncode(samples, sampleRate, channels) {
  channels = channels || 1;
  const n = samples.length;
  const buffer = new ArrayBuffer(44 + n * 2);
  const view = new DataView(buffer);
  writeStr(view, 0, 'RIFF');
  view.setUint32(4, 36 + n * 2, true);
  writeStr(view, 8, 'WAVE');
  writeStr(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * channels * 2, true);
  view.setUint16(32, channels * 2, true);
  view.setUint16(34, 16, true);
  writeStr(view, 36, 'data');
  view.setUint32(40, n * 2, true);
  let off = 44;
  for (let i = 0; i < n; i++) {
    const v = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(off, v < 0 ? v * 0x8000 : v * 0x7fff, true);
    off += 2;
  }
  return new Uint8Array(buffer);
}

function writeStr(view, offset, str) {
  for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
}

export function wavDecode(bytes) {
  const view = bytes instanceof DataView ? bytes : new DataView(bytes.buffer || bytes, bytes.byteOffset || 0, bytes.byteLength || bytes.length);
  let offset = 0;
  let sampleRate = 16000;
  let channels = 1;
  let bits = 16;
  let dataOffset = -1;
  let dataLen = 0;
  const tag = (o) => String.fromCharCode(view.getUint8(o), view.getUint8(o + 1), view.getUint8(o + 2), view.getUint8(o + 3));
  if (tag(0) !== 'RIFF') throw new Error('not a wav file');
  offset = 12;
  while (offset + 8 <= view.byteLength) {
    const id = tag(offset);
    const size = view.getUint32(offset + 4, true);
    if (id === 'fmt ') {
      channels = view.getUint16(offset + 10, true);
      sampleRate = view.getUint32(offset + 12, true);
      bits = view.getUint16(offset + 22, true);
    } else if (id === 'data') {
      dataOffset = offset + 8;
      dataLen = size;
      break;
    }
    offset += 8 + size + (size % 2);
  }
  if (dataOffset < 0) throw new Error('no data chunk');
  const count = Math.floor(dataLen / (bits / 8) / channels);
  const out = new Float32Array(count * channels);
  for (let i = 0; i < count * channels; i++) {
    const p = dataOffset + i * (bits / 8);
    if (bits === 16) out[i] = view.getInt16(p, true) / 32768;
    else if (bits === 8) out[i] = (view.getUint8(p) - 128) / 128;
    else if (bits === 32) out[i] = view.getFloat32(p, true);
    else out[i] = view.getInt16(p, true) / 32768;
  }
  if (channels > 1) {
    const mono = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      let s = 0;
      for (let c = 0; c < channels; c++) s += out[i * channels + c];
      mono[i] = s / channels;
    }
    return { samples: mono, sampleRate, channels: 1 };
  }
  return { samples: out, sampleRate, channels };
}

export function resample(signal, fromRate, toRate) {
  if (fromRate === toRate) return Float32Array.from(signal);
  const ratio = toRate / fromRate;
  const outLen = Math.floor(signal.length * ratio);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const src = i / ratio;
    const i0 = Math.floor(src);
    const i1 = Math.min(signal.length - 1, i0 + 1);
    const t = src - i0;
    out[i] = signal[i0] * (1 - t) + signal[i1] * t;
  }
  return out;
}

export function peakNormalize(signal, target) {
  target = target === undefined ? 0.95 : target;
  let peak = 0;
  for (let i = 0; i < signal.length; i++) peak = Math.max(peak, Math.abs(signal[i]));
  if (peak < 1e-9) return signal;
  const g = target / peak;
  const out = new Float32Array(signal.length);
  for (let i = 0; i < signal.length; i++) out[i] = signal[i] * g;
  return out;
}

export function rms(signal) {
  let s = 0;
  for (let i = 0; i < signal.length; i++) s += signal[i] * signal[i];
  return Math.sqrt(s / Math.max(1, signal.length));
}

export function zeroCrossingRate(signal) {
  let c = 0;
  for (let i = 1; i < signal.length; i++) if ((signal[i - 1] < 0) !== (signal[i] < 0)) c++;
  return c / Math.max(1, signal.length);
}

export function addNoise(signal, snrDb, rng, noise) {
  const s = Math.max(rms(signal), 0.02);
  const wanted = s / Math.pow(10, snrDb / 20);
  const out = new Float32Array(signal.length);
  if (noise && noise.length) {
    const nr = Math.max(rms(noise), 1e-9);
    const scale = wanted / nr;
    for (let i = 0; i < signal.length; i++) out[i] = signal[i] + noise[i % noise.length] * scale;
  } else {
    for (let i = 0; i < signal.length; i++) out[i] = signal[i] + wanted * rng.gauss();
  }
  return out;
}

export function shiftPitch(signal, sampleRate, semitones) {
  const factor = Math.pow(2, semitones / 12);
  const out = new Float32Array(signal.length);
  for (let i = 0; i < signal.length; i++) {
    const src = i * factor;
    const i0 = Math.floor(src);
    const i1 = Math.min(signal.length - 1, i0 + 1);
    const t = src - i0;
    out[i] = (signal[i0] || 0) * (1 - t) + (signal[i1] || 0) * t;
  }
  return out;
}

export function timeStretch(signal, factor) {
  const outLen = Math.floor(signal.length * factor);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const src = i / factor;
    const i0 = Math.floor(src);
    const i1 = Math.min(signal.length - 1, i0 + 1);
    const t = src - i0;
    out[i] = (signal[i0] || 0) * (1 - t) + (signal[i1] || 0) * t;
  }
  return out;
}

export function autocorrelation(signal, sampleRate, fmin, fmax) {
  fmin = fmin || 80;
  fmax = fmax || 1000;
  const minLag = Math.floor(sampleRate / fmax);
  const maxLag = Math.ceil(sampleRate / fmin);
  let best = 0;
  let bestVal = -Infinity;
  for (let lag = minLag; lag <= Math.min(maxLag, signal.length - 1); lag++) {
    let s = 0;
    for (let i = 0; i + lag < signal.length; i++) s += signal[i] * signal[i + lag];
    if (s > bestVal) {
      bestVal = s;
      best = lag;
    }
  }
  return best > 0 ? sampleRate / best : 0;
}

export { Fft, nextPow2, stft, istft, hannWindow };
