export function nextPow2(n) {
  let p = 1;
  while (p < n) p *= 2;
  return p;
}

export class Fft {
  constructor(size) {
    this.n = nextPow2(size);
    this.rev = new Int32Array(this.n);
    const bits = Math.log2(this.n);
    for (let i = 0; i < this.n; i++) {
      let r = 0;
      for (let b = 0; b < bits; b++) if (i & (1 << b)) r |= 1 << (bits - 1 - b);
      this.rev[i] = r;
    }
    this.cos = new Float32Array(this.n / 2);
    this.sin = new Float32Array(this.n / 2);
    for (let i = 0; i < this.n / 2; i++) {
      this.cos[i] = Math.cos((2 * Math.PI * i) / this.n);
      this.sin[i] = Math.sin((2 * Math.PI * i) / this.n);
    }
  }

  forward(re, im) {
    const n = this.n;
    const outRe = new Float32Array(n);
    const outIm = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const r = this.rev[i];
      outRe[r] = re[i] || 0;
      outIm[r] = im ? im[i] || 0 : 0;
    }
    for (let size = 2; size <= n; size *= 2) {
      const half = size / 2;
      const step = n / size;
      for (let i = 0; i < n; i += size) {
        for (let j = 0; j < half; j++) {
          const ci = j * step;
          const wr = this.cos[ci];
          const wi = -this.sin[ci];
          const aRe = outRe[i + j];
          const aIm = outIm[i + j];
          const bRe = outRe[i + j + half];
          const bIm = outIm[i + j + half];
          const tRe = bRe * wr - bIm * wi;
          const tIm = bRe * wi + bIm * wr;
          outRe[i + j] = aRe + tRe;
          outIm[i + j] = aIm + tIm;
          outRe[i + j + half] = aRe - tRe;
          outIm[i + j + half] = aIm - tIm;
        }
      }
    }
    return { re: outRe, im: outIm };
  }

  inverse(re, im, scale) {
    const n = this.n;
    const conjIm = new Float32Array(n);
    for (let i = 0; i < n; i++) conjIm[i] = -im[i];
    const r = this.forward(re, conjIm);
    const s = scale === undefined ? 1 / n : scale;
    const outRe = new Float32Array(n);
    const outIm = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      outRe[i] = r.re[i] * s;
      outIm[i] = -r.im[i] * s;
    }
    return { re: outRe, im: outIm };
  }
}

export function hannWindow(n) {
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / Math.max(1, n - 1)));
  return w;
}

export function hammingWindow(n) {
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / Math.max(1, n - 1));
  return w;
}

export function stft(signal, frameSize, hop, window) {
  const n = nextPow2(frameSize);
  const fft = new Fft(n);
  const win = window || hannWindow(n);
  const frames = Math.max(1, Math.floor((signal.length - frameSize) / hop) + 1);
  const bins = n / 2 + 1;
  const mag = new Float32Array(frames * bins);
  const phase = new Float32Array(frames * bins);
  const re = new Float32Array(n);
  const im = new Float32Array(n);
  for (let f = 0; f < frames; f++) {
    re.fill(0);
    im.fill(0);
    const start = f * hop;
    for (let i = 0; i < n; i++) {
      const idx = start + i;
      if (idx < signal.length && i < frameSize) re[i] = signal[idx] * win[i];
    }
    const r = fft.forward(re, im);
    for (let b = 0; b < bins; b++) {
      mag[f * bins + b] = Math.sqrt(r.re[b] * r.re[b] + r.im[b] * r.im[b]);
      phase[f * bins + b] = Math.atan2(r.im[b], r.re[b]);
    }
  }
  return { mag, phase, frames, bins, frameSize: n, hop };
}

export function istft(mag, phase, frames, bins, frameSize, hop, win) {
  const fft = new Fft(frameSize);
  const window = win || hannWindow(frameSize);
  const outLen = (frames - 1) * hop + frameSize;
  const out = new Float32Array(outLen);
  const norm = new Float32Array(outLen);
  const re = new Float32Array(frameSize);
  const im = new Float32Array(frameSize);
  for (let f = 0; f < frames; f++) {
    for (let b = 0; b < bins; b++) {
      const m = mag[f * bins + b];
      const p = phase[f * bins + b];
      re[b] = m * Math.cos(p);
      im[b] = m * Math.sin(p);
      if (b > 0 && b < frameSize - b) {
        re[frameSize - b] = re[b];
        im[frameSize - b] = -im[b];
      }
    }
    const r = fft.inverse(re, im);
    const start = f * hop;
    for (let i = 0; i < frameSize; i++) {
      if (start + i >= outLen) break;
      out[start + i] += r.re[i] * window[i % window.length];
      norm[start + i] += window[i % window.length] * window[i % window.length];
    }
  }
  let maxNorm = 0;
  for (let i = 0; i < outLen; i++) {
    if (norm[i] > maxNorm) maxNorm = norm[i];
  }
  const thresh = maxNorm * 0.1;
  for (let i = 0; i < outLen; i++) {
    const nv = norm[i];
    if (nv <= 1e-9) {
      out[i] = 0;
      continue;
    }
    out[i] = nv < thresh ? out[i] / thresh : out[i] / nv;
  }
  return out;
}

export function powerSpectrumDb(mag, ref) {
  const r = ref === undefined ? 1e-6 : ref;
  const out = new Float32Array(mag.length);
  for (let i = 0; i < mag.length; i++) {
    out[i] = 10 * Math.log10(Math.max(r * r, mag[i] * mag[i]));
  }
  return out;
}
