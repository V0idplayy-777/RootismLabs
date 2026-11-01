import { Rng } from '../core/tensor.js';

export const IMU_ACTIVITIES = ['stationary', 'walking', 'running', 'stairs_up', 'stairs_down', 'fall', 'wrist_flick', 'shake'];

export function imuWindow(rng, activity, length, sampleRate, channels) {
  sampleRate = sampleRate || 50;
  channels = channels || 3;
  const out = new Float32Array(length * channels);
  const t0 = rng.range(0, 10);
  for (let i = 0; i < length; i++) {
    const t = t0 + i / sampleRate;
    let ax = 0;
    let ay = 0;
    let az = 0.15;
    switch (activity) {
      case 'stationary': {
        ax = 0.02 * rng.gauss();
        ay = 0.02 * rng.gauss();
        az = 1 + 0.015 * rng.gauss();
        break;
      }
      case 'walking': {
        const f = 1.7 + rng.range(0, 0) * 0;
        const w = 2 * Math.PI * f;
        ax = 0.35 * Math.sin(w * t) + 0.12 * Math.sin(2 * w * t) + 0.05 * rng.gauss();
        ay = 0.28 * Math.sin(w * t + 1.1) + 0.1 * Math.sin(3 * w * t) + 0.05 * rng.gauss();
        az = 1 + 0.2 * Math.sin(2 * w * t + 0.4) + 0.05 * rng.gauss();
        break;
      }
      case 'running': {
        const w = 2 * Math.PI * 2.7;
        ax = 1.1 * Math.sin(w * t) + 0.45 * Math.sin(2 * w * t + 0.3) + 0.12 * rng.gauss();
        ay = 0.8 * Math.sin(w * t + 0.9) + 0.35 * Math.sin(3 * w * t) + 0.12 * rng.gauss();
        az = 1 + 0.7 * Math.sin(2 * w * t + 0.7) + 0.12 * rng.gauss();
        break;
      }
      case 'stairs_up': {
        const w = 2 * Math.PI * 1.3;
        ax = 0.5 * Math.sin(w * t) + 0.2 * Math.sin(2 * w * t) + 0.08 * rng.gauss();
        ay = 0.6 * Math.sin(w * t + 1.4) + 0.15 * rng.gauss();
        az = 1.1 + 0.35 * Math.sin(2 * w * t) + 0.08 * rng.gauss();
        break;
      }
      case 'stairs_down': {
        const w = 2 * Math.PI * 1.5;
        ax = 0.55 * Math.sin(w * t + 2.2) + 0.22 * Math.sin(2 * w * t) + 0.09 * rng.gauss();
        ay = 0.5 * Math.sin(w * t + 0.5) + 0.18 * rng.gauss();
        az = 0.85 + 0.4 * Math.sin(2 * w * t + 1.1) + 0.09 * rng.gauss();
        break;
      }
      case 'fall': {
        const impact = Math.exp(-Math.pow((i - length * 0.35) / (length * 0.05), 2));
        const after = i > length * 0.45 ? 1 : 0;
        ax = 3.2 * impact * Math.sin(i * 0.7) + after * 0.05 * rng.gauss() + (1 - after) * 0.15 * rng.gauss();
        ay = 2.6 * impact * Math.cos(i * 0.5) + after * 0.05 * rng.gauss() + (1 - after) * 0.15 * rng.gauss();
        az = 1 - 1.6 * impact + after * 0.95 + after * 0.03 * rng.gauss();
        break;
      }
      case 'wrist_flick': {
        const flick = Math.exp(-Math.pow((i - length * 0.5) / (length * 0.09), 2));
        ax = 2.4 * flick * Math.sin(i * 0.9) + 0.06 * rng.gauss();
        ay = 1.2 * flick * Math.cos(i * 0.6) + 0.06 * rng.gauss();
        az = 1 + 0.5 * flick + 0.05 * rng.gauss();
        break;
      }
      case 'shake': {
        ax = 1.6 * Math.sin(2 * Math.PI * 5.5 * t) + 0.3 * rng.gauss();
        ay = 1.4 * Math.sin(2 * Math.PI * 5.1 * t + 1) + 0.3 * rng.gauss();
        az = 1 + 0.9 * Math.sin(2 * Math.PI * 4.6 * t) + 0.3 * rng.gauss();
        break;
      }
      default: {
        ax = 0.05 * rng.gauss();
        ay = 0.05 * rng.gauss();
        az = 1 + 0.05 * rng.gauss();
      }
    }
    const vals = [ax, ay, az];
    for (let c = 0; c < channels; c++) {
      if (c < 3) out[i * channels + c] = vals[c];
      else out[i * channels + c] = 0.1 * rng.gauss() + (c === 3 ? vals[0] * 0.8 : vals[1] * 0.8);
    }
  }
  return out;
}

export function imuStats(window, length, channels) {
  const out = [];
  for (let c = 0; c < channels; c++) {
    let mean = 0;
    for (let i = 0; i < length; i++) mean += window[i * channels + c];
    mean /= length;
    let variance = 0;
    let max = -Infinity;
    let min = Infinity;
    let mad = 0;
    for (let i = 0; i < length; i++) {
      const v = window[i * channels + c];
      variance += (v - mean) * (v - mean);
      max = Math.max(max, v);
      min = Math.min(min, v);
      mad += Math.abs(v - mean);
    }
    variance /= length;
    out.push(mean, Math.sqrt(variance), max - min, mad / length, zeroCrossings(window, length, channels, c, mean), dominantFreq(window, length, channels, c));
  }
  let magSum = 0;
  for (let i = 0; i < length; i++) {
    let m = 0;
    for (let c = 0; c < Math.min(3, channels); c++) m += window[i * channels + c] * window[i * channels + c];
    magSum += Math.sqrt(m);
  }
  const magMean = magSum / length;
  let magVar = 0;
  for (let i = 0; i < length; i++) {
    let m = 0;
    for (let c = 0; c < Math.min(3, channels); c++) m += window[i * channels + c] * window[i * channels + c];
    magVar += (Math.sqrt(m) - magMean) * (Math.sqrt(m) - magMean);
  }
  out.push(magMean, Math.sqrt(magVar / length));
  return out;
}

function zeroCrossings(window, length, channels, c, mean) {
  let z = 0;
  for (let i = 1; i < length; i++) {
    if ((window[(i - 1) * channels + c] - mean < 0) !== (window[i * channels + c] - mean < 0)) z++;
  }
  return z / length;
}

function dominantFreq(window, length, channels, c) {
  const signal = new Float32Array(length);
  for (let i = 0; i < length; i++) signal[i] = window[i * channels + c];
  let best = 0;
  let bestPower = 0;
  for (let k = 1; k < Math.min(32, Math.floor(length / 4)); k++) {
    let re = 0;
    let im = 0;
    for (let i = 0; i < length; i++) {
      const a = 2 * Math.PI * k * i / length;
      re += signal[i] * Math.cos(a);
      im -= signal[i] * Math.sin(a);
    }
    const p = re * re + im * im;
    if (p > bestPower) {
      bestPower = p;
      best = k;
    }
  }
  return best / length;
}

export function forecastSeries(rng, kind, length, horizon) {
  const out = new Float32Array(length + horizon);
  const trend = rng.range(-0.002, 0.006);
  const phase = rng.range(0, Math.PI * 2);
  const level = rng.range(0.2, 0.7);
  for (let i = 0; i < length + horizon; i++) {
    let v = level + trend * i;
    switch (kind) {
      case 'traffic': {
        v += 0.25 * Math.sin(2 * Math.PI * i / 24 + phase);
        v += 0.12 * Math.sin(2 * Math.PI * i / 12 + phase * 1.7);
        v += 0.06 * Math.sin(2 * Math.PI * i / 168);
        if (rng.next() < 0.02) v += rng.range(0.15, 0.4);
        break;
      }
      case 'battery': {
        v = 1 - 0.006 * i - 0.05 * Math.sin(2 * Math.PI * i / 40 + phase) * 0.3;
        if (rng.next() < 0.05) v -= rng.range(0.02, 0.08);
        break;
      }
      case 'sales': {
        v += 0.3 * Math.sin(2 * Math.PI * i / 7 + phase);
        v += 0.15 * (i % 7 === 5 || i % 7 === 6 ? 1 : 0);
        if (rng.next() < 0.03) v += rng.range(0.3, 0.8);
        break;
      }
      case 'energy': {
        v += 0.28 * Math.sin(2 * Math.PI * i / 24 + phase) + 0.1 * Math.sin(2 * Math.PI * i / 6 + phase);
        break;
      }
      default: {
        v += 0.2 * Math.sin(2 * Math.PI * i / 12 + phase);
      }
    }
    v += rng.gauss() * 0.03;
    out[i] = v;
  }
  return out;
}

export const METRIC_NAMES = ['cpu', 'memory', 'disk_io', 'net_in', 'net_out', 'latency', 'errors', 'queue_depth'];

export function anomalyWindow(rng, dims, length, kind) {
  const out = new Float32Array(length * dims);
  const phases = [];
  const periods = [];
  const levels = [];
  const amps = [];
  for (let d = 0; d < dims; d++) {
    phases.push(rng.range(0, Math.PI * 2));
    periods.push(6 + rng.int(30));
    levels.push(rng.range(0.2, 0.6));
    amps.push(rng.range(0.05, 0.25));
  }
  const coupling = [];
  for (let d = 0; d < dims; d++) coupling.push(rng.range(0, 0.6));
  const anomalyStart = kind && kind !== 'normal' ? Math.floor(length * rng.range(0.3, 0.7)) : -1;
  const anomalyLen = kind && kind !== 'normal' ? Math.max(3, Math.floor(length * rng.range(0.08, 0.3))) : 0;
  for (let i = 0; i < length; i++) {
    const inAnomaly = anomalyStart >= 0 && i >= anomalyStart && i < anomalyStart + anomalyLen;
    let common = 0;
    for (let d = 0; d < dims; d++) {
      const base = levels[d] + amps[d] * Math.sin(2 * Math.PI * i / periods[d] + phases[d]);
      const noise = rng.gauss() * 0.03;
      let v = base + noise + common * coupling[d];
      if (inAnomaly) {
        switch (kind) {
          case 'spike':
            if (d === 0 || d === dims - 1) v += rng.range(0.8, 2.0) * (i === anomalyStart ? 1 : Math.exp(-(i - anomalyStart) / 3));
            break;
          case 'level_shift':
            v += 0.6 * (d % 3 === 0 ? 1 : 0.3);
            break;
          case 'oscillation':
            v += 0.35 * Math.sin(2 * Math.PI * i / 2.3 + d);
            break;
          case 'flatline':
            v = levels[d] + 0.001 * rng.gauss();
            break;
          case 'correlation_break':
            v += rng.gauss() * 0.35;
            break;
          default:
            v += rng.range(0.4, 1.2);
        }
      }
      out[i * dims + d] = v;
      if (d === 0) common = v - levels[0];
    }
  }
  return { data: out, anomalyStart, anomalyLen, kind: kind || 'normal' };
}

export function mouseTrace(rng, kind, points) {
  const out = new Float32Array(points * 5);
  let x = rng.range(0.1, 0.3);
  let y = rng.range(0.1, 0.3);
  const tx = rng.range(0.6, 0.9);
  const ty = rng.range(0.5, 0.9);
  let vx = 0;
  let vy = 0;
  for (let i = 0; i < points; i++) {
    const t = i / points;
    let dx;
    let dy;
    let dt;
    let pressure;
    let curvature;
    if (kind === 'human') {
      const ease = Math.sin(Math.PI * Math.min(1, t * 1.05));
      const jitter = rng.gauss() * 0.006;
      vx = vx * 0.72 + ((tx - x) * 0.14 + jitter) * (0.4 + ease);
      vy = vy * 0.72 + ((ty - y) * 0.14 + rng.gauss() * 0.006) * (0.4 + ease);
      dx = vx;
      dy = vy;
      dt = Math.max(4, 16 + rng.gauss() * 7);
      pressure = 0.4 + 0.4 * ease + rng.gauss() * 0.06;
      curvature = Math.abs(rng.gauss()) * 0.35 + 0.1 * Math.sin(i * 0.7);
      if (rng.next() < 0.04) {
        dx *= 0.05;
        dy *= 0.05;
        dt += rng.range(60, 400);
      }
    } else {
      const steps = points;
      dx = (tx - 0.2) / steps;
      dy = (ty - 0.2) / steps;
      dt = 8;
      pressure = 0.5;
      curvature = 0;
      if (kind === 'bezier_bot') {
        const bez = bezier(t, rng);
        dx = bez[0];
        dy = bez[1];
        dt = 8 + Math.round(t * 4);
      }
    }
    x += dx;
    y += dy;
    out[i * 5] = dx * 20;
    out[i * 5 + 1] = dy * 20;
    out[i * 5 + 2] = Math.min(1, dt / 100);
    out[i * 5 + 3] = Math.max(0, Math.min(1, pressure));
    out[i * 5 + 4] = Math.max(0, Math.min(1, curvature));
  }
  return out;
}

function bezier(t, rng) {
  const p0 = 0.2;
  const p1 = rng.range(0.3, 0.6);
  const p2 = rng.range(0.5, 0.8);
  const p3 = rng.range(0.6, 0.9);
  const mt = 1 - t;
  const v = 3 * mt * mt * (p1 - p0) + 6 * mt * t * (p2 - p1) + 3 * t * t * (p3 - p2);
  return [v * 0.01, v * 0.008];
}

export function keystrokeTrace(rng, kind, points) {
  const out = new Float32Array(points * 3);
  const humanDwell = rng.range(70, 140);
  const humanFlight = rng.range(90, 200);
  for (let i = 0; i < points; i++) {
    if (kind === 'human') {
      out[i * 3] = (humanDwell + rng.gauss() * 28) / 300;
      out[i * 3 + 1] = (humanFlight + rng.gauss() * 42 + (i % 5 === 4 ? 90 : 0)) / 500;
      out[i * 3 + 2] = rng.next() < 0.06 ? 1 : 0;
    } else if (kind === 'macro') {
      out[i * 3] = 0.28 + rng.gauss() * 0.004;
      out[i * 3 + 1] = 0.2 + rng.gauss() * 0.003;
      out[i * 3 + 2] = 0;
    } else {
      out[i * 3] = 0.03 + rng.gauss() * 0.002;
      out[i * 3 + 1] = 0.02 + rng.gauss() * 0.002;
      out[i * 3 + 2] = 0;
    }
  }
  return out;
}

export function behaviorFeatures(trace, points, stride) {
  const per = trace.length / points;
  const out = [];
  for (const key of ['mean', 'std', 'min', 'max', 'p10', 'p50', 'p90', 'skew', 'kurtosis', 'zcr', 'autocorr', 'entropy']) {
    for (let c = 0; c < per; c++) {
      const col = new Float32Array(points);
      for (let i = 0; i < points; i++) col[i] = trace[i * per + c];
      out.push(featureStat(col, key));
    }
  }
  return Float32Array.from(out);
}

function featureStat(col, key) {
  const n = col.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += col[i];
  mean /= n;
  switch (key) {
    case 'mean': return mean;
    case 'min': {
      let m = Infinity;
      for (let i = 0; i < n; i++) m = Math.min(m, col[i]);
      return m;
    }
    case 'max': {
      let m = -Infinity;
      for (let i = 0; i < n; i++) m = Math.max(m, col[i]);
      return m;
    }
    case 'zcr': {
      let z = 0;
      for (let i = 1; i < n; i++) if ((col[i - 1] - mean < 0) !== (col[i] - mean < 0)) z++;
      return z / n;
    }
    case 'autocorr': {
      let num = 0;
      let den = 0;
      for (let i = 0; i < n; i++) den += (col[i] - mean) * (col[i] - mean);
      for (let i = 1; i < n; i++) num += (col[i] - mean) * (col[i - 1] - mean);
      return den > 0 ? num / den : 0;
    }
    case 'entropy': {
      const bins = 8;
      let min = Infinity;
      let max = -Infinity;
      for (let i = 0; i < n; i++) {
        min = Math.min(min, col[i]);
        max = Math.max(max, col[i]);
      }
      const counts = new Array(bins).fill(0);
      for (let i = 0; i < n; i++) {
        let b = Math.floor((col[i] - min) / Math.max(1e-9, max - min) * bins);
        if (b >= bins) b = bins - 1;
        counts[b]++;
      }
      let h = 0;
      for (const c of counts) {
        if (!c) continue;
        const p = c / n;
        h -= p * Math.log2(p);
      }
      return h;
    }
    default: {
      let m2 = 0;
      let m3 = 0;
      let m4 = 0;
      for (let i = 0; i < n; i++) {
        const d = col[i] - mean;
        m2 += d * d;
        m3 += d * d * d;
        m4 += d * d * d * d;
      }
      const varr = m2 / n;
      const sd = Math.sqrt(varr) || 1e-9;
      if (key === 'std') return sd;
      if (key === 'skew') return m3 / n / Math.pow(sd, 3);
      if (key === 'kurtosis') return m4 / n / Math.pow(sd, 4) - 3;
      const sorted = Array.from(col).sort((a, b) => a - b);
      if (key === 'p10') return sorted[Math.floor(n * 0.1)];
      if (key === 'p50') return sorted[Math.floor(n * 0.5)];
      if (key === 'p90') return sorted[Math.floor(n * 0.9)];
      return mean;
    }
  }
}

export { featureStat };
