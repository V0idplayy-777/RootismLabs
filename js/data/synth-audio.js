import { Rng } from '../core/tensor.js';

export function synthTone(sampleRate, duration, freq, opts) {
  opts = opts || {};
  const n = Math.round(sampleRate * duration);
  const out = new Float32Array(n);
  const harmonics = opts.harmonics || 1;
  const attack = opts.attack === undefined ? 0.01 : opts.attack;
  const decay = opts.decay === undefined ? 0 : opts.decay;
  const vibrato = opts.vibrato || 0;
  const vibratoRate = opts.vibratoRate || 5;
  const phase0 = opts.phase || 0;
  for (let i = 0; i < n; i++) {
    const t = i / sampleRate;
    let f = freq;
    if (vibrato) f *= 1 + vibrato * Math.sin(2 * Math.PI * vibratoRate * t + phase0);
    let v = 0;
    for (let h = 1; h <= harmonics; h++) {
      v += Math.sin(2 * Math.PI * f * h * t + phase0 * h) / Math.pow(h, opts.rolloff === undefined ? 1.2 : opts.rolloff);
    }
    let env = 1;
    if (attack > 0 && t < attack) env = t / attack;
    if (decay > 0) env *= Math.exp(-decay * t);
    if (opts.gate) {
      const gatePeriod = opts.gate;
      const inGate = (t % gatePeriod) < gatePeriod * 0.55;
      env *= inGate ? 1 : 0.02;
    }
    out[i] = v * env;
  }
  return out;
}

export function synthNoise(sampleRate, duration, kind, rng) {
  const n = Math.round(sampleRate * duration);
  const out = new Float32Array(n);
  if (kind === 'white') {
    for (let i = 0; i < n; i++) out[i] = rng.gauss() * 0.5;
  } else if (kind === 'pink') {
    let b0 = 0;
    let b1 = 0;
    let b2 = 0;
    let b3 = 0;
    let b4 = 0;
    let b5 = 0;
    let b6 = 0;
    for (let i = 0; i < n; i++) {
      const w = rng.gauss();
      b0 = 0.99886 * b0 + w * 0.0555179;
      b1 = 0.99332 * b1 + w * 0.0750759;
      b2 = 0.96900 * b2 + w * 0.1538520;
      b3 = 0.86650 * b3 + w * 0.3104856;
      b4 = 0.55000 * b4 + w * 0.5329522;
      b5 = -0.7616 * b5 - w * 0.0168980;
      out[i] = (b0 + b1 + b2 + b3 + b4 + b5 + b6 + w * 0.5362) * 0.06;
      b6 = w * 0.115926;
    }
  } else if (kind === 'babble') {
    const voices = 4;
    const parts = [];
    for (let v = 0; v < voices; v++) parts.push(synthSpeechLike(sampleRate, duration, rng, { syllableRate: 3 + rng.next() * 3, f0: 90 + rng.next() * 140, intelligible: false }));
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (const p of parts) s += p[i];
      out[i] = s / voices;
    }
  } else if (kind === 'hum') {
    const f = 50 + rng.int(3) * 10;
    for (let i = 0; i < n; i++) {
      const t = i / sampleRate;
      out[i] = 0.25 * Math.sin(2 * Math.PI * f * t) + 0.12 * Math.sin(2 * Math.PI * 2 * f * t) + 0.03 * rng.gauss();
    }
  } else {
    for (let i = 0; i < n; i++) out[i] = rng.gauss() * 0.3;
  }
  return out;
}

export function formantEnvelope(freq, formants) {
  let v = 0;
  for (const f of formants) {
    const d = (freq - f.f) / f.bw;
    v += f.gain * Math.exp(-0.5 * d * d);
  }
  return v;
}

export function synthSpeechLike(sampleRate, duration, rng, opts) {
  opts = opts || {};
  const n = Math.round(sampleRate * duration);
  const out = new Float32Array(n);
  const f0Base = opts.f0 === undefined ? 110 + rng.range(0, 90) : opts.f0;
  const syllableRate = opts.syllableRate === undefined ? 4.5 : opts.syllableRate;
  const syllables = Math.max(1, Math.round(duration * syllableRate));
  const voiced = [];
  for (let s = 0; s < syllables; s++) voiced.push(rng.next() < (opts.unvoicedRate === undefined ? 0.22 : opts.unvoicedRate) ? 0 : 1);
  const formantSets = [
    [{ f: 500, bw: 90, gain: 1 }, { f: 1500, bw: 120, gain: 0.55 }, { f: 2500, bw: 180, gain: 0.28 }],
    [{ f: 350, bw: 80, gain: 1 }, { f: 2100, bw: 130, gain: 0.5 }, { f: 2700, bw: 180, gain: 0.25 }],
    [{ f: 700, bw: 90, gain: 1 }, { f: 1100, bw: 110, gain: 0.6 }, { f: 2450, bw: 170, gain: 0.3 }],
    [{ f: 450, bw: 85, gain: 1 }, { f: 900, bw: 100, gain: 0.5 }, { f: 2900, bw: 200, gain: 0.3 }],
    [{ f: 620, bw: 90, gain: 1 }, { f: 1900, bw: 130, gain: 0.45 }, { f: 2600, bw: 180, gain: 0.22 }]
  ];
  const perSyl = n / syllables;
  const sylAmps = [];
  let phase = 0;
  for (let i = 0; i < n; i++) {
    const t = i / sampleRate;
    const sIdx = Math.min(syllables - 1, Math.floor(i / perSyl));
    const local = (i % perSyl) / perSyl;
    const sylEnv = Math.pow(Math.sin(Math.PI * Math.min(1, local * 1.05)), 0.7);
    const f0 = f0Base * (1 + 0.06 * Math.sin(2 * Math.PI * 3.7 * t + rng.int(1)) + 0.02 * rng.gauss());
    if (voiced[sIdx]) {
      const fs = formantSets[sIdx % formantSets.length];
      let v = 0;
      if (!sylAmps[sIdx]) {
        const harm = Math.min(24, Math.floor(sampleRate / 2 / f0Base));
        const amps = new Float32Array(harm + 1);
        for (let h = 1; h <= harm; h++) amps[h] = formantEnvelope(f0Base * h, fs) / Math.pow(h, 1.05);
        sylAmps[sIdx] = amps;
      }
      const amps = sylAmps[sIdx];
      for (let h = 1; h < amps.length; h++) {
        const amp = amps[h];
        if (amp < 1e-4) continue;
        v += amp * Math.sin(phase * h);
      }
      phase += 2 * Math.PI * f0 / sampleRate;
      if (phase > 2 * Math.PI * 1000) phase -= 2 * Math.PI * 1000;
      out[i] = v * sylEnv * 0.28;
    } else {
      const fricative = rng.next() < 0.5;
      out[i] = (fricative ? rng.gauss() * 0.5 : rng.gauss() * 0.22) * sylEnv * 0.4;
    }
  }
  return out;
}

export function synthKeyword(sampleRate, duration, rng, phrase) {
  const syllables = phrase.syllables;
  const n = Math.round(sampleRate * duration);
  const out = new Float32Array(n);
  const segLen = Math.floor(n / syllables.length);
  const f0 = phrase.f0Base * (1 + rng.range(-0.12, 0.12));
  for (let s = 0; s < syllables.length; s++) {
    const syl = syllables[s];
    const start = Math.round(s * segLen * rng.range(0.94, 1.06));
    const len = Math.round(segLen * syl.dur * rng.range(0.9, 1.1));
    const formants = syl.formants.map((f) => ({ f: f * rng.range(0.94, 1.06), bw: f * 0.16, gain: 1 }));
    const harm = Math.min(20, Math.floor(sampleRate / 2 / f0));
    const amps = new Float32Array(harm + 1);
    for (let h = 1; h <= harm; h++) {
      const freq = f0 * h;
      let amp = 0;
      for (const fm of formants) {
        const d = (freq - fm.f) / fm.bw;
        amp += fm.gain * Math.exp(-0.5 * d * d);
      }
      amps[h] = amp / Math.pow(h, 1.1);
    }
    for (let i = 0; i < len; i++) {
      const gi = start + i;
      if (gi < 0 || gi >= n) continue;
      const t = i / sampleRate;
      const env = Math.pow(Math.sin(Math.PI * Math.min(1, i / len)), 0.6);
      let v = 0;
      if (syl.voiced) {
        const drift = 1 + 0.03 * Math.sin(2 * Math.PI * 5 * t) + syl.pitchCurve * (i / len);
        for (let h = 1; h <= harm; h++) {
          const amp = amps[h];
          if (amp < 1e-4) continue;
          v += amp * Math.sin(2 * Math.PI * f0 * drift * h * t + syl.phase);
        }
        v *= 0.3;
      } else {
        v = rng.gauss() * 0.42;
        let shaped = 0;
        for (const fm of formants) {
          const d = Math.log2(Math.max(60, fm.f) / 3000);
          shaped += Math.exp(-2 * d * d);
        }
        v *= 0.4 + 0.6 * Math.min(1, shaped);
      }
      out[gi] += v * env;
    }
  }
  return out;
}

export const WAKE_PHRASES = [
  { name: 'hey rootism', syllables: [
    { voiced: false, dur: 0.7, formants: [1800, 3200], pitchCurve: 0, phase: 0.2 },
    { voiced: true, dur: 1.0, formants: [620, 1200, 2500], pitchCurve: 0.04, phase: 0.5 },
    { voiced: true, dur: 0.9, formants: [500, 1500, 2600], pitchCurve: -0.03, phase: 1.1 },
    { voiced: true, dur: 1.0, formants: [400, 1900, 2700], pitchCurve: -0.06, phase: 1.7 }
  ], f0Base: 132 },
  { name: 'okay lab', syllables: [
    { voiced: true, dur: 0.9, formants: [560, 1400, 2500], pitchCurve: 0.05, phase: 0.3 },
    { voiced: false, dur: 0.6, formants: [2200, 3600], pitchCurve: 0, phase: 0.9 },
    { voiced: true, dur: 1.0, formants: [700, 1150, 2450], pitchCurve: -0.02, phase: 1.4 },
    { voiced: true, dur: 0.8, formants: [430, 900, 2600], pitchCurve: -0.08, phase: 2.0 }
  ], f0Base: 148 }
];

export const NEGATIVE_PHRASES = [
  { name: 'turn it off', syllables: [
    { voiced: true, dur: 0.9, formants: [400, 1100, 2400], pitchCurve: 0, phase: 0.4 },
    { voiced: false, dur: 0.5, formants: [2600, 4000], pitchCurve: 0, phase: 1.0 },
    { voiced: true, dur: 0.8, formants: [300, 800, 2300], pitchCurve: -0.05, phase: 1.6 }
  ], f0Base: 118 },
  { name: 'what time is it', syllables: [
    { voiced: false, dur: 0.6, formants: [2000, 3400], pitchCurve: 0, phase: 0.1 },
    { voiced: true, dur: 0.9, formants: [520, 1700, 2600], pitchCurve: 0.03, phase: 0.8 },
    { voiced: true, dur: 0.8, formants: [680, 1200, 2500], pitchCurve: -0.02, phase: 1.3 },
    { voiced: true, dur: 0.9, formants: [350, 900, 2350], pitchCurve: -0.06, phase: 1.9 }
  ], f0Base: 126 },
  { name: 'play some music', syllables: [
    { voiced: true, dur: 0.9, formants: [500, 1300, 2450], pitchCurve: 0.04, phase: 0.2 },
    { voiced: true, dur: 0.8, formants: [640, 1050, 2400], pitchCurve: 0, phase: 0.9 },
    { voiced: false, dur: 0.5, formants: [2400, 3800], pitchCurve: 0, phase: 1.4 },
    { voiced: true, dur: 1.0, formants: [430, 1800, 2650], pitchCurve: -0.04, phase: 2.0 }
  ], f0Base: 140 },
  { name: 'good morning', syllables: [
    { voiced: true, dur: 0.9, formants: [560, 1450, 2500], pitchCurve: 0.05, phase: 0.5 },
    { voiced: true, dur: 0.8, formants: [480, 1000, 2400], pitchCurve: -0.02, phase: 1.1 },
    { voiced: true, dur: 0.9, formants: [700, 1250, 2550], pitchCurve: -0.05, phase: 1.7 }
  ], f0Base: 122 },
  { name: 'set a timer', syllables: [
    { voiced: false, dur: 0.5, formants: [2800, 4200], pitchCurve: 0, phase: 0.3 },
    { voiced: true, dur: 0.8, formants: [520, 1600, 2500], pitchCurve: 0.02, phase: 0.9 },
    { voiced: true, dur: 0.9, formants: [390, 1900, 2700], pitchCurve: -0.03, phase: 1.5 },
    { voiced: true, dur: 0.8, formants: [460, 1100, 2400], pitchCurve: -0.06, phase: 2.1 }
  ], f0Base: 136 }
];

export function mixSignals(list, gains) {
  const n = Math.max(...list.map((s) => s.length));
  const out = new Float32Array(n);
  list.forEach((s, i) => {
    const g = gains ? gains[i] : 1;
    for (let j = 0; j < s.length; j++) out[j] += s[j] * g;
  });
  return out;
}

export function applyEnvelope(signal, attack, release, sampleRate) {
  const out = new Float32Array(signal);
  const a = Math.round(attack * sampleRate);
  const r = Math.round(release * sampleRate);
  for (let i = 0; i < Math.min(a, out.length); i++) out[i] *= i / Math.max(1, a);
  for (let i = 0; i < Math.min(r, out.length); i++) {
    const j = out.length - 1 - i;
    out[j] *= i / Math.max(1, r);
  }
  return out;
}

export function soundEvent(rng, kind, sampleRate, duration) {
  const n = Math.round(sampleRate * duration);
  const out = new Float32Array(n);
  switch (kind) {
    case 'doorbell': {
      const a = synthTone(sampleRate, duration * 0.5, 880, { harmonics: 3, decay: 4, rolloff: 2 });
      const b = synthTone(sampleRate, duration * 0.6, 660, { harmonics: 3, decay: 3.5, rolloff: 2 });
      for (let i = 0; i < a.length; i++) out[i] += a[i];
      const off = Math.round(n * 0.42);
      for (let i = 0; i < b.length && off + i < n; i++) out[off + i] += b[i];
      break;
    }
    case 'smoke_alarm': {
      const beep = synthTone(sampleRate, duration, 3100, { harmonics: 2, gate: 0.52, rolloff: 3 });
      for (let i = 0; i < n; i++) out[i] = beep[i] * 0.6;
      break;
    }
    case 'glass_break': {
      for (let burst = 0; burst < 5 + rng.int(6); burst++) {
        const start = rng.int(Math.max(1, n - 400));
        const len = 200 + rng.int(2000);
        for (let i = 0; i < len && start + i < n; i++) {
          const env = Math.exp(-i / (len * 0.25));
          out[start + i] += (rng.gauss() * 0.7 + 0.3 * Math.sin(2 * Math.PI * (3000 + rng.range(0, 4000)) * i / sampleRate)) * env;
        }
      }
      break;
    }
    case 'baby_cry': {
      const f0 = 420 + rng.range(0, 160);
      const cryAmps = new Float32Array(13);
      for (let h = 1; h <= 12; h++) {
        cryAmps[h] = formantEnvelope(f0 * h, [
          { f: 600, bw: 120, gain: 1 },
          { f: 1400, bw: 160, gain: 0.8 },
          { f: 3000, bw: 260, gain: 0.6 }
        ]) / Math.pow(h, 0.9);
      }
      for (let i = 0; i < n; i++) {
        const t = i / sampleRate;
        const wail = 1 + 0.12 * Math.sin(2 * Math.PI * 3.2 * t);
        const env = 0.35 + 0.65 * Math.abs(Math.sin(2 * Math.PI * 2.1 * t));
        let v = 0;
        for (let h = 1; h <= 12; h++) {
          if (cryAmps[h] < 1e-4) continue;
          v += cryAmps[h] * Math.sin(2 * Math.PI * f0 * wail * h * t);
        }
        out[i] = v * env * 0.22;
      }
      break;
    }
    case 'dog_bark': {
      const barks = 2 + rng.int(3);
      for (let b = 0; b < barks; b++) {
        const start = Math.round((b / barks) * n * 0.8 + rng.range(0, n * 0.05));
        const len = Math.round(sampleRate * rng.range(0.08, 0.16));
        const f0 = 180 + rng.range(0, 220);
        const barkAmps = new Float32Array(11);
        for (let h = 1; h <= 10; h++) {
          barkAmps[h] = formantEnvelope(f0 * h, [{ f: 500, bw: 200, gain: 1 }, { f: 1600, bw: 400, gain: 0.7 }]) / h;
        }
        for (let i = 0; i < len && start + i < n; i++) {
          const t = i / sampleRate;
          const env = Math.exp(-t * 22) * (1 - Math.exp(-t * 400));
          let v = 0;
          for (let h = 1; h <= 10; h++) {
            if (barkAmps[h] < 1e-4) continue;
            v += barkAmps[h] * Math.sin(2 * Math.PI * f0 * h * t);
          }
          out[start + i] += (v * 0.5 + rng.gauss() * 0.15) * env;
        }
      }
      break;
    }
    case 'knock': {
      const knocks = 2 + rng.int(3);
      for (let k = 0; k < knocks; k++) {
        const start = Math.round((k / knocks) * n * 0.7 + rng.range(0, n * 0.06));
        const len = Math.round(sampleRate * 0.07);
        for (let i = 0; i < len && start + i < n; i++) {
          const t = i / sampleRate;
          const env = Math.exp(-t * 60);
          out[start + i] += (Math.sin(2 * Math.PI * 150 * t) * 0.7 + Math.sin(2 * Math.PI * 320 * t) * 0.3 + rng.gauss() * 0.2) * env;
        }
      }
      break;
    }
    case 'siren': {
      for (let i = 0; i < n; i++) {
        const t = i / sampleRate;
        const f = 1100 + 500 * Math.sin(2 * Math.PI * 0.8 * t);
        out[i] = 0.4 * Math.sin(2 * Math.PI * f * t) + 0.12 * Math.sin(2 * Math.PI * f * 2 * t);
      }
      break;
    }
    case 'phone_ring': {
      const ring = synthTone(sampleRate, duration, 440, { harmonics: 1, gate: 1.0 });
      const ring2 = synthTone(sampleRate, duration, 480, { harmonics: 1, gate: 1.0 });
      for (let i = 0; i < n; i++) out[i] = (ring[i] + ring2[i]) * 0.3;
      break;
    }
    case 'microwave_beep': {
      const beep = synthTone(sampleRate, duration, 2400, { harmonics: 1, gate: 0.35 });
      for (let i = 0; i < n; i++) out[i] = beep[i] * 0.4;
      break;
    }
    case 'speech': {
      const s = synthSpeechLike(sampleRate, duration, rng, {});
      for (let i = 0; i < n; i++) out[i] = s[i];
      break;
    }
    case 'music': {
      const tempo = 100 + rng.range(0, 40);
      const beat = 60 / tempo;
      const notes = [220, 261.6, 293.7, 329.6, 392, 440];
      for (let i = 0; i < n; i++) {
        const t = i / sampleRate;
        const beatIdx = Math.floor(t / beat);
        const f = notes[(beatIdx * 3 + rng.int(1)) % notes.length] * (beatIdx % 4 === 0 ? 1 : 1);
        const env = Math.exp(-((t % beat) / beat) * 2.5);
        out[i] = (Math.sin(2 * Math.PI * f * t) * 0.4 + Math.sin(2 * Math.PI * f * 2 * t) * 0.15) * env * 0.5;
      }
      break;
    }
    default: {
      for (let i = 0; i < n; i++) out[i] = rng.gauss() * 0.05;
    }
  }
  return out;
}

export function rhythmTrack(rng, sampleRate, duration, bpm) {
  const n = Math.round(sampleRate * duration);
  const out = new Float32Array(n);
  const onsets = [];
  const beatLen = 60 / bpm;
  const subdivision = rng.next() < 0.4 ? 2 : 1;
  const pattern = [];
  const steps = Math.ceil(duration / (beatLen / subdivision));
  for (let s = 0; s < steps; s++) {
    const accent = s % (4 * subdivision) === 0;
    const hit = accent || rng.next() < (subdivision === 2 ? 0.45 : 0.7);
    pattern.push(hit ? (accent ? 1 : 0.6 + rng.range(0, 0.3)) : 0);
  }
  for (let s = 0; s < pattern.length; s++) {
    if (!pattern[s]) continue;
    const start = Math.round(s * (beatLen / subdivision) * sampleRate);
    const len = Math.round(sampleRate * 0.05);
    for (let i = 0; i < len && start + i < n; i++) {
      const t = i / sampleRate;
      const env = Math.exp(-t * 90);
      out[start + i] += (rng.gauss() * 0.8 + Math.sin(2 * Math.PI * 120 * t) * 0.4) * env * pattern[s];
    }
    onsets.push(start / sampleRate);
  }
  return { signal: out, onsets, bpm };
}

export function pinkNoiseLike(rng, n) {
  const out = new Float32Array(n);
  let b0 = 0;
  let b1 = 0;
  let b2 = 0;
  for (let i = 0; i < n; i++) {
    const w = rng.gauss();
    b0 = 0.99765 * b0 + w * 0.0990460;
    b1 = 0.96300 * b1 + w * 0.2965164;
    b2 = 0.57000 * b2 + w * 1.0526913;
    out[i] = (b0 + b1 + b2 + w * 0.1848) * 0.2;
  }
  return out;
}

export { Rng };
