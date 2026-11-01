import { Rng } from './tensor.js';

export const OPTIMIZERS = {
  sgd: { label: 'SGD', fields: ['lr', 'weightDecay'] },
  momentum: { label: 'SGD + Momentum', fields: ['lr', 'momentum', 'weightDecay', 'nesterov'] },
  adam: { label: 'Adam', fields: ['lr', 'beta1', 'beta2', 'eps', 'weightDecay'] },
  adamw: { label: 'AdamW', fields: ['lr', 'beta1', 'beta2', 'eps', 'weightDecay'] },
  rmsprop: { label: 'RMSprop', fields: ['lr', 'rho', 'eps', 'weightDecay'] },
  adagrad: { label: 'Adagrad', fields: ['lr', 'eps', 'initialAccumulator'] },
  adadelta: { label: 'Adadelta', fields: ['rho', 'eps'] },
  lion: { label: 'Lion', fields: ['lr', 'beta1', 'beta2', 'weightDecay'] }
};

export const DEFAULT_OPT = {
  name: 'adam',
  lr: 0.003,
  momentum: 0.9,
  nesterov: false,
  beta1: 0.9,
  beta2: 0.999,
  rho: 0.95,
  eps: 1e-8,
  weightDecay: 0,
  initialAccumulator: 0.1
};

export const SCHEDULES = {
  constant: { label: 'Constant' },
  linearWarmup: { label: 'Linear warmup + constant' },
  warmupCosine: { label: 'Warmup + cosine decay' },
  cosine: { label: 'Cosine decay' },
  cosineRestart: { label: 'Cosine with warm restarts' },
  exponential: { label: 'Exponential decay' },
  step: { label: 'Step decay' },
  inverseSqrt: { label: 'Inverse sqrt' },
  oneCycle: { label: 'One cycle' },
  plateau: { label: 'Reduce on plateau' }
};

export const DEFAULT_SCHEDULE = {
  name: 'warmupCosine',
  warmupSteps: 100,
  minLrRatio: 0.02,
  decayRate: 0.96,
  stepSize: 10,
  restartPeriod: 200,
  maxLrMultiplier: 4,
  plateauFactor: 0.5,
  plateauPatience: 3
};

export function lrAt(cfg, sched, step, totalSteps) {
  const base = cfg.lr;
  const warm = Math.max(0, sched.warmupSteps | 0);
  const minLr = base * (sched.minLrRatio === undefined ? 0.02 : sched.minLrRatio);
  const t = Math.max(0, Math.min(step, Math.max(1, totalSteps)));
  switch (sched.name) {
    case 'constant':
      return base;
    case 'linearWarmup': {
      if (warm > 0 && step < warm) return base * ((step + 1) / warm);
      const p = totalSteps > warm ? (step - warm) / Math.max(1, totalSteps - warm) : 0;
      return base + (minLr - base) * p;
    }
    case 'warmupCosine': {
      if (warm > 0 && step < warm) return base * ((step + 1) / warm);
      const p = totalSteps > warm ? (step - warm) / Math.max(1, totalSteps - warm) : 0;
      return minLr + 0.5 * (base - minLr) * (1 + Math.cos(Math.PI * Math.min(1, p)));
    }
    case 'cosine': {
      const p = t / Math.max(1, totalSteps);
      return minLr + 0.5 * (base - minLr) * (1 + Math.cos(Math.PI * Math.min(1, p)));
    }
    case 'cosineRestart': {
      const period = Math.max(1, sched.restartPeriod | 0);
      const p = (step % period) / period;
      return minLr + 0.5 * (base - minLr) * (1 + Math.cos(Math.PI * p));
    }
    case 'exponential':
      return base * Math.pow(sched.decayRate === undefined ? 0.96 : sched.decayRate, step / Math.max(1, totalSteps / 10));
    case 'step':
      return base * Math.pow(sched.decayRate === undefined ? 0.5 : sched.decayRate, Math.floor(step / Math.max(1, sched.stepSize | 0)));
    case 'inverseSqrt':
      return base / Math.sqrt(Math.max(1, step + 1));
    case 'oneCycle': {
      const peak = base * (sched.maxLrMultiplier || 4);
      const p = t / Math.max(1, totalSteps);
      if (p < 0.3) return base + (peak - base) * (p / 0.3);
      const q = (p - 0.3) / 0.7;
      return minLr + 0.5 * (peak - minLr) * (1 + Math.cos(Math.PI * q));
    }
    default:
      return base;
  }
}

export class Optimizer {
  constructor(params, cfg, sched) {
    this.params = params;
    this.cfg = Object.assign({}, DEFAULT_OPT, cfg || {});
    this.sched = Object.assign({}, DEFAULT_SCHEDULE, sched || {});
    this.stepCount = 0;
    this.totalSteps = 1000;
    this.plateauLr = this.cfg.lr;
    this.plateauBest = Infinity;
    this.plateauWait = 0;
    this.state = new Map();
    for (const p of params) {
      this.state.set(p, {
        m: new Float32Array(p.d.length),
        v: new Float32Array(p.d.length),
        acc: new Float32Array(p.d.length).fill(this.cfg.initialAccumulator || 0.1)
      });
    }
  }

  setTotalSteps(n) {
    this.totalSteps = Math.max(1, n);
  }

  currentLr() {
    if (this.sched.name === 'plateau') return this.plateauLr;
    return lrAt(this.cfg, this.sched, this.stepCount, this.totalSteps);
  }

  reportPlateau(metric) {
    if (this.sched.name !== 'plateau') return;
    if (!isFinite(metric)) return;
    if (metric < this.plateauBest - 1e-6) {
      this.plateauBest = metric;
      this.plateauWait = 0;
    } else {
      this.plateauWait++;
      if (this.plateauWait >= (this.sched.plateauPatience || 3)) {
        this.plateauWait = 0;
        this.plateauLr = Math.max(1e-7, this.plateauLr * (this.sched.plateauFactor || 0.5));
      }
    }
  }

  apply(lrOverride) {
    const lr = lrOverride === undefined ? this.currentLr() : lrOverride;
    const cfg = this.cfg;
    const wd = cfg.weightDecay || 0;
    const decoupled = cfg.name === 'adamw' || cfg.name === 'lion';
    this.stepCount++;
    const t = this.stepCount;
    for (const p of this.params) {
      const st = this.state.get(p);
      const g = p.g;
      if (!g) continue;
      const d = p.d;
      const n = d.length;
      switch (cfg.name) {
        case 'sgd': {
          for (let i = 0; i < n; i++) {
            const gi = g[i] + (decoupled ? 0 : wd * d[i]);
            d[i] -= lr * gi;
          }
          break;
        }
        case 'momentum': {
          const mu = cfg.momentum === undefined ? 0.9 : cfg.momentum;
          for (let i = 0; i < n; i++) {
            const gi = g[i] + wd * d[i];
            st.m[i] = mu * st.m[i] + gi;
            d[i] -= lr * (cfg.nesterov ? gi + mu * st.m[i] : st.m[i]);
          }
          break;
        }
        case 'adam':
        case 'adamw': {
          const b1 = cfg.beta1 === undefined ? 0.9 : cfg.beta1;
          const b2 = cfg.beta2 === undefined ? 0.999 : cfg.beta2;
          const eps = cfg.eps || 1e-8;
          const c1 = 1 - Math.pow(b1, t);
          const c2 = 1 - Math.pow(b2, t);
          for (let i = 0; i < n; i++) {
            const gi = g[i];
            st.m[i] = b1 * st.m[i] + (1 - b1) * gi;
            st.v[i] = b2 * st.v[i] + (1 - b2) * gi * gi;
            let upd = (st.m[i] / c1) / (Math.sqrt(st.v[i] / c2) + eps);
            if (decoupled) upd += wd * d[i];
            d[i] -= lr * upd;
          }
          break;
        }
        case 'rmsprop': {
          const rho = cfg.rho === undefined ? 0.95 : cfg.rho;
          const eps = cfg.eps || 1e-8;
          for (let i = 0; i < n; i++) {
            const gi = g[i];
            st.v[i] = rho * st.v[i] + (1 - rho) * gi * gi;
            d[i] -= lr * gi / (Math.sqrt(st.v[i]) + eps);
          }
          break;
        }
        case 'adagrad': {
          const eps = cfg.eps || 1e-8;
          for (let i = 0; i < n; i++) {
            const gi = g[i];
            st.acc[i] += gi * gi;
            d[i] -= lr * gi / (Math.sqrt(st.acc[i]) + eps);
          }
          break;
        }
        case 'adadelta': {
          const rho = cfg.rho === undefined ? 0.95 : cfg.rho;
          const eps = cfg.eps || 1e-6;
          for (let i = 0; i < n; i++) {
            const gi = g[i];
            st.v[i] = rho * st.v[i] + (1 - rho) * gi * gi;
            const upd = Math.sqrt(st.acc[i] + eps) / Math.sqrt(st.v[i] + eps) * gi;
            st.acc[i] = rho * st.acc[i] + (1 - rho) * upd * upd;
            d[i] -= upd;
          }
          break;
        }
        case 'lion': {
          const b1 = cfg.beta1 === undefined ? 0.9 : cfg.beta1;
          const b2 = cfg.beta2 === undefined ? 0.99 : cfg.beta2;
          for (let i = 0; i < n; i++) {
            const c = b1 * st.m[i] + (1 - b1) * g[i];
            d[i] -= lr * (Math.sign(c) + wd * d[i]);
            st.m[i] = b2 * st.m[i] + (1 - b2) * g[i];
          }
          break;
        }
        default: {
          for (let i = 0; i < n; i++) d[i] -= lr * g[i];
        }
      }
    }
    return lr;
  }
}

export class Ema {
  constructor(params, decay) {
    this.decay = decay === undefined ? 0.999 : decay;
    this.shadow = params.map((p) => new Float32Array(p.d));
    this.params = params;
  }
  update() {
    const d = this.decay;
    for (let i = 0; i < this.params.length; i++) {
      const p = this.params[i].d;
      const s = this.shadow[i];
      for (let j = 0; j < s.length; j++) s[j] = d * s[j] + (1 - d) * p[j];
    }
  }
  apply() {
    this.backup = this.params.map((p) => new Float32Array(p.d));
    for (let i = 0; i < this.params.length; i++) this.params[i].d.set(this.shadow[i]);
  }
  restore() {
    if (!this.backup) return;
    for (let i = 0; i < this.params.length; i++) this.params[i].d.set(this.backup[i]);
    this.backup = null;
  }
}

export function makeRng(seed) {
  return new Rng(seed);
}
