export function fnv1a(str, seed) {
  let h = (seed === undefined ? 2166136261 : seed) >>> 0;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h >>> 0;
}

export function hash32(str, seed) {
  let h = (seed === undefined ? 0x811c9dc5 : seed) >>> 0;
  for (let i = 0; i < str.length; i++) {
    h = Math.imul(h ^ str.charCodeAt(i), 0x01000193) >>> 0;
    h = ((h << 13) | (h >>> 19)) >>> 0;
  }
  h = Math.imul(h ^ (h >>> 16), 0x7feb352d) >>> 0;
  h = Math.imul(h ^ (h >>> 15), 0x846ca68b) >>> 0;
  return (h ^ (h >>> 16)) >>> 0;
}

export const DEFAULT_TEXT_SPEC = {
  dim: 512,
  charN: [2, 5],
  wordN: [1, 2],
  signed: true,
  norm: 'l2',
  lowercase: true,
  useBiasTerm: true
};

export function normalizeText(text, spec) {
  let t = String(text);
  if (spec.lowercase !== false) t = t.toLowerCase();
  return t;
}

export function tokenizeWords(text) {
  return String(text).toLowerCase().split(/[^a-z0-9_'@.$\-]+/).filter((w) => w.length > 0);
}

export function charNgrams(text, minN, maxN) {
  const padded = '^' + text + '$';
  const out = [];
  for (let n = minN; n <= maxN; n++) {
    for (let i = 0; i + n <= padded.length; i++) out.push(padded.slice(i, i + n));
  }
  return out;
}

export function wordNgrams(words, minN, maxN) {
  const out = [];
  for (let n = minN; n <= maxN; n++) {
    for (let i = 0; i + n <= words.length; i++) out.push(words.slice(i, i + n).join(' '));
  }
  return out;
}

export function hashFeatures(text, spec) {
  spec = Object.assign({}, DEFAULT_TEXT_SPEC, spec || {});
  const dim = spec.dim || 512;
  const out = new Float32Array(dim);
  const t = normalizeText(text, spec);
  const grams = [];
  if (spec.charN && spec.charN[1] >= spec.charN[0]) {
    const cg = charNgrams(t, spec.charN[0], spec.charN[1]);
    for (const g of cg) grams.push(['c', g]);
  }
  if (spec.wordN && spec.wordN[1] >= spec.wordN[0]) {
    const words = tokenizeWords(t);
    const wg = wordNgrams(words, spec.wordN[0], spec.wordN[1]);
    for (const g of wg) grams.push(['w', g]);
  }
  for (const [kind, g] of grams) {
    const h = hash32(kind + g, 0x9e3779b1);
    const idx = h % dim;
    const sign = spec.signed === false ? 1 : ((h >>> 31) & 1) ? -1 : 1;
    out[idx] += sign;
  }
  if (spec.useBiasTerm !== false) out[dim - 1] += 1;
  return normalizeVector(out, spec.norm);
}

export function normalizeVector(v, norm) {
  if (!norm || norm === 'none') return v;
  if (norm === 'l1') {
    let s = 0;
    for (let i = 0; i < v.length; i++) s += Math.abs(v[i]);
    if (s > 0) for (let i = 0; i < v.length; i++) v[i] /= s;
    return v;
  }
  if (norm === 'binary') {
    for (let i = 0; i < v.length; i++) v[i] = v[i] !== 0 ? 1 : 0;
    return v;
  }
  if (norm === 'log') {
    for (let i = 0; i < v.length; i++) v[i] = v[i] > 0 ? Math.log1p(v[i]) : -Math.log1p(-v[i]);
    return v;
  }
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  s = Math.sqrt(s);
  if (s > 0) for (let i = 0; i < v.length; i++) v[i] /= s;
  return v;
}

export function buildCharVocab(texts, maxVocab, minFreq) {
  const freq = new Map();
  for (const t of texts) {
    const s = String(t);
    for (let i = 0; i < s.length; i++) {
      const c = s[i];
      freq.set(c, (freq.get(c) || 0) + 1);
    }
  }
  const entries = Array.from(freq.entries()).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  const kept = entries.filter((e) => (minFreq ? e[1] >= minFreq : true)).slice(0, maxVocab || 256);
  const chars = kept.map((e) => e[0]);
  const index = new Map();
  chars.forEach((c, i) => index.set(c, i + 3));
  return {
    chars,
    index,
    padId: 0,
    unkId: 1,
    eosId: 2,
    size: chars.length + 3,
    frequencies: kept.map((e) => e[1])
  };
}

export function encodeChars(text, vocab, maxLen, mode) {
  const s = String(text);
  const ids = new Int32Array(maxLen);
  const len = Math.min(s.length, mode === 'tail' ? maxLen : maxLen);
  let n = 0;
  const start = mode === 'tail' ? Math.max(0, s.length - maxLen) : 0;
  for (let i = start; i < s.length && n < maxLen; i++) {
    const id = vocab.index.get(s[i]);
    ids[n++] = id === undefined ? vocab.unkId : id;
  }
  return { ids, length: n };
}

export function buildWordVocab(texts, maxVocab, minFreq) {
  const freq = new Map();
  for (const t of texts) {
    for (const w of tokenizeWords(t)) freq.set(w, (freq.get(w) || 0) + 1);
  }
  const entries = Array.from(freq.entries()).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  const kept = entries.filter((e) => (minFreq ? e[1] >= minFreq : true)).slice(0, maxVocab || 2000);
  const words = kept.map((e) => e[0]);
  const index = new Map();
  words.forEach((w, i) => index.set(w, i + 3));
  return {
    words,
    index,
    padId: 0,
    unkId: 1,
    eosId: 2,
    size: words.length + 3,
    frequencies: kept.map((e) => e[1])
  };
}

export function encodeWords(text, vocab, maxLen) {
  const ws = tokenizeWords(text);
  const ids = new Int32Array(maxLen);
  let n = 0;
  for (let i = 0; i < ws.length && n < maxLen; i++) {
    const id = vocab.index.get(ws[i]);
    ids[n++] = id === undefined ? vocab.unkId : id;
  }
  return { ids, length: n };
}

export const TYPO_OPS = ['swap', 'drop', 'duplicate', 'replace', 'case', 'space', 'leet'];

export function applyTypo(text, rng, rate, allowed) {
  const ops = allowed && allowed.length ? allowed : TYPO_OPS;
  const chars = String(text).split('');
  const n = Math.max(1, Math.round(chars.length * rate));
  for (let k = 0; k < n; k++) {
    if (!chars.length) break;
    const op = ops[rng.int(ops.length)];
    const i = rng.int(chars.length);
    switch (op) {
      case 'swap': {
        if (i + 1 < chars.length) {
          const t = chars[i];
          chars[i] = chars[i + 1];
          chars[i + 1] = t;
        }
        break;
      }
      case 'drop':
        chars.splice(i, 1);
        break;
      case 'duplicate':
        chars.splice(i, 0, chars[i]);
        break;
      case 'replace': {
        const pool = 'abcdefghijklmnopqrstuvwxyz0123456789';
        chars[i] = pool[rng.int(pool.length)];
        break;
      }
      case 'case':
        chars[i] = rng.next() < 0.5 ? chars[i].toUpperCase() : chars[i].toLowerCase();
        break;
      case 'space': {
        if (i > 0 && chars[i] !== ' ') chars.splice(i, 0, ' ');
        break;
      }
      case 'leet': {
        const map = { a: '4', e: '3', i: '1', o: '0', s: '5', t: '7' };
        const lower = chars[i].toLowerCase();
        if (map[lower]) chars[i] = map[lower];
        break;
      }
      default:
        break;
    }
  }
  return chars.join('');
}

export function wordDrop(text, rng, rate) {
  const words = String(text).split(/\s+/);
  return words.filter(() => rng.next() > rate).join(' ') || words[0];
}

export function wordShuffle(text, rng, window) {
  const words = String(text).split(/\s+/);
  const w = window || 2;
  for (let i = 0; i < words.length - 1; i++) {
    const j = Math.min(words.length - 1, i + 1 + rng.int(w));
    if (rng.next() < 0.5) {
      const t = words[i];
      words[i] = words[j];
      words[j] = t;
    }
  }
  return words.join(' ');
}

export function truncateMiddle(text, keep) {
  const s = String(text);
  if (s.length <= keep) return s;
  const half = Math.floor(keep / 2);
  return s.slice(0, half) + s.slice(s.length - (keep - half));
}

export const PII_PATTERNS = {
  EMAIL: /[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}/g,
  PHONE: /(?:\+?\d{1,3}[\s\-]?)?(?:\(\d{2,4}\)[\s\-]?)?\d{3,4}[\s\-]?\d{3,4}(?:[\s\-]?\d{2,4})?/g,
  CARD: /\b(?:\d[ \-]?){13,19}\b/g,
  IP: /\b(?:\d{1,3}\.){3}\d{1,3}\b/g,
  KEY: /\b(?:sk|pk|api|ghp|xox[baprs])[-_][A-Za-z0-9_\-]{8,}\b/gi,
  SSN: /\b\d{3}-\d{2}-\d{4}\b/g,
  DATE: /\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b/g
};

export function findPii(text) {
  const out = [];
  for (const [kind, re] of Object.entries(PII_PATTERNS)) {
    const rx = new RegExp(re.source, re.flags.includes('g') ? re.flags : re.flags + 'g');
    let m;
    while ((m = rx.exec(text)) !== null) {
      out.push({ kind, start: m.index, end: m.index + m[0].length, text: m[0] });
      if (m[0].length === 0) rx.lastIndex++;
    }
  }
  out.sort((a, b) => a.start - b.start || b.end - a.end);
  const filtered = [];
  let lastEnd = -1;
  for (const e of out) {
    if (e.start >= lastEnd) {
      filtered.push(e);
      lastEnd = e.end;
    }
  }
  return filtered;
}

export function redact(text) {
  const spans = findPii(text);
  let out = '';
  let pos = 0;
  for (const s of spans) {
    out += text.slice(pos, s.start) + '[' + s.kind + ']';
    pos = s.end;
  }
  return out + text.slice(pos);
}

export function passwordGuessRankBits(pw, charModel) {
  let bits = 0;
  for (let i = 0; i < pw.length; i++) {
    const ctx = pw.slice(Math.max(0, i - 3), i);
    bits += charModel.nll(ctx, pw[i]);
  }
  return bits;
}

export class CharMarkov {
  constructor(order, alpha) {
    this.order = order || 3;
    this.alpha = alpha === undefined ? 0.35 : alpha;
    this.table = new Map();
    this.vocab = [];
    this.index = new Map();
    this.total = 0;
  }

  fit(texts) {
    const counts = new Map();
    for (const t of texts) {
      for (let i = 0; i < t.length; i++) counts.set(t[i], (counts.get(t[i]) || 0) + 1);
    }
    this.vocab = Array.from(counts.keys()).sort();
    this.index = new Map();
    this.vocab.forEach((c, i) => this.index.set(c, i));
    const V = this.vocab.length;
    for (const t of texts) {
      const s = '^'.repeat(this.order) + t + '$';
      for (let i = this.order; i < s.length; i++) {
        const ctx = s.slice(i - this.order, i);
        const ch = s[i];
        const ci = this.index.get(ch);
        if (ci === undefined) continue;
        let row = this.table.get(ctx);
        if (!row) {
          row = new Float32Array(V);
          this.table.set(ctx, row);
        }
        row[ci] += 1;
        this.total++;
      }
    }
    return this;
  }

  probs(ctx) {
    const V = this.vocab.length;
    const row = this.table.get(ctx);
    const out = new Float32Array(V);
    let sum = 0;
    for (let i = 0; i < V; i++) {
      const c = (row ? row[i] : 0) + this.alpha;
      out[i] = c;
      sum += c;
    }
    for (let i = 0; i < V; i++) out[i] /= sum;
    return out;
  }

  nll(ctx, ch) {
    const ci = this.index.get(ch);
    if (ci === undefined) return Math.log2(this.vocab.length + 8);
    const key = ctx.length >= this.order ? ctx.slice(ctx.length - this.order) : '^'.repeat(this.order - ctx.length) + ctx;
    const p = this.probs(key);
    return -Math.log2(Math.max(1e-12, p[ci]));
  }

  sample(rng, length) {
    let ctx = '^'.repeat(this.order);
    let out = '';
    for (let i = 0; i < length; i++) {
      const p = this.probs(ctx);
      let r = rng.next();
      let pick = 0;
      for (let j = 0; j < p.length; j++) {
        r -= p[j];
        if (r <= 0) {
          pick = j;
          break;
        }
      }
      const ch = this.vocab[pick];
      if (ch === '$') break;
      out += ch;
      ctx = (ctx + ch).slice(-this.order);
    }
    return out;
  }
}

export function bitsPerChar(model, text) {
  let bits = 0;
  let ctx = '^'.repeat(model.order);
  for (let i = 0; i < text.length; i++) {
    bits += model.nll(ctx, text[i]);
    ctx = (ctx + text[i]).slice(-model.order);
  }
  return { bits, bpc: bits / Math.max(1, text.length), chars: text.length };
}

export class WordMarkov {
  constructor(order, alpha) {
    this.order = order || 2;
    this.alpha = alpha === undefined ? 0.5 : alpha;
    this.table = new Map();
    this.starts = [];
    this.vocab = [];
  }

  fit(texts) {
    const counts = new Map();
    for (const t of texts) {
      const words = String(t).toLowerCase().split(/\s+/).filter(Boolean);
      if (words.length < this.order + 1) continue;
      for (const w of words) counts.set(w, (counts.get(w) || 0) + 1);
      this.starts.push(words.slice(0, this.order));
      const seq = words.concat(['~']);
      for (let i = this.order; i < seq.length; i++) {
        const ctx = seq.slice(i - this.order, i).join(' ');
        let row = this.table.get(ctx);
        if (!row) {
          row = new Map();
          this.table.set(ctx, row);
        }
        row.set(seq[i], (row.get(seq[i]) || 0) + 1);
      }
    }
    this.vocab = Array.from(counts.keys());
    return this;
  }

  next(ctxWords, rng) {
    const ctx = ctxWords.join(' ');
    const row = this.table.get(ctx);
    if (!row) return null;
    let total = 0;
    for (const v of row.values()) total += v;
    total += this.alpha * this.vocab.length;
    let r = rng.next() * total;
    for (const [w, c] of row) {
      r -= c;
      if (r <= 0) return w;
    }
    return this.vocab[rng.int(this.vocab.length)];
  }

  sample(rng, minWords, maxWords) {
    if (!this.starts.length) return '';
    let cur = this.starts[rng.int(this.starts.length)].slice();
    const out = cur.slice();
    const limit = maxWords || 14;
    let guard = 0;
    while (out.length < limit && guard < limit * 4) {
      guard++;
      const w = this.next(cur, rng);
      if (w === null) {
        cur = this.starts[rng.int(this.starts.length)].slice();
        continue;
      }
      if (w === '~') break;
      out.push(w);
      cur.push(w);
      if (cur.length > this.order) cur.shift();
    }
    if (out.length < (minWords || 4)) {
      const extra = this.starts[rng.int(this.starts.length)];
      for (const w of extra) if (out.length < (minWords || 4)) out.push(w);
    }
    return out.join(' ');
  }

  nll(text) {
    const words = String(text).toLowerCase().split(/\s+/).filter(Boolean);
    if (words.length <= this.order) return 0;
    let total = 0;
    for (let i = this.order; i < words.length; i++) {
      const ctx = words.slice(i - this.order, i).join(' ');
      const row = this.table.get(ctx);
      let p;
      if (!row) {
        p = this.alpha / (this.alpha * this.vocab.length + 1);
      } else {
        let s = 0;
        for (const v of row.values()) s += v;
        p = ((row.get(words[i]) || 0) + this.alpha) / (s + this.alpha * this.vocab.length);
      }
      total -= Math.log(Math.max(1e-12, p));
    }
    return total / (words.length - this.order);
  }
}
