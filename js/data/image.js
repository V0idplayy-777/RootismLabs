import { Rng } from '../core/tensor.js';

export function blank(c, h, w, fill) {
  const out = new Float32Array(c * h * w);
  if (fill) out.fill(fill);
  return out;
}

export function idx(c, h, w, ch, y, x) {
  return ((c * h + y) * w + x) * 0 + (ch * h + y) * w + x;
}

export function setPx(img, C, H, W, y, x, values) {
  if (y < 0 || y >= H || x < 0 || x >= W) return;
  for (let c = 0; c < C; c++) img[(c * H + y) * W + x] = values[c === 0 && values.length === 1 ? 0 : c];
}

export function getPx(img, C, H, W, y, x) {
  const out = new Array(C);
  for (let c = 0; c < C; c++) out[c] = img[(c * H + y) * W + x];
  return out;
}

export function fillRect(img, C, H, W, x0, y0, x1, y1, color) {
  for (let y = Math.max(0, y0); y < Math.min(H, y1); y++) {
    for (let x = Math.max(0, x0); x < Math.min(W, x1); x++) setPx(img, C, H, W, y, x, color);
  }
}

export function fillEllipse(img, C, H, W, cx, cy, rx, ry, color, rotation) {
  const rot = rotation || 0;
  const cos = Math.cos(rot);
  const sin = Math.sin(rot);
  const span = Math.max(rx, ry) + 2;
  for (let y = Math.max(0, Math.floor(cy - span)); y < Math.min(H, Math.ceil(cy + span)); y++) {
    for (let x = Math.max(0, Math.floor(cx - span)); x < Math.min(W, Math.ceil(cx + span)); x++) {
      const dx = x - cx;
      const dy = y - cy;
      const rx2 = dx * cos + dy * sin;
      const ry2 = -dx * sin + dy * cos;
      if ((rx2 * rx2) / (rx * rx) + (ry2 * ry2) / (ry * ry) <= 1) setPx(img, C, H, W, y, x, color);
    }
  }
}

export function fillPolygon(img, C, H, W, pts, color) {
  let minY = Infinity;
  let maxY = -Infinity;
  let minX = Infinity;
  let maxX = -Infinity;
  for (const p of pts) {
    minY = Math.min(minY, p[1]);
    maxY = Math.max(maxY, p[1]);
    minX = Math.min(minX, p[0]);
    maxX = Math.max(maxX, p[0]);
  }
  for (let y = Math.max(0, Math.floor(minY)); y < Math.min(H, Math.ceil(maxY)); y++) {
    for (let x = Math.max(0, Math.floor(minX)); x < Math.min(W, Math.ceil(maxX)); x++) {
      if (pointInPolygon(x + 0.5, y + 0.5, pts)) setPx(img, C, H, W, y, x, color);
    }
  }
}

export function pointInPolygon(x, y, pts) {
  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const xi = pts[i][0];
    const yi = pts[i][1];
    const xj = pts[j][0];
    const yj = pts[j][1];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

export function drawLine(img, C, H, W, x0, y0, x1, y1, color, thickness) {
  const t = thickness || 1;
  const steps = Math.ceil(Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0)) * 2) + 1;
  for (let i = 0; i <= steps; i++) {
    const x = Math.round(x0 + (x1 - x0) * i / steps);
    const y = Math.round(y0 + (y1 - y0) * i / steps);
    fillEllipse(img, C, H, W, x, y, t / 2, t / 2, color);
  }
}

export function valueNoise(rng, w, h, cells, octaves, persistence) {
  octaves = octaves || 4;
  persistence = persistence === undefined ? 0.5 : persistence;
  const out = new Float32Array(w * h);
  let amp = 1;
  let total = 0;
  let c = cells || 4;
  for (let o = 0; o < octaves; o++) {
    const gw = c + 1;
    const gh = c + 1;
    const grid = new Float32Array(gw * gh);
    for (let i = 0; i < grid.length; i++) grid[i] = rng.next();
    for (let y = 0; y < h; y++) {
      const gy = y / h * c;
      const y0 = Math.floor(gy);
      const fy = smooth(gy - y0);
      for (let x = 0; x < w; x++) {
        const gx = x / w * c;
        const x0 = Math.floor(gx);
        const fx = smooth(gx - x0);
        const a = grid[y0 * gw + x0];
        const b = grid[y0 * gw + Math.min(gw - 1, x0 + 1)];
        const cc = grid[Math.min(gh - 1, y0 + 1) * gw + x0];
        const d = grid[Math.min(gh - 1, y0 + 1) * gw + Math.min(gw - 1, x0 + 1)];
        const top = a + (b - a) * fx;
        const bot = cc + (d - cc) * fx;
        out[y * w + x] += amp * (top + (bot - top) * fy);
      }
    }
    total += amp;
    amp *= persistence;
    c *= 2;
  }
  for (let i = 0; i < out.length; i++) out[i] /= total;
  return out;
}

function smooth(t) {
  return t * t * (3 - 2 * t);
}

export function bilinearResize(src, C, sH, sW, dH, dW) {
  const out = new Float32Array(C * dH * dW);
  for (let c = 0; c < C; c++) {
    for (let y = 0; y < dH; y++) {
      const sy = (y + 0.5) * sH / dH - 0.5;
      const y0 = Math.max(0, Math.min(sH - 1, Math.floor(sy)));
      const y1 = Math.max(0, Math.min(sH - 1, y0 + 1));
      const fy = sy - Math.floor(sy);
      for (let x = 0; x < dW; x++) {
        const sx = (x + 0.5) * sW / dW - 0.5;
        const x0 = Math.max(0, Math.min(sW - 1, Math.floor(sx)));
        const x1 = Math.max(0, Math.min(sW - 1, x0 + 1));
        const fx = sx - Math.floor(sx);
        const v = src[(c * sH + y0) * sW + x0] * (1 - fx) * (1 - fy) +
          src[(c * sH + y0) * sW + x1] * fx * (1 - fy) +
          src[(c * sH + y1) * sW + x0] * (1 - fx) * fy +
          src[(c * sH + y1) * sW + x1] * fx * fy;
        out[(c * dH + y) * dW + x] = v;
      }
    }
  }
  return out;
}

export function areaDownsample(src, C, sH, sW, factor) {
  const dH = Math.max(1, Math.floor(sH / factor));
  const dW = Math.max(1, Math.floor(sW / factor));
  const out = new Float32Array(C * dH * dW);
  for (let c = 0; c < C; c++) {
    for (let y = 0; y < dH; y++) {
      for (let x = 0; x < dW; x++) {
        let s = 0;
        let n = 0;
        for (let dy = 0; dy < factor; dy++) {
          for (let dx = 0; dx < factor; dx++) {
            const sy = y * factor + dy;
            const sx = x * factor + dx;
            if (sy < sH && sx < sW) {
              s += src[(c * sH + sy) * sW + sx];
              n++;
            }
          }
        }
        out[(c * dH + y) * dW + x] = n ? s / n : 0;
      }
    }
  }
  return out;
}

export function bicubicSample(src, C, sH, sW, y, x) {
  const out = new Float32Array(C);
  const iy = Math.floor(y);
  const ix = Math.floor(x);
  const fy = y - iy;
  const fx = x - ix;
  for (let c = 0; c < C; c++) {
    let v = 0;
    for (let m = -1; m <= 2; m++) {
      const wy = cubicWeight(fy - m);
      const sy = Math.max(0, Math.min(sH - 1, iy + m));
      for (let n = -1; n <= 2; n++) {
        const wx = cubicWeight(fx - n);
        const sx = Math.max(0, Math.min(sW - 1, ix + n));
        v += wy * wx * src[(c * sH + sy) * sW + sx];
      }
    }
    out[c] = v;
  }
  return out;
}

function cubicWeight(t) {
  const a = -0.5;
  const at = Math.abs(t);
  if (at <= 1) return (a + 2) * at * at * at - (a + 3) * at * at + 1;
  if (at < 2) return a * at * at * at - 5 * a * at * at + 8 * a * at - 4 * a;
  return 0;
}

export function bicubicResize(src, C, sH, sW, dH, dW) {
  const out = new Float32Array(C * dH * dW);
  for (let y = 0; y < dH; y++) {
    const sy = (y + 0.5) * sH / dH - 0.5;
    for (let x = 0; x < dW; x++) {
      const sx = (x + 0.5) * sW / dW - 0.5;
      const v = bicubicSample(src, C, sH, sW, sy, sx);
      for (let c = 0; c < C; c++) out[(c * dH + y) * dW + x] = v[c];
    }
  }
  return out;
}

export function addGaussianNoise(img, rng, std) {
  const out = new Float32Array(img.length);
  for (let i = 0; i < img.length; i++) out[i] = img[i] + std * rng.gauss();
  return out;
}

export function addPoissonNoise(img, rng, scale) {
  const out = new Float32Array(img.length);
  for (let i = 0; i < img.length; i++) {
    const lambda = Math.max(0, img[i]) * scale;
    let L = Math.exp(-lambda);
    let k = 0;
    let p = 1;
    do {
      k++;
      p *= rng.next();
    } while (p > L && k < 200);
    out[i] = (k - 1) / scale;
  }
  return out;
}

export function gaussianBlur(img, C, H, W, radius) {
  const r = Math.max(1, Math.round(radius || 1));
  const kernel = [];
  let sum = 0;
  for (let i = -r; i <= r; i++) {
    const v = Math.exp(-(i * i) / (2 * (r * 0.7) * (r * 0.7)));
    kernel.push(v);
    sum += v;
  }
  for (let i = 0; i < kernel.length; i++) kernel[i] /= sum;
  const tmp = new Float32Array(img.length);
  const out = new Float32Array(img.length);
  for (let c = 0; c < C; c++) {
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        let v = 0;
        for (let k = -r; k <= r; k++) {
          const sx = Math.max(0, Math.min(W - 1, x + k));
          v += kernel[k + r] * img[(c * H + y) * W + sx];
        }
        tmp[(c * H + y) * W + x] = v;
      }
    }
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        let v = 0;
        for (let k = -r; k <= r; k++) {
          const sy = Math.max(0, Math.min(H - 1, y + k));
          v += kernel[k + r] * tmp[(c * H + sy) * W + x];
        }
        out[(c * H + y) * W + x] = v;
      }
    }
  }
  return out;
}

export function flipHorizontal(img, C, H, W) {
  const out = new Float32Array(img.length);
  for (let c = 0; c < C; c++) {
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) out[(c * H + y) * W + x] = img[(c * H + y) * W + (W - 1 - x)];
    }
  }
  return out;
}

export function flipVertical(img, C, H, W) {
  const out = new Float32Array(img.length);
  for (let c = 0; c < C; c++) {
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) out[(c * H + y) * W + x] = img[(c * H + (H - 1 - y)) * W + x];
    }
  }
  return out;
}

export function rotate90(img, C, H, W, times) {
  let cur = img;
  let ch = H;
  let cw = W;
  for (let t = 0; t < ((times % 4) + 4) % 4; t++) {
    const out = new Float32Array(cur.length);
    for (let c = 0; c < C; c++) {
      for (let y = 0; y < ch; y++) {
        for (let x = 0; x < cw; x++) out[(c * cw + x) * ch + (ch - 1 - y)] = cur[(c * ch + y) * cw + x];
      }
    }
    const tmp = ch;
    ch = cw;
    cw = tmp;
    cur = out;
  }
  return { data: cur, h: ch, w: cw };
}

export function randomCrop(img, C, H, W, size, rng) {
  const y0 = rng.int(Math.max(1, H - size + 1));
  const x0 = rng.int(Math.max(1, W - size + 1));
  const out = new Float32Array(C * size * size);
  for (let c = 0; c < C; c++) {
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        out[(c * size + y) * size + x] = img[(c * H + y0 + y) * W + x0 + x];
      }
    }
  }
  return out;
}

export function adjustBrightness(img, delta) {
  const out = new Float32Array(img.length);
  for (let i = 0; i < img.length; i++) out[i] = img[i] + delta;
  return out;
}

export function adjustContrast(img, factor) {
  const out = new Float32Array(img.length);
  for (let i = 0; i < img.length; i++) out[i] = (img[i] - 0.5) * factor + 0.5;
  return out;
}

export function cutout(img, C, H, W, rng, size) {
  const out = new Float32Array(img);
  const s = size || Math.max(2, Math.floor(Math.min(H, W) / 4));
  const y0 = rng.int(Math.max(1, H - s));
  const x0 = rng.int(Math.max(1, W - s));
  for (let c = 0; c < C; c++) {
    for (let y = y0; y < Math.min(H, y0 + s); y++) {
      for (let x = x0; x < Math.min(W, x0 + s); x++) out[(c * H + y) * W + x] = 0;
    }
  }
  return out;
}

export function jpegQuantize(img, C, H, W, quality) {
  const levels = Math.max(2, Math.round((quality === undefined ? 0.4 : quality) * 32));
  const out = new Float32Array(img.length);
  for (let i = 0; i < img.length; i++) out[i] = Math.round(img[i] * levels) / levels;
  return out;
}

export function synthNatural(rng, C, H, W) {
  const img = new Float32Array(C * H * W);
  const base = valueNoise(rng, W, H, 3, 4, 0.55);
  const detail = valueNoise(rng, W, H, 12, 3, 0.5);
  const palette = [];
  for (let c = 0; c < C; c++) palette.push([rng.range(0.05, 0.55), rng.range(0.35, 0.95)]);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const v = base[y * W + x] * 0.75 + detail[y * W + x] * 0.25;
      const edge = Math.abs(v - 0.5) < 0.06 ? 0.25 : 0;
      for (let c = 0; c < C; c++) {
        const lo = palette[c][0];
        const hi = palette[c][1];
        img[(c * H + y) * W + x] = clamp01(lo + (hi - lo) * v + edge * (c === 0 ? 0.2 : -0.1));
      }
    }
  }
  const shapes = 2 + rng.int(4);
  for (let s = 0; s < shapes; s++) {
    const kind = rng.int(3);
    const color = [rng.range(0, 1), rng.range(0, 1), rng.range(0, 1)].slice(0, C);
    if (kind === 0) {
      fillEllipse(img, C, H, W, rng.range(0, W), rng.range(0, H), rng.range(2, W / 4), rng.range(2, H / 4), color, rng.range(0, 3.14));
    } else if (kind === 1) {
      const pts = [];
      const n = 3 + rng.int(4);
      const cx = rng.range(0.2, 0.8) * W;
      const cy = rng.range(0.2, 0.8) * H;
      const r = rng.range(0.1, 0.3) * Math.min(H, W);
      for (let i = 0; i < n; i++) {
        const a = (i / n) * Math.PI * 2 + rng.range(-0.2, 0.2);
        pts.push([cx + Math.cos(a) * r * rng.range(0.7, 1.2), cy + Math.sin(a) * r * rng.range(0.7, 1.2)]);
      }
      fillPolygon(img, C, H, W, pts, color);
    } else {
      drawLine(img, C, H, W, rng.range(0, W), rng.range(0, H), rng.range(0, W), rng.range(0, H), color, 1 + rng.int(2));
    }
  }
  return img;
}

export function synthFace(rng, H, W, withBox) {
  const C = 3;
  const img = new Float32Array(C * H * W);
  const bgL = rng.range(0.05, 0.45);
  const bg = valueNoise(rng, W, H, 4, 2, 0.5);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const v = bgL + bg[y * W + x] * 0.25;
      for (let c = 0; c < C; c++) img[(c * H + y) * W + x] = clamp01(v + (c - 1) * 0.03);
    }
  }
  const scale = Math.min(H, W);
  const cx = rng.range(0.3, 0.7) * W;
  const cy = rng.range(0.3, 0.7) * H;
  const rw = scale * rng.range(0.16, 0.3);
  const rh = rw * rng.range(1.1, 1.45);
  const rot = rng.range(-0.35, 0.35);
  const skin = [rng.range(0.45, 0.95), rng.range(0.3, 0.72), rng.range(0.2, 0.55)];
  fillEllipse(img, C, H, W, cx, cy, rw, rh, skin, rot);
  const eyeY = cy - rh * 0.18;
  const eyeDx = rw * 0.42;
  const eyeR = rw * 0.16;
  fillEllipse(img, C, H, W, cx - eyeDx, eyeY, eyeR, eyeR * 0.75, [0.94, 0.94, 0.95], rot);
  fillEllipse(img, C, H, W, cx + eyeDx, eyeY, eyeR, eyeR * 0.75, [0.94, 0.94, 0.95], rot);
  const pupil = [0.08, 0.07, 0.09];
  fillEllipse(img, C, H, W, cx - eyeDx + rng.range(-1, 1), eyeY, eyeR * 0.45, eyeR * 0.45, pupil, rot);
  fillEllipse(img, C, H, W, cx + eyeDx + rng.range(-1, 1), eyeY, eyeR * 0.45, eyeR * 0.45, pupil, rot);
  const noseY = cy + rh * 0.08;
  drawLine(img, C, H, W, cx, noseY - rh * 0.1, cx + rw * 0.08, noseY + rh * 0.06, [skin[0] * 0.75, skin[1] * 0.7, skin[2] * 0.7], Math.max(1, rw * 0.08));
  const mouthY = cy + rh * 0.45;
  const mouthW = rw * rng.range(0.35, 0.6);
  const open = rng.range(0.05, 0.3);
  fillEllipse(img, C, H, W, cx, mouthY, mouthW * 0.5, mouthW * 0.5 * open, [rng.range(0.25, 0.6), rng.range(0.1, 0.25), rng.range(0.15, 0.28)], rot);
  if (rng.next() < 0.4) {
    fillEllipse(img, C, H, W, cx, cy - rh * 0.85, rw * rng.range(1.0, 1.2), rh * rng.range(0.35, 0.6),
      [rng.range(0.05, 0.3), rng.range(0.03, 0.2), rng.range(0.02, 0.15)], rot);
  }
  const box = {
    x: clamp01((cx - rw * 1.15) / W),
    y: clamp01((cy - rh * 1.2) / H),
    w: clamp01((rw * 2.3) / W),
    h: clamp01((rh * 2.4) / H)
  };
  box.w = Math.min(box.w, 1 - box.x);
  box.h = Math.min(box.h, 1 - box.y);
  return withBox ? { image: img, box } : img;
}

export function synthHand(rng, H, W, withBox) {
  const C = 3;
  const img = new Float32Array(C * H * W);
  const bgL = rng.range(0.05, 0.5);
  for (let i = 0; i < img.length / C; i++) {
    const v = bgL + rng.range(-0.03, 0.03);
    for (let c = 0; c < C; c++) img[c * (img.length / C) + i] = clamp01(v);
  }
  const cx = rng.range(0.35, 0.65) * W;
  const cy = rng.range(0.4, 0.7) * H;
  const s = Math.min(H, W) * rng.range(0.18, 0.3);
  const skin = [rng.range(0.4, 0.95), rng.range(0.28, 0.7), rng.range(0.2, 0.55)];
  const rot = rng.range(-0.6, 0.6);
  fillEllipse(img, C, H, W, cx, cy, s * 0.75, s * 0.9, skin, rot);
  const fingers = 4 + (rng.next() < 0.7 ? 1 : 0);
  for (let i = 0; i < fingers; i++) {
    const spread = (i - (fingers - 1) / 2) * 0.28;
    const a = rot - Math.PI / 2 + spread;
    const len = s * rng.range(0.7, 1.25);
    const ex = cx + Math.cos(a) * (s * 0.55 + len * 0.5);
    const ey = cy + Math.sin(a) * (s * 0.55 + len * 0.5);
    const cos = Math.cos(a);
    const sin = Math.sin(a);
    fillEllipse(img, C, H, W, ex, ey, s * 0.2, len * 0.5, skin, Math.atan2(sin, cos) + Math.PI / 2);
  }
  const box = {
    x: clamp01((cx - s * 1.3) / W),
    y: clamp01((cy - s * 1.7) / H),
    w: clamp01((s * 2.6) / W),
    h: clamp01((s * 2.8) / H)
  };
  box.w = Math.min(box.w, 1 - box.x);
  box.h = Math.min(box.h, 1 - box.y);
  return withBox ? { image: img, box } : img;
}

export function synthSubject(rng, H, W) {
  const C = 3;
  const img = new Float32Array(C * H * W);
  const mask = new Float32Array(H * W);
  const bgTone = [rng.range(0, 1), rng.range(0, 1), rng.range(0, 1)];
  const bgNoise = valueNoise(rng, W, H, 6, 3, 0.5);
  const pattern = rng.int(3);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let v = bgNoise[y * W + x];
      if (pattern === 1) v = 0.5 + 0.5 * Math.sin((x / W) * Math.PI * rng.range(2, 8) + y * 0.1);
      if (pattern === 2) v = ((Math.floor(x / 4) + Math.floor(y / 4)) % 2) * 0.4 + v * 0.3;
      for (let c = 0; c < C; c++) {
        img[(c * H + y) * W + x] = clamp01(bgTone[c] * 0.6 + v * 0.5 - 0.15);
      }
    }
  }
  const cx = rng.range(0.3, 0.7) * W;
  const cy = rng.range(0.3, 0.7) * H;
  const r = Math.min(H, W) * rng.range(0.18, 0.36);
  const fg = [rng.range(0, 1), rng.range(0, 1), rng.range(0, 1)];
  const kind = rng.int(3);
  const limbs = 3 + rng.int(5);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const dx = x - cx;
      const dy = y - cy;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const ang = Math.atan2(dy, dx);
      let inside = false;
      if (kind === 0) inside = dist <= r;
      else if (kind === 1) inside = Math.abs(dx) <= r * 0.75 && Math.abs(dy) <= r * 1.1;
      else inside = dist <= r * (0.72 + 0.28 * Math.cos(limbs * ang + rng.int(1) * 0));
      if (inside) {
        mask[y * W + x] = 1;
        const shade = 0.85 + 0.3 * (1 - dist / (r * 1.4));
        for (let c = 0; c < C; c++) img[(c * H + y) * W + x] = clamp01(fg[c] * shade);
      }
    }
  }
  return { image: img, mask };
}

export function synthSaliency(rng, H, W) {
  const C = 3;
  const img = new Float32Array(C * H * W);
  const sal = new Float32Array(H * W);
  const bg = valueNoise(rng, W, H, 5, 3, 0.5);
  const tone = rng.range(0.25, 0.6);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const v = tone + bg[y * W + x] * 0.3 - 0.15;
      for (let c = 0; c < C; c++) img[(c * H + y) * W + x] = clamp01(v + (c - 1) * 0.04);
    }
  }
  const n = 1 + rng.int(2);
  const items = [];
  for (let i = 0; i < n; i++) {
    const cx = rng.range(0.15, 0.85) * W;
    const cy = rng.range(0.15, 0.85) * H;
    const r = Math.min(H, W) * rng.range(0.1, 0.26);
    const color = [rng.range(0, 1), rng.range(0, 1), rng.range(0, 1)];
    const contrast = 0.4 + rng.range(0, 0.6);
    items.push({ cx, cy, r, color, contrast });
  }
  for (const it of items) {
    fillEllipse(img, C, H, W, it.cx, it.cy, it.r, it.r * rng.range(0.8, 1.25), it.color, rng.range(0, 3));
  }
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let best = 0;
      for (const it of items) {
        const dx = x - it.cx;
        const dy = y - it.cy;
        const d = Math.sqrt(dx * dx + dy * dy);
        const v = Math.max(0, 1 - d / (it.r * 1.6)) * it.contrast;
        best = Math.max(best, v);
      }
      sal[y * W + x] = best;
    }
  }
  return { image: img, saliency: sal, primary: items[0] };
}

export function stylize(img, C, H, W, rng, style) {
  const out = new Float32Array(img.length);
  const palette = [];
  const nColors = 3 + rng.int(4);
  for (let i = 0; i < nColors; i++) palette.push([rng.range(0, 1), rng.range(0, 1), rng.range(0, 1)]);
  const lum = new Float32Array(H * W);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let s = 0;
      for (let c = 0; c < C; c++) s += img[(c * H + y) * W + x];
      lum[y * W + x] = s / C;
    }
  }
  const levels = style.levels || 5;
  const brush = style.brush || 2;
  const warp = style.warp || 1.5;
  const noise = valueNoise(rng, W, H, 5, 3, 0.5);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const wx = Math.max(0, Math.min(W - 1, Math.round(x + (noise[y * W + x] - 0.5) * warp * brush)));
      const wy = Math.max(0, Math.min(H - 1, Math.round(y + (noise[y * W + x] - 0.5) * warp * brush * 0.6)));
      const v = lum[wy * W + wx];
      const q = Math.min(levels - 1, Math.floor(v * levels));
      const col = palette[q % palette.length];
      const grain = (noise[y * W + x] - 0.5) * 0.12;
      for (let c = 0; c < C; c++) {
        out[(c * H + y) * W + x] = clamp01(col[c] * 0.85 + grain + (v - q / levels) * 0.2);
      }
    }
  }
  return out;
}

export function explicitLikePattern(rng, C, H, W, kind) {
  const img = new Float32Array(C * H * W);
  const skinTone = [rng.range(0.6, 0.95), rng.range(0.4, 0.7), rng.range(0.35, 0.6)];
  const bg = rng.range(0.05, 0.35);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      for (let c = 0; c < C; c++) img[(c * H + y) * W + x] = clamp01(bg + (c === 0 ? 0.05 : 0));
    }
  }
  if (kind === 0) {
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const v = 0.5 + 0.5 * Math.sin(x * 0.6 + y * 0.4 + rng.next() * 0.02);
        for (let c = 0; c < C; c++) img[(c * H + y) * W + x] = clamp01(bg + v * 0.2);
      }
    }
    const blobs = 3 + rng.int(4);
    for (let i = 0; i < blobs; i++) {
      fillEllipse(img, C, H, W, rng.range(0, W), rng.range(0, H), rng.range(0.08, 0.3) * W, rng.range(0.08, 0.3) * H, skinTone, rng.range(0, 3));
    }
  } else if (kind === 1) {
    const cx = W * rng.range(0.35, 0.65);
    const cy = H * rng.range(0.3, 0.6);
    fillEllipse(img, C, H, W, cx, cy, W * 0.22, H * 0.3, skinTone, rng.range(-0.3, 0.3));
    fillEllipse(img, C, H, W, cx, cy - H * 0.32, W * 0.14, H * 0.16, skinTone, 0);
    for (let i = 0; i < 4; i++) {
      const a = rng.range(0.6, 2.5);
      fillEllipse(img, C, H, W, cx + Math.cos(a) * W * 0.28, cy + Math.sin(a) * H * 0.2, W * 0.07, H * 0.18, skinTone, a);
    }
  } else {
    const base = valueNoise(rng, W, H, 3, 2, 0.5);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const v = base[y * W + x];
        for (let c = 0; c < C; c++) img[(c * H + y) * W + x] = clamp01(v * 0.8 + (c - 1) * 0.05);
      }
    }
    const n = 2 + rng.int(5);
    for (let i = 0; i < n; i++) {
      const pts = [];
      const cx = rng.range(0.2, 0.8) * W;
      const cy = rng.range(0.2, 0.8) * H;
      const r = rng.range(0.1, 0.3) * Math.min(H, W);
      const k = 3 + rng.int(5);
      for (let j = 0; j < k; j++) {
        const a = (j / k) * Math.PI * 2;
        pts.push([cx + Math.cos(a) * r * rng.range(0.6, 1.3), cy + Math.sin(a) * r * rng.range(0.6, 1.3)]);
      }
      fillPolygon(img, C, H, W, pts, [rng.range(0.1, 0.9), rng.range(0.1, 0.9), rng.range(0.1, 0.9)]);
    }
  }
  return img;
}

export function clamp01(v) {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

export function toRgba(img, C, H, W) {
  const out = new Uint8ClampedArray(H * W * 4);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const o = (y * W + x) * 4;
      for (let c = 0; c < 3; c++) {
        const v = c < C ? img[(c * H + y) * W + x] : 0;
        out[o + c] = Math.round(clamp01(v * 0.5 + 0.5) * 255);
      }
      out[o + 3] = 255;
    }
  }
  return out;
}

export function fromRgba(rgba, C, H, W) {
  const out = new Float32Array(C * H * W);
  for (let c = 0; c < C; c++) {
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        out[(c * H + y) * W + x] = rgba[(y * W + x) * 4 + c] / 255 * 2 - 1;
      }
    }
  }
  return out;
}

export function normalizeImage(img, mode) {
  const out = new Float32Array(img.length);
  if (mode === 'zero_one') {
    for (let i = 0; i < img.length; i++) out[i] = img[i];
    return out;
  }
  for (let i = 0; i < img.length; i++) out[i] = img[i] * 2 - 1;
  return out;
}
