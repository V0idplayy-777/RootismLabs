import { crc32 } from './hash.js';

const LOCAL_SIG = 0x04034b50;
const CENTRAL_SIG = 0x02014b50;
const END_SIG = 0x06054b50;
const UTF8_FLAG = 0x0800;

function encoder() {
  return new TextEncoder();
}

function dosTime(date) {
  const d = date || new Date();
  const time = (d.getUTCHours() << 11) | (d.getUTCMinutes() << 5) | Math.floor(d.getUTCSeconds() / 2);
  const day = ((Math.max(1980, d.getUTCFullYear()) - 1980) << 9) | ((d.getUTCMonth() + 1) << 5) | d.getUTCDate();
  return { time: time & 0xffff, date: day & 0xffff };
}

function asBytes(data) {
  if (data instanceof Uint8Array) return data;
  if (ArrayBuffer.isView(data)) return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  if (data instanceof ArrayBuffer) return new Uint8Array(data);
  if (typeof data === 'string') return encoder().encode(data);
  return new Uint8Array(0);
}

export function zipWrite(entries, options) {
  const opts = options || {};
  const enc = encoder();
  const stamp = dosTime(opts.date || null);
  const local = [];
  const central = [];
  let offset = 0;

  for (const entry of entries) {
    const name = String(entry.name || 'file');
    const nameBytes = enc.encode(name);
    const data = asBytes(entry.data);
    const crc = data.length ? crc32(data) >>> 0 : 0;
    const comment = entry.comment ? enc.encode(String(entry.comment)) : new Uint8Array(0);

    const head = new Uint8Array(30 + nameBytes.length + (entry.extra ? entry.extra.length : 0));
    const hv = new DataView(head.buffer);
    hv.setUint32(0, LOCAL_SIG, true);
    hv.setUint16(4, 20, true);
    hv.setUint16(6, UTF8_FLAG, true);
    hv.setUint16(8, 0, true);
    hv.setUint16(10, stamp.time, true);
    hv.setUint16(12, stamp.date, true);
    hv.setUint32(14, crc, true);
    hv.setUint32(18, data.length, true);
    hv.setUint32(22, data.length, true);
    hv.setUint16(26, nameBytes.length, true);
    hv.setUint16(28, entry.extra ? entry.extra.length : 0, true);
    head.set(nameBytes, 30);
    if (entry.extra) head.set(entry.extra, 30 + nameBytes.length);
    local.push(head, data);

    const cd = new Uint8Array(46 + nameBytes.length + comment.length);
    const cv = new DataView(cd.buffer);
    cv.setUint32(0, CENTRAL_SIG, true);
    cv.setUint16(4, 0x031e, true);
    cv.setUint16(6, 20, true);
    cv.setUint16(8, UTF8_FLAG, true);
    cv.setUint16(10, 0, true);
    cv.setUint16(12, stamp.time, true);
    cv.setUint16(14, stamp.date, true);
    cv.setUint32(16, crc, true);
    cv.setUint32(20, data.length, true);
    cv.setUint32(24, data.length, true);
    cv.setUint16(28, nameBytes.length, true);
    cv.setUint16(30, 0, true);
    cv.setUint16(32, comment.length, true);
    cv.setUint16(34, 0, true);
    cv.setUint16(36, 0, true);
    cv.setUint32(38, 0, true);
    cv.setUint32(42, offset >>> 0, true);
    cd.set(nameBytes, 46);
    if (comment.length) cd.set(comment, 46 + nameBytes.length);
    central.push(cd);

    offset += head.length + data.length;
  }

  const cdSize = central.reduce((a, c) => a + c.length, 0);
  const end = new Uint8Array(22);
  const ev = new DataView(end.buffer);
  ev.setUint32(0, END_SIG, true);
  ev.setUint16(4, 0, true);
  ev.setUint16(6, 0, true);
  ev.setUint16(8, entries.length, true);
  ev.setUint16(10, entries.length, true);
  ev.setUint32(12, cdSize >>> 0, true);
  ev.setUint32(16, offset >>> 0, true);
  ev.setUint16(20, 0, true);

  const chunks = local.concat(central, [end]);
  const total = chunks.reduce((a, c) => a + c.length, 0);
  const out = new Uint8Array(total);
  let p = 0;
  for (const c of chunks) {
    out.set(c, p);
    p += c.length;
  }
  return out;
}

export function zipRead(bytes) {
  const data = asBytes(bytes);
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  let endOffset = -1;
  for (let i = data.length - 22; i >= 0 && i >= data.length - 65558; i--) {
    if (view.getUint32(i, true) === END_SIG) {
      endOffset = i;
      break;
    }
  }
  if (endOffset < 0) throw new Error('not a zip archive: end of central directory not found');
  const count = view.getUint16(endOffset + 10, true);
  const cdSize = view.getUint32(endOffset + 12, true);
  let cdOffset = view.getUint32(endOffset + 16, true);
  if (cdOffset === 0xffffffff || count === 0xffff || cdSize === 0xffffffff) {
    for (let i = endOffset - 20; i >= 0; i--) {
      if (view.getUint32(i, true) === 0x06064b50) {
        cdOffset = Number(view.getBigUint64(i + 48, true));
        break;
      }
    }
  }
  const dec = new TextDecoder();
  const out = [];
  let p = cdOffset;
  for (let n = 0; n < count; n++) {
    if (view.getUint32(p, true) !== CENTRAL_SIG) throw new Error('corrupt central directory at entry ' + n);
    const method = view.getUint16(p + 10, true);
    const crc = view.getUint32(p + 16, true);
    const compSize = view.getUint32(p + 20, true);
    const uncompSize = view.getUint32(p + 24, true);
    const nameLen = view.getUint16(p + 28, true);
    const extraLen = view.getUint16(p + 30, true);
    const commentLen = view.getUint16(p + 32, true);
    const localOffset = view.getUint32(p + 42, true);
    const name = dec.decode(data.subarray(p + 46, p + 46 + nameLen));
    p += 46 + nameLen + extraLen + commentLen;

    const lNameLen = view.getUint16(localOffset + 26, true);
    const lExtraLen = view.getUint16(localOffset + 28, true);
    const start = localOffset + 30 + lNameLen + lExtraLen;
    if (method !== 0) throw new Error('unsupported compression method ' + method + ' in ' + name);
    const payload = data.slice(start, start + uncompSize);
    const actual = payload.length ? crc32(payload) >>> 0 : 0;
    out.push({ name, data: payload, crc, size: uncompSize, compSize, crcOk: actual === crc });
  }
  return out;
}

export function zipGet(bytes, name) {
  const entries = zipRead(bytes);
  for (const e of entries) if (e.name === name) return e.data;
  return null;
}

export function zipVerify(bytes) {
  const entries = zipRead(bytes);
  const bad = entries.filter((e) => !e.crcOk).map((e) => e.name);
  return { entries: entries.length, totalBytes: entries.reduce((a, e) => a + e.size, 0), corrupt: bad, ok: bad.length === 0 };
}
