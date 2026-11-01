import { toFloat16Array, toBFloat16Array, fromFloat16Array, fromBFloat16Array } from '../core/quant.js';

const DTYPE_BYTES = { F64: 8, F32: 4, F16: 2, BF16: 2, I64: 8, I32: 4, I16: 2, I8: 1, U8: 1, BOOL: 1 };

const DTYPE_FOR = {
  float32: 'F32',
  float64: 'F64',
  float16: 'F16',
  bfloat16: 'BF16',
  int64: 'I64',
  int32: 'I32',
  int16: 'I16',
  int8: 'I8',
  uint8: 'U8',
  bool: 'BOOL'
};

function shapeSize(shape) {
  let n = 1;
  for (const d of shape) n *= d;
  return n;
}

function rawView(typed) {
  return new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength);
}

const ALREADY_ENCODED = {
  F16: Uint16Array,
  BF16: Uint16Array,
  I8: Int8Array,
  U8: Uint8Array,
  I32: Int32Array,
  I64: BigInt64Array,
  F32: Float32Array,
  F64: Float64Array
};

function encodePayload(tensor) {
  const dtype = DTYPE_FOR[tensor.dtype] || tensor.dtype;
  const values = tensor.values;
  const expectedType = ALREADY_ENCODED[dtype];
  if (expectedType && values instanceof expectedType) return { dtype, bytes: rawView(values) };
  if (dtype === 'F32') {
    return { dtype, bytes: values instanceof Float32Array ? rawView(values) : rawView(Float32Array.from(values)) };
  }
  if (dtype === 'F16') {
    const src = values instanceof Float32Array ? values : Float32Array.from(values);
    return { dtype, bytes: rawView(toFloat16Array(src)) };
  }
  if (dtype === 'BF16') {
    const src = values instanceof Float32Array ? values : Float32Array.from(values);
    return { dtype, bytes: rawView(toBFloat16Array(src)) };
  }
  if (dtype === 'I8') {
    const src = values instanceof Int8Array ? values : Int8Array.from(Array.from(values, (v) => Math.round(v)));
    return { dtype, bytes: rawView(src) };
  }
  if (dtype === 'U8') {
    const src = values instanceof Uint8Array ? values : Uint8Array.from(Array.from(values, (v) => Math.round(v)));
    return { dtype, bytes: rawView(src) };
  }
  if (dtype === 'I32') {
    const out = new Int32Array(values.length);
    for (let i = 0; i < values.length; i++) out[i] = Math.round(values[i]);
    return { dtype, bytes: rawView(out) };
  }
  if (dtype === 'I64') {
    const out = new BigInt64Array(values.length);
    for (let i = 0; i < values.length; i++) out[i] = BigInt(Math.round(Number(values[i])));
    return { dtype, bytes: rawView(out) };
  }
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) out[i] = values[i];
  return { dtype: 'F32', bytes: rawView(out) };
}

export function writeSafetensors(tensors, metadata) {
  const header = {};
  const payloads = [];
  let offset = 0;

  if (metadata && Object.keys(metadata).length) {
    const meta = {};
    for (const k of Object.keys(metadata)) {
      const v = metadata[k];
      meta[k] = typeof v === 'string' ? v : JSON.stringify(v);
    }
    header.__metadata__ = meta;
  }

  for (const t of tensors) {
    const shape = t.shape && t.shape.length ? t.shape.slice() : [t.values.length];
    const expected = shapeSize(shape);
    if (expected !== t.values.length) throw new Error('tensor ' + t.name + ' shape ' + JSON.stringify(shape) + ' does not match ' + t.values.length + ' values');
    const enc = encodePayload(t);
    const size = enc.bytes.length;
    header[t.name] = { dtype: enc.dtype, shape, data_offsets: [offset, offset + size] };
    payloads.push(enc.bytes);
    offset += size;
  }

  const headerJson = JSON.stringify(header);
  const headerBytes = new TextEncoder().encode(headerJson);
  const prefix = new Uint8Array(8);
  new DataView(prefix.buffer).setBigUint64(0, BigInt(headerBytes.length), true);

  const total = 8 + headerBytes.length + offset;
  const out = new Uint8Array(total);
  out.set(prefix, 0);
  out.set(headerBytes, 8);
  let p = 8 + headerBytes.length;
  for (const chunk of payloads) {
    out.set(chunk, p);
    p += chunk.length;
  }
  return { bytes: out, header, headerBytes: headerBytes.length, dataBytes: offset, totalBytes: total };
}

export function readSafetensors(bytes) {
  const data = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const headerLen = Number(view.getBigUint64(0, true));
  const headerJson = new TextDecoder().decode(data.subarray(8, 8 + headerLen));
  const header = JSON.parse(headerJson);
  const base = 8 + headerLen;
  const tensors = [];
  for (const name of Object.keys(header)) {
    if (name === '__metadata__') continue;
    const entry = header[name];
    const [start, end] = entry.data_offsets;
    const raw = data.slice(base + start, base + end);
    const rv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
    const count = shapeSize(entry.shape);
    let values;
    switch (entry.dtype) {
      case 'F32': {
        values = new Float32Array(count);
        for (let i = 0; i < count; i++) values[i] = rv.getFloat32(i * 4, true);
        break;
      }
      case 'F16': {
        const h = new Uint16Array(count);
        for (let i = 0; i < count; i++) h[i] = rv.getUint16(i * 2, true);
        values = fromFloat16Array(h);
        break;
      }
      case 'BF16': {
        const h = new Uint16Array(count);
        for (let i = 0; i < count; i++) h[i] = rv.getUint16(i * 2, true);
        values = fromBFloat16Array(h);
        break;
      }
      case 'I8': {
        values = new Int8Array(count);
        for (let i = 0; i < count; i++) values[i] = rv.getInt8(i);
        break;
      }
      case 'U8': {
        values = new Uint8Array(count);
        for (let i = 0; i < count; i++) values[i] = rv.getUint8(i);
        break;
      }
      case 'I32': {
        values = new Int32Array(count);
        for (let i = 0; i < count; i++) values[i] = rv.getInt32(i * 4, true);
        break;
      }
      case 'I64': {
        values = new Float64Array(count);
        for (let i = 0; i < count; i++) values[i] = Number(rv.getBigInt64(i * 8, true));
        break;
      }
      case 'F64': {
        values = new Float64Array(count);
        for (let i = 0; i < count; i++) values[i] = rv.getFloat64(i * 8, true);
        break;
      }
      default: throw new Error('unsupported safetensors dtype ' + entry.dtype);
    }
    tensors.push({ name, dtype: entry.dtype, shape: entry.shape, values, raw });
  }
  return { tensors, metadata: header.__metadata__ || null, headerBytes: headerLen, dataBytes: data.length - base };
}

export function safetensorsFromStates(states, options) {
  const opts = options || {};
  const precision = opts.precision || 'fp32';
  const map = { fp32: 'float32', fp16: 'float16', bf16: 'bfloat16', int8: 'int8', uint8: 'uint8' };
  const dtype = map[precision] || 'float32';
  const tensors = [];
  const names = opts.names || Object.keys(states);
  for (const name of names) {
    const node = states[name];
    const values = node && node.v ? node.v : node;
    if (!values || values.length === undefined) continue;
    const shape = node && node.shape ? node.shape.slice() : [values.length];
    tensors.push({ name: name.replace(/\./g, '__'), values: values instanceof Float32Array ? values : Float32Array.from(values), shape, originalName: name });
  }
  return { tensors, dtype };
}

export function safetensorsLayout(tensors) {
  let offset = 0;
  const rows = [];
  for (const t of tensors) {
    const enc = encodePayload(t);
    rows.push({ name: t.name, dtype: enc.dtype, shape: t.shape || [t.values.length], elements: t.values.length, bytes: enc.bytes.length, offset });
    offset += enc.bytes.length;
  }
  return rows;
}

export function bytesForDtype(dtype) {
  return DTYPE_BYTES[dtype] || 0;
}
