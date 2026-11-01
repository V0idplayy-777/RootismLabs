export const WIRE_VARINT = 0;
export const WIRE_FIXED64 = 1;
export const WIRE_BYTES = 2;
export const WIRE_FIXED32 = 5;

const enc = new TextEncoder();

export function encodeVarint(value) {
  let v = Number(value);
  if (!Number.isFinite(v)) v = 0;
  if (v < 0) {
    let x = BigInt(Math.trunc(v)) & 0xffffffffffffffffn;
    const out = [];
    while (x > 0x7fn) {
      out.push(Number(x & 0x7fn) | 0x80);
      x >>= 7n;
    }
    out.push(Number(x & 0x7fn));
    return Uint8Array.from(out);
  }
  const out = [];
  while (v > 0x7f) {
    out.push((v & 0x7f) | 0x80);
    v = Math.floor(v / 128);
  }
  out.push(v & 0x7f);
  return Uint8Array.from(out);
}

export function decodeVarint(bytes, offset) {
  let result = 0;
  let shift = 1;
  let i = offset;
  let big = null;
  while (i < bytes.length) {
    const b = bytes[i];
    i += 1;
    const part = b & 0x7f;
    const place = i - offset - 1;
    if (big === null && shift > 4398046511104) {
      big = BigInt(result) + (BigInt(part) << BigInt(place * 7));
    } else if (big === null) {
      result += part * shift;
    } else {
      big += BigInt(part) << BigInt(place * 7);
    }
    if ((b & 0x80) === 0) {
      if (big === null) return [result, i];
      const masked = big & 0xffffffffffffffffn;
      const signed = masked > 0x7fffffffffffffffn ? masked - 0x10000000000000000n : masked;
      const asNumber = Number(signed);
      return [Number.isSafeInteger(asNumber) ? asNumber : signed, i];
    }
    shift *= 128;
  }
  throw new Error('truncated varint at offset ' + offset);
}

export class ProtoWriter {
  constructor() {
    this.chunks = [];
    this.length = 0;
  }

  raw(bytes) {
    const b = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    this.chunks.push(b);
    this.length += b.length;
    return this;
  }

  byte(value) {
    this.chunks.push(Uint8Array.of(value & 0xff));
    this.length += 1;
    return this;
  }

  uvarint(value) {
    return this.raw(encodeVarint(value));
  }

  tag(field, wire) {
    if (!Number.isInteger(field) || field < 1) throw new Error('invalid protobuf field number: ' + field);
    return this.uvarint(field * 8 + wire);
  }

  varintField(field, value) {
    if (value === undefined || value === null) return this;
    this.tag(field, WIRE_VARINT);
    return this.uvarint(value);
  }

  boolField(field, value) {
    if (value === undefined || value === null) return this;
    this.tag(field, WIRE_VARINT);
    return this.byte(value ? 1 : 0);
  }

  fixed32Field(field, value) {
    const b = new Uint8Array(4);
    new DataView(b.buffer).setUint32(0, value >>> 0, true);
    this.tag(field, WIRE_FIXED32);
    return this.raw(b);
  }

  floatField(field, value) {
    const b = new Uint8Array(4);
    new DataView(b.buffer).setFloat32(0, value, true);
    this.tag(field, WIRE_FIXED32);
    return this.raw(b);
  }

  doubleField(field, value) {
    const b = new Uint8Array(8);
    new DataView(b.buffer).setFloat64(0, value, true);
    this.tag(field, WIRE_FIXED64);
    return this.raw(b);
  }

  int64Field(field, value) {
    const b = new Uint8Array(8);
    new DataView(b.buffer).setBigInt64(0, BigInt(Math.round(Number(value))), true);
    this.tag(field, WIRE_FIXED64);
    return this.raw(b);
  }

  bytesField(field, bytes) {
    if (bytes === undefined || bytes === null) return this;
    const b = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    this.tag(field, WIRE_BYTES);
    this.uvarint(b.length);
    return this.raw(b);
  }

  stringField(field, text) {
    if (text === undefined || text === null || text === '') return this;
    return this.bytesField(field, enc.encode(String(text)));
  }

  messageField(field, writer) {
    if (!writer) return this;
    const payload = writer.finish();
    this.tag(field, WIRE_BYTES);
    this.uvarint(payload.length);
    return this.raw(payload);
  }

  packedFloats(field, values) {
    const b = new Uint8Array(values.length * 4);
    const dv = new DataView(b.buffer);
    for (let i = 0; i < values.length; i++) dv.setFloat32(i * 4, values[i], true);
    this.tag(field, WIRE_BYTES);
    this.uvarint(b.length);
    return this.raw(b);
  }

  repeatedVarint(field, values) {
    for (const v of values) this.varintField(field, v);
    return this;
  }

  packedInt64(field, values) {
    const b = new Uint8Array(values.length * 8);
    const dv = new DataView(b.buffer);
    for (let i = 0; i < values.length; i++) dv.setBigInt64(i * 8, BigInt(Math.round(Number(values[i]))), true);
    this.tag(field, WIRE_BYTES);
    this.uvarint(b.length);
    return this.raw(b);
  }

  finish() {
    const out = new Uint8Array(this.length);
    let p = 0;
    for (const c of this.chunks) {
      out.set(c, p);
      p += c.length;
    }
    this.chunks = [out];
    this.length = out.length;
    return out;
  }
}

export function parseMessage(bytes) {
  const data = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const fields = {};
  let i = 0;
  while (i < data.length) {
    const [key, next] = decodeVarint(data, i);
    i = next;
    const field = Math.floor(key / 8);
    const wire = key % 8;
    let value;
    if (wire === WIRE_VARINT) {
      const [v, n] = decodeVarint(data, i);
      i = n;
      value = v;
    } else if (wire === WIRE_FIXED64) {
      value = new DataView(data.buffer, data.byteOffset + i, 8).getFloat64(0, true);
      i += 8;
    } else if (wire === WIRE_BYTES) {
      const [len, n] = decodeVarint(data, i);
      i = n;
      value = data.slice(i, i + len);
      i += len;
    } else if (wire === WIRE_FIXED32) {
      value = new DataView(data.buffer, data.byteOffset + i, 4).getFloat32(0, true);
      i += 4;
    } else {
      throw new Error('unsupported wire type ' + wire + ' at field ' + field);
    }
    if (!fields[field]) fields[field] = [];
    fields[field].push({ wire, value });
  }
  return fields;
}

export function first(fields, n) {
  const list = fields[n];
  return list && list.length ? list[0].value : undefined;
}

export function asString(value) {
  if (value instanceof Uint8Array) return new TextDecoder().decode(value);
  return String(value);
}

export function asFloats(value) {
  if (!(value instanceof Uint8Array)) return [];
  const n = Math.floor(value.length / 4);
  const out = new Float32Array(n);
  const dv = new DataView(value.buffer, value.byteOffset, value.length);
  for (let i = 0; i < n; i++) out[i] = dv.getFloat32(i * 4, true);
  return out;
}

export function asInt64s(value) {
  if (!(value instanceof Uint8Array)) return [];
  const n = Math.floor(value.length / 8);
  const out = [];
  const dv = new DataView(value.buffer, value.byteOffset, value.length);
  for (let i = 0; i < n; i++) out.push(Number(dv.getBigInt64(i * 8, true)));
  return out;
}

export const ONNX_FIELDS = {
  model: { irVersion: 1, opsetImport: 8, producerName: 2, producerVersion: 3, domain: 4, modelVersion: 5, docString: 6, graph: 7, metadataProps: 14 },
  opset: { domain: 1, version: 2 },
  graph: { node: 1, name: 2, initializer: 5, docString: 10, input: 11, output: 12, valueInfo: 13 },
  node: { input: 1, output: 2, name: 3, opType: 4, attribute: 5, docString: 6, domain: 7 },
  attribute: { name: 1, float: 2, int: 3, string: 4, tensor: 5, graph: 6, floats: 7, ints: 8, strings: 9, tensors: 10, graphs: 11, type: 20 },
  tensor: { dims: 1, dataType: 2, floatData: 4, int32Data: 5, int64Data: 7, name: 8, rawData: 9, stringData: 11, doubleData: 10, docString: 13, externalData: 6, dataLocation: 14 },
  valueInfo: { name: 1, type: 2, docString: 3 },
  typeProto: { tensorType: 1, sequenceType: 2, mapType: 3 },
  tensorType: { elemType: 1, shape: 2 },
  shape: { dim: 1 },
  dim: { dimValue: 1, dimParam: 2 },
  stringEntry: { key: 1, value: 2 }
};

export const ONNX_DTYPE = {
  undefined: 0,
  float32: 1,
  uint8: 2,
  int8: 3,
  uint16: 4,
  int16: 5,
  int32: 6,
  int64: 7,
  string: 8,
  bool: 9,
  float16: 10,
  double: 11,
  uint32: 12,
  uint64: 13,
  complex64: 14,
  complex128: 15,
  bfloat16: 16
};

export const ONNX_ATTR = {
  undefined: 0,
  float: 1,
  int: 2,
  string: 3,
  tensor: 4,
  graph: 5,
  floats: 6,
  ints: 7,
  strings: 8,
  tensors: 9,
  graphs: 10,
  sparseTensor: 11,
  typeProto: 12,
  sparseTensors: 13,
  typeProtos: 14
};
