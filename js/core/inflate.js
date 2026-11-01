const LENGTH_BASE = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const LENGTH_EXTRA = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const DIST_BASE = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
const DIST_EXTRA = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
const CL_ORDER = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

const FIXED_LIT_LENGTHS = (() => {
  const out = new Uint8Array(288);
  for (let i = 0; i < 144; i++) out[i] = 8;
  for (let i = 144; i < 256; i++) out[i] = 9;
  for (let i = 256; i < 280; i++) out[i] = 7;
  for (let i = 280; i < 288; i++) out[i] = 8;
  return out;
})();

const FIXED_DIST_LENGTHS = new Uint8Array(30).fill(5);

class BitReader {
  constructor(bytes, offset) {
    this.bytes = bytes;
    this.pos = offset || 0;
    this.bit = 0;
  }

  bits(n) {
    let value = 0;
    for (let i = 0; i < n; i++) {
      if (this.pos >= this.bytes.length) throw new Error('inflate: unexpected end of input');
      const byte = this.bytes[this.pos];
      value |= ((byte >> this.bit) & 1) << i;
      this.bit++;
      if (this.bit === 8) {
        this.bit = 0;
        this.pos++;
      }
    }
    return value >>> 0;
  }

  align() {
    if (this.bit) {
      this.bit = 0;
      this.pos++;
    }
  }
}

function buildHuffman(lengths) {
  const maxBits = 15;
  const counts = new Int32Array(maxBits + 1);
  for (let i = 0; i < lengths.length; i++) counts[lengths[i]]++;
  counts[0] = 0;
  const offsets = new Int32Array(maxBits + 2);
  for (let b = 1; b <= maxBits; b++) offsets[b + 1] = offsets[b] + counts[b];
  const symbols = new Int32Array(lengths.length);
  for (let s = 0; s < lengths.length; s++) {
    if (lengths[s]) symbols[offsets[lengths[s]]++] = s;
  }
  const maxLen = (() => {
    let m = 1;
    for (let b = maxBits; b >= 1; b--) if (counts[b]) { m = b; break; }
    return m;
  })();
  const tableSize = 1 << maxLen;
  const tableCodes = new Int32Array(tableSize);
  const tableLengths = new Uint8Array(tableSize);
  let code = 0;
  let symbolIndex = 0;
  for (let len = 1; len <= maxLen; len++) {
    for (let i = 0; i < counts[len]; i++) {
      const symbol = symbols[symbolIndex++];
      const reversed = reverseBits(code, len);
      const fill = 1 << len;
      for (let slot = reversed; slot < tableSize; slot += fill) {
        tableCodes[slot] = symbol;
        tableLengths[slot] = len;
      }
      code++;
    }
    code <<= 1;
  }
  return { tableCodes, tableLengths, maxLen };
}

function reverseBits(value, len) {
  let out = 0;
  for (let i = 0; i < len; i++) {
    out = (out << 1) | ((value >> i) & 1);
  }
  return out;
}

function decodeSymbol(reader, tree) {
  let index = 0;
  for (let len = 1; len <= tree.maxLen; len++) {
    index |= reader.bits(1) << (len - 1);
    if (tree.tableLengths[index] === len) return tree.tableCodes[index];
  }
  throw new Error('inflate: bad huffman code');
}

function readDynamicTrees(reader) {
  const hlit = reader.bits(5) + 257;
  const hdist = reader.bits(5) + 1;
  const hclen = reader.bits(4) + 4;
  const clLengths = new Uint8Array(19);
  for (let i = 0; i < hclen; i++) clLengths[CL_ORDER[i]] = reader.bits(3);
  const clTree = buildHuffman(clLengths);
  const lengths = new Uint8Array(hlit + hdist);
  let i = 0;
  while (i < lengths.length) {
    const symbol = decodeSymbol(reader, clTree);
    if (symbol < 16) {
      lengths[i++] = symbol;
    } else if (symbol === 16) {
      if (!i) throw new Error('inflate: repeat with no previous length');
      const prev = lengths[i - 1];
      const repeat = 3 + reader.bits(2);
      for (let r = 0; r < repeat; r++) lengths[i++] = prev;
    } else if (symbol === 17) {
      const repeat = 3 + reader.bits(3);
      for (let r = 0; r < repeat; r++) lengths[i++] = 0;
    } else {
      const repeat = 11 + reader.bits(7);
      for (let r = 0; r < repeat; r++) lengths[i++] = 0;
    }
  }
  return {
    literal: buildHuffman(lengths.subarray(0, hlit)),
    distance: buildHuffman(lengths.subarray(hlit, hlit + hdist))
  };
}

class ByteWriter {
  constructor(initial, limit) {
    this.data = new Uint8Array(Math.max(1024, initial || 0));
    this.length = 0;
    this.limit = limit || 0;
  }

  ensure(extra) {
    if (this.limit && this.length + extra > this.limit) throw new Error('inflate: output exceeds ' + this.limit + ' bytes');
    if (this.length + extra <= this.data.length) return;
    let size = this.data.length * 2;
    while (size < this.length + extra) size *= 2;
    const next = new Uint8Array(size);
    next.set(this.data.subarray(0, this.length));
    this.data = next;
  }

  push(value) {
    this.ensure(1);
    this.data[this.length++] = value;
  }

  copyFrom(distance, length) {
    if (distance > this.length) throw new Error('inflate: distance ' + distance + ' beyond start of output');
    this.ensure(length);
    let src = this.length - distance;
    for (let i = 0; i < length; i++) this.data[this.length++] = this.data[src++];
  }

  finish() {
    return this.data.slice(0, this.length);
  }
}

function inflateBlock(reader, out, literalTree, distanceTree) {
  for (;;) {
    const symbol = decodeSymbol(reader, literalTree);
    if (symbol < 256) {
      out.push(symbol);
    } else if (symbol === 256) {
      return;
    } else {
      const lengthIndex = symbol - 257;
      if (lengthIndex >= LENGTH_BASE.length) throw new Error('inflate: invalid length symbol ' + symbol);
      const length = LENGTH_BASE[lengthIndex] + (LENGTH_EXTRA[lengthIndex] ? reader.bits(LENGTH_EXTRA[lengthIndex]) : 0);
      const distSymbol = decodeSymbol(reader, distanceTree);
      if (distSymbol >= DIST_BASE.length) throw new Error('inflate: invalid distance symbol ' + distSymbol);
      const distance = DIST_BASE[distSymbol] + (DIST_EXTRA[distSymbol] ? reader.bits(DIST_EXTRA[distSymbol]) : 0);
      out.copyFrom(distance, length);
    }
  }
}

export function inflateRaw(input, options) {
  const opts = options || {};
  const bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
  const start = opts.offset || 0;
  const reader = new BitReader(bytes, start);
  const out = new ByteWriter(opts.expectedSize, opts.maxBytes || 0);
  let final = false;
  let blocks = 0;
  while (!final) {
    final = reader.bits(1) === 1;
    const type = reader.bits(2);
    blocks++;
    if (type === 0) {
      reader.align();
      if (reader.pos + 4 > bytes.length) throw new Error('inflate: truncated stored block header');
      const len = bytes[reader.pos] | (bytes[reader.pos + 1] << 8);
      const nlen = bytes[reader.pos + 2] | (bytes[reader.pos + 3] << 8);
      if ((len ^ 0xffff) !== nlen) throw new Error('inflate: bad stored block length');
      reader.pos += 4;
      if (reader.pos + len > bytes.length) throw new Error('inflate: truncated stored block body');
      out.ensure(len);
      for (let i = 0; i < len; i++) out.data[out.length++] = bytes[reader.pos + i];
      reader.pos += len;
    } else if (type === 1) {
      inflateBlock(reader, out, buildHuffman(FIXED_LIT_LENGTHS), buildHuffman(FIXED_DIST_LENGTHS));
    } else if (type === 2) {
      const trees = readDynamicTrees(reader);
      inflateBlock(reader, out, trees.literal, trees.distance);
    } else {
      throw new Error('inflate: reserved block type 3');
    }
  }
  void blocks;
  return out.finish();
}

export function inflateZlib(input, options) {
  const bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
  if (bytes.length < 2) throw new Error('inflate: zlib stream too short');
  const cmf = bytes[0];
  const flg = bytes[1];
  if ((cmf & 0x0f) !== 8) throw new Error('inflate: not a deflate stream');
  if (((cmf << 8) | flg) % 31 !== 0) throw new Error('inflate: bad zlib header checksum');
  if (flg & 0x20) throw new Error('inflate: preset dictionary is not supported');
  return inflateRaw(bytes, Object.assign({}, options || {}, { offset: (options && options.offset ? options.offset : 0) + 2 }));
}

export function crc32Of(bytes) {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) {
    c ^= bytes[i];
    for (let k = 0; k < 8; k++) c = (c >>> 1) ^ (0xedb88320 & -(c & 1));
  }
  return (c ^ 0xffffffff) >>> 0;
}

export function deflateStore(input, options) {
  const opts = options || {};
  const bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
  const maxBlock = 65535;
  const chunks = [];
  let header = null;
  if (opts.zlib) {
    const cmf = 0x78;
    const level = opts.level === 0 ? 0 : opts.level === 1 ? 1 : opts.level >= 9 ? 3 : 2;
    const flg = (level << 6);
    const check = ((cmf << 8) | flg) % 31;
    header = new Uint8Array([cmf, flg + (31 - check)]);
    chunks.push(header);
  }
  let offset = 0;
  let first = true;
  while (first || offset < bytes.length) {
    first = false;
    const len = Math.min(maxBlock, bytes.length - offset);
    const isFinal = offset + len >= bytes.length ? 1 : 0;
    const head = new Uint8Array(5);
    head[0] = isFinal;
    head[1] = len & 0xff;
    head[2] = (len >> 8) & 0xff;
    head[3] = ~len & 0xff;
    head[4] = (~len >> 8) & 0xff;
    chunks.push(head);
    chunks.push(bytes.subarray(offset, offset + len));
    offset += len;
  }
  if (opts.zlib) {
    const adler = adler32(bytes);
    chunks.push(new Uint8Array([(adler >>> 24) & 0xff, (adler >>> 16) & 0xff, (adler >>> 8) & 0xff, adler & 0xff]));
  }
  let total = 0;
  for (const c of chunks) total += c.length;
  const out = new Uint8Array(total);
  let p = 0;
  for (const c of chunks) {
    out.set(c, p);
    p += c.length;
  }
  return out;
}

export function adler32(bytes) {
  let a = 1;
  let b = 0;
  for (let i = 0; i < bytes.length; i++) {
    a = (a + bytes[i]) % 65521;
    b = (b + a) % 65521;
  }
  return ((b << 16) | a) >>> 0;
}
