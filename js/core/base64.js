const ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
const LOOKUP = (() => {
  const map = new Int16Array(128).fill(-1);
  for (let i = 0; i < ALPHABET.length; i++) map[ALPHABET.charCodeAt(i)] = i;
  map['-'.charCodeAt(0)] = 62;
  map['_'.charCodeAt(0)] = 63;
  return map;
})();

export function base64Encode(bytes) {
  const data = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  let out = '';
  let i = 0;
  for (; i + 2 < data.length; i += 3) {
    const v = (data[i] << 16) | (data[i + 1] << 8) | data[i + 2];
    out += ALPHABET[(v >> 18) & 63] + ALPHABET[(v >> 12) & 63] + ALPHABET[(v >> 6) & 63] + ALPHABET[v & 63];
  }
  const rest = data.length - i;
  if (rest === 1) {
    const v = data[i] << 16;
    out += ALPHABET[(v >> 18) & 63] + ALPHABET[(v >> 12) & 63] + '==';
  } else if (rest === 2) {
    const v = (data[i] << 16) | (data[i + 1] << 8);
    out += ALPHABET[(v >> 18) & 63] + ALPHABET[(v >> 12) & 63] + ALPHABET[(v >> 6) & 63] + '=';
  }
  return out;
}

export function base64Decode(text) {
  const clean = String(text).replace(/[\r\n\t ]/g, '');
  let pad = 0;
  if (clean.endsWith('==')) pad = 2;
  else if (clean.endsWith('=')) pad = 1;
  const usable = clean.length - (clean.endsWith('=') ? (pad === 2 ? 2 : 1) : 0);
  const out = new Uint8Array(Math.floor((usable * 6) / 8));
  let buffer = 0;
  let bits = 0;
  let p = 0;
  for (let i = 0; i < usable; i++) {
    const code = clean.charCodeAt(i);
    const value = code < 128 ? LOOKUP[code] : -1;
    if (value < 0) throw new Error('base64: illegal character at ' + i);
    buffer = (buffer << 6) | value;
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out[p++] = (buffer >> bits) & 0xff;
    }
  }
  void pad;
  return out.length === p ? out : out.slice(0, p);
}

export function base64ToFloat32(text) {
  const bytes = base64Decode(text);
  const aligned = bytes.byteOffset % 4 === 0 ? bytes : bytes.slice();
  return new Float32Array(aligned.buffer, aligned.byteOffset, aligned.byteLength / 4);
}

export function float32ToBase64(values) {
  const src = values instanceof Float32Array ? values : Float32Array.from(values);
  return base64Encode(new Uint8Array(src.buffer, src.byteOffset, src.byteLength));
}

export function utf8Encode(text) {
  if (typeof TextEncoder !== 'undefined') return new TextEncoder().encode(text);
  const out = [];
  for (const ch of String(text)) {
    const c = ch.codePointAt(0);
    if (c < 0x80) out.push(c);
    else if (c < 0x800) out.push(0xc0 | (c >> 6), 0x80 | (c & 63));
    else if (c < 0x10000) out.push(0xe0 | (c >> 12), 0x80 | ((c >> 6) & 63), 0x80 | (c & 63));
    else out.push(0xf0 | (c >> 18), 0x80 | ((c >> 12) & 63), 0x80 | ((c >> 6) & 63), 0x80 | (c & 63));
  }
  return Uint8Array.from(out);
}

export function utf8Decode(bytes) {
  if (typeof TextDecoder !== 'undefined') return new TextDecoder().decode(bytes);
  let out = '';
  let i = 0;
  while (i < bytes.length) {
    const b = bytes[i++];
    if (b < 0x80) out += String.fromCharCode(b);
    else if (b < 0xe0) out += String.fromCharCode(((b & 31) << 6) | (bytes[i++] & 63));
    else if (b < 0xf0) out += String.fromCharCode(((b & 15) << 12) | ((bytes[i++] & 63) << 6) | (bytes[i++] & 63));
    else {
      const c = ((b & 7) << 18) | ((bytes[i++] & 63) << 12) | ((bytes[i++] & 63) << 6) | (bytes[i++] & 63);
      out += String.fromCodePoint(c);
    }
  }
  return out;
}
