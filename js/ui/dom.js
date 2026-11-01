export function el(tag, attrs, children) {
  const node = document.createElement(tag);
  if (attrs) setAttrs(node, attrs);
  if (children !== undefined && children !== null) mount(node, children);
  return node;
}

export function setAttrs(node, attrs) {
  for (const key of Object.keys(attrs)) {
    const value = attrs[key];
    if (value === null || value === undefined || value === false) continue;
    if (key === 'class') node.className = value;
    else if (key === 'style' && typeof value === 'object') Object.assign(node.style, value);
    else if (key === 'text') node.textContent = String(value);
    else if (key === 'html') node.innerHTML = String(value);
    else if (key === 'value') node.value = value;
    else if (key === 'checked') node.checked = !!value;
    else if (key === 'disabled') node.disabled = !!value;
    else if (key === 'dataset') Object.assign(node.dataset, value);
    else if (key.slice(0, 2) === 'on' && typeof value === 'function') node.addEventListener(key.slice(2).toLowerCase(), value);
    else if (value === true) node.setAttribute(key, '');
    else node.setAttribute(key, String(value));
  }
  return node;
}

export function mount(node, children) {
  const list = Array.isArray(children) ? children : [children];
  for (const child of list) {
    if (child === null || child === undefined || child === false) continue;
    node.appendChild(typeof child === 'object' ? child : document.createTextNode(String(child)));
  }
  return node;
}

export function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
  return node;
}

export function replace(node, children) {
  return mount(clear(node), children);
}

export function frag(children) {
  return mount(document.createDocumentFragment(), children);
}

export function on(node, type, handler, options) {
  node.addEventListener(type, handler, options);
  return () => node.removeEventListener(type, handler, options);
}

export function svgEl(tag, attrs, children) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
  if (attrs) {
    for (const key of Object.keys(attrs)) {
      const value = attrs[key];
      if (value === null || value === undefined || value === false) continue;
      node.setAttribute(key, String(value));
    }
  }
  if (children) mount(node, children);
  return node;
}

export function debounce(fn, ms) {
  let timer = null;
  return function debounced() {
    const args = arguments;
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => { timer = null; fn.apply(null, args); }, ms || 120);
  };
}

export function throttle(fn, ms) {
  let last = 0;
  let timer = null;
  return function throttled() {
    const args = arguments;
    const now = Date.now();
    const wait = (ms || 100) - (now - last);
    if (wait <= 0) {
      last = now;
      fn.apply(null, args);
    } else if (!timer) {
      timer = setTimeout(() => { timer = null; last = Date.now(); fn.apply(null, args); }, wait);
    }
  };
}

export function readFileBytes(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(new Uint8Array(reader.result));
    reader.onerror = () => reject(reader.error || new Error('could not read ' + file.name));
    reader.readAsArrayBuffer(file);
  });
}

export function readFileText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error || new Error('could not read ' + file.name));
    reader.readAsText(file);
  });
}

export function download(filename, bytes, mime) {
  const blob = bytes instanceof Blob ? bytes : new Blob([bytes], { type: mime || 'application/octet-stream' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}

export function downloadJson(filename, value) {
  download(filename, JSON.stringify(value, null, 2), 'application/json');
}

export function pickFiles(accept, multiple) {
  return new Promise((resolve) => {
    const input = el('input', { type: 'file', accept: accept || '', multiple: multiple !== false, class: 'sr' });
    input.addEventListener('change', () => {
      resolve(Array.from(input.files || []));
      input.remove();
    });
    document.body.appendChild(input);
    input.click();
  });
}

export function copyText(value) {
  if (navigator.clipboard && navigator.clipboard.writeText) return navigator.clipboard.writeText(value);
  const area = el('textarea', { value, class: 'sr' });
  document.body.appendChild(area);
  area.select();
  let ok = false;
  try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
  area.remove();
  return Promise.resolve(ok);
}

let tooltipNode = null;

export function showTooltip(html, x, y) {
  hideTooltip();
  tooltipNode = el('div', { class: 'tooltip', html });
  document.body.appendChild(tooltipNode);
  const rect = tooltipNode.getBoundingClientRect();
  const left = Math.max(8, Math.min(x + 12, window.innerWidth - rect.width - 10));
  const top = Math.max(8, Math.min(y + 14, window.innerHeight - rect.height - 10));
  tooltipNode.style.left = left + 'px';
  tooltipNode.style.top = top + 'px';
}

export function hideTooltip() {
  if (tooltipNode) { tooltipNode.remove(); tooltipNode = null; }
}

export function attachTooltip(node, makeHtml) {
  node.addEventListener('mouseenter', (event) => {
    const html = typeof makeHtml === 'function' ? makeHtml() : makeHtml;
    if (html) showTooltip(html, event.clientX, event.clientY);
  });
  node.addEventListener('mousemove', (event) => {
    if (tooltipNode) {
      const rect = tooltipNode.getBoundingClientRect();
      tooltipNode.style.left = Math.max(8, Math.min(event.clientX + 12, window.innerWidth - rect.width - 10)) + 'px';
      tooltipNode.style.top = Math.max(8, Math.min(event.clientY + 14, window.innerHeight - rect.height - 10)) + 'px';
    }
  });
  node.addEventListener('mouseleave', hideTooltip);
  return node;
}

export function modal(title, bodyNodes, footerNodes, options) {
  const back = el('div', { class: 'modal-back' });
  const close = () => back.remove();
  const box = el('div', { class: 'modal', style: options && options.width ? { maxWidth: options.width } : null }, [
    el('header', {}, [
      el('h2', { text: title }),
      el('div', { class: 'spacer' }),
      el('button', { class: 'ghost sm', text: 'Close', onclick: close })
    ]),
    el('div', { class: 'body' }, bodyNodes),
    footerNodes ? el('footer', {}, footerNodes) : null
  ]);
  back.appendChild(box);
  back.addEventListener('mousedown', (event) => { if (event.target === back) close(); });
  document.body.appendChild(back);
  return { close, root: back, box };
}

export function confirmDialog(message, okLabel, onOk) {
  const handle = modal('Confirm', [el('p', { text: message })], [
    el('button', { class: 'ghost', text: 'Cancel', onclick: () => handle.close() }),
    el('button', { class: 'danger', text: okLabel || 'Delete', onclick: () => { handle.close(); onOk(); } })
  ], { width: '440px' });
  return handle;
}

export function field(labelText, control, hint) {
  return el('label', { class: 'field' }, [
    el('span', { class: 'lbl', text: labelText }),
    control,
    hint ? el('span', { class: 'tiny faint', text: hint }) : null
  ]);
}

export function numberField(labelText, value, options, onChange) {
  const opts = options || {};
  const input = el('input', {
    type: 'number',
    value: value,
    min: opts.min,
    max: opts.max,
    step: opts.step || 'any',
    disabled: opts.disabled,
    oninput: (event) => {
      const parsed = parseFloat(event.target.value);
      onChange(Number.isFinite(parsed) ? parsed : null, event);
    }
  });
  if (opts.width) input.style.width = opts.width;
  return field(labelText, input, opts.hint);
}

export function selectField(labelText, value, choices, onChange, opts) {
  const options = opts || {};
  const select = el('select', {
    disabled: options.disabled,
    onchange: (event) => onChange(event.target.value, event)
  }, (Array.isArray(choices) ? choices : Object.keys(choices).map((k) => ({ value: k, label: choices[k] }))).map((choice) => {
    const item = typeof choice === 'string' ? { value: choice, label: choice } : choice;
    return el('option', { value: item.value, text: item.label, selected: String(item.value) === String(value) });
  }));
  if (options.width) select.style.width = options.width;
  return field(labelText, select, options.hint);
}

export function sliderField(labelText, value, options, onChange) {
  const opts = options || {};
  const readout = el('span', { class: 'num small amber', text: opts.format ? opts.format(value) : String(value) });
  const input = el('input', {
    type: 'range',
    min: opts.min,
    max: opts.max,
    step: opts.step || 1,
    value: value,
    disabled: opts.disabled,
    style: { width: '100%' },
    oninput: (event) => {
      const next = parseFloat(event.target.value);
      readout.textContent = opts.format ? opts.format(next) : String(next);
      onChange(next, event);
    }
  });
  return el('label', { class: 'field' }, [
    el('div', { class: 'row between' }, [el('span', { class: 'lbl', text: labelText }), readout]),
    input,
    opts.hint ? el('span', { class: 'tiny faint', text: opts.hint }) : null
  ]);
}

export function checkboxRow(labelText, checked, onChange, hint) {
  const input = el('input', { type: 'checkbox', checked: !!checked, onchange: (event) => onChange(event.target.checked, event) });
  const row = el('label', { class: 'inline', style: { width: '100%' } }, [input, el('span', { text: labelText })]);
  return hint ? el('div', {}, [row, el('div', { class: 'tiny faint', text: hint, style: { marginLeft: '19px' } })]) : row;
}

export function statCard(key, value, sub, tone) {
  return el('div', { class: 'stat' }, [
    el('div', { class: 'k', text: key }),
    el('div', { class: 'v' + (tone ? ' ' + tone : ''), text: value }),
    sub ? el('div', { class: 's', text: sub }) : null
  ]);
}

export function kvList(pairs) {
  const list = el('dl', { class: 'kv' });
  for (const pair of pairs) {
    if (!pair) continue;
    list.appendChild(el('dt', { text: pair[0] }));
    const dd = el('dd');
    if (pair[1] && typeof pair[1] === 'object' && pair[1].nodeType) dd.appendChild(pair[1]);
    else dd.textContent = pair[1] === null || pair[1] === undefined ? '—' : String(pair[1]);
    list.appendChild(dd);
  }
  return list;
}

export function sectionTitle(label, count) {
  return el('div', { class: 'section-title' }, [
    el('span', { text: label }),
    count === undefined || count === null ? null : el('span', { class: 'count', text: count })
  ]);
}

export function busy(node, isBusy, label) {
  if (!node) return node;
  node.style.opacity = isBusy ? '0.55' : '';
  node.style.pointerEvents = isBusy ? 'none' : '';
  if (isBusy && label) node.setAttribute('aria-busy', 'true');
  return node;
}

export function fitCanvas(canvas, width, height) {
  const ratio = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(width));
  const h = Math.max(1, Math.round(height));
  if (canvas.width !== Math.round(w * ratio) || canvas.height !== Math.round(h * ratio)) {
    canvas.width = Math.round(w * ratio);
    canvas.height = Math.round(h * ratio);
  }
  canvas.style.width = w + 'px';
  canvas.style.height = h + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return ctx;
}

export function measureCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  return { width: rect.width || canvas.clientWidth || 320, height: rect.height || canvas.clientHeight || 160 };
}

export function unique(values) {
  const seen = new Set();
  const out = [];
  for (const value of values) if (!seen.has(value)) { seen.add(value); out.push(value); }
  return out;
}

export function clamp(value, lo, hi) {
  return Math.min(hi, Math.max(lo, value));
}

export function nextFrame() {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

export function yieldToBrowser() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

export function setFavicon() {
  const size = 64;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.fillStyle = '#16181a';
  ctx.fillRect(0, 0, size, size);
  ctx.strokeStyle = '#e0a33e';
  ctx.lineWidth = 4;
  ctx.strokeRect(4, 4, size - 8, size - 8);
  ctx.strokeStyle = '#4fa3a5';
  ctx.lineWidth = 5;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(15, 47);
  ctx.lineTo(26, 22);
  ctx.lineTo(37, 37);
  ctx.lineTo(49, 16);
  ctx.stroke();
  ctx.fillStyle = '#e0a33e';
  ctx.beginPath();
  ctx.arc(26, 22, 4.5, 0, Math.PI * 2);
  ctx.fill();
  const link = document.createElement('link');
  link.rel = 'icon';
  link.href = canvas.toDataURL('image/png');
  document.head.appendChild(link);
}

export function markSvg(scale) {
  const s = scale || 18;
  return svgEl('svg', { width: s, height: s, viewBox: '0 0 18 18' }, [
    svgEl('rect', { x: 1, y: 1, width: 16, height: 16, rx: 2, fill: 'none', stroke: '#e0a33e', 'stroke-width': 1.2 }),
    svgEl('path', { d: 'M4.5 13.2 L7.4 6.4 L10.2 10.4 L13.5 4.8', fill: 'none', stroke: '#4fa3a5', 'stroke-width': 1.4, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }),
    svgEl('circle', { cx: 7.4, cy: 6.4, r: 1.15, fill: '#e0a33e' })
  ]);
}
