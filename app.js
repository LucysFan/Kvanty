/* =======================================================================
   Интерактивный дашборд комплекса синтеза фазового ДОЭ.
   Тонкий клиент: весь тяжёлый счёт выполнен build-скриптом
   (scripts/build_dashboard.py). Здесь — только рисование готовых массивов,
   офскрин-канвасы кадров (создаются один раз) и зонды без пересчётов
   (ТЗ, п. 12.1.2, 12.6.1).
   ======================================================================= */
'use strict';

const N = 128, NP = 4;
const S = {                      // состояние приложения
  variant: 'nn',                 // 'nn' | 'gs'
  plane: 0,                      // активная плоскость 0..3
  quant: true,                   // показывать 8-битную (квантованную) фазу
  flash: 0,                      // A6: вспышка разницы float↔8 бит
  z: 0,                          // текущий индекс z-стопки
  playing: true,
  speed: 1,
  loop: true,
  autopause: true,
  planeMode: 'swipe',
  swipe: 0.5,
  blend: 0.5,
  protocol: 'primary',
  measMode: 'probe',
  roi: null,
  probePX: 64, probePY: 64,
  trainIdx: 0,
  physStep: 0,
  trainPlaying: false,
  fps: 0,
};

const D = {};                    // данные
const CACHE = { frames: {}, small: {} };

/* ------------------------------------------------------------------ утилиты */
const $ = s => document.querySelector(s);
const $$ = s => Array.from(document.querySelectorAll(s));
const clamp = (v, a, b) => v < a ? a : v > b ? b : v;
const MA = m => m.annotations || {};   // (совместимость)

function fmt(v, d = 3) { return (v === undefined || v === null || Number.isNaN(v)) ? '—' : (+v).toFixed(d); }

/** Uint8/float массив → офскрин-канвас 128×128 (создаётся один раз). */
function toCanvas(data, offset, gray = true, scale = 1) {
  const c = document.createElement('canvas');
  c.width = c.height = N;
  const ctx = c.getContext('2d');
  const img = ctx.createImageData(N, N);
  const px = img.data;
  for (let i = 0; i < N * N; i++) {
    const o = i * 4;
    let v = data[offset + i];
    if (scale !== 1) v = v * scale;
    px[o] = px[o + 1] = px[o + 2] = v;
    px[o + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

/** Циклическая colormap для фазы: код 0..255 → RGB. */
function cyclicLUT() {
  const lut = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;                       // 0..1 соответствует −π..π
    // «twilight»-подобная замкнутая палитра
    const r = 0.5 + 0.5 * Math.cos(2 * Math.PI * (t + 0.10));
    const g = 0.5 + 0.5 * Math.cos(2 * Math.PI * (t + 0.45));
    const b = 0.5 + 0.5 * Math.cos(2 * Math.PI * (t + 0.78));
    lut[i * 3] = clamp(r * 255, 0, 255) | 0;
    lut[i * 3 + 1] = clamp(g * 255, 0, 255) | 0;
    lut[i * 3 + 2] = clamp(b * 255, 0, 255) | 0;
  }
  return lut;
}
const LUT = cyclicLUT();

function phaseCanvas(codes, offset = 0) {
  const c = document.createElement('canvas');
  c.width = c.height = N;
  const ctx = c.getContext('2d');
  const img = ctx.createImageData(N, N);
  for (let i = 0; i < N * N; i++) {
    const v = codes[offset + i];
    img.data[i * 4] = LUT[v * 3];
    img.data[i * 4 + 1] = LUT[v * 3 + 1];
    img.data[i * 4 + 2] = LUT[v * 3 + 2];
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

function drawScaled(ctx, src, w, h) {
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, w, h);
  ctx.drawImage(src, 0, 0, w, h);
}

/* ------------------------------------------------------------- загрузка данных */
async function loadBundle() {
  // 1) window.HDOE_BUNDLE (data_bundle.js) — офлайн, из файла
  if (window.HDOE_BUNDLE) {
    const b = window.HDOE_BUNDLE;
    const toU8 = s => new Uint8Array(b.b64[s] ? b64ToBuf(b.b64[s]) : []);
    const toF32 = s => new Float32Array(b.b64[s] ? b64ToBuf(b.b64[s]) : []);
    // массивы лежат как base64(gzip(raw)) — распаковка асинхронная
    const bufs = {};
    for (const key of Object.keys(b.b64 || {})) bufs[key] = await b64ToBufGz(b.b64[key]);
    const u8 = key => new Uint8Array(bufs[key] || new ArrayBuffer(0));
    const f32 = key => new Float32Array(bufs[key] || new ArrayBuffer(0));
    const keys = Object.keys(bufs);
    D.manifest = b.manifest;
    D.targets = u8('targets');
    D.phase = { nn: f32('phase_nn'), gs: f32('phase_gs') };
    D.doe = { nn: u8('doe_nn'), gs: u8('doe_gs') };
    // z-стопки упакованы дельта-кодированием по времени (см. build_dashboard.py)
    const unDelta = (arr, npx) => {
      const n = Math.floor(arr.length / npx);
      for (let k = 1; k < n; k++) {
        const o = k * npx, p = o - npx;
        for (let i = 0; i < npx; i++) arr[o + i] = (arr[o + i] + arr[p + i]) & 255;
      }
      return arr;
    };
    D.zstack = { nn: unDelta(u8('zstack_nn'), N * N), gs: unDelta(u8('zstack_gs'), N * N) };
    D.recon = { nn: f32('recon_nn'), gs: f32('recon_gs') };
    D.ssimMap = { nn: f32('ssim_map_nn'), gs: f32('ssim_map_gs') };
    D.zgrid = f32('zgrid');
    D.trainMasks = u8('train_masks');
    D.trainRecon = u8('train_recon');
    D.bitdepth = { nn: u8('bitdepth_nn'), gs: u8('bitdepth_gs') };
    D.zprobe = { nn: f32('zprobe_nn'), gs: f32('zprobe_gs') };
    D.physSteps = u8('phys_steps');
    D.manifest.loaded_keys = keys;
    D.filesUsed = keys.length;
    D.trainSteps = (b.json.train_steps || {}).steps || [];
    D.trainLog = b.json.train_log || [];
    D.tests = b.json.tests || {};
    return;
  }
  // 2) fetch — при запуске через любой статический сервер
  const load = p => fetch(p).then(r => { if (!r.ok) throw new Error(p); return r.arrayBuffer(); });
  const j = p => fetch(p).then(r => r.json());
  D.manifest = await j('data/manifest.json');
  const sh = D.manifest.layout.shapes;
  const u8 = async p => new Uint8Array(await load('data/' + p));
  const f32 = async p => new Float32Array(await load('data/' + p));
  D.targets = await u8('targets.u8.bin');
  D.phase = { nn: await f32('phase_nn.f32.bin'), gs: await f32('phase_gs.f32.bin') };
  D.doe = { nn: await u8('doe_nn.u8.bin'), gs: await u8('doe_gs.u8.bin') };
  D.zstack = { nn: await u8('zstack_nn.u8.bin'), gs: await u8('zstack_gs.u8.bin') };
  D.recon = { nn: await f32('recon_nn.f32.bin'), gs: await f32('recon_gs.f32.bin') };
  D.ssimMap = { nn: await f32('ssim_map_nn.f32.bin'), gs: await f32('ssim_map_gs.f32.bin') };
  D.zgrid = await f32('zgrid.f32.bin');
  D.trainMasks = await u8('train_masks.u8.bin');
  D.trainRecon = await u8('train_recon.u8.bin');
  D.bitdepth = { nn: await u8('bitdepth_nn.u8.bin'), gs: await u8('bitdepth_gs.u8.bin') };
  D.zprobe = { nn: await f32('zprobe_nn.f32.bin'), gs: await f32('zprobe_gs.f32.bin') };
  D.physSteps = await u8('phys_steps.u8.bin');
  D.trainSteps = (await j('data/train_steps.json')).steps || [];
  D.trainLog = await j('data/train_log.json');
  D.tests = await j('data/tests.json');
  void sh;
}

function b64ToBuf(b64) {
  const bin = atob(b64);
  const buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  return buf.buffer;
}

/** base64(gzip(raw)) → ArrayBuffer. Распаковка через DecompressionStream
 *  (встроен в браузеры 2023+) с фолбэком на pako-less реализацию gunzip. */
async function b64ToBufGz(b64) {
  const raw = new Uint8Array(b64ToBuf(b64));
  if (typeof DecompressionStream === 'undefined') {
    return gunzipFallback(raw).buffer;
  }
  const ds = new DecompressionStream('gzip');
  const stream = new Blob([raw]).stream().pipeThrough(ds);
  return new Response(stream).arrayBuffer();
}

/** Минимальный inflate (RFC 1951) + gzip-обёртка (RFC 1952) — фолбэк для
 *  старых браузеров. Используется только если нет DecompressionStream. */
function gunzipFallback(src) {
  // проверка gzip-заголовка
  if (!(src[0] === 0x1f && src[1] === 0x8b)) return src;
  let p = 10;                                  // за пределами заголовка
  const flg = src[3];
  if (flg & 4) { const xlen = src[p] | (src[p + 1] << 8); p += 2 + xlen; }
  if (flg & 8) { while (src[p++]) {} }
  if (flg & 16) { while (src[p++]) {} }
  if (flg & 2) p += 2;
  const out = [];
  const LBASE = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59,
    67, 83, 99, 115, 131, 163, 195, 227, 258];
  const LEXT = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
    5, 5, 5, 5, 0];
  const DBASE = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385,
    513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
  const DEXT = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10,
    11, 11, 12, 12, 13, 13];
  let bitbuf = 0, bitcnt = 0;
  const bits = n => {
    while (bitcnt < n) { bitbuf |= src[p++] << bitcnt; bitcnt += 8; }
    const v = bitbuf & ((1 << n) - 1); bitbuf >>>= n; bitcnt -= n; return v;
  };
  function buildTree(lengths) {
    const maxLen = Math.max(...lengths);
    const blCount = new Array(maxLen + 1).fill(0);
    for (const l of lengths) if (l) blCount[l]++;
    const nextCode = new Array(maxLen + 1).fill(0);
    let code = 0;
    for (let b = 1; b <= maxLen; b++) { code = (code + blCount[b - 1]) << 1; nextCode[b] = code; }
    const map = new Map();
    for (let i = 0; i < lengths.length; i++) {
      const l = lengths[i];
      if (!l) continue;
      map.set(l + ':' + nextCode[l], i);
      nextCode[l]++;
    }
    return { map, maxLen };
  }
  function decodeSym(tree) {
    let code = 0;
    for (let l = 1; l <= tree.maxLen; l++) {
      code = (code << 1) | bits(1);
      const s = tree.map.get(l + ':' + code);
      if (s !== undefined) return s;
    }
    throw new Error('inflate: битый код Хаффмана');
  }
  const LENLENS = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];
  let fixedLit = null, fixedDist = null;
  let last = 0;
  while (!last) {
    last = bits(1);
    const type = bits(2);
    if (type === 0) {                          // stored
      bitbuf = 0; bitcnt = 0;
      const len = src[p] | (src[p + 1] << 8); p += 4;
      for (let i = 0; i < len; i++) out.push(src[p++]);
      continue;
    }
    let lit, dist;
    if (type === 1) {                          // fixed Huffman
      if (!fixedLit) {
        const l = [];
        for (let i = 0; i < 288; i++) l.push(i < 144 ? 8 : i < 256 ? 9 : i < 280 ? 7 : 8);
        fixedLit = buildTree(l);
        fixedDist = buildTree(new Array(30).fill(5));
      }
      lit = fixedLit; dist = fixedDist;
    } else if (type === 2) {                   // dynamic Huffman
      const hlit = bits(5) + 257, hdist = bits(5) + 1, hclen = bits(4) + 4;
      const cl = new Array(19).fill(0);
      for (let i = 0; i < hclen; i++) cl[LENLENS[i]] = bits(3);
      const clTree = buildTree(cl);
      const lens = [];
      while (lens.length < hlit + hdist) {
        const sym = decodeSym(clTree);
        if (sym < 16) lens.push(sym);
        else if (sym === 16) { const r = 3 + bits(2), v = lens[lens.length - 1]; for (let i = 0; i < r; i++) lens.push(v); }
        else if (sym === 17) { const r = 3 + bits(3); for (let i = 0; i < r; i++) lens.push(0); }
        else { const r = 11 + bits(7); for (let i = 0; i < r; i++) lens.push(0); }
      }
      lit = buildTree(lens.slice(0, hlit));
      dist = buildTree(lens.slice(hlit));
    } else throw new Error('inflate: недопустимый тип блока');
    for (;;) {
      const sym = decodeSym(lit);
      if (sym === 256) break;
      if (sym < 256) { out.push(sym); continue; }
      const li = sym - 257;
      const len = LBASE[li] + bits(LEXT[li]);
      const dsym = decodeSym(dist);
      const d = DBASE[dsym] + bits(DEXT[dsym]);
      const start = out.length - d;
      for (let i = 0; i < len; i++) out.push(out[start + i]);
    }
  }
  return new Uint8Array(out);
}

/* ------------------------------------------------------------------ кэши */
function frameCount() { return D.zgrid.length; }

function frame(variant, k) {
  const key = variant + ':' + k;
  if (!CACHE.frames[key]) {
    CACHE.frames[key] = toCanvas(D.zstack[variant], k * N * N);
  }
  return CACHE.frames[key];
}

function reconFrame(variant, plane) {
  const key = 'recon:' + variant + ':' + plane;
  if (!CACHE.small[key]) {
    const src = D.recon[variant];
    const c = document.createElement('canvas');
    c.width = c.height = N;
    const ctx = c.getContext('2d');
    const img = ctx.createImageData(N, N);
    const off = plane * N * N;
    for (let i = 0; i < N * N; i++) {
      const v = clamp(src[off + i], 0, 1) * 255;
      img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = v;
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    CACHE.small[key] = c;
  }
  return CACHE.small[key];
}

function targetFrame(i) {
  const key = 'target:' + i;
  if (!CACHE.small[key]) CACHE.small[key] = toCanvas(D.targets, i * N * N);
  return CACHE.small[key];
}

function ssimFrame(variant, plane) {
  const key = 'ssim:' + variant + ':' + plane;
  if (!CACHE.small[key]) {
    const src = D.ssimMap[variant];
    const c = document.createElement('canvas');
    c.width = c.height = N;
    const ctx = c.getContext('2d');
    const img = ctx.createImageData(N, N);
    const off = plane * N * N;
    for (let i = 0; i < N * N; i++) {
      const v = src[off + i];                  // −1..1
      const t = clamp((v + 0.2) / 1.2, 0, 1);
      // сине-жёлтая шкала: 0 → тёмно-синий, 1 → яркий жёлто-белый
      img.data[i * 4] = 255 * clamp(1.4 * t - 0.15, 0, 1);
      img.data[i * 4 + 1] = 255 * clamp(1.15 * t - 0.05, 0, 1);
      img.data[i * 4 + 2] = 255 * clamp(0.75 - 0.75 * t + 0.25, 0, 1);
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    CACHE.small[key] = c;
  }
  return CACHE.small[key];
}

/* ---------------------------------------------------------------- шапка */
function renderHeader() {
  const m = D.manifest, c = m.constants;
  const chips = [
    `λ = <b>${(c.lam * 1e9).toFixed(0)} нм</b>`,
    `Δx = <b>${(c.dx * 1e6).toFixed(0)} мкм</b>`,
    `N = <b>${c.N}×${c.N}</b>`,
    `z = <b>${c.zs_mm.map(v => v.toFixed(1)).join(' / ')} мм</b>`,
    `z<sub>crit</sub> = <b>${c.z_crit_mm.toFixed(4)} мм</b>`,
    `seed = <b>${c.seed}</b>`,
    `${Object.entries(m.versions).map(([k, v]) => `${k} ${v}`).join(' · ')}`,
  ];
  $('#params').innerHTML = chips.map(t => `<span class="chip">${t}</span>`).join('');

  const t = D.tests || {};
  const items = [
    ['8.1', 'Парсеваль', t.parseval],
    ['8.2', 'Линза', t.lens],
    ['8.3', 'Квантование', t.quantization],
    ['8.4', 'Градиент', t.gradient],
    ['8.5', 'Кросс-схема', t.cross_scheme],
    ['8.6', 'Граница', t.boundary],
    ['8.7', 'Форматы', t.formats],
    ['8.9', 'Физика в сети', t.netphys],
  ];
  const tested = items.filter(([, , r]) => r);
  const passed = tested.filter(([, , r]) => (r && (r.passed !== false))).length;
  $('#badges').innerHTML = items.map(([num, name, r]) => {
    const ok = r ? (r.passed !== false) : false;
    return `<span class="badge ${ok ? '' : 'fail'}" data-test="${num}">${ok ? '✓' : '✕'} ${num} ${name}</span>`;
  }).join('') + `<span class="badge info">тесты: ${passed}/${tested.length || items.length}</span>`;
  $$('.badge[data-test]').forEach(b => b.onclick = () => showTest(b.dataset.test));
}

function showTest(num) {
  const t = D.tests || {};
  const map = { '8.1': 'parseval', '8.2': 'lens', '8.3': 'quantization', '8.4': 'gradient',
                '8.5': 'cross_scheme', '8.6': 'boundary', '8.7': 'formats',
                '8.9': 'netphys' };
  const key = map[num];
  const r = t[key] || {};
  let body = '';
  const rows = (arr) => arr.map(o => `<tr>` + Object.entries(o).map(([k, v]) =>
        `<td>${typeof v === 'number' ? (+v).toPrecision(6) : v}</td>`).join('') + `</tr>`).join('');
  const head = (arr) => arr.length ? `<tr>` + Object.keys(arr[0]).map(k => `<th>${k}</th>`).join('') + `</tr>` : '';
  const titles = {
    parseval: '8.1 Парсеваль: Σ|U₀|²Δx² = Σ|U_z|²Δx′² (погрешность ≤ 1e−3)',
    lens: '8.2 Линза: φ = −πr²/(λf) при z = f → максимум в центральном пикселе',
    quantization: '8.3 Квантование: round-trip 8 бит, ошибка ≤ δ/2 = 0.01232 рад',
    gradient: '8.4 Градиент: gradcheck комплексного поля (complex128) + численная сверка',
    cross_scheme: '8.5 Кросс-схема: одно-БПФ vs угловой спектр (4× апсемплинг), SSIM ≥ 0.99',
    boundary: '8.6 Граница применимости одно-БПФ-схемы',
    formats: '8.7 Форматы выходных PNG (128×128, 8 бит, режим L)',
    netphys: '8.9 Физика внутри сети: один прямой проход, PCC совпадает с metrics.json, '
           + 'градиент доходит до U-Net через K шагов физического слоя',
  };
  body += `<div class="formula">${titles[key] || ''}</div>`;
  if (key === 'parseval' && t.parseval_rows) {
    body += `<table>${head(t.parseval_rows)}${rows(t.parseval_rows)}</table>`;
  } else if (key === 'lens' && t.lens_rows) {
    body += `<table>${head(t.lens_rows)}${rows(t.lens_rows)}</table>`;
  } else if (key === 'formats' && t.formats) {
    body += `<table>${head(t.formats)}${rows(t.formats)}</table>`;
  } else {
    body += `<table>${Object.entries(r).map(([k, v]) =>
      `<tr><td>${k}</td><td>${typeof v === 'number' ? (+v).toPrecision(8) : JSON.stringify(v)}</td></tr>`).join('')}</table>`;
  }
  if (key === 'boundary') {
    body += `\n<div class="formula">z₁ = ${(D.manifest.constants.zs_mm[0]).toFixed(1)} мм при z_crit = ` +
            `${D.manifest.constants.z_crit_mm.toFixed(4)} мм: запас ` +
            `${((D.manifest.constants.zs_mm[0] / D.manifest.constants.z_crit_mm - 1) * 100).toFixed(2)} % — ` +
            `штатная ситуация, требующая контроля (п. 4.5.2).</div>`;
  }
  openModal(`Тест ${num}`, body);
}

function openModal(title, html) {
  $('#modal-title').textContent = title;
  $('#modal-body').innerHTML = html;
  $('#modal').classList.remove('hidden');
}

/* ------------------------------------------------------------- 1. цели */
function renderTargets() {
  const grid = $('#targets-grid');
  grid.innerHTML = '';
  for (let i = 0; i < NP; i++) {
    const cell = document.createElement('div');
    cell.className = 'target-cell' + (i === S.plane ? ' active' : '');
    cell.dataset.plane = i;
    const c = document.createElement('canvas');
    c.width = N; c.height = N;
    const ctx = c.getContext('2d');
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(targetFrame(i), 0, 0);
    cell.appendChild(c);
    const tag = document.createElement('div');
    tag.className = 'tag';
    tag.textContent = `T${i + 1} · ${D.manifest.constants.zs_mm[i].toFixed(1)} мм`;
    cell.appendChild(tag);
    cell.onclick = () => setPlane(i);
    grid.appendChild(cell);
  }
  const row = D.manifest.file_metrics[S.variant][S.plane];
  $('#target-caption').innerHTML =
    `z${S.plane + 1} = <b>${row.z_mm.toFixed(1)} мм</b> · Δx′ ${(D.manifest.constants.out_dx[S.plane] * 1e6).toFixed(3)} мкм · ` +
    `PCC <b>${fmt(row.pcc)}</b> · SSIM <b>${fmt(row.ssim)}</b>`;
}

function setPlane(i) {
  S.plane = i;
  renderTargets();
  renderPlanes();
  renderMetrics();
  renderMeas();
}

/* -------------------------------------------------------------- 2. ДОЭ */
function phaseCodes() { return D.doe[S.variant]; }

function renderDoe() {
  const cv = $('#doe-phase');
  const ctx = cv.getContext('2d');
  const codes = phaseCodes();
  drawScaled(ctx, phaseCanvas(codes), cv.width, cv.height);
  if (S.flash > 0) {                 // A6: усиленная подсветка разницы float↔8 бит
    const ph = D.phase[S.variant];
    const img = ctx.getImageData(0, 0, cv.width, cv.height);
    const sX = cv.width / N, sY = cv.height / N;
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const q = -Math.PI + codes[y * N + x] * 2 * Math.PI / 255;
      const d = Math.abs(q - ph[y * N + x]);
      const v = clamp(d / 0.012, 0, 1) * S.flash;
      if (v > 0.02) {
        for (let dy = 0; dy < sY; dy++) for (let dx = 0; dx < sX; dx++) {
          const o = ((y * sY + dy) * cv.width + (x * sX + dx)) * 4;
          img.data[o] = img.data[o] * (1 - v) + 255 * v;
          img.data[o + 1] = img.data[o + 1] * (1 - v) + 90 * v;
          img.data[o + 2] = img.data[o + 2] * (1 - v) + 90 * v;
        }
      }
    }
    ctx.putImageData(img, 0, 0);
  }
  // 8-битное представление
  const g = $('#doe-gray');
  drawScaled(g.getContext('2d'), toCanvas(codes, 0), N, N);
  // гистограмма с лестницей 256 уровней
  drawHistogram(codes, D.phase[S.variant]);
  const ph = D.phase[S.variant];
  $('#doe-note').innerHTML =
    `φ ∈ [${fmt(Math.min.apply(null, ph), 2)}, ${fmt(Math.max.apply(null, ph), 2)}] рад<br>` +
    `δ = ${(2 * Math.PI / 255).toFixed(4)} рад · кодов ${new Set(codes).size}/256`;
}

function drawHistogram(codes, phf) {
  const cv = $('#doe-hist');
  const W = cv.width = cv.clientWidth * (window.devicePixelRatio || 1);
  const H = cv.height = 96 * (window.devicePixelRatio || 1);
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, W, H);
  const h = new Array(256).fill(0);
  for (let i = 0; i < N * N; i++) h[codes[i]]++;
  const hmax = Math.max(...h, 1);
  // лестница уровней фазы
  ctx.fillStyle = '#1a2c46';
  for (let i = 0; i < 256; i++) {
    const x = i / 256 * W, w = W / 256;
    ctx.fillRect(x, H - 6, Math.max(1, w - 0.4), 6);
  }
  ctx.fillStyle = '#4dd6c1';
  for (let i = 0; i < 256; i++) {
    const bh = h[i] / hmax * (H - 20);
    ctx.fillRect(i / 256 * W, H - 6 - bh, Math.max(1, W / 256 - (W > 400 ? 0.6 : 0)), bh);
  }
  // маркеры ±π
  ctx.fillStyle = '#8ea1c2';
  ctx.font = `${10 * (window.devicePixelRatio || 1)}px monospace`;
  ctx.fillText('−π', 3, H - 8);
  ctx.fillText('+π', W - 18, H - 8);
  ctx.fillStyle = '#ffb347';
  ctx.fillRect(W / 2 - 0.5, 0, 1, H - 6);
  if (phf) {
    ctx.fillStyle = '#7aa2ff';
    ctx.fillText('коды 0…255', W / 2 + 4, 12 * (window.devicePixelRatio || 1));
  }
}

function doeProbe(ev) {
  const cv = $('#doe-phase');
  const r = cv.getBoundingClientRect();
  const px = clamp(((ev.clientX - r.left) / r.width * N) | 0, 0, N - 1);
  const py = clamp(((ev.clientY - r.top) / r.height * N) | 0, 0, N - 1);
  const code = phaseCodes()[py * N + px];
  const phi = -Math.PI + code * 2 * Math.PI / 255;
  const phf = D.phase[S.variant][py * N + px];
  const mm = `(${(px * 10).toFixed(0)}, ${(py * 10).toFixed(0)}) мкм`;
  $('#doe-probe').innerHTML =
    `px (${px}, ${py}) · ${mm} · код <b>${code}</b> · φ = <b>${phi.toFixed(4)}</b> рад ` +
    `(${(phi * 180 / Math.PI).toFixed(2)}°) · float ${fmt(phf, 4)} · Δ ${fmt(Math.abs(phi - phf), 5)}`;
}

/* --------------------------------------------------- 3. пропагация (A1/A2) */
function renderPropStatic() {
  const m = D.manifest, c = m.constants;
  const track = $('#zscale-track');
  const z0 = D.zgrid[0], z1 = D.zgrid[D.zgrid.length - 1];
  const pos = z => ((z - z0) / (z1 - z0)) * 100;
  let html = '';
  c.zs.forEach((z, i) => {
    html += `<div class="marker" style="left:${pos(z).toFixed(2)}%" data-plane="${i}">z${i + 1}</div>`;
  });
  html += `<div class="crit" style="left:${pos(c.z_crit).toFixed(2)}%">z<sub>crit</sub></div>`;
  track.innerHTML = html;
  track.querySelectorAll('.marker').forEach(el => el.onclick = () => {
    const p = +el.dataset.plane;
    const z = c.zs[p];
    let best = 0, bd = 1e9;
    for (let k = 0; k < D.zgrid.length; k++) {
      const d = Math.abs(D.zgrid[k] - z);
      if (d < bd) { bd = d; best = k; }
    }
    S.playing = false; $('#prop-play').textContent = '▶ пуск';
    setZ(best);
  });
  S.z = nearestZIndex(c.zs[0]);
  $('#prop-slider').max = D.zgrid.length - 1;
  $('#prop-slider').value = S.z;
  drawCone();
}

function nearestZIndex(z) {
  let best = 0, bd = 1e9;
  for (let k = 0; k < D.zgrid.length; k++) {
    const d = Math.abs(D.zgrid[k] - z);
    if (d < bd) { bd = d; best = k; }
  }
  return best;
}

function setZ(k) {
  S.z = clamp(k | 0, 0, D.zgrid.length - 1);
  $('#prop-slider').value = S.z;
  renderZReadout();
  drawCone();
}

function renderZReadout() {
  const z = D.zgrid[S.z];
  const c = D.manifest.constants;
  const dxo = c.lam * z / (c.N * c.dx);
  const s = dxo / c.dx;
  const page = c.N * dxo * 1e3;
  const marg = (z / c.z_crit - 1) * 100;
  const plane = c.zs.findIndex(v => Math.abs(v - z) < 0.01e-3);
  $('#z-readout').innerHTML =
    `z = <b>${(z * 1e3).toFixed(2)} мм</b>` +
    ` · Δx′ = <b>${(dxo * 1e6).toFixed(4)} мкм</b>` +
    ` · s = Δx′/Δx = <b>${s.toFixed(5)}</b>` +
    ` · страница ${page.toFixed(4)} мм` +
    ` · запас до z<sub>crit</sub> <b>${marg.toFixed(2)} %</b>` +
    (plane >= 0 ? ` · <b style="color:#ffd08a">плоскость z${plane + 1}</b>` : '');
  // автопауза на z_i
  if (S.playing && S.autopause && plane >= 0 && S._lastPlane !== plane) {
    S._lastPlane = plane;
    S.playing = false;
    $('#prop-play').textContent = '▶ пуск';
    S._pauseTimer = performance.now() + 900;
  }
  if (plane < 0) S._lastPlane = -1;
}

/** A2: световой конус (SVG), синхронный с анимацией. */
function drawCone() {
  const svg = $('#cone');
  const c = D.manifest.constants;
  const W = 220, H = 320, cx = W / 2;
  const z0 = D.zgrid[0], z1 = D.zgrid[D.zgrid.length - 1];
  const z = D.zgrid[S.z];
  const yOf = zz => 20 + ((zz - z0) / (z1 - z0)) * (H - 50);
  const spread = zz => 18 + ((zz - z0) / (z1 - z0)) * 62;
  let s = '';
  s += `<defs><linearGradient id="beam" x1="0" x2="0" y1="0" y2="1">
        <stop offset="0" stop-color="#4dd6c1" stop-opacity=".55"/>
        <stop offset="1" stop-color="#4dd6c1" stop-opacity=".05"/></linearGradient></defs>`;
  s += `<polygon points="${cx - 40},${yOf(z0)} ${cx + 40},${yOf(z0)} ${cx + spread(z1)},${yOf(z1)} ${cx - spread(z1)},${yOf(z1)}" fill="url(#beam)"/>`;
  // волновые фронты
  for (let i = 0; i < 9; i++) {
    const zz = z0 + (i / 9) * (z1 - z0);
    s += `<line x1="${cx - spread(zz)}" y1="${yOf(zz)}" x2="${cx + spread(zz)}" y2="${yOf(zz)}"
          stroke="#2c6b6b" stroke-width="1" stroke-dasharray="3 4"/>`;
  }
  // плоскости
  c.zs.forEach((zz, i) => {
    const y = yOf(zz);
    s += `<line x1="${cx - spread(zz)}" y1="${y}" x2="${cx + spread(zz)}" y2="${y}" stroke="#7aa2ff" stroke-width="1.5"/>`;
    s += `<text x="${cx + spread(zz) + 4}" y="${y + 3}" fill="#7aa2ff" font-size="9" font-family="monospace">z${i + 1}</text>`;
  });
  // z_crit
  const yc = yOf(c.z_crit);
  s += `<line x1="${cx - 80}" y1="${yc}" x2="${cx + 80}" y2="${yc}" stroke="#ffb347" stroke-width="1" stroke-dasharray="5 4"/>`;
  s += `<text x="4" y="${yc + 3}" fill="#ffb347" font-size="9" font-family="monospace">z_crit</text>`;
  // текущая плоскость
  const y = yOf(z), sp = spread(z);
  s += `<line x1="${cx - sp}" y1="${y}" x2="${cx + sp}" y2="${y}" stroke="#4dd6c1" stroke-width="2.5"/>
        <circle cx="${cx - sp}" cy="${y}" r="2.4" fill="#4dd6c1"/><circle cx="${cx + sp}" cy="${y}" r="2.4" fill="#4dd6c1"/>`;
  s += `<rect x="${cx - 34}" y="${yOf(z0) - 14}" width="68" height="10" rx="2" fill="#1d4a44" stroke="#2f6d63"/>
        <text x="${cx}" y="${yOf(z0) - 6}" fill="#b8fff0" font-size="7.5" text-anchor="middle" font-family="monospace">ДОЭ 128×128</text>`;
  svg.innerHTML = s;
}

function propProbe(ev) {
  const cv = $('#prop-canvas');
  const r = cv.getBoundingClientRect();
  const px = clamp(((ev.clientX - r.left) / r.width * N) | 0, 0, N - 1);
  const py = clamp(((ev.clientY - r.top) / r.height * N) | 0, 0, N - 1);
  const v = D.zstack[S.variant][S.z * N * N + py * N + px];
  const z = D.zgrid[S.z];
  const dxo = D.manifest.constants.lam * z / (N * D.manifest.constants.dx);
  $('#prop-probe').innerHTML =
    `(${px}, ${py}) · ${(px * dxo * 1e6).toFixed(1)}, ${(py * dxo * 1e6).toFixed(1)} мкм · ` +
    `I = <b>${v}</b>/255 (${(v / 255).toFixed(3)}) · z = ${(z * 1e3).toFixed(2)} мм`;
}

/* -------------------------------------------------------- 4. плоскости */
function renderPlanes() {
  const grid = $('#planes-grid');
  grid.innerHTML = '';
  for (let i = 0; i < NP; i++) {
    const cell = document.createElement('div');
    cell.className = 'plane-cell';
    const fm = D.manifest.file_metrics[S.variant][i];
    cell.innerHTML = `<h3><span>плоскость ${i + 1} · z<sub>${i + 1}</sub></span><span class="m">PCC ${fmt(fm.pcc, 2)}</span></h3>`;
    const view = document.createElement('div');
    view.className = 'view';
    const cv = document.createElement('canvas');
    cv.width = cv.height = 160;
    view.appendChild(cv);
    const labL = document.createElement('div'); labL.className = 'lab l'; labL.textContent = 'цель';
    const labR = document.createElement('div'); labR.className = 'lab r'; labR.textContent = 'z';
    view.append(labL, labR);
    cell.appendChild(view);
    grid.appendChild(cell);

    const draw = () => {
      const ctx = cv.getContext('2d');
      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, cv.width, cv.height);
      const T = targetFrame(i), R = reconFrame(S.variant, i);
      if (S.planeMode === 'swipe' || S.planeMode === 'blend') {
        const ch = (S.planeMode === 'swipe') ? S.swipe : S.blend;
        ctx.drawImage(T, 0, 0, cv.width, cv.height);
        ctx.save();
        ctx.beginPath();
        ctx.rect(cv.width * ch, 0, cv.width * (1 - ch), cv.height);
        ctx.clip();
        ctx.drawImage(R, 0, 0, cv.width, cv.height);
        ctx.restore();
        if (S.planeMode === 'swipe') {
          ctx.strokeStyle = '#4dd6c1'; ctx.lineWidth = 2;
          ctx.beginPath(); ctx.moveTo(cv.width * ch, 0); ctx.lineTo(cv.width * ch, cv.height); ctx.stroke();
        }
        labL.textContent = 'цель';
        labR.textContent = S.variant.toUpperCase();
      } else if (S.planeMode === 'side') {
        ctx.drawImage(T, 0, 0, cv.width / 2, cv.height);
        ctx.drawImage(R, cv.width / 2, 0, cv.width / 2, cv.height);
        labL.textContent = 'цель'; labR.textContent = S.variant.toUpperCase();
      } else if (S.planeMode === 'diff') {
        const imT = T.getContext('2d').getImageData(0, 0, N, N);
        const imR = R.getContext('2d').getImageData(0, 0, N, N);
        const out = ctx.createImageData(N, N);
        for (let k = 0; k < N * N; k++) {
          const d = (imT.data[k * 4] - imR.data[k * 4]) / 255;
          const v = clamp(Math.abs(d) * 3, 0, 1);
          out.data[k * 4] = d > 0 ? 255 * v : 30 * v;
          out.data[k * 4 + 1] = 60 * v;
          out.data[k * 4 + 2] = d < 0 ? 255 * v : 30 * v;
          out.data[k * 4 + 3] = 255;
        }
        const tmp = document.createElement('canvas'); tmp.width = tmp.height = N;
        tmp.getContext('2d').putImageData(out, 0, 0);
        ctx.drawImage(tmp, 0, 0, cv.width, cv.height);
        labL.textContent = 'цель − ' + S.variant; labR.textContent = '';
      }
    };
    draw();
    // A5: swipe-шторка мышью
    let dragging = false;
    const setSwipe = ev => {
      const r = view.getBoundingClientRect();
      const v = clamp((ev.clientX - r.left) / r.width, 0, 1);
      if (S.planeMode === 'swipe') S.swipe = v;
      else if (S.planeMode === 'blend') S.blend = v;
      else if (S.planeMode === 'side') S.blend = v;
      else return;
      renderPlanes();
    };
    view.addEventListener('mousedown', e => { dragging = true; setSwipe(e); });
    window.addEventListener('mouseup', () => { dragging = false; });
    view.addEventListener('mousemove', e => { if (dragging) setSwipe(e); });
    view.addEventListener('click', () => setPlane(i));
  }
}

/* ----------------------------------------------------------- 5. метрики */
function renderMetrics(retrigger = true) {
  const fm = D.manifest.file_metrics;
  const nn = fm.nn, gs = fm.gs;
  const proto = S.protocol;
  const pick = r => proto === 'primary' ? [r.pcc, r.ssim] : [r.pcc_raw !== undefined ? r.pcc_raw : r.pcc,
                                                             r.ssim_raw !== undefined ? r.ssim_raw : r.ssim];
  // счётчики (A4: count-up)
  const cur = pick(nn[S.plane]);
  countUp('#c-pcc', cur[0]);
  countUp('#c-ssim', cur[1]);
  const meanN = proto === 'primary' ? D.manifest.metrics.mean : { pcc: avg(nn.map(r => pick(r)[0])), ssim: avg(nn.map(r => pick(r)[1])) };
  const meanG = gs ? { pcc: avg(gs.map(r => pick(r)[0])), ssim: avg(gs.map(r => pick(r)[1])) } : null;
  $('#c-mean-nn').innerHTML = `${fmt(meanN.pcc)}<small> PCC</small> / ${fmt(meanN.ssim)}<small> SSIM</small>`;
  $('#c-mean-gs').innerHTML = meanG ? `${fmt(meanG.pcc)}<small> PCC</small> / ${fmt(meanG.ssim)}<small> SSIM</small>` : '—';

  // столбики
  let bars = '';
  for (let i = 0; i < NP; i++) {
    const a = pick(nn[i]);
    bars += `<div class="mbar"><span>z${i + 1}</span><div class="track" data-plane="${i}">
      <div class="fill" style="width:${(clamp(a[0], 0, 1) * 100).toFixed(1)}%"></div></div>
      <span class="val">${fmt(a[0], 2)}</span></div>`;
    if (gs) {
      const b = pick(gs[i]);
      bars += `<div class="mbar"><span>GS z${i + 1}</span><div class="track" data-plane="${i}">
        <div class="fill gs" style="width:${(clamp(b[0], 0, 1) * 100).toFixed(1)}%"></div></div>
        <span class="val">${fmt(b[0], 2)}</span></div>`;
    }
  }
  $('#metrics-bars').innerHTML = bars;
  $$('#metrics-bars .track').forEach(el => el.onclick = () => setPlane(+el.dataset.plane));

  // таблицы обоих протоколов
  const table = (rows, cls) => `<table class="mini"><tr><th>плоскость</th><th>z, мм</th><th>PCC</th><th>SSIM</th>
      <th>PCC справ.</th><th>SSIM справ.</th></tr>` +
    rows.map((r, i) => `<tr class="${i === S.plane ? 'best' : ''}"><td>${i + 1}</td><td>${r.z_mm.toFixed(1)}</td>
      <td>${fmt(r.pcc)}</td><td>${fmt(r.ssim)}</td>
      <td>${fmt(r.pcc_raw !== undefined ? r.pcc_raw : r.pcc)}</td>
      <td>${fmt(r.ssim_raw !== undefined ? r.ssim_raw : r.ssim)}</td></tr>`).join('') +
    `<tr><td><b>среднее</b></td><td></td><td><b>${fmt(avg(rows.map(r => r.pcc)))}</b></td>
      <td><b>${fmt(avg(rows.map(r => r.ssim)))}</b></td>
      <td><b>${fmt(avg(rows.map(r => r.pcc_raw !== undefined ? r.pcc_raw : r.pcc)))}</b></td>
      <td><b>${fmt(avg(rows.map(r => r.ssim_raw !== undefined ? r.ssim_raw : r.ssim)))}</b></td></tr></table>`;
  $('#metrics-tables').innerHTML =
    `<div><div class="plot-title">NN (сеть)</div>${table(nn)}</div>` +
    (gs ? `<div><div class="plot-title">GS (бейзлайн)</div>${table(gs)}</div>` : '');
  void retrigger;
}

function avg(a) { return a.reduce((x, y) => x + y, 0) / a.length; }

function countUp(sel, target) {
  const el = $(sel);
  if (!el) return;
  el.dataset.target = target;
  if (el.dataset.cur === undefined) {           // первое заполнение — сразу
    el.dataset.cur = target;
    el.innerHTML = `${fmt(target)}<small> ${sel === '#c-pcc' ? 'PCC' : 'SSIM'}</small>`;
    return;
  }
  if (el._anim) return;
  el._anim = true;
  const step = () => {
    const tgt = +el.dataset.target;
    let cur = +(el.dataset.cur || 0);
    cur += (tgt - cur) * 0.18;
    if (Math.abs(tgt - cur) < 0.0005) cur = tgt;
    el.dataset.cur = cur;
    el.innerHTML = `${fmt(cur)}<small> ${sel === '#c-pcc' ? 'PCC' : 'SSIM'}</small>`;
    if (cur !== tgt) requestAnimationFrame(step); else el._anim = false;
  };
  requestAnimationFrame(step);
}

/* --------------------------------------------------------- 6. измерения */
function renderMeas() {
  const cv = $('#meas-canvas');
  const ctx = cv.getContext('2d');
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (S.measMode === 'zprobe' || S.measMode === 'probe' || S.measMode === 'roi') {
    ctx.drawImage(reconFrame(S.variant, S.plane), 0, 0, cv.width, cv.height);
  }
  // рамка ROI
  if (S.roi) {
    const s = cv.width / N;
    ctx.strokeStyle = '#4dd6c1';
    ctx.lineWidth = 2;
    ctx.strokeRect(S.roi.x0 * s, S.roi.y0 * s, (S.roi.x1 - S.roi.x0) * s, (S.roi.y1 - S.roi.y0) * s);
  }
  // SSIM-карта
  const sc = $('#ssim-canvas');
  drawScaled(sc.getContext('2d'), ssimFrame(S.variant, S.plane), sc.width, sc.height);
  drawProfiles();
  drawZProbe();
  renderRuler();
}

function measEvent(ev, kind) {
  const cv = $('#meas-canvas');
  const r = cv.getBoundingClientRect();
  const px = clamp(((ev.clientX - r.left) / r.width * N) | 0, 0, N - 1);
  const py = clamp(((ev.clientY - r.top) / r.height * N) | 0, 0, N - 1);
  S.probePX = px; S.probePY = py;
  if (kind === 'down' && S.measMode === 'roi') { S._roi0 = [px, py]; S.roi = { x0: px, y0: py, x1: px + 1, y1: py + 1 }; }
  if (kind === 'move' && S.measMode === 'roi' && S._roi0) {
    S.roi = { x0: Math.min(S._roi0[0], px), y0: Math.min(S._roi0[1], py),
              x1: Math.max(S._roi0[0], px) + 1, y1: Math.max(S._roi0[1], py) + 1 };
  }
  const rec = reconFrame(S.variant, S.plane);
  const data = rec.getContext('2d').getImageData(px, py, 1, 1).data;
  const v = data[0];
  const z = D.manifest.constants.zs[S.plane];
  const dxo = D.manifest.constants.lam * z / (N * D.manifest.constants.dx);
  const sm = D.ssimMap[S.variant][S.plane * N * N + py * N + px];
  $('#meas-probe').innerHTML =
    `(${px}, ${py}) · ${(px * dxo * 1e6).toFixed(1)}, ${(py * dxo * 1e6).toFixed(1)} мкм · ` +
    `I = ${v}/255 (${(v / 255).toFixed(3)}) · SSIM<sub>лок</sub> ${fmt(sm, 3)}`;
  if (S.measMode === 'zprobe') drawZProbe(px, py);
  if (S.measMode === 'roi') updateROI();
  drawProfiles(px, py);
}

function renderMeasureUI() {
  const plotTitle = S.measMode === 'roi' ? 'ROI-статистика' : '';
  $('#roi-readout').innerHTML = plotTitle
    ? 'потяните рамку мышью по изображению'
    : 'переключите режим «ROI», чтобы измерить область';
}

function drawProfiles(px, py) {
  px = px === undefined ? S.probePX : px;
  py = py === undefined ? S.probePY : py;
  const rec = reconFrame(S.variant, S.plane);
  const d = rec.getContext('2d').getImageData(0, 0, N, N).data;
  const row = new Float32Array(N), col = new Float32Array(N);
  for (let x = 0; x < N; x++) row[x] = d[(py * N + x) * 4] / 255;
  for (let y = 0; y < N; y++) col[y] = d[(y * N + px) * 4] / 255;
  plotLine($('#prof-h'), row, py + 1, 'H');
  plotLine($('#prof-v'), col, px + 1, 'V');
  $('#prof-h-val').textContent = `строка y = ${py}`;
  $('#prof-v-val').textContent = `столбец x = ${px}`;
}

function plotLine(cv, arr, cursor, axis) {
  const dpr = window.devicePixelRatio || 1;
  cv.width = cv.clientWidth * dpr;
  cv.height = 90 * dpr;
  const ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = '#1b2b46';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) { const y = H * i / 4; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }
  ctx.strokeStyle = S.variant === 'nn' ? '#4dd6c1' : '#ffa94d';
  ctx.lineWidth = 1.6 * dpr;
  ctx.beginPath();
  for (let i = 0; i < arr.length; i++) {
    const x = i / (arr.length - 1) * W;
    const y = H - 3 * dpr - arr[i] * (H - 8 * dpr);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  }
  ctx.stroke();
  const cx = (cursor / arr.length) * W;
  ctx.strokeStyle = '#4dd6c166';
  ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, H); ctx.stroke();
  void axis;
}

function drawZProbe(px, py) {
  px = px === undefined ? S.probePX : px;
  py = py === undefined ? S.probePY : py;
  const curves = D.zprobe[S.variant];
  const pix = D.manifest.zprobe_pixels;
  let idx = pix.findIndex(([x, y]) => x === px && y === py);
  if (idx < 0) {
    // ближайший предпросчитанный пиксель (пересчётов нет — п. 12.6.1)
    let bd = 1e9;
    pix.forEach(([x, y], i) => { const dd = (x - px) ** 2 + (y - py) ** 2; if (dd < bd) { bd = dd; idx = i; } });
  }
  const K = D.zgrid.length;
  const arr = curves.subarray(idx * K, (idx + 1) * K);
  const cv = $('#zprobe');
  const dpr = window.devicePixelRatio || 1;
  cv.width = cv.clientWidth * dpr;
  cv.height = 110 * dpr;
  const ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  let mx = 0;
  for (let i = 0; i < K; i++) mx = Math.max(mx, arr[i]);
  ctx.strokeStyle = S.variant === 'nn' ? '#4dd6c1' : '#ffa94d';
  ctx.lineWidth = 1.5 * dpr;
  ctx.beginPath();
  for (let i = 0; i < K; i++) {
    const x = i / (K - 1) * W, y = H - 6 * dpr - (arr[i] / (mx || 1)) * (H - 14 * dpr);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  }
  ctx.stroke();
  // маркеры плоскостей
  const n = D.manifest.constants.zs_mm;
  const z0 = D.manifest.constants.zstack.z0, z1 = D.manifest.constants.zstack.z1;
  n.forEach((z, i) => {
    const x = ((z * 1e-3 - z0) / (z1 - z0)) * W;
    ctx.strokeStyle = '#7aa2ff88';
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    ctx.fillStyle = '#7aa2ff';
    ctx.font = `${9 * dpr}px monospace`;
    ctx.fillText('z' + (i + 1), x + 2, 10 * dpr);
  });
  const xc = (S.z / (K - 1)) * W;
  ctx.strokeStyle = '#ffffff55';
  ctx.beginPath(); ctx.moveTo(xc, 0); ctx.lineTo(xc, H); ctx.stroke();
  const [cx, cy] = pix[idx];
  $('#zprobe-val').textContent = `пиксель (${cx}, ${cy}) · I(z)`;
}

function updateROI() {
  if (!S.roi) return;
  const rec = reconFrame(S.variant, S.plane);
  const { x0, y0, x1, y1 } = S.roi;
  const src = D.recon[S.variant];
  const off = S.plane * N * N;
  let sum = 0, sum2 = 0, mn = 1e9, mx = -1e9, n = 0, total = 0;
  let sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
  for (let i = 0; i < N * N; i++) total += src[off + i];
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
    const v = src[off + y * N + x];
    sum += v; sum2 += v * v; mn = Math.min(mn, v); mx = Math.max(mx, v); n++;
    const t = D.targets[S.plane * N * N + y * N + x] / 255;
    sa += t; sb += v; saa += t * t; sbb += v * v; sab += t * v;
  }
  const mean = sum / n, std = Math.sqrt(Math.max(0, sum2 / n - mean * mean));
  const ev = sum / (total + 1e-9);
  const pcc = (n * sab - sa * sb) / (Math.sqrt(n * saa - sa * sa) * Math.sqrt(n * sbb - sb * sb) + 1e-9);
  // средний локальный SSIM внутри ROI
  const sm = D.ssimMap[S.variant];
  let ss = 0, cnt = 0;
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) { ss += sm[off + y * N + x]; cnt++; }
  $('#roi-readout').innerHTML =
    `ROI [${x0}:${x1}]×[${y0}:${y1}] · mean ${fmt(mean)} · std ${fmt(std)} · min ${fmt(mn)} · max ${fmt(mx)}<br>` +
    `доля энергии ${(ev * 100).toFixed(2)} % · PCC ${fmt(pcc)} · средний SSIM ${fmt(ss / Math.max(cnt, 1))}`;
  void rec;
}

function renderRuler() {
  const c = D.manifest.constants;
  const rows = [];
  const z = c.zs[S.plane];
  for (let i = 1; i < c.zs.length; i++) {
    rows.push([`Δz(z${i + 1} − z${i})`, `${((c.zs[i] - c.zs[i - 1]) * 1e3).toFixed(2)} мм`]);
  }
  const dxo = c.lam * z / (c.N * c.dx);
  rows.push([`активная плоскость`, `z${S.plane + 1} = ${(z * 1e3).toFixed(2)} мм`]);
  rows.push([`Δx′(z)`, `${(dxo * 1e6).toFixed(4)} мкм`]);
  rows.push([`s(z) = Δx′/Δx`, (dxo / c.dx).toFixed(5)]);
  rows.push([`страница`, `${(c.N * dxo * 1e3).toFixed(4)} мм`]);
  const marg = (z / c.z_crit - 1) * 100;
  rows.push([`запас до z_crit`, `<span class="${marg < 1 ? 'warn' : ''}">${marg.toFixed(2)} %</span>`]);
  $('#zruler').innerHTML = `<h4>линейка по z</h4>` +
    rows.map(([k, v]) => `<div class="row"><span>${k}</span><b>${v}</b></div>`).join('');
}

/* ----------------------------------------------------------- 7. обучение */
function renderTrain() {
  const masks = D.trainMasks;
  const M_ = masks.length / (N * N);
  if (!M_) return;
  const i = clamp(S.trainIdx, 0, M_ - 1);
  const mc = $('#train-mask');
  drawScaled(mc.getContext('2d'), phaseCanvas(masks, i * N * N), mc.width, mc.height);
  const rc = $('#train-recon');
  if (D.trainRecon.length >= (i + 1) * N * N) {
    drawScaled(rc.getContext('2d'), toCanvas(D.trainRecon, i * N * N), rc.width, rc.height);
  }
  const steps = D.trainSteps || [];
  $('#train-step-label').textContent =
    `снапшот ${i + 1}/${M_} · шаг ${steps[i] !== undefined ? steps[i] : '—'}`;
  const sl = $('#train-slider');
  sl.max = M_ - 1;
  sl.value = i;
  drawTrainCurve(i);
}

function drawTrainCurve(idx) {
  const log = D.trainLog || [];
  const cv = $('#train-loss');
  const dpr = window.devicePixelRatio || 1;
  cv.width = cv.clientWidth * dpr;
  cv.height = 150 * dpr;
  const ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  if (!log.length) { ctx.fillStyle = '#8ea1c2'; ctx.font = `${12 * dpr}px monospace`; ctx.fillText('лог обучения отсутствует', 12, 24); return; }
  const keys = ['loss', 'val_ssim'];
  const colors = { loss: '#4dd6c1', val_ssim: '#ffa94d' };
  const series = keys.map(k => log.map(r => ({ st: r.step, v: r[k] })).filter(p => p.v !== undefined));
  const xmax = Math.max(...log.map(r => r.step));
  ctx.strokeStyle = '#16233a';
  for (let i = 0; i <= 4; i++) { const y = H * i / 4; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }
  keys.forEach((k, si) => {
    const s = series[si];
    if (!s.length) return;
    const vmin = Math.min(...s.map(p => p.v)), vmax = Math.max(...s.map(p => p.v));
    ctx.strokeStyle = colors[k];
    ctx.lineWidth = 1.6 * dpr;
    ctx.beginPath();
    s.forEach((p, i) => {
      const x = p.st / xmax * W;
      const y = H - 8 * dpr - ((p.v - vmin) / (vmax - vmin + 1e-9)) * (H - 16 * dpr);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
  });
  // текущий шаг
  const steps = D.trainSteps || [];
  const st = steps[Math.min(idx, steps.length - 1)] || 0;
  const x = st / xmax * W;
  ctx.strokeStyle = '#ffffff66';
  ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
  ctx.font = `${10 * dpr}px monospace`;
  ctx.fillStyle = '#4dd6c1'; ctx.fillText('loss', 6, 13 * dpr);
  ctx.fillStyle = '#ffa94d'; ctx.fillText('val SSIM', 6, 26 * dpr);
  ctx.fillStyle = '#8ea1c2'; ctx.fillText(`шаг ${st}`, W - 80 * dpr, 13 * dpr);
}

/* ------------------------------------------------------------- 8. физика */
function renderPhysics() {
  const c = D.manifest.constants;
  const rows = D.manifest.control_table;
  $('#crit-note').innerHTML =
    `z<sub>crit</sub> = <b>${c.z_crit_mm.toFixed(3)} мм</b> · запас z₁ ` +
    `<span style="color:#ffb347">${((c.zs_mm[0] / c.z_crit_mm - 1) * 100).toFixed(2)} %</span>`;
  drawCrit();
  drawLens();
  // таблица 4.7
  let html = `<table><tr><th>№</th><th>z, мм</th><th>Δx′, мкм</th><th>страница, мм</th><th>s = Δx′/Δx</th>
    <th>чирп у края, π</th><th>запас, %</th><th>N_F</th></tr>`;
  rows.forEach(r => {
    html += `<tr data-plane="${r.plane - 1}" class="${r.plane - 1 === S.plane ? 'hl' : ''}">
      <td>${r.plane}</td><td>${r.z_mm.toFixed(1)}</td><td>${r.dx_out_um.toFixed(4)}</td>
      <td>${r.page_mm.toFixed(4)}</td><td>${r.s_i.toFixed(5)}</td>
      <td>${r.chirp_frac_pi.toFixed(5)}</td><td>${r.margin_pct.toFixed(2)}</td>
      <td>${r.fresnel_number.toFixed(1)}</td></tr>`;
  });
  html += `</table>`;
  $('#phys-table').innerHTML = html;
  $$('#phys-table tr[data-plane]').forEach(tr => {
    tr.onmouseenter = () => setPlane(+tr.dataset.plane);
  });
}

function drawCrit() {
  const c = D.manifest.constants;
  const cv = $('#crit-canvas');
  const dpr = window.devicePixelRatio || 1;
  cv.width = cv.clientWidth * dpr;
  cv.height = 120 * dpr;
  const ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height;
  const z0 = 23.9, z1 = 28.0;
  const X = z => (z - z0) / (z1 - z0) * (W - 20 * dpr) + 10 * dpr;
  ctx.clearRect(0, 0, W, H);
  // ось z
  ctx.strokeStyle = '#22304d';
  ctx.beginPath(); ctx.moveTo(X(z0), H * 0.62); ctx.lineTo(X(z1), H * 0.62); ctx.stroke();
  // зона недопустимая
  ctx.fillStyle = '#ff6b6b22';
  ctx.fillRect(0, 0, X(c.z_crit_mm), H);
  ctx.strokeStyle = '#ffb347';
  ctx.setLineDash([4, 3]);
  ctx.beginPath(); ctx.moveTo(X(c.z_crit_mm), 0); ctx.lineTo(X(c.z_crit_mm), H); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#ffb347';
  ctx.font = `${10 * dpr}px monospace`;
  ctx.fillText('z_crit', X(c.z_crit_mm) + 3, 12 * dpr);
  c.zs_mm.forEach((z, i) => {
    ctx.fillStyle = i === S.plane ? '#4dd6c1' : '#7aa2ff';
    ctx.beginPath(); ctx.arc(X(z), H * 0.62, 5 * dpr, 0, 7); ctx.fill();
    ctx.fillStyle = '#cfe0f7';
    ctx.fillText(`z${i + 1}=${z}`, X(z) - 12 * dpr, H * 0.62 - 12 * dpr);
    // запас
    ctx.fillStyle = '#8ea1c2';
    ctx.fillText(`${((z / c.z_crit_mm - 1) * 100).toFixed(2)}%`, X(z) - 12 * dpr, H * 0.62 + 20 * dpr);
  });
  ctx.fillStyle = '#8ea1c2';
  ctx.fillText('z, мм →', W - 50 * dpr, H - 6 * dpr);
}

function drawLens() {
  const prof = D.tests.lens_profile;
  const cv = $('#lens-canvas');
  const dpr = window.devicePixelRatio || 1;
  cv.width = cv.clientWidth * dpr;
  cv.height = 110 * dpr;
  const ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  if (!prof) return;
  const mx = Math.max(...prof);
  ctx.strokeStyle = '#4dd6c1';
  ctx.lineWidth = 1.5 * dpr;
  ctx.beginPath();
  prof.forEach((v, i) => {
    const x = i / (prof.length - 1) * W;
    const y = H - 6 * dpr - Math.sqrt(Math.max(0, v / mx)) * (H - 14 * dpr);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();
  const xc = 64 / 127 * W;
  ctx.strokeStyle = '#ffb34766';
  ctx.beginPath(); ctx.moveTo(xc, 0); ctx.lineTo(xc, H); ctx.stroke();
  ctx.fillStyle = '#8ea1c2';
  ctx.font = `${10 * dpr}px monospace`;
  ctx.fillText('√I профиль, центр = максимум', 6, 12 * dpr);
}

/* ------------------------------------------------------------- главный цикл */
let _lastT = 0, _frames = 0, _fpsT = 0;
function tick(t) {
  const dt = t - _lastT;
  _lastT = t;
  // FPS
  _frames++;
  if (t - _fpsT > 500) {
    S.fps = _frames / ((t - _fpsT) / 1000);
    _fpsT = t; _frames = 0;
    $('#fps').textContent = `${S.fps.toFixed(0)} fps`;
  }
  // A1: z-sweep
  if (S.playing && dt > 0) {
    if (S._pauseTimer && t < S._pauseTimer) {
      // пауза-«задержка» на плоскости
    } else {
      S._pauseTimer = 0;
      S._acc = (S._acc || 0) + dt;
      const per = 110 / S.speed;
      if (S._acc >= per) {
        S._acc = 0;
        let k = S.z + 1;
        if (k >= D.zgrid.length) {
          if (S.loop) k = 0; else { S.playing = false; $('#prop-play').textContent = '▶ пуск'; k = S.z; }
        }
        setZ(k);
      }
    }
  }
  // отрисовка кадра пропагации
  const cv = $('#prop-canvas');
  if (cv && D.zstack && D.zstack[S.variant]) {
    drawScaled(cv.getContext('2d'), frame(S.variant, S.z), cv.width, cv.height);
  }
  // затухание вспышки A6
  if (S.flash > 0) {
    S.flash = Math.max(0, S.flash - dt / 900);
    renderDoe();
  }
  // A3: play обучения
  if (S.trainPlaying) {
    S._tacc = (S._tacc || 0) + dt;
    if (S._tacc > 650) {
      S._tacc = 0;
      const M_ = D.trainMasks.length / (N * N);
      S.trainIdx = (S.trainIdx + 1) % M_;
      renderTrain();
    }
  }
  requestAnimationFrame(tick);
}

/* --------------------------------------------------------------- обработчики */
function wire() {
  // глобальный переключатель NN/GS (п. 12.5.7)
  $$('.variant-switch button').forEach(b => b.onclick = () => {
    S.variant = b.dataset.variant;
    $$('.variant-switch button').forEach(x => x.classList.toggle('active', x === b));
    refreshAll();
  });
  // таблица 4.7 в модальном окне (панель 8)
  const pm = $('#phys-modal');
  if (pm) pm.onclick = () => openModal('Таблица 4.7 — расчётные параметры', $('#phys-table').innerHTML);
  // ДОЭ
  $('#doe-quant').onchange = e => { S.quant = e.target.checked; renderDoe(); };
  $('#doe-flash').onclick = () => { S.flash = 1; renderDoe(); };
  $('#doe-phase').onmousemove = doeProbe;
  // пропагация
  $('#prop-play').onclick = () => {
    S.playing = !S.playing;
    $('#prop-play').textContent = S.playing ? '⏸ пауза' : '▶ пуск';
    if (S.playing) S._lastPlane = -1;
  };
  $('#prop-speed').oninput = e => { S.speed = +e.target.value; $('#prop-speed-val').textContent = '×' + S.speed; };
  $('#prop-loop').onchange = e => S.loop = e.target.checked;
  $('#prop-autopause').onchange = e => S.autopause = e.target.checked;
  $('#prop-slider').oninput = e => { S.playing = false; $('#prop-play').textContent = '▶ пуск'; setZ(+e.target.value); };
  $('#prop-canvas').onmousemove = propProbe;
  // плоскости
  $$('#plane-mode button').forEach(b => b.onclick = () => {
    S.planeMode = b.dataset.mode;
    $$('#plane-mode button').forEach(x => x.classList.toggle('active', x === b));
    renderPlanes();
  });
  // протокол метрик
  $$('#protocol-switch button').forEach(b => b.onclick = () => {
    S.protocol = b.dataset.proto;
    $$('#protocol-switch button').forEach(x => x.classList.toggle('active', x === b));
    renderMetrics();
  });
  // измерения
  $$('#meas-mode button').forEach(b => b.onclick = () => {
    S.measMode = b.dataset.mmode;
    $$('#meas-mode button').forEach(x => x.classList.toggle('active', x === b));
    renderMeasureUI();
    renderMeas();
    updateROI();
  });
  const mc = $('#meas-canvas');
  mc.addEventListener('mousedown', e => measEvent(e, 'down'));
  mc.addEventListener('mousemove', e => measEvent(e, 'move'));
  window.addEventListener('mouseup', () => { S._roi0 = null; });
  // обучение
  $('#train-slider').oninput = e => { S.trainIdx = +e.target.value; renderTrain(); };
  $('#train-play').onclick = () => { S.trainPlaying = !S.trainPlaying; $('#train-play').textContent = S.trainPlaying ? '⏸ пауза' : '▶ play'; };
  // модальное окно
  $('#modal-close').onclick = () => $('#modal').classList.add('hidden');
  $('#modal').onclick = e => { if (e.target.id === 'modal') $('#modal').classList.add('hidden'); };
  // A8: деградация битности — переключение по щелчку на панели «Плоскости»
  wireBitdepth();
  window.addEventListener('keydown', e => {
    if (e.key === ' ') { e.preventDefault(); $('#prop-play').click(); }
    if (e.key === 'ArrowRight') setZ(S.z + 1);
    if (e.key === 'ArrowLeft') setZ(S.z - 1);
    if (e.key === 'n' || e.key === 'N') $$('.variant-switch button')[0].click();
    if (e.key === 'g' || e.key === 'G') $$('.variant-switch button')[1].click();
    if (e.key === 't') toggleTour();
  });
  window.addEventListener('resize', () => { renderPlanes(); drawHistogram(phaseCodes(), D.phase[S.variant]); renderPhysics(); renderMetrics(false); drawStepsCurve(D.manifest.phys_steps); });
}

/* --------------------------------- физика внутри сети: шаги K (панель 2) */
function renderPhysSteps() {
  const row = $('#steps-row');
  if (!row) return;
  if (!D.physSteps || !D.physSteps.length) { $('#phys-steps').style.display = 'none'; return; }
  const nFrames = D.physSteps.length / (N * N);
  const meta = (D.manifest.phys_steps || []);
  const total = D.manifest.phys_total || 0;
  S.physStep = Math.max(0, Math.min(S.physStep, nFrames - 1));
  row.innerHTML = '';
  for (let i = 0; i < nFrames; i++) {
    const cell = document.createElement('div');
    cell.className = 'step-cell' + (i === S.physStep ? ' active' : '');
    const cv = document.createElement('canvas');
    cv.width = cv.height = 96;
    drawScaled(cv.getContext('2d'), phaseCanvas(D.physSteps, i * N * N), 96, 96);
    const cap = document.createElement('span');
    const pcc = meta[i] ? meta[i].pcc.toFixed(3) : '—';
    const kk = meta[i] && meta[i].step !== undefined ? meta[i].step : i;
    cap.innerHTML = `${i === 0 ? 'U-Net' : (i === nFrames - 1 ? `K=${total} · 8 бит` : `K=${kk}`)}<b>${pcc}</b>`;
    cell.appendChild(cv);
    cell.appendChild(cap);
    cell.onclick = () => { S.physStep = i; renderPhysSteps(); };
    row.appendChild(cell);
  }
  drawStepsCurve(meta);
  const p = $('#steps-probe');
  if (p) {
    const m = meta[S.physStep];
    const kk = m && m.step !== undefined ? m.step : S.physStep;
    p.innerHTML = m
      ? `шаг физики K = <b>${kk}</b> / ${total || '—'} · PCC <b>${m.pcc.toFixed(4)}</b>` +
        (m.per_plane ? ` · плоскости ${m.per_plane.map(v => v.toFixed(3)).join(' · ')}` : '')
      : '';
  }
}

/* кривая качества по шагам физики — клик/перетаскивание выбирает кадр */
function drawStepsCurve(meta) {
  const cv = $('#steps-curve');
  if (!cv || !meta || !meta.length) return;
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth || 400, h = 58;
  cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const vals = meta.map(m => m.pcc);
  const v0 = Math.min(...vals) - 0.01, v1 = Math.max(...vals) + 0.01;
  const X = i => 2 + (w - 4) * (meta.length > 1 ? i / (meta.length - 1) : 0.5);
  const Y = v => h - 12 - (h - 20) * ((v - v0) / (v1 - v0 || 1));
  ctx.beginPath();
  vals.forEach((v, i) => i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v)));
  const grad = ctx.createLinearGradient(0, 0, w, 0);
  grad.addColorStop(0, 'rgba(122,162,255,.25)');
  grad.addColorStop(0.15, 'rgba(122,162,255,.9)');
  grad.addColorStop(1, '#4dd6c1');
  ctx.lineWidth = 1.6; ctx.strokeStyle = grad; ctx.stroke();
  ctx.lineTo(X(vals.length - 1), h - 2); ctx.lineTo(X(0), h - 2); ctx.closePath();
  ctx.fillStyle = 'rgba(122,162,255,.10)'; ctx.fill();
  // маркер выбранного кадра
  const i = Math.min(S.physStep, vals.length - 1);
  ctx.beginPath(); ctx.arc(X(i), Y(vals[i]), 3.2, 0, Math.PI * 2);
  ctx.fillStyle = '#4dd6c1'; ctx.fill();
  ctx.strokeStyle = 'rgba(77,214,193,.35)'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(X(i), 2); ctx.lineTo(X(i), h - 2); ctx.stroke();
  // подписи начала и конца
  ctx.font = '9px ui-monospace, monospace'; ctx.fillStyle = 'rgba(190,200,215,.7)';
  ctx.fillText(vals[0].toFixed(3), 3, 9);
  const last = vals[vals.length - 1].toFixed(3);
  ctx.fillText(last, w - ctx.measureText(last).width - 3, Y(vals[vals.length - 1]) - 4);
  // скраб
  const pick = ev => {
    const r = cv.getBoundingClientRect();
    const t = Math.max(0, Math.min(1, (ev.clientX - r.left - 2) / (r.width - 4)));
    S.physStep = Math.round(t * (meta.length - 1));
    renderPhysSteps();
  };
  cv.style.cursor = 'pointer';
  cv.onpointerdown = ev => { cv.setPointerCapture(ev.pointerId); pick(ev); };
  cv.onpointermove = ev => { if (ev.buttons) pick(ev); };
}

/* ------------------------------------------------ A8: деградация битности */
function wireBitdepth() {
  const bar = document.createElement('div');
  bar.id = 'bitbar';
  bar.style.cssText = 'display:flex;gap:8px;align-items:center;margin-top:10px;flex-wrap:wrap';
  bar.innerHTML = `<span class="plot-title" style="margin:0">A8 · битность фазы:</span>` +
    [8, 6, 4, 2].map((b, i) => `<button data-bit="${b}" class="${b === 8 ? 'active primary' : ''}">${b} бит</button>`).join('') +
    `<span class="hint">реконструкция z₂ из маски, квантованной в b бит (предпросчитано)</span>`;
  $('#panel-planes').appendChild(bar);
  $$('#bitbar button').forEach(b => b.onclick = () => {
    const bit = +b.dataset.bit;
    $$('#bitbar button').forEach(x => x.classList.toggle('primary', x === b));
    const levels = [8, 6, 4, 2];
    const idx = levels.indexOf(bit);
    const src = D.bitdepth[S.variant];
    const c = toCanvas(src, idx * N * N);
    CACHE.small['bitprev'] = c;
    const cv = $('#prop-canvas');
    drawScaled(cv.getContext('2d'), c, cv.width, cv.height);
    S.playing = false;
    $('#prop-play').textContent = '▶ пуск';
    $('#prop-probe').innerHTML = `<b>${bit} бит</b> (${2 ** bit} уровней фазы) · z₂ = 25.3 мм · ` +
      `шаг ${(2 * Math.PI / (2 ** bit - 1)).toFixed(4)} рад`;
  });
}

/* ------------------------------------------------------------- авто-тур */
const TOUR = [
  { t: 'Шапка: параметры и тесты', d: 'λ = 532 нм, Δx = 10 мкм, N = 128, z₁…z₄ = 24,1…27,7 мм, z_crit = 24,0602 мм. Бейджи — пройденные тесты 8.1–8.7; клик открывает числа.', ms: 3500 },
  { t: 'Цели и ДОЭ', d: 'Четыре цели (канал i ↔ z_i). Маска ДОЭ: фаза в циклической палитре и 8-битное представление. Наведите курсор — код и фаза в рад.', ms: 4200, act: () => { S.flash = 1; renderDoe(); } },
  { t: 'z-свип', d: 'Покадровое проигрывание стопки интенсивностей (шаг 0,02 мм): изображения «собираются» на z_i, между плоскостями — спеклы. Автопауза на каждой z_i.', ms: 5000, act: () => { S.playing = true; $('#prop-play').textContent = '⏸ пауза'; S.autopause = true; } },
  { t: 'Измерения', d: 'z-зонд I(z) по кликнутому пикселю, ROI-статистика, профили H/V, локальная SSIM-карта.', ms: 4500, act: () => { S.playing = false; S.measMode = 'roi'; renderMeas(); S.roi = { x0: 40, y0: 40, x1: 88, y1: 88 }; updateROI(); } },
  { t: 'Плоскости и метрики', d: 'Шторка цель↔реконструкция по худшей плоскости; столбики PCC NN vs GS и таблицы обоих протоколов.', ms: 4500, act: () => { setPlane(0); S.planeMode = 'swipe'; S.swipe = 0.5; renderPlanes(); } },
  { t: 'Обучение', d: 'Снапшоты маски каждые 100 шагов с синхронной кривой потерь и контрольного SSIM.', ms: 4000, act: () => { S.trainPlaying = true; } },
  { t: 'Физика', d: 'Граница применимости z_crit и положения z_i; тест линзы (максимум в центральном пикселе); таблица Δx′(z), s_i.', ms: 3200, act: () => { S.trainPlaying = false; } },
  { t: 'Итог', d: 'Деградация битности 8→6→4→2 и итоговая таблица метрик.', ms: 3000, act: () => { $$('#bitbar button')[3].click(); } },
];
let _tourTimer = null, _tourIdx = 0;
function refreshAll() {
  renderPlanes();
  renderMetrics(false);
  renderMeas();
  renderDoe();
  renderPhysics();
}

function toggleTour() {
  if (_tourTimer) { clearInterval(_tourTimer); _tourTimer = null; $('#tour').classList.add('hidden'); return; }
  _tourIdx = 0;
  const el = $('#tour');
  el.classList.remove('hidden');
  const show = () => {
    const st = TOUR[_tourIdx];
    if (!st) { clearInterval(_tourTimer); _tourTimer = null; el.classList.add('hidden'); return; }
    $('#tour-step').textContent = `авто-тур ${_tourIdx + 1}/${TOUR.length} · сценарий раздела 13`;
    $('#tour-text').textContent = `${st.t} — ${st.d}`;
    $('#tour-progress').style.width = '0%';
    if (st.act) st.act();
    let p = 0;
    const step = st.ms / 30;
    clearInterval(_tourTimer);
    _tourTimer = setInterval(() => {
      p += 100 / (st.ms / 33);
      $('#tour-progress').style.width = Math.min(100, p) + '%';
    }, 33);
    _tourIdx++;
    setTimeout(() => { if (_tourTimer) show(); }, st.ms);
  };
  show();
  void TOUR;
}

/* --------------------------------------------------------------- запуск */
async function main() {
  try {
    await loadBundle();
  } catch (err) {
    document.body.insertAdjacentHTML('afterbegin',
      `<div style="padding:20px;background:#2a0f0f;border:1px solid #5b2b2b;border-radius:12px;margin-bottom:12px">
       <b>Не удалось загрузить бандл данных.</b><br>Откройте <code>index.html</code> двойным кликом
       (файл <code>data_bundle.js</code> должен лежать рядом) или запустите
       <code>python -m http.server</code> в каталоге dashboard.<br><small>${err}</small></div>`);
    return;
  }
  renderHeader();
  renderTargets();
  renderDoe();
  renderPropStatic();
  setZ(S.z);
  renderPlanes();
  renderMetrics();
  renderMeas();
  renderMeasureUI();
  renderTrain();
  renderPhysics();
  renderPhysSteps();
  wire();
  $('#footer-info').innerHTML =
    `seed ${D.manifest.constants.seed} · офлайн`;
  // кнопка авто-тура
  document.body.insertAdjacentHTML('beforeend',
    `<button class="tour-btn" onclick="toggleTour()" title="авто-тур по сценарию (T)">▶ тур</button>`);
  await initRunner();       // пульт запуска задач (только под serve_dashboard.py)
  requestAnimationFrame(tick);
}
// глобальные ссылки для инлайновых обработчиков (index.html) и авто-проверок
window.refreshAll = refreshAll;
window.toggleTour = toggleTour;
window.HDOE_DEBUG = { loadBundle, main, renderPlanes, renderMetrics, renderPhysics, b64ToBufGz };
// перечитать данные после пересборки бандла (кнопка запуска, см. RUNNER ниже)
window.HDOE_RELOAD = async () => {
  try {
    CACHE.frames = {}; CACHE.small = {};
    await loadBundle();
    renderHeader(); renderTargets(); renderDoe(); renderPropStatic(); setZ(S.z);
    renderPlanes(); renderMetrics(); renderMeas(); renderMeasureUI(); renderTrain();
    renderPhysics(); renderPhysSteps();
    $('#footer-info').innerHTML = `seed ${D.manifest.constants.seed} · данные перечитаны ` +
      `${new Date().toLocaleTimeString('ru-RU')}`;
  } catch (e) { /* оставим прежние данные */ }
};

document.addEventListener('DOMContentLoaded', main);

/* ===================================================================
   Пульт запуска задач. Работает только когда дашборд отдан локальным
   сервером scripts/serve_dashboard.py (он же и запускает этапы).
   При открытии index.html двойным кликом (file://) пульт скрыт:
   дашборд остаётся полностью офлайн-статическим, как требует ТЗ.
   =================================================================== */
const RUN = { mode: 'off', available: false, poll: null, since: 0, stages: [], pipelines: [] };

async function probeRunner() {
  if (location.protocol === 'file:') return false;
  try {
    const r = await fetch('api/state', { cache: 'no-store' });
    if (!r.ok) return false;
    const js = await r.json();
    RUN.available = true;
    RUN.stages = js.stages || [];
    RUN.pipelines = js.pipelines || [];
    RUN.state = js.state || {};
    RUN.files = js.files || {};
    RUN.metrics = js.metrics || null;
    RUN.python = js.python;
    return true;
  } catch (e) {
    return false;
  }
}

function buildRunner() {
  if (!RUN.available) return;
  const btn = $('#run-btn');
  if (btn) btn.classList.remove('hidden');
  $('#rp-pipelines').innerHTML = RUN.pipelines
    .map(p => `<button data-pl="${p.id}" title="${p.stages.join(' → ')}">${p.title}</button>`)
    .join('');
  const groups = {};
  RUN.stages.forEach(s => { (groups[s.group] = groups[s.group] || []).push(s); });
  $('#rp-stages').innerHTML = Object.entries(groups).map(([g, items]) =>
    `<div style="grid-column:1/-1" class="plot-title">${g}</div>` +
    items.map(s => `<button class="rp-stage" data-stage="${s.id}">
        <b>${s.title}</b><span>${s.hint || ''}</span></button>`).join('')).join('');
  void btn;

  $('#rp-pipelines').querySelectorAll('button').forEach(b => {
    b.onclick = () => runStart({ pipeline: b.dataset.pl });
  });
  $('#rp-stages').querySelectorAll('.rp-stage').forEach(b => {
    b.onclick = () => runStart({ stages: [b.dataset.stage] });
  });
  $('#rp-close').onclick = () => $('#runpanel').classList.add('hidden');
  $('#rp-stop').onclick = () => fetch('api/stop', { method: 'POST' }).then(refreshRunState);
  $('#rp-copy').onclick = () => {
    const t = $('#rp-log').textContent;
    (navigator.clipboard ? navigator.clipboard.writeText(t) : Promise.reject())
      .then(() => flashStatus('лог скопирован', 'ok'))
      .catch(() => flashStatus('копирование недоступно', 'fail'));
  };
  // приём картинки: перетаскивание + выбор файла
  const drop = $('#rp-drop'), file = $('#rp-file');
  file.onchange = () => { if (file.files[0]) uploadImage(file.files[0]); };
  ['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, e => {
    e.preventDefault(); drop.classList.add('hover');
  }));
  ['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => {
    e.preventDefault(); drop.classList.remove('hover');
  }));
  drop.addEventListener('drop', e => {
    const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) uploadImage(f);
  });
  refreshRunState();
}

function flashStatus(text, kind) {
  const el = $('#rp-status');
  el.textContent = text;
  el.className = 'rp-status' + (kind === 'busy' ? ' busy' : kind === 'fail' ? ' fail' : '');
}

async function refreshRunState() {
  try {
    const js = await (await fetch('api/state', { cache: 'no-store' })).json();
    RUN.state = js.state; RUN.files = js.files; RUN.metrics = js.metrics;
  } catch (e) { return; }
  const st = RUN.state || {};
  document.querySelectorAll('.rp-stage').forEach(b => {
    b.classList.toggle('active', b.dataset.stage === st.running);
  });
  const busy = !!st.running;
  $('#run-btn').classList.toggle('busy', busy);
  $('#run-btn').textContent = busy
    ? `⏳ ${(RUN.stages.find(s => s.id === st.running) || {}).title || 'работа'}`
    : '▶ запуск';
  if (busy) {
    flashStatus('выполняется', 'busy');
  } else if (st.returncode === 0) {
    flashStatus('готово', 'ok');
  } else if (st.returncode) {
    flashStatus(`ошибка ${st.returncode}`, 'fail');
  } else {
    flashStatus('готов', 'ok');
  }
  // что уже посчитано — подписываем на плитках этапов
  const F = RUN.files || {};
  const label = (key, txt) => {
    const el = document.querySelector(`.rp-stage[data-stage="${key}"]`);
    if (!el || !F[txt]) return;
    let s = el.querySelector('small');
    if (!s) { s = document.createElement('small'); el.appendChild(s); }
    s.textContent = `готово · ${F[txt].mtime_str}`;
  };
  label('evaluate', 'doe'); label('tests', 'tests'); label('report', 'report');
  label('video', 'video'); label('bundle', 'bundle'); label('train', 'ckpt');
  label('sweep', 'sweep');
}

async function runStart(payload) {
  const btn = $('#run-btn');
  btn.classList.add('busy');
  $('#runpanel').classList.remove('hidden');
  $('#rp-log').textContent = '';
  RUN.since = 0;
  try {
    const r = await fetch('api/run', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const js = await r.json();
    if (!js.ok) { flashStatus(js.message || 'не удалось', 'fail'); return; }
    flashStatus('выполняется', 'busy');
    runPoll();
  } catch (e) {
    flashStatus('нет связи с пультом', 'fail');
  }
}

function runPoll() {
  if (RUN.poll) return;
  const step = async () => {
    if (!RUN.available) return;
    let js = null;
    try {
      js = await (await fetch(`api/log?since=${RUN.since}`, { cache: 'no-store' })).json();
    } catch (e) {
      RUN.poll = setTimeout(step, 1500); return;
    }
    if (js.lines && js.lines.length) {
      const pre = $('#rp-log');
      const tail = js.lines.join('');
      pre.insertAdjacentHTML('beforeend', escapeHtml(tail));
      RUN.since = js.total;
      pre.scrollTop = pre.scrollHeight;
    }
    if (js.running) {
      RUN.poll = setTimeout(step, 700);
      refreshRunStateQuiet();
    } else {
      RUN.poll = null;
      await refreshRunState();
      $('#rp-log').insertAdjacentHTML('beforeend', '\n— этап завершён —\n');
      // после счёта перечитываем бандл, если он пересобран
      await maybeReloadBundle();
    }
  };
  RUN.poll = setTimeout(step, 300);
}

async function refreshRunStateQuiet() {
  try {
    const js = await (await fetch('api/state', { cache: 'no-store' })).json();
    RUN.state = js.state;
    document.querySelectorAll('.rp-stage').forEach(b => {
      b.classList.toggle('active', b.dataset.stage === (js.state || {}).running);
    });
  } catch (e) { /* тихо */ }
}

const escapeHtml = s => s.replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

async function uploadImage(file) {
  flashStatus('загрузка картинки…', 'busy');
  try {
    const r = await fetch('api/upload', { method: 'POST', body: file });
    const js = await r.json();
    if (!js.ok) { flashStatus(js.message || 'не принято', 'fail'); return; }
    flashStatus(`цель ${js.width}×${js.height} принята — жми «Цели из картинки»`, 'ok');
  } catch (e) {
    flashStatus('загрузка не удалась', 'fail');
  }
}

/* После пересборки бандла подменяем data_bundle.js свежими данными,
   чтобы числа на панелях совпали с новыми metrics.json без F5. */
async function maybeReloadBundle() {
  try {
    const r = await fetch('data_bundle.js', { cache: 'no-store' });
    const code = await r.text();
    if (!code || code.length < 1000) return;
    const el = document.createElement('script');
    el.textContent = code;
    document.head.appendChild(el);
    if (window.HDOE_BUNDLE && window.HDOE_RELOAD) window.HDOE_RELOAD();
  } catch (e) { /* тихо */ }
}

/* горячие клавиши пульта */
function wireRunnerKeys() {
  window.addEventListener('keydown', e => {
    if (!RUN.available) return;
    if (e.key === 'r' || e.key === 'R') {
      e.preventDefault();
      const p = $('#runpanel');
      p.classList.toggle('hidden');
      if (!p.classList.contains('hidden')) refreshRunState();
    }
  });
}

async function initRunner() {
  const ok = await probeRunner();
  if (!ok) return;
  buildRunner();
  wireRunnerKeys();
  const btn = $('#run-btn');
  if (btn) btn.onclick = () => {
    const p = $('#runpanel');
    p.classList.toggle('hidden');
    if (!p.classList.contains('hidden')) refreshRunState();
  };
  RUN.poll = null;
  runPoll();
}

window.HDOE_RUNNER = { RUN, runStart, initRunner, refreshRunState };
