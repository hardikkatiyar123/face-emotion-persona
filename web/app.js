// Face Emotion Persona: live emotion detection with a persona for every mood.
// Everything runs in the browser; no video is uploaded.
// Three emotion models, picked in the Model menu:
//   HSEmotion: EfficientNet-B0 trained on AffectNet (web/models/hsemotion/), run with onnxruntime-web
//   Built-in:  face-api.js' small expression model
//   Custom:    YOUR model from training/train.py (web/model/model.json)
// Faces are always found with face-api.js' Tiny Face Detector.

const FACE_MODELS = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.13/model/';
const STABLE_MS = 500;     // an emotion must stay on top this long before the persona changes
const FADE_MS = 300;       // persona cross-fade (half out, half in)
const TRACK_TTL_MS = 700;  // forget a face this long after it was last seen
const SMOOTH_MS = 450;     // time constant of the score smoothing (same at any frame rate)
const MIN_CONFIDENCE = 0.4; // below this, or ...
const MIN_MARGIN = 0.12;    // ... when the top two are closer than this, show "Uncertain"
const INK_ON_COLOR = '#0a0e16';
const UNCERTAIN = { label: 'Uncertain 🤔', color: '#c3cad6' };

// Face detector input size adapts to the frame rate: bigger finds faces better, but is slower
const DETECTOR_SIZES = [320, 416, 512];
const TUNE_EVERY_MS = 3000;
const FPS_LOW = 15, FPS_HIGH = 24;

// HSEmotion, preprocessing as in EmotiEffLib's own code (emotiefflib/facial_analysis.py):
// tight RGB face crop, resized to 224x224, scaled to 0–1, ImageNet mean/std, NCHW.
const HSE = {
  url: 'models/hsemotion/enet_b0_8_best_vgaf.onnx',
  size: 224,
  mean: [0.485, 0.456, 0.406],
  std: [0.229, 0.224, 0.225],
  // Output order (AffectNet, 8 classes). Contempt has no match in our 7 emotions, so it's dropped.
  labels: ['angry', null, 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'],
  fastMs: 50, // made the default model only if one face takes less than this on this device
};

const EMOTIONS = [
  { key: 'happy',    label: 'Happy',     color: '#ffc23d', emoji: '☀️', persona: 'Sunshine Mode',         tagline: 'Main character energy' },
  { key: 'sad',      label: 'Sad',       color: '#5b9cff', emoji: '🌧️', persona: 'Rainy Day Poet',        tagline: 'Writing sad songs' },
  { key: 'angry',    label: 'Angry',     color: '#ff5a4e', emoji: '🌋', persona: 'Volcano Mode',          tagline: 'Do not disturb' },
  { key: 'surprise', label: 'Surprised', color: '#b07cff', emoji: '🤯', persona: 'Plot Twist Detected',   tagline: 'Didn’t see that coming' },
  { key: 'fear',     label: 'Scared',    color: '#2fd4c0', emoji: '👻', persona: 'Horror Movie Survivor', tagline: 'Made it to the credits' },
  { key: 'disgust',  label: 'Disgusted', color: '#9bd23c', emoji: '🤢', persona: 'Food Critic',           tagline: 'Not impressed' },
  { key: 'neutral',  label: 'Neutral',   color: '#9aa6b8', emoji: '🗿', persona: 'Stone Cold Chill',      tagline: 'Unbothered. Moisturized.' },
];
const BY_KEY = Object.fromEntries(EMOTIONS.map(e => [e.key, e]));

// Different names for the same emotion (face-api.js vs FER-2013 folders)
const ALIASES = {
  happy: 'happy', sad: 'sad', angry: 'angry', neutral: 'neutral',
  surprise: 'surprise', surprised: 'surprise',
  fear: 'fear', fearful: 'fear',
  disgust: 'disgust', disgusted: 'disgust',
};

const els = {
  video: document.getElementById('video'),
  canvas: document.getElementById('overlay'),
  stage: document.querySelector('.stage'),
  startBtn: document.getElementById('start'),
  startLabel: document.getElementById('start-label'),
  toggle: document.getElementById('toggle'),
  snapshot: document.getElementById('snapshot'),
  model: document.getElementById('model'),
  modelNote: document.getElementById('model-note'),
  status: document.getElementById('status'),
  readout: document.getElementById('readout'),
  readoutPct: document.getElementById('readout-pct'),
  caption: document.getElementById('readout-caption'),
  bars: document.getElementById('bars'),
  persona: document.getElementById('persona'),
  personaEmoji: document.getElementById('persona-emoji'),
  personaName: document.getElementById('persona-name'),
  personaTagline: document.getElementById('persona-tagline'),
  announce: document.getElementById('announce'),
  fps: document.getElementById('fps'),
  flash: document.getElementById('flash'),
};
const ctx = els.canvas.getContext('2d');
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');

let stream = null;
let running = false;
let session = 0; // bumped on every start/stop so detections from an old session are dropped
let customModel = null;
let customLabels = null;
let hseSession = null;
let userPickedModel = false; // once the user picks a model, don't switch it for them
let detectorOptions = null;
let detectorLevel = 1, maxDetectorLevel = DETECTOR_SIZES.length - 1;
let tuneFrom = 0, tuneFrames = 0;

const hseCanvas = document.createElement('canvas');
hseCanvas.width = hseCanvas.height = HSE.size;
const hseCtx = hseCanvas.getContext('2d', { willReadFrequently: true });

// Each face is followed across frames as a "track" with its own smoothed scores and persona
let tracks = [];
let lastFaces = [];   // [{ box, track }] from the latest frame
let mainTrack = null; // the closest (biggest) face, shown in the side panel
let cardKey = null;
let cardTimer = null;
let fps = 0, lastFrameAt = 0, fpsShownAt = 0;

const shown = Object.fromEntries(EMOTIONS.map(e => [e.key, 0]));
const barEls = {};

function setStatus(msg, isError = false) {
  els.status.textContent = msg;
  els.status.classList.toggle('error', isError);
}

function buildBars() {
  for (const e of EMOTIONS) {
    const li = document.createElement('li');
    li.className = 'bar-row';
    li.innerHTML = `
      <span>${e.label}</span>
      <span class="bar-track"><span class="bar-fill" style="--c:${e.color}"></span></span>
      <span class="bar-value">0%</span>`;
    els.bars.appendChild(li);
    barEls[e.key] = { row: li, fill: li.querySelector('.bar-fill'), value: li.querySelector('.bar-value') };
  }
}

// Both start buttons (big one on the start screen, small one in the controls)
function setStartEnabled(enabled, label = 'Start camera') {
  els.startBtn.disabled = !enabled;
  els.startLabel.textContent = label;
  els.toggle.disabled = !enabled;
  els.toggle.textContent = label;
}

// ---------- Model loading ----------
async function init() {
  buildBars();
  try {
    try { await faceapi.tf.setBackend('webgl'); } catch { /* falls back to CPU */ }
    await faceapi.tf.ready();
    await Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri(FACE_MODELS),
      faceapi.nets.faceExpressionNet.loadFromUri(FACE_MODELS),
    ]);
    setDetectorLevel(1);
  } catch (err) {
    console.error(err);
    els.startLabel.textContent = 'Models failed to load';
    setStatus('Face models failed to load. Check your internet connection and reload the page.', true);
    return;
  }

  await loadCustomModel();
  setStartEnabled(true);
  setStatus('Ready. Start the camera when you are.');
  loadHSEmotion(); // big download: runs in the background, the camera can start meanwhile
}

async function loadCustomModel() {
  try {
    // manifest.json always exists, so checking for a trained model never logs a 404.
    // train.py sets "customModel" to true when it exports one.
    const manifest = await (await fetch('model/manifest.json', { cache: 'no-store' })).json();
    if (!manifest.customModel) return; // no trained model deployed: that's fine
    const res = await fetch('model/labels.json', { cache: 'no-store' });
    if (!res.ok) throw new Error(`labels.json: HTTP ${res.status}`);
    customLabels = (await res.json()).map(l => ALIASES[String(l).toLowerCase()] || String(l).toLowerCase());
    customModel = await faceapi.tf.loadLayersModel('model/model.json');
    const opt = els.model.querySelector('option[value="custom"]');
    opt.disabled = false;
    opt.textContent = 'My trained model';
    if (!userPickedModel) els.model.value = 'custom';
  } catch (err) {
    console.warn('Custom model not loaded:', err);
    customModel = null;
  }
}

async function loadHSEmotion() {
  const opt = els.model.querySelector('option[value="hsemotion"]');
  try {
    if (typeof ort === 'undefined') throw new Error('onnxruntime-web did not load');
    // Multi-threading would need cross-origin isolation, which blocks the CDN scripts
    ort.env.wasm.numThreads = 1;
    ort.env.logLevel = 'error';
    const providers = navigator.gpu ? ['webgpu', 'wasm'] : ['wasm'];
    const session = await ort.InferenceSession.create(HSE.url, { executionProviders: providers, logSeverityLevel: 3 });

    // Speed test on this device: one warm-up run, then the average of three
    const n = HSE.size;
    const blank = new ort.Tensor('float32', new Float32Array(3 * n * n), [1, 3, n, n]);
    await session.run({ [session.inputNames[0]]: blank });
    const t0 = performance.now();
    for (let i = 0; i < 3; i++) await session.run({ [session.inputNames[0]]: blank });
    const ms = Math.round((performance.now() - t0) / 3);

    hseSession = session;
    const fast = ms < HSE.fastMs;
    opt.disabled = false;
    opt.textContent = fast ? 'HSEmotion (most accurate)' : 'HSEmotion (most accurate, slow here)';
    els.modelNote.textContent = fast
      ? `HSEmotion ready: ${ms} ms per face on this device.`
      : `HSEmotion takes ${ms} ms per face on this device, so it isn't the default. Pick it for accuracy over speed.`;
    if (fast && !userPickedModel) {
      els.model.value = 'hsemotion';
      onModelChange();
    }
  } catch (err) {
    console.warn('HSEmotion model not loaded:', err);
    opt.textContent = 'HSEmotion (could not load)';
    els.modelNote.textContent = 'The HSEmotion model could not load. The other models still work.';
  }
}

function activeModel() {
  const v = els.model.value;
  if (v === 'hsemotion' && hseSession) return 'hsemotion';
  if (v === 'custom' && customModel) return 'custom';
  return 'builtin';
}

function setDetectorLevel(level) {
  detectorLevel = level;
  detectorOptions = new faceapi.TinyFaceDetectorOptions({ inputSize: DETECTOR_SIZES[level], scoreThreshold: 0.5 });
}

// Every few seconds: use a smaller detector if the frame rate is low, a bigger one if there's headroom.
// A size that proved too slow isn't tried again until the model changes, so it doesn't bounce.
function tuneDetector(now) {
  tuneFrames++;
  if (!tuneFrom) tuneFrom = now;
  if (now - tuneFrom < TUNE_EVERY_MS) return;
  const rate = (tuneFrames * 1000) / (now - tuneFrom);
  if (rate < FPS_LOW && detectorLevel > 0) {
    maxDetectorLevel = detectorLevel - 1;
    setDetectorLevel(detectorLevel - 1);
  } else if (rate > FPS_HIGH && detectorLevel < maxDetectorLevel) {
    setDetectorLevel(detectorLevel + 1);
  }
  tuneFrom = now;
  tuneFrames = 0;
}

function resetDetectorTuning() {
  maxDetectorLevel = DETECTOR_SIZES.length - 1;
  setDetectorLevel(1);
  tuneFrom = 0;
  tuneFrames = 0;
}

function onModelChange() {
  resetTracking();
  resetDetectorTuning();
  for (const k in shown) shown[k] = 0;
}

// ---------- Camera ----------
async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus('This browser cannot use the camera here. Open the site over https:// or on localhost.', true);
    return;
  }
  setStartEnabled(false, 'Starting…');
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
  } catch (err) {
    const msg = err.name === 'NotAllowedError'
      ? 'Camera access is blocked. Allow the camera in your browser’s site settings, then press Start camera again.'
      : err.name === 'NotFoundError'
        ? 'No camera found. Connect a webcam and try again.'
        : `The camera could not start (${err.name}). Close other apps using it and try again.`;
    setStatus(msg, true);
    setStartEnabled(true);
    return;
  }
  els.video.srcObject = stream;
  try {
    await els.video.play();
  } catch (err) {
    console.error(err);
    stream.getTracks().forEach(t => t.stop());
    stream = null;
    els.video.srcObject = null;
    setStatus(`The video could not play (${err.name}). Press Start camera to try again.`, true);
    setStartEnabled(true);
    return;
  }
  sizeCanvas();
  // The big start button disappears now; keep keyboard focus on a visible control
  const focusToggle = document.activeElement === els.startBtn;
  els.stage.classList.add('live');
  els.toggle.disabled = false;
  els.toggle.textContent = 'Stop camera';
  els.snapshot.disabled = false;
  if (focusToggle) els.toggle.focus();
  setStatus('Detecting…');
  fps = 0;
  lastFrameAt = 0;
  tuneFrom = 0;
  tuneFrames = 0;
  running = true;
  loop(++session);
}

function stopCamera() {
  running = false;
  session++;
  stream?.getTracks().forEach(t => t.stop());
  stream = null;
  els.video.srcObject = null;
  resetTracking();
  ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
  els.stage.classList.remove('live');
  els.stage.style.aspectRatio = '';
  if (document.activeElement === els.snapshot) els.toggle.focus();
  els.snapshot.disabled = true;
  setStartEnabled(true);
  els.fps.textContent = '';
  for (const k in shown) shown[k] = 0;
  renderBars(null);
  els.readout.textContent = '—';
  els.readout.style.color = '';
  els.readoutPct.textContent = '';
  els.caption.textContent = 'Current expression';
  setCard(null);
  setStatus('Camera stopped.');
}

function resetTracking() {
  tracks = [];
  lastFaces = [];
  mainTrack = null;
}

// Match the canvas to the camera resolution (it can change, e.g. when a phone rotates)
function sizeCanvas() {
  const { videoWidth: w, videoHeight: h } = els.video;
  if (!w || !h || (els.canvas.width === w && els.canvas.height === h)) return;
  els.canvas.width = w;
  els.canvas.height = h;
  els.stage.style.aspectRatio = `${w} / ${h}`;
}

// ---------- Detection loop ----------
async function loop(id) {
  if (id !== session) return;
  try {
    await detect(id);
  } catch (err) {
    console.error(err);
  }
  if (id === session) requestAnimationFrame(() => loop(id));
}

async function detect(id) {
  const video = els.video;
  if (video.readyState < 2) return;
  const model = activeModel();
  let faces;

  // After each await, bail out if the camera was stopped (or restarted) meanwhile,
  // otherwise a late result would draw over the stopped camera.
  if (model === 'builtin') {
    const results = await faceapi.detectAllFaces(video, detectorOptions).withFaceExpressions();
    if (id !== session) return;
    faces = results.map(r => ({ box: r.detection.box, probs: fromBuiltin(r.expressions) }));
  } else {
    const dets = await faceapi.detectAllFaces(video, detectorOptions);
    if (id !== session) return;
    faces = [];
    for (const d of dets) {
      const probs = model === 'hsemotion' ? await classifyHSEmotion(d.box) : classifyCustom(d.box);
      if (id !== session) return;
      faces.push({ box: d.box, probs });
    }
  }

  const now = performance.now();
  sizeCanvas();
  updateTracks(faces, now);
  tickFps(now);
  tuneDetector(now);
  draw(now);
  updatePanel();
}

function fromBuiltin(expressions) {
  const probs = {};
  for (const [name, p] of Object.entries(expressions)) {
    const key = ALIASES[name];
    if (key) probs[key] = p;
  }
  return probs;
}

// Same preprocessing as train.py: grayscale, 48x48, values 0–1
function classifyCustom(box) {
  const tf = faceapi.tf;
  const vw = els.video.videoWidth, vh = els.video.videoHeight;
  const pad = 0.1;
  const x = Math.max(0, Math.floor(box.x - box.width * pad));
  const y = Math.max(0, Math.floor(box.y - box.height * pad));
  const w = Math.min(vw - x, Math.floor(box.width * (1 + 2 * pad)));
  const h = Math.min(vh - y, Math.floor(box.height * (1 + 2 * pad)));
  if (w < 10 || h < 10) return {};

  const scores = tf.tidy(() => {
    const frame = tf.browser.fromPixels(els.video);
    const face = frame.slice([y, x, 0], [h, w, 3]).toFloat();
    const gray = face.mul(tf.tensor1d([0.2989, 0.587, 0.114])).sum(-1).expandDims(-1);
    const input = tf.image.resizeBilinear(gray, [48, 48]).div(255).expandDims(0);
    return customModel.predict(input).dataSync();
  });

  const probs = {};
  customLabels.forEach((label, i) => { probs[label] = scores[i]; });
  return probs;
}

async function classifyHSEmotion(box) {
  const n = HSE.size, plane = n * n;
  const vw = els.video.videoWidth, vh = els.video.videoHeight;
  const x = Math.max(0, Math.floor(box.x));
  const y = Math.max(0, Math.floor(box.y));
  const w = Math.min(vw - x, Math.floor(box.width));
  const h = Math.min(vh - y, Math.floor(box.height));
  if (w < 10 || h < 10) return {};

  hseCtx.drawImage(els.video, x, y, w, h, 0, 0, n, n);
  const px = hseCtx.getImageData(0, 0, n, n).data; // RGBA
  const input = new Float32Array(3 * plane);
  const [mr, mg, mb] = HSE.mean, [sr, sg, sb] = HSE.std;
  for (let i = 0, p = 0; i < plane; i++, p += 4) {
    input[i] = (px[p] / 255 - mr) / sr;
    input[i + plane] = (px[p + 1] / 255 - mg) / sg;
    input[i + 2 * plane] = (px[p + 2] / 255 - mb) / sb;
  }
  const out = await hseSession.run({ [hseSession.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, n, n]) });
  const logits = out[hseSession.outputNames[0]].data;

  // Softmax over the 7 emotions we use (the model outputs raw scores)
  const used = HSE.labels.map((key, i) => [key, logits[i]]).filter(([key]) => key);
  const max = Math.max(...used.map(([, v]) => v));
  const exps = used.map(([key, v]) => [key, Math.exp(v - max)]);
  const sum = exps.reduce((s, [, v]) => s + v, 0);
  return Object.fromEntries(exps.map(([key, v]) => [key, v / sum]));
}

function topEmotion(probs) {
  let best = 'neutral', bestP = 0;
  for (const [k, p] of Object.entries(probs)) if (p > bestP) { best = k; bestP = p; }
  return { key: best, p: bestP };
}

// Top emotion, flagged uncertain when it's weak or nearly tied with the runner-up
function readEmotion(probs) {
  const [first = ['neutral', 0], second = ['', 0]] = Object.entries(probs).sort((a, b) => b[1] - a[1]);
  const uncertain = first[1] < MIN_CONFIDENCE || first[1] - second[1] < MIN_MARGIN;
  return { key: first[0], p: first[1], uncertain };
}

// ---------- Face tracking + persona stability ----------
function centerOf(box) {
  return { x: box.x + box.width / 2, y: box.y + box.height / 2 };
}

function newTrack(box, probs) {
  const t = {
    probs: Object.fromEntries(EMOTIONS.map(e => [e.key, probs[e.key] || 0])),
    candidate: null, since: 0,           // emotion currently on top, and since when
    stable: null, prev: null, changedAt: 0, // persona shown, the one before it, and when it switched
    cx: 0, cy: 0, lastSeen: 0,
  };
  tracks.push(t);
  return t;
}

function updateTracks(faces, now) {
  // Greedy nearest-centre matching: each face keeps its track (and persona) between frames
  const pairs = [];
  faces.forEach((f, fi) => {
    const c = centerOf(f.box);
    const maxDist = Math.max(f.box.width, f.box.height) * 0.75;
    tracks.forEach((t, ti) => {
      const d = Math.hypot(c.x - t.cx, c.y - t.cy);
      if (d < maxDist) pairs.push({ fi, ti, d });
    });
  });
  pairs.sort((a, b) => a.d - b.d);
  const matched = new Array(faces.length).fill(null);
  const used = new Set();
  for (const { fi, ti } of pairs) {
    if (matched[fi] || used.has(ti)) continue;
    matched[fi] = tracks[ti];
    used.add(ti);
  }

  lastFaces = faces.map((f, fi) => {
    const t = matched[fi] || newTrack(f.box, f.probs);
    updateTrack(t, f.box, f.probs, now);
    return { box: f.box, track: t };
  });
  tracks = tracks.filter(t => now - t.lastSeen < TRACK_TTL_MS);
  mainTrack = lastFaces.length
    ? lastFaces.reduce((a, b) => (b.box.area > a.box.area ? b : a)).track
    : null;
}

function updateTrack(t, box, probs, now) {
  const c = centerOf(box);
  t.cx = c.x;
  t.cy = c.y;
  // Exponential smoothing by elapsed time, so it behaves the same at 8 or 30 fps
  const a = t.lastSeen ? 1 - Math.exp(-(now - t.lastSeen) / SMOOTH_MS) : 1;
  t.lastSeen = now;
  for (const e of EMOTIONS) t.probs[e.key] += a * ((probs[e.key] || 0) - t.probs[e.key]);

  // The persona only switches once the same emotion has been on top, and certain, for STABLE_MS.
  // While uncertain, the current persona stays.
  const r = readEmotion(t.probs);
  const top = r.uncertain ? null : r.key;
  if (top !== t.candidate) {
    t.candidate = top;
    t.since = now;
  }
  if (t.candidate && t.candidate !== t.stable && now - t.since >= STABLE_MS) {
    t.prev = t.stable;
    t.stable = t.candidate;
    t.changedAt = now;
  }
}

// Which persona to draw for a track right now, and how opaque (fade old out, then new in)
function personaFrame(t, now) {
  if (!t.stable) return null;
  if (reducedMotion.matches) return { key: t.stable, alpha: 1 };
  const p = Math.min(1, (now - t.changedAt) / FADE_MS);
  if (!t.prev) return { key: t.stable, alpha: p };
  return p < 0.5 ? { key: t.prev, alpha: 1 - 2 * p } : { key: t.stable, alpha: 2 * p - 1 };
}

function tickFps(now) {
  if (lastFrameAt) {
    const inst = 1000 / Math.max(1, now - lastFrameAt);
    fps = fps ? fps * 0.9 + inst * 0.1 : inst;
  }
  lastFrameAt = now;
  if (fps && now - fpsShownAt > 500) {
    els.fps.textContent = `${Math.round(fps)} fps · ${DETECTOR_SIZES[detectorLevel]}px`;
    fpsShownAt = now;
  }
}

// ---------- Drawing ----------
// The part of the frame visible on screen (object-fit: cover may crop it) in canvas pixels,
// plus k = canvas pixels per CSS pixel, so text keeps the same on-screen size everywhere.
function visibleBounds() {
  const W = els.canvas.width, H = els.canvas.height;
  const sw = els.stage.clientWidth, sh = els.stage.clientHeight;
  const scale = Math.max(sw / W, sh / H);
  if (!scale) return { x: 0, y: 0, w: W, h: H, k: 1 };
  const w = sw / scale, h = sh / scale;
  return { x: (W - w) / 2, y: (H - h) / 2, w, h, k: 1 / scale };
}

function draw(now) {
  ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
  drawFaces(ctx, visibleBounds(), now);
}

function drawFaces(c, b, now) {
  const W = c.canvas.width;
  for (const { box, track } of lastFaces) {
    const r = readEmotion(track.probs);
    const e = r.uncertain ? UNCERTAIN : BY_KEY[r.key] || BY_KEY.neutral;
    const x = W - box.x - box.width; // mirror to match the mirrored video
    const { y, width: w, height: h } = box;

    // Rounded, glowing face box
    c.save();
    c.lineWidth = 3 * b.k;
    c.strokeStyle = e.color;
    c.shadowColor = e.color;
    c.shadowBlur = 18 * b.k;
    roundRect(c, x, y, w, h, Math.min(w, h) * 0.12);
    c.stroke();
    c.restore();

    drawLabel(c, r.uncertain ? e.label : `${e.label} ${Math.round(r.p * 100)}%`, e.color, x, y, h, b);

    const pf = personaFrame(track, now);
    if (pf && pf.alpha > 0) drawPersona(c, BY_KEY[pf.key], pf.alpha, x, y, w, h, b);
  }
}

// "Happy 86%" pill above the box (inside the top of the box if there's no room)
function drawLabel(c, text, color, x, y, h, b) {
  const k = b.k;
  c.font = `600 ${15 * k}px Inter, system-ui, sans-serif`;
  const pw = c.measureText(text).width + 20 * k, ph = 28 * k;
  let py = y - ph - 8 * k;
  if (py < b.y + 4 * k) py = y + 8 * k;
  const px = clamp(x, b.x + 4 * k, b.x + b.w - pw - 4 * k);
  c.fillStyle = color;
  roundRect(c, px, py, pw, ph, ph / 2);
  c.fill();
  c.fillStyle = INK_ON_COLOR;
  c.textBaseline = 'middle';
  c.fillText(text, px + 10 * k, py + ph / 2 + k);
}

// Persona card below the box (inside the bottom of the box if there's no room)
function drawPersona(c, e, alpha, x, y, w, h, b) {
  const k = b.k;
  const pad = 12 * k, titleSize = 16 * k, tagSize = 13 * k, lineGap = 4 * k;
  const titleFont = `700 ${titleSize}px "Bricolage Grotesque", Inter, system-ui, sans-serif`;
  const tagFont = `400 ${tagSize}px Inter, system-ui, sans-serif`;
  const title = `${e.emoji} ${e.persona}`;
  c.font = titleFont;
  const tw = c.measureText(title).width;
  c.font = tagFont;
  const gw = c.measureText(e.tagline).width;

  const pw = Math.max(tw, gw) + pad * 2;
  const ph = pad * 2 + titleSize * 1.2 + lineGap + tagSize * 1.2;
  let py = y + h + 10 * k;
  if (py + ph > b.y + b.h - 4 * k) py = y + h - ph - 10 * k;
  const px = clamp(x + w / 2 - pw / 2, b.x + 4 * k, b.x + b.w - pw - 4 * k);

  c.save();
  c.globalAlpha = alpha;
  c.fillStyle = 'rgba(10, 14, 22, 0.78)';
  roundRect(c, px, py, pw, ph, 14 * k);
  c.fill();
  c.globalAlpha = alpha * 0.7;
  c.strokeStyle = e.color;
  c.lineWidth = 1.5 * k;
  c.stroke();
  c.globalAlpha = alpha;
  c.textBaseline = 'top';
  c.font = titleFont;
  c.fillStyle = '#eef2f8';
  c.fillText(title, px + pad, py + pad);
  c.font = tagFont;
  c.fillStyle = '#b4bfd0';
  c.fillText(e.tagline, px + pad, py + pad + titleSize * 1.2 + lineGap);
  c.restore();
}

function roundRect(c, x, y, w, h, r) {
  r = Math.max(0, Math.min(r, w / 2, h / 2));
  c.beginPath();
  c.moveTo(x + r, y);
  c.arcTo(x + w, y, x + w, y + h, r);
  c.arcTo(x + w, y + h, x, y + h, r);
  c.arcTo(x, y + h, x, y, r);
  c.arcTo(x, y, x + w, y, r);
  c.closePath();
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(v, hi));
}

// ---------- Side panel ----------
function updatePanel() {
  if (mainTrack) {
    Object.assign(shown, mainTrack.probs);
    els.caption.textContent = lastFaces.length > 1 ? `Closest of ${lastFaces.length} faces` : 'Current expression';
  } else {
    els.caption.textContent = 'Looking for a face…';
    for (const k in shown) shown[k] *= 0.85;
  }

  const top = mainTrack ? readEmotion(shown) : null;
  renderBars(top && !top.uncertain ? top.key : null);

  if (top) {
    const e = top.uncertain ? UNCERTAIN : BY_KEY[top.key];
    els.readout.textContent = e.label;
    els.readout.style.color = e.color;
    els.readoutPct.textContent = top.uncertain ? '' : `${Math.round(top.p * 100)}%`;
  }
  if (mainTrack?.stable && mainTrack.stable !== cardKey) setCard(mainTrack.stable);
}

function renderBars(topKey) {
  for (const e of EMOTIONS) {
    const pct = Math.round(shown[e.key] * 100);
    barEls[e.key].fill.style.width = `${pct}%`;
    barEls[e.key].value.textContent = `${pct}%`;
    barEls[e.key].row.classList.toggle('top', e.key === topKey);
  }
}

// Persona card: fades out, swaps content, fades back in. Also sets the page's mood colour.
function setCard(key) {
  cardKey = key;
  const e = key ? BY_KEY[key] : null;
  document.documentElement.style.setProperty('--mood', (e || BY_KEY.neutral).color);
  if (e) els.announce.textContent = `${e.label}: ${e.persona}. ${e.tagline}`;

  const apply = () => {
    els.personaEmoji.textContent = e ? e.emoji : '🙂';
    els.personaName.textContent = e ? e.persona : 'Waiting for a face';
    els.personaTagline.textContent = e ? e.tagline : 'Start the camera to meet your persona.';
    els.persona.classList.remove('swap');
  };
  clearTimeout(cardTimer);
  if (reducedMotion.matches) {
    apply();
    return;
  }
  els.persona.classList.add('swap');
  cardTimer = setTimeout(apply, FADE_MS / 2);
}

// ---------- Snapshot ----------
function takeSnapshot() {
  if (!running) return;
  const W = els.canvas.width, H = els.canvas.height;
  const snap = document.createElement('canvas');
  snap.width = W;
  snap.height = H;
  const c = snap.getContext('2d');

  c.save();
  c.translate(W, 0);
  c.scale(-1, 1); // mirrored, like the preview
  c.drawImage(els.video, 0, 0, W, H);
  c.restore();
  const b = { ...visibleBounds(), x: 0, y: 0, w: W, h: H };
  drawFaces(c, b, performance.now());
  drawWatermark(c, b);

  snap.toBlob(blob => {
    if (!blob) {
      setStatus('The snapshot could not be saved. Try again.', true);
      return;
    }
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = snapshotName();
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    setStatus(`Snapshot saved as ${a.download}`);
  }, 'image/png');

  els.flash.classList.remove('on');
  void els.flash.offsetWidth; // restart the animation
  els.flash.classList.add('on');
}

function drawWatermark(c, b) {
  const k = b.k;
  c.save();
  c.font = `700 ${14 * k}px "Bricolage Grotesque", Inter, system-ui, sans-serif`;
  c.textBaseline = 'bottom';
  c.fillStyle = 'rgba(255, 255, 255, 0.85)';
  c.shadowColor = 'rgba(0, 0, 0, 0.6)';
  c.shadowBlur = 6 * k;
  c.fillText('Face Emotion Persona', 12 * k, c.canvas.height - 10 * k);
  c.restore();
}

function snapshotName() {
  const d = new Date();
  const p = n => String(n).padStart(2, '0');
  return `persona-${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}.png`;
}

// ---------- Wire up ----------
els.startBtn.addEventListener('click', () => { if (!running) startCamera(); });
els.toggle.addEventListener('click', () => (running ? stopCamera() : startCamera()));
els.snapshot.addEventListener('click', takeSnapshot);
els.model.addEventListener('change', () => {
  userPickedModel = true;
  onModelChange();
});
init();
