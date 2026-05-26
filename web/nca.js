// In-browser inference loop for the Growing Neural Cellular Automata model.
//
// Each frame:
//   1. Build an ort.Tensor view over our persistent state buffer.
//   2. Run the ONNX model -> delta update of the same shape.
//   3. Apply alive mask (3x3 max-pool over alpha > threshold) and a stochastic
//      fire mask, then state += update * alive * fire.
//   4. Composite RGBA channels onto a 2D canvas at native grid resolution
//      (CSS upscales with pixelated rendering).
//
// The ONNX model itself is just two conv layers — no random ops, no control
// flow — so it is compatible with onnxruntime-web's WebGL execution provider.

const MODEL_URL = "./nca.onnx";
const META_URL = "./nca.json";

const ui = {
  canvas: document.getElementById("cv"),
  play: document.getElementById("playBtn"),
  reset: document.getElementById("resetBtn"),
  step: document.getElementById("stepBtn"),
  backend: document.getElementById("backend"),
  speed: document.getElementById("speed"),
  speedVal: document.getElementById("speedVal"),
  fire: document.getElementById("fire"),
  fireVal: document.getElementById("fireVal"),
  stats: document.getElementById("stats"),
  badge: document.getElementById("backendBadge"),
};

const state = {
  meta: null,
  session: null,
  activeBackend: null,
  buffer: null,        // Float32Array, length = 1*H*W*C
  inputTensor: null,
  inputName: null,
  outputName: null,
  step: 0,
  fps: 0,
  fpsAcc: 0,
  fpsFrames: 0,
  fpsLast: performance.now(),
  playing: true,
  fireRate: 0.5,
  stepsPerFrame: 1,
};

function seed(buf, meta) {
  buf.fill(0);
  const cy = (meta.height / 2) | 0;
  const cx = (meta.width / 2) | 0;
  const base = (cy * meta.width + cx) * meta.channels;
  for (let c = 3; c < meta.channels; c++) buf[base + c] = 1.0;
}

async function loadMeta() {
  const r = await fetch(META_URL);
  if (!r.ok) throw new Error(`failed to fetch ${META_URL}: ${r.status}`);
  return r.json();
}

async function createSession(backend) {
  // Try the requested execution provider, fall back to WASM if WebGL is
  // unavailable (e.g. headless browsers, no GPU).
  const providers = backend === "webgl" ? ["webgl", "wasm"] : ["wasm"];
  try {
    const session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: providers,
      graphOptimizationLevel: "all",
    });
    const used = providers[0];
    return { session, used };
  } catch (err) {
    if (backend === "webgl") {
      console.warn("WebGL backend failed, falling back to WASM:", err);
      const session = await ort.InferenceSession.create(MODEL_URL, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      return { session, used: "wasm" };
    }
    throw err;
  }
}

async function init(backend = "webgl") {
  state.meta = state.meta || await loadMeta();
  const { meta } = state;
  ui.canvas.width = meta.width;
  ui.canvas.height = meta.height;
  const { session, used } = await createSession(backend);
  state.session = session;
  state.activeBackend = used;
  state.inputName = session.inputNames[0];
  state.outputName = session.outputNames[0];

  ui.badge.textContent = used.toUpperCase();
  ui.badge.className = "badge " + (used === "webgl" ? "ok" : "warn");

  state.buffer = new Float32Array(1 * meta.height * meta.width * meta.channels);
  state.inputTensor = new ort.Tensor("float32", state.buffer,
    [1, meta.height, meta.width, meta.channels]);
  seed(state.buffer, meta);
  state.step = 0;
}

// 3x3 max-pool on the alpha channel, comparing each pooled value against
// `alive_threshold`. Operates on the flat NHWC buffer in-place — no allocation.
const aliveScratch = { arr: null };
function computeAliveMask(buf, meta) {
  const { height: H, width: W, channels: C, alive_threshold: T, alpha_channel: A } = meta;
  if (!aliveScratch.arr || aliveScratch.arr.length !== H * W) {
    aliveScratch.arr = new Uint8Array(H * W);
  }
  const alive = aliveScratch.arr;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let m = 0;
      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= H) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= W) continue;
          const v = buf[(yy * W + xx) * C + A];
          if (v > m) m = v;
        }
      }
      alive[y * W + x] = m > T ? 1 : 0;
    }
  }
  return alive;
}

async function runStep() {
  const { meta, session, buffer, inputTensor, inputName, outputName, fireRate } = state;
  const results = await session.run({ [inputName]: inputTensor });
  const update = results[outputName].data;
  const alive = computeAliveMask(buffer, meta);
  const { height: H, width: W, channels: C } = meta;
  for (let i = 0; i < H * W; i++) {
    const a = alive[i];
    if (!a) continue;
    if (Math.random() >= fireRate) continue;
    const base = i * C;
    for (let c = 0; c < C; c++) buffer[base + c] += update[base + c];
  }
  state.step++;
}

function renderToCanvas() {
  const { buffer, meta } = state;
  const { height: H, width: W, channels: C, alpha_channel: A, rgb_channels: RGB } = meta;
  const ctx = ui.canvas.getContext("2d");
  const img = ctx.createImageData(W, H);
  for (let i = 0; i < H * W; i++) {
    const base = i * C;
    const r = Math.max(0, Math.min(1, buffer[base + RGB[0]]));
    const g = Math.max(0, Math.min(1, buffer[base + RGB[1]]));
    const b = Math.max(0, Math.min(1, buffer[base + RGB[2]]));
    const a = Math.max(0, Math.min(1, buffer[base + A]));
    img.data[i * 4 + 0] = (r * 255) | 0;
    img.data[i * 4 + 1] = (g * 255) | 0;
    img.data[i * 4 + 2] = (b * 255) | 0;
    img.data[i * 4 + 3] = (a * 255) | 0;
  }
  ctx.putImageData(img, 0, 0);
}

function countAlive() {
  const { buffer, meta } = state;
  const { height: H, width: W, channels: C, alpha_channel: A } = meta;
  let n = 0;
  for (let i = 0; i < H * W; i++) {
    if (buffer[i * C + A] > 0.1) n++;
  }
  return n;
}

function updateStats() {
  const now = performance.now();
  state.fpsFrames++;
  state.fpsAcc += now - state.fpsLast;
  state.fpsLast = now;
  if (state.fpsAcc > 500) {
    state.fps = (1000 * state.fpsFrames / state.fpsAcc) | 0;
    state.fpsAcc = 0;
    state.fpsFrames = 0;
  }
  ui.stats.innerHTML = `
    backend: <b>${state.activeBackend}</b><br>
    model:   <b>${state.meta.height}×${state.meta.width}×${state.meta.channels}</b><br>
    step:    <b>${state.step}</b><br>
    fps:     <b>${state.fps}</b><br>
    alive:   <b>${countAlive()}</b>
  `;
}

async function frame() {
  if (state.playing) {
    for (let i = 0; i < state.stepsPerFrame; i++) await runStep();
  }
  renderToCanvas();
  updateStats();
  requestAnimationFrame(frame);
}

// --- Damage interaction ----------------------------------------------------
// Clicking the canvas zeros out a small disc — replicates the regeneration
// experiment from the training notebook (cells should heal themselves).
function damageAt(clientX, clientY) {
  const rect = ui.canvas.getBoundingClientRect();
  const x = ((clientX - rect.left) / rect.width * state.meta.width) | 0;
  const y = ((clientY - rect.top) / rect.height * state.meta.height) | 0;
  const r = Math.max(3, ((Math.min(state.meta.width, state.meta.height) * 0.18) | 0));
  const { buffer, meta } = state;
  const { height: H, width: W, channels: C } = meta;
  for (let yy = Math.max(0, y - r); yy < Math.min(H, y + r); yy++) {
    for (let xx = Math.max(0, x - r); xx < Math.min(W, x + r); xx++) {
      if ((xx - x) ** 2 + (yy - y) ** 2 <= r * r) {
        const base = (yy * W + xx) * C;
        for (let c = 0; c < C; c++) buffer[base + c] = 0;
      }
    }
  }
}

ui.canvas.addEventListener("click", (e) => damageAt(e.clientX, e.clientY));

ui.play.addEventListener("click", () => {
  state.playing = !state.playing;
  ui.play.textContent = state.playing ? "Pause" : "Play";
});
ui.reset.addEventListener("click", () => {
  seed(state.buffer, state.meta);
  state.step = 0;
});
ui.step.addEventListener("click", async () => {
  const wasPlaying = state.playing;
  state.playing = false;
  await runStep();
  ui.play.textContent = "Play";
  state.playing = false;
  if (wasPlaying) {} // intentionally stay paused so user can step repeatedly
});
ui.backend.addEventListener("change", async () => {
  ui.badge.textContent = "switching…";
  ui.badge.className = "badge";
  await init(ui.backend.value);
});
ui.speed.addEventListener("input", () => {
  state.stepsPerFrame = parseInt(ui.speed.value, 10);
  ui.speedVal.textContent = state.stepsPerFrame;
});
ui.fire.addEventListener("input", () => {
  state.fireRate = parseFloat(ui.fire.value);
  ui.fireVal.textContent = state.fireRate.toFixed(2);
});

(async () => {
  try {
    await init("webgl");
    requestAnimationFrame(frame);
  } catch (err) {
    console.error(err);
    ui.badge.textContent = "error";
    ui.badge.className = "badge warn";
    ui.stats.innerHTML = `<b style="color:#fca5a5">${err.message}</b>`;
  }
})();
