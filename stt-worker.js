// On-phone transcriber: NVIDIA Parakeet TDT 0.6B v3 (CC-BY-4.0) on raw ONNX Runtime Web.
// Encoder fp16 on WebGPU, decoder+joiner on WASM. Mel front end and TDT greedy loop from
// parakeet.js 1.4.4 (MIT); see LICENSE-parakeet.txt. Audio never leaves the phone here.
// Chosen by tools/stt_bench: matches gpt-4o-transcribe on phone-mic replays (5-17% WER on the S23).
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.1/dist/ort.all.bundle.min.mjs";
import { JsPreprocessor } from "./mel.js";

const Q = new URLSearchParams(self.location.search);
const BASE = Q.get("m") || "https://huggingface.co/striimit/parakeet-tdt-0.6b-v3-webgpu/resolve/main/";
const FILES = { enc: "encoder-model.fp16.onnx", dec: "decoder_joint-model.onnx", vocab: "vocab.txt" };
const SIZES = { enc: 1239245959, dec: 72520893, vocab: 93939 };
const CACHE = "parakeet-v3-fp16";   // not halo-*: the app's service worker never prunes it
const HIDDEN = 640, LAYERS = 2, MAX_PER_FRAME = 10;

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.1/dist/";
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

let encS = null, decS = null, vocab = null, blank = -1, running = false, fresh = false;
const mel = new JsPreprocessor({ nMels: 128 });
const queue = [];
const got = { enc: 0, dec: 0, vocab: 0 };
const total = SIZES.enc + SIZES.dec + SIZES.vocab;

// Model files live in Cache Storage after the first download (1.3 GB, once).
async function fetchCached(key) {
  const url = BASE + FILES[key];
  const cache = await caches.open(CACHE);
  let r = await cache.match(url);
  if (!r) {
    // Hugging Face rate-limits resolves (429 seen once on the S23): back off and retry.
    let res;
    for (let i = 0; ; i++) {
      res = await fetch(url);
      if (res.ok || i >= 4 || ![429, 500, 502, 503, 504].includes(res.status)) break;
      const wait = Math.min(30, +res.headers.get("retry-after") || 2 ** (i + 1));
      postMessage({ kind: "progress", pct: 0, note: `busy (HTTP ${res.status}), retrying in ${wait} s` });
      await new Promise(r => setTimeout(r, wait * 1000));
    }
    if (!res.ok) throw new Error(FILES[key] + " HTTP " + res.status);
    let last = -1;
    const count = new TransformStream({ transform(ch, c) {
      got[key] += ch.byteLength;
      const pct = Math.floor(100 * (got.enc + got.dec + got.vocab) / total);
      if (pct > last) { last = pct; postMessage({ kind: "progress", pct }); }
      c.enqueue(ch);
    } });
    await cache.put(url, new Response(res.body.pipeThrough(count),
      { headers: { "content-type": key === "vocab" ? "text/plain" : "application/octet-stream" } }));
    r = await cache.match(url);
    if (!r) throw new Error("could not store " + FILES[key]);
    fresh = true;
  }
  got[key] = SIZES[key];
  return r;
}

async function load() {
  const t = performance.now();
  try {
    if (!navigator.gpu || !(await navigator.gpu.requestAdapter())) throw new Error("no WebGPU adapter");
    const [rv, rd, re] = [await fetchCached("vocab"), await fetchCached("dec"), await fetchCached("enc")];
    // Loading straight after a 1.3 GB download took 139 s on the S23 (renderer at 3.3 GB) vs 5 s
    // from cache: hand back to the page, which starts a clean worker that loads from cache.
    if (fresh) { postMessage({ kind: "restart" }); self.close(); return; }
    vocab = [];
    for (const line of (await rv.text()).split(/\r?\n/)) {
      const [tok, id] = line.split(/\s+/);
      if (tok && id !== undefined && !isNaN(+id)) vocab[+id] = tok.replace(/\u2581/g, " ");
    }
    blank = vocab.findIndex(x => x === "<blk>");
    if (blank < 0) throw new Error("vocab has no <blk>");
    postMessage({ kind: "progress", pct: 100, note: "loading decoder" });
    decS = await ort.InferenceSession.create(new Uint8Array(await rd.arrayBuffer()),
      { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    // Hand ORT a disk-backed blob URL (the path the encoder bench loaded in 14 s) so this
    // worker never holds its own reference to the 1.24 GB buffer.
    postMessage({ kind: "progress", pct: 100, note: "loading encoder" });
    const encUrl = URL.createObjectURL(await re.blob());
    try { encS = await ort.InferenceSession.create(encUrl, { executionProviders: ["webgpu"], graphOptimizationLevel: "all" }); }
    finally { URL.revokeObjectURL(encUrl); }
    const tLoad = (performance.now() - t) / 1000;
    postMessage({ kind: "progress", pct: 100, note: "warming up" });
    const warm = new Float32Array(16000 * 2);
    for (let i = 0; i < warm.length; i++) warm[i] = (Math.random() - 0.5) * 0.01;
    await transcribe(warm);   // compile the GPU kernels before the first real piece
    postMessage({ kind: "ready", sec: +((performance.now() - t) / 1000).toFixed(1), load: +tLoad.toFixed(1),
                  threads: ort.env.wasm.numThreads });
    pump();
  } catch (e) {
    postMessage({ kind: "fail", error: String(e && e.message || e) });
  }
}

const zeros = () => new ort.Tensor("float32", new Float32Array(LAYERS * HIDDEN), [LAYERS, 1, HIDDEN]);
const tokArr = new Int32Array(1);
const tokT = new ort.Tensor("int32", tokArr, [1, 1]);
const lenT = new ort.Tensor("int32", new Int32Array([1]), [1]);

// One piece of 16 kHz audio -> text. Returns {text, enc, dec} with stage seconds.
// skip: leading seconds that are context only (already transcribed as the previous piece).
// until: seconds from the start after which audio is right context only (0 = keep to the end).
// Parakeet on short pieces lost 2-5x the words it does on 30 s windows (tools/device_app_stt.py),
// so each piece is encoded with left context and only its own tokens are kept.
const FRAME_S = 0.08;   // 8x subsampling of 10 ms mel frames
async function transcribe(pcm, skip = 0, until = 0) {
  const t0 = performance.now();
  const { features, length: T } = mel.process(pcm);
  if (T < 16) return { text: "", enc: 0, dec: 0 };
  const eo = await encS.run({
    audio_signal: new ort.Tensor("float32", features, [1, 128, T]),
    length: new ort.Tensor("int64", BigInt64Array.from([BigInt(T)]), [1]),
  });
  const out = eo.outputs ?? Object.values(eo)[0];
  const [, D, Tall] = out.dims;
  let Tenc = Tall;
  for (const [k, v] of Object.entries(eo)) if (v !== out && /len/i.test(k)) Tenc = Math.min(Tall, Number(v.data[0]));
  const src = out.data, frames = new Float32Array(Tenc * D);
  for (let t = 0; t < Tenc; t++) for (let d = 0; d < D; d++) frames[t * D + d] = src[d * Tall + t];
  const t1 = performance.now();

  // TDT greedy decode (parakeet.js / onnx-asr): the LSTM state only advances on a real token;
  // the duration head says how many encoder frames to skip.
  const fbuf = new Float32Array(D), fT = new ort.Tensor("float32", fbuf, [1, D, 1]);
  let s1 = zeros(), s2 = zeros(), last = blank, emitted = 0;
  const ids = [], at = [];
  for (let t = 0; t < Tenc;) {
    fbuf.set(frames.subarray(t * D, (t + 1) * D));
    tokArr[0] = last;
    const o = await decS.run({ encoder_outputs: fT, targets: tokT, target_length: lenT,
                               input_states_1: s1, input_states_2: s2 });
    const lg = o.outputs.data, V = vocab.length;
    let best = 0, bv = -Infinity;
    for (let i = 0; i < V; i++) if (lg[i] > bv) { bv = lg[i]; best = i; }
    let step = 0, sv = -Infinity;
    for (let i = V; i < lg.length; i++) if (lg[i] > sv) { sv = lg[i]; step = i - V; }
    if (best !== blank) {
      s1 = o.output_states_1; s2 = o.output_states_2;
      ids.push(best); at.push(t); last = best; emitted++;
    }
    if (step > 0) { t += step; emitted = 0; }
    else if (best === blank || emitted >= MAX_PER_FRAME) { t++; emitted = 0; }
  }
  // A word belongs to the piece its first token falls in: keep [skip, until) of the audio,
  // one frame of slack for timestamp jitter, the same rule on both edges so joins neither
  // drop nor repeat a word.
  const wordAt = (sec) => {
    const f = sec / FRAME_S - 1;
    let k = 0;
    while (k < ids.length && !(at[k] >= f && (vocab[ids[k]] || "").startsWith(" "))) k++;
    return k;
  };
  const k = skip > 0 ? wordAt(skip) : 0;
  const k2 = until > 0 ? wordAt(until) : ids.length;
  const text = ids.slice(k, Math.max(k, k2)).map(i => vocab[i] || "").join("")
    .replace(/^\s+/, "").replace(/\s+(?=[^\w\s])/g, "").replace(/\s+/g, " ").trim();
  return { text, enc: (t1 - t0) / 1000, dec: (performance.now() - t1) / 1000 };
}

async function pump() {
  if (running || !encS) return;
  running = true;
  while (queue.length) {
    const job = queue.shift();
    const t = performance.now();
    let r;
    try { r = await transcribe(job.pcm, job.skip || 0, job.until || 0); }
    catch (e) { postMessage({ kind: "segerr", id: job.id, error: String(e && e.message || e) }); continue; }
    postMessage({ kind: "seg", id: job.id, start: job.start, end: job.end, text: r.text,
                  sec: +((performance.now() - t) / 1000).toFixed(3), enc: +r.enc.toFixed(3), dec: +r.dec.toFixed(3),
                  audio: +(job.pcm.length / 16000).toFixed(2), skip: job.skip });
  }
  running = false;
}

onmessage = e => {
  const m = e.data;
  if (m.kind === "seg") {
    const job = { id: m.id, start: m.start, end: m.end, pcm: m.pcm, skip: m.skip || 0, until: m.until || 0 };
    if (m.urgent) queue.unshift(job); else queue.push(job);
    pump();
  } else if (m.kind === "drop") {       // commit jobs a tap job supersedes; the in-flight one still finishes
    const ids = new Set(m.ids), gone = queue.filter(j => ids.has(j.id)).map(j => j.id);
    for (let i = queue.length - 1; i >= 0; i--) if (ids.has(queue[i].id)) queue.splice(i, 1);
    postMessage({ kind: "dropped", ids: gone });
  } else if (m.kind === "clear") {
    queue.length = 0;
  }
};

load();
