// Network-first app shell, cached for offline launch. API calls and the model download are cross-origin and pass through.
// Only halo-* caches are ours to prune; parakeet-v3-fp16 holds the on-phone model (1.3 GB) and must survive updates.
const C = "halo-v6", SHELL = ["./", "index.html", "manifest.json", "icon-192.png", "icon-512.png", "webm-muxer.min.js", "stt-worker.js", "mel.js"];
self.addEventListener("install", e => { self.skipWaiting(); e.waitUntil(caches.open(C).then(c => c.addAll(SHELL))); });
self.addEventListener("activate", e => e.waitUntil(
  caches.keys().then(ks => Promise.all(ks.filter(k => k.startsWith("halo-") && k !== C).map(k => caches.delete(k)))).then(() => self.clients.claim())));
self.addEventListener("fetch", e => {
  const r = e.request;
  if (r.method !== "GET" || new URL(r.url).origin !== location.origin) return;
  e.respondWith(fetch(r).then(res => {
    const copy = res.clone(); caches.open(C).then(c => c.put(r, copy)); return res;
  }).catch(() => caches.match(r, { ignoreSearch: true }).then(x => x || caches.match("./"))));
});
