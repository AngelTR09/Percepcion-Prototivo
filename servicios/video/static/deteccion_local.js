// Deteccion de VIDEO ejecutada EN EL NAVEGADOR (onnxruntime-web), con el mismo modelo best.pt exportado a ONNX.
// Asi los recuadros siguen la imagen en vivo con el procesador de quien mira, sin depender de la CPU del servidor
// (en un plan gratuito de Render cada fotograma tardaba ~3 s). Replica la logica de modulos/deteccion.py de VIDEO:
//   conf >= 0.15 y NMS (IoU 0.45) por clase, y luego el umbral propio de cada clase.
(function (global) {
  "use strict";
  let lado = 416;                  // lado de entrada: 416 rapida, 512 equilibrada, 640 precisa (el modelo acepta cualquier multiplo de 32)
  const CONF_MIN = 0.15, IOU = 0.45;
  const CLASES = ["persona", "mochila", "telefono", "cuaderno", "libro", "audifonos", "reloj", "laptop"];   // orden de best.pt
  const UMBRAL = { persona: 0.20, mochila: 0.15, telefono: 0.15, audifonos: 0.15, reloj: 0.15, cuaderno: 0.80, libro: 0.80, laptop: 0.25 };
  const NOMBRES = { persona: "Persona", mochila: "Mochila", telefono: "Telefono", audifonos: "Audifonos", reloj: "Reloj", cuaderno: "Cuaderno/Libro", libro: "Cuaderno/Libro", laptop: "Laptop" };
  const PERMITIDOS = new Set(["persona", "mochila"]);
  const ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/";

  let sesion = null, cargando = null, lienzo = null, ctx = null;

  // ---- partes puras (se prueban en Node) ----

  // RGBA 640x640 (con el letterbox ya aplicado) -> tensor CHW float32 en [0, 1]
  function aTensor(rgba, n0) {
    const L = n0 || lado, n = L * L, out = new Float32Array(3 * n);
    for (let i = 0, j = 0; i < n; i++, j += 4) {
      out[i] = rgba[j] / 255; out[n + i] = rgba[j + 1] / 255; out[2 * n + i] = rgba[j + 2] / 255;
    }
    return out;
  }

  function iou(a, b) {
    const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1), x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
    return union > 0 ? inter / union : 0;
  }

  // salida: Float32Array [1, 4 + 8, 8400] (cx, cy, w, h y puntaje de cada clase). `g` = geometria del letterbox.
  function posprocesar(salida, anclas, g) {
    const nc = CLASES.length, cand = [];
    for (let i = 0; i < anclas; i++) {
      let mejor = 0, c = -1;
      for (let k = 0; k < nc; k++) { const p = salida[(4 + k) * anclas + i]; if (p > mejor) { mejor = p; c = k; } }
      if (mejor < CONF_MIN) continue;
      const cx = salida[i], cy = salida[anclas + i], w = salida[2 * anclas + i], h = salida[3 * anclas + i];
      cand.push({ c, conf: mejor, x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2 });
    }
    cand.sort((a, b) => b.conf - a.conf);
    const elegidos = [];
    for (const d of cand) {                                   // NMS por clase
      if (elegidos.every((e) => e.c !== d.c || iou(e, d) <= IOU)) elegidos.push(d);
    }
    const objetos = [];
    for (const d of elegidos) {
      const clase = CLASES[d.c];
      if (d.conf < (UMBRAL[clase] ?? 0.25)) continue;         // umbral propio de cada clase (VIDEO)
      const cl = (v, max) => Math.max(0, Math.min(max, v));
      objetos.push({
        clase, etiqueta: NOMBRES[clase], confianza: Math.round(d.conf * 10000) / 10000, permitido: PERMITIDOS.has(clase),
        caja: { x1: Math.round(cl((d.x1 - g.ox) / g.r, g.ancho)), y1: Math.round(cl((d.y1 - g.oy) / g.r, g.alto)),
                x2: Math.round(cl((d.x2 - g.ox) / g.r, g.ancho)), y2: Math.round(cl((d.y2 - g.oy) / g.r, g.alto)) },
      });
    }
    return objetos;
  }

  function geometria(ancho, alto, n0) {                      // letterbox: escala y margenes
    const L = n0 || lado, r = Math.min(L / ancho, L / alto), nw = Math.round(ancho * r), nh = Math.round(alto * r);
    return { r, nw, nh, ox: Math.floor((L - nw) / 2), oy: Math.floor((L - nh) / 2), ancho, alto };
  }

  function fijarLado(n) {                                    // 416 | 512 | 640
    lado = n;
    if (lienzo) { lienzo.width = lienzo.height = lado; }
  }

  // ---- parte del navegador ----

  function cargarScript(src) {
    return new Promise((ok, mal) => { const s = document.createElement("script"); s.src = src; s.onload = ok; s.onerror = () => mal(new Error("no se pudo cargar " + src)); document.head.append(s); });
  }

  // Descarga el motor y el modelo (~12 MB, un solo archivo para los tres tamanos). Devuelve true si quedo listo; si falla, el llamador usa el servidor.
  function cargar() {
    if (sesion) return Promise.resolve(true);
    if (!cargando) {
      cargando = (async () => {
        if (typeof WebAssembly !== "object") return false;
        await cargarScript(ORT + "ort.min.js");
        global.ort.env.wasm.wasmPaths = ORT;
        global.ort.env.wasm.numThreads = 1;                   // sin aislamiento entre origenes no hay hilos
        sesion = await global.ort.InferenceSession.create("/modelo/best.onnx", { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
        lienzo = document.createElement("canvas"); lienzo.width = lienzo.height = lado;
        ctx = lienzo.getContext("2d", { willReadFrequently: true });
        return true;
      })().catch(() => { cargando = null; return false; });
    }
    return cargando;
  }

  // `fuente`: <video>. Devuelve {ancho, alto, latencia_ms, objetos} con cajas en pixeles del video.
  async function detectar(fuente) {
    const inicio = performance.now();
    const ancho = fuente.videoWidth, alto = fuente.videoHeight, g = geometria(ancho, alto);
    ctx.fillStyle = "rgb(114,114,114)"; ctx.fillRect(0, 0, lado, lado);
    ctx.drawImage(fuente, 0, 0, ancho, alto, g.ox, g.oy, g.nw, g.nh);
    const tensor = new global.ort.Tensor("float32", aTensor(ctx.getImageData(0, 0, lado, lado).data), [1, 3, lado, lado]);
    const res = await sesion.run({ images: tensor });
    const salida = res.output0;
    return { ancho, alto, latencia_ms: Math.round(performance.now() - inicio), objetos: posprocesar(salida.data, salida.dims[2], g) };
  }

  const api = { cargar, detectar, fijarLado, aTensor, posprocesar, geometria, get lado() { return lado; } };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else global.DeteccionLocal = api;
})(typeof window !== "undefined" ? window : globalThis);
