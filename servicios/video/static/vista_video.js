// Interfaz del servicio VIDEO: camara en vivo con la deteccion del grupo VIDEO.
// Dibuja lo que dibuja VIDEO (azul = permitido, rojo = no permitido), lleva su contador de alerta (2,5 s)
// y genera la alerta con captura y sonido. Los hallazgos se entregan al servicio IMAGEN (via /api/hallazgo),
// que tiene la ultima palabra: si no los verifica, son falsos positivos y no llegan a PREDICCION.
(function () {
  "use strict";
  const TIEMPO_ALERTA = 2.5;   // segundos: la mitad de los 5 s de VIDEO
  // Todo lo que se dibuja en rojo (no permitido) dispara alerta. El principal.py original de VIDEO omitia "laptop" aunque la dibuja en rojo.
  const ALERTA = new Set(["telefono", "audifonos", "reloj", "cuaderno", "libro", "laptop"]);
  const ANCHO_ENVIO = 640;
  const PAUSA_MS = 100;
  const ENTREGA_CADA_MS = 3000;
  const TOLERANCIA_MS = 1000;   // un parpadeo del detector no reinicia el contador

  const $ = (id) => document.getElementById(id);
  let flujo = null, ultimas = null, corriendo = false, entregando = false, modoLocal = false, fps = 0;
  let inicio = {}, ultimaVista = {}, alertada = {}, ultimaEntrega = 0;
  const lienzoEnvio = document.createElement("canvas");

  const umbral = () => (Number($("umbral-video")?.value) || 50) / 100;

  function estado(texto, vivo) {
    $("v-estado-texto").textContent = texto;
    $("v-estado").className = "chip" + (vivo ? " vivo" : "");
  }

  async function iniciar() {
    if (flujo) return;
    try {
      flujo = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
    } catch (e) { estado("No se pudo acceder a la cámara", false); return; }
    const v = $("v-video");
    v.srcObject = flujo; v.classList.add("active");
    $("v-placeholder").style.display = "none";
    $("v-iniciar").hidden = true; $("v-detener").hidden = false;
    estado("Detectando…", true);
    corriendo = true; ciclo();
    activarLocal();
  }

  // Detector en el navegador: los recuadros siguen la imagen en vivo sin depender de la CPU del servidor.
  function modo(texto, ok) { const c = $("v-modo"); c.textContent = texto; c.className = "chip " + (ok ? "ok" : "aviso"); }

  async function activarLocal() {
    if (modoLocal || typeof DeteccionLocal === "undefined") return;
    modo("⏳ Cargando detector del navegador (12 MB)…", false);
    DeteccionLocal.fijarLado(Number($("calidad-video")?.value) || 416);
    modoLocal = await DeteccionLocal.cargar();
    modo(modoLocal ? "⚡ Detección en tu navegador" : "🖥️ Detección en el servidor (el navegador no pudo cargar el detector)", modoLocal);
  }

  function detener() {
    corriendo = false;
    if (flujo) flujo.getTracks().forEach((t) => t.stop());
    flujo = null; ultimas = null; inicio = {}; ultimaVista = {}; alertada = {}; fps = 0;
    const v = $("v-video"); v.srcObject = null; v.classList.remove("active");
    $("v-placeholder").style.display = ""; $("v-iniciar").hidden = false; $("v-detener").hidden = true;
    estado("Cámara detenida", false); dibujar(); lista();
  }

  // ---- deteccion ----
  async function ciclo() {
    if (!corriendo) return;
    const v = $("v-video");
    let espera = PAUSA_MS;
    if (modoLocal && v.readyState >= 2 && v.videoWidth > 0 && !v.paused) {
      try {
        ultimas = await DeteccionLocal.detectar(v);
        fps = fps ? fps * 0.8 + (1000 / Math.max(1, ultimas.latencia_ms)) * 0.2 : 1000 / Math.max(1, ultimas.latencia_ms);
        $("v-modo").textContent = `⚡ Detección en tu navegador · ${fps.toFixed(1)} fps`;
        if (corriendo) { estado("Detectando…", true); procesar(); }
      } catch (e) { modoLocal = false; modo("🖥️ Detección en el servidor (falló la del navegador)", false); }
      if (corriendo) { dibujar(); setTimeout(ciclo, 0); }
      return;
    }
    if (v.readyState >= 2 && v.videoWidth > 0 && !v.paused) {
      try {
        const ancho = Math.min(ANCHO_ENVIO, v.videoWidth);
        lienzoEnvio.width = ancho; lienzoEnvio.height = Math.round(v.videoHeight * ancho / v.videoWidth);
        lienzoEnvio.getContext("2d").drawImage(v, 0, 0, lienzoEnvio.width, lienzoEnvio.height);
        const blob = await new Promise((ok) => lienzoEnvio.toBlob(ok, "image/jpeg", 0.7));
        const r = await fetch("/api/detectar", { method: "POST", headers: { "Content-Type": "image/jpeg" }, body: blob });
        if (r.status === 503) { ultimas = null; espera = 3000; estado("Detector de VIDEO no disponible", false); }
        else if (r.ok) { ultimas = await r.json(); if (corriendo) { estado("Detectando…", true); procesar(); } }
        else espera = 1500;
      } catch (e) { espera = 2000; }
    }
    if (corriendo) { dibujar(); setTimeout(ciclo, espera); }
  }

  // ---- contador de alerta (logica de principal.py de VIDEO) ----
  function procesar() {
    const ahora = performance.now();
    const presentes = new Set();
    for (const o of ultimas.objetos) {
      if (!ALERTA.has(o.clase)) continue;
      presentes.add(o.clase);
      ultimaVista[o.clase] = ahora;
      if (!(o.clase in inicio)) { inicio[o.clase] = ahora; alertada[o.clase] = false; }
      if ((ahora - inicio[o.clase]) / 1000 >= TIEMPO_ALERTA && !alertada[o.clase]) { alertada[o.clase] = true; nuevaAlerta(o); }
    }
    for (const c of Object.keys(inicio)) {
      if (!presentes.has(c) && ahora - (ultimaVista[c] || 0) > TOLERANCIA_MS) { delete inicio[c]; delete alertada[c]; }
    }
    lista();
    entregarAImagen();
  }

  function nuevaAlerta(o) {
    const v = $("v-video");
    const lienzo = document.createElement("canvas");
    lienzo.width = Math.min(480, v.videoWidth); lienzo.height = Math.round(v.videoHeight * lienzo.width / v.videoWidth);
    lienzo.getContext("2d").drawImage(v, 0, 0, lienzo.width, lienzo.height);
    const fig = document.createElement("figure"); fig.dataset.clase = o.clase;
    const img = document.createElement("img"); img.src = lienzo.toDataURL("image/jpeg", 0.8); img.alt = "Captura de la alerta";
    const cap = document.createElement("figcaption");
    cap.textContent = `${new Date().toLocaleTimeString("es")} · ${o.etiqueta} ${o.confianza.toFixed(2)}`;
    fig.append(img, cap);
    $("v-alertas").prepend(fig); $("v-alertas-vacio").hidden = true;
    try { new Audio("/api/sonido-alerta").play().catch(() => {}); } catch (e) { /* sin sonido */ }
  }

  function lista() {
    const ul = $("v-lista"); ul.replaceChildren();
    const objetos = ultimas ? ultimas.objetos : [];
    if (!objetos.length) { const li = document.createElement("li"); li.textContent = "Sin detecciones."; ul.append(li); return; }
    for (const o of objetos) {
      const li = document.createElement("li");
      li.textContent = `${o.etiqueta} ${o.confianza.toFixed(2)} · ${o.permitido ? "permitido" : "NO permitido"}`;
      li.className = o.permitido ? "v-ok" : "v-mal"; ul.append(li);
    }
  }

  // ---- entrega a IMAGEN ----
  // Del MISMO fotograma se toman el objeto prohibido (para IMAGEN) y la persona (para el analisis facial).
  function instantanea() {
    const v = $("v-video");
    const lienzo = document.createElement("canvas");
    lienzo.width = v.videoWidth; lienzo.height = v.videoHeight;
    lienzo.getContext("2d").drawImage(v, 0, 0);
    return lienzo;
  }

  function recortar(foto, o, margen) {
    const k = foto.width / ultimas.ancho;
    const mx = (o.caja.x2 - o.caja.x1) * margen, my = (o.caja.y2 - o.caja.y1) * margen;
    const x1 = Math.max(0, (o.caja.x1 - mx) * k), y1 = Math.max(0, (o.caja.y1 - my) * k);
    const x2 = Math.min(foto.width, (o.caja.x2 + mx) * k), y2 = Math.min(foto.height, (o.caja.y2 + my) * k);
    const lienzo = document.createElement("canvas");
    lienzo.width = Math.max(1, Math.round(x2 - x1)); lienzo.height = Math.max(1, Math.round(y2 - y1));
    lienzo.getContext("2d").drawImage(foto, x1, y1, x2 - x1, y2 - y1, 0, 0, lienzo.width, lienzo.height);
    return lienzo;
  }

  // Reduce un lienzo a un ancho maximo (el analisis en un servidor lento cuesta segun los pixeles).
  function limitar(lienzo, anchoMax) {
    if (lienzo.width <= anchoMax) return lienzo;
    const r = document.createElement("canvas");
    r.width = anchoMax; r.height = Math.round(lienzo.height * anchoMax / lienzo.width);
    r.getContext("2d").drawImage(lienzo, 0, 0, r.width, r.height);
    return r;
  }

  async function entregarAImagen() {
    if (entregando || !ultimas || performance.now() - ultimaEntrega < ENTREGA_CADA_MS) return;
    const c = ultimas.objetos.filter((o) => ALERTA.has(o.clase) && o.confianza >= umbral())
      .sort((a, b) => b.confianza - a.confianza)[0];
    if (!c) return;
    entregando = true; ultimaEntrega = performance.now();
    const foto = instantanea();
    const p = ultimas.objetos.filter((o) => o.clase === "persona").sort((a, b) => b.confianza - a.confianza)[0];
    const lienzoPersona = limitar(p ? recortar(foto, p, 0.25) : foto, 720);   // sin persona detectada: el fotograma completo
    const cuerpo = {
      clase: c.clase, etiqueta: c.etiqueta, confianza: c.confianza, umbral: umbral(),
      objeto_b64: recortar(foto, c, 0.12).toDataURL("image/jpeg", 0.92),
      persona: {
        dataUrl: lienzoPersona.toDataURL("image/jpeg", 0.88), detectada: Boolean(p), confianza: p ? p.confianza : null,
        completa: limitar(foto, 800).toDataURL("image/jpeg", 0.85),   // respaldo si el recorte no incluye el rostro
      },
    };
    $("v-imagen-estado").className = "chip"; $("v-imagen-estado").textContent = "🖼️ IMAGEN: verificando…";
    try {
      const r = await fetch("/api/hallazgo", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(cuerpo) });
      if (!r.ok) throw new Error("HTTP " + r.status);
      marcar(c, await r.json());
    } catch (e) {
      $("v-imagen-estado").className = "chip mal"; $("v-imagen-estado").textContent = "🖼️ IMAGEN: no disponible";
    } finally { entregando = false; }
  }

  // IMAGEN tiene la ultima palabra: la alerta queda marcada como confirmada o como falso positivo.
  function marcar(c, r) {
    const chip = $("v-imagen-estado");
    chip.className = "chip " + (r.pasa_a_prediccion ? "ok" : "mal");
    chip.textContent = `🖼️ IMAGEN: ${r.reclasificado_desde ? r.reclasificado_desde + " → " + r.etiqueta : c.etiqueta} ${r.pasa_a_prediccion ? "confirmado → PREDICCIÓN" : "falso positivo, descartado"}`;
    const fig = $("v-alertas").querySelector(`figure[data-clase="${c.clase}"]`);
    if (!fig) return;
    const cap = fig.querySelector("figcaption");
    cap.dataset.base = cap.dataset.base || cap.textContent;
    cap.textContent = `${cap.dataset.base} · IMAGEN: ${r.pasa_a_prediccion ? "confirmada" : "falso positivo, descartada"}`;
    cap.className = r.pasa_a_prediccion ? "v-ok" : "v-mal";
  }

  // ---- dibujo (mismo estilo que dibujar_detecciones de VIDEO) ----
  function dibujar() {
    const canvas = $("v-canvas"), ctx = canvas.getContext("2d"), v = $("v-video");
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    ctx.clearRect(0, 0, w, h);
    if (!ultimas || !v.videoWidth) return;
    const escala = Math.max(w / v.videoWidth, h / v.videoHeight);   // object-fit: cover
    const dx = (w - v.videoWidth * escala) / 2, dy = (h - v.videoHeight * escala) / 2;
    const k = v.videoWidth / ultimas.ancho;
    ctx.lineWidth = 3; ctx.font = "600 14px Inter, system-ui, sans-serif"; ctx.textBaseline = "top";
    for (const o of ultimas.objetos) {
      const c = o.caja, color = o.permitido ? "#2563EB" : "#EF4444";
      const x = dx + c.x1 * k * escala, y = dy + c.y1 * k * escala;
      ctx.strokeStyle = color; ctx.strokeRect(x, y, (c.x2 - c.x1) * k * escala, (c.y2 - c.y1) * k * escala);
      const t = `${o.etiqueta} ${o.confianza.toFixed(2)}`, ancho = ctx.measureText(t).width + 10, ye = y >= 22 ? y - 22 : y;
      ctx.fillStyle = color; ctx.fillRect(x - 1.5, ye, ancho, 22);
      ctx.fillStyle = "#FFFFFF"; ctx.fillText(t, x + 4, ye + 4);
    }
    const ahora = performance.now();
    let mayor = 0, activa = false;
    for (const c of Object.keys(inicio)) { mayor = Math.max(mayor, (ahora - inicio[c]) / 1000); if (alertada[c]) activa = true; }
    ctx.font = "700 16px Inter, system-ui, sans-serif";
    ctx.fillStyle = "rgba(15,23,42,.65)"; ctx.fillRect(10, 10, 215, 28);
    ctx.fillStyle = "#34D399"; ctx.fillText(`Tiempo: ${Math.min(mayor, TIEMPO_ALERTA).toFixed(1)}s / ${TIEMPO_ALERTA}s`, 18, 16);
    if (activa) {
      ctx.fillStyle = "rgba(127,29,29,.85)"; ctx.fillRect(10, 44, 330, 28);
      ctx.fillStyle = "#FFFFFF"; ctx.fillText("ALERTA: MATERIAL NO PERMITIDO", 18, 50);
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    $("v-iniciar").addEventListener("click", iniciar);
    $("v-detener").addEventListener("click", detener);
    $("calidad-video").addEventListener("change", () => { if (typeof DeteccionLocal !== "undefined") DeteccionLocal.fijarLado(Number($("calidad-video").value)); fps = 0; });
    window.addEventListener("resize", dibujar);
    fetch("/api/estado").then((r) => r.json()).then((d) => { $("estado-detector-video").textContent = d.detector.replace(/_/g, " "); }).catch(() => {});
  });
})();
