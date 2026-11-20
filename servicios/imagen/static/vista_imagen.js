// Interfaz del servicio IMAGEN: muestra el ultimo analisis que hizo del objeto que le entrego VIDEO.
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const pct = (v) => Math.round(v * 100) + "%";
  const ESTILO = { verificado: ["ok", "✅ verificado"], probable: ["medio", "🟡 probable"], no_verificado: ["mal", "❌ no verificado"] };
  let vistoTs = 0, vistoCompleto = null;

  function el(tag, texto, clase) { const n = document.createElement(tag); if (texto != null) n.textContent = texto; if (clase) n.className = clase; return n; }

  function pintar(u) {
    const h = u.hallazgo, v = u.resultado;
    $("i-video").className = "chip aviso";
    $("i-video").textContent = h.reclasificado_desde
      ? `🎥 VIDEO: ${h.reclasificado_desde} ${pct(h.confianza)} → IMAGEN lo reclasifica como ${h.etiqueta}`
      : `🎥 VIDEO: ${h.etiqueta} ${pct(h.confianza)} (≥ ${pct(h.umbral)})`;
    const [clase, etiqueta] = ESTILO[v.veredicto] || ["", v.veredicto];
    $("i-imagen").className = "chip " + clase; $("i-imagen").textContent = `🖼️ IMAGEN: ${h.etiqueta} ${etiqueta}`;
    const pp = $("i-prediccion");
    if (!v.pasa_a_prediccion) { pp.className = "chip mal"; pp.textContent = "🔮 PREDICCIÓN: no recibe nada · falso positivo"; }
    else if (u.prediccion.entregado) { pp.className = "chip ok"; pp.textContent = "🔮 PREDICCIÓN: recibido"; }
    else { pp.className = "chip mal"; pp.textContent = "🔮 PREDICCIÓN: no disponible"; }
    $("i-ayuda").textContent = `Último hallazgo a las ${new Date(u.ts * 1000).toLocaleTimeString("es")}. IMAGEN analizó el recorte de ${v.tamano_recorte[0]} × ${v.tamano_recorte[1]} px que VIDEO entregó (${u.tiempo_ms} ms).`;

    const det = $("i-detalle"); det.replaceChildren();
    const motivos = el("div"); motivos.append(el("h4", "Por qué este veredicto"));
    const ul = el("ul"); v.motivos.forEach((m) => ul.append(el("li", m))); ul.append(el("li", v.decision)); motivos.append(ul);
    const medidas = el("div"); medidas.append(el("h4", "Medidas de IMAGEN"));
    const dl = el("dl"); const filas = [];
    if (v.forma) filas.push(["Proporción (largo/ancho)", v.forma.aspecto], ["Vértices del contorno", v.forma.vertices],
      ["Rectangularidad", v.forma.rectangularidad], ["Solidez", v.forma.solidez], ["Cobertura del contorno", pct(v.forma.cobertura)]);
    filas.push(["VIDEO sobre el recorte mejorado", v.redeteccion.confirmada ? "confirma " + pct(v.redeteccion.confianza) : "no confirma"]);
    filas.forEach(([k, val]) => dl.append(el("dt", k), el("dd", String(val))));
    medidas.append(dl); det.append(motivos, medidas); det.hidden = false;

    const cont = $("i-secciones"); cont.replaceChildren();
    if (!u.completo) {
      const espera = el("section", null, "tarjeta seccion");
      espera.append(el("h2", "Preparando las imágenes de evidencia…"), el("p", "El veredicto ya está decidido. IMAGEN está calculando los filtros, el mejoramiento, la segmentación y los bordes de los temas del curso; aparecerán aquí en unos segundos.", "ayuda"));
      cont.append(espera);
    }
    for (const s of v.secciones) {
      const card = el("section", null, "tarjeta seccion");
      const cab = el("div", null, "cab"); cab.append(el("span", s.semana, "semana"), el("h2", s.titulo));
      card.append(cab, el("p", s.descripcion, "ayuda"));
      const rejilla = el("div", null, "pasos");
      for (const it of s.items) {
        const fig = el("figure"); const img = el("img"); img.src = it.imagen; img.alt = it.titulo;
        const cap = el("figcaption"); cap.append(el("strong", it.titulo), it.nota);
        fig.append(img, cap); rejilla.append(fig);
      }
      card.append(rejilla); cont.append(card);
    }
  }

  async function consultar() {
    try {
      const u = await (await fetch("/api/ultimo")).json();
      if (u.ts && (u.ts !== vistoTs || u.completo !== vistoCompleto)) { vistoTs = u.ts; vistoCompleto = u.completo; pintar(u); }
    } catch (e) { /* el servicio se esta reiniciando */ }
  }

  document.addEventListener("DOMContentLoaded", () => { consultar(); setInterval(consultar, 1500); });
})();
